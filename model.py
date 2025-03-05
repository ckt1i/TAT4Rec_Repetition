import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class ItemEmbedding(nn.Module):
    
    def __init__(self, num_items, d_model):
        """
        num_items: 物品的种类数量
        d_model: 物品嵌入维度（embedding size）
        """
        super(ItemEmbedding, self).__init__()

        # 物品 ID 的嵌入层 (num_items, d_model)
        self.item_embedding = nn.Embedding(num_items, d_model)

    def forward(self, item_ids):
        """
        item_ids: 物品 ID 张量 (batch_size, seq_len)
        returns: 物品嵌入向量 (batch_size, seq_len, d_model)
        """
        return self.item_embedding(item_ids)

class TimeStampEmbedding(nn.Module):
    
    def __init__(self, d_model, num_bins, window_size):
        """
        d_model: 物品嵌入维度
        num_bins: 时间戳的分箱数量
        window_size: 滑动窗口大小
        """
        super(TimeStampEmbedding , self).__init__()

        self.d_model = d_model
        self.num_bins = num_bins
        self.window_size = window_size

        # 时间戳的嵌入层 (num_bins, d_model)
        self.time_embedding = nn.Embedding(num_bins, d_model)

        # 预计算窗口函数参数
        self.register_buffer("div_term", torch.arange(0, num_bins).float())

    def forward(self, timestamps):
        """
        timestamps: 时间戳张量 (batch_size, seq_len)
        
        weighted_embeddings: 时间戳嵌入向量 (batch_size, seq_len, d_model)
        """
        # 计算相对时间间隔
        related_time = timestamps - timestamps[:, :1]

        # 归一化并裁剪
        max_timegap = torch.max(related_time)
        scaled_timegap = torch.clip((related_time / max_timegap) , 0 , self.num_bins - 1).long()

        # 获取嵌入向量
        embeddings = self.time_embedding(scaled_timegap)

        # 计算窗口函数加权
        window_weights = torch.cos(math.pi * (self.div_term - scaled_timegap.unsqueeze(-1)) / self.window_size) ** 2
        window_weights = torch.clamp(window_weights , min=0.0)

        tmp1 = embeddings.unsqueeze(-2)
        tmp2 = window_weights.unsqueeze(-1)

        weight_embeddings = (embeddings.unsqueeze(-2) * window_weights.unsqueeze(-1)).sum(dim=-2)

        return weight_embeddings


def ScaledDotProductAttention(Q, K, V, mask=None):
    """
    Q: Query matrix (batch_size , seq_len , embed_size)
    K: Key matrix (batch_size , seq_len , embed_size)
    V: Value matrix (batch_size , seq_len , embed_size)
    mask : mask matrix 

    output: Output matrix with attention applied (batch_size , seq_len , embed_size)
    attention_matrix: Attention matrix (batch_size , seq_len , seq_len)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_matrix = F.softmax(scores, dim=-1)

    output = torch.matmul(attention_matrix, V)

    return output , attention_matrix

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size , num_heads):
        """
        embed_size : Embedding size of the input
        num_heads : Number of heads in the multi-head attention
        """

        super(MultiHeadAttention , self).__init__()

        assert embed_size % num_heads == 0

        self.embed_size = embed_size
        self.num_hed = num_heads
        self.head_dim = embed_size // num_heads

        self.W_Q = nn.Linear(embed_size , embed_size)
        self.W_K = nn.Linear(embed_size , embed_size)
        self.W_V = nn.Linear(embed_size , embed_size)

        self.W_O = nn.Linear(embed_size , embed_size)

    def forward(self , Q , K , V , mask=None):
        """
        Q : Query matrix (batch_size , seq_len , embed_size)
        K : Key matrix (batch_size , seq_len , embed_size)
        V : Value matrix (batch_size , seq_len , embed_size)
        mask : mask matrix (batch_size , seq_len , seq_len)
        """

        batch_size = Q.size(0)

        # Get the number of the sequence length of the query and key matrices
        seq_len_q = Q.size(1)
        seq_len_k = K.size(1)

        # 将线性变换后的“共享”矩阵拆分为多头，调整维度为 (batch_size, h, seq_len, d_k)
        # d_k 就是每个注意力头的维度
        Q = self.W_Q(Q).view(batch_size , seq_len_q , self.num_hed , -1).transpose(1, 2)
        K = self.W_K(K).view(batch_size , seq_len_k , self.num_hed , -1).transpose(1, 2)
        V = self.W_V(V).view(batch_size , seq_len_k , self.num_hed , -1).transpose(1, 2)

        # Scaled Dot-Product Attention
        scaled_attention, _ = ScaledDotProductAttention(Q , K , V , mask)

        # Merge the multi-head attention output via concat fuction and reversed into (batch_size , seq_len , embed_size)
        concat_output = scaled_attention.transpose(1, 2).contiguous().view(batch_size , -1 , self.embed_size)

        # Linear transformation
        output = self.W_O(concat_output)

        return output

class LayerNormalization(nn.Module):   

    def __init__(self, feature_size , epsilon = 1e-9):
        """
        feature_size: Feature size of the input. (Normalization dimention of the features)
        epsilon: Epsilon value for avoid division by zero
        """
        super(LayerNormalization, self).__init__()

        self.gamma = nn.Parameter(torch.ones(feature_size)) # learnable parameters
        self.beta = nn.Parameter(torch.zeros(feature_size)) # learnable parameters
        self.epsilon = epsilon

    def forward(self, X):

        mean = X.mean(-1, keepdim=True)
        std = X.std(-1, keepdim=True)

        return self.gamma * (X - mean) / (std + self.epsilon) + self.beta

class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model , num_heads , dropout=0.1 , epsilon = 1e-9):
        """
        d_model : Feature size of the input. (Normalization dimention of the features)
        dropout : Dropout rate
        epsilon : Epsilon value for avoid division by zero
        """
        super(ResidualAttentionBlock , self).__init__()

        self.multi_head_attention = MultiHeadAttention(embed_size = d_model, num_heads = num_heads)
        
        self.normalization = LayerNormalization(d_model , epsilon)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self , Q , K , V , mask = None):

        # Apply multi-head attention
        multi_head_attention_output = self.multi_head_attention(Q , K , V , mask)

        # Apply dropout
        multi_head_attention_output = self.dropout(multi_head_attention_output)

        # Residual connection 
        output = Q + multi_head_attention_output

        # Apply normalization
        output = self.normalization(output)

        return output


class FeedForward(nn.Module):

    def __init__(self , d_model , d_ff):
        """
        d_model : Feature size of the input. (Normalization dimention of the features)
        d_ff : Hidden layer size of the feed forward network
        dropout : Dropout rate
        """
        super(FeedForward , self).__init__()

        self.linear1 = nn.Linear(d_model , d_ff)
        self.linear2 = nn.Linear(d_ff , d_model)


    def forward(self , X):

        return self.linear2(F.relu(self.linear1(X)))


class EncoderLayer(nn.Module):

    def __init__(self , d_embed , num_heads ,  dropout=0.1 , epsilon = 1e-9):
        
        super(EncoderLayer, self).__init__()

        self.self_attention = ResidualAttentionBlock(d_embed , num_heads , dropout , epsilon)
        self.feedforward = FeedForward(d_embed , d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self , X , mask = None):
        
        # Apply Residual Attention Block
        output = self.self_attention(X , X , X , mask)

        # Apply dropout
        output = self.dropout(output)

        # Apply Feed Forward Network
        output = self.feedforward(output)

        return output

class EncoderBlock(nn.Module):

    def __init__(self, embed_size, num_heads, num_layers, dropout=0.1, epsilon=1e-9):
        """
        embed_size : Embedding size of the input
        num_heads : Number of attention heads
        num_heads : Number of encoder layers to stack , h in the Paper
        num_layers : Number of encoder layers to stack , P in the Paper
        dropout : Dropout rate
        epsilon : Small constant for numerical stability in layer normalization
        """
        super(EncoderBlock, self).__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, num_heads, dropout, epsilon) 
             for _ in range(num_layers)])
        
        self.feedforward = FeedForward(embed_size, num_heads)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, mask=None):
       
        for layer in self.layers:
            X = layer(X, mask)

        Encode_output = X

        X = self.feedforward(X)

        X = self.dropout(X)

        X = X + Encode_output
        
        return X , Encode_output


class DecoderLayer(nn.Module):
    
    def __init__(self , d_embed , num_heads , dropout=0.1 , epsilon = 1e-9):

        super(DecoderLayer , self).__init__()

        self.self_attention = ResidualAttentionBlock(d_embed , num_heads , dropout , epsilon)
        self.cross_attention = ResidualAttentionBlock(d_embed , num_heads , dropout , epsilon)
        self.feedforward = FeedForward(d_embed , d_embed)

    def forward(self , X , Encode_output , self_mask = None , cross_mask = None):

        # Apply Residual Attention Block
        self_output = self.self_attention(X , X , X , self_mask)

        # Apply Residual Attention Block
        cross_output = self.cross_attention(self_output , Encode_output , Encode_output , cross_mask)

        # Apply Feed Forward Network
        output = self_output + self.feedforward(cross_output)

        return output

class DecoderBlock(nn.Module):
    
    def __init__(self , embed_size , num_heads , num_layers , dropout=0.1 , epsilon=1e-9):

        super(DecoderBlock , self).__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(embed_size , num_heads , dropout , epsilon)
             for _ in range(num_layers)])

        self.feedforward = FeedForward(embed_size , num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self , X , Encode_output , self_mask = None , cross_mask = None):

        for layer in self.layers:
            X = layer(X , Encode_output , self_mask , cross_mask)

        Decode_output = X

        X = self.feedforward(X)

        X = self.dropout(X)

        X = X + Decode_output

        return X , Decode_output
        

class PredictionLayer(nn.Module):
    
    def __init__(self, d_model, num_items):
        """
        d_model: Item embedding dimension (same as model hidden size)
        num_items: Total number of items in the recommendation system
        """
        super(PredictionLayer, self).__init__()

        # 共享的 Item Embedding 矩阵 (num_items, d_model)
        self.item_embeddings = nn.Parameter(torch.randn(num_items, d_model))

    def forward(self, decoder_output):
        """
        decoder_output: (batch_size, seq_len, d_model)
        returns: preference scores (batch_size, seq_len, num_items)
        """
        # 计算与物品嵌入的点积，获得匹配分数
        scores = torch.matmul(decoder_output, self.item_embeddings.T)  # (batch_size, seq_len, num_items)
        
        return scores


class TAT4SRec(nn.Module):

    def __init__(self , num_items , d_model , num_heads , num_layers , num_bins , dropout=0.1 , epsilon=1e-9):
        """
        num_items : Total number of items in the recommendation system
        d_model : Item embedding dimension (same as model hidden size)
        num_heads : Number of attention heads
        num_layers : Number of encoder and decoder layers to stack
        num_bins: NUmber of bins for the timestamp embedding
        dropout : Dropout rate
        epsilon : Small constant for numerical stability in layer normalization
        """
        super(TAT4SRec , self).__init__()

        self.itemembedding = ItemEmbedding(num_items , d_model)
        self.timestampembedding = TimeStampEmbedding(d_model , num_bins , window_size = 1024)

        self.encoder = EncoderBlock(d_model , num_heads , num_layers , dropout , epsilon)
        self.decoder = DecoderBlock(d_model , num_heads , num_layers , dropout , epsilon)

        self.prediction = PredictionLayer(d_model , num_items)

    def forward(self , item_ids , timestamps):
        
        # Embedding layer
        item_embeddings = self.itemembedding(item_ids)
        timestamp_embeddings = self.timestampembedding(timestamps)

        # Encoder
        encoder_output , Encode_output = self.encoder(timestamp_embeddings)

        # Decoder
        decoder_output , Decode_output = self.decoder(item_embeddings , Encode_output)

        # Prediction
        scores = self.prediction(decoder_output)

        return scores
    

# Test the Model
def main():
    # 假设参数
    batch_size = 2
    seq_len = 10
    num_items = 1000  # 物品总数
    d_model = 64  # 物品嵌入维度
    num_heads = 4  # 注意力头数
    num_layers = 2  # 编码器和解码器层数
    num_bins = 100  # 时间戳分箱数

    # 创建物品 ID 和时间戳输入张量
    item_ids = torch.randint(0, num_items, (batch_size, seq_len))
    timestamps = torch.randint(0, 1000, (batch_size, seq_len))

    # 初始化模型
    model = TAT4SRec(num_items, d_model, num_heads, num_layers, num_bins)

    # 计算预测分数
    scores = model(item_ids, timestamps)

    print(scores.shape)  # (batch_size, seq_len, num_items)

if __name__ == "__main__":
    main()