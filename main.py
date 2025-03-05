import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataprocessing import Read_user_csv_Data, RecSysDataset
from model import TAT4SRec


def train(model, data_loader, loss_fn, optimizer, num_epochs=10, device="cpu"):
   
    model.train()  # 进入训练模式

    for epoch in range(num_epochs):
        total_loss = 0

        for item_ids, timestamps, targets, negatives in data_loader:
            # 发送数据到 GPU
            item_ids, timestamps, targets, negatives = (
                item_ids.to(device),
                timestamps.to(device),
                targets.to(device),
                negatives.to(device),
            )

            # 检查输入
            if torch.isnan(item_ids).any() or torch.isnan(timestamps).any() or \
               torch.isnan(targets).any() or torch.isnan(negatives).any():
                print("NaN detected in input!")
                break

            # 前向传播
            scores = model(item_ids, timestamps)  # (batch_size, seq_len, num_items)
            if torch.isnan(scores).any():
                print("NaN detected in scores!")
                break

            # 计算目标物品的预测分数
            pos_scores = scores.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
            
            neg_scores = scores.gather(2, negatives.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)

            # 计算二元交叉熵损失
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(pos_scores, torch.ones_like(pos_scores)) + loss_fn(neg_scores, torch.zeros_like(neg_scores))
            
            # 反向传播 & 更新参数
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 限制梯度范数
            optimizer.step()

            total_loss += loss.item()

        # 打印每个 Epoch 的损失
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def main():
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取用户交互数据
    user_path = "Data/Raw_User_Data.csv"  # 你的数据文件路径

    # 读取数据，并获得 item_id 映射表
    users_interactions, users_timestamps, item2idx = Read_user_csv_Data(user_path)

    # 统计物品总数
    num_items = len(item2idx) + 1

    # 训练参数
    d_model = 64  # 物品嵌入维度
    num_heads = 8  # 多头注意力
    num_layers = 2  # 编码器/解码器层数
    num_bins = 100  # 时间分桶数
    batch_size = 128 # 批大小
    learning_rate = 0.001 # 学习率
    num_epochs = 20 # 训练轮数
    seq_len = 50

    # 创建数据集 & DataLoader
    dataset = RecSysDataset(users_interactions , users_timestamps, num_items, seq_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = TAT4SRec(num_items, d_model, num_heads, num_layers, num_bins).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 进行优化

    # 训练模型
    train(model, data_loader, loss_fn, optimizer, num_epochs, device)

if __name__ == "__main__":
    main()