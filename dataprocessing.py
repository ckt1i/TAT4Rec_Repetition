import random
import torch
from torch.utils.data import Dataset

def Read_user_csv_Data(User_path):

    users_interactions = []
    users_timestamps = []
    user_interactions = []
    user_timestamps = []
    item2idx = {}  # item_id → int 映射表
    current_user = None
    next_idx = 1  

    with open(User_path, 'r') as f:
            
        for line in f:

            user_id, item_id, rating, timestamp, purchase = line.strip().split(',')
            
            # 如果 item_id 不在映射表中，分配新索引
            if item_id not in item2idx:
                item2idx[item_id] = next_idx
                next_idx += 1

            user_interaction = item2idx[item_id]  # 转换 item_id 为整数索引
            user_timestamp = int(timestamp) 
            
            if user_id != current_user and current_user != None: 

                users_interactions.append(user_interactions)
                users_timestamps.append(user_timestamps)
                
                user_interactions = []
                user_timestamps = []

                current_user = user_id
            
            if current_user == None:

                current_user = user_id

            user_interactions.append(user_interaction)
            user_timestamps.append(user_timestamp)
        
    return users_interactions , users_timestamps , item2idx


def negative_sampling(pos_samples, num_items, num_neg=1):
    """
    负采样：确保负样本不在正样本中
    """
    neg_samples = []
    for pos in pos_samples:
        neg = pos
        while neg in pos_samples:  # 直到选出一个未出现的负样本
            neg = random.randint(0, num_items - 1)
        neg_samples.append(neg)
    return neg_samples


class RecSysDataset(Dataset):
    def __init__(self, interactions, timestamps, num_items, seq_len=50):
        self.interactions = interactions
        self.timestamps = timestamps
        self.num_items = num_items
        self.seq_len = seq_len

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        item_seq = self.interactions[idx]  
        time_seq = self.timestamps[idx] 

        # 截断或填充序列到固定长度 seq_len
        if len(item_seq) >= self.seq_len:
            item_seq = item_seq[-self.seq_len:]
            time_seq = time_seq[-self.seq_len:]
        else:
            pad_len = self.seq_len - len(item_seq)
            item_seq = [0] * pad_len + item_seq  # 物品填充 0
            time_seq = [0] * pad_len + time_seq  # 时间戳填充 0

        # 目标物品：右移一位
        target_seq = item_seq[1:] + [0]  # 最后一个时间步用 0 填充

        # 负采样
        neg_seq = negative_sampling(item_seq, self.num_items)

        return torch.tensor(item_seq), torch.tensor(time_seq), torch.tensor(target_seq), torch.tensor(neg_seq)
    
