import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataprocessing import Read_user_csv_Data, RecSysDataset , split_train_test
from model import TAT4SRec
from train import *


def main():
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取用户交互数据
    user_path = "Data/User_Data.csv"  # 你的数据文件路径

    # 读取数据，并获得 item_id 映射表
    users_interactions, users_timestamps, item2idx = Read_user_csv_Data(user_path)

    # 划分训练集和测试集
    train_interactions, train_timestamps, test_interactions, test_timestamps = split_train_test(users_interactions, users_timestamps, train_size=0.2)

    # 统计物品总数
    num_items = len(item2idx) + 1

    # 训练参数
    d_model = 64  # 物品嵌入维度
    num_heads = 8  # 多头注意力
    num_layers = 2  # 编码器/解码器层数
    num_bins = 100  # 时间分桶数
    batch_size = 128 # 批大小是
    learning_rate = 0.001 # 学习率
    num_epochs = 100 # 训练轮数
    seq_len = 50

    # 创建数据集 & DataLoader
    train_dataset = RecSysDataset(train_interactions, train_timestamps, num_items, seq_len)
    test_dataset = RecSysDataset(test_interactions, test_timestamps, num_items, seq_len)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = TAT4SRec(num_items, d_model, num_heads, num_layers, num_bins).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 进行优化

    # 训练模型
    train(model, train_data_loader, test_data_loader , loss_fn, optimizer, num_epochs, device)
    print("Final Evaluation:", evaluate(model, test_data_loader, device))

if __name__ == "__main__":
    main()