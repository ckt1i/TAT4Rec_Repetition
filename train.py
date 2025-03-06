import torch
import torch.nn as nn
import numpy as np


def train(model, train_loader, test_loader , loss_fn, optimizer, num_epochs=10, device="cpu"):

    model.train()  # 进入训练模式
    best_ndcg10 = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        total_loss = 0

        for item_ids, timestamps, targets, negatives in train_loader:
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
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 每 5 个 epoch 评估一次
        if (epoch + 1) % 5 == 0:
            results = evaluate(model, test_loader, device)
            print(f"Evaluation at Epoch {epoch+1}: {results}")

            # 选取最佳模型
            if results["NDCG@10"] > best_ndcg10:
                best_ndcg10 = results["NDCG@10"]
                best_model_state = model.state_dict()

    # 训练结束，加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Best model with NDCG@10 = {best_ndcg10:.4f} loaded.")


def dcg_at_k(r, k):
    """ 计算 DCG@K """
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    """ 计算 NDCG@K """
    r = np.asfarray(r)[:k]
    dcg_max = dcg_at_k(np.sort(r)[::-1], k) 
    return dcg_at_k(r, k) / dcg_max if dcg_max > 0 else 0.0

def hit_at_k(r, k):
    """ 计算 Hit@K """
    r = np.asfarray(r)[:k]
    return 1.0 if np.sum(r[:k]) > 0 else 0.0

def mrr(r):
    """ 计算 MRR """
    for i, rel in enumerate(r):
        if rel:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_ranking(predictions, targets, k_values=[5, 10]):
    """
    计算 NDCG@K, Hit@K, MRR 指标
    """
    results = {f'NDCG@{k}': 0.0 for k in k_values}
    results.update({f'Hit@{k}': 0.0 for k in k_values})
    results["MRR"] = 0.0
    num_users = len(targets)

    for pred, target in zip(predictions, targets):
        sorted_indices = np.argsort(pred)[::-1]  # 按得分降序排序
        relevance = np.isin(sorted_indices, target).astype(int)  # 计算相关性

        for k in k_values:
            results[f'NDCG@{k}'] += ndcg_at_k(relevance, k)
            results[f'Hit@{k}'] += hit_at_k(relevance, k)

        results["MRR"] += mrr(relevance)

    # 取平均值
    for key in results:
        results[key] /= num_users
        results[key] = round(results[key], 4)


    return results



def evaluate(model, test_loader, device = "cuda"):
    """
    评估模型，计算 NDCG@K, Hit@K, MRR
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for item_ids, timestamps, targets, _ in test_loader:
            item_ids, timestamps, targets = (
                item_ids.to(device),
                timestamps.to(device),
                targets.to(device),
            )

            scores = model(item_ids, timestamps) # (batch_size, seq_len, num_items)
            scores = scores[:, -1, :] # 只取最后一个时间步的预测
            scores = scores.cpu().numpy() # 转换为 NumPy 数组
            all_predictions.extend(scores)
            all_targets.extend(targets.cpu().numpy())

    return evaluate_ranking(all_predictions, all_targets, k_values=[5, 10])