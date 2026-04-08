import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------- 配置 -----------
MAX_REGION_MEMORY_LEN = 256  # 每类最多存多少个样本特征（默认256）
FEATURE_DIM = 256  # 单个特征维度
criterion = nn.CrossEntropyLoss()
# ----------------------------


# 初始化 memory：每类一个 [256, 0] 的张量
def region_memory(num_class, device):
    return {
        f'region_class{cls_id}': torch.zeros(FEATURE_DIM, 0, device=device)
        for cls_id in range(num_class)
    }


# 更新 memory：限制长度 + detach + no_grad
@torch.no_grad()
def update_region_memory(region_dic, keys, vals, cat):
    if cat not in vals:
        return region_dic
    idx = list(vals).index(cat)
    new_key = keys[idx].unsqueeze(-1).detach()  # shape: [256, 1]

    memory = region_dic[f'region_class{cat}']
    updated = torch.cat((memory, new_key), dim=1)

    if updated.shape[1] > MAX_REGION_MEMORY_LEN:
        updated = updated[:, -MAX_REGION_MEMORY_LEN:]

    region_dic[f'region_class{cat}'] = updated
    return region_dic


# 类别级对比损失计算（正类 vs 其他类 memory）
def compute_contrast_loss(l_pos, l_neg, device, temperature=0.2):
    N = l_pos.size(0)
    logits = torch.cat((l_pos, l_neg), dim=1)  # shape: [N, 1+neg_samples]
    logits /= temperature
    labels = torch.zeros(N, dtype=torch.long).to(device)  # 正类是第0个
    return criterion(logits, labels)


# region contrast loss 总体流程
def region_contrast_loss(num_class, keys, vals, region_memory, device):
    contrast_loss = torch.tensor(0., device=device)

    for cls_id in range(num_class):
        if cls_id not in vals:
            continue

        query = keys[list(vals).index(cls_id)]  # shape: [256]
        query = query.detach()

        pos_bank = region_memory[f'region_class{cls_id}'].clone().detach()  # [256, N_pos]
        if pos_bank.shape[1] == 0:
            continue  # 没有正类样本

        l_pos = torch.matmul(query.unsqueeze(0), pos_bank)  # [1, N_pos]

        # 构造负类 memory
        l_neg = []
        for other_id in range(num_class):
            if other_id == cls_id:
                continue
            neg_bank = region_memory[f'region_class{other_id}'].clone().detach()
            if neg_bank.shape[1] > 0:
                l_neg.append(torch.matmul(query.unsqueeze(0), neg_bank))  # [1, N_neg]

        if len(l_neg) == 0:
            continue  # 没有负类，不计算损失
        l_neg = torch.cat(l_neg, dim=1)  # [1, N_total_neg]

        contrast_loss += compute_contrast_loss(l_pos, l_neg, device=device)

    return contrast_loss
