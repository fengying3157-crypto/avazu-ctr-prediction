# =============================================================================
# 脚本：13_train_rare_category_lr.py
# 任务：低频截断阈值对照 —— 两组类别LR训练（min_count=2 vs min_count=5）
# 说明：
# 1. 直接读取 12 脚本产出的两套已保存编码，不重新编码
# 2. 两组各自用自己的 vocab_sizes 构建 LR，参数、optimizer 完全独立
# 3. 纯 BCE 训练，weight_decay=0，无 L2 正则
# =============================================================================
import os
import json
import numpy as np
import torch
from ctr_models import LR
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

SCRIPT_PATH = os.path.abspath(__file__)
PROJ_ROOT = os.path.dirname(SCRIPT_PATH)

EXP_ROOT = os.path.join(PROJ_ROOT, "data", "avazu", "processed", "rare_category_compare")
GROUP_DIRS = {
    2: os.path.join(EXP_ROOT, "min_count_2"),
    5: os.path.join(EXP_ROOT, "min_count_5"),
}

def load_group(group_dir):
    """读取一组已保存的编码与配置"""
    with open(os.path.join(group_dir, "feature_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feature_cols = cfg["feature_cols"]
    # 词表大小直接读配置：已含编号0，不能拿词表字典长度当大小
    vocab_sizes = [int(cfg["vocab_sizes"][col]) for col in feature_cols]

    npz = np.load(os.path.join(group_dir, "encoded_features.npz"))
    x_train = torch.from_numpy(npz["x_train"]).long()
    y_train = torch.from_numpy(npz["y_train"]).float()
    x_val = torch.from_numpy(npz["x_val"]).long()
    y_val = torch.from_numpy(npz["y_val"]).float()
    return feature_cols, vocab_sizes, x_train, y_train, x_val, y_val

data_2 = load_group(GROUP_DIRS[2])
data_5 = load_group(GROUP_DIRS[5])

for mc, (feature_cols, vocab_sizes, x_train, y_train, x_val, y_val) in [(2, data_2), (5, data_5)]:
    print(f"==== min_count={mc} ====")
    print(f"字段数 {len(feature_cols)}，词表大小之和 {sum(vocab_sizes)}")
    print(f"x_train {tuple(x_train.shape)}  y_train {tuple(y_train.shape)}")
    print(f"x_val   {tuple(x_val.shape)}    y_val   {tuple(y_val.shape)}")
    print(f"预期LR参数量（词表之和+1个bias）：{sum(vocab_sizes) + 1}")

    # 标签必须是一维
    assert y_train.dim() == 1 and y_val.dim() == 1, "标签不是一维"

    # 编码上界自检：每个字段最大编号必须小于本字段词表大小
    for i, col in enumerate(feature_cols):
        max_id = int(torch.max(torch.max(x_train[:, i]), torch.max(x_val[:, i])))
        assert max_id < vocab_sizes[i], f"❌ {col} 编码越界：max={max_id}, vocab_size={vocab_sizes[i]}"
    print("✅ 编码上界自检通过")

# 两组字段顺序必须完全一致，否则统计表逐字段对照不成立
assert data_2[0] == data_5[0], "❌ 两组 feature_cols 顺序不一致"
print("✅ 两组字段顺序一致")

# -------------------------- 第二步：构建LR模型，全部参数置零，校验初始logits --------------------------
def build_zero_lr(vocab_sizes):
    model = LR(vocab_sizes)
    # 不硬编码emb_lin，遍历全部参数置0
    for p in model.parameters():
        nn.init.zeros_(p)
    return model

# 分别构建两个独立LR
lr_2 = build_zero_lr(data_2[1])
lr_5 = build_zero_lr(data_5[1])

# 校验参数量
cnt_2 = sum(p.numel() for p in lr_2.parameters())
cnt_5 = sum(p.numel() for p in lr_5.parameters())
print(f"\n==== 参数数量校验 ====")
print(f"min_count=2 LR总参数量：{cnt_2}（预期12770）")
print(f"min_count=5 LR总参数量：{cnt_5}（预期4690）")
assert cnt_2 == 12770, f"min_count=2 参数数量不对，得到{cnt_2}"
assert cnt_5 == 4690, f"min_count=5 参数数量不对，得到{cnt_5}"
print("✅ 参数量校验通过")

# 拿一小批样本测试初始输出是否接近0
_, _, x_train_2, y_train_2, _, _ = data_2
_, _, x_train_5, y_train_5, _, _ = data_5

with torch.no_grad():
    logits_2 = lr_2(x_train_2[:4])
    logits_5 = lr_5(x_train_5[:4])

print("\n==== 初始logits校验（全部参数置零） ====")
print(f"min_count=2 前4条logits：{logits_2.squeeze().tolist()}")
print(f"min_count=5 前4条logits：{logits_5.squeeze().tolist()}")

# 初始logits应该全部≈0
assert torch.allclose(logits_2, torch.zeros_like(logits_2), atol=1e-6), "min_count=2 初始logits不为0"
assert torch.allclose(logits_5, torch.zeros_like(logits_5), atol=1e-6), "min_count=5 初始logits不为0"
print("✅ 两组初始logits校验通过，参数置零成功")

# -------------------------- 第三步：DataLoader、训练、评估函数 --------------------------
def get_loaders(x_tr, y_tr, x_val, y_val, seed=123):
    train_ds = TensorDataset(x_tr, y_tr)
    val_ds = TensorDataset(x_val, y_val)
    # 训练集固定种子，保证两组采样顺序对齐
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    return train_loader, val_loader

def train_one_epoch(model, loader, opt):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x, y in loader:
        opt.zero_grad()
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.shape[0]
        total_samples += x.shape[0]
    avg_bce = total_loss / total_samples
    return avg_bce

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits = []
    all_y = []
    for x, y in loader:
        logits = model(x)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
    all_logits = torch.cat(all_logits)
    all_y = torch.cat(all_y)
    prob = torch.sigmoid(all_logits)
    logloss = F.binary_cross_entropy(prob, all_y).item()
    auc = roc_auc_score(all_y.numpy(), prob.numpy())
    pred = (prob >= 0.5).float()
    acc = (pred == all_y).float().mean().item()
    return logloss, auc, acc

# 构建loader和optimizer
train_loader_2, val_loader_2 = get_loaders(data_2[2], data_2[3], data_2[4], data_2[5], seed=123)
train_loader_5, val_loader_5 = get_loaders(data_5[2], data_5[3], data_5[4], data_5[5], seed=123)

opt_2 = Adam(lr_2.parameters(), lr=0.01, weight_decay=0.0)
opt_5 = Adam(lr_5.parameters(), lr=0.01, weight_decay=0.0)

# 记录历史
history_2 = []
history_5 = []
epoch_num = 5

print("\n==== 开始训练 5 轮 ====")
for epoch in range(1, epoch_num+1):
    # min_count=2
    tr_bce_2 = train_one_epoch(lr_2, train_loader_2, opt_2)
    val_ll_2, val_auc_2, val_acc_2 = evaluate(lr_2, val_loader_2)
    history_2.append({
        "epoch": epoch,
        "train_bce": tr_bce_2,
        "val_logloss": val_ll_2,
        "val_auc": val_auc_2,
        "val_acc": val_acc_2
    })

    # min_count=5
    tr_bce_5 = train_one_epoch(lr_5, train_loader_5, opt_5)
    val_ll_5, val_auc_5, val_acc_5 = evaluate(lr_5, val_loader_5)
    history_5.append({
        "epoch": epoch,
        "train_bce": tr_bce_5,
        "val_logloss": val_ll_5,
        "val_auc": val_auc_5,
        "val_acc": val_acc_5
    })

    print(f"Epoch {epoch}")
    print(f"  mc=2 | train_bce:{tr_bce_2:.6f}, val_logloss:{val_ll_2:.6f}, val_auc:{val_auc_2:.6f}, val_acc:{val_acc_2:.6f}")
    print(f"  mc=5 | train_bce:{tr_bce_5:.6f}, val_logloss:{val_ll_5:.6f}, val_auc:{val_auc_5:.6f}, val_acc:{val_acc_5:.6f}")


# -------------------------- 第四步：保存历史、模型、实验配置，重载校验 --------------------------
# 保存各组history和模型
def save_group(model, history, group_dir):
    torch.save(model.state_dict(), os.path.join(group_dir, "lr_epoch5.pt"))
    with open(os.path.join(group_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

save_group(lr_2, history_2, GROUP_DIRS[2])
save_group(lr_5, history_5, GROUP_DIRS[5])

# 根目录公共实验配置
common_cfg = {
    "task": "LR on different min_count encoding",
    "lr": 0.01,
    "weight_decay": 0.0,
    "epochs": 5,
    "train_batch": 512,
    "val_batch": 2048,
    "seed": 123,
    "model_type": "LR",
    "group_info": {
        "min_count_2": {"param_count":12770},
        "min_count_5": {"param_count":4690}
    }
}
with open(os.path.join(EXP_ROOT, "experiment_config.json"), "w", encoding="utf-8") as f:
    json.dump(common_cfg, f, indent=2, ensure_ascii=False)
print("\n✅ 模型、history、公共配置保存完成")

# 重载校验：取同一批验证样本，对比保存前后logits
def reload_check(group_dir, vocab_sizes, x_val, y_val, ref_model):
    model = LR(vocab_sizes)
    model.load_state_dict(torch.load(os.path.join(group_dir, "lr_epoch5.pt")))
    model.eval()
    # 取前128条验证样本
    x_batch = x_val[:128]
    with torch.no_grad():
        logits_new = model(x_batch).squeeze()
        logits_old = ref_model(x_batch).squeeze()
    ok = torch.allclose(logits_new, logits_old, atol=1e-5)
    # 重载后完整评估，单独构造val loader
    val_ds = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    ll, auc, acc = evaluate(model, val_loader)
    return ok, ll, auc, acc

print("\n==== 重载校验 ====")
ok2, ll2_reload, auc2_reload, acc2_reload = reload_check(GROUP_DIRS[2], data_2[1], data_2[4], data_2[5], lr_2)
ok5, ll5_reload, auc5_reload, acc5_reload = reload_check(GROUP_DIRS[5], data_5[1], data_5[4], data_5[5], lr_5)

print(f"min_count=2 重载logits一致：{ok2}，重载val_logloss:{ll2_reload:.6f}, auc:{auc2_reload:.6f}, acc:{acc2_reload:.6f}")
print(f"min_count=5 重载logits一致：{ok5}，重载val_logloss:{ll5_reload:.6f}, auc:{auc5_reload:.6f}, acc:{acc5_reload:.6f}")
assert ok2 and ok5, "重载后logits不一致"
print("✅ 重载校验全部通过")
