from pathlib import Path

import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd

from ctr_models import FM

# ===================== 路径配置 =====================
DATA_DIR = Path(__file__).resolve().parent / "data" / "avazu" / "processed"
npz_path = f"{DATA_DIR}/encoded_features.npz"
config_path = f"{DATA_DIR}/feature_config.json"
out_lambda0 = f"{DATA_DIR}/fm_lambda0.pt"
out_lambda1e4 = f"{DATA_DIR}/fm_lambda1e-4.pt"
hist_path = f"{DATA_DIR}/fm_l2_history.json"

# ===================== 加载数据与配置 =====================
with open(config_path, "r", encoding="utf-8") as f:
    feature_config = json.load(f)
vocab_sizes = [int(v) for v in feature_config["vocab_sizes"].values()]

data = np.load(npz_path)
x_train = torch.from_numpy(data["x_train"]).long()
y_train = torch.from_numpy(data["y_train"]).float().squeeze(-1)
x_val = torch.from_numpy(data["x_val"]).long()
y_val = torch.from_numpy(data["y_val"]).float().squeeze(-1)

print("x_train shape", x_train.shape, "y_train shape", y_train.shape)
print("x_val shape", x_val.shape, "y_val shape", y_val.shape)

# 数据集
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)

# ===================== 固定初始化，保证两组起点完全一致 =====================
torch.manual_seed(42)
# 基准模型，执行指定初始化
base_fm = FM(vocab_sizes,latent_dim=8)
base_fm.init_spec()
#保存初始权重，两个实验组都从这一份开始
init_state = copy.deepcopy(base_fm.state_dict())

# 模型1：λ=0，无正则
fm0 = FM(vocab_sizes,latent_dim=8)
fm0.load_state_dict(init_state)

#模型2 λ=0.0001，二阶emb加L2
fm1 = FM(vocab_sizes,latent_dim=8)
fm1.load_state_dict(init_state)

# 校验：训练前参数完全一致
# 校验：训练前参数完全相等
def check_param_equal(m1, m2):
    for (p1, p2) in zip(m1.parameters(), m2.parameters()):
        if not torch.allclose(p1, p2):
            return False
    return True
print("初始化参数对齐校验：", check_param_equal(fm0, fm1))

# 优化器：weight_decay=0，我们自己手动算二阶L2惩罚
opt0 = optim.Adam(fm0.parameters(), lr=0.01, weight_decay=0)
opt1 = optim.Adam(fm1.parameters(), lr=0.01, weight_decay=0)

# 两个独立dataloader，同一个种子123，保证批次顺序完全一样
gen0 = torch.Generator()
gen0.manual_seed(123)
gen1 = torch.Generator()
gen1.manual_seed(123)

train_loader0 = DataLoader(train_ds, batch_size=512, shuffle=True, generator=gen0)
train_loader1 = DataLoader(train_ds, batch_size=512, shuffle=True, generator=gen1)
val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)

loss_bce = nn.BCEWithLogitsLoss(reduction="mean")
lam = 1e-4


# ===================== 验证评估函数（只算BCE，不带正则惩罚） =====================
def evaluate(model, loader, bce_criterion):
    model.eval()
    total_loss = 0.0
    y_true_list = []
    logit_list = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = bce_criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            y_true_list.append(yb.cpu())
            logit_list.append(logits.cpu())
    y_true = torch.cat(y_true_list).numpy()
    logits_all = torch.cat(logit_list).numpy()
    prob = 1.0 / (1.0 + np.exp(-logits_all))
    avg_logloss = total_loss / len(loader.dataset)
    auc = roc_auc_score(y_true, prob)
    acc = np.mean((prob >= 0.5) == y_true)
    return avg_logloss, auc, acc

# ===================== 单轮训练函数 =====================
def train_one_epoch(model, loader, opt, bce_criterion, lam):
    model.train()
    bce_sum = 0.0
    reg_sum = 0.0
    total_target_sum = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)
        bce_loss = bce_criterion(logits, yb)
        # 只对二阶emb做L2惩罚： (λ/2)*sum(||V_f||_F^2)
        l2_v = model.get_second_l2()
        reg_term = 0.5 * lam * l2_v
        total_loss = bce_loss + reg_term

        total_loss.backward()
        opt.step()

        bce_sum += bce_loss.item() * len(yb)
        reg_sum += reg_term.item()* len(yb)
        total_target_sum += total_loss.item() * len(yb)

    # 本轮平均
    avg_bce = bce_sum / len(loader.dataset)
    avg_reg = reg_sum / len(loader.dataset)
    avg_target = total_target_sum / len(loader.dataset)
    # 二阶emb整体F范数
    with torch.no_grad():
        total_v_sq = model.get_second_l2()
        v_f_norm = torch.sqrt(total_v_sq).item()
    return avg_bce, avg_reg, avg_target, v_f_norm

# ===================== 主训练循环 =====================
epoch_num = 5
history0 = []
history1 = []

for ep in range(epoch_num):
    # 训练 λ=0
    bce0, reg0, target0, norm0 = train_one_epoch(fm0, train_loader0, opt0, loss_bce, lam=0.0)
    val_ll0, val_auc0, val_acc0 = evaluate(fm0, val_loader, loss_bce)
    history0.append({
        "epoch": ep + 1,
        "train_bce": bce0,
        "train_reg": reg0,
        "train_total": target0,
        "v_norm": norm0,
        "val_logloss": val_ll0,
        "val_auc": val_auc0,
        "val_acc": val_acc0
    })

    # 训练 λ=1e-4
    bce1, reg1, target1, norm1 = train_one_epoch(fm1, train_loader1, opt1, loss_bce, lam=lam)
    val_ll1, val_auc1, val_acc1 = evaluate(fm1, val_loader, loss_bce)
    history1.append({
        "epoch": ep + 1,
        "train_bce": bce1,
        "train_reg": reg1,
        "train_total": target1,
        "v_norm": norm1,
        "val_logloss": val_ll1,
        "val_auc": val_auc1,
        "val_acc": val_acc1
    })

    print(f"==== Epoch {ep + 1}/{epoch_num} ====")
    print(f"λ=0   | train_bce:{bce0:.6f} | reg:{reg0:.6f} | total:{target0:.6f} | v_norm:{norm0:.4f} | val_ll:{val_ll0:.6f} | auc:{val_auc0:.4f} | acc:{val_acc0:.4f}")
    print(f"λ=1e-4| train_bce:{bce1:.6f} | reg:{reg1:.6f} | total:{target1:.6f} | v_norm:{norm1:.4f} | val_ll:{val_ll1:.6f} | auc:{val_auc1:.4f} | acc:{val_acc1:.4f}")

# ===================== 保存权重与历史 =====================
torch.save(fm0.state_dict(), out_lambda0)
torch.save(fm1.state_dict(), out_lambda1e4)

import json
with open(hist_path, "w", encoding="utf-8") as f:
    json.dump({"lambda0": history0, "lambda1e-4": history1}, f, indent=2)
print("历史记录、权重保存完成")

# ===================== 保存实验配置 JSON =====================
config_save = {
    "experiment": "FM second-order embedding L2 regularization comparison",
    "compared_groups": {
        "lambda0": {"lambda": 0.0, "regularize_scope": "none"},
        "lambda1e-4": {"lambda": 1e-4, "regularize_scope": "second_emb_only"}
    },
    "regularize_scope_detail": "只对二阶Embedding权重施加L2惩罚；一阶embedding和bias不参与正则；惩罚项不除以样本数",
    "loss_formula": "J = batch_mean_BCE + (lambda/2) * sum_f ||V_f||_F^2",
    "init": {
        "first_emb": 0.0,
        "bias": 0.0,
        "second_emb": {"dist": "normal", "mean": 0.0, "std": 0.01},
        "global_seed": 42
    },
    "optimizer": {"name": "Adam", "lr": 0.01, "weight_decay": 0},
    "epochs": 5,
    "batch_size": {"train": 512, "val": 2048},
    "loader_seed": 123,
    "data_source": {
        "features": "encoded_features.npz",
        "config": "feature_config.json"
    },
    "field_order": list(feature_config["vocab_sizes"].keys())
}
config_path_out = f"{DATA_DIR}/fm_l2_config.json"
with open(config_path_out, "w", encoding="utf-8") as f:
    json.dump(config_save, f, ensure_ascii=False, indent=2)
print(f"配置保存完成：{config_path_out}")
print("字段顺序：", config_save["field_order"])


# ===================== 重载校验 =====================
def reload_check(path, model):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt)
    ll, auc, acc = evaluate(model, val_loader, loss_bce)
    return ll, auc, acc

fm0_reload = FM(vocab_sizes, latent_dim=8)
fm0_reload.init_spec()
fm1_reload = FM(vocab_sizes, latent_dim=8)
fm1_reload.init_spec()

ll0_r, auc0_r, acc0_r = reload_check(out_lambda0, fm0_reload)
ll1_r, auc1_r, acc1_r = reload_check(out_lambda1e4, fm1_reload)
print("\n==== 重载校验 ====")
print(f"λ=0   reload: logloss={ll0_r:.6f}, auc={auc0_r:.4f}, acc={acc0_r:.4f}")
print(f"λ=1e-4 reload: logloss={ll1_r:.6f}, auc={auc1_r:.4f}, acc={acc1_r:.4f}")

# ===================== 末轮：高置信误判统计（和07误差分析逻辑一致） =====================

def calc_bad_stat(model, loader):
    model.eval()
    xs = []
    ys = []
    logits_all = []
    with torch.no_grad():
        for xb, yb in loader:
            lo = model(xb)
            xs.append(xb)
            ys.append(yb)
            logits_all.append(lo)
    y_true = torch.cat(ys).cpu().numpy()
    logits_all = torch.cat(logits_all).cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logits_all))
    df = pd.DataFrame({"label": y_true, "prob": prob})

    # 先算loss放进df
    loss_arr = - y_true * np.log(prob + 1e-12) - (1 - y_true) * np.log(1 - prob + 1e-12)
    df["loss"] = loss_arr

    bad0 = df[(df["label"] == 0) & (df["prob"] >= 0.9)]
    bad1 = df[(df["label"] == 1) & (df["prob"] <= 0.1)]
    bad_all = pd.concat([bad0, bad1])

    total_loss_all = df["loss"].sum()
    bad_sum_loss = bad_all["loss"].sum()
    bad_ratio = bad_sum_loss / total_loss_all
    return len(bad_all), bad_ratio


cnt0, ratio0 = calc_bad_stat(fm0, val_loader)
cnt1, ratio1 = calc_bad_stat(fm1, val_loader)

print("\n==== 末轮高置信误判统计 ====")
print(f"λ=0   高置信误判数量：{cnt0}，占自身验证损失比例：{ratio0:.2%}")
print(f"λ=1e-4 高置信误判数量：{cnt1}，占自身验证损失比例：{ratio1:.2%}")