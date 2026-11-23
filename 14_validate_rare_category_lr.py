# =============================================================================
# 脚本：14_validate_rare_category_lr.py
# 任务：验证集误差切片 + 概率校准（沿用 13 脚本训练好的两组第5轮 LR）
# 说明：
# 1. 两组模型各自用自己的 vocab_sizes / 验证编码 / lr_epoch5.pt
# 2. 逐样本损失用 BCEWithLogitsLoss(reduction="none")
# 3. 先复现全验证集 LogLoss，再进入切片与分桶
# =============================================================================
import os
import json
import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ctr_models import LR
from collections import Counter

SCRIPT_PATH = os.path.abspath(__file__)
PROJ_ROOT = os.path.dirname(SCRIPT_PATH)

EXP_ROOT = os.path.join(PROJ_ROOT, "data", "avazu", "processed", "rare_category_compare")
DIAG_DIR = os.path.join(EXP_ROOT, "diagnostics")
GROUP_DIRS = {
    2: os.path.join(EXP_ROOT, "min_count_2"),
    5: os.path.join(EXP_ROOT, "min_count_5"),
}

# 从12脚本复制过来的路径
TRAIN_CSV = os.path.join(PROJ_ROOT, "data", "avazu", "processed", "train.csv.gz")
VAL_CSV = os.path.join(PROJ_ROOT, "data", "avazu", "processed", "val.csv.gz")

N_VAL = 9571
def load_group(group_dir):
    """读取配置、词表、验证张量和第5轮权重"""
    with open(os.path.join(group_dir, "feature_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feature_cols = cfg["feature_cols"]
    vocab_sizes = [int(cfg["vocab_sizes"][c]) for c in feature_cols]

    with open(os.path.join(group_dir, "feature_vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    npz = np.load(os.path.join(group_dir, "encoded_features.npz"))
    x_val = torch.from_numpy(npz["x_val"]).long()
    y_val = torch.from_numpy(npz["y_val"]).float()

    model = LR(vocab_sizes)
    model.load_state_dict(torch.load(os.path.join(group_dir, "lr_epoch5.pt"), map_location="cpu"))
    model.eval()
    return feature_cols, vocab_sizes, vocab, x_val, y_val, model

@torch.no_grad()
def get_logits(model, x, batch_size=2048):
    outs = []
    for i in range(0, len(x), batch_size):
        outs.append(model(x[i:i + batch_size]))
    return torch.cat(outs)

data_2 = load_group(GROUP_DIRS[2])
data_5 = load_group(GROUP_DIRS[5])

logits_2 = get_logits(data_2[5], data_2[3])
logits_5 = get_logits(data_5[5], data_5[3])

criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
loss_2 = criterion(logits_2, data_2[4])
loss_5 = criterion(logits_5, data_5[4])

print("==== 逐样本损失与LogLoss复现 ====")
print(f"min_count=2 全验证集LogLoss：{loss_2.mean().item():.6f}（应复现 0.394597）")
print(f"min_count=5 全验证集LogLoss：{loss_5.mean().item():.6f}（应复现 0.393683）")
assert abs(loss_2.mean().item() - 0.394597) < 1e-5, "min_count=2 LogLoss 复现失败"
assert abs(loss_5.mean().item() - 0.393683) < 1e-5, "min_count=5 LogLoss 复现失败"
print("✅ 两组全验证集LogLoss复现成功")

# 校验A：两组标签与原始CSV逐行一致
raw_val_df = pd.read_csv(gzip.open(VAL_CSV, "rt", encoding="utf-8"), dtype=str)
assert len(raw_val_df) == N_VAL, f"验证CSV行数不对：{len(raw_val_df)}"
raw_y_val = torch.from_numpy(raw_val_df["click"].astype(np.float32).values)
assert torch.equal(raw_y_val, data_2[4]), "min_count=2 标签与CSV不一致"
assert torch.equal(raw_y_val, data_5[4]), "min_count=5 标签与CSV不一致"
assert torch.equal(data_2[4], data_5[4]), "两组标签不一致"
print("✅ 两组验证标签与原始CSV逐行一致")

# 校验B：两组标签形状与模型输出对齐
assert logits_2.shape == (N_VAL,) and logits_5.shape == (N_VAL,)
print("✅ logits 形状均为 (9571,)")

# 校验C：用保存的词表重编码 device_ip，应与NPZ这一列相同
ip_idx = data_2[0].index("device_ip")
assert ip_idx == data_5[0].index("device_ip"), "两组 device_ip 列位置不一致"

for mc, (feature_cols, vocab_sizes, vocab, x_val, y_val, model) in [(2, data_2), (5, data_5)]:
    ip_map = {k: int(v) for k, v in vocab["device_ip"].items()}
    # 与 12 脚本完全一致的编码规则：缺失/低频/未见 → 0
    re_ip = raw_val_df["device_ip"].map(ip_map).fillna(0).astype(np.int64).values
    npz_ip = x_val[:, ip_idx].numpy()
    diff_cnt = int((re_ip != npz_ip).sum())
    assert diff_cnt == 0, f"min_count={mc} device_ip 重编码与NPZ不一致，差异{diff_cnt}条"
    print(f"✅ min_count={mc} device_ip 重编码与NPZ完全一致，零率 {float((npz_ip == 0).mean()):.6f}")

# ===================== 第二步：device_ip频次切片、分组指标 =====================
# 读取训练集，统计device_ip在训练集中的频次
raw_train_df = pd.read_csv(gzip.open(TRAIN_CSV, "rt", encoding="utf-8"), dtype=str)
train_ip_counter = Counter(raw_train_df["device_ip"].dropna())

# 把验证集原始df带上损失、logits，方便分组
val_df = raw_val_df.copy()
val_df["loss_2"] = loss_2.detach().cpu().numpy()
val_df["loss_5"] = loss_5.detach().cpu().numpy()
val_df["click"] = val_df["click"].astype(float)

# 给每一行打上分组标签
def get_group_label(ip_str):
    if pd.isna(ip_str):
        return "缺失"
    cnt = train_ip_counter.get(ip_str, 0)
    if cnt == 0:
        return "严格未见"
    elif cnt == 1:
        return "低频"
    elif 2 <= cnt <= 4:
        return "编码发生变化"
    else:
        return "两组均保留"

val_df["group"] = val_df["device_ip"].apply(get_group_label)

# 分组汇总
group_rows = []
group_list = ["缺失", "严格未见", "低频", "编码发生变化", "两组均保留"]
for g in group_list:
    sub = val_df[val_df["group"] == g]
    n_g = len(sub)
    if n_g == 0:
        avg_loss2 = np.nan
        avg_loss5 = np.nan
        ctr = np.nan
    else:
        avg_loss2 = sub["loss_2"].mean()
        avg_loss5 = sub["loss_5"].mean()
        ctr = sub["click"].mean()
    delta_g = avg_loss5 - avg_loss2
    contrib_g = (n_g / N_VAL) * delta_g
    group_rows.append({
        "group": g,
        "n_g": n_g,
        "actual_ctr": ctr,
        "avg_loss_2": avg_loss2,
        "avg_loss_5": avg_loss5,
        "delta_g": delta_g,
        "contrib_g": contrib_g
    })

slice_df = pd.DataFrame(group_rows)
print("\n==== device_ip频次切片指标 ====")
print(slice_df.to_string(index=False, float_format="%.6f"))

# 校验：样本总数之和=9571；贡献总和≈-0.000914
sum_n = slice_df["n_g"].sum()
sum_contrib = slice_df["contrib_g"].sum()
print(f"\n分组样本总数：{sum_n}，期望 {N_VAL}")
print(f"贡献总和：{sum_contrib:.6f}，期望≈-0.000914")
assert sum_n == N_VAL, "分组样本加总不等于9571"
assert abs(sum_contrib - (-0.000914)) < 1e-5, "贡献总和和全局差值对不上"
print("✅ 切片分组校验通过")


# ===================== 第三步：概率校准分桶 =====================
# 计算预测概率
p_2 = torch.sigmoid(logits_2).cpu().numpy()
p_5 = torch.sigmoid(logits_5).cpu().numpy()
y_true = val_df["click"].values

# 分桶协议：内部边界分桶 + 最后一个桶闭区间
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bin_labels = [f"[{bins[i]:.1f},{bins[i+1]:.1f})" for i in range(len(bins)-1)]
bin_labels[-1] = "[0.9,1.0]"            # 最后一桶闭区间，p=1.0 也能落桶
internal_edges = bins[1:-1]              # 9 个内部边界：0.1 ~ 0.9

calib_rows = []
for mid, model_name, p_arr in zip(["2", "5"], ["min_count=2", "min_count=5"], [p_2, p_5]):
    # side="right"：边界值归入右侧桶，即左闭右开
    bin_idx = np.searchsorted(internal_edges, p_arr, side="right")
    assert bin_idx.min() >= 0 and bin_idx.max() <= 9, "分桶索引越界"
    print(f"{model_name} 预测概率范围：min={p_arr.min():.6f}, max={p_arr.max():.6f}")
    for b_idx, b_label in enumerate(bin_labels):
        mask = bin_idx == b_idx
        n_bin = int(mask.sum())
        if n_bin == 0:
            mean_pred = np.nan
            mean_true = np.nan
            diff = np.nan
        else:
            mean_pred = float(np.mean(p_arr[mask]))
            mean_true = float(np.mean(y_true[mask]))
            diff = mean_pred - mean_true
        calib_rows.append({
            "model": model_name,
            "bin_range": b_label,
            "n_bin": n_bin,
            "mean_pred_prob": mean_pred,
            "actual_ctr": mean_true,
            "pred_minus_true": diff
        })


calib_df = pd.DataFrame(calib_rows)
print("\n==== 概率校准分桶指标 ====")
print(calib_df.to_string(index=False, float_format="%.6f"))

# 样本量充足口径：每桶 ≥ 100 条
MIN_BIN_N = 100
adequate = calib_df[calib_df["n_bin"] >= MIN_BIN_N]
print(f"\n==== 样本量充足桶（n≥{MIN_BIN_N}）====\n{adequate.to_string(index=False, float_format='%.6f')}")

for m in ["min_count=2", "min_count=5"]:
    assert calib_df[calib_df["model"] == m]["n_bin"].sum() == N_VAL, f"{m} 分桶合计不对"
print("✅ 分桶计数与协议校验通过")


# ===================== 第四步：保存两个诊断csv =====================
os.makedirs(DIAG_DIR, exist_ok=True)
slice_df.to_csv(os.path.join(DIAG_DIR, "slice_metrics.csv"), index=False, encoding="utf-8-sig")
calib_df.to_csv(os.path.join(DIAG_DIR, "calibration_bins.csv"), index=False, encoding="utf-8-sig")
print(f"\n✅ 诊断表格已保存到 {DIAG_DIR}")
print("  - slice_metrics.csv")
print("  - calibration_bins.csv")
