# =============================================================================
# 脚本：15_final_holdout_evaluation.py
# 任务：最终留出集评估（FM 主模型 / 阈值5 LR / 完整DeepFM / 常数基线）
# 说明：
# 1. 先在验证集复现四组已知指标，确认编码与评估链路正确
# 2. 再在 2014-10-30 留出集上正式评估一次
# 3. 四组共用同一批记录、同一套评估函数
# =============================================================================
import os
import json
import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from ctr_models import LR, FM, DeepFM

SCRIPT_PATH = os.path.abspath(__file__)
PROJ_ROOT = os.path.dirname(SCRIPT_PATH)

PROC = os.path.join(PROJ_ROOT, "data", "avazu", "processed")
FM_DIR = os.path.join(PROC, "fm_deepfm_compare")
RARE5_DIR = os.path.join(PROC, "rare_category_compare", "min_count_5")

TRAIN_CSV = os.path.join(PROC, "train.csv.gz")
VAL_CSV = os.path.join(PROC, "val.csv.gz")
HOLDOUT_CSV = os.path.join(PROC, "holdout.csv.gz")

# 预期值：第10个实验 / 阈值实验已记录的第5轮验证指标
EXPECT_VAL = {
    "FM": 0.389312,
    "LR_mc5": 0.393683,
    "DeepFM": 0.405970,
    "Const": 0.433471,
}

# 常数基线的AUC与Accuracy预期（AUC恒为0.5；Acc@0.5等于负类占比）
EXPECT_CONST = {
    "val": {"AUC": 0.5, "Acc@0.5": 0.844321},
    "holdout": {"AUC": 0.5, "Acc@0.5": 0.834584},
}

def load_cfg(cfg_path, vocab_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feature_cols = cfg["feature_cols"]
    vocab_sizes = [int(cfg["vocab_sizes"][c]) for c in feature_cols]
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return feature_cols, vocab_sizes, vocab

def read_csv(path):
    df = pd.read_csv(gzip.open(path, "rt", encoding="utf-8"), dtype=str)
    # 小时规则与预处理一致：hour 为 YYMMDDHH，取最后两位
    df["hour_of_day"] = df["hour"].str[-2:]
    return df

def encode(df, feature_cols, vocab):
    cols = []
    for col in feature_cols:
        m = {k: int(v) for k, v in vocab[col].items()}
        cols.append(df[col].map(m).fillna(0).astype(np.int64).values.reshape(-1, 1))
    return torch.from_numpy(np.concatenate(cols, axis=1))

def check_range(x, vocab_sizes, feature_cols, tag):
    for i, col in enumerate(feature_cols):
        mx = int(x[:, i].max())
        assert mx < vocab_sizes[i], f"{tag} 字段 {col} 越界：max={mx}, vocab_size={vocab_sizes[i]}"
    print(f"✅ {tag} 22字段编码范围合法")

def predict_logits(model, x, bs=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            outs.append(model(x[i:i + bs]))   # 三个模型输出均为 [B]
    return torch.cat(outs)

def metrics_from_logits(logits, y):
    ll = F.binary_cross_entropy_with_logits(logits, y).item()
    prob = torch.sigmoid(logits)
    auc = roc_auc_score(y.numpy(), prob.numpy())
    acc = (((prob >= 0.5).float()) == y).float().mean().item()
    return ll, auc, acc, prob


def metrics_from_prob(prob, y):
    """给常数基线用：输入已经是概率，不经过logits"""
    ll = F.binary_cross_entropy(prob, y).item()
    auc = roc_auc_score(y.numpy(), prob.numpy())
    acc = (((prob >= 0.5).float()) == y).float().mean().item()
    return ll, auc, acc, prob

# -------------------------- 资源加载 --------------------------
fc2, vs2, vocab2 = load_cfg(os.path.join(PROC, "feature_config.json"),
                            os.path.join(PROC, "feature_vocab.json"))
fc5, vs5, vocab5 = load_cfg(os.path.join(RARE5_DIR, "feature_config.json"),
                            os.path.join(RARE5_DIR, "feature_vocab.json"))
assert fc2 == fc5 == fc5, "两组字段顺序不一致"
print(f"阈值2词表合计 {sum(vs2)}，阈值5词表合计 {sum(vs5)}")

fm = FM(vs2, latent_dim=8)
fm.load_state_dict(torch.load(os.path.join(FM_DIR, "fm_epoch5.pt"), map_location="cpu", weights_only=True))
deepfm = DeepFM(vs2, k=8, use_fm2=True)
deepfm.load_state_dict(torch.load(os.path.join(FM_DIR, "deepfm_epoch5.pt"), map_location="cpu", weights_only=True))
lr5 = LR(vs5)
lr5.load_state_dict(torch.load(os.path.join(RARE5_DIR, "lr_epoch5.pt"), map_location="cpu", weights_only=True))
print("✅ 三组权重加载完成（FM/DeepFM 用阈值2词表，LR 用阈值5词表）")

# -------------------------- 第二步：验证集链路检查 --------------------------
val_df = read_csv(VAL_CSV)
assert len(val_df) == 9571, f"验证集行数不对：{len(val_df)}"

y_np = val_df["click"].astype(np.float32).values.copy()
y_val = torch.from_numpy(y_np)
x_val_2 = encode(val_df, fc2, vocab2)
x_val_5 = encode(val_df, fc5, vocab5)

check_range(x_val_2, vs2, fc2, "阈值2")
check_range(x_val_5, vs5, fc5, "阈值5")

# 最强的一条检查：重新编码结果必须与预处理阶段保存的 x_val 逐元素相同（同时校验顺序与编码）
npz_root = np.load(os.path.join(PROC, "encoded_features.npz"))
npz_5 = np.load(os.path.join(RARE5_DIR, "encoded_features.npz"))
assert np.array_equal(x_val_2.numpy(), npz_root["x_val"]), "阈值2 重编码与已存 x_val 不一致"
assert np.array_equal(x_val_5.numpy(), npz_5["x_val"]), "阈值5 重编码与已存 x_val 不一致"
assert np.array_equal(y_val.numpy(), npz_root["y_val"].squeeze()), "验证标签与已存 y_val 不一致"
print("✅ 验证集重编码与已存 NPZ 逐元素一致（记录顺序、字段顺序、编码规则全部对齐）")

# 常数基线：训练集点击率，用训练标签重新计算
train_df = read_csv(TRAIN_CSV)
p_train = float(train_df["click"].astype(np.float32).mean())
print(f"训练集点击率（常数基线概率）：{p_train:.6f}")

# 四组在验证集上的指标
logits_fm = predict_logits(fm, x_val_2)
logits_deepfm = predict_logits(deepfm, x_val_2)
logits_lr5 = predict_logits(lr5, x_val_5)

# 改成正确顺序
const_prob = torch.full_like(y_val, p_train)
ll_const = F.binary_cross_entropy(const_prob, y_val).item()

rows = []
for name, logits in [("FM", logits_fm), ("LR_mc5", logits_lr5), ("DeepFM", logits_deepfm)]:
    ll, auc, acc, _ = metrics_from_logits(logits, y_val)
    rows.append((name, ll, auc, acc))

print("\n==== 验证集指标复现（四组） ====")
print(f"{'方案':<10}{'LogLoss':>10}{'AUC':>10}{'Acc@0.5':>10}{'预期':>10}")
for name, ll, auc, acc in rows:
    print(f"{name:<10}{ll:>10.6f}{auc:>10.6f}{acc:>10.6f}{EXPECT_VAL[name]:>10.6f}")
print(f"{'Const':<10}{ll_const:>10.6f}{'-':>10}{'-':>10}{EXPECT_VAL['Const']:>10.6f}")

assert abs(rows[0][1] - EXPECT_VAL["FM"]) < 1e-5, "FM 验证 LogLoss 未复现"
assert abs(rows[1][1] - EXPECT_VAL["LR_mc5"]) < 1e-5, "阈值5 LR 验证 LogLoss 未复现"
assert abs(rows[2][1] - EXPECT_VAL["DeepFM"]) < 1e-5, "DeepFM 验证 LogLoss 未复现"
assert abs(ll_const - EXPECT_VAL["Const"]) < 1e-5, "常数基线验证 LogLoss 未复现"
print("✅ 四组验证集指标全部复现")

# 收集验证指标，供metrics.json写入
val_metric_dict = {}
for name, ll, auc, acc in rows:
    val_metric_dict[name] = {"LogLoss": ll, "AUC": auc, "Acc@0.5": acc}
# 常数基线固定指标
val_metric_dict["Const"] = {
    "LogLoss": ll_const,
    "AUC": 0.500000,
    "Acc@0.5": 0.844321
}


# -------------------------- 第三步：留出集正式评估 --------------------------
print("\n==== 开始留出集 holdout 评估 ====")
holdout_df = read_csv(HOLDOUT_CSV)
# 校验样本数量
assert len(holdout_df) == 10525, f"留出集数量异常，实际：{len(holdout_df)}，预期：10525"
# 校验日期：hour字段前缀是141030
holdout_hour_raw = holdout_df["hour"].values
assert all(h.startswith("141030") for h in holdout_hour_raw), "留出集包含非10-30日期样本"
print("✅ 留出集：10525条，全部为2014-10-30")

# 校验ID：留出集ID和训练、验证ID没有重叠
train_id_set = set(train_df["id"].values)
val_id_set = set(val_df["id"].values)
holdout_id_set = set(holdout_df["id"].values)
assert holdout_id_set.isdisjoint(train_id_set), "ERROR：留出集ID 和训练集ID 重叠！"
assert holdout_id_set.isdisjoint(val_id_set), "ERROR：留出集ID 和验证集ID 重叠！"
print("✅ 留出集ID与训练、验证ID无交集")


# 提取标签、ID
holdout_id = holdout_df["id"].values
y_holdout_np = holdout_df["click"].astype(np.float32).values.copy()
y_holdout = torch.from_numpy(y_holdout_np)

# 两套编码
x_holdout_2 = encode(holdout_df, fc2, vocab2)
x_holdout_5 = encode(holdout_df, fc5, vocab5)
check_range(x_holdout_2, vs2, fc2, "留出集-阈值2")
check_range(x_holdout_5, vs5, fc5, "留出集-阈值5")

# 四组预测
logits_fm_h = predict_logits(fm, x_holdout_2)
logits_deepfm_h = predict_logits(deepfm, x_holdout_2)
logits_lr5_h = predict_logits(lr5, x_holdout_5)
const_prob_h = torch.full_like(y_holdout, p_train)

# 计算指标
ll_fm_h, auc_fm_h, acc_fm_h, prob_fm = metrics_from_logits(logits_fm_h, y_holdout)
ll_lr5_h, auc_lr5_h, acc_lr5_h, prob_lr5 = metrics_from_logits(logits_lr5_h, y_holdout)
ll_deepfm_h, auc_deepfm_h, acc_deepfm_h, prob_deepfm = metrics_from_logits(logits_deepfm_h, y_holdout)
ll_const_h = F.binary_cross_entropy(const_prob_h, y_holdout).item()

# 打印留出集结果
print("\n==== 留出集指标（holdout 2014-10-30） ====")
print(f"{'方案':<10}{'LogLoss':>10}{'AUC':>10}{'Acc@0.5':>10}")
print(f"{'FM':<10}{ll_fm_h:>10.6f}{auc_fm_h:>10.6f}{acc_fm_h:>10.6f}")
print(f"{'LR_mc5':<10}{ll_lr5_h:>10.6f}{auc_lr5_h:>10.6f}{acc_lr5_h:>10.6f}")
print(f"{'DeepFM':<10}{ll_deepfm_h:>10.6f}{auc_deepfm_h:>10.6f}{acc_deepfm_h:>10.6f}")
print(f"{'Const':<10}{ll_const_h:>10.6f}{'-':>10}{'-':>10}")

# -------------------------- 第四步：保存输出文件 --------------------------
out_dir = os.path.join(PROC, "final_holdout_eval")
os.makedirs(out_dir, exist_ok=True)

# config.json
config_info = {
    "main_model": "FM",
    "main_metric": "LogLoss",
    "threshold_acc": 0.5,
    "holdout_date": "2014-10-30",
    "schemes": {
        "FM": {
            "weight_path": "fm_deepfm_compare/fm_epoch5.pt",
            "vocab_set": "min_count_2",
            "latent_dim": 8
        },
        "LR_mc5": {
            "weight_path": "rare_category_compare/min_count_5/lr_epoch5.pt",
            "vocab_set": "min_count_5"
        },
        "DeepFM": {
            "weight_path": "fm_deepfm_compare/deepfm_epoch5.pt",
            "vocab_set": "min_count_2",
            "k": 8,
            "use_fm2": True
        },
        "Const": {
            "train_click_rate": p_train
        }
    }
}
with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config_info, f, indent=2, ensure_ascii=False)

# metrics.json
metrics_info = {
    "validation": val_metric_dict,

    "holdout": {
        "FM": {"LogLoss": ll_fm_h, "AUC": auc_fm_h, "Acc@0.5": acc_fm_h},
        "LR_mc5": {"LogLoss": ll_lr5_h, "AUC": auc_lr5_h, "Acc@0.5": acc_lr5_h},
        "DeepFM": {"LogLoss": ll_deepfm_h, "AUC": auc_deepfm_h, "Acc@0.5": acc_deepfm_h},
        "Const": {
    "LogLoss": ll_const_h,
    "AUC": 0.500000,
    "Acc@0.5": 0.834584
}

    },
    "holdout_sample_num": 10525,
    "holdout_click_rate": float(y_holdout.mean().item())
}
with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_info, f, indent=2, ensure_ascii=False)

# predictions.csv
pred_df = pd.DataFrame({
    "id": holdout_id,
    "click_true": y_holdout_np,
    "prob_FM": prob_fm.numpy(),
    "prob_LR_mc5": prob_lr5.numpy(),
    "prob_DeepFM": prob_deepfm.numpy(),
    "prob_Const": np.full_like(prob_fm.numpy(), p_train)
})
pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

print(f"\n✅ 全部结果保存至：{out_dir}")
print("  - config.json")
print("  - metrics.json")
print("  - predictions.csv")