import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path

# 导入我们刚刚单独写好的模型类
from ctr_models import LR, FM

# ========== 路径配置 ==========
BASE_DIR = Path(__file__).parent
PROC_DIR = BASE_DIR / "data" / "avazu" / "processed"

# 权重文件
lr_ckpt_path = PROC_DIR / "lr_compare_model.pt"
fm_ckpt_path = PROC_DIR / "fm_compare_model.pt"
# 特征配置
config_path = PROC_DIR / "feature_config.json"
# 编码后的验证集
npz_path = PROC_DIR / "encoded_features.npz"
# 输出误差表
out_csv = PROC_DIR / "lr_fm_val_errors.csv"

# ========== 加载配置和验证数据 ==========
with open(config_path, "r", encoding="utf-8") as f:
    feature_config = json.load(f)
vocab_sizes = [int(v) for v in feature_config["vocab_sizes"].values()]

data = np.load(npz_path)
x_val = torch.from_numpy(data["x_val"]).long()
y_val = torch.from_numpy(data["y_val"]).float()

print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)

# ========== 加载模型权重 ==========
fm_k = 8
lr_model = LR(vocab_sizes)
fm_model = FM(vocab_sizes, latent_dim=fm_k)

lr_model.load_state_dict(torch.load(lr_ckpt_path, map_location="cpu"))
fm_model.load_state_dict(torch.load(fm_ckpt_path, map_location="cpu"))

lr_model.eval()
fm_model.eval()
#把权重读进来、设为评估模式，还没有前向计算
# 做前向推理、算逐样本 loss

# ========== 前向推理 + 逐样本损失 ==========
loss_fn = nn.BCEWithLogitsLoss(reduction="none")
with torch.no_grad():
    logits_lr = lr_model(x_val)
    logits_fm = fm_model(x_val)

    #概率sigmod
    prob_lr = torch.sigmoid(logits_lr)
    prob_fm = torch.sigmoid(logits_fm)

    # 把标签从 [N,1] 压成 [N]，和logits对齐
    y_val = y_val.squeeze(-1)

    loss_lr = loss_fn(logits_lr,y_val)
    loss_fm = loss_fn(logits_fm,y_val)

loss_diff = loss_fm - loss_lr

# 校验：平均损失，核对和之前指标是否匹配
mean_loss_lr = loss_lr.mean().item()
mean_loss_fm = loss_fm.mean().item()
mean_diff = loss_diff.mean().item()

print(f'LR mean loss: {mean_loss_lr:.6f}')
print(f'FM mean loss: {mean_loss_fm:.6f}')
print(f'Mean diff(FM-LR): {mean_diff:.6f}')

# ========== 构造结果表并保存csv ==========
df = pd.DataFrame({
    'row_idx':np.arange(len(y_val)),
    'label':y_val.cpu().numpy(),
    'lr_prob':prob_lr.cpu().numpy(),
    'fm_prob':prob_fm.cpu().numpy(),
    'lr_loss':loss_lr.cpu().numpy(),
    'fm_loss':loss_fm.cpu().numpy(),
    'fm_minus_lr_loss':loss_diff.cpu().numpy()
})
df.to_csv(out_csv,index=False,encoding='utf-8-sig')
print(f'csv保存成功：{out_csv}')

# 分组统计 y=0、y=1 的样本数、组内平均损失、每组对总差值的贡献。
# ========== 按 y=0 / y=1 分组统计 ==========
df = pd.read_csv(out_csv)
total_n = len(df)

group0 = df[df['label'] ==0]
group1 = df[df['label'] ==1]

sum_diff0 = group0['fm_minus_lr_loss'].sum()
sum_diff1 = group1['fm_minus_lr_loss'].sum()
contrib0 = sum_diff0 / total_n
contrib1 = sum_diff1 / total_n

print("==== 分组统计 ====")
print(f"负样本 y=0：数量 {len(group0)}，LR平均loss {group0['lr_loss'].mean():.6f}，FM平均loss {group0['fm_loss'].mean():.6f}，贡献 {contrib0:.6f}")
print(f"正样本 y=1：数量 {len(group1)}，LR平均loss {group1['lr_loss'].mean():.6f}，FM平均loss {group1['fm_loss'].mean():.6f}，贡献 {contrib1:.6f}")
print(f"贡献合计：{contrib0 + contrib1:.6f}（应接近0.117756）")

# ========== 自信错误样本统计 ==========
# LR自信错样本
lr_bad0 = df[(df["label"] == 0) & (df["lr_prob"] >= 0.9)]
lr_bad1 = df[(df["label"] == 1) & (df["lr_prob"] <= 0.1)]
# FM自信错样本
fm_bad0 = df[(df["label"] == 0) & (df["fm_prob"] >= 0.9)]
fm_bad1 = df[(df["label"] == 1) & (df["fm_prob"] <= 0.1)]

print("==== 自信错误样本统计 ====")
print(f"LR | y=0且prob≥0.9：数量 {len(lr_bad0)}，这批样本FM-LR损失差之和：{lr_bad0['fm_minus_lr_loss'].sum():.6f}")
print(f"LR | y=1且prob≤0.1：数量 {len(lr_bad1)}，这批样本FM-LR损失差之和：{lr_bad1['fm_minus_lr_loss'].sum():.6f}")
print(f"FM | y=0且prob≥0.9：数量 {len(fm_bad0)}，这批样本FM-LR损失差之和：{fm_bad0['fm_minus_lr_loss'].sum():.6f}")
print(f"FM | y=1且prob≤0.1：数量 {len(fm_bad1)}，这批样本FM-LR损失差之和：{fm_bad1['fm_minus_lr_loss'].sum():.6f}")

# 按 fm_minus_lr_loss 降序，取前10条
top10 = df.sort_values("fm_minus_lr_loss", ascending=False).head(10)
print("\n==== FM-LR损失差最大前10条样本 ====")
print(top10[["row_idx","label","lr_prob","fm_prob","lr_loss","fm_loss","fm_minus_lr_loss"]])


# ========== 新增：统计高置信误判样本占模型自身总损失的比例 ==========
lr_bad_all = pd.concat([lr_bad0, lr_bad1])
fm_bad_all = pd.concat([fm_bad0, fm_bad1])

for name, group in [
    ("lr", lr_bad_all),
    ("fm", fm_bad_all),
]:
    loss_col = f"{name}_loss"
    own_loss = group[loss_col].sum()
    loss_share = own_loss / df[loss_col].sum()
    print(f"{name}：高置信误判{len(group)}条，"
          f"自身损失和={own_loss:.6f}，"
          f"占自身总损失={loss_share:.2%}")
