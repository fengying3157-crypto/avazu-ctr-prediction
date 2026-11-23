from pathlib import Path

import torch
import json
import numpy as np
from ctr_models import LR, FM, DeepFM
import torch.utils.data as data_utils

# ===================== 加载配置、词表、编码数据 =====================
DATA_DIR = Path(__file__).resolve().parent / "data" / "avazu" / "processed"

with open(f"{DATA_DIR}/feature_config.json","r",encoding="utf-8") as f:
    feat_config = json.load(f)
with open(f"{DATA_DIR}/feature_vocab.json","r",encoding="utf-8") as f:
    vocab_info = json.load(f)

feature_cols = feat_config["feature_cols"]
vocab_sizes = [
    feat_config["vocab_sizes"][col]
    for col in feature_cols
]
print(f"num_fields={len(vocab_sizes)}")

data_npz = np.load(f"{DATA_DIR}/encoded_features.npz")
print("npz里面所有key：", list(data_npz.keys()))

X_all = torch.from_numpy(data_npz["x_train"]).long()
# 取小批量样本用于校验
x_1 = X_all[:1]    # batch=1
x_3 = X_all[:3]    # batch=3

# ===================== 1. 形状检查 =====================
deepfm = DeepFM(vocab_sizes, k=8)
out1 = deepfm(x_1)
out3 = deepfm(x_3)
print("\n==== 形状检查 ====")
print(f"input x_1 shape: {x_1.shape}, output shape: {out1.shape}")
print(f"input x_3 shape: {x_3.shape}, output shape: {out3.shape}")
assert out1.shape == torch.Size([1]), "batch=1输出必须是[1]"
assert out3.shape == torch.Size([3]), "batch=3输出必须是[3]"
print("✅ 形状校验通过")

# ===================== 2. 求和检查：logits == part_lr + part_fm2 + part_dnn =====================
logits, plr, pfm2, pdnn = deepfm(x_3, return_parts=True)
sum_parts = plr + pfm2 + pdnn
close_res = torch.allclose(logits, sum_parts, atol=1e-6)
print("\n==== 分项求和检查 ====")
print(f"allclose(logits, sum(parts)): {close_res}")
assert close_res
print("✅ 分项求和校验通过")

# ===================== 3. FM退化检查：DeepFM去掉DNN分支，输出等价FM一阶+二阶计算 =====================
fm = FM(vocab_sizes, latent_dim=8)

with torch.no_grad():
    # 复制一阶权重
    for dst, src in zip(fm.first_embeds, deepfm.emb_lin):
        dst.weight.copy_(src.weight)

    # 复制二阶向量
    for dst, src in zip(fm.second_embeds, deepfm.emb_shared):
        dst.weight.copy_(src.weight)

    fm.bias.copy_(deepfm.bias)

    _, plr, pfm2, _ = deepfm(x_3, return_parts=True)
    deepfm_fm_only = plr + pfm2
    fm_output = fm(x_3)

    same = torch.allclose(
        deepfm_fm_only, fm_output, atol=1e-6
    )
    max_diff = (deepfm_fm_only - fm_output).abs().max().item()

print("与同参数FM输出接近：", same)
print("最大绝对差：", max_diff)
assert same

# ===================== 4. 梯度路径检查 =====================
emb_shared_params = list(deepfm.emb_shared.parameters())
# 4.1 只对二阶贡献反向传播
deepfm.zero_grad()
_, _, pfm2, _ = deepfm(x_3, return_parts=True)
loss_fm2 = pfm2.sum()
loss_fm2.backward()
fm2_grad_has_nonzero = any((p.grad is not None) and torch.any(p.grad != 0) for p in emb_shared_params)
fm2_grad_all_finite = all(
    torch.isfinite(p.grad).all().item()
    for p in emb_shared_params
    if p.grad is not None
)
assert fm2_grad_has_nonzero and fm2_grad_all_finite

# 4.2 只对DNN贡献反向传播
deepfm.zero_grad()
_, _, _, pdnn = deepfm(x_3, return_parts=True)
loss_dnn = pdnn.sum()
loss_dnn.backward()
dnn_grad_has_nonzero = any((p.grad is not None) and torch.any(p.grad != 0) for p in emb_shared_params)
dnn_grad_all_finite = all(
    torch.isfinite(p.grad).all().item()
    for p in emb_shared_params
    if p.grad is not None
)
assert dnn_grad_has_nonzero and dnn_grad_all_finite

print("\n==== 梯度路径检查 ====")
print(f"二阶路径：共享Emb存在非零梯度={fm2_grad_has_nonzero}, 梯度全部有限={fm2_grad_all_finite}")
print(f"DNN路径：共享Emb存在非零梯度={dnn_grad_has_nonzero}, 梯度全部有限={dnn_grad_all_finite}")
print("✅ 梯度路径校验通过")

print("\n======= 全部四项检查完成 =======")
print("一句话说明共享参数：DeepFM中emb_shared只查表一次，取出的E张量同时供给FM二阶交互计算与DNN输入，两组分支共用同一套隐向量权重。")

# 第 1 步：固定抽取 32 条样本，确认正负两类
# ===================== 任务五：固定32条样本训练 =====================
torch.manual_seed(42)
y_all = torch.from_numpy(data_npz["y_train"]).float()

# 固定抽取32条，一次确定，反复使用
n_samp = 32
idx = torch.randperm(len(X_all))[:n_samp]
X_fix = X_all[idx]
y_fix = y_all[idx]

# 确认正负两类都出现
n_pos = (y_fix == 1).sum().item()
n_neg = (y_fix == 0).sum().item()
print(f"\n固定32条样本：正样本={n_pos}, 负样本={n_neg}")
assert n_pos > 0 and n_neg > 0, "样本必须同时含正负两类"
print("✅ 正负两类齐全")

# 新建DeepFM，无额外正则，CPU
deepfm_train = DeepFM(vocab_sizes, k=8)
loss_fn = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(deepfm_train.parameters(), lr=0.001)

# 把标签拉成一维 [32]
y_fix_1d = y_fix.squeeze(-1)
# ---------- 初始状态：更新0次 ----------
with torch.no_grad():
    logits_init = deepfm_train(X_fix)
    loss_init = loss_fn(logits_init, y_fix_1d).item()
    prob_init = torch.sigmoid(logits_init)
    pred_init = (prob_init >= 0.5).float()
    acc_init = (pred_init == y_fix_1d).float().mean().item()

print(f"\n更新0次(初始) | BCE loss: {loss_init:.6f} | Acc: {acc_init:.6f}")

# ---------- 训练循环，共200步 ----------
total_steps = 200
for step in range(1,total_steps +1):
    opt.zero_grad()
    logits = deepfm_train(X_fix)
    loss = loss_fn(logits,y_fix_1d)
    loss.backward()
    opt.step()

    if step % 50 == 0:
        with torch.no_grad():
            logits_eval = deepfm_train(X_fix)
            loss_eval = loss_fn(logits_eval, y_fix_1d).item()

            prob_eval = torch.sigmoid(logits_eval)
            pred_eval = (prob_eval >= 0.5).float()
            acc_eval = (
                (pred_eval == y_fix_1d).float().mean().item()
            )

        print(
            f"更新{step}次 | BCE loss: {loss_eval:.6f} "
            f"| Acc: {acc_eval:.6f}"
        )