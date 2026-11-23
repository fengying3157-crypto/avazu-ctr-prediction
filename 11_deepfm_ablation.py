# =============================================================================
# 脚本：11_deepfm_ablation.py
# 任务：DeepFM二阶交互项消融对照实验
# 研究问题：在已有DNN分支的前提下，显式FM二阶交互项是否改善验证集表现
# 组别定义：
#   full: use_fm2=True  → 一阶 + 二阶FM交互 + DNN
#   ablate: use_fm2=False → 一阶 + DNN（保留共享embedding供DNN使用，移除二阶贡献）
# 公平约束：
#   1. 两组由同一个初始化模型复制而来，初始参数完全一致
#   2. 共享embedding保留，仅关闭二阶项的输出贡献；参数总量相同
#   3. 数据、k=8、DNN结构176→64→32→1、lr=0.001、λ=1e-4、仅共享embedding L2正则
#   4. 独立Generator种子=123；初始化种子=42；训练5轮，固定对比第5轮指标
#   5. 【修复】两组各自绑定自己的共享embedding做L2正则，不再共用同一份参数列表
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from ctr_models import DeepFM


def get_paths():
    """
    路径函数，新建本次消融实验输出文件夹
    """
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent

    data_proc_dir = root_dir / "data" / "avazu" / "processed"
    out_dir = data_proc_dir / "deepfm_ablation"
    out_dir.mkdir(exist_ok=True, parents=True)
    return data_proc_dir, out_dir


def load_data(data_proc_dir):
    """
    加载特征配置、词表、编码数据，和10_train_fm_deepfm保持一致
    返回：vocab_sizes, train_ds, val_ds, feature_cols
    """
    with open(data_proc_dir / "feature_config.json", "r", encoding="utf-8") as f:
        feat_config = json.load(f)
    feature_cols = feat_config["feature_cols"]
    vocab_dict = feat_config["vocab_sizes"]
    vocab_sizes = [int(vocab_dict[col]) for col in feature_cols]

    data_npz = np.load(data_proc_dir / "encoded_features.npz")
    x_train_np = data_npz["x_train"]
    y_train_np = data_npz["y_train"]
    x_val_np = data_npz["x_val"]
    y_val_np = data_npz["y_val"]

    x_train = torch.from_numpy(x_train_np).long()
    y_train = torch.from_numpy(y_train_np).float()
    x_val = torch.from_numpy(x_val_np).long()
    y_val = torch.from_numpy(y_val_np).float()

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    print(f"字段数量：{len(vocab_sizes)}")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    return vocab_sizes, train_ds, val_ds, feature_cols


def build_dataloaders(train_ds, val_ds):
    """
    给full组、ablation组各自一套独立dataloader，generator同种子123
    """
    gen_full = torch.Generator()
    gen_full.manual_seed(123)
    train_loader_full = DataLoader(
        train_ds,
        batch_size=512,
        shuffle=True,
        generator=gen_full
    )
    val_loader_full = DataLoader(val_ds, batch_size=2048, shuffle=False)

    gen_ablate = torch.Generator()
    gen_ablate.manual_seed(123)
    train_loader_ablate = DataLoader(
        train_ds,
        batch_size=512,
        shuffle=True,
        generator=gen_ablate
    )
    val_loader_ablate = DataLoader(val_ds, batch_size=2048, shuffle=False)

    return train_loader_full, val_loader_full, train_loader_ablate, val_loader_ablate


def create_ablation_pair(vocab_sizes):
    """
    只初始化一次use_fm2=True的模型，复制全部参数给消融组use_fm2=False
    校验1：两组所有参数逐张量相等
    校验2：同一输入下 full - ablate ≈ y2（二阶项）
    返回：model_full, model_ablate, reg_full, reg_ablate
    """
    torch.manual_seed(42)
    # 原始基准模型
    base_model = DeepFM(vocab_sizes, k=8, use_fm2=True)
    # 消融模型，开关关闭
    model_ablate = DeepFM(vocab_sizes, k=8, use_fm2=False)

    # 参数完整复制
    with torch.no_grad():
        model_ablate.load_state_dict(base_model.state_dict())

    # 校验1：参数全部相等
    param_ok = True
    for (n1, p1), (n2, p2) in zip(base_model.named_parameters(), model_ablate.named_parameters()):
        if not torch.equal(p1, p2):
            param_ok = False
    print(f"同源参数对齐校验：{param_ok}")

    # 校验2：小批次输入，差值等于二阶项
    x_test = torch.randint(0, 5, (4, len(vocab_sizes)))
    with torch.no_grad():
        out_full = base_model(x_test)
        out_ablate = model_ablate(x_test)
        diff = out_full - out_ablate
        # 手动单独算二阶项
        emb_list = [base_model.emb_shared[i](x_test[:, i]) for i in range(len(vocab_sizes))]
        emb = torch.stack(emb_list, dim=1)
        sum_sq = emb.sum(dim=1) ** 2
        sq_sum = (emb ** 2).sum(dim=1)
        y2_ref = 0.5 * (sum_sq - sq_sum).sum(dim=1)
    diff_ok = torch.allclose(diff, y2_ref, atol=1e-6)
    print(f"输出差值等于二阶项校验：{diff_ok}")

    # 正则参数：两组各自独立的共享embedding列表
    reg_full = list(base_model.emb_shared.parameters())
    reg_ablate = list(model_ablate.emb_shared.parameters())

    return base_model, model_ablate, reg_full, reg_ablate


def train_one_epoch(model, loader, opt, reg_params, lam):
    # 归属检查：确保正则列表里的参数都属于当前模型，防止串用别组参数
    model_param_ids = {id(p) for p in model.parameters()}
    assert all(id(p) in model_param_ids for p in reg_params), (
        "正则参数列表包含其他模型的参数"
    )
    model.train()
    total_bce = 0.0
    total_reg = 0.0
    total_samples = 0
    loss_fn = nn.BCEWithLogitsLoss()
    for xb, yb in loader:
        yb = yb.squeeze(-1)
        logits = model(xb)
        bce = loss_fn(logits, yb)
        # L2正则：lambda/2 * sum(w^2)
        reg = 0.0
        for p in reg_params:
            reg += torch.sum(p ** 2)
        reg = lam * 0.5 * reg
        loss = bce + reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        # 按样本加权统计
        bs = yb.shape[0]
        total_bce += bce.item() * bs
        total_reg += reg.item() * bs
        total_samples += bs
    avg_bce = total_bce / total_samples
    avg_reg = total_reg / total_samples
    avg_total = avg_bce + avg_reg
    return avg_bce, avg_reg, avg_total


def evaluate(model, loader):
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in loader:
            yb = yb.squeeze(-1)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y)
    loss_fn = nn.BCEWithLogitsLoss()
    val_logloss = loss_fn(logits, y_true).item()
    prob = torch.sigmoid(logits)
    val_auc = roc_auc_score(y_true.numpy(), prob.numpy())
    val_acc = ((prob >= 0.5) == y_true.bool()).float().mean().item()
    return val_logloss, val_auc, val_acc


if __name__ == "__main__":
    data_proc_dir, out_dir = get_paths()
    print(f"预处理目录：{data_proc_dir}")
    print(f"消融实验输出目录：{out_dir}")

    vocab_sizes, train_ds, val_ds, feature_cols = load_data(data_proc_dir)
    train_loader_full, val_loader_full, train_loader_ablate, val_loader_ablate = build_dataloaders(train_ds, val_ds)
    print("full训练loader批次数量：", len(train_loader_full))
    print("ablate训练loader批次数量：", len(train_loader_ablate))

    model_full, model_ablate, reg_full, reg_ablate = create_ablation_pair(vocab_sizes)
    x_small, y_small = train_ds[:4]
    out_full = model_full(x_small)
    out_ablate = model_ablate(x_small)
    print(f"full输出shape: {out_full.shape}")
    print(f"ablate输出shape: {out_ablate.shape}")

    # 超参配置
    lr = 0.001
    lam = 1e-4
    epochs = 5
    opt_full = optim.Adam(model_full.parameters(), lr=lr)
    opt_ablate = optim.Adam(model_ablate.parameters(), lr=lr)

    exp_config = {
        "feature_cols": feature_cols,
        "vocab_sizes": vocab_sizes,
        "k": 8,
        "lr": lr,
        "lambda": lam,
        "reg_scope": "only shared embedding, include index 0; linear/bias/mlp no reg",
        "reg_formula": "lambda / 2 * sum(weight ** 2)",
        "optimizer_weight_decay": 0,
        "epochs": epochs,
        "train_bs": 512,
        "val_bs": 2048,
        "seed_data": 123,
        "seed_init": 42,
        "dnn_struct": [176, 64, 32, 1],
        "group_full": {"use_fm2": True, "struct": "lr + fm2 + dnn"},
        "group_ablate": {"use_fm2": False, "struct": "lr + dnn"},
        "compare_rule": "compare epoch 5 fixed, not pick best epoch"
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)

    history = {"full": [], "ablate": []}
    for epoch in range(1, epochs + 1):
        print(f"\n==== Epoch {epoch}/{epochs} ====")
        # full组训练评估
        full_bce, full_reg, full_total = train_one_epoch(model_full, train_loader_full, opt_full, reg_full, lam)
        full_ll, full_auc, full_acc = evaluate(model_full, val_loader_full)
        history["full"].append({
            "epoch": epoch,
            "train_bce": full_bce,
            "train_reg": full_reg,
            "train_total": full_total,
            "val_logloss": full_ll,
            "val_auc": full_auc,
            "val_acc": full_acc
        })
        print(f"Full(+fm2) | train_bce:{full_bce:.6f} | reg:{full_reg:.6f} | total:{full_total:.6f} | val_ll:{full_ll:.6f} | auc:{full_auc:.4f} | acc:{full_acc:.4f}")

        # 消融组训练评估
        ablate_bce, ablate_reg, ablate_total = train_one_epoch(model_ablate, train_loader_ablate, opt_ablate, reg_ablate, lam)
        ablate_ll, ablate_auc, ablate_acc = evaluate(model_ablate, val_loader_ablate)
        history["ablate"].append({
            "epoch": epoch,
            "train_bce": ablate_bce,
            "train_reg": ablate_reg,
            "train_total": ablate_total,
            "val_logloss": ablate_ll,
            "val_auc": ablate_auc,
            "val_acc": ablate_acc
        })
        print(f"Ablate(-fm2)| train_bce:{ablate_bce:.6f} | reg:{ablate_reg:.6f} | total:{ablate_total:.6f} | val_ll:{ablate_ll:.6f} | auc:{ablate_auc:.4f} | acc:{ablate_acc:.4f}")

    # 保存产物
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    torch.save(model_full.state_dict(), out_dir / "full_epoch5.pt")
    torch.save(model_ablate.state_dict(), out_dir / "ablate_epoch5.pt")
    print("\n✅ 配置、历史、权重保存完成")

    # 重载校验：注意构造模型时带上use_fm2标记
    print("\n==== 重载权重校验 ====")
    full_reload = DeepFM(vocab_sizes, k=8, use_fm2=True)
    ablate_reload = DeepFM(vocab_sizes, k=8, use_fm2=False)
    full_reload.load_state_dict(torch.load(out_dir / "full_epoch5.pt", weights_only=True))
    ablate_reload.load_state_dict(torch.load(out_dir / "ablate_epoch5.pt", weights_only=True))

    fr_ll, fr_auc, fr_acc = evaluate(full_reload, val_loader_full)
    ar_ll, ar_auc, ar_acc = evaluate(ablate_reload, val_loader_ablate)
    print(f"Full重载校验  | val_ll:{fr_ll:.6f} | auc:{fr_auc:.4f} | acc:{fr_acc:.4f}")
    print(f"Ablate重载校验| val_ll:{ar_ll:.6f} | auc:{ar_auc:.4f} | acc:{ar_acc:.4f}")
    print("✅ 重载校验完成")
