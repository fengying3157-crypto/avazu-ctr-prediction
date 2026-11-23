# =============================================================================
# 脚本：10_train_fm_deepfm.py
# 任务：FM 与 DeepFM 控制变量对照实验（Avazu点击率预估）
# 实验核心目标：
# 固定所有训练条件、对齐初始化参数，仅差别为 DeepFM 额外拥有DNN深层分支
# 以此判断：在FM基础上增加DNN能否改善验证集效果
# 控制要点：
# 1. DeepFM一阶emb、共享二阶emb、bias复制给FM，初始共有参数完全一致
# 2. FM、DeepFM各自独立DataLoader随机生成器，种子同为123，保证批次顺序一致
# 3. 共用同一套训练、验证函数；分开记录BCE、正则损失、总目标损失
# 4. 一共训练5轮；保存实验配置、每轮指标、第5轮权重，并且做加载复现校验
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
# 导入ctr_models里面已经实现好的模型
from ctr_models import FM, DeepFM

def get_paths():
    """
    自动定位项目路径，避免硬编码绝对路径
    返回：
        data_proc_dir：预处理好的数据集目录
        out_dir：本次对照实验结果输出目录
    """
    # 获取当前脚本的绝对路径
    script_path = Path(__file__).resolve()
    # avazu文件夹
    # 项目根目录
    root_dir = script_path.parent

    # 存放npz、feature_config.json等预处理文件
    data_proc_dir = root_dir / "data" / "avazu" / "processed"
    # 新建专门文件夹存放本次实验产出
    out_dir = data_proc_dir / "fm_deepfm_compare"
    # 文件夹不存在就创建，存在不会报错
    out_dir.mkdir(exist_ok=True, parents=True)
    return data_proc_dir, out_dir



def load_data(data_proc_dir):
    """
    加载特征配置、词表、编码后的训练验证数据
    参数：
        data_proc_dir: 预处理文件夹路径
    返回：
        vocab_sizes: list，每个特征的词表大小
        train_ds: 训练集TensorDataset
        val_ds: 验证集TensorDataset
    """
    # 加载特征配置
    with open(data_proc_dir / "feature_config.json", "r", encoding="utf-8") as f:
        feat_config = json.load(f)
    feature_cols = feat_config["feature_cols"]
    vocab_dict = feat_config["vocab_sizes"]
    # 按照feature_cols顺序取出，并且转成int
    vocab_sizes = [int(vocab_dict[col]) for col in feature_cols]

    # 加载编码好的数据
    data_npz = np.load(data_proc_dir / "encoded_features.npz")
    x_train_np = data_npz["x_train"]
    y_train_np = data_npz["y_train"]
    x_val_np = data_npz["x_val"]
    y_val_np = data_npz["y_val"]

    # 转为torch张量，输入是整数索引，标签是浮点
    x_train = torch.from_numpy(x_train_np).long()
    y_train = torch.from_numpy(y_train_np).float()
    x_val = torch.from_numpy(x_val_np).long()
    y_val = torch.from_numpy(y_val_np).float()

    # 组装数据集
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    # 打印信息校验
    print(f"字段数量：{len(vocab_sizes)}")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

    return vocab_sizes, train_ds, val_ds, feature_cols


# FM 和 DeepFM 分开创建两个独立的随机生成器 Generator，种子都固定 123，不能共用同一个 Generator
def build_dataloaders(train_ds, val_ds):
    """
    分别构造两组独立的DataLoader（给FM、DeepFM各自一套）
    两组Generator种子都固定123，互相独立，保证读取批次顺序一致
    返回：
        train_loader_fm, val_loader_fm
        train_loader_deepfm, val_loader_deepfm
    """
    # FM 的随机生成器
    gen_fm = torch.Generator()
    gen_fm.manual_seed(123)
    train_loader_fm = DataLoader(
        train_ds,
        batch_size=512,
        shuffle=True,
        generator=gen_fm
    )
    val_loader_fm = DataLoader(
        val_ds,
        batch_size=2048,
        shuffle=False
    )

    # DeepFM 的随机生成器，独立对象，同样种子123
    gen_deepfm = torch.Generator()
    gen_deepfm.manual_seed(123)
    train_loader_deepfm = DataLoader(
        train_ds,
        batch_size=512,
        shuffle=True,
        generator=gen_deepfm
    )
    val_loader_deepfm = DataLoader(
        val_ds,
        batch_size=2048,
        shuffle=False
    )

    return train_loader_fm, val_loader_fm, train_loader_deepfm, val_loader_deepfm

def create_and_align_models(vocab_sizes):
    """
    创建DeepFM、FM，并对齐共有参数初始值
    返回：
        fm, deepfm
        reg_fm: FM需要正则的参数列表（二阶emb）
        reg_deepfm: DeepFM需要正则的参数列表（共享emb）
    """
    torch.manual_seed(42)

    # 先创建DeepFM
    deepfm = DeepFM(vocab_sizes, k=8)
    # 再创建FM
    fm = FM(vocab_sizes, latent_dim=8)

    # 关闭梯度，复制参数
    with torch.no_grad():
        # 复制一阶embedding
        for i in range(len(vocab_sizes)):
            fm.first_embeds[i].weight.copy_(deepfm.emb_lin[i].weight)
        # 复制二阶共享embedding
        for i in range(len(vocab_sizes)):
            fm.second_embeds[i].weight.copy_(deepfm.emb_shared[i].weight)
        # 复制全局偏置
        fm.bias.copy_(deepfm.bias)

    # 校验参数是否相等
    ok = True
    for i in range(len(vocab_sizes)):
        if not torch.equal(fm.first_embeds[i].weight, deepfm.emb_lin[i].weight):
            ok = False
        if not torch.equal(fm.second_embeds[i].weight, deepfm.emb_shared[i].weight):
            ok = False
    if not torch.equal(fm.bias, deepfm.bias):
        ok = False

    print(f"参数对齐校验结果：{ok}")

    # 提前转为list！parameters()是生成器，不转list遍历一次就空了
    reg_fm = list(fm.second_embeds.parameters())
    reg_deepfm = list(deepfm.emb_shared.parameters())

    return fm, deepfm, reg_fm, reg_deepfm

def train_one_epoch(model, loader, optimizer, reg_params, lam):
    """
    训练一轮，统一逻辑给FM、DeepFM共用
    返回：本轮平均bce、平均正则损失、平均总目标损失
    """
    model.train()
    total_bce = 0.0
    total_reg = 0.0
    total_total = 0.0
    total_samples = 0

    loss_fn = nn.BCEWithLogitsLoss()

    for xb, yb in loader:
        # yb现在是 [B,1]，squeeze成 [B] 和 logits对齐
        yb = yb.squeeze(-1)
        logits = model(xb)

        bce = loss_fn(logits, yb)

        # 计算L2正则
        reg = 0.0
        for p in reg_params:
            reg += torch.sum(p ** 2)
        reg = reg * lam / 2.0

        loss_total = bce + reg

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # 样本加权累加
        batch_n = xb.shape[0]
        total_bce += bce.item() * batch_n
        total_reg += reg.item() * batch_n
        total_total += loss_total.item() * batch_n
        total_samples += batch_n

    avg_bce = total_bce / total_samples
    avg_reg = total_reg / total_samples
    avg_total = total_total / total_samples
    return avg_bce, avg_reg, avg_total

def evaluate(model, loader):
    """
    验证集评估，统一逻辑给FM、DeepFM共用
    返回：val_logloss, val_auc, val_acc
    """
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
    # 纯BCE，也就是LogLoss
    loss_fn = nn.BCEWithLogitsLoss()
    val_logloss = loss_fn(logits, y_true).item()

    # sigmoid得到概率
    prob = torch.sigmoid(logits)
    val_auc = roc_auc_score(y_true.numpy(), prob.numpy())
    val_acc = ((prob >= 0.5) == y_true.bool()).float().mean().item()

    return val_logloss, val_auc, val_acc



if __name__ == "__main__":
    # ========== 1. 加载基础资源 ==========
    data_proc_dir, out_dir = get_paths()
    print(f"预处理数据目录：{data_proc_dir}")
    print(f"实验输出目录：{out_dir}")

    vocab_sizes, train_ds, val_ds, feature_cols = load_data(data_proc_dir)
    train_loader_fm, val_loader_fm, train_loader_deepfm, val_loader_deepfm = build_dataloaders(train_ds, val_ds)
    print("FM训练loader批次数量：", len(train_loader_fm))
    print("DeepFM训练loader批次数量：", len(train_loader_deepfm))

    fm, deepfm, reg_fm, reg_deepfm = create_and_align_models(vocab_sizes)
    x_small, y_small = train_ds[:4]
    out_fm = fm(x_small)
    out_deepfm = deepfm(x_small)
    print(f"FM小批次输出shape: {out_fm.shape}")
    print(f"DeepFM小批次输出shape: {out_deepfm.shape}")

    # ========== 2. 超参配置 ==========
    lr = 0.001
    lam = 1e-4
    epochs = 5
    # 各自独立优化器
    opt_fm = optim.Adam(fm.parameters(), lr=lr)
    opt_deepfm = optim.Adam(deepfm.parameters(), lr=lr)

    # 保存实验配置
    exp_config = {
        "vocab_sizes": vocab_sizes,
        "feature_cols": feature_cols,
        "k": 8,
        "lr": lr,
        "lambda": lam,
        "reg_scope": "only second/shared embedding, include index 0; linear/bias/mlp no reg",
        "reg_formula": "lambda / 2 * sum(weight ** 2)",
        "optimizer_weight_decay": 0,
        "epochs": epochs,
        "train_bs": 512,
        "val_bs": 2048,
        "seed_data": 123,
        "seed_init": 42,
        "dnn_struct": [176, 64, 32, 1],
        "compare_rule": "compare epoch 5 fixed, not pick best epoch"
    }

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)

    # ========== 3. 训练循环 ==========
    history = {"fm": [], "deepfm": []}
    for epoch in range(1, epochs + 1):
        print(f"\n==== Epoch {epoch}/{epochs} ====")

        # ----- FM一轮训练+验证 -----
        fm_bce, fm_reg, fm_total = train_one_epoch(fm, train_loader_fm, opt_fm, reg_fm, lam)
        fm_val_ll, fm_val_auc, fm_val_acc = evaluate(fm, val_loader_fm)
        history["fm"].append({
            "epoch": epoch,
            "train_bce": fm_bce,
            "train_reg": fm_reg,
            "train_total": fm_total,
            "val_logloss": fm_val_ll,
            "val_auc": fm_val_auc,
            "val_acc": fm_val_acc
        })
        print(f"FM   | train_bce:{fm_bce:.6f} | reg:{fm_reg:.6f} | total:{fm_total:.6f} | val_ll:{fm_val_ll:.6f} | auc:{fm_val_auc:.4f} | acc:{fm_val_acc:.4f}")

        # ----- DeepFM一轮训练+验证 -----
        deepfm_bce, deepfm_reg, deepfm_total = train_one_epoch(deepfm, train_loader_deepfm, opt_deepfm, reg_deepfm, lam)
        deepfm_val_ll, deepfm_val_auc, deepfm_val_acc = evaluate(deepfm, val_loader_deepfm)
        history["deepfm"].append({
            "epoch": epoch,
            "train_bce": deepfm_bce,
            "train_reg": deepfm_reg,
            "train_total": deepfm_total,
            "val_logloss": deepfm_val_ll,
            "val_auc": deepfm_val_auc,
            "val_acc": deepfm_val_acc
        })
        print(f"DeepFM | train_bce:{deepfm_bce:.6f} | reg:{deepfm_reg:.6f} | total:{deepfm_total:.6f} | val_ll:{deepfm_val_ll:.6f} | auc:{deepfm_val_auc:.4f} | acc:{deepfm_val_acc:.4f}")

    # ========== 4. 保存历史和第5轮权重 ==========
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    torch.save(fm.state_dict(), out_dir / "fm_epoch5.pt")
    torch.save(deepfm.state_dict(), out_dir / "deepfm_epoch5.pt")
    print("\n✅ 模型权重、配置、历史指标保存完成")

    # ========== 5. 重载校验：读回来验证指标复现 ==========
    print("\n==== 重载权重校验 ====")
    fm_reload = FM(vocab_sizes, latent_dim=8)
    deepfm_reload = DeepFM(vocab_sizes, k=8)
    fm_reload.load_state_dict(torch.load(out_dir / "fm_epoch5.pt", weights_only=True))
    deepfm_reload.load_state_dict(torch.load(out_dir / "deepfm_epoch5.pt", weights_only=True))

    fm_re_ll, fm_re_auc, fm_re_acc = evaluate(fm_reload, val_loader_fm)
    deepfm_re_ll, deepfm_re_auc, deepfm_re_acc = evaluate(deepfm_reload, val_loader_deepfm)
    print(f"FM重载校验 | val_ll:{fm_re_ll:.6f} | auc:{fm_re_auc:.4f} | acc:{fm_re_acc:.4f}")
    print(f"DeepFM重载校验 | val_ll:{deepfm_re_ll:.6f} | auc:{deepfm_re_auc:.4f} | acc:{deepfm_re_acc:.4f}")
    print("✅ 重载校验完成，脚本编写完毕！")



