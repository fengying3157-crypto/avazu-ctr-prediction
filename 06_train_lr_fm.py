from pathlib import Path

import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
DATA_DIR = Path(__file__).resolve().parent / "data" / "avazu" / "processed"

# ---------------------- 路径配置 ----------------------
npz_path = f"{DATA_DIR}/encoded_features.npz"
config_path = f"{DATA_DIR}/feature_config.json"

# ---------------------- 加载编码数据 ----------------------
data = np.load(npz_path)
x_train = torch.from_numpy(data["x_train"]).long()
y_train = torch.from_numpy(data["y_train"]).float()
x_val = torch.from_numpy(data["x_val"]).long()
y_val = torch.from_numpy(data["y_val"]).float()

# ---------------------- 加载词表配置 vocab_sizes ----------------------
with open(config_path, "r", encoding="utf-8") as f:
    feature_config = json.load(f)
vocab_dict = feature_config["vocab_sizes"]
# 按之前固定的特征顺序取出size，转int
feature_order = ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'hour_of_day']
vocab_sizes = [int(vocab_dict[name]) for name in feature_order]
num_fields = len(vocab_sizes)
fm_k = 8


print(f"num_fields={num_fields}, fm_k={fm_k}")
print(f"X_train shape: {x_train.shape}, y_train shape:{y_train.shape}")
print(f"X_val shape: {x_val.shape}, y_val shape:{y_val.shape}")

# 写 **LR 模型类**，严格遵守协议：一阶权重、偏置全部初始化为 0。
class LR(nn.Module):
    def __init__(self, vocab_sizes):
        super().__init__()
        self.num_fields = len(vocab_sizes)
        # 一阶embedding：就是每个特征单独的权重，维度=1
        self.first_embeds = nn.ModuleList()
        for vs in vocab_sizes:
            emb_layer = nn.Embedding(vs, 1)
            nn.init.zeros_(emb_layer.weight)# 要求：一阶权重初始化为0
            self.first_embeds.append(emb_layer)
        # 全局偏置，初始化为0
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        # x: [B, num_fields]，每个位置是类别id
        first_list = []
        for i in range(self.num_fields):
            xi = x[:, i]
            w_i = self.first_embeds[i](xi)  # [B, 1]
            first_list.append(w_i)
        first_sum = torch.sum(torch.cat(first_list,dim=1),dim=1,keepdim=True)#【B，1】
        logits = first_sum + self.bias
        return logits


#定义FM
class FM(nn.Module):

    def __init__(self, vocab_sizes,k=8) :
        super().__init__()
        self.num_fields = len(vocab_sizes)
        self.k = k
        #一阶部分，和LR一样，初试为0
        self.first_embeds = nn.ModuleList()
        for vs in vocab_sizes:
            emb_layer = nn.Embedding(vs,1)
            nn.init.zeros_(emb_layer.weight)
            self.first_embeds.append(emb_layer)
        self.bias = nn.Parameter(torch.zeros(1))

        #FM二阶隐向量v,k=8,正太初始化，std = 0.01
        self.second_embeds = nn.ModuleList()
        for vs in vocab_sizes:
            v_layer = nn.Embedding(vs,k)
            nn.init.normal_(v_layer.weight,mean=0,std=0.01)
            self.second_embeds.append(v_layer)

    def forward(self,x):
        #x;[B,num_fields]
        #一阶项
        first_list = []
        for i in range(self.num_fields):
            xi = x[:,i]
            w_i = self.first_embeds[i](xi)
            first_list.append(w_i)
        first_sum = torch.sum(torch.cat(first_list,dim=1),dim=1,keepdim=True)

        #二阶项 FM公式
        v_list = []
        for i in range(self.num_fields):
            xi = x[:,i]
            vi = self.second_embeds[i](xi)
            v_list.append(vi)
        vx = torch.stack(v_list,dim=1)# [B, field, k]

        sum_square = torch.sum(vx,dim=1) ** 2
        square_sum = torch.sum(vx**2,dim=1)
        second_term = 0.5 * torch.sum(sum_square - square_sum,dim=1,keepdim=True)

        logits = first_sum + second_term + self.bias
        return logits


#通过三个测试：写**共用验证函数**
def evaluate(model,loader,criterion):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb,yb in loader:
            logits = model(xb)
            loss = criterion(logits,yb)
            total_loss += loss.item() * xb.size(0)
            prob = torch.sigmoid(logits)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(prob.cpu().numpy())
        avg_loss = total_loss / len(loader.dataset)
        auc = roc_auc_score(y_true,y_pred)
        acc = np.mean((np.array(y_pred) >= 0.5) == np.array(y_true)).item()
        return avg_loss,auc,acc


#测试一下
if __name__ == "__main__":
    torch.manual_seed(42)  # ✅ 新增全局种子42，放在最前面，模型初始化之前

    lr_model = LR(vocab_sizes)
    fm_model = FM(vocab_sizes, k=8)
    x_test = x_train[:3]
    y_test = y_train[:3]

    out_lr = lr_model(x_test)
    out_fm = fm_model(x_test)
    print("LR test logits shape:", out_lr.shape)
    print("FM test logits shape:", out_fm.shape)
    print(f"原生FM与LR输出差值（带二阶）：{torch.abs(out_lr - out_fm).max().item():.8f}")

    # ---------------- 校验2：FR暴力两两交叉核对FM二阶公式（现在v非零！有效校验） ----------------
    v_list = []
    for i in range(fm_model.num_fields):
        xi = x_test[:, i]
        vi = fm_model.second_embeds[i](xi)
        v_list.append(vi)
    vx = torch.stack(v_list, dim=1)

    B = vx.shape[0]
    fr_second = torch.zeros(B, 1)
    for i in range(fm_model.num_fields):
        for j in range(i + 1, fm_model.num_fields):
            dot_val = torch.sum(vx[:, i, :] * vx[:, j, :], dim=1, keepdim=True)
            fr_second += dot_val

    sum_square = torch.sum(vx, dim=1) ** 2
    square_sum = torch.sum(vx ** 2, dim=1)
    vec_second = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

    diff2 = torch.abs(fr_second - vec_second).max()
    print(f"\n校验2｜向量化FM和FR暴力两两最大差值：{diff2.item():.8f}")

    # ---------------- 校验1：二阶Emb清零，FM退化成LR（挪到FR校验之后） ----------------
    for emb in fm_model.second_embeds:
        nn.init.zeros_(emb.weight)
    out_fm_no_second = fm_model(x_test)
    diff1 = torch.abs(out_lr - out_fm_no_second).max()
    print(f"\n校验1｜二阶清零后，LR与FM最大差值：{diff1.item():.8f}")

    # ---------------- 校验3：梯度检查 ----------------
    fm_model = FM(vocab_sizes, k=8)
    logits = fm_model(x_test)
    loss = nn.BCEWithLogitsLoss()(logits, y_test)
    loss.backward()

    grad_sample = fm_model.second_embeds[0].weight.grad
    grad_has_nonzero = torch.any(grad_sample != 0.0).item()
    has_nan = torch.any(torch.isnan(grad_sample)).item()
    has_inf = torch.any(torch.isinf(grad_sample)).item()

    print(f"\n校验3｜二阶Emb梯度是否存在非零：{grad_has_nonzero}")
    print(f"校验3｜梯度有无NaN：{has_nan}")
    print(f"校验3｜梯度有无Inf：{has_inf}")


    # ---------------- 构造数据集、Dataloader ----------------
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    gen_lr = torch.Generator()
    gen_lr.manual_seed(123)
    gen_fm = torch.Generator()
    gen_fm.manual_seed(123)
    train_loader_lr = DataLoader(train_ds,batch_size=512,shuffle=True,generator=gen_lr)
    train_loader_fm = DataLoader(train_ds,batch_size=512,shuffle=True,generator=gen_fm)
    val_loader = DataLoader(val_ds,batch_size=2048,shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    lr_model = LR(vocab_sizes)
    fm_model = FM(vocab_sizes,k=8)

    opt_lr = optim.Adam(lr_model.parameters(),lr = 0.01)
    opt_fm = optim.Adam(fm_model.parameters(),lr= 0.01)

    epochs = 5
    history = []  # ✅ 新增：用来存每一轮指标
    print('\n=======准备开始训练========')
    #加上**两个模型并行训练的循环**：
    for epoch in range(epochs):
        #----------训练lr----------
        lr_model.train()
        train_loss_lr = 0.0
        for xb,yb in train_loader_lr:
            opt_lr.zero_grad()
            logits = lr_model(xb)
            loss = criterion(logits,yb)
            loss.backward()
            opt_lr.step()
            train_loss_lr += loss.item() * xb.size(0)

        train_loss_lr /= len(train_loader_lr.dataset)
        val_loss_lr,val_auc_lr,val_acc_lr = evaluate(lr_model,val_loader, criterion)

        # ----------训练lr----------
        fm_model.train()
        train_loss_fm = 0.0
        for xb, yb in train_loader_fm:
            opt_fm.zero_grad()
            logits = fm_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt_fm.step()
            train_loss_fm += loss.item() * xb.size(0)

        train_loss_fm /= len(train_loader_fm.dataset)
        val_loss_fm, val_auc_fm, val_acc_fm = evaluate(fm_model, val_loader, criterion)
        history.append({
            "epoch": epoch + 1,
            "lr_train_logloss": train_loss_lr,
            "lr_val_logloss": val_loss_lr,
            "lr_val_auc": val_auc_lr,
            "lr_val_acc": val_acc_lr,
            "fm_train_logloss": train_loss_fm,
            "fm_val_logloss": val_loss_fm,
            "fm_val_auc": val_auc_fm,
            "fm_val_acc": val_acc_fm,
        })

        print(f'\n==== Epoch {epoch + 1}/{epochs} ====')
        print(f'LR  | train_logloss:{train_loss_lr:.6f} | val_logloss:{val_loss_lr:.6f} | val_auc:{val_auc_lr:.6f} | val_acc:{val_acc_lr:.6f}')
        print(f'FM  | train_logloss:{train_loss_fm:.6f} | val_logloss:{val_loss_fm:.6f} | val_auc:{val_auc_fm:.6f} | val_acc:{val_acc_fm:.6f}')


# ---------------- 保存实验配置、训练历史、模型权重 ----------------
exp_config = {
        "task": "LR vs FM Avazu CTR",
        "seed_global": 42,
        "seed_loader_lr":123,
        "seed_loader_fm":123,
        "lr":0.01,
        "epochs":5,
        "train_batch":512,
        "val_batch":2048,
        "n_feat":22,
        "fm_k":8,
        "train_size": len(train_loader_lr.dataset),
        "val_size": len(val_loader.dataset),
        "vocab_sizes": vocab_sizes
    }
with open(f"{DATA_DIR}/exp_config_lr_fm.json","w",encoding="utf-8") as f:
        json.dump(exp_config, f, ensure_ascii=False, indent=2)
with open(f"{DATA_DIR}/exp_history_lr_fm.json","w",encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 保存末轮权重，文件名 lr_compare_model.pt / fm_compare_model.pt
torch.save(lr_model.state_dict(), f"{DATA_DIR}/lr_compare_model.pt")
torch.save(fm_model.state_dict(), f"{DATA_DIR}/fm_compare_model.pt")
print("\n✅ 权重、实验配置、历史指标保存完成")

# ---------------- 重载校验 ----------------
lr_reload = LR(vocab_sizes)
lr_reload.load_state_dict(torch.load(f"{DATA_DIR}/lr_compare_model.pt", map_location="cpu"))
fm_reload = FM(vocab_sizes, k=8)
fm_reload.load_state_dict(torch.load(f"{DATA_DIR}/fm_compare_model.pt", map_location="cpu"))

# 取同一个验证batch对比logits
xb_test, yb_test = next(iter(val_loader))
with torch.no_grad():
    logits_old_lr = lr_model(xb_test)
    logits_new_lr = lr_reload(xb_test)
    logits_old_fm = fm_model(xb_test)
    logits_new_fm = fm_reload(xb_test)

diff_lr = torch.abs(logits_old_lr - logits_new_lr).max().item()
diff_fm = torch.abs(logits_old_fm - logits_new_fm).max().item()
print(f"\n==== 重载一致性校验 ====")
print(f"LR logits最大误差：{diff_lr:.8f}")
print(f"FM logits最大误差：{diff_fm:.8f}")

# 重载模型重跑完整验证集核对指标
val_loss_lr_re, val_auc_lr_re, val_acc_lr_re = evaluate(lr_reload, val_loader, criterion)
val_loss_fm_re, val_auc_fm_re, val_acc_fm_re = evaluate(fm_reload, val_loader, criterion)
print(f"\n==== 重载后验证指标 ====")
print(f"LR  | val_logloss:{val_loss_lr_re:.6f} | val_auc:{val_auc_lr_re:.6f} | val_acc:{val_acc_lr_re:.6f}")
print(f"FM  | val_logloss:{val_loss_fm_re:.6f} | val_auc:{val_auc_fm_re:.6f} | val_acc:{val_acc_fm_re:.6f}")


