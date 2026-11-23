#LR 模型结构再对齐一遍：
# 22 个离散特征，每个特征单独一个 Embedding 层（输出维度 = 1）；
# 全部 embedding 输出相加，再加全局 bias，直接输出 logits；
# 没有二阶交互，就是类别 LR；id=0 正常参与训练，不用 padding_idx。
# 第 1 步：导入包 + 脚本相对路径定位 + 加载 npz 和 config
import os
import json
from typing import Any

import numpy as np
import pandas as pd
import  torch
from  torch import  nn
from  torch.utils.data import  TensorDataset,DataLoader
from sklearn.metrics import roc_auc_score,log_loss

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# 脚本路径定位
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.normpath(
    os.path.join(script_dir, "data", "avazu", "processed")
)
npz_path = os.path.join(out_dir, "encoded_features.npz")
config_path = os.path.join(out_dir, "feature_config.json")

# 加载编码数据
data = np.load(npz_path)
x_train = torch.from_numpy(data["x_train"]).long()
y_train = torch.from_numpy(data["y_train"])
x_val = torch.from_numpy(data["x_val"]).long()
y_val = torch.from_numpy(data["y_val"])

# 加载配置（vocab_sizes已经包含0，embedding直接用，不用+1）
with open(config_path, "r", encoding="utf-8") as f:
    feature_config = json.load(f)
feature_cols = feature_config["feature_cols"]
vocab_sizes = feature_config["vocab_sizes"]

print("X_train shape:", x_train.shape, x_train.dtype)
print("X_val shape:", x_val.shape, x_val.dtype)
print("特征数量：", len(feature_cols))
print("device_ip embedding行数：", vocab_sizes["device_ip"])

# 第 2 步：搭建 CategoryLR 模型类
#  为 22 个特征，各自创建一个 Embedding 层，输出维度 = 1
#  所有 embedding 结果求和，再加一个标量偏置，输出 logits
#全部参数初始化为 0；不设置 padding_idx，0 号 embedding 正常训练

class CategoryLR(nn.Module):
    def __init__(self,feature_cols,vocab_sizes) :
        super().__init__()
        self.feature_cols = feature_cols
        self.embeds = nn.ModuleDict()
        for col in feature_cols:
            vs = vocab_sizes[col]
            #每个特征embedding输出一维，就是一节权重
            emb = nn.Embedding(num_embeddings=vs,embedding_dim=1)
            nn.init.zeros_(emb.weight)
            self.embeds[col] = emb
        # 全局偏置
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        batch_size = x.shape[0]# x: [B, 22]
        total = 0.0
        for i,col in enumerate(self.feature_cols):
            xi = x[:,i]
            wi = self.embeds[col](xi)#[B,1]
            total += wi
        logits = total + self.bias
        return logits

# 第 3 步：实例化模型，构造 Dataset、DataLoader，设置超参
model = CategoryLR(feature_cols,vocab_sizes)

train_dataset= TensorDataset(x_train,y_train)
val_dataset = TensorDataset(x_val,y_val)

train_dataloader = DataLoader(dataset= train_dataset,batch_size=512,shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset,batch_size=2048,shuffle=False)

loss_fn =nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.01)

# 测试前向 + 梯度检查
xb,yb = next(iter(train_dataloader))
logits = model(xb)
print('测试Batch logits shape: ',logits.shape)
loss = loss_fn(logits,yb)
loss.backward()

#随便看一个embedding梯度
first_col = feature_cols[0]
print(f'{first_col} embedding权重梯度均值： ',model.embeds[first_col].weight.grad.mean().item())

# 第 4 步：写验证函数，一次性跑验证集，算出 loss、auc、acc
def evaluate(model,loader,loss_fn):
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb,yb in loader:
            logits = model(xb)
            all_logits.append(logits)
            all_y.append(yb)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    loss = loss_fn(logits,y).item()
    p = torch.sigmoid(logits).cpu().numpy()
    y_np = y.cpu().numpy()
    auc = roc_auc_score(y_np,p)
    pred = (p>0.5).astype(np.float32)
    acc = np.mean(pred == y_np)
    return loss,auc,acc

# 第 5 步：主训练循环，5 轮；记录每轮指标，保存训练历史、模型权重
epochs = 5
history = []
for epoch in range(1,epochs+1):
    #训练
    model.train()
    train_loss_sum = 0.0
    for xb,yb in train_dataloader:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits,yb)
        loss.backward()
        opt.step()

        train_loss_sum += loss.item() * xb.shape[0]
    train_bce = train_loss_sum / len(train_dataset)

    #验证
    val_loss,val_auc,val_acc = evaluate(model,val_dataloader,loss_fn)
    row = {
        "epoch": epoch,
        "train_bce": train_bce,
        "val_logloss": val_loss,
        "val_auc": val_auc,
        "val_acc": val_acc
    }
    history.append(row)
    print(f'epoch:{epoch:2d} | train_bce:{train_bce:.6f}   |  val_logloss:{val_loss:.6f}   |val_auc:{val_auc:.6f} | val_acc:{val_acc:.6f}')

# 保存权重和历史
torch.save(model.state_dict(),os.path.join(out_dir,'lr_model.pt'))
with open(os.path.join(out_dir, "lr_history.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)
print("\n✅ LR训练完成，权重、历史记录已保存")