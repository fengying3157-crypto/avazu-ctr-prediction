#ctr_models.py 只干一件事：放模型结构，没有训练代码、没有打印、没有文件读写
# ctr_models.py 就是 “模型说明书”，只写长什么样；训练脚本是 “训练车间”；误差分析脚本是 “质检车间”。说明书单独拿出来，车间都来查说明书，车间之间不互相串门
from typing import Any

import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__(self, vocab_sizes: list[int]):
        super().__init__()
        self.first_embeds = nn.ModuleList()
        self.bias = nn.Parameter(torch.zeros(1))
        for vs in vocab_sizes:
            self.first_embeds.append(nn.Embedding(vs, 1))

    def forward(self, x):
        out = 0.0
        for i, emb in enumerate(self.first_embeds):
            out = out + emb(x[:, i])
        out = out.squeeze(-1)
        out = out + self.bias
        return out

class FM(nn.Module):
    def __init__(self, vocab_sizes: list[int], latent_dim=8):
        super().__init__()
        self.first_embeds = nn.ModuleList()
        self.second_embeds = nn.ModuleList()
        self.bias = nn.Parameter(torch.zeros(1))
        for vs in vocab_sizes:
            self.first_embeds.append(nn.Embedding(vs, 1))
            self.second_embeds.append(nn.Embedding(vs, latent_dim))

    def init_spec(self):
        # 本次实验要求的初始化：一阶、bias=0；二阶 N(0,0.01)
        for emb in self.first_embeds:
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.bias)
        for emb in self.second_embeds:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)


    def forward(self, x):
        B = x.shape[0]
        lin = 0.0
        for i, emb in enumerate(self.first_embeds):
            lin = lin + emb(x[:, i])
        lin = lin.squeeze(-1)
        lin += self.bias

        v_list = []
        for i, emb in enumerate(self.second_embeds):
            v_list.append(emb(x[:, i]))
        v = torch.stack(v_list, dim=1)
        sum_square = torch.sum(v, dim=1) ** 2
        square_sum = torch.sum(v ** 2, dim=1)
        inter = 0.5 * torch.sum(sum_square - square_sum, dim=-1)
        logits = lin + inter
        return logits

    def get_second_l2(self):
        # 单独取出所有二阶emb，计算F范数平方和，用于正则项
        total = 0.0
        for emb in self.second_embeds:
            total = total + torch.sum(emb.weight ** 2)
        return total

class DeepFM(nn.Module):
    def __init__(self, vocab_sizes:list[int],k: int = 8,use_fm2: bool = True) :
        super().__init__()
        self.num_fields = len(vocab_sizes)
        self.k = k
        self.use_fm2 = use_fm2

        #一阶emb:每个字段单独Embedding，输出维度1
        self.emb_lin = nn.ModuleList([
            nn.Embedding(v,1) for v in vocab_sizes
        ])
        #共享二阶emb:【只定义这一组】，FM二阶和DNN公用
        self.emb_shared = nn.ModuleList([
            nn.Embedding(v,k) for v in vocab_sizes
        ])
        #全局偏置
        self.bias = nn.Parameter(torch.zeros(1))

        #DNN部分
        self.dnn = nn.Sequential(
            nn.Linear(self.num_fields * k,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

        # ========== 初始化规则 ==========
        # 一阶emb、bias初始化为0
        for emb in self.emb_lin:
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.bias)
        #共享emb:N（0,0.01)
        for emb in self.emb_shared:
            nn.init.normal_(emb.weight,mean=0.0,std=0.01)
        # DNN用pytorch默认初始化，不用手动改

    def forward(self,x:torch.Tensor,return_parts:bool = False):
        """
                x: [B, num_fields]
                return_parts=True时，返回 (logits, part_lr, part_fm2, part_dnn)
                """
        B = x.shape[0]

        # ---------- 一阶分支（LR部分）----------
        lin_emb = []
        for i in range(self.num_fields):
            field_idx = x[:, i]
            e = self.emb_lin[i](field_idx)  # [B,1]
            lin_emb.append(e)
        lin_stack = torch.stack(lin_emb, dim=1)  # [B,22,1]
        part_lr = lin_stack.sum(dim=(1, 2)) + self.bias  # [B]

        # ---------- 共享Embedding查表，给FM和DNN共用 ----------
        shared_emb = []
        for i in range(self.num_fields):
            field_idx = x[:, i]
            e = self.emb_shared[i](field_idx)  # [B,8]
            shared_emb.append(e)
        E = torch.stack(shared_emb, dim=1)  # [B,22,8]

        # ---------- FM二阶分支 ----------
        sum_sq = E.sum(dim=1) ** 2  # [B,8]
        sq_sum = (E ** 2).sum(dim=1)  # [B,8]
        part_fm2 = 0.5 * (sum_sq - sq_sum).sum(dim=1)  # [B]

        # ---------- DNN深层分支 ----------
        dnn_in = E.flatten(1)  # [B, 22*8=176]
        part_dnn = self.dnn(dnn_in).squeeze(-1)  # [B]

        # 三项相加得到logits
        # 根据use_fm2开关决定是否加入二阶项
        if self.use_fm2:
            logits = part_lr + part_fm2 + part_dnn
        else:
            logits = part_lr + part_dnn

        if return_parts:
            return logits, part_lr, part_fm2, part_dnn
        return logits