import numpy as np
import json
import pandas as pd
from collections import Counter
import os

#路径引入
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.normpath(
    os.path.join(script_dir, "data", "avazu", "processed")
)
train_path = os.path.join(out_dir, "train.csv.gz")
val_path = os.path.join(out_dir, "val.csv.gz")


#读取，全部读为字符串
df_train = pd.read_csv(train_path,dtype = str)
df_val = pd.read_csv(val_path,dtype = str)

print('训练集形状：',df_train.shape)
print('验证集形状：',df_val.shape)

#Step2 选出要使用的 21 个原始类别字段,从hour提取hour_of_day，凑齐一共 22 个输入特征
cat_cols = [
    'C1','banner_pos','site_id','site_domain','site_category',
    'app_id','app_domain','app_category','device_id','device_ip',
    'device_model','device_type','device_conn_type','C14','C15',
    'C16','C17','C18','C19','C20','C21']

#构造hour of day
df_train['hour_of_day'] = df_train['hour'].str[-2:]
df_val['hour_of_day'] = df_val['hour'].str[-2:]

#最终22个
feature_cols = cat_cols + ['hour_of_day']
print('特征数量： ',len(feature_cols))
print('特征列表: ',feature_cols)

# 每一个特征单独建词表,只看训练集；频次≥2 才分配 id，从 1 开始；低频 / 缺失 / 未知全部映射 0；每个字段一套独立 vocab
min_count = 2
vocab_dict = {}

for col in  feature_cols:
    counter = Counter(df_train[col].dropna())
    this_vocab = {}
    idx = 1
    for val,cnt in  counter.items():
        if cnt >= min_count:
            this_vocab[val] = idx
            idx += 1
    vocab_dict[col] = this_vocab
    print(f'{col} 词表大小为： {len(this_vocab)}')

# 构造配置文件
# Embedding总行数包含编号0，区别于有效类别数。
vocab_sizes = {col: len(vocab_dict[col]) + 1 for col in feature_cols}
feature_config = {
    "feature_cols": feature_cols,
    "min_count": min_count,
    "vocab_sizes": vocab_sizes,
    "hour_of_day_rule": "extract last two characters from hour string (%y%m%d%H)",
    "zero_rule": "low frequency, unseen and missing value mapped to 0"
}
config_path = os.path.join(out_dir, "feature_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(feature_config, f, ensure_ascii=False, indent=2)
print("✅ feature_config.json 已保存")



#写编码转换函数，批量把 df_train、df_val 转成整数数组 X_train、X_va
def encode_df(df,feature_cols,vocab_dict):
    out = []
    for col in feature_cols:
        vocab = vocab_dict[col]
        #不在词表返回0
        arr = df[col].apply(lambda x: vocab.get(x,0)).values
        out.append(arr)
    #按列堆叠，shape[样本数，22]
    mat = np.column_stack(out)
    return mat


x_train = encode_df(df_train,feature_cols,vocab_dict)
x_val = encode_df(df_val,feature_cols,vocab_dict)

#标签
y_train = df_train['click'].astype(np.float32).values.reshape(-1,1)
y_val = df_val['click'].astype(np.float32).values.reshape(-1,1)

print('x_train 形状： ',x_train.shape,x_train.dtype)
print('y_train 形状： ',y_train.shape,y_train.dtype)
print('x_val 形状： ',x_val.shape,x_val.dtype)
print('y_val 形状： ',y_val.shape,y_val.dtype)

#检验一下
# 1、检查编码没有负数；每个编码值 ≤ 对应字段词表最大 id
# 2、每个字段在验证集上 0（OOV）的占比，这个后面写到报告里
#检验1 无负数
assert np.all(x_train >= 0),'x_train 存在负编码'
assert np.all(x_val >= 0),'x_val 存在负编码'
print('检验全部编码非负')

# 检验2 每个字段的id不超过词表上限
for idx,col in enumerate(feature_cols):
    vocab = vocab_dict[col]
    max_id = max(vocab.values()) if vocab else 0
    assert x_train[:,idx].max()  <= max_id,f'{col}训练集编码越界'
    assert x_val[:, idx].max() <= max_id, f'{col}验证集编码越界'
print('检验所有字段编码没有超出词表范围')


#统计验证集每个字段0的占比
print('======验证集OOV（编码0）占比=======')
for idx,col in enumerate(feature_cols):
    zero_rate = np.mean(x_val[:,idx] == 0)
    print(f'{col} : {zero_rate}')



#验证检验后保存词表 json、保存 numpy 数组，然后重载回来校验编码是否一模一样
vocab_save_path = os.path.join(out_dir, "feature_vocab.json")
with open(vocab_save_path, "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

#保存编码矩阵
np.savez_compressed(
    os.path.join(out_dir, "encoded_features.npz"),
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val
)
print("✅ 文件保存完成")

# ---------- 重载校验 ----------
with open(vocab_save_path, "r", encoding="utf-8") as f:
    vocab_reload = json.load(f)

data = np.load(os.path.join(out_dir, "encoded_features.npz"))
x_train_reload = data["x_train"]
x_val_reload = data["x_val"]

# 校验1：npz数组本身保存正确
assert np.array_equal(x_train, x_train_reload), "训练集npz重载不一致！"
assert np.array_equal(x_val, x_val_reload), "验证集npz重载不一致！"

# 校验2：用重载出来的词表，重新跑一遍编码，必须和原始X一致
x_train_reencoded = encode_df(df_train, feature_cols, vocab_reload)
x_val_reencoded = encode_df(df_val, feature_cols, vocab_reload)
assert np.array_equal(x_train, x_train_reencoded), "从json词表重载重编码，训练集不一致！"
assert np.array_equal(x_val, x_val_reencoded), "从json词表重载重编码，验证集不一致！"

print("✅ 重载校验：npz数组校验 + json词表重编码校验全部通过")
