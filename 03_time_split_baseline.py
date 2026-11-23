import os
import gzip
import pandas as  pd
import numpy as  np
from  datetime import date
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score


# 脚本路径定位
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir, "data", "avazu", "processed")
data_root = os.path.normpath(data_root)

# 输入抽样文件路径
sample_path = os.path.join(data_root, "sample.csv.gz")

#划分三个部分
train_out = os.path.join(data_root,'train.csv.gz')
val_out = os.path.join(data_root,'val.csv.gz')
holdout_out = os.path.join(data_root,'holdout.csv.gz')
#划分时间
train_start = '2014-10-21'
train_end = '2014-10-28'
val_date = '2014-10-29'
holdout_date = '2014-10-30'

#读取文件、
df = pd.read_csv(sample_path,dtype = str)
# dt_date 转日期类型
df['dt_date'] = pd.to_datetime(df['dt_date']).dt.date
#click转为int
df['click'] = df['click'].astype(int)

# 检查click只有0、1
click_unique = sorted(df['click'].unique().tolist())
print(f'click字段唯一值：{click_unique}')
assert click_unique == [0,1]
print(f'全部样本总行数：{len(df)}')

#检验通过开始划分
df_train = df[(df["dt_date"] >= pd.to_datetime("2014-10-21").date()) & (df["dt_date"] <= pd.to_datetime("2014-10-28").date())]
df_val   = df[df["dt_date"] == pd.to_datetime("2014-10-29").date()]
df_hold  = df[df["dt_date"] == pd.to_datetime("2014-10-30").date()]


print(f"训练集行数：{len(df_train)}")
print(f"验证集行数：{len(df_val)}")
print(f"留出集行数：{len(df_hold)}")
print(f"三份合计行数：{len(df_train)+len(df_val)+len(df_hold)}")

#检验代码
# 校验：三份都不为空
assert len(df_train) > 0, "训练集为空！"
assert len(df_val) > 0, "验证集为空！"
assert len(df_hold) > 0, "留出集为空！"

# 检查id跨集合无重复
id_train = set(df_train["id"].unique())
id_val = set(df_val["id"].unique())
id_hold = set(df_hold["id"].unique())

overlap_train_val = id_train & id_val
overlap_train_hold = id_train & id_hold
overlap_val_hold = id_val & id_hold

print(f"训练集&验证集重复id数量：{len(overlap_train_val)}")
print(f"训练集&留出集重复id数量：{len(overlap_train_hold)}")
print(f"验证集&留出集重复id数量：{len(overlap_val_hold)}")

assert len(overlap_train_val) == 0, "训练集和验证集存在重复id！"
assert len(overlap_train_hold) == 0, "训练集和留出集存在重复id！"
assert len(overlap_val_hold) == 0, "验证集和留出集存在重复id！"
print("✅ ID去重校验通过，三份数据id互不重叠")

#检查完成然后保存各个文件
# 保存三份压缩csv
df_train.to_csv(train_out, index=False, compression="gzip")
df_val.to_csv(val_out, index=False, compression="gzip")
df_hold.to_csv(holdout_out, index=False, compression="gzip")

print("✅ 时间切分文件保存完成")
print(f"训练集：{train_out}")
print(f"验证集：{val_out}")
print(f"留出集：{holdout_out}")


# 做常数基线部分
print('=======常数基线计算========')
train_click_sum = df_train['click'].sum()
train_total = len(df_train)
p0  = train_click_sum/train_total

#验证集
y_val_true = df_val['click'].values
y_val_pred_prob = np.full_like(y_val_true,fill_value=p0,dtype=np.float64)

#计算三个指标
ll = log_loss(y_val_true,y_val_pred_prob)
auc = roc_auc_score(y_val_true,y_val_pred_prob)
# 阈值0.5判断类别
y_preb_label = (y_val_pred_prob >= 0.5).astype(int)
acc = accuracy_score(y_val_true,y_preb_label)

print("\n==================== 常数基线结果 ====================")
print(f"训练集总行数：{train_total}，训练集点击1总数：{train_click_sum}")
print(f"训练集点击率 p0 = {p0:.6f}")
print(f"验证集总行数：{len(df_val)}，验证集真实点击率：{y_val_true.mean():.6f}")
print(f"常数基线 LogLoss: {ll:.6f}")
print(f"常数基线 AUC: {auc:.6f}")
print(f"常数基线 Accuracy(阈值0.5): {acc:.6f}")