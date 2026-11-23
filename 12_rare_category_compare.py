# =============================================================================
# 脚本：12_rare_category_compare.py
# 任务：低频类别截断对照实验，min_count=2 vs min_count=5
# 说明：
# 1. 样本完全不变，仅修改建词表时的频次阈值
# 2. 低频、缺失、验证集新类别统一编码为0
# 3. 验证集不参与词表构建，只用训练集统计频次
# 4. 输出两个子目录 min_count_2 / min_count_5，附带统计表格 category_stats.csv
# =============================================================================
import os
import json
import gzip
import pandas as pd
import numpy as np
from collections import Counter

# 根据脚本文件位置，定位项目根目录
SCRIPT_PATH = os.path.abspath(__file__)
PROJ_ROOT = os.path.dirname(SCRIPT_PATH)

# 输入路径
train_csv_in = os.path.join(PROJ_ROOT, "data", "avazu", "processed", "train.csv.gz")
val_csv_in = os.path.join(PROJ_ROOT, "data", "avazu", "processed", "val.csv.gz")
feat_config_in = os.path.join(PROJ_ROOT, "data", "avazu", "processed", "feature_config.json")

# 输出根目录
out_root = os.path.join(PROJ_ROOT, "data", "avazu", "processed", "rare_category_compare")
out_dir_2 = os.path.join(out_root, "min_count_2")
out_dir_5 = os.path.join(out_root, "min_count_5")

# 建立文件夹
os.makedirs(out_dir_2, exist_ok=True)
os.makedirs(out_dir_5, exist_ok=True)

# 读取原始特征配置，拿到固定字段顺序
with open(feat_config_in, "r", encoding="utf-8") as f:
    base_feat_config = json.load(f)
feature_cols = base_feat_config["feature_cols"]
print(f"固定特征字段顺序，一共 {len(feature_cols)} 个字段：")
print(feature_cols)

# -------------------------- 第二步：读取数据，构造衍生字段 hour_of_day --------------------------
def load_df(csv_path):
    # dtype=str 全部读成字符串，避免类别被自动转数字
    df = pd.read_csv(gzip.open(csv_path, "rt", encoding="utf-8"), dtype=str)
    # hour 原始格式类似 "14102100"，取最后两位作为小时
    df["hour_of_day"] = df["hour"].str[-2:]
    return df

# 读训练、验证集
train_df = load_df(train_csv_in)
val_df = load_df(val_csv_in)

# 提取标签，转成float一维数组
y_train = train_df["click"].astype(np.float32).values
y_val = val_df["click"].astype(np.float32).values

# 核对数量
print(f"train 行数: {len(train_df)}")
print(f"val 行数: {len(val_df)}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

def fit_vocab(train_df, feature_cols, min_count):
    """
    只用训练集统计频次，构建词表
    返回：vocab_dict: key=字段名，value={类别字符串:编号}
    编号规则：频次>=min_count的类别，从1开始顺序编号；其余归0
    """
    vocab_dict = {}
    for col in feature_cols:
        # 排除空值统计
        vals = train_df[col].dropna().tolist()
        cnt = Counter(vals)
        # 保留频次 >= min_count 的类别
        keep_items = [k for k, v in cnt.items() if v >= min_count]
        # 从1开始编号
        field_vocab = {cat: idx+1 for idx, cat in enumerate(keep_items)}
        vocab_dict[col] = field_vocab
    return vocab_dict

def encode_df(df, feature_cols, vocab_dict):
    """
    用已经建好的词表编码df
    不在词表、缺失、低频全部映射为0
    返回 np.int64 矩阵 [N, num_fields]
    """
    arr_list = []
    for col in feature_cols:
        field_vocab = vocab_dict[col]
        # map：找不到就是0
        ser = df[col].map(field_vocab).fillna(0).astype(np.int64)
        arr_list.append(ser.values.reshape(-1,1))
    x_mat = np.concatenate(arr_list, axis=1)
    return x_mat

# 分别构建两套词表
vocab2 = fit_vocab(train_df, feature_cols, min_count=2)
vocab5 = fit_vocab(train_df, feature_cols, min_count=5)

# 分别编码训练、验证集
x_train_2 = encode_df(train_df, feature_cols, vocab2)
x_val_2   = encode_df(val_df, feature_cols, vocab2)
x_train_5 = encode_df(train_df, feature_cols, vocab5)
x_val_5   = encode_df(val_df, feature_cols, vocab5)

print("==== 编码完成，形状核对 ====")
print(f"min_count=2 | x_train: {x_train_2.shape}, x_val: {x_val_2.shape}")
print(f"min_count=5 | x_train: {x_train_5.shape}, x_val: {x_val_5.shape}")

# -------------------------- 第四步：统计指标，生成 category_stats.csv --------------------------
# -------------------------- 第四步：统计指标，生成 category_stats.csv --------------------------
def calc_stats(train_df, val_df, x_train_mat, x_val_mat, feature_cols, vocab_dict, min_count):
    rows = []
    total_val = len(val_df)
    for idx, col in enumerate(feature_cols):
        # 1. 保留的类别数量
        field_vocab = vocab_dict[col]
        kept_categories = len(field_vocab)
        vocab_size = kept_categories + 1  # +0号

        # 2. train_zero_rate：训练集编码等于0的比例
        train_col = x_train_mat[:, idx]
        train_zero_rate = np.mean(train_col == 0)

        # 3. val_zero_rate：验证集编码等于0的比例
        val_col = x_val_mat[:, idx]
        val_zero_rate = np.mean(val_col == 0)

        # 4. val_strict_unseen_rate：原始类别不在【训练原始集合】里的比例
        train_unique_set = set(train_df[col].dropna().unique())
        val_series = val_df[col]
        # 非缺失，并且不在训练原始集合
        mask_not_missing = val_series.notna()
        mask_unseen = ~val_series.isin(train_unique_set)
        mask_target = mask_not_missing & mask_unseen
        val_strict_unseen_rate = mask_target.sum() / total_val

        row = {
            "min_count": min_count,
            "field": col,
            "kept_categories": kept_categories,
            "vocab_size": vocab_size,
            "train_zero_rate": train_zero_rate,
            "val_zero_rate": val_zero_rate,
            "val_strict_unseen_rate": val_strict_unseen_rate
        }
        rows.append(row)
    return rows

stats_rows_2 = calc_stats(train_df, val_df, x_train_2, x_val_2, feature_cols, vocab2, min_count=2)
stats_rows_5 = calc_stats(train_df, val_df, x_train_5, x_val_5, feature_cols, vocab5, min_count=5)

all_stats = stats_rows_2 + stats_rows_5
stats_df = pd.DataFrame(all_stats)
stats_csv_path = os.path.join(out_root, "category_stats.csv")
stats_df.to_csv(stats_csv_path, index=False, encoding="utf-8-sig")
print(f"统计表格已保存到：{stats_csv_path}")

# 打印你需要的4个字段：device_ip、device_id、site_id、app_id
print("\n==== 关键字段统计输出 ====")
show_fields = ["device_ip", "device_id", "site_id", "app_id"]
for f in show_fields:
    print(stats_df[stats_df["field"] == f].to_string(index=False))


# -------------------------- 第五步：各项合规校验 --------------------------
print("\n==== 开始校验 ====")
ok_flag = True

# 校验1：内存标签形状与取值合法性（跨组一致性与原CSV核对放到保存后执行）
assert y_train.shape == (len(train_df),), f"y_train shape 错误：{y_train.shape}"
assert y_val.shape == (len(val_df),), f"y_val shape 错误：{y_val.shape}"
assert set(np.unique(y_train)) <= {0.0, 1.0}, "y_train 取值超出 {0,1}"
assert set(np.unique(y_val)) <= {0.0, 1.0}, "y_val 取值超出 {0,1}"
print("✅ 校验1：标签形状合法、取值仅含 0/1")


# 校验2：形状 + dtype 双断言，不满足直接抛异常终止
def check_shape_dtype(name, arr, expected_shape, expected_dtype):
    if arr.shape != expected_shape:
        raise ValueError(f"❌ {name} 形状不符：{arr.shape}，期望 {expected_shape}")
    if arr.dtype != expected_dtype:
        raise ValueError(f"❌ {name} dtype 不符：{arr.dtype}，期望 {expected_dtype}")
    print(f"✅ {name} 形状 {arr.shape} dtype {arr.dtype}")

n_field = len(feature_cols)
check_shape_dtype("x_train_2", x_train_2, (len(train_df), n_field), np.int64)
check_shape_dtype("x_val_2",   x_val_2,   (len(val_df), n_field),   np.int64)
check_shape_dtype("x_train_5", x_train_5, (len(train_df), n_field), np.int64)
check_shape_dtype("x_val_5",   x_val_5,   (len(val_df), n_field),   np.int64)
check_shape_dtype("y_train",   y_train,   (len(train_df),),         np.float32)
check_shape_dtype("y_val",     y_val,     (len(val_df),),           np.float32)

# 校验3：每个字段编号范围：0 ≤ val < vocab_size
def check_code_range(x_mat, vocab_dict, feature_cols):
    for i,col in enumerate(feature_cols):
        vs = vocab_dict[col]
        vsize = len(vs)+1
        col_data = x_mat[:,i]
        min_c = col_data.min()
        max_c = col_data.max()
        if min_c >=0 and max_c < vsize:
            continue
        else:
            print(f"❌ 字段 {col} 越界：min={min_c},max={max_c},vocab_size={vsize}")
            return False
    return True

if check_code_range(x_train_2, vocab2, feature_cols) and check_code_range(x_val_2, vocab2, feature_cols):
    print("✅ min_count=2 编码范围校验通过")
else:
    ok_flag=False
if check_code_range(x_train_5, vocab5, feature_cols) and check_code_range(x_val_5, vocab5, feature_cols):
    print("✅ min_count=5 编码范围校验通过")
else:
    ok_flag=False

# 校验4：阈值5保留类别数 ≤ 阈值2
# 校验5：阈值5的0比例 ≥ 阈值2
for f in feature_cols:
    row2 = stats_df[(stats_df["field"]==f) & (stats_df["min_count"]==2)].iloc[0]
    row5 = stats_df[(stats_df["field"]==f) & (stats_df["min_count"]==5)].iloc[0]
    if row5["kept_categories"] > row2["kept_categories"]:
        print(f"❌ {f}：min5保留类别 > min2")
        ok_flag=False
    if row5["train_zero_rate"] < row2["train_zero_rate"]:
        print(f"❌ {f}：{f} min5训练零率 小于 min2")
        ok_flag=False
    if row5["val_zero_rate"] < row2["val_zero_rate"]:
        print(f"❌ {f}：{f} min5验证零率 小于 min2")
        ok_flag=False


# 校验6：同一字段严格未见率相等
for f in feature_cols:
    row2 = stats_df[(stats_df["field"]==f) & (stats_df["min_count"]==2)].iloc[0]
    row5 = stats_df[(stats_df["field"]==f) & (stats_df["min_count"]==5)].iloc[0]
    if not np.isclose(row2["val_strict_unseen_rate"], row5["val_strict_unseen_rate"]):
        print(f"❌ {f} 严格未见率不一致")
        ok_flag=False

if ok_flag:
    print("\n🎉 全部校验通过！")
else:
    print("\n⚠️ 存在校验失败项，请检查")

# -------------------------- 第六步：保存文件 + 重载一致性校验 --------------------------
def save_and_check(out_dir, vocab_dict, x_train, y_train, x_val, y_val, min_count):
    # 构造feature_config
    save_config = {
        "feature_cols": feature_cols,
        "min_count": min_count,
        "vocab_sizes": {col: len(vocab_dict[col]) + 1 for col in feature_cols},
        "note": "hour_of_day由hour最后两位生成；0代表缺失/低频/未登录类别"
    }
    # 保存词表json
    with open(os.path.join(out_dir, "feature_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    # 保存配置json
    with open(os.path.join(out_dir, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump(save_config, f, ensure_ascii=False, indent=2)
    # 保存npz
    np.savez_compressed(
        os.path.join(out_dir, "encoded_features.npz"),
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val
    )
    print(f"✅ {min_count} 保存完成，目录：{out_dir}")

    # 重载核对
    reload_npz = np.load(os.path.join(out_dir, "encoded_features.npz"))
    rx_train = reload_npz["x_train"]
    ry_train = reload_npz["y_train"]
    rx_val = reload_npz["x_val"]
    ry_val = reload_npz["y_val"]

    # 数组完全相等
    if np.array_equal(rx_train, x_train) and np.array_equal(rx_val, x_val):
        print(f"✅ {min_count} 重载npz编码矩阵一致")
    else:
        print(f"❌ {min_count} 重载编码矩阵不一致")

    if np.allclose(ry_train, y_train) and np.allclose(ry_val, y_val):
        print(f"✅ {min_count} 重载标签一致")
    else:
        print(f"❌ {min_count} 重载标签不一致")

    # 重载词表，重新编码验证是否和原编码一样
    with open(os.path.join(out_dir, "feature_vocab.json"), "r", encoding="utf-8") as f:
        reload_vocab = json.load(f)
    re_x_val = encode_df(val_df, feature_cols, reload_vocab)
    if np.array_equal(re_x_val, x_val):
        print(f"✅ {min_count} 重载词表二次编码结果一致")
    else:
        print(f"❌ {min_count} 重载词表二次编码不一致")


# 分别保存两套
save_and_check(out_dir_2, vocab2, x_train_2, y_train, x_val_2, y_val, min_count=2)
save_and_check(out_dir_5, vocab5, x_train_5, y_train, x_val_5, y_val, min_count=5)

# 统计词表大小总和
sum_vocab2 = sum([len(vocab2[c]) + 1 for c in feature_cols])
sum_vocab5 = sum([len(vocab5[c]) + 1 for c in feature_cols])
print(f"\n==== 汇总信息 ====")
print(f"min_count=2 全部字段词表size之和：{sum_vocab2}")
print(f"min_count=5 全部字段词表size之和：{sum_vocab5}")

# -------------------------- 保存后交叉校验：两组标签一致性 + 与原始CSV核对 --------------------------
print("\n==== 保存后交叉校验 ====")

npz2 = np.load(os.path.join(out_dir_2, "encoded_features.npz"))
npz5 = np.load(os.path.join(out_dir_5, "encoded_features.npz"))

# 1. 两组之间标签逐元素比对
assert np.array_equal(npz2["y_train"], npz5["y_train"]), "两组 y_train 不一致"
assert np.array_equal(npz2["y_val"], npz5["y_val"]), "两组 y_val 不一致"
print("✅ 两组 y_train / y_val 完全一致")

# 2. 重新从原始CSV读标签，核对，不复用内存变量
raw_train_df = pd.read_csv(gzip.open(train_csv_in, "rt", encoding="utf-8"), dtype=str)
raw_val_df = pd.read_csv(gzip.open(val_csv_in, "rt", encoding="utf-8"), dtype=str)
raw_y_train = raw_train_df["click"].astype(np.float32).values
raw_y_val = raw_val_df["click"].astype(np.float32).values

assert np.array_equal(npz2["y_train"], raw_y_train), "min_count=2 y_train 与原始CSV不符"
assert np.array_equal(npz5["y_train"], raw_y_train), "min_count=5 y_train 与原始CSV不符"
assert np.array_equal(npz2["y_val"], raw_y_val), "min_count=2 y_val 与原始CSV不符"
assert np.array_equal(npz5["y_val"], raw_y_val), "min_count=5 y_val 与原始CSV不符"
print("✅ 两组保存的标签与原始CSV 逐元素一致")

# 3. 样本顺序对齐（行数一致；编码矩阵可不同，不强求相等）
assert npz2["x_train"].shape[0] == npz5["x_train"].shape[0] == len(raw_train_df), "训练集行数不对齐"
assert npz2["x_val"].shape[0] == npz5["x_val"].shape[0] == len(raw_val_df), "验证集行数不对齐"
print("✅ 两组样本行数对齐，顺序同源")
print("\n🎉 保存后交叉校验全部通过")
