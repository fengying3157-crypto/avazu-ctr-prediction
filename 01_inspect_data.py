# 01_inspect_data.py
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent
data_path = project_root / "data" / "avazu" / "raw" / "train.gz"

df = pd.read_csv(data_path, nrows=1000, dtype=str, compression="gzip")

print("=" * 60)
print("Avazu数据集预览【仅前1000行，不代表全量数据】")
print("=" * 60)

# 1. 行列数、全部列名、前5行
rows, cols = df.shape
print(f"\n1. 基础信息【仅前1000行】")
print(f"样本行数：{rows}，字段数：{cols}")
print("全部列名：")
print(df.columns.tolist())
print("\n前5行：")
print(df.head())

# 2. click取值与数量
print(f"\n2. click分布【仅前1000行】")
print(df["click"].value_counts())

# 3. hour样例、最小值、最大值
hour_series = pd.to_numeric(df["hour"])
print(f"\n3. hour字段【仅前1000行】")
print("hour样例：", df["hour"].head().tolist())
print(f"hour最小值：{hour_series.min()}, hour最大值：{hour_series.max()}")

# 4. 各列缺失数量
print(f"\n4. 各列缺失值数量【仅前1000行】")
print(df.isna().sum())

# 5. 各列不同取值数量
print(f"\n5. 各列唯一值数量【仅前1000行】")
print(df.nunique())
