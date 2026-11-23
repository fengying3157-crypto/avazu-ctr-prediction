import os
import time
import numpy as np
import gzip
import pandas as pd
from datetime import datetime
from collections import Counter
# 处理 hour 字段解析、合法性校验、提取日期
def parse_hour(s):
    try:
         dt = datetime.strptime(str(s),'%y%m%d%H')## 按 %y%m%d%H 解析
         return dt.date()
    except Exception:
        return None





rng = np.random.default_rng(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
in_gz_path = os.path.join(script_dir, "data", "avazu", "raw", "train.gz")
in_gz_path = os.path.normpath(in_gz_path)

# 输出路径
out_gz_path = os.path.join(script_dir, 'data','avazu','processed','sample.csv.gz')
out_gz_path = os.path.normpath(out_gz_path)
#每块10w行,每行以 0.0025 的概率保留。
chunk_size = 100000
sample_prob = 0.0025
max_SAMPLE_LIMIT = 200000 #选中总量超过20w就停止报错

#加入计数器，存放选中数据的列表，计时起点
#容器：保存每一块抽出来的数据
all_selected = []
#累计统计变量
total_scan_rows = 0#累计扫描行数
total_selected_rows = 0#选中行数
#记录开始时间，用来最后算耗时
start = time.time()
raw_date_counts = Counter()
#开始循环
# 写打开文件、分块循环的外层框架，先把读取、抽样、计数打印做出来，
# 暂时不加 hour 解析、上限截断报错
with gzip.open(in_gz_path,'rt',encoding='utf-8') as f:
    for chunk in pd.read_csv(f,chunksize=chunk_size,dtype = str):
        #当前块行数
        n_chunk = len(chunk)
        #累计扫描行数加上当前块，当前扫描的
        total_scan_rows += n_chunk
        #按概率取样
        mask = rng.random(size = n_chunk) < sample_prob
        selected = chunk[mask].copy()
        n_sel_prev = len(selected) # 解析hour之前的数量
        # 检查当前块的全部原始记录，而不仅是选中记录
        if not chunk["hour"].str.fullmatch(r"\d{8}", na=False).all():
            raise ValueError("hour存在缺失值或不符合8位数字格式的记录")

        parsed_hour = pd.to_datetime(
            chunk["hour"],
            format="%y%m%d%H",
            errors="coerce",
        )

        if parsed_hour.isna().any():
            raise ValueError("hour存在无法解析的日期，停止并检查")

        chunk_dates = parsed_hour.dt.strftime("%Y-%m-%d")

        # 累计原始数据每天的行数
        raw_date_counts.update(chunk_dates.value_counts().to_dict())

        # selected保留了原始索引，按索引匹配对应日期
        selected["dt_date"] = chunk_dates.loc[selected.index]
        n_sel = len(selected)
        total_selected_rows  += n_sel

        print(f'本块：扫描了{n_chunk}行,过滤前选择{n_sel_prev}，筛选后选中了{n_sel}行，累计选中：{total_selected_rows}')

        #就可以放入总表了
        all_selected.append(selected)
        #需要加入最大选择就停止
        if total_selected_rows > max_SAMPLE_LIMIT:
            raise RuntimeError(f'抽样总数{total_selected_rows},超过上限{max_SAMPLE_LIMIT},停止，不再抽文件放入')

# 用 `pd.concat` 把 all_selected 合并成一张完整大表
df_sample = pd.concat(all_selected,ignore_index=True)

#拿到抽样数据的日期集合
sample_dates = set(df_sample['dt_date'].unique())

# 【校验：使用第一次扫描收集的日期，两边均为YYYY-MM-DD字符串】
raw_dates = set(raw_date_counts)

# 判断：原始所有日期，抽样结果里是不是都存在
date_cover_ok = raw_dates.issubset((sample_dates))
if not  date_cover_ok:
    raise RuntimeError(f'检验失败：抽样数据缺少日期{sorted(raw_dates - sample_dates)}')

cost_time = time.time() - start

print('=============抽样完成汇总==========')
print(f"总扫描原始行数：{total_scan_rows}")
print(f"最终有效选中行数：{len(df_sample)}")
print(f"原始数据日期列表：{sorted(raw_dates)}")
print(f"抽样数据日期列表：{sorted(sample_dates)}")
print(f"日期覆盖校验：{'通过' if date_cover_ok else '失败'}")
print(f"总耗时：{cost_time:.2f} 秒")
print("原始数据每天的行数：", dict(sorted(raw_date_counts.items())))
print(
    "抽样数据每天的行数：",
    df_sample["dt_date"].value_counts().sort_index().to_dict(),
)
#保存压缩文件
os.makedirs(os.path.dirname(out_gz_path), exist_ok=True)
df_sample.to_csv(out_gz_path,index=False,compression='gzip')
print(f"文件已保存到：{out_gz_path}")
print("✅ 全部任务完成标志")
