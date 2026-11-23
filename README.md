# Avazu 广告点击率预估

给定一条广告曝光记录，预测用户点击的概率。项目实现了常数基线、类别LR、FM和DeepFM，按时间顺序训练与评估，并通过正则化、结构消融、低频阈值对照和误差切片分析模型表现。

## 数据与环境

原始数据来自 [Kaggle Avazu Click-Through Rate Prediction](https://www.kaggle.com/competitions/avazu-ctr-prediction/data)。将带 `click` 标签的 `train.gz` 放在项目根目录下的 `data/avazu/raw/train.gz`。抽样脚本按每块10万行读取，以 `0.0025` 的概率独立保留每条记录，随机种子为42，共获得100,747条开发样本。

| 数据集 | 日期（2014年） | 样本数 | 用途 |
| --- | --- | ---: | --- |
| 训练集 | 10月21日～28日 | 80,651 | 建词表、训练模型、计算常数基线 |
| 验证集 | 10月29日 | 9,571 | 比较方案、误差切片与概率校准 |
| 最终留出集 | 10月30日 | 10,525 | 评估冻结后的方案 |

每条记录使用22个类别字段，其中 `hour_of_day` 由 `hour` 的末两位生成。词表仅根据训练集构建，低频、未见和缺失类别映射到可学习的0号位置。

从仓库根目录运行下面的命令。建议使用 Python 3.12，并在安装依赖后执行脚本。记录本页结果时使用 Python 3.12.9、PyTorch 2.14.1+cu126、NumPy 2.5.2、pandas 3.0.6 和 scikit-learn 1.9.1。训练与评估在 CPU 上运行。

```bash
python -m venv .venv
# Windows Git Bash
source .venv/Scripts/activate
# macOS / Linux 使用 source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 从原始数据运行

准备 `train.gz` 后，依次执行下列命令。

```bash
# 流式抽样并保存样本
python -X utf8 02_sample_data.py

# 按日期划分并计算常数基线
python -X utf8 03_time_split_baseline.py

# 构建原阈值2词表，生成训练和验证编码
python -X utf8 04_encode_features.py

# 在统一训练设置下训练FM和完整DeepFM，保存第5轮权重
python -X utf8 10_train_fm_deepfm.py

# 生成阈值2与阈值5各自的词表、编码与类别统计
python -X utf8 12_rare_category_compare.py

# 在两套编码上分别训练类别LR
python -X utf8 13_train_rare_category_lr.py

# 使用两个LR做验证集误差切片和概率分桶
python -X utf8 14_validate_rare_category_lr.py

# 先复算验证指标，再评估冻结方案的最终留出表现
python -X utf8 15_final_holdout_evaluation.py
```

## 使用已有产物复算最终指标

已有下列文件时，从仓库根目录直接运行最终评估。

```bash
python -X utf8 15_final_holdout_evaluation.py
```

所有输入均位于 `data/avazu/processed/` 下。评估脚本读取 `train.csv.gz`、`val.csv.gz` 和 `holdout.csv.gz`。其中训练标签用于计算常数预测概率，验证CSV用于重编码并校验原有指标。

| 方案 | 权重 | 配套编码产物 |
| --- | --- | --- |
| FM | `fm_deepfm_compare/fm_epoch5.pt` | `processed/` 根目录中的原阈值2产物 |
| 阈值5类别LR | `rare_category_compare/min_count_5/lr_epoch5.pt` | `rare_category_compare/min_count_5/` 下的阈值5产物 |
| 完整DeepFM | `fm_deepfm_compare/deepfm_epoch5.pt` | `processed/` 根目录中的原阈值2产物，模型使用 `use_fm2=True` |
| 常数基线 | 训练集点击率约为0.172657 | 每条样本使用同一个预测概率 |

每套编码产物包含 `feature_config.json`、`feature_vocab.json` 和 `encoded_features.npz`。配置提供字段顺序和含0号位置的词表大小，词表用于重编码，NPZ用于逐元素核对验证编码。脚本会检查留出日期、样本数以及留出ID与训练、验证ID的交集。

输出写入 `data/avazu/processed/final_holdout_eval/`。`config.json` 记录评估方案，`metrics.json` 记录两份数据的指标，`predictions.csv` 保存留出样本ID、真实标签及四组预测概率。数据与权重需在本地通过上述步骤生成，仓库的 `/data/` 目录由 `.gitignore` 排除。

## 结果

在打开留出集之前，依据固定第5轮的验证LogLoss选择FM作为主模型。各方案使用同一批验证记录和同一批留出记录，主指标为LogLoss，辅助指标为ROC-AUC和Accuracy@0.5。

| 方案 | 验证LogLoss ↓ | 验证AUC ↑ | 验证Acc ↑ | 留出LogLoss ↓ | 留出AUC ↑ | 留出Acc ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FM | **0.389312** | **0.728148** | 0.843695 | **0.404425** | **0.723975** | 0.835629 |
| 阈值5类别LR | 0.393683 | 0.720809 | 0.842545 | 0.408212 | 0.718205 | **0.836485** |
| 完整DeepFM | 0.405970 | 0.709393 | 0.839620 | 0.415877 | 0.712646 | 0.831544 |
| 常数基线 | 0.433471 | 0.500000 | **0.844321** | 0.448728 | 0.500000 | 0.834584 |

FM在10月30日的留出LogLoss比验证集高0.015113，在四组中仍最低。四组按LogLoss的排序一致。结果对应这次时间划分、模型配置和固定权重。

## 一次实际排错

在 [二阶项消融实验](11_deepfm_ablation.py) 的首次运行中，完整组和消融组传入了同一份正则参数列表。消融组日志里出现了非零正则损失，但列表里的22张Embedding表都属于完整组，消融组自身的Embedding没有收到这项梯度。这使两组同时在二阶项开关和实际L2作用上产生差异，首轮结果因此标为无效对照。

修复时为两组分别构造正则参数列表，并在训练入口通过参数对象的 `id` 检查归属。重新按相同协议训练后，检查两组惩罚梯度均作用于本组Embedding，并重载两组末轮权重复算验证指标。修复后的第5轮，去掉二阶项的组LogLoss略低，完整组AUC略高，两个指标未显示一致方向的收益。
