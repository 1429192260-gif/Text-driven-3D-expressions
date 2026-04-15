# Text-Driven 3D Expressions

本项目研究中文文本驱动的 3D 面部表情参数生成，当前已经形成一条比较完整的实验链路：

- 受控映射：`text + emotion + intensity -> expression params`
- 文本直推两阶段管线：`text -> emotion/intensity -> expression params`
- 微博弱监督数据构建与扩充实验
- 跨域训练与泛化评估
- 文本前端误差分析与案例分析

项目已经不再是单一的 MLP 训练脚本，而是一个围绕“文本到表情参数”任务展开的数据、训练、评估和分析仓库。

## 1. 项目结构

### 根目录核心文件

- `README.md`：项目说明
- `requirements.txt`：Python 依赖
- `rule_mapping.py`：情绪到表情参数的规则先验
- `visualize.py`：简单的表情可视化

### 主要目录

- `scripts/`：训练、预测、评估、数据构建、误差分析脚本
- `models/`：表情映射器、文本前端分类器，以及本地前端模型目录
- `utils/`：文本编码、语义特征、Hugging Face 本地前端等工具
- `data/`：人工数据、微博候选数据、弱监督样本和训练集
- `docs/`：实验总结、跨域结果、前端对比和误差分析文档
- `outputs/`：训练得到的 checkpoint 和指标文件

## 2. 方法概览

### 2.1 表情参数预测器

当前支持两类主模型：

- `models/mlp_mapper.py`
  直接将文本特征与控制特征拼接后映射到表情参数。

- `models/prior_fusion_mapper.py`
  在文本特征和控制特征之外引入规则先验 `rule_mapping.py`，学习“先验 + 残差修正”的表达方式。

### 2.2 文本前端

文本前端用于从原始文本中预测情绪类别，再估计强度，供下游表情参数模型使用。当前支持两种方式：

- 学习式前端：`models/text_only_classifier.py`
- 本地 Hugging Face 前端：`utils/hf_emotion_frontend.py`

### 2.3 额外文本特征

项目除了文本编码器 embedding，还支持手工语义特征：

- `utils/text_encoder.py`
- `utils/text_features.py`

这些特征会用于：

- 表情参数映射实验
- 文本前端分类实验
- 前端消融分析

## 3. 数据组织

目前数据已经分成多层，不建议再把它理解成“一个 `train.json`”。

### 3.1 手工标注主数据

- `data/full_samples_804.json`
- `data/manual_samples_5class_190.json`
- `data/manual_samples_8class_304.json`

这些文件用于主实验、严格测试集或跨域目标域评估。

### 3.2 微博候选与清洗数据

- `data/weibo_cleaned.json`
- `data/weibo_candidate_240.json`
- `data/weibo_candidate_refined.json`

这些文件对应从微博语料中筛选候选样本的中间过程。

### 3.3 微博弱监督/自动挖掘数据

- `data/weibo_emotion_mined_150.json`
- `data/weibo_emotion_mined_500.json`
- `data/weibo_emotion_mined_8class_800.json`
- `data/weibo_expression_samples_108.json`
- `data/weibo_expression_samples_500.json`
- `data/weibo_expression_samples_8class_800.json`

这些文件用于数据扩充和跨域实验。

### 3.4 向量化训练集

- `data/train_240.json`
- `data/train_304.json`
- `data/train_412.json`
- `data/train_804.json`
- `data/train_weak_only.json`

这些文件适合直接喂给表情参数模型训练脚本。

## 4. 数据构建链路

当前微博实验的数据链路大致是下面这条：

1. 预处理语料
   - `scripts/prepare_weibo_corpus.py`

2. 从语料中筛候选文本
   - `scripts/select_weibo_candidates.py`
   - `scripts/refine_weibo_candidates.py`

3. 自动挖掘情绪样本
   - `scripts/mine_weibo_emotion_samples.py`
   - `scripts/mine_weibo_emotion_samples_500.py`
   - `scripts/mine_weibo_emotion_samples_8class_800.py`

4. 生成表情数据集
   - `scripts/build_weibo_expression_dataset.py`
   - `scripts/build_weibo_expression_dataset_500.py`
   - `scripts/build_weibo_expression_dataset_8class_800.py`

5. 将参数字典转为训练向量
   - `scripts/build_dataset.py`

如果你只关心最终训练，不需要从头跑完整链路，可以直接使用 `data/` 里已经生成好的目标数据文件。

## 5. 主要实验脚本

### 5.1 表情参数预测实验

- `scripts/train_mlp.py`
  单次训练一个表情参数模型，支持：
  - `mlp`
  - `prior_fusion`
  - 是否启用语义特征
  - 是否启用规则先验

- `scripts/run_experiments.py`
  批量运行主实验，对比：
  - Baseline MLP
  - Semantic MLP
  - Prior Only
  - Prior-Fusion

### 5.2 文本前端实验

- `scripts/train_text_only.py`
  训练文本前端情绪分类器。

- `scripts/text_only_predict.py`
  文本直推两阶段预测：
  `text -> predicted emotion/intensity -> expression params`

- `scripts/evaluate_text_only_pipeline.py`
  评估文本前端带来的误差，并与“上界控制输入”进行比较。

### 5.3 分析脚本

- `scripts/case_analysis_text_only.py`
- `scripts/per_emotion_error_analysis_text_only.py`

这类脚本用于查看：

- 前端失败案例
- 不同情绪类别上的误差分布
- 文本前端替换后对下游表情参数的影响

### 5.4 跨域实验

- `scripts/train_cross_domain_expression.py`

用于：

- 在微博弱监督数据上训练
- 在人工标注数据上测试
- 评估跨域泛化能力

## 6. 推荐运行方式

### 6.1 环境准备

- Python 3.10+
- 建议使用虚拟环境

安装依赖：

```bash
pip install -r requirements.txt
```

### 6.2 运行主表情实验

```bash
python scripts/run_experiments.py \
  --data-path data/full_samples_804.json \
  --split-mode source_holdout \
  --summary-path docs/experiment_summary_804_source_holdout.md
```

说明：

- `source_holdout` 是当前更可信的划分方式
- 这个设置会尽量让测试集保留人工样本，避免增强数据泄漏到测试阶段

### 6.3 训练文本前端

```bash
python scripts/train_text_only.py \
  --data-path data/full_samples_804.json \
  --epochs 30 \
  --batch-size 16 \
  --run-name text_only_base \
  --split-mode source_holdout
```

### 6.4 受控预测

```bash
python scripts/predict.py \
  --checkpoint outputs/prior_fusion_full_mapper.pt \
  --text "今天总算顺利了一次，心里一下轻松了不少。" \
  --emotion happy \
  --intensity 0.6 \
  --no-vis
```

### 6.5 文本直推预测

```bash
python scripts/text_only_predict.py \
  --text "完全没想到会是这个结果，我一下愣住了。" \
  --frontend-backend learned \
  --classifier-checkpoint outputs/text_only_base_classifier.pt \
  --expression-checkpoint outputs/prior_fusion_full_mapper.pt \
  --no-vis
```

### 6.6 跨域实验示例

```bash
python scripts/train_cross_domain_expression.py \
  --train-data-path data/weibo_expression_samples_500.json \
  --test-data-path data/manual_samples_5class_190.json \
  --run-name weibo500_to_manual190_priorfusion \
  --model-type prior_fusion
```

## 7. 建议优先阅读的结果文档

如果只看关键结果，建议从下面几份文档开始：

- `docs/experiment_summary_804_source_holdout.md`
  主实验对比，适合回答“当前最有效的表情参数模型是什么”。

- `docs/text_only_frontend_ablation_summary.md`
  文本前端方案和设置对比，适合回答“文本前端选哪种更合适”。

- `docs/text_only_vs_upper_bound_804.json`
  文本直推两阶段管线与上界控制输入的差距。

- `docs/cross_domain_weibo500_to_manual190.md`
  微博弱监督数据训练到人工数据测试的跨域结果。

- `docs/cross_domain_weibo800_to_manual304_8class.md`
  8 类情绪场景下的跨域结果。

## 8. 输出文件说明

训练和评估结果主要写入 `outputs/` 与 `docs/`：

- `outputs/*_mapper.pt`
  表情参数模型 checkpoint

- `outputs/*_classifier.pt`
  文本前端模型 checkpoint

- `outputs/*_metrics.json`
  训练或测试指标

- `docs/*.md`
  适合直接写进论文或报告的实验摘要

- `docs/*.json`
  更细的结构化评估结果

## 9. 当前默认结论

基于仓库中当前的实验组织，建议默认采用：

- 表情参数模型：`prior_fusion`
- 数据划分：`source_holdout`
- 文本前端：学习式前端作为主线，对比本地 HF 前端
- 结果汇报：优先报告人工测试集上的表现，不只看随机划分结果

这是一个实验仓库，不是产品仓库。读代码时更合理的顺序是：

1. 先看 `README`
2. 再看 `scripts/run_experiments.py`
3. 然后看 `scripts/train_text_only.py` 和 `scripts/evaluate_text_only_pipeline.py`
4. 最后再进入 `docs/` 看具体结论
