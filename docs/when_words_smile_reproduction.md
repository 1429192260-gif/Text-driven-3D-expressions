# When Words Smile 复现记录

## 目标

复现论文 `When Words Smile: Generating Diverse Emotional Facial Expressions from Text` 的基础流程，并为后续加入中文情绪先验分支做准备。

当前复现优先级：

1. 获取官方代码和 README。
2. 准备官方数据目录结构。
3. 跑通数据读取与格式检查。
4. 在官方 checkpoint 可用时跑推理与验证。
5. 再考虑训练完整模型。

## 官方资源

- 论文页：https://aclanthology.org/2025.emnlp-main.1374/
- PDF：https://aclanthology.org/2025.emnlp-main.1374.pdf
- 官方项目页：https://walkermitty.github.io/EmoAva
- 官方 GitHub：https://github.com/WalkerMitty/EmoAva
- 数据与 baseline model：https://drive.google.com/drive/folders/1YqNRyk4QliBTpaz8hQl_4_HgQgB6zssK

## 重要限制

官方 README 说明，训练集访问需要填写 `license_agreement.pdf` 并邮件申请，同时还需要获得 MELD 和 MEMOR 数据集权限。

因此复现应分两层：

- `light reproduction`：使用官方已发布的测试集、预训练模型或样例结果，跑通推理、验证、可视化。
- `full reproduction`：拿到完整训练集后，按官方训练流程复现训练结果。

## 官方数据目录

官方代码期望数据放在：

```text
external/EmoAva/dataset/
```

需要文件：

```text
train_stage1_text.pkl
train_stage1_exps.pkl
test_stage1_text.pkl
test_stage1_exps.pkl
dev_stage1_text.pkl
dev_stage1_exps.pkl
stage1_mean.npy
stage1_std.npy
```

其中：

- `*_text.pkl` 是文本列表。
- `*_exps.pkl` 是对应的 3D expression parameter sequence。
- `stage1_mean.npy` 与 `stage1_std.npy` 用于参数标准化/反标准化。

## 官方运行命令

### 环境

```shell
conda create -n cteg python=3.10
conda activate cteg
cd external/EmoAva/src
pip install -r requirements.txt
```

可视化需要额外安装 PyTorch3D，并准备 FLAME 相关资源。

### 训练

编辑：

```text
external/EmoAva/src/train.sh
```

设置 `pretrained_path` 为本地 `bert-base-cased` 路径，然后运行：

```shell
bash train.sh
```

### 推理

编辑：

```text
external/EmoAva/src/infer.sh
```

设置：

```text
-model "path/to/output/model.chkpt"
-tokenizer_path "bert-base-cased"
-data_source "path/to/EmoAva/dataset"
```

然后运行：

```shell
bash infer.sh
```

### 验证

先在 `infer.sh` 中设置 `infer_mode` 为 `p`，生成：

```text
para_result.pt
```

再运行：

```shell
python evaluate.py \
  --para_predict para_result.pt \
  --split test \
  --tokenizer_path bert-base-cased
```

官方验证脚本会输出 continuous PPL metric。

### 可视化

```shell
python visualize.py \
  --exp_path ../dataset/test_stage1_exps.pkl \
  --output all_videos
```

可视化需要从 BFM_to_FLAME 获取：

```text
FLAME_albedo_from_BFM.npz
```

并放到：

```text
external/EmoAva/src/data/
```

## 当前本地状态

官方代码、数据、checkpoint 已经放入本地并完成第一轮轻量复现。

本地路径：

```text
external/EmoAva
external/EmoAva/src
external/EmoAva/dataset
external/EmoAva/checkpoints/model.chkpt
models/bert-base-cased
```

数据检查结果：

```text
train: 12000
dev: 1500
test: 1500
expression sequence shape: T x 53
```

本地环境：

```text
conda env: cteg
python: 3.10
torch: 2.2.2
transformers: 4.39.3
```

已完成官方小样本推理：

```shell
conda run -n cteg python translate.py ^
  -model "..\checkpoints\model.chkpt" ^
  -tokenizer_path "..\..\..\models\bert-base-cased" ^
  -save_path "..\..\..\outputs\when_words_smile_repro" ^
  -data_source "..\dataset" ^
  -save_name "result_5_parallel.pt" ^
  -max_seq_len 64 ^
  -src_len 128 ^
  -batch_size 2 ^
  -infer_mode "p" ^
  -seed 42 ^
  -cvae ^
  -no_cuda
```

输出文件：

```shell
outputs/when_words_smile_repro/result_5_serial.pt
outputs/when_words_smile_repro/result_5_parallel.pt
```

轻量评价结果：

```text
Samples: 5
Sequence length evaluated: 63
MAE: 0.402966
RMSE: 0.574025
Smoothness: 0.366579
Acceleration: 0.609487
```

结果文档：

```text
docs/when_words_smile_repro_subset_eval.md
```

兼容性修改：

- `external/EmoAva/src/translate.py` 中 `torch.load` 增加 `weights_only=False`，用于兼容 PyTorch 2.6+ 的 checkpoint 加载规则。

早期网络问题记录：

GitHub HTTPS clone 在本机连接失败；SSH 认证成功，但 clone 超时。后续由用户手动下载 zip 解压。

## 下一步

## 完整 test 集复现结果

已完成完整 `test` 集并行推理：

```shell
conda run -n cteg python translate.py ^
  -model "..\checkpoints\model.chkpt" ^
  -tokenizer_path "..\..\..\models\bert-base-cased" ^
  -save_path "..\..\..\outputs\when_words_smile_repro" ^
  -data_source "..\dataset" ^
  -save_name "test_parallel_full.pt" ^
  -max_seq_len 256 ^
  -src_len 128 ^
  -batch_size 64 ^
  -infer_mode "p" ^
  -seed 42 ^
  -cvae ^
  -no_cuda
```

输出：

```text
outputs/when_words_smile_repro/test_parallel_full.pt
shape: 1500 x 255 x 53
```

轻量全量评价：

```text
Samples: 1500
Sequence length evaluated: 255
MAE: 0.379475
RMSE: 0.563167
Smoothness: 0.386786
Acceleration: 0.663121
```

结果文档：

```text
docs/when_words_smile_repro_full_eval.md
```

## 官方 PPL 评价状态

官方 `evaluate.py` 在 Windows 中文用户名路径下使用 `joblib` 多进程时，会触发 ASCII 编码问题：

```text
UnicodeEncodeError: 'ascii' codec can't encode character
```

已将 `n_jobs=16` 临时改为 `n_jobs=1`，但单进程计算 official continuous PPL 在 CPU 上超过 30 分钟仍未完成。

后续如果需要官方 PPL，可选路线：

1. 将项目复制到纯英文路径后恢复 `n_jobs=16` 再跑。
2. 重写 PPL 计算为更快的向量化/批处理版本。
3. 在 Linux 或无中文路径环境下运行官方评价脚本。

## 后续

1. 优先基于当前完整推理结果做可视化和案例分析。
2. 再开始加入中文情绪先验分支。
3. 如导师要求严格复现官方 PPL，再单独处理评价脚本性能问题。

## 复现后如何接入我们的改进

先复现官方 text-to-expression sequence：

```text
text -> text encoder -> expression parameter sequence
```

再加入我们的中文情绪先验分支：

```text
text -> Chinese-Emotion-Small + rules_v4 -> emotion prior
```

最终形成：

```text
text embedding + emotion prior + temporal profile
  -> prior-guided sequence generator
  -> expression parameter sequence
```

对比实验：

1. 官方/简化 `When Words Smile` baseline。
2. `text + emotion/intensity`。
3. `text + rules_v4 prior`。
4. `text + rules_v4 prior + temporal profile`。
