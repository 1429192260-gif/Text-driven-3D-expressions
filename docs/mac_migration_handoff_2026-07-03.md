# Mac Migration Handoff - 2026-07-03

这份文档用于把当前 Windows 工作区安全迁移到 Mac，并尽量保留项目上下文、研究判断、实验路线和后续工作记忆。它不是普通安装说明，而是一次完整交接清单。

## 0. 迁移目标

目标不是单纯把文件复制到 Mac，而是保证后续可以继续研究：

- 能在 Mac 上跑通中文文本到表情参数预测主线。
- 能继续复现实验结果和查看已有实验报告。
- 能保留当前研究脉络，包括方法演化、数学模型设计思想、关键结论和待改进方向。
- 能把高风险的 3D 官方渲染链与主线实验分开迁移，避免一开始被 CUDA / PyTorch3D / DECA 环境卡住。

## 1. 当前项目状态快照

- 当前分支：`codex/3d-expression-backend`
- 远端仓库：`git@github.com:1429192260-gif/Text-driven-3D-expressions.git`
- 最近提交：`65a63bc Add numpy compatibility for chumpy`
- 当前工作区不是完全干净状态，存在已修改和未跟踪文件。
- `.gitignore` 已忽略大资产目录，例如 `outputs/`、`models/Chinese-Emotion-Small/`、`models/bert-base-cased/`、`external/EmoAva/checkpoints/`、`external/EmoAva/dataset/`。

重要提醒：只在 Mac 上 `git clone` 不会拿到所有模型、checkpoint、微博清洗数据和 EmoAva 资产。

## 2. 当前研究记忆包

项目核心任务：

```text
中文文本 -> 情绪类别/强度前端 -> 表情参数预测后端 -> 3D 表情生成/可视化
```

主线数学抽象：

```text
x: 文本
e: 情绪类别
a: 情绪强度
y: 22 维表情参数向量
```

受控后端任务：

```text
f(x, e, a) -> y
```

text-only 两阶段任务：

```text
g(x) -> e_hat
q(x, e_hat) -> a_hat
f(x, e_hat, a_hat) -> y_hat
```

目前最重要的方法判断：

- `Baseline MLP` 是无结构先验的直接函数逼近。
- `Semantic MLP` 的本质是输入表示增强，即加入中文语义 cue 特征 `s(x)`。
- `Prior-Fusion` 的本质是把直接回归改写为 `规则先验 + 残差学习`：

```text
y_hat = p(e, a) + h_theta(phi(x), control, p(e, a))
```

- `rules_v4` 的本质不是简单规则堆叠，而是对预训练中文情绪模型输出做任务空间对齐、强度校准和局部决策修正。
- 微博数据扩展是弱监督伪标签扩展，收益是扩大文本分布覆盖，风险是标签噪声和规则偏差。
- `source_holdout` 是比随机切分更可信的评估方式，因为它尽量让测试集保留人工样本，避免增强数据泄漏。

## 3. 关键代码地图

核心后端：

- `models/mlp_mapper.py`：Baseline MLP。
- `models/prior_fusion_mapper.py`：Prior-Fusion 后端。
- `scripts/train_mlp.py`：受控表情参数预测训练主入口。
- `scripts/run_experiments.py`：后端四组对比实验入口。
- `rule_mapping.py`：情绪/强度到表情参数的规则先验。
- `utils/param_utils.py`：表情参数字典与 22 维向量互转。

文本前端：

- `models/text_only_classifier.py`：学习式 text-only 情绪分类器。
- `scripts/train_text_only.py`：训练 text-only 前端。
- `scripts/text_only_predict.py`：文本直推完整管线。
- `scripts/evaluate_text_only_pipeline.py`：比较 text-only 与 upper-bound。
- `utils/hf_emotion_frontend.py`：本地 `Chinese-Emotion-Small` + `rules_v4` 前端。

数据构建：

- `scripts/prepare_weibo_corpus.py`：微博语料清洗。
- `scripts/mine_weibo_emotion_samples_8class_800.py`：8 类微博样本挖掘。
- `scripts/build_weibo_expression_dataset_8class_800.py`：弱监督微博表情参数构建。
- `scripts/train_cross_domain_expression.py`：微博训练到人工样本测试的跨域实验。

EmoAva / 3D 后端：

- `external/EmoAva/`：外部 EmoAva 代码和数据。
- `scripts/train_emotion_prior_fusion_v3.py` 到 `scripts/train_mixture_gate_v6.py`：后续 3D 表达后端实验。
- `scripts/render_emoava_official_deca.py`：官方 DECA/FLAME 渲染相关，高迁移风险。

## 4. 迁移资产分层

### 4.1 必须通过 Git 迁移

这些应尽量提交到 Git，或至少明确带到 Mac：

- `scripts/`
- `models/*.py`
- `utils/`
- `rule_mapping.py`
- `README.md`
- `requirements.txt`
- `docs/*.md`
- 小型核心数据文件，如 `data/full_samples_804.json`、`data/train_804.json`

### 4.2 必须单独拷贝的大资产

这些通常被 `.gitignore` 忽略，不能依赖 Git：

- `models/Chinese-Emotion-Small/`
- `models/bert-base-cased/`
- `outputs/` 中需要复现实验的 checkpoint，尤其 `.pt` 和 `.chkpt`
- `external/EmoAva/checkpoints/`
- `external/EmoAva/dataset/`
- `external/EmoAva/src/data/`
- `data/weibo_cleaned.json`

### 4.3 不建议迁移

- `.venv/`：Mac 上重新创建。
- `__pycache__/`
- `.idea/` 和 `.vscode/` 可选。
- 大量中间 `outputs/` 如果后续不用，可以先不迁移。

## 5. Windows 端收尾清单

1. 确认当前工作区状态：

```bash
git status --short
```

2. 决定哪些未提交文件要纳入迁移。当前重点关注：

```text
models/text_only_classifier.py
scripts/evaluate_text_only_pipeline.py
scripts/train_text_only.py
utils/hf_emotion_frontend.py
utils/text_features.py
scripts/build_weibo_expression_dataset_8class_800.py
scripts/mine_weibo_emotion_samples_8class_800.py
scripts/train_cross_domain_expression.py
docs/chat_history_summary_2026-04-17.md
docs/text_only_frontend_ablation_summary.md
docs/text_only_frontend_ablation_strict.md
docs/cross_domain_weibo800_to_manual304_8class.md
data/manual_samples_5class_190.json
data/manual_samples_8class_304.json
data/weibo_emotion_mined_8class_800.json
data/weibo_expression_samples_8class_800.json
```

3. 如果确认这些是有效研究成果，建议提交：

```bash
git add <selected files>
git commit -m "Add migration-ready experiment state"
git push
```

4. 导出大资产清单：

```powershell
Get-ChildItem models,outputs,data,external -Recurse -File |
  Select-Object FullName,Length |
  Sort-Object Length -Descending |
  Export-Csv migration_asset_inventory.csv -NoTypeInformation -Encoding UTF8
```

5. 单独打包大资产。建议先只打包最小可运行资产：

```text
models/Chinese-Emotion-Small/
outputs/<当前要复现实验的 checkpoint>
data/full_samples_804.json
data/manual_samples_8class_304.json
data/weibo_expression_samples_8class_800.json
```

## 6. Mac 端环境构建清单

### 6.1 基础工具

安装 Xcode Command Line Tools：

```bash
xcode-select --install
```

建议安装 Homebrew：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

建议安装 Git、Python 管理工具：

```bash
brew install git pyenv
```

### 6.2 Python 版本建议

推荐先用 Python 3.10，原因：

- 和当前项目中外部 EmoAva 依赖更接近。
- 对 `chumpy`、旧版 `numpy`、`torchvision` 兼容性更稳。
- 避免 Python 3.12 带来的旧包兼容风险。

安装：

```bash
pyenv install 3.10.14
pyenv local 3.10.14
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 6.3 主项目核心环境

先安装主线依赖：

```bash
pip install torch torchvision torchaudio
pip install matplotlib requests sentence-transformers transformers numpy scipy tqdm scikit-learn
```

如果只跑文本到表情参数主线，通常不需要先安装 EmoAva 的全部依赖。

### 6.4 EmoAva 独立环境

如果要跑 `external/EmoAva` 或官方 3D 渲染，建议单独建一个环境：

```bash
python -m venv .venv-emoava
source .venv-emoava/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r external/EmoAva/src/requirements.txt
```

注意：

- `pytorch3d` 在 Mac 上可能需要特殊处理。
- 自定义 CUDA rasterizer 在 Mac 上基本不能按原 CUDA 方式编译。
- 官方 DECA/FLAME 渲染应作为第二阶段，不要作为主线迁移验收项。

## 7. Mac 端项目恢复流程

1. 克隆代码：

```bash
git clone git@github.com:1429192260-gif/Text-driven-3D-expressions.git
cd Text-driven-3D-expressions
git checkout codex/3d-expression-backend
```

2. 还原大资产到相同相对路径：

```text
models/Chinese-Emotion-Small/
models/bert-base-cased/
outputs/
external/EmoAva/checkpoints/
external/EmoAva/dataset/
external/EmoAva/src/data/
```

3. 确认关键文件存在：

```bash
test -d models/Chinese-Emotion-Small
test -f data/full_samples_804.json
test -d outputs
```

4. 先做 import smoke test：

```bash
python -m py_compile scripts/train_mlp.py
python -m py_compile scripts/train_text_only.py
python -m py_compile scripts/text_only_predict.py
python -m py_compile utils/hf_emotion_frontend.py
```

5. 验证核心包：

```bash
python - <<'PY'
import torch
import transformers
import sentence_transformers
import matplotlib
print("torch", torch.__version__)
print("mps available", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
print("transformers", transformers.__version__)
print("ok")
PY
```

## 8. Mac 端主线验收脚本

先用低成本验证，不要一上来跑完整训练。

### 8.1 受控预测验收

```bash
python scripts/predict.py \
  --checkpoint outputs/prior_fusion_full_mapper.pt \
  --text "今天终于顺利了一次，心里轻松了不少。" \
  --emotion happy \
  --intensity 0.6 \
  --no-vis
```

如果 checkpoint 名称不同，需要改成实际存在的 `.pt` 文件。

### 8.2 text-only 前端验收

```bash
python scripts/text_only_predict.py \
  --text "完全没想到会是这个结果，我一下子愣住了。" \
  --frontend-backend hf_local \
  --expression-checkpoint outputs/prior_fusion_full_mapper.pt \
  --no-vis
```

### 8.3 评估脚本验收

```bash
python scripts/evaluate_text_only_pipeline.py \
  --data-path data/full_samples_804.json \
  --frontend-backend hf_local \
  --hf-mode rules_v4 \
  --expression-checkpoint outputs/prior_fusion_full_mapper.pt \
  --output-path docs/mac_migration_smoke_text_only.json
```

### 8.4 小规模训练验收

```bash
python scripts/train_mlp.py \
  --data-path data/full_samples_240.json \
  --epochs 1 \
  --batch-size 4 \
  --run-name mac_smoke \
  --model-type prior_fusion \
  --split-mode source_holdout
```

## 9. 风险和处理策略

### 9.1 Git 不包含大资产

风险：Mac 上代码到了，但模型和 checkpoint 不在。

处理：按第 4 节单独拷贝大资产，且保持相对路径一致。

### 9.2 Python 版本不兼容

风险：Python 3.12 对旧 `numpy/chumpy` 不友好。

处理：优先 Python 3.10。

### 9.3 Apple Silicon 与 PyTorch

风险：部分脚本只判断 CUDA，不自动用 MPS。

处理：主线可以先 CPU 跑通；后续再考虑把设备选择扩展为 `cuda -> mps -> cpu`。

### 9.4 EmoAva 官方 3D 渲染

风险：CUDA rasterizer、PyTorch3D、DECA 相关依赖在 Mac 上成本高。

处理：迁移验收先不要求官方 3D 渲染；主线稳定后再单独开任务。

### 9.5 中文路径与文件名

风险：Mac 终端、Git、压缩包解压时可能出现中文路径编码问题。

处理：尽量用 UTF-8 压缩工具；核心代码和脚本保持英文路径；中文资料可作为文档资产迁移。

## 10. 后续研究路线记忆

下一步最值得继续推进的研究不是盲目加模型，而是明确改哪个数学对象：

- 改 `phi(x)`：换文本表示或更好中文语义编码。
- 改 `s(x)`：更系统地设计中文情绪 cue。
- 改 `g(x)`：提升 text-only 情绪前端。
- 改 `q(x,e)`：把 intensity 从规则估计升级为可学习或半监督估计。
- 改 `p(e,a)`：更细粒度的表情先验模板。
- 改 `h_theta`：继续优化 Prior-Fusion 残差建模。
- 改训练分布：更稳地利用微博弱监督和人工样本。
- 改评估：增加更严格的 ablation、跨域、按情绪类别分析。

推荐学习/讲解顺序：

```text
Baseline MLP
-> Semantic MLP
-> Prior-Fusion
-> text-only front-end
-> HF + mapping + intensity calibration
-> rules_v4
-> weakly supervised Weibo expansion
-> source_holdout and cross-domain evaluation
-> EmoAva / 3D backend integration
```

## 11. Mac 迁移完成标准

迁移可以认为完成，当且仅当以下项目通过：

- Git 代码能在 Mac 上 checkout 到正确分支。
- 核心大资产已还原到相同相对路径。
- Python 3.10 虚拟环境可创建并安装核心依赖。
- `py_compile` 通过核心脚本。
- `predict.py` 能用已有 checkpoint 输出表情参数。
- `text_only_predict.py` 能使用 `hf_local` 或 `learned` 前端完成预测。
- 至少一个 1 epoch smoke training 能跑完。
- 至少一个评估脚本能写出 JSON/Markdown 结果。
- 当前研究记忆文档已保留在 `docs/`，Mac 上可继续查看。

