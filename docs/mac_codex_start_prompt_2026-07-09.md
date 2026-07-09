# Mac Codex Start Prompt - 2026-07-09

把下面这段发给 Mac 上新开的 Codex，用于无缝接入本项目迁移任务。

```text
你现在是 Mac 端 Codex，正在和 Windows 端 Codex 协作，把一个中文文本驱动 3D 表情参数预测项目从 Windows 迁移到 macOS。

你的角色：
- 负责 Mac 端环境构建、代码拉取、迁移包解压、资产归位、运行验证和报错反馈。
- 不要从零重构项目；优先按仓库已有文档和脚本执行。
- 遇到依赖、路径、checkpoint、模型缺失问题时，先定位缺什么，再把明确缺失项反馈给 Windows 端。

请先阅读这些文件：
- docs/mac_migration_handoff_2026-07-03.md
- docs/mac_migration_asset_manifest_2026-07-03.md
- requirements-mac-core.txt
- README.md

当前 Git 状态：
- 仓库：git@github.com:1429192260-gif/Text-driven-3D-expressions.git
- 分支：codex/3d-expression-backend
- 迁移提交：7c5404e Prepare Mac migration handoff

Windows 端已经准备好的手动迁移包：
- project_mac_transfer_minimal.zip
- 大小约 797MB
- 内含：models/Chinese-Emotion-Small/、核心 data 文件、主线 outputs checkpoint、SHA256SUMS.txt、迁移文档

Mac 端第一阶段目标：
1. clone 仓库并 checkout 到 codex/3d-expression-backend。
2. 创建 Python 3.10 虚拟环境。
3. 安装 requirements-mac-core.txt。
4. 将 project_mac_transfer_minimal.zip 解压到仓库根目录，保持相对路径。
5. 跑 py_compile 验证核心脚本。
6. 跑 predict.py 验证受控表情参数预测。
7. 跑 text_only_predict.py 验证 hf_local + rules_v4 前端。
8. 跑 1 epoch smoke training 验证训练链。

建议命令：

mkdir -p ~/Projects
cd ~/Projects
git clone git@github.com:1429192260-gif/Text-driven-3D-expressions.git
cd Text-driven-3D-expressions
git checkout codex/3d-expression-backend

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-mac-core.txt

unzip ~/Downloads/project_mac_transfer_minimal.zip -d .

python -m py_compile scripts/train_mlp.py scripts/train_text_only.py scripts/text_only_predict.py scripts/evaluate_text_only_pipeline.py utils/hf_emotion_frontend.py

python scripts/predict.py \
  --checkpoint outputs/prior_fusion_full_mapper.pt \
  --text "今天终于顺利了一次，心里轻松了不少。" \
  --emotion happy \
  --intensity 0.6 \
  --no-vis

python scripts/text_only_predict.py \
  --text "完全没想到会是这个结果，我一下子愣住了。" \
  --frontend-backend hf_local \
  --expression-checkpoint outputs/prior_fusion_full_mapper.pt \
  --no-vis

python scripts/train_mlp.py \
  --data-path data/full_samples_240.json \
  --epochs 1 \
  --batch-size 4 \
  --run-name mac_smoke \
  --model-type prior_fusion \
  --split-mode source_holdout

协作方式：
- 如果成功，请汇报：Python 版本、torch 版本、MPS 是否可用、三个验证脚本结果。
- 如果失败，请不要笼统说失败。请汇报：执行的命令、完整错误栈、当前目录、缺失文件路径或缺失 Python 包名。
- 不要优先处理 external/EmoAva 官方 3D 渲染。第一阶段只验收文本到表情参数主线。
- 如果需要新资产，请明确列出相对路径，例如 outputs/xxx.pt 或 external/EmoAva/checkpoints/model.chkpt。
```

## Windows Codex 分工

Windows 端继续负责：

- 保留源工作区和迁移包。
- 根据 Mac 端反馈补发缺失模型、checkpoint、数据文件。
- 必要时更新 Git 分支和迁移文档。
- 不把 `.venv/`、完整 `outputs/`、官方 3D 大资产强行塞入 Git。

## Mac Codex 分工

Mac 端负责：

- 建环境。
- 解压资产。
- 跑 smoke tests。
- 把错误精确反馈回来。
- 第一阶段不深入修官方 DECA / PyTorch3D / CUDA rasterizer。

## 第一阶段通过标准

- 仓库能 checkout 到 `codex/3d-expression-backend`。
- `requirements-mac-core.txt` 能安装完成。
- `models/Chinese-Emotion-Small/` 存在。
- `outputs/prior_fusion_full_mapper.pt` 存在。
- `python -m py_compile` 核心脚本通过。
- `scripts/predict.py` 能输出表情参数。
- `scripts/text_only_predict.py --frontend-backend hf_local` 能输出情绪、强度和表情参数。
- `scripts/train_mlp.py --epochs 1` 能完成一次 smoke training。

