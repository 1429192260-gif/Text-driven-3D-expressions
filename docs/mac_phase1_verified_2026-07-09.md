# Mac Phase 1 Verification - 2026-07-09

This document records the verified macOS migration state for the core text-to-expression pipeline.

## Scope

Verified:

- Python 3.10 project-local environment.
- Core macOS requirements.
- Minimal transfer package assets.
- Syntax compilation for core scripts.
- Controlled expression prediction.
- Text-only prediction with local Hugging Face frontend.
- One-epoch smoke training.
- Lightweight text-only pipeline evaluation.

Not included in phase 1:

- `external/EmoAva` official 3D rendering.
- DECA / FLAME / PyTorch3D / CUDA rasterizer migration.
- Global system environment changes.

## Repository

- Repository: `git@github.com:1429192260-gif/Text-driven-3D-expressions.git`
- Branch: `codex/3d-expression-backend`
- Verified HEAD: `8ba9f6779ab44281f2d4b9809ec753c2c66f4ab3`
- macOS checkout path:

```text
/Users/cyz/Documents/Codex/2026-07-09/mac-codex-windows-macos-clone-checkout/work/Text-driven-3D-expressions
```

## Environment

The host did not have a usable `python3.10` on PATH. To avoid global changes, the Mac environment was built locally in the repository:

- `.uv-bootstrap/`: temporary local bootstrap environment using system Python 3.9.
- `.uv-python/`: local CPython 3.10.20 managed by `uv`.
- `.venv/`: project virtual environment.

No Homebrew package was installed. No shell startup file or global PATH was changed.

Verified versions:

```text
Python 3.10.20
torch 2.13.0
torchvision 0.28.0
torchaudio 2.11.0
transformers 5.13.0
sentence-transformers 5.6.0
matplotlib 3.10.9
MPS available: True
```

## Transfer Assets

The transfer package was provided as an already-unpacked directory:

```text
/Users/cyz/Downloads/project_mac_transfer_minimal
```

It was synced into the repository root.

Restored core assets:

```text
models/Chinese-Emotion-Small/
outputs/prior_fusion_full_mapper.pt
outputs/prior_fusion_804_source_holdout_strict_mapper.pt
outputs/text_only_emotion8_base_classifier.pt
data/weibo_cleaned.json
SHA256SUMS.txt
```

`SHA256SUMS.txt` uses Windows path separators and CRLF line endings. Verification passed after temporary path normalization; the BOM-affected first line was verified separately.

## Verified Commands

Compile check:

```bash
.venv/bin/python -m py_compile scripts/train_mlp.py scripts/train_text_only.py scripts/text_only_predict.py scripts/evaluate_text_only_pipeline.py utils/hf_emotion_frontend.py
```

Controlled prediction:

```bash
.venv/bin/python scripts/predict.py \
  --checkpoint outputs/prior_fusion_full_mapper.pt \
  --text "今天终于顺利了一次，心里轻松了不少。" \
  --emotion happy \
  --intensity 0.6 \
  --no-vis
```

Output included:

```text
browInnerUp: 0.0639
eyeSquintLeft: 0.1651
eyeSquintRight: 0.1540
mouthSmileLeft: 0.3698
mouthSmileRight: 0.3473
```

Text-only prediction:

```bash
.venv/bin/python scripts/text_only_predict.py \
  --text "完全没想到会是这个结果，我一下子愣住了。" \
  --frontend-backend hf_local \
  --expression-checkpoint outputs/prior_fusion_full_mapper.pt \
  --no-vis
```

Output included:

```text
predicted_emotion: surprise (p=0.8481)
raw_frontend_label: surprise
estimated_intensity_value: 0.72
jawOpen: 0.3293
```

One-epoch smoke training:

```bash
.venv/bin/python scripts/train_mlp.py \
  --data-path data/full_samples_240.json \
  --epochs 1 \
  --batch-size 4 \
  --run-name mac_smoke \
  --model-type prior_fusion \
  --split-mode source_holdout
```

Output:

```text
Epoch 001 | train_loss=0.000260 | val_loss=0.000169 | val_mae=0.005059
Encoder: hashing-char-256
Best checkpoint saved to outputs/mac_smoke_mapper.pt
Metrics saved to outputs/mac_smoke_metrics.json
Test metrics | loss=0.000256 | mae=0.006266 | rmse=0.016008
```

Text-only pipeline evaluation:

```bash
.venv/bin/python scripts/evaluate_text_only_pipeline.py \
  --data-path data/full_samples_804.json \
  --frontend-backend hf_local \
  --hf-mode rules_v4 \
  --expression-checkpoint outputs/prior_fusion_full_mapper.pt \
  --output-path docs/mac_migration_smoke_text_only.json
```

Result summary:

```text
test_count: 40
test_source: manual_only
upper_bound.mae: 0.007883421145379543
upper_bound.rmse: 0.02072555013000965
upper_bound.active_mae: 0.03476075828075409
text_only.mae: 0.018926359713077545
text_only.rmse: 0.05776036158204079
text_only.active_mae: 0.07847034931182861
```

Full evaluation output:

```text
docs/mac_migration_smoke_text_only.json
```

## Working Tree Notes

Expected local-only ignored paths:

```text
.uv-bootstrap/
.uv-python/
.venv/
models/Chinese-Emotion-Small/
outputs/
data/weibo_cleaned.json
```

The transfer package contains CRLF-formatted data files. Git may show several tracked `data/*.json` files as modified even when `git diff --stat` only reports `.gitignore`. This is a line-ending artifact from the Windows transfer package. Do not normalize these files unless the team decides to commit a deliberate line-ending cleanup.

The only intentional tracked source-control edit made during Mac setup was adding these local environment directories to `.gitignore`:

```text
.uv-bootstrap/
.uv-python/
```

## Phase 1 Status

Phase 1 core pipeline migration is verified on macOS.
