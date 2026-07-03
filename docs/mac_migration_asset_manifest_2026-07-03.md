# Mac Migration Asset Manifest - 2026-07-03

This manifest records the non-Git assets that must be copied separately when moving the project to macOS.

## Summary

| Path | Approx size | Git status | Migration role |
| --- | ---: | --- | --- |
| `outputs/` | 3962 MB | ignored | Experiment checkpoints, reports, generated figures |
| `models/` | 2831 MB | partially ignored | Local HF models and code-side model definitions |
| `external/` | 822 MB | partially ignored | EmoAva code, checkpoints, data, DECA assets |
| `data/` | 200 MB | partially tracked | Manual data, weak-supervision data, cleaned Weibo corpus |
| `.venv/` | 123 MB | ignored | Do not migrate; recreate on macOS |

## Required For Core Text-To-Expression Pipeline

Copy these to the same relative paths on the Mac:

| Source path | Reason |
| --- | --- |
| `models/Chinese-Emotion-Small/` | Required for `hf_local` text-only frontend and `rules_v4` experiments |
| `data/full_samples_804.json` | Main 804-sample experiment dataset |
| `data/manual_samples_8class_304.json` | Manual 8-class target/evaluation dataset |
| `data/manual_samples_5class_190.json` | Manual 5-class target/evaluation dataset |
| `data/weibo_expression_samples_8class_800.json` | 8-class weakly supervised Weibo expression dataset |
| `outputs/*.pt` needed by current experiments | Expression mapper and text frontend checkpoints |

## Required For Reproducing Weibo Data Construction

| Source path | Reason |
| --- | --- |
| `data/weibo_cleaned.json` | Cleaned Weibo corpus; ignored by Git |
| `data/微博文本情感分析数据-数据集/` | Raw segmented Weibo corpus |
| `data/weibo_emotion_mined_8class_800.json` | Mined 8-class Weibo emotion samples |
| `data/weibo_expression_samples_8class_800.json` | Generated weak-supervision labels |

## Required For EmoAva / Official 3D Backend

Copy these only when the core pipeline is already working on macOS:

| Source path | Reason |
| --- | --- |
| `external/EmoAva/checkpoints/` | Official EmoAva model checkpoint |
| `external/EmoAva/dataset/` | EmoAva train/dev/test data |
| `external/EmoAva/src/data/` | DECA/FLAME assets |
| `external/EmoAva/resource/` | Visualization examples |
| `models/bert-base-cased/` | Local BERT model used by some EmoAva experiments |
| `outputs/when_words_smile_*` | Reproduction and prior-fusion experiment checkpoints |

## Largest Known Files

| File | Approx size |
| --- | ---: |
| `models/Chinese-Emotion-Small/model.safetensors` | 1064 MB |
| `external/EmoAva/checkpoints/model.chkpt` | 578 MB |
| `outputs/when_words_smile_repro/train_smoke/model_smoke.chkpt` | 577 MB |
| `models/bert-base-cased/tf_model.h5` | 502 MB |
| `models/bert-base-cased/pytorch_model.bin` | 416 MB |
| `models/bert-base-cased/model.safetensors` | 416 MB |
| `models/bert-base-cased/flax_model.msgpack` | 413 MB |
| `external/EmoAva/dataset/train_stage1_exps.pkl` | 128 MB |
| `data/weibo_cleaned.json` | 103 MB |

## Copy Strategy

Recommended first transfer:

```text
models/Chinese-Emotion-Small/
data/full_samples_804.json
data/manual_samples_8class_304.json
data/manual_samples_5class_190.json
data/weibo_expression_samples_8class_800.json
selected outputs/*.pt checkpoints
```

Recommended second transfer:

```text
outputs/
models/bert-base-cased/
external/EmoAva/checkpoints/
external/EmoAva/dataset/
external/EmoAva/src/data/
data/weibo_cleaned.json
data/微博文本情感分析数据-数据集/
```

Do not transfer:

```text
.venv/
__pycache__/
.git/objects as a manual copy unless you are intentionally moving the whole repository folder
```

