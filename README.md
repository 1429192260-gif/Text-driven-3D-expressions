# Text-to-Expression MLP

This project maps input text to a fixed facial-expression parameter vector. The current workflow supports both:
- controlled prediction: `text + emotion + intensity -> params`
- text-only pipeline: `text -> emotion/intensity -> params`

## Project Structure

- `scripts/build_dataset.py`: converts raw samples into train format
- `scripts/train_mlp.py`: trains the expression-parameter predictor
- `scripts/run_experiments.py`: runs ablation experiments for the parameter predictor
- `scripts/train_text_only.py`: trains a text-only emotion + intensity classifier
- `scripts/predict.py`: runs controlled inference with provided emotion and intensity
- `scripts/text_only_predict.py`: runs text-only inference and automatically predicts emotion/intensity first
- `models/mlp_mapper.py`: baseline parameter mapper
- `models/prior_fusion_mapper.py`: prior-fusion parameter mapper
- `models/text_only_classifier.py`: joint emotion/intensity classifier
- `utils/param_utils.py`: parameter vector conversion and clamping helpers
- `visualize.py`: simple matplotlib-based face visualization

## Environment

- Python 3.10+
- Recommended: create and activate a virtual environment before installing dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

## Main Data Files

- `data/full_samples_804.json`: raw expression samples with source metadata
- `data/train_804.json`: vectorized parameter training data
- `data/text_only_labels_804.json`: text-only labels for emotion and intensity

## Workflows

### 1. Train the expression predictor

```bash
python scripts/run_experiments.py --data-path data/full_samples_804.json --split-mode source_holdout --summary-path docs/experiment_summary_804_source_holdout.md
```

### 2. Train the text-only classifier

```bash
python scripts/train_text_only.py --data-path data/full_samples_804.json --epochs 30 --batch-size 16 --run-name text_only_base --split-mode source_holdout
```

### 3. Controlled prediction

```bash
python scripts/predict.py --checkpoint outputs/prior_fusion_full_mapper.pt --text "今天总算顺利了一次，心里一下轻松了不少。" --emotion happy --intensity 0.6 --no-vis
```

### 4. Text-only prediction

```bash
python scripts/text_only_predict.py --text "完全没想到会是这个结果，我一下愣住了。" --classifier-checkpoint outputs/text_only_base_classifier.pt --expression-checkpoint outputs/prior_fusion_full_mapper.pt --no-vis
```

## Notes

- The encoder falls back to `hashing-char-256` if the local sentence-transformer checkpoint is unavailable.
- `source_holdout` is the more trustworthy split mode for paper experiments because the test set can be kept manual-only.
- Trained weights are written to `outputs/`.
