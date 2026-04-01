# Text-to-Expression MLP

This project maps input text, emotion labels, and intensity values to a fixed facial-expression parameter vector.

## Project Structure

- `scripts/build_dataset.py`: converts `data/raw_data.json` into `data/train.json`
- `scripts/train_mlp.py`: trains the MLP model and saves weights to `outputs/mlp_mapper.pt`
- `scripts/predict.py`: runs inference and prints predicted expression parameters
- `models/mlp_mapper.py`: MLP model definition
- `utils/param_utils.py`: parameter vector conversion and clamping helpers
- `visualize.py`: simple matplotlib-based face visualization

## Environment

- Python 3.10+
- Recommended: create and activate a virtual environment before installing dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

Raw data is expected in `data/raw_data.json`. Each sample should contain:

```json
{
  "text": "Ê¾ÀýÎÄ±¾",
  "emotion": "happy",
  "intensity": 0.8,
  "params": {
    "mouthSmileLeft": 0.6,
    "mouthSmileRight": 0.6
  }
}
```

After preprocessing, `scripts/build_dataset.py` writes `data/train.json`, where `params` is converted into a fixed-length `param_vector`.

## Workflow

1. Build the training dataset:

```bash
python scripts/build_dataset.py
```

2. Train the model:

```bash
python scripts/train_mlp.py
```

3. Run prediction:

```bash
python scripts/predict.py
```

## Notes

- The encoder model used by training and prediction is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Trained weights are written to `outputs/mlp_mapper.pt`.
- `outputs/` is ignored by Git, so trained model files are not committed by default.
- `test_params.py` is a simple connectivity check for the Hugging Face mirror endpoint.
