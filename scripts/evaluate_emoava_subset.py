import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def pad_or_truncate(seq: np.ndarray, max_len: int, dim: int = 53) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((max_len, dim), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=bool)
    usable = min(len(seq), max_len)
    if usable:
        out[:usable] = np.asarray(seq[:usable], dtype=np.float32)
        mask[:usable] = True
    return out, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pred = torch.load(args.prediction, map_location="cpu")
    if pred.ndim != 3:
        raise SystemExit(f"Expected prediction tensor [N,T,53], got {tuple(pred.shape)}")
    pred_np = pred.numpy().astype(np.float32)
    n, t, dim = pred_np.shape
    max_len = args.max_len or t

    dataset_dir = Path(args.dataset_dir)
    gold_path = dataset_dir / f"{args.split}_stage1_exps.pkl"
    text_path = dataset_dir / f"{args.split}_stage1_text.pkl"
    gold = load_pickle(gold_path)[:n]
    texts = load_pickle(text_path)[:n]

    gold_padded = []
    masks = []
    for seq in gold:
        padded, mask = pad_or_truncate(np.asarray(seq), max_len, dim)
        gold_padded.append(padded)
        masks.append(mask)
    gold_np = np.stack(gold_padded)
    mask_np = np.stack(masks)
    pred_np = pred_np[:, :max_len, :]

    abs_err = np.abs(pred_np - gold_np)
    mae = float(abs_err[mask_np].mean())
    rmse = float(np.sqrt(((pred_np - gold_np) ** 2)[mask_np].mean()))
    smoothness = float(np.abs(np.diff(pred_np, axis=1)).mean())
    acceleration = float(np.abs(pred_np[:, 2:] - 2 * pred_np[:, 1:-1] + pred_np[:, :-2]).mean())

    lines = [
        "# When Words Smile Subset Evaluation",
        "",
        f"- Prediction: `{args.prediction}`",
        f"- Dataset: `{dataset_dir}`",
        f"- Split: `{args.split}`",
        f"- Samples: `{n}`",
        f"- Sequence length evaluated: `{max_len}`",
        "",
        "## Metrics",
        "",
        f"- MAE: `{mae:.6f}`",
        f"- RMSE: `{rmse:.6f}`",
        f"- Smoothness: `{smoothness:.6f}`",
        f"- Acceleration: `{acceleration:.6f}`",
        "",
        "## First Samples",
        "",
    ]
    for i, text in enumerate(texts[:5]):
        lines.append(f"{i + 1}. {text}")

    report = "\n".join(lines)
    print(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
