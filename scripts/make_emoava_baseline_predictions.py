import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


def pad_or_truncate(arr: np.ndarray, max_length: int, dim: int = 53) -> np.ndarray:
    padded = np.zeros((max_length, dim), dtype=np.float32)
    if arr.shape[0] >= max_length:
        padded[:max_length] = arr[:max_length]
    else:
        padded[: arr.shape[0]] = arr
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["random", "shuffle", "mean"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-len", type=int, default=255)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "external" / "EmoAva" / "src"))
    import em_dataloader

    em_dataloader.random = random
    load_dataset_split_batch = em_dataloader.load_dataset_split_batch

    baseline_arg = None if args.baseline == "mean" else args.baseline
    _, exps, _ = load_dataset_split_batch(args.split, -1, baseline_arg)
    if args.baseline == "mean":
        mean = np.load(repo / "external" / "EmoAva" / "dataset" / "stage1_mean.npy")
        pred = np.stack([np.tile(mean.reshape(1, 53), (args.max_len, 1)) for _ in exps])
    else:
        pred = np.stack([pad_or_truncate(np.asarray(x[1:], dtype=np.float32), args.max_len) for x in exps])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(pred, dtype=torch.float32), output)
    print(f"saved {args.baseline}: {output} {pred.shape}")


if __name__ == "__main__":
    main()
