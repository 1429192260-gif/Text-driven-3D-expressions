import argparse
import pickle
from pathlib import Path

import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def describe_split(dataset_dir: Path, split: str):
    text_path = dataset_dir / f"{split}_stage1_text.pkl"
    exps_path = dataset_dir / f"{split}_stage1_exps.pkl"
    if not text_path.exists() or not exps_path.exists():
        print(f"[{split}] missing: {text_path.name} or {exps_path.name}")
        return

    texts = load_pickle(text_path)
    exps = load_pickle(exps_path)
    print(f"[{split}] texts: {len(texts)}")
    print(f"[{split}] exps:  {len(exps)}")
    if texts:
        print(f"[{split}] first text: {texts[0]}")
    if exps:
        arr = np.asarray(exps[0])
        print(f"[{split}] first exp shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"[{split}] first exp min/max: {arr.min():.6f}/{arr.max():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="external/EmoAva/dataset",
        help="Path to the official EmoAva dataset directory.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    print(f"Dataset dir: {dataset_dir.resolve()}")
    if not dataset_dir.exists():
        raise SystemExit("Dataset directory does not exist.")

    for split in ["train", "dev", "test"]:
        describe_split(dataset_dir, split)

    for name in ["stage1_mean.npy", "stage1_std.npy"]:
        path = dataset_dir / name
        if path.exists():
            arr = np.load(path)
            print(f"[{name}] shape: {arr.shape}, dtype: {arr.dtype}")
        else:
            print(f"[{name}] missing")


if __name__ == "__main__":
    main()
