import argparse
import pickle
from pathlib import Path

import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


def split_summary(dataset_dir: Path, split: str) -> dict:
    texts = load_pickle(dataset_dir / f"{split}_stage1_text.pkl")
    exps = load_pickle(dataset_dir / f"{split}_stage1_exps.pkl")
    lengths = np.array([len(x) for x in exps], dtype=np.float32)
    text_words = np.array([len(t.split()) for t in texts], dtype=np.float32)
    all_params = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1, 53) for x in exps], axis=0)
    energy = np.linalg.norm(all_params, axis=-1)
    return {
        "count": len(texts),
        "sequence_length": describe(lengths),
        "text_words": describe(text_words),
        "param_mean_abs": float(np.abs(all_params).mean()),
        "param_std": float(all_params.std()),
        "energy": describe(energy),
        "first_text": texts[0],
        "first_shape": tuple(np.asarray(exps[0]).shape),
    }


def format_desc(desc: dict[str, float]) -> str:
    return (
        f"min={desc['min']:.2f}, p25={desc['p25']:.2f}, mean={desc['mean']:.2f}, "
        f"median={desc['median']:.2f}, p75={desc['p75']:.2f}, max={desc['max']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--output", default="docs/emoava_dataset_summary.md")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    lines = ["# EmoAva Dataset Summary", ""]
    for split in ["train", "dev", "test"]:
        summary = split_summary(dataset_dir, split)
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Count: `{summary['count']}`",
                f"- First shape: `{summary['first_shape']}`",
                f"- First text: {summary['first_text']}",
                f"- Sequence length: `{format_desc(summary['sequence_length'])}`",
                f"- Text words: `{format_desc(summary['text_words'])}`",
                f"- Param mean abs: `{summary['param_mean_abs']:.6f}`",
                f"- Param std: `{summary['param_std']:.6f}`",
                f"- Frame energy: `{format_desc(summary['energy'])}`",
                "",
            ]
        )

    mean = np.load(dataset_dir / "stage1_mean.npy")
    std = np.load(dataset_dir / "stage1_std.npy")
    lines.extend(
        [
            "## Normalization",
            "",
            f"- stage1_mean shape: `{mean.shape}`",
            f"- stage1_std shape: `{std.shape}`",
            f"- mean abs avg: `{float(np.abs(mean).mean()):.6f}`",
            f"- std avg: `{float(std.mean()):.6f}`",
            "",
        ]
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
