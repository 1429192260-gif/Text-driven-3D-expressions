import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", default="outputs/when_words_smile_repro/test_parallel_full.pt")
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--output", default="docs/when_words_smile_param_error_analysis.md")
    args = parser.parse_args()

    pred = torch.load(args.prediction, map_location="cpu").numpy()
    gold = load_pickle(Path(args.dataset_dir) / "test_stage1_exps.pkl")

    sum_abs = np.zeros(53, dtype=np.float64)
    count = 0
    for i, gold_i in enumerate(gold):
        g = np.asarray(gold_i, dtype=np.float32)
        usable = min(len(g), pred.shape[1])
        g = g[:usable]
        p = pred[i, :usable]
        sum_abs += np.abs(p - g).sum(axis=0)
        count += len(g)
    mae_dim = sum_abs / count
    ranked = sorted(enumerate(mae_dim), key=lambda x: x[1], reverse=True)

    groups = {
        "exp_0_9": range(0, 10),
        "exp_10_19": range(10, 20),
        "exp_20_29": range(20, 30),
        "exp_30_39": range(30, 40),
        "exp_40_49": range(40, 50),
        "jaw_50_52": range(50, 53),
    }
    lines = ["# When Words Smile Parameter Error Analysis", ""]
    lines.extend(["## Group MAE", ""])
    for name, indices in groups.items():
        val = float(np.mean([mae_dim[i] for i in indices]))
        lines.append(f"- {name}: `{val:.6f}`")
    lines.extend(["", "## Top Error Dimensions", ""])
    for idx, val in ranked[:15]:
        lines.append(f"- dim {idx}: `{val:.6f}`")
    lines.extend(["", "## Lowest Error Dimensions", ""])
    for idx, val in ranked[-10:]:
        lines.append(f"- dim {idx}: `{val:.6f}`")

    out = Path(args.output)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
