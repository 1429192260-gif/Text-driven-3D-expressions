import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def sequence_energy(seq: np.ndarray) -> np.ndarray:
    return np.linalg.norm(seq, axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pred = torch.load(args.prediction, map_location="cpu").numpy()
    dataset_dir = Path(args.dataset_dir)
    texts = load_pickle(dataset_dir / f"{args.split}_stage1_text.pkl")
    gold = load_pickle(dataset_dir / f"{args.split}_stage1_exps.pkl")

    lines = ["# When Words Smile Case Analysis", ""]
    for i in range(min(args.num_cases, len(texts), pred.shape[0])):
        gold_i = np.asarray(gold[i], dtype=np.float32)
        pred_i = pred[i, : len(gold_i)]
        mae = float(np.abs(pred_i - gold_i).mean())
        gold_energy = sequence_energy(gold_i)
        pred_energy = sequence_energy(pred_i)
        lines.extend(
            [
                f"## Case {i + 1}",
                "",
                f"- Text: {texts[i]}",
                f"- Gold length: `{len(gold_i)}`",
                f"- MAE: `{mae:.6f}`",
                f"- Gold peak frame: `{int(gold_energy.argmax())}`",
                f"- Pred peak frame: `{int(pred_energy.argmax())}`",
                f"- Gold mean energy: `{float(gold_energy.mean()):.6f}`",
                f"- Pred mean energy: `{float(pred_energy.mean()):.6f}`",
                f"- Gold max energy: `{float(gold_energy.max()):.6f}`",
                f"- Pred max energy: `{float(pred_energy.max()):.6f}`",
                "",
            ]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
