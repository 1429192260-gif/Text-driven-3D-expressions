import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


PARAM_GROUPS = {
    "exp_0_9": list(range(0, 10)),
    "exp_10_19": list(range(10, 20)),
    "exp_20_29": list(range(20, 30)),
    "exp_30_39": list(range(30, 40)),
    "jaw_50_52": list(range(50, 53)),
}


def group_energy(seq: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.linalg.norm(seq[:, indices], axis=-1)


def plot_case(case_idx: int, text: str, pred: np.ndarray, gold: np.ndarray, output_dir: Path):
    usable = min(len(gold), len(pred))
    pred = pred[:usable]
    gold = gold[:usable]
    frames = np.arange(usable)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    pred_energy = np.linalg.norm(pred, axis=-1)
    gold_energy = np.linalg.norm(gold, axis=-1)
    axes[0].plot(frames, gold_energy, label="gold", linewidth=2)
    axes[0].plot(frames, pred_energy, label="prediction", linewidth=2)
    axes[0].set_title(f"Case {case_idx + 1}: {text[:90]}")
    axes[0].set_ylabel("overall energy")
    axes[0].legend()

    for name, indices in PARAM_GROUPS.items():
        axes[1].plot(frames, group_energy(pred, indices), label=name)
    axes[1].set_ylabel("pred group energy")
    axes[1].legend(ncol=3, fontsize=8)

    error = np.abs(pred - gold).mean(axis=-1)
    axes[2].plot(frames, error, color="#c44e52", linewidth=2)
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("frame MAE")

    output = output_dir / f"case_{case_idx + 1:02d}.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_baseline_summary(predictions: dict[str, np.ndarray], output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, pred in predictions.items():
        energy = np.linalg.norm(pred, axis=-1).mean(axis=0)
        ax.plot(energy, label=name, linewidth=2)
    ax.set_title("Average expression energy over time")
    ax.set_xlabel("frame")
    ax.set_ylabel("mean L2 energy")
    ax.legend()
    output = output_dir / "baseline_energy_summary.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", default="outputs/when_words_smile_repro/test_parallel_full.pt")
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--output-dir", default="outputs/when_words_smile_repro/plots")
    parser.add_argument("--num-cases", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = torch.load(args.prediction, map_location="cpu").numpy()
    dataset_dir = Path(args.dataset_dir)
    texts = load_pickle(dataset_dir / "test_stage1_text.pkl")
    gold = load_pickle(dataset_dir / "test_stage1_exps.pkl")

    outputs = []
    for i in range(min(args.num_cases, len(texts))):
        outputs.append(plot_case(i, texts[i], pred[i], np.asarray(gold[i]), output_dir))

    baseline_paths = {
        "ours_checkpoint": Path("outputs/when_words_smile_repro/test_parallel_full.pt"),
        "mean": Path("outputs/when_words_smile_repro/baseline_mean.pt"),
        "random": Path("outputs/when_words_smile_repro/baseline_random.pt"),
        "shuffle": Path("outputs/when_words_smile_repro/baseline_shuffle.pt"),
    }
    available = {
        name: torch.load(path, map_location="cpu").numpy()
        for name, path in baseline_paths.items()
        if path.exists()
    }
    if available:
        outputs.append(plot_baseline_summary(available, output_dir))

    report = output_dir / "README.md"
    report.write_text(
        "# When Words Smile Reproduction Plots\n\n"
        + "\n".join(f"- `{p.name}`" for p in outputs)
        + "\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
