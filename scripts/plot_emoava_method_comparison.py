import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_METHODS = {
    "gold": None,
    "baseline": "outputs/when_words_smile_repro/test_parallel_full.pt",
    "v2": "outputs/when_words_smile_prior_v2/test_prior_adapter.pt",
    "v3_full": "outputs/when_words_smile_prior_v3/test_prior_fusion.pt",
    "v3_no_emotion": "outputs/when_words_smile_prior_v3/test_prior_fusion_no_emotion.pt",
    "v3_no_intensity": "outputs/when_words_smile_prior_v3/test_prior_fusion_no_intensity.pt",
}


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_predictions(repo: Path, method_paths: dict[str, str | None]) -> dict[str, np.ndarray]:
    predictions = {}
    for name, rel_path in method_paths.items():
        if rel_path is None:
            continue
        path = repo / rel_path
        if path.exists():
            predictions[name] = torch.load(path, map_location="cpu").numpy().astype(np.float32)
    return predictions


def sequence_mae(pred: np.ndarray, gold: np.ndarray) -> float:
    usable = min(len(pred), len(gold))
    return float(np.abs(pred[:usable] - gold[:usable]).mean())


def frame_mae(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    usable = min(len(pred), len(gold))
    return np.abs(pred[:usable] - gold[:usable]).mean(axis=-1)


def energy(seq: np.ndarray, length: int) -> np.ndarray:
    return np.linalg.norm(seq[:length], axis=-1)


def pick_cases(predictions: dict[str, np.ndarray], gold: list[np.ndarray], target: str, count: int):
    baseline = predictions["baseline"]
    candidate = predictions[target]
    rows = []
    for i, gold_i in enumerate(gold[: min(len(gold), baseline.shape[0], candidate.shape[0])]):
        gold_np = np.asarray(gold_i, dtype=np.float32)
        base_mae = sequence_mae(baseline[i], gold_np)
        cand_mae = sequence_mae(candidate[i], gold_np)
        rows.append((i, base_mae, cand_mae, base_mae - cand_mae))
    rows.sort(key=lambda x: x[3], reverse=True)
    improved = rows[:count]
    worsened = list(reversed(rows[-count:]))
    return improved, worsened, rows


def plot_case(
    case_idx: int,
    text: str,
    gold: np.ndarray,
    predictions: dict[str, np.ndarray],
    methods: list[str],
    output: Path,
):
    usable = min(len(gold), *(predictions[name][case_idx].shape[0] for name in methods))
    frames = np.arange(usable)
    gold = gold[:usable]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(frames, energy(gold, usable), label="gold", color="black", linewidth=2.4)
    for name in methods:
        axes[0].plot(frames, energy(predictions[name][case_idx], usable), label=name, linewidth=1.8)
    axes[0].set_title(f"Case {case_idx + 1}: {text[:100]}")
    axes[0].set_ylabel("L2 expression energy")
    axes[0].legend(ncol=3, fontsize=8)

    for name in methods:
        axes[1].plot(frames, frame_mae(predictions[name][case_idx], gold), label=name, linewidth=1.8)
    axes[1].set_ylabel("frame MAE")
    axes[1].legend(ncol=3, fontsize=8)

    labels = methods
    values = [sequence_mae(predictions[name][case_idx], gold) for name in labels]
    colors = ["#9aa0a6", "#6aaed6", "#7ccba2", "#f4a261", "#c77dff"][: len(labels)]
    axes[2].bar(labels, values, color=colors)
    axes[2].set_ylabel("case MAE")
    axes[2].tick_params(axis="x", rotation=20)
    for x, y in zip(labels, values):
        axes[2].text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    texts: list[str],
    improved: list[tuple[int, float, float, float]],
    worsened: list[tuple[int, float, float, float]],
    target: str,
):
    lines = [
        "# V3 Method Comparison Case Analysis",
        "",
        f"- Target method: `{target}`",
        "- Improvement is computed as `baseline_case_mae - target_case_mae`.",
        "",
        "## Most Improved Cases",
        "",
        "| Case | Baseline MAE | Target MAE | Improvement | Text |",
        "|---:|---:|---:|---:|---|",
    ]
    for idx, base, cand, improvement in improved:
        text = texts[idx].replace("|", "/")
        lines.append(f"| {idx + 1} | {base:.6f} | {cand:.6f} | {improvement:.6f} | {text[:120]} |")
    lines.extend(
        [
            "",
            "## Weak Or Regressed Cases",
            "",
            "| Case | Baseline MAE | Target MAE | Improvement | Text |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for idx, base, cand, improvement in worsened:
        text = texts[idx].replace("|", "/")
        lines.append(f"| {idx + 1} | {base:.6f} | {cand:.6f} | {improvement:.6f} | {text[:120]} |")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--output-dir", default="outputs/when_words_smile_prior_v3/comparison_plots")
    parser.add_argument("--target", default="v3_no_emotion")
    parser.add_argument("--num-cases", type=int, default=5)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dataset_dir = repo / args.dataset_dir
    texts = load_pickle(dataset_dir / "test_stage1_text.pkl")
    gold = [np.asarray(x, dtype=np.float32) for x in load_pickle(dataset_dir / "test_stage1_exps.pkl")]
    predictions = load_predictions(repo, DEFAULT_METHODS)
    if "baseline" not in predictions:
        raise SystemExit("Missing baseline prediction.")
    if args.target not in predictions:
        raise SystemExit(f"Missing target prediction: {args.target}")

    methods = [name for name in ["baseline", "v2", "v3_full", args.target, "v3_no_intensity"] if name in predictions]
    methods = list(dict.fromkeys(methods))
    improved, worsened, _ = pick_cases(predictions, gold, args.target, args.num_cases)
    output_dir = repo / args.output_dir
    write_summary(output_dir, texts, improved, worsened, args.target)

    for rank, (idx, _, _, _) in enumerate(improved, start=1):
        plot_case(idx, texts[idx], gold[idx], predictions, methods, output_dir / f"improved_{rank:02d}_case_{idx + 1:04d}.png")
    for rank, (idx, _, _, _) in enumerate(worsened, start=1):
        plot_case(idx, texts[idx], gold[idx], predictions, methods, output_dir / f"regressed_{rank:02d}_case_{idx + 1:04d}.png")
    print(output_dir / "README.md")


if __name__ == "__main__":
    main()
