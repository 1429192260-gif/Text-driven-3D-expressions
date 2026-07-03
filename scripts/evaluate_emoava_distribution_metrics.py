import argparse
import csv
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch


DEFAULT_METHODS = {
    "baseline": "outputs/when_words_smile_repro/test_parallel_full.pt",
    "v4_global": "outputs/when_words_smile_prior_v4/test_uncertainty_prior_fusion_no_prior_branch.pt",
    "v5_full": "outputs/when_words_smile_prior_v5/test_learned_affect_prior_fusion.pt",
    "v6_frame": "outputs/when_words_smile_prior_v6/test_mixture_gate.pt",
    "v6_sample": "outputs/when_words_smile_prior_v6/test_mixture_gate_sample.pt",
}


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_prediction(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu")
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def parse_methods(raw: str, repo: Path) -> dict[str, Path]:
    methods = {}
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in DEFAULT_METHODS:
            known = ", ".join(DEFAULT_METHODS)
            raise KeyError(f"Unknown method `{name}`. Available methods: {known}")
        methods[name] = repo / DEFAULT_METHODS[name]
    if not methods:
        raise ValueError("No methods were selected.")
    return methods


def align_prediction(pred: np.ndarray, gold: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for idx, gold_seq in enumerate(gold):
        seq = np.asarray(pred[idx, : len(gold_seq)], dtype=np.float64)
        if seq.ndim != 2 or seq.shape[1] != 53:
            raise ValueError(f"Prediction {idx} has shape {seq.shape}; expected [T, 53].")
        out.append(seq)
    return out


def concat_frames(sequences: list[np.ndarray], dims: slice | np.ndarray | list[int]) -> np.ndarray:
    return np.concatenate([seq[:, dims] for seq in sequences], axis=0)


def concat_deltas(sequences: list[np.ndarray], dims: slice | np.ndarray | list[int]) -> np.ndarray:
    deltas = [np.diff(seq, axis=0)[:, dims] for seq in sequences if len(seq) > 1]
    if not deltas:
        raise ValueError("No sequence has at least two frames; dynamic metrics cannot be computed.")
    return np.concatenate(deltas, axis=0)


def sequence_stats(sequences: list[np.ndarray]) -> np.ndarray:
    features = []
    for seq in sequences:
        if len(seq) > 1:
            delta = np.diff(seq, axis=0)
            mean_abs_delta = np.mean(np.abs(delta), axis=0)
            energy = np.linalg.norm(delta, axis=1)
            energy_stats = np.array([np.mean(energy), np.std(energy)], dtype=np.float64)
        else:
            mean_abs_delta = np.zeros(seq.shape[1], dtype=np.float64)
            energy_stats = np.zeros(2, dtype=np.float64)
        features.append(
            np.concatenate(
                [
                    np.mean(seq, axis=0),
                    np.std(seq, axis=0),
                    mean_abs_delta,
                    energy_stats,
                ]
            )
        )
    return np.asarray(features, dtype=np.float64)


def frechet_distance(real: np.ndarray, fake: np.ndarray, eps: float = 1e-6) -> float:
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("Frechet inputs must be 2-D feature matrices.")
    if real.shape[1] != fake.shape[1]:
        raise ValueError(f"Feature dimensions differ: {real.shape[1]} vs {fake.shape[1]}.")

    mu_real = np.mean(real, axis=0)
    mu_fake = np.mean(fake, axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_fake = np.cov(fake, rowvar=False)

    cov_real = np.atleast_2d(cov_real) + np.eye(real.shape[1]) * eps
    cov_fake = np.atleast_2d(cov_fake) + np.eye(fake.shape[1]) * eps
    # For symmetric positive semi-definite covariances, Tr(sqrt(C1 C2)) can be
    # computed from the eigenvalues of C1 C2 without importing scipy.sqrtm.
    eigvals = np.linalg.eigvals(cov_real @ cov_fake)
    eigvals = np.real(eigvals)
    eigvals = np.maximum(eigvals, 0.0)
    trace_covmean = float(np.sum(np.sqrt(eigvals)))
    diff = mu_real - mu_fake
    value = diff @ diff + np.trace(cov_real) + np.trace(cov_fake) - 2.0 * trace_covmean
    return float(max(value, 0.0))


def paired_metrics(gold: list[np.ndarray], pred: list[np.ndarray]) -> dict[str, float]:
    rows = []
    for gold_seq, pred_seq in zip(gold, pred):
        n = min(len(gold_seq), len(pred_seq))
        gold_seq = gold_seq[:n]
        pred_seq = pred_seq[:n]
        diff = pred_seq - gold_seq
        row = {
            "mae_all53": float(np.mean(np.abs(diff))),
            "mae_expression50": float(np.mean(np.abs(diff[:, :50]))),
            "mae_jaw3": float(np.mean(np.abs(diff[:, 50:53]))),
            "rmse_all53": float(np.sqrt(np.mean(diff * diff))),
        }
        if len(gold_seq) > 1:
            delta_diff = np.diff(pred_seq, axis=0) - np.diff(gold_seq, axis=0)
            row["mae_delta_all53"] = float(np.mean(np.abs(delta_diff)))
            row["mae_delta_expression50"] = float(np.mean(np.abs(delta_diff[:, :50])))
            row["mae_delta_jaw3"] = float(np.mean(np.abs(delta_diff[:, 50:53])))
        else:
            row["mae_delta_all53"] = np.nan
            row["mae_delta_expression50"] = np.nan
            row["mae_delta_jaw3"] = np.nan
        rows.append(row)
    keys = rows[0].keys()
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}


def evaluate_method(gold: list[np.ndarray], pred: list[np.ndarray]) -> dict[str, float]:
    metrics = paired_metrics(gold, pred)
    gold = [gold_seq[: min(len(gold_seq), len(pred_seq))] for gold_seq, pred_seq in zip(gold, pred)]
    pred = [pred_seq[: len(gold_seq)] for gold_seq, pred_seq in zip(gold, pred)]

    metrics["fed_frame_all53"] = frechet_distance(concat_frames(gold, slice(None)), concat_frames(pred, slice(None)))
    metrics["fed_frame_expression50"] = frechet_distance(concat_frames(gold, slice(0, 50)), concat_frames(pred, slice(0, 50)))
    metrics["fed_frame_jaw3"] = frechet_distance(concat_frames(gold, slice(50, 53)), concat_frames(pred, slice(50, 53)))
    metrics["fdd_delta_all53"] = frechet_distance(concat_deltas(gold, slice(None)), concat_deltas(pred, slice(None)))
    metrics["fdd_delta_expression50"] = frechet_distance(concat_deltas(gold, slice(0, 50)), concat_deltas(pred, slice(0, 50)))
    metrics["fdd_delta_jaw3"] = frechet_distance(concat_deltas(gold, slice(50, 53)), concat_deltas(pred, slice(50, 53)))
    metrics["fvd_like_sequence_stats"] = frechet_distance(sequence_stats(gold), sequence_stats(pred))
    return metrics


def write_csv(path: Path, rows: list[dict]):
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], baseline: dict):
    main_cols = [
        "method",
        "mae_all53",
        "mae_delta_all53",
        "fed_frame_expression50",
        "fdd_delta_expression50",
        "fed_frame_jaw3",
        "fvd_like_sequence_stats",
    ]
    lines = [
        "# EmoAva Distribution Metrics",
        "",
        "Lower is better for every metric in this report.",
        "",
        "- `FED` follows the Frechet-distance idea behind FID, but uses 53-D FLAME expression/jaw parameters instead of Inception image features.",
        "- `FDD` is the same distribution distance on frame-to-frame deltas, so it focuses on expression dynamics.",
        "- `fvd_like_sequence_stats` compares sequence-level statistics and is a lightweight substitute for FVD when we only have 3D parameter sequences.",
        "- LSE-D/LSE-C are not reported here because the current task is text-only and has no audio stream.",
        "",
        "## Main Results",
        "",
        "| " + " | ".join(main_cols) + " |",
        "| " + " | ".join(["---"] * len(main_cols)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                row[col] if isinstance(row[col], str) else f"{row[col]:.6f}"
                for col in main_cols
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Relative Improvement Over Baseline",
            "",
            "Positive means the method is better than baseline.",
            "",
            "| method | mae_all53_rel_improve | fed_expression50_rel_improve | fdd_delta_expression50_rel_improve | sequence_stats_rel_improve |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        method = row["method"]
        values = []
        for key in [
            "mae_all53",
            "fed_frame_expression50",
            "fdd_delta_expression50",
            "fvd_like_sequence_stats",
        ]:
            if baseline[key] == 0:
                values.append(0.0)
            else:
                values.append((baseline[key] - row[key]) / baseline[key] * 100.0)
        lines.append(
            f"| {method} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% | {values[3]:.2f}% |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate paired and Frechet-style metrics for EmoAva predictions.")
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--methods", default="baseline,v4_global,v5_full,v6_frame,v6_sample")
    parser.add_argument("--output-dir", default="outputs/emoava_official_render_analysis")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dataset_dir = repo / args.dataset_dir
    gold = [np.asarray(x, dtype=np.float64) for x in load_pickle(dataset_dir / f"{args.split}_stage1_exps.pkl")]
    methods = parse_methods(args.methods, repo)

    rows = []
    for method, path in methods.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction for `{method}`: {path}")
        pred = align_prediction(load_prediction(path), gold)
        metrics = evaluate_method(gold, pred)
        rows.append({"method": method, **metrics})

    out_dir = repo / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "distribution_metrics.csv", rows)
    (out_dir / "distribution_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    baseline = next(row for row in rows if row["method"] == "baseline")
    write_markdown(out_dir / "distribution_metrics.md", rows, baseline)
    print(out_dir / "distribution_metrics.md")


if __name__ == "__main__":
    main()
