import argparse
import json
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("outputs")

RUNS = [
    {
        "run_name": "baseline_mlp",
        "title": "Baseline MLP",
        "description": "text embedding + emotion + intensity",
        "args": ["--model-type", "mlp", "--no-semantic-features", "--no-prior"],
    },
    {
        "run_name": "semantic_mlp",
        "title": "Semantic MLP",
        "description": "baseline + semantic cue features",
        "args": ["--model-type", "mlp", "--use-semantic-features", "--no-prior"],
    },
    {
        "run_name": "prior_only",
        "title": "Prior Only",
        "description": "expression prior + residual refinement without semantic cues",
        "args": ["--model-type", "prior_fusion", "--no-semantic-features", "--use-prior"],
    },
    {
        "run_name": "prior_fusion_full",
        "title": "Prior-Fusion",
        "description": "semantic features + expression prior + residual refinement",
        "args": ["--model-type", "prior_fusion", "--use-semantic-features", "--use-prior"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run all text-expression experiments.")
    parser.add_argument("--data-path", default="data/train.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["random", "source_holdout"], default="random")
    parser.add_argument("--summary-path", default="docs/experiment_summary_240.md")
    return parser.parse_args()


def run_one(config, data_path, epochs, batch_size, seed, split_mode):
    cmd = [
        sys.executable,
        "scripts/train_mlp.py",
        "--data-path", data_path,
        "--run-name", config["run_name"],
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--split-mode", split_mode,
    ] + config["args"]
    subprocess.run(cmd, check=True)


def load_metrics(run_name):
    path = OUTPUT_DIR / f"{run_name}_metrics.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value):
    return f"{value:.6f}"


def build_main_table(results):
    lines = []
    lines.append("| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for result in results:
        metrics = result["metrics"]["test_metrics"]
        lines.append(
            f"| {result['title']} | {result['description']} | {format_float(metrics['mae'])} | {format_float(metrics['rmse'])} | {format_float(metrics['active_mae'])} | {format_float(metrics['loss'])} |"
        )
    return lines


def build_group_table(results, group_key, title):
    groups = sorted({group for result in results for group in result["metrics"]["test_metrics"][group_key].keys()})
    lines = []
    lines.append(f"## {title}")
    lines.append("")
    header = "| Group | " + " | ".join(result["title"] for result in results) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(results)) + " |"
    lines.append(header)
    lines.append(sep)
    for group in groups:
        row = [group]
        for result in results:
            metrics = result["metrics"]["test_metrics"][group_key].get(group)
            value = format_float(metrics["mae"]) if metrics else "-"
            row.append(value)
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_source_note(results):
    config = results[0]["metrics"]["config"]
    split_sources = config.get("split_sources", {})
    lines = ["## Split Diagnostics", ""]
    lines.append(f"- split_mode: {config.get('split_mode', 'random')}")
    lines.append(f"- train_manual: {split_sources.get('train_manual', 0)}")
    lines.append(f"- train_augmented: {split_sources.get('train_augmented', 0)}")
    lines.append(f"- val_manual: {split_sources.get('val_manual', 0)}")
    lines.append(f"- val_augmented: {split_sources.get('val_augmented', 0)}")
    lines.append(f"- test_manual: {split_sources.get('test_manual', 0)}")
    lines.append(f"- test_augmented: {split_sources.get('test_augmented', 0)}")
    return lines


def build_summary(results, epochs, batch_size, seed, split_mode):
    lines = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append(f"- epochs: {epochs}")
    lines.append(f"- batch_size: {batch_size}")
    lines.append(f"- seed: {seed}")
    lines.append(f"- split_mode: {split_mode}")
    lines.append("")
    lines.extend(build_main_table(results))
    lines.append("")
    lines.extend(build_source_note(results))
    lines.append("")
    lines.extend(build_group_table(results, "per_emotion", "Per-Emotion Test MAE"))
    lines.append("")
    lines.extend(build_group_table(results, "per_intensity_bucket", "Per-Intensity Test MAE"))
    if any(result["metrics"]["test_metrics"].get("per_source") for result in results):
        lines.append("")
        lines.extend(build_group_table(results, "per_source", "Per-Source Test MAE"))
    lines.append("")
    lines.append("## Writing Notes")
    lines.append("")
    lines.append("- `Semantic MLP` vs `Baseline MLP` isolates the value of handcrafted semantic cue features.")
    lines.append("- `Prior Only` vs `Baseline MLP` isolates the effect of expression priors without semantic enhancement.")
    lines.append("- `Prior-Fusion` vs `Prior Only` shows whether semantic cues still help once prior guidance is introduced.")
    lines.append("- When `split_mode=source_holdout`, the test set is intended to be dominated by manual samples, which is more conservative than random mixing.")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    for config in RUNS:
        run_one(config, data_path=args.data_path, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, split_mode=args.split_mode)

    results = []
    for config in RUNS:
        metrics = load_metrics(config["run_name"])
        results.append({**config, "metrics": metrics})

    summary = build_summary(results, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, split_mode=args.split_mode)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
