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
    parser.add_argument("--summary-path", default="docs/experiment_summary_240.md")
    return parser.parse_args()


def run_one(config, data_path, epochs, batch_size, seed):
    cmd = [
        sys.executable,
        "scripts/train_mlp.py",
        "--data-path", data_path,
        "--run-name", config["run_name"],
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
    ] + config["args"]
    subprocess.run(cmd, check=True)


def load_metrics(run_name):
    path = OUTPUT_DIR / f"{run_name}_metrics.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value):
    return f"{value:.6f}"


def build_summary(results, epochs, batch_size, seed):
    lines = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append(f"- epochs: {epochs}")
    lines.append(f"- batch_size: {batch_size}")
    lines.append(f"- seed: {seed}")
    lines.append("")
    lines.append("| Run | Setting | Test MAE | Test RMSE | Active MAE | Test Loss |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for result in results:
        metrics = result["metrics"]["test_metrics"]
        lines.append(
            f"| {result['title']} | {result['description']} | {format_float(metrics['mae'])} | {format_float(metrics['rmse'])} | {format_float(metrics['active_mae'])} | {format_float(metrics['loss'])} |"
        )

    lines.append("")
    lines.append("## Writing Notes")
    lines.append("")
    lines.append("- `Baseline MLP` can serve as the core baseline without semantic enhancement or explicit prior guidance.")
    lines.append("- `Semantic MLP` is the first ablation to verify whether handcrafted semantic cues improve parameter prediction.")
    lines.append("- `Prior-Fusion` is the main method and tests whether expression priors plus residual correction stabilize learning in small-sample settings.")
    lines.append("- If `Prior-Fusion` is much better than the other models, the paper should state clearly that the current labels are strongly aligned with rule-based priors.")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    for config in RUNS:
        run_one(config, data_path=args.data_path, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)

    results = []
    for config in RUNS:
        metrics = load_metrics(config["run_name"])
        results.append({**config, "metrics": metrics})

    summary = build_summary(results, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
