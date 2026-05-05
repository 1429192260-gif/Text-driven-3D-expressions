import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import multivariate_normal


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(np.math.erf)(x / np.sqrt(2.0)))


def official_frame_prob(gold: np.ndarray, pred: np.ndarray, sigma: float, threshold: float) -> float:
    cov = np.diag([sigma**2] * len(pred))
    mvn = multivariate_normal(mean=pred, cov=cov)
    return float(mvn.cdf(gold + threshold) - mvn.cdf(gold - threshold))


def fast_frame_prob(gold: np.ndarray, pred: np.ndarray, sigma: float, threshold: float) -> float:
    upper = ((gold + threshold) - pred) / sigma
    lower = ((gold - threshold) - pred) / sigma
    return float(np.prod(normal_cdf(upper)) - np.prod(normal_cdf(lower)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", default="outputs/when_words_smile_repro/test_parallel_full.pt")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--output", default="docs/when_words_smile_ppl_formula_check.md")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "external" / "EmoAva" / "src"))
    from em_dataloader import get_dataloader, load_dataset_split_batch
    from transformers import BertTokenizer

    pred = torch.load(args.prediction, map_location="cpu")
    src_texts, tgt_exps, tgt_exps_steps = load_dataset_split_batch("test", -1, None)
    tokenizer = BertTokenizer.from_pretrained(repo / "models" / "bert-base-cased")
    tgt_texts = [" ".join(["[UNK]"] * i) for i in tgt_exps_steps]
    loader = get_dataloader(
        src_texts,
        tgt_texts,
        tgt_exps,
        {
            "tokenizer": tokenizer,
            "batch_size": 1,
            "src_len": 128,
            "trg_len": 256,
            "shuffle": False,
        },
    )
    batch = next(iter(loader))
    gold = batch["tgt_exp_gold"][0].numpy()
    mask = batch["tgt_gold_mask"][0].numpy().astype(bool)
    pred_np = pred[0].numpy()

    rows = []
    checked = 0
    for frame_idx in range(len(mask)):
        if not mask[frame_idx]:
            break
        off = official_frame_prob(gold[frame_idx], pred_np[frame_idx], args.sigma, args.threshold)
        fast = fast_frame_prob(gold[frame_idx], pred_np[frame_idx], args.sigma, args.threshold)
        rows.append((frame_idx, off, fast, abs(off - fast)))
        checked += 1
        if checked >= args.frames:
            break

    lines = [
        "# PPL Formula Check",
        "",
        "This compares the official SciPy `multivariate_normal.cdf(x+t) - cdf(x-t)` frame probability with the diagonal-covariance fast form.",
        "",
        "| Frame | Official | Fast | Abs Diff |",
        "|---:|---:|---:|---:|",
    ]
    for frame_idx, off, fast, diff in rows:
        lines.append(f"| {frame_idx} | {off:.12e} | {fast:.12e} | {diff:.12e} |")
    lines.append("")
    lines.append(f"Max abs diff: `{max(r[3] for r in rows):.12e}`")

    out = Path(args.output)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
