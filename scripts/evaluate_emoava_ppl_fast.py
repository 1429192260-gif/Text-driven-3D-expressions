import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(np.math.erf)(x / np.sqrt(2.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    emoava_src = repo / "external" / "EmoAva" / "src"
    sys.path.insert(0, str(emoava_src))
    from em_dataloader import get_dataloader, load_dataset_split_batch
    from transformers import BertTokenizer

    pred = torch.load(args.prediction, map_location="cpu")
    src_texts, tgt_exps, tgt_exps_steps = load_dataset_split_batch(args.split, -1, None)
    tokenizer = BertTokenizer.from_pretrained(repo / "models" / "bert-base-cased")
    tgt_texts = [" ".join(["[UNK]"] * i) for i in tgt_exps_steps]
    loader = get_dataloader(
        src_texts,
        tgt_texts,
        tgt_exps,
        {
            "tokenizer": tokenizer,
            "batch_size": 256,
            "src_len": 128,
            "trg_len": 256,
            "shuffle": False,
        },
    )
    gold_batches = []
    mask_batches = []
    for batch in loader:
        gold_batches.append(batch["tgt_exp_gold"])
        mask_batches.append(batch["tgt_gold_mask"])
    gold = torch.cat(gold_batches, dim=0).numpy()
    mask = torch.cat(mask_batches, dim=0).numpy().astype(bool)
    pred_np = pred.numpy()
    if pred_np.shape != gold.shape:
        raise SystemExit(f"Shape mismatch: pred {pred_np.shape}, gold {gold.shape}")

    # Official metric uses mvn.cdf(upper) - mvn.cdf(lower). With diagonal
    # covariance, each multivariate CDF is the product of 1D Gaussian CDFs.
    upper = ((gold + args.threshold) - pred_np) / args.sigma
    lower = ((gold - args.threshold) - pred_np) / args.sigma
    cdf_upper = np.clip(normal_cdf(upper), args.eps, 1.0)
    cdf_lower = np.clip(normal_cdf(lower), args.eps, 1.0)
    log_upper = np.log(cdf_upper).sum(axis=-1)
    log_lower = np.log(cdf_lower).sum(axis=-1)
    ratio = np.exp(np.minimum(log_lower - log_upper, 0.0))
    prob_frame = np.exp(log_upper) * np.maximum(1.0 - ratio, args.eps)
    log2_prob_frame = np.log2(np.maximum(prob_frame, args.eps))
    valid_counts = mask.sum(axis=1)
    accum = (log2_prob_frame * mask).sum(axis=1)
    h_i = -accum / valid_counts
    ppl = float(2 ** h_i.mean())

    lines = [
        "# When Words Smile Fast PPL Evaluation",
        "",
        f"- Prediction: `{args.prediction}`",
        f"- Split: `{args.split}`",
        f"- Samples: `{pred_np.shape[0]}`",
        f"- Shape: `{pred_np.shape}`",
        f"- Sigma: `{args.sigma}`",
        f"- Threshold: `{args.threshold}`",
        "",
        f"perplexity `{ppl:.2f}`",
    ]
    report = "\n".join(lines)
    print(report)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
