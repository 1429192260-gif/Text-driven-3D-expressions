import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def pad_gold(seq: np.ndarray, max_len: int, dim: int = 53) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((max_len, dim), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=bool)
    usable = min(len(seq), max_len)
    out[:usable] = seq[:usable]
    mask[:usable] = True
    return out, mask


def interpolate_profiles(knots: torch.Tensor, length: int) -> torch.Tensor:
    # Linear interpolation from 5 learned knots to T frames.
    device = knots.device
    src = torch.linspace(0, 1, knots.shape[-1], device=device)
    dst = torch.linspace(0, 1, length, device=device)
    right = torch.bucketize(dst, src).clamp(1, len(src) - 1)
    left = right - 1
    denom = (src[right] - src[left]).clamp_min(1e-6)
    w = (dst - src[left]) / denom
    return knots[..., left] * (1 - w) + knots[..., right] * w


class PriorAdapter(nn.Module):
    def __init__(self, num_emotions: int, dim: int = 53, knots: int = 5):
        super().__init__()
        self.delta_knots = nn.Parameter(torch.zeros(num_emotions, dim, knots))
        self.global_scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, pred: torch.Tensor, emotion_ids: torch.Tensor, intensities: torch.Tensor):
        batch, length, dim = pred.shape
        profiles = interpolate_profiles(self.delta_knots, length)
        selected = profiles[emotion_ids].transpose(1, 2)
        factors = 1.0 + torch.tanh(self.global_scale) * intensities[:, None, None] * torch.tanh(selected)
        return pred * factors


def build_prior_features(texts: list[str]):
    from apply_emotion_prior_to_emoava_sequence import infer_prior

    emotions = ["angry", "calm", "concern", "happy", "neutral", "sad", "surprise"]
    emotion_to_id = {name: i for i, name in enumerate(emotions)}
    ids = []
    intensities = []
    rows = []
    for text in texts:
        emotion, intensity, scores = infer_prior(text)
        ids.append(emotion_to_id.get(emotion, emotion_to_id["neutral"]))
        intensities.append(intensity)
        rows.append((emotion, intensity, scores))
    return emotions, torch.tensor(ids, dtype=torch.long), torch.tensor(intensities, dtype=torch.float32), rows


def load_split(prediction: Path, dataset_dir: Path, split: str):
    pred = torch.load(prediction, map_location="cpu").float()
    texts = load_pickle(dataset_dir / f"{split}_stage1_text.pkl")
    gold_raw = load_pickle(dataset_dir / f"{split}_stage1_exps.pkl")
    gold = []
    mask = []
    for seq in gold_raw[: pred.shape[0]]:
        g, m = pad_gold(np.asarray(seq, dtype=np.float32), pred.shape[1], pred.shape[2])
        gold.append(g)
        mask.append(m)
    return pred, torch.tensor(np.stack(gold)), torch.tensor(np.stack(mask)), texts[: pred.shape[0]]


def masked_l1(pred: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor):
    err = torch.abs(pred - gold).mean(dim=-1)
    return err[mask].mean()


def evaluate(model, pred, gold, mask, emotion_ids, intensities):
    model.eval()
    with torch.no_grad():
        adjusted = model(pred, emotion_ids, intensities)
        loss = masked_l1(adjusted, gold, mask)
    return float(loss.item()), adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-prediction", default="outputs/when_words_smile_prior_v1/dev_parallel_full.pt")
    parser.add_argument("--test-prediction", default="outputs/when_words_smile_repro/test_parallel_full.pt")
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--output", default="outputs/when_words_smile_prior_v2/test_prior_adapter.pt")
    parser.add_argument("--checkpoint", default="outputs/when_words_smile_prior_v2/adapter_v2.pt")
    parser.add_argument("--report", default="docs/when_words_smile_prior_v2_train_report.md")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--train-size", type=int, default=1200)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    dataset_dir = repo / args.dataset_dir
    dev_pred, dev_gold, dev_mask, dev_texts = load_split(repo / args.dev_prediction, dataset_dir, "dev")
    test_pred, _, _, test_texts = load_split(repo / args.test_prediction, dataset_dir, "test")
    emotions, dev_emotion_ids, dev_intensities, _ = build_prior_features(dev_texts)
    _, test_emotion_ids, test_intensities, test_prior_rows = build_prior_features(test_texts)

    train_size = min(args.train_size, dev_pred.shape[0] - 1)
    train_idx = torch.arange(train_size)
    val_idx = torch.arange(train_size, dev_pred.shape[0])

    model = PriorAdapter(num_emotions=len(emotions), dim=dev_pred.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        adjusted = model(dev_pred[train_idx], dev_emotion_ids[train_idx], dev_intensities[train_idx])
        loss = masked_l1(adjusted, dev_gold[train_idx], dev_mask[train_idx])
        reg = 0.001 * (model.delta_knots**2).mean()
        total = loss + reg
        total.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            val_loss, _ = evaluate(
                model,
                dev_pred[val_idx],
                dev_gold[val_idx],
                dev_mask[val_idx],
                dev_emotion_ids[val_idx],
                dev_intensities[val_idx],
            )
            history.append((epoch, float(loss.item()), val_loss))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_adjusted = evaluate(model, test_pred, test_pred, torch.ones(test_pred.shape[:2], dtype=torch.bool), test_emotion_ids, test_intensities)

    out = repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(test_adjusted, out)
    ckpt = repo / args.checkpoint
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "emotions": emotions, "best_val_mae": best_val, "history": history}, ckpt)

    counts = {}
    for emotion, _, _ in test_prior_rows:
        counts[emotion] = counts.get(emotion, 0) + 1
    report_lines = [
        "# Prior Adapter V2 Training Report",
        "",
        f"- Dev prediction: `{args.dev_prediction}`",
        f"- Train size: `{len(train_idx)}`",
        f"- Val size: `{len(val_idx)}`",
        f"- Epochs: `{args.epochs}`",
        f"- Best val MAE: `{best_val:.6f}`",
        f"- Output: `{args.output}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "## Emotion Counts On Test",
        "",
    ]
    for emotion, count in sorted(counts.items()):
        report_lines.append(f"- {emotion}: `{count}`")
    report_lines.extend(["", "## History", "", "| Epoch | Train MAE | Val MAE |", "|---:|---:|---:|"])
    for epoch, train_loss, val_loss in history:
        report_lines.append(f"| {epoch} | {train_loss:.6f} | {val_loss:.6f} |")
    report = repo / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(report_lines), encoding="utf-8")
    print(report)
    print(out)


if __name__ == "__main__":
    main()
