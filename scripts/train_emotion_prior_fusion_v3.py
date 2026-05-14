import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


EMOTIONS = ["angry", "calm", "concern", "happy", "neutral", "sad", "surprise"]
SCORE_EMOTIONS = ["angry", "calm", "concern", "happy", "sad", "surprise"]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def pad_gold(seq: np.ndarray, max_len: int, dim: int = 53) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((max_len, dim), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=bool)
    usable = min(len(seq), max_len)
    if usable:
        out[:usable] = seq[:usable]
        mask[:usable] = True
    return out, mask


def text_statistics(text: str) -> list[float]:
    words = text.split()
    chars = len(text)
    return [
        min(chars / 160.0, 1.0),
        min(len(words) / 32.0, 1.0),
        min(text.count("!") / 4.0, 1.0),
        min(text.count("?") / 4.0, 1.0),
        min((text.count(",") + text.count(";") + text.count(":")) / 8.0, 1.0),
        1.0 if text.isupper() and chars > 2 else 0.0,
    ]


def build_prior_features(texts: list[str], ablation: str, text_embeddings: torch.Tensor | None = None):
    from apply_emotion_prior_to_emoava_sequence import infer_prior

    emotion_to_id = {name: i for i, name in enumerate(EMOTIONS)}
    ids = []
    intensities = []
    scores = []
    stats = []
    embeddings = []
    rows = []
    if text_embeddings is None:
        text_embeddings = torch.empty(len(texts), 0, dtype=torch.float32)
    for text in texts:
        emotion, intensity, raw_scores = infer_prior(text)
        top_score = max(raw_scores.values()) if raw_scores else 0.0
        if ablation == "confidence_prior" and top_score < 2.0:
            emotion = "neutral"
            intensity = 0.0
        if ablation in {"no_emotion", "soft_prior", "no_prior"}:
            emotion = "neutral"
        if ablation in {"no_intensity", "soft_prior", "no_prior"}:
            intensity = 0.0
        score_row = [min(raw_scores.get(name, 0.0) / 3.0, 1.0) for name in SCORE_EMOTIONS]
        stat_row = text_statistics(text)
        if ablation in {"no_scores", "no_prior"}:
            score_row = [0.0 for _ in score_row]
        if ablation in {"no_text_stats", "no_prior"}:
            stat_row = [0.0 for _ in stat_row]
        embedding = text_embeddings[len(ids)].float()
        if ablation == "no_bert":
            embedding = torch.zeros_like(embedding)
        ids.append(emotion_to_id.get(emotion, emotion_to_id["neutral"]))
        intensities.append(float(intensity))
        scores.append(score_row)
        stats.append(stat_row)
        embeddings.append(embedding)
        rows.append((emotion, float(intensity), raw_scores, text))

    return {
        "emotion_ids": torch.tensor(ids, dtype=torch.long),
        "intensities": torch.tensor(intensities, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "stats": torch.tensor(stats, dtype=torch.float32),
        "embeddings": torch.stack(embeddings) if embeddings else torch.empty(len(texts), 0),
        "rows": rows,
    }


def load_or_build_text_embeddings(
    texts: list[str],
    split: str,
    repo: Path,
    bert_model: str | None,
    cache_dir: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not bert_model:
        return None

    cache_root = repo / cache_dir
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{split}_bert_mean_{max_length}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu").float()
        if cached.shape[0] == len(texts):
            return cached

    from transformers import AutoModel, AutoTokenizer

    model_path = repo / bert_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outputs.append(pooled.cpu())
    embeddings = torch.cat(outputs, dim=0).float()
    torch.save(embeddings, cache_path)
    return embeddings


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


def make_time_features(length: int, device: torch.device, disabled: bool = False) -> torch.Tensor:
    pos = torch.linspace(0.0, 1.0, length, device=device)
    if disabled:
        return torch.zeros(length, 6, device=device)
    return torch.stack(
        [
            pos,
            pos.square(),
            torch.sin(math.pi * pos),
            torch.cos(math.pi * pos),
            torch.sin(2.0 * math.pi * pos),
            torch.cos(2.0 * math.pi * pos),
        ],
        dim=-1,
    )


def masked_l1(pred: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = torch.abs(pred - gold).mean(dim=-1)
    return err[mask].mean()


def masked_temporal_l1(pred: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    diff_pred = pred[:, 1:] - pred[:, :-1]
    diff_gold = gold[:, 1:] - gold[:, :-1]
    diff_mask = mask[:, 1:] & mask[:, :-1]
    err = torch.abs(diff_pred - diff_gold).mean(dim=-1)
    return err[diff_mask].mean()


class PriorFusionAdapter(nn.Module):
    def __init__(
        self,
        num_emotions: int,
        score_dim: int,
        stats_dim: int,
        text_emb_dim: int,
        exp_dim: int = 53,
        hidden: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.emotion_emb = nn.Embedding(num_emotions, 16)
        sample_dim = 16 + 1 + score_dim + stats_dim + text_emb_dim
        self.sample_proj = nn.Sequential(
            nn.Linear(sample_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )
        self.exp_proj = nn.Linear(exp_dim, hidden)
        self.time_proj = nn.Linear(6, hidden)
        self.fusion = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Linear(hidden, exp_dim)
        self.gate_head = nn.Linear(hidden, exp_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.08))

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(
        self,
        pred: torch.Tensor,
        emotion_ids: torch.Tensor,
        intensities: torch.Tensor,
        scores: torch.Tensor,
        stats: torch.Tensor,
        text_embeddings: torch.Tensor,
        time_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emotion_feature = self.emotion_emb(emotion_ids)
        sample = torch.cat([emotion_feature, intensities[:, None], scores, stats, text_embeddings], dim=-1)
        sample_hidden = self.sample_proj(sample)[:, None, :]
        time_hidden = self.time_proj(time_features)[None, :, :]
        exp_hidden = self.exp_proj(pred)
        hidden = self.fusion(torch.tanh(exp_hidden + sample_hidden + time_hidden))
        gate = torch.sigmoid(self.gate_head(hidden))
        residual = torch.tanh(self.residual_scale) * torch.tanh(self.residual_head(hidden))
        adjusted = pred + gate * residual
        return adjusted, residual, gate


def select_batch(features: dict[str, torch.Tensor], idx: torch.Tensor, device: torch.device):
    return (
        features["emotion_ids"][idx].to(device),
        features["intensities"][idx].to(device),
        features["scores"][idx].to(device),
        features["stats"][idx].to(device),
        features["embeddings"][idx].to(device),
    )


def evaluate(
    model: PriorFusionAdapter,
    pred: torch.Tensor,
    gold: torch.Tensor,
    mask: torch.Tensor,
    features: dict[str, torch.Tensor],
    time_features: torch.Tensor,
    batch_size: int,
    device: torch.device,
):
    model.eval()
    losses = []
    outputs = []
    with torch.no_grad():
        for start in range(0, pred.shape[0], batch_size):
            end = min(start + batch_size, pred.shape[0])
            idx = torch.arange(start, end)
            batch_pred = pred[idx].to(device)
            batch_gold = gold[idx].to(device)
            batch_mask = mask[idx].to(device)
            emotion_ids, intensities, scores, stats, text_embeddings = select_batch(features, idx, device)
            adjusted, _, _ = model(
                batch_pred,
                emotion_ids,
                intensities,
                scores,
                stats,
                text_embeddings,
                time_features,
            )
            losses.append(masked_l1(adjusted, batch_gold, batch_mask).detach().cpu())
            outputs.append(adjusted.detach().cpu())
    return float(torch.stack(losses).mean().item()), torch.cat(outputs, dim=0)


def train_epoch(
    model: PriorFusionAdapter,
    optimizer: torch.optim.Optimizer,
    pred: torch.Tensor,
    gold: torch.Tensor,
    mask: torch.Tensor,
    features: dict[str, torch.Tensor],
    indices: torch.Tensor,
    time_features: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dynamic_weight: float,
    residual_weight: float,
):
    model.train()
    order = indices[torch.randperm(len(indices))]
    total_loss = 0.0
    total_batches = 0
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch_pred = pred[idx].to(device)
        batch_gold = gold[idx].to(device)
        batch_mask = mask[idx].to(device)
        emotion_ids, intensities, scores, stats, text_embeddings = select_batch(features, idx, device)
        adjusted, residual, gate = model(
            batch_pred,
            emotion_ids,
            intensities,
            scores,
            stats,
            text_embeddings,
            time_features,
        )
        reconstruction = masked_l1(adjusted, batch_gold, batch_mask)
        dynamic = masked_temporal_l1(adjusted, batch_gold, batch_mask)
        reg = residual_weight * ((residual.square().mean()) + 0.1 * gate.square().mean())
        loss = reconstruction + dynamic_weight * dynamic + reg
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def save_report(
    path: Path,
    args: argparse.Namespace,
    history: list[tuple[int, float, float]],
    best_val: float,
    counts: dict[str, int],
):
    lines = [
        "# When Words Smile Prior V3 Training Report",
        "",
        "## Method",
        "",
        "V3 uses a lightweight decoder-side prior-fusion adapter. Compared with V2, it no longer learns only an emotion-time multiplicative factor. It combines base expression frames, text-derived emotion prior features, text statistics, and temporal position features to predict a gated residual sequence.",
        "",
        "```text",
        "h_t = Fusion(W_p p_t + W_e e + W_s s + W_x x_text + W_t tau_t)",
        "r_t = tanh(alpha) * tanh(W_r h_t)",
        "g_t = sigmoid(W_g h_t)",
        "p'_t = p_t + g_t * r_t",
        "```",
        "",
        "Training loss:",
        "",
        "```text",
        "L = MAE(p', y) + lambda_dyn * MAE(Delta p', Delta y) + lambda_reg * ||r||^2",
        "```",
        "",
        "## Settings",
        "",
        f"- Dev prediction: `{args.dev_prediction}`",
        f"- Test prediction: `{args.test_prediction}`",
        f"- Ablation: `{args.ablation}`",
        f"- Train size: `{args.train_size}`",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Learning rate: `{args.lr}`",
        f"- Hidden size: `{args.hidden}`",
        f"- BERT model: `{args.bert_model or 'disabled'}`",
        f"- Dynamic loss weight: `{args.dynamic_weight}`",
        f"- Residual regularization weight: `{args.residual_weight}`",
        f"- Best val MAE: `{best_val:.6f}`",
        f"- Output: `{args.output}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "## Test Prior Counts",
        "",
    ]
    for emotion, count in sorted(counts.items()):
        lines.append(f"- {emotion}: `{count}`")
    lines.extend(["", "## History", "", "| Epoch | Train Loss | Val MAE |", "|---:|---:|---:|"])
    for epoch, train_loss, val_loss in history:
        lines.append(f"| {epoch} | {train_loss:.6f} | {val_loss:.6f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-prediction", default="outputs/when_words_smile_prior_v1/dev_parallel_full.pt")
    parser.add_argument("--test-prediction", default="outputs/when_words_smile_repro/test_parallel_full.pt")
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--output", default="outputs/when_words_smile_prior_v3/test_prior_fusion.pt")
    parser.add_argument("--checkpoint", default="outputs/when_words_smile_prior_v3/prior_fusion_v3.pt")
    parser.add_argument("--report", default="docs/when_words_smile_prior_v3_train_report.md")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--dynamic-weight", type=float, default=0.08)
    parser.add_argument("--residual-weight", type=float, default=0.0005)
    parser.add_argument(
        "--ablation",
        choices=[
            "full",
            "soft_prior",
            "confidence_prior",
            "no_prior",
            "no_emotion",
            "no_intensity",
            "no_scores",
            "no_text_stats",
            "no_time",
            "no_bert",
        ],
        default="full",
    )
    parser.add_argument("--bert-model", default=None)
    parser.add_argument("--bert-cache-dir", default="outputs/when_words_smile_prior_v3/text_embeddings")
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--bert-max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    dataset_dir = repo / args.dataset_dir
    dev_pred, dev_gold, dev_mask, dev_texts = load_split(repo / args.dev_prediction, dataset_dir, "dev")
    test_pred, _, _, test_texts = load_split(repo / args.test_prediction, dataset_dir, "test")
    dev_embeddings = load_or_build_text_embeddings(
        dev_texts,
        "dev",
        repo,
        args.bert_model,
        args.bert_cache_dir,
        args.bert_batch_size,
        args.bert_max_length,
        device,
    )
    test_embeddings = load_or_build_text_embeddings(
        test_texts,
        "test",
        repo,
        args.bert_model,
        args.bert_cache_dir,
        args.bert_batch_size,
        args.bert_max_length,
        device,
    )
    dev_features = build_prior_features(dev_texts, args.ablation, dev_embeddings)
    test_features = build_prior_features(test_texts, args.ablation, test_embeddings)

    train_size = min(args.train_size, dev_pred.shape[0] - 1)
    train_idx = torch.arange(train_size)
    val_idx = torch.arange(train_size, dev_pred.shape[0])
    model = PriorFusionAdapter(
        num_emotions=len(EMOTIONS),
        score_dim=len(SCORE_EMOTIONS),
        stats_dim=6,
        text_emb_dim=dev_features["embeddings"].shape[-1],
        exp_dim=dev_pred.shape[-1],
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    time_features = make_time_features(dev_pred.shape[1], device, disabled=args.ablation == "no_time")

    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            optimizer,
            dev_pred,
            dev_gold,
            dev_mask,
            dev_features,
            train_idx,
            time_features,
            args.batch_size,
            device,
            args.dynamic_weight,
            args.residual_weight,
        )
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            val_loss, _ = evaluate(
                model,
                dev_pred[val_idx],
                dev_gold[val_idx],
                dev_mask[val_idx],
                {k: (v[val_idx] if isinstance(v, torch.Tensor) else v) for k, v in dev_features.items() if k != "rows"},
                time_features,
                args.batch_size,
                device,
            )
            history.append((epoch, train_loss, val_loss))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_mae={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_adjusted = evaluate(
        model,
        test_pred,
        test_pred,
        torch.ones(test_pred.shape[:2], dtype=torch.bool),
        test_features,
        time_features,
        args.batch_size,
        device,
    )

    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(test_adjusted, output)
    checkpoint = repo / args.checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "emotions": EMOTIONS,
            "score_emotions": SCORE_EMOTIONS,
            "best_val_mae": best_val,
            "history": history,
            "args": vars(args),
        },
        checkpoint,
    )

    counts = {}
    for emotion, _, _, _ in test_features["rows"]:
        counts[emotion] = counts.get(emotion, 0) + 1
    save_report(repo / args.report, args, history, best_val, counts)
    print(repo / args.report)
    print(output)


if __name__ == "__main__":
    main()
