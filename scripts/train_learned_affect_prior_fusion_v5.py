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
    lowered = text.lower()
    words = text.split()
    chars = len(text)
    return [
        min(chars / 160.0, 1.0),
        min(len(words) / 32.0, 1.0),
        min(text.count("!") / 4.0, 1.0),
        min(text.count("?") / 4.0, 1.0),
        min((text.count(",") + text.count(";") + text.count(":")) / 8.0, 1.0),
        1.0 if text.isupper() and chars > 2 else 0.0,
        1.0 if "..." in text else 0.0,
        1.0 if any(token in lowered for token in ["not", "no", "never", "n't"]) else 0.0,
    ]


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


def build_text_features(texts: list[str], text_embeddings: torch.Tensor | None = None):
    from apply_emotion_prior_to_emoava_sequence import infer_prior

    stats = []
    scores = []
    rows = []
    if text_embeddings is None:
        text_embeddings = torch.empty(len(texts), 0, dtype=torch.float32)
    for i, text in enumerate(texts):
        emotion, intensity, raw_scores = infer_prior(text)
        score_row = [min(raw_scores.get(name, 0.0) / 3.0, 1.0) for name in SCORE_EMOTIONS]
        stats.append(text_statistics(text))
        scores.append(score_row)
        rows.append((emotion, float(intensity), raw_scores, text))
    return {
        "stats": torch.tensor(stats, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "embeddings": text_embeddings.float(),
        "rows": rows,
    }


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


def make_time_features(length: int, device: torch.device) -> torch.Tensor:
    pos = torch.linspace(0.0, 1.0, length, device=device)
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


class LearnedAffectPriorFusionAdapter(nn.Module):
    def __init__(
        self,
        score_dim: int,
        stats_dim: int,
        text_emb_dim: int,
        affect_dim: int = 8,
        exp_dim: int = 53,
        hidden: int = 64,
        dropout: float = 0.05,
        use_global_branch: bool = True,
        use_prior_branch: bool = True,
        use_text_prior: bool = True,
    ):
        super().__init__()
        self.use_global_branch = use_global_branch
        self.use_prior_branch = use_prior_branch
        self.use_text_prior = use_text_prior
        text_dim = score_dim + stats_dim + text_emb_dim
        self.text_prior_net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.affect_logits = nn.Linear(hidden, affect_dim)
        self.confidence_head = nn.Linear(hidden, 1)
        self.intensity_head = nn.Linear(hidden, 1)

        self.exp_proj = nn.Linear(exp_dim, hidden)
        self.time_proj = nn.Linear(6, hidden)
        self.global_net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prior_sample_proj = nn.Sequential(
            nn.Linear(affect_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.global_residual_head = nn.Linear(hidden, exp_dim)
        self.global_gate_head = nn.Linear(hidden, exp_dim)
        self.prior_residual_head = nn.Linear(hidden, exp_dim)
        self.prior_gate_head = nn.Linear(hidden, exp_dim)
        self.global_scale = nn.Parameter(torch.tensor(0.08))
        self.prior_scale = nn.Parameter(torch.tensor(0.04))

        for head in [self.global_residual_head, self.prior_residual_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.global_gate_head.bias, -2.0)
        nn.init.constant_(self.prior_gate_head.bias, -3.0)
        nn.init.constant_(self.confidence_head.bias, -1.5)

    def forward(
        self,
        pred: torch.Tensor,
        scores: torch.Tensor,
        stats: torch.Tensor,
        embeddings: torch.Tensor,
        time_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_input = torch.cat([scores, stats, embeddings], dim=-1)
        text_hidden = self.text_prior_net(text_input)
        affect_probs = torch.softmax(self.affect_logits(text_hidden), dim=-1)
        confidence = torch.sigmoid(self.confidence_head(text_hidden)).squeeze(-1)
        intensity = torch.sigmoid(self.intensity_head(text_hidden)).squeeze(-1)

        if not self.use_text_prior:
            affect_probs = torch.zeros_like(affect_probs)
            affect_probs[:, 0] = 1.0
            confidence = torch.zeros_like(confidence)
            intensity = torch.zeros_like(intensity)

        exp_hidden = self.exp_proj(pred)
        time_hidden = self.time_proj(time_features)[None, :, :]

        global_hidden = self.global_net(torch.cat([exp_hidden, time_hidden.expand_as(exp_hidden)], dim=-1))
        global_gate = torch.sigmoid(self.global_gate_head(global_hidden))
        global_residual = torch.tanh(self.global_scale) * torch.tanh(self.global_residual_head(global_hidden))

        sample = torch.cat([affect_probs, confidence[:, None], intensity[:, None]], dim=-1)
        prior_sample = self.prior_sample_proj(sample)[:, None, :].expand_as(exp_hidden)
        prior_hidden = self.prior_net(torch.cat([exp_hidden, time_hidden.expand_as(exp_hidden), prior_sample], dim=-1))
        prior_gate = torch.sigmoid(self.prior_gate_head(prior_hidden)) * confidence[:, None, None]
        prior_residual = torch.tanh(self.prior_scale) * torch.tanh(self.prior_residual_head(prior_hidden))

        if not self.use_global_branch:
            global_gate = torch.zeros_like(global_gate)
            global_residual = torch.zeros_like(global_residual)
        if not self.use_prior_branch:
            prior_gate = torch.zeros_like(prior_gate)
            prior_residual = torch.zeros_like(prior_residual)

        adjusted = pred + global_gate * global_residual + prior_gate * prior_residual
        residual = global_residual + prior_residual
        gate = global_gate + prior_gate
        return adjusted, residual, gate, prior_gate, affect_probs, confidence


def select_batch(features: dict[str, torch.Tensor], idx: torch.Tensor, device: torch.device):
    return (
        features["scores"][idx].to(device),
        features["stats"][idx].to(device),
        features["embeddings"][idx].to(device),
    )


def slice_features(features: dict[str, torch.Tensor], idx: torch.Tensor):
    return {k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in features.items() if k != "rows"}


def entropy_regularizer(probs: torch.Tensor) -> torch.Tensor:
    entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1)
    return entropy.mean()


def evaluate(
    model: LearnedAffectPriorFusionAdapter,
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
    confidences = []
    with torch.no_grad():
        for start in range(0, pred.shape[0], batch_size):
            end = min(start + batch_size, pred.shape[0])
            idx = torch.arange(start, end)
            batch_pred = pred[idx].to(device)
            batch_gold = gold[idx].to(device)
            batch_mask = mask[idx].to(device)
            scores, stats, embeddings = select_batch(features, idx, device)
            adjusted, _, _, _, _, confidence = model(batch_pred, scores, stats, embeddings, time_features)
            losses.append(masked_l1(adjusted, batch_gold, batch_mask).detach().cpu())
            outputs.append(adjusted.detach().cpu())
            confidences.append(confidence.detach().cpu())
    return float(torch.stack(losses).mean().item()), torch.cat(outputs, dim=0), torch.cat(confidences, dim=0)


def train_epoch(
    model: LearnedAffectPriorFusionAdapter,
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
    prior_gate_weight: float,
    entropy_weight: float,
    confidence_weight: float,
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
        scores, stats, embeddings = select_batch(features, idx, device)
        adjusted, residual, _, prior_gate, affect_probs, confidence = model(
            batch_pred,
            scores,
            stats,
            embeddings,
            time_features,
        )
        reconstruction = masked_l1(adjusted, batch_gold, batch_mask)
        dynamic = masked_temporal_l1(adjusted, batch_gold, batch_mask)
        reg = residual_weight * residual.square().mean() + prior_gate_weight * prior_gate.square().mean()
        entropy_reg = entropy_weight * entropy_regularizer(affect_probs)
        confidence_reg = confidence_weight * confidence.mean()
        loss = reconstruction + dynamic_weight * dynamic + reg + entropy_reg + confidence_reg
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
    rows: list[tuple[str, float, dict[str, float], str]],
    confidences: torch.Tensor,
):
    hard_counts = {}
    for emotion, *_ in rows:
        hard_counts[emotion] = hard_counts.get(emotion, 0) + 1
    lines = [
        "# When Words Smile Prior V5 Training Report",
        "",
        "## Method",
        "",
        "V5 replaces the hand-crafted soft affect prior with a learned text-to-affect prior network. The prior network receives keyword-score features, text statistics, and optional frozen BERT sentence embeddings, then predicts a latent affect distribution, confidence, and intensity.",
        "",
        "```text",
        "z_text = PriorNet(x_text)",
        "a = softmax(W_a z_text)",
        "c = sigmoid(W_c z_text)",
        "s = sigmoid(W_s z_text)",
        "p'_t = p_t + g_global(t) * r_global(t) + c * g_prior(t) * r_prior(t)",
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
        f"- Affect dim: `{args.affect_dim}`",
        f"- BERT model: `{args.bert_model or 'disabled'}`",
        f"- Dynamic loss weight: `{args.dynamic_weight}`",
        f"- Residual regularization weight: `{args.residual_weight}`",
        f"- Prior gate regularization weight: `{args.prior_gate_weight}`",
        f"- Entropy weight: `{args.entropy_weight}`",
        f"- Confidence weight: `{args.confidence_weight}`",
        f"- Best val MAE: `{best_val:.6f}`",
        f"- Output: `{args.output}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "## Test Learned Prior Statistics",
        "",
        f"- Mean confidence: `{float(confidences.mean()):.6f}`",
        f"- Median confidence: `{float(confidences.median()):.6f}`",
        "",
        "## Hard Rule Counts For Reference Only",
        "",
    ]
    for emotion, count in sorted(hard_counts.items()):
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
    parser.add_argument("--output", default="outputs/when_words_smile_prior_v5/test_learned_affect_prior_fusion.pt")
    parser.add_argument("--checkpoint", default="outputs/when_words_smile_prior_v5/learned_affect_prior_fusion_v5.pt")
    parser.add_argument("--report", default="docs/when_words_smile_prior_v5_train_report.md")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--affect-dim", type=int, default=8)
    parser.add_argument("--dynamic-weight", type=float, default=0.08)
    parser.add_argument("--residual-weight", type=float, default=0.0005)
    parser.add_argument("--prior-gate-weight", type=float, default=0.0002)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--confidence-weight", type=float, default=0.0001)
    parser.add_argument("--bert-model", default=None)
    parser.add_argument("--bert-cache-dir", default="outputs/when_words_smile_prior_v5/text_embeddings")
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--bert-max-length", type=int, default=64)
    parser.add_argument(
        "--ablation",
        choices=["full", "no_prior_branch", "no_global_branch", "no_text_prior"],
        default="full",
    )
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
    dev_features = build_text_features(dev_texts, dev_embeddings)
    test_features = build_text_features(test_texts, test_embeddings)

    train_size = min(args.train_size, dev_pred.shape[0] - 1)
    train_idx = torch.arange(train_size)
    val_idx = torch.arange(train_size, dev_pred.shape[0])
    model = LearnedAffectPriorFusionAdapter(
        score_dim=len(SCORE_EMOTIONS),
        stats_dim=8,
        text_emb_dim=dev_features["embeddings"].shape[-1],
        affect_dim=args.affect_dim,
        exp_dim=dev_pred.shape[-1],
        hidden=args.hidden,
        dropout=args.dropout,
        use_global_branch=args.ablation != "no_global_branch",
        use_prior_branch=args.ablation != "no_prior_branch",
        use_text_prior=args.ablation != "no_text_prior",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    time_features = make_time_features(dev_pred.shape[1], device)

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
            args.prior_gate_weight,
            args.entropy_weight,
            args.confidence_weight,
        )
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            val_loss, _, _ = evaluate(
                model,
                dev_pred[val_idx],
                dev_gold[val_idx],
                dev_mask[val_idx],
                slice_features(dev_features, val_idx),
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
    _, test_adjusted, test_confidences = evaluate(
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
            "best_val_mae": best_val,
            "history": history,
            "args": vars(args),
        },
        checkpoint,
    )
    save_report(repo / args.report, args, history, best_val, test_features["rows"], test_confidences)
    print(repo / args.report)
    print(output)


if __name__ == "__main__":
    main()
