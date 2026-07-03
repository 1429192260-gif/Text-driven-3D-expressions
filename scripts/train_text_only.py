import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.text_only_classifier import TextOnlyClassifier
from utils.text_encoder import load_text_encoder
from utils.text_features import extract_semantic_features

EMOTION_LIST = ["happy", "sad", "angry", "surprise", "disgust", "concern", "bored", "calm"]
AUGMENTED_PREFIXES = ("weibo_",)
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def is_augmented_source(source):
    source = source or "manual"
    return source.startswith(AUGMENTED_PREFIXES)


def normalize_item(item):
    normalized = dict(item)
    normalized.setdefault("source", "manual")
    return normalized


def split_subset(items, train_ratio, val_ratio, rng, allow_test=True):
    items = [dict(item) for item in items]
    rng.shuffle(items)
    n = len(items)
    if n == 0:
        return [], [], []
    if allow_test:
        train_end = max(1, int(round(n * train_ratio)))
        val_count = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if train_end + val_count >= n:
            val_count = 1 if n - train_end > 1 else 0
        val_end = min(n, train_end + val_count)
        return items[:train_end], items[train_end:val_end], items[val_end:]
    train_end = max(1, int(round(n * (train_ratio / max(train_ratio + val_ratio, 1e-8)))))
    train_end = min(train_end, n - 1) if n > 1 else n
    return items[:train_end], items[train_end:], []


def split_data(data, split_mode="source_holdout", train_ratio=0.7, val_ratio=0.15, seed=42):
    grouped = {}
    for item in data:
        grouped.setdefault(item["emotion"], []).append(dict(item))

    train_data, val_data, test_data = [], [], []
    rng = random.Random(seed)

    for emotion_items in grouped.values():
        if split_mode == "random":
            rng.shuffle(emotion_items)
            n = len(emotion_items)
            train_end = max(1, int(round(n * train_ratio)))
            val_count = max(1, int(round(n * val_ratio))) if n >= 3 else 0
            if train_end + val_count >= n:
                val_count = 1 if n - train_end > 1 else 0
            val_end = min(n, train_end + val_count)
            train_data.extend(emotion_items[:train_end])
            val_data.extend(emotion_items[train_end:val_end])
            test_data.extend(emotion_items[val_end:])
            continue

        manual_items = [item for item in emotion_items if not is_augmented_source(item.get("source", "manual"))]
        augmented_items = [item for item in emotion_items if is_augmented_source(item.get("source", "manual"))]
        manual_train, manual_val, manual_test = split_subset(manual_items, train_ratio, val_ratio, rng, allow_test=True)
        aug_train, aug_val, _ = split_subset(augmented_items, train_ratio, val_ratio, rng, allow_test=False)
        if not manual_test and manual_val:
            manual_test = [manual_val.pop()]
        if not manual_test and manual_train:
            manual_test = [manual_train.pop()]
        train_data.extend(manual_train + aug_train)
        val_data.extend(manual_val + aug_val)
        test_data.extend(manual_test)

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)
    return train_data, val_data, test_data


class ClassificationDataset(Dataset):
    def __init__(self, data, encoder, use_semantic_features=True):
        self.items = []
        texts = [item["text"] for item in data]
        embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        for item, text_emb in zip(data, embeddings):
            semantic = extract_semantic_features(item["text"]) if use_semantic_features else []
            features = list(text_emb) + semantic
            emotion_idx = EMOTION_LIST.index(item["emotion"])
            self.items.append({
                "features": torch.tensor(features, dtype=torch.float32),
                "emotion": torch.tensor(emotion_idx, dtype=torch.long),
                "source": item.get("source", "manual"),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch):
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "emotion": torch.stack([item["emotion"] for item in batch]),
        "source": [item["source"] for item in batch],
    }


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()
    emo_logits_all, emo_targets_all = [], []
    sources = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            emo_targets = batch["emotion"].to(device)
            emo_logits = model(features)
            loss = ce(emo_logits, emo_targets)
            total_loss += loss.item()
            emo_logits_all.append(emo_logits.cpu())
            emo_targets_all.append(emo_targets.cpu())
            sources.extend(batch["source"])
    emo_logits = torch.cat(emo_logits_all)
    emo_targets = torch.cat(emo_targets_all)
    emo_pred = emo_logits.argmax(dim=-1)
    emotion_acc = (emo_pred == emo_targets).float().mean().item()
    per_source = {}
    for source in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == source]
        source_pred = emo_pred[idx]
        source_target = emo_targets[idx]
        per_source[source] = {
            "emotion_acc": (source_pred == source_target).float().mean().item(),
            "count": len(idx),
        }
    return {
        "loss": total_loss / max(len(loader), 1),
        "emotion_acc": emotion_acc,
        "per_source": per_source,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train text-only emotion classifier.")
    parser.add_argument("--data-path", default="data/full_samples_804.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default="text_only_emotion")
    parser.add_argument("--encoder-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["random", "source_holdout"], default="source_holdout")
    parser.add_argument("--no-semantic-features", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = [normalize_item(item) for item in json.load(f) if item.get("emotion") in EMOTION_LIST]

    train_data, val_data, test_data = split_data(data, split_mode=args.split_mode, seed=args.seed)
    encoder = load_text_encoder(args.encoder_name)
    use_semantic_features = not args.no_semantic_features

    train_dataset = ClassificationDataset(train_data, encoder, use_semantic_features=use_semantic_features)
    val_dataset = ClassificationDataset(val_data, encoder, use_semantic_features=use_semantic_features)
    test_dataset = ClassificationDataset(test_data, encoder, use_semantic_features=use_semantic_features)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    sample = train_dataset[0]
    model = TextOnlyClassifier(
        input_dim=sample["features"].shape[0],
        hidden_dim=args.hidden_dim,
        num_emotions=len(EMOTION_LIST),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            emo_targets = batch["emotion"].to(device)
            emo_logits = model(features)
            loss = ce(emo_logits, emo_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        if val_metrics["emotion_acc"] > best_val:
            best_val = val_metrics["emotion_acc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} | "
                f"val_emotion_acc={val_metrics['emotion_acc']:.4f}"
            )

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.run_name
    checkpoint_path = os.path.join(args.output_dir, f"{tag}_classifier.pt")
    metrics_path = os.path.join(args.output_dir, f"{tag}_classifier_metrics.json")

    checkpoint = {
        "state_dict": best_state,
        "encoder_name": getattr(encoder, "name", args.encoder_name),
        "use_semantic_features": use_semantic_features,
        "input_dim": sample["features"].shape[0],
        "hidden_dim": args.hidden_dim,
        "emotion_list": EMOTION_LIST,
    }
    torch.save(checkpoint, checkpoint_path)

    metrics_payload = {
        "config": {
            "run_name": tag,
            "encoder_name": getattr(encoder, "name", args.encoder_name),
            "use_semantic_features": use_semantic_features,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "split_mode": args.split_mode,
            "splits": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
            "split_sources": {
                "train_augmented": sum(is_augmented_source(item.get('source', 'manual')) for item in train_data),
                "train_manual": sum(not is_augmented_source(item.get('source', 'manual')) for item in train_data),
                "val_augmented": sum(is_augmented_source(item.get('source', 'manual')) for item in val_data),
                "val_manual": sum(not is_augmented_source(item.get('source', 'manual')) for item in val_data),
                "test_augmented": sum(is_augmented_source(item.get('source', 'manual')) for item in test_data),
                "test_manual": sum(not is_augmented_source(item.get('source', 'manual')) for item in test_data),
            },
        },
        "best_val_emotion_acc": best_val,
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"Encoder: {getattr(encoder, 'name', args.encoder_name)}")
    print(f"Best classifier saved to {checkpoint_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Test metrics | emotion_acc={test_metrics['emotion_acc']:.4f}")


if __name__ == "__main__":
    main()
