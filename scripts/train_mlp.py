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

from models.mlp_mapper import MLPMapper
from models.prior_fusion_mapper import PriorFusionMapper
from rule_mapping import rule_mapping
from utils.param_utils import params_dict_to_vector
from utils.text_encoder import load_text_encoder
from utils.text_features import extract_semantic_features

EMOTION_LIST = [
    "happy", "sad", "angry", "surprise",
    "disgust", "concern", "bored", "calm",
]

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
AUGMENTED_PREFIXES = ("weibo_",)


def emotion_to_onehot(emotion):
    vec = [0.0] * len(EMOTION_LIST)
    vec[EMOTION_LIST.index(emotion)] = 1.0
    return vec


def intensity_to_bucket(intensity):
    if intensity < 0.45:
        return "weak"
    if intensity < 0.75:
        return "medium"
    return "strong"


def normalize_item(item):
    normalized = dict(item)
    if "param_vector" not in normalized:
        normalized["param_vector"] = params_dict_to_vector(normalized["params"])
    normalized.setdefault("source", "manual")
    normalized.setdefault("source_label", "")
    return normalized


def is_augmented_source(source):
    source = source or "manual"
    return source.startswith(AUGMENTED_PREFIXES)


class ExpressionDataset(Dataset):
    def __init__(self, data, encoder, use_semantic_features=True, use_prior=True):
        self.items = []
        texts = [item["text"] for item in data]
        embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        for item, text_emb in zip(data, embeddings):
            emo_vec = emotion_to_onehot(item["emotion"])
            intensity_value = float(item["intensity"])
            intensity = [intensity_value]
            semantic_features = extract_semantic_features(item["text"]) if use_semantic_features else []
            control_features = emo_vec + intensity + semantic_features
            prior_vector = rule_mapping(item["emotion"], intensity_value) if use_prior else [0.0] * len(item["param_vector"])
            mlp_input = list(text_emb) + control_features

            self.items.append({
                "text_features": torch.tensor(text_emb, dtype=torch.float32),
                "control_features": torch.tensor(control_features, dtype=torch.float32),
                "prior_vector": torch.tensor(prior_vector, dtype=torch.float32),
                "mlp_input": torch.tensor(mlp_input, dtype=torch.float32),
                "target": torch.tensor(item["param_vector"], dtype=torch.float32),
                "emotion": item["emotion"],
                "source": item.get("source", "manual"),
                "intensity_value": intensity_value,
                "intensity_bucket": intensity_to_bucket(intensity_value),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch):
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    return collated


def build_model(model_type, sample):
    if model_type == "mlp":
        return MLPMapper(
            input_dim=sample["mlp_input"].shape[0],
            output_dim=sample["target"].shape[0],
        )

    if model_type == "prior_fusion":
        return PriorFusionMapper(
            text_dim=sample["text_features"].shape[0],
            control_dim=sample["control_features"].shape[0],
            prior_dim=sample["prior_vector"].shape[0],
            output_dim=sample["target"].shape[0],
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def forward_model(model, batch, model_type):
    if model_type == "mlp":
        return model(batch["mlp_input"])
    return model(batch["text_features"], batch["control_features"], batch["prior_vector"])


def split_data_random(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    grouped = {}
    for item in data:
        grouped.setdefault(item["emotion"], []).append(dict(item))

    train_data, val_data, test_data = [], [], []
    rng = random.Random(seed)

    for emotion_items in grouped.values():
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

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)
    return train_data, val_data, test_data


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


def split_data_source_holdout(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    grouped = {}
    for item in data:
        grouped.setdefault(item["emotion"], []).append(dict(item))

    train_data, val_data, test_data = [], [], []
    rng = random.Random(seed)

    for emotion, emotion_items in grouped.items():
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


def split_data(data, split_mode="random", train_ratio=0.7, val_ratio=0.15, seed=42):
    if split_mode == "random":
        return split_data_random(data, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    if split_mode == "source_holdout":
        return split_data_source_holdout(data, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def compute_metrics(predictions, targets):
    diff = predictions - targets
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    active_mask = (targets > 0.05).float()
    active_denominator = max(active_mask.sum().item(), 1.0)
    active_mae = ((diff.abs() * active_mask).sum() / active_denominator).item()
    return {"mae": mae, "rmse": rmse, "active_mae": active_mae}


def compute_group_metrics(predictions, targets, groups):
    stats = {}
    unique_groups = sorted(set(groups))
    for group in unique_groups:
        indices = [i for i, g in enumerate(groups) if g == group]
        group_pred = predictions[indices]
        group_target = targets[indices]
        group_metrics = compute_metrics(group_pred, group_target)
        group_metrics["count"] = len(indices)
        stats[group] = group_metrics
    return stats


def evaluate(model, loader, criterion, model_type, device):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    emotions = []
    intensity_buckets = []
    sources = []

    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            pred = forward_model(model, tensor_batch, model_type)
            loss = criterion(pred, tensor_batch["target"])
            total_loss += loss.item()
            preds.append(pred.cpu())
            targets.append(tensor_batch["target"].cpu())
            emotions.extend(batch["emotion"])
            intensity_buckets.extend(batch["intensity_bucket"])
            sources.extend(batch["source"])

    predictions = torch.cat(preds, dim=0)
    labels = torch.cat(targets, dim=0)
    metrics = compute_metrics(predictions, labels)
    metrics["loss"] = total_loss / max(len(loader), 1)
    metrics["per_emotion"] = compute_group_metrics(predictions, labels, emotions)
    metrics["per_intensity_bucket"] = compute_group_metrics(predictions, labels, intensity_buckets)
    metrics["per_source"] = compute_group_metrics(predictions, labels, sources)
    return metrics


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = [normalize_item(item) for item in json.load(f)]

    train_data, val_data, test_data = split_data(data, split_mode=args.split_mode, seed=args.seed)
    encoder = load_text_encoder(args.encoder_name)

    train_dataset = ExpressionDataset(train_data, encoder, args.use_semantic_features, args.use_prior)
    val_dataset = ExpressionDataset(val_data, encoder, args.use_semantic_features, args.use_prior)
    test_dataset = ExpressionDataset(test_data, encoder, args.use_semantic_features, args.use_prior)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    sample = train_dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_type, sample).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            pred = forward_model(model, tensor_batch, args.model_type)
            loss = criterion(pred, tensor_batch["target"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, criterion, args.model_type, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_active_mae": val_metrics["active_mae"],
        })

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} | val_mae={val_metrics['mae']:.6f}")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, args.model_type, device)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.run_name or args.model_type
    checkpoint_path = os.path.join(args.output_dir, f"{tag}_mapper.pt")
    metrics_path = os.path.join(args.output_dir, f"{tag}_metrics.json")

    checkpoint = {
        "model_type": args.model_type,
        "encoder_name": getattr(encoder, "name", args.encoder_name),
        "use_semantic_features": args.use_semantic_features,
        "use_prior": args.use_prior,
        "state_dict": best_state,
        "emotion_list": EMOTION_LIST,
        "control_feature_dim": sample["control_features"].shape[0],
        "text_feature_dim": sample["text_features"].shape[0],
        "output_dim": sample["target"].shape[0],
    }
    torch.save(checkpoint, checkpoint_path)

    metrics_payload = {
        "config": {
            "run_name": tag,
            "model_type": args.model_type,
            "encoder_name": getattr(encoder, "name", args.encoder_name),
            "use_semantic_features": args.use_semantic_features,
            "use_prior": args.use_prior,
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
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"Encoder: {getattr(encoder, 'name', args.encoder_name)}")
    print(f"Best checkpoint saved to {checkpoint_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Test metrics | loss={test_metrics['loss']:.6f} | mae={test_metrics['mae']:.6f} | rmse={test_metrics['rmse']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train text-to-expression models.")
    parser.add_argument("--data-path", default="data/train.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--model-type", choices=["mlp", "prior_fusion"], default="prior_fusion")
    parser.add_argument("--encoder-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["random", "source_holdout"], default="random")
    parser.add_argument("--use-semantic-features", dest="use_semantic_features", action="store_true")
    parser.add_argument("--no-semantic-features", dest="use_semantic_features", action="store_false")
    parser.set_defaults(use_semantic_features=True)
    parser.add_argument("--use-prior", dest="use_prior", action="store_true")
    parser.add_argument("--no-prior", dest="use_prior", action="store_false")
    parser.set_defaults(use_prior=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
