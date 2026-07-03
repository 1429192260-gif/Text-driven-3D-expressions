import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.train_mlp import (
    ExpressionDataset,
    build_model,
    collate_batch,
    evaluate,
    forward_model,
    load_text_encoder,
    normalize_item,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train on source dataset and evaluate on target dataset.')
    parser.add_argument('--train-data-path', required=True)
    parser.add_argument('--test-data-path', required=True)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--model-type', choices=['mlp', 'prior_fusion'], default='prior_fusion')
    parser.add_argument('--encoder-name', default='paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--use-semantic-features', dest='use_semantic_features', action='store_true')
    parser.add_argument('--no-semantic-features', dest='use_semantic_features', action='store_false')
    parser.set_defaults(use_semantic_features=True)
    parser.add_argument('--use-prior', dest='use_prior', action='store_true')
    parser.add_argument('--no-prior', dest='use_prior', action='store_false')
    parser.set_defaults(use_prior=True)
    return parser.parse_args()


def split_train_val(data, val_ratio, seed):
    grouped = {}
    for item in data:
        grouped.setdefault(item['emotion'], []).append(dict(item))
    train_data, val_data = [], []
    rng = random.Random(seed)
    for emotion_items in grouped.values():
        rng.shuffle(emotion_items)
        n = len(emotion_items)
        val_count = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        split = max(1, n - val_count)
        train_data.extend(emotion_items[:split])
        val_data.extend(emotion_items[split:])
    rng.shuffle(train_data)
    rng.shuffle(val_data)
    return train_data, val_data


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.train_data_path, 'r', encoding='utf-8') as f:
        train_source = [normalize_item(item) for item in json.load(f)]
    with open(args.test_data_path, 'r', encoding='utf-8') as f:
        test_data = [normalize_item(item) for item in json.load(f)]

    train_data, val_data = split_train_val(train_source, args.val_ratio, args.seed)
    encoder = load_text_encoder(args.encoder_name)

    train_dataset = ExpressionDataset(train_data, encoder, args.use_semantic_features, args.use_prior)
    val_dataset = ExpressionDataset(val_data, encoder, args.use_semantic_features, args.use_prior)
    test_dataset = ExpressionDataset(test_data, encoder, args.use_semantic_features, args.use_prior)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    sample = train_dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model_type, sample).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            pred = forward_model(model, tensor_batch, args.model_type)
            loss = criterion(pred, tensor_batch['target'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, criterion, args.model_type, device)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_metrics['loss'], 'val_mae': val_metrics['mae']})
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} | val_mae={val_metrics['mae']:.6f}")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, args.model_type, device)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f'{args.run_name}_mapper.pt')
    metrics_path = os.path.join(args.output_dir, f'{args.run_name}_metrics.json')

    checkpoint = {
        'model_type': args.model_type,
        'encoder_name': getattr(encoder, 'name', args.encoder_name),
        'use_semantic_features': args.use_semantic_features,
        'use_prior': args.use_prior,
        'state_dict': best_state,
        'control_feature_dim': sample['control_features'].shape[0],
        'text_feature_dim': sample['text_features'].shape[0],
        'output_dim': sample['target'].shape[0],
    }
    torch.save(checkpoint, checkpoint_path)

    payload = {
        'config': {
            'run_name': args.run_name,
            'train_data_path': args.train_data_path,
            'test_data_path': args.test_data_path,
            'model_type': args.model_type,
            'encoder_name': getattr(encoder, 'name', args.encoder_name),
            'use_semantic_features': args.use_semantic_features,
            'use_prior': args.use_prior,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
            'splits': {'train': len(train_data), 'val': len(val_data), 'test': len(test_data)},
        },
        'best_val_loss': best_val,
        'test_metrics': test_metrics,
        'history': history,
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'Best checkpoint saved to {checkpoint_path}')
    print(f'Metrics saved to {metrics_path}')
    print(f"Test metrics | loss={test_metrics['loss']:.6f} | mae={test_metrics['mae']:.6f} | rmse={test_metrics['rmse']:.6f} | active_mae={test_metrics['active_mae']:.6f}")


if __name__ == '__main__':
    main()
