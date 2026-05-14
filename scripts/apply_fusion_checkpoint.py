import argparse
import sys
from pathlib import Path

import torch


def apply_v4(repo: Path, checkpoint_path: Path, prediction_path: Path, split: str, output_path: Path, batch_size: int, no_cuda: bool):
    sys.path.insert(0, str(repo / "scripts"))
    from train_uncertainty_prior_fusion_v4 import (
        EMOTIONS,
        UncertaintyPriorFusionAdapter,
        build_uncertainty_features,
        evaluate,
        load_split,
        make_time_features,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    settings = ckpt.get("args", {})
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    dataset_dir = repo / settings.get("dataset_dir", "external/EmoAva/dataset")
    pred, gold, mask, texts = load_split(prediction_path, dataset_dir, split)
    ablation = settings.get("ablation", "full")
    features = build_uncertainty_features(
        texts,
        ablation,
        float(settings.get("temperature", 0.85)),
        float(settings.get("evidence_scale", 3.0)),
    )
    model = UncertaintyPriorFusionAdapter(
        num_emotions=len(EMOTIONS),
        stats_dim=6,
        exp_dim=pred.shape[-1],
        hidden=int(settings.get("hidden", 64)),
        dropout=float(settings.get("dropout", 0.05)),
        use_global_branch=ablation != "no_global_branch",
        use_prior_branch=ablation != "no_prior_branch",
    ).to(device)
    model.load_state_dict(ckpt["model"])
    time_features = make_time_features(pred.shape[1], device)
    _, adjusted = evaluate(model, pred, gold, mask, features, time_features, batch_size, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adjusted, output_path)
    print(output_path)


def apply_v5(repo: Path, checkpoint_path: Path, prediction_path: Path, split: str, output_path: Path, batch_size: int, no_cuda: bool):
    sys.path.insert(0, str(repo / "scripts"))
    from train_learned_affect_prior_fusion_v5 import (
        SCORE_EMOTIONS,
        LearnedAffectPriorFusionAdapter,
        build_text_features,
        evaluate,
        load_or_build_text_embeddings,
        load_split,
        make_time_features,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    settings = ckpt.get("args", {})
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    dataset_dir = repo / settings.get("dataset_dir", "external/EmoAva/dataset")
    pred, gold, mask, texts = load_split(prediction_path, dataset_dir, split)
    embeddings = load_or_build_text_embeddings(
        texts,
        split,
        repo,
        settings.get("bert_model"),
        settings.get("bert_cache_dir", "outputs/when_words_smile_prior_v5/text_embeddings"),
        int(settings.get("bert_batch_size", 32)),
        int(settings.get("bert_max_length", 64)),
        device,
    )
    features = build_text_features(texts, embeddings)
    ablation = settings.get("ablation", "full")
    model = LearnedAffectPriorFusionAdapter(
        score_dim=len(SCORE_EMOTIONS),
        stats_dim=8,
        text_emb_dim=features["embeddings"].shape[-1],
        affect_dim=int(settings.get("affect_dim", 8)),
        exp_dim=pred.shape[-1],
        hidden=int(settings.get("hidden", 64)),
        dropout=float(settings.get("dropout", 0.05)),
        use_global_branch=ablation != "no_global_branch",
        use_prior_branch=ablation != "no_prior_branch",
        use_text_prior=ablation != "no_text_prior",
    ).to(device)
    model.load_state_dict(ckpt["model"])
    time_features = make_time_features(pred.shape[1], device)
    _, adjusted, _ = evaluate(model, pred, gold, mask, features, time_features, batch_size, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adjusted, output_path)
    print(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["v4", "v5"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    checkpoint_path = repo / args.checkpoint
    prediction_path = repo / args.prediction
    output_path = repo / args.output
    if args.kind == "v4":
        apply_v4(repo, checkpoint_path, prediction_path, args.split, output_path, args.batch_size, args.no_cuda)
    else:
        apply_v5(repo, checkpoint_path, prediction_path, args.split, output_path, args.batch_size, args.no_cuda)


if __name__ == "__main__":
    main()
