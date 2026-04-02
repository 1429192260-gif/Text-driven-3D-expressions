import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_mapper import MLPMapper
from models.prior_fusion_mapper import PriorFusionMapper
from rule_mapping import rule_mapping
from utils.param_utils import clamp_params_dict, vector_to_params_dict
from utils.text_encoder import load_text_encoder
from utils.text_features import extract_semantic_features

EMOTION_LIST = [
    "happy", "sad", "angry", "surprise",
    "disgust", "concern", "bored", "calm",
]

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def emotion_to_onehot(emotion):
    vec = [0.0] * len(EMOTION_LIST)
    vec[EMOTION_LIST.index(emotion)] = 1.0
    return vec


def build_features(text, emotion, intensity, encoder, use_semantic_features=True, use_prior=True, output_dim=22):
    text_emb = encoder.encode(text, convert_to_numpy=True)
    semantic_features = extract_semantic_features(text) if use_semantic_features else []
    control_features = emotion_to_onehot(emotion) + [float(intensity)] + semantic_features
    prior_vector = rule_mapping(emotion, float(intensity)) if use_prior else [0.0] * output_dim
    return {
        "text_features": torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0),
        "control_features": torch.tensor(control_features, dtype=torch.float32).unsqueeze(0),
        "prior_vector": torch.tensor(prior_vector, dtype=torch.float32).unsqueeze(0),
        "mlp_input": torch.tensor(list(text_emb) + control_features, dtype=torch.float32).unsqueeze(0),
    }


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_type = checkpoint["model_type"]
        if model_type == "mlp":
            model = MLPMapper(input_dim=checkpoint["text_feature_dim"] + checkpoint["control_feature_dim"], output_dim=checkpoint["output_dim"])
        else:
            model = PriorFusionMapper(text_dim=checkpoint["text_feature_dim"], control_dim=checkpoint["control_feature_dim"], prior_dim=checkpoint["output_dim"], output_dim=checkpoint["output_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        return model, checkpoint

    model = MLPMapper(input_dim=384 + len(EMOTION_LIST) + 1, output_dim=22)
    model.load_state_dict(checkpoint)
    return model, {"model_type": "mlp", "encoder_name": DEFAULT_MODEL_NAME, "use_semantic_features": False, "use_prior": False, "output_dim": 22}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict facial expression parameters from text.")
    parser.add_argument("--checkpoint", default="outputs/prior_fusion_mapper.pt")
    parser.add_argument("--text", default="????????????")
    parser.add_argument("--emotion", default="happy", choices=EMOTION_LIST)
    parser.add_argument("--intensity", type=float, default=0.8)
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model, checkpoint = load_model(args.checkpoint)
    encoder = load_text_encoder(checkpoint.get("encoder_name", DEFAULT_MODEL_NAME))
    model.eval()

    features = build_features(
        args.text,
        args.emotion,
        args.intensity,
        encoder,
        use_semantic_features=checkpoint.get("use_semantic_features", False),
        use_prior=checkpoint.get("use_prior", False),
        output_dim=checkpoint.get("output_dim", 22),
    )

    with torch.no_grad():
        if checkpoint["model_type"] == "mlp":
            pred = model(features["mlp_input"])[0].tolist()
        else:
            pred = model(features["text_features"], features["control_features"], features["prior_vector"])[0].tolist()

    params = clamp_params_dict(vector_to_params_dict(pred))

    print("????:", args.text)
    print("????:", args.emotion)
    print("???:", args.intensity)
    print("???:", getattr(encoder, "name", checkpoint.get("encoder_name", DEFAULT_MODEL_NAME)))
    print("????:")
    for k, v in params.items():
        if abs(v) > 0.05:
            print(f"{k}: {v:.4f}")

    if not args.no_vis:
        from visualize import draw_face
        draw_face(params, title=args.text)


if __name__ == "__main__":
    main()
