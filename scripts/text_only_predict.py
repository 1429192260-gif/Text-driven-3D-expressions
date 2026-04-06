import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_mapper import MLPMapper
from models.prior_fusion_mapper import PriorFusionMapper
from models.text_only_classifier import TextOnlyClassifier
from rule_mapping import rule_mapping
from utils.param_utils import clamp_params_dict, vector_to_params_dict
from utils.text_encoder import load_text_encoder
from utils.text_features import extract_semantic_features

EMOTION_LIST = [
    "happy", "sad", "angry", "surprise",
    "disgust", "concern", "bored", "calm",
]
INTENSITY_TO_VALUE = {
    "weak": 0.35,
    "medium": 0.60,
    "strong": 0.82,
}
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def emotion_to_onehot(emotion):
    vec = [0.0] * len(EMOTION_LIST)
    vec[EMOTION_LIST.index(emotion)] = 1.0
    return vec


def load_expression_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["model_type"] == "mlp":
        model = MLPMapper(input_dim=checkpoint["text_feature_dim"] + checkpoint["control_feature_dim"], output_dim=checkpoint["output_dim"])
    else:
        model = PriorFusionMapper(text_dim=checkpoint["text_feature_dim"], control_dim=checkpoint["control_feature_dim"], prior_dim=checkpoint["output_dim"], output_dim=checkpoint["output_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def load_classifier(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = TextOnlyClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_emotions=len(checkpoint["emotion_list"]),
        num_intensity=len(checkpoint["intensity_labels"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def build_classifier_features(text, encoder, use_semantic_features):
    text_emb = encoder.encode(text, convert_to_numpy=True)
    semantic = extract_semantic_features(text) if use_semantic_features else []
    return torch.tensor(list(text_emb) + semantic, dtype=torch.float32).unsqueeze(0)


def build_expression_features(text, emotion, intensity, encoder, use_semantic_features, use_prior, output_dim):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Text-only facial expression prediction.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--classifier-checkpoint", default="outputs/text_only_classifier.pt")
    parser.add_argument("--expression-checkpoint", default="outputs/prior_fusion_full_mapper.pt")
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    classifier, classifier_ckpt = load_classifier(args.classifier_checkpoint)
    classifier_encoder = load_text_encoder(classifier_ckpt.get("encoder_name", DEFAULT_MODEL_NAME))

    clf_features = build_classifier_features(args.text, classifier_encoder, classifier_ckpt.get("use_semantic_features", True))
    with torch.no_grad():
        emotion_logits, intensity_logits = classifier(clf_features)
        emotion_probs = F.softmax(emotion_logits, dim=-1)[0]
        intensity_probs = F.softmax(intensity_logits, dim=-1)[0]

    emotion_idx = int(torch.argmax(emotion_probs).item())
    intensity_idx = int(torch.argmax(intensity_probs).item())
    predicted_emotion = classifier_ckpt["emotion_list"][emotion_idx]
    predicted_bucket = classifier_ckpt["intensity_labels"][intensity_idx]
    predicted_intensity = INTENSITY_TO_VALUE[predicted_bucket]

    expression_model, expr_ckpt = load_expression_model(args.expression_checkpoint)
    expr_encoder = load_text_encoder(expr_ckpt.get("encoder_name", DEFAULT_MODEL_NAME))
    features = build_expression_features(
        args.text,
        predicted_emotion,
        predicted_intensity,
        expr_encoder,
        use_semantic_features=expr_ckpt.get("use_semantic_features", False),
        use_prior=expr_ckpt.get("use_prior", False),
        output_dim=expr_ckpt.get("output_dim", 22),
    )

    with torch.no_grad():
        if expr_ckpt["model_type"] == "mlp":
            pred = expression_model(features["mlp_input"])[0].tolist()
        else:
            pred = expression_model(features["text_features"], features["control_features"], features["prior_vector"])[0].tolist()

    params = clamp_params_dict(vector_to_params_dict(pred))

    print(f"text: {args.text}")
    print(f"predicted_emotion: {predicted_emotion} (p={emotion_probs[emotion_idx].item():.4f})")
    print(f"predicted_intensity_bucket: {predicted_bucket} (p={intensity_probs[intensity_idx].item():.4f})")
    print(f"predicted_intensity_value: {predicted_intensity:.2f}")
    print("predicted_params:")
    for k, v in params.items():
        if abs(v) > 0.05:
            print(f"{k}: {v:.4f}")

    if not args.no_vis:
        from visualize import draw_face
        draw_face(params, title=args.text)


if __name__ == "__main__":
    main()
