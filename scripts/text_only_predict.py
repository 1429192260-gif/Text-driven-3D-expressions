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
from utils.hf_emotion_frontend import DEFAULT_LOCAL_MODEL_DIR, HFEmotionFrontend
from utils.param_utils import clamp_params_dict, vector_to_params_dict
from utils.text_encoder import load_text_encoder
from utils.text_features import extract_semantic_features

EMOTION_LIST = [
    "happy", "sad", "angry", "surprise",
    "disgust", "concern", "bored", "calm",
]
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

EMOTION_INTENSITY_RULES = {
    'happy': 0.68,
    'sad': 0.54,
    'angry': 0.72,
    'surprise': 0.74,
    'disgust': 0.62,
    'concern': 0.56,
    'bored': 0.48,
    'calm': 0.42,
}

BOOST_WORDS = ['非常', '特别', '太', '超级', '真的', '好', '太太']
LOW_WORDS = ['有点', '一点', '还算', '稍微']


def estimate_intensity_from_text(text, emotion):
    intensity = EMOTION_INTENSITY_RULES.get(emotion, 0.6)
    intensity += min(sum(text.count(w) for w in BOOST_WORDS), 3) * 0.04
    intensity -= min(sum(text.count(w) for w in LOW_WORDS), 2) * 0.05
    intensity += min(text.count('!') + text.count('！'), 3) * 0.03
    intensity += min(text.count('?') + text.count('？'), 2) * 0.02 if emotion == 'surprise' else 0.0
    return max(0.3, min(0.9, round(intensity, 2)))


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
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def build_classifier_features(text, encoder, use_semantic_features):
    text_emb = encoder.encode(text, convert_to_numpy=True)
    semantic = extract_semantic_features(text) if use_semantic_features else []
    return torch.tensor(list(text_emb) + semantic, dtype=torch.float32).unsqueeze(0)


def build_expression_features(text, emotion, intensity, encoder, use_semantic_features, use_prior, output_dim, control_feature_dim):
    text_emb = encoder.encode(text, convert_to_numpy=True)
    semantic_features = extract_semantic_features(text) if use_semantic_features else []
    control_features = emotion_to_onehot(emotion) + [float(intensity)] + semantic_features
    if len(control_features) < control_feature_dim:
        control_features = control_features + [0.0] * (control_feature_dim - len(control_features))
    elif len(control_features) > control_feature_dim:
        control_features = control_features[:control_feature_dim]
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
    parser.add_argument("--frontend-backend", choices=["learned", "hf_local"], default="learned")
    parser.add_argument("--hf-model-dir", default=DEFAULT_LOCAL_MODEL_DIR)
    parser.add_argument("--classifier-checkpoint", default="outputs/text_only_emotion_classifier.pt")
    parser.add_argument("--expression-checkpoint", default="outputs/prior_fusion_full_mapper.pt")
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frontend_backend == "hf_local":
        frontend = HFEmotionFrontend(model_dir=args.hf_model_dir)
        frontend_pred = frontend.predict(args.text)
        predicted_emotion = frontend_pred["emotion"]
        predicted_intensity = frontend_pred["intensity"]
        emotion_confidence = frontend_pred["confidence"]
        raw_label = frontend_pred["raw_label"]
    else:
        classifier, classifier_ckpt = load_classifier(args.classifier_checkpoint)
        classifier_encoder = load_text_encoder(classifier_ckpt.get("encoder_name", DEFAULT_MODEL_NAME))
        clf_features = build_classifier_features(args.text, classifier_encoder, classifier_ckpt.get("use_semantic_features", True))
        with torch.no_grad():
            emotion_logits = classifier(clf_features)
            emotion_probs = F.softmax(emotion_logits, dim=-1)[0]

        emotion_idx = int(torch.argmax(emotion_probs).item())
        predicted_emotion = classifier_ckpt["emotion_list"][emotion_idx]
        predicted_intensity = estimate_intensity_from_text(args.text, predicted_emotion)
        emotion_confidence = round(float(emotion_probs[emotion_idx].item()), 4)
        raw_label = predicted_emotion

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
        control_feature_dim=expr_ckpt.get("control_feature_dim", len(EMOTION_LIST) + 1),
    )

    with torch.no_grad():
        if expr_ckpt["model_type"] == "mlp":
            pred = expression_model(features["mlp_input"])[0].tolist()
        else:
            pred = expression_model(features["text_features"], features["control_features"], features["prior_vector"])[0].tolist()

    params = clamp_params_dict(vector_to_params_dict(pred))

    print(f"text: {args.text}")
    print(f"frontend_backend: {args.frontend_backend}")
    print(f"predicted_emotion: {predicted_emotion} (p={emotion_confidence:.4f})")
    print(f"raw_frontend_label: {raw_label}")
    print(f"estimated_intensity_value: {predicted_intensity:.2f}")
    print("predicted_params:")
    for k, v in params.items():
        if abs(v) > 0.05:
            print(f"{k}: {v:.4f}")

    if not args.no_vis:
        from visualize import draw_face
        draw_face(params, title=args.text)


if __name__ == "__main__":
    main()
