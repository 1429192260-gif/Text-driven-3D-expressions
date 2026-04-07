import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_mapper import MLPMapper
from models.prior_fusion_mapper import PriorFusionMapper
from models.text_only_classifier import TextOnlyClassifier
from rule_mapping import rule_mapping
from utils.hf_emotion_frontend import DEFAULT_LOCAL_MODEL_DIR, HFEmotionFrontend
from utils.param_utils import clamp_params_dict, params_dict_to_vector
from utils.text_encoder import load_text_encoder
from utils.text_features import extract_semantic_features

EMOTION_LIST = [
    "happy", "sad", "angry", "surprise",
    "disgust", "concern", "bored", "calm",
]
AUGMENTED_PREFIXES = ("weibo_",)
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


def normalize_item(item):
    normalized = dict(item)
    if 'param_vector' not in normalized:
        normalized['param_vector'] = params_dict_to_vector(normalized['params'])
    normalized.setdefault('source', 'manual')
    return normalized


def is_augmented_source(source):
    source = source or 'manual'
    return source.startswith(AUGMENTED_PREFIXES)


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
        grouped.setdefault(item['emotion'], []).append(dict(item))
    train_data, val_data, test_data = [], [], []
    rng = random.Random(seed)
    for emotion_items in grouped.values():
        manual_items = [item for item in emotion_items if not is_augmented_source(item.get('source', 'manual'))]
        augmented_items = [item for item in emotion_items if is_augmented_source(item.get('source', 'manual'))]
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint['model_type'] == 'mlp':
        model = MLPMapper(input_dim=checkpoint['text_feature_dim'] + checkpoint['control_feature_dim'], output_dim=checkpoint['output_dim'])
    else:
        model = PriorFusionMapper(text_dim=checkpoint['text_feature_dim'], control_dim=checkpoint['control_feature_dim'], prior_dim=checkpoint['output_dim'], output_dim=checkpoint['output_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def load_classifier(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = TextOnlyClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_emotions=len(checkpoint['emotion_list']),
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def build_expression_features(text, emotion, intensity, encoder, expr_ckpt):
    text_emb = encoder.encode(text, convert_to_numpy=True)
    semantic_features = extract_semantic_features(text) if expr_ckpt.get('use_semantic_features', False) else []
    control_features = emotion_to_onehot(emotion) + [float(intensity)] + semantic_features
    control_dim = expr_ckpt.get('control_feature_dim', len(control_features))
    if len(control_features) < control_dim:
        control_features = control_features + [0.0] * (control_dim - len(control_features))
    elif len(control_features) > control_dim:
        control_features = control_features[:control_dim]
    prior_vector = rule_mapping(emotion, float(intensity)) if expr_ckpt.get('use_prior', False) else [0.0] * expr_ckpt.get('output_dim', 22)
    return {
        'text_features': torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0),
        'control_features': torch.tensor(control_features, dtype=torch.float32).unsqueeze(0),
        'prior_vector': torch.tensor(prior_vector, dtype=torch.float32).unsqueeze(0),
        'mlp_input': torch.tensor(list(text_emb) + control_features, dtype=torch.float32).unsqueeze(0),
    }


def run_expression(model, expr_ckpt, features):
    with torch.no_grad():
        if expr_ckpt['model_type'] == 'mlp':
            pred = model(features['mlp_input'])[0]
        else:
            pred = model(features['text_features'], features['control_features'], features['prior_vector'])[0]
    return pred


def compute_metrics(predictions, targets):
    diff = predictions - targets
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    active_mask = (targets > 0.05).float()
    active_denominator = max(active_mask.sum().item(), 1.0)
    active_mae = ((diff.abs() * active_mask).sum() / active_denominator).item()
    return {'mae': mae, 'rmse': rmse, 'active_mae': active_mae}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate text-only pipeline against upper bound.')
    parser.add_argument('--data-path', default='data/full_samples_804.json')
    parser.add_argument('--expression-checkpoint', default='outputs/prior_fusion_full_mapper.pt')
    parser.add_argument('--frontend-backend', choices=['learned', 'hf_local'], default='learned')
    parser.add_argument('--classifier-checkpoint', default='outputs/text_only_emotion_base_classifier.pt')
    parser.add_argument('--hf-model-dir', default=DEFAULT_LOCAL_MODEL_DIR)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-path', default='docs/text_only_vs_upper_bound_804.json')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = [normalize_item(item) for item in json.load(f)]

    _, _, test_data = split_data_source_holdout(data, seed=args.seed)
    test_data = [item for item in test_data if item.get('source', 'manual') == 'manual']

    expression_model, expr_ckpt = load_expression_model(args.expression_checkpoint)
    expr_encoder = load_text_encoder(expr_ckpt.get('encoder_name', DEFAULT_MODEL_NAME))

    classifier = None
    clf_ckpt = None
    clf_encoder = None
    hf_frontend = None
    if args.frontend_backend == 'hf_local':
        hf_frontend = HFEmotionFrontend(model_dir=args.hf_model_dir)
    else:
        classifier, clf_ckpt = load_classifier(args.classifier_checkpoint)
        clf_encoder = load_text_encoder(clf_ckpt.get('encoder_name', DEFAULT_MODEL_NAME))

    upper_preds = []
    text_only_preds = []
    targets = []
    records = []

    for item in test_data:
        target = torch.tensor(item['param_vector'], dtype=torch.float32)
        upper_features = build_expression_features(item['text'], item['emotion'], float(item['intensity']), expr_encoder, expr_ckpt)
        upper_pred = run_expression(expression_model, expr_ckpt, upper_features)

        if args.frontend_backend == 'hf_local':
            frontend_pred = hf_frontend.predict(item['text'])
            pred_emotion = frontend_pred['emotion']
            pred_intensity = frontend_pred['intensity']
            emotion_confidence = frontend_pred['confidence']
            raw_label = frontend_pred['raw_label']
        else:
            clf_features = torch.tensor(list(clf_encoder.encode(item['text'], convert_to_numpy=True)) + (extract_semantic_features(item['text']) if clf_ckpt.get('use_semantic_features', True) else []), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emotion_logits = classifier(clf_features)
                emotion_probs = F.softmax(emotion_logits, dim=-1)[0]
            pred_emotion = clf_ckpt['emotion_list'][int(torch.argmax(emotion_probs).item())]
            pred_intensity = estimate_intensity_from_text(item['text'], pred_emotion)
            emotion_confidence = round(float(emotion_probs.max().item()), 4)
            raw_label = pred_emotion
        text_features = build_expression_features(item['text'], pred_emotion, pred_intensity, expr_encoder, expr_ckpt)
        text_pred = run_expression(expression_model, expr_ckpt, text_features)

        upper_preds.append(upper_pred)
        text_only_preds.append(text_pred)
        targets.append(target)
        records.append({
            'text': item['text'],
            'true_emotion': item['emotion'],
            'true_intensity': float(item['intensity']),
            'pred_emotion': pred_emotion,
            'pred_intensity': pred_intensity,
            'emotion_confidence': emotion_confidence,
            'raw_frontend_label': raw_label,
        })

    upper_preds = torch.stack(upper_preds)
    text_only_preds = torch.stack(text_only_preds)
    targets = torch.stack(targets)

    result = {
        'config': {
            'data_path': args.data_path,
            'expression_checkpoint': args.expression_checkpoint,
            'classifier_checkpoint': args.classifier_checkpoint,
            'frontend_backend': args.frontend_backend,
            'hf_model_dir': args.hf_model_dir,
            'seed': args.seed,
            'test_count': len(test_data),
            'test_source': 'manual_only',
        },
        'upper_bound': compute_metrics(upper_preds, targets),
        'text_only': compute_metrics(text_only_preds, targets),
        'records_preview': records[:10],
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'saved to {args.output_path}')


if __name__ == '__main__':
    main()
