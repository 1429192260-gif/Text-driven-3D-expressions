import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.text_only_classifier import TextOnlyClassifier
from scripts.evaluate_text_only_pipeline import (
    build_expression_features,
    compute_metrics,
    load_expression_model,
    normalize_item,
    run_expression,
    split_data_source_holdout,
)
from utils.hf_emotion_frontend import HFEmotionFrontend
from utils.text_encoder import load_text_encoder
from utils.text_features import extract_semantic_features

DATA_PATH = 'data/full_samples_804.json'
EXPR_CKPT = 'outputs/prior_fusion_804_source_holdout_strict_mapper.pt'
LEARNED_CKPT = 'outputs/text_only_emotion8_base_classifier.pt'
OUTPUT_JSON = 'docs/per_emotion_error_analysis_text_only.json'
OUTPUT_MD = 'docs/per_emotion_error_analysis_text_only.md'
EMOTION_ORDER = ['happy', 'sad', 'angry', 'surprise', 'disgust', 'concern', 'bored', 'calm']
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


def estimate_intensity_from_text(text, emotion):
    intensity = EMOTION_INTENSITY_RULES.get(emotion, 0.6)
    intensity += min(sum(text.count(w) for w in BOOST_WORDS), 3) * 0.04
    intensity -= min(sum(text.count(w) for w in LOW_WORDS), 2) * 0.05
    intensity += min(text.count('!') + text.count('！'), 3) * 0.03
    intensity += min(text.count('?') + text.count('？'), 2) * 0.02 if emotion == 'surprise' else 0.0
    return max(0.3, min(0.9, round(intensity, 2)))


def predict_learned_batch(texts, checkpoint_path):
    classifier, ckpt = load_classifier(checkpoint_path)
    encoder = load_text_encoder(ckpt.get('encoder_name', 'paraphrase-multilingual-MiniLM-L12-v2'))
    features = []
    for text in texts:
        feat = list(encoder.encode(text, convert_to_numpy=True))
        if ckpt.get('use_semantic_features', True):
            feat += extract_semantic_features(text)
        features.append(feat)
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        emotion_logits = classifier(features)
        probs = F.softmax(emotion_logits, dim=-1)
    preds = []
    for idx, text in enumerate(texts):
        pred_emotion = ckpt['emotion_list'][int(torch.argmax(probs[idx]).item())]
        preds.append({
            'emotion': pred_emotion,
            'intensity': estimate_intensity_from_text(text, pred_emotion),
            'confidence': round(float(probs[idx].max().item()), 4),
            'raw_frontend_label': pred_emotion,
        })
    return preds


def evaluate_variant(name, predictions, test_data, expr_model, expr_ckpt, expr_encoder):
    grouped_preds = defaultdict(list)
    grouped_targets = defaultdict(list)
    emotion_correct = defaultdict(int)
    emotion_count = defaultdict(int)

    for item, pred in zip(test_data, predictions):
        target = torch.tensor(item['param_vector'], dtype=torch.float32)
        features = build_expression_features(item['text'], pred['emotion'], pred['intensity'], expr_encoder, expr_ckpt)
        pred_vector = run_expression(expr_model, expr_ckpt, features)
        grouped_preds[item['emotion']].append(pred_vector)
        grouped_targets[item['emotion']].append(target)
        emotion_correct[item['emotion']] += int(pred['emotion'] == item['emotion'])
        emotion_count[item['emotion']] += 1

    metrics = {}
    for emotion in EMOTION_ORDER:
        preds = torch.stack(grouped_preds[emotion])
        targets = torch.stack(grouped_targets[emotion])
        m = compute_metrics(preds, targets)
        metrics[emotion] = {
            'param_mae': round(m['mae'], 6),
            'param_rmse': round(m['rmse'], 6),
            'active_mae': round(m['active_mae'], 6),
            'emotion_acc': round(emotion_correct[emotion] / max(emotion_count[emotion], 1), 6),
            'count': emotion_count[emotion],
        }
    return metrics


def main():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = [normalize_item(item) for item in json.load(f)]
    _, _, test_data = split_data_source_holdout(data, seed=42)
    test_data = [item for item in test_data if item.get('source', 'manual') == 'manual']
    texts = [item['text'] for item in test_data]

    expr_model, expr_ckpt = load_expression_model(EXPR_CKPT)
    expr_encoder = load_text_encoder(expr_ckpt.get('encoder_name', 'paraphrase-multilingual-MiniLM-L12-v2'))

    hf_raw = HFEmotionFrontend(mode='raw')
    hf_rules = HFEmotionFrontend(mode='rules_v4')

    predictions = {
        'learned': predict_learned_batch(texts, LEARNED_CKPT),
        'hf_raw': [hf_raw.predict(text) for text in texts],
        'hf_rules_v4': [hf_rules.predict(text) for text in texts],
    }

    result = {
        'config': {
            'data_path': DATA_PATH,
            'expression_checkpoint': EXPR_CKPT,
            'learned_checkpoint': LEARNED_CKPT,
            'test_count': len(test_data),
            'test_source': 'manual_only_source_holdout',
        },
        'per_emotion': {},
    }

    for name, preds in predictions.items():
        result['per_emotion'][name] = evaluate_variant(name, preds, test_data, expr_model, expr_ckpt, expr_encoder)

    Path(OUTPUT_JSON).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = []
    lines.append('# Per-Emotion Error Analysis for Text-Only Front-Ends')
    lines.append('')
    lines.append('- setting: manual-only source-holdout test set')
    lines.append('')
    lines.append('## Parameter MAE by Emotion')
    lines.append('')
    lines.append('| Emotion | learned | hf_raw | hf_rules_v4 |')
    lines.append('| --- | ---: | ---: | ---: |')
    for emotion in EMOTION_ORDER:
        lines.append(
            f"| {emotion} | {result['per_emotion']['learned'][emotion]['param_mae']:.6f} | {result['per_emotion']['hf_raw'][emotion]['param_mae']:.6f} | {result['per_emotion']['hf_rules_v4'][emotion]['param_mae']:.6f} |"
        )
    lines.append('')
    lines.append('## Emotion Accuracy by Emotion')
    lines.append('')
    lines.append('| Emotion | learned | hf_raw | hf_rules_v4 |')
    lines.append('| --- | ---: | ---: | ---: |')
    for emotion in EMOTION_ORDER:
        lines.append(
            f"| {emotion} | {result['per_emotion']['learned'][emotion]['emotion_acc']:.6f} | {result['per_emotion']['hf_raw'][emotion]['emotion_acc']:.6f} | {result['per_emotion']['hf_rules_v4'][emotion]['emotion_acc']:.6f} |"
        )
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append('- `param_mae` measures final expression parameter error after passing through the fixed backend.')
    lines.append('- `emotion_acc` measures whether the front-end predicted the correct emotion category on that emotion subset.')
    lines.append('- This analysis helps identify which fine-grained emotions benefit most from Chinese rule enhancement.')

    Path(OUTPUT_MD).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(OUTPUT_JSON)
    print(OUTPUT_MD)


if __name__ == '__main__':
    main()
