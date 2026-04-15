import json
import os
import random
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
from utils.param_utils import params_dict_to_vector, vector_to_params_dict
from utils.text_encoder import load_text_encoder
from utils.text_features import extract_semantic_features

DEFAULT_DATA_PATH = 'data/full_samples_804.json'
DEFAULT_EXPR_CKPT = 'outputs/prior_fusion_804_source_holdout_strict_mapper.pt'
DEFAULT_LEARNED_CKPT = 'outputs/text_only_emotion8_base_classifier.pt'
DEFAULT_OUTPUT_JSON = 'docs/case_analysis_text_only_frontends.json'
DEFAULT_OUTPUT_MD = 'docs/case_analysis_text_only_frontends.md'
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


def predict_learned(text, checkpoint_path):
    classifier, ckpt = load_classifier(checkpoint_path)
    encoder = load_text_encoder(ckpt.get('encoder_name', 'paraphrase-multilingual-MiniLM-L12-v2'))
    features = torch.tensor(
        list(encoder.encode(text, convert_to_numpy=True)) + (extract_semantic_features(text) if ckpt.get('use_semantic_features', True) else []),
        dtype=torch.float32,
    ).unsqueeze(0)
    with torch.no_grad():
        emotion_logits = classifier(features)
        probs = F.softmax(emotion_logits, dim=-1)[0]
    pred_emotion = ckpt['emotion_list'][int(torch.argmax(probs).item())]
    pred_intensity = estimate_intensity_from_text(text, pred_emotion)
    return {
        'emotion': pred_emotion,
        'intensity': pred_intensity,
        'confidence': round(float(probs.max().item()), 4),
        'raw_frontend_label': pred_emotion,
    }


def pick_cases(test_data, per_emotion=2, seed=42):
    grouped = defaultdict(list)
    for item in test_data:
        grouped[item['emotion']].append(item)
    rng = random.Random(seed)
    cases = []
    for emotion in EMOTION_ORDER:
        items = list(grouped.get(emotion, []))
        items.sort(key=lambda x: (float(x['intensity']), x['text']))
        if len(items) <= per_emotion:
            cases.extend(items)
            continue
        # one lower-intensity and one higher-intensity sample when possible
        chosen = [items[0], items[-1]] if per_emotion == 2 else []
        if per_emotion > 2:
            middle = items[1:-1]
            rng.shuffle(middle)
            chosen.extend(middle[:per_emotion - 2])
        # dedup while keeping order
        seen = set()
        uniq = []
        for item in chosen:
            key = item['text']
            if key not in seen:
                uniq.append(item)
                seen.add(key)
        cases.extend(uniq[:per_emotion])
    return cases


def top_param_diffs(pred_vector, target_vector, k=5):
    diffs = []
    pred_dict = vector_to_params_dict(pred_vector.tolist())
    target_dict = vector_to_params_dict(target_vector.tolist())
    for name in pred_dict:
        diffs.append((name, abs(pred_dict[name] - target_dict[name]), pred_dict[name], target_dict[name]))
    diffs.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            'name': name,
            'abs_diff': round(diff, 4),
            'pred': round(pred, 4),
            'target': round(target, 4),
        }
        for name, diff, pred, target in diffs[:k]
    ]


def main():
    with open(DEFAULT_DATA_PATH, 'r', encoding='utf-8') as f:
        data = [normalize_item(item) for item in json.load(f)]
    _, _, test_data = split_data_source_holdout(data, seed=42)
    test_data = [item for item in test_data if item.get('source', 'manual') == 'manual']
    cases = pick_cases(test_data, per_emotion=2, seed=42)

    expr_model, expr_ckpt = load_expression_model(DEFAULT_EXPR_CKPT)
    expr_encoder = load_text_encoder(expr_ckpt.get('encoder_name', 'paraphrase-multilingual-MiniLM-L12-v2'))
    hf_raw = HFEmotionFrontend(mode='raw')
    hf_rules = HFEmotionFrontend(mode='rules_v4')

    records = []
    summary = {
        'learned': {'mae_sum': 0.0, 'count': 0},
        'hf_raw': {'mae_sum': 0.0, 'count': 0},
        'hf_rules_v4': {'mae_sum': 0.0, 'count': 0},
    }

    for item in cases:
        target_vector = torch.tensor(item['param_vector'], dtype=torch.float32)
        variants = {
            'learned': predict_learned(item['text'], DEFAULT_LEARNED_CKPT),
            'hf_raw': hf_raw.predict(item['text']),
            'hf_rules_v4': hf_rules.predict(item['text']),
        }
        result_variants = {}
        for key, pred in variants.items():
            features = build_expression_features(item['text'], pred['emotion'], pred['intensity'], expr_encoder, expr_ckpt)
            pred_vector = run_expression(expr_model, expr_ckpt, features)
            metrics = compute_metrics(pred_vector.unsqueeze(0), target_vector.unsqueeze(0))
            summary[key]['mae_sum'] += metrics['mae']
            summary[key]['count'] += 1
            result_variants[key] = {
                'pred_emotion': pred['emotion'],
                'pred_intensity': pred['intensity'],
                'confidence': pred['confidence'],
                'raw_frontend_label': pred.get('raw_frontend_label', pred.get('raw_label', '')), 
                'param_mae': round(metrics['mae'], 6),
                'top_param_diffs': top_param_diffs(pred_vector, target_vector),
            }
        records.append({
            'text': item['text'],
            'true_emotion': item['emotion'],
            'true_intensity': float(item['intensity']),
            'variants': result_variants,
        })

    payload = {
        'config': {
            'data_path': DEFAULT_DATA_PATH,
            'expression_checkpoint': DEFAULT_EXPR_CKPT,
            'learned_checkpoint': DEFAULT_LEARNED_CKPT,
            'case_count': len(records),
            'selection': 'manual-only source_holdout test set, 2 cases per emotion',
        },
        'summary': {
            key: {'mean_case_param_mae': round(value['mae_sum'] / max(value['count'], 1), 6), 'count': value['count']}
            for key, value in summary.items()
        },
        'records': records,
    }

    Path(DEFAULT_OUTPUT_JSON).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = []
    lines.append('# Case Analysis: Text-Only Front-Ends')
    lines.append('')
    lines.append('- setting: manual-only source-holdout test set')
    lines.append(f"- cases: {len(records)}")
    lines.append('')
    lines.append('## Mean Case Param MAE')
    lines.append('')
    lines.append('| Front-End | Mean Case Param MAE |')
    lines.append('| --- | ---: |')
    for key in ['learned', 'hf_raw', 'hf_rules_v4']:
        lines.append(f"| {key} | {payload['summary'][key]['mean_case_param_mae']:.6f} |")
    lines.append('')
    for idx, record in enumerate(records, start=1):
        lines.append(f"## Case {idx}: {record['true_emotion']}")
        lines.append('')
        lines.append(f"- text: {record['text']}")
        lines.append(f"- true_emotion: {record['true_emotion']}")
        lines.append(f"- true_intensity: {record['true_intensity']:.2f}")
        lines.append('')
        lines.append('| Variant | Pred Emotion | Pred Intensity | Confidence | Param MAE | Raw Label |')
        lines.append('| --- | --- | ---: | ---: | ---: | --- |')
        for key in ['learned', 'hf_raw', 'hf_rules_v4']:
            v = record['variants'][key]
            lines.append(f"| {key} | {v['pred_emotion']} | {v['pred_intensity']:.2f} | {v['confidence']:.4f} | {v['param_mae']:.6f} | {v['raw_frontend_label']} |")
        lines.append('')
        lines.append('Top parameter differences (`hf_rules_v4`):')
        for diff in record['variants']['hf_rules_v4']['top_param_diffs']:
            lines.append(f"- {diff['name']}: pred={diff['pred']}, target={diff['target']}, abs_diff={diff['abs_diff']}")
        lines.append('')

    Path(DEFAULT_OUTPUT_MD).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(DEFAULT_OUTPUT_JSON)
    print(DEFAULT_OUTPUT_MD)


if __name__ == '__main__':
    main()
