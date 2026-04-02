import argparse
import json
import re
from pathlib import Path

INPUT_PATH = Path('data/weibo_cleaned.json')
OUTPUT_PATH = Path('data/weibo_candidate_240.json')
PREVIEW_PATH = Path('docs/weibo_candidate_240_preview.md')
TARGET_PER_LABEL = {'0': 80, '1': 80, '2': 80}
EMOTION_MAP = {'0': 'happy', '1': 'angry_or_disgust', '2': 'sad'}
BAD_PATTERNS = [
    r'http[s]?://',
    r'www\.',
    r'@',
    r'转发',
    r'回复',
    r'关注',
    r'抽奖',
    r'奖品',
    r'优惠',
    r'活动',
    r'点击',
    r'链接',
    r'群号',
    r'报名',
    r'投票',
    r'分享',
    r'来自:',
    r'来自：',
    r'直播',
    r'下载',
    r't\.sina',
]


def score_text(text: str) -> float:
    score = 0.0
    length = len(text)
    if 8 <= length <= 38:
        score += 3.0
    elif 39 <= length <= 55:
        score += 1.5
    else:
        score -= 2.0

    if re.search(r'[。！？…~]', text):
        score += 0.8
    if re.search(r'我|今天|真的|有点|怎么|为什么|感觉|太|好|难受|无语|气死|开心|喜欢|难过', text):
        score += 1.2
    if re.search(r'哈哈|呵呵|呜|唉|哎|啊|呀', text):
        score += 0.6
    if re.search(r'[A-Za-z]{5,}|\d{4,}', text):
        score -= 1.0
    if text.count('...') or text.count('…') >= 2:
        score += 0.3
    return score


def is_valid(text: str) -> bool:
    text = text.strip()
    if len(text) < 6 or len(text) > 60:
        return False
    if any(re.search(p, text, flags=re.IGNORECASE) for p in BAD_PATTERNS):
        return False
    if text.count('【') >= 1 or text.count('】') >= 1:
        return False
    if re.search(r'^[\W_]+$', text):
        return False
    if text.count('的') > 8 and len(text) > 35:
        return False
    return True


def dedupe_key(text: str) -> str:
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[。！？!?,，~…\.]+', '', text)
    return text.lower()


with INPUT_PATH.open('r', encoding='utf-8') as f:
    records = json.load(f)

buckets = {k: [] for k in TARGET_PER_LABEL}
seen = set()

for item in records:
    label = item['label_id']
    if label not in TARGET_PER_LABEL:
        continue
    text = item['text'].strip()
    if not is_valid(text):
        continue
    key = dedupe_key(text)
    if key in seen:
        continue
    seen.add(key)
    candidate = {
        'text': text,
        'source_label': label,
        'suggested_emotion': EMOTION_MAP[label],
        'score': round(score_text(text), 3),
        'source_file': item['source_file'],
        'line_no': item['line_no'],
    }
    buckets[label].append(candidate)

selected = []
for label, need in TARGET_PER_LABEL.items():
    ranked = sorted(buckets[label], key=lambda x: (-x['score'], len(x['text'])))
    chosen = ranked[:need]
    selected.extend(chosen)
    buckets[label] = chosen

OUTPUT_PATH.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')

lines = [
    '# 微博候选样本 240 条',
    '',
    '该文件由脚本自动筛选，供后续人工映射到表情参数。',
    '',
    '| label | suggested_emotion | count |',
    '| --- | --- | ---: |',
]
for label in ['0', '1', '2']:
    lines.append(f"| {label} | {EMOTION_MAP[label]} | {len(buckets[label])} |")
    lines.append('')
    lines.append(f'## label {label}')
    lines.append('')
    for item in buckets[label][:20]:
        lines.append(f"- {item['text']}")
    lines.append('')

PREVIEW_PATH.write_text('\n'.join(lines), encoding='utf-8')
print(f'Saved {len(selected)} candidates to {OUTPUT_PATH}')
for label in ['0', '1', '2']:
    print(label, len(buckets[label]))
