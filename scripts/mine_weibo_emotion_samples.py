import json
import re
from pathlib import Path

INPUT_PATH = Path('data/weibo_cleaned.json')
OUTPUT_PATH = Path('data/weibo_emotion_mined_150.json')
PREVIEW_PATH = Path('docs/weibo_emotion_mined_150_preview.md')
TARGETS = {'happy': 50, 'angry': 50, 'sad': 50}
LEXICON = {
    'happy': ['开心','高兴','喜欢','哈哈','呵呵','幸福','轻松','有爱','美好','可爱','满足','舒服','真棒','太棒','好开心','笑死'],
    'angry': ['生气','气死','愤怒','火大','烦死','恶心','讨厌','受不了','太过分','无耻','混蛋','怒','崩溃','缺德','可恶'],
    'sad': ['难过','伤心','心酸','难受','无语','郁闷','想哭','悲伤','低落','疲惫','好累','累死','心疼','失落','孤单']
}
BAD = ['抽奖','关注','转发','活动','报名','投票','群号','下载','直播','t.sina','http','www.','@','【','】']

with INPUT_PATH.open('r', encoding='utf-8') as f:
    data = json.load(f)


def valid(text):
    if len(text) < 6 or len(text) > 48:
        return False
    if any(x.lower() in text.lower() for x in BAD):
        return False
    if re.search(r'[A-Za-z]{6,}|\d{4,}', text):
        return False
    if text.count('的') > 5 and len(text) > 24:
        return False
    return True


def score(text, emotion):
    s = 0.0
    n = len(text)
    if 8 <= n <= 24:
        s += 3
    elif 25 <= n <= 36:
        s += 2
    else:
        s += 0.5
    if '我' in text:
        s += 1
    if re.search(r'[。！？~…]$', text):
        s += 0.8
    hits = sum(1 for k in LEXICON[emotion] if k in text)
    s += hits * 1.4
    if text.count('...') >= 1 or text.count('…') >= 1:
        s += 0.2
    if text.count('，') + text.count(',') > 2:
        s -= 0.5
    return round(s, 3)


def normalize(text):
    return re.sub(r'[。！？!?,，~…\.\s]+', '', text)

selected = {k: [] for k in TARGETS}
seen = set()

for item in data:
    text = item['text'].strip()
    if not valid(text):
        continue
    key = normalize(text)
    if key in seen:
        continue
    matched = [emo for emo, words in LEXICON.items() if any(w in text for w in words)]
    if len(matched) != 1:
        continue
    emotion = matched[0]
    seen.add(key)
    selected[emotion].append({
        'text': text,
        'suggested_emotion': emotion,
        'source_label': item['label_id'],
        'score': score(text, emotion),
        'source_file': item['source_file'],
        'line_no': item['line_no'],
    })

final = []
for emotion, need in TARGETS.items():
    ranked = sorted(selected[emotion], key=lambda x: (-x['score'], len(x['text'])))
    picked = ranked[:need]
    selected[emotion] = picked
    final.extend(picked)

OUTPUT_PATH.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# 按情绪词挖掘的微博候选集', '']
for emotion in ['happy','angry','sad']:
    lines.append(f'## {emotion} ({len(selected[emotion])})')
    lines.append('')
    for item in selected[emotion][:25]:
        lines.append(f"- {item['text']}")
    lines.append('')
PREVIEW_PATH.write_text('\n'.join(lines), encoding='utf-8')
print('saved', len(final), 'to', OUTPUT_PATH)
for emotion in ['happy','angry','sad']:
    print(emotion, len(selected[emotion]))
