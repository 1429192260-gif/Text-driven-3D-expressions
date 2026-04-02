import json
import re
from pathlib import Path

INPUT_PATH = Path('data/weibo_candidate_240.json')
OUTPUT_PATH = Path('data/weibo_candidate_refined.json')
PREVIEW_PATH = Path('docs/weibo_candidate_refined_preview.md')
TARGET_PER_LABEL = {'0': 50, '1': 50, '2': 50}

with INPUT_PATH.open('r', encoding='utf-8') as f:
    data = json.load(f)

bad_substrings = [
    '抽奖','关注','转发','链接','活动','报名','投票','群号','下载','直播','t.sina',
    '你叼','傻x','操','妈的','死全家','cnm','tm','靠靠靠','滚','首都就是好',
    '...？','?？','？？？','。。。','......','我的...我的','要我的...','哈哈...我承认我笑了',
]

positive_cues = ['开心','高兴','喜欢','哈哈','呵呵','好吃','太棒','有爱','美好','幸福','轻松','可爱','好玩','满足','舒服']
negative_anger_cues = ['气死','生气','愤怒','火大','烦死','恶心','讨厌','受不了','太过分','无耻','混蛋','崩溃','怒']
negative_sad_cues = ['难过','伤心','心酸','失落','难受','无语','郁闷','想哭','悲伤','低落','疲惫','累','孤单','心疼']


def clean_score(text, label):
    score = 0.0
    n = len(text)
    if 8 <= n <= 28:
        score += 3.0
    elif 29 <= n <= 40:
        score += 2.0
    elif 41 <= n <= 52:
        score += 0.5
    else:
        score -= 2.0
    if re.search(r'[。！？~…]$', text):
        score += 0.8
    if '我' in text:
        score += 0.8
    if label == '0' and any(k in text for k in positive_cues):
        score += 1.5
    if label == '1' and any(k in text for k in negative_anger_cues):
        score += 1.5
    if label == '2' and any(k in text for k in negative_sad_cues):
        score += 1.5
    if re.search(r'[A-Za-z]{4,}|\d{3,}', text):
        score -= 1.0
    if text.count('，') + text.count(',') > 2:
        score -= 0.6
    return score


def is_good(text, label):
    if len(text) < 6 or len(text) > 52:
        return False
    if any(s.lower() in text.lower() for s in bad_substrings):
        return False
    if 'http' in text.lower() or '@' in text or '【' in text or '】' in text:
        return False
    if text.count('的') > 5 and len(text) > 28:
        return False
    if re.search(r'[A-Za-z]{8,}', text):
        return False
    if label == '0' and not any(k in text for k in positive_cues):
        return False
    if label == '1' and not any(k in text for k in negative_anger_cues):
        return False
    if label == '2' and not any(k in text for k in negative_sad_cues):
        return False
    return True

buckets = {'0': [], '1': [], '2': []}
seen = set()
for item in data:
    label = item['source_label']
    text = item['text'].strip()
    key = re.sub(r'[。！？!?,，~…\.\s]+', '', text)
    if key in seen:
        continue
    seen.add(key)
    if not is_good(text, label):
        continue
    item['refined_score'] = round(clean_score(text, label), 3)
    buckets[label].append(item)

selected = []
for label, need in TARGET_PER_LABEL.items():
    ranked = sorted(buckets[label], key=lambda x: (-x['refined_score'], len(x['text'])))
    chosen = ranked[:need]
    selected.extend(chosen)
    buckets[label] = chosen

OUTPUT_PATH.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')

lines = ['# 精筛微博候选集', '', '| label | count |', '| --- | ---: |']
for label in ['0','1','2']:
    lines.append(f'| {label} | {len(buckets[label])} |')
    lines.append('')
    lines.append(f'## label {label}')
    lines.append('')
    for item in buckets[label][:25]:
        lines.append(f"- {item['text']}")
    lines.append('')
PREVIEW_PATH.write_text('\n'.join(lines), encoding='utf-8')

print('saved', len(selected), 'to', OUTPUT_PATH)
for label in ['0','1','2']:
    print(label, len(buckets[label]))
