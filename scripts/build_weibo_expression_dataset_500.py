import json
import re
from pathlib import Path
from statistics import mean

MINED_PATH = Path('data/weibo_emotion_mined_500.json')
OUTPUT_RAW = Path('data/weibo_expression_samples_500.json')
OUTPUT_PREVIEW = Path('docs/weibo_expression_samples_500_preview.md')
BASE_RAW = Path('data/full_samples_304.json')
MERGED_RAW = Path('data/full_samples_804.json')
TARGET_COUNTS = {'happy': 100, 'angry': 100, 'sad': 100, 'surprise': 100, 'calm': 100}

BLACKLIST = [
    '回复', '转发', '抽奖', '关注', '报名', '活动', '链接', '下载', '直播',
    't.sina', 'http', 'www.', '@', '【', '】', '草你', '操', 'cnm', 'tm',
    '我了个擦', '你丫的', '混蛋', '妈的', '死全家'
]

INTENSITY_HINTS = {
    'happy': {'strong': ['好开心', '幸福', '太棒', '真棒', '笑死我了', '超喜欢', '特别开心'], 'mid': ['开心', '高兴', '喜欢', '哈哈', '呵呵', '有爱', '可爱', '轻松', '满足'], 'soft': ['不错', '还行', '好玩', '舒服']},
    'angry': {'strong': ['愤怒', '气死', '爆发', '出离愤怒', '太过分', '真可恶'], 'mid': ['生气', '讨厌', '烦死', '受不了', '可恶', '恶心', '火大', '怒了'], 'soft': ['烦', '缺德']},
    'sad': {'strong': ['想哭', '伤心', '悲伤', '心酸', '真的好难受'], 'mid': ['难过', '难受', '郁闷', '低落', '失落', '心疼', '疲惫', '委屈'], 'soft': ['好累', '累', '有点难过']},
    'surprise': {'strong': ['震惊', '怎么可能', '真的假的', '我的天'], 'mid': ['惊讶', '居然', '竟然', '没想到', '想不到', '不会吧', '天啊'], 'soft': ['突然']},
    'calm': {'strong': ['宁静', '安然', '惬意'], 'mid': ['平静', '安静', '安稳', '放松', '安心', '淡定', '悠闲'], 'soft': ['轻松', '舒服', '静静']}
}


def stable_unit(text: str, salt: str) -> float:
    value = sum((i + 1) * ord(ch) for i, ch in enumerate(text + salt))
    return (value % 1000) / 1000.0


def has_bad_pattern(text: str) -> bool:
    lower = text.lower()
    return any(token.lower() in lower for token in BLACKLIST)


def is_clean_text(text: str) -> bool:
    if len(text) < 6 or len(text) > 42:
        return False
    if has_bad_pattern(text):
        return False
    if text.count('，') + text.count(',') > 2:
        return False
    if re.search(r'[A-Za-z]{6,}|\d{4,}', text):
        return False
    if text.count('的') > 4 and len(text) > 24:
        return False
    return True


def normalize(text: str) -> str:
    return re.sub(r'[。！？!?,，~…\.\s]+', '', text)


def score_text(text: str, emotion: str) -> float:
    score = 0.0
    n = len(text)
    if 8 <= n <= 26:
        score += 3.0
    elif 27 <= n <= 34:
        score += 2.0
    else:
        score += 0.5
    if '我' in text:
        score += 0.8
    if re.search(r'[。！？!~…]$', text):
        score += 0.5
    for tier, weight in [('strong', 2.0), ('mid', 1.2), ('soft', 0.6)]:
        score += sum(weight for token in INTENSITY_HINTS[emotion][tier] if token in text)
    if text.count('！！') >= 1 or text.count('!?') >= 1 or text.count('？！') >= 1:
        score += 0.4
    if text.count('...') >= 1 or text.count('…') >= 1:
        score += 0.1
    return round(score, 3)


def estimate_intensity(text: str, emotion: str) -> float:
    base = {'happy': 0.52, 'angry': 0.58, 'sad': 0.5, 'surprise': 0.6, 'calm': 0.42}[emotion]
    bonus = 0.0
    for token in INTENSITY_HINTS[emotion]['strong']:
        if token in text:
            bonus += 0.12
    for token in INTENSITY_HINTS[emotion]['mid']:
        if token in text:
            bonus += 0.06
    for token in INTENSITY_HINTS[emotion]['soft']:
        if token in text:
            bonus += 0.03
    bonus += min(text.count('!') + text.count('！'), 3) * 0.03
    bonus += min(text.count('~'), 3) * 0.015
    bonus += 0.02 if '有点' in text else 0.0
    bonus += stable_unit(text, 'intensity') * 0.03 - 0.015
    intensity = max(0.28, min(0.9, base + bonus))
    return round(intensity, 2)


def build_params(emotion: str, intensity: float, text: str) -> dict:
    u1 = stable_unit(text, 'a')
    u2 = stable_unit(text, 'b')
    u3 = stable_unit(text, 'c')

    def asym(value: float, spread: float):
        left = max(0.0, min(1.0, value - spread * (u1 - 0.5)))
        right = max(0.0, min(1.0, value + spread * (u2 - 0.5)))
        return round(left, 2), round(right, 2)

    if emotion == 'happy':
        smile = 0.52 * intensity + 0.06
        squint = 0.28 * intensity
        brow = 0.12 * intensity
        left_smile, right_smile = asym(smile, 0.08)
        left_sq, right_sq = asym(squint, 0.05)
        return {'mouthSmileLeft': left_smile, 'mouthSmileRight': right_smile, 'eyeSquintLeft': left_sq, 'eyeSquintRight': right_sq, 'browInnerUp': round(max(0.0, min(1.0, brow + (u3 - 0.5) * 0.03)), 2)}
    if emotion == 'angry':
        brow = 0.68 * intensity + 0.04
        press = 0.5 * intensity + 0.03
        cheek = 0.22 * intensity
        left_brow, right_brow = asym(brow, 0.07)
        left_press, right_press = asym(press, 0.06)
        return {'browDownLeft': left_brow, 'browDownRight': right_brow, 'mouthPressLeft': left_press, 'mouthPressRight': right_press, 'cheekPuff': round(max(0.0, min(1.0, cheek + (u3 - 0.5) * 0.04)), 2)}
    if emotion == 'sad':
        frown = 0.58 * intensity + 0.03
        brow = 0.5 * intensity + 0.05
        squint = 0.24 * intensity
        left_frown, right_frown = asym(frown, 0.06)
        left_sq, right_sq = asym(squint, 0.04)
        return {'mouthFrownLeft': left_frown, 'mouthFrownRight': right_frown, 'browInnerUp': round(max(0.0, min(1.0, brow + (u3 - 0.5) * 0.04)), 2), 'eyeSquintLeft': left_sq, 'eyeSquintRight': right_sq}
    if emotion == 'surprise':
        brow = 0.62 * intensity + 0.06
        wide = 0.72 * intensity + 0.04
        jaw = 0.42 * intensity + 0.02
        left_brow, right_brow = asym(brow, 0.06)
        left_wide, right_wide = asym(wide, 0.05)
        return {'browOuterUpLeft': left_brow, 'browOuterUpRight': right_brow, 'eyeWideLeft': left_wide, 'eyeWideRight': right_wide, 'jawOpen': round(max(0.0, min(1.0, jaw + (u3 - 0.5) * 0.04)), 2)}
    smile = 0.08 * intensity + 0.02
    left_smile, right_smile = asym(smile, 0.03)
    return {'mouthSmileLeft': left_smile, 'mouthSmileRight': right_smile, 'browInnerUp': round(max(0.0, min(1.0, 0.04 * intensity + (u3 - 0.5) * 0.02)), 2)}


with MINED_PATH.open('r', encoding='utf-8') as f:
    mined = json.load(f)

buckets = {emotion: [] for emotion in TARGET_COUNTS}
seen = set()
for item in mined:
    emotion = item['suggested_emotion']
    text = item['text'].strip()
    if emotion not in buckets:
        continue
    rank = score_text(text, emotion)
    buckets[emotion].append((rank, text, item))

selected = []
selected_keys = set()
for emotion, need in TARGET_COUNTS.items():
    ranked = sorted(buckets[emotion], key=lambda x: (-x[0], len(x[1]), x[1]))
    picked = []
    for rank, text, item in ranked:
        key = normalize(text)
        if key in selected_keys:
            continue
        if len(picked) < need and is_clean_text(text):
            picked.append((rank, text, item))
            selected_keys.add(key)
    if len(picked) < need:
        for rank, text, item in ranked:
            key = normalize(text)
            if key in selected_keys:
                continue
            picked.append((rank, text, item))
            selected_keys.add(key)
            if len(picked) >= need:
                break
    for rank, text, item in picked[:need]:
        intensity = estimate_intensity(text, emotion)
        selected.append({'text': text, 'emotion': emotion, 'intensity': intensity, 'params': build_params(emotion, intensity, text), 'source': 'weibo_mined_large', 'source_label': item['source_label']})

OUTPUT_RAW.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
with BASE_RAW.open('r', encoding='utf-8') as f:
    base = json.load(f)
merged = base + selected
MERGED_RAW.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')

lines = ['# 微博补充样本 500 条', '', f'- total: {len(selected)}', f'- mean intensity: {mean(item["intensity"] for item in selected):.3f}', '']
for emotion in ['happy', 'angry', 'sad', 'surprise', 'calm']:
    items = [item for item in selected if item['emotion'] == emotion]
    lines.append(f'## {emotion} ({len(items)})')
    lines.append('')
    for item in items[:20]:
        lines.append(f"- [{item['intensity']}] {item['text']}")
    lines.append('')
OUTPUT_PREVIEW.write_text('\n'.join(lines), encoding='utf-8')
print('saved raw', len(selected), 'to', OUTPUT_RAW)
print('saved merged', len(merged), 'to', MERGED_RAW)
for emotion in ['happy', 'angry', 'sad', 'surprise', 'calm']:
    print(emotion, sum(1 for item in selected if item['emotion'] == emotion))
