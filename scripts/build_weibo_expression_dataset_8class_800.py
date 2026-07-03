import json
import re
from pathlib import Path
from statistics import mean

MINED_PATH = Path('data/weibo_emotion_mined_8class_800.json')
OUTPUT_RAW = Path('data/weibo_expression_samples_8class_800.json')
OUTPUT_PREVIEW = Path('docs/weibo_expression_samples_8class_800_preview.md')

TARGET_COUNTS = {
    'happy': 100,
    'angry': 100,
    'sad': 100,
    'surprise': 100,
    'calm': 100,
    'disgust': 100,
    'concern': 100,
    'bored': 100,
}

INTENSITY_HINTS = {
    'happy': {'strong': ['好开心', '幸福', '太棒', '真棒', '笑死我了', '超喜欢', '特别开心'], 'mid': ['开心', '高兴', '喜欢', '哈哈', '呵呵', '有爱', '可爱', '轻松', '满足'], 'soft': ['不错', '还行', '好玩', '舒服']},
    'angry': {'strong': ['愤怒', '气死', '爆发', '出离愤怒', '太过分', '真可恶'], 'mid': ['生气', '讨厌', '烦死', '受不了', '可恶', '火大', '怒了', '不爽', '冒火'], 'soft': ['烦', '缺德']},
    'sad': {'strong': ['想哭', '伤心', '悲伤', '心酸', '真的好难受', '高兴不起来'], 'mid': ['难过', '难受', '郁闷', '低落', '失落', '心疼', '疲惫', '委屈'], 'soft': ['好累', '累', '有点难过']},
    'surprise': {'strong': ['震惊', '怎么可能', '真的假的', '我的天', '吓死'], 'mid': ['惊讶', '居然', '竟然', '没想到', '想不到', '不会吧', '天啊'], 'soft': ['突然']},
    'calm': {'strong': ['宁静', '安然', '惬意'], 'mid': ['平静', '安静', '安稳', '放松', '安心', '淡定', '悠闲'], 'soft': ['轻松', '舒服', '静静']},
    'disgust': {'strong': ['恶心', '倒胃口', '反胃'], 'mid': ['嫌弃', '膈应', '不舒服', '不想再看', '不想听细节'], 'soft': ['别说了', '第二眼']},
    'concern': {'strong': ['心慌', '放心不下', '希望没事'], 'mid': ['担心', '害怕', '没底', '不放心', '怎么还', '没回消息'], 'soft': ['多想', '再确认']},
    'bored': {'strong': ['无聊死了', '提不起兴趣', '盯着时钟'], 'mid': ['无聊', '没意思', '发呆', '走神', '好慢', '真慢', '犯困'], 'soft': ['没劲', '数了好几遍']},
}


def stable_unit(text: str, salt: str) -> float:
    value = sum((i + 1) * ord(ch) for i, ch in enumerate(text + salt))
    return (value % 1000) / 1000.0


def estimate_intensity(text: str, emotion: str) -> float:
    base = {'happy': 0.52, 'angry': 0.56, 'sad': 0.5, 'surprise': 0.58, 'calm': 0.42, 'disgust': 0.54, 'concern': 0.48, 'bored': 0.44}[emotion]
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
    bonus -= 0.04 if '有点' in text else 0.0
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
    if emotion == 'disgust':
        brow = 0.44 * intensity + 0.03
        squint = 0.50 * intensity + 0.02
        frown = 0.40 * intensity + 0.02
        left_brow, right_brow = asym(brow, 0.05)
        left_sq, right_sq = asym(squint, 0.05)
        left_frown, right_frown = asym(frown, 0.05)
        return {'browDownLeft': left_brow, 'browDownRight': right_brow, 'eyeSquintLeft': left_sq, 'eyeSquintRight': right_sq, 'mouthFrownLeft': left_frown, 'mouthFrownRight': right_frown, 'cheekPuff': round(max(0.0, min(1.0, 0.16 * intensity + (u3 - 0.5) * 0.03)), 2)}
    if emotion == 'concern':
        brow = 0.46 * intensity + 0.04
        wide = 0.30 * intensity + 0.02
        press = 0.24 * intensity + 0.02
        left_wide, right_wide = asym(wide, 0.04)
        left_press, right_press = asym(press, 0.04)
        return {'browInnerUp': round(max(0.0, min(1.0, brow + (u3 - 0.5) * 0.03)), 2), 'eyeWideLeft': left_wide, 'eyeWideRight': right_wide, 'mouthPressLeft': left_press, 'mouthPressRight': right_press}
    if emotion == 'bored':
        look_down = 0.34 * intensity + 0.04
        look_left = 0.22 * intensity + 0.02
        frown = 0.16 * intensity + 0.01
        left_frown, right_frown = asym(frown, 0.03)
        left_brow, right_brow = asym(0.22 * intensity + 0.02, 0.03)
        return {'eyeLookDown': round(max(0.0, min(1.0, look_down + (u3 - 0.5) * 0.03)), 2), 'eyeLookLeft': round(max(0.0, min(1.0, look_left + (u2 - 0.5) * 0.03)), 2), 'mouthFrownLeft': left_frown, 'mouthFrownRight': right_frown, 'browDownLeft': left_brow, 'browDownRight': right_brow}
    smile = 0.08 * intensity + 0.02
    left_smile, right_smile = asym(smile, 0.03)
    return {'mouthSmileLeft': left_smile, 'mouthSmileRight': right_smile, 'browInnerUp': round(max(0.0, min(1.0, 0.04 * intensity + (u3 - 0.5) * 0.02)), 2)}


with MINED_PATH.open('r', encoding='utf-8') as f:
    mined = json.load(f)

selected = []
for item in mined:
    text = item['text'].strip()
    emotion = item['suggested_emotion']
    if emotion not in TARGET_COUNTS:
        continue
    intensity = estimate_intensity(text, emotion)
    selected.append({'text': text, 'emotion': emotion, 'intensity': intensity, 'params': build_params(emotion, intensity, text), 'source': 'weibo_mined_8class', 'source_label': item['source_label']})

OUTPUT_RAW.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# 微博 8 类表达样本 800 条', '', f'- total: {len(selected)}', f'- mean intensity: {mean(item["intensity"] for item in selected):.3f}', '']
for emotion in ['happy', 'angry', 'sad', 'surprise', 'calm', 'disgust', 'concern', 'bored']:
    items = [item for item in selected if item['emotion'] == emotion]
    lines.append(f'## {emotion} ({len(items)})')
    lines.append('')
    for item in items[:20]:
        lines.append(f"- [{item['intensity']}] {item['text']}")
    lines.append('')
OUTPUT_PREVIEW.write_text('\n'.join(lines), encoding='utf-8')
print('saved raw', len(selected), 'to', OUTPUT_RAW)
for emotion in ['happy', 'angry', 'sad', 'surprise', 'calm', 'disgust', 'concern', 'bored']:
    print(emotion, sum(1 for item in selected if item['emotion'] == emotion))
