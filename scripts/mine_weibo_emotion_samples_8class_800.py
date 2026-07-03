import json
import re
from pathlib import Path

INPUT_PATH = Path('data/weibo_cleaned.json')
OUTPUT_PATH = Path('data/weibo_emotion_mined_8class_800.json')
PREVIEW_PATH = Path('docs/weibo_emotion_mined_8class_800_preview.md')
TARGETS = {
    'happy': 100,
    'angry': 100,
    'sad': 100,
    'surprise': 100,
    'calm': 100,
    'disgust': 100,
    'concern': 100,
    'bored': 100,
}
LEXICON = {
    'happy': ['开心', '高兴', '喜欢', '哈哈', '呵呵', '幸福', '轻松', '有爱', '美好', '可爱', '满足', '舒服', '真棒', '太棒', '好开心', '笑死'],
    'angry': ['生气', '气死', '愤怒', '火大', '烦死', '讨厌', '受不了', '太过分', '无耻', '崩溃', '缺德', '可恶', '怒了', '不爽', '冒火'],
    'sad': ['难过', '伤心', '心酸', '难受', '郁闷', '想哭', '悲伤', '低落', '好累', '累死', '心疼', '失落', '孤单', '委屈', '高兴不起来'],
    'surprise': ['惊讶', '震惊', '居然', '竟然', '没想到', '想不到', '突然', '我的天', '天啊', '真的假的', '怎么可能', '吓死', '不会吧'],
    'calm': ['平静', '安静', '安稳', '轻松', '放松', '舒服', '安心', '淡定', '悠闲', '宁静', '安然', '静静', '惬意'],
    'disgust': ['恶心', '倒胃口', '反胃', '嫌弃', '膈应', '不想再看', '不想听细节', '不舒服', '别说了', '看第二眼'],
    'concern': ['担心', '心慌', '害怕', '没底', '不放心', '希望没事', '怎么还', '多想', '再确认', '没回消息', '放心不下'],
    'bored': ['无聊', '没意思', '发呆', '走神', '好慢', '真慢', '犯困', '没劲', '提不起兴趣', '盯着时钟', '数了好几遍'],
}
BAD = ['抽奖', '关注', '转发', '活动', '报名', '投票', '群号', '下载', '直播', 't.sina', 'http', 'www.', '@', '【', '】', '回复']

with INPUT_PATH.open('r', encoding='utf-8') as f:
    data = json.load(f)


def valid(text):
    if len(text) < 6 or len(text) > 42:
        return False
    lower = text.lower()
    if any(x.lower() in lower for x in BAD):
        return False
    if re.search(r'[A-Za-z]{6,}|\d{4,}', text):
        return False
    if text.count('的') > 5 and len(text) > 24:
        return False
    if text.count('，') + text.count(',') > 2:
        return False
    return True


def score(text, emotion):
    s = 0.0
    n = len(text)
    if 8 <= n <= 24:
        s += 3.2
    elif 25 <= n <= 34:
        s += 2.0
    else:
        s += 0.6
    if '我' in text:
        s += 0.7
    if re.search(r'[。！？~…]$', text):
        s += 0.7
    hits = sum(1 for k in LEXICON[emotion] if k in text)
    s += hits * 1.5
    if text.count('...') >= 1 or text.count('…') >= 1:
        s += 0.2
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
    ranked = sorted(selected[emotion], key=lambda x: (-x['score'], len(x['text']), x['text']))
    picked = ranked[:need]
    selected[emotion] = picked
    final.extend(picked)

OUTPUT_PATH.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# 按情绪词挖掘的微博候选集 8 类 800 条', '']
for emotion in ['happy', 'angry', 'sad', 'surprise', 'calm', 'disgust', 'concern', 'bored']:
    lines.append(f'## {emotion} ({len(selected[emotion])})')
    lines.append('')
    for item in selected[emotion][:30]:
        lines.append(f"- {item['text']}")
    lines.append('')
PREVIEW_PATH.write_text('\n'.join(lines), encoding='utf-8')
print('saved', len(final), 'to', OUTPUT_PATH)
for emotion in ['happy', 'angry', 'sad', 'surprise', 'calm', 'disgust', 'concern', 'bored']:
    print(emotion, len(selected[emotion]))
