import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


HF_LABEL_MAPPING = {
    0: "calm",
    1: "concern",
    2: "happy",
    3: "angry",
    4: "sad",
    5: "question",
    6: "surprise",
    7: "disgust",
}

DEFAULT_LOCAL_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "Chinese-Emotion-Small")
)

BOOST_WORDS = ["非常", "特别", "太", "超级", "真的", "极其", "十分"]
LOW_WORDS = ["有点", "一点", "稍微", "还算", "略微"]
SOFTEN_WORDS = ["好像", "想", "还是", "先", "一下", "不自觉", "我不是", "只是", "感觉", "似乎"]

KEYWORD_RULES = {
    "happy": {
        "轻快": 2.0,
        "顺眼": 2.0,
        "笑": 1.6,
        "开心": 1.8,
        "高兴": 1.8,
        "暖洋洋": 2.0,
        "亮了一下": 2.0,
        "最爱的": 1.6,
        "喜欢": 1.4,
        "真好": 1.6,
    },
    "sad": {
        "睡不着": 1.8,
        "难过": 2.0,
        "失落": 2.0,
        "提不起精神": 2.0,
        "胸口闷": 2.0,
        "一个人待着": 1.8,
        "不想说话": 1.6,
        "过去这么久": 1.4,
        "委屈": 2.2,
        "高兴不起来": 3.0,
        "提不起兴趣": 1.2,
    },
    "angry": {
        "不爽": 2.2,
        "冒火": 2.3,
        "爆发": 2.2,
        "生气": 2.0,
        "犯躁": 2.4,
        "忍": 1.4,
        "无语": 1.8,
        "气": 0.6,
    },
    "surprise": {
        "居然": 2.2,
        "咦": 2.2,
        "天啊": 2.2,
        "没想到": 2.4,
        "突然": 1.8,
        "怎么会": 2.0,
        "太意外": 2.2,
        "真的假的": 2.3,
        "不一样": 1.8,
        "惊到": 2.4,
        "一下子": 1.6,
    },
    "disgust": {
        "倒胃口": 2.4,
        "不想听细节": 2.4,
        "不想再看第二眼": 2.4,
        "离这件事远点": 2.1,
        "恶心": 2.4,
        "反胃": 2.4,
        "不舒服": 1.4,
        "别说了": 1.8,
    },
    "concern": {
        "多想": 2.0,
        "没底": 2.0,
        "心跳得好快": 2.4,
        "再确认": 2.1,
        "确认一下": 2.0,
        "担心": 2.2,
        "希望没事": 2.2,
        "怎么还": 2.2,
        "不会吧": 1.8,
        "没回消息": 2.2,
    },
    "bored": {
        "无聊": 2.4,
        "没意思": 2.3,
        "好慢": 2.0,
        "真慢": 2.0,
        "犯困": 2.0,
        "发呆": 2.2,
        "没劲": 2.0,
        "到点下班": 2.2,
        "数了好几遍": 2.2,
        "走神": 2.2,
        "提不起兴趣": 2.4,
        "盯着时钟": 2.4,
    },
    "calm": {
        "平静": 2.2,
        "安静": 2.0,
        "放松": 2.0,
        "慢慢": 1.4,
        "休息": 1.8,
        "安心": 2.0,
        "舒缓": 2.0,
        "落回来了": 2.4,
        "忙完了": 1.6,
        "呼吸声": 1.6,
        "安安稳稳": 2.2,
        "不错": 0.8,
    },
}

BASE_INTENSITY = {
    "happy": 0.58,
    "sad": 0.52,
    "angry": 0.60,
    "surprise": 0.60,
    "disgust": 0.54,
    "concern": 0.48,
    "bored": 0.43,
    "calm": 0.36,
}

MAX_INTENSITY = {
    "happy": 0.82,
    "sad": 0.84,
    "angry": 0.82,
    "surprise": 0.86,
    "disgust": 0.78,
    "concern": 0.78,
    "bored": 0.78,
    "calm": 0.76,
}

RAW_BASE_INTENSITY = {
    "happy": 0.64,
    "sad": 0.55,
    "angry": 0.70,
    "surprise": 0.72,
    "disgust": 0.62,
    "concern": 0.56,
    "bored": 0.48,
    "calm": 0.42,
}


def _map_raw_label_to_emotion(raw_emotion: str) -> str:
    if raw_emotion == 'question':
        return 'concern'
    return raw_emotion


def estimate_intensity_raw(text, emotion, confidence):
    intensity = RAW_BASE_INTENSITY.get(emotion, 0.6)
    intensity += min(sum(text.count(word) for word in BOOST_WORDS), 3) * 0.04
    intensity -= min(sum(text.count(word) for word in LOW_WORDS), 2) * 0.05
    intensity += min(text.count('!') + text.count('！'), 3) * 0.03
    if emotion == 'surprise':
        intensity += min(text.count('?') + text.count('？'), 2) * 0.02
    return max(0.30, min(0.90, round(intensity, 2)))


def _keyword_scores(text):
    scores = {emotion: 0.0 for emotion in KEYWORD_RULES}
    for emotion, mapping in KEYWORD_RULES.items():
        for phrase, weight in mapping.items():
            if phrase in text:
                scores[emotion] += weight
    if '高兴不起来' in text:
        scores['happy'] -= 1.6
    if '不是生气' in text:
        scores['angry'] -= 1.4
    return scores


def _select_emotion(text, raw_emotion):
    scores = _keyword_scores(text)
    if raw_emotion == "question":
        scores["concern"] += 1.2
    elif raw_emotion in scores:
        scores[raw_emotion] += 0.8
    best_emotion, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score >= 1.6:
        if best_emotion == "angry" and scores.get("disgust", 0.0) >= best_score - 0.15 and "生气" not in text and "冒火" not in text and "不爽" not in text:
            return "disgust", scores
        return best_emotion, scores
    if raw_emotion == "question":
        return "concern", scores
    return raw_emotion, scores


def estimate_intensity_rules(text, emotion, confidence, keyword_score=0.0):
    intensity = BASE_INTENSITY.get(emotion, 0.5)
    intensity += max(0.0, float(confidence) - 0.45) * 0.12
    intensity += min(keyword_score, 3.2) * 0.03
    intensity += min(sum(text.count(word) for word in BOOST_WORDS), 2) * 0.025
    intensity -= min(sum(text.count(word) for word in LOW_WORDS), 2) * 0.06
    intensity -= min(sum(text.count(word) for word in SOFTEN_WORDS), 3) * 0.025
    intensity += min(text.count("!") + text.count("！"), 2) * 0.025
    if emotion in {"surprise", "concern"}:
        intensity += min(text.count("?") + text.count("？"), 2) * 0.015
    if emotion == "calm":
        intensity -= 0.03
    if emotion == "concern":
        intensity -= 0.02
    if emotion in {"angry", "disgust"} and "有点" in text:
        intensity -= 0.05
    if emotion == "surprise" and ("有点" in text or "一下" in text):
        intensity -= 0.05
    if emotion == 'surprise' and any(token in text for token in ['惊到', '一下子', '天啊']):
        intensity += 0.08
    if emotion == 'bored' and any(token in text for token in ['提不起兴趣', '盯着时钟', '发呆']):
        intensity += 0.10
    if emotion == 'calm' and any(token in text for token in ['安安稳稳', '不错']):
        intensity += 0.10
    if emotion == 'sad' and any(token in text for token in ['委屈', '高兴不起来']):
        intensity += 0.08
    if emotion == 'disgust' and '倒胃口' in text and '太' in text:
        intensity -= 0.04
    upper = MAX_INTENSITY.get(emotion, 0.82)
    return max(0.30, min(upper, round(intensity, 2)))


class HFEmotionFrontend:
    def __init__(self, model_dir=DEFAULT_LOCAL_MODEL_DIR, device=None, mode='rules_v4'):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
        label_id = int(torch.argmax(probs).item())
        raw_emotion = HF_LABEL_MAPPING.get(label_id, "calm")
        confidence = float(probs[label_id].item())

        if self.mode == 'raw':
            emotion = _map_raw_label_to_emotion(raw_emotion)
            intensity = estimate_intensity_raw(text, emotion, confidence)
            scores = {}
        else:
            emotion, scores = _select_emotion(text, raw_emotion)
            intensity = estimate_intensity_rules(text, emotion, confidence, scores.get(emotion, 0.0))

        return {
            "emotion": emotion,
            "intensity": intensity,
            "confidence": round(confidence, 4),
            "raw_label": raw_emotion,
            "label_id": label_id,
            "mode": self.mode,
            "keyword_scores": {k: round(v, 2) for k, v in scores.items() if abs(v) > 0.01},
        }
