import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


EMOTION_KEYWORDS = {
    "happy": ["great", "beautiful", "good", "love", "happy", "glad", "nice", "wonderful", "smile"],
    "sad": ["dead", "sorry", "sad", "cry", "hurt", "alone", "miss", "bad", "wrong"],
    "angry": ["hell", "damn", "throw", "hate", "angry", "mad", "stupid", "shut", "ridiculous"],
    "surprise": ["what", "wow", "really", "surprise", "unbelievable", "amazing", "suddenly"],
    "concern": ["should", "could", "would", "maybe", "worry", "careful", "problem", "difference"],
    "calm": ["okay", "fine", "sure", "right", "yes", "well"],
}

TEMPORAL_PROFILES = {
    "happy": np.array([0.65, 0.90, 1.08, 1.00, 0.82], dtype=np.float32),
    "sad": np.array([0.75, 0.88, 1.00, 1.05, 0.98], dtype=np.float32),
    "angry": np.array([0.70, 1.02, 1.12, 1.08, 0.92], dtype=np.float32),
    "surprise": np.array([0.55, 1.15, 1.10, 0.78, 0.62], dtype=np.float32),
    "concern": np.array([0.72, 0.95, 1.04, 1.04, 0.90], dtype=np.float32),
    "calm": np.array([0.88, 0.95, 0.98, 0.95, 0.88], dtype=np.float32),
    "neutral": np.array([0.82, 0.95, 1.00, 0.95, 0.82], dtype=np.float32),
}

EMOTION_GROUPS = {
    "happy": np.r_[0:10, 20:30],
    "sad": np.r_[10:20, 30:40],
    "angry": np.r_[10:20, 30:40, 50:53],
    "surprise": np.r_[20:30, 50:53],
    "concern": np.r_[10:30],
    "calm": np.r_[0:53],
    "neutral": np.r_[0:53],
}


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def infer_prior(text: str) -> tuple[str, float, dict[str, float]]:
    lowered = text.lower()
    scores = {}
    for emotion, words in EMOTION_KEYWORDS.items():
        score = 0.0
        for word in words:
            if word in lowered:
                score += 1.0
        scores[emotion] = score
    emotion, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        emotion = "neutral"
    intensity = min(1.0, 0.35 + 0.18 * score)
    if "!" in text:
        intensity = min(1.0, intensity + 0.10)
    if "?" in text and emotion in {"surprise", "concern", "neutral"}:
        emotion = "surprise" if score > 0 else "concern"
        intensity = min(1.0, intensity + 0.08)
    return emotion, float(intensity), scores


def interpolate_profile(profile: np.ndarray, length: int) -> np.ndarray:
    src_x = np.linspace(0.0, 1.0, len(profile))
    dst_x = np.linspace(0.0, 1.0, length)
    return np.interp(dst_x, src_x, profile).astype(np.float32)


def apply_prior(seq: np.ndarray, emotion: str, intensity: float, alpha: float) -> np.ndarray:
    profile = interpolate_profile(TEMPORAL_PROFILES.get(emotion, TEMPORAL_PROFILES["neutral"]), len(seq))
    factor = 1.0 + alpha * intensity * (profile - 1.0)
    out = seq.copy()
    group = EMOTION_GROUPS.get(emotion, EMOTION_GROUPS["neutral"])
    out[:, group] = out[:, group] * factor[:, None]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-report", default=None)
    args = parser.parse_args()

    pred = torch.load(args.prediction, map_location="cpu").numpy().astype(np.float32)
    texts = load_pickle(Path(args.dataset_dir) / f"{args.split}_stage1_text.pkl")
    adjusted = np.empty_like(pred)
    prior_rows = []
    for i, text in enumerate(texts[: pred.shape[0]]):
        emotion, intensity, scores = infer_prior(text)
        adjusted[i] = apply_prior(pred[i], emotion, intensity, args.alpha)
        prior_rows.append((i, text, emotion, intensity, scores))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(adjusted, dtype=torch.float32), output)
    print(f"saved {output} {adjusted.shape}")

    if args.prior_report:
        lines = ["# Emotion Prior Report", "", f"- Alpha: `{args.alpha}`", ""]
        counts = {}
        for _, _, emotion, _, _ in prior_rows:
            counts[emotion] = counts.get(emotion, 0) + 1
        lines.append("## Counts")
        lines.append("")
        for emotion, count in sorted(counts.items()):
            lines.append(f"- {emotion}: `{count}`")
        lines.extend(["", "## First Samples", ""])
        for i, text, emotion, intensity, scores in prior_rows[:20]:
            non_zero = {k: v for k, v in scores.items() if v > 0}
            lines.append(f"{i + 1}. `{emotion}` intensity `{intensity:.2f}` text: {text} scores: `{non_zero}`")
        report = Path(args.prior_report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
