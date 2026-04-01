import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 然后再导入 sentence_transformers
from sentence_transformers import SentenceTransformer
import torch
# ... 其余代码
import os
import sys
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_mapper import MLPMapper
from utils.param_utils import vector_to_params_dict, clamp_params_dict

EMOTION_LIST = [
    "happy", "sad", "angry", "surprise",
    "disgust", "concern", "bored", "calm"
]

def emotion_to_onehot(emotion):
    vec = [0.0] * len(EMOTION_LIST)
    vec[EMOTION_LIST.index(emotion)] = 1.0
    return vec

def build_input(text, emotion, intensity, encoder):
    text_emb = encoder.encode(text)
    emo_vec = emotion_to_onehot(emotion)
    intensity_vec = [float(intensity)]
    x = list(text_emb) + emo_vec + intensity_vec
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

def main():
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    input_dim = 384 + 8 + 1   # MiniLM embedding + emotion onehot + intensity
    output_dim = 22

    model = MLPMapper(input_dim, output_dim)
    model.load_state_dict(torch.load("outputs/mlp_mapper.pt", map_location="cpu"))
    model.eval()

    text = "真恶心，快拿走"
    emotion = "surprise"
    intensity = 0.9

    x = build_input(text, emotion, intensity, encoder)

    with torch.no_grad():
        pred = model(x)[0].tolist()

    params = vector_to_params_dict(pred)
    params = clamp_params_dict(params)

    print("输入文本：", text)
    print("预测参数：")
    for k, v in params.items():
        if v > 0.05:
            print(f"{k}: {v:.4f}")
    from visualize import draw_face

    draw_face(params, title=text)

if __name__ == "__main__":
    main()

