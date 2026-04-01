import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_mapper import MLPMapper

EMOTION_LIST = [
    "happy", "sad", "angry", "surprise",
    "disgust", "concern", "bored", "calm"
]

def emotion_to_onehot(emotion):
    vec = [0.0] * len(EMOTION_LIST)
    vec[EMOTION_LIST.index(emotion)] = 1.0
    return vec

class ExpressionDataset(Dataset):
    def __init__(self, path, encoder):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.encoder = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text_emb = self.encoder.encode(item["text"])
        emo_vec = emotion_to_onehot(item["emotion"])
        intensity = [float(item["intensity"])]

        x = list(text_emb) + emo_vec + intensity
        y = item["param_vector"]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main():
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    dataset = ExpressionDataset("data/train.json", encoder)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    sample_x, sample_y = dataset[0]
    input_dim = len(sample_x)
    output_dim = len(sample_y)

    model = MLPMapper(input_dim=input_dim, output_dim=output_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100

    for epoch in range(num_epochs):
        total_loss = 0.0

        for x, y in loader:
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/mlp_mapper.pt")
    print("Model saved to outputs/mlp_mapper.pt")

if __name__ == "__main__":
    main()