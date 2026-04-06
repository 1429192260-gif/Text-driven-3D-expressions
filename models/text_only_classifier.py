import torch
import torch.nn as nn


class TextOnlyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_emotions, num_intensity):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)
        self.intensity_head = nn.Linear(hidden_dim, num_intensity)

    def forward(self, features):
        shared = self.backbone(features)
        emotion_logits = self.emotion_head(shared)
        intensity_logits = self.intensity_head(shared)
        return emotion_logits, intensity_logits
