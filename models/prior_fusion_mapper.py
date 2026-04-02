import torch
import torch.nn as nn


class PriorFusionMapper(nn.Module):
    def __init__(self, text_dim, control_dim, prior_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.control_branch = nn.Sequential(
            nn.Linear(control_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + prior_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        self.residual_scale = 0.35

    def forward(self, text_features, control_features, prior_vector):
        text_repr = self.text_branch(text_features)
        control_repr = self.control_branch(control_features)
        fused = torch.cat([text_repr, control_repr, prior_vector], dim=-1)
        residual = self.fusion(fused) * self.residual_scale
        prediction = prior_vector + residual
        return torch.clamp(prediction, min=0.0, max=1.0)
