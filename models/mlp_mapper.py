import torch
import torch.nn as nn

class MLPMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()   # 大部分参数是 0~1
        )

    def forward(self, x):
        return self.net(x)