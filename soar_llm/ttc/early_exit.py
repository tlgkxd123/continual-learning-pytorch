"""Early exit classifiers at specified layers."""

import torch
import torch.nn as nn


class EarlyExitClassifier(nn.Module):
    """Small MLP: hidden -> exit probability."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # [B, S, H] -> [B, S, 1] -> mean over seq for per-sample score
        return self.mlp(hidden).mean(dim=1)
