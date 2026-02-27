"""Confidence-gated refinement loop and verifier head."""

from typing import Optional

import torch
import torch.nn as nn


class VerifierHead(nn.Module):
    """2-layer MLP on mean-pooled sequence -> confidence 0-1."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-6))
        else:
            pooled = hidden.mean(dim=1)
        return self.mlp(pooled).squeeze(-1)


class RefinementLoop:
    """Run up to N refinement passes when confidence < threshold."""

    def __init__(self, max_loops: int = 8, threshold: float = 0.8):
        self.max_loops = max_loops
        self.threshold = threshold

    def should_continue(self, confidence: float, loop: int) -> bool:
        return loop < self.max_loops and confidence < self.threshold
