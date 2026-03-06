"""RLVR: reward-linked value regularization for continual learning."""

from typing import Dict, Optional

import torch
import torch.nn as nn


class RLVR:
    """Keep parameters close to a reward-verified reference model."""

    def __init__(self, lambda_: float = 1000.0):
        self.lambda_ = lambda_
        self.reference_params: Dict[str, torch.Tensor] = {}

    def snapshot_reference(self, model: nn.Module):
        """Save current trainable parameters as RLVR reference."""
        self.reference_params = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def penalty(self, model: nn.Module, reward: Optional[float] = None) -> torch.Tensor:
        """Return RLVR penalty, optionally scaled by reward confidence."""
        first_param = next(model.parameters(), None)
        if first_param is None:
            return torch.tensor(0.0)
        if not self.reference_params:
            return torch.tensor(0.0, device=first_param.device)

        loss = torch.tensor(0.0, device=first_param.device)
        for n, p in model.named_parameters():
            ref = self.reference_params.get(n)
            if ref is not None:
                loss = loss + (p - ref.to(p.device)).pow(2).sum()

        scale = 1.0
        if reward is not None:
            scale = max(0.0, 1.0 - float(reward))
        return self.lambda_ * scale * loss
