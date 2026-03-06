"""GRPO: grouped relative proximal optimization regularizer."""

from typing import Dict, Optional

import torch
import torch.nn as nn


class GRPO:
    """Group-relative proximal regularization over parameter drift."""

    def __init__(self, lambda_: float = 1000.0, group_size: int = 4, eps: float = 1e-6):
        self.lambda_ = lambda_
        self.group_size = max(1, group_size)
        self.eps = eps
        self.reference_params: Dict[str, torch.Tensor] = {}

    def snapshot_reference(self, model: nn.Module):
        """Save current trainable parameters as GRPO reference."""
        self.reference_params = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards per group (or globally for short vectors)."""
        if rewards.numel() == 0:
            return rewards
        if rewards.numel() <= self.group_size:
            return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + self.eps)

        normalized = torch.empty_like(rewards)
        for i in range(0, rewards.numel(), self.group_size):
            chunk = rewards[i : i + self.group_size]
            normalized[i : i + self.group_size] = (chunk - chunk.mean()) / (
                chunk.std(unbiased=False) + self.eps
            )
        return normalized

    def penalty(self, model: nn.Module, rewards: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return GRPO penalty weighted by group-relative reward signal."""
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
        if rewards is not None and rewards.numel() > 0:
            relative = self.normalize_rewards(rewards).mean().item()
            scale = max(0.0, 1.0 - relative)
        return self.lambda_ * scale * loss
