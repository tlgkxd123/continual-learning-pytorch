"""EWC++: Fisher Information masking for continual learning."""

from typing import Dict

import torch
import torch.nn as nn


class EWCPlus:
    """Diagonal Fisher per adapter/block. Penalty: lambda * sum(F_i * (theta - theta_old)^2)."""

    def __init__(self, lambda_: float = 1000.0):
        self.lambda_ = lambda_
        self.fisher: Dict[str, torch.Tensor] = {}
        self.opt_params: Dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        max_batches: int = 100,
    ):
        """Accumulate diagonal Fisher from model gradients on data."""
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            model.zero_grad()
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch["input_ids"].to(device)
            out = model(input_ids=x, labels=x)
            loss = out.get("loss") if isinstance(out, dict) else getattr(out, "loss", None)
            if loss is None:
                continue
            if not isinstance(loss, torch.Tensor):
                continue
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n].add_(p.grad.pow(2))

        self.fisher = fisher
        self.opt_params = {n: p.detach().clone() for n, p in model.named_parameters() if n in fisher}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC penalty term."""
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.opt_params:
                loss = loss + (self.fisher[n] * (p - self.opt_params[n]).pow(2)).sum()
        return self.lambda_ * loss
