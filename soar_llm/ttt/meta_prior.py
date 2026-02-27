"""MAML-style meta-learned prior for TTT adapter init."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adapter import TTTRouter
from ..config import SOARConfig


class MAMLTrainer:
    """Inner loop: 1-3 steps on mini-task. Outer loop: meta-update adapter init."""

    def __init__(
        self,
        adapter: TTTRouter,
        inner_lr: float = 1e-3,
        outer_lr: float = 1e-4,
        inner_steps: int = 3,
    ):
        self.adapter = adapter
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

    def inner_loop(
        self,
        base_model,
        input_ids: torch.Tensor,
        mask_ratio: float = 0.15,
    ) -> torch.Tensor:
        """Run inner loop on one mini-task: mask tokens, predict, return loss."""
        b, s = input_ids.shape
        mask = torch.rand(b, s, device=input_ids.device) < mask_ratio
        mask = mask & (input_ids != 0)
        labels = input_ids.clone()
        labels[~mask] = -100

        with torch.enable_grad():
            for _ in range(self.inner_steps):
                out = base_model(input_ids=input_ids)
                logits = out.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                loss.backward()
                for p in self.adapter.parameters():
                    if p.grad is not None:
                        p.data.sub_(p.grad, alpha=self.inner_lr)
                        p.grad.zero_()
        return loss.detach()

    def meta_step(
        self,
        base_model,
        support_batch: torch.Tensor,
        query_batch: torch.Tensor,
        meta_optimizer: torch.optim.Optimizer,
    ) -> float:
        """One meta-update: inner on support, evaluate on query, meta-backprop."""
        meta_optimizer.zero_grad()
        # Clone adapter for inner loop (functional MAML style)
        old_params = [p.clone() for p in self.adapter.parameters()]
        _ = self.inner_loop(base_model, support_batch)
        # Query loss
        with torch.no_grad():
            out = base_model(input_ids=query_batch)
        labels = query_batch[:, 1:].contiguous()
        logits = out.logits[:, :-1].contiguous()
        query_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=0,
        )
        query_loss.backward()
        meta_optimizer.step()
        # Restore adapter
        for p, old in zip(self.adapter.parameters(), old_params):
            p.data.copy_(old)
        return query_loss.item()
