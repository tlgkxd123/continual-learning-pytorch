"""Dynamic RL: reward computation and adaptive learning-rate for TTT."""

import math

import torch
import torch.nn.functional as F


class DynamicRL:
    """Compute per-step reward from model output quality and adapt optimizer LR.

    High reward  → confident outputs → allow larger adapter updates.
    Low  reward  → uncertain outputs → conservative (small) updates.
    """

    # ------------------------------------------------------------------ rewards

    def compute_confidence_reward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,  # noqa: ARG002  (kept for API symmetry)
    ) -> float:
        """Mean max-softmax-probability over all positions.  Range ∈ (0, 1]."""
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            return float(probs.max(dim=-1).values.mean().item())

    def compute_perplexity_reward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> float:
        """Reward = exp(−PPL / 100).  Lower perplexity ⟹ higher reward ∈ (0, 1]."""
        with torch.no_grad():
            labels = input_ids[:, 1:].contiguous()
            shift_logits = logits[:, :-1].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                labels.view(-1),
            )
            ppl = torch.exp(loss).item()
        return math.exp(-ppl / 100.0)

    # --------------------------------------------------------------- LR control

    def adapt_lr(
        self,
        optimizer: torch.optim.Optimizer,
        reward: float,
        base_lr: float,
        min_scale: float = 0.1,
        max_scale: float = 2.0,
    ) -> float:
        """Scale each param-group LR by a reward-derived factor.

        Returns the new LR applied to the first param group.
        """
        reward = max(0.0, min(1.0, float(reward)))
        scale = min_scale + (max_scale - min_scale) * reward
        new_lr = base_lr * scale
        for group in optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr
