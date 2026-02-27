"""ER-ACE style replay buffer with surprise-driven selection."""

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ReplaySample:
    """Single replay sample with loss for priority."""

    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    loss: float


class ReplayBuffer:
    """~100K tokens. 5% of batch from replay. Surprise = high loss -> keep."""

    def __init__(self, max_tokens: int = 100_000, sample_ratio: float = 0.05):
        self.max_tokens = max_tokens
        self.sample_ratio = sample_ratio
        self._samples: deque = deque()
        self._total_tokens = 0

    def add(self, input_ids: torch.Tensor, loss: float, attention_mask: Optional[torch.Tensor] = None):
        """Add sample. Prefer high-loss (surprise). Evict low-loss when full."""
        tokens = input_ids.numel()
        sample = ReplaySample(input_ids=input_ids, attention_mask=attention_mask, loss=loss)
        while self._total_tokens + tokens > self.max_tokens and self._samples:
            old = self._samples[0]
            if old.loss >= loss:
                break
            self._samples.popleft()
            self._total_tokens -= old.input_ids.numel()
        if self._total_tokens + tokens <= self.max_tokens:
            self._samples.append(sample)
            self._total_tokens += tokens

    def sample(self, batch_size: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        """Sample ~5% of batch from replay. Returns dict or None if empty."""
        n = max(1, int(batch_size * self.sample_ratio))
        if not self._samples:
            return None
        chosen = random.choices(list(self._samples), k=min(n, len(self._samples)))
        input_ids = torch.cat([s.input_ids for s in chosen], dim=0).to(device)
        masks = [s.attention_mask for s in chosen if s.attention_mask is not None]
        attention_mask = torch.cat(masks, dim=0).to(device) if masks else None
        return {"input_ids": input_ids, "attention_mask": attention_mask}
