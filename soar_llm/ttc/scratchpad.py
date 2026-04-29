"""Scratchpad: hidden <think> buffer, tokens excluded from output."""

from typing import List, Optional

import torch


class ScratchpadBuffer:
    """<think> tokens: generate internally, attend into output, exclude from final text."""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self._buffer: List[int] = []

    def is_active(self) -> bool:
        return len(self._buffer) > 0

    def add(self, token_ids: List[int]):
        """Add tokens to scratchpad."""
        self._buffer.extend(token_ids)
        if len(self._buffer) > self.max_tokens:
            self._buffer = self._buffer[-self.max_tokens:]

    def clear(self):
        self._buffer.clear()

    def get_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        """Return buffer as tensor for attention context."""
        if not self._buffer:
            return None
        return torch.tensor([self._buffer], dtype=torch.long, device=device)

    def filter_output(self, output_ids: List[int], think_start: int, think_end: int) -> List[int]:
        """Remove <think>...</think> span from output."""
        return output_ids[:think_start] + output_ids[think_end:]
