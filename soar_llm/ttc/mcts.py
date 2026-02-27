"""MCTS-lite: tree search with PRM scoring for planning."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch


@dataclass
class MCTSNode:
    """Single node in search tree."""

    token_ids: List[int]
    score: float
    children: List["MCTSNode"]


class MCTSDecoder:
    """Branch N continuations, score with PRM, follow best branch."""

    def __init__(
        self,
        beam_width: int = 8,
        max_depth: int = 32,
        scorer: Optional[Callable[[torch.Tensor], float]] = None,
    ):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self._scorer = scorer

    def decode(
        self,
        generate_fn: Callable[[torch.Tensor, int], torch.Tensor],
        prompt: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        """Generate beam_width branches, score, return best sequence and score."""
        best_seq = None
        best_score = -1e9
        for _ in range(self.beam_width):
            seq = generate_fn(prompt, self.max_depth)
            if self._scorer is not None:
                score = self._scorer(seq)
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_seq = seq
        return best_seq or prompt, best_score
