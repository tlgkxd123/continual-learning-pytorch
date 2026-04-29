"""SOAR model configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SOARConfig:
    """Configuration for SOAR LLM (Qwen3.5-0.8B base)."""

    # Base model
    model_name: str = "Qwen/Qwen3.5-0.8B"
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 8
    intermediate_size: int = 3584
    vocab_size: int = 248320

    # TTT (Test-Time Training)
    ttt_adapter_ratio: float = 0.001
    ttt_bottleneck_dim: int = 42
    ttt_enabled: bool = True

    # Shampoo-lite
    shampoo_update_freq: int = 1
    shampoo_momentum: float = 0.9
    shampoo_eps: float = 1e-6

    # Early exit (24 layers; use 8, 16, 24)
    early_exit_layers: List[int] = field(default_factory=lambda: [8, 16, 24])
    early_exit_threshold: float = 0.7

    # Continual Learning
    replay_buffer_size: int = 100_000
    replay_sample_ratio: float = 0.05
    ewc_lambda: float = 1000.0
    lora_rank: int = 8
    lora_alpha: int = 16
    ttt_replay_weight: float = 0.05
    ttt_ewc_weight: float = 0.05
    ttt_rlvr_weight: float = 0.1
    ttt_grpo_weight: float = 0.1

    # Refinement
    max_refinement_loops: int = 8
    confidence_threshold: float = 0.8

    # Scratchpad
    scratchpad_max_tokens: int = 4096

    # Agent
    max_tool_calls_per_turn: int = 5
    working_memory_tokens: int = 128_000

    # MCTS
    mcts_beam_width: int = 8
    mcts_max_depth: int = 32

    # Matryoshka Slicing (Register-Level Dynamic Precision)
    # See docs/DESIGN_MATRYOSHKA_SLICING.md
    matryoshka_enabled: bool = False
    matryoshka_precision: str = "8"  # "2" | "4" | "8" | "auto" (SM-aware)
