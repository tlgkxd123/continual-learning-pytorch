"""Plastic adapter layers for per-sample TTT updates."""

import math
from typing import Optional

import torch
import torch.nn as nn

from ..config import SOARConfig


class PlasticAdapter(nn.Module):
    """Small down->up projection, residual: x + adapter(adapter(x))."""

    def __init__(self, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class TTTRouter(nn.Module):
    """Per-block plastic adapters for TTT. ~0.1% of block params."""

    def __init__(self, config: SOARConfig):
        super().__init__()
        self.config = config
        bottleneck = config.ttt_bottleneck_dim or max(32, config.hidden_size // 24)
        self.adapters_attn = nn.ModuleList([
            PlasticAdapter(config.hidden_size, bottleneck)
            for _ in range(config.num_layers)
        ])
        self.adapters_ffn = nn.ModuleList([
            PlasticAdapter(config.hidden_size, bottleneck)
            for _ in range(config.num_layers)
        ])

    def forward_attn_adapter(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if layer_idx < len(self.adapters_attn):
            return self.adapters_attn[layer_idx](x)
        return x

    def forward_ffn_adapter(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if layer_idx < len(self.adapters_ffn):
            return self.adapters_ffn[layer_idx](x)
        return x

    def _get_layers_and_attn_names(self, base_model):
        """Detect layer list and attention module name for different architectures."""
        if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            blocks = base_model.transformer.h
            return blocks, "gpt2"
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            blocks = base_model.model.layers
            return blocks, "llama_style"
        raise ValueError("Unsupported model architecture for TTT hooks")

    def register_hooks(self, base_model):
        """Register forward hooks on base model to inject adapter outputs."""
        blocks, arch = self._get_layers_and_attn_names(base_model)
        handles = []

        for i, block in enumerate(blocks):
            if i >= len(self.adapters_attn):
                break
            attn_ad = self.adapters_attn[i]
            ffn_ad = self.adapters_ffn[i]

            if arch == "gpt2":
                attn_mod = block.attn
                ffn_mod = block.mlp
            else:
                attn_mod = getattr(block, "self_attn", None) or getattr(block, "linear_attn", None)
                ffn_mod = block.mlp

            if attn_mod is not None:
                def attn_hook(m, inputs, output, ad=attn_ad):
                    if isinstance(output, tuple):
                        h = output[0]
                        rest = output[1:]
                        return (ad(h),) + rest
                    return ad(output)
                handles.append(attn_mod.register_forward_hook(attn_hook))

            def ffn_hook(m, inputs, output, ad=ffn_ad):
                return ad(output)
            handles.append(ffn_mod.register_forward_hook(ffn_hook))

        return handles
