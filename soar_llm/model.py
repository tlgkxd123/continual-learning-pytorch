"""SOAR model: HF base + TTT adapters + TTC + CL gate + agent."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

from .config import SOARConfig


class SOARModel(nn.Module):
    """Assembles HF base + TTT adapters + early exit + refinement + agent detection."""

    def __init__(self, config: Optional[SOARConfig] = None):
        super().__init__()
        self.config = config or SOARConfig()
        self._base = GPT2LMHeadModel.from_pretrained(self.config.model_name)
        for p in self._base.parameters():
            p.requires_grad = False

        self._ttt_adapters = None
        self._early_exit_classifiers = None
        self._verifier_head = None
        self._hooks: List[Any] = []

    def _lazy_init_ttt(self):
        """Lazy init TTT adapters (imported here to avoid circular deps)."""
        if self._ttt_adapters is None:
            from .ttt.adapter import TTTRouter
            self._ttt_adapters = TTTRouter(self.config).to(
                next(self._base.parameters()).device
            )
        return self._ttt_adapters

    def _lazy_init_early_exit(self):
        """Lazy init early exit classifiers."""
        if self._early_exit_classifiers is None:
            from .ttc.early_exit import EarlyExitClassifier
            self._early_exit_classifiers = nn.ModuleDict()
            for layer_idx in self.config.early_exit_layers:
                self._early_exit_classifiers[str(layer_idx)] = EarlyExitClassifier(
                    self.config.hidden_size
                ).to(next(self._base.parameters()).device)
        return self._early_exit_classifiers

    def _lazy_init_verifier(self):
        """Lazy init verifier head."""
        if self._verifier_head is None:
            from .ttc.refinement import VerifierHead
            self._verifier_head = VerifierHead(self.config.hidden_size).to(
                next(self._base.parameters()).device
            )
        return self._verifier_head

    def _ensure_ttt_hooks(self):
        """Register TTT hooks if not already active."""
        if not self._hooks:
            adapters = self._lazy_init_ttt()
            self._hooks = adapters.register_hooks(self._base)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_ttt: bool = False,
        use_early_exit: bool = False,
        return_hidden_per_layer: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through base model with optional TTT and early exit."""
        if use_ttt:
            self._ensure_ttt_hooks()
        outputs = self._base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=use_early_exit or return_hidden_per_layer,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states

        if use_early_exit and hidden_states:
            classifiers = self._lazy_init_early_exit()
            for layer_idx in self.config.early_exit_layers:
                if str(layer_idx) in classifiers and layer_idx < len(hidden_states):
                    hs = hidden_states[layer_idx]
                    exit_prob = classifiers[str(layer_idx)](hs)
                    if exit_prob.mean() > self.config.early_exit_threshold:
                        # Use this layer's hidden for logits
                        logits = self._base.lm_head(hidden_states[layer_idx])
                        break

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=-100
            )

        result = {"loss": loss, "logits": logits}
        if return_hidden_per_layer and hidden_states:
            result["hidden_states"] = hidden_states
        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens. kwargs passed to HF generate."""
        return self._base.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def get_ttt_params(self) -> List[nn.Parameter]:
        """Parameters that receive gradients during TTT."""
        if self._ttt_adapters is None:
            return []
        return list(self._ttt_adapters.parameters())

    def freeze_base(self):
        """Ensure base weights are frozen."""
        for p in self._base.parameters():
            p.requires_grad = False
