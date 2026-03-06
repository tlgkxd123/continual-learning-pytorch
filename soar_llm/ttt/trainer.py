"""TTTContinualTrainer: test-time training with continual-learning regularisers.

Combines:
  * TTT adapter hooks (PlasticAdapter / TTTRouter) — only these params are
    updated; base-model weights stay frozen.
  * Shampoo-lite optimiser — lightweight 2nd-order preconditioner.
  * RLVR  — reward-linked value regularisation.
  * Dynamic GRPO — group-relative policy optimisation penalty.
  * EWC+  — elastic weight consolidation (diagonal Fisher).
  * Replay buffer — 5 % of each step sampled from past contexts.
  * DynamicRL  — adaptive LR scaled by output-confidence reward.
"""

import threading
from typing import Optional

import torch
import torch.nn.functional as F

from ..config import SOARConfig
from ..continual.dynamic_rl import DynamicRL
from ..continual.ewc_plus import EWCPlus
from ..continual.grpo import GRPO
from ..continual.replay_buffer import ReplayBuffer
from ..continual.rlvr import RLVR
from .adapter import TTTRouter
from .shampoo_lite import ShampooLite

# How often (in steps) to re-snapshot RLVR / GRPO reference weights.
_REFERENCE_SNAPSHOT_FREQ = 50


class TTTContinualTrainer:
    """On-the-fly adapter training triggered after every inference call.

    Usage::

        trainer = TTTContinualTrainer(model, config, device)
        # … generate response …
        trainer.train_step(input_ids, reward=confidence_score)
        print(trainer.stats)
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        config: SOARConfig,
        device: torch.device,
        ttt_lr: float = 1e-4,
        replay_weight: float = 0.05,
    ) -> None:
        self.base_model = base_model
        self.config = config
        self.device = device
        self.ttt_lr = ttt_lr
        self.replay_weight = replay_weight

        # ---- TTT adapter --------------------------------------------------- #
        self.ttt_router = TTTRouter(config).to(device)
        try:
            self._hooks = self.ttt_router.register_hooks(base_model)
        except (ValueError, AttributeError):
            # Graceful fallback: adapters are trained but not injected.
            self._hooks = []
        self.hooks_active = bool(self._hooks)

        # Freeze base-model params so only adapter params are trained.
        for p in base_model.parameters():
            p.requires_grad_(False)

        # ---- Optimiser ----------------------------------------------------- #
        self.optimizer = ShampooLite(self.ttt_router.parameters(), lr=ttt_lr)

        # ---- Continual-learning regularisers ------------------------------- #
        self.rlvr = RLVR(lambda_=config.ewc_lambda)
        self.grpo = GRPO(lambda_=config.ewc_lambda, group_size=4)
        self.ewc = EWCPlus(lambda_=config.ewc_lambda)
        self.replay_buffer = ReplayBuffer(
            config.replay_buffer_size, config.replay_sample_ratio
        )
        self.dynamic_rl = DynamicRL()

        # Take initial reference snapshots for RLVR / GRPO.
        self.rlvr.snapshot_reference(self.ttt_router)
        self.grpo.snapshot_reference(self.ttt_router)

        # ---- Internal state ------------------------------------------------ #
        self.step_count: int = 0
        self.running_loss: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ private

    def _lm_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Causal LM next-token-prediction loss."""
        out = self.base_model(input_ids=input_ids)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    # ------------------------------------------------------------------ public

    def train_step(
        self,
        input_ids: torch.Tensor,
        reward: Optional[float] = None,
        rewards_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        """One TTT gradient step.  Thread-safe (holds an internal lock).

        Args:
            input_ids:      [1, seq_len] tokenised context (prompt + response).
            reward:         Scalar reward ∈ [0, 1] for RLVR scaling.
                            ``None`` disables reward-based RLVR scaling.
            rewards_tensor: Group rewards for dynamic GRPO.
                            ``None`` applies a uniform GRPO penalty.

        Returns:
            Scalar total-loss value.
        """
        with self._lock:
            self.base_model.eval()
            self.ttt_router.train()
            self.optimizer.zero_grad()

            with torch.enable_grad():
                # 1. Self-supervised LM loss on current context.
                lm = self._lm_loss(input_ids)

                # 2. RLVR — keep adapter close to reward-verified reference.
                rlvr_pen = self.rlvr.penalty(self.ttt_router, reward=reward)

                # 3. Dynamic GRPO — group-relative policy penalty.
                grpo_pen = self.grpo.penalty(
                    self.ttt_router, rewards=rewards_tensor
                )

                # 4. EWC+ — Fisher-weighted parameter drift penalty.
                # EWC penalty returns 0.0 (float) before Fisher is computed;
                # guard against that to keep the computation graph valid.
                ewc_pen = self.ewc.penalty(self.ttt_router)
                if not isinstance(ewc_pen, torch.Tensor):
                    ewc_pen = torch.tensor(0.0, device=self.device)

                # 5. Replay — small fraction of past contexts.
                replay_loss = torch.tensor(0.0, device=self.device)
                replay_batch = self.replay_buffer.sample(1, self.device)
                if replay_batch is not None:
                    r_ids = replay_batch["input_ids"]
                    if r_ids.numel() > 1:
                        replay_loss = self._lm_loss(r_ids[:1])

                total = (
                    lm
                    + rlvr_pen
                    + grpo_pen
                    + ewc_pen
                    + self.replay_weight * replay_loss
                )
                total.backward()

            self.optimizer.step()

            # Dynamically adapt LR based on reward confidence.
            if reward is not None:
                self.dynamic_rl.adapt_lr(self.optimizer, reward, self.ttt_lr)

            # Store context in replay buffer for future steps.
            self.replay_buffer.add(input_ids.detach().cpu(), lm.item())

            # Update running stats.
            self.step_count += 1
            loss_val = total.item()
            self.running_loss = 0.9 * self.running_loss + 0.1 * loss_val

            # Periodically re-snapshot RLVR / GRPO reference weights.
            if self.step_count % _REFERENCE_SNAPSHOT_FREQ == 0:
                self.rlvr.snapshot_reference(self.ttt_router)
                self.grpo.snapshot_reference(self.ttt_router)

            self.ttt_router.eval()
            return loss_val

    def compute_reward(
        self, logits: torch.Tensor, input_ids: torch.Tensor
    ) -> float:
        """Convenience wrapper: confidence-based self-supervised reward."""
        return self.dynamic_rl.compute_confidence_reward(logits, input_ids)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks from the base model."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.hooks_active = False

    @property
    def stats(self) -> dict:
        """Return a JSON-serialisable stats snapshot."""
        return {
            "step_count": self.step_count,
            "running_loss": round(self.running_loss, 4),
            "ttt_lr": round(self.ttt_lr, 6),
            "hooks_active": self.hooks_active,
            "replay_tokens": self.replay_buffer.total_tokens,
        }
