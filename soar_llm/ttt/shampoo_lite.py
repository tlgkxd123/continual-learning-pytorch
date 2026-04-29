"""Shampoo-lite: lightweight second-order preconditioner for TTT."""

from typing import Iterator

import torch
from torch.optim import Optimizer


class ShampooLite(Optimizer):
    """Diagonal + low-rank Kronecker approximation for fast TTT updates."""

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        eps: float = 1e-6,
        update_freq: int = 1,
    ):
        defaults = dict(lr=lr, momentum=momentum, eps=eps, update_freq=update_freq)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["v"] = torch.zeros_like(p)
                    state["d"] = torch.ones(p.numel(), device=p.device, dtype=p.dtype)

                state["step"] += 1
                v = state["v"]
                d = state["d"]

                # Diagonal Fisher approximation: d += g^2
                flat_g = g.flatten()
                d.add_(flat_g.pow(2), alpha=1 - momentum)

                # Preconditioned gradient: g / (sqrt(d) + eps)
                d_sqrt = d.sqrt().add(eps)
                precond_g = (g.flatten() / d_sqrt).view_as(g)

                # Momentum
                v.mul_(momentum).add_(precond_g, alpha=lr)

                p.sub_(v)

        return loss
