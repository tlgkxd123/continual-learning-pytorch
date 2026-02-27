"""Expert slot reservation: soft-lock old, allocate new for novel domains."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class ExpertReserver:
    """Router over adapter slots. Old slots: low lr. New slots for novel domains."""

    def __init__(self, num_base_slots: int = 4, max_slots: int = 16):
        self.num_base_slots = num_base_slots
        self.max_slots = max_slots
        self.slot_lr_mult: Dict[int, float] = {i: 0.01 for i in range(num_base_slots)}
        self.num_allocated = num_base_slots

    def allocate_new_slot(self) -> Optional[int]:
        """Allocate a new expert slot. Returns slot idx or None if full."""
        if self.num_allocated >= self.max_slots:
            return None
        idx = self.num_allocated
        self.num_allocated += 1
        self.slot_lr_mult[idx] = 1.0
        return idx

    def get_lr_multiplier(self, param_name: str, slot_idx: int) -> float:
        """Return learning rate multiplier for param in given slot."""
        return self.slot_lr_mult.get(slot_idx, 1.0)
