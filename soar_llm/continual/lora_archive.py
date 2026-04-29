"""LoRA snapshot archive: save/merge by task context."""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class LoRAArchive:
    """Save LoRA deltas (~50MB) after each fine-tune. Merge by domain at inference."""

    def __init__(self, archive_dir: str = "./lora_archive"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: Dict[str, str] = {}

    def save(self, lora_state: Dict[str, torch.Tensor], domain: str) -> str:
        """Save LoRA state. Returns path."""
        path = self.archive_dir / f"{domain}.pt"
        torch.save(lora_state, path)
        self.snapshots[domain] = str(path)
        return str(path)

    def load(self, domain: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load LoRA state for domain."""
        path = self.snapshots.get(domain) or str(self.archive_dir / f"{domain}.pt")
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def merge_into_model(self, model: nn.Module, domain: str, alpha: float = 1.0) -> bool:
        """Merge loaded LoRA into model weights."""
        state = self.load(domain)
        if state is None:
            return False
        for name, delta in state.items():
            if "." in name and hasattr(model, "get_parameter"):
                try:
                    p = dict(model.named_parameters()).get(name)
                    if p is not None:
                        p.data.add_(delta.to(p.device), alpha=alpha)
                except Exception:
                    pass
        return True
