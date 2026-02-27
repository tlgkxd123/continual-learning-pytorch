"""Agent memory: Working, Episodic, Semantic, Procedural."""

from typing import Any, Dict, List, Optional

import torch


class AgentMemory:
    """Working (KV), Episodic (vector DB), Semantic (LoRA), Procedural (cached graphs)."""

    def __init__(
        self,
        working_size: int = 128_000,
    ):
        self.working_size = working_size
        self._episodic: List[Dict[str, Any]] = []
        self._procedural: Dict[str, Any] = {}

    def add_episodic(self, key: str, value: Any, embedding: Optional[torch.Tensor] = None):
        """Add to episodic memory (simplified list; can swap for FAISS)."""
        self._episodic.append({"key": key, "value": value, "embedding": embedding})

    def add_procedural(self, task: str, call_graph: Dict):
        """Cache tool-call graph for common task."""
        self._procedural[task] = call_graph

    def get_procedural(self, task: str) -> Optional[Dict]:
        return self._procedural.get(task)

    def get_episodic(self, query: str, k: int = 5) -> List[Any]:
        """Retrieve k nearest (simplified: return last k)."""
        return [e["value"] for e in self._episodic[-k:]]
