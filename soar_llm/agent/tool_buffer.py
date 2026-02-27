"""Tool call buffer; inject <TOOL_RESULT> back into stream."""

from typing import Any, Dict, List, Optional


class ToolBuffer:
    """Buffer for tool calls; inject results as <TOOL_RESULT>...</TOOL_RESULT>."""

    def __init__(self):
        self._pending: List[Dict[str, Any]] = []
        self._results: List[str] = []

    def add_call(self, tool: str, args: Dict[str, Any]):
        self._pending.append({"tool": tool, "args": args})

    def add_result(self, result: str):
        self._results.append(result)

    def format_result_tokens(self, result: str) -> str:
        return f"<TOOL_RESULT>{result}</TOOL_RESULT>"

    def get_injectable_text(self) -> str:
        """Get all results as single injectable string."""
        return " ".join(
            self.format_result_tokens(r) for r in self._results
        )

    def clear(self):
        self._pending.clear()
        self._results.clear()
