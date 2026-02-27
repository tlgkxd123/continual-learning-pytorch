"""Native tool tokens and argument schema."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

TOOL_TOKENS = [
    "<SEARCH>",
    "<CODE_EXEC>",
    "<FILE_READ>",
    "<API_CALL>",
    "<MEMORY_WRITE>",
    "<SPAWN_AGENT>",
]

TOOL_ARG_PATTERN = re.compile(
    r"(<SEARCH>|<CODE_EXEC>|<FILE_READ>|<API_CALL>|<MEMORY_WRITE>|<SPAWN_AGENT>)\s*(\{[^}]*\})"
)


def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse model output for <TOOL>{"arg": "val"} patterns."""
    result = []
    for m in TOOL_ARG_PATTERN.finditer(text):
        tool_name = m.group(1).strip("<>")
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        result.append((tool_name, args))
    return result
