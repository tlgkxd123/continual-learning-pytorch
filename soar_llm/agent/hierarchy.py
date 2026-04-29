"""Hierarchical agent: Orchestrator -> Subagents -> Tools."""

from enum import Enum
from typing import Callable, Dict



class AgentState(Enum):
    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"
    RETRY = "retry"
    COMPLETE = "complete"


class AgentHierarchy:
    """Orchestrator plans; subagents execute; tools perform actions."""

    def __init__(
        self,
        orchestrator_fn: Callable[[str], str],
        tool_executor: Callable[[str, Dict], str],
    ):
        self.orchestrator_fn = orchestrator_fn
        self.tool_executor = tool_executor
        self.state = AgentState.PLAN
        self._error_model: Dict[str, int] = {}

    def run(self, task: str, max_steps: int = 10) -> str:
        """Execute task: plan -> execute -> reflect -> retry/complete."""
        self.state = AgentState.PLAN
        plan = self.orchestrator_fn(task)
        for _ in range(max_steps):
            if self.state == AgentState.COMPLETE:
                return plan
            if self.state == AgentState.EXECUTE:
                self._execute_tools(plan)
                self.state = AgentState.REFLECT
            elif self.state == AgentState.REFLECT:
                self.state = AgentState.COMPLETE
        return plan

    def _execute_tools(self, plan: str) -> str:
        """Execute tool calls from plan. Placeholder."""
        return ""
