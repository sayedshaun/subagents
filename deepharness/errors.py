from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .providers.base import TokenUsage

__all__ = [
    "ConcurrentUpdateError",
    "ConfigurationError",
    "DeepHarnessError",
    "ExecutionError",
    "HumanInputRequired",
    "MCPError",
    "OutputValidationError",
    "OutsideWorkspace",
    "ProviderError",
    "StepLimitExceeded",
    "TokenBudgetExceeded",
    "ToolDenied",
    "ToolNotFoundError",
]


class DeepHarnessError(Exception):
    """Base class for all errors raised by deepharness."""


class ConfigurationError(DeepHarnessError, ValueError):
    """Raised when a component is built or used in an invalid configuration.

    Also a ValueError - an invalid argument is what this is, and graph/ raised
    bare ValueErrors before it existed.
    """


class ToolNotFoundError(DeepHarnessError, KeyError):
    """Raised when a Toolbox is asked for a tool that isn't registered."""


class OutsideWorkspace(DeepHarnessError, ValueError):
    """Raised when a tool is asked for a path outside its workspace root.

    Reaches the model as that call's result, since a model that asked for the
    wrong path can correct itself - but the read never happens.
    """

    def __init__(self, path: str, root: Any):
        self.path = path
        self.root = root
        super().__init__(f"path {path!r} is outside the workspace root {root}")


class OutputValidationError(DeepHarnessError):
    """Raised when a model's structured answer does not fit the output= shape.

    Handed back to the model as the failing call's result rather than ending
    the run, so it can correct the fields and answer again.
    """


class MCPError(DeepHarnessError):
    """Raised when an MCP server refuses a call, fails, or answers unusably.

    A tool call that fails on the server reaches the model as that call's
    result, same as a local tool raising - the run continues, and the model gets
    a turn to try something else.
    """


class ProviderError(DeepHarnessError):
    """Raised when an LLM provider request fails after retries."""


class ToolDenied(DeepHarnessError):
    """Raised when a Toolbox's approve= callback refuses a gated call.

    An error rather than a canned result, so the caller decides what a refusal
    means; inside an Agent it reaches the model as that call's result.
    """

    def __init__(self, name: str, arguments: dict[str, Any]):
        self.name = name
        self.arguments = arguments
        super().__init__(f"call to {name!r} was not approved")


class HumanInputRequired(DeepHarnessError):
    """Raised by a tool to pause the agent and wait for a human answer."""

    def __init__(self, question: str):
        self.question = question
        super().__init__(question)


class ExecutionError(DeepHarnessError):
    """Raised when a node function raises during graph execution."""

    def __init__(self, node_name: str, original: Exception):
        self.node_name = node_name
        self.original = original
        super().__init__(f"Node '{node_name}' failed: {original!r}")


class ConcurrentUpdateError(DeepHarnessError):
    """Raised when parallel branches write the same state field with no reducer.

    Silently picking a winner would drop one branch's work, so the graph
    refuses to guess: declare a reducer on the field, or give each branch its
    own field.
    """

    def __init__(self, field_name: str, writer_count: int):
        self.field_name = field_name
        self.writer_count = writer_count
        super().__init__(
            f"{writer_count} concurrent branches wrote '{field_name}' and it "
            f"declares no reducer; add field(metadata={{'reducer': ...}}) to it "
            f"or give each branch its own field"
        )


class StepLimitExceeded(DeepHarnessError):
    """Raised when a graph run exceeds max_steps, usually a loop that never exits.

    state carries the partial result so a run that hits the limit can still
    be inspected, matching TokenBudgetExceeded.
    """

    def __init__(self, limit: int, state: Any | None = None):
        self.limit = limit
        self.state = state
        super().__init__(f"Graph exceeded its limit of {limit} steps")


class TokenBudgetExceeded(DeepHarnessError):
    """Raised when an Agent's cumulative token usage exceeds its Budget.tokens.

    state carries the agent's partial result, so the tokens already paid for
    aren't lost with the exception - inspect it, or resume from its messages.
    """

    def __init__(
        self,
        agent_name: str,
        usage: TokenUsage,
        budget: int,
        state: dict[str, Any] | None = None,
    ):
        self.agent_name = agent_name
        self.usage = usage
        self.budget = budget
        self.state = state
        super().__init__(
            f"{agent_name} used {usage.total_tokens} tokens, "
            f"exceeding its budget of {budget}"
        )
