"""What happens to a single turn: recording it, and gating it on a human.

Both halves are about a turn's outcome rather than the loop's mechanics, and
neither needs an Agent - only the transcript, the tools, and the calls the model
asked for. Keeping them here leaves loop.py to the loop itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import ConfigurationError, HumanInputRequired
from ..tools.permissions import Decision, Permissions
from ..tools.toolbox import Toolbox
from .context import truncate
from .state import AgentState, Message, PendingHumanInput, as_dict


def prepare(state: AgentState, system: str | None) -> list[dict[str, Any]]:
    """The transcript to send, as wire-form dicts, system prompt in front.

    Entries are normalized because a caller resuming a paused run appends a
    Message of their own, and a provider must never be handed one.
    """
    messages = [as_dict(message) for message in state.messages]
    if system and not any(m["role"] == "system" for m in messages):
        messages.insert(0, Message.system(system).to_dict())
    return messages


def record_request(messages: list[dict[str, Any]], response: Any) -> None:
    """Record the assistant turn that asked for tools."""
    messages.append(
        Message.ai(
            response.blocks or response.content,
            tool_calls=[
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in response.tool_calls
            ],
        ).to_dict()
    )


def render(result: Any, *, limit: int | None = None) -> tuple[str, bool]:
    """One tool outcome as the text the model sees, and whether it failed.

    Shared with the ToolFinished event rather than rendered twice, so a caller
    watching a run cannot be shown something the model was never sent.
    """
    failed = isinstance(result, Exception)
    content = f"Error: {result!r}" if failed else str(result)
    return truncate(content, limit), failed


def record_results(
    messages: list[dict[str, Any]],
    calls: list[Any],
    results: list[Any],
    *,
    limit: int | None = None,
) -> list[PendingHumanInput]:
    """Record one turn's tool outcomes, returning any that need a human.

    Tools are arbitrary user code, so the dispatcher catches broadly and hands
    the exception here as a value: a failing tool becomes an error message the
    model gets a turn to correct, instead of killing the run and taking the whole
    message history with it.
    """
    pending: list[PendingHumanInput] = []
    for call, result in zip(calls, results, strict=True):
        if isinstance(result, HumanInputRequired):
            pending.append(PendingHumanInput(call.id, call.name, result.question))
            continue
        content, _ = render(result, limit=limit)
        messages.append(
            Message.tool(content, name=call.name, call_id=call.id).to_dict()
        )
    return pending


@dataclass(slots=True)
class ApprovedCall:
    """A gated call the human allowed, replayed with the model's arguments.

    Shaped like a provider's ToolCall so the dispatcher cannot tell the
    difference between a fresh call and a replayed one.
    """

    id: str | None
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Ruling:
    """One turn's calls, split by what the run is allowed to do with them."""

    allowed: list[Any]
    paused: list[PendingHumanInput]
    denied: list[Any]
    asking: list[Any] = field(default_factory=list)
    """Gated calls the toolbox will put to its approver as it runs them."""


def rule(
    tools: Toolbox, calls: list[Any], permissions: Permissions | None = None
) -> Ruling:
    """Split a turn's calls into the ones to run, to ask about, and to refuse.

    Decided before dispatch rather than inside the tool, so a gated call cannot
    run by accident - and the model cannot route around the gate by declining to
    ask. A policy decides per call; without one, or for a call no rule matches,
    the tool's own requires_approval stands. A toolbox that can ask settles an
    "ask" itself, so the call goes to it rather than pausing the run.
    """
    allowed: list[Any] = []
    paused: list[PendingHumanInput] = []
    denied: list[Any] = []
    asking: list[Any] = []
    for call in calls:
        match _decide(tools, call, permissions):
            case "deny":
                denied.append(call)
            case "ask" if tools.can_ask:
                asking.append(call)
            case "ask":
                arguments = dict(call.arguments)
                paused.append(
                    PendingHumanInput(
                        call_id=call.id,
                        name=call.name,
                        question=f"Run {call.name} with {arguments}?",
                        arguments=arguments,
                    )
                )
            case _:
                allowed.append(call)
    return Ruling(allowed=allowed, paused=paused, denied=denied, asking=asking)


def _decide(tools: Toolbox, call: Any, permissions: Permissions | None) -> Decision:
    if permissions is not None:
        decision = permissions.decide(call.name, call.arguments)
        if decision is not None:
            return decision
    if call.name in tools and tools.get(call.name).requires_approval:
        return "ask"
    return "allow"


DENIED = "Denied by policy: this call is not permitted."
"""A permission rule refused the call outright."""

NOT_RUN = "Not run: the turn stopped for approval of another call."
"""The turn paused on a gated call, so this one was left unrun."""


def record_unrun(messages: list[dict[str, Any]], calls: list[Any], note: str) -> None:
    """Account for calls that were requested but never ran, and say why.

    Recorded rather than dropped for two reasons: the model needs to learn it
    cannot take that route, and a vendor rejects a transcript in which a
    requested call has no result at all - which is what a resumed run would
    otherwise send.
    """
    for call in calls:
        messages.append(Message.tool(note, name=call.name, call_id=call.id).to_dict())


def settle(
    state: AgentState, messages: list[dict[str, Any]], agent_name: str
) -> list[ApprovedCall]:
    """Apply the caller's rulings, returning the calls still to run.

    Rejections are recorded here as that call's result, so the model learns it
    was refused and can say so instead of retrying forever.
    """
    outstanding = [pending for pending in state.paused if pending.needs_approval]
    if not outstanding:
        return []
    if any(pending.approved is None for pending in outstanding):
        raise ConfigurationError(
            f"{agent_name} is paused on {outstanding[0].name}; call "
            f"state.approve() or state.reject() before running it again"
        )
    for pending in outstanding:
        if not pending.approved:
            messages.append(
                Message.tool(
                    "Denied by the user.",
                    name=pending.name,
                    call_id=pending.call_id,
                ).to_dict()
            )
    return [
        ApprovedCall(pending.call_id, pending.name, pending.arguments or {})
        for pending in outstanding
        if pending.approved
    ]
