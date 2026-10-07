from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from deepharness.providers.base import (
    LLM,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
)

from ..errors import (
    ConfigurationError,
    HumanInputRequired,
    OutputValidationError,
    TokenBudgetExceeded,
)
from ..tools.permissions import Permissions
from ..tools.toolbox import Ctx, Toolbox, ToolSpec
from . import turn
from .context import ContextPolicy
from .events import StepStarted, ToolFinished, ToolStarted
from .output import FINAL_TOOL, coerce, final_tool_schema, find_final
from .state import (
    AgentState,
    Budget,
    Finished,
    Message,
    PendingHumanInput,
    StopReason,
)

AgentEvent = (
    TextDelta | ThinkingDelta | StepStarted | ToolStarted | ToolFinished | Finished
)
"""What streaming a run emits: the run's progress, then the final state.

Prose arrives as TextDelta, reasoning as ThinkingDelta, and the rest is what the
loop is doing between those - a step beginning, a tool starting and finishing -
ending with the one Finished that carries the AgentState. A caller interested in
text alone wants astream() and never sees these."""


@dataclass(slots=True)
class _Ask:
    """A model call the loop needs its driver to perform."""

    messages: list[dict[str, Any]]


@dataclass(slots=True)
class _Dispatch:
    """One turn's tool calls, for the driver to run however it runs them."""

    calls: list[Any]
    ask: list[bool]
    """Per call, whether the toolbox puts it to its approver first."""


class Agent:
    """A model plus tools, run as a think/act loop over an AgentState.

    Ask the model, dispatch whatever tools it requests, repeat until it answers
    or the budget runs out. Without a model the run is a passthrough, which is
    what makes an Agent usable as a placeholder node in a Graph.

    The parts worth knowing before reading the loop:

    * Only stop_reason == "answer" means the model actually replied. Every other
      reason leaves output empty or partial, so state.answered is the check to
      make before trusting it.
    * run() is a real synchronous path, not arun() wrapped in an event loop, so
      it raises for a tool that turns out to be `async def`. arun() dispatches a
      turn's tools concurrently, sync ones included - those go to threads.
    * A failing tool does not end the run: the error becomes that call's result
      so the model gets a turn to correct itself.
    * total_usage accumulates across every model call this instance makes, not
      per run, and Budget(tokens=...) turns crossing it into TokenBudgetExceeded
      with the partial state attached.
    * Permissions decides per call what may run, what needs a human and what
      is refused outright; a call no rule matches falls back to the tool's own
      requires_approval flag. A deny or ask rule naming a tool that is not
      registered is refused at construction, because a misspelled one matches
      nothing and fails open.
    * ContextPolicy bounds what the transcript costs: each tool result is
      truncated as it is recorded, and the model is sent a pruned view while
      state.messages keeps every message.
    * Every public entry point - arun, run, astream, stream - is the same loop;
      only the I/O differs. See astream_events for the one async driver.

    Pausing has two flavours, and they resolve differently: a tool marked
    requires_approval has not run yet (approve it and it runs), while a tool
    raising HumanInputRequired is asking a question (your answer becomes its
    result). See deepharness/agent/turn.py. A Toolbox given approve= asks
    about gated calls itself, so the run carries on instead of pausing.
    """

    __slots__ = (
        "_budget",
        "_context",
        "_final_schema",
        "_model",
        "_name",
        "_output",
        "_permissions",
        "_system",
        "_tools",
        "_total_usage",
    )

    def __init__(
        self,
        model: LLM | None = None,
        *,
        tools: Iterable[Callable[..., Any]] | Toolbox = (),
        system: str | None = None,
        name: str = "agent",
        budget: Budget | None = None,
        context: ContextPolicy | None = None,
        permissions: Permissions | None = None,
        output: type | None = None,
    ):
        self._model = model
        self._tools = tools if isinstance(tools, Toolbox) else Toolbox(tools)
        self._system = system
        self._name = name
        self._budget = budget or Budget()
        self._context = context or ContextPolicy()
        self._permissions = permissions
        _check_rules(self._tools, permissions)
        self._output = output
        self._final_schema = final_tool_schema(output) if output is not None else None
        self._total_usage = TokenUsage(0, 0, 0)

    # Read-only views: an agent's configuration is settled at construction, and
    # total_usage is live state no caller should be able to reset or inflate.

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> LLM | None:
        return self._model

    @property
    def tools(self) -> Toolbox:
        return self._tools

    @property
    def system(self) -> str | None:
        return self._system

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def context(self) -> ContextPolicy:
        return self._context

    @property
    def permissions(self) -> Permissions | None:
        """What the run may do without asking; None leaves it to each tool."""
        return self._permissions

    @property
    def output(self) -> type | None:
        return self._output

    @property
    def total_usage(self) -> TokenUsage:
        """Cumulative usage across every model call this agent has made."""
        return self._total_usage

    def as_tool(
        self, *, name: str | None = None, description: str | None = None
    ) -> Callable[..., Any]:
        """Wrap this Agent as a tool callable from another Agent's toolbox.

        The wrapped tool takes a single `input` string, runs it through
        arun() as a user message, and returns the resulting output text.
        It's async, so register it with an Agent that calls arun() -
        call_sync() raises ConfigurationError for async tools, same as any
        other async tool.

        The sub-agent inherits the caller's deps, so a delegated run keeps the
        database handle or tenant the parent was given.
        """

        async def call(input: str, ctx: Ctx) -> str:
            result = await self.arun(input, deps=ctx.deps)
            return result.output

        call.__name__ = name or self._name
        call._tool_spec = ToolSpec(  # type: ignore[attr-defined]
            name=name or self._name,
            description=description
            or self._system
            or f"Delegate a task to the '{self._name}' agent.",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
            func=call,
            ctx_params=("ctx",),
        )
        return call

    def _account_for_usage(
        self,
        response: Any,
        state: AgentState,
        messages: list[dict[str, Any]],
    ) -> None:
        if response.usage is None:
            return

        self._total_usage = self._total_usage + response.usage
        if (
            self._budget.tokens is not None
            and self._total_usage.total_tokens > self._budget.tokens
        ):
            raise TokenBudgetExceeded(
                self._name,
                self._total_usage,
                self._budget.tokens,
                state=self._result(state, messages, response.content, "token_budget"),
            )

    def _result(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        output: Any,
        stop_reason: StopReason,
        paused: list[PendingHumanInput] | None = None,
    ) -> AgentState:
        return AgentState(
            messages=messages,
            output=output,
            usage=self._total_usage,
            stop_reason=stop_reason,
            paused=paused or [],
        )

    def _turns(
        self, state: AgentState, messages: list[dict[str, Any]]
    ) -> Generator[_Ask | _Dispatch, Any, AgentState]:
        """The think/act loop itself, with the I/O lifted out of it.

        The loop yields the work it needs done - a model call, or a round of
        tool calls - and its driver performs that work and sends the result
        back. run() can then stay a genuinely synchronous path and arun() a
        concurrent one, while everything they agree on (the step budget, stop
        reasons, pausing for a human, feeding a failed tool back to the model)
        lives here once instead of being maintained in two copies.

        A run that starts from a paused state settles the approvals first: the
        allowed calls run now, with the arguments the model originally sent, and
        only then does the loop go back to the model with their results.
        """
        approved = turn.settle(state, messages, self._name)
        if approved:
            # A human already approved these; the toolbox must not ask again.
            results = yield _Dispatch(approved, [False] * len(approved))
            turn.record_results(
                messages, approved, results, limit=self._context.tool_result_chars
            )

        for _ in range(self._budget.steps):
            response = yield _Ask(messages)
            self._account_for_usage(response, state, messages)

            final = find_final(response) if self._final_schema else None
            if final is not None:
                turn.record_request(messages, response)
                try:
                    answer = coerce(self._output, final.arguments)
                except OutputValidationError as exc:
                    # Same courtesy a failing tool gets: hand the model the
                    # error so it can call FINAL_TOOL again with valid fields.
                    messages.append(
                        Message.tool(
                            f"Error: {exc}", name=FINAL_TOOL, call_id=final.id
                        ).to_dict()
                    )
                    continue
                return self._result(state, messages, answer, "answer")

            if not response.tool_calls:
                messages.append(
                    Message.ai(response.blocks or response.content).to_dict()
                )
                if self._final_schema is not None:
                    # output= was asked for, so plain prose is not an answer yet.
                    messages.append(
                        Message.human(f"Answer by calling {FINAL_TOOL}.").to_dict()
                    )
                    continue
                cut_short = response.finish_reason != "stop"
                return self._result(
                    state,
                    messages,
                    response.content,
                    "truncated" if cut_short else "answer",
                )

            if not self._tools:
                raise ConfigurationError(
                    f"{self._name} received tool calls but has no tools registered"
                )

            turn.record_request(messages, response)
            wanted = [call for call in response.tool_calls if call.name != FINAL_TOOL]
            ruling = turn.rule(self._tools, wanted, self._permissions)
            turn.record_unrun(messages, ruling.denied, turn.DENIED)
            if ruling.paused:
                # Nothing in this turn runs until the human rules on the gated
                # call: letting the rest run first would half-apply a turn the
                # human may be about to refuse.
                turn.record_unrun(messages, ruling.allowed, turn.NOT_RUN)
                return self._result(state, messages, "", "paused", paused=ruling.paused)

            if ruling.allowed:
                results = yield _Dispatch(ruling.allowed, ruling.ask)
                pending = turn.record_results(
                    messages,
                    ruling.allowed,
                    results,
                    limit=self._context.tool_result_chars,
                )
                if pending:
                    return self._result(state, messages, "", "paused", paused=pending)

        return self._result(state, messages, "", "step_budget")

    def _passthrough(self, state: AgentState) -> AgentState:
        """Without a model an Agent is inert - a placeholder node in a Graph.

        Returns the state untouched rather than announcing itself: printing from
        inside the agent would make this path untestable without capturing
        stdout, and a library has no business writing to a caller's console.
        """
        return state

    def _schemas(self) -> list[dict[str, Any]] | None:
        schemas = self._tools.schemas()
        if self._final_schema is not None:
            schemas.append(self._final_schema)
        return schemas or None

    async def arun(self, state: Any = None, *, deps: Any = None) -> AgentState:
        """Run to completion. Tool calls in the same turn dispatch concurrently."""
        async for event in self.astream_events(state, deps=deps):
            if isinstance(event, Finished):
                return event.state
        raise AssertionError("a run always ends with Finished")  # pragma: no cover

    def run(self, state: Any = None, *, deps: Any = None) -> AgentState:
        """Synchronous counterpart to arun(). Raises if a tool is `async def`."""
        for event in self.stream_events(state, deps=deps):
            if isinstance(event, Finished):
                return event.state
        raise AssertionError("a run always ends with Finished")  # pragma: no cover

    async def astream_events(
        self, state: Any = None, *, deps: Any = None
    ) -> AsyncIterator[AgentEvent]:
        """Drive one run, emitting prose as it arrives then the final state.

        The only async driver: arun() consumes this and keeps the last event, so
        the loop's mechanics - budget, approvals, tool dispatch - exist once
        rather than once per public method. Providers that cannot really stream
        still work here; their turn simply arrives as a single delta.
        """
        state = AgentState.of(state)
        if self._model is None:
            yield Finished(self._passthrough(state))
            return

        ctx = Ctx(state=state, deps=deps)
        turns = self._turns(state, turn.prepare(state, self._system))
        schemas = self._schemas()
        outcome: Any = None
        step = 0
        try:
            while True:
                request = turns.send(outcome)
                if isinstance(request, _Ask):
                    step += 1
                    yield StepStarted(step)
                    async for event in self._model.astream_events(
                        self._context.prune(request.messages), tools=schemas
                    ):
                        if isinstance(event, TextDelta | ThinkingDelta):
                            yield event
                        else:
                            outcome = event.response
                else:
                    for call in request.calls:
                        yield ToolStarted(call.name, call.arguments, call.id)
                    outcome = await asyncio.gather(
                        *(
                            self._call_tool(call.name, call.arguments, ctx, ask)
                            for call, ask in zip(
                                request.calls, request.ask, strict=True
                            )
                        )
                    )
                    for call, result in zip(request.calls, outcome, strict=True):
                        if (finished := self._finished(call, result)) is not None:
                            yield finished
        except StopIteration as done:
            yield Finished(done.value)

    def stream_events(
        self, state: Any = None, *, deps: Any = None
    ) -> Iterator[AgentEvent]:
        """Synchronous counterpart to astream_events()."""
        state = AgentState.of(state)
        if self._model is None:
            yield Finished(self._passthrough(state))
            return

        ctx = Ctx(state=state, deps=deps)
        turns = self._turns(state, turn.prepare(state, self._system))
        schemas = self._schemas()
        outcome: Any = None
        step = 0
        try:
            while True:
                request = turns.send(outcome)
                if isinstance(request, _Ask):
                    step += 1
                    yield StepStarted(step)
                    for event in self._model.stream_events(
                        self._context.prune(request.messages), tools=schemas
                    ):
                        if isinstance(event, TextDelta | ThinkingDelta):
                            yield event
                        else:
                            outcome = event.response
                else:
                    results: list[Any] = []
                    for call, ask in zip(request.calls, request.ask, strict=True):
                        yield ToolStarted(call.name, call.arguments, call.id)
                        result = self._call_tool_sync(
                            call.name, call.arguments, ctx, ask
                        )
                        results.append(result)
                        if (finished := self._finished(call, result)) is not None:
                            yield finished
                    outcome = results
        except StopIteration as done:
            yield Finished(done.value)

    async def astream(
        self, state: Any = None, *, deps: Any = None
    ) -> AsyncIterator[str]:
        """Just the text, for the common "print as it types" case."""
        async for event in self.astream_events(state, deps=deps):
            if isinstance(event, TextDelta):
                yield event.text

    def stream(self, state: Any = None, *, deps: Any = None) -> Iterator[str]:
        """Synchronous counterpart to astream()."""
        for event in self.stream_events(state, deps=deps):
            if isinstance(event, TextDelta):
                yield event.text

    def _finished(self, call: Any, result: Any) -> ToolFinished | None:
        """How one dispatched call ended, or None if it is not over.

        A tool that asked a human has produced no result yet - the run is about
        to pause on it - so it gets no event rather than one reporting its own
        question as an error.
        """
        if isinstance(result, HumanInputRequired):
            return None
        content, failed = turn.render(result, limit=self._context.tool_result_chars)
        return ToolFinished(call.name, content, failed, call.id)

    async def _call_tool(
        self, name: str, arguments: dict[str, Any], ctx: Ctx, ask: bool = False
    ) -> Any:
        # ask is explicit either way: the permission rules have already decided
        # this call, and an allow rule must not be re-asked by the toolbox.
        try:
            return await self._tools.call(name, ctx=ctx, ask=ask, **arguments)
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - see turn.record_results
            return exc

    def _call_tool_sync(
        self, name: str, arguments: dict[str, Any], ctx: Ctx, ask: bool = False
    ) -> Any:
        try:
            return self._tools.call_sync(name, ctx=ctx, ask=ask, **arguments)
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - see turn.record_results
            return exc


def _check_rules(tools: Toolbox, permissions: Permissions | None) -> None:
    """Refuse a gating rule that names a tool this agent does not have.

    A misspelled rule matches no call at all, which for a deny rule means the
    thing it was written to forbid runs. Silence is the worst outcome here, so
    it is an error at construction rather than a surprise at runtime.

    Only the gating rules are checked - see Permissions.gates - and pattern
    rules are left alone, since they are meant not to name one tool.
    """
    if permissions is None or not tools:
        return
    unknown = sorted(
        {
            rule.tool
            for rule in permissions.gates
            if not rule.is_pattern and rule.tool not in tools
        }
    )
    if unknown:
        known = ", ".join(sorted(tools.names())) or "none"
        raise ConfigurationError(
            f"permission rules deny or ask about unregistered tools: "
            f"{', '.join(unknown)}. Registered tools: {known}. A rule that "
            f"names no tool matches no call, so it would silently allow what it "
            f"was written to stop"
        )
