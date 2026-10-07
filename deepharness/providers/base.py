from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any, Literal

from .content import Block, Text, Thinking, text_of


def token_usage(usage: Any) -> TokenUsage | None:
    """Normalize a vendor's parsed Usage into TokenUsage, if it sent one."""
    if usage is None:
        return None
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cached_tokens=usage.cached_tokens,
        cache_write_tokens=usage.cache_write_tokens,
    )


def without_none(payload: Any) -> dict[str, Any]:
    """A request payload as a dict, minus fields that were never set.

    Vendors do not treat an explicit null the same as an absent key - sending
    "tools": null where the API expects a list is an error at several of them -
    so unset optionals are dropped rather than serialized.
    """
    return {
        f.name: value
        for f in fields(payload)
        if (value := getattr(payload, f.name)) is not None
    }


class ReasoningLevel(StrEnum):
    """StrEnum so members serialize as plain strings (JSON payloads, dict
    keys) without extra conversion at the call sites."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def budget(self) -> int:
        """Anthropic and Gemini take a raw thinking-token budget, not an
        effort label; this is the token count each level maps to for them."""
        match self:
            case ReasoningLevel.LOW:
                return 1024
            case ReasoningLevel.MEDIUM:
                return 4096
            case ReasoningLevel.HIGH:
                return 16000


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the model.

    id is the vendor's identifier for this specific call (OpenAI's
    tool_calls[].id, Anthropic's tool_use block id) - carried through so the
    result can be linked back to it on the next turn. None for vendors with
    no such concept (Gemini correlates by name/position instead).
    """

    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass(slots=True)
class TokenUsage:
    """Token counts for one completion, normalized across vendors.

    cached_tokens is the part of prompt_tokens the vendor served from its prompt
    cache rather than recomputing, and cache_write_tokens is what it charged to
    put a prefix there. Both are reported rather than deducted: they are already
    inside prompt_tokens, and a cached token still occupies the context window
    even when it costs less - so Budget keeps counting the full figure.

    Both default to zero, so a vendor that does not cache, or does not say,
    reports nothing rather than a guess.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


FinishReason = Literal["stop", "length", "filtered", "other"]
"""Why the model stopped generating, normalized across vendors.

Anything but "stop" means the text is cut short: a vendor returns a partial
answer in the same shape as a whole one, so without this a caller cannot tell
half a sentence from a finished reply.
"""


@dataclass(slots=True)
class CompletionResponse:
    """Normalized result of a provider completion, independent of vendor format.

    `content` stays the turn's prose, because that is what almost every caller
    wants, and `blocks` carries everything the model actually returned -
    including the thinking a later turn may have to replay. The two are kept in
    step here rather than by each provider: give either one and the other is
    derived, so they cannot disagree.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    finish_reason: FinishReason = "stop"
    blocks: list[Block] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.blocks and self.content:
            self.blocks = [Text(self.content)]
        elif self.blocks and not self.content:
            self.content = text_of(self.blocks)

    @property
    def thinking(self) -> str:
        """The reasoning the model reported, if it reported any."""
        return "".join(
            block.text for block in self.blocks if isinstance(block, Thinking)
        )


@dataclass(slots=True)
class TextDelta:
    """A chunk of the model's prose, as it arrives."""

    text: str


@dataclass(slots=True)
class ThinkingDelta:
    """A chunk of the model's reasoning, as it arrives.

    Separate from TextDelta so a caller can show it differently, or not at all:
    run together with the answer it reads as one confused voice.
    """

    text: str


@dataclass(slots=True)
class Completed:
    """The whole turn, once the stream ends: text, tool calls and usage."""

    response: CompletionResponse


StreamEvent = TextDelta | ThinkingDelta | Completed
"""What a streaming call emits.

Text alone is not enough to drive an agent: a turn may ask for tools instead of
answering, and the vendor sends those in the same stream, fragmented. So a
stream yields deltas as they arrive and finishes with the assembled response.
"""


class LLM(ABC):
    """The interface agent/ and graph/ depend on: send messages, get a reply.

    Deliberately narrow and transport-agnostic. A provider does not have to
    speak HTTP - a local model, a fake for tests, or a queue-backed worker
    implements generate() and agenerate() and works everywhere; the streaming
    methods default to delivering the whole turn as one delta, so only a backend
    that really streams overrides them. Vendors that do speak HTTP share their
    request sequence through RestCompletions (see rest.py) rather than through
    this class.

    __slots__ is empty here rather than absent: a base class without it hands
    every subclass a __dict__, which would make the providers' own __slots__
    declarations save nothing.
    """

    __slots__ = ()

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        """Send messages and optional tool schemas; return a normalized response."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        """Synchronous counterpart to agenerate(), for use outside an event loop."""

    async def astream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a turn as TextDeltas, ending with a Completed.

        The default is the whole turn in one delta, because a backend that cannot
        stream still has to be usable here: callers - Agent included - then need
        one code path instead of two, and get the text either way rather than an
        error or an empty iterator. Providers that really stream override this.
        """
        response = await self.agenerate(messages, tools=tools)
        for event in as_deltas(response):
            yield event
        yield Completed(response)

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        """Synchronous counterpart to astream_events()."""
        response = self.generate(messages, tools=tools)
        yield from as_deltas(response)
        yield Completed(response)

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Just the text, for the common "print as it types" case."""
        async for event in self.astream_events(messages, tools=tools):
            if isinstance(event, TextDelta):
                yield event.text

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """Synchronous counterpart to astream()."""
        for event in self.stream_events(messages, tools=tools):
            if isinstance(event, TextDelta):
                yield event.text


def as_deltas(response: CompletionResponse) -> list[TextDelta | ThinkingDelta]:
    """One whole turn as the deltas a streaming turn would have emitted.

    For a backend that cannot stream: its callers still get the same event
    sequence, in block order, rather than a special case of their own.
    """
    events: list[TextDelta | ThinkingDelta] = []
    for block in response.blocks:
        if isinstance(block, Thinking) and block.text:
            events.append(ThinkingDelta(block.text))
        elif isinstance(block, Text) and block.text:
            events.append(TextDelta(block.text))
    return events
