# API reference

## Imports

A first program's names come from the root; everything else comes from the layer that owns it,
so an import path says which part of the library a name belongs to.

```python
from deepharness import Agent, Graph, Message, OpenAI, tool  # ~27 high-level names
from deepharness.agent import ContextPolicy, StepStarted
from deepharness.graph import concat, merge_dicts
from deepharness.providers import Caching, Fallback, Image, Text
from deepharness.tools import Permissions, ToolName, file_tools, shell_tool
from deepharness.prebuilt import DeepResearch
from deepharness.errors import ProviderError
```

| Module | Holds |
| --- | --- |
| `deepharness` | `Agent`, `AgentState`, `Budget`, `Message`, `Toolbox`, `Ctx`, `tool`, `Finished`, `TextDelta`, `Graph`, `Executor`, `DeepHarnessError`, and all 15 providers |
| `deepharness.agent` | the loop's own types — `ContextPolicy`, `estimate_tokens`, the progress events, `StopReason`, `PendingHumanInput`, `ToolSpec`, `FINAL_TOOL` |
| `deepharness.graph` | `NodeSpec`, `concat`, `merge_dicts` |
| `deepharness.providers` | `LLM` and the wire types, content blocks (`Text`, `Image`, `Document`, `Thinking`), `ReasoningLevel`, and the wrappers `Caching`/`Fallback`/`RateLimited`/`Retrying`/`Wrapping` |
| `deepharness.tools` | `Permissions`, `Rule`, `ToolName`, `Workspace`, `file_tools`, `shell_tool`, `MCPServer`, `Transport`, `TavilySearch` |
| `deepharness.prebuilt` | `DeepResearch`, `Finding`, `ResearchResult` and its events |
| `deepharness.errors` | every exception; only `DeepHarnessError` is re-exported at the root |

The root is kept small on purpose: each name there is a promise about stability, and a flat
namespace of everything stops being discoverable long before it is complete.

The public surface, importable from `deepharness` unless noted otherwise. This page covers
signatures and behavior only — see the [guide](../guide/agents.md) for narrative
explanations and examples.

## Agents

### `Agent`

```python
Agent(
    model: LLM | None = None,
    *,
    tools: Iterable[Callable] | Toolbox = (),
    system: str | None = None,
    name: str = "agent",
    budget: Budget | None = None,
    context: ContextPolicy | None = None,
    permissions: Permissions | None = None,
    output: type | None = None,
)
```

Runs a think/act loop against `model`: request a completion, dispatch any requested tool
calls, repeat until the model stops calling tools or the budget's step limit is reached.

| Member | Signature | Description |
| --- | --- | --- |
| `arun` | `async def arun(state: Any = None, *, deps: Any = None) -> AgentState` | Async run. Tool calls in the same turn dispatch concurrently. |
| `run` | `def run(state: Any = None, *, deps: Any = None) -> AgentState` | Sync run. Raises if a registered tool is `async def`. |
| `astream` | `async def astream(state=None, *, deps=None) -> AsyncIterator[str]` | Text deltas as they arrive; tools still dispatch. |
| `astream_events` | `async def astream_events(...) -> AsyncIterator[AgentEvent]` | Text deltas, progress events and a final `Finished(state)`. |
| `stream` / `stream_events` | sync counterparts | Same, outside an event loop. |
| `total_usage` | `TokenUsage` | Cumulative token usage across every call made by this agent instance. Read-only. |
| `budget` | `Budget` | The run's limits; defaults to `Budget()` when none is passed. Read-only. |
| `context` | `ContextPolicy` | What the transcript may cost; defaults to `ContextPolicy()`. Read-only. |
| `permissions` | `Permissions \| None` | Per-call policy; `None` leaves each call to the tool's own `requires_approval`. Read-only. |
| `tools` | `Toolbox` | Always a `Toolbox` — an iterable passed as `tools=` is wrapped in one. Read-only. |
| `output` | `type \| None` | A dataclass; when set, `state.output` is a validated instance of it. |

### `AgentState`

```python
AgentState(
    messages: list[dict] = [],
    output: Any = None,
    usage: TokenUsage = TokenUsage(0, 0, 0),
    stop_reason: StopReason | None = None,
    paused: list[PendingHumanInput] = [],
)
```

What a run consumed and produced. `stop_reason` is one of `"answer"`, `"step_budget"`,
`"paused"`, `"token_budget"` or `"truncated"`. `answered` is `True` only when
`stop_reason == "answer"`.
`AgentState.of(value)` builds one from a prompt string, a list of messages, a dict of known
fields, or an existing state; an unknown dict key raises `ConfigurationError`.
`to_dict()`/`from_dict(data)` round-trip the whole state as JSON-able data, including any
call paused on an approval, so a run can be stored wherever the caller likes and resumed in
another process. Structured `output` is stored as plain data and returns as a dict.

`approve(call_id=None)` / `reject(call_id=None)` rule on calls waiting in `paused`, returning the
state so a resume reads as `await agent.arun(state.approve())`. Both raise `ConfigurationError`
when nothing matches.

### `PendingHumanInput`

```python
PendingHumanInput(
    call_id: str | None,
    name: str,
    question: str,
    arguments: dict | None = None,
    approved: bool | None = None,
)
```

One paused call. `needs_approval` is `True` when it came from a `requires_approval` tool — those
have not run yet and carry the `arguments` to run with. Otherwise the pause came from a tool
raising `HumanInputRequired`, and the human's answer becomes that call's result.

### `Budget`

```python
Budget(steps: int = 10, tokens: int | None = None)
```

Frozen dataclass bounding one run. `steps` caps think/act turns — spending them stops the run
with a `"step_budget"` stop reason rather than raising, since the run is truncated but already
paid for. `tokens` caps cumulative usage and raises `TokenBudgetExceeded` when crossed.
`Budget(steps=1)` makes an agent single-shot: one model call, and no turn to react to a tool
result. Non-positive values raise `ConfigurationError`.

`state` is a dict with a `messages` key (a list of [`Message`](#message)/dict). The returned
dict adds `output` (the model's final text) and `usage` (a `TokenUsage`).

Without a `model`, `run`/`arun` are a no-op passthrough — useful as a placeholder while
wiring a graph.

### `ContextPolicy`

```python
ContextPolicy(
    max_tokens: int | None = None,
    tool_result_chars: int | None = 8000,
    keep_last: int = 4,
)
```

Frozen dataclass bounding what a run sends. `tool_result_chars` truncates each tool result as
it is recorded, eliding the middle. `max_tokens` prunes the transcript the model is sent —
oldest turns first, with a note in their place — while `state.messages` keeps every message;
it is `None` (no pruning) by default. Pruning never drops the leading system prompt, never
orphans a tool result, and always keeps the last `keep_last` messages, so a transcript whose
tail alone exceeds the budget is sent over it. Non-positive values raise `ConfigurationError`.

| Member | Signature | Description |
| --- | --- | --- |
| `prune` | `def prune(messages: list[dict]) -> list[dict]` | The transcript as it should be sent; returns the list unchanged when it fits. Override to prune differently. |

### `estimate_tokens`

```python
estimate_tokens(messages: list[dict]) -> int
```

Approximate token cost of a transcript, at roughly four characters per token. A heuristic, not
a tokenizer — a real count would mean a per-vendor dependency.

### `AgentEvent`

```python
AgentEvent = TextDelta | StepStarted | ToolStarted | ToolFinished | Finished
```

What `astream_events()`/`stream_events()` emit.

| Event | Fields | Emitted |
| --- | --- | --- |
| `StepStarted` | `step: int` | Before each model call; 1-based, capped by `Budget.steps`. |
| `ToolStarted` | `name: str`, `arguments: dict`, `call_id: str \| None` | Before a tool runs, with the arguments the model sent. |
| `ToolFinished` | `name: str`, `result: str`, `failed: bool`, `call_id: str \| None` | After a tool returns or raises. `result` is the (truncated) text the model will read. Not emitted for a tool that asked a human — it has no result yet. |
| `TextDelta` | `text: str` | As the model's prose arrives. |
| `ThinkingDelta` | `text: str` | As the model's reasoning arrives, kept apart from the answer. |
| `Finished` | `state: AgentState` | Once, last, carrying the run's result. |

### `TokenBudgetExceeded`

```python
TokenBudgetExceeded(agent_name: str, usage: TokenUsage, budget: int)
```

Raised by `Agent` the moment cumulative `total_usage` crosses `Budget.tokens`, checked right
after a model response — before any further tool dispatch or model call.

## Messages & sessions

### `Message`

A `dict` subclass — every constructor below returns a plain `{"role": ..., "content": ...}`
style dict, so it's interchangeable with hand-built message dicts anywhere one is expected.

| Constructor | Produces |
| --- | --- |
| `Message.system(content: str)` | `{"role": "system", "content": ...}` |
| `Message.human(content: str)` | `{"role": "user", "content": ...}` |
| `Message.ai(content: str, *, tool_calls: list[dict] \| None = None)` | `{"role": "assistant", "content": ...}` |
| `Message.tool(content: str, *, name: str, call_id: str \| None = None)` | `{"role": "tool", "name": ..., "content": ...}` |

## Tools

### `tool`

```python
@tool
@tool(
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
)
```

Decorates a function so it can be registered as a callable tool. Builds a JSON schema from
the function's signature (parameter types, required-ness) and docstring (summary plus
`Args:`/`:param:` descriptions). Handles containers, `Literal`, `Enum` and unions — see [Tools](../guide/tools.md). Works on
both sync and async functions. `requires_approval=True` makes the agent pause before every call
to it and run it only once approved.

### `Ctx`

```python
Ctx(state: Any = None, deps: Any = None)
```

Injected into any tool parameter annotated `Ctx`, and hidden from that tool's schema. `deps` is
whatever the run was given; `state` is the run's `AgentState`.

### `Toolbox`

```python
Toolbox(tools: Iterable[Callable] = ())
```

| Method | Signature | Description |
| --- | --- | --- |
| `register` | `def register(func: Callable) -> Callable` | Registers a function as a tool. |
| `get` | `def get(name: str) -> ToolSpec` | Looks up a registered tool by name. |
| `schemas` | `def schemas() -> list[dict]` | Returns tool schemas, ready to pass to a provider. |
| `call` | `async def call(name: str, **kwargs) -> Any` | Invokes a tool by name, awaiting it if async. |
| `call_sync` | `def call_sync(name: str, **kwargs) -> Any` | Invokes a tool synchronously; raises if it's async. |

### `Workspace`

```python
Workspace(root: str | Path = ".")
```

The one directory a tool may touch. `resolve(path)` returns an absolute path inside the root
or raises `OutsideWorkspace`; resolution is symlink-aware, so a link inside the root pointing
out of it is refused. `relative(path)` renders a resolved path as the model should see it. A
root that is not an existing directory raises `ConfigurationError`.

### `file_tools`

```python
file_tools(
    workspace: Workspace | str | Path = ".",
    *,
    max_bytes: int = 200_000,
    max_matches: int = 200,
    writable: bool = True,
) -> list[Callable]
```

Filesystem tools confined to one workspace: `read_file`, `list_files`, `search_files`, and —
unless `writable=False` — `write_file` and `edit_file`, both gated with
`requires_approval=True`. `read_file` numbers lines and stops at `max_bytes`, naming the offset
to continue from. `edit_file` refuses an `old` string that occurs more than once.

### `shell_tool`

```python
shell_tool(
    workspace: Workspace | str | Path = ".",
    *,
    timeout: int = 60,
    max_chars: int = 30_000,
    requires_approval: bool = True,
) -> Callable
```

A `run_command` tool that runs a shell command with the workspace as its working directory,
reporting stdout, then stderr, then a non-zero exit code. Gated by default, and not a sandbox:
the workspace bounds where a command starts, not what it can reach.

### `MCPServer` / `MCPTool`

```python
MCPServer.stdio(command: str | Iterable[str], *, env=None, requires_approval=True)
MCPServer.http(url: str, *, headers=None, client=None, requires_approval=True)
MCPServer(transport: Transport, *, requires_approval: bool = True)
```

A Model Context Protocol server's tools as callables. The handshake runs on first use;
`async with` closes the transport.

| Member | Signature | Description |
| --- | --- | --- |
| `tools` | `async def tools() -> list[Callable]` | The server's tools, ready for `Agent(tools=...)`. Async, so use `arun()`. |
| `list_tools` | `async def list_tools() -> list[MCPTool]` | The same tools as data: `name`, `description`, `input_schema`, `read_only`. |
| `call` | `async def call(name, arguments) -> str` | Run one tool; raises `MCPError` if the server reports failure. |
| `connect` / `aclose` | `async def ...() -> None` | Handshake and shutdown; both are idempotent. |

A tool the server did not mark read-only is gated with `requires_approval=True`. A
protocol-level failure raises `MCPError`.

`Transport` is the seam — `request(method, params)`, `notify(method, params)`, `aclose()` —
with two implementations: `StdioTransport` (a subprocess speaking newline-delimited JSON,
reading past any notifications to its own reply) and `HTTPTransport` (streamable HTTP, JSON or
SSE replies, carrying any `Mcp-Session-Id`). The deprecated `2024-11-05` HTTP+SSE transport and
OAuth are not implemented, and only MCP's tools are — not resources, prompts or sampling.
Implement `Transport` yourself and pass it to `MCPServer(transport)` to cover a server these
two do not reach.

### `Permissions` / `Rule`

```python
Permissions(
    *,
    allow: Iterable[RuleLike] = (),
    ask: Iterable[RuleLike] = (),
    deny: Iterable[RuleLike] = (),
)
Rule(tool: str, arguments: dict[str, str] | None = None)

RuleLike = str | Rule | Callable[..., Any]   # a name or pattern, a Rule, or the tool itself
```

Decides per call what may run. `decide(name, arguments) -> "allow" | "ask" | "deny" | None`,
where `None` means no rule applied and the tool's own `requires_approval` stands. `deny` beats
`allow` beats `ask`.

A rule may be written three ways: the decorated tool itself (read for the name it is
registered under), a `ToolName` member, or a string name or `fnmatch` pattern.
`Rule` narrows by argument patterns, also `fnmatch`; an argument a rule mentions but the call
omits does not match. `Rule.is_pattern` says whether a rule names a tool or a shape.

`Permissions.rules` is every rule; `Permissions.gates` is the deny and ask rules — the ones
whose silence would be unsafe. `Agent` refuses at construction when a gate names a tool it has
not registered, since a misspelled deny rule matches nothing and allows what it was written to
stop. Allow rules and pattern rules are not checked.

### `ToolName`

```python
ToolName.READ_FILE  # "read_file"; also LIST_FILES, SEARCH_FILES, WRITE_FILE,
ToolName.RUN_COMMAND  # "run_command"                             EDIT_FILE
```

The names the built-in tools register under, as one `StrEnum` — a member is a `str`, so it
works anywhere a name or a `Rule`'s tool is expected. One enum rather than one per module, so
nothing sits a letter of case away from the factory that builds the tools.

## Graphs & execution

### `Graph`

```python
Graph(state_type: type)
```

| Method | Signature | Description |
| --- | --- | --- |
| `add` | `def add(*, name: str \| None = None, start: bool = False, end: bool = False)` | Decorator that registers a function as a node. |
| `connect` | `def connect(source: Callable \| str, target: Callable \| str, *, condition: Callable[[Any], bool] \| None = None, loop: bool = False)` | Declares an edge, optionally conditional. `loop=True` marks a back-edge, re-running the loop head and everything downstream of it. |
| `build` | `def build() -> Executor` | Validates the graph (has a start node, fully reachable, no cycle whose back-edge is unmarked) and returns an `Executor`. |

### `Executor`

```python
async def run(state: Any = None, *, max_steps: int = 50) -> Any
def diagram() -> str
```

Omitting `state` builds one from the type the `Graph` was declared with, so a state whose
fields all have defaults needs no argument. Runs the graph wave by wave: each wave's ready
nodes execute concurrently, results merge back
into the state field-by-field, and the next wave is whichever nodes now have all predecessors
satisfied. A field written by two or more concurrent branches is combined by its
[reducer](../guide/graph.md#reducers), or raises `ConcurrentUpdateError` if it declares none.
Taking a `loop=True` edge re-runs the loop head and everything downstream of it; exceeding
`max_steps` raises `StepLimitExceeded`. Returned by `Graph.build()` — not constructed
directly.

`diagram()` returns a box-drawing picture of the graph as text — layers top to bottom by
wave, `═` for start and end nodes, `▽` for a conditional edge, and back-edges routed up the
right margin. It returns the string rather than printing it. See
[Seeing the shape](../guide/graph.md#seeing-the-shape).

### `concat` / `merge_dicts`

```python
concat(base: list, values: list[list]) -> list
merge_dicts(base: dict, values: list[dict]) -> dict
```

Built-in reducers for merging concurrent writes to one field. Declare one with
`field(metadata={"reducer": concat})`.

### `ExecutionError`

```python
ExecutionError(node_name: str, original: Exception)
```

Raised by `Executor.run` when a node function raises; wraps the original exception with the
name of the node that failed.

## Providers

Every provider below implements the same interface:

| Method | Returns |
| --- | --- |
| `await agenerate(messages: list[dict], *, tools: list[dict] \| None = None)` | `CompletionResponse` |
| `generate(messages: list[dict], *, tools: list[dict] \| None = None)` | `CompletionResponse` |
| `async for event in astream_events(messages, *, tools=None)` | `TextDelta`, then `Completed(response)` |
| `for event in stream_events(messages, *, tools=None)` | `TextDelta`, then `Completed(response)` |
| `async for chunk in astream(messages, *, tools=None)` | text deltas (filters the above) |
| `for chunk in stream(messages, *, tools=None)` | text deltas |

A streaming turn ends with `Completed`, carrying the assembled `CompletionResponse` — tool
calls included, reassembled from however the vendor fragmented them. `agenerate`/`generate` are
the required pair — implement those two and a custom provider works
everywhere, transport regardless. The streaming pair is optional: the base class raises
`NotImplementedError` naming the provider, so a backend that cannot stream needs no stub.
Vendors that speak REST share their request sequence through `RestCompletions` rather than by
inheriting it (see `providers/rest.py`).

### `TokenUsage`

```python
TokenUsage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
)
```

`cached_tokens` is the part of `prompt_tokens` the vendor served from its prompt cache;
`cache_write_tokens` is what it charged to put a prefix there. Both are **inside**
`prompt_tokens`, reported rather than deducted — a cached token still occupies the context
window, so `Budget(tokens=…)` keeps counting the full figure. Read from
`prompt_tokens_details.cached_tokens` (OpenAI, DeepSeek, llama.cpp),
`cache_read_input_tokens` / `cache_creation_input_tokens` (Anthropic) and
`cachedContentTokenCount` (Gemini); zero when a vendor does not report them. `+` sums every
field, which is how `agent.total_usage` accumulates.

### Content blocks

```python
Text(text: str)
Image(data: str | None = None, media_type: str = "image/png", url: str | None = None)
Document(data: str, media_type: str = "application/pdf", name: str | None = None)
Thinking(text: str, signature: str | None = None)

Block = Text | Image | Document | Thinking
Content = str | list[Block]
```

What a message's `content` may be. `Image.from_path(path)` / `Document.from_path(path)` encode
a local file and infer its media type; `Image.from_url(url)` passes a URL through. An `Image`
with both `data` and `url`, or neither, raises `ConfigurationError`, as does a suffix no media
type is known for. Providers render blocks into their own wire shape; Gemini rejects an image
URL it cannot send. Text-only content is still sent — and saved — as a plain string.

`Thinking.signature` is Anthropic's attestation, replayed unmodified with the turn because
Anthropic requires it back after a tool call; an unsigned thinking block is left out rather
than sent.

### Response types

```python
ToolCall(name: str, arguments: dict, id: str | None = None)
TokenUsage(prompt_tokens: int, completion_tokens: int, total_tokens: int)
CompletionResponse(content: str, tool_calls: list[ToolCall] = [], usage: TokenUsage | None = None)
```

### Direct providers

| Class | Signature |
| --- | --- |
| `Anthropic` | `Anthropic(model: str, api_key: str \| None = None, max_tokens: int = 4096, *, reasoning_effort=None, cache_prompt: bool = False, max_concurrency: int \| None = None)` |
| `OpenAI` | `OpenAI(model: str, api_key: str \| None = None, *, base_url: str \| None = None, temperature: float \| None = None, reasoning_effort=None, stream_usage: bool = True, max_concurrency: int \| None = None)` |
| `Gemini` | `Gemini(model: str, api_key: str \| None = None, *, reasoning_effort=None, max_concurrency: int \| None = None)` |

`api_key` falls back to the vendor's standard environment variable when omitted:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

`max_concurrency` caps that provider's in-flight requests, held across retries and for a
stream's whole body — a fan-out of agents otherwise opens one connection per branch.
`cache_prompt=True` (Anthropic) puts a cache breakpoint on the system prompt and the last tool
definition, the part of a harness request that repeats every turn.

Each provider holds an HTTP connection pool. Call `await model.aclose()` — or
`model.close()` from synchronous code — when you are done with one.

### Wrapping a provider

```python
Fallback(primary: LLM, *others: LLM, on: tuple[type[BaseException], ...] = (ProviderError,))
Caching(llm: LLM, *, maxsize: int = 256, ttl: float | None = None)
RateLimited(llm: LLM, *, rps: float, burst: int | None = None)
Retrying(llm: LLM, *, attempts: int = 2, backoff: float = 0.5)
```

Each is itself an `LLM`, so one wrapper covers `generate`, `agenerate` and both streaming
paths, and they compose: `Caching(RateLimited(Fallback(a, b), rps=2))`. `inner` reaches the
provider underneath (`models` on `Fallback`), and `aclose()`/`close()` pass down.

| Wrapper | Behaviour |
| --- | --- |
| `Fallback` | Tries each provider in order, catching only `on` — `ProviderError` by default, so a bug in your own code is not mistaken for a flaky vendor. All failing raises one `ProviderError` with the last as `__cause__`. A stream failing before its first event falls back; one failing partway raises. |
| `Caching` | In-process LRU keyed on messages **and** tool schemas, with optional `ttl`. Hits are copied on the way out; a streamed hit is replayed in block order then `Completed`. Exposes `hits`, `misses`, `clear()`. No single-flight: concurrent identical calls all miss. Only sound in front of a deterministic setup. |
| `RateLimited` | Token bucket at `rps`, capacity `burst` (default `max(1, int(rps))`). A caller reserves its slot then waits its own turn. Guarded by a `threading.Lock`, so one instance works across event loops. |
| `Retrying` | Re-asks when a turn has no text **and** no tool calls. Transport-level retries already happen in `HTTPClient`. A stream is retried only while nothing has been emitted. |

Non-positive limits raise `ConfigurationError`, as does a `Fallback` with nothing to fall back
to. `Wrapping` is the shared base if you write your own: subclass it and override only the
methods you change.

### OpenAI-compatible gateways

All of these share `OpenAI`'s constructor shape and read their own API key from the
environment automatically:

`Groq`, `Together`, `DeepSeek`, `Mistral`, `XAI`, `OpenRouter`, `Fireworks`, `Cerebras`.

Local servers — no API key required: `Ollama`, `VLLM`, `LMStudio`, `LlamaCpp`.

For any other OpenAI-compatible endpoint, construct `OpenAI` directly with an explicit
`base_url` and `api_key`.

## Prebuilt workflows

### `DeepResearch`

```python
DeepResearch(
    model: LLM,
    *,
    tools: Iterable[Callable] | Toolbox = (),
    max_sub_questions: int = 5,
    n_parallel: int | None = None,
    sequential: bool = False,
    planner_system: str = PLANNER_SYSTEM,
    researcher_system: str = RESEARCHER_SYSTEM,
    synthesizer_system: str = SYNTHESIZER_SYSTEM,
    budget: Budget | None = None,
)
```

Plans sub-questions, researches each with its own `Agent` (`tools` reach every researcher, not
the planner or synthesizer), then synthesizes one report. See
[Deep research](../guide/research.md) for the full picture, including `n_parallel` vs
`sequential`.

| Member | Signature | Description |
| --- | --- | --- |
| `arun` | `async def arun(query: str) -> ResearchResult` | Drains `astream_events` and returns its `ResearchFinished` result. |
| `astream_events` | `async def astream_events(query: str) -> AsyncIterator[ResearchEvent]` | The actual driver. Yields `Planning`, `Planned`, `Researching`, `Researched`, `Synthesizing`, `TextDelta` (the report as it is written), then a final `ResearchFinished(result)`. |
| `tools` | `Toolbox` | Always a `Toolbox`. Read-only. |
| `max_sub_questions` | `int` | The planner's cap. Read-only. |
| `sequential` | `bool` | Whether researchers run one at a time, seeing earlier answers. Read-only. |
| `n_parallel` | `int \| None` | The concurrency cap; `None` leaves it to the planner. Read-only. |

Raises `ConfigurationError` at construction for `max_sub_questions < 1`, `n_parallel < 1`, or
passing both `sequential=True` and `n_parallel=`.

### `ResearchResult`

```python
ResearchResult(
    query: str,
    sub_questions: list[str],
    findings: list[Finding],
    report: str,
    usage: TokenUsage,
)
```

`findings` is in the order the answers arrived: completion order when parallel, planned order
when `sequential=True`. `usage` covers every agent the run made — planner, every researcher, and
the synthesizer.

### `Finding`

```python
Finding(question: str, answer: str)
```

One sub-question and the answer a researcher reached for it.

### Events

Importable from `deepharness.prebuilt.research` (not re-exported from `deepharness` itself):

```python
Planning(query: str)
Planned(sub_questions: list[str])
Researching(count: int)
Researched(index: int, question: str)
Synthesizing(findings: int)
ResearchFinished(result: ResearchResult)
```

`ResearchFinished` is named apart from `Agent`'s `Finished` on purpose, even though it fills the
same role: the two wrap different things (`result` vs `state`), and the two streams are commonly
read side by side — `from deepharness import Finished` would silently match nothing against a
`DeepResearch` stream, since it names the wrong class.

`ResearchEvent` is the union of these plus `TextDelta`. Typed rather than formatted prose, so a
caller can count, route, or record events instead of matching substrings against a sentence
meant for a human.
