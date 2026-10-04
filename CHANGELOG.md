# Changelog

Notable changes, newest first. This project is pre-1.0: while the major version is `0`, a
breaking change raises the minor version.

## Unreleased

### Breaking

- `save_session` and `load_session` are removed. Where a run is stored is the application's
  choice, so the library keeps only the serialization: `state.to_dict()` and
  `AgentState.from_dict(data)`. The equivalent of the old pair is
  `Path(p).write_text(json.dumps(state.to_dict()))` and
  `AgentState.from_dict(json.loads(Path(p).read_text()))`.

### Added

- `Toolbox(approve=fn)` checks `fn(name, arguments)` before running a `requires_approval`
  call, wherever the toolbox is called from; a refusal raises `ToolDenied`. Without it an
  `Agent` pauses as before.

## 0.3.0 — 2026-09-19

The release that makes an agent usable for long runs: tools that work in a directory, rules
about what they may do, a bounded context window, and progress you can watch.

### Breaking

- **Imports are layered.** `deepharness` now exports ~27 high-level names instead of 81;
  everything else comes from the package that owns it — `deepharness.agent`,
  `deepharness.graph`, `deepharness.providers`, `deepharness.tools`, `deepharness.prebuilt`,
  `deepharness.errors`. `Agent`, `Graph`, `tool`, `Message`, `Budget`, `Toolbox`, `Ctx`,
  `AgentState`, `Finished`, `TextDelta`, `Executor`, `DeepHarnessError` and all 15 providers
  are unchanged; `ContextPolicy`, `Permissions`, `file_tools`, `save_session`, `DeepResearch`,
  the content blocks, the progress events, the provider wrappers and the specific exceptions
  all moved one level down. No aliases are kept: an import that moved fails loudly rather than
  working for one more release.
- `load_session(path)` returns an `AgentState` rather than `list[dict]`, and `save_session`
  writes the whole state instead of the transcript alone. That is what lets a run paused on an
  approval resume in another process — a messages-only file cannot carry what it is waiting
  for. Passing the result to `run()`/`arun()` is unchanged; iterating it is not. A file holding
  a bare JSON array still loads as a transcript.
- Each tool result is truncated to 8000 characters as it is recorded
  (`ContextPolicy.tool_result_chars`). Pass `None` to keep the old unbounded behaviour.
- A turn that pauses for approval now records its unrun calls in the transcript. Without it, a
  resumed run sends a tool call no result answers, which vendors reject.
- `StreamAccumulator.feed()` returns a `StreamEvent` instead of `str | None`, so a payload can
  carry reasoning rather than prose. Only affects code implementing a provider by hand.

### Added

- **Working in a directory.** `file_tools()` gives `read_file`, `list_files`, `search_files`,
  `write_file` and `edit_file`; `shell_tool()` gives `run_command`. Every path resolves through
  `Workspace`, which refuses anything outside its root — symlinks included — and raises
  `OutsideWorkspace`, which reaches the model as that call's result. Writers and `run_command`
  are gated by default.
- **Permissions.** `Permissions(allow=…, ask=…, deny=…)` with `Rule("run_command", {"command":
  "git *"})` decides per call, from the arguments the model actually sent. `deny` beats `allow`
  beats `ask`; a call no rule matches falls back to the tool's own `requires_approval`. A rule
  can name a tool by passing the tool itself or a `ToolName` member instead of a string, and a deny or ask rule naming an unregistered tool is refused at construction —
  a misspelled one would match nothing and allow what it was written to stop.
- **Context management.** `ContextPolicy` truncates each tool result and, with `max_tokens`
  set, prunes the view the model is sent while `state.messages` keeps every message. Pruning
  never drops the system prompt, never orphans a tool result and always keeps the last
  `keep_last` messages. `estimate_tokens()` is the heuristic behind it.
- **Progress events.** `astream_events()` now emits `StepStarted`, `ToolStarted`,
  `ToolFinished` and `ThinkingDelta` alongside `TextDelta`. `ToolFinished.result` is the text
  the model will read, truncation included.
- **Content blocks.** A message's content may be `Text`, `Image`, `Document` or `Thinking`
  blocks — `Message.human([Text("what changed?"), Image.from_path("ui.png")])`. Text-only
  content is still sent, and saved, as a plain string.
- **Reasoning round-trips.** Thinking arrives as `ThinkingDelta`, is available as
  `response.thinking`, is kept in the transcript, and is replayed to Anthropic with its
  signature — required after a tool call, and previously lost. OpenAI-compatible servers are
  read too: `reasoning_content` (llama.cpp, DeepSeek) and `reasoning` (OpenRouter), in both
  the response and the stream.
- **Provider wrappers.** `Fallback`, `Caching`, `RateLimited` and `Retrying` are themselves
  `LLM`s, so one wrapper covers the sync path, the async path and streaming:
  `Caching(RateLimited(Fallback(a, b), rps=2))`. `Wrapping` is the base for your own.
- **MCP client.** `MCPServer.stdio(...)` / `MCPServer.http(...)` turns a Model Context Protocol
  server's tools into ordinary toolbox entries, over stdio or streamable HTTP, with no new
  dependency. `Transport` is the extension point for anything those two do not reach.
- **Anthropic prompt caching.** `cache_prompt=True` puts a cache breakpoint on the system
  prompt and the last tool definition.
- **Cache counts in usage.** `TokenUsage` gains `cached_tokens` and `cache_write_tokens`, read
  from each vendor's own field, so a turn whose prompt was mostly a cache read is no longer
  indistinguishable from one that paid for all of it. Both are inside `prompt_tokens`;
  `Budget` still counts the full figure.
- **Concurrency cap.** `max_concurrency=` on any provider bounds its in-flight requests, held
  across retries and for a stream's whole body.
- **DeepResearch.** `DeepResearch` plans sub-questions, researches them in parallel and
  synthesizes one report, streaming its progress through `astream_events()`.
- **Web search.** `TavilySearch` wraps Tavily's API as a tool.
- **`LlamaCpp`** provider for llama.cpp's server.
- **Graph diagrams.** A built `Graph` renders as a terminal diagram.

### Fixed

- A capped provider used in a second event loop no longer raises: the concurrency gate is now
  per loop, because a contended asyncio primitive binds to the loop that awaited it.
- A stdio MCP server that writes notifications before its reply is understood; the client reads
  past them to the message matching its request id.
- Gemini thinking tokens count as completion tokens, so the three usage numbers add up.
- A provider-truncated answer surfaces as `stop_reason == "truncated"` for OpenAI, Anthropic
  and Gemini rather than looking like a finished reply.
- A provider with no API key sends no auth header at all, which is the normal case for a local
  server.
- Four documentation snippets used names they never imported.

## 0.2.0 — 2026-09-04

- A graph run builds its own default state when none is given.
- A gateway can opt out of asking for streamed usage (`stream_usage=False`).
- Provider requests get a real timeout and retry transient failures with backoff.
- Gemini's API key moves from the query string to a header, keeping it out of error messages.
- Prompt tokens are kept in streamed usage; truncated tool arguments no longer kill a stream.
- Registering two different tools under one name is refused rather than silently shadowed.
- Graph state is validated, raising `ConfigurationError` instead of a bare `ValueError`.
- `OutputValidationError` is exported from the package root.

## 0.1.0 — 2026-09-01

First release: the agent think/act loop, `Toolbox` and `@tool`, the graph engine with parallel
branches and merging, typed dataclass state, and providers for OpenAI, Anthropic, Gemini and
the OpenAI-compatible gateways.
