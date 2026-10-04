<div align="center">
<h1><img src="https://raw.githubusercontent.com/sayedshaun/deepharness/main/docs/assets/logo.png" width="220" alt="DeepHarness logo"><br>DeepHarness</h1>

**A harness for LLM agents: workspace tools, permissions, and typed, concurrent workflows.**

Build agents that read and change a codebase under an explicit permission policy, and compose
them as a graph of plain Python functions: parallel branches, typed state, and 15 LLM providers
behind one interface.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![CI](https://github.com/sayedshaun/deepharness/actions/workflows/test.yml/badge.svg)](https://github.com/sayedshaun/deepharness/actions/workflows/test.yml) [![PyPI](https://img.shields.io/pypi/v/deepharness?logo=pypi&logoColor=white&color=3775A9)](https://pypi.org/project/deepharness/) [![Dependencies: httpx only](https://img.shields.io/badge/dependencies-httpx%20only-6E63F5)](https://github.com/sayedshaun/deepharness/blob/main/pyproject.toml) [![Async native](https://img.shields.io/badge/async-native-0EA5E9)](#quickstart) [![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff) [![License: MIT](https://img.shields.io/badge/license-MIT-22C55E)](LICENSE) [![Read the docs](https://img.shields.io/badge/docs-read%20the%20docs-3776AB?logo=materialformkdocs&logoColor=white)](https://sayedshaun.github.io/deepharness/)

[Install](#install) · [Quickstart](#quickstart) · [Harness](#working-in-a-directory) · [Graphs](#graphs) · [Providers](#providers) · [Docs](https://sayedshaun.github.io/deepharness/) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

</div>

---

## Why DeepHarness

- **One dependency. That's it.** `httpx`. Providers talk to vendor REST APIs
  directly — no vendor SDKs, no transitive dependency sprawl.
- **Your state is a dataclass.** Not an untyped dict. Your editor autocompletes it, your type
  checker checks it.
- **Nodes are just functions.** Sync or async, decorated in place. No base class to inherit,
  no runner object to construct.
- **Parallel by default.** Independent branches run concurrently via `asyncio`; sync functions
  are offloaded to threads so a blocking call never stalls the event loop.
- **One provider interface, a dozen vendors.** `OpenAI`, `Anthropic`, `Gemini`, and OpenAI-compatible
  gateways (`Groq`, `Together`, `Fireworks`, `DeepSeek`, `Mistral`, `Cerebras`, `OpenRouter`, `XAI`,
  `Ollama`, `LMStudio`, `VLLM`, `LlamaCpp`) all share the same `LLM` interface.
- **A harness, not just a loop.** Workspace-confined file and shell tools, MCP servers, a
  per-call permission policy, a bounded context window, and progress events — the parts a run
  needs once it is long enough to matter.

## Install

```bash
pip install deepharness
```

Or from source:

```bash
pip install git+https://github.com/sayedshaun/deepharness.git
```

Requires Python 3.11 or newer.

## Quickstart

An `Agent` runs a think/act loop against a model: ask for a response, dispatch any tool calls
it requests, repeat until the model answers with no tool calls.

```python
from deepharness import Agent, OpenAI, tool


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."


agent = Agent(
    OpenAI(model="gpt-4o-mini"),  # reads OPENAI_API_KEY from the environment
    tools=[get_weather],
)

result = agent.run("Weather in Oslo?")
print(result.output)
```

Every entry point comes in both flavours — `run()` / `arun()` and `stream()` / `astream()` —
so the same agent drops into a script or an event loop unchanged:

```python
result = await agent.arun("Weather in Oslo?")
```

### Agents as tools

An `Agent` can itself be handed to another agent as a tool via `as_tool()`, so one agent can
delegate a task to another:

```python
researcher = Agent(OpenAI(model="gpt-4o-mini"), name="researcher")

editor = Agent(
    OpenAI(model="gpt-4o-mini"),
    name="editor",
    tools=[researcher.as_tool(description="Look up facts on a topic.")],
)
```

## Working in a directory

An agent that can read a codebase and change it, with the sharp edges gated:

```python
from deepharness import Agent
from deepharness.agent import ContextPolicy
from deepharness.tools import Permissions, Rule, ToolName, file_tools, shell_tool

agent = Agent(
    OpenAI(model="gpt-4o-mini"),
    tools=[*file_tools("."), shell_tool(".")],
    context=ContextPolicy(max_tokens=120_000),
    permissions=Permissions(
        allow=[ToolName.READ_FILE, ToolName.LIST_FILES, ToolName.SEARCH_FILES],
        ask=[ToolName.WRITE_FILE, ToolName.EDIT_FILE, ToolName.RUN_COMMAND],
        deny=[Rule(ToolName.RUN_COMMAND, {"command": "*rm -rf*"})],
    ),
)

state = await agent.arun("What does the executor do? Add a docstring if it lacks one.")
```

A full walkthrough — approvals, context, sessions, and a sixty-line script that runs — is in
[Building a coding agent](https://sayedshaun.github.io/deepharness/guide/coding-agent/).

Every path is resolved inside the workspace root, `deny` beats `allow` beats `ask`, and a run
that needs a human stops with `stop_reason == "paused"` — resumable later, in another process,
because `state.to_dict()`/`AgentState.from_dict()` round-trip the whole state, pending approval
included.

The rest of what a long run needs:

- **Watch it work.** `astream_events()` emits `StepStarted`, `ToolStarted`, `ToolFinished` and
  `ThinkingDelta` alongside the text, so a tool call is visible rather than dead air.
- **Stay inside the window.** `ContextPolicy` truncates each tool result and prunes the view
  the model is sent, while `state.messages` keeps everything.
- **Wrap the model, not the loop.** [`Caching`, `RateLimited`, `Retrying` and
  `Fallback`](#wrapping-a-provider) are themselves `LLM`s, so they work on the sync path, the
  async path and streaming alike.
- **More than text.** `Message.human([Text("what changed?"), Image.from_path("ui.png")])`
  sends images and PDFs; a thinking model's reasoning arrives as `ThinkingDelta` and is
  replayed where the vendor requires it.
- **Tools from elsewhere.** An [MCP](https://modelcontextprotocol.io) server's tools join the
  same toolbox via `MCPServer.stdio(...)` or `MCPServer.http(...)`.

### Wrapping a provider

Caching, rate limiting, retrying and falling back are all "do something around a model call,
then delegate" — so each one is just another `LLM`. They compose at the call site, where the
order is visible:

```python
from deepharness import Anthropic, OpenAI
from deepharness.providers import Caching, Fallback, RateLimited

llm = Caching(
    RateLimited(Fallback(OpenAI("gpt-4o-mini"), Anthropic("claude-sonnet-4-5")), rps=2)
)

agent = Agent(llm, tools=file_tools("."))
```

| Wrapper | What it does |
| --- | --- |
| `Fallback(primary, *others)` | Moves to the next provider when one raises `ProviderError`. A stream that fails *before* its first event falls back; one that fails partway does not, since those deltas already reached you. |
| `Caching(llm, maxsize=256, ttl=None)` | Serves a repeated request from an LRU. Put it in front of a deterministic setup only. |
| `RateLimited(llm, rps=…, burst=…)` | Token bucket. Each caller reserves a slot and waits its own turn, so simultaneous requests leave in order at the configured rate. |
| `Retrying(llm, attempts=2)` | Asks again when a turn comes back with no text *and* no tool call — the transport already retries 429s and 5xx. |

Because these are `LLM` implementations rather than a middleware stack, one wrapper covers
`run()`, `arun()` and streaming at once — and `Fallback(on=(ProviderError,))` deliberately does
not catch bare `Exception`, so a bug in your own code never reads as a flaky vendor.

## Graphs

For fixed, known-shape workflows — run these steps, some in parallel, then merge — wire plain
functions (or agents) into a `Graph` instead:

```python
import asyncio
from dataclasses import dataclass

from deepharness import Agent, Graph, OpenAI


@dataclass
class State:
    sales: str = ""
    churn: str = ""
    summary: str = ""


graph = Graph(State)

analyst = Agent(
    OpenAI(model="gpt-4o-mini"),
    system="Summarise the quarter in one sentence for an exec audience.",
)


@graph.add(start=True)
async def fetch_sales(state: State) -> State:
    await asyncio.sleep(0.5)
    state.sales = "Sales up 12% QoQ."
    return state


@graph.add(start=True)
async def fetch_churn(state: State) -> State:
    await asyncio.sleep(0.3)
    state.churn = "Churn down to 4%."
    return state


@graph.add(end=True)
async def summarize(state: State) -> State:
    result = await analyst.arun(f"{state.sales} {state.churn}")
    state.summary = result.output
    return state


graph.connect(fetch_sales, summarize)
graph.connect(fetch_churn, summarize)

executor = graph.build()
result = asyncio.run(executor.run())

print(result.summary)
```

`fetch_sales` and `fetch_churn` both run in the first wave, concurrently. `summarize` waits
for both before running, then hands the merged state to an `Agent` — a node is just a
function, so an LLM call inside one needs no special wiring.

`executor.diagram()` draws the wiring, with nodes that run in the same wave side by side:

```text
╭═════════════╮   ╭═════════════╮
│ fetch_sales │   │ fetch_churn │
╰═════════════╯   ╰═════════════╯
       ╰────────╮        │
                ├────────╯
                ▼
          ╭═══════════╮
          │ summarize │
          ╰═══════════╯
```

### Agent or Graph?

Reach for `as_tool()` when an LLM should decide at runtime whether and which sub-agent to
call. Reach for `Graph` when the workflow's shape — which steps, which run in parallel, in
what order — is known ahead of time.

## Providers

Pass `api_key=` explicitly, or leave it out and let the provider read its own environment
variable:

| Provider | Environment variable |
| --- | --- |
| `OpenAI` | `OPENAI_API_KEY` |
| `Anthropic` | `ANTHROPIC_API_KEY` |
| `Gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

OpenAI-compatible gateways each read their own key — `GROQ_API_KEY`, `TOGETHER_API_KEY`,
`DEEPSEEK_API_KEY`, and so on — and local runtimes (`Ollama`, `LMStudio`, `VLLM`, `LlamaCpp`)
need no key
at all. Swapping vendors means swapping the constructor; nothing else changes:

```python
from deepharness import Anthropic, Gemini, Groq, Ollama

Anthropic(model="claude-sonnet-4-5")
Gemini(model="gemini-2.5-flash")
Groq(model="llama-3.3-70b-versatile")
Ollama(model="llama3.2")  # no key needed
```

## Contributing

Issues and pull requests are welcome. To get a working checkout:

```bash
git clone https://github.com/sayedshaun/deepharness.git
cd deepharness
pip install -e ".[dev]"
make test
```

Two conventions are worth knowing before you open a pull request:

- **The dependency budget is one.** `httpx`, and nothing else at runtime. A feature that needs
  a third-party library almost always has a standard-library shape — reach for that instead.
- **Tests must not touch the network.** Inject an `httpx` client or transport so the suite stays
  deterministic and offline.

Run `make fmt` and `make test` before pushing; CI runs the same checks on Python 3.11, 3.12 and
3.13. Keep each commit focused on a single change, with a one-line message prefixed by a
Conventional Commits type (`feat:`, `fix:`, `docs:`, and so on).

Full setup, commands, and design conventions live in **[CONTRIBUTING.md](CONTRIBUTING.md)**; the
rules the code itself is held to are in **[AGENTS.md](AGENTS.md)**.

## License

Released under the **[MIT License](LICENSE)** — Copyright (c) 2026 Sayed Shaun.

You may use, modify, distribute, and sell this software, in open-source or commercial work,
provided the copyright notice and licence text travel with it. It comes with no warranty.

---

<div align="center">
<sub>Simple, modular, typed, dependency-light.</sub>
</div>
