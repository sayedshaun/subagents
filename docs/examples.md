# Examples

Short, self-contained pieces, in the order they get harder. Each one is the smallest code that
shows the thing it is about — copy it, swap the model for a real provider, and it runs.

## An agent with one tool

The think/act loop at its smallest: one model, one tool, one answer.

```python
import asyncio

from deepharness import Agent, OpenAI, tool


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."


agent = Agent(OpenAI("gpt-4o-mini"), tools=[get_weather])

state = asyncio.run(agent.arun("Weather in Oslo?"))
print(state.answered, state.output)
```

`answered` is the check worth making: it is `True` only when the model actually replied. Every
other `stop_reason` leaves `output` empty or partial.

## A graph with no model at all

The clearest way to see what nodes, state, and edges are — no LLM involved.

```python
import asyncio
from dataclasses import dataclass

from deepharness import Graph


@dataclass
class State:
    total: int = 0


graph = Graph(State)


@graph.add(start=True)
def double(state: State) -> State:
    state.total = 21
    return state


@graph.add()
def add_one(state: State) -> State:
    state.total += 1
    return state


graph.connect(double, add_one)

print(asyncio.run(graph.build().run()))  # State(total=22)
```

## Tools that fail, and tools that run together

Two tool calls in one turn run concurrently, and a tool that raises is reported back to the
model rather than ending the run:

```python
@tool
def check_flights(city: str) -> str:
    """Look up flights. Fails when the airline API is down."""
    raise RuntimeError("airline API timed out")


agent = Agent(model, tools=[check_flights])
state = await agent.arun("Any flights to Paris?")
```

The failing call becomes `Error: RuntimeError('airline API timed out')` in the transcript, so
the model gets a turn to apologize or try something else. Sync tools run in a thread, so a
blocking one does not stall the others in the same turn.

## Specialists in parallel

Three agents run as concurrent `start=True` nodes, then a fourth node joins their output:

```python
graph.connect(sales, synthesize)
graph.connect(churn, synthesize)
graph.connect(support, synthesize)
```

Wall-clock time is close to the slowest branch, not the sum of the three — the point of marking
independent work `start=True`. Each branch writes its own field, so no reducer is needed.

## Many branches into one field

When branches write the **same** field, declare a reducer so they combine instead of
overwriting (see [state merging](guide/graph.md#parallel-execution-and-state-merging)):

```python
from dataclasses import dataclass, field

from deepharness.graph import concat


@dataclass
class State:
    findings: list[str] = field(default_factory=list, metadata={"reducer": concat})
```

Without one, the merge raises `ConcurrentUpdateError` — a loud mistake rather than a silent lost
write. For a fan-out whose width is only known at runtime, the graph cannot help: nodes are
registered up front, so use `asyncio.gather` inside a single node instead.

## A refinement loop

A back-edge marked `loop=True` turns a chain into a loop. `draft` and `critique` re-run until
the score clears the bar, while `load_rubric` sits upstream of the head and runs once:

```python
graph.connect(load_rubric, critique)
graph.connect(draft, critique)
graph.connect(critique, draft, loop=True, condition=lambda s: s.score < 0.8)
```

See [loops](guide/graph.md#loops) for the re-entry rules and `max_steps`.

## Gating a risky tool on a human

Mark the tool and the agent pauses *before* running it, every time:

```python
from deepharness import tool


@tool(requires_approval=True)
def wire_transfer(amount_usd: int, to: str) -> str:
    """Send money."""
    return f"sent ${amount_usd:,} to {to}"


state = await agent.arun("Pay the Acme invoice")
print(state.stop_reason)  # "paused"
print(state.paused[0].question)  # Run wire_transfer with {'amount_usd': 50000, ...}?

state = await agent.arun(state.approve())  # now it runs
```

`reject()` instead of `approve()` records "Denied by the user." as the result, so the model can
say so rather than retrying. The gate is on the tool, not in the prompt, so a model cannot skip
it by not asking. The pause is an ordinary returned state, not an exception, so it can be
persisted and resumed in another process.

## Streaming, with tools still working

```python
async for chunk in agent.astream("What is 17 * 23?"):
    print(chunk, end="", flush=True)
```

Every turn is streamed, tool turns included — those simply yield no text. Use
`astream_events()` when you also need the final state; see
[streaming](guide/agents.md#streaming).

## Dependencies a tool can reach

A parameter annotated `Ctx` is filled by the runtime and hidden from the model, so a tool reads
per-request dependencies without a global:

```python
from deepharness import Ctx, tool


@tool
def lookup_plan(customer: str, ctx: Ctx) -> str:
    """Look up a customer's plan."""
    return ctx.deps.db.plan_for(customer, tenant=ctx.deps.tenant)


state = await agent.arun("What plan is Acme on?", deps=Deps(db=db, tenant="acme"))
```

## An agent working in a directory

For the whole thing assembled — permissions, approvals, context and sessions — see
[building a coding agent](guide/coding-agent.md). The short version:

`file_tools()` and `shell_tool()` are the difference between an agent that talks about a
codebase and one that reads it. Both close over a `Workspace`, so every path the model sends is
resolved inside that root:

```python
import asyncio

from deepharness import Agent, OpenAI
from deepharness.tools import file_tools, shell_tool

agent = Agent(
    OpenAI("gpt-4o-mini"),
    tools=[*file_tools("."), shell_tool(".")],
    system="Work in the repository you are given. Read before you edit.",
)

state = asyncio.run(
    agent.arun("Which module defines the executor, and what does it do?")
)
print(state.output)
```

A path that escapes the root raises `OutsideWorkspace`, which reaches the model as that call's
result — so it can correct itself while the read never happens. `write_file` and `edit_file`
are gated out of the box; `file_tools(".", writable=False)` hands over the read-only three
instead, which is a stronger guarantee than a rule because there is nothing left to rule on.

## Allowing `git log` but not `git push`

`requires_approval` is per tool, which stops being enough when the tool is `run_command`.
`Permissions` decides per call, from the arguments the model actually sent:

```python
from deepharness import Agent
from deepharness.tools import Permissions, Rule, ToolName, shell_tool

agent = Agent(
    llm,
    tools=[shell_tool(".")],
    permissions=Permissions(
        allow=[Rule(ToolName.RUN_COMMAND, {"command": "git log*"})],
        ask=[ToolName.RUN_COMMAND],
        deny=[Rule(ToolName.RUN_COMMAND, {"command": "*rm -rf*"})],
    ),
)
```

`deny` beats `allow` beats `ask`, so widening a policy can never quietly override a refusal
already written down. A denied call never runs and the model is told so, which lets it find
another way instead of retrying. A call no rule matches falls back to the tool's own
`requires_approval`.

## Keeping a long run inside the window

A run that reads files pays for every result again on every later turn. `ContextPolicy` bounds
both ends of that:

```python
from deepharness import Agent
from deepharness.agent import ContextPolicy
from deepharness.tools import file_tools

agent = Agent(
    llm,
    tools=file_tools("."),
    context=ContextPolicy(max_tokens=100_000, tool_result_chars=8_000),
)
```

`tool_result_chars` truncates each result as it is recorded, eliding the middle. `max_tokens`
prunes the view the model is sent — oldest turns first — while `state.messages` keeps every
message, so nothing is lost to you that only had to be kept from the provider. See
[context management](guide/agents.md#context-management).

## Watching a run work

Text alone makes an agent look stalled, because most of a long run is tool calls the model
never narrates:

```python
from deepharness import Finished, TextDelta
from deepharness.agent import StepStarted, ToolFinished, ToolStarted

async for event in agent.astream_events("Add a docstring to the executor"):
    match event:
        case StepStarted(step):
            print(f"\n— step {step}")
        case ToolStarted(name, arguments, _):
            print(f"{name}({arguments}) …")
        case ToolFinished(name, result, failed, _):
            print(f"{name} {'failed' if failed else 'ok'}: {result[:60]}")
        case TextDelta(text):
            print(text, end="", flush=True)
        case Finished(state):
            print(f"\nstopped because: {state.stop_reason}")
```

`ToolFinished.result` is the text the model will read, truncation included, so what you show
cannot drift from what it saw.

## Wrapping the model instead of the loop

Caching, rate limiting, retrying and falling back are all `LLM`s, so they stack in front of a
provider and work on every path at once:

```python
from deepharness import Agent, Anthropic, OpenAI
from deepharness.providers import Caching, Fallback, RateLimited

llm = Caching(
    RateLimited(Fallback(OpenAI("gpt-4o-mini"), Anthropic("claude-sonnet-4-5")), rps=2)
)

agent = Agent(llm, tools=[get_weather])
```

Read it outside-in: the cache answers first, then the limiter, then the fallback talks to a
vendor. `Fallback` catches `ProviderError` only — a `TypeError` in your own code should not
read as a flaky model — and a stream that fails partway through raises rather than restarting,
because those deltas already reached you. See
[wrapping a provider](guide/providers.md#wrapping-a-provider).

## Asking about an image

Message content is a string until it needs to be more:

```python
from deepharness import Message
from deepharness.providers import Image, Text

state = await agent.arun(
    [
        Message.human(
            [Text("What changed in this screenshot?"), Image.from_path("ui.png")]
        )
    ]
)
```

`Image.from_path()` encodes the file and infers its media type; `Document.from_path("x.pdf")`
sends a file to read. Each provider renders these into its own wire shape. Content that is only
text is still sent as a plain string, so nothing changes for an ordinary conversation.

## Resuming a run that stopped for a human

The pause is a returned state, not an exception, so it survives a process boundary:

```python
state = await agent.arun("Deploy the release branch")
if state.stop_reason == "paused":
    store.put("run", json.dumps(state.to_dict()))  # pending approval included

# ... in another process, once someone has looked at it
state = AgentState.from_dict(json.loads(store.get("run")))
state = await agent.arun(state.approve())  # the gated call runs now
```

`store` is whatever you keep runs in. See
[session persistence](guide/agents.md#session-persistence).

## Tools from an MCP server

An [MCP](https://modelcontextprotocol.io) server's tools join the same toolbox as your own:

```python
from deepharness import Agent
from deepharness.tools import MCPServer

async with MCPServer.stdio(
    ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
) as mcp:
    agent = Agent(llm, tools=await mcp.tools())
    state = await agent.arun("What is in the repo?")
```

They are async, so drive the agent with `arun()`. A tool the server did not mark read-only is
gated by default — it is code in another process that this side cannot inspect. See
[MCP](guide/tools.md#tools-from-an-mcp-server).

## The agent loop, as a graph

`Agent` runs think/act as an internal Python loop. Rebuilding it out of graph nodes gives you a
place to insert an approval gate, a budget check, or a re-planning step mid-cycle:

```python
graph.connect(think, act, condition=lambda s: bool(s.pending_calls))
graph.connect(act, think, loop=True)
```

## Researching a question end to end

`DeepResearch` plans sub-questions, researches each with its own `Agent`, and synthesizes one
report:

```python
import asyncio

from deepharness import OpenAI
from deepharness.prebuilt import DeepResearch
from deepharness.tools import TavilySearch

search = TavilySearch()
research = DeepResearch(OpenAI("gpt-4o-mini"), tools=[search.as_tool()])

result = asyncio.run(research.arun("How do UK master's student visas work?"))
print(result.report)
```

Watch it work instead of waiting on a blank screen by iterating `astream_events()` — see
[Deep research](guide/research.md#watching-a-run-live).
