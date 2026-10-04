# Agents

An `Agent` runs a think/act loop against a model: ask for a response, dispatch any tool calls
it requests, repeat until the model answers with no tool calls or the step budget is spent.

```python
from deepharness import Agent, Budget, Message, tool
from deepharness import OpenAI


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."


agent = Agent(
    OpenAI(model="gpt-4o-mini", api_key="sk-..."),
    system="You are a concise assistant.",
    tools=[get_weather],
)

result = await agent.arun("Weather in Oslo?")
print(result.output)
```

`tools=` accepts any number of functions — decorated with [`@tool`](tools.md) or plain — an
agent isn't limited to one. See the [tools guide](tools.md) for how a function becomes a tool.

## `run()` vs `arun()`

Same split as providers: `arun()` is the async path — tool calls requested in the same turn
run **concurrently**, sync tools included: those go to a thread so one blocking call cannot
stall the rest. `run()` is a real synchronous path rather than a wrapper, so it raises
if a registered tool turns out to be `async def` — there's no event loop here to await it:

```python
result = agent.run("Weather in Oslo?")  # sync, sync tools only
result = await agent.arun("Weather in Oslo?")  # async, concurrent tool calls
```

## State

`arun`/`run` return an `AgentState` — a dataclass, not a dict:

| Field | What it holds |
| --- | --- |
| `messages` | The transcript, as wire-form dicts. |
| `output` | The answer: text, or an `output=` instance when one is set. |
| `usage` | `TokenUsage` for this run. |
| `stop_reason` | Why the loop ended: `"answer"`, `"step_budget"`, `"paused"`, `"token_budget"`, `"truncated"`. |
| `paused` | Any `PendingHumanInput` waiting on a human; empty otherwise. |
| `answered` | `True` only when `stop_reason == "answer"` — check this before trusting `output`. |

For input, pass whatever is convenient — a prompt string, a list of messages, or an
`AgentState` when you are resuming one:

```python
await agent.arun("Weather in Oslo?")
await agent.arun([Message.human("Weather in Oslo?")])
await agent.arun(previous_state)
```

A dict of known fields still works, but a key the agent does not own raises
`ConfigurationError` rather than being dropped silently — an agent owns its own state, so keep
a graph's fields on the graph's state.

## Streaming

`astream()` yields the model's prose as it arrives, and tools still run:

```python
async for chunk in agent.astream("What is 17 * 23?"):
    print(chunk, end="", flush=True)
```

Every turn is streamed, not just the answering one — an agent cannot know in advance whether a
turn will answer or call a tool, so the provider hands back the assembled turn either way and a
tool turn simply yields no text.

When you need the result too, use `astream_events()`. It yields `TextDelta` as text arrives and
one final `Finished` carrying the `AgentState`, because an async generator cannot return a
value — and in between, what the loop is doing:

```python
from deepharness import Finished, TextDelta
from deepharness.agent import StepStarted, ToolFinished, ToolStarted

async for event in agent.astream_events("What is 17 * 23?"):
    match event:
        case StepStarted(step):
            print(f"\n— step {step}")
        case TextDelta(text):
            print(text, end="", flush=True)
        case ToolStarted(name, arguments, _):
            print(f"{name}({arguments}) …")
        case ToolFinished(name, result, failed, _):
            print(f"{name} {'failed' if failed else 'ok'}: {result}")
        case Finished(state):
            print(f"\nstopped because: {state.stop_reason}")
```

Most of a long run's wall clock goes on tool calls the model never narrates, so text alone
makes an agent look stalled. `StepStarted.step` is 1-based and capped by `Budget.steps`;
`ToolFinished.result` is the text the model will read — already truncated by the
[context policy](#context-management), so what you display is what the model saw. A tool that
raised sets `failed`, and the run still continues, since the error goes back to the model as
that call's result. A tool that asked a human reports no `ToolFinished`: it has no result yet,
and the run is about to pause on it.

`stream()`/`stream_events()` are the synchronous counterparts, and there tools run one at a
time, so each tool's pair of events surrounds its own call rather than the whole turn's.
`astream()`/`stream()` yield text alone and never see the rest.

## Token usage and budgets

Every provider normalizes the vendor's token counts into a `TokenUsage(prompt_tokens,
completion_tokens, total_tokens)`. `Agent` accumulates it across every model call on
`agent.total_usage`, and the final `state.usage` reflects that running total:

```python
agent = Agent(llm, name="assistant", budget=Budget(tokens=50_000))

state = await agent.arun("...")
print(
    state.usage
)  # TokenUsage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
print(agent.total_usage)  # same object — persists across multiple arun()/run() calls
```

`Budget` bounds a run two ways, and they fail differently. `Budget(tokens=...)` raises
`TokenBudgetExceeded` the moment cumulative usage crosses it — checked right after each model
response, so a run already over budget won't dispatch further tool calls or make another model
call. `Budget(steps=...)` (default 10) caps think/act turns instead, and spending them returns
normally with `stop_reason == "step_budget"` and an empty `output`.

`Budget(steps=1)` is the single-shot case: one model call, no turn to react to a tool result.
Handy for a classify-or-extract step, but an agent with tools will stop at `"step_budget"`
rather than answering whenever it calls one.

## Context management

A think/act loop grows its own transcript: every turn appends the model's request and whatever
its tools returned, and every later turn pays for all of it again. `ContextPolicy` bounds that
from both ends.

```python
from deepharness import Agent
from deepharness.agent import ContextPolicy

agent = Agent(
    llm,
    tools=[read_file],
    context=ContextPolicy(max_tokens=100_000, tool_result_chars=8_000),
)
```

`tool_result_chars` (default 8000) bounds each tool result as it is recorded, eliding the
middle and saying how much went missing — the middle rather than the tail because a long
result usually says what it is at the top and how it ended at the bottom. This is on by
default: one oversized result is the usual way a long run dies, since it is re-sent with every
turn after it.

`max_tokens` bounds the whole transcript and is **off** by default. Set it somewhere under the
model's real window and the loop sends a pruned view — oldest turns dropped, with a note in
their place — while `state.messages` keeps every message, so nothing is lost to you that only
had to be kept from the provider. Pruning never drops the leading system prompt, never orphans
a tool result from the turn that requested it, and always keeps the last `keep_last` messages
(default 4). A transcript whose tail alone exceeds the budget is therefore sent over it: a
bound is not worth breaking a turn's tool-call pairing for.

The token figure behind `max_tokens` is an estimate (`estimate_tokens`, roughly four
characters per token) rather than a real tokenizer, which would mean a per-vendor dependency.
Treat it as a soft bound that keeps a long run away from a hard provider error, not as a way
to predict a bill. Subclass `ContextPolicy` and override `prune()` to choose differently — the
loop asks for a view of the transcript and does not care how it was chosen.

## Structured output

Pass `output=` a dataclass and `state.output` becomes a validated instance of it instead of
prose:

```python
from dataclasses import dataclass


@dataclass
class Weather:
    city: str
    celsius: int


agent = Agent(llm, output=Weather, tools=[get_weather])

state = await agent.arun("Weather in Oslo?")
print(state.output.celsius)  # 22
```

It works by offering the model one extra tool, `final_answer`, whose parameters are the
model's schema — so it behaves the same on every provider, with no vendor-specific JSON mode
involved. Two consequences worth knowing:

- If the model replies with prose instead of calling `final_answer`, that is not treated as an
  answer: the agent asks it to call the tool and keeps going, bounded by the step budget. A
  model that never complies ends at `stop_reason == "step_budget"`.
- If the arguments don't fit, an `OutputValidationError` goes back to the model as that call's
  result — the same courtesy a failing tool gets — so it can try again with valid fields. Every
  bad field is reported at once, so one round-trip fixes them all.
- Fields are checked, not coerced from anything: `list[str]`, `Literal`, `Enum`, `X | None` and
  nested dataclasses all validate, an `int` is accepted where a `float` is declared, and `true`
  is rejected for an `int` field even though Python calls a bool an int.

## Human in the loop

Two different things a human can be needed for, and they resolve differently.

**Approval — the call has not run yet.** Mark the tool and the agent pauses *before* running it:

```python
@tool(requires_approval=True)
def wire_transfer(amount_usd: int, to: str) -> str:
    """Send money."""
    return f"sent ${amount_usd:,} to {to}"


state = await agent.arun("Pay the Acme invoice")
print(state.stop_reason)  # "paused"
print(state.paused[0].question)  # Run wire_transfer with {'amount_usd': 50000, ...}?

state = await agent.arun(state.approve())  # runs it now, with the model's arguments
# or
state = await agent.arun(state.reject())  # records "Denied by the user." instead
```

`approve()`/`reject()` return the state, so a resume is one line. With no `call_id` they resolve
every pending call; pass one to rule on a single call. Resuming a paused run without deciding
raises `ConfigurationError` rather than silently continuing.

To decide during the run instead of pausing it, give the `Toolbox` an `approve=` callback. It is
called with each gated call's name and arguments before the call runs, and may be `async`. `True`
runs the call; `False` raises `ToolDenied`, which an `Agent` hands to the model as that call's
result. How the callback decides is up to you:

```python
from deepharness.agent import Toolbox


def approve(name: str, arguments: dict) -> bool:
    return arguments["amount_usd"] <= 1_000


agent = Agent(model, tools=Toolbox([wire_transfer], approve=approve))
```

The gate belongs to the toolbox, so it also holds when a `Graph` node or your own code calls
`tools.call(...)` directly; catch `ToolDenied` there to decide what a refusal means. Without
`approve=` the run pauses as above.

The gate lives on the tool, not in the prompt, so a model cannot route around it by declining to
ask. And if a turn requests a gated call alongside ordinary ones, **nothing** in that turn runs
until the ruling — a half-applied turn the human is about to refuse would be worse than waiting.
Those unrun calls are still recorded as not run, because a requested call that no result answers
is a transcript vendors reject when the run resumes.

For anything finer than per-tool — allowing `git log` but not `git push`, both of which are
`run_command` — see [Permissions](tools.md#permissions-deciding-per-call).

**A question — the tool wants to ask you something.** Raise `HumanInputRequired` and the human's
answer becomes that call's result:

```python
from deepharness.errors import HumanInputRequired


@tool
def confirm(question: str) -> str:
    """Ask the operator something."""
    raise HumanInputRequired(question)


state = await agent.arun("...")
pending = state.paused[0]
state.messages.append(
    Message.tool("yes, proceed", name=pending.name, call_id=pending.call_id)
)
state = await agent.arun(state)
```

The difference matters: an approval defers execution, a question substitutes a result. Use
`pending.needs_approval` to tell them apart.

## Session persistence

`save_session`/`load_session` round-trip a whole `AgentState` through JSON, so a run resumes
across process runs:

```python
from deepharness import Agent, Message
from deepharness.agent import load_session, save_session

state = load_session("session.json")  # an empty AgentState if the file is new
state.messages.append(Message.human("Continue where we left off.").to_dict())

state = await agent.arun(state)
save_session("session.json", state)
```

The whole state is saved, not just the transcript — usage, stop reason, and any call
[paused on an approval](#human-in-the-loop). That last one is the point: a run waiting on a
human is the one most worth resuming later, and it is the one a messages-only file cannot
carry.

```python
state = load_session("session.json")
if state.stop_reason == "paused":
    state = await agent.arun(state.approve())  # the gated tool runs now
```

`save_session` also accepts a bare message list for the conversational case, and
`load_session` reads a file holding a bare JSON array as a transcript — so a session written
by hand, or by a version that saved messages alone, still loads. Structured `output` is stored
as plain data and comes back as a dict rather than your dataclass; reconstructing the type
would mean trusting an import path out of a file.

## Messages

`Message` is a `dict` subclass with role-named constructors, so it's a drop-in replacement for
`{"role": ..., "content": ...}` everywhere a message is expected:

```python
from deepharness import Message

Message.system("You are a concise assistant.")  # {"role": "system", "content": "..."}
Message.human("What's the weather in Oslo?")  # {"role": "user", "content": "..."}
Message.ai("It's 22°C and sunny.")  # {"role": "assistant", "content": "..."}
Message.tool(
    "22°C, sunny", name="get_weather"
)  # {"role": "tool", "name": "...", "content": "..."}
```

You only need the `tool_calls=`/`call_id=` forms yourself if you're building messages by hand
instead of going through `Agent` — it manages that round-trip for you.

## Agents as nodes

See [Graphs → Agents as nodes](graph.md#agents-as-nodes) for wiring multiple agents into a
multi-agent workflow.
