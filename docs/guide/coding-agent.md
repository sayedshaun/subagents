# Building a coding agent

An agent that reads a codebase and changes it, assembled one decision at a time. Each stage
runs; each one fixes a problem the stage before it has.

```python
import asyncio

from deepharness import Agent, OpenAI
from deepharness.tools import file_tools

agent = Agent(
    OpenAI("gpt-4o-mini"),
    tools=file_tools("."),
    system="Work in the repository you are given. Read before you edit.",
)

state = asyncio.run(agent.arun("What does the executor do?"))
print(state.output)
```

That already works: `file_tools()` gives the model `read_file`, `list_files`, `search_files`,
`write_file` and `edit_file`, every path resolved inside the directory you named. A path that
climbs out of it raises `OutsideWorkspace`, which reaches the model as that call's result — so
it corrects itself, and the read never happens.

Two things are wrong with it, and the rest of this page is fixing them: it can edit your files
with no say from you, and it will eventually send a request larger than the model's context
window.

## Read-only first

The strongest guarantee is not a rule, it is an absent tool. If the job is understanding rather
than changing, hand over the three read-only tools and there is nothing left to rule on:

```python
agent = Agent(llm, tools=file_tools(".", writable=False))
```

`read_file` numbers lines and stops at `max_bytes`, telling the model which offset to continue
from — without that, one read of a large file fills the window and every later turn pays for it
again.

## A shell, and why it stops

```python
from deepharness.tools import file_tools, shell_tool

agent = Agent(llm, tools=[*file_tools("."), shell_tool(".")], system=SYSTEM)
```

`run_command` runs through a shell — pipes and redirection are the point — with the workspace
as its working directory. It is **not a sandbox**: the workspace bounds where a command starts,
not what it can reach. So it arrives gated, as do `write_file` and `edit_file`: the run stops
before the call and waits.

```python
state = await agent.arun("Add a docstring to Executor.run")

print(state.stop_reason)  # "paused"
print(state.paused[0].name)  # "edit_file"
print(state.paused[0].question)  # Run edit_file with {'path': ..., 'old': ...}?
```

Nothing else in that turn ran either. A turn half-applied while a human is deciding would be
worse than waiting, so the other calls are recorded as not run and the whole turn waits.

## Ruling on it

`approve()` runs the call with the arguments the model sent; `reject()` tells it no. Both
return the state, so resuming is one line:

```python
state = await agent.arun(state.approve())  # or state.reject()
```

A loop around that is the whole of a terminal coding agent:

```python
from deepharness import AgentState


async def drive(agent: Agent, prompt: str) -> AgentState:
    state = await agent.arun(prompt)
    while state.stop_reason == "paused":
        for call in state.paused:
            print(f"\n{call.question}")
            allowed = input("[y/N] ").strip().lower() == "y"
            state.approve(call.call_id) if allowed else state.reject(call.call_id)
        state = await agent.arun(state)
    return state
```

## Deciding without being asked every time

Being asked about every `run_command` gets old, and `requires_approval` cannot tell `git log`
from `git push` — they are the same tool. `Permissions` decides per call, from the arguments
the model actually sent:

```python
from deepharness.tools import Permissions, Rule, ToolName, file_tools, shell_tool

permissions = Permissions(
    allow=[
        ToolName.READ_FILE,
        ToolName.LIST_FILES,
        ToolName.SEARCH_FILES,
        Rule(ToolName.RUN_COMMAND, {"command": "git log*"}),
        Rule(ToolName.RUN_COMMAND, {"command": "pytest*"}),
    ],
    ask=[ToolName.WRITE_FILE, ToolName.EDIT_FILE, ToolName.RUN_COMMAND],
    deny=[
        Rule(ToolName.RUN_COMMAND, {"command": "*rm -rf*"}),
        Rule(ToolName.RUN_COMMAND, {"command": "*git push*"}),
    ],
)

agent = Agent(llm, tools=[*file_tools("."), shell_tool(".")], permissions=permissions)
```

`deny` beats `allow` beats `ask`, so widening a policy can never quietly override a refusal
already written down. A denied call never runs and the model is told, which lets it find
another way rather than retrying. A call no rule matches falls back to the tool's own
`requires_approval` — so a policy narrows what tools already declare instead of replacing it.

A deny or ask rule naming a tool the agent does not have is refused at construction: a
misspelled deny matches nothing, and a deny that matches nothing allows exactly what it was
written to stop.

## Keeping it inside the window

A coding agent reads files, and every result it reads is re-sent with every later turn.
`ContextPolicy` bounds both ends of that:

```python
from deepharness.agent import ContextPolicy

agent = Agent(
    llm,
    tools=[*file_tools("."), shell_tool(".")],
    permissions=permissions,
    context=ContextPolicy(max_tokens=100_000, tool_result_chars=8_000),
)
```

`tool_result_chars` truncates each result as it is recorded, eliding the middle and saying how
much went missing. `max_tokens` prunes the view the model is sent — oldest turns first, with a
note in their place — while `state.messages` keeps every message. Pruning never drops the
system prompt, never orphans a tool result, and always keeps the most recent turns.

## Watching it work

Most of a long run is tool calls the model never narrates, so text alone makes it look stalled:

```python
from deepharness import Finished, TextDelta
from deepharness.agent import StepStarted, ToolFinished, ToolStarted

async for event in agent.astream_events("Add a docstring to Executor.run"):
    match event:
        case StepStarted(step):
            print(f"\n— step {step}")
        case ToolStarted(name, arguments, _):
            print(f"{name}({arguments}) …")
        case ToolFinished(name, result, failed, _):
            print(f"{name} {'failed' if failed else 'ok'}: {result[:70]}")
        case TextDelta(text):
            print(text, end="", flush=True)
        case Finished(state):
            print(f"\nstopped because: {state.stop_reason}")
```

`ToolFinished.result` is the text the model will read, truncation included, so what you show
cannot drift from what it saw. A thinking model's reasoning arrives as `ThinkingDelta` and is
worth rendering differently — it is the model's working, not its answer.

## Stopping for the day

The pause is a returned state, not an exception, so the whole run survives a process boundary —
approval included:

```python
import json
from pathlib import Path

from deepharness import AgentState

Path("run.json").write_text(json.dumps(state.to_dict()))

# tomorrow, in another process
state = AgentState.from_dict(json.loads(Path("run.json").read_text()))
if state.stop_reason == "paused":
    state = await agent.arun(state.approve())
```

## All of it

```python
import asyncio
import json
from pathlib import Path

from deepharness import Agent, AgentState, OpenAI
from deepharness.agent import ContextPolicy
from deepharness.tools import Permissions, Rule, ToolName, file_tools, shell_tool

SYSTEM = """You are a coding agent working inside one repository.

Read before you change anything: list_files and search_files to find your way,
read_file before an edit, edit_file for part of a file and write_file for a new
one, run_command for anything else. Keep prose short - the tool calls show your
work."""

SESSION = Path("run.json")


def build(root: str = ".") -> Agent:
    return Agent(
        OpenAI("gpt-4o-mini"),
        tools=[*file_tools(root), shell_tool(root)],
        system=SYSTEM,
        context=ContextPolicy(max_tokens=100_000, tool_result_chars=8_000),
        permissions=Permissions(
            allow=[
                ToolName.READ_FILE,
                ToolName.LIST_FILES,
                ToolName.SEARCH_FILES,
                Rule(ToolName.RUN_COMMAND, {"command": "git log*"}),
                Rule(ToolName.RUN_COMMAND, {"command": "pytest*"}),
            ],
            ask=[ToolName.WRITE_FILE, ToolName.EDIT_FILE, ToolName.RUN_COMMAND],
            deny=[Rule(ToolName.RUN_COMMAND, {"command": "*rm -rf*"})],
        ),
    )


async def drive(agent: Agent, state: AgentState) -> AgentState:
    """Run until it answers, asking about anything the policy gates."""
    state = await agent.arun(state)
    while state.stop_reason == "paused":
        for call in state.paused:
            print(f"\n{call.question}")
            if input("[y/N] ").strip().lower() == "y":
                state.approve(call.call_id)
            else:
                state.reject(call.call_id)
        state = await agent.arun(state)
    return state


async def main() -> None:
    agent = build(".")
    state = (
        AgentState.from_dict(json.loads(SESSION.read_text()))
        if SESSION.exists()
        else AgentState()
    )
    while (prompt := input("\n> ").strip()) not in ("", "quit"):
        state.messages.append({"role": "user", "content": prompt})
        state = await drive(agent, state)
        print(f"\n{state.output}")
        SESSION.write_text(json.dumps(state.to_dict()))


asyncio.run(main())
```

Sixty lines, and every decision in it is visible: which tools exist, what runs unasked, what
stops, how big a request may get, and where the conversation lives between runs.
