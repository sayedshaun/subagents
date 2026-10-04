"""requires_approval: a gated call defers until a human allows it."""

from dataclasses import dataclass

import pytest

from deepharness.agent import Agent, Toolbox, tool
from deepharness.errors import ConfigurationError, ToolDenied
from deepharness.graph import Graph
from deepharness.providers.base import CompletionResponse, ToolCall
from deepharness.tools import Permissions

from .test_agent import ScriptedProvider

SENT: list[int] = []


@tool(requires_approval=True)
def wire_transfer(amount_usd: int) -> str:
    """Send money."""
    SENT.append(amount_usd)
    return f"sent ${amount_usd}"


@tool
def balance() -> str:
    """Report the balance."""
    return "$1,000,000"


def transfer_turn(amount=50_000, call_id="t1"):
    return CompletionResponse(
        content="",
        tool_calls=[
            ToolCall(id=call_id, name="wire_transfer", arguments={"amount_usd": amount})
        ],
    )


@pytest.fixture(autouse=True)
def _clear():
    SENT.clear()


def test_the_decorator_records_the_gate_on_the_tool():
    assert wire_transfer._tool_spec.requires_approval is True
    assert balance._tool_spec.requires_approval is False


async def test_a_gated_call_pauses_without_running():
    provider = ScriptedProvider([transfer_turn()])
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay Acme $50,000")

    assert state.stop_reason == "paused"
    assert SENT == [], "the tool must not have run before approval"
    pending = state.paused[0]
    assert pending.name == "wire_transfer"
    assert pending.needs_approval
    assert pending.arguments == {"amount_usd": 50_000}


async def test_approving_actually_runs_the_call():
    provider = ScriptedProvider([transfer_turn(), CompletionResponse(content="Sent.")])
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay Acme $50,000")
    state = await agent.arun(state.approve())

    assert SENT == [50_000], "approval must run the gated tool"
    assert state.output == "Sent."
    assert state.answered
    assert any(
        m["role"] == "tool" and m["content"] == "sent $50000" for m in state.messages
    )


async def test_rejecting_tells_the_model_instead_of_running():
    provider = ScriptedProvider(
        [transfer_turn(), CompletionResponse(content="Understood, cancelled.")]
    )
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay Acme $50,000")
    state = await agent.arun(state.reject())

    assert SENT == []
    assert state.output == "Understood, cancelled."
    assert any(
        m["role"] == "tool" and m["content"] == "Denied by the user."
        for m in state.messages
    )


async def test_the_arguments_the_model_sent_are_the_ones_that_run():
    provider = ScriptedProvider(
        [transfer_turn(amount=25), CompletionResponse(content="done")]
    )
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay $25")
    await agent.arun(state.approve())

    assert SENT == [25]


async def test_resuming_without_deciding_is_an_error():
    provider = ScriptedProvider([transfer_turn()])
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay Acme $50,000")

    with pytest.raises(ConfigurationError, match="state.approve"):
        await agent.arun(state)


def test_approving_a_call_that_is_not_paused_is_an_error():
    from deepharness.agent import AgentState

    with pytest.raises(ConfigurationError, match="no paused call"):
        AgentState().approve()


async def test_one_call_can_be_approved_by_id_and_another_rejected():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="wire_transfer", arguments={"amount_usd": 1}),
                    ToolCall(id="b", name="wire_transfer", arguments={"amount_usd": 2}),
                ],
            ),
            CompletionResponse(content="one sent, one cancelled"),
        ]
    )
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay both")
    assert len(state.paused) == 2

    state.approve("a")
    state.reject("b")
    state = await agent.arun(state)

    assert SENT == [1]
    assert any(m["content"] == "Denied by the user." for m in state.messages)
    assert any(m["content"] == "sent $1" for m in state.messages)


async def test_an_ungated_tool_in_the_same_turn_waits_for_the_ruling():
    """A turn is not half-applied while a human is deciding."""
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="balance", arguments={}),
                    ToolCall(id="b", name="wire_transfer", arguments={"amount_usd": 5}),
                ],
            ),
            CompletionResponse(content="done"),
        ]
    )
    agent = Agent(provider, tools=[balance, wire_transfer])

    state = await agent.arun("check then pay")

    assert state.stop_reason == "paused"
    # The ungated tool did not run, but it was requested - so it is accounted
    # for, or the resumed transcript has a tool call no result answers.
    assert not any("$1,000,000" in m["content"] for m in state.messages)
    skipped = [m for m in state.messages if m["role"] == "tool"]
    assert [m["name"] for m in skipped] == ["balance"]
    assert skipped[0]["content"].startswith("Not run")


def test_approval_works_on_the_sync_path_too():
    provider = ScriptedProvider([transfer_turn(), CompletionResponse(content="Sent.")])
    agent = Agent(provider, tools=[wire_transfer])

    state = agent.run("pay Acme $50,000")
    state = agent.run(state.approve())

    assert SENT == [50_000]
    assert state.output == "Sent."


async def test_a_toolbox_that_can_ask_runs_the_call_without_pausing():
    provider = ScriptedProvider([transfer_turn(), CompletionResponse(content="Sent.")])
    asked = []

    def approve(name, arguments):
        asked.append((name, arguments))
        return True

    agent = Agent(provider, tools=Toolbox([wire_transfer], approve=approve))

    state = await agent.arun("pay Acme $50,000")

    assert asked == [("wire_transfer", {"amount_usd": 50_000})]
    assert SENT == [50_000]
    assert state.answered
    assert state.paused == []


async def test_a_toolbox_refusal_tells_the_model():
    provider = ScriptedProvider(
        [transfer_turn(), CompletionResponse(content="Understood, cancelled.")]
    )
    agent = Agent(
        provider, tools=Toolbox([wire_transfer], approve=lambda name, args: False)
    )

    state = await agent.arun("pay Acme $50,000")

    assert SENT == []
    assert state.output == "Understood, cancelled."
    assert any("ToolDenied" in m["content"] for m in state.messages)


async def test_ungated_calls_run_alongside_one_the_toolbox_approved():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="balance", arguments={}),
                    ToolCall(id="b", name="wire_transfer", arguments={"amount_usd": 5}),
                ],
            ),
            CompletionResponse(content="done"),
        ]
    )
    asked = []
    toolbox = Toolbox(
        [balance, wire_transfer], approve=lambda name, args: asked.append(name) or True
    )
    agent = Agent(provider, tools=toolbox)

    state = await agent.arun("check then pay")

    assert asked == ["wire_transfer"], "only the gated tool is asked about"
    assert SENT == [5]
    results = {
        m["tool_call_id"]: m["content"] for m in state.messages if m["role"] == "tool"
    }
    assert results == {"a": "$1,000,000", "b": "sent $5"}


async def test_a_permission_allow_is_not_asked_about_again():
    provider = ScriptedProvider([transfer_turn(), CompletionResponse(content="Sent.")])
    asked = []
    toolbox = Toolbox(
        [wire_transfer], approve=lambda name, args: asked.append(name) or True
    )
    agent = Agent(
        provider, tools=toolbox, permissions=Permissions(allow=[wire_transfer])
    )

    await agent.arun("pay Acme $50,000")

    assert asked == []
    assert SENT == [50_000]


async def test_a_permission_ask_on_an_ungated_tool_goes_to_the_approver():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="", tool_calls=[ToolCall(id="a", name="balance", arguments={})]
            ),
            CompletionResponse(content="done"),
        ]
    )
    toolbox = Toolbox([balance], approve=lambda name, args: False)
    agent = Agent(provider, tools=toolbox, permissions=Permissions(ask=[balance]))

    state = await agent.arun("check")

    assert any("ToolDenied" in m["content"] for m in state.messages)


def test_a_toolbox_that_can_ask_works_on_the_sync_path():
    provider = ScriptedProvider([transfer_turn(), CompletionResponse(content="Sent.")])
    agent = Agent(
        provider, tools=Toolbox([wire_transfer], approve=lambda name, args: True)
    )

    state = agent.run("pay Acme $50,000")

    assert SENT == [50_000]
    assert state.output == "Sent."


async def test_the_toolbox_asks_when_called_directly():
    toolbox = Toolbox(
        [wire_transfer, balance], approve=lambda name, args: args["amount_usd"] < 100
    )

    assert await toolbox.call("wire_transfer", amount_usd=10) == "sent $10"
    with pytest.raises(ToolDenied) as denied:
        await toolbox.call("wire_transfer", amount_usd=500)
    assert denied.value.name == "wire_transfer"
    assert denied.value.arguments == {"amount_usd": 500}
    assert toolbox.call_sync("balance") == "$1,000,000"
    assert SENT == [10]


async def test_an_async_approver_is_awaited():
    async def approve(name, arguments):
        return True

    toolbox = Toolbox([wire_transfer], approve=approve)

    assert await toolbox.call("wire_transfer", amount_usd=1) == "sent $1"


def test_an_async_approver_is_refused_on_the_sync_path():
    async def approve(name, arguments):
        return True

    toolbox = Toolbox([wire_transfer], approve=approve)

    with pytest.raises(ConfigurationError, match="async"):
        toolbox.call_sync("wire_transfer", amount_usd=1)
    assert SENT == []


def test_a_toolbox_without_an_approver_does_not_gate():
    """The agent pauses before the call reaches it; the toolbox just runs it."""
    assert (
        Toolbox([wire_transfer]).call_sync("wire_transfer", amount_usd=1) == "sent $1"
    )


async def test_a_graph_node_calling_a_gated_tool_is_asked():
    toolbox = Toolbox([wire_transfer], approve=lambda name, args: False)

    @dataclass
    class Payment:
        result: str = ""

    graph = Graph(Payment)

    @graph.add(start=True, end=True)
    async def pay(state: Payment) -> Payment:
        try:
            state.result = await toolbox.call("wire_transfer", amount_usd=50)
        except ToolDenied:
            state.result = "skipped"
        return state

    state = await graph.build().run()

    assert state.result == "skipped"
    assert SENT == []
