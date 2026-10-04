"""A run stored as data and picked up again, pause and all."""

import json

import pytest

from deepharness.agent import (
    Agent,
    AgentState,
    Message,
    PendingHumanInput,
    tool,
)
from deepharness.errors import ConfigurationError
from deepharness.providers.base import LLM, CompletionResponse, TokenUsage, ToolCall


class Scripted(LLM):
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def agenerate(self, messages, *, tools=None):
        return self.generate(messages, tools=tools)

    def generate(self, messages, *, tools=None):
        self.calls += 1
        return self.responses[self.calls - 1]


@tool(requires_approval=True)
def deploy(target: str) -> str:
    """Ship it, once a human says so."""
    return f"deployed to {target}"


def round_trip(state: AgentState) -> AgentState:
    """Through JSON text, as any store the caller picks would see it."""
    return AgentState.from_dict(json.loads(json.dumps(state.to_dict())))


def test_messages_round_trip():
    messages = [
        Message.system("be helpful").to_dict(),
        Message.human("hi").to_dict(),
        Message.ai("hello!").to_dict(),
    ]

    assert round_trip(AgentState(messages=messages)).messages == messages


def test_a_bare_message_list_builds_a_state():
    assert AgentState.of([{"role": "user", "content": "hi"}]).messages == [
        {"role": "user", "content": "hi"}
    ]


def test_usage_and_stop_reason_survive():
    state = AgentState(
        messages=[Message.human("hi").to_dict()],
        output="hello",
        usage=TokenUsage(3, 4, 7),
        stop_reason="answer",
    )

    assert round_trip(state) == state


def test_structured_output_is_stored_as_plain_data():
    from dataclasses import dataclass

    @dataclass
    class Answer:
        city: str

    assert round_trip(AgentState(output=Answer(city="Oslo"))).output == {"city": "Oslo"}


def test_a_run_paused_on_an_approval_resumes_from_stored_data():
    model = Scripted(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(name="deploy", arguments={"target": "prod"}, id="1")
                ],
            ),
            CompletionResponse(content="shipped"),
        ]
    )
    agent = Agent(model, tools=[deploy])

    paused = agent.run("deploy to prod")
    assert paused.stop_reason == "paused"

    # A different process: the pause has to survive serialization, not memory.
    resumed = round_trip(paused)
    assert resumed.paused == [
        PendingHumanInput(
            call_id="1",
            name="deploy",
            question="Run deploy with {'target': 'prod'}?",
            arguments={"target": "prod"},
        )
    ]

    state = agent.run(resumed.approve())

    assert state.answered
    assert "deployed to prod" in state.messages[-2]["content"]


def test_data_with_an_unexpected_key_is_an_error():
    with pytest.raises(ConfigurationError, match="unknown state keys"):
        AgentState.from_dict({"messages": [], "temperature": 0.5})
