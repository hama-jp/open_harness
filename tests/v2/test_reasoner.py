"""Tests for the Reasoner."""

import pytest

from open_harness_v2.core.reasoner import ActionType, Reasoner, ReasonerDecision
from open_harness_v2.types import LLMResponse, ToolCall


class TestReasonerDecisions:
    def test_tool_calls_execute(self):
        reasoner = Reasoner()
        response = LLMResponse(
            content='{"tool": "shell", "args": {"command": "ls"}}',
            tool_calls=[ToolCall(name="shell", arguments={"command": "ls"})],
        )
        decision = reasoner.decide(response)
        assert decision.action == ActionType.EXECUTE_TOOLS
        assert len(decision.tool_calls) == 1
        assert decision.tool_calls[0].name == "shell"

    def test_text_response_done(self):
        reasoner = Reasoner()
        response = LLMResponse(content="Here's the answer: 42")
        decision = reasoner.decide(response)
        assert decision.action == ActionType.RESPOND
        assert "42" in decision.response_text

    def test_error_response(self):
        reasoner = Reasoner()
        response = LLMResponse(
            content="[LLM API Error: timeout]",
            finish_reason="error",
        )
        decision = reasoner.decide(response)
        assert decision.action == ActionType.ERROR

    def test_empty_response(self):
        reasoner = Reasoner()
        response = LLMResponse()
        decision = reasoner.decide(response)
        assert decision.action == ActionType.ERROR
        assert "Empty response" in decision.error

    def test_thinking_preserved(self):
        reasoner = Reasoner()
        response = LLMResponse(
            content="Answer: 42",
            thinking="Let me think about this...",
        )
        decision = reasoner.decide(response)
        assert decision.thinking == "Let me think about this..."


class TestStepLimit:
    def test_step_limit_reached(self):
        reasoner = Reasoner(max_steps=2)

        # Step 1: OK
        r1 = LLMResponse(
            content="tool call",
            tool_calls=[ToolCall(name="shell", arguments={})],
        )
        d1 = reasoner.decide(r1)
        assert d1.action == ActionType.EXECUTE_TOOLS

        # Step 2: OK
        r2 = LLMResponse(
            content="tool call",
            tool_calls=[ToolCall(name="shell", arguments={})],
        )
        d2 = reasoner.decide(r2)
        assert d2.action == ActionType.EXECUTE_TOOLS

        # Step 3: over limit
        r3 = LLMResponse(content="more work")
        d3 = reasoner.decide(r3)
        assert d3.action == ActionType.ERROR
        assert "Step limit" in d3.error

    def test_step_count_tracking(self):
        reasoner = Reasoner()
        assert reasoner.step_count == 0
        reasoner.decide(LLMResponse(content="hello"))
        assert reasoner.step_count == 1

    def test_reset(self):
        reasoner = Reasoner()
        reasoner.decide(LLMResponse(content="hello"))
        assert reasoner.step_count == 1
        reasoner.reset()
        assert reasoner.step_count == 0
