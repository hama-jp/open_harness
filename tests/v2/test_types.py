"""Tests for Open Harness v2 shared types."""

from open_harness_v2.types import (
    AgentEvent,
    EventType,
    LLMResponse,
    ToolCall,
    ToolParameter,
    ToolResult,
)


class TestToolCall:
    def test_basic(self):
        tc = ToolCall(name="shell", arguments={"command": "ls"})
        assert tc.name == "shell"
        assert tc.arguments == {"command": "ls"}
        assert tc.raw == ""

    def test_with_raw(self):
        tc = ToolCall(name="read_file", arguments={"path": "x.py"}, raw='{"tool":"read_file"}')
        assert tc.raw == '{"tool":"read_file"}'


class TestToolResult:
    def test_success_message(self):
        r = ToolResult(success=True, output="file contents")
        assert r.to_message() == "file contents"

    def test_error_message_with_output(self):
        r = ToolResult(success=False, output="partial", error="timeout")
        assert "[Tool Error] timeout" in r.to_message()
        assert "partial" in r.to_message()

    def test_error_message_no_output(self):
        r = ToolResult(success=False, output="", error="not found")
        assert r.to_message() == "[Tool Error] not found"


class TestToolParameter:
    def test_defaults(self):
        p = ToolParameter(name="path", type="string", description="File path")
        assert p.required is True
        assert p.default is None
        assert p.enum is None


class TestLLMResponse:
    def test_has_tool_calls_empty(self):
        r = LLMResponse(content="hello")
        assert r.has_tool_calls is False

    def test_has_tool_calls_present(self):
        tc = ToolCall(name="shell", arguments={})
        r = LLMResponse(tool_calls=[tc])
        assert r.has_tool_calls is True

    def test_defaults(self):
        r = LLMResponse()
        assert r.content == ""
        assert r.thinking == ""
        assert r.finish_reason == ""
        assert r.usage == {}
        assert r.model == ""
        assert r.latency_ms == 0


class TestAgentEvent:
    def test_creation(self):
        ev = AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"})
        assert ev.type == EventType.AGENT_STARTED
        assert ev.data["goal"] == "test"
        assert ev.timestamp > 0

    def test_default_data(self):
        ev = AgentEvent(type=EventType.AGENT_DONE)
        assert ev.data == {}


class TestEventType:
    def test_values(self):
        assert EventType.AGENT_STARTED.value == "agent.started"
        assert EventType.TOOL_EXECUTED.value == "tool.executed"
        assert EventType.LLM_STREAMING.value == "llm.streaming"
