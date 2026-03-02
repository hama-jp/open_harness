"""Tests for ConsoleRenderer — EventBus subscriber."""

from __future__ import annotations

import pytest
from io import StringIO
from rich.console import Console

from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType
from open_harness_v2.ui.renderer import ConsoleRenderer


def _make_console() -> tuple[Console, StringIO]:
    """Create a Console that writes to a StringIO buffer (no ANSI codes)."""
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=120, highlight=False)
    return console, buf


@pytest.fixture
def renderer_setup():
    """Create renderer + event bus wired together."""
    console, buf = _make_console()
    bus = EventBus()
    renderer = ConsoleRenderer(console, verbose=True)
    renderer.attach(bus)
    return bus, console, buf


@pytest.mark.anyio
async def test_agent_started(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "Hello world"}))
    output = buf.getvalue()
    assert "Goal:" in output
    assert "Hello world" in output


@pytest.mark.anyio
async def test_thinking(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(type=EventType.LLM_THINKING, data={"thinking": "Let me consider..."}))
    output = buf.getvalue()
    assert "thinking:" in output
    assert "Let me consider" in output


@pytest.mark.anyio
async def test_tool_executing(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(
        type=EventType.TOOL_EXECUTING,
        data={"tool": "read_file", "args": {"path": "/tmp/test.txt"}},
    ))
    output = buf.getvalue()
    assert "read_file" in output


@pytest.mark.anyio
async def test_tool_executed_success(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(
        type=EventType.TOOL_EXECUTED,
        data={"tool": "read_file", "success": True, "output": "file contents here"},
    ))
    output = buf.getvalue()
    assert "read_file" in output
    assert "file contents here" in output


@pytest.mark.anyio
async def test_tool_error(renderer_setup):
    """TOOL_ERROR events should render via the dedicated error handler."""
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(
        type=EventType.TOOL_ERROR,
        data={"tool": "write_file", "success": False, "error": "permission denied"},
    ))
    output = buf.getvalue()
    assert "failed" in output
    assert "permission denied" in output


@pytest.mark.anyio
async def test_policy_violation(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(
        type=EventType.POLICY_VIOLATION,
        data={"message": "Write blocked", "rule": "denied_path"},
    ))
    output = buf.getvalue()
    assert "Policy:" in output
    assert "Write blocked" in output


@pytest.mark.anyio
async def test_reasoner_decision_verbose(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(
        type=EventType.REASONER_DECISION,
        data={"step": 3, "action": "execute_tools"},
    ))
    output = buf.getvalue()
    assert "Step 3" in output
    assert "execute_tools" in output


@pytest.mark.anyio
async def test_reasoner_decision_not_verbose():
    """Non-verbose mode should not show reasoner decisions."""
    console, buf = _make_console()
    bus = EventBus()
    renderer = ConsoleRenderer(console, verbose=False)
    renderer.attach(bus)

    await bus.emit(AgentEvent(
        type=EventType.REASONER_DECISION,
        data={"step": 1, "action": "respond"},
    ))
    output = buf.getvalue()
    assert "Step 1" not in output


@pytest.mark.anyio
async def test_llm_response_verbose(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(
        type=EventType.LLM_RESPONSE,
        data={"model": "qwen3-8b", "latency_ms": 1234.5},
    ))
    output = buf.getvalue()
    assert "qwen3-8b" in output
    assert "1234" in output or "1235" in output


@pytest.mark.anyio
async def test_llm_response_not_verbose():
    """Non-verbose mode should not show LLM response details."""
    console, buf = _make_console()
    bus = EventBus()
    renderer = ConsoleRenderer(console, verbose=False)
    renderer.attach(bus)

    await bus.emit(AgentEvent(
        type=EventType.LLM_RESPONSE,
        data={"model": "qwen3-8b", "latency_ms": 500},
    ))
    output = buf.getvalue()
    assert output.strip() == ""


@pytest.mark.anyio
async def test_agent_done(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(
        type=EventType.AGENT_DONE,
        data={"response": "The answer is 4.", "steps": 2},
    ))
    output = buf.getvalue()
    assert "The answer is 4" in output
    assert "2 step" in output


@pytest.mark.anyio
async def test_agent_error(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(
        type=EventType.AGENT_ERROR,
        data={"error": "Connection refused"},
    ))
    output = buf.getvalue()
    assert "Error:" in output
    assert "Connection refused" in output


@pytest.mark.anyio
async def test_agent_cancelled(renderer_setup):
    bus, console, buf = renderer_setup
    await bus.emit(AgentEvent(type=EventType.AGENT_CANCELLED, data={}))
    output = buf.getvalue()
    assert "Cancelled" in output


@pytest.mark.anyio
async def test_detach():
    """After detach, events should not be rendered."""
    console, buf = _make_console()
    bus = EventBus()
    renderer = ConsoleRenderer(console, verbose=True)
    renderer.attach(bus)

    await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "before"}))
    assert "before" in buf.getvalue()

    renderer.detach(bus)
    await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "after"}))
    assert "after" not in buf.getvalue()


@pytest.mark.anyio
async def test_long_output_truncated(renderer_setup):
    """Very long tool output should be truncated in the display."""
    bus, console, buf = renderer_setup
    long_output = "x" * 5000
    await bus.emit(AgentEvent(
        type=EventType.TOOL_EXECUTED,
        data={"tool": "shell", "success": True, "output": long_output},
    ))
    output = buf.getvalue()
    assert "5000 chars total" in output


@pytest.mark.anyio
async def test_thinking_truncated(renderer_setup):
    """Long thinking text should be truncated."""
    bus, console, buf = renderer_setup
    long_thinking = "a" * 500
    await bus.emit(AgentEvent(
        type=EventType.LLM_THINKING,
        data={"thinking": long_thinking},
    ))
    output = buf.getvalue()
    assert "..." in output
