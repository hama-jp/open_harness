"""Tests for the Executor."""

import pytest

from open_harness_v2.core.executor import Executor, ExecutionResult
from open_harness_v2.events.bus import EventBus
from open_harness_v2.policy.engine import PolicyEngine
from open_harness_v2.config import PolicySpec
from open_harness_v2.tools.base import Tool
from open_harness_v2.tools.registry import ToolRegistry
from open_harness_v2.types import AgentEvent, EventType, ToolCall, ToolParameter, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "Echo input"
    parameters = [ToolParameter(name="text", type="string", description="Text to echo")]
    max_output = 5000

    async def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        return ToolResult(success=True, output=f"Echo: {text}")


class FailTool(Tool):
    name = "fail"
    description = "Always fails"
    parameters = []
    max_output = 5000

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=False, output="", error="Always fails")


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(FailTool())
    return reg


@pytest.fixture
def event_bus():
    return EventBus()


class TestSequentialExecution:
    async def test_single_tool(self, registry, event_bus):
        executor = Executor(registry, event_bus=event_bus)
        result = await executor.execute([
            ToolCall(name="echo", arguments={"text": "hello"}),
        ])
        assert len(result.results) == 1
        tc, tr = result.results[0]
        assert tc.name == "echo"
        assert tr.success is True
        assert "Echo: hello" in tr.output

    async def test_multiple_tools(self, registry, event_bus):
        executor = Executor(registry, event_bus=event_bus)
        result = await executor.execute([
            ToolCall(name="echo", arguments={"text": "a"}),
            ToolCall(name="echo", arguments={"text": "b"}),
        ])
        assert len(result.results) == 2
        assert result.all_succeeded is True

    async def test_failing_tool(self, registry, event_bus):
        executor = Executor(registry, event_bus=event_bus)
        result = await executor.execute([
            ToolCall(name="fail", arguments={}),
        ])
        assert result.all_succeeded is False

    async def test_unknown_tool(self, registry, event_bus):
        executor = Executor(registry, event_bus=event_bus)
        result = await executor.execute([
            ToolCall(name="nonexistent", arguments={}),
        ])
        assert result.all_succeeded is False
        assert "Unknown tool" in result.results[0][1].error

    async def test_events_emitted(self, registry, event_bus):
        events = []
        event_bus.subscribe("*", lambda e: events.append(e))

        executor = Executor(registry, event_bus=event_bus)
        await executor.execute([
            ToolCall(name="echo", arguments={"text": "hello"}),
        ])

        event_types = [e.type for e in events]
        assert EventType.TOOL_EXECUTING in event_types
        assert EventType.TOOL_EXECUTED in event_types


class TestPolicyChecks:
    async def test_policy_violation(self, registry, event_bus):
        policy = PolicyEngine(PolicySpec(disabled_tools=["echo"]))
        executor = Executor(registry, policy, event_bus)

        result = await executor.execute([
            ToolCall(name="echo", arguments={"text": "test"}),
        ])

        assert len(result.violations) == 1
        assert result.violations[0][1].rule == "disabled_tool"
        # Violation should also appear in results as a failed ToolResult
        assert not result.all_succeeded

    async def test_budget_tracking(self, registry, event_bus):
        policy = PolicyEngine(PolicySpec())
        executor = Executor(registry, policy, event_bus)

        await executor.execute([
            ToolCall(name="echo", arguments={"text": "test"}),
        ])

        # echo is not in TOOL_CATEGORIES so budget won't increment specific counters
        # but it should have been recorded
        assert sum(policy.budget.tool_calls.values()) == 1


class TestConcurrentExecution:
    async def test_concurrent_tools(self, registry, event_bus):
        executor = Executor(registry, event_bus=event_bus)
        result = await executor.execute(
            [
                ToolCall(name="echo", arguments={"text": "a"}),
                ToolCall(name="echo", arguments={"text": "b"}),
            ],
            concurrent=True,
        )
        assert len(result.results) == 2
        assert result.all_succeeded is True

    async def test_concurrent_with_policy_violation(self, registry, event_bus):
        policy = PolicyEngine(PolicySpec(disabled_tools=["fail"]))
        executor = Executor(registry, policy, event_bus)

        result = await executor.execute(
            [
                ToolCall(name="echo", arguments={"text": "ok"}),
                ToolCall(name="fail", arguments={}),
            ],
            concurrent=True,
        )

        # echo should succeed, fail should be blocked by policy
        assert len(result.violations) == 1
        success_count = sum(1 for _, r in result.results if r.success)
        assert success_count == 1


class TestExecutionResult:
    def test_all_succeeded_empty(self):
        r = ExecutionResult()
        assert r.all_succeeded is True

    def test_all_succeeded_with_violations(self):
        r = ExecutionResult()
        r.violations.append((ToolCall(name="x", arguments={}), None))
        assert r.all_succeeded is False
