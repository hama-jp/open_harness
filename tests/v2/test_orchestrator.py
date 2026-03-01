"""End-to-end integration tests for the Orchestrator.

Uses mocked LLM + tools to prove the full loop works:
1. Mock LLM returns tool call → Reasoner decides EXECUTE_TOOLS → Executor runs tool → context updated
2. Mock LLM returns text → Reasoner decides RESPOND → loop ends with DONE event
3. Cancel signal → loop terminates gracefully
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from open_harness_v2.config import HarnessConfig, ProfileSpec
from open_harness_v2.core.context import AgentContext
from open_harness_v2.core.orchestrator import Orchestrator
from open_harness_v2.events.bus import EventBus
from open_harness_v2.llm.client import AsyncLLMClient
from open_harness_v2.llm.middleware import LLMRequest, MiddlewarePipeline
from open_harness_v2.llm.router import ModelRouter
from open_harness_v2.policy.engine import PolicyEngine
from open_harness_v2.config import PolicySpec
from open_harness_v2.tools.base import Tool
from open_harness_v2.tools.registry import ToolRegistry
from open_harness_v2.types import (
    AgentEvent,
    EventType,
    LLMResponse,
    ToolCall,
    ToolParameter,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------

class MockFileTool(Tool):
    name = "read_file"
    description = "Read a file"
    parameters = [ToolParameter(name="path", type="string", description="File path")]
    max_output = 5000

    async def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        return ToolResult(success=True, output=f"Contents of {path}: hello world")


class MockShellTool(Tool):
    name = "shell"
    description = "Run shell command"
    parameters = [ToolParameter(name="command", type="string", description="Command")]
    max_output = 5000

    async def execute(self, **kwargs) -> ToolResult:
        cmd = kwargs.get("command", "")
        return ToolResult(success=True, output=f"$ {cmd}\nOK")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(MockFileTool())
    reg.register(MockShellTool())
    return reg


def _make_router(config: HarnessConfig | None = None) -> ModelRouter:
    config = config or HarnessConfig(
        profiles={"local": ProfileSpec(models=["test-model"])},
    )
    # We won't use the real client — pipeline will be mocked
    return ModelRouter(config, client=AsyncMock(spec=AsyncLLMClient))


def _make_mock_pipeline(responses: list[LLMResponse]) -> MiddlewarePipeline:
    """Create a pipeline that returns responses from the list in order."""
    call_count = {"n": 0}

    async def mock_execute(request: LLMRequest) -> LLMResponse:
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    pipeline = AsyncMock(spec=MiddlewarePipeline)
    pipeline.execute = mock_execute
    return pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolCallThenRespond:
    """Test: LLM returns tool call → tools execute → LLM returns text → done."""

    async def test_full_loop(self):
        registry = _make_registry()
        router = _make_router()
        event_bus = EventBus()

        # Track events
        events: list[AgentEvent] = []
        event_bus.subscribe("*", lambda e: events.append(e))

        # Mock LLM: first call returns tool call, second returns text
        responses = [
            LLMResponse(
                content='{"tool": "read_file", "args": {"path": "main.py"}}',
                tool_calls=[ToolCall(name="read_file", arguments={"path": "main.py"})],
                usage={"total_tokens": 100},
            ),
            LLMResponse(
                content="I read the file. It contains hello world.",
                usage={"total_tokens": 80},
            ),
        ]
        pipeline = _make_mock_pipeline(responses)

        orchestrator = Orchestrator(
            router=router,
            registry=registry,
            event_bus=event_bus,
            pipeline=pipeline,
            max_steps=10,
        )

        result = await orchestrator.run("Read main.py and tell me what's in it")

        # Should return the final text response
        assert "hello world" in result

        # Check events
        event_types = [e.type for e in events]
        assert EventType.AGENT_STARTED in event_types
        assert EventType.LLM_RESPONSE in event_types
        assert EventType.REASONER_DECISION in event_types
        assert EventType.TOOL_EXECUTING in event_types
        assert EventType.TOOL_EXECUTED in event_types
        assert EventType.AGENT_DONE in event_types


class TestDirectTextResponse:
    """Test: LLM returns text immediately → done."""

    async def test_immediate_response(self):
        registry = _make_registry()
        router = _make_router()

        responses = [
            LLMResponse(content="The answer is 42."),
        ]
        pipeline = _make_mock_pipeline(responses)

        orchestrator = Orchestrator(
            router=router,
            registry=registry,
            pipeline=pipeline,
        )

        result = await orchestrator.run("What is the answer?")
        assert "42" in result


class TestMultipleToolCalls:
    """Test: Multiple tool calls in sequence."""

    async def test_two_tools_then_respond(self):
        registry = _make_registry()
        router = _make_router()

        responses = [
            # First: shell tool
            LLMResponse(
                content='shell command',
                tool_calls=[ToolCall(name="shell", arguments={"command": "ls"})],
                usage={"total_tokens": 50},
            ),
            # Second: read_file tool
            LLMResponse(
                content='read file',
                tool_calls=[ToolCall(name="read_file", arguments={"path": "x.py"})],
                usage={"total_tokens": 50},
            ),
            # Third: final response
            LLMResponse(content="Done! Found the files."),
        ]
        pipeline = _make_mock_pipeline(responses)

        orchestrator = Orchestrator(
            router=router,
            registry=registry,
            pipeline=pipeline,
            max_steps=10,
        )

        result = await orchestrator.run("List files and read x.py")
        assert "Done" in result


class TestCancellation:
    """Test: Cancel signal terminates the loop."""

    async def test_cancel_during_execution(self):
        registry = _make_registry()
        router = _make_router()
        event_bus = EventBus()

        events: list[AgentEvent] = []
        event_bus.subscribe("*", lambda e: events.append(e))

        # Mock pipeline that introduces a small delay (simulating real LLM call)
        call_count = {"n": 0}

        async def slow_execute(request: LLMRequest) -> LLMResponse:
            call_count["n"] += 1
            await asyncio.sleep(0.01)  # simulate LLM latency
            return LLMResponse(
                content="tool",
                tool_calls=[ToolCall(name="shell", arguments={"command": "echo hi"})],
            )

        pipeline = AsyncMock(spec=MiddlewarePipeline)
        pipeline.execute = slow_execute

        orchestrator = Orchestrator(
            router=router,
            registry=registry,
            event_bus=event_bus,
            pipeline=pipeline,
            max_steps=1000,
        )

        # Cancel after a short delay
        async def cancel_soon():
            await asyncio.sleep(0.05)
            orchestrator.cancel()

        cancel_task = asyncio.create_task(cancel_soon())
        result = await orchestrator.run("Keep going forever")
        await cancel_task

        assert "cancelled" in result.lower()

        event_types = [e.type for e in events]
        assert EventType.AGENT_CANCELLED in event_types

        # Should have stopped well before the step limit
        assert call_count["n"] < 20


class TestStepLimit:
    """Test: Step limit stops the loop."""

    async def test_step_limit_hit(self):
        registry = _make_registry()
        router = _make_router()

        # LLM keeps returning tool calls
        tool_response = LLMResponse(
            content="tool",
            tool_calls=[ToolCall(name="shell", arguments={"command": "echo"})],
        )
        pipeline = _make_mock_pipeline([tool_response])

        orchestrator = Orchestrator(
            router=router,
            registry=registry,
            pipeline=pipeline,
            max_steps=3,
        )

        result = await orchestrator.run("Do a lot of work")
        assert "Step limit" in result or "error" in result.lower()


class TestPolicyIntegration:
    """Test: Policy violations are handled."""

    async def test_policy_blocks_tool(self):
        registry = _make_registry()
        router = _make_router()
        policy = PolicyEngine(PolicySpec(disabled_tools=["shell"]))

        responses = [
            # LLM tries to use shell (blocked)
            LLMResponse(
                content="use shell",
                tool_calls=[ToolCall(name="shell", arguments={"command": "rm -rf /"})],
            ),
            # After seeing the error, LLM responds with text
            LLMResponse(content="I can't run that command."),
        ]
        pipeline = _make_mock_pipeline(responses)

        orchestrator = Orchestrator(
            router=router,
            registry=registry,
            policy=policy,
            pipeline=pipeline,
            max_steps=10,
        )

        result = await orchestrator.run("Delete everything")
        assert "can't" in result.lower() or "cannot" in result.lower()


class TestTokenBudget:
    """Test: Token budget causes loop to stop."""

    async def test_token_budget_exceeded(self):
        registry = _make_registry()
        router = _make_router()
        policy = PolicyEngine(PolicySpec(max_tokens_per_goal=100))

        responses = [
            LLMResponse(
                content="tool",
                tool_calls=[ToolCall(name="shell", arguments={"command": "echo"})],
                usage={"total_tokens": 150},  # exceeds budget
            ),
            LLMResponse(content="more"),
        ]
        pipeline = _make_mock_pipeline(responses)

        orchestrator = Orchestrator(
            router=router,
            registry=registry,
            policy=policy,
            pipeline=pipeline,
        )

        result = await orchestrator.run("Work")
        assert "budget" in result.lower() or "token" in result.lower()


class TestCustomContext:
    """Test: Pre-built context is used."""

    async def test_custom_context(self):
        registry = _make_registry()
        router = _make_router()

        responses = [
            LLMResponse(content="Done with custom context."),
        ]
        pipeline = _make_mock_pipeline(responses)

        ctx = AgentContext()
        ctx.system.role = "You are a custom agent."
        ctx.system.tools_description = "echo(text) - Echo text"

        orchestrator = Orchestrator(
            router=router,
            registry=registry,
            pipeline=pipeline,
        )

        result = await orchestrator.run("Hello", context=ctx)
        assert "Done" in result
