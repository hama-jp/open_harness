"""Tests for v0.4.5: External agent parallel execution."""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from open_harness.agent import Agent, AgentEvent, _EXTERNAL_AGENT_TOOLS, _LoopState
from open_harness.llm.client import ToolCall
from open_harness.tools.base import ToolResult


def _make_agent():
    """Create an Agent with mocked dependencies."""
    mock_config = MagicMock()
    mock_config.llm.default_provider = "test"
    mock_config.llm.providers = {"test": MagicMock(base_url="http://localhost:1234/v1")}
    mock_config.llm.default_model = "test-model"
    mock_config.llm.timeout = 30
    mock_config.policy = MagicMock()
    mock_config.memory = MagicMock()
    mock_config.memory.max_turns = 50
    mock_config.memory.db_path = ":memory:"
    mock_config.compensation = MagicMock()
    mock_config.compensation.thinking_mode = "none"

    mock_tools = MagicMock()
    mock_tools.list_tools.return_value = []
    mock_memory = MagicMock()
    mock_project = MagicMock()
    mock_project.root = "/tmp/test_project"
    mock_project.info = {"has_git": False}

    with patch("open_harness.agent.ModelRouter"), \
         patch("open_harness.agent.Compensator"), \
         patch("open_harness.agent.PolicyEngine"), \
         patch("open_harness.agent.load_policy"), \
         patch("open_harness.agent.Planner"), \
         patch("open_harness.agent.PlanCritic"), \
         patch("open_harness.agent.ProjectMemoryStore"), \
         patch("open_harness.agent.ProjectMemoryEngine"), \
         patch("open_harness.agent.CheckpointEngine"):
        agent = Agent(mock_config, mock_tools, mock_memory, mock_project)
    return agent


class TestLoopState:
    def test_default_values(self):
        state = _LoopState()
        assert state.writes_since_snapshot == 0
        assert state.test_failed_in_batch is False

    def test_mutation(self):
        state = _LoopState()
        state.writes_since_snapshot = 3
        state.test_failed_in_batch = True
        assert state.writes_since_snapshot == 3
        assert state.test_failed_in_batch is True


class TestToolCallSplitting:
    def test_external_tool_names(self):
        """Verify the set of external agent tool names."""
        assert _EXTERNAL_AGENT_TOOLS == {"codex", "claude_code", "gemini_cli"}

    def test_split_local_and_external(self):
        """Tool calls should be correctly split into local and external."""
        tool_calls = [
            ToolCall(name="read_file", arguments={"path": "foo.py"}),
            ToolCall(name="codex", arguments={"prompt": "fix bug"}),
            ToolCall(name="shell", arguments={"command": "ls"}),
            ToolCall(name="gemini_cli", arguments={"prompt": "review"}),
        ]

        local_calls = [tc for tc in tool_calls if tc.name not in _EXTERNAL_AGENT_TOOLS]
        external_calls = [tc for tc in tool_calls if tc.name in _EXTERNAL_AGENT_TOOLS]

        assert len(local_calls) == 2
        assert local_calls[0].name == "read_file"
        assert local_calls[1].name == "shell"

        assert len(external_calls) == 2
        assert external_calls[0].name == "codex"
        assert external_calls[1].name == "gemini_cli"

    def test_all_local(self):
        """When all calls are local, external list should be empty."""
        tool_calls = [
            ToolCall(name="read_file", arguments={"path": "foo.py"}),
            ToolCall(name="write_file", arguments={"path": "bar.py", "content": "x"}),
        ]

        local_calls = [tc for tc in tool_calls if tc.name not in _EXTERNAL_AGENT_TOOLS]
        external_calls = [tc for tc in tool_calls if tc.name in _EXTERNAL_AGENT_TOOLS]

        assert len(local_calls) == 2
        assert len(external_calls) == 0

    def test_all_external(self):
        """When all calls are external, local list should be empty."""
        tool_calls = [
            ToolCall(name="codex", arguments={"prompt": "a"}),
            ToolCall(name="claude_code", arguments={"prompt": "b"}),
            ToolCall(name="gemini_cli", arguments={"prompt": "c"}),
        ]

        local_calls = [tc for tc in tool_calls if tc.name not in _EXTERNAL_AGENT_TOOLS]
        external_calls = [tc for tc in tool_calls if tc.name in _EXTERNAL_AGENT_TOOLS]

        assert len(local_calls) == 0
        assert len(external_calls) == 3


class TestProcessLocalToolCall:
    def test_local_tool_call_yields_events(self):
        """_process_local_tool_call should yield tool_call and tool_result events."""
        agent = _make_agent()
        tc = ToolCall(name="read_file", arguments={"path": "test.py"})
        messages = []
        loop_state = _LoopState()

        # Mock policy to allow
        agent.policy.check.return_value = None
        agent.tools.execute.return_value = ToolResult(
            success=True, output="file contents here")

        events = list(agent._process_local_tool_call(
            tc, messages, checkpoint=None, step=1, loop_state=loop_state))

        event_types = [e.type for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types

        # Messages should be appended
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"

    def test_local_tool_call_policy_violation(self):
        """_process_local_tool_call should handle policy violations."""
        agent = _make_agent()
        tc = ToolCall(name="shell", arguments={"command": "rm -rf /"})
        messages = []
        loop_state = _LoopState()

        violation = MagicMock()
        violation.rule = "blocked_command"
        violation.message = "Dangerous command"
        agent.policy.check.return_value = violation

        events = list(agent._process_local_tool_call(
            tc, messages, checkpoint=None, step=1, loop_state=loop_state))

        # Should still yield tool_call and tool_result
        event_types = [e.type for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types

        # tool_result should indicate failure
        result_events = [e for e in events if e.type == "tool_result"]
        assert not result_events[0].metadata["success"]


class TestExecuteExternalParallel:
    def _make_agent_with_mock_ratelimiter(self):
        """Create agent with a mocked rate_limiter."""
        agent = _make_agent()
        agent.rate_limiter = MagicMock()
        return agent

    def test_parallel_execution_yields_results_in_order(self):
        """Results should be yielded in the original call order."""
        agent = self._make_agent_with_mock_ratelimiter()
        calls = [
            ToolCall(name="codex", arguments={"prompt": "task A"}),
            ToolCall(name="gemini_cli", arguments={"prompt": "task B"}),
        ]
        messages = []
        loop_state = _LoopState()

        # Mock: no rate limiting
        agent.rate_limiter.get_best_agent.side_effect = lambda name: (name, None)
        agent.policy.check.return_value = None

        # Mock execution: each returns a result with the tool name
        def mock_execute(tool_name, args):
            return ToolResult(success=True, output=f"result from {tool_name}")

        agent.tools.execute.side_effect = mock_execute

        events = list(agent._execute_external_parallel(
            calls, messages, checkpoint=None, step=1, loop_state=loop_state))

        # Find tool_result events
        result_events = [e for e in events if e.type == "tool_result"]
        assert len(result_events) == 2
        assert "codex" in result_events[0].data
        assert "gemini_cli" in result_events[1].data

        # Messages should be appended in order
        assert len(messages) == 4  # 2 calls * (assistant + user)

    def test_parallel_execution_policy_violation(self):
        """Policy violations should be caught before parallel execution."""
        agent = self._make_agent_with_mock_ratelimiter()
        calls = [
            ToolCall(name="codex", arguments={"prompt": "task A"}),
        ]
        messages = []
        loop_state = _LoopState()

        agent.rate_limiter.get_best_agent.side_effect = lambda name: (name, None)
        violation = MagicMock()
        violation.rule = "budget_exceeded"
        violation.message = "Too many external calls"
        agent.policy.check.return_value = violation

        events = list(agent._execute_external_parallel(
            calls, messages, checkpoint=None, step=1, loop_state=loop_state))

        result_events = [e for e in events if e.type == "tool_result"]
        assert len(result_events) == 1
        assert not result_events[0].metadata["success"]

        # tools.execute should NOT have been called
        agent.tools.execute.assert_not_called()

    def test_parallel_execution_with_rate_limit_fallback(self):
        """Rate-limited agents should fall back to alternatives."""
        agent = self._make_agent_with_mock_ratelimiter()
        calls = [
            ToolCall(name="codex", arguments={"prompt": "task"}),
        ]
        messages = []
        loop_state = _LoopState()

        # Pre-check: codex is available
        agent.rate_limiter.get_best_agent.return_value = ("codex", None)
        agent.policy.check.return_value = None

        # Execution returns rate limit error
        rate_limit_result = ToolResult(
            success=False, output="", error="429 Too many requests")
        fallback_result = ToolResult(
            success=True, output="result from claude_code")

        agent.tools.execute.side_effect = [rate_limit_result, fallback_result]
        agent.rate_limiter.record_rate_limit.return_value = MagicMock(
            human_remaining=lambda: "15m")
        agent.rate_limiter.get_fallback.return_value = "claude_code"

        from open_harness.tools.rate_limiter import AgentRateLimiter
        with patch.object(AgentRateLimiter, "is_rate_limit_error", return_value=True):
            events = list(agent._execute_external_parallel(
                calls, messages, checkpoint=None, step=1, loop_state=loop_state))

        # Should have compensation events for rate limiting
        comp_events = [e for e in events if e.type == "compensation"]
        assert len(comp_events) >= 1

    def test_thread_pool_is_used_for_execution(self):
        """Verify ThreadPoolExecutor is used for parallel execution."""
        agent = self._make_agent_with_mock_ratelimiter()
        calls = [
            ToolCall(name="codex", arguments={"prompt": "a"}),
            ToolCall(name="gemini_cli", arguments={"prompt": "b"}),
        ]
        messages = []
        loop_state = _LoopState()

        agent.rate_limiter.get_best_agent.side_effect = lambda name: (name, None)
        agent.policy.check.return_value = None

        execution_threads = []

        import threading
        def mock_execute(tool_name, args):
            execution_threads.append(threading.current_thread().name)
            return ToolResult(success=True, output=f"ok from {tool_name}")

        agent.tools.execute.side_effect = mock_execute

        events = list(agent._execute_external_parallel(
            calls, messages, checkpoint=None, step=1, loop_state=loop_state))

        # Both should have been executed
        assert len(execution_threads) == 2


class TestBuildToolPromptMode:
    def test_plan_mode_includes_planning_context(self):
        """build_tool_prompt with mode='plan' should include planning instructions."""
        from open_harness.llm.compensator import build_tool_prompt

        prompt = build_tool_prompt("tools here", mode="plan")
        assert "Planning Context" in prompt
        assert "PLAN mode" in prompt
        assert "clarifying questions" in prompt
        assert "Delegate planning tasks" in prompt

    def test_goal_mode_excludes_planning_context(self):
        """build_tool_prompt with mode='goal' should NOT include planning instructions."""
        from open_harness.llm.compensator import build_tool_prompt

        prompt = build_tool_prompt("tools here", mode="goal")
        assert "Planning Context" not in prompt

    def test_default_mode_is_plan(self):
        """Default mode should be 'plan'."""
        from open_harness.llm.compensator import build_tool_prompt

        prompt = build_tool_prompt("tools here")
        assert "Planning Context" in prompt
