"""Tests for Issue 2: Task/Goal Cancellation."""

import inspect
from unittest.mock import MagicMock, patch

from open_harness.agent import Agent


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


class TestCancelEvent:
    def test_cancel_sets_event_on_agent(self):
        """cancel() should set the Agent's internal _cancel_event."""
        agent = _make_agent()
        assert not agent._cancel_event.is_set()
        agent.cancel()
        assert agent._cancel_event.is_set()

    def test_cancel_event_clear(self):
        """_cancel_event can be cleared after being set."""
        agent = _make_agent()
        agent.cancel()
        assert agent._cancel_event.is_set()
        agent._cancel_event.clear()
        assert not agent._cancel_event.is_set()

    def test_agent_has_cancel_method(self):
        assert hasattr(Agent, "cancel")
        assert callable(getattr(Agent, "cancel"))

    def test_agent_init_accepts_user_input_fn(self):
        """Agent.__init__ should accept user_input_fn parameter."""
        sig = inspect.signature(Agent.__init__)
        assert "user_input_fn" in sig.parameters
