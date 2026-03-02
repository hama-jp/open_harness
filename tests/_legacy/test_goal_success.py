"""Tests for Issue 2: Goal/task success state."""

from open_harness.agent import AgentEvent


class TestGoalSuccessState:
    def test_done_event_has_success_field(self):
        """Done events should include a success field in metadata."""
        event = AgentEvent("done", "completed", {"success": True})
        assert event.metadata["success"] is True

        event = AgentEvent("done", "failed", {"success": False})
        assert event.metadata["success"] is False

    def test_canceled_event_is_failure(self):
        """A canceled done event should have success=False."""
        event = AgentEvent("done", "[Canceled by user]", {"success": False})
        assert event.metadata["success"] is False

    def test_step_limit_event_is_failure(self):
        """Hitting step limit should have success=False."""
        event = AgentEvent(
            "done",
            "[Reached 50 steps. Use /tier large or simplify the goal.]",
            {"success": False},
        )
        assert event.metadata["success"] is False

    def test_normal_completion_is_success(self):
        """Normal text response should have success=True."""
        event = AgentEvent(
            "done",
            "I've completed the task.",
            {"success": True, "latency_ms": 1500, "steps": 3},
        )
        assert event.metadata["success"] is True
