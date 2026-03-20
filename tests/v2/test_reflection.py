"""Tests for the Reflector — self-assessment and progress tracking."""

import pytest

from open_harness_v2.core.reflection import (
    ActionOutcome,
    ProgressSignal,
    Reflector,
    ReflectionResult,
)


class TestReflectorBasics:
    """Basic Reflector functionality."""

    def test_initial_state(self):
        r = Reflector()
        assert r.total_actions == 0
        assert r.success_rate == 0.0

    def test_set_goal_resets(self):
        r = Reflector()
        r.record(_ok("shell", {"command": "ls"}))
        r.set_goal("new goal")
        assert r.total_actions == 0

    def test_record_tracks_count(self):
        r = Reflector()
        r.record(_ok("shell", {"command": "ls"}))
        r.record(_ok("read_file", {"path": "x.py"}))
        assert r.total_actions == 2

    def test_success_rate(self):
        r = Reflector()
        r.record(_ok("shell", {"command": "ls"}))
        r.record(_fail("shell", {"command": "bad"}))
        assert r.success_rate == 0.5


class TestReflectorSignals:
    """Tests for signal detection."""

    def test_no_actions_returns_uncertain(self):
        r = Reflector()
        result = r.reflect()
        assert result.signal == ProgressSignal.UNCERTAIN

    def test_all_success_returns_advancing(self):
        r = Reflector()
        for i in range(5):
            r.record(_ok(f"tool_{i}", {"arg": str(i)}, output=f"output_{i}"))
        result = r.reflect()
        assert result.signal == ProgressSignal.ADVANCING

    def test_consecutive_failures_detected(self):
        r = Reflector()
        for i in range(4):
            r.record(_fail("shell", {"command": f"bad_{i}"}))
        result = r.reflect()
        assert result.signal in (ProgressSignal.BLOCKED, ProgressSignal.STALLED)

    def test_loop_detection(self):
        r = Reflector()
        # Same tool with same args repeated
        for _ in range(5):
            r.record(_ok("shell", {"command": "ls"}))
        result = r.reflect()
        assert result.signal in (ProgressSignal.STALLED, ProgressSignal.REGRESSING)

    def test_good_diversity_is_advancing(self):
        r = Reflector()
        tools = ["read_file", "shell", "write_file", "git_status", "search_files"]
        for i, tool in enumerate(tools):
            r.record(_ok(tool, {"arg": str(i)}, output=f"output_{i}"))
        result = r.reflect()
        assert result.signal == ProgressSignal.ADVANCING

    def test_same_tool_failing_is_blocked(self):
        r = Reflector()
        r.record(_ok("read_file", {"path": "a.py"}))
        for i in range(4):
            r.record(_fail("shell", {"command": f"test_{i}"}))
        result = r.reflect()
        assert result.signal in (ProgressSignal.BLOCKED, ProgressSignal.STALLED)


class TestReflectorSuggestions:
    """Tests for recovery suggestions."""

    def test_blocked_suggests_replan(self):
        r = Reflector()
        for _ in range(4):
            r.record(_fail("shell", {"command": "bad"}))
        result = r.reflect()
        assert result.should_replan or result.suggestions

    def test_stalled_suggests_different_tool(self):
        r = Reflector()
        for _ in range(4):
            r.record(_ok("shell", {"command": "ls"}))
        result = r.reflect()
        if result.signal == ProgressSignal.STALLED:
            assert any("different" in s.lower() or "switch" in s.lower()
                       for s in result.suggestions) or result.should_replan


class TestContextInjection:
    """Tests for context injection generation."""

    def test_empty_returns_empty(self):
        r = Reflector()
        assert r.get_context_injection() == ""

    def test_non_empty_returns_block(self):
        r = Reflector()
        r.record(_ok("shell", {"command": "ls"}))
        r.record(_fail("shell", {"command": "bad"}))
        block = r.get_context_injection()
        assert "Progress Status" in block
        assert "1/2" in block  # success rate

    def test_warns_on_consecutive_failures(self):
        r = Reflector()
        for _ in range(3):
            r.record(_fail("shell", {"command": "bad"}))
        block = r.get_context_injection()
        assert "WARNING" in block
        assert "3 consecutive failures" in block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(
    tool: str, args: dict, step: int = 1, output: str = "some output"
) -> ActionOutcome:
    return ActionOutcome(
        step_number=step,
        tool_name=tool,
        tool_args=args,
        success=True,
        output_snippet=output,
    )


def _fail(
    tool: str, args: dict, step: int = 1, error: str = "command failed"
) -> ActionOutcome:
    return ActionOutcome(
        step_number=step,
        tool_name=tool,
        tool_args=args,
        success=False,
        output_snippet="",
        error=error,
    )
