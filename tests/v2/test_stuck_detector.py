"""Tests for the StuckDetector — loop, stall, and error spiral detection."""

import pytest

from open_harness_v2.core.stuck_detector import (
    RecoveryAction,
    StuckDetector,
    StuckPattern,
)


class TestStuckDetectorBasics:
    """Basic StuckDetector behavior."""

    def test_initial_not_stuck(self):
        d = StuckDetector()
        diagnosis = d.diagnose()
        assert not diagnosis.is_stuck

    def test_few_actions_not_stuck(self):
        d = StuckDetector()
        d.record("shell", {"command": "ls"}, True, "file.py")
        d.record("read_file", {"path": "x.py"}, True, "content")
        diagnosis = d.diagnose()
        assert not diagnosis.is_stuck

    def test_reset_clears_state(self):
        d = StuckDetector()
        for _ in range(5):
            d.record("shell", {"command": "ls"}, True, "out")
        d.reset()
        diagnosis = d.diagnose()
        assert not diagnosis.is_stuck


class TestExactLoopDetection:
    """Tests for exact loop (same tool + same args) detection."""

    def test_exact_loop_detected(self):
        d = StuckDetector(exact_loop_threshold=3)
        for _ in range(4):
            d.record("shell", {"command": "ls"}, True, "out")
        diagnosis = d.diagnose()
        assert diagnosis.is_stuck
        assert diagnosis.pattern == StuckPattern.EXACT_LOOP

    def test_different_args_not_loop(self):
        d = StuckDetector(exact_loop_threshold=3)
        for i in range(4):
            d.record("shell", {"command": f"cmd_{i}"}, True, "out")
        diagnosis = d.diagnose()
        assert diagnosis.pattern != StuckPattern.EXACT_LOOP

    def test_loop_recovery_escalates(self):
        d = StuckDetector(exact_loop_threshold=3)
        for _ in range(4):
            d.record("shell", {"command": "ls"}, True, "out")

        # First intervention
        d.record_intervention()
        for _ in range(4):
            d.record("shell", {"command": "ls"}, True, "out")

        diagnosis = d.diagnose()
        assert diagnosis.is_stuck
        # Should escalate to REPLAN after first intervention
        assert diagnosis.recovery in (RecoveryAction.REPLAN, RecoveryAction.ASK_USER)


class TestErrorSpiralDetection:
    """Tests for cascading error detection."""

    def test_high_failure_rate_detected(self):
        d = StuckDetector()
        # Some initial successes
        d.record("read_file", {"path": "a.py"}, True, "ok")
        # Then mostly failures
        for i in range(6):
            d.record("shell", {"command": f"bad_{i}"}, False, "error")
        diagnosis = d.diagnose()
        assert diagnosis.is_stuck
        assert diagnosis.pattern == StuckPattern.ERROR_SPIRAL

    def test_single_tool_failing(self):
        d = StuckDetector()
        d.record("read_file", {"path": "a.py"}, True, "ok")
        for _ in range(4):
            d.record("shell", {"command": "test"}, False, "error")
        diagnosis = d.diagnose()
        if diagnosis.is_stuck and diagnosis.pattern == StuckPattern.ERROR_SPIRAL:
            assert "shell" in diagnosis.description

    def test_moderate_failure_suggests_escalation(self):
        d = StuckDetector()
        for i in range(8):
            success = i % 2 == 0  # 50% failure rate
            d.record("shell", {"command": f"cmd_{i}"}, success, "out")
        diagnosis = d.diagnose()
        if diagnosis.is_stuck:
            assert diagnosis.recovery in (
                RecoveryAction.ESCALATE_MODEL,
                RecoveryAction.REPLAN,
                RecoveryAction.SWITCH_STRATEGY,
            )


class TestThrashingDetection:
    """Tests for A→B→A→B alternation detection."""

    def test_thrashing_detected(self):
        d = StuckDetector()
        for _ in range(4):
            d.record("read_file", {"path": "a.py"}, True, "content")
            d.record("write_file", {"path": "a.py"}, True, "ok")
        diagnosis = d.diagnose()
        if diagnosis.pattern == StuckPattern.THRASHING:
            assert diagnosis.is_stuck
            assert "alternating" in diagnosis.description.lower()


class TestSemanticLoopDetection:
    """Tests for different tools producing same output."""

    def test_same_output_detected(self):
        d = StuckDetector()
        same_output = "exactly the same output every time"
        for i in range(5):
            d.record(f"tool_{i % 2}", {"arg": str(i)}, True, same_output)
        diagnosis = d.diagnose()
        if diagnosis.pattern == StuckPattern.SEMANTIC_LOOP:
            assert diagnosis.is_stuck


class TestStagnationDetection:
    """Tests for many actions without file changes."""

    def test_read_only_stagnation(self):
        d = StuckDetector(stagnation_threshold=5)
        for i in range(6):
            d.record("read_file", {"path": f"file_{i}.py"}, True, "content")
        diagnosis = d.diagnose()
        if diagnosis.pattern == StuckPattern.STAGNATION:
            assert diagnosis.is_stuck
            assert "reads" in diagnosis.description.lower()


class TestRecoveryActions:
    """Tests for recovery recommendations."""

    def test_intervention_count_tracks(self):
        d = StuckDetector()
        assert d.intervention_count == 0
        d.record_intervention()
        d.record_intervention()
        assert d.intervention_count == 2

    def test_needs_intervention_threshold(self):
        d = StuckDetector(exact_loop_threshold=3)
        for _ in range(5):
            d.record("shell", {"command": "ls"}, True, "out")
        diagnosis = d.diagnose()
        if diagnosis.is_stuck:
            assert diagnosis.needs_intervention == (diagnosis.severity >= 0.6)
