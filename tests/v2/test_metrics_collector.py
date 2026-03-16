"""Tests for the MetricsCollector — KPI tracking via EventBus."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from open_harness_v2.events.bus import EventBus
from open_harness_v2.metrics.collector import (
    GoalMetrics,
    MetricsCollector,
    SessionStats,
)
from open_harness_v2.types import AgentEvent, EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: EventType, data: dict | None = None) -> AgentEvent:
    return AgentEvent(type=event_type, data=data or {})


# ---------------------------------------------------------------------------
# GoalMetrics unit tests
# ---------------------------------------------------------------------------

class TestGoalMetrics:
    def test_elapsed_s(self):
        m = GoalMetrics(started_at=100.0, finished_at=105.5)
        assert m.elapsed_s == pytest.approx(5.5)

    def test_elapsed_s_not_finished(self):
        m = GoalMetrics(started_at=100.0)
        assert m.elapsed_s == 0.0

    def test_p95_latency_empty(self):
        m = GoalMetrics()
        assert m.p95_latency_ms == 0.0

    def test_p95_latency(self):
        m = GoalMetrics(latencies_ms=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # p95 index = int(10 * 0.95) = 9 → value 100
        assert m.p95_latency_ms == 100.0

    def test_avg_latency(self):
        m = GoalMetrics(latencies_ms=[10, 20, 30])
        assert m.avg_latency_ms == 20.0

    def test_to_dict(self):
        m = GoalMetrics(
            goal="test goal",
            started_at=1000.0,
            finished_at=1010.0,
            success=True,
            steps=5,
            total_tokens=1000,
        )
        d = m.to_dict()
        assert d["goal"] == "test goal"
        assert d["elapsed_s"] == 10.0
        assert d["success"] is True
        assert d["steps"] == 5


# ---------------------------------------------------------------------------
# SessionStats unit tests
# ---------------------------------------------------------------------------

class TestSessionStats:
    def test_success_rate_zero(self):
        s = SessionStats()
        assert s.success_rate == 0.0

    def test_success_rate(self):
        s = SessionStats(total_goals=10, successful_goals=7)
        assert s.success_rate == pytest.approx(0.7)

    def test_cost_per_success(self):
        s = SessionStats(total_tokens=10000, successful_goals=5)
        assert s.cost_per_success == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# MetricsCollector integration tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_collector_tracks_successful_goal():
    """Full goal lifecycle: started → llm_response → tool_executed → done."""
    bus = EventBus()
    collector = MetricsCollector(metrics_dir=None)  # in-memory only
    collector.attach(bus)

    # Goal starts
    await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": "Fix bug"}))
    assert collector.current is not None
    assert collector.current.goal == "Fix bug"

    # LLM responds
    await bus.emit(_make_event(EventType.LLM_RESPONSE, {
        "model": "gpt-4",
        "has_tool_calls": True,
        "content_length": 100,
        "latency_ms": 500.0,
        "usage": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
    }))
    assert collector.current.llm_calls == 1
    assert collector.current.total_tokens == 300

    # Tool executes
    await bus.emit(_make_event(EventType.TOOL_EXECUTED, {
        "tool": "edit_file",
        "success": True,
        "output": "ok",
        "output_length": 2,
        "elapsed_ms": 50,
    }))
    assert collector.current.tool_calls == 1

    # Goal done
    await bus.emit(_make_event(EventType.AGENT_DONE, {
        "response": "Fixed!",
        "steps": 3,
    }))
    assert collector.current is None
    assert len(collector.history) == 1
    assert collector.history[0].success is True
    assert collector.history[0].steps == 3
    assert collector.session.total_goals == 1
    assert collector.session.successful_goals == 1


@pytest.mark.anyio
async def test_collector_tracks_failed_goal():
    bus = EventBus()
    collector = MetricsCollector(metrics_dir=None)
    collector.attach(bus)

    await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": "Crash"}))
    await bus.emit(_make_event(EventType.AGENT_ERROR, {
        "error": "Something broke",
    }))

    assert len(collector.history) == 1
    assert collector.history[0].success is False
    assert collector.history[0].error == "Something broke"
    assert collector.session.success_rate == 0.0


@pytest.mark.anyio
async def test_collector_tracks_cancelled_goal():
    bus = EventBus()
    collector = MetricsCollector(metrics_dir=None)
    collector.attach(bus)

    await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": "Cancel me"}))
    await bus.emit(_make_event(EventType.AGENT_CANCELLED, {
        "response": "Agent cancelled",
        "steps": 1,
    }))

    assert len(collector.history) == 1
    assert collector.history[0].success is False
    assert collector.history[0].error == "cancelled"


@pytest.mark.anyio
async def test_collector_tracks_external_delegations():
    bus = EventBus()
    collector = MetricsCollector(metrics_dir=None)
    collector.attach(bus)

    await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": "Delegate"}))
    await bus.emit(_make_event(EventType.TOOL_EXECUTED, {
        "tool": "claude_code",
        "success": True,
        "output": "done",
        "output_length": 4,
        "elapsed_ms": 5000,
    }))
    await bus.emit(_make_event(EventType.AGENT_DONE, {
        "response": "Done",
        "steps": 2,
    }))

    assert collector.history[0].external_delegations == 1


@pytest.mark.anyio
async def test_collector_tracks_tool_errors():
    bus = EventBus()
    collector = MetricsCollector(metrics_dir=None)
    collector.attach(bus)

    await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": "Errors"}))
    await bus.emit(_make_event(EventType.TOOL_ERROR, {
        "tool": "shell",
        "error": "permission denied",
    }))
    await bus.emit(_make_event(EventType.AGENT_DONE, {
        "response": "Done with errors",
        "steps": 1,
    }))

    assert collector.history[0].tool_errors == 1
    assert collector.session.total_tool_errors == 1


@pytest.mark.anyio
async def test_collector_multiple_goals():
    bus = EventBus()
    collector = MetricsCollector(metrics_dir=None)
    collector.attach(bus)

    # Goal 1: success
    await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": "G1"}))
    await bus.emit(_make_event(EventType.LLM_RESPONSE, {
        "model": "m1",
        "has_tool_calls": False,
        "content_length": 10,
        "latency_ms": 100.0,
        "usage": {"total_tokens": 500},
    }))
    await bus.emit(_make_event(EventType.AGENT_DONE, {"response": "ok", "steps": 1}))

    # Goal 2: failure
    await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": "G2"}))
    await bus.emit(_make_event(EventType.LLM_RESPONSE, {
        "model": "m2",
        "has_tool_calls": False,
        "content_length": 10,
        "latency_ms": 200.0,
        "usage": {"total_tokens": 800},
    }))
    await bus.emit(_make_event(EventType.AGENT_ERROR, {"error": "fail"}))

    assert collector.session.total_goals == 2
    assert collector.session.successful_goals == 1
    assert collector.session.success_rate == pytest.approx(0.5)
    assert collector.session.total_tokens == 1300


@pytest.mark.anyio
async def test_collector_summary():
    bus = EventBus()
    collector = MetricsCollector(metrics_dir=None)
    collector.attach(bus)

    await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": "G1"}))
    await bus.emit(_make_event(EventType.AGENT_DONE, {"response": "ok", "steps": 1}))

    summary = collector.summary()
    assert "goals:1" in summary
    assert "success:1" in summary
    assert "rate:100%" in summary


@pytest.mark.anyio
async def test_collector_persists_to_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_dir = Path(tmpdir) / "metrics"
        bus = EventBus()
        collector = MetricsCollector(metrics_dir=metrics_dir)
        collector.attach(bus)

        await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": "persist test"}))
        await bus.emit(_make_event(EventType.AGENT_DONE, {"response": "ok", "steps": 1}))

        filepath = metrics_dir / "goals.jsonl"
        assert filepath.exists()
        lines = filepath.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["goal"] == "persist test"
        assert data["success"] is True


@pytest.mark.anyio
async def test_collector_load_historical():
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_dir = Path(tmpdir) / "metrics"
        bus = EventBus()
        collector = MetricsCollector(metrics_dir=metrics_dir)
        collector.attach(bus)

        # Write two goals
        for i in range(2):
            await bus.emit(_make_event(EventType.AGENT_STARTED, {"goal": f"G{i}"}))
            await bus.emit(_make_event(EventType.AGENT_DONE, {"response": "ok", "steps": 1}))

        historical = collector.load_historical()
        assert len(historical) == 2
        assert historical[0]["goal"] == "G0"
        assert historical[1]["goal"] == "G1"
