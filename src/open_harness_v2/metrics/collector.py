"""MetricsCollector — EventBus subscriber that aggregates session KPIs.

Subscribes to agent lifecycle and LLM events to compute per-goal and
session-level metrics: success rate, cost-per-success, p95 latency, etc.

Persists metrics to a JSONL file for historical analysis.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType

_logger = logging.getLogger(__name__)

# Default storage location
_DEFAULT_METRICS_DIR = Path.home() / ".cache" / "open_harness" / "metrics"


@dataclass
class GoalMetrics:
    """Metrics for a single goal execution."""

    goal: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    success: bool = False
    steps: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    external_delegations: int = 0
    error: str = ""

    @property
    def elapsed_s(self) -> float:
        if self.finished_at and self.started_at:
            return self.finished_at - self.started_at
        return 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal[:200],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round(self.elapsed_s, 2),
            "success": self.success,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "models_used": list(set(self.models_used)),
            "external_delegations": self.external_delegations,
            "error": self.error[:200] if self.error else "",
        }


@dataclass
class SessionStats:
    """Aggregated statistics across multiple goals in a session."""

    total_goals: int = 0
    successful_goals: int = 0
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tool_errors: int = 0
    all_latencies_ms: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_goals == 0:
            return 0.0
        return self.successful_goals / self.total_goals

    @property
    def p95_latency_ms(self) -> float:
        if not self.all_latencies_ms:
            return 0.0
        sorted_lat = sorted(self.all_latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def cost_per_success(self) -> float:
        """Tokens per successful goal (proxy for cost)."""
        if self.successful_goals == 0:
            return 0.0
        return self.total_tokens / self.successful_goals


_EXTERNAL_TOOL_NAMES = frozenset({"claude_code", "codex", "gemini_cli"})


class MetricsCollector:
    """Collects per-goal and session-level metrics from EventBus events.

    Attach to an EventBus to automatically track all goal executions.
    Optionally persists completed goal metrics to a JSONL file.

    Parameters
    ----------
    metrics_dir:
        Directory for JSONL persistence. If None, metrics are only
        kept in memory for the current session.
    """

    def __init__(self, metrics_dir: Path | None = _DEFAULT_METRICS_DIR) -> None:
        self._metrics_dir = metrics_dir
        self._current: GoalMetrics | None = None
        self._history: list[GoalMetrics] = []
        self._session = SessionStats()

        if self._metrics_dir:
            self._metrics_dir.mkdir(parents=True, exist_ok=True)

    def attach(self, bus: EventBus) -> None:
        """Subscribe to relevant EventBus events."""
        bus.subscribe(EventType.AGENT_STARTED, self._on_agent_started)
        bus.subscribe(EventType.AGENT_DONE, self._on_agent_done)
        bus.subscribe(EventType.AGENT_ERROR, self._on_agent_error)
        bus.subscribe(EventType.AGENT_CANCELLED, self._on_agent_cancelled)
        bus.subscribe(EventType.LLM_RESPONSE, self._on_llm_response)
        bus.subscribe(EventType.TOOL_EXECUTED, self._on_tool_executed)
        bus.subscribe(EventType.TOOL_ERROR, self._on_tool_error)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_agent_started(self, event: AgentEvent) -> None:
        self._current = GoalMetrics(
            goal=event.data.get("goal", ""),
            started_at=event.timestamp,
        )

    def _on_llm_response(self, event: AgentEvent) -> None:
        if not self._current:
            return
        self._current.llm_calls += 1
        latency = event.data.get("latency_ms", 0)
        if latency:
            self._current.latencies_ms.append(latency)
        usage = event.data.get("usage", {})
        self._current.total_tokens += usage.get("total_tokens", 0)
        self._current.input_tokens += usage.get("input_tokens", 0)
        self._current.output_tokens += usage.get("output_tokens", 0)
        model = event.data.get("model", "")
        if model:
            self._current.models_used.append(model)

    def _on_tool_executed(self, event: AgentEvent) -> None:
        if not self._current:
            return
        self._current.tool_calls += 1
        tool_name = event.data.get("tool", "")
        if tool_name in _EXTERNAL_TOOL_NAMES:
            self._current.external_delegations += 1

    def _on_tool_error(self, event: AgentEvent) -> None:
        if not self._current:
            return
        self._current.tool_errors += 1

    def _on_agent_done(self, event: AgentEvent) -> None:
        if not self._current:
            return
        self._current.finished_at = event.timestamp
        self._current.success = True
        self._current.steps = event.data.get("steps", 0)
        self._finalize()

    def _on_agent_error(self, event: AgentEvent) -> None:
        if not self._current:
            return
        self._current.finished_at = event.timestamp
        self._current.success = False
        self._current.error = event.data.get("error", "")
        self._current.steps = event.data.get("steps", 0)
        self._finalize()

    def _on_agent_cancelled(self, event: AgentEvent) -> None:
        if not self._current:
            return
        self._current.finished_at = event.timestamp
        self._current.success = False
        self._current.error = "cancelled"
        self._current.steps = event.data.get("steps", 0)
        self._finalize()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        """Finalize current goal metrics and update session stats."""
        if not self._current:
            return

        metrics = self._current
        self._current = None
        self._history.append(metrics)

        # Update session stats
        self._session.total_goals += 1
        if metrics.success:
            self._session.successful_goals += 1
        self._session.total_tokens += metrics.total_tokens
        self._session.total_llm_calls += metrics.llm_calls
        self._session.total_tool_calls += metrics.tool_calls
        self._session.total_tool_errors += metrics.tool_errors
        self._session.all_latencies_ms.extend(metrics.latencies_ms)

        # Persist to JSONL
        self._persist(metrics)

    def _persist(self, metrics: GoalMetrics) -> None:
        """Append goal metrics to JSONL file."""
        if not self._metrics_dir:
            return
        try:
            filepath = self._metrics_dir / "goals.jsonl"
            with open(filepath, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        except OSError:
            _logger.warning("Failed to persist metrics", exc_info=True)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    @property
    def session(self) -> SessionStats:
        """Current session statistics."""
        return self._session

    @property
    def history(self) -> list[GoalMetrics]:
        """Completed goal metrics for this session."""
        return list(self._history)

    @property
    def current(self) -> GoalMetrics | None:
        """In-progress goal metrics (None if no goal is running)."""
        return self._current

    def summary(self) -> str:
        """Human-readable session summary for /status display."""
        s = self._session
        parts = [
            f"goals:{s.total_goals}",
            f"success:{s.successful_goals}",
            f"rate:{s.success_rate:.0%}",
            f"tokens:{s.total_tokens:,}",
        ]
        if s.all_latencies_ms:
            parts.append(f"p95:{s.p95_latency_ms:.0f}ms")
        if s.successful_goals > 0:
            parts.append(f"tokens/success:{s.cost_per_success:,.0f}")
        return " | ".join(parts)

    def load_historical(self, limit: int = 100) -> list[dict[str, Any]]:
        """Load recent historical metrics from JSONL file."""
        if not self._metrics_dir:
            return []
        filepath = self._metrics_dir / "goals.jsonl"
        if not filepath.exists():
            return []
        try:
            lines = filepath.read_text().strip().split("\n")
            recent = lines[-limit:] if len(lines) > limit else lines
            return [json.loads(line) for line in recent if line.strip()]
        except (OSError, json.JSONDecodeError):
            _logger.warning("Failed to load historical metrics", exc_info=True)
            return []
