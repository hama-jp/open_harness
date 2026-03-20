"""Reflection — self-assessment of progress and quality after each action.

The Reflector analyzes tool execution results to determine:
1. Whether the action moved closer to the goal
2. Whether the approach is working or needs adjustment
3. What to do next based on accumulated evidence

This enables the agent to course-correct autonomously rather than
blindly executing tools until a step limit is hit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_logger = logging.getLogger(__name__)


class ProgressSignal(Enum):
    """Signals derived from analyzing action outcomes."""

    ADVANCING = "advancing"      # Making clear progress
    STALLED = "stalled"          # No meaningful progress
    REGRESSING = "regressing"    # Moving away from goal
    BLOCKED = "blocked"          # Hit a hard obstacle
    UNCERTAIN = "uncertain"      # Can't determine progress


@dataclass
class ActionOutcome:
    """Record of a single action and its result."""

    step_number: int
    tool_name: str
    tool_args: dict[str, Any]
    success: bool
    output_snippet: str  # first N chars of output
    error: str = ""
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class ReflectionResult:
    """Output of a reflection cycle."""

    signal: ProgressSignal
    confidence: float  # 0.0 - 1.0
    assessment: str  # human-readable assessment
    suggestions: list[str] = field(default_factory=list)
    should_replan: bool = False
    should_escalate: bool = False  # escalate model tier


class Reflector:
    """Analyzes action outcomes to assess progress toward the goal.

    Maintains a sliding window of recent outcomes for pattern detection.
    Works without LLM calls — uses heuristic analysis for speed.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._window_size = window_size
        self._outcomes: list[ActionOutcome] = []
        self._goal: str = ""
        self._consecutive_failures: int = 0
        self._consecutive_same_tool: int = 0
        self._last_tool: str = ""
        self._unique_tools_used: set[str] = set()
        self._files_touched: set[str] = set()

    def set_goal(self, goal: str) -> None:
        """Set the current goal for context-aware reflection."""
        self._goal = goal
        self.reset()

    def reset(self) -> None:
        """Reset state for a new goal."""
        self._outcomes.clear()
        self._consecutive_failures = 0
        self._consecutive_same_tool = 0
        self._last_tool = ""
        self._unique_tools_used.clear()
        self._files_touched.clear()

    def record(self, outcome: ActionOutcome) -> None:
        """Record an action outcome for analysis."""
        self._outcomes.append(outcome)
        if len(self._outcomes) > self._window_size * 2:
            self._outcomes = self._outcomes[-self._window_size:]

        # Track patterns
        self._unique_tools_used.add(outcome.tool_name)

        if outcome.success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        if outcome.tool_name == self._last_tool:
            self._consecutive_same_tool += 1
        else:
            self._consecutive_same_tool = 0
        self._last_tool = outcome.tool_name

        # Track file access
        path = outcome.tool_args.get("path", "")
        if path:
            self._files_touched.add(path)

    def reflect(self) -> ReflectionResult:
        """Analyze recent outcomes and produce a reflection."""
        if not self._outcomes:
            return ReflectionResult(
                signal=ProgressSignal.UNCERTAIN,
                confidence=0.0,
                assessment="No actions taken yet.",
            )

        # Gather signals
        signals: list[tuple[ProgressSignal, float, str]] = []

        # 1. Check failure patterns
        signals.append(self._check_failures())

        # 2. Check for loops (same tool with same args)
        signals.append(self._check_loops())

        # 3. Check for tool diversity (sign of exploration vs. stuck)
        signals.append(self._check_diversity())

        # 4. Check for error pattern escalation
        signals.append(self._check_error_patterns())

        # 5. Check output quality
        signals.append(self._check_output_quality())

        # Aggregate signals
        return self._aggregate(signals)

    def get_context_injection(self) -> str:
        """Generate a context block summarizing progress for the LLM."""
        if not self._outcomes:
            return ""

        recent = self._outcomes[-5:]
        total = len(self._outcomes)
        successes = sum(1 for o in self._outcomes if o.success)

        lines = [
            f"## Progress Status (step {total})",
            f"Success rate: {successes}/{total} actions succeeded",
        ]

        if self._consecutive_failures > 0:
            lines.append(
                f"WARNING: {self._consecutive_failures} consecutive failures"
            )

        if self._consecutive_same_tool > 2:
            lines.append(
                f"NOTE: Same tool '{self._last_tool}' used "
                f"{self._consecutive_same_tool + 1} times in a row"
            )

        # Recent action summary
        lines.append("Recent actions:")
        for o in recent:
            status = "OK" if o.success else "FAIL"
            lines.append(f"  [{status}] {o.tool_name}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Signal checks
    # ------------------------------------------------------------------

    def _check_failures(self) -> tuple[ProgressSignal, float, str]:
        """Check consecutive failure patterns."""
        if self._consecutive_failures == 0:
            return (ProgressSignal.ADVANCING, 0.3, "Last action succeeded")
        if self._consecutive_failures == 1:
            return (ProgressSignal.UNCERTAIN, 0.4, "Single failure — may be transient")
        if self._consecutive_failures == 2:
            return (ProgressSignal.STALLED, 0.6, "Two consecutive failures")
        if self._consecutive_failures >= 3:
            return (
                ProgressSignal.BLOCKED, 0.8,
                f"{self._consecutive_failures} consecutive failures — "
                f"approach may be fundamentally wrong",
            )
        return (ProgressSignal.UNCERTAIN, 0.2, "")

    def _check_loops(self) -> tuple[ProgressSignal, float, str]:
        """Detect repetitive tool calls (same tool + same args)."""
        if len(self._outcomes) < 3:
            return (ProgressSignal.UNCERTAIN, 0.1, "Too few actions to detect loops")

        recent = self._outcomes[-5:]
        # Check for exact duplicates (same tool + same args)
        seen: set[str] = set()
        duplicates = 0
        for o in recent:
            key = f"{o.tool_name}:{sorted(o.tool_args.items())}"
            if key in seen:
                duplicates += 1
            seen.add(key)

        if duplicates >= 3:
            return (
                ProgressSignal.REGRESSING, 0.9,
                "Detected loop: same tool with same arguments repeated",
            )
        if duplicates >= 2:
            return (
                ProgressSignal.STALLED, 0.7,
                "Possible loop: repeated tool calls with same arguments",
            )
        if self._consecutive_same_tool > 3:
            return (
                ProgressSignal.STALLED, 0.5,
                f"Same tool '{self._last_tool}' used {self._consecutive_same_tool + 1} "
                f"times consecutively",
            )
        return (ProgressSignal.ADVANCING, 0.2, "No loops detected")

    def _check_diversity(self) -> tuple[ProgressSignal, float, str]:
        """Check tool usage diversity as a progress indicator."""
        if len(self._outcomes) < 3:
            return (ProgressSignal.UNCERTAIN, 0.1, "")

        recent_tools = [o.tool_name for o in self._outcomes[-5:]]
        unique_recent = len(set(recent_tools))
        total_recent = len(recent_tools)

        ratio = unique_recent / total_recent
        if ratio > 0.6:
            return (
                ProgressSignal.ADVANCING, 0.4,
                "Good tool diversity — exploring multiple approaches",
            )
        if ratio < 0.3 and total_recent >= 4:
            return (
                ProgressSignal.STALLED, 0.5,
                "Low tool diversity — may be stuck in a rut",
            )
        return (ProgressSignal.UNCERTAIN, 0.2, "")

    def _check_error_patterns(self) -> tuple[ProgressSignal, float, str]:
        """Check for escalating error patterns."""
        if len(self._outcomes) < 3:
            return (ProgressSignal.UNCERTAIN, 0.1, "")

        recent = self._outcomes[-5:]
        errors = [o for o in recent if not o.success]

        if not errors:
            return (ProgressSignal.ADVANCING, 0.3, "No recent errors")

        # Check if errors are the same type (same tool failing)
        error_tools = [e.tool_name for e in errors]
        if len(set(error_tools)) == 1 and len(errors) >= 2:
            return (
                ProgressSignal.BLOCKED, 0.7,
                f"Tool '{error_tools[0]}' failing repeatedly — "
                f"likely need a different approach",
            )

        return (ProgressSignal.STALLED, 0.4, f"{len(errors)} errors in recent actions")

    def _check_output_quality(self) -> tuple[ProgressSignal, float, str]:
        """Check if tool outputs are producing useful information."""
        if not self._outcomes:
            return (ProgressSignal.UNCERTAIN, 0.1, "")

        recent = [o for o in self._outcomes[-5:] if o.success]
        if not recent:
            return (ProgressSignal.UNCERTAIN, 0.2, "No successful actions to evaluate")

        # Check for empty or very short outputs
        empty_count = sum(
            1 for o in recent if len(o.output_snippet.strip()) < 10
        )
        if empty_count > len(recent) // 2:
            return (
                ProgressSignal.STALLED, 0.5,
                "Recent tool outputs are mostly empty — may not be finding useful data",
            )

        return (ProgressSignal.ADVANCING, 0.3, "Tool outputs contain data")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, signals: list[tuple[ProgressSignal, float, str]],
    ) -> ReflectionResult:
        """Aggregate multiple signals into a single reflection result."""
        if not signals:
            return ReflectionResult(
                signal=ProgressSignal.UNCERTAIN,
                confidence=0.0,
                assessment="No signals to analyze.",
            )

        # Weight by confidence
        weighted: dict[ProgressSignal, float] = {}
        assessments: list[str] = []

        for signal, confidence, assessment in signals:
            weighted[signal] = weighted.get(signal, 0) + confidence
            if assessment and confidence > 0.3:
                assessments.append(assessment)

        # Pick the signal with the highest weighted score
        best_signal = max(weighted, key=lambda s: weighted[s])
        best_confidence = min(1.0, weighted[best_signal] / max(1.0, sum(weighted.values())) * 2)

        # Generate suggestions
        suggestions: list[str] = []
        should_replan = False
        should_escalate = False

        if best_signal == ProgressSignal.BLOCKED:
            suggestions.append("Try a completely different approach")
            suggestions.append("Re-read error messages carefully for clues")
            should_replan = True
        elif best_signal == ProgressSignal.REGRESSING:
            suggestions.append("Stop repeating the same action")
            suggestions.append("Analyze why previous attempts failed before retrying")
            should_replan = True
        elif best_signal == ProgressSignal.STALLED:
            suggestions.append("Consider a different tool or approach")
            if self._consecutive_same_tool > 2:
                suggestions.append(
                    f"Switch from '{self._last_tool}' to another tool"
                )
            if self._consecutive_failures >= 2:
                should_escalate = True

        return ReflectionResult(
            signal=best_signal,
            confidence=best_confidence,
            assessment=" | ".join(assessments) if assessments else "Progress nominal",
            suggestions=suggestions,
            should_replan=should_replan,
            should_escalate=should_escalate,
        )

    @property
    def total_actions(self) -> int:
        return len(self._outcomes)

    @property
    def success_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(1 for o in self._outcomes if o.success) / len(self._outcomes)

    @property
    def outcomes(self) -> list[ActionOutcome]:
        return list(self._outcomes)
