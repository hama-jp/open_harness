"""Stuck detector — identifies when the agent is spinning its wheels.

Detects multiple patterns of unproductive behavior:
1. **Exact loops**: Same tool + same arguments repeated
2. **Semantic loops**: Different tools but producing the same output
3. **Thrashing**: Rapidly alternating between two states
4. **Stagnation**: Many actions without meaningful file changes
5. **Error spirals**: Cascading errors from a root cause

Each pattern has a specific recovery recommendation.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

_logger = logging.getLogger(__name__)


class StuckPattern(Enum):
    """Types of stuck behavior detected."""

    EXACT_LOOP = "exact_loop"
    SEMANTIC_LOOP = "semantic_loop"
    THRASHING = "thrashing"
    STAGNATION = "stagnation"
    ERROR_SPIRAL = "error_spiral"
    NOT_STUCK = "not_stuck"


class RecoveryAction(Enum):
    """Recommended recovery actions."""

    REPLAN = "replan"                    # Generate a new plan
    SWITCH_STRATEGY = "switch_strategy"  # Try a different approach
    ESCALATE_MODEL = "escalate_model"    # Use a larger model
    SKIP_STEP = "skip_step"             # Skip the current step
    ASK_USER = "ask_user"               # Request human guidance
    FORCE_DIFFERENT_TOOL = "force_different_tool"  # Inject hint to use different tool
    RESET_AND_RETRY = "reset_and_retry"  # Checkpoint rollback + retry


@dataclass
class StuckDiagnosis:
    """Result of stuck detection analysis."""

    is_stuck: bool
    pattern: StuckPattern
    severity: float  # 0.0 (not stuck) - 1.0 (definitely stuck)
    description: str
    recovery: RecoveryAction
    recovery_hint: str = ""  # Specific guidance for the recovery action

    @property
    def needs_intervention(self) -> bool:
        """True if severity warrants automatic intervention."""
        return self.severity >= 0.6


@dataclass
class _ActionFingerprint:
    """Compact representation of an action for comparison."""

    tool: str
    args_hash: str
    output_hash: str
    success: bool
    step: int


class StuckDetector:
    """Detects stuck patterns from a sequence of agent actions.

    Usage::

        detector = StuckDetector()
        detector.record("shell", {"command": "ls"}, True, "file.py\\n")
        diagnosis = detector.diagnose()
        if diagnosis.is_stuck:
            ... handle recovery ...
    """

    def __init__(
        self,
        window_size: int = 15,
        exact_loop_threshold: int = 3,
        stagnation_threshold: int = 8,
    ) -> None:
        self._window_size = window_size
        self._exact_loop_threshold = exact_loop_threshold
        self._stagnation_threshold = stagnation_threshold
        self._history: list[_ActionFingerprint] = []
        self._step_counter: int = 0
        self._files_modified: set[str] = set()
        self._last_files_modified_count: int = 0
        self._intervention_count: int = 0

    def record(
        self,
        tool_name: str,
        args: dict,
        success: bool,
        output: str,
        files_modified: set[str] | None = None,
    ) -> None:
        """Record an action for stuck detection."""
        self._step_counter += 1
        fingerprint = _ActionFingerprint(
            tool=tool_name,
            args_hash=_hash_dict(args),
            output_hash=_hash_str(output[:500]),
            success=success,
            step=self._step_counter,
        )
        self._history.append(fingerprint)
        if len(self._history) > self._window_size * 2:
            self._history = self._history[-self._window_size:]

        if files_modified:
            self._files_modified.update(files_modified)

    def diagnose(self) -> StuckDiagnosis:
        """Analyze the action history for stuck patterns.

        Returns the highest-severity diagnosis found.
        """
        if len(self._history) < 3:
            return StuckDiagnosis(
                is_stuck=False,
                pattern=StuckPattern.NOT_STUCK,
                severity=0.0,
                description="Too few actions to diagnose",
                recovery=RecoveryAction.REPLAN,
            )

        # Check patterns in order of severity
        checks = [
            self._check_exact_loop,
            self._check_error_spiral,
            self._check_thrashing,
            self._check_semantic_loop,
            self._check_stagnation,
        ]

        worst = StuckDiagnosis(
            is_stuck=False,
            pattern=StuckPattern.NOT_STUCK,
            severity=0.0,
            description="No stuck patterns detected",
            recovery=RecoveryAction.REPLAN,
        )

        for check in checks:
            diagnosis = check()
            if diagnosis.severity > worst.severity:
                worst = diagnosis

        return worst

    def record_intervention(self) -> None:
        """Record that an intervention was applied."""
        self._intervention_count += 1

    @property
    def intervention_count(self) -> int:
        return self._intervention_count

    def reset(self) -> None:
        """Reset for a new goal."""
        self._history.clear()
        self._step_counter = 0
        self._files_modified.clear()
        self._last_files_modified_count = 0
        self._intervention_count = 0

    # ------------------------------------------------------------------
    # Pattern detectors
    # ------------------------------------------------------------------

    def _check_exact_loop(self) -> StuckDiagnosis:
        """Detect exact repetition: same tool + same args."""
        recent = self._history[-self._window_size:]
        if len(recent) < self._exact_loop_threshold:
            return self._not_stuck()

        # Count (tool, args_hash) pairs
        pair_counts = Counter(
            (fp.tool, fp.args_hash) for fp in recent
        )
        most_common_pair, count = pair_counts.most_common(1)[0]

        if count >= self._exact_loop_threshold:
            tool_name = most_common_pair[0]
            severity = min(1.0, count / self._exact_loop_threshold * 0.7)

            # Escalate recovery based on intervention history
            if self._intervention_count >= 2:
                recovery = RecoveryAction.ASK_USER
                hint = (
                    f"The agent has been stuck calling '{tool_name}' with the "
                    f"same arguments {count} times. Previous recovery attempts "
                    f"({self._intervention_count}) failed. Human guidance needed."
                )
            elif self._intervention_count >= 1:
                recovery = RecoveryAction.REPLAN
                hint = (
                    f"'{tool_name}' called {count} times with same args. "
                    f"Previous strategy switch failed. Generate a completely "
                    f"different plan."
                )
            else:
                recovery = RecoveryAction.FORCE_DIFFERENT_TOOL
                hint = (
                    f"Stop calling '{tool_name}' with the same arguments. "
                    f"Try a different tool or different arguments."
                )

            return StuckDiagnosis(
                is_stuck=True,
                pattern=StuckPattern.EXACT_LOOP,
                severity=severity,
                description=(
                    f"Exact loop detected: '{tool_name}' called {count} times "
                    f"with identical arguments"
                ),
                recovery=recovery,
                recovery_hint=hint,
            )

        return self._not_stuck()

    def _check_error_spiral(self) -> StuckDiagnosis:
        """Detect cascading errors (>50% of recent actions failing)."""
        recent = self._history[-8:]
        if len(recent) < 4:
            return self._not_stuck()

        failures = sum(1 for fp in recent if not fp.success)
        failure_rate = failures / len(recent)

        if failure_rate >= 0.75:
            # Check if it's the same tool failing
            failed_tools = [fp.tool for fp in recent if not fp.success]
            tool_counts = Counter(failed_tools)
            dominant_tool, dominant_count = tool_counts.most_common(1)[0]

            if dominant_count >= 3:
                return StuckDiagnosis(
                    is_stuck=True,
                    pattern=StuckPattern.ERROR_SPIRAL,
                    severity=0.9,
                    description=(
                        f"Error spiral: '{dominant_tool}' failing repeatedly "
                        f"({failures}/{len(recent)} actions failed)"
                    ),
                    recovery=RecoveryAction.SWITCH_STRATEGY,
                    recovery_hint=(
                        f"Tool '{dominant_tool}' is consistently failing. "
                        f"Abandon this approach and try something different. "
                        f"Consider reading error messages to understand root cause."
                    ),
                )

            return StuckDiagnosis(
                is_stuck=True,
                pattern=StuckPattern.ERROR_SPIRAL,
                severity=0.7,
                description=f"High failure rate: {failures}/{len(recent)} recent actions failed",
                recovery=RecoveryAction.REPLAN,
                recovery_hint="Multiple tools failing. Re-analyze the problem and create a new plan.",
            )

        if failure_rate >= 0.5:
            return StuckDiagnosis(
                is_stuck=True,
                pattern=StuckPattern.ERROR_SPIRAL,
                severity=0.5,
                description=f"Elevated failure rate: {failures}/{len(recent)} actions failed",
                recovery=RecoveryAction.ESCALATE_MODEL,
                recovery_hint="Moderate failure rate. Try escalating to a more capable model.",
            )

        return self._not_stuck()

    def _check_thrashing(self) -> StuckDiagnosis:
        """Detect alternating between two states (A→B→A→B)."""
        recent = self._history[-8:]
        if len(recent) < 6:
            return self._not_stuck()

        # Check for A-B-A-B pattern in tool names
        tools = [fp.tool for fp in recent]
        alternations = 0
        for i in range(2, len(tools)):
            if tools[i] == tools[i - 2] and tools[i] != tools[i - 1]:
                alternations += 1

        if alternations >= 3:
            tool_a = tools[-2]
            tool_b = tools[-1]
            return StuckDiagnosis(
                is_stuck=True,
                pattern=StuckPattern.THRASHING,
                severity=0.7,
                description=(
                    f"Thrashing detected: alternating between "
                    f"'{tool_a}' and '{tool_b}'"
                ),
                recovery=RecoveryAction.REPLAN,
                recovery_hint=(
                    f"Stop alternating between '{tool_a}' and '{tool_b}'. "
                    f"Step back and think about what you're trying to achieve."
                ),
            )

        return self._not_stuck()

    def _check_semantic_loop(self) -> StuckDiagnosis:
        """Detect different tools producing the same output."""
        recent = self._history[-self._window_size:]
        if len(recent) < 4:
            return self._not_stuck()

        # Count output hashes
        output_counts = Counter(fp.output_hash for fp in recent if fp.success)
        if not output_counts:
            return self._not_stuck()

        most_common_output, count = output_counts.most_common(1)[0]
        if count >= 4:
            return StuckDiagnosis(
                is_stuck=True,
                pattern=StuckPattern.SEMANTIC_LOOP,
                severity=0.6,
                description=(
                    f"Semantic loop: {count} actions produced the same output"
                ),
                recovery=RecoveryAction.SWITCH_STRATEGY,
                recovery_hint=(
                    "Multiple different actions are producing identical results. "
                    "The current approach is not making progress. Try something fundamentally different."
                ),
            )

        return self._not_stuck()

    def _check_stagnation(self) -> StuckDiagnosis:
        """Detect many actions without meaningful progress (no new files changed)."""
        if len(self._history) < self._stagnation_threshold:
            return self._not_stuck()

        recent = self._history[-self._stagnation_threshold:]
        successes = sum(1 for fp in recent if fp.success)

        # If mostly succeeding but no file changes since last check
        current_modified = len(self._files_modified)
        if (
            successes > self._stagnation_threshold // 2
            and current_modified == self._last_files_modified_count
        ):
            # Check if only read operations
            read_tools = {"read_file", "list_dir", "search_files", "git_status",
                          "git_diff", "git_log"}
            all_reads = all(
                fp.tool in read_tools for fp in recent if fp.success
            )
            if all_reads and len(recent) >= self._stagnation_threshold:
                return StuckDiagnosis(
                    is_stuck=True,
                    pattern=StuckPattern.STAGNATION,
                    severity=0.5,
                    description=(
                        f"Stagnation: {len(recent)} actions (all reads) "
                        f"without any file modifications"
                    ),
                    recovery=RecoveryAction.FORCE_DIFFERENT_TOOL,
                    recovery_hint=(
                        "You've been reading files without making changes. "
                        "If you have enough information, start implementing. "
                        "If not, clearly identify what's blocking you."
                    ),
                )

        self._last_files_modified_count = current_modified
        return self._not_stuck()

    @staticmethod
    def _not_stuck() -> StuckDiagnosis:
        return StuckDiagnosis(
            is_stuck=False,
            pattern=StuckPattern.NOT_STUCK,
            severity=0.0,
            description="No stuck patterns detected",
            recovery=RecoveryAction.REPLAN,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_dict(d: dict) -> str:
    """Stable hash of a dictionary for fingerprinting."""
    try:
        serialized = str(sorted(d.items()))
    except Exception:
        serialized = str(d)
    return hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()[:12]


def _hash_str(s: str) -> str:
    """Hash a string for fingerprinting."""
    return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()[:12]
