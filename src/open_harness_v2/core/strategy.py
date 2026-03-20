"""Strategy — adaptive approach management for autonomous goal execution.

Manages multiple strategies for achieving a goal and switches between
them when the current approach is failing. Strategies define how the
agent should approach a problem, not the specific steps.

Inspired by how human developers work: when one approach fails, they
don't retry the same thing — they switch to a different strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
_logger = logging.getLogger(__name__)


class ApproachType(Enum):
    """High-level approaches the agent can take."""

    DIRECT = "direct"              # Solve it directly with available tools
    DELEGATE = "delegate"          # Delegate to external agent
    DECOMPOSE = "decompose"        # Break into smaller sub-problems
    RESEARCH_FIRST = "research_first"  # Read/understand before acting
    INCREMENTAL = "incremental"    # Small changes with frequent verification
    MINIMAL = "minimal"            # Smallest possible change


@dataclass
class Strategy:
    """A specific approach configuration for the agent."""

    approach: ApproachType
    description: str
    system_hint: str  # Injected into system prompt
    priority: int = 0  # Higher = tried first
    attempts: int = 0
    max_attempts: int = 2
    success: bool | None = None  # None = not yet tried

    @property
    def is_exhausted(self) -> bool:
        return self.attempts >= self.max_attempts

    @property
    def is_available(self) -> bool:
        return not self.is_exhausted and self.success is not True


@dataclass
class StrategyState:
    """Current state of strategy management."""

    current: Strategy | None = None
    history: list[tuple[Strategy, str]] = field(default_factory=list)  # (strategy, outcome)


class StrategyManager:
    """Manages and switches between strategies for goal achievement.

    Strategies are pre-defined approaches that modify how the agent
    behaves (via system prompt hints). When one strategy fails, the
    manager selects the next best option.

    Usage::

        mgr = StrategyManager()
        mgr.initialize("implement feature X", has_external_agents=True)
        strategy = mgr.current_strategy
        # ... agent uses strategy.system_hint ...
        # if stuck:
        mgr.mark_failed("kept getting syntax errors")
        next_strategy = mgr.next_strategy()
    """

    def __init__(self) -> None:
        self._strategies: list[Strategy] = []
        self._current_idx: int = -1
        self._state = StrategyState()

    def initialize(
        self,
        goal: str,
        has_external_agents: bool = False,
        goal_complexity: str = "medium",
    ) -> Strategy:
        """Initialize strategies based on goal analysis.

        Returns the first (highest priority) strategy.
        """
        self._strategies = self._generate_strategies(
            goal, has_external_agents, goal_complexity,
        )
        self._current_idx = 0
        self._state = StrategyState()

        if self._strategies:
            strategy = self._strategies[0]
            self._state.current = strategy
            strategy.attempts += 1
            _logger.info(
                "Initial strategy: %s (%s)",
                strategy.approach.value, strategy.description,
            )
            return strategy

        # Fallback
        return Strategy(
            approach=ApproachType.DIRECT,
            description="Direct approach (fallback)",
            system_hint="Solve the goal directly with available tools.",
        )

    @property
    def current_strategy(self) -> Strategy | None:
        return self._state.current

    def mark_failed(self, reason: str = "") -> None:
        """Mark the current strategy as failed."""
        if self._state.current:
            self._state.current.success = False
            self._state.history.append((self._state.current, f"Failed: {reason}"))
            _logger.info(
                "Strategy failed: %s — %s",
                self._state.current.approach.value, reason,
            )

    def mark_succeeded(self) -> None:
        """Mark the current strategy as succeeded."""
        if self._state.current:
            self._state.current.success = True
            self._state.history.append((self._state.current, "Succeeded"))

    def next_strategy(self) -> Strategy | None:
        """Select the next available strategy.

        Returns None if all strategies are exhausted.
        """
        for strategy in self._strategies:
            if strategy.is_available and strategy is not self._state.current:
                self._state.current = strategy
                strategy.attempts += 1
                _logger.info(
                    "Switching to strategy: %s (%s)",
                    strategy.approach.value, strategy.description,
                )
                return strategy

        # If all primary strategies exhausted, try incrementing attempts
        # on the least-tried strategies
        least_tried = sorted(
            [s for s in self._strategies if not s.is_exhausted],
            key=lambda s: s.attempts,
        )
        if least_tried:
            strategy = least_tried[0]
            self._state.current = strategy
            strategy.attempts += 1
            _logger.info(
                "Retrying strategy: %s (attempt %d)",
                strategy.approach.value, strategy.attempts,
            )
            return strategy

        _logger.warning("All strategies exhausted")
        return None

    def get_system_hint(self) -> str:
        """Get the current strategy's system prompt hint."""
        if self._state.current:
            return self._state.current.system_hint
        return ""

    def get_context_block(self) -> str:
        """Generate a context block showing strategy state."""
        if not self._state.current:
            return ""

        lines = [
            f"## Current Strategy: {self._state.current.approach.value}",
            self._state.current.system_hint,
        ]

        if self._state.history:
            lines.append(f"\nPrevious attempts: {len(self._state.history)}")
            for strategy, outcome in self._state.history[-3:]:
                lines.append(f"  - {strategy.approach.value}: {outcome}")

        return "\n".join(lines)

    @property
    def attempts_made(self) -> int:
        return len(self._state.history)

    @property
    def all_exhausted(self) -> bool:
        return all(s.is_exhausted for s in self._strategies)

    # ------------------------------------------------------------------
    # Strategy generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_strategies(
        goal: str,
        has_external_agents: bool,
        goal_complexity: str,
    ) -> list[Strategy]:
        """Generate ordered strategies based on goal characteristics."""
        strategies: list[Strategy] = []
        goal_lower = goal.lower()

        # Always include direct approach first
        strategies.append(Strategy(
            approach=ApproachType.DIRECT,
            description="Solve directly with available tools",
            system_hint=(
                "Solve this goal directly. Use tools efficiently: "
                "read what you need, make changes, verify results. "
                "Don't over-analyze — act decisively."
            ),
            priority=100,
        ))

        # Research-first for complex or investigative tasks
        is_investigative = any(
            w in goal_lower
            for w in ("debug", "investigate", "find", "diagnose", "understand", "analyze")
        )
        if is_investigative or goal_complexity == "heavy":
            strategies.append(Strategy(
                approach=ApproachType.RESEARCH_FIRST,
                description="Understand the problem before acting",
                system_hint=(
                    "IMPORTANT: Before making any changes, thoroughly understand the "
                    "problem. Read relevant files, check git history, understand the "
                    "codebase structure. Only start implementing after you have a "
                    "clear understanding of what needs to change and why."
                ),
                priority=90,
            ))

        # Incremental for code modification tasks
        is_code_change = any(
            w in goal_lower
            for w in ("fix", "add", "implement", "refactor", "change", "update",
                      "create", "modify", "write")
        )
        if is_code_change:
            strategies.append(Strategy(
                approach=ApproachType.INCREMENTAL,
                description="Make small changes and verify frequently",
                system_hint=(
                    "Make changes INCREMENTALLY: one small change at a time, "
                    "then verify it works before moving to the next change. "
                    "Run tests after each change. If a change breaks something, "
                    "revert it and try differently."
                ),
                priority=70,
            ))

        # Minimal approach for bug fixes
        is_bugfix = any(
            w in goal_lower for w in ("fix", "bug", "error", "issue", "broken")
        )
        if is_bugfix:
            strategies.append(Strategy(
                approach=ApproachType.MINIMAL,
                description="Make the smallest possible change to fix the issue",
                system_hint=(
                    "Apply the MINIMAL change needed. Don't refactor, don't "
                    "improve unrelated code. Find the exact root cause and "
                    "fix only that. Verify the fix with a targeted test."
                ),
                priority=60,
            ))

        # Decompose for complex multi-step tasks
        if goal_complexity == "heavy":
            strategies.append(Strategy(
                approach=ApproachType.DECOMPOSE,
                description="Break into independent sub-tasks",
                system_hint=(
                    "This task is complex. Break it into smaller, independent "
                    "sub-tasks. Complete each sub-task fully (including "
                    "verification) before moving to the next. If a sub-task "
                    "fails, skip it and continue with others."
                ),
                priority=50,
            ))

        # External delegation
        if has_external_agents and goal_complexity in ("medium", "heavy"):
            strategies.append(Strategy(
                approach=ApproachType.DELEGATE,
                description="Delegate to external agent (Claude/Codex/Gemini)",
                system_hint=(
                    "Use an external agent (claude_code, codex, or gemini_cli) "
                    "to handle this task. Provide the agent with clear "
                    "instructions, relevant file paths, and specific requirements. "
                    "Review and verify the agent's output."
                ),
                priority=40,
            ))

        # Sort by priority (highest first)
        strategies.sort(key=lambda s: s.priority, reverse=True)
        return strategies
