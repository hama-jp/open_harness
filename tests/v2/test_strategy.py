"""Tests for the StrategyManager — adaptive approach management."""

import pytest

from open_harness_v2.core.strategy import (
    ApproachType,
    Strategy,
    StrategyManager,
)


class TestStrategyManagerInit:
    """Tests for strategy initialization."""

    def test_initialize_returns_strategy(self):
        mgr = StrategyManager()
        s = mgr.initialize("Fix the bug")
        assert s is not None
        assert s.approach is not None

    def test_direct_always_first(self):
        mgr = StrategyManager()
        s = mgr.initialize("anything")
        assert s.approach == ApproachType.DIRECT

    def test_heavy_includes_decompose(self):
        mgr = StrategyManager()
        mgr.initialize("Refactor the entire codebase across all files",
                        goal_complexity="heavy")
        approaches = {s.approach for s in mgr._strategies}
        assert ApproachType.DECOMPOSE in approaches

    def test_bugfix_includes_minimal(self):
        mgr = StrategyManager()
        mgr.initialize("Fix the authentication bug")
        approaches = {s.approach for s in mgr._strategies}
        assert ApproachType.MINIMAL in approaches

    def test_investigative_includes_research(self):
        mgr = StrategyManager()
        mgr.initialize("Debug the slow query in the dashboard")
        approaches = {s.approach for s in mgr._strategies}
        assert ApproachType.RESEARCH_FIRST in approaches

    def test_external_agents_include_delegate(self):
        mgr = StrategyManager()
        mgr.initialize("Fix bug", has_external_agents=True, goal_complexity="medium")
        approaches = {s.approach for s in mgr._strategies}
        assert ApproachType.DELEGATE in approaches

    def test_no_external_agents_no_delegate(self):
        mgr = StrategyManager()
        mgr.initialize("Fix bug", has_external_agents=False)
        approaches = {s.approach for s in mgr._strategies}
        assert ApproachType.DELEGATE not in approaches


class TestStrategySwitching:
    """Tests for strategy switching."""

    def test_mark_failed_and_next(self):
        mgr = StrategyManager()
        first = mgr.initialize("Fix the bug in auth.py")
        assert first.approach == ApproachType.DIRECT

        mgr.mark_failed("kept getting errors")
        second = mgr.next_strategy()
        assert second is not None
        assert second.approach != first.approach

    def test_all_exhausted(self):
        mgr = StrategyManager()
        mgr.initialize("Simple task", goal_complexity="light")

        # Exhaust all strategies
        for _ in range(20):
            mgr.mark_failed("nope")
            nxt = mgr.next_strategy()
            if nxt is None:
                break

        # Eventually should be exhausted
        assert mgr.all_exhausted or mgr.next_strategy() is None

    def test_mark_succeeded(self):
        mgr = StrategyManager()
        mgr.initialize("Fix bug")
        mgr.mark_succeeded()
        assert mgr.current_strategy.success is True


class TestStrategyHints:
    """Tests for system prompt hints."""

    def test_get_system_hint(self):
        mgr = StrategyManager()
        mgr.initialize("Fix bug")
        hint = mgr.get_system_hint()
        assert hint  # Should not be empty
        assert isinstance(hint, str)

    def test_get_context_block(self):
        mgr = StrategyManager()
        mgr.initialize("Fix bug")
        block = mgr.get_context_block()
        assert "Strategy" in block or "strategy" in block

    def test_context_block_shows_history(self):
        mgr = StrategyManager()
        mgr.initialize("Fix the login bug")
        mgr.mark_failed("syntax errors")
        mgr.next_strategy()
        block = mgr.get_context_block()
        assert "Previous attempts" in block or "previous" in block.lower()


class TestStrategyState:
    """Tests for strategy state tracking."""

    def test_attempts_made(self):
        mgr = StrategyManager()
        mgr.initialize("task")
        assert mgr.attempts_made == 0
        mgr.mark_failed("fail")
        assert mgr.attempts_made == 1

    def test_strategy_is_exhausted(self):
        s = Strategy(
            approach=ApproachType.DIRECT,
            description="test",
            system_hint="hint",
            max_attempts=2,
        )
        s.attempts = 2
        assert s.is_exhausted
        assert not s.is_available
