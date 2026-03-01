"""Tests for Issue 7: Token/cost monitoring."""

from open_harness.policy import PolicyConfig, PolicyEngine


class TestTokenBudget:
    def test_unlimited_by_default(self):
        """Default max_tokens_per_goal=0 means no limit."""
        engine = PolicyEngine(PolicyConfig())
        engine.begin_goal()
        engine.record_usage({"total_tokens": 100000})
        assert engine.check_token_budget() is None

    def test_budget_exceeded(self):
        """Should return reason when budget is exceeded."""
        config = PolicyConfig(max_tokens_per_goal=1000)
        engine = PolicyEngine(config)
        engine.begin_goal()

        engine.record_usage({"total_tokens": 500})
        assert engine.check_token_budget() is None

        engine.record_usage({"total_tokens": 600})
        result = engine.check_token_budget()
        assert result is not None
        assert "exceeded" in result.lower()
        assert "1100" in result  # actual usage
        assert "1000" in result  # budget

    def test_budget_reset_on_new_goal(self):
        """begin_goal() should reset token counter."""
        config = PolicyConfig(max_tokens_per_goal=1000)
        engine = PolicyEngine(config)
        engine.begin_goal()
        engine.record_usage({"total_tokens": 900})

        engine.begin_goal()
        assert engine._token_usage == 0
        assert engine.check_token_budget() is None

    def test_record_usage_accumulates(self):
        """Multiple record_usage calls should accumulate."""
        engine = PolicyEngine(PolicyConfig(max_tokens_per_goal=500))
        engine.begin_goal()

        engine.record_usage({"total_tokens": 100})
        engine.record_usage({"total_tokens": 200})
        engine.record_usage({"total_tokens": 300})
        assert engine._token_usage == 600
        assert engine.check_token_budget() is not None

    def test_empty_usage_dict(self):
        """Should handle empty usage dict gracefully."""
        engine = PolicyEngine(PolicyConfig(max_tokens_per_goal=100))
        engine.begin_goal()
        engine.record_usage({})
        assert engine._token_usage == 0
        assert engine.check_token_budget() is None

    def test_missing_total_tokens_key(self):
        """Should handle usage dict without total_tokens."""
        engine = PolicyEngine(PolicyConfig(max_tokens_per_goal=100))
        engine.begin_goal()
        engine.record_usage({"prompt_tokens": 50, "completion_tokens": 30})
        assert engine._token_usage == 0
