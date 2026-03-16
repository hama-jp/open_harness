"""Tests for GoalComplexity classifier in the Orchestrator."""

import pytest

from open_harness_v2.core.orchestrator import GoalComplexity


class TestGoalComplexity:
    """Test the goal complexity classifier."""

    def test_light_question(self):
        assert GoalComplexity.classify("What is Python?") == GoalComplexity.LIGHT

    def test_light_explain(self):
        assert GoalComplexity.classify("Explain how the router works") == GoalComplexity.LIGHT

    def test_light_show(self):
        assert GoalComplexity.classify("Show me the version") == GoalComplexity.LIGHT

    def test_light_status(self):
        assert GoalComplexity.classify("status") == GoalComplexity.LIGHT

    def test_light_describe(self):
        assert GoalComplexity.classify("Describe the architecture") == GoalComplexity.LIGHT

    def test_heavy_refactor(self):
        assert GoalComplexity.classify(
            "Refactor the entire codebase to use async/await"
        ) == GoalComplexity.HEAVY

    def test_heavy_multi_file_implement(self):
        assert GoalComplexity.classify(
            "Implement a new feature across all files in the project"
        ) == GoalComplexity.HEAVY

    def test_heavy_build_with_tests(self):
        assert GoalComplexity.classify(
            "Build a new module and write tests for it"
        ) == GoalComplexity.HEAVY

    def test_medium_single_fix(self):
        result = GoalComplexity.classify("Fix the login bug")
        assert result in (GoalComplexity.MEDIUM, GoalComplexity.HEAVY)

    def test_medium_add_feature(self):
        result = GoalComplexity.classify("Add feature to the settings page")
        assert result in (GoalComplexity.MEDIUM, GoalComplexity.HEAVY)

    def test_medium_generic(self):
        assert GoalComplexity.classify("Update the config parsing logic") == GoalComplexity.MEDIUM

    def test_empty_goal(self):
        # Empty goals default to MEDIUM
        assert GoalComplexity.classify("") == GoalComplexity.MEDIUM

    def test_short_question(self):
        assert GoalComplexity.classify("What is X?") == GoalComplexity.LIGHT

    def test_list_query(self):
        assert GoalComplexity.classify("List all endpoints") == GoalComplexity.LIGHT
