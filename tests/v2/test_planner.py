"""Tests for the Planner — goal decomposition and dynamic re-planning."""

import pytest

from open_harness_v2.core.planner import ExecutionPlan, PlanStep, Planner


class TestPlanStep:
    """Tests for PlanStep data model."""

    def test_initial_state(self):
        step = PlanStep(id=1, description="Read the file")
        assert step.status == "pending"
        assert step.attempts == 0
        assert not step.is_terminal
        assert step.can_retry

    def test_terminal_states(self):
        for status in ("completed", "skipped", "failed"):
            step = PlanStep(id=1, description="x", status=status)
            assert step.is_terminal

    def test_can_retry_respects_max(self):
        step = PlanStep(id=1, description="x", max_attempts=2)
        step.attempts = 2
        assert not step.can_retry

    def test_can_retry_not_completed(self):
        step = PlanStep(id=1, description="x", status="completed")
        assert not step.can_retry


class TestExecutionPlan:
    """Tests for ExecutionPlan lifecycle."""

    def _make_plan(self, n: int = 3) -> ExecutionPlan:
        return ExecutionPlan(
            goal="test goal",
            steps=[
                PlanStep(id=i + 1, description=f"Step {i + 1}")
                for i in range(n)
            ],
        )

    def test_progress_ratio_empty(self):
        plan = ExecutionPlan(goal="x")
        assert plan.progress_ratio == 0.0

    def test_progress_ratio(self):
        plan = self._make_plan(4)
        plan.steps[0].status = "completed"
        plan.steps[1].status = "completed"
        assert plan.progress_ratio == 0.5

    def test_advance_sets_in_progress(self):
        plan = self._make_plan()
        step = plan.advance()
        assert step is not None
        assert step.status == "in_progress"
        assert step.attempts == 1

    def test_advance_skips_terminal(self):
        plan = self._make_plan()
        plan.steps[0].status = "completed"
        step = plan.advance()
        assert step is not None
        assert step.id == 2

    def test_advance_returns_none_when_complete(self):
        plan = self._make_plan(2)
        plan.steps[0].status = "completed"
        plan.steps[1].status = "completed"
        assert plan.advance() is None
        assert plan.is_complete

    def test_complete_current(self):
        plan = self._make_plan()
        plan.advance()
        plan.complete_current("done")
        assert plan.steps[0].status == "completed"
        assert plan.steps[0].result_summary == "done"
        assert plan.current_step_idx == 1

    def test_fail_current_retries(self):
        plan = self._make_plan()
        plan.advance()  # step 1, attempt 1
        plan.fail_current("error")
        # Should go back to pending for retry
        assert plan.steps[0].status == "pending"
        assert plan.steps[0].attempts == 1

    def test_fail_current_exhausted(self):
        plan = self._make_plan()
        step = plan.advance()
        step.attempts = 3  # max_attempts is 3
        plan.fail_current("final error")
        assert plan.steps[0].status == "failed"

    def test_skip_current(self):
        plan = self._make_plan()
        plan.advance()
        plan.skip_current("not needed")
        assert plan.steps[0].status == "skipped"
        assert plan.current_step_idx == 1

    def test_to_context_block_empty(self):
        plan = ExecutionPlan(goal="x")
        assert plan.to_context_block() == ""

    def test_to_context_block_format(self):
        plan = self._make_plan(2)
        plan.steps[0].status = "completed"
        block = plan.to_context_block()
        assert "1/2 complete" in block
        assert "[x]" in block
        assert "[ ]" in block

    def test_to_dict(self):
        plan = self._make_plan(2)
        d = plan.to_dict()
        assert d["goal"] == "test goal"
        assert len(d["steps"]) == 2
        assert d["progress"] == "0/2"


class TestPlanner:
    """Tests for the Planner (heuristic mode)."""

    async def test_heuristic_question(self):
        planner = Planner()
        plan = await planner.create_plan("What is the main function?")
        assert len(plan.steps) == 1
        assert "answer" in plan.steps[0].description.lower()

    async def test_heuristic_code_change(self):
        planner = Planner()
        plan = await planner.create_plan("Fix the login bug in auth.py")
        assert len(plan.steps) >= 2
        # Should include analyze + implement + verify steps
        descriptions = " ".join(s.description.lower() for s in plan.steps)
        assert "analyze" in descriptions or "understand" in descriptions

    async def test_heuristic_default(self):
        planner = Planner()
        plan = await planner.create_plan("Do something unusual")
        assert len(plan.steps) >= 1

    async def test_llm_plan_with_mock(self):
        """Test LLM-driven planning with a mock."""
        async def mock_llm(messages):
            return '["Read the code", "Make changes", "Run tests"]'

        planner = Planner(llm_call=mock_llm, tool_names=["shell", "read_file"])
        plan = await planner.create_plan("Fix the bug")
        assert len(plan.steps) == 3
        assert plan.steps[0].description == "Read the code"

    async def test_llm_plan_fallback_on_error(self):
        """Test fallback to heuristic when LLM fails."""
        async def broken_llm(messages):
            raise RuntimeError("LLM down")

        planner = Planner(llm_call=broken_llm)
        plan = await planner.create_plan("Fix the bug")
        # Should fall back to heuristic
        assert len(plan.steps) >= 1

    async def test_parse_plan_response_json_array(self):
        result = Planner._parse_plan_response('["step 1", "step 2"]')
        assert result == ["step 1", "step 2"]

    async def test_parse_plan_response_code_block(self):
        result = Planner._parse_plan_response(
            '```json\n["step 1", "step 2"]\n```'
        )
        assert result == ["step 1", "step 2"]

    async def test_parse_plan_response_numbered_list(self):
        result = Planner._parse_plan_response(
            "1. First thing\n2. Second thing\n3. Third thing"
        )
        assert result == ["First thing", "Second thing", "Third thing"]

    async def test_replan_with_mock(self):
        async def mock_llm(messages):
            return '["New step 1", "New step 2"]'

        planner = Planner(llm_call=mock_llm)
        plan = ExecutionPlan(
            goal="Fix bug",
            steps=[
                PlanStep(id=1, description="Read code", status="completed"),
                PlanStep(id=2, description="Old fix", status="pending"),
            ],
        )
        revised = await planner.replan(plan, "Fix approach didn't work")
        assert revised.revision_count == 1
        # Should keep completed step and add new ones
        completed = [s for s in revised.steps if s.status == "completed"]
        assert len(completed) >= 1
