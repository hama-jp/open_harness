"""Planner — LLM-driven goal decomposition and dynamic re-planning.

Decomposes a high-level goal into actionable sub-steps, tracks plan
execution, and supports dynamic re-planning when the agent encounters
obstacles or discovers new information.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in an execution plan."""

    id: int
    description: str
    status: str = "pending"  # pending | in_progress | completed | skipped | failed
    result_summary: str = ""
    attempts: int = 0
    max_attempts: int = 3

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "skipped", "failed")

    @property
    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts and self.status != "completed"


@dataclass
class ExecutionPlan:
    """A structured plan for achieving a goal."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    current_step_idx: int = 0
    revision_count: int = 0

    @property
    def current_step(self) -> PlanStep | None:
        if 0 <= self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return all(s.is_terminal for s in self.steps)

    @property
    def progress_ratio(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.is_terminal)
        return done / len(self.steps)

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "completed")

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")

    def advance(self) -> PlanStep | None:
        """Move to the next non-terminal step. Returns it, or None if done."""
        while self.current_step_idx < len(self.steps):
            step = self.steps[self.current_step_idx]
            if not step.is_terminal:
                step.status = "in_progress"
                step.attempts += 1
                return step
            self.current_step_idx += 1
        return None

    def complete_current(self, summary: str = "") -> None:
        step = self.current_step
        if step:
            step.status = "completed"
            step.result_summary = summary
            self.current_step_idx += 1

    def fail_current(self, reason: str = "") -> None:
        step = self.current_step
        if step:
            if step.can_retry:
                step.status = "pending"  # will retry
                step.result_summary = f"Failed (attempt {step.attempts}): {reason}"
            else:
                step.status = "failed"
                step.result_summary = reason

    def skip_current(self, reason: str = "") -> None:
        step = self.current_step
        if step:
            step.status = "skipped"
            step.result_summary = reason
            self.current_step_idx += 1

    def to_context_block(self) -> str:
        """Render the plan as a context block for the LLM."""
        if not self.steps:
            return ""
        lines = [
            f"## Execution Plan ({self.completed_count}/{len(self.steps)} complete)",
        ]
        for step in self.steps:
            icon = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
                "skipped": "[-]",
                "failed": "[!]",
            }.get(step.status, "[ ]")
            line = f"  {icon} {step.id}. {step.description}"
            if step.result_summary and step.is_terminal:
                line += f"  ({step.result_summary[:80]})"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "status": s.status,
                    "result_summary": s.result_summary,
                }
                for s in self.steps
            ],
            "progress": f"{self.completed_count}/{len(self.steps)}",
        }


# ---------------------------------------------------------------------------
# Plan generation prompt
# ---------------------------------------------------------------------------

_PLAN_GENERATION_PROMPT = """\
You are a planning assistant. Given a goal, decompose it into a concrete, \
ordered list of actionable steps.

Rules:
- Each step should be a single, clear action (not vague).
- Steps should be in execution order (dependencies respected).
- Include verification steps where appropriate (e.g., "Run tests to verify").
- Keep the plan concise: 3-10 steps for most goals.
- For simple goals (questions, explanations), use 1-2 steps.
- Each step should be achievable with the available tools.

Respond with ONLY a JSON array of step descriptions:
["Step 1 description", "Step 2 description", ...]

Goal: {goal}

Available tools: {tools}
"""

_REPLAN_PROMPT = """\
The current plan needs revision. Here is the situation:

Original goal: {goal}

Current plan state:
{plan_state}

Reason for re-planning: {reason}

Create a revised plan for the REMAINING work. Only include steps that still \
need to be done. Respond with ONLY a JSON array:
["Remaining step 1", "Remaining step 2", ...]
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """Decomposes goals into execution plans and supports re-planning.

    The Planner can operate in two modes:
    1. **LLM-driven**: Uses the LLM to generate plans (requires a pipeline).
    2. **Heuristic**: Falls back to a single-step plan when no LLM is available.
    """

    def __init__(
        self,
        llm_call: Any = None,  # async callable: (messages, model) -> LLMResponse
        tool_names: list[str] | None = None,
    ) -> None:
        self._llm_call = llm_call
        self._tool_names = tool_names or []

    async def create_plan(self, goal: str) -> ExecutionPlan:
        """Create an execution plan for the given goal.

        Tries LLM-driven planning first, falls back to heuristic.
        """
        plan = ExecutionPlan(goal=goal)

        if self._llm_call:
            steps = await self._llm_plan(goal)
            if steps:
                for i, desc in enumerate(steps, 1):
                    plan.steps.append(PlanStep(id=i, description=desc))
                _logger.info(
                    "Created %d-step plan for: %.80s",
                    len(plan.steps), goal,
                )
                return plan

        # Heuristic fallback
        plan.steps = self._heuristic_plan(goal)
        return plan

    async def replan(
        self, plan: ExecutionPlan, reason: str,
    ) -> ExecutionPlan:
        """Revise the plan based on new information or obstacles.

        Preserves completed steps and replaces remaining ones.
        """
        plan.revision_count += 1

        if self._llm_call:
            new_steps = await self._llm_replan(plan, reason)
            if new_steps:
                # Keep completed/skipped steps, replace the rest
                kept = [s for s in plan.steps if s.is_terminal]
                next_id = max((s.id for s in kept), default=0) + 1
                for i, desc in enumerate(new_steps):
                    kept.append(PlanStep(id=next_id + i, description=desc))
                plan.steps = kept
                plan.current_step_idx = len(
                    [s for s in plan.steps if s.is_terminal]
                )
                _logger.info(
                    "Re-planned (revision %d): %d remaining steps",
                    plan.revision_count,
                    len(new_steps),
                )
                return plan

        # Fallback: just mark current as failed and add a generic retry
        plan.fail_current(reason)
        return plan

    # ------------------------------------------------------------------
    # LLM-driven planning
    # ------------------------------------------------------------------

    async def _llm_plan(self, goal: str) -> list[str]:
        """Use LLM to generate a plan."""
        try:
            prompt = _PLAN_GENERATION_PROMPT.format(
                goal=goal,
                tools=", ".join(self._tool_names) if self._tool_names else "general tools",
            )
            messages = [
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": prompt},
            ]
            response = await self._llm_call(messages)
            return self._parse_plan_response(response)
        except Exception as e:
            _logger.warning("LLM planning failed: %s", e)
            return []

    async def _llm_replan(
        self, plan: ExecutionPlan, reason: str,
    ) -> list[str]:
        """Use LLM to revise the plan."""
        try:
            prompt = _REPLAN_PROMPT.format(
                goal=plan.goal,
                plan_state=plan.to_context_block(),
                reason=reason,
            )
            messages = [
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": prompt},
            ]
            response = await self._llm_call(messages)
            return self._parse_plan_response(response)
        except Exception as e:
            _logger.warning("LLM re-planning failed: %s", e)
            return []

    @staticmethod
    def _parse_plan_response(response_text: str) -> list[str]:
        """Parse a JSON array of step descriptions from LLM output."""
        text = response_text.strip()

        # Try direct JSON parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(s).strip() for s in result if str(s).strip()]
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(1))
                if isinstance(result, list):
                    return [str(s).strip() for s in result if str(s).strip()]
            except json.JSONDecodeError:
                pass

        # Try extracting bare JSON array
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return [str(s).strip() for s in result if str(s).strip()]
            except json.JSONDecodeError:
                pass

        # Last resort: numbered list
        lines = text.split("\n")
        steps = []
        for line in lines:
            line = line.strip()
            m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
            if m:
                steps.append(m.group(1).strip())
        return steps

    # ------------------------------------------------------------------
    # Heuristic planning
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_plan(goal: str) -> list[PlanStep]:
        """Generate a simple heuristic plan without LLM."""
        goal_lower = goal.lower()

        # Question / explanation → single step
        if any(
            goal_lower.startswith(w)
            for w in ("what ", "how ", "why ", "explain ", "describe ")
        ):
            return [
                PlanStep(id=1, description=f"Answer: {goal}"),
            ]

        # Code change → analyze + implement + verify
        if any(
            w in goal_lower
            for w in ("fix", "add", "implement", "create", "refactor", "change", "update")
        ):
            return [
                PlanStep(id=1, description="Analyze the codebase to understand current state"),
                PlanStep(id=2, description=f"Implement the changes: {goal}"),
                PlanStep(id=3, description="Verify changes work correctly (run tests if available)"),
            ]

        # Default: single execution step
        return [
            PlanStep(id=1, description=goal),
        ]
