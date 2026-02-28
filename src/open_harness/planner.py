"""Planner-Critic-Executor loop for autonomous goal execution.

Breaks goals into discrete steps, executes each with verification,
and self-corrects on failure. Degrades gracefully to direct execution
if planning fails (critical for weak local LLMs).

Architecture:
  Planner  → creates Plan (list of PlanSteps) from a goal
  PlanCritic → validates Plan with rule-based checks (no LLM call)
  Agent.run_goal() orchestrates: plan → execute step → snapshot → next
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Hard limits to keep plans small and manageable for weak LLMs
MAX_PLAN_STEPS = 5
PLANNING_MAX_TOKENS = 2048


@dataclass
class PlanStep:
    """A single step in a plan."""
    step_id: str
    title: str
    instruction: str
    success_criteria: list[str] = field(default_factory=list)
    max_agent_steps: int = 12

    def to_prompt(self) -> str:
        criteria = "\n".join(f"  - {c}" for c in self.success_criteria) if self.success_criteria else "  - Step completes without errors"
        return (
            f"## Step: {self.title}\n\n"
            f"{self.instruction}\n\n"
            f"Success criteria:\n{criteria}\n\n"
            f"Focus ONLY on this step. Do not work on other steps."
        )


@dataclass
class Plan:
    """A structured plan for achieving a goal."""
    goal: str
    steps: list[PlanStep]
    assumptions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"Plan ({len(self.steps)} steps):"]
        for i, s in enumerate(self.steps, 1):
            parts.append(f"  {i}. {s.title}")
        return "\n".join(parts)


@dataclass
class StepResult:
    """Result of executing a single plan step."""
    step_id: str
    success: bool
    summary: str
    attempts: int = 1


@dataclass
class PlanFailure:
    """Describes why planning failed."""
    reason: str
    raw_output: str = ""
    recoverable: bool = True


# -----------------------------------------------------------------------
# Planner — creates plans from goals
# -----------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """You are a planning assistant. Given a goal, break it into a small number of concrete steps.

RULES:
- Maximum {max_steps} steps. Fewer is better.
- Each step must be independently verifiable.
- Steps should be ordered by dependency.
- Be specific and actionable — no vague steps.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{
  "steps": [
    {{
      "title": "Short title",
      "instruction": "Detailed instruction for what to do",
      "success_criteria": ["How to verify this step succeeded"]
    }}
  ],
  "assumptions": ["Any assumptions about the project"]
}}"""

REPLAN_PROMPT = """The original goal was: {goal}

Completed steps:
{completed}

Step "{failed_title}" FAILED: {failure_reason}

Create a revised plan for the REMAINING work only. The completed steps are already done.
Respond with ONLY a JSON object in the same format as before."""


class Planner:
    """Creates structured plans from goals using the LLM."""

    def __init__(self, router: Any, max_steps: int = MAX_PLAN_STEPS):
        self.router = router
        self.max_steps = min(max_steps, MAX_PLAN_STEPS)

    def create_plan(
        self,
        goal: str,
        context: str = "",
        tier: str = "small",
    ) -> tuple[Plan | None, PlanFailure | None]:
        """Generate a plan for the given goal.

        Returns (Plan, None) on success or (None, PlanFailure) on failure.
        """
        system = PLAN_SYSTEM_PROMPT.format(max_steps=self.max_steps)
        user_msg = f"GOAL: {goal}"
        if context:
            user_msg += f"\n\nCONTEXT:\n{context}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = self.router.chat(
                messages=messages,
                tier=tier,
                max_tokens=PLANNING_MAX_TOKENS,
                temperature=0.2,
            )
        except Exception as e:
            return None, PlanFailure(reason=f"LLM error: {e}")

        if not response.content:
            return None, PlanFailure(reason="Empty response from LLM")

        return self._parse_plan(goal, response.content)

    def replan_remaining(
        self,
        goal: str,
        completed: list[PlanStep],
        failed_step: PlanStep,
        failure_reason: str,
        tier: str = "small",
    ) -> tuple[Plan | None, PlanFailure | None]:
        """Create a revised plan after a step failure."""
        completed_text = "\n".join(
            f"  {i+1}. {s.title} (DONE)" for i, s in enumerate(completed)
        ) or "  (none)"

        system = PLAN_SYSTEM_PROMPT.format(max_steps=self.max_steps)
        user_msg = REPLAN_PROMPT.format(
            goal=goal,
            completed=completed_text,
            failed_title=failed_step.title,
            failure_reason=failure_reason,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = self.router.chat(
                messages=messages, tier=tier,
                max_tokens=PLANNING_MAX_TOKENS, temperature=0.2,
            )
        except Exception as e:
            return None, PlanFailure(reason=f"Replan LLM error: {e}")

        if not response.content:
            return None, PlanFailure(reason="Empty replan response")

        return self._parse_plan(goal, response.content)

    def _parse_plan(self, goal: str, raw: str) -> tuple[Plan | None, PlanFailure | None]:
        """Parse LLM output into a Plan."""
        # Try to extract JSON from the response
        json_str = _extract_json(raw)
        if not json_str:
            return None, PlanFailure(
                reason="Could not extract JSON from planner output",
                raw_output=raw[:500],
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return None, PlanFailure(
                reason=f"Invalid JSON: {e}",
                raw_output=raw[:500],
            )

        steps_raw = data.get("steps", [])
        if not isinstance(steps_raw, list) or not steps_raw:
            return None, PlanFailure(reason="No steps in plan", raw_output=raw[:500])

        # Enforce step limit
        steps_raw = steps_raw[:self.max_steps]

        steps = []
        for i, s in enumerate(steps_raw):
            if not isinstance(s, dict):
                continue
            title = str(s.get("title", f"Step {i+1}"))
            instruction = str(s.get("instruction", title))
            criteria = s.get("success_criteria", [])
            if isinstance(criteria, str):
                criteria = [criteria]
            elif not isinstance(criteria, list):
                criteria = []
            criteria = [str(c) for c in criteria]

            steps.append(PlanStep(
                step_id=f"step_{i+1}",
                title=title,
                instruction=instruction,
                success_criteria=criteria,
            ))

        if not steps:
            return None, PlanFailure(reason="No valid steps parsed", raw_output=raw[:500])

        assumptions = data.get("assumptions", [])
        if isinstance(assumptions, str):
            assumptions = [assumptions]
        elif not isinstance(assumptions, list):
            assumptions = []

        return Plan(goal=goal, steps=steps, assumptions=[str(a) for a in assumptions]), None


# -----------------------------------------------------------------------
# PlanCritic — rule-based plan validation (no LLM call)
# -----------------------------------------------------------------------

class PlanCritic:
    """Validates plans with rule-based checks. No LLM call needed."""

    def __init__(self, max_steps: int = MAX_PLAN_STEPS):
        self.max_steps = max_steps

    def validate(self, plan: Plan) -> list[str]:
        """Return list of issues. Empty = plan is OK."""
        issues = []

        if not plan.steps:
            issues.append("Plan has no steps")
            return issues

        if len(plan.steps) > self.max_steps:
            issues.append(f"Too many steps ({len(plan.steps)} > {self.max_steps})")

        for step in plan.steps:
            if not step.title or not step.title.strip():
                issues.append(f"Step {step.step_id} has empty title")
            if not step.instruction or not step.instruction.strip():
                issues.append(f"Step {step.step_id} has empty instruction")
            if len(step.instruction) < 10:
                issues.append(f"Step {step.step_id} instruction too vague: '{step.instruction}'")

        # Check for duplicate titles (likely a copy-paste hallucination)
        titles = [s.title.lower().strip() for s in plan.steps]
        if len(set(titles)) < len(titles):
            issues.append("Plan contains duplicate step titles (possible hallucination)")

        return issues


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _extract_json(text: str) -> str | None:
    """Extract JSON object from potentially messy LLM output."""
    # Try the whole text first
    text = text.strip()
    if text.startswith("{"):
        return text

    # Try to find JSON in markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # Try to find any JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return None
