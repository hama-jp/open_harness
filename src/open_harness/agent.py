"""Core agent loop — interactive and autonomous goal-driven modes."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator

from open_harness.checkpoint import CheckpointEngine
from open_harness.config import HarnessConfig
from open_harness.llm.client import LLMResponse, ToolCall
from open_harness.llm.compensator import (
    Compensator,
    build_autonomous_prompt,
    build_tool_prompt,
    truncate_tool_output,
)
from open_harness.llm.router import ModelRouter
from open_harness.memory.store import MemoryStore
from open_harness.planner import Plan, PlanCritic, PlanStep, Planner, StepResult
from open_harness.policy import PolicyEngine, load_policy
from open_harness.project import ProjectContext
from open_harness.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

MAX_INTERACTIVE_STEPS = 15
MAX_GOAL_STEPS = 50  # Autonomous mode gets more room


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: str  # thinking, text, tool_call, tool_result, compensation, status, done
    data: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent:
    """ReAct agent with interactive and autonomous goal-driven modes."""

    def __init__(
        self,
        config: HarnessConfig,
        tools: ToolRegistry,
        memory: MemoryStore,
        project: ProjectContext | None = None,
    ):
        self.config = config
        self.tools = tools
        self.memory = memory
        self.project = project or ProjectContext()
        self.router = ModelRouter(config)
        self.compensator = Compensator(config.compensation)
        self.policy = PolicyEngine(load_policy(config.policy))
        self.policy.set_project_root(self.project.root)
        self.planner = Planner(self.router)
        self.plan_critic = PlanCritic()
        self._interactive_prompt: str | None = None
        self._autonomous_prompt: str | None = None

    @property
    def interactive_prompt(self) -> str:
        if self._interactive_prompt is None:
            self._interactive_prompt = build_tool_prompt(
                self.tools.get_prompt_description(),
                self.config.compensation.thinking_mode,
            )
        return self._interactive_prompt

    @property
    def autonomous_prompt(self) -> str:
        if self._autonomous_prompt is None:
            self._autonomous_prompt = build_autonomous_prompt(
                self.tools.get_prompt_description(),
                self.project.to_prompt(),
                self.config.compensation.thinking_mode,
            )
        return self._autonomous_prompt

    def invalidate_prompts(self):
        """Call after tools or project context change."""
        self._interactive_prompt = None
        self._autonomous_prompt = None

    # ------------------------------------------------------------------
    # Interactive mode (existing behavior)
    # ------------------------------------------------------------------

    def run_stream(self, user_message: str) -> Generator[AgentEvent, None, None]:
        """Interactive single-turn with streaming."""
        self.compensator.reset()
        self.policy.begin_goal()  # reset budgets per turn
        self.memory.add_turn("user", user_message)
        messages = [
            {"role": "system", "content": self.interactive_prompt},
            *self.memory.get_messages(),
        ]
        yield from self._agent_loop(messages, MAX_INTERACTIVE_STEPS)

    def run(self, user_message: str) -> str:
        final = ""
        for ev in self.run_stream(user_message):
            if ev.type == "done":
                final = ev.data
        return final

    # ------------------------------------------------------------------
    # Autonomous goal mode — the core of "self-driving"
    # ------------------------------------------------------------------

    def run_goal(self, goal: str) -> Generator[AgentEvent, None, None]:
        """Run autonomously toward a goal.

        Attempts to plan first (break into steps), then execute each step.
        Falls back to direct execution if planning fails.
        """
        self.compensator.reset()
        self.policy.begin_goal()

        yield AgentEvent("status", f"Goal: {goal} [policy: {self.policy.config.mode}]")
        yield AgentEvent("status", f"Project: {self.project.info['type']} @ {self.project.info['root']}")

        # Checkpoint engine for transactional safety
        ckpt = CheckpointEngine(
            self.project.root,
            has_git=self.project.info.get("has_git", False),
        )
        ckpt_status = ckpt.begin()
        yield AgentEvent("status", f"Checkpoint: {ckpt_status}")

        try:
            # Phase 1: Try to create a plan
            yield AgentEvent("status", "Planning...")
            plan, err = self.planner.create_plan(
                goal, context=self.project.to_prompt(),
            )

            if plan:
                issues = self.plan_critic.validate(plan)
                if issues:
                    yield AgentEvent("status", f"Plan rejected: {'; '.join(issues)}")
                    plan = None

            if plan:
                yield AgentEvent("status", plan.summary())
                yield from self._run_planned_goal(goal, plan, ckpt)
            else:
                # Fallback: direct execution without planning
                reason = err.reason if err else "critic rejected"
                yield AgentEvent("compensation",
                    f"Planning failed ({reason}); switching to direct mode")
                yield from self._run_direct_goal(goal, ckpt)

        finally:
            yield AgentEvent("status", f"Budget: {self.policy.budget.summary()}")
            finish_status = ckpt.finish(keep_changes=True)
            if finish_status:
                yield AgentEvent("status", f"Finish: {finish_status}")

    def _run_direct_goal(
        self, goal: str, ckpt: CheckpointEngine,
    ) -> Generator[AgentEvent, None, None]:
        """Direct autonomous execution without planning (fallback)."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.autonomous_prompt},
            {"role": "user", "content": (
                f"GOAL: {goal}\n\n"
                "Work autonomously to achieve this goal. "
                "Do not ask me questions — just do it."
            )},
        ]
        yield from self._agent_loop(messages, MAX_GOAL_STEPS, checkpoint=ckpt)

    def _run_planned_goal(
        self, goal: str, plan: Plan, ckpt: CheckpointEngine,
    ) -> Generator[AgentEvent, None, None]:
        """Execute a planned goal step by step with checkpoints."""
        completed: list[PlanStep] = []
        results: list[StepResult] = []
        steps_per_step = MAX_GOAL_STEPS // max(len(plan.steps), 1)

        for i, step in enumerate(plan.steps):
            yield AgentEvent("status",
                f"Step {i+1}/{len(plan.steps)}: {step.title}")

            # Execute this step
            step_result = yield from self._execute_plan_step(
                goal, step, completed, ckpt, steps_per_step,
            )
            results.append(step_result)

            if step_result.success:
                completed.append(step)
                # Snapshot after each successful step
                snap = ckpt.snapshot(f"step {i+1} complete: {step.title}")
                if snap:
                    yield AgentEvent("status", f"Snapshot: {snap.commit_hash}")
            else:
                # Step failed — try replanning once
                yield AgentEvent("compensation",
                    f"Step '{step.title}' failed: {step_result.summary}")

                # Rollback to last successful snapshot
                if ckpt.snapshots:
                    rb = ckpt.rollback(ckpt.snapshots[-1])
                    yield AgentEvent("status", f"Rollback: {rb}")

                new_plan, err = self.planner.replan_remaining(
                    goal, completed, step, step_result.summary,
                )

                if new_plan and not self.plan_critic.validate(new_plan):
                    yield AgentEvent("status",
                        f"Replanned: {new_plan.summary()}")
                    # Continue with new plan
                    yield from self._run_planned_goal(
                        goal, new_plan, ckpt)
                    return
                else:
                    # Replanning failed — fall back to direct execution
                    reason = err.reason if err else "critic rejected replan"
                    yield AgentEvent("compensation",
                        f"Replan failed ({reason}); switching to direct mode")
                    remaining_context = (
                        f"GOAL: {goal}\n\n"
                        f"Completed steps so far:\n"
                        + "\n".join(f"  - {s.title}" for s in completed)
                        + f"\n\nFailed step: {step.title} ({step_result.summary})\n"
                        f"Continue working to achieve the goal."
                    )
                    messages: list[dict[str, Any]] = [
                        {"role": "system", "content": self.autonomous_prompt},
                        {"role": "user", "content": remaining_context},
                    ]
                    yield from self._agent_loop(
                        messages, MAX_GOAL_STEPS, checkpoint=ckpt)
                    return

        # All steps completed
        yield AgentEvent("status",
            f"Plan completed: {len(completed)}/{len(plan.steps)} steps succeeded")

    def _execute_plan_step(
        self,
        goal: str,
        step: PlanStep,
        completed: list[PlanStep],
        ckpt: CheckpointEngine,
        max_steps: int,
    ) -> Generator[AgentEvent, None, StepResult]:
        """Execute a single plan step using the agent loop."""
        context_parts = [f"Overall goal: {goal}"]
        if completed:
            context_parts.append("Completed steps:")
            for s in completed:
                context_parts.append(f"  - {s.title}")
        context_parts.append(f"\nCurrent task:\n{step.to_prompt()}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.autonomous_prompt},
            {"role": "user", "content": "\n".join(context_parts)},
        ]

        # Collect events from the agent loop
        final_text = ""
        had_errors = False
        for event in self._agent_loop(messages, max_steps, checkpoint=ckpt):
            yield event
            if event.type == "done":
                final_text = event.data
            elif event.type == "tool_result" and not event.metadata.get("success"):
                had_errors = True

        # Determine success: the step "succeeded" if the agent completed
        # without hitting the step limit and produced output
        success = bool(final_text) and "[Reached" not in final_text
        return StepResult(
            step_id=step.step_id,
            success=success,
            summary=final_text[:200] if final_text else "No output",
        )

    # ------------------------------------------------------------------
    # Shared agent loop
    # ------------------------------------------------------------------

    def _agent_loop(
        self,
        messages: list[dict[str, Any]],
        max_steps: int,
        tier: str | None = None,
        checkpoint: CheckpointEngine | None = None,
    ) -> Generator[AgentEvent, None, None]:
        """Core ReAct loop shared by interactive and goal modes."""
        tier = tier or self.router.current_tier
        step = 0
        _writes_since_snapshot = 0

        while step < max_steps:
            step += 1
            yield AgentEvent("status", f"[{step}/{max_steps}] {tier}")

            response = yield from self._stream_llm(messages, tier)

            # API error
            if response.finish_reason == "error":
                comp = self.compensator.next_strategy(
                    messages, response.content, "API error", tier)
                if comp and comp.success:
                    yield AgentEvent("compensation", comp.notes)
                    if comp.modified_messages:
                        messages = comp.modified_messages
                    if comp.escalated_tier:
                        tier = comp.escalated_tier
                    continue
                self.memory.add_turn("assistant", response.content)
                yield AgentEvent("done", response.content)
                return

            # Tool call
            if response.has_tool_call:
                tc = response.tool_calls[0]
                yield AgentEvent("tool_call", tc.name, {"tool": tc.name, "args": tc.arguments})

                # Policy check before execution
                violation = self.policy.check(tc.name, tc.arguments)
                if violation:
                    result = ToolResult(
                        success=False, output="",
                        error=f"[Policy: {violation.rule}] {violation.message}",
                    )
                else:
                    result = self.tools.execute(tc.name, tc.arguments)
                    self.policy.record(tc.name)

                output = truncate_tool_output(result.to_message(), 8000)
                yield AgentEvent("tool_result", output,
                                 {"success": result.success, "tool": tc.name})

                # Auto-snapshot after file writes (every 5 writes)
                if checkpoint and tc.name in ("write_file", "edit_file") and result.success:
                    _writes_since_snapshot += 1
                    if _writes_since_snapshot >= 5:
                        snap = checkpoint.snapshot(f"after {_writes_since_snapshot} writes (step {step})")
                        if snap:
                            yield AgentEvent("status", f"Snapshot: {snap.commit_hash}")
                        _writes_since_snapshot = 0

                # Auto-rollback on test failure (if checkpoint active)
                if (checkpoint and tc.name == "run_tests"
                        and not result.success and checkpoint.snapshots):
                    yield AgentEvent("compensation",
                        "Tests failed — rolling back to last snapshot")
                    rb = checkpoint.rollback(checkpoint.snapshots[-1])
                    yield AgentEvent("status", f"Rollback: {rb}")
                    # Tell the LLM about the rollback
                    messages.append({"role": "assistant",
                        "content": f'{{"tool": "{tc.name}", "args": {_safe_json(tc.arguments)}}}'})
                    messages.append({"role": "user",
                        "content": f"[Tool Result for {tc.name}]\n{output}\n\n"
                                   f"[ROLLBACK] Changes have been rolled back to the last "
                                   f"snapshot. Try a different approach."})
                    continue

                messages.append({"role": "assistant",
                    "content": f'{{"tool": "{tc.name}", "args": {_safe_json(tc.arguments)}}}'})
                messages.append({"role": "user",
                    "content": f"[Tool Result for {tc.name}]\n{output}"})
                continue

            # Malformed tool call
            if self._looks_like_failed_tool_call(response.content):
                comp = self.compensator.next_strategy(
                    messages, response.content, "Malformed tool call", tier)
                if comp and comp.success:
                    yield AgentEvent("compensation", comp.notes)
                    if comp.modified_messages:
                        messages = comp.modified_messages
                    if comp.escalated_tier:
                        tier = comp.escalated_tier
                    continue

            # Auto-snapshot before finishing (capture final state)
            if checkpoint and _writes_since_snapshot > 0:
                snap = checkpoint.snapshot(f"goal complete (step {step})")
                if snap:
                    yield AgentEvent("status", f"Final snapshot: {snap.commit_hash}")

            # Text response — done
            self.memory.add_turn("assistant", response.content)
            yield AgentEvent("done", response.content,
                             {"latency_ms": response.latency_ms, "steps": step})
            return

        # Step limit — try escalation
        comp = self.compensator.on_step_limit(messages, tier, step)
        if comp and comp.success:
            yield AgentEvent("compensation", comp.notes)
            messages = comp.modified_messages or messages
            tier = comp.escalated_tier or tier
            yield from self._agent_loop(messages, max_steps, tier=tier,
                                        checkpoint=checkpoint)
            return

        yield AgentEvent("done",
            f"[Reached {max_steps} steps. Use /tier large or simplify the goal.]")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stream_llm(self, messages, tier) -> Generator[AgentEvent, None, LLMResponse]:
        gen = self.router.chat_stream(messages=messages, tier=tier, temperature=0.3)
        response: LLMResponse | None = None
        try:
            while True:
                etype, data = next(gen)
                if etype == "thinking":
                    yield AgentEvent("thinking", data)
                elif etype == "text":
                    yield AgentEvent("text", data)
        except StopIteration as e:
            response = e.value
        if response is None:
            response = LLMResponse(content="", finish_reason="error")
        return response

    @staticmethod
    def _looks_like_failed_tool_call(content: str) -> bool:
        indicators = [
            '"tool"' in content and '"args"' in content and "{" in content,
            "```json" in content and '"tool"' in content,
            content.strip().startswith("{") and '"tool_call"' in content,
        ]
        return any(indicators) and len(content) < 500



def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)
