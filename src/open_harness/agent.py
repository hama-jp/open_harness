"""Core agent loop — interactive and autonomous goal-driven modes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Generator

from open_harness.checkpoint import CheckpointEngine
from open_harness.config import HarnessConfig
from open_harness.llm.client import LLMResponse
from open_harness.llm.compensator import (
    Compensator,
    build_autonomous_prompt,
    build_tool_prompt,
    truncate_tool_output,
)
from open_harness.llm.router import ModelRouter
from open_harness.memory.project_memory import (
    ProjectMemoryEngine,
    ProjectMemoryStore,
    build_memory_block,
)
from open_harness.memory.store import MemoryStore
from open_harness.planner import Plan, PlanCritic, PlanStep, Planner, StepResult
from open_harness.policy import PolicyEngine, load_policy
from open_harness.project import ProjectContext
from open_harness.tools.base import ToolRegistry, ToolResult
from open_harness.tools.rate_limiter import AgentRateLimiter

logger = logging.getLogger(__name__)

MAX_INTERACTIVE_STEPS = 15
MAX_GOAL_STEPS = 50  # Autonomous mode gets more room

# Tool names that correspond to external agents (used for rate-limit detection).
_EXTERNAL_AGENT_TOOLS = {"codex", "claude_code", "gemini_cli"}


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: str  # thinking, text, tool_call, tool_result, compensation, status, done, summary
    data: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalTracker:
    """Collects statistics during a goal execution for the final summary."""

    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    compensations: list[str] = field(default_factory=list)
    rollbacks: int = 0
    snapshots: int = 0
    planned: bool = False
    plan_steps_total: int = 0
    plan_steps_ok: int = 0
    replans: int = 0
    direct_fallback: bool = False
    rate_limit_fallbacks: int = 0
    files_written: list[str] = field(default_factory=list)

    def on_event(self, event: AgentEvent):
        """Update tracker from an agent event."""
        if event.type == "tool_call":
            self.tool_calls += 1
            name = event.metadata.get("tool", "?")
            self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
            # Track file writes
            if name in ("write_file", "edit_file"):
                args = event.metadata.get("args", {})
                path = args.get("path", "") if isinstance(args, dict) else ""
                if path:
                    self.files_written.append(path)
        elif event.type == "tool_result":
            if event.metadata.get("success"):
                self.tool_successes += 1
            else:
                self.tool_failures += 1
        elif event.type == "compensation":
            self.compensations.append(event.data)
            if "rolling back" in event.data.lower() or "rollback" in event.data.lower():
                self.rollbacks += 1
            if "direct mode" in event.data.lower():
                self.direct_fallback = True
            if "rate-limited" in event.data.lower() or "retrying with" in event.data.lower():
                self.rate_limit_fallbacks += 1
        elif event.type == "status":
            if event.data.startswith("Snapshot:"):
                self.snapshots += 1
            elif event.data.startswith("Replanned:"):
                self.replans += 1

    def build_summary(self) -> str:
        """Build a human-readable summary of the goal execution."""
        lines: list[str] = []

        # Execution mode
        if self.planned:
            lines.append(
                f"Plan: {self.plan_steps_ok}/{self.plan_steps_total} steps succeeded")
            if self.replans:
                lines.append(f"Replans: {self.replans}")
            if self.direct_fallback:
                lines.append("Fell back to direct mode")
        else:
            lines.append("Mode: direct (no plan)")

        # Tools
        lines.append(
            f"Tool calls: {self.tool_calls} "
            f"(OK: {self.tool_successes}, FAIL: {self.tool_failures})")
        if self.tool_counts:
            top = sorted(self.tool_counts.items(), key=lambda x: -x[1])[:5]
            breakdown = ", ".join(f"{n}: {c}" for n, c in top)
            lines.append(f"  {breakdown}")

        # Files
        unique_files = list(dict.fromkeys(self.files_written))
        if unique_files:
            lines.append(f"Files modified: {len(unique_files)}")
            for f in unique_files[:10]:
                lines.append(f"  {f}")
            if len(unique_files) > 10:
                lines.append(f"  ... and {len(unique_files) - 10} more")

        # Snapshots / Rollbacks
        if self.snapshots or self.rollbacks:
            parts = []
            if self.snapshots:
                parts.append(f"snapshots: {self.snapshots}")
            if self.rollbacks:
                parts.append(f"rollbacks: {self.rollbacks}")
            lines.append(f"Checkpoints: {', '.join(parts)}")

        # Rate limit fallbacks
        if self.rate_limit_fallbacks:
            lines.append(f"Rate-limit fallbacks: {self.rate_limit_fallbacks}")

        # Compensations (trial & error)
        non_rollback = [c for c in self.compensations
                        if "rolling back" not in c.lower() and "rollback" not in c.lower()]
        if non_rollback:
            lines.append(f"Compensations: {len(non_rollback)}")
            for c in non_rollback[:5]:
                lines.append(f"  ~ {c}")

        return "\n".join(lines)


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
        self.project_memory_store = ProjectMemoryStore(config.memory.db_path)
        self.project_memory = ProjectMemoryEngine(
            self.project_memory_store, str(self.project.root))
        # Rate-limit fallback for external agents
        available_ext = [
            t.name for t in self.tools.list_tools()
            if t.name in _EXTERNAL_AGENT_TOOLS
        ]
        self.rate_limiter = AgentRateLimiter(available_agents=available_ext)
        # Session-level checkpoint for interactive mode git protection
        self._session_checkpoint = CheckpointEngine(
            self.project.root,
            has_git=self.project.info.get("has_git", False),
        )
        self._session_checkpoint_started = False
        self._interactive_prompt: str | None = None
        self._autonomous_prompt: str | None = None
        self._interaction_count: int = 0

    @property
    def interactive_prompt(self) -> str:
        if self._interactive_prompt is None:
            tool_names = [t.name for t in self.tools.list_tools()]
            self._interactive_prompt = build_tool_prompt(
                self.tools.get_prompt_description(),
                self.config.compensation.thinking_mode,
                available_tools=tool_names,
                agent_configs=self.config.external_agents,
            )
        return self._interactive_prompt

    @property
    def autonomous_prompt(self) -> str:
        if self._autonomous_prompt is None:
            memory_block = build_memory_block(
                self.project_memory_store, str(self.project.root))
            project_ctx = self.project.to_prompt()
            if memory_block:
                project_ctx += f"\n\n{memory_block}"
            tool_names = [t.name for t in self.tools.list_tools()]
            self._autonomous_prompt = build_autonomous_prompt(
                self.tools.get_prompt_description(),
                project_ctx,
                self.config.compensation.thinking_mode,
                available_tools=tool_names,
                agent_configs=self.config.external_agents,
            )
        return self._autonomous_prompt

    def invalidate_prompts(self):
        """Call after tools or project context change."""
        self._interactive_prompt = None
        self._autonomous_prompt = None

    def close(self):
        """Finish the session checkpoint (merge changes back)."""
        if self._session_checkpoint_started:
            self._session_checkpoint.finish(keep_changes=True)
            self._session_checkpoint_started = False

    # ------------------------------------------------------------------
    # Interactive mode (existing behavior)
    # ------------------------------------------------------------------

    def run_stream(self, user_message: str) -> Generator[AgentEvent, None, None]:
        """Interactive single-turn with streaming."""
        self.compensator.reset()
        self.policy.begin_goal()  # reset budgets per turn
        self.memory.add_turn("user", user_message)
        # Periodic memory prune (every 20 interactions)
        self._interaction_count += 1
        if self._interaction_count % 20 == 0:
            self.project_memory.on_session_end()

        # Lazy-start session checkpoint for git protection
        if not self._session_checkpoint_started:
            status = self._session_checkpoint.begin()
            self._session_checkpoint_started = True
            logger.info("Session checkpoint: %s", status)

        messages = [
            {"role": "system", "content": self.interactive_prompt},
            *self.memory.get_messages(),
        ]
        yield from self._agent_loop(
            messages, MAX_INTERACTIVE_STEPS,
            checkpoint=self._session_checkpoint,
        )

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

        Emits a ``summary`` event at the end with trial-and-error statistics.
        """
        self.compensator.reset()
        self.policy.begin_goal()

        tracker = GoalTracker()

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
                tracker.planned = True
                tracker.plan_steps_total = len(plan.steps)
                yield AgentEvent("status", plan.summary())
                yield from self._tracked_goal(
                    self._run_planned_goal(goal, plan, ckpt), tracker)
            else:
                # Fallback: direct execution without planning
                reason = err.reason if err else "critic rejected"
                comp_event = AgentEvent("compensation",
                    f"Planning failed ({reason}); switching to direct mode")
                tracker.on_event(comp_event)
                yield comp_event
                yield from self._tracked_goal(
                    self._run_direct_goal(goal, ckpt), tracker)

        finally:
            yield AgentEvent("status", f"Budget: {self.policy.budget.summary()}")
            finish_status = ckpt.finish(keep_changes=True)
            if finish_status:
                yield AgentEvent("status", f"Finish: {finish_status}")

            # Emit goal summary
            summary_text = tracker.build_summary()
            yield AgentEvent("summary", summary_text, {
                "tool_calls": tracker.tool_calls,
                "tool_failures": tracker.tool_failures,
                "rollbacks": tracker.rollbacks,
                "compensations": len(tracker.compensations),
                "files_modified": len(set(tracker.files_written)),
            })

            # Flush learned memories and invalidate cached prompt
            self.project_memory.on_session_end()
            self._autonomous_prompt = None

    def _tracked_goal(
        self,
        events: Generator[AgentEvent, None, None],
        tracker: GoalTracker,
    ) -> Generator[AgentEvent, None, None]:
        """Wrap a goal generator to feed every event into the tracker."""
        for event in events:
            tracker.on_event(event)
            # Track planned step results from done events
            if event.type == "done" and event.metadata.get("planned"):
                tracker.plan_steps_ok = event.metadata.get("steps", 0)
            yield event

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
        self,
        goal: str,
        plan: Plan,
        ckpt: CheckpointEngine,
        completed: list[PlanStep] | None = None,
        _replan_depth: int = 0,
    ) -> Generator[AgentEvent, None, None]:
        """Execute a planned goal step by step with checkpoints."""
        if completed is None:
            completed = []
        # Track the snapshot taken after the last successful step
        last_good_snapshot = ckpt.snapshots[-1] if ckpt.snapshots else None

        for i, step in enumerate(plan.steps):
            yield AgentEvent("status",
                f"Step {i+1}/{len(plan.steps)}: {step.title}")

            # Use step's max_agent_steps if set, otherwise divide budget
            max_steps = step.max_agent_steps or (
                MAX_GOAL_STEPS // max(len(plan.steps), 1))

            # Execute this step
            step_result = yield from self._execute_plan_step(
                goal, step, completed, ckpt, max_steps,
            )

            if step_result.success:
                completed.append(step)
                # Snapshot after each successful step
                snap = ckpt.snapshot(f"step {i+1} complete: {step.title}")
                if snap:
                    yield AgentEvent("status", f"Snapshot: {snap.commit_hash}")
                    last_good_snapshot = snap
            else:
                # Step failed — try replanning (capped at 1 retry)
                yield AgentEvent("compensation",
                    f"Step '{step.title}' failed: {step_result.summary}")

                # Rollback to last SUCCESSFUL step snapshot (not latest)
                if last_good_snapshot:
                    rb = ckpt.rollback(last_good_snapshot)
                    yield AgentEvent("status", f"Rollback: {rb}")

                if _replan_depth >= 1:
                    # Already replanned once — fall back to direct mode
                    yield AgentEvent("compensation",
                        "Max replan attempts reached; switching to direct mode")
                    yield from self._fallback_to_direct(
                        goal, completed, step, step_result, ckpt)
                    return

                new_plan, err = self.planner.replan_remaining(
                    goal, completed, step, step_result.summary,
                )

                if new_plan and not self.plan_critic.validate(new_plan):
                    yield AgentEvent("status",
                        f"Replanned: {new_plan.summary()}")
                    yield from self._run_planned_goal(
                        goal, new_plan, ckpt,
                        completed=list(completed),  # copy to prevent corruption on failure
                        _replan_depth=_replan_depth + 1,
                    )
                    return
                else:
                    reason = err.reason if err else "critic rejected replan"
                    yield AgentEvent("compensation",
                        f"Replan failed ({reason}); switching to direct mode")
                    yield from self._fallback_to_direct(
                        goal, completed, step, step_result, ckpt)
                    return

        # All steps completed — emit a final summary done event
        summary = (
            f"Plan completed: {len(completed)}/{len(plan.steps)} steps succeeded.\n"
            + "\n".join(f"  - {s.title}" for s in completed)
        )
        yield AgentEvent("done", summary, {"steps": len(completed), "planned": True})

    def _fallback_to_direct(
        self,
        goal: str,
        completed: list[PlanStep],
        failed_step: PlanStep,
        step_result: StepResult,
        ckpt: CheckpointEngine,
    ) -> Generator[AgentEvent, None, None]:
        """Fall back to direct execution after plan/replan failure."""
        completed_text = "\n".join(f"  - {s.title}" for s in completed)
        remaining_context = (
            f"GOAL: {goal}\n\n"
            f"Completed steps so far:\n{completed_text}\n\n"
            f"Failed step: {failed_step.title} ({step_result.summary})\n"
            f"Continue working to achieve the goal."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.autonomous_prompt},
            {"role": "user", "content": remaining_context},
        ]
        yield from self._agent_loop(
            messages, MAX_GOAL_STEPS, checkpoint=ckpt)

    def _execute_plan_step(
        self,
        goal: str,
        step: PlanStep,
        completed: list[PlanStep],
        ckpt: CheckpointEngine,
        max_steps: int,
    ) -> Generator[AgentEvent, None, StepResult]:
        """Execute a single plan step using the agent loop.

        Filters internal 'done' events (emits them as 'status' instead)
        so only the top-level run_planned_goal emits the final 'done'.
        """
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
        last_tool_ok = True  # tracks most recent tool result
        for event in self._agent_loop(messages, max_steps, checkpoint=ckpt):
            if event.type == "done":
                # Don't forward step-level done — it would confuse consumers
                final_text = event.data
                yield AgentEvent("status", f"Step '{step.title}' finished")
            else:
                yield event
                if event.type == "tool_result":
                    last_tool_ok = event.metadata.get("success", True)

        # Step succeeds if it completed without step limit AND last tool
        # didn't fail (agent may have recovered from earlier errors)
        hit_limit = "[Reached" in final_text
        success = bool(final_text) and not hit_limit and last_tool_ok
        return StepResult(
            step_id=step.step_id,
            success=success,
            summary=final_text[:200] if final_text else "No output",
            attempts=1,
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

                # --- Rate-limit pre-check: swap to fallback before execution ---
                actual_tool = tc.name
                fallback_reason: str | None = None
                if tc.name in _EXTERNAL_AGENT_TOOLS:
                    actual_tool, fallback_reason = self.rate_limiter.get_best_agent(tc.name)
                    if fallback_reason:
                        yield AgentEvent("compensation", fallback_reason)

                yield AgentEvent("tool_call", actual_tool, {"tool": actual_tool, "args": tc.arguments})

                # Policy check before execution
                violation = self.policy.check(actual_tool, tc.arguments)
                if violation:
                    result = ToolResult(
                        success=False, output="",
                        error=f"[Policy: {violation.rule}] {violation.message}",
                    )
                else:
                    result = self.tools.execute(actual_tool, tc.arguments)
                    self.policy.record(actual_tool)

                # --- Rate-limit post-check: detect limit hit in output ---
                if (actual_tool in _EXTERNAL_AGENT_TOOLS
                        and not result.success
                        and AgentRateLimiter.is_rate_limit_error(result.to_message())):
                    entry = self.rate_limiter.record_rate_limit(
                        actual_tool, result.to_message())
                    yield AgentEvent("compensation",
                        f"{actual_tool} rate-limited (cooldown {entry.human_remaining()})")

                    # Try another fallback
                    retry_agent = self.rate_limiter.get_fallback(actual_tool)
                    if retry_agent:
                        yield AgentEvent("compensation",
                            f"Retrying with {retry_agent}")
                        yield AgentEvent("tool_call", retry_agent,
                            {"tool": retry_agent, "args": tc.arguments})
                        result = self.tools.execute(retry_agent, tc.arguments)
                        self.policy.record(retry_agent)
                        actual_tool = retry_agent

                        # Check if retry also hit rate limit
                        if (not result.success
                                and AgentRateLimiter.is_rate_limit_error(result.to_message())):
                            self.rate_limiter.record_rate_limit(
                                retry_agent, result.to_message())
                            yield AgentEvent("compensation",
                                f"{retry_agent} also rate-limited")
                    else:
                        # All external agents are rate-limited — tell the LLM
                        # so it can try a different approach instead of looping
                        cooldown_info = self.rate_limiter.status_summary()
                        result = ToolResult(
                            success=False, output="",
                            error=(
                                f"All external agents are rate-limited. "
                                f"{cooldown_info} "
                                f"Use local tools (shell, read_file, write_file, "
                                f"edit_file) to proceed without external agents."
                            ),
                        )
                        yield AgentEvent("compensation",
                            "All external agents rate-limited — falling back to local tools")

                output = truncate_tool_output(result.to_message(), 8000)
                yield AgentEvent("tool_result", output,
                                 {"success": result.success, "tool": actual_tool})

                # Auto-learn from tool usage
                self.project_memory.on_tool_result(
                    actual_tool, tc.arguments, result.success, output)

                # Auto-snapshot after file writes (every 5 writes)
                if checkpoint and actual_tool in ("write_file", "edit_file") and result.success:
                    _writes_since_snapshot += 1
                    if _writes_since_snapshot >= 5:
                        snap = checkpoint.snapshot(f"after {_writes_since_snapshot} writes (step {step})")
                        if snap:
                            yield AgentEvent("status", f"Snapshot: {snap.commit_hash}")
                        _writes_since_snapshot = 0

                # Auto-rollback on test failure (if checkpoint active)
                if (checkpoint and actual_tool == "run_tests"
                        and not result.success and checkpoint.snapshots):
                    yield AgentEvent("compensation",
                        "Tests failed — rolling back to last snapshot")
                    rb = checkpoint.rollback(checkpoint.snapshots[-1])
                    yield AgentEvent("status", f"Rollback: {rb}")
                    # Tell the LLM about the rollback
                    messages.append({"role": "assistant",
                        "content": f'{{"tool": "{actual_tool}", "args": {_safe_json(tc.arguments)}}}'})
                    messages.append({"role": "user",
                        "content": f"[Tool Result for {actual_tool}]\n{output}\n\n"
                                   f"[ROLLBACK] Changes have been rolled back to the last "
                                   f"snapshot. Try a different approach."})
                    continue

                messages.append({"role": "assistant",
                    "content": f'{{"tool": "{actual_tool}", "args": {_safe_json(tc.arguments)}}}'})
                messages.append({"role": "user",
                    "content": f"[Tool Result for {actual_tool}]\n{output}"})
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
        except KeyboardInterrupt:
            # Gracefully close the generator so the httpx stream is cleaned up
            gen.close()
            response = LLMResponse(content="[Interrupted]", finish_reason="interrupted")
            yield AgentEvent("done", "[Interrupted by user]")
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
        # Allow up to 2000 chars — weak LLMs often wrap tool calls in prose
        return any(indicators) and len(content) < 2000



def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)
