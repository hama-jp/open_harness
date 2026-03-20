"""Orchestrator — autonomous agent main loop with Plan→Execute→Reflect→Verify.

    Plan → Execute → Reflect → (Re-plan if stuck) → Verify → Done

The Orchestrator owns no business logic itself; it wires the Planner,
Reasoner, Executor, Reflector, StuckDetector, Verifier, and
StrategyManager together, driving the agent toward goal completion.

Key autonomous behaviors:
1. **Goal decomposition**: Breaks goals into executable plans
2. **Progress tracking**: Monitors advancement toward the goal
3. **Stuck detection**: Identifies loops, stalls, and error spirals
4. **Self-recovery**: Switches strategies, re-plans, escalates models
5. **Verification**: Validates work before declaring completion
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from open_harness_v2.core.context import AgentContext
from open_harness_v2.core.executor import Executor
from open_harness_v2.core.planner import ExecutionPlan, Planner
from open_harness_v2.core.reasoner import ActionType, QualitySignal, Reasoner
from open_harness_v2.core.reflection import (
    ActionOutcome,
    ProgressSignal,
    Reflector,
)
from open_harness_v2.core.strategy import StrategyManager
from open_harness_v2.core.stuck_detector import RecoveryAction, StuckDetector
from open_harness_v2.core.verifier import VerificationStatus, Verifier
from open_harness_v2.events.bus import EventBus
from open_harness_v2.llm.middleware import LLMRequest, MiddlewarePipeline
from open_harness_v2.llm.router import ModelRouter
from open_harness_v2.policy.engine import PolicyEngine
from open_harness_v2.tools.registry import ToolRegistry
from open_harness_v2.types import AgentEvent, EventType

_logger = logging.getLogger(__name__)

# External agent tool names — when available, non-trivial tasks MUST be
# delegated to at least one of these before the agent is allowed to respond.
_EXTERNAL_AGENTS = frozenset({"claude_code", "codex", "gemini_cli"})

_DELEGATION_REDIRECT = (
    "STOP. You MUST NOT answer this yourself. "
    "You have external agent tools available (claude_code, codex, gemini_cli). "
    "Use one of them NOW to handle this task. "
    "Give the agent the FILE PATHS and a clear task description. "
    "The external agent can read files itself — do NOT read the files yourself."
)

# Reflection interval: run reflection every N tool executions
_REFLECT_INTERVAL = 3

# Maximum recovery attempts before giving up
_MAX_RECOVERIES = 5

# ---------------------------------------------------------------------------
# Goal complexity classification
# ---------------------------------------------------------------------------

class GoalComplexity:
    """Classifies goal text into complexity tiers for delegation decisions."""

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"

    # Patterns that indicate a lightweight / quick task
    _LIGHT_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\b(what|who|where|when|why|how)\b.*\?$", re.I),
        re.compile(r"\b(explain|describe|list|show|tell me|summarize)\b", re.I),
        re.compile(r"\b(status|version|help|info)\b", re.I),
    ]

    # Patterns that indicate a heavyweight / complex task
    _HEAVY_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\b(refactor|rewrite|migrate|redesign|implement|build)\b", re.I),
        re.compile(r"\b(fix.+bug|debug|investigate.+issue)\b", re.I),
        re.compile(r"\bmulti.?file\b", re.I),
        re.compile(r"\b(across|all files|entire|whole|codebase)\b", re.I),
        re.compile(r"\b(add feature|new feature|create.+module)\b", re.I),
        re.compile(r"\b(test suite|write tests|add tests)\b", re.I),
    ]

    @classmethod
    def classify(cls, goal: str) -> str:
        """Classify a goal string into LIGHT, MEDIUM, or HEAVY."""
        text = goal.strip()

        # Short goals are usually lightweight
        if len(text) < 30:
            for pat in cls._LIGHT_PATTERNS:
                if pat.search(text):
                    return cls.LIGHT

        # Check for heavy indicators
        heavy_score = sum(1 for pat in cls._HEAVY_PATTERNS if pat.search(text))
        if heavy_score >= 2:
            return cls.HEAVY

        # Check for light indicators
        light_score = sum(1 for pat in cls._LIGHT_PATTERNS if pat.search(text))
        if light_score > 0 and heavy_score == 0:
            return cls.LIGHT

        # Single heavy indicator → medium
        if heavy_score == 1:
            return cls.MEDIUM

        return cls.MEDIUM


# ---------------------------------------------------------------------------
# Enhanced autonomous system prompt
# ---------------------------------------------------------------------------

_AUTONOMOUS_SYSTEM_ROLE = """\
You are an autonomous AI agent with access to tools. You work independently \
to achieve goals without human intervention.

## Core Principles
1. **Act decisively**: Don't ask for permission — use your tools to solve problems.
2. **Verify your work**: After making changes, confirm they work correctly.
3. **Adapt when stuck**: If an approach isn't working, try something different.
4. **Be thorough**: Read error messages carefully. Understand before you act.
5. **Track progress**: Complete tasks systematically, one step at a time.

## Problem-Solving Protocol
1. UNDERSTAND: Read relevant files and understand the current state.
2. PLAN: Identify what needs to change and in what order.
3. EXECUTE: Make changes using the appropriate tools.
4. VERIFY: Check that changes work (run tests, read output).
5. ITERATE: If something failed, analyze why and try a different approach.

## Anti-Patterns to Avoid
- Do NOT repeat the same failing action — try something different.
- Do NOT read the same file multiple times without making changes between reads.
- Do NOT give up after a single failure — analyze the error and adapt.
- Do NOT provide a final answer without having actually completed the work.
- Do NOT skip verification — always confirm your changes work.\
"""


class Orchestrator:
    """Async main loop for autonomous agent execution.

    Implements the Plan→Execute→Reflect→Verify cycle:

    1. **Plan**: Decompose the goal into steps (via Planner)
    2. **Execute**: Run tools via the LLM→Reasoner→Executor chain
    3. **Reflect**: Assess progress after each action (via Reflector)
    4. **Recover**: If stuck, switch strategies or re-plan (via StuckDetector)
    5. **Verify**: Validate goal completion before responding (via Verifier)

    Parameters
    ----------
    router:
        Model router for tier-based model selection.
    registry:
        Tool registry with all available tools.
    policy:
        Policy engine for guardrails (optional).
    event_bus:
        Event bus for UI decoupling (optional).
    pipeline:
        Middleware pipeline for LLM calls (optional).
    max_steps:
        Maximum steps before stopping.
    context_budget:
        Token budget for context assembly (0 = unlimited).
    enable_planning:
        Enable LLM-driven goal decomposition.
    enable_reflection:
        Enable self-reflection after tool executions.
    enable_verification:
        Enable goal-completion verification.
    """

    def __init__(
        self,
        router: ModelRouter,
        registry: ToolRegistry,
        policy: PolicyEngine | None = None,
        event_bus: EventBus | None = None,
        pipeline: MiddlewarePipeline | None = None,
        max_steps: int = 50,
        context_budget: int = 0,
        approval_engine: Any = None,
        hook_engine: Any = None,
        sandbox_engine: Any = None,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        enable_verification: bool = True,
    ) -> None:
        self._router = router
        self._registry = registry
        self._policy = policy
        self._event_bus = event_bus or EventBus()
        self._pipeline = pipeline or MiddlewarePipeline(router.get_client())
        self._reasoner = Reasoner(max_steps=max_steps)
        self._executor = Executor(
            registry, policy, self._event_bus,
            approval_engine=approval_engine,
            hook_engine=hook_engine,
            sandbox_engine=sandbox_engine,
        )
        self._context_budget = context_budget
        self._cancelled = False
        self.system_extra: str = ""  # injected by CLI (e.g. project memory)

        # Autonomous subsystems
        self._enable_planning = enable_planning
        self._enable_reflection = enable_reflection
        self._enable_verification = enable_verification

        self._reflector = Reflector()
        self._stuck_detector = StuckDetector()
        self._strategy_mgr = StrategyManager()
        self._verifier = Verifier()
        self._planner = Planner(
            tool_names=[t.name for t in registry.list_tools()],
        )

        # Tracking state
        self._actions_taken: list[dict[str, Any]] = []
        self._files_modified: set[str] = set()
        self._recovery_count = 0

    async def run(self, goal: str, context: AgentContext | None = None) -> str:
        """Run the agent loop until completion or cancellation.

        Parameters
        ----------
        goal:
            The user's goal / prompt.
        context:
            Optional pre-built context. If None, a fresh context is created.

        Returns
        -------
        str
            The final text response from the agent.
        """
        self._cancelled = False
        self._reasoner.reset()
        self._reflector.set_goal(goal)
        self._stuck_detector.reset()
        self._actions_taken.clear()
        self._files_modified.clear()
        self._recovery_count = 0

        if self._policy:
            self._policy.begin_goal()

        # Track external agent usage for delegation enforcement
        used_tools: set[str] = set()
        available_ext = _EXTERNAL_AGENTS & {
            t.get("function", t).get("name", "")
            for t in (self._registry.get_openai_schemas() or [])
        }
        delegation_retries = 0

        # Classify goal complexity
        goal_complexity = GoalComplexity.classify(goal)
        _logger.info("Goal complexity: %s", goal_complexity)

        # Initialize strategy
        strategy = self._strategy_mgr.initialize(
            goal,
            has_external_agents=bool(available_ext),
            goal_complexity=goal_complexity,
        )
        await self._emit(EventType.STRATEGY_INITIALIZED, {
            "approach": strategy.approach.value,
            "description": strategy.description,
        })

        # Create execution plan
        plan: ExecutionPlan | None = None
        if self._enable_planning and goal_complexity != GoalComplexity.LIGHT:
            plan = await self._planner.create_plan(goal)
            if plan and plan.steps:
                await self._emit(EventType.PLAN_CREATED, plan.to_dict())

        # Set up context
        ctx = context or AgentContext()
        ctx.system.role = _AUTONOMOUS_SYSTEM_ROLE
        if self.system_extra:
            ctx.system.extra = self.system_extra
        if not ctx.system.tools_description:
            ctx.system.tools_description = self._registry.get_compact_prompt_description()

        # Inject strategy and plan into goal state
        ctx.goal_state.goal = goal
        ctx.goal_state.strategy_hint = strategy.system_hint
        ctx.goal_state.max_steps = self._reasoner._max_steps
        if plan:
            ctx.goal_state.plan_block = plan.to_context_block()

        ctx.add_user_message(goal)

        await self._emit(EventType.AGENT_STARTED, {"goal": goal})

        final_response = ""
        consecutive_empty = 0
        tool_exec_since_reflect = 0

        try:
            while not self._cancelled:
                await asyncio.sleep(0)

                # Update step counter in goal state
                ctx.goal_state.step_number = self._reasoner.step_count + 1

                # Warn when approaching step limit
                steps_used = self._reasoner.step_count
                max_steps = self._reasoner._max_steps
                if steps_used > 0 and steps_used == max_steps - 5:
                    _logger.warning(
                        "Approaching step limit: %d/%d steps used",
                        steps_used, max_steps,
                    )
                    ctx.add_user_message(
                        f"[System] You are approaching the step limit "
                        f"({steps_used}/{max_steps}). Wrap up your work soon. "
                        f"Summarize what you've accomplished and what remains."
                    )

                # 1. Build messages from context
                messages = ctx.to_messages(budget=self._context_budget)

                # 2. Call LLM via pipeline
                tools = self._registry.get_openai_schemas()
                request = LLMRequest(
                    messages=messages,
                    model=self._router.current_model,
                    tools=tools or None,
                    tool_choice="auto" if tools else None,
                )
                response = await self._pipeline.execute(request)

                await self._emit(EventType.LLM_RESPONSE, {
                    "model": response.model,
                    "has_tool_calls": response.has_tool_calls,
                    "content_length": len(response.content),
                    "latency_ms": response.latency_ms,
                    "usage": response.usage,
                })

                # Track token usage
                if self._policy and response.usage:
                    self._policy.record_usage(response.usage)
                    budget_msg = self._policy.check_token_budget()
                    if budget_msg:
                        _logger.warning(budget_msg)
                        final_response = budget_msg
                        break

                # 3. Reasoner decides next action
                decision = self._reasoner.decide(response)

                await self._emit(EventType.REASONER_DECISION, {
                    "action": decision.action.value,
                    "step": self._reasoner.step_count,
                    "quality": decision.quality.value,
                })

                if decision.thinking:
                    await self._emit(EventType.LLM_THINKING, {
                        "thinking": decision.thinking,
                    })

                # 4. Act on decision
                if decision.action == ActionType.RESPOND:
                    # === Delegation enforcement ===
                    should_delegate = (
                        available_ext
                        and not (used_tools & _EXTERNAL_AGENTS)
                        and delegation_retries < 2
                        and (
                            goal_complexity == GoalComplexity.HEAVY
                            or (
                                goal_complexity == GoalComplexity.MEDIUM
                                and delegation_retries >= 1
                            )
                        )
                    )
                    if should_delegate:
                        delegation_retries += 1
                        _logger.info(
                            "Delegation redirect #%d (complexity=%s)",
                            delegation_retries, goal_complexity,
                        )
                        ctx.add_assistant_message(decision.response_text)
                        ctx.cycle_working()
                        ctx.add_user_message(_DELEGATION_REDIRECT)
                        continue

                    # === Premature completion check ===
                    if (
                        decision.quality == QualitySignal.WEAK
                        and self._reasoner.step_count <= 2
                        and goal_complexity != GoalComplexity.LIGHT
                    ):
                        _logger.info(
                            "Weak response at step %d — pushing agent to do more",
                            self._reasoner.step_count,
                        )
                        ctx.add_assistant_message(decision.response_text)
                        ctx.cycle_working()
                        ctx.add_user_message(
                            "[System] Your response seems incomplete. "
                            "Use your tools to actually accomplish the goal, "
                            "don't just describe what you would do."
                        )
                        continue

                    # === Verification before final response ===
                    if (
                        self._enable_verification
                        and goal_complexity != GoalComplexity.LIGHT
                        and self._actions_taken
                    ):
                        verification = await self._verifier.quick_verify(
                            goal,
                            decision.response_text,
                            self._reflector.success_rate,
                            len(self._files_modified),
                        )
                        await self._emit(EventType.VERIFICATION_RESULT, {
                            "status": verification.status.value,
                            "summary": verification.summary,
                        })

                        if (
                            verification.status == VerificationStatus.FAILED
                            and self._recovery_count < _MAX_RECOVERIES
                        ):
                            self._recovery_count += 1
                            ctx.add_assistant_message(decision.response_text)
                            ctx.cycle_working()
                            feedback = (
                                "[System] Verification failed: "
                                + "; ".join(verification.checks_failed)
                            )
                            if verification.suggestions:
                                feedback += "\nSuggestions: " + "; ".join(
                                    verification.suggestions
                                )
                            ctx.add_user_message(feedback)
                            continue

                    final_response = decision.response_text
                    ctx.add_assistant_message(decision.response_text)
                    self._strategy_mgr.mark_succeeded()
                    break

                elif decision.action == ActionType.EXECUTE_TOOLS:
                    ctx.add_assistant_message(response.content)
                    ctx.cycle_working()

                    # Execute tools
                    exec_result = await self._executor.execute(decision.tool_calls)

                    # Track usage and results
                    for tc, result in exec_result.results:
                        used_tools.add(tc.name)

                        # Record for reflection
                        outcome = ActionOutcome(
                            step_number=self._reasoner.step_count,
                            tool_name=tc.name,
                            tool_args=tc.arguments,
                            success=result.success,
                            output_snippet=result.output[:300] if result.output else "",
                            error=result.error,
                        )
                        self._reflector.record(outcome)

                        # Record for stuck detection
                        files_changed = None
                        if tc.name in ("write_file", "edit_file"):
                            path = tc.arguments.get("path", "")
                            if path and result.success:
                                self._files_modified.add(path)
                                files_changed = {path}
                        self._stuck_detector.record(
                            tc.name, tc.arguments, result.success,
                            result.output[:500] if result.output else result.error,
                            files_changed,
                        )

                        # Record for verification
                        self._actions_taken.append({
                            "tool_name": tc.name,
                            "args": tc.arguments,
                            "success": result.success,
                            "output": result.output[:200] if result.output else "",
                            "error": result.error,
                        })

                    # Add results to working layer
                    for tc, result in exec_result.results:
                        ctx.add_tool_result(tc.name, result.to_message())

                    tool_exec_since_reflect += 1

                    # === Periodic reflection ===
                    if (
                        self._enable_reflection
                        and tool_exec_since_reflect >= _REFLECT_INTERVAL
                    ):
                        tool_exec_since_reflect = 0
                        reflection = self._reflector.reflect()

                        await self._emit(EventType.REFLECTION_RESULT, {
                            "signal": reflection.signal.value,
                            "confidence": reflection.confidence,
                            "assessment": reflection.assessment,
                        })

                        # Update context with progress info
                        ctx.goal_state.progress_block = (
                            self._reflector.get_context_injection()
                        )
                        if plan:
                            ctx.goal_state.plan_block = plan.to_context_block()

                        # === Stuck detection ===
                        if reflection.signal in (
                            ProgressSignal.BLOCKED,
                            ProgressSignal.REGRESSING,
                            ProgressSignal.STALLED,
                        ):
                            diagnosis = self._stuck_detector.diagnose()

                            if diagnosis.needs_intervention:
                                await self._emit(EventType.STUCK_DETECTED, {
                                    "pattern": diagnosis.pattern.value,
                                    "severity": diagnosis.severity,
                                    "description": diagnosis.description,
                                    "recovery": diagnosis.recovery.value,
                                })

                                recovery_applied = await self._apply_recovery(
                                    ctx, plan, diagnosis.recovery,
                                    diagnosis.recovery_hint,
                                    goal, goal_complexity,
                                )
                                if recovery_applied:
                                    self._stuck_detector.record_intervention()
                                    continue

                    # Update plan progress if applicable
                    if plan and plan.current_step and exec_result.all_succeeded:
                        # Heuristic: if all tools succeeded, advance the plan
                        # The LLM will naturally work through plan steps
                        pass  # Plan advancement is driven by reflection

                elif decision.action == ActionType.ERROR:
                    error_msg = decision.error or "Agent encountered an error"

                    # Guard: empty response recovery
                    if "Empty response" in error_msg:
                        consecutive_empty += 1
                        if consecutive_empty < 3:
                            _logger.warning(
                                "Empty response #%d — escalating model",
                                consecutive_empty,
                            )
                            self._router.escalate()
                            continue
                    else:
                        consecutive_empty = 0

                    # Try recovery before giving up
                    if self._recovery_count < _MAX_RECOVERIES:
                        diagnosis = self._stuck_detector.diagnose()
                        if diagnosis.recovery != RecoveryAction.ASK_USER:
                            recovery_applied = await self._apply_recovery(
                                ctx, plan, RecoveryAction.ESCALATE_MODEL,
                                f"Error: {error_msg}",
                                goal, goal_complexity,
                            )
                            if recovery_applied:
                                continue

                    final_response = error_msg
                    await self._emit(EventType.AGENT_ERROR, {
                        "error": final_response,
                    })
                    break

                else:
                    # CONTINUE or unknown — keep going
                    consecutive_empty = 0
                    ctx.add_assistant_message(response.content)

        except asyncio.CancelledError:
            self._cancelled = True
        except Exception as e:
            _logger.exception("Orchestrator error")
            final_response = f"Agent error: {type(e).__name__}: {e}"
            await self._emit(EventType.AGENT_ERROR, {"error": final_response})

        # Handle cancellation
        if self._cancelled and not final_response:
            final_response = "Agent cancelled"

        done_type = (
            EventType.AGENT_CANCELLED if self._cancelled
            else EventType.AGENT_DONE
        )
        await self._emit(done_type, {
            "response": final_response[:500],
            "steps": self._reasoner.step_count,
            "files_modified": list(self._files_modified),
            "success_rate": self._reflector.success_rate,
            "recovery_count": self._recovery_count,
        })

        return final_response

    def cancel(self) -> None:
        """Request cancellation of the running loop."""
        self._cancelled = True

    # ------------------------------------------------------------------
    # Recovery mechanism
    # ------------------------------------------------------------------

    async def _apply_recovery(
        self,
        ctx: AgentContext,
        plan: ExecutionPlan | None,
        recovery_action: RecoveryAction,
        hint: str,
        goal: str,
        goal_complexity: str,
    ) -> bool:
        """Apply a recovery action. Returns True if recovery was applied."""
        if self._recovery_count >= _MAX_RECOVERIES:
            _logger.warning("Maximum recovery attempts reached (%d)", _MAX_RECOVERIES)
            return False

        self._recovery_count += 1

        _logger.info(
            "Applying recovery #%d: %s",
            self._recovery_count, recovery_action.value,
        )

        if recovery_action == RecoveryAction.ESCALATE_MODEL:
            escalated = self._router.escalate()
            if escalated:
                await self._emit(EventType.RECOVERY_APPLIED, {
                    "action": "escalate_model",
                    "new_model": self._router.current_model,
                })
                return True
            # If can't escalate, fall through to replan
            recovery_action = RecoveryAction.REPLAN

        if recovery_action == RecoveryAction.FORCE_DIFFERENT_TOOL:
            ctx.add_user_message(
                f"[System] RECOVERY: {hint}\n"
                f"You MUST use a DIFFERENT tool or different arguments. "
                f"Repeating the same action will not work."
            )
            ctx.goal_state.recovery_hint = hint
            await self._emit(EventType.RECOVERY_APPLIED, {
                "action": "force_different_tool",
                "hint": hint,
            })
            return True

        if recovery_action == RecoveryAction.SWITCH_STRATEGY:
            next_strategy = self._strategy_mgr.next_strategy()
            if next_strategy:
                ctx.goal_state.strategy_hint = next_strategy.system_hint
                ctx.add_user_message(
                    f"[System] STRATEGY CHANGE: Switching to "
                    f"'{next_strategy.approach.value}' approach.\n"
                    f"{next_strategy.system_hint}"
                )
                await self._emit(EventType.STRATEGY_SWITCHED, {
                    "approach": next_strategy.approach.value,
                    "description": next_strategy.description,
                })
                return True
            # All strategies exhausted
            await self._emit(EventType.STRATEGY_EXHAUSTED, {})

        if recovery_action == RecoveryAction.REPLAN:
            if plan and self._enable_planning:
                plan = await self._planner.replan(plan, hint)
                ctx.goal_state.plan_block = plan.to_context_block()
                ctx.add_user_message(
                    f"[System] REPLANNED: The plan has been revised.\n"
                    f"{plan.to_context_block()}\n"
                    f"Follow the updated plan."
                )
                await self._emit(EventType.PLAN_REVISED, plan.to_dict())
                return True
            # No plan to revise, try strategy switch
            next_strategy = self._strategy_mgr.next_strategy()
            if next_strategy:
                ctx.goal_state.strategy_hint = next_strategy.system_hint
                ctx.add_user_message(
                    f"[System] RECOVERY: {hint}\n"
                    f"New approach: {next_strategy.system_hint}"
                )
                return True

        if recovery_action == RecoveryAction.SKIP_STEP:
            if plan and plan.current_step:
                plan.skip_current(hint)
                next_step = plan.advance()
                if next_step:
                    ctx.goal_state.plan_block = plan.to_context_block()
                    ctx.add_user_message(
                        f"[System] Skipped step due to: {hint}\n"
                        f"Moving to: {next_step.description}"
                    )
                    return True

        if recovery_action == RecoveryAction.RESET_AND_RETRY:
            # Clear recovery hint and try fresh
            ctx.goal_state.recovery_hint = ""
            ctx.add_user_message(
                f"[System] RESET: Previous approach failed. "
                f"Start fresh. Re-read the goal and try a new approach.\n"
                f"Goal: {goal}"
            )
            return True

        # ASK_USER or unhandled — don't apply automatic recovery
        return False

    async def _emit(self, event_type: EventType, data: dict[str, Any]) -> None:
        await self._event_bus.emit(AgentEvent(type=event_type, data=data))
