"""Orchestrator — thin main loop that ties everything together.

    context → LLM → reasoner → executor → loop

The Orchestrator owns no business logic itself; it simply wires
the Reasoner, Executor, and Context together and emits events.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from open_harness_v2.core.context import AgentContext
from open_harness_v2.core.executor import Executor
from open_harness_v2.core.reasoner import ActionType, Reasoner
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
        """Classify a goal string into LIGHT, MEDIUM, or HEAVY.

        - LIGHT: Questions, explanations, status checks — no delegation needed.
        - MEDIUM: Moderate tasks — delegation optional, self-solve first.
        - HEAVY: Complex multi-step tasks — delegation recommended.
        """
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


class Orchestrator:
    """Async main loop for agent execution.

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
        Middleware pipeline for LLM calls (optional). If not provided,
        a basic pipeline is created from the router's client.
    max_steps:
        Maximum steps before stopping.
    context_budget:
        Token budget for context assembly (0 = unlimited).
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

        if self._policy:
            self._policy.begin_goal()

        # Track external agent usage for delegation enforcement
        used_tools: set[str] = set()
        available_ext = _EXTERNAL_AGENTS & {
            t.get("function", t).get("name", "")
            for t in (self._registry.get_openai_schemas() or [])
        }
        # How many times we've redirected the LLM to delegate
        delegation_retries = 0

        # Classify goal complexity for staged delegation
        goal_complexity = GoalComplexity.classify(goal)
        _logger.info("Goal complexity: %s", goal_complexity)

        # Set up context
        ctx = context or AgentContext()
        if self.system_extra:
            ctx.system.extra = self.system_extra
        if not ctx.system.tools_description:
            ctx.system.tools_description = self._registry.get_compact_prompt_description()
        ctx.add_user_message(goal)

        await self._emit(EventType.AGENT_STARTED, {"goal": goal})

        final_response = ""
        consecutive_empty = 0  # guard against empty response loops

        try:
            while not self._cancelled:
                # Yield to event loop so cancellation can be detected
                await asyncio.sleep(0)

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
                        f"({steps_used}/{max_steps}). Wrap up your work soon."
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
                })

                if decision.thinking:
                    await self._emit(EventType.LLM_THINKING, {
                        "thinking": decision.thinking,
                    })

                # 4. Act on decision
                if decision.action == ActionType.RESPOND:
                    # Staged delegation enforcement:
                    # - LIGHT goals: never force delegation (answer directly)
                    # - MEDIUM goals: allow self-solve, delegate only on retry
                    # - HEAVY goals: force delegation before answering
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
                            "Delegation redirect #%d (complexity=%s) — "
                            "LLM tried to respond without using an external agent",
                            delegation_retries,
                            goal_complexity,
                        )
                        ctx.add_assistant_message(decision.response_text)
                        ctx.cycle_working()
                        ctx.add_user_message(_DELEGATION_REDIRECT)
                        continue

                    final_response = decision.response_text
                    ctx.add_assistant_message(decision.response_text)
                    break

                elif decision.action == ActionType.EXECUTE_TOOLS:
                    # Record the assistant message (tool call)
                    ctx.add_assistant_message(response.content)

                    # Promote previous working messages to history
                    ctx.cycle_working()

                    # Execute tools
                    exec_result = await self._executor.execute(decision.tool_calls)

                    # Track which tools were used
                    for tc, _result in exec_result.results:
                        used_tools.add(tc.name)

                    # Add results to working layer
                    for tc, result in exec_result.results:
                        ctx.add_tool_result(tc.name, result.to_message())

                    # Continue the loop for the next LLM call

                elif decision.action == ActionType.ERROR:
                    error_msg = decision.error or "Agent encountered an error"

                    # Guard: treat consecutive empty responses as recoverable
                    # before giving up — try escalating the model first.
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

        # Handle cancellation (either from self.cancel() or CancelledError)
        if self._cancelled and not final_response:
            final_response = "Agent cancelled"

        done_type = (
            EventType.AGENT_CANCELLED if self._cancelled
            else EventType.AGENT_DONE
        )
        await self._emit(done_type, {
            "response": final_response[:500],
            "steps": self._reasoner.step_count,
        })

        return final_response

    def cancel(self) -> None:
        """Request cancellation of the running loop."""
        self._cancelled = True

    async def _emit(self, event_type: EventType, data: dict[str, Any]) -> None:
        await self._event_bus.emit(AgentEvent(type=event_type, data=data))
