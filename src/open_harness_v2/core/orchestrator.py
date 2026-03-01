"""Orchestrator — thin main loop that ties everything together.

    context → LLM → reasoner → executor → loop

The Orchestrator owns no business logic itself; it simply wires
the Reasoner, Executor, and Context together and emits events.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from open_harness_v2.core.context import AgentContext
from open_harness_v2.core.executor import Executor
from open_harness_v2.core.reasoner import ActionType, Reasoner
from open_harness_v2.events.bus import EventBus
from open_harness_v2.llm.client import AsyncLLMClient
from open_harness_v2.llm.middleware import LLMRequest, MiddlewarePipeline
from open_harness_v2.llm.router import ModelRouter
from open_harness_v2.policy.engine import PolicyEngine
from open_harness_v2.tools.registry import ToolRegistry
from open_harness_v2.types import AgentEvent, EventType, LLMResponse

_logger = logging.getLogger(__name__)


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
    ) -> None:
        self._router = router
        self._registry = registry
        self._policy = policy
        self._event_bus = event_bus or EventBus()
        self._pipeline = pipeline or MiddlewarePipeline(router.get_client())
        self._reasoner = Reasoner(max_steps=max_steps)
        self._executor = Executor(registry, policy, self._event_bus)
        self._context_budget = context_budget
        self._cancelled = False

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

        # Set up context
        ctx = context or AgentContext()
        if not ctx.system.tools_description:
            ctx.system.tools_description = self._registry.get_compact_prompt_description()
        ctx.add_user_message(goal)

        await self._emit(EventType.AGENT_STARTED, {"goal": goal})

        final_response = ""

        try:
            while not self._cancelled:
                # Yield to event loop so cancellation can be detected
                await asyncio.sleep(0)

                # 1. Build messages from context
                messages = ctx.to_messages(budget=self._context_budget)

                # 2. Call LLM via pipeline
                request = LLMRequest(
                    messages=messages,
                    model=self._router.current_model,
                )
                response = await self._pipeline.execute(request)

                await self._emit(EventType.LLM_RESPONSE, {
                    "model": response.model,
                    "has_tool_calls": response.has_tool_calls,
                    "content_length": len(response.content),
                    "latency_ms": response.latency_ms,
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

                    # Add results to working layer
                    for tc, result in exec_result.results:
                        ctx.add_tool_result(tc.name, result.to_message())

                    # Continue the loop for the next LLM call

                elif decision.action == ActionType.ERROR:
                    final_response = decision.error or "Agent encountered an error"
                    await self._emit(EventType.AGENT_ERROR, {
                        "error": final_response,
                    })
                    break

                else:
                    # CONTINUE or unknown — keep going
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
