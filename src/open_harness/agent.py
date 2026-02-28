"""Core agent loop - ReAct pattern with weak LLM compensation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from open_harness.config import HarnessConfig
from open_harness.llm.client import LLMResponse, ToolCall
from open_harness.llm.compensator import Compensator, build_tool_prompt
from open_harness.llm.router import ModelRouter
from open_harness.memory.store import MemoryStore
from open_harness.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 20  # Safety limit for agent loop iterations


@dataclass
class AgentStep:
    """Record of a single agent step."""
    step_type: str  # "llm_call", "tool_call", "tool_result", "compensation"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Agent:
    """ReAct agent with weak LLM compensation.

    Flow:
    1. User message → build context → LLM call
    2. LLM responds with text or tool call
    3. If tool call: execute → feed result back → go to 2
    4. If text: return to user
    5. If malformed: compensate (retry/refine/escalate) → go to 2
    """

    def __init__(
        self,
        config: HarnessConfig,
        tools: ToolRegistry,
        memory: MemoryStore,
        on_step: Callable[[AgentStep], None] | None = None,
    ):
        self.config = config
        self.tools = tools
        self.memory = memory
        self.router = ModelRouter(config)
        self.compensator = Compensator(config.compensation)
        self.on_step = on_step or (lambda s: None)
        self._system_prompt: str | None = None
        self._steps: list[AgentStep] = []

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_tool_prompt(
                self.tools.get_prompt_description(),
                self.config.compensation.thinking_mode,
            )
        return self._system_prompt

    def _record_step(self, step_type: str, content: str, **kwargs: Any):
        step = AgentStep(step_type=step_type, content=content, metadata=kwargs)
        self._steps.append(step)
        self.on_step(step)

    def run(self, user_message: str) -> str:
        """Process a user message through the agent loop.

        Returns the final text response to the user.
        """
        self.compensator.reset()
        self._steps.clear()

        # Add user message to memory
        self.memory.add_turn("user", user_message)

        # Build initial message list
        messages = self._build_messages()

        tier = self.router.current_tier
        step_count = 0

        while step_count < MAX_AGENT_STEPS:
            step_count += 1

            # Call LLM
            self._record_step("llm_call", f"Calling {tier} model...")
            response = self.router.chat(
                messages=messages,
                tier=tier,
                temperature=0.3,
            )

            self._record_step(
                "llm_response",
                response.content,
                thinking=response.thinking,
                latency_ms=response.latency_ms,
                model=response.model,
                usage=response.usage,
            )

            if response.finish_reason == "error":
                # API error - try compensation
                comp = self.compensator.next_strategy(
                    messages, response.content, "API error", tier
                )
                if comp and comp.success:
                    self._record_step("compensation", comp.notes, strategy=comp.strategy)
                    if comp.modified_messages:
                        messages = comp.modified_messages
                    if comp.escalated_tier:
                        tier = comp.escalated_tier
                    continue
                else:
                    self.memory.add_turn("assistant", response.content)
                    return response.content

            # Check for tool calls
            if response.has_tool_call:
                tc = response.tool_calls[0]  # Process one tool at a time
                return self._handle_tool_call(tc, messages, tier, step_count)

            # No tool call - check if the response looks like a failed tool call attempt
            if self._looks_like_failed_tool_call(response.content):
                comp = self.compensator.next_strategy(
                    messages,
                    response.content,
                    "Response looks like a malformed tool call",
                    tier,
                )
                if comp and comp.success:
                    self._record_step("compensation", comp.notes, strategy=comp.strategy)
                    if comp.modified_messages:
                        messages = comp.modified_messages
                    if comp.escalated_tier:
                        tier = comp.escalated_tier
                    continue

            # Clean text response - return to user
            final = response.content
            self.memory.add_turn("assistant", final)
            return final

        # Exceeded step limit
        return "[Agent reached maximum steps. Please try a simpler request.]"

    def _handle_tool_call(
        self,
        tc: ToolCall,
        messages: list[dict[str, Any]],
        tier: str,
        step_count: int,
    ) -> str:
        """Handle a tool call and continue the agent loop."""
        self._record_step(
            "tool_call",
            f"Calling tool: {tc.name}",
            tool=tc.name,
            args=tc.arguments,
        )

        # Execute tool
        result = self.tools.execute(tc.name, tc.arguments)
        self._record_step(
            "tool_result",
            result.to_message(),
            success=result.success,
            tool=tc.name,
        )

        # Add tool call and result to messages
        messages.append({
            "role": "assistant",
            "content": f'{{"tool": "{tc.name}", "args": {_safe_json(tc.arguments)}}}',
        })
        messages.append({
            "role": "user",
            "content": f"[Tool Result for {tc.name}]\n{result.to_message()}",
        })

        # Continue the agent loop
        while step_count < MAX_AGENT_STEPS:
            step_count += 1

            self._record_step("llm_call", f"Continuing with {tier} model...")
            response = self.router.chat(
                messages=messages,
                tier=tier,
                temperature=0.3,
            )

            self._record_step(
                "llm_response",
                response.content,
                thinking=response.thinking,
                latency_ms=response.latency_ms,
            )

            if response.has_tool_call:
                tc = response.tool_calls[0]
                self._record_step(
                    "tool_call",
                    f"Calling tool: {tc.name}",
                    tool=tc.name,
                    args=tc.arguments,
                )

                result = self.tools.execute(tc.name, tc.arguments)
                self._record_step(
                    "tool_result",
                    result.to_message(),
                    success=result.success,
                    tool=tc.name,
                )

                messages.append({
                    "role": "assistant",
                    "content": f'{{"tool": "{tc.name}", "args": {_safe_json(tc.arguments)}}}',
                })
                messages.append({
                    "role": "user",
                    "content": f"[Tool Result for {tc.name}]\n{result.to_message()}",
                })
                continue

            # Text response - done
            final = response.content
            self.memory.add_turn("assistant", final)
            return final

        return "[Agent reached maximum steps. Please try a simpler request.]"

    def _build_messages(self) -> list[dict[str, Any]]:
        """Build the message list for the LLM."""
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        msgs.extend(self.memory.get_messages())
        return msgs

    def _looks_like_failed_tool_call(self, content: str) -> bool:
        """Check if the response looks like a failed attempt at a tool call."""
        indicators = [
            '"tool"' in content and '"args"' in content and "{" in content,
            "```json" in content and '"tool"' in content,
            content.strip().startswith("{") and '"tool_call"' in content,
        ]
        return any(indicators) and len(content) < 500


def _safe_json(obj: Any) -> str:
    """Safely serialize to JSON string."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)
