"""Core agent loop - ReAct pattern with weak LLM compensation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

from open_harness.config import HarnessConfig
from open_harness.llm.client import LLMResponse, ToolCall
from open_harness.llm.compensator import Compensator, build_tool_prompt
from open_harness.llm.router import ModelRouter
from open_harness.memory.store import MemoryStore
from open_harness.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 20


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: str  # "thinking", "text", "tool_call", "tool_result", "compensation", "status", "done"
    data: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent:
    """ReAct agent with streaming and weak LLM compensation."""

    def __init__(
        self,
        config: HarnessConfig,
        tools: ToolRegistry,
        memory: MemoryStore,
    ):
        self.config = config
        self.tools = tools
        self.memory = memory
        self.router = ModelRouter(config)
        self.compensator = Compensator(config.compensation)
        self._system_prompt: str | None = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_tool_prompt(
                self.tools.get_prompt_description(),
                self.config.compensation.thinking_mode,
            )
        return self._system_prompt

    def run_stream(self, user_message: str) -> Generator[AgentEvent, None, None]:
        """Process a user message, yielding events as they happen.

        Events:
          status       - progress info (e.g. "Calling model...")
          thinking     - LLM thinking content
          text         - displayable text chunk (stream to terminal)
          tool_call    - about to execute a tool
          tool_result  - tool execution result
          compensation - retry/escalation info
          done         - final complete response
        """
        self.compensator.reset()
        self.memory.add_turn("user", user_message)
        messages = self._build_messages()
        tier = self.router.current_tier

        for step in range(MAX_AGENT_STEPS):
            yield AgentEvent("status", f"Calling {tier} model...")

            # --- stream from LLM ---
            response = yield from self._stream_llm(messages, tier)

            if response.finish_reason == "error":
                comp = self.compensator.next_strategy(
                    messages, response.content, "API error", tier,
                )
                if comp and comp.success:
                    yield AgentEvent("compensation", comp.notes, {"strategy": comp.strategy})
                    if comp.modified_messages:
                        messages = comp.modified_messages
                    if comp.escalated_tier:
                        tier = comp.escalated_tier
                    continue
                self.memory.add_turn("assistant", response.content)
                yield AgentEvent("done", response.content)
                return

            # --- tool call ---
            if response.has_tool_call:
                tc = response.tool_calls[0]
                yield AgentEvent("tool_call", tc.name, {"tool": tc.name, "args": tc.arguments})

                result = self.tools.execute(tc.name, tc.arguments)
                yield AgentEvent(
                    "tool_result",
                    result.to_message(),
                    {"success": result.success, "tool": tc.name},
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

            # --- failed tool call? ---
            if self._looks_like_failed_tool_call(response.content):
                comp = self.compensator.next_strategy(
                    messages, response.content, "Malformed tool call", tier,
                )
                if comp and comp.success:
                    yield AgentEvent("compensation", comp.notes, {"strategy": comp.strategy})
                    if comp.modified_messages:
                        messages = comp.modified_messages
                    if comp.escalated_tier:
                        tier = comp.escalated_tier
                    continue

            # --- normal text response ---
            self.memory.add_turn("assistant", response.content)
            yield AgentEvent("done", response.content, {"latency_ms": response.latency_ms})
            return

        yield AgentEvent("done", "[Agent reached maximum steps.]")

    # ---- non-streaming fallback (kept for simple use cases) ----

    def run(self, user_message: str) -> str:
        """Non-streaming convenience wrapper. Returns final text."""
        final = ""
        for event in self.run_stream(user_message):
            if event.type == "done":
                final = event.data
        return final

    # ---- internal helpers ----

    def _stream_llm(
        self,
        messages: list[dict[str, Any]],
        tier: str,
    ) -> Generator[AgentEvent, None, LLMResponse]:
        """Stream LLM response, yielding thinking/text events.

        Returns the final LLMResponse.
        """
        gen = self.router.chat_stream(messages=messages, tier=tier, temperature=0.3)

        response: LLMResponse | None = None
        try:
            while True:
                event_type, data = next(gen)
                if event_type == "thinking":
                    yield AgentEvent("thinking", data)
                elif event_type == "text":
                    yield AgentEvent("text", data)
        except StopIteration as e:
            response = e.value

        if response is None:
            response = LLMResponse(content="", finish_reason="error")

        return response

    def _build_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            *self.memory.get_messages(),
        ]

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
