"""Reasoner — interprets LLM output and decides the next action.

The Reasoner is a pure function: it takes an LLMResponse and returns a
``ReasonerDecision`` describing what the orchestrator should do next.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

from open_harness_v2.types import LLMResponse, ToolCall

_logger = logging.getLogger(__name__)


class ActionType(enum.Enum):
    """What the orchestrator should do after reasoning."""

    EXECUTE_TOOLS = "execute_tools"  # Run the extracted tool calls
    RESPOND = "respond"              # Final text response — end the loop
    CONTINUE = "continue"            # LLM produced content but loop should continue
    ERROR = "error"                  # Unrecoverable error


@dataclass
class ReasonerDecision:
    """Output of the Reasoner."""

    action: ActionType
    tool_calls: list[ToolCall] = field(default_factory=list)
    response_text: str = ""
    thinking: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Reasoner:
    """Interprets LLM output and decides the next action.

    Decision logic:
    1. If the response has tool calls → EXECUTE_TOOLS
    2. If the response has text content and no tool calls → RESPOND (done)
    3. If the response is an error → ERROR
    4. If the response is empty → ERROR
    """

    def __init__(self, max_steps: int = 50) -> None:
        self._max_steps = max_steps
        self._step_count = 0

    def decide(self, response: LLMResponse) -> ReasonerDecision:
        """Analyze an LLM response and produce a decision."""
        self._step_count += 1

        # Check step limit
        if self._step_count > self._max_steps:
            _logger.warning("Step limit reached (%d)", self._max_steps)
            return ReasonerDecision(
                action=ActionType.ERROR,
                error=f"Step limit reached ({self._max_steps}). Stopping.",
                response_text=response.content,
                thinking=response.thinking,
            )

        # Error response from LLM
        if response.finish_reason == "error":
            return ReasonerDecision(
                action=ActionType.ERROR,
                error=response.content or "LLM returned an error",
                thinking=response.thinking,
            )

        # Empty response
        if not response.content and not response.tool_calls:
            return ReasonerDecision(
                action=ActionType.ERROR,
                error="Empty response from LLM",
                thinking=response.thinking,
            )

        # Has tool calls → execute them
        if response.has_tool_calls:
            return ReasonerDecision(
                action=ActionType.EXECUTE_TOOLS,
                tool_calls=response.tool_calls,
                response_text=response.content,
                thinking=response.thinking,
            )

        # Text response only → done
        return ReasonerDecision(
            action=ActionType.RESPOND,
            response_text=response.content,
            thinking=response.thinking,
        )

    def reset(self) -> None:
        """Reset step counter for a new goal."""
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count
