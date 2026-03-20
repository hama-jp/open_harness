"""Reasoner — interprets LLM output and decides the next action.

The Reasoner is a pure function: it takes an LLMResponse and returns a
``ReasonerDecision`` describing what the orchestrator should do next.

Enhanced with progress-aware decision making:
- Detects potential premature completion
- Recommends strategy adjustments based on response patterns
- Tracks response quality trends
"""

from __future__ import annotations

import enum
import logging
import re
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


class QualitySignal(enum.Enum):
    """Quality assessment of the LLM response."""

    GOOD = "good"            # Substantive, on-task response
    WEAK = "weak"            # Vague or superficial
    REPETITIVE = "repetitive"  # Repeats previous content
    OFF_TRACK = "off_track"  # Doesn't address the goal


@dataclass
class ReasonerDecision:
    """Output of the Reasoner."""

    action: ActionType
    tool_calls: list[ToolCall] = field(default_factory=list)
    response_text: str = ""
    thinking: str = ""
    error: str = ""
    quality: QualitySignal = QualitySignal.GOOD
    metadata: dict[str, Any] = field(default_factory=dict)


class Reasoner:
    """Interprets LLM output and decides the next action.

    Decision logic:
    1. If the response has tool calls → EXECUTE_TOOLS
    2. If the response has text content and no tool calls → RESPOND (done)
    3. If the response is an error → ERROR
    4. If the response is empty → ERROR

    Enhanced:
    - Tracks response patterns for quality assessment
    - Detects premature completion attempts
    - Provides quality signals to the orchestrator
    """

    def __init__(self, max_steps: int = 50) -> None:
        self._max_steps = max_steps
        self._step_count = 0
        self._recent_responses: list[str] = []
        self._tool_call_count = 0
        self._text_response_count = 0

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
            self._tool_call_count += 1
            quality = self._assess_tool_call_quality(response)
            return ReasonerDecision(
                action=ActionType.EXECUTE_TOOLS,
                tool_calls=response.tool_calls,
                response_text=response.content,
                thinking=response.thinking,
                quality=quality,
            )

        # Text response only → done
        self._text_response_count += 1
        quality = self._assess_response_quality(response.content)

        # Track for repetition detection
        self._recent_responses.append(response.content[:200])
        if len(self._recent_responses) > 5:
            self._recent_responses.pop(0)

        return ReasonerDecision(
            action=ActionType.RESPOND,
            response_text=response.content,
            thinking=response.thinking,
            quality=quality,
        )

    def reset(self) -> None:
        """Reset step counter for a new goal."""
        self._step_count = 0
        self._recent_responses.clear()
        self._tool_call_count = 0
        self._text_response_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------

    def _assess_tool_call_quality(self, response: LLMResponse) -> QualitySignal:
        """Assess the quality of a tool call decision."""
        # Check for duplicate tool calls in the same response
        names = [tc.name for tc in response.tool_calls]
        if len(names) != len(set(names)):
            return QualitySignal.REPETITIVE

        return QualitySignal.GOOD

    def _assess_response_quality(self, content: str) -> QualitySignal:
        """Assess the quality of a text response."""
        if not content or len(content.strip()) < 20:
            return QualitySignal.WEAK

        # Check for vague / non-committal responses
        vague_patterns = [
            r"\bI('m| am) not sure\b",
            r"\bI can't determine\b",
            r"\bI don't have enough\b",
            r"\bI would need to\b",
            r"\bLet me know if\b",
        ]
        vague_count = sum(
            1 for p in vague_patterns if re.search(p, content, re.I)
        )
        if vague_count >= 2:
            return QualitySignal.WEAK

        # Check for repetition with recent responses
        if self._recent_responses:
            content_lower = content[:200].lower()
            for prev in self._recent_responses:
                if prev.lower() == content_lower:
                    return QualitySignal.REPETITIVE

        return QualitySignal.GOOD
