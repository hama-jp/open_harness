"""Shared data types for Open Harness v2."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tool types
# ---------------------------------------------------------------------------

@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # string, integer, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolCall:
    """Parsed tool call from LLM response."""

    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> str:
        if self.success:
            return self.output
        if self.output:
            return f"[Tool Error] {self.error}\n{self.output}"
        return f"[Tool Error] {self.error}"


# ---------------------------------------------------------------------------
# LLM types
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Unified response from LLM."""

    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(enum.Enum):
    """Event types emitted by the agent system."""

    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_DONE = "agent.done"
    AGENT_ERROR = "agent.error"
    AGENT_CANCELLED = "agent.cancelled"

    # LLM events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_STREAMING = "llm.streaming"
    LLM_THINKING = "llm.thinking"
    LLM_ERROR = "llm.error"

    # Tool events
    TOOL_EXECUTING = "tool.executing"
    TOOL_EXECUTED = "tool.executed"
    TOOL_ERROR = "tool.error"

    # Reasoner events
    REASONER_DECISION = "reasoner.decision"

    # Context events
    CONTEXT_COMPRESSED = "context.compressed"

    # Policy events
    POLICY_VIOLATION = "policy.violation"


@dataclass
class AgentEvent:
    """Event emitted by the agent system via the EventBus."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
