"""Typed context object with layer-based compression.

Each layer knows how to compress itself. The ``AgentContext`` assembles them
into an OpenAI-compatible message list within a token budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)

# Rough chars-per-token ratio for budget estimation
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        total += _estimate_tokens(str(content))
    return total


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

@dataclass
class SystemLayer:
    """System prompt — never compressed.

    Contains role description, tool descriptions, project context, and
    memory/rules.
    """

    role: str = "You are an autonomous AI agent with access to tools."
    tools_description: str = ""
    project_context: str = ""
    extra: str = ""

    def to_messages(self) -> list[dict[str, Any]]:
        parts = [self.role]
        if self.tools_description:
            parts.append(f"\n## Available Tools\n\n{self.tools_description}")
        if self.project_context:
            parts.append(f"\n## Project Context\n\n{self.project_context}")
        if self.extra:
            parts.append(f"\n{self.extra}")
        return [{"role": "system", "content": "\n".join(parts)}]


@dataclass
class PlanLayer:
    """Current plan — compressed to current + next N steps."""

    steps: list[str] = field(default_factory=list)
    current_step: int = 0
    _lookahead: int = 2  # show current step + N next steps

    def to_messages(self) -> list[dict[str, Any]]:
        if not self.steps:
            return []
        visible = self.steps[self.current_step:self.current_step + self._lookahead + 1]
        if not visible:
            return []
        lines = [f"## Current Plan (step {self.current_step + 1}/{len(self.steps)})"]
        for i, step in enumerate(visible):
            marker = "→" if i == 0 else " "
            lines.append(f"  {marker} {self.current_step + i + 1}. {step}")
        return [{"role": "system", "content": "\n".join(lines)}]

    def advance(self) -> bool:
        """Move to the next step. Returns False if already at the end."""
        if self.current_step >= len(self.steps) - 1:
            return False
        self.current_step += 1
        return True

    @property
    def is_complete(self) -> bool:
        return not self.steps or self.current_step >= len(self.steps)


@dataclass
class HistoryLayer:
    """Compressible past turns.

    Uses two-level compression:
    - L1: Adjacent tool-call + tool-result pairs → one-line summary
    - L2: Consecutive L1 summaries → aggregated count
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    _protected_tail: int = 6  # keep last N messages uncompressed

    def add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def to_messages(self, budget: int | None = None) -> list[dict[str, Any]]:
        if budget is None or _estimate_messages_tokens(self.messages) <= budget:
            return list(self.messages)
        return self._compress(budget)

    def _compress(self, budget: int) -> list[dict[str, Any]]:
        """L1+L2 compression to fit within token budget."""
        if len(self.messages) <= self._protected_tail:
            return list(self.messages)

        # Split into compressible and protected
        compressible = self.messages[:-self._protected_tail]
        protected = self.messages[-self._protected_tail:]

        # L1: Summarize tool-call/result pairs
        compressed = self._l1_compress(compressible)

        # If still over budget, apply L2
        if _estimate_messages_tokens(compressed + protected) > budget:
            compressed = self._l2_compress(compressed)

        # If still over budget, drop oldest compressed entries
        while (
            compressed
            and _estimate_messages_tokens(compressed + protected) > budget
        ):
            compressed.pop(0)

        return compressed + protected

    @staticmethod
    def _l1_compress(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compress adjacent assistant(tool_call) + user(tool_result) pairs."""
        result: list[dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            content = msg.get("content", "") or ""

            # Detect tool call + result pair
            if (
                msg.get("role") == "assistant"
                and i + 1 < len(messages)
                and messages[i + 1].get("role") == "user"
            ):
                next_content = messages[i + 1].get("content", "") or ""
                if "[Tool Result" in str(next_content):
                    # Summarize the pair
                    tool_name = _extract_tool_name(content)
                    status = "OK" if "success" in str(next_content).lower() or "[Tool Result" in str(next_content) else "error"
                    summary = f"[Tool: {tool_name} → {status}]"
                    result.append({
                        "role": "user",
                        "content": summary,
                    })
                    i += 2
                    continue

            result.append(msg)
            i += 1
        return result

    @staticmethod
    def _l2_compress(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge consecutive L1 summaries into aggregated counts."""
        result: list[dict[str, Any]] = []
        summary_batch: list[str] = []

        def flush_batch():
            if summary_batch:
                count = len(summary_batch)
                result.append({
                    "role": "user",
                    "content": f"[{count} tool calls summarized]",
                })
                summary_batch.clear()

        for msg in messages:
            content = msg.get("content", "") or ""
            if isinstance(content, str) and content.startswith("[Tool:"):
                summary_batch.append(content)
            else:
                flush_batch()
                result.append(msg)

        flush_batch()
        return result


@dataclass
class WorkingLayer:
    """Recent tool results — protected from compression but truncated."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    _max_per_result: int = 3000  # max chars per tool result

    def add(self, message: dict[str, Any]) -> None:
        # Truncate long tool results
        content = message.get("content", "")
        if isinstance(content, str) and len(content) > self._max_per_result:
            half = self._max_per_result // 2
            truncated = (
                content[:half]
                + f"\n\n[...{len(content) - self._max_per_result} chars truncated...]\n\n"
                + content[-half:]
            )
            message = {**message, "content": truncated}
        self.messages.append(message)

    def to_messages(self) -> list[dict[str, Any]]:
        return list(self.messages)

    def promote_to_history(self, history: HistoryLayer) -> None:
        """Move all working messages to history (after they're no longer 'recent')."""
        for msg in self.messages:
            history.add(msg)
        self.messages.clear()


# ---------------------------------------------------------------------------
# Main context
# ---------------------------------------------------------------------------

class AgentContext:
    """Structured context with layer-specific compression.

    Usage::

        ctx = AgentContext()
        ctx.system.role = "You are ..."
        ctx.system.tools_description = registry.get_prompt_description()
        ctx.history.add({"role": "user", "content": goal})
        messages = ctx.to_messages(budget=8000)
    """

    def __init__(self) -> None:
        self.system = SystemLayer()
        self.plan = PlanLayer()
        self.history = HistoryLayer()
        self.working = WorkingLayer()

    def to_messages(self, budget: int = 0) -> list[dict[str, Any]]:
        """Convert all layers to an OpenAI-compatible message list.

        Parameters
        ----------
        budget:
            Target token budget (0 = unlimited).  System + plan + working
            are always included; history is compressed to fit.
        """
        system_msgs = self.system.to_messages()
        plan_msgs = self.plan.to_messages()
        working_msgs = self.working.to_messages()

        if budget <= 0:
            # No budget constraint — return everything
            return system_msgs + plan_msgs + self.history.messages + working_msgs

        # Calculate remaining budget for history
        fixed_tokens = (
            _estimate_messages_tokens(system_msgs)
            + _estimate_messages_tokens(plan_msgs)
            + _estimate_messages_tokens(working_msgs)
        )
        history_budget = max(0, budget - fixed_tokens)
        history_msgs = self.history.to_messages(budget=history_budget)

        return system_msgs + plan_msgs + history_msgs + working_msgs

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.history.add({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.history.add({"role": "assistant", "content": content})

    def add_tool_result(self, tool_name: str, result_text: str) -> None:
        """Add a tool result to the working layer."""
        self.working.add({
            "role": "user",
            "content": f"[Tool Result for {tool_name}]\n{result_text}",
        })

    def cycle_working(self) -> None:
        """Promote working messages to history for the next turn."""
        self.working.promote_to_history(self.history)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tool_name(text: str) -> str:
    """Try to extract a tool name from an assistant message."""
    # Look for {"tool": "name"...}
    import re
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', str(text))
    if m:
        return m.group(1)
    return "unknown"
