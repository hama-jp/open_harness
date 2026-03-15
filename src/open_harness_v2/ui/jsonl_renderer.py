"""JSONL renderer — streams events as newline-delimited JSON for CI/automation.

Inspired by Codex CLI's non-interactive mode. Each agent event is emitted
as a single JSON line to stdout, making it easy to pipe into other tools.

Usage::

    renderer = JSONLRenderer()
    renderer.attach(event_bus)
    # Events are printed as JSONL to stdout
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType


class JSONLRenderer:
    """Streams agent events as JSONL to stdout.

    Parameters
    ----------
    output:
        File-like object for output (default: sys.stdout).
    include_tool_output:
        Whether to include full tool output in events.
    """

    def __init__(
        self,
        output: Any = None,
        include_tool_output: bool = True,
    ) -> None:
        self._output = output or sys.stdout
        self._include_tool_output = include_tool_output
        self._start_time = time.monotonic()

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to all events on the given bus."""
        event_bus.subscribe("*", self._handle)

    def _handle(self, event: AgentEvent) -> None:
        """Serialize and emit an event as JSONL."""
        data = self._serialize(event)
        line = json.dumps(data, default=str, ensure_ascii=False)
        self._output.write(line + "\n")
        self._output.flush()

    def _serialize(self, event: AgentEvent) -> dict[str, Any]:
        """Convert an AgentEvent to a JSON-serializable dict."""
        elapsed = time.monotonic() - self._start_time
        result: dict[str, Any] = {
            "type": event.type.value,
            "timestamp": event.timestamp,
            "elapsed_s": round(elapsed, 3),
        }

        data = event.data

        if event.type == EventType.AGENT_STARTED:
            result["goal"] = data.get("goal", "")

        elif event.type == EventType.AGENT_DONE:
            result["response"] = data.get("response", "")
            result["steps"] = data.get("steps", 0)

        elif event.type == EventType.AGENT_ERROR:
            result["error"] = data.get("error", "")

        elif event.type == EventType.AGENT_CANCELLED:
            result["response"] = data.get("response", "")

        elif event.type == EventType.TOOL_EXECUTING:
            result["tool"] = data.get("tool", "")
            result["args"] = data.get("args", {})

        elif event.type == EventType.TOOL_EXECUTED:
            result["tool"] = data.get("tool", "")
            result["success"] = data.get("success", True)
            result["elapsed_ms"] = data.get("elapsed_ms", 0)
            if self._include_tool_output:
                output = data.get("output", "")
                # Limit output size in JSONL mode
                if len(output) > 2000:
                    output = output[:2000] + f"... ({len(output)} chars total)"
                result["output"] = output

        elif event.type == EventType.TOOL_ERROR:
            result["tool"] = data.get("tool", "")
            result["error"] = data.get("error", "")
            result["elapsed_ms"] = data.get("elapsed_ms", 0)

        elif event.type == EventType.LLM_RESPONSE:
            result["model"] = data.get("model", "")
            result["latency_ms"] = data.get("latency_ms", 0)
            result["usage"] = data.get("usage", {})

        elif event.type == EventType.LLM_THINKING:
            thinking = data.get("thinking", "")
            result["thinking"] = thinking[:500] if thinking else ""

        elif event.type == EventType.POLICY_VIOLATION:
            result["tool"] = data.get("tool", "")
            result["rule"] = data.get("rule", "")
            result["message"] = data.get("message", "")

        elif event.type == EventType.REASONER_DECISION:
            result["action"] = data.get("action", "")
            result["step"] = data.get("step", 0)

        else:
            # Pass through any other data
            result["data"] = data

        return result
