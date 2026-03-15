"""Console renderer — EventBus subscriber that draws events with Rich.

The renderer knows nothing about the Orchestrator or agent internals.
It simply subscribes to the EventBus and renders each event to the console.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType

_logger = logging.getLogger(__name__)

# Sensitive parameter names — values are masked in display
_SENSITIVE_PARAMS = frozenset({
    "api_key", "token", "password", "secret", "credential",
})

# Long parameter values are truncated to this length
_ARG_DISPLAY_LIMIT = 120


def _format_tool_args(args: dict[str, Any]) -> str:
    """Format tool arguments for human-readable display.

    - Long string values are truncated with an ellipsis
    - Multi-line values show first line + line count
    - Sensitive parameter values are masked
    """
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if any(s in k.lower() for s in _SENSITIVE_PARAMS):
            parts.append(f"{k}=***")
            continue
        if isinstance(v, str):
            if "\n" in v:
                first_line = v.split("\n", 1)[0]
                line_count = v.count("\n") + 1
                display = first_line[:80] + f" ({line_count} lines)"
            elif len(v) > _ARG_DISPLAY_LIMIT:
                display = v[:_ARG_DISPLAY_LIMIT] + "..."
            else:
                display = v
            parts.append(f'{k}="{display}"')
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


class ConsoleRenderer:
    """Render agent events to a Rich Console.

    Parameters
    ----------
    console:
        Rich Console instance for output.
    verbose:
        If True, show detailed events (reasoner decisions, LLM latency).
    """

    def __init__(self, console: Console, verbose: bool = False) -> None:
        self._console = console
        self._verbose = verbose
        self._step_count = 0
        self._goal_start: float = 0.0
        self._llm_waiting_since: float = 0.0

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to all events on the given bus."""
        event_bus.subscribe("*", self._handle)

    def detach(self, event_bus: EventBus) -> None:
        """Unsubscribe from the event bus."""
        event_bus.unsubscribe("*", self._handle)

    def _handle(self, event: AgentEvent) -> None:
        """Dispatch an event to the appropriate renderer."""
        dispatch = {
            EventType.AGENT_STARTED: self._on_agent_started,
            EventType.LLM_THINKING: self._on_thinking,
            EventType.TOOL_EXECUTING: self._on_tool_executing,
            EventType.TOOL_EXECUTED: self._on_tool_executed,
            EventType.TOOL_ERROR: self._on_tool_error,
            EventType.POLICY_VIOLATION: self._on_policy_violation,
            EventType.REASONER_DECISION: self._on_reasoner_decision,
            EventType.LLM_RESPONSE: self._on_llm_response,
            EventType.AGENT_DONE: self._on_agent_done,
            EventType.AGENT_ERROR: self._on_agent_error,
            EventType.AGENT_CANCELLED: self._on_agent_cancelled,
        }
        handler = dispatch.get(event.type)
        if handler:
            try:
                handler(event.data)
            except Exception:
                _logger.exception("Renderer error for event %s", event.type)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_agent_started(self, data: dict[str, Any]) -> None:
        goal = data.get("goal", "")
        self._step_count = 0
        self._goal_start = time.monotonic()
        self._console.print(f"[dim]Goal: {goal}[/dim]")

    def _on_thinking(self, data: dict[str, Any]) -> None:
        thinking = data.get("thinking", "")
        if thinking:
            preview = thinking[:200] + ("..." if len(thinking) > 200 else "")
            self._console.print(f"[dim italic]thinking: {preview}[/dim italic]")

    def _on_tool_executing(self, data: dict[str, Any]) -> None:
        name = data.get("tool", "")
        args = data.get("args", {})
        args_str = _format_tool_args(args)
        step_label = f"[dim][{self._step_count}][/dim] " if self._step_count else ""
        self._console.print(f"{step_label}[yellow]> {name}[/yellow] {args_str}")

    def _on_tool_executed(self, data: dict[str, Any]) -> None:
        name = data.get("tool", "")
        output = data.get("output", "")
        elapsed = data.get("elapsed_ms", 0)
        title_text = name
        if elapsed:
            title_text += f" ({elapsed:.0f}ms)"
        title = f"[green]{title_text}[/green]"

        if len(output) > 1000:
            output = output[:1000] + f"\n... ({len(output)} chars total)"

        if output:
            self._console.print(Panel(output, title=title, border_style="dim"))
        else:
            self._console.print(f"  {title} [dim](no output)[/dim]")

    def _on_tool_error(self, data: dict[str, Any]) -> None:
        name = data.get("tool", "")
        error = data.get("error", "Unknown error")
        title = f"[red]{name} (failed)[/red]"
        self._console.print(Panel(str(error), title=title, border_style="red"))

    def _on_policy_violation(self, data: dict[str, Any]) -> None:
        tool = data.get("tool", "")
        message = data.get("message", "Unknown violation")
        rule = data.get("rule", "")
        prefix = f"[bold]{tool}[/bold]: " if tool else ""
        self._console.print(
            f"[red]Policy: {prefix}{message}[/red]", highlight=False,
        )
        if self._verbose and rule:
            self._console.print(f"  [dim]rule: {rule}[/dim]")

    def _on_reasoner_decision(self, data: dict[str, Any]) -> None:
        step = data.get("step", "?")
        action = data.get("action", "?")
        self._step_count = step if isinstance(step, int) else self._step_count
        if not self._verbose:
            return
        self._console.print(f"[dim]Step {step}: {action}[/dim]")

    def _on_llm_response(self, data: dict[str, Any]) -> None:
        if not self._verbose:
            return
        model = data.get("model", "?")
        latency = data.get("latency_ms", 0)
        tokens = data.get("usage", {})
        token_info = ""
        if tokens:
            total = tokens.get("total_tokens", 0)
            if total:
                token_info = f" [{total} tokens]"
        self._console.print(f"[dim]{model} — {latency:.0f}ms{token_info}[/dim]")

    def _on_agent_done(self, data: dict[str, Any]) -> None:
        response = data.get("response", "")
        steps = data.get("steps", 0)
        if response:
            self._console.print()
            self._console.print(Markdown(response))
        elapsed = time.monotonic() - self._goal_start if self._goal_start else 0
        if elapsed > 0:
            self._console.print(
                f"\n[dim]Completed in {steps} step(s), {elapsed:.1f}s[/dim]"
            )
        elif self._verbose:
            self._console.print(f"\n[dim]Completed in {steps} step(s)[/dim]")

    def _on_agent_error(self, data: dict[str, Any]) -> None:
        error = data.get("error", "Unknown error")
        self._console.print(f"\n[red]Error: {error}[/red]")

    def _on_agent_cancelled(self, data: dict[str, Any]) -> None:
        self._console.print("\n[yellow]Cancelled[/yellow]")
