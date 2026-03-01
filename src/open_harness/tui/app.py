"""Textual TUI for Open Harness — two-pane agent interface."""

from __future__ import annotations

import io
import re as _re
import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import (
    Collapsible, Footer, Header, Input, OptionList,
    RichLog, Static, TextArea,
)
from textual.worker import get_current_worker

from open_harness.agent import Agent, AgentEvent
from open_harness.cli import (
    _expand_at_references,
    _notifications,
    handle_command,
)
from open_harness.config import HarnessConfig
from open_harness.memory.store import MemoryStore
from open_harness.project import ProjectContext
from open_harness.tasks.queue import TaskQueueManager, TaskStatus, TaskStore
from open_harness.tools.base import ToolRegistry

_MARKUP_RE = _re.compile(r"\[/?[^\]]*\]")


class SelectableLog(RichLog):
    """RichLog that tracks plain text for copy-mode support."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._plain_lines: list[str] = []

    def write(
        self,
        content: Any,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
    ) -> "SelectableLog":
        if isinstance(content, str):
            self._plain_lines.append(_MARKUP_RE.sub("", content))
        elif isinstance(content, Markdown):
            self._plain_lines.append(content.markup)
        elif isinstance(content, Panel):
            renderable = content.renderable
            if isinstance(renderable, str):
                self._plain_lines.append(renderable)
            else:
                self._plain_lines.append(str(renderable))
        else:
            self._plain_lines.append(str(content))
        return super().write(  # type: ignore[return-value]
            content, width=width, expand=expand, shrink=shrink, scroll_end=scroll_end,
        )

    def get_plain_text(self) -> str:
        return "\n".join(self._plain_lines)

    def clear(self) -> "SelectableLog":
        self._plain_lines.clear()
        return super().clear()  # type: ignore[return-value]


class CopyScreen(ModalScreen[None]):
    """Modal screen for selecting and copying output text."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("ctrl+c", "copy_selected", "Copy", show=False),
    ]

    DEFAULT_CSS = """
    CopyScreen {
        align: center middle;
    }
    CopyScreen #copy-container {
        width: 90%;
        height: 85%;
        border: solid $accent;
        background: $surface;
    }
    CopyScreen #copy-header {
        dock: top;
        width: 100%;
        height: auto;
        padding: 0 1;
        background: $boost;
    }
    CopyScreen #copy-area {
        width: 100%;
        height: 1fr;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="copy-container"):
            yield Static(
                "[bold]Copy Mode[/bold] — "
                "Select text with mouse or Shift+arrows, "
                "Ctrl+C to copy, Escape to close",
                id="copy-header",
            )
            yield TextArea(self._text, read_only=True, id="copy-area")

    def on_mount(self) -> None:
        self.query_one("#copy-area", TextArea).focus()

    def action_copy_selected(self) -> None:
        """Copy selected text; fallback if TextArea didn't handle Ctrl+C."""
        area = self.query_one("#copy-area", TextArea)
        if area.selected_text:
            self.app.copy_to_clipboard(area.selected_text)


class StreamingDisplay:
    """Adaptor: renders AgentEvents to a Console writing to a StringIO."""

    def __init__(self, console: Console):
        self.con = console
        self._streaming = False

    def handle(self, event: AgentEvent):
        from open_harness.cli import StreamingDisplay as _CliDisplay
        # Delegate to a throwaway CLI display backed by our console
        if not hasattr(self, "_cli"):
            self._cli = _CliDisplay(self.con)
        self._cli.handle(event)


class HarnessApp(App):
    """Two-pane Textual TUI for Open Harness."""

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("tab", "cycle_mode", "Mode", show=True),
        Binding("ctrl+s", "toggle_sidebar", "Sidebar", show=True),
        Binding("ctrl+a", "toggle_agent_panel", "Agents", show=True),
        Binding("ctrl+c", "copy_mode", "Copy", show=True),
        Binding("ctrl+l", "clear_log", "Clear", show=True),
        Binding("f1", "show_help", "Help", show=True),
    ]

    def __init__(
        self,
        *,
        config: HarnessConfig,
        project: ProjectContext,
        agent: Agent,
        tools: ToolRegistry,
        memory: MemoryStore,
        task_queue: TaskQueueManager,
        task_store: TaskStore,
        version: str,
    ):
        super().__init__()
        self.harness_config = config
        self.project = project
        self.agent = agent
        self.tools = tools
        self.memory = memory
        self.task_queue = task_queue
        self.task_store = task_store
        self.version = version

        # State
        self._modes = ["plan", "goal"]
        self._mode_index = 0
        self._agent_running = False
        self._instruction_queue: list[str] = []
        self._text_buffer = ""

        # Stats
        self._tool_calls = 0
        self._tool_ok = 0
        self._tool_fail = 0
        self._tool_counts: dict[str, int] = {}
        self._compensations = 0
        self._agent_start_time: float | None = None

        # Plan tracking
        self._plan_text = ""

        # Sub-agent tracking
        self._active_agents: dict[str, float] = {}  # tool_name → start_time
        self._agent_panel_timer: Timer | None = None

        # Input history: (display_text, scroll_anchor)
        self._input_history: list[tuple[str, int]] = []

    @property
    def current_mode(self) -> str:
        return self._modes[self._mode_index]

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="output-area"):
                yield SelectableLog(id="output", highlight=True, markup=True, wrap=True)
                yield RichLog(id="agent-panel", highlight=True, markup=True, wrap=True)
            with VerticalScroll(id="sidebar"):
                with Collapsible(title="History", collapsed=False):
                    yield OptionList(id="history-list")
                with Collapsible(title="Plan", collapsed=False):
                    yield Static("No plan yet.", id="plan-content")
                with Collapsible(title="Queue", collapsed=False):
                    yield Static("No queued instructions.", id="queue-content")
                with Collapsible(title="Agents", collapsed=False):
                    yield Static("No active agents.", id="agents-content")
                with Collapsible(title="Tasks", collapsed=False):
                    yield Static("No background tasks.", id="tasks-content")
                with Collapsible(title="Stats", collapsed=True):
                    yield Static("Waiting for execution...", id="stats-content")
        yield Input(placeholder="Type a message or /command...", id="user-input")
        yield Footer()

    def on_mount(self) -> None:
        tool_names = [t.name for t in self.tools.list_tools()]
        self.title = (
            f"Open Harness v{self.version} | "
            f"{self.harness_config.llm.default_tier} | "
            f"{self.agent.policy.config.mode} | "
            f"{len(tool_names)} tools"
        )
        output = self.query_one("#output", RichLog)
        output.write(
            f"[bold bright_white]Open Harness v{self.version}[/bold bright_white]  "
            f"[dim]Self-driving AI agent for local LLMs[/dim]"
        )
        output.write(
            f"[dim]Project: {self.project.info['type']} @ {self.project.info['root']}[/dim]"
        )
        output.write(
            f"[dim]Tools ({len(tool_names)}): {', '.join(tool_names)}[/dim]"
        )
        output.write(
            f"[dim]Mode: {self.current_mode} (Tab to cycle)[/dim]\n"
        )
        # Agent panel starts hidden
        agent_panel = self.query_one("#agent-panel", RichLog)
        agent_panel.border_title = "Sub-Agent Streaming Output"
        agent_panel.display = False
        self.query_one("#user-input", Input).focus()
        self.set_interval(2.0, self._check_task_notifications)

    # ------------------------------------------------------------------
    # Bindings
    # ------------------------------------------------------------------

    def action_cycle_mode(self) -> None:
        # Don't steal Tab from the input widget when it is focused
        focused = self.focused
        if isinstance(focused, Input):
            return
        self._mode_index = (self._mode_index + 1) % len(self._modes)
        output = self.query_one("#output", RichLog)
        output.write(f"[bold]Mode: {self.current_mode}[/bold]")

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    def action_toggle_agent_panel(self) -> None:
        panel = self.query_one("#agent-panel", RichLog)
        panel.display = not panel.display

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "history-list":
            idx = event.option_index
            if 0 <= idx < len(self._input_history):
                _, anchor = self._input_history[idx]
                output = self.query_one("#output", RichLog)
                output.scroll_to(y=anchor, animate=True)

    def action_copy_mode(self) -> None:
        output = self.query_one("#output", SelectableLog)
        text = output.get_plain_text()
        if text.strip():
            self.push_screen(CopyScreen(text))

    def action_clear_log(self) -> None:
        self.query_one("#output", SelectableLog).clear()

    def action_show_help(self) -> None:
        output = self.query_one("#output", RichLog)
        output.write(Panel(
            "[bold]Modes (Tab outside input to cycle):[/bold]\n"
            "  plan    - Discuss, explore, and plan with the agent\n"
            "  goal    - Agent works autonomously\n\n"
            "[bold]Commands:[/bold]\n"
            "  /goal <task>   - Run a goal\n"
            "  /submit <task> - Submit to background\n"
            "  /tasks         - List background tasks\n"
            "  /result <id>   - Show task result\n"
            "  /cancel [id]   - Cancel running/queued task\n"
            "  /model [tier]  - Show/set model tier\n"
            "  /tier [name]   - Show/set tier\n"
            "  /policy [mode] - Show/set policy\n"
            "  /tools         - List tools\n"
            "  /project       - Show project context\n"
            "  /memory        - Show learned memories\n"
            "  /clear         - Clear conversation\n"
            "  /quit          - Exit\n\n"
            "[bold]Keys:[/bold]\n"
            "  Tab        - Cycle mode (when not in input)\n"
            "  Ctrl+S     - Toggle sidebar\n"
            "  Ctrl+A     - Toggle agent panel\n"
            "  Ctrl+C     - Copy mode (select & copy text)\n"
            "  Ctrl+L     - Clear log\n"
            "  Ctrl+Q     - Quit",
            title="Help",
            border_style="cyan",
        ))

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        if text.startswith("/"):
            if text.strip().lower() in ("/quit", "/exit", "/q"):
                self.exit()
                return
            self._handle_slash_command(text)
            return

        if self._agent_running:
            self._instruction_queue.append(text)
            self._refresh_queue_section()
            output = self.query_one("#output", RichLog)
            output.write(f"[yellow]Queued: {text[:60]}[/yellow]")
            return

        self._run_input(text)

    def _handle_slash_command(self, cmd: str) -> None:
        output = self.query_one("#output", RichLog)

        # Commands that start the agent
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/goal" and arg:
            self._run_input(arg, force_mode="goal")
            return
        if command in ("/submit", "/bg") and arg:
            self._run_input(arg, force_mode="submit")
            return

        # Delegate to existing CLI command handler via StringIO capture
        buf = io.StringIO()
        cap_console = Console(file=buf, force_terminal=True, width=80)
        from open_harness.cli import StreamingDisplay as CliDisplay
        cap_display = CliDisplay(cap_console)
        result = handle_command(cmd, self.agent, self.harness_config, cap_display)
        captured = buf.getvalue()
        if captured.strip():
            output.write(captured.rstrip())
        elif result is True:
            pass  # command handled, no output
        elif not result:
            output.write(f"[red]Unknown command: {cmd}[/red]")

    def _run_input(self, text: str, force_mode: str | None = None) -> None:
        mode = force_mode or self.current_mode

        if mode == "submit":
            task = self.task_queue.submit(text)
            output = self.query_one("#output", RichLog)
            output.write(f"[green]Task {task.id} queued: {task.goal[:60]}[/green]")
            output.write(f"[dim]Check: /tasks | /result {task.id}[/dim]")
            self._refresh_tasks_section()
            return

        text = _expand_at_references(text, self.project.root)
        self._agent_running = True
        self._text_buffer = ""
        self._agent_start_time = time.monotonic()

        # Record input history with scroll anchor
        output = self.query_one("#output", RichLog)
        anchor = output.line_count
        mode_tag = f"[dim]({mode})[/dim] " if mode != self.current_mode else ""
        output.write(f"[bold bright_white]> {mode_tag}{text[:120]}[/bold bright_white]")
        display = f"[{mode}] {text[:40]}" if len(text) > 40 else f"[{mode}] {text}"
        self._input_history.append((display, anchor))
        self._refresh_history_section()

        self.run_agent(text, mode)

    # ------------------------------------------------------------------
    # Agent worker (runs in thread)
    # ------------------------------------------------------------------

    @work(thread=True)
    def run_agent(self, user_input: str, mode: str) -> None:
        worker = get_current_worker()
        gen = (
            self.agent.run_stream(user_input)
            if mode == "plan"
            else self.agent.run_goal(user_input)
        )
        try:
            for event in gen:
                if worker.is_cancelled:
                    break
                self.call_from_thread(self._handle_agent_event, event)
        except Exception as e:
            self.call_from_thread(self._show_error, str(e))
        finally:
            self.call_from_thread(self._on_agent_done)

    # ------------------------------------------------------------------
    # Event → widget mapping
    # ------------------------------------------------------------------

    def _handle_agent_event(self, event: AgentEvent) -> None:
        output = self.query_one("#output", RichLog)

        if event.type == "status":
            self._flush_text_buffer()
            output.write(f"[dim]{event.data}[/dim]")
            self._update_plan_from_status(event.data)

        elif event.type == "thinking":
            self._flush_text_buffer()
            first = event.data.split("\n")[0][:80]
            output.write(f"[dim italic]thinking: {first}...[/dim italic]")

        elif event.type == "text":
            self._text_buffer += event.data

        elif event.type == "tool_call":
            self._flush_text_buffer()
            tool = event.metadata.get("tool", "?")
            args = str(event.metadata.get("args", {}))
            if len(args) > 120:
                args = args[:120] + "..."
            output.write(f"[yellow]> {tool}[/yellow] [dim]{args}[/dim]")
            # Stats
            self._tool_calls += 1
            self._tool_counts[tool] = self._tool_counts.get(tool, 0) + 1
            self._refresh_stats_section()

        elif event.type == "tool_result":
            ok = event.metadata.get("success", False)
            if ok:
                self._tool_ok += 1
            else:
                self._tool_fail += 1
            icon = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
            out = event.data
            if len(out) > 600:
                out = out[:600] + "\n..."
            if out.strip():
                output.write(Panel(out, title=f"{icon} {event.metadata.get('tool', '')}",
                                   border_style="dim", expand=False))
            self._refresh_stats_section()

        elif event.type == "compensation":
            self._flush_text_buffer()
            output.write(f"[magenta]~ {event.data}[/magenta]")
            self._compensations += 1
            self._refresh_stats_section()

        elif event.type == "agent_progress":
            tool = event.metadata.get("tool", "?")
            color_map = {
                "codex": "cyan",
                "claude_code": "magenta",
                "gemini_cli": "yellow",
            }
            color = color_map.get(tool, "white")
            panel = self.query_one("#agent-panel", RichLog)
            panel.write(f"[{color}][{tool}][/{color}] {event.data}")
            # Auto-show panel
            if not panel.display:
                panel.display = True
            # Track active agent
            self._active_agents[tool] = time.monotonic()
            self._refresh_agents_section()

        elif event.type == "agent_done":
            tool = event.metadata.get("tool", "?")
            success = event.metadata.get("success", False)
            icon = "[green]OK[/green]" if success else "[red]FAIL[/red]"
            panel = self.query_one("#agent-panel", RichLog)
            panel.write(f"[bold]{icon} {tool} finished[/bold]")
            self._active_agents.pop(tool, None)
            self._refresh_agents_section()
            # Auto-hide after 3s if no active agents
            if not self._active_agents:
                self._schedule_agent_panel_hide()

        elif event.type == "done":
            self._flush_text_buffer()
            if event.data:
                output.write(Markdown(event.data))

        elif event.type == "summary":
            self._flush_text_buffer()
            output.write("")
            output.write(Panel(
                event.data,
                title="Goal Summary",
                border_style="cyan",
                expand=False,
            ))
            self._refresh_stats_section()

    def _flush_text_buffer(self) -> None:
        if self._text_buffer:
            output = self.query_one("#output", RichLog)
            output.write(Markdown(self._text_buffer))
            self._text_buffer = ""

    def _show_error(self, msg: str) -> None:
        output = self.query_one("#output", RichLog)
        output.write(f"[red]Error: {msg}[/red]")

    def _on_agent_done(self) -> None:
        self._flush_text_buffer()
        self._agent_running = False
        output = self.query_one("#output", RichLog)
        if self._agent_start_time:
            elapsed = time.monotonic() - self._agent_start_time
            output.write(f"[dim]({elapsed:.1f}s)[/dim]\n")
            self._agent_start_time = None

        # Clear sub-agent tracking
        self._active_agents.clear()
        self._refresh_agents_section()

        # Auto-dequeue
        if self._instruction_queue:
            next_input = self._instruction_queue.pop(0)
            self._refresh_queue_section()
            output.write(f"[yellow]Running queued: {next_input[:60]}[/yellow]")
            self._run_input(next_input)

    # ------------------------------------------------------------------
    # Agent panel auto-hide / toggle
    # ------------------------------------------------------------------

    def _schedule_agent_panel_hide(self) -> None:
        if self._agent_panel_timer:
            self._agent_panel_timer.stop()
        self._agent_panel_timer = self.set_timer(3.0, self._hide_agent_panel)

    def _hide_agent_panel(self) -> None:
        if not self._active_agents:
            self.query_one("#agent-panel", RichLog).display = False
        self._agent_panel_timer = None

    # ------------------------------------------------------------------
    # Sidebar section updates
    # ------------------------------------------------------------------

    def _update_plan_from_status(self, status_text: str) -> None:
        """Parse plan-related status events and update the Plan section."""
        plan_widget = self.query_one("#plan-content", Static)

        if status_text.startswith("Plan ("):
            # Full plan summary: "Plan (3 steps):\n  1. Title\n  ..."
            self._plan_text = status_text
            plan_widget.update(self._plan_text)
        elif status_text.startswith("Step ") and "/" in status_text:
            # "Step 2/3: Title" — mark current step
            if self._plan_text:
                lines = self._plan_text.split("\n")
                try:
                    parts = status_text.split("/", 1)
                    current = int(parts[0].replace("Step ", ""))
                    updated: list[str] = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped and stripped[0].isdigit():
                            num = int(stripped.split(".")[0])
                            if num < current:
                                updated.append(f"  [green]{stripped}[/green]")
                            elif num == current:
                                updated.append(f"  [bold yellow]> {stripped}[/bold yellow]")
                            else:
                                updated.append(f"  [dim]{stripped}[/dim]")
                        else:
                            updated.append(line)
                    display_text = "\n".join(updated)
                    plan_widget.update(display_text)
                except (ValueError, IndexError):
                    pass
        elif "finished" in status_text.lower() and "step" in status_text.lower():
            pass  # will be updated on next Step N/M status

    def _refresh_history_section(self) -> None:
        option_list = self.query_one("#history-list", OptionList)
        option_list.clear_options()
        for i, (display, _) in enumerate(self._input_history):
            option_list.add_option(f"{i + 1}. {display}")

    def _refresh_queue_section(self) -> None:
        queue_widget = self.query_one("#queue-content", Static)
        if not self._instruction_queue:
            queue_widget.update("No queued instructions.")
        else:
            lines = ["[bold]Queued instructions:[/bold]"]
            for i, instr in enumerate(self._instruction_queue, 1):
                lines.append(f"  {i}. {instr[:60]}")
            queue_widget.update("\n".join(lines))

    def _refresh_agents_section(self) -> None:
        widget = self.query_one("#agents-content", Static)
        if not self._active_agents:
            widget.update("[dim]No active agents.[/dim]")
            return
        lines = []
        for tool, start in self._active_agents.items():
            elapsed = time.monotonic() - start
            lines.append(f"  [yellow]{tool}[/yellow]  {elapsed:.0f}s")
        widget.update("\n".join(lines))

    def _refresh_tasks_section(self) -> None:
        tasks_widget = self.query_one("#tasks-content", Static)
        tasks = self.task_queue.list_tasks(limit=15)
        if not tasks:
            tasks_widget.update("No background tasks.")
            return
        lines: list[str] = []
        for t in tasks:
            status_map = {
                TaskStatus.QUEUED: "[dim]queued[/dim]",
                TaskStatus.RUNNING: "[yellow]running[/yellow]",
                TaskStatus.SUCCEEDED: "[green]OK[/green]",
                TaskStatus.FAILED: "[red]FAIL[/red]",
                TaskStatus.CANCELED: "[dim]canceled[/dim]",
            }
            elapsed = f"{t.elapsed:.0f}s" if t.elapsed is not None else "-"
            status = status_map.get(t.status, str(t.status))
            lines.append(f"  {t.id}  {status:20s}  {t.goal[:30]:30s}  {elapsed}")
        tasks_widget.update("\n".join(lines))

    def _refresh_stats_section(self) -> None:
        stats_widget = self.query_one("#stats-content", Static)
        elapsed = ""
        if self._agent_start_time:
            elapsed = f"{time.monotonic() - self._agent_start_time:.1f}s"

        tool_names = [t.name for t in self.tools.list_tools()]
        breakdown = ""
        if self._tool_counts:
            top = sorted(self._tool_counts.items(), key=lambda x: -x[1])[:5]
            breakdown = "\n    " + ", ".join(f"{n}: {c}" for n, c in top)

        lines = [
            f"  Mode:     {self.current_mode}",
            f"  Tier:     {self.harness_config.llm.default_tier}",
            f"  Tools:    {len(tool_names)} (OK: {self._tool_ok}, FAIL: {self._tool_fail}){breakdown}",
            f"  Comp:     {self._compensations}",
        ]
        if elapsed:
            lines.append(f"  Elapsed:  {elapsed}")
        stats_widget.update("\n".join(lines))

    def _check_task_notifications(self) -> None:
        """Poll for background task completion notifications."""
        import queue as _queue_mod
        changed = False
        while not _notifications.empty():
            try:
                task = _notifications.get_nowait()
            except _queue_mod.Empty:
                break
            changed = True
            output = self.query_one("#output", RichLog)
            icon = "[green]OK[/green]" if task.status == TaskStatus.SUCCEEDED else "[red]FAIL[/red]"
            output.write(f"\n{icon} Task {task.id} complete: {task.goal[:50]}")
            if task.result_text:
                output.write(f"[dim]{task.result_text[:100]}[/dim]")
            elif task.error_text:
                output.write(f"[red]{task.error_text[:100]}[/red]")
        if changed:
            self._refresh_tasks_section()
