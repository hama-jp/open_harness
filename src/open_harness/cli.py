"""CLI interface for Open Harness with streaming and task queue."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from open_harness.agent import Agent, AgentEvent
from open_harness.config import HarnessConfig, load_config
from open_harness.memory.store import MemoryStore
from open_harness.project import ProjectContext
from open_harness.tasks.queue import TaskQueueManager, TaskRecord, TaskStatus, TaskStore
from open_harness.tools.base import ToolRegistry
from open_harness.tools.external import CodexTool, GeminiCliTool
from open_harness.tools.file_ops import (
    EditFileTool, ListDirectoryTool, ReadFileTool, SearchFilesTool, WriteFileTool,
)
from open_harness.tools.git_tools import (
    GitBranchTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool,
)
from open_harness.tools.shell import ShellTool
from open_harness.tools.testing import TestRunnerTool

console = Console()


def setup_tools(config: HarnessConfig, project: ProjectContext) -> ToolRegistry:
    registry = ToolRegistry()

    # Core
    registry.register(ShellTool(config.tools.shell))
    registry.register(ReadFileTool(config.tools.file))
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ListDirectoryTool())
    registry.register(SearchFilesTool())

    # Testing
    test_cmd = project.info.get("test_command") or "python -m pytest"
    registry.register(TestRunnerTool(test_command=test_cmd, cwd=str(project.root)))

    # Git
    if project.info.get("has_git"):
        registry.register(GitStatusTool())
        registry.register(GitDiffTool())
        registry.register(GitCommitTool())
        registry.register(GitBranchTool())
        registry.register(GitLogTool())

    # External agents
    for name, tool_cls, flag_key in [
        ("codex", CodexTool, "codex"),
        ("gemini", GeminiCliTool, "gemini"),
    ]:
        ext_cfg = config.external_agents.get(flag_key)
        if ext_cfg and ext_cfg.enabled:
            tool = tool_cls(ext_cfg.command)
            if tool.available:
                registry.register(tool)

    return registry


def create_agent_factory(config: HarnessConfig, project: ProjectContext):
    """Factory that creates isolated Agent instances for background tasks."""
    def factory() -> Agent:
        tools = setup_tools(config, project)
        memory = MemoryStore(
            config.memory.db_path,
            max_turns=config.memory.max_conversation_turns,
        )
        return Agent(config, tools, memory, project)
    return factory


class StreamingDisplay:
    """Renders agent events to the terminal in real time."""

    def __init__(self, con: Console):
        self.con = con
        self._streaming = False

    def handle(self, event: AgentEvent):
        if event.type == "status":
            self.con.print(f"[dim]{event.data}[/dim]", end="\r")

        elif event.type == "thinking":
            first = event.data.split("\n")[0][:80]
            self.con.print(f"[dim italic]thinking: {first}...[/dim italic]")

        elif event.type == "text":
            if not self._streaming:
                self._streaming = True
                self.con.print()
            self.con.print(event.data, end="", highlight=False)

        elif event.type == "tool_call":
            self._flush()
            tool = event.metadata.get("tool", "?")
            args = str(event.metadata.get("args", {}))
            if len(args) > 120:
                args = args[:120] + "..."
            self.con.print(f"[yellow]> {tool}[/yellow] [dim]{args}[/dim]")

        elif event.type == "tool_result":
            ok = event.metadata.get("success", False)
            icon = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
            out = event.data
            if len(out) > 600:
                out = out[:600] + "\n..."
            if out.strip():
                self.con.print(Panel(out, title=f"{icon} {event.metadata.get('tool', '')}",
                                     border_style="dim", expand=False))

        elif event.type == "compensation":
            self._flush()
            self.con.print(f"[magenta]~ {event.data}[/magenta]")

        elif event.type == "done":
            if self._streaming:
                self.con.print()
                self._streaming = False
            elif event.data:
                self.con.print()
                self.con.print(Markdown(event.data))

    def _flush(self):
        if self._streaming:
            self.con.print()
            self._streaming = False


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

import queue as _queue_mod

# Module-level task queue (set up in main)
_task_queue: TaskQueueManager | None = None
# Thread-safe notification queue to avoid interleaving console output
_notifications: _queue_mod.Queue[TaskRecord] = _queue_mod.Queue()


def _on_task_complete(task: TaskRecord):
    """Callback when a background task finishes (called from worker thread)."""
    _notifications.put(task)
    # Terminal bell (safe from any thread)
    print("\a", end="", flush=True)


def _drain_notifications():
    """Show any pending task completion notifications (called from main thread)."""
    while not _notifications.empty():
        try:
            task = _notifications.get_nowait()
        except _queue_mod.Empty:
            break
        icon = "[green]OK[/green]" if task.status == TaskStatus.SUCCEEDED else "[red]FAIL[/red]"
        console.print(f"\n{icon} Task {task.id} complete: {task.goal[:50]}")
        if task.result_text:
            console.print(f"[dim]{task.result_text[:100]}[/dim]")
        elif task.error_text:
            console.print(f"[red]{task.error_text[:100]}[/red]")


def handle_command(cmd: str, agent: Agent, config: HarnessConfig, display: StreamingDisplay) -> bool | str:
    """Handle /commands. Returns True if handled, 'quit' to exit."""
    global _task_queue
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        return "quit"

    elif command == "/clear":
        agent.memory.clear_conversation()
        console.print("[dim]Conversation cleared.[/dim]")
        return True

    elif command == "/tier":
        if arg:
            agent.router.current_tier = arg
            console.print(f"[dim]Tier: {agent.router.current_tier}[/dim]")
        else:
            for name, desc in agent.router.list_tiers().items():
                m = " *" if name == agent.router.current_tier else ""
                console.print(f"  {name}: {desc}{m}")
        return True

    elif command == "/tools":
        for tool in agent.tools.list_tools():
            console.print(f"  [bold]{tool.name}[/bold]: {tool.description[:80]}")
        return True

    elif command == "/project":
        console.print(Panel(agent.project.to_prompt(), title="Project Context", border_style="blue"))
        return True

    elif command == "/goal":
        if not arg:
            console.print("[red]Usage: /goal <description of what you want to achieve>[/red]")
            return True
        # Warn if background task is running (LLM concurrency)
        if _task_queue and _task_queue.is_busy():
            console.print("[yellow]Warning: a background task is running. "
                          "LLM requests may queue.[/yellow]")
        start = time.monotonic()
        for event in agent.run_goal(arg):
            display.handle(event)
        elapsed = time.monotonic() - start
        console.print(f"\n[dim]Goal completed in {elapsed:.1f}s[/dim]\n")
        return True

    elif command in ("/submit", "/bg"):
        if not arg:
            console.print("[red]Usage: /submit <goal>[/red]")
            return True
        if not _task_queue:
            console.print("[red]Task queue not initialized.[/red]")
            return True
        task = _task_queue.submit(arg)
        console.print(f"[green]Task {task.id} queued: {task.goal[:60]}[/green]")
        console.print(f"[dim]Log: {task.log_path}[/dim]")
        console.print(f"[dim]Check: /tasks | /result {task.id}[/dim]")
        return True

    elif command in ("/tasks", "/status"):
        if not _task_queue:
            console.print("[dim]Task queue not initialized.[/dim]")
            return True
        tasks = _task_queue.list_tasks(limit=15)
        if not tasks:
            console.print("[dim]No tasks submitted yet. Use /submit <goal>[/dim]")
            return True

        table = Table(title="Tasks", show_lines=False, border_style="dim")
        table.add_column("ID", style="bold", width=10)
        table.add_column("Status", width=10)
        table.add_column("Goal", max_width=50)
        table.add_column("Time", width=8)

        status_styles = {
            TaskStatus.QUEUED: "[dim]queued[/dim]",
            TaskStatus.RUNNING: "[yellow]running[/yellow]",
            TaskStatus.SUCCEEDED: "[green]OK[/green]",
            TaskStatus.FAILED: "[red]FAIL[/red]",
            TaskStatus.CANCELED: "[dim]canceled[/dim]",
        }

        for t in tasks:
            elapsed = ""
            if t.elapsed is not None:
                elapsed = f"{t.elapsed:.0f}s"
            table.add_row(
                t.id,
                status_styles.get(t.status, str(t.status)),
                t.goal[:50],
                elapsed,
            )
        console.print(table)
        return True

    elif command == "/result":
        if not _task_queue:
            console.print("[dim]Task queue not initialized.[/dim]")
            return True
        if not arg:
            console.print("[red]Usage: /result <task_id>[/red]")
            return True
        task = _task_queue.get_task(arg)
        if not task:
            console.print(f"[red]Task '{arg}' not found.[/red]")
            return True
        console.print(f"[bold]Task {task.id}[/bold]: {task.goal}")
        console.print(f"Status: {task.status.value}")
        if task.elapsed is not None:
            console.print(f"[dim]Time: {task.elapsed:.1f}s[/dim]")
        if task.result_text:
            console.print(Panel(task.result_text, title="Result", border_style="green"))
        if task.error_text:
            console.print(Panel(task.error_text, title="Error", border_style="red"))
        if task.log_path:
            console.print(f"[dim]Log: {task.log_path}[/dim]")
        return True

    elif command == "/policy":
        p = agent.policy.config
        console.print(f"[bold]Policy mode:[/bold] {p.mode}")
        budgets = []
        if p.max_file_writes:
            budgets.append(f"file writes: {p.max_file_writes}")
        if p.max_shell_commands:
            budgets.append(f"shell commands: {p.max_shell_commands}")
        if p.max_git_commits:
            budgets.append(f"git commits: {p.max_git_commits}")
        if p.max_external_calls:
            budgets.append(f"external calls: {p.max_external_calls}")
        if budgets:
            console.print(f"[dim]Budgets: {', '.join(budgets)}[/dim]")
        else:
            console.print("[dim]Budgets: unlimited[/dim]")
        if p.denied_paths:
            console.print(f"[dim]Denied paths: {len(p.denied_paths)} patterns[/dim]")
        if p.blocked_shell_patterns:
            console.print(f"[dim]Blocked shell: {len(p.blocked_shell_patterns)} patterns[/dim]")
        if p.disabled_tools:
            console.print(f"[dim]Disabled tools: {', '.join(p.disabled_tools)}[/dim]")
        if arg and arg in ("safe", "balanced", "full"):
            from open_harness.policy import PRESETS, PolicyConfig
            agent.policy.config = PolicyConfig(**PRESETS[arg])
            console.print(f"[green]Policy switched to: {arg}[/green]")
        return True

    elif command == "/memory":
        memories = agent.project_memory_store.get_memories(
            str(agent.project.root), limit=20)
        runbooks = agent.project_memory_store.get_runbooks(
            str(agent.project.root), limit=5)
        if not memories and not runbooks:
            console.print("[dim]No project memories yet. Use /goal to start learning.[/dim]")
        else:
            if memories:
                console.print("[bold]Learned memories:[/bold]")
                for m in memories:
                    icon = {"pattern": "P", "structure": "S", "error": "E", "runbook": "R"}.get(m.kind, "?")
                    pin = " [yellow]*[/yellow]" if m.pinned else ""
                    console.print(f"  [{icon}] {m.value} [dim](score:{m.score:.1f} seen:{m.seen_count}){pin}[/dim]")
            if runbooks:
                console.print("[bold]Runbooks:[/bold]")
                for rb in runbooks:
                    console.print(f"  [bold]{rb.title}[/bold] ({rb.usage_count} uses, {rb.success_count} ok)")
                    for j, step in enumerate(rb.steps[:5], 1):
                        console.print(f"    {j}. {step}")
        return True

    elif command == "/help":
        console.print("""
[bold]Interactive:[/bold]
  (type normally)  - Chat with the agent, it will use tools as needed

[bold]Autonomous:[/bold]
  /goal <task>     - Agent works autonomously until task is complete
  /submit <task>   - Submit goal to background queue
  /tasks           - List all tasks and their status
  /result <id>     - Show detailed result of a task

[bold]Settings:[/bold]
  /tier [name]     - Show or set model tier (small/medium/large)
  /policy [mode]   - Show or set policy (safe/balanced/full)
  /tools           - List available tools
  /project         - Show detected project context
  /memory          - Show learned project memories
  /clear           - Clear conversation
  /quit            - Exit
        """)
        return True

    return False


@click.command()
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
@click.option("--tier", "-t", default=None, help="Model tier")
@click.option("--goal", "-g", "goal_text", default=None, help="Run a goal non-interactively and exit")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(config_path: str | None, tier: str | None, goal_text: str | None, verbose: bool):
    """Open Harness - self-driving AI agent for local LLMs."""
    global _task_queue

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    console.print(Panel(
        "[bold]Open Harness[/bold] v0.2.0\n"
        "Self-driving AI agent for local LLMs\n"
        "[dim]Type /help for commands, /goal <task> for autonomous mode[/dim]",
        border_style="blue",
    ))

    config = load_config(config_path)
    if tier:
        config.llm.default_tier = tier

    # Detect project
    project = ProjectContext()
    pinfo = project.info
    console.print(f"[dim]Project: {pinfo['type']} @ {pinfo['root']}[/dim]")
    if pinfo.get("test_command"):
        console.print(f"[dim]Tests: {pinfo['test_command']}[/dim]")

    try:
        model_cfg = config.llm.models.get(config.llm.default_tier)
        if model_cfg:
            console.print(f"[dim]Model: {model_cfg.model} ({config.llm.default_tier})[/dim]")
    except Exception:
        pass

    # Main agent for interactive use
    tools = setup_tools(config, project)
    memory = MemoryStore(config.memory.db_path, max_turns=config.memory.max_conversation_turns)
    agent = Agent(config, tools, memory, project)
    display = StreamingDisplay(console)

    tool_names = [t.name for t in tools.list_tools()]
    console.print(f"[dim]Tools ({len(tool_names)}): {', '.join(tool_names)}[/dim]")

    # Task queue with isolated agent factory
    task_store = TaskStore(config.memory.db_path)
    agent_factory = create_agent_factory(config, project)
    _task_queue = TaskQueueManager(
        task_store, agent_factory, on_complete=_on_task_complete)
    _task_queue.start()
    console.print(f"[dim]Task queue: ready[/dim]")
    console.print()

    # Non-interactive goal mode
    if goal_text:
        start = time.monotonic()
        for event in agent.run_goal(goal_text):
            display.handle(event)
        console.print(f"\n[dim]({time.monotonic() - start:.1f}s)[/dim]")
        _task_queue.shutdown()
        task_store.close()
        agent.router.close()
        memory.close()
        return

    # Interactive REPL
    try:
        while True:
            try:
                user_input = console.input("[bold green]> [/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                _drain_notifications()
                continue

            _drain_notifications()

            if user_input.startswith("/"):
                result = handle_command(user_input, agent, config, display)
                if result == "quit":
                    console.print("[dim]Goodbye![/dim]")
                    break
                if result:
                    continue

            try:
                start = time.monotonic()
                for event in agent.run_stream(user_input):
                    display.handle(event)
                console.print(f"\n[dim]({time.monotonic() - start:.1f}s)[/dim]\n")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if verbose:
                    console.print_exception()
    finally:
        _task_queue.shutdown()
        task_store.close()
        agent.router.close()
        memory.close()


if __name__ == "__main__":
    main()
