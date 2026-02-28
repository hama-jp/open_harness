"""CLI interface for Open Harness with streaming and goal mode."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from open_harness.agent import Agent, AgentEvent
from open_harness.config import HarnessConfig, load_config
from open_harness.memory.store import MemoryStore
from open_harness.project import ProjectContext
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
# Background goal runner
# ---------------------------------------------------------------------------

class BackgroundGoal:
    """Runs a goal in a background thread with logging."""

    def __init__(self, agent: Agent, goal: str, log_path: Path):
        self.agent = agent
        self.goal = goal
        self.log_path = log_path
        self.thread: threading.Thread | None = None
        self.running = False
        self.completed = False
        self.result = ""

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = True
        self.thread.start()

    def _run(self):
        with open(self.log_path, "w") as f:
            f.write(f"=== Goal: {self.goal} ===\n")
            f.write(f"=== Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            try:
                for event in self.agent.run_goal(self.goal):
                    ts = time.strftime("%H:%M:%S")
                    if event.type == "status":
                        f.write(f"[{ts}] {event.data}\n")
                    elif event.type == "tool_call":
                        f.write(f"[{ts}] TOOL: {event.metadata.get('tool')} {event.metadata.get('args', {})}\n")
                    elif event.type == "tool_result":
                        ok = "OK" if event.metadata.get("success") else "FAIL"
                        f.write(f"[{ts}] RESULT ({ok}): {event.data[:500]}\n")
                    elif event.type == "thinking":
                        f.write(f"[{ts}] THINKING: {event.data[:200]}...\n")
                    elif event.type == "text":
                        f.write(event.data)
                    elif event.type == "compensation":
                        f.write(f"[{ts}] COMPENSATE: {event.data}\n")
                    elif event.type == "done":
                        f.write(f"\n\n=== DONE ===\n{event.data}\n")
                        self.result = event.data
                    f.flush()
            except Exception as e:
                f.write(f"\n\n=== ERROR ===\n{e}\n")
                self.result = f"[Error: {e}]"
            finally:
                self.running = False
                self.completed = True
                f.write(f"\n=== Finished: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

_bg_goals: list[BackgroundGoal] = []


def handle_command(cmd: str, agent: Agent, config: HarnessConfig, display: StreamingDisplay) -> bool | str:
    """Handle /commands. Returns True if handled, 'quit' to exit."""
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
        start = time.monotonic()
        for event in agent.run_goal(arg):
            display.handle(event)
        elapsed = time.monotonic() - start
        console.print(f"\n[dim]Goal completed in {elapsed:.1f}s[/dim]\n")
        return True

    elif command == "/bg":
        if not arg:
            console.print("[red]Usage: /bg <goal>[/red]")
            return True
        log_dir = Path.home() / ".open_harness" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"goal_{int(time.time())}.log"
        bg = BackgroundGoal(agent, arg, log_path)
        bg.start()
        _bg_goals.append(bg)
        console.print(f"[green]Background goal #{len(_bg_goals)} started[/green]")
        console.print(f"[dim]Log: {log_path}[/dim]")
        console.print(f"[dim]Check status: /status[/dim]")
        return True

    elif command == "/status":
        if not _bg_goals:
            console.print("[dim]No background goals.[/dim]")
        for i, bg in enumerate(_bg_goals, 1):
            status = "running" if bg.running else ("done" if bg.completed else "unknown")
            icon = {"running": "[yellow]...[/yellow]", "done": "[green]OK[/green]"}.get(status, "?")
            console.print(f"  #{i} {icon} {bg.goal[:60]}")
            if bg.completed and bg.result:
                console.print(f"      [dim]{bg.result[:100]}[/dim]")
            console.print(f"      [dim]{bg.log_path}[/dim]")
        return True

    elif command == "/help":
        console.print("""
[bold]Interactive:[/bold]
  (type normally)  - Chat with the agent, it will use tools as needed

[bold]Autonomous:[/bold]
  /goal <task>     - Agent works autonomously until task is complete
  /bg <task>       - Run goal in background (check with /status)
  /status          - Check background goal status

[bold]Settings:[/bold]
  /tier [name]     - Show or set model tier (small/medium/large)
  /tools           - List available tools
  /project         - Show detected project context
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
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    console.print(Panel(
        "[bold]Open Harness[/bold] v0.1.0\n"
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

    tools = setup_tools(config, project)
    memory = MemoryStore(config.memory.db_path, max_turns=config.memory.max_conversation_turns)
    agent = Agent(config, tools, memory, project)
    display = StreamingDisplay(console)

    tool_names = [t.name for t in tools.list_tools()]
    console.print(f"[dim]Tools ({len(tool_names)}): {', '.join(tool_names)}[/dim]")
    console.print()

    # Non-interactive goal mode
    if goal_text:
        start = time.monotonic()
        for event in agent.run_goal(goal_text):
            display.handle(event)
        console.print(f"\n[dim]({time.monotonic() - start:.1f}s)[/dim]")
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
                continue

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
        agent.router.close()
        memory.close()


if __name__ == "__main__":
    main()
