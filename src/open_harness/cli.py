"""CLI interface for Open Harness with streaming and task queue."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from open_harness.agent import Agent, AgentEvent
from open_harness.completer import AtFileCompleter
from open_harness.config import HarnessConfig, load_config
from open_harness.memory.store import MemoryStore
from open_harness.project import ProjectContext
from open_harness.tasks.queue import TaskQueueManager, TaskRecord, TaskStatus, TaskStore
from open_harness.tools.base import ToolRegistry
from open_harness.tools.external import ClaudeCodeTool, CodexTool, GeminiCliTool
from open_harness.tools.file_ops import (
    EditFileTool, ListDirectoryTool, ReadFileTool, SearchFilesTool, WriteFileTool,
)
from open_harness.tools.git_tools import (
    GitBranchTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool,
)
from open_harness.tools.shell import ShellTool
from open_harness.tools.testing import TestRunnerTool

console = Console()

# Repository root: <repo>/src/open_harness/cli.py → <repo>
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_version() -> str:
    """Read version from pyproject.toml."""
    toml = _REPO_ROOT / "pyproject.toml"
    if toml.exists():
        for line in toml.read_text().splitlines():
            if line.strip().startswith("version"):
                # version = "0.2.0"
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "unknown"


def self_update() -> bool:
    """Pull latest code from git and reinstall the package.

    Returns True if an update was applied, False if already up-to-date.
    Exits the process on failure so the caller doesn't continue with stale code.
    """
    repo = _REPO_ROOT
    git_dir = repo / ".git"
    if not git_dir.is_dir():
        console.print(f"[red]Not a git repository: {repo}[/red]")
        return False

    current_ver = get_version()
    console.print(f"[dim]Current version: v{current_ver}[/dim]")
    console.print(f"[dim]Updating from {repo} ...[/dim]")

    # 1. git fetch + check
    fetch = subprocess.run(
        ["git", "fetch"], cwd=repo, capture_output=True, text=True, timeout=30)
    if fetch.returncode != 0:
        console.print(f"[red]git fetch failed: {fetch.stderr.strip()}[/red]")
        return False

    # Compare local HEAD vs remote
    local = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    remote = subprocess.run(
        ["git", "rev-parse", "@{u}"], cwd=repo, capture_output=True, text=True)

    if remote.returncode != 0:
        console.print("[yellow]No upstream branch configured. Trying git pull anyway.[/yellow]")
    elif local.stdout.strip() == remote.stdout.strip():
        console.print(f"[green]Already up-to-date (v{current_ver}).[/green]")
        return False

    # 2. git pull
    console.print("[dim]git pull ...[/dim]")
    pull = subprocess.run(
        ["git", "pull", "--ff-only"], cwd=repo, capture_output=True, text=True, timeout=60)
    if pull.returncode != 0:
        console.print(f"[red]git pull failed: {pull.stderr.strip()}[/red]")
        console.print("[dim]Hint: commit or stash local changes first.[/dim]")
        return False

    # Show what changed
    if pull.stdout.strip():
        console.print(f"[dim]{pull.stdout.strip()}[/dim]")

    # 3. reinstall package (prefer uv, fall back to pip)
    if shutil.which("uv"):
        install_cmd = ["uv", "pip", "install", "-e", str(repo), "-q"]
        install_label = "uv pip install -e ."
    else:
        install_cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo), "-q"]
        install_label = "pip install -e ."
    console.print(f"[dim]{install_label} ...[/dim]")
    pip = subprocess.run(
        install_cmd, cwd=repo, capture_output=True, text=True, timeout=120)
    if pip.returncode != 0:
        console.print(f"[yellow]{install_label} warning: {pip.stderr.strip()[:200]}[/yellow]")

    new_ver = get_version()
    console.print(f"[green]Updated: v{current_ver} -> v{new_ver}[/green]")
    console.print("[dim]Please restart harness to apply changes.[/dim]")
    return True


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
        ("claude_code", ClaudeCodeTool, "claude"),
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

        elif event.type == "summary":
            self._flush()
            self.con.print()
            self.con.print(Panel(
                event.data,
                title="Goal Summary",
                border_style="cyan",
                expand=False,
            ))

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


def _list_dir_tree(path: Path, max_depth: int = 2, _prefix: str = "", _depth: int = 0) -> str:
    """Return a simple tree listing of a directory."""
    if _depth >= max_depth:
        return ""
    lines: list[str] = []
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return f"{_prefix}(permission denied)\n"
    for entry in entries:
        if entry.name.startswith(".") or entry.name in ("__pycache__", "node_modules", ".venv"):
            continue
        if entry.is_dir():
            lines.append(f"{_prefix}{entry.name}/")
            lines.append(_list_dir_tree(entry, max_depth, _prefix + "  ", _depth + 1))
        else:
            lines.append(f"{_prefix}{entry.name}")
    return "\n".join(lines)


def _expand_at_references(text: str, project_root: Path, max_size: int = 50_000) -> str:
    """Expand @path references into inline file content."""
    import re

    pattern = r"(?<![a-zA-Z0-9])@([\w./\-_]+)"
    refs = re.findall(pattern, text)
    if not refs:
        return text

    attachments: list[str] = []
    clean_text = text
    seen: set[str] = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        p = (project_root / ref).resolve()
        # Security: don't allow traversal outside project root
        try:
            p.relative_to(project_root)
        except ValueError:
            continue
        if p.is_file():
            try:
                content = p.read_text(errors="replace")[:max_size]
            except OSError:
                continue
            attachments.append(f"[File: {ref}]\n```\n{content}\n```")
            clean_text = clean_text.replace(f"@{ref}", f"`{ref}`", 1)
        elif p.is_dir():
            listing = _list_dir_tree(p, max_depth=2)
            attachments.append(f"[Directory: {ref}]\n{listing}")
            clean_text = clean_text.replace(f"@{ref}", f"`{ref}/`", 1)

    if attachments:
        return clean_text + "\n\n" + "\n\n".join(attachments)
    return text


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

    elif command == "/model":
        from urllib.parse import urlparse

        def _tier_line(tier_name: str, model_cfg, is_current: bool) -> str:
            provider_cfg = config.llm.providers.get(model_cfg.provider)
            location = ""
            if provider_cfg:
                host = urlparse(provider_cfg.base_url).hostname or ""
                if host in ("localhost", "127.0.0.1", "::1"):
                    location = " (local)"
                else:
                    location = f" ({host})"
            mark = "  *" if is_current else ""
            return (
                f"  {tier_name}:  {model_cfg.model} "
                f"@ {model_cfg.provider}{location} "
                f"max_tokens={model_cfg.max_tokens}{mark}"
            )

        if arg:
            prev = agent.router.current_tier
            agent.router.current_tier = arg
            if agent.router.current_tier == prev and arg != prev:
                console.print(f"[red]Unknown tier: {arg}[/red]")
            else:
                mcfg = agent.router.get_model_config()
                console.print(_tier_line(agent.router.current_tier, mcfg, True))
        else:
            console.print("[bold]Model tiers:[/bold]")
            for name, _mcfg in config.llm.models.items():
                console.print(_tier_line(name, _mcfg, name == agent.router.current_tier))
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
        arg = _expand_at_references(arg, agent.project.root)
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
        arg = _expand_at_references(arg, agent.project.root)
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

    elif command == "/update":
        self_update()
        return True

    elif command == "/help":
        console.print("""
[bold]Modes (Shift+Tab to cycle):[/bold]
  chat    [green]>[/green]       - Chat with the agent, it will use tools as needed
  goal    [yellow]goal>[/yellow]  - Agent works autonomously until task is complete
  submit  [blue]bg>[/blue]    - Submit goal to background queue

[bold]Autonomous:[/bold]
  /goal <task>     - Run a goal regardless of current mode
  /submit <task>   - Submit to background regardless of current mode
  /tasks           - List all tasks and their status
  /result <id>     - Show detailed result of a task

[bold]File references:[/bold]
  @path/to/file    - Attach file content (Tab to complete)

[bold]Settings:[/bold]
  /model [tier]    - Show all model tiers with details, or switch tier
  /tier [name]     - Show or set model tier (small/medium/large)
  /policy [mode]   - Show or set policy (safe/balanced/full)
  /tools           - List available tools
  /project         - Show detected project context
  /memory          - Show learned project memories
  /update          - Update Open Harness to latest version
  /clear           - Clear conversation
  /quit            - Exit
        """)
        return True

    return False


@click.command()
@click.option("--config", "-c", "config_path", default=None,
              help="Path to open_harness.yaml (auto-detected from CWD, ~/.open_harness/, or repo root)")
@click.option("--tier", "-t", default=None, help="Model tier")
@click.option("--goal", "-g", "goal_text", default=None, help="Run a goal non-interactively and exit")
@click.option("--update", "-u", "do_update", is_flag=True, help="Update Open Harness to latest version and exit")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(config_path: str | None, tier: str | None, goal_text: str | None,
         do_update: bool, verbose: bool):
    """Open Harness - self-driving AI agent for local LLMs."""
    global _task_queue

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if do_update:
        self_update()
        return

    version = get_version()

    # Gradient ASCII art banner
    _banner_lines = [
        r"   ____                      _   _                                ",
        r"  / __ \                    | | | |                               ",
        r" | |  | |_ __   ___ _ __   | |_| | __ _ _ __ _ __   ___  ___ ___ ",
        r" | |  | | '_ \ / _ \ '_ \  |  _  |/ _` | '__| '_ \ / _ \/ __/ __|",
        r" | |__| | |_) |  __/ | | | | | | | (_| | |  | | | |  __/\__ \__ \\",
        r"  \____/| .__/ \___|_| |_| |_| |_|\__,_|_|  |_| |_|\___||___/___/",
        r"        | |                                                       ",
        r"        |_|                                                       ",
    ]
    # Blue-cyan gradient using Rich markup
    _gradient = [
        "bold bright_blue",
        "bold blue",
        "bold cyan",
        "bold bright_cyan",
        "bold cyan",
        "bold blue",
        "bold bright_blue",
        "bold blue",
    ]
    for line, style in zip(_banner_lines, _gradient):
        console.print(f"[{style}]{line}[/{style}]")
    console.print(
        f"  [bold bright_white]v{version}[/bold bright_white]"
        "  [dim]Self-driving AI agent for local LLMs[/dim]"
    )
    console.print(
        "  [dim]Type /help for commands, /goal <task> for autonomous mode[/dim]\n"
    )

    config, config_file = load_config(config_path)
    if config_file:
        console.print(f"[dim]Config: {config_file}[/dim]")
    else:
        console.print("[dim]Config: defaults (no open_harness.yaml found)[/dim]")
    if tier:
        config.llm.default_tier = tier

    # Detect project
    project = ProjectContext()
    pinfo = project.info
    console.print(f"[dim]Project: {pinfo['type']} @ {pinfo['root']}[/dim]")
    if pinfo.get("test_command"):
        console.print(f"[dim]Tests: {pinfo['test_command']}[/dim]")

    # Ensure project is under git so file edits can be reverted
    git_status = project.ensure_git()
    if git_status == "auto-initialized git":
        console.print(f"[yellow]Git: auto-initialized (to allow safe file editing)[/yellow]")
    else:
        console.print(f"[dim]Git: {git_status}[/dim]")

    try:
        model_cfg = config.llm.models.get(config.llm.default_tier)
        if model_cfg:
            provider_cfg = config.llm.providers.get(model_cfg.provider)
            location = ""
            if provider_cfg:
                from urllib.parse import urlparse
                host = urlparse(provider_cfg.base_url).hostname or ""
                if host in ("localhost", "127.0.0.1", "::1"):
                    location = " (local)"
                else:
                    location = f" ({host})"
            console.print(
                f"[dim]Model: {model_cfg.model} ({config.llm.default_tier})"
                f" @ {model_cfg.provider}{location}[/dim]"
            )
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

    # Interactive REPL with mode cycling (Shift+Tab)
    _modes = ["chat", "goal", "submit"]
    _mode_index = [0]  # list to allow mutation in closure

    _mode_colors = {"chat": "ansigreen", "goal": "ansiyellow", "submit": "ansiblue"}
    _mode_labels = {
        "chat": "chat",
        "goal": "goal (autonomous)",
        "submit": "background",
    }

    def _get_prompt():
        mode = _modes[_mode_index[0]]
        color = _mode_colors[mode]
        cols = shutil.get_terminal_size().columns
        line = "─" * cols
        return HTML(f"<dim>{line}</dim>\n<{color}><b>❯ </b></{color}>")

    def _get_toolbar():
        mode = _modes[_mode_index[0]]
        cols = shutil.get_terminal_size().columns
        line = "─" * cols
        color = _mode_colors[mode]
        return HTML(
            f"<dim>{line}</dim>\n"
            f"  <{color}><b>⏵⏵ {_mode_labels[mode]}</b></{color}>"
            f"  <dim>(shift+tab to cycle)</dim>"
        )

    kb = KeyBindings()

    @kb.add("s-tab")
    def _cycle_mode(event):
        _mode_index[0] = (_mode_index[0] + 1) % len(_modes)

    _pt_style = Style.from_dict({"bottom-toolbar": "noreverse"})
    history_path = Path(os.path.expanduser("~/.open_harness/history"))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    completer = AtFileCompleter(project.root)
    session = PromptSession(
        completer=completer,
        complete_while_typing=False,
        key_bindings=kb, bottom_toolbar=_get_toolbar, style=_pt_style,
        history=FileHistory(str(history_path)),
    )

    try:
        while True:
            try:
                user_input = session.prompt(_get_prompt).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                _drain_notifications()
                continue

            _drain_notifications()

            # Slash commands work in any mode
            if user_input.startswith("/"):
                result = handle_command(user_input, agent, config, display)
                if result == "quit":
                    console.print("[dim]Goodbye![/dim]")
                    break
                if result:
                    continue

            # Expand @file references
            user_input = _expand_at_references(user_input, project.root)

            mode = _modes[_mode_index[0]]
            try:
                start = time.monotonic()

                if mode == "chat":
                    for event in agent.run_stream(user_input):
                        display.handle(event)
                    console.print(f"\n[dim]({time.monotonic() - start:.1f}s)[/dim]\n")

                elif mode == "goal":
                    if _task_queue and _task_queue.is_busy():
                        console.print("[yellow]Warning: a background task is running. "
                                      "LLM requests may queue.[/yellow]")
                    for event in agent.run_goal(user_input):
                        display.handle(event)
                    console.print(f"\n[dim]Goal completed in {time.monotonic() - start:.1f}s[/dim]\n")

                elif mode == "submit":
                    if not _task_queue:
                        console.print("[red]Task queue not initialized.[/red]")
                        continue
                    task = _task_queue.submit(user_input)
                    console.print(f"[green]Task {task.id} queued: {task.goal[:60]}[/green]")
                    console.print(f"[dim]Log: {task.log_path}[/dim]")
                    console.print(f"[dim]Check: /tasks | /result {task.id}[/dim]")

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if verbose:
                    console.print_exception()
    finally:
        agent.close()  # finish session checkpoint (merge git changes)
        if _task_queue:
            _task_queue.shutdown()
        task_store.close()
        agent.router.close()
        memory.close()


if __name__ == "__main__":
    main()
