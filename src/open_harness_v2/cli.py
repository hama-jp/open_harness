"""CLI entry point for Open Harness v2.

Usage::

    harness                          # interactive REPL
    harness "Fix the bug"            # one-shot (positional argument)
    harness --config path.yaml       # config file
    harness --profile api            # profile switch
    harness -v                       # verbose event log
    harness update                   # self-update via git pull
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from open_harness_v2 import __version__
from open_harness_v2.config import HarnessConfig, _SEARCH_PATHS, load_config
from open_harness_v2.core.orchestrator import Orchestrator
from open_harness_v2.events.bus import EventBus
from open_harness_v2.hooks import HookEngine
from open_harness_v2.hooks.engine import load_hooks
from open_harness_v2.llm.error_recovery import ErrorRecoveryMiddleware
from open_harness_v2.llm.middleware import MiddlewarePipeline
from open_harness_v2.llm.prompt_optimizer import PromptOptimizerMiddleware
from open_harness_v2.llm.router import ModelRouter
from open_harness_v2.checkpoint import CheckpointEngine
from open_harness_v2.memory import MemoryStore, ProjectMemory, SessionMemory
from open_harness_v2.policy.engine import PolicyEngine
from open_harness_v2.project_instructions import load_project_instructions
from open_harness_v2.skills import SkillRegistry
from open_harness_v2.tasks import TaskManager, TaskStore
from open_harness_v2.todo import TodoManager
from open_harness_v2.tools.builtin import register_builtins
from open_harness_v2.tools.registry import ToolRegistry
from open_harness_v2.ui.renderer import ConsoleRenderer

_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _config_exists() -> bool:
    """Return True if a config file is found in default search paths."""
    return any(p.exists() for p in _SEARCH_PATHS)


def _find_config_display(config_path: str | None) -> str:
    """Return a human-readable config source string."""
    if config_path:
        return config_path
    for p in _SEARCH_PATHS:
        if p.exists():
            return str(p)
    return "defaults"


def _project_hash() -> str:
    """Derive a stable project ID from the working directory."""
    return hashlib.sha256(os.getcwd().encode()).hexdigest()[:12]


def _print_banner(
    console: Console,
    config: HarnessConfig,
    config_path: str | None,
    num_tools: int,
    num_skills: int = 0,
    has_hooks: bool = False,
    has_instructions: bool = False,
) -> None:
    """Print a compact startup banner with status info."""
    from urllib.parse import urlparse

    prof = config.active_profile
    host = urlparse(prof.url).hostname or "unknown"
    if host in ("localhost", "127.0.0.1", "::1"):
        location = "local"
    else:
        location = host

    # ASCII art banner with blue-cyan gradient
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

    console.print()
    for line, style in zip(_banner_lines, _gradient):
        console.print(f"[{style}]{line}[/{style}]")
    console.print(
        f"  [bold bright_white]v{__version__}[/bold bright_white]"
        "  [dim]Async-first AI agent harness[/dim]"
    )
    console.print("[dim]─[/dim]" * min(console.width, 50))

    # Status grid — two columns, compact
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", min_width=8)
    grid.add_column(min_width=20)
    grid.add_column(style="dim", min_width=8)
    grid.add_column(min_width=16)

    grid.add_row(
        "model",
        f"[bold]{prof.models[0]}[/bold]" if prof.models else "[dim]none[/dim]",
        "policy",
        config.policy.mode,
    )
    grid.add_row(
        "provider",
        f"{prof.provider} [dim]({location})[/dim]",
        "tools",
        str(num_tools),
    )
    grid.add_row(
        "profile",
        config.profile,
        "config",
        f"[dim]{_find_config_display(config_path)}[/dim]",
    )

    # Show skills, hooks, and project instructions status
    extras: list[str] = []
    if num_skills > 0:
        extras.append(f"[bold green]{num_skills} skills[/bold green]")
    if has_hooks:
        extras.append("[bold green]hooks[/bold green]")
    if has_instructions:
        extras.append("[bold green]HARNESS.md[/bold green]")
    if extras:
        grid.add_row("extras", ", ".join(extras), "", "")

    # Detect available external agents (quick shutil.which check only;
    # actually running --version is slow and blocks startup).
    import shutil

    _ext_agents = [
        ("claude", "Claude Code"),
        ("codex", "Codex"),
        ("gemini", "Gemini CLI"),
    ]
    available: list[str] = []
    for cmd, label in _ext_agents:
        if shutil.which(cmd) is not None:
            available.append(label)
    if available:
        agents_str = "[bold green]" + "[/bold green], [bold green]".join(available) + "[/bold green]"
    else:
        agents_str = "[dim]none[/dim]"
    grid.add_row("agents", agents_str, "", "")

    console.print(grid)
    console.print()


# ------------------------------------------------------------------
# Component wiring
# ------------------------------------------------------------------

def _build_components(
    config_path: str | None,
    profile: str | None,
    verbose: bool,
) -> tuple[Orchestrator, EventBus, ConsoleRenderer, Console, HarnessConfig]:
    """Wire up all components and return the orchestrator + UI pieces."""
    config = load_config(config_path)

    # Override profile if specified on CLI
    if profile:
        config.profile = profile

    # Auto-start Ollama if the active profile uses it
    if config.active_profile.api_type == "ollama":
        from open_harness_v2.setup import ensure_ollama

        native_url = config.active_profile.url.rstrip("/").removesuffix("/v1")
        ensure_ollama(native_url)

    # Memory
    store = MemoryStore()
    project_memory = ProjectMemory(store, project_id=_project_hash())
    session_memory = SessionMemory(store)

    # Core components
    router = ModelRouter(config)
    registry = ToolRegistry()
    register_builtins(registry, project_memory=project_memory)
    registry.discover()

    policy = PolicyEngine(config.policy)
    policy.set_project_root(Path.cwd())
    event_bus = EventBus()

    # Wire memory to EventBus
    session_memory.attach(event_bus)

    # Skills — discover from builtin, user, and project dirs
    skill_registry = SkillRegistry()
    skill_registry.discover(project_root=Path.cwd())

    # Hooks — merge config-level hooks with project-level .harness/hooks.yaml
    hooks_config = config.hooks
    if hooks_config is None:
        hooks_config = load_hooks(project_root=Path.cwd())
    hook_engine = HookEngine(hooks_config)
    hook_engine.attach(event_bus)

    # Project instructions (HARNESS.md)
    project_instructions = load_project_instructions(project_root=Path.cwd())

    # Todo manager (session-scoped task tracking)
    todo_manager = TodoManager()

    # UI
    console = Console()
    renderer = ConsoleRenderer(console, verbose=verbose)
    renderer.attach(event_bus)

    # LLM middleware pipeline
    pipeline = MiddlewarePipeline(router.get_client())
    pipeline.use(PromptOptimizerMiddleware())
    pipeline.use(ErrorRecoveryMiddleware(
        tool_names=[t.name for t in registry.list_tools()],
        on_escalate=lambda model, req: (
            router.escalate() and router.current_model or model
        ),
    ))

    # Orchestrator
    orchestrator = Orchestrator(
        router=router,
        registry=registry,
        policy=policy,
        event_bus=event_bus,
        pipeline=pipeline,
        max_steps=config.max_steps,
    )

    # Checkpoint — EventBus subscriber, only active in git repos with non-full policy
    checkpoint = CheckpointEngine(Path.cwd(), config.policy)
    checkpoint.attach(event_bus)

    # Clean up orphan branches from crashed sessions
    CheckpointEngine.cleanup_orphan_branches(Path.cwd())

    # Task queue — factory creates fresh orchestrators for background tasks
    task_store = TaskStore()

    def _orchestrator_factory() -> Orchestrator:
        task_policy = PolicyEngine(config.policy)
        task_policy.set_project_root(Path.cwd())
        task_pipeline = MiddlewarePipeline(router.get_client())
        task_pipeline.use(PromptOptimizerMiddleware())
        task_pipeline.use(ErrorRecoveryMiddleware(
            tool_names=[t.name for t in registry.list_tools()],
            on_escalate=lambda model, req: (
                router.escalate() and router.current_model or model
            ),
        ))
        return Orchestrator(
            router=router,
            registry=registry,
            policy=task_policy,
            event_bus=event_bus,
            pipeline=task_pipeline,
            max_steps=config.max_steps,
        )

    task_manager = TaskManager(task_store, event_bus, _orchestrator_factory)

    # Attach references for CLI access
    orchestrator._project_memory = project_memory  # type: ignore[attr-defined]
    orchestrator._session_memory = session_memory  # type: ignore[attr-defined]
    orchestrator._memory_store = store  # type: ignore[attr-defined]
    orchestrator._task_manager = task_manager  # type: ignore[attr-defined]
    orchestrator._task_store = task_store  # type: ignore[attr-defined]
    orchestrator._skill_registry = skill_registry  # type: ignore[attr-defined]
    orchestrator._hook_engine = hook_engine  # type: ignore[attr-defined]
    orchestrator._todo_manager = todo_manager  # type: ignore[attr-defined]
    orchestrator._project_instructions = project_instructions  # type: ignore[attr-defined]

    return orchestrator, event_bus, renderer, console, config


# ------------------------------------------------------------------
# REPL
# ------------------------------------------------------------------

_REPL_COMMANDS: dict[str, str] = {
    "/tools": "List available tools",
    "/skills": "List available skills (slash commands)",
    "/model": "Show current model and tier",
    "/status": "Show current session status (budget, memory, tasks, todos)",
    "/remember": "Save a fact: /remember <key> <value>",
    "/forget": "Remove a fact: /forget <key>",
    "/memories": "List all project memories",
    "/todo": "Add a todo: /todo <description>",
    "/done": "Complete a todo: /done <id>",
    "/todos": "List all todos",
    "/submit": "Run goal in background: /submit <goal>",
    "/tasks": "List background tasks",
    "/result": "Show task result: /result <id>",
    "/cancel": "Cancel a task: /cancel <id>",
    "/update": "Self-update (git pull + reinstall)",
    "/quit": "Exit the REPL",
    "/help": "Show this help",
}


def _suggest_command(cmd: str) -> str | None:
    """Suggest the closest matching REPL command for a typo."""
    cmd_lower = cmd.lower()
    # Check prefix matches first
    matches = [c for c in _REPL_COMMANDS if c.startswith(cmd_lower)]
    if len(matches) == 1:
        return matches[0]
    # Check substring matches
    matches = [c for c in _REPL_COMMANDS if cmd_lower[1:] in c]
    if len(matches) == 1:
        return matches[0]
    return None


async def _run_repl(
    orchestrator: Orchestrator,
    event_bus: EventBus,
    console: Console,
    router: ModelRouter | None = None,
    registry: ToolRegistry | None = None,
    project_memory: ProjectMemory | None = None,
    task_manager: TaskManager | None = None,
    session_memory: SessionMemory | None = None,
    skill_registry: SkillRegistry | None = None,
    todo_manager: TodoManager | None = None,
) -> None:
    """Run the interactive REPL loop."""
    # Session continuity: load previous conversation and bind for auto-save
    session_id = f"repl-{_project_hash()}"
    session_messages: list[dict[str, Any]] = []
    if session_memory:
        session_messages = await session_memory.load(session_id)
        session_memory.bind(session_id, session_messages)
        if session_messages:
            _logger.debug(
                "Restored %d messages from session %s",
                len(session_messages), session_id,
            )

    console.print(
        "[dim]Type [bold]/help[/bold] for commands, "
        "[bold]/quit[/bold] to exit[/dim]\n"
    )

    # Set up prompt_toolkit session with persistent history and completion
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import WordCompleter

    history_path = Path.home() / ".cache" / "open_harness" / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Build completion list: REPL commands + skill slash commands
    completion_words = list(_REPL_COMMANDS.keys())
    if skill_registry:
        for name in skill_registry.skill_names():
            completion_words.append(f"/{name}")
    completer = WordCompleter(completion_words, sentence=True)
    session = PromptSession(
        history=FileHistory(str(history_path)),
        completer=completer,
        complete_while_typing=False,
    )

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: session.prompt(">>> ")
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        text = user_input.strip()
        if not text:
            continue

        # Handle REPL commands
        if text.startswith("/"):
            cmd = text.split()[0].lower()
            if cmd == "/quit":
                console.print("[dim]Bye![/dim]")
                break
            elif cmd == "/help":
                for name, desc in _REPL_COMMANDS.items():
                    console.print(f"  [bold]{name:<10}[/bold] {desc}")
                continue
            elif cmd == "/tools":
                if registry:
                    tools = registry.list_tools()
                    if tools:
                        for tool in tools:
                            console.print(
                                f"  [bold]{tool.name:<18}[/bold] {tool.description}"
                            )
                    else:
                        console.print("  [dim]No tools registered[/dim]")
                else:
                    console.print("  [dim]Tool registry not available[/dim]")
                continue
            elif cmd == "/model":
                if router:
                    console.print(
                        f"  model: [bold]{router.current_model}[/bold]  "
                        f"tier: {router.current_tier}"
                    )
                else:
                    console.print("  [dim]Router not available[/dim]")
                continue
            elif cmd == "/status":
                # Session overview: model, policy budget, memory count, tasks
                if router:
                    console.print(
                        f"  model: [bold]{router.current_model}[/bold]  "
                        f"tier: {router.current_tier}"
                    )
                pol = orchestrator._policy
                if pol:
                    console.print(
                        f"  budget: {pol.budget.summary()}"
                    )
                if project_memory:
                    facts = await project_memory.list_all()
                    console.print(f"  memories: {len(facts)} fact(s)")
                console.print(f"  session history: {len(session_messages)} message(s)")
                if task_manager:
                    tasks = task_manager.list_tasks()
                    running = sum(1 for t in tasks if t.status.value == "running")
                    console.print(
                        f"  tasks: {len(tasks)} total, {running} running"
                    )
                if todo_manager and todo_manager.list_all():
                    console.print(f"  todos: {todo_manager.summary()}")
                if skill_registry:
                    console.print(
                        f"  skills: {len(skill_registry.list_skills())} loaded"
                    )
                continue
            elif cmd == "/remember":
                parts = text.split(maxsplit=2)
                if len(parts) < 3:
                    console.print("  [yellow]Usage: /remember <key> <value>[/yellow]")
                else:
                    if project_memory:
                        await project_memory.remember(parts[1], parts[2])
                        orchestrator.system_extra = await project_memory.build_context_block()
                        console.print(f"  Remembered: [bold]{parts[1]}[/bold]")
                    else:
                        console.print("  [dim]Memory not available[/dim]")
                continue
            elif cmd == "/forget":
                parts = text.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("  [yellow]Usage: /forget <key>[/yellow]")
                else:
                    if project_memory:
                        await project_memory.forget(parts[1])
                        orchestrator.system_extra = await project_memory.build_context_block()
                        console.print(f"  Forgot: [bold]{parts[1]}[/bold]")
                    else:
                        console.print("  [dim]Memory not available[/dim]")
                continue
            elif cmd == "/memories":
                if project_memory:
                    facts = await project_memory.list_all()
                    if facts:
                        for key, value in facts:
                            console.print(f"  [bold]{key:<20}[/bold] {value}")
                    else:
                        console.print("  [dim]No memories stored[/dim]")
                else:
                    console.print("  [dim]Memory not available[/dim]")
                continue
            elif cmd == "/submit":
                goal_text = text[len("/submit"):].strip()
                if not goal_text:
                    console.print("  [yellow]Usage: /submit <goal>[/yellow]")
                elif task_manager:
                    record = await task_manager.submit(goal_text)
                    console.print(
                        f"  Submitted task [bold]{record.id}[/bold]: {goal_text}"
                    )
                else:
                    console.print("  [dim]Task manager not available[/dim]")
                continue
            elif cmd == "/tasks":
                if task_manager:
                    tasks = task_manager.list_tasks()
                    if tasks:
                        for t in tasks:
                            elapsed = f" ({t.elapsed:.1f}s)" if t.elapsed else ""
                            console.print(
                                f"  [bold]{t.id}[/bold]  "
                                f"{t.status.value:<10} "
                                f"{t.goal[:40]}{elapsed}"
                            )
                    else:
                        console.print("  [dim]No tasks[/dim]")
                else:
                    console.print("  [dim]Task manager not available[/dim]")
                continue
            elif cmd == "/result":
                parts = text.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("  [yellow]Usage: /result <id>[/yellow]")
                elif task_manager:
                    t = task_manager.get_task(parts[1])
                    if t:
                        console.print(f"  [bold]{t.id}[/bold]  {t.status.value}")
                        console.print(f"  Goal: {t.goal}")
                        if t.result:
                            console.print(f"  Result: {t.result[:500]}")
                        if t.error:
                            console.print(f"  [red]Error: {t.error}[/red]")
                    else:
                        console.print(f"  [yellow]Task not found: {parts[1]}[/yellow]")
                else:
                    console.print("  [dim]Task manager not available[/dim]")
                continue
            elif cmd == "/cancel":
                parts = text.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("  [yellow]Usage: /cancel <id>[/yellow]")
                elif task_manager:
                    ok = await task_manager.cancel(parts[1])
                    if ok:
                        console.print(f"  Cancelled: [bold]{parts[1]}[/bold]")
                    else:
                        console.print(f"  [yellow]Cannot cancel: {parts[1]}[/yellow]")
                else:
                    console.print("  [dim]Task manager not available[/dim]")
                continue
            elif cmd == "/skills":
                if skill_registry:
                    skills = skill_registry.list_skills()
                    if skills:
                        for skill in skills:
                            args_hint = ""
                            if skill.args == "required":
                                args_hint = " <args> (required)"
                            elif skill.args == "optional":
                                args_hint = " [args]"
                            console.print(
                                f"  [bold]/{skill.name:<14}[/bold] "
                                f"{skill.description}{args_hint}"
                                f"  [dim]({skill.source})[/dim]"
                            )
                    else:
                        console.print("  [dim]No skills loaded[/dim]")
                else:
                    console.print("  [dim]Skill registry not available[/dim]")
                continue
            elif cmd == "/todo":
                parts = text.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("  [yellow]Usage: /todo <description>[/yellow]")
                elif todo_manager:
                    item = todo_manager.add(parts[1])
                    console.print(f"  Added: {item.to_display()}")
                else:
                    console.print("  [dim]Todo manager not available[/dim]")
                continue
            elif cmd == "/done":
                parts = text.split(maxsplit=1)
                if len(parts) < 2:
                    console.print("  [yellow]Usage: /done <id>[/yellow]")
                elif todo_manager:
                    try:
                        item_id = int(parts[1])
                    except ValueError:
                        console.print("  [yellow]Invalid ID[/yellow]")
                        continue
                    if todo_manager.complete(item_id):
                        console.print(f"  Completed: #{item_id}")
                    else:
                        console.print(f"  [yellow]Todo not found: #{item_id}[/yellow]")
                else:
                    console.print("  [dim]Todo manager not available[/dim]")
                continue
            elif cmd == "/todos":
                if todo_manager:
                    items = todo_manager.list_all()
                    if items:
                        for item in items:
                            style = "dim" if item.status.value == "completed" else ""
                            display = item.to_display()
                            if style:
                                console.print(f"  [{style}]{display}[/{style}]")
                            else:
                                console.print(f"  {display}")
                        console.print(f"  [dim]{todo_manager.summary()}[/dim]")
                    else:
                        console.print("  [dim]No todos[/dim]")
                else:
                    console.print("  [dim]Todo manager not available[/dim]")
                continue
            elif cmd == "/update":
                from open_harness_v2.update import self_update
                self_update(console)
                continue
            else:
                # Check if it's a skill invocation (e.g., /commit, /review)
                skill_name = cmd.lstrip("/")
                if skill_registry:
                    skill = skill_registry.get(skill_name)
                    if skill:
                        # Extract args after the command
                        skill_args = text[len(cmd):].strip()
                        if skill.args == "required" and not skill_args:
                            console.print(
                                f"  [yellow]/{skill_name} requires arguments: "
                                f"{skill.args_description}[/yellow]"
                            )
                            continue
                        # Expand skill into a goal and run it
                        text = skill.expand(skill_args)
                        console.print(
                            f"  [dim]Running skill: [bold]/{skill_name}[/bold][/dim]"
                        )
                        # Fall through to the goal execution below
                    else:
                        suggestion = _suggest_command(cmd)
                        if suggestion:
                            console.print(
                                f"[yellow]Unknown command: {cmd}[/yellow]  "
                                f"[dim]Did you mean [bold]{suggestion}[/bold]?[/dim]"
                            )
                        else:
                            console.print(
                                f"[yellow]Unknown command: {cmd}[/yellow]  "
                                f"[dim]Type /help for commands, /skills for skills[/dim]"
                            )
                        continue
                else:
                    suggestion = _suggest_command(cmd)
                    if suggestion:
                        console.print(
                            f"[yellow]Unknown command: {cmd}[/yellow]  "
                            f"[dim]Did you mean [bold]{suggestion}[/bold]?[/dim]"
                        )
                    else:
                        console.print(
                            f"[yellow]Unknown command: {cmd}[/yellow]  "
                            f"[dim]Type /help for commands[/dim]"
                        )
                    continue

        # Run the goal — Ctrl+C cancels the current goal
        task: asyncio.Task[str] | None = None
        try:
            # Build context with session history for continuity
            from open_harness_v2.core.context import AgentContext

            ctx = AgentContext()
            for msg in session_messages:
                ctx.history.add(msg)

            session_messages.append({"role": "user", "content": text})

            task = asyncio.create_task(orchestrator.run(text, context=ctx))

            # Install SIGINT handler to cancel running goal
            original_handler = signal.getsignal(signal.SIGINT)

            def _cancel_goal(sig: int, frame: object) -> None:
                if task and not task.done():
                    orchestrator.cancel()

            signal.signal(signal.SIGINT, _cancel_goal)

            response = await task

            # Restore original handler
            signal.signal(signal.SIGINT, original_handler)

            # Save assistant response to session
            session_messages.append({"role": "assistant", "content": response})
            if session_memory:
                await session_memory.save(session_id, session_messages)

        except asyncio.CancelledError:
            console.print("\n[yellow]Cancelled[/yellow]")
            # Remove the unanswered user message
            if session_messages and session_messages[-1].get("role") == "user":
                session_messages.pop()
        except Exception as exc:
            console.print(f"\n[red]Error: {exc}[/red]")
            _logger.exception("REPL goal error")
            if session_messages and session_messages[-1].get("role") == "user":
                session_messages.pop()

        console.print()  # blank line between goals


# ------------------------------------------------------------------
# Click entry point
# ------------------------------------------------------------------

@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("goal", required=False, default=None)
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
@click.option("--profile", default=None, help="Profile name to use.")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed event log.")
@click.version_option(version=__version__, prog_name="harness")
@click.pass_context
def main(
    ctx: click.Context,
    goal: str | None,
    config_path: str | None,
    profile: str | None,
    verbose: bool,
) -> None:
    """Open Harness v2 — async-first AI agent harness.

    Run with a GOAL argument for one-shot mode, or without for interactive REPL.
    """
    # If a subcommand was invoked (e.g. "harness update"), skip the default logic.
    if ctx.invoked_subcommand is not None:
        return

    # Click consumes the first positional token as GOAL before checking
    # subcommands.  Detect this and redirect.
    if goal and goal in main.commands:
        ctx.invoke(main.commands[goal])
        return

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(name)s %(levelname)s %(message)s",
    )

    # Offer setup wizard when no config file exists
    if config_path is None and not _config_exists():
        cons = Console()
        cons.print("[yellow]No config file found.[/yellow]")
        if click.confirm("Run setup wizard?", default=True):
            from open_harness_v2.setup import run_setup_wizard

            run_setup_wizard()
        else:
            cons.print("[dim]Using defaults.[/dim]")

    orchestrator, event_bus, renderer, console, config = _build_components(
        config_path, profile, verbose,
    )

    # Extract components from orchestrator internals for REPL commands
    router = orchestrator._router
    registry = orchestrator._registry
    project_memory: ProjectMemory = orchestrator._project_memory  # type: ignore[attr-defined]
    session_memory: SessionMemory = orchestrator._session_memory  # type: ignore[attr-defined]
    memory_store: MemoryStore = orchestrator._memory_store  # type: ignore[attr-defined]
    task_manager: TaskManager = orchestrator._task_manager  # type: ignore[attr-defined]
    task_store: TaskStore = orchestrator._task_store  # type: ignore[attr-defined]
    skill_registry: SkillRegistry = orchestrator._skill_registry  # type: ignore[attr-defined]
    hook_engine: HookEngine = orchestrator._hook_engine  # type: ignore[attr-defined]
    todo_manager: TodoManager = orchestrator._todo_manager  # type: ignore[attr-defined]
    project_instructions_text: str = orchestrator._project_instructions  # type: ignore[attr-defined]

    # Run everything in a single event loop so that httpx connections,
    # background tasks, etc. are all cleaned up properly.
    async def _main_async() -> None:
        # Inject project memory + project instructions into system prompt
        block = await project_memory.build_context_block()
        if project_instructions_text:
            block = f"{project_instructions_text}\n\n{block}" if block else project_instructions_text
        orchestrator.system_extra = block

        # Print startup banner (skip for one-shot mode)
        if not goal:
            _print_banner(
                console, config, config_path,
                len(registry.list_tools()),
                num_skills=len(skill_registry.list_skills()),
                has_hooks=hook_engine.has_hooks,
                has_instructions=bool(project_instructions_text),
            )

        if goal:
            await orchestrator.run(goal)
        else:
            await _run_repl(
                orchestrator, event_bus, console,
                router, registry, project_memory, task_manager,
                session_memory, skill_registry, todo_manager,
            )

        # Cleanup — drain background tasks before closing shared resources
        await task_manager.shutdown()
        await router.get_client().close()
        await memory_store.close()
        await task_store.close()

    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        console.print("\n[dim]Bye![/dim]")


@main.command("update")
def update_cmd() -> None:
    """Self-update: git pull + reinstall."""
    from open_harness_v2.update import self_update

    console = Console()
    self_update(console)


if __name__ == "__main__":
    main()
