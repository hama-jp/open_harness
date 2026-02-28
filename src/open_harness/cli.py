"""CLI interface for Open Harness."""

from __future__ import annotations

import logging
import os
import sys
import time

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from open_harness.agent import Agent, AgentStep
from open_harness.config import HarnessConfig, load_config
from open_harness.llm.router import ModelRouter
from open_harness.memory.store import MemoryStore
from open_harness.tools.base import ToolRegistry
from open_harness.tools.external import CodexTool, GeminiCliTool
from open_harness.tools.file_ops import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from open_harness.tools.shell import ShellTool

console = Console()


def setup_tools(config: HarnessConfig) -> ToolRegistry:
    """Register all available tools."""
    registry = ToolRegistry()

    # Core tools
    registry.register(ShellTool(config.tools.shell))
    registry.register(ReadFileTool(config.tools.file))
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ListDirectoryTool())
    registry.register(SearchFilesTool())

    # External agents
    if config.external_agents.get("codex", None) and config.external_agents["codex"].enabled:
        codex = CodexTool(config.external_agents["codex"].command)
        if codex.available:
            registry.register(codex)
            console.print("[dim]  codex CLI available[/dim]")

    if config.external_agents.get("gemini", None) and config.external_agents["gemini"].enabled:
        gemini = GeminiCliTool(config.external_agents["gemini"].command)
        if gemini.available:
            registry.register(gemini)
            console.print("[dim]  gemini CLI available[/dim]")

    return registry


def on_step(step: AgentStep):
    """Callback for agent steps - shows progress to user."""
    if step.step_type == "tool_call":
        tool_name = step.metadata.get("tool", "?")
        args = step.metadata.get("args", {})
        args_short = str(args)
        if len(args_short) > 100:
            args_short = args_short[:100] + "..."
        console.print(f"[yellow]> {tool_name}[/yellow] [dim]{args_short}[/dim]")

    elif step.step_type == "tool_result":
        success = step.metadata.get("success", False)
        icon = "[green]OK[/green]" if success else "[red]FAIL[/red]"
        # Show truncated output
        output = step.content
        if len(output) > 500:
            output = output[:500] + "\n..."
        if output.strip():
            console.print(Panel(
                output,
                title=f"{icon} {step.metadata.get('tool', '')}",
                border_style="dim",
                expand=False,
            ))

    elif step.step_type == "compensation":
        console.print(f"[magenta]~ Compensating: {step.content}[/magenta]")

    elif step.step_type == "llm_call":
        console.print(f"[dim]{step.content}[/dim]", end="\r")

    elif step.step_type == "llm_response":
        latency = step.metadata.get("latency_ms", 0)
        if latency:
            console.print(f"[dim]({latency:.0f}ms)[/dim]", end=" ")
        thinking = step.metadata.get("thinking", "")
        if thinking:
            # Show first line of thinking
            first_line = thinking.split("\n")[0][:80]
            console.print(f"[dim italic]thinking: {first_line}...[/dim italic]")


def handle_command(cmd: str, agent: Agent, config: HarnessConfig) -> bool:
    """Handle special commands. Returns True if command was handled."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()

    if command in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/dim]")
        return True

    elif command == "/clear":
        agent.memory.clear_conversation()
        console.print("[dim]Conversation cleared.[/dim]")
        return True

    elif command == "/tier":
        if len(parts) > 1:
            agent.router.current_tier = parts[1]
            console.print(f"[dim]Model tier set to: {agent.router.current_tier}[/dim]")
        else:
            tiers = agent.router.list_tiers()
            current = agent.router.current_tier
            for name, desc in tiers.items():
                marker = " *" if name == current else ""
                console.print(f"  {name}: {desc}{marker}")
        return True

    elif command == "/tools":
        for tool in agent.tools.list_tools():
            console.print(f"  [bold]{tool.name}[/bold]: {tool.description}")
        return True

    elif command == "/help":
        console.print("""
[bold]Commands:[/bold]
  /quit, /exit  - Exit the harness
  /clear        - Clear conversation history
  /tier [name]  - Show or set model tier (small/medium/large)
  /tools        - List available tools
  /help         - Show this help
        """)
        return True

    return False


@click.command()
@click.option("--config", "-c", "config_path", default=None, help="Path to config file")
@click.option("--tier", "-t", default=None, help="Model tier to use (small/medium/large)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(config_path: str | None, tier: str | None, verbose: bool):
    """Open Harness - AI agent harness for local LLMs."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Banner
    console.print(Panel(
        "[bold]Open Harness[/bold] v0.1.0\n"
        "AI agent harness optimized for local LLMs\n"
        "[dim]Type /help for commands, /quit to exit[/dim]",
        border_style="blue",
    ))

    # Load config
    config = load_config(config_path)
    if tier:
        config.llm.default_tier = tier

    # Show current model
    try:
        model_cfg = config.llm.models.get(config.llm.default_tier)
        if model_cfg:
            console.print(f"[dim]Model: {model_cfg.model} ({config.llm.default_tier})[/dim]")
    except Exception:
        pass

    # Setup
    tools = setup_tools(config)
    memory = MemoryStore(config.memory.db_path)
    agent = Agent(config, tools, memory, on_step=on_step)

    console.print(f"[dim]Tools: {', '.join(t.name for t in tools.list_tools())}[/dim]")
    console.print()

    # REPL
    try:
        while True:
            try:
                user_input = console.input("[bold green]> [/bold green]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                if handle_command(user_input, agent, config):
                    if user_input.strip().split()[0] in ("/quit", "/exit", "/q"):
                        break
                    continue

            # Run agent
            try:
                start = time.monotonic()
                response = agent.run(user_input)
                elapsed = time.monotonic() - start

                console.print()
                console.print(Markdown(response))
                console.print(f"\n[dim]({elapsed:.1f}s)[/dim]")
                console.print()

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if verbose:
                    console.print_exception()

    finally:
        agent.router.close()
        memory.close()


if __name__ == "__main__":
    main()
