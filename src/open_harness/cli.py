"""CLI interface for Open Harness with streaming support."""

from __future__ import annotations

import logging
import sys
import time

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from open_harness.agent import Agent, AgentEvent
from open_harness.config import HarnessConfig, load_config
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
    registry = ToolRegistry()
    registry.register(ShellTool(config.tools.shell))
    registry.register(ReadFileTool(config.tools.file))
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ListDirectoryTool())
    registry.register(SearchFilesTool())

    if config.external_agents.get("codex") and config.external_agents["codex"].enabled:
        codex = CodexTool(config.external_agents["codex"].command)
        if codex.available:
            registry.register(codex)
            console.print("[dim]  codex CLI available[/dim]")

    if config.external_agents.get("gemini") and config.external_agents["gemini"].enabled:
        gemini = GeminiCliTool(config.external_agents["gemini"].command)
        if gemini.available:
            registry.register(gemini)
            console.print("[dim]  gemini CLI available[/dim]")

    return registry


class StreamingDisplay:
    """Handles real-time display of agent events."""

    def __init__(self, console: Console):
        self.console = console
        self._text_buffer = ""
        self._streaming = False

    def handle_event(self, event: AgentEvent):
        if event.type == "status":
            self.console.print(f"[dim]{event.data}[/dim]", end="\r")

        elif event.type == "thinking":
            first_line = event.data.split("\n")[0][:80]
            self.console.print(f"[dim italic]thinking: {first_line}...[/dim italic]")

        elif event.type == "text":
            if not self._streaming:
                self._streaming = True
                self.console.print()  # blank line before response
            self._text_buffer += event.data
            # Print chunk immediately — raw text, not markdown
            self.console.print(event.data, end="", highlight=False)

        elif event.type == "tool_call":
            self._flush_stream()
            tool = event.metadata.get("tool", "?")
            args = event.metadata.get("args", {})
            args_short = str(args)
            if len(args_short) > 100:
                args_short = args_short[:100] + "..."
            self.console.print(f"[yellow]> {tool}[/yellow] [dim]{args_short}[/dim]")

        elif event.type == "tool_result":
            success = event.metadata.get("success", False)
            icon = "[green]OK[/green]" if success else "[red]FAIL[/red]"
            output = event.data
            if len(output) > 500:
                output = output[:500] + "\n..."
            if output.strip():
                self.console.print(Panel(
                    output,
                    title=f"{icon} {event.metadata.get('tool', '')}",
                    border_style="dim",
                    expand=False,
                ))

        elif event.type == "compensation":
            self._flush_stream()
            self.console.print(f"[magenta]~ Compensating: {event.data}[/magenta]")

        elif event.type == "done":
            if self._streaming:
                # Already streamed text — just add newline
                self.console.print()
                self._streaming = False
                self._text_buffer = ""
            elif event.data:
                # Non-streamed response (e.g., after tool calls where text wasn't streamed)
                self.console.print()
                self.console.print(Markdown(event.data))

    def _flush_stream(self):
        if self._streaming:
            self.console.print()
            self._streaming = False
            self._text_buffer = ""


def handle_command(cmd: str, agent: Agent, config: HarnessConfig) -> bool:
    """Handle /commands. Returns True if handled."""
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
@click.option("--tier", "-t", default=None, help="Model tier (small/medium/large)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def main(config_path: str | None, tier: str | None, verbose: bool):
    """Open Harness - AI agent harness for local LLMs."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    console.print(Panel(
        "[bold]Open Harness[/bold] v0.1.0\n"
        "AI agent harness optimized for local LLMs\n"
        "[dim]Type /help for commands, /quit to exit[/dim]",
        border_style="blue",
    ))

    config = load_config(config_path)
    if tier:
        config.llm.default_tier = tier

    try:
        model_cfg = config.llm.models.get(config.llm.default_tier)
        if model_cfg:
            console.print(f"[dim]Model: {model_cfg.model} ({config.llm.default_tier})[/dim]")
    except Exception:
        pass

    tools = setup_tools(config)
    memory = MemoryStore(config.memory.db_path, max_turns=config.memory.max_conversation_turns)
    agent = Agent(config, tools, memory)
    display = StreamingDisplay(console)

    console.print(f"[dim]Tools: {', '.join(t.name for t in tools.list_tools())}[/dim]")
    console.print()

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

            if user_input.startswith("/"):
                if handle_command(user_input, agent, config):
                    if user_input.strip().split()[0] in ("/quit", "/exit", "/q"):
                        break
                    continue

            try:
                start = time.monotonic()
                for event in agent.run_stream(user_input):
                    display.handle_event(event)
                elapsed = time.monotonic() - start
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
