"""Tool registry with plugin discovery for Open Harness v2."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from open_harness_v2.tools.base import Tool
from open_harness_v2.types import ToolResult

_logger = logging.getLogger(__name__)


def _smart_truncate(text: str, max_length: int) -> str:
    """Intelligent truncation: keep head and tail with middle summary.

    For shell-like output, keeps first 25% + last 75% for error visibility.
    """
    if len(text) <= max_length:
        return text
    head_size = max_length // 4
    tail_size = max_length - head_size
    omitted = len(text) - max_length
    return (
        text[:head_size]
        + f"\n\n... [{omitted} chars truncated] ...\n\n"
        + text[-tail_size:]
    )


class ToolRegistry:
    """Registry of available tools with async execution and plugin discovery."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def tool_names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments.

        Applies per-tool output truncation after execution.
        Returns an error ToolResult if the tool is unknown or raises.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}. Available: {', '.join(self._tools.keys())}",
            )
        try:
            result = await tool.execute(**arguments)
            # Apply per-tool output truncation
            max_out = getattr(tool, "max_output", 5000)
            if max_out > 0 and len(result.output) > max_out:
                result = ToolResult(
                    success=result.success,
                    output=_smart_truncate(result.output, max_out),
                    error=result.error,
                    metadata=result.metadata,
                )
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' execution failed: {type(e).__name__}: {e}",
            )

    def discover(self) -> None:
        """Load tools from entry_points group ``open_harness.tools``.

        Each entry point should be a callable that returns a Tool instance
        or a Tool subclass (which will be instantiated).
        """
        eps = entry_points()
        # Python 3.12+ returns a SelectableGroups; 3.9+ supports .select()
        if hasattr(eps, "select"):
            group = eps.select(group="open_harness.tools")
        else:
            group = eps.get("open_harness.tools", [])

        for ep in group:
            try:
                obj = ep.load()
                # If it's a class, instantiate it
                if isinstance(obj, type) and issubclass(obj, Tool):
                    tool = obj()
                elif isinstance(obj, Tool):
                    tool = obj
                elif callable(obj):
                    tool = obj()
                else:
                    _logger.warning(
                        "Entry point %s did not return a Tool: %s", ep.name, type(obj)
                    )
                    continue
                self.register(tool)
                _logger.info("Discovered plugin tool: %s", tool.name)
            except Exception:
                _logger.exception("Failed to load tool plugin: %s", ep.name)

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schemas for all registered tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_prompt_description(self) -> str:
        """Return detailed prompt descriptions for all registered tools."""
        descs = [t.to_prompt_description() for t in self._tools.values()]
        return "\n\n".join(descs)

    def get_compact_prompt_description(self) -> str:
        """Compact one-line-per-tool description for token-efficient prompts."""
        return "\n".join(t.to_compact_description() for t in self._tools.values())
