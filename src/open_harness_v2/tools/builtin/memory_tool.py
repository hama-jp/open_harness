"""LLM-callable tools for the project memory system.

These tools let the LLM explicitly store and retrieve facts about the
current project.  They require a ``ProjectMemory`` instance to be
injected at construction time.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from open_harness_v2.tools.base import Tool
from open_harness_v2.types import ToolParameter, ToolResult

if TYPE_CHECKING:
    from open_harness_v2.memory.project import ProjectMemory


class RememberTool(Tool):
    """Save a fact about the current project."""

    name = "remember"
    description = "Save a key/value fact about the current project for future reference."
    parameters = [
        ToolParameter(name="key", type="string", description="Short identifier for the fact"),
        ToolParameter(name="value", type="string", description="The fact content to remember"),
    ]

    def __init__(self, project_memory: ProjectMemory) -> None:
        self._memory = project_memory

    async def execute(self, **kwargs: Any) -> ToolResult:
        key = kwargs.get("key", "")
        value = kwargs.get("value", "")
        if not key or not value:
            return ToolResult(success=False, output="", error="Both key and value are required.")
        await self._memory.remember(key, value)
        return ToolResult(success=True, output=f"Remembered: {key}")


class ForgetTool(Tool):
    """Remove a previously remembered fact."""

    name = "forget"
    description = "Remove a previously saved fact about the current project."
    parameters = [
        ToolParameter(name="key", type="string", description="The key of the fact to forget"),
    ]

    def __init__(self, project_memory: ProjectMemory) -> None:
        self._memory = project_memory

    async def execute(self, **kwargs: Any) -> ToolResult:
        key = kwargs.get("key", "")
        if not key:
            return ToolResult(success=False, output="", error="Key is required.")
        existing = await self._memory.recall(key)
        if existing is None:
            return ToolResult(success=False, output="", error=f"No fact found for key: {key}")
        await self._memory.forget(key)
        return ToolResult(success=True, output=f"Forgot: {key}")
