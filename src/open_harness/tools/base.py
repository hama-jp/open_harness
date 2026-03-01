"""Tool system base classes and registry."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # string, integer, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> str:
        if self.success:
            return self.output
        return f"[Tool Error] {self.error}\n{self.output}" if self.output else f"[Tool Error] {self.error}"


class Tool(ABC):
    """Base class for all tools."""

    name: str
    description: str
    parameters: list[ToolParameter]

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_prompt_description(self) -> str:
        """Generate a text description for prompt-based tool calling."""
        params_desc = []
        for p in self.parameters:
            req = "required" if p.required else "optional"
            line = f"  - {p.name} ({p.type}, {req}): {p.description}"
            if p.default is not None:
                line += f" (default: {p.default})"
            if p.enum:
                line += f" (options: {', '.join(p.enum)})"
            params_desc.append(line)

        params_str = "\n".join(params_desc) if params_desc else "  (none)"
        return f"### {self.name}\n{self.description}\nParameters:\n{params_str}"


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_prompt_description(self) -> str:
        descs = [t.to_prompt_description() for t in self._tools.values()]
        return "\n\n".join(descs)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}. Available: {', '.join(self._tools.keys())}",
            )
        try:
            return tool.execute(**arguments)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' execution failed: {type(e).__name__}: {e}",
            )
