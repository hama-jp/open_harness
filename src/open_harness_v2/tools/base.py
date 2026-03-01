"""Async Tool abstract base class for Open Harness v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from open_harness_v2.types import ToolParameter, ToolResult


class Tool(ABC):
    """Base class for all tools.

    Subclasses must set ``name``, ``description``, ``parameters`` as class
    attributes and implement the async ``execute()`` method.
    """

    name: str
    description: str
    parameters: list[ToolParameter]
    max_output: int = 5000  # Per-tool output limit (chars). Override in subclasses.

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool asynchronously."""

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties: dict[str, Any] = {}
        required: list[str] = []
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
        params_desc: list[str] = []
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

    def to_compact_description(self) -> str:
        """One-line compact description for token-efficient prompts."""
        params = ", ".join(
            f"{p.name}: {p.type}" + ("?" if not p.required else "")
            for p in self.parameters
        )
        return f"{self.name}({params}) - {self.description}"
