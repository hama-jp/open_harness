"""Prompt optimizer middleware — injects format hints for weak models.

Adds tool-calling format examples and thinking directives to the system
prompt so that smaller / weaker models produce parseable output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from open_harness_v2.types import LLMResponse

from .middleware import LLMRequest, Middleware, NextFn

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool format hint (injected into system prompt)
# ---------------------------------------------------------------------------

_TOOL_FORMAT_HINT = """\
When you need to use a tool, respond with EXACTLY this JSON (nothing else):
{"tool": "tool_name", "args": {"param1": "value1"}}

RULES:
- Output ONLY the JSON when calling a tool -- no other text around it
- ONE tool call per response
- To respond to the user, just write normal text (no JSON)
"""

_THINKING_HINTS = {
    "always": "Use <think>...</think> for ALL reasoning before responding.\n",
    "auto": "Use <think>...</think> for complex reasoning. Skip for simple tasks.\n",
    "never": "/no_think\n",
}


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@dataclass
class PromptOptimizerMiddleware:
    """Middleware that enriches the system prompt for weak models.

    Parameters
    ----------
    thinking_mode:
        ``"auto"`` | ``"always"`` | ``"never"`` — controls thinking hints.
    inject_tool_format:
        If ``True``, append tool-format examples to the system prompt
        when ``request.tools`` is non-empty.
    extra_instructions:
        Additional lines to inject into the system prompt.
    """

    thinking_mode: str = "auto"
    inject_tool_format: bool = True
    extra_instructions: list[str] = field(default_factory=list)

    async def process(
        self,
        request: LLMRequest,
        next_fn: NextFn,
    ) -> LLMResponse:
        """Optionally modify the system prompt, then forward."""
        additions: list[str] = []

        # Thinking directive
        hint = _THINKING_HINTS.get(self.thinking_mode, "")
        if hint:
            additions.append(hint)

        # Tool format examples
        if self.inject_tool_format and request.tools:
            additions.append(_TOOL_FORMAT_HINT)
            # Append a mini-summary of available tools
            tool_names = _extract_tool_names(request.tools)
            if tool_names:
                additions.append(
                    f"Available tools: {', '.join(tool_names)}\n"
                )

        # Extra user-provided instructions
        additions.extend(self.extra_instructions)

        if additions:
            request = _inject_into_system_prompt(
                request, "\n".join(additions),
            )

        return await next_fn(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    """Extract tool names from an OpenAI-style tools list."""
    names: list[str] = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "")
        if name:
            names.append(name)
    return names


def _inject_into_system_prompt(
    request: LLMRequest,
    addition: str,
) -> LLMRequest:
    """Return a shallow copy of *request* with *addition* appended to the
    system message.  If there is no system message, one is created."""
    messages = list(request.messages)
    if messages and messages[0].get("role") == "system":
        original = messages[0].get("content", "")
        messages[0] = {
            "role": "system",
            "content": f"{original}\n\n{addition}",
        }
    else:
        messages.insert(0, {"role": "system", "content": addition})

    return LLMRequest(
        messages=messages,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        tools=request.tools,
        tool_choice=request.tool_choice,
        context_length=request.context_length,
        metadata=request.metadata,
    )
