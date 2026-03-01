"""Tool call extraction and stream processing utilities.

Adapted from v1 ``open_harness.llm.client`` with imports pointing to
``open_harness_v2.types.ToolCall``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Generator

from open_harness_v2.types import ToolCall

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thinking extraction
# ---------------------------------------------------------------------------

def _extract_thinking(text: str) -> tuple[str, str]:
    """Extract ``<think>...</think>`` blocks from response text.

    Returns (thinking_text, cleaned_text).
    """
    pattern = r"<think>(.*?)</think>"
    thinking_parts = re.findall(pattern, text, re.DOTALL)
    thinking = "\n".join(thinking_parts).strip()
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return thinking, cleaned


# ---------------------------------------------------------------------------
# JSON / tool-call parsing helpers
# ---------------------------------------------------------------------------

def _extract_balanced_json(text: str, start: int) -> str | None:
    """Extract a balanced JSON object starting at *start* (must be ``{``).

    Handles nested braces and quoted strings so that
    ``{"args": {"k": "v"}}`` is captured in full.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_parse_tool_json(raw: str) -> ToolCall | None:
    """Try to parse a single JSON string as a tool call."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Single repair pass: strip markdown fences, trailing prose, fix quotes
        cleaned = raw.strip()
        for prefix in ("```json", "```"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    if "tool" in data and "args" in data:
        args = data.get("args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"prompt": args}
        return ToolCall(name=data["tool"], arguments=args, raw=raw)
    elif "tool_call" in data:
        tc = data["tool_call"]
        args = tc.get("arguments", tc.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {"prompt": args}
        return ToolCall(
            name=tc.get("name", tc.get("tool", "")),
            arguments=args,
            raw=raw,
        )
    return None


def _parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Extract tool calls from free-form text.

    Uses a unified search: fenced blocks -> bare ``{"tool":...}`` ->
    whole text -> alt format.  Short-circuits on first successful match.
    """
    calls: list[ToolCall] = []

    # Strategy 1: fenced code block
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)

    # Strategy 2: bare {"tool": ...} with balanced brace matching
    if not matches:
        bare_starts = [m.start() for m in re.finditer(r'\{"tool"\s*:', text)]
        for pos in bare_starts:
            obj = _extract_balanced_json(text, pos)
            if obj:
                matches.append(obj)

    # Strategy 3: entire text is a JSON object
    if not matches:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            matches = [stripped]

    # Strategy 4: alternate {"tool_call": ...} format
    if not matches:
        alt_starts = [m.start() for m in re.finditer(r'\{"tool_call"\s*:', text)]
        for pos in alt_starts:
            obj = _extract_balanced_json(text, pos)
            if obj:
                matches.append(obj)

    for match in matches:
        call = _try_parse_tool_json(match)
        if call:
            calls.append(call)

    return calls


# ---------------------------------------------------------------------------
# ToolCallParser — schema-aware with short-circuit regex
# ---------------------------------------------------------------------------

class ToolCallParser:
    """Schema-aware parser for tool calls from free-form text.

    Pre-compiles a combined regex from registered tool names for fast
    first-match short-circuiting.  Falls back to generic parsing.
    """

    def __init__(self, tool_names: list[str] | None = None):
        self._tool_names = set(tool_names) if tool_names else set()
        if self._tool_names:
            escaped = "|".join(re.escape(n) for n in self._tool_names)
            self._known_tool_pattern = re.compile(
                r'\{\s*"tool"\s*:\s*"(' + escaped + r')"'
            )
        else:
            self._known_tool_pattern = None

    def parse(self, text: str) -> list[ToolCall]:
        """Parse tool calls with short-circuit on known tool names."""
        if self._known_tool_pattern:
            match = self._known_tool_pattern.search(text)
            if match:
                obj = _extract_balanced_json(text, text.rfind("{", 0, match.end()))
                if obj:
                    call = _try_parse_tool_json(obj)
                    if call:
                        return [call]
        return _parse_tool_calls_from_text(text)


# ---------------------------------------------------------------------------
# NativeToolCallAccumulator — streaming native function-calling
# ---------------------------------------------------------------------------

class NativeToolCallAccumulator:
    """Accumulate native function-calling tool_calls from streaming deltas.

    OpenAI-compatible providers send tool calls as incremental chunks:
    each chunk has an ``index``, a ``function.name`` (first chunk only),
    and ``function.arguments`` fragments that must be concatenated.
    """

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, str]] = {}

    def feed(self, delta: dict[str, Any]) -> None:
        """Process ``delta.tool_calls`` from a single SSE chunk."""
        tc_list = delta.get("tool_calls")
        if not tc_list:
            return
        for tc in tc_list:
            idx = tc.get("index", 0)
            func = tc.get("function", {})
            if idx not in self._calls:
                self._calls[idx] = {"name": "", "arguments": ""}
            if func.get("name"):
                self._calls[idx]["name"] = func["name"]
            if func.get("arguments"):
                self._calls[idx]["arguments"] += func["arguments"]

    def has_calls(self) -> bool:
        return bool(self._calls)

    def finalize(self) -> list[ToolCall]:
        """Parse accumulated fragments into complete ToolCall objects."""
        result: list[ToolCall] = []
        for idx in sorted(self._calls):
            entry = self._calls[idx]
            name = entry["name"]
            raw_args = entry["arguments"]
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}
            if name:
                result.append(
                    ToolCall(
                        name=name,
                        arguments=args,
                        raw=json.dumps(
                            {"function": {"name": name, "arguments": raw_args}}
                        ),
                    )
                )
        return result


# ---------------------------------------------------------------------------
# StreamProcessor — handles <think> tags and detects tool calls
# ---------------------------------------------------------------------------

class StreamProcessor:
    """Processes SSE chunks from a streaming LLM response.

    States:
      init      - haven't determined response type yet
      thinking  - inside <think> block
      detecting - after thinking, buffering first chars to detect text vs tool
      text      - streaming normal text
      tool      - buffering a tool call (not displayed until complete)
    """

    _FENCE_PREFIXES = ("```json", "```\n{", "```{")
    _THINKING_YIELD_INTERVAL = 200

    def __init__(self) -> None:
        self.buffer = ""
        self.thinking = ""
        self.content_start = 0
        self.displayed_up_to = 0
        self.state = "init"
        self._thinking_yielded_at = 0

    def feed(self, chunk: str) -> Generator[tuple[str, str], None, None]:
        """Feed a chunk.  Yields ``(event_type, data)`` pairs.

        event_type: ``"thinking"``, ``"text"``, ``"tool_buffering"``
        """
        self.buffer += chunk

        changed = True
        while changed:
            changed = False

            if self.state == "init":
                stripped = self.buffer.lstrip()
                if stripped.startswith("<think>"):
                    self.state = "thinking"
                    changed = True
                elif len(stripped) >= 7 or (
                    stripped and not stripped.startswith("<")
                ):
                    self.content_start = len(self.buffer) - len(stripped)
                    self.displayed_up_to = self.content_start
                    self.state = "detecting"
                    changed = True

            elif self.state == "thinking":
                end_idx = self.buffer.find("</think>")
                if end_idx >= 0:
                    think_start = self.buffer.find("<think>") + len("<think>")
                    self.thinking = self.buffer[think_start:end_idx].strip()
                    self.content_start = end_idx + len("</think>")
                    self.displayed_up_to = self.content_start
                    yield ("thinking", self.thinking)
                    self.state = "detecting"
                    changed = True
                else:
                    think_start = self.buffer.find("<think>")
                    if think_start >= 0:
                        partial_len = (
                            len(self.buffer) - think_start - len("<think>")
                        )
                        if (
                            partial_len - self._thinking_yielded_at
                            >= self._THINKING_YIELD_INTERVAL
                        ):
                            self._thinking_yielded_at = partial_len
                            snippet = self.buffer[
                                think_start + len("<think>") :
                            ].strip()
                            last_line = snippet.split("\n")[-1][:80]
                            yield ("thinking", last_line)

            elif self.state == "detecting":
                content = self.buffer[self.content_start :].lstrip()
                if not content:
                    break
                if content.startswith("{"):
                    self.state = "tool"
                elif any(content.startswith(p) for p in self._FENCE_PREFIXES):
                    self.state = "tool"
                elif len(content) > 8:
                    self.state = "text"
                    changed = True

            elif self.state == "text":
                new = self.buffer[self.displayed_up_to :]
                if new:
                    self.displayed_up_to = len(self.buffer)
                    yield ("text", new)

            elif self.state == "tool":
                break

    def finish(self) -> tuple[str, str, list[ToolCall]]:
        """Call when stream ends.  Returns ``(thinking, content, tool_calls)``."""
        content = self.buffer[self.content_start :].strip()

        if self.state == "thinking":
            think_start = self.buffer.find("<think>")
            if think_start >= 0:
                self.thinking = self.buffer[
                    think_start + len("<think>") :
                ].strip()
            content = ""

        tool_calls: list[ToolCall] = []
        if self.state == "tool":
            tool_calls = _parse_tool_calls_from_text(content)

        # Fallback: even in text state, check full content
        if not tool_calls and content:
            tool_calls = _parse_tool_calls_from_text(content)

        return self.thinking, content, tool_calls

    @property
    def undisplayed_text(self) -> str:
        """Text that hasn't been yielded yet."""
        return self.buffer[self.displayed_up_to :].strip()
