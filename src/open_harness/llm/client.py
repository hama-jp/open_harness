"""OpenAI-compatible LLM client for local and remote providers."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import httpx

from open_harness.config import ProviderConfig


@dataclass
class ToolCall:
    """Parsed tool call from LLM response."""
    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class LLMResponse:
    """Unified response from LLM."""
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0

    @property
    def has_tool_call(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def clean_content(self) -> str:
        return self.content


def _extract_thinking(text: str) -> tuple[str, str]:
    """Extract <think>...</think> blocks from response text."""
    pattern = r"<think>(.*?)</think>"
    thinking_parts = re.findall(pattern, text, re.DOTALL)
    thinking = "\n".join(thinking_parts).strip()
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return thinking, cleaned


def _extract_balanced_json(text: str, start: int) -> str | None:
    """Extract a balanced JSON object starting at *start* (must be '{').

    Handles nested braces and quoted strings so that `{"args": {"k": "v"}}`
    is captured in full rather than truncated at the first '}'.
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
                return text[start:i + 1]
    return None


def _parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Try to extract tool calls from free-form text."""
    calls = []

    # Strategy 1: fenced code block — ```json ... ```
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
        try:
            data = json.loads(match)
            if "tool" in data and "args" in data:
                args = data.get("args", {})
                # Ensure args is a dict (weak LLMs may emit a JSON string)
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {"prompt": args}
                calls.append(ToolCall(
                    name=data["tool"],
                    arguments=args,
                    raw=match,
                ))
            elif "tool_call" in data:
                tc = data["tool_call"]
                args = tc.get("arguments", tc.get("args", {}))
                # Ensure args is a dict
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {"prompt": args}
                calls.append(ToolCall(
                    name=tc.get("name", tc.get("tool", "")),
                    arguments=args,
                    raw=match,
                ))
        except (json.JSONDecodeError, KeyError):
            continue

    return calls


# ---------------------------------------------------------------------------
# Stream processor: handles <think> tags and detects tool calls during streaming
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

    # Fenced code block markers that indicate the LLM is wrapping a tool call
    _FENCE_PREFIXES = ("```json", "```\n{", "```{")

    def __init__(self):
        self.buffer = ""
        self.thinking = ""
        self.content_start = 0
        self.displayed_up_to = 0
        self.state = "init"

    def feed(self, chunk: str) -> Generator[tuple[str, str], None, None]:
        """Feed a chunk. Yields (event_type, data) pairs.

        event_type: "thinking", "text", "tool_buffering"
        """
        self.buffer += chunk

        # Re-evaluate state machine
        changed = True
        while changed:
            changed = False

            if self.state == "init":
                stripped = self.buffer.lstrip()
                if stripped.startswith("<think>"):
                    self.state = "thinking"
                    changed = True
                elif len(stripped) >= 7 or (stripped and not stripped.startswith("<")):
                    # Not a think block
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

            elif self.state == "detecting":
                content = self.buffer[self.content_start:].lstrip()
                if not content:
                    break
                if content.startswith("{"):
                    self.state = "tool"
                    # Don't yield anything - buffer the tool call
                elif any(content.startswith(p) for p in self._FENCE_PREFIXES):
                    # Fenced JSON block (e.g. ```json\n{"tool":...}\n```)
                    self.state = "tool"
                elif len(content) > 8:
                    # Need enough chars to distinguish ```json from normal text
                    self.state = "text"
                    changed = True

            elif self.state == "text":
                new = self.buffer[self.displayed_up_to:]
                if new:
                    self.displayed_up_to = len(self.buffer)
                    yield ("text", new)

            elif self.state == "tool":
                # Silently buffer
                break

    def finish(self) -> tuple[str, str, list[ToolCall]]:
        """Call when stream ends. Returns (thinking, content, tool_calls)."""
        content = self.buffer[self.content_start:].strip()

        if self.state == "thinking":
            # Stream ended inside thinking block (incomplete)
            think_start = self.buffer.find("<think>")
            if think_start >= 0:
                self.thinking = self.buffer[think_start + len("<think>"):].strip()
            content = ""

        tool_calls: list[ToolCall] = []
        if self.state == "tool":
            tool_calls = _parse_tool_calls_from_text(content)

        # Fallback: even in text state, check if the full content looks like
        # a tool call (weak LLMs may wrap JSON in prose or fences)
        if not tool_calls and content:
            tool_calls = _parse_tool_calls_from_text(content)

        return self.thinking, content, tool_calls

    @property
    def undisplayed_text(self) -> str:
        """Text that hasn't been yielded yet (for tool or detecting states)."""
        return self.buffer[self.displayed_up_to:].strip()


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class LLMClient:
    """Client for OpenAI-compatible LLM APIs (LM Studio, Ollama, etc.)."""

    def __init__(self, provider: ProviderConfig, timeout: float = 120):
        self.provider = provider
        # Local LLMs can take a while for first token, but stream chunks quickly
        self.client = httpx.Client(
            base_url=provider.base_url,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=30, read=300),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """Send a non-streaming chat completion request."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        start = time.monotonic()
        try:
            resp = self.client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return LLMResponse(content=f"[LLM API Error: {e}]", finish_reason="error")

        latency = (time.monotonic() - start) * 1000
        try:
            data = resp.json()
        except Exception:
            return LLMResponse(content="[LLM API Error: invalid JSON response]", finish_reason="error")

        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(content="[LLM API Error: empty choices]", finish_reason="error")
        choice = choices[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "") or ""
        finish = choice.get("finish_reason", "")

        thinking, clean_content = _extract_thinking(raw_content)

        tool_calls: list[ToolCall] = []
        native_calls = message.get("tool_calls", [])
        if native_calls:
            for tc in native_calls:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    name=func.get("name", ""),
                    arguments=args,
                    raw=json.dumps(tc),
                ))

        if not tool_calls and clean_content:
            tool_calls = _parse_tool_calls_from_text(clean_content)

        return LLMResponse(
            content=clean_content,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=data.get("usage", {}),
            model=data.get("model", model),
            raw_response=data,
            latency_ms=latency,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Generator[tuple[str, str], None, LLMResponse]:
        """Streaming chat completion. Yields (event_type, data) tuples.

        event_type: "thinking", "text", "tool_buffering"

        Returns final LLMResponse when generator is exhausted (use .send(None)
        or iterate to completion — the return value is accessible via
        StopIteration.value).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start = time.monotonic()
        processor = StreamProcessor()
        model_name = model

        try:
            with self.client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line.startswith("data: "):
                        continue
                    data_str = raw_line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    chunk = delta.get("content", "")
                    if not chunk:
                        continue

                    model_name = data.get("model", model)

                    # Process chunk through state machine
                    for event in processor.feed(chunk):
                        yield event

        except httpx.HTTPError as e:
            latency = (time.monotonic() - start) * 1000
            return LLMResponse(
                content=f"[LLM API Error: {e}]",
                finish_reason="error",
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return LLMResponse(
                content=f"[LLM Stream Error: {e}]",
                finish_reason="error",
                latency_ms=latency,
            )

        latency = (time.monotonic() - start) * 1000
        thinking, content, tool_calls = processor.finish()

        return LLMResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason="stop",
            model=model_name,
            latency_ms=latency,
        )

    def close(self):
        self.client.close()
