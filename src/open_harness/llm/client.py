"""OpenAI-compatible LLM client for local and remote providers."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import httpx

from open_harness.config import ProviderConfig

_logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_BACKOFF_BASE = 1  # seconds — exponential: 1, 2, 4

# OOM detection keywords in Ollama error responses
_OOM_KEYWORDS = (
    "out of memory", "oom", "exit status 2", "not enough memory", "alloc",
    "unexpectedly stopped", "resource limitations",
)


def _is_oom_error(body: str) -> bool:
    """Check if an Ollama error body indicates an out-of-memory condition."""
    lower = body.lower()
    return any(kw in lower for kw in _OOM_KEYWORDS)


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
# Native tool call accumulator for streaming responses
# ---------------------------------------------------------------------------

class NativeToolCallAccumulator:
    """Accumulate native function-calling tool_calls from streaming deltas.

    OpenAI-compatible providers send tool calls as incremental chunks:
    each chunk has an ``index``, a ``function.name`` (first chunk only),
    and ``function.arguments`` fragments that must be concatenated.
    """

    def __init__(self):
        self._calls: dict[int, dict[str, str]] = {}  # index -> {name, arguments}

    def feed(self, delta: dict[str, Any]):
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
                result.append(ToolCall(
                    name=name,
                    arguments=args,
                    raw=json.dumps({"function": {"name": name, "arguments": raw_args}}),
                ))
        return result


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

    # Yield a partial thinking event every N characters so the UI stays alive
    _THINKING_YIELD_INTERVAL = 200

    def __init__(self):
        self.buffer = ""
        self.thinking = ""
        self.content_start = 0
        self.displayed_up_to = 0
        self.state = "init"
        self._thinking_yielded_at = 0

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
                else:
                    # Yield partial thinking so the UI stays responsive
                    think_start = self.buffer.find("<think>")
                    if think_start >= 0:
                        partial_len = len(self.buffer) - think_start - len("<think>")
                        if partial_len - self._thinking_yielded_at >= self._THINKING_YIELD_INTERVAL:
                            self._thinking_yielded_at = partial_len
                            snippet = self.buffer[think_start + len("<think>"):].strip()
                            # Show last line as progress
                            last_line = snippet.split("\n")[-1][:80]
                            yield ("thinking", last_line)

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
        self._api_type = provider.api_type  # "openai" or "ollama"
        _headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }

        # For Ollama native API, strip /v1 from base_url
        base_url = provider.base_url
        if self._api_type == "ollama":
            base_url = base_url.rstrip("/").removesuffix("/v1")

        # Non-streaming: generous read timeout for slow models
        self.client = httpx.Client(
            base_url=base_url,
            headers=_headers,
            timeout=httpx.Timeout(timeout, connect=30, read=300),
        )
        # Streaming: shorter read timeout — chunks arrive frequently once
        # generation starts, so a 60s gap means something is wrong.
        self._stream_client = httpx.Client(
            base_url=base_url,
            headers=_headers,
            timeout=httpx.Timeout(timeout, connect=30, read=60),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        context_length: int = 0,
    ) -> LLMResponse:
        """Send a non-streaming chat completion request."""
        if self._api_type == "ollama":
            return self._chat_ollama(messages, model, max_tokens, temperature,
                                     tools, tool_choice, context_length)
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
        # Merge provider-specific extra params
        if self.provider.extra_params:
            payload.update(self.provider.extra_params)

        start = time.monotonic()
        resp = None
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.client.post("/chat/completions", json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    _logger.warning(
                        "LLM API returned %d (attempt %d/%d), retrying...",
                        resp.status_code, attempt + 1, _MAX_RETRIES)
                    time.sleep(_BACKOFF_BASE * (2 ** attempt))
                    continue
                resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                last_error = e
                _logger.warning(
                    "LLM API timeout (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, e)
                time.sleep(_BACKOFF_BASE * (2 ** attempt))
            except httpx.HTTPError as e:
                last_error = e
                # Don't retry client errors (4xx) except 429
                if hasattr(e, "response") and e.response is not None:
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        return LLMResponse(content=f"[LLM API Error: {e}]", finish_reason="error")
                _logger.warning(
                    "LLM API error (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, e)
                time.sleep(_BACKOFF_BASE * (2 ** attempt))
        else:
            err = last_error or "exhausted retries"
            return LLMResponse(content=f"[LLM API Error: {err}]", finish_reason="error")
        if resp is None:
            return LLMResponse(content="[LLM API Error: no response]", finish_reason="error")

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

    # ------------------------------------------------------------------
    # Ollama native API (/api/chat) — supports think: false properly
    # ------------------------------------------------------------------

    def _chat_ollama(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        context_length: int = 0,
    ) -> LLMResponse:
        """Non-streaming chat via Ollama native API."""
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if context_length > 0:
            options["num_ctx"] = context_length
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if tools:
            payload["tools"] = tools
        if self.provider.extra_params:
            payload.update(self.provider.extra_params)

        start = time.monotonic()
        resp = None
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.client.post("/api/chat", json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    # Check for OOM — reduce num_ctx and retry
                    body = ""
                    try:
                        body = resp.text
                    except Exception:
                        pass
                    if resp.status_code == 500 and _is_oom_error(body):
                        current_ctx = payload.get("options", {}).get("num_ctx", 0)
                        if current_ctx > 8192:
                            new_ctx = current_ctx // 2
                            payload.setdefault("options", {})["num_ctx"] = new_ctx
                            _logger.warning(
                                "Ollama OOM with num_ctx=%d, reducing to %d",
                                current_ctx, new_ctx)
                        else:
                            _logger.warning("Ollama OOM at minimum num_ctx, retrying...")
                    else:
                        _logger.warning(
                            "Ollama API returned %d (attempt %d/%d), retrying...",
                            resp.status_code, attempt + 1, _MAX_RETRIES)
                    time.sleep(_BACKOFF_BASE * (2 ** attempt))
                    continue
                resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                last_error = e
                _logger.warning("Ollama API timeout (attempt %d/%d): %s",
                                attempt + 1, _MAX_RETRIES, e)
                time.sleep(_BACKOFF_BASE * (2 ** attempt))
            except httpx.HTTPError as e:
                last_error = e
                if hasattr(e, "response") and e.response is not None:
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        return LLMResponse(content=f"[Ollama API Error: {e}]", finish_reason="error")
                _logger.warning("Ollama API error (attempt %d/%d): %s",
                                attempt + 1, _MAX_RETRIES, e)
                time.sleep(_BACKOFF_BASE * (2 ** attempt))
        else:
            err = last_error or "exhausted retries"
            return LLMResponse(content=f"[Ollama API Error: {err}]", finish_reason="error")
        if resp is None:
            return LLMResponse(content="[Ollama API Error: no response]", finish_reason="error")

        latency = (time.monotonic() - start) * 1000
        try:
            data = resp.json()
        except Exception:
            return LLMResponse(content="[Ollama API Error: invalid JSON]", finish_reason="error")

        message = data.get("message", {})
        content = message.get("content", "")
        thinking, clean_content = _extract_thinking(content)

        # Native tool calls from Ollama
        tool_calls: list[ToolCall] = []
        native_calls = message.get("tool_calls", [])
        if native_calls:
            for tc in native_calls:
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(ToolCall(
                    name=func.get("name", ""),
                    arguments=args,
                    raw=json.dumps(tc),
                ))

        if not tool_calls and clean_content:
            tool_calls = _parse_tool_calls_from_text(clean_content)

        # Build usage from Ollama-native fields
        usage: dict[str, int] = {}
        if "prompt_eval_count" in data:
            usage["prompt_tokens"] = data["prompt_eval_count"]
        if "eval_count" in data:
            usage["completion_tokens"] = data["eval_count"]
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        return LLMResponse(
            content=clean_content,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason=data.get("done_reason", "stop"),
            usage=usage,
            model=data.get("model", model),
            raw_response=data,
            latency_ms=latency,
        )

    def _chat_stream_ollama(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        context_length: int = 0,
    ) -> Generator[tuple[str, str], None, LLMResponse]:
        """Streaming chat via Ollama native API."""
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if context_length > 0:
            options["num_ctx"] = context_length
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        if self.provider.extra_params:
            payload.update(self.provider.extra_params)

        start = time.monotonic()
        processor = StreamProcessor()
        model_name = model
        stream_usage: dict[str, int] = {}

        for attempt in range(_MAX_RETRIES):
            try:
                with self._stream_client.stream(
                    "POST", "/api/chat", json=payload
                ) as resp:
                    if resp.status_code in (429, 500, 502, 503, 504):
                        # Check for OOM — reduce num_ctx and retry
                        body = ""
                        try:
                            body = resp.read().decode(errors="replace")
                        except Exception:
                            pass
                        if resp.status_code == 500 and _is_oom_error(body):
                            current_ctx = payload.get("options", {}).get("num_ctx", 0)
                            if current_ctx > 8192:
                                new_ctx = current_ctx // 2
                                payload.setdefault("options", {})["num_ctx"] = new_ctx
                                _logger.warning(
                                    "Ollama OOM with num_ctx=%d, reducing to %d",
                                    current_ctx, new_ctx)
                            else:
                                _logger.warning("Ollama OOM at minimum num_ctx, retrying...")
                        else:
                            _logger.warning("Ollama stream returned %d, retrying...",
                                            resp.status_code)
                        time.sleep(_BACKOFF_BASE * (2 ** attempt))
                        processor = StreamProcessor()
                        continue

                    for line in resp.iter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        msg = data.get("message", {})
                        chunk = msg.get("content", "")
                        model_name = data.get("model", model)

                        # Final chunk — extract usage
                        if data.get("done"):
                            if "prompt_eval_count" in data:
                                stream_usage["prompt_tokens"] = data["prompt_eval_count"]
                            if "eval_count" in data:
                                stream_usage["completion_tokens"] = data["eval_count"]
                            if stream_usage:
                                stream_usage["total_tokens"] = (
                                    stream_usage.get("prompt_tokens", 0) +
                                    stream_usage.get("completion_tokens", 0))
                            break

                        if not chunk:
                            continue

                        for event in processor.feed(chunk):
                            yield event
                break  # success
            except httpx.TimeoutException:
                _logger.warning("Ollama stream timeout (attempt %d/%d)",
                                attempt + 1, _MAX_RETRIES)
                time.sleep(_BACKOFF_BASE * (2 ** attempt))
                processor = StreamProcessor()
            except httpx.HTTPError as e:
                if hasattr(e, "response") and e.response is not None:
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        return LLMResponse(content=f"[Ollama Error: {e}]", finish_reason="error")
                _logger.warning("Ollama stream error (attempt %d/%d): %s",
                                attempt + 1, _MAX_RETRIES, e)
                time.sleep(_BACKOFF_BASE * (2 ** attempt))
                processor = StreamProcessor()
        else:
            return LLMResponse(content="[Ollama Error: exhausted retries]", finish_reason="error")

        latency = (time.monotonic() - start) * 1000
        thinking, content, tool_calls = processor.finish()

        return LLMResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage=stream_usage,
            model=model_name,
            latency_ms=latency,
        )

    # ------------------------------------------------------------------
    # OpenAI-compatible streaming
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        context_length: int = 0,
    ) -> Generator[tuple[str, str], None, LLMResponse]:
        """Streaming chat completion. Yields (event_type, data) tuples.

        event_type: "thinking", "text", "tool_buffering"

        Returns final LLMResponse when generator is exhausted (use .send(None)
        or iterate to completion — the return value is accessible via
        StopIteration.value).
        """
        if self._api_type == "ollama":
            return (yield from self._chat_stream_ollama(
                messages, model, max_tokens, temperature, context_length))
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        # Merge provider-specific extra params (e.g. think: false for Ollama)
        if self.provider.extra_params:
            payload.update(self.provider.extra_params)

        start = time.monotonic()
        processor = StreamProcessor()
        native_tc = NativeToolCallAccumulator()
        model_name = model
        stream_usage: dict[str, int] = {}

        try:
            # Retry only the connection phase. Once we start yielding chunks,
            # we cannot retry without corrupting downstream consumers.
            _chunks_yielded = False
            success = False
            for _attempt in range(_MAX_RETRIES):
                try:
                    with self._stream_client.stream("POST", "/chat/completions", json=payload) as resp:
                        if resp.status_code in (429, 500, 502, 503, 504):
                            _logger.warning(
                                "LLM stream API returned %d (attempt %d/%d), retrying...",
                                resp.status_code, _attempt + 1, _MAX_RETRIES)
                            time.sleep(_BACKOFF_BASE * (2 ** _attempt))
                            processor = StreamProcessor()
                            continue
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

                            # Issue 1: accumulate native tool call chunks
                            native_tc.feed(delta)

                            chunk = delta.get("content", "")
                            if not chunk:
                                continue

                            model_name = data.get("model", model)

                            # Issue 7: capture usage from final chunk
                            if "usage" in data and data["usage"]:
                                stream_usage = data["usage"]

                            for event in processor.feed(chunk):
                                _chunks_yielded = True
                                yield event
                    success = True
                    break
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    # If we already yielded chunks, retrying would corrupt output
                    if _chunks_yielded:
                        latency = (time.monotonic() - start) * 1000
                        return LLMResponse(
                            content=f"[LLM API Error: stream interrupted after partial output: {e}]",
                            finish_reason="error",
                            latency_ms=latency,
                        )
                    _logger.warning(
                        "LLM stream error (attempt %d/%d): %s",
                        _attempt + 1, _MAX_RETRIES, e)
                    if _attempt < _MAX_RETRIES - 1:
                        time.sleep(_BACKOFF_BASE * (2 ** _attempt))
                        processor = StreamProcessor()
                        native_tc = NativeToolCallAccumulator()
                        continue
                    latency = (time.monotonic() - start) * 1000
                    return LLMResponse(
                        content=f"[LLM API Error: {e}]",
                        finish_reason="error",
                        latency_ms=latency,
                    )

            if not success:
                latency = (time.monotonic() - start) * 1000
                return LLMResponse(
                    content="[LLM API Error: exhausted retries]",
                    finish_reason="error",
                    latency_ms=latency,
                )

        except KeyboardInterrupt:
            latency = (time.monotonic() - start) * 1000
            thinking, content, tool_calls = processor.finish()
            return LLMResponse(
                content=content or "[Interrupted]",
                thinking=thinking,
                finish_reason="interrupted",
                model=model_name,
                latency_ms=latency,
            )
        except httpx.HTTPError as e:
            latency = (time.monotonic() - start) * 1000
            return LLMResponse(
                content=f"[LLM API Error: {e}]",
                finish_reason="error",
                latency_ms=latency,
            )
        except (json.JSONDecodeError, ValueError, OSError) as e:
            latency = (time.monotonic() - start) * 1000
            return LLMResponse(
                content=f"[LLM Stream Error: {type(e).__name__}: {e}]",
                finish_reason="error",
                latency_ms=latency,
            )

        latency = (time.monotonic() - start) * 1000
        thinking, content, tool_calls = processor.finish()

        # Issue 1: merge native function-calling tool calls
        if native_tc.has_calls():
            tool_calls = native_tc.finalize() + tool_calls

        return LLMResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage=stream_usage,
            model=model_name,
            latency_ms=latency,
        )

    def close(self):
        self.client.close()
        self._stream_client.close()
