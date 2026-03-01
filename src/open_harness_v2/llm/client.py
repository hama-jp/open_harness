"""Async OpenAI-compatible LLM client for local and remote providers.

Async counterpart of ``open_harness.llm.client.LLMClient``.  Uses
``httpx.AsyncClient`` and exposes ``async def chat()`` /
``async def chat_stream()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator

import httpx

from open_harness_v2.config import ProfileSpec
from open_harness_v2.types import LLMResponse, ToolCall

from .response_parser import (
    NativeToolCallAccumulator,
    StreamProcessor,
    _extract_thinking,
    _parse_tool_calls_from_text,
)

_logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_BACKOFF_BASE = 1  # seconds -- exponential: 1, 2, 4

# OOM detection keywords in Ollama error responses
_OOM_KEYWORDS = (
    "out of memory",
    "oom",
    "exit status 2",
    "not enough memory",
    "alloc",
    "unexpectedly stopped",
    "resource limitations",
)


def _is_oom_error(body: str) -> bool:
    """Check if an Ollama error body indicates an out-of-memory condition."""
    lower = body.lower()
    return any(kw in lower for kw in _OOM_KEYWORDS)


class AsyncLLMClient:
    """Async client for OpenAI-compatible LLM APIs (LM Studio, Ollama, etc.)."""

    def __init__(self, profile: ProfileSpec, timeout: float = 120) -> None:
        self.profile = profile
        self._api_type = profile.api_type  # "openai" or "ollama"

        headers = {
            "Authorization": f"Bearer {profile.api_key}",
            "Content-Type": "application/json",
        }

        # For Ollama native API, strip /v1 from url
        base_url = profile.url
        if self._api_type == "ollama":
            base_url = base_url.rstrip("/").removesuffix("/v1")

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=30, read=300),
        )
        self._stream_client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=30, read=60),
        )

    # ------------------------------------------------------------------
    # Non-streaming chat
    # ------------------------------------------------------------------

    async def chat(
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
            return await self._chat_ollama(
                messages, model, max_tokens, temperature,
                tools, tool_choice, context_length,
            )
        return await self._chat_openai(
            messages, model, max_tokens, temperature,
            tools, tool_choice,
        )

    async def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
    ) -> LLMResponse:
        """Non-streaming chat via OpenAI-compatible API."""
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
        if self.profile.extra_params:
            payload.update(self.profile.extra_params)

        start = time.monotonic()
        resp = None
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    _logger.warning(
                        "LLM API returned %d (attempt %d/%d), retrying...",
                        resp.status_code, attempt + 1, _MAX_RETRIES,
                    )
                    await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                    continue
                resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                last_error = e
                _logger.warning(
                    "LLM API timeout (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, e,
                )
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
            except httpx.HTTPError as e:
                last_error = e
                if hasattr(e, "response") and e.response is not None:
                    if (
                        400 <= e.response.status_code < 500
                        and e.response.status_code != 429
                    ):
                        return LLMResponse(
                            content=f"[LLM API Error: {e}]",
                            finish_reason="error",
                        )
                _logger.warning(
                    "LLM API error (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, e,
                )
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
        else:
            err = last_error or "exhausted retries"
            return LLMResponse(
                content=f"[LLM API Error: {err}]", finish_reason="error",
            )
        if resp is None:
            return LLMResponse(
                content="[LLM API Error: no response]", finish_reason="error",
            )

        latency = (time.monotonic() - start) * 1000
        try:
            data = resp.json()
        except Exception:
            return LLMResponse(
                content="[LLM API Error: invalid JSON response]",
                finish_reason="error",
            )

        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(
                content="[LLM API Error: empty choices]", finish_reason="error",
            )
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
                tool_calls.append(
                    ToolCall(
                        name=func.get("name", ""),
                        arguments=args,
                        raw=json.dumps(tc),
                    )
                )

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
    # Ollama native API (/api/chat)
    # ------------------------------------------------------------------

    async def _chat_ollama(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        context_length: int,
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
        if self.profile.extra_params:
            payload.update(self.profile.extra_params)

        start = time.monotonic()
        resp = None
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.post("/api/chat", json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
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
                                current_ctx, new_ctx,
                            )
                        else:
                            _logger.warning(
                                "Ollama OOM at minimum num_ctx, retrying...",
                            )
                    else:
                        _logger.warning(
                            "Ollama API returned %d (attempt %d/%d), retrying...",
                            resp.status_code, attempt + 1, _MAX_RETRIES,
                        )
                    await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                    continue
                resp.raise_for_status()
                break
            except httpx.TimeoutException as e:
                last_error = e
                _logger.warning(
                    "Ollama API timeout (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, e,
                )
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
            except httpx.HTTPError as e:
                last_error = e
                if hasattr(e, "response") and e.response is not None:
                    if (
                        400 <= e.response.status_code < 500
                        and e.response.status_code != 429
                    ):
                        return LLMResponse(
                            content=f"[Ollama API Error: {e}]",
                            finish_reason="error",
                        )
                _logger.warning(
                    "Ollama API error (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, e,
                )
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
        else:
            err = last_error or "exhausted retries"
            return LLMResponse(
                content=f"[Ollama API Error: {err}]", finish_reason="error",
            )
        if resp is None:
            return LLMResponse(
                content="[Ollama API Error: no response]", finish_reason="error",
            )

        latency = (time.monotonic() - start) * 1000
        try:
            data = resp.json()
        except Exception:
            return LLMResponse(
                content="[Ollama API Error: invalid JSON]", finish_reason="error",
            )

        message = data.get("message", {})
        content = message.get("content", "")
        thinking, clean_content = _extract_thinking(content)

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
                tool_calls.append(
                    ToolCall(
                        name=func.get("name", ""),
                        arguments=args,
                        raw=json.dumps(tc),
                    )
                )

        if not tool_calls and clean_content:
            tool_calls = _parse_tool_calls_from_text(clean_content)

        usage: dict[str, int] = {}
        if "prompt_eval_count" in data:
            usage["prompt_tokens"] = data["prompt_eval_count"]
        if "eval_count" in data:
            usage["completion_tokens"] = data["eval_count"]
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0,
            )

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

    # ------------------------------------------------------------------
    # Streaming: OpenAI-compatible
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        context_length: int = 0,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Streaming chat completion.  Yields ``(event_type, data)`` tuples.

        event_type: ``"thinking"``, ``"text"``, ``"tool_buffering"``

        After exhaustion, call ``last_response`` to get the final
        ``LLMResponse``.
        """
        if self._api_type == "ollama":
            async for event in self._chat_stream_ollama(
                messages, model, max_tokens, temperature, context_length,
            ):
                yield event
            return

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if self.profile.extra_params:
            payload.update(self.profile.extra_params)

        start = time.monotonic()
        processor = StreamProcessor()
        native_tc = NativeToolCallAccumulator()
        model_name = model
        stream_usage: dict[str, int] = {}
        chunks_yielded = False

        for attempt in range(_MAX_RETRIES):
            try:
                async with self._stream_client.stream(
                    "POST", "/chat/completions", json=payload,
                ) as resp:
                    if resp.status_code in (429, 500, 502, 503, 504):
                        _logger.warning(
                            "LLM stream API returned %d (attempt %d/%d), retrying...",
                            resp.status_code, attempt + 1, _MAX_RETRIES,
                        )
                        await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                        processor = StreamProcessor()
                        continue
                    resp.raise_for_status()

                    async for raw_line in resp.aiter_lines():
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

                        native_tc.feed(delta)

                        chunk = delta.get("content", "")
                        if not chunk:
                            continue

                        model_name = data.get("model", model)

                        if "usage" in data and data["usage"]:
                            stream_usage = data["usage"]

                        for event in processor.feed(chunk):
                            chunks_yielded = True
                            yield event
                break  # success
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if chunks_yielded:
                    latency = (time.monotonic() - start) * 1000
                    self._last_response = LLMResponse(
                        content=f"[LLM API Error: stream interrupted: {e}]",
                        finish_reason="error",
                        latency_ms=latency,
                    )
                    return
                _logger.warning(
                    "LLM stream error (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, e,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                    processor = StreamProcessor()
                    native_tc = NativeToolCallAccumulator()
                    continue
                latency = (time.monotonic() - start) * 1000
                self._last_response = LLMResponse(
                    content=f"[LLM API Error: {e}]",
                    finish_reason="error",
                    latency_ms=latency,
                )
                return
            except httpx.HTTPError as e:
                latency = (time.monotonic() - start) * 1000
                self._last_response = LLMResponse(
                    content=f"[LLM API Error: {e}]",
                    finish_reason="error",
                    latency_ms=latency,
                )
                return

        latency = (time.monotonic() - start) * 1000
        thinking, content, tool_calls = processor.finish()

        if native_tc.has_calls():
            tool_calls = native_tc.finalize() + tool_calls

        self._last_response = LLMResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage=stream_usage,
            model=model_name,
            latency_ms=latency,
        )

    # ------------------------------------------------------------------
    # Streaming: Ollama native
    # ------------------------------------------------------------------

    async def _chat_stream_ollama(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        context_length: int,
    ) -> AsyncGenerator[tuple[str, str], None]:
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
        if self.profile.extra_params:
            payload.update(self.profile.extra_params)

        start = time.monotonic()
        processor = StreamProcessor()
        model_name = model
        stream_usage: dict[str, int] = {}

        for attempt in range(_MAX_RETRIES):
            try:
                async with self._stream_client.stream(
                    "POST", "/api/chat", json=payload,
                ) as resp:
                    if resp.status_code in (429, 500, 502, 503, 504):
                        body = ""
                        try:
                            body = (await resp.aread()).decode(errors="replace")
                        except Exception:
                            pass
                        if resp.status_code == 500 and _is_oom_error(body):
                            current_ctx = payload.get("options", {}).get(
                                "num_ctx", 0,
                            )
                            if current_ctx > 8192:
                                new_ctx = current_ctx // 2
                                payload.setdefault("options", {})["num_ctx"] = new_ctx
                                _logger.warning(
                                    "Ollama OOM with num_ctx=%d, reducing to %d",
                                    current_ctx, new_ctx,
                                )
                            else:
                                _logger.warning(
                                    "Ollama OOM at minimum num_ctx, retrying...",
                                )
                        else:
                            _logger.warning(
                                "Ollama stream returned %d, retrying...",
                                resp.status_code,
                            )
                        await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                        processor = StreamProcessor()
                        continue

                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        msg = data.get("message", {})
                        chunk = msg.get("content", "")
                        model_name = data.get("model", model)

                        if data.get("done"):
                            if "prompt_eval_count" in data:
                                stream_usage["prompt_tokens"] = data[
                                    "prompt_eval_count"
                                ]
                            if "eval_count" in data:
                                stream_usage["completion_tokens"] = data[
                                    "eval_count"
                                ]
                            if stream_usage:
                                stream_usage["total_tokens"] = (
                                    stream_usage.get("prompt_tokens", 0)
                                    + stream_usage.get("completion_tokens", 0)
                                )
                            break

                        if not chunk:
                            continue

                        for event in processor.feed(chunk):
                            yield event
                break  # success
            except httpx.TimeoutException:
                _logger.warning(
                    "Ollama stream timeout (attempt %d/%d)",
                    attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                processor = StreamProcessor()
            except httpx.HTTPError as e:
                if hasattr(e, "response") and e.response is not None:
                    if (
                        400 <= e.response.status_code < 500
                        and e.response.status_code != 429
                    ):
                        self._last_response = LLMResponse(
                            content=f"[Ollama Error: {e}]",
                            finish_reason="error",
                        )
                        return
                _logger.warning(
                    "Ollama stream error (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, e,
                )
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                processor = StreamProcessor()
        else:
            self._last_response = LLMResponse(
                content="[Ollama Error: exhausted retries]",
                finish_reason="error",
            )
            return

        latency = (time.monotonic() - start) * 1000
        thinking, content, tool_calls = processor.finish()

        self._last_response = LLMResponse(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason="stop",
            usage=stream_usage,
            model=model_name,
            latency_ms=latency,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def last_response(self) -> LLMResponse | None:
        """The final LLMResponse from the last ``chat_stream()`` call."""
        return getattr(self, "_last_response", None)

    async def close(self) -> None:
        """Close underlying HTTP clients."""
        await self._client.aclose()
        await self._stream_client.aclose()
