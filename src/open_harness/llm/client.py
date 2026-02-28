"""OpenAI-compatible LLM client for local and remote providers."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from open_harness.config import ModelConfig, ProviderConfig


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
        """Content with thinking tags stripped."""
        return self.content


def _extract_thinking(text: str) -> tuple[str, str]:
    """Extract <think>...</think> blocks from response text.

    Returns (thinking_content, remaining_text).
    """
    pattern = r"<think>(.*?)</think>"
    thinking_parts = re.findall(pattern, text, re.DOTALL)
    thinking = "\n".join(thinking_parts).strip()
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return thinking, cleaned


def _parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Try to extract tool calls from free-form text.

    Handles various formats local LLMs might produce:
    1. {"tool": "name", "args": {...}}
    2. ```json\n{"tool": "name", "args": {...}}\n```
    3. Native-style function calls embedded in text
    """
    calls = []

    # Try code block extraction first
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)

    # Also try bare JSON objects
    if not matches:
        bare_json_pattern = r'(\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{.*?\}\s*\})'
        matches = re.findall(bare_json_pattern, text, re.DOTALL)

    # Also try tool_call format
    if not matches:
        alt_pattern = r'(\{"tool_call"\s*:\s*\{.*?\}\s*\})'
        matches = re.findall(alt_pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            if "tool" in data and "args" in data:
                calls.append(ToolCall(
                    name=data["tool"],
                    arguments=data.get("args", {}),
                    raw=match,
                ))
            elif "tool_call" in data:
                tc = data["tool_call"]
                calls.append(ToolCall(
                    name=tc.get("name", tc.get("tool", "")),
                    arguments=tc.get("arguments", tc.get("args", {})),
                    raw=match,
                ))
        except (json.JSONDecodeError, KeyError):
            continue

    return calls


class LLMClient:
    """Client for OpenAI-compatible LLM APIs (LM Studio, Ollama, etc.)."""

    def __init__(self, provider: ProviderConfig, timeout: float = 120):
        self.provider = provider
        self.client = httpx.Client(
            base_url=provider.base_url,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
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
        """Send a chat completion request."""
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
            return LLMResponse(
                content=f"[LLM API Error: {e}]",
                finish_reason="error",
            )

        latency = (time.monotonic() - start) * 1000
        data = resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "") or ""
        finish = choice.get("finish_reason", "")

        # Extract thinking blocks
        thinking, clean_content = _extract_thinking(raw_content)

        # Parse tool calls - native format first
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

        # Fallback: try to parse tool calls from text
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

    def close(self):
        self.client.close()
