"""Tests for AsyncLLMClient with mocked httpx responses."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from open_harness_v2.config import ProfileSpec
from open_harness_v2.llm.client import AsyncLLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def openai_profile() -> ProfileSpec:
    return ProfileSpec(
        provider="lmstudio",
        url="http://localhost:1234/v1",
        api_key="test-key",
        api_type="openai",
        models=["small-model", "large-model"],
    )


@pytest.fixture
def ollama_profile() -> ProfileSpec:
    return ProfileSpec(
        provider="ollama",
        url="http://localhost:11434/v1",
        api_key="no-key",
        api_type="ollama",
        models=["qwen3-8b"],
    )


def _make_openai_response(
    content: str = "Hello!",
    finish_reason: str = "stop",
    tool_calls: list | None = None,
    model: str = "small-model",
) -> dict:
    """Build a mock OpenAI-compatible chat completion JSON."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "model": model,
    }


def _make_ollama_response(
    content: str = "Hello!",
    done_reason: str = "stop",
    tool_calls: list | None = None,
    model: str = "qwen3-8b",
) -> dict:
    """Build a mock Ollama /api/chat response JSON."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "message": message,
        "done_reason": done_reason,
        "model": model,
        "prompt_eval_count": 15,
        "eval_count": 25,
    }


def _mock_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response with JSON body."""
    resp = httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "http://test"),
    )
    return resp


# ---------------------------------------------------------------------------
# Tests: OpenAI-compatible chat
# ---------------------------------------------------------------------------

class TestOpenAIChat:
    async def test_basic_chat(self, openai_profile: ProfileSpec):
        client = AsyncLLMClient(openai_profile)
        mock_resp = _mock_response(_make_openai_response(content="World"))

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model="small-model",
            )

        assert result.content == "World"
        assert result.finish_reason == "stop"
        assert result.model == "small-model"
        assert result.usage["total_tokens"] == 30
        assert result.latency_ms > 0
        await client.close()

    async def test_tool_calls_from_native(self, openai_profile: ProfileSpec):
        tc_data = [
            {
                "function": {
                    "name": "shell",
                    "arguments": '{"command": "ls"}',
                },
            }
        ]
        client = AsyncLLMClient(openai_profile)
        mock_resp = _mock_response(
            _make_openai_response(content="", tool_calls=tc_data),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await client.chat(
                messages=[{"role": "user", "content": "list files"}],
                model="small-model",
            )

        assert result.has_tool_calls
        assert result.tool_calls[0].name == "shell"
        assert result.tool_calls[0].arguments == {"command": "ls"}
        await client.close()

    async def test_tool_calls_from_text(self, openai_profile: ProfileSpec):
        """When the model outputs a JSON tool call in content text."""
        content = '{"tool": "read_file", "args": {"path": "main.py"}}'
        client = AsyncLLMClient(openai_profile)
        mock_resp = _mock_response(_make_openai_response(content=content))

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await client.chat(
                messages=[{"role": "user", "content": "read main.py"}],
                model="small-model",
            )

        assert result.has_tool_calls
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "main.py"}
        await client.close()

    async def test_thinking_extraction(self, openai_profile: ProfileSpec):
        content = "<think>Let me think about this.</think>The answer is 42."
        client = AsyncLLMClient(openai_profile)
        mock_resp = _mock_response(_make_openai_response(content=content))

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await client.chat(
                messages=[{"role": "user", "content": "What is the answer?"}],
                model="small-model",
            )

        assert result.thinking == "Let me think about this."
        assert result.content == "The answer is 42."
        await client.close()

    async def test_timeout_retries_then_error(self, openai_profile: ProfileSpec):
        client = AsyncLLMClient(openai_profile)

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ReadTimeout("timed out")
            # Patch asyncio.sleep to not actually sleep
            with patch("open_harness_v2.llm.client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.chat(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="small-model",
                )

        assert result.finish_reason == "error"
        assert "timed out" in result.content
        await client.close()

    async def test_http_error_no_retry_on_4xx(self, openai_profile: ProfileSpec):
        """4xx errors (except 429) should not be retried."""
        client = AsyncLLMClient(openai_profile)
        error_response = httpx.Response(
            status_code=400,
            json={"error": "bad request"},
            request=httpx.Request("POST", "http://test"),
        )
        exc = httpx.HTTPStatusError(
            "bad request",
            request=httpx.Request("POST", "http://test"),
            response=error_response,
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = exc
            result = await client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model="small-model",
            )

        assert result.finish_reason == "error"
        # Should only be called once (no retry)
        assert mock_post.call_count == 1
        await client.close()


# ---------------------------------------------------------------------------
# Tests: Ollama native chat
# ---------------------------------------------------------------------------

class TestOllamaChat:
    async def test_basic_ollama_chat(self, ollama_profile: ProfileSpec):
        client = AsyncLLMClient(ollama_profile)
        mock_resp = _mock_response(
            _make_ollama_response(content="Ollama says hi"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model="qwen3-8b",
            )

        assert result.content == "Ollama says hi"
        assert result.usage["prompt_tokens"] == 15
        assert result.usage["completion_tokens"] == 25
        assert result.usage["total_tokens"] == 40
        # Verify it posted to /api/chat (Ollama endpoint)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "/api/chat" or call_args[1].get("url") == "/api/chat"
        await client.close()

    async def test_ollama_tool_calls(self, ollama_profile: ProfileSpec):
        tc_data = [
            {
                "function": {
                    "name": "shell",
                    "arguments": {"command": "pwd"},
                },
            }
        ]
        client = AsyncLLMClient(ollama_profile)
        mock_resp = _mock_response(
            _make_ollama_response(content="", tool_calls=tc_data),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await client.chat(
                messages=[{"role": "user", "content": "pwd"}],
                model="qwen3-8b",
            )

        assert result.has_tool_calls
        assert result.tool_calls[0].name == "shell"
        await client.close()

    async def test_ollama_context_length(self, ollama_profile: ProfileSpec):
        client = AsyncLLMClient(ollama_profile)
        mock_resp = _mock_response(_make_ollama_response())

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model="qwen3-8b",
                context_length=32768,
            )

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["num_ctx"] == 32768
        await client.close()
