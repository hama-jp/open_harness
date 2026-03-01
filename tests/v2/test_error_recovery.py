"""Tests for error classification and recovery middleware."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from open_harness_v2.llm.error_recovery import ErrorClassifier, ErrorRecoveryMiddleware
from open_harness_v2.llm.middleware import LLMRequest
from open_harness_v2.types import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# ErrorClassifier tests
# ---------------------------------------------------------------------------

class TestErrorClassifier:
    def test_empty_response(self):
        ec = ErrorClassifier()
        assert ec.classify("some error", "") == "empty_response"
        assert ec.classify("", "   ") == "empty_response"

    def test_malformed_json(self):
        ec = ErrorClassifier()
        result = ec.classify("parse error", '{"tool": "shell", "args": {')
        assert result == "malformed_json"

    def test_wrong_tool_name(self):
        ec = ErrorClassifier(tool_names=["shell", "read_file"])
        result = ec.classify(
            "Unknown tool: shel. Please use a valid tool.",
            '{"tool": "shel", "args": {}}',
        )
        assert result == "wrong_tool_name"

    def test_missing_args(self):
        ec = ErrorClassifier()
        result = ec.classify(
            "Missing required argument: path",
            '{"tool": "read_file", "args": {}}',
        )
        assert result == "missing_args"

    def test_prose_wrapped(self):
        ec = ErrorClassifier()
        result = ec.classify(
            "No tool call detected",
            'I think I should use the shell. {"tool": "shell", "args": {"command": "ls"}}',
        )
        assert result == "prose_wrapped"

    def test_unknown(self):
        ec = ErrorClassifier()
        result = ec.classify("something went wrong", "Just some random text here.")
        assert result == "unknown"

    def test_valid_json_not_malformed(self):
        """Valid JSON with tool key should not be classified as malformed."""
        ec = ErrorClassifier()
        result = ec.classify(
            "No tool call detected",
            '{"tool": "shell", "args": {"command": "ls"}}',
        )
        # The JSON is valid, so it's not malformed — it's prose_wrapped
        # because the classifier checks the content pattern
        assert result == "prose_wrapped"


class TestSuggestTool:
    def test_substring_match(self):
        ec = ErrorClassifier(tool_names=["shell", "read_file", "write_file"])
        assert ec.suggest_tool("shel") == "shell"

    def test_superset_match(self):
        ec = ErrorClassifier(tool_names=["shell", "read_file"])
        assert ec.suggest_tool("shell_command") == "shell"

    def test_prefix_match(self):
        ec = ErrorClassifier(tool_names=["read_file", "write_file"])
        assert ec.suggest_tool("read") == "read_file"

    def test_no_match(self):
        ec = ErrorClassifier(tool_names=["shell", "read_file"])
        assert ec.suggest_tool("xyz_unknown") is None

    def test_empty_registry(self):
        ec = ErrorClassifier()
        assert ec.suggest_tool("shell") is None

    def test_dash_normalization(self):
        ec = ErrorClassifier(tool_names=["read_file"])
        assert ec.suggest_tool("read-file") == "read_file"


# ---------------------------------------------------------------------------
# ErrorRecoveryMiddleware tests
# ---------------------------------------------------------------------------

class TestErrorRecoveryMiddleware:
    async def test_no_recovery_needed(self):
        """When the response has tool calls, no recovery happens."""
        mw = ErrorRecoveryMiddleware(max_retries=3)

        async def next_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(
                content='{"tool": "shell"}',
                tool_calls=[ToolCall(name="shell", arguments={"command": "ls"})],
                finish_reason="stop",
            )

        req = LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m")
        result = await mw.process(req, next_fn)

        assert result.has_tool_calls
        assert result.tool_calls[0].name == "shell"

    async def test_recovery_on_empty_response(self):
        """Empty response triggers recovery."""
        call_count = 0

        async def next_fn(req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content="", finish_reason="stop")
            return LLMResponse(
                content="recovered",
                tool_calls=[ToolCall(name="shell", arguments={})],
                finish_reason="stop",
            )

        mw = ErrorRecoveryMiddleware(max_retries=3)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m")
        result = await mw.process(req, next_fn)

        assert call_count == 2
        assert result.has_tool_calls

    async def test_recovery_on_error_finish_reason(self):
        """Error finish_reason triggers recovery."""
        call_count = 0

        async def next_fn(req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content="[LLM API Error: timeout]",
                    finish_reason="error",
                )
            return LLMResponse(content="Hello!", finish_reason="stop")

        mw = ErrorRecoveryMiddleware(max_retries=3)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m")
        result = await mw.process(req, next_fn)

        assert call_count == 2
        assert result.content == "Hello!"

    async def test_max_retries_exhausted(self):
        """After max_retries, the last response is returned."""
        call_count = 0

        async def next_fn(req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            return LLMResponse(content="", finish_reason="stop")

        mw = ErrorRecoveryMiddleware(max_retries=2)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m")
        result = await mw.process(req, next_fn)

        # 1 initial + 2 retries = 3 calls
        assert call_count == 3
        assert result.content == ""

    async def test_prose_wrapped_no_retry(self):
        """prose_wrapped errors are not retried (parser handles them)."""
        call_count = 0

        async def next_fn(req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            return LLMResponse(
                content='Let me use shell. {"tool": "shell", "args": {"command": "ls"}}',
                finish_reason="stop",
            )

        mw = ErrorRecoveryMiddleware(max_retries=3)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m")
        result = await mw.process(req, next_fn)

        assert call_count == 1

    async def test_escalate_model_callback(self):
        """When escalation strategy fires, on_escalate is called."""
        call_count = 0
        models_used: list[str] = []

        def escalate(current_model: str, req: LLMRequest) -> str:
            return "big-model"

        async def next_fn(req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            models_used.append(req.model)
            if call_count <= 2:
                return LLMResponse(content="", finish_reason="stop")
            return LLMResponse(
                content="recovered",
                tool_calls=[ToolCall(name="shell", arguments={})],
                finish_reason="stop",
            )

        mw = ErrorRecoveryMiddleware(
            max_retries=3,
            strategies=["refine_prompt", "add_examples", "escalate_model"],
            on_escalate=escalate,
        )
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}], model="small")
        result = await mw.process(req, next_fn)

        assert result.has_tool_calls
        # 3rd call should use escalated model
        assert models_used[-1] == "big-model"

    async def test_refine_prompt_adds_correction(self):
        """refine_prompt strategy adds correction message."""
        messages_at_retry: list[list[dict]] = []

        async def next_fn(req: LLMRequest) -> LLMResponse:
            messages_at_retry.append(list(req.messages))
            if len(messages_at_retry) == 1:
                return LLMResponse(content="", finish_reason="stop")
            return LLMResponse(
                content="ok",
                tool_calls=[ToolCall(name="shell", arguments={})],
                finish_reason="stop",
            )

        mw = ErrorRecoveryMiddleware(
            max_retries=1,
            strategies=["refine_prompt"],
        )
        req = LLMRequest(
            messages=[{"role": "user", "content": "do something"}],
            model="m",
        )
        await mw.process(req, next_fn)

        # The second call should have extra messages
        assert len(messages_at_retry) == 2
        retry_msgs = messages_at_retry[1]
        # Should have: original user + assistant (failed) + user (correction)
        assert len(retry_msgs) == 3
        assert retry_msgs[1]["role"] == "assistant"
        assert retry_msgs[2]["role"] == "user"
        assert "try again" in retry_msgs[2]["content"].lower()

    async def test_normal_text_response_no_retry(self):
        """A normal text response (non-empty, no tool calls) is not retried."""
        call_count = 0

        async def next_fn(req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            return LLMResponse(
                content="Here is your answer: 42",
                finish_reason="stop",
            )

        mw = ErrorRecoveryMiddleware(max_retries=3)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m")
        result = await mw.process(req, next_fn)

        assert call_count == 1
        assert result.content == "Here is your answer: 42"
