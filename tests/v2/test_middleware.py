"""Tests for the middleware pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from open_harness_v2.llm.middleware import LLMRequest, MiddlewarePipeline
from open_harness_v2.types import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RecordingMiddleware:
    """Middleware that records calls and optionally modifies request/response."""

    def __init__(
        self,
        name: str,
        modify_request: dict[str, Any] | None = None,
        modify_response: dict[str, Any] | None = None,
    ):
        self.name = name
        self.modify_request = modify_request
        self.modify_response = modify_response
        self.seen_requests: list[LLMRequest] = []

    async def process(self, request: LLMRequest, next_fn) -> LLMResponse:
        self.seen_requests.append(request)

        # Optionally modify the request
        if self.modify_request:
            for key, value in self.modify_request.items():
                setattr(request, key, value)

        response = await next_fn(request)

        # Optionally modify the response
        if self.modify_response:
            for key, value in self.modify_response.items():
                setattr(response, key, value)

        return response


class FakeClient:
    """Minimal fake AsyncLLMClient for pipeline tests."""

    def __init__(self, response: LLMResponse | None = None):
        self._response = response or LLMResponse(content="default response")
        self.received_kwargs: dict[str, Any] = {}

    async def chat(self, **kwargs) -> LLMResponse:
        self.received_kwargs = kwargs
        return self._response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLMRequest:
    def test_defaults(self):
        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
        )
        assert req.max_tokens == 4096
        assert req.temperature == 0.3
        assert req.tools is None
        assert req.tool_choice is None
        assert req.context_length == 0
        assert req.metadata == {}

    def test_custom_values(self):
        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="big-model",
            max_tokens=8192,
            temperature=0.7,
            tools=[{"function": {"name": "shell"}}],
            tool_choice="auto",
            context_length=32768,
            metadata={"key": "value"},
        )
        assert req.max_tokens == 8192
        assert req.temperature == 0.7
        assert req.context_length == 32768
        assert req.metadata["key"] == "value"


class TestMiddlewarePipeline:
    async def test_empty_pipeline(self):
        """Pipeline with no middleware calls client directly."""
        fake = FakeClient(LLMResponse(content="direct"))
        pipeline = MiddlewarePipeline(fake)

        result = await pipeline.execute(
            LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m"),
        )

        assert result.content == "direct"

    async def test_single_middleware(self):
        """A single middleware wraps the client call."""
        fake = FakeClient(LLMResponse(content="from client"))
        pipeline = MiddlewarePipeline(fake)

        mw = RecordingMiddleware("A", modify_response={"content": "modified by A"})
        pipeline.use(mw)

        result = await pipeline.execute(
            LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m"),
        )

        assert result.content == "modified by A"
        assert len(mw.seen_requests) == 1

    async def test_middleware_ordering(self):
        """Middleware is executed in registration order (first = outermost)."""
        order: list[str] = []

        class OrderTracker:
            def __init__(self, name: str):
                self.name = name

            async def process(self, request, next_fn):
                order.append(f"{self.name}_before")
                response = await next_fn(request)
                order.append(f"{self.name}_after")
                return response

        fake = FakeClient()
        pipeline = MiddlewarePipeline(fake)
        pipeline.use(OrderTracker("A"))
        pipeline.use(OrderTracker("B"))
        pipeline.use(OrderTracker("C"))

        await pipeline.execute(
            LLMRequest(messages=[], model="m"),
        )

        assert order == [
            "A_before", "B_before", "C_before",
            "C_after", "B_after", "A_after",
        ]

    async def test_middleware_modifies_request(self):
        """Middleware can modify the request before passing it down."""
        fake = FakeClient()
        pipeline = MiddlewarePipeline(fake)
        pipeline.use(RecordingMiddleware("temp", modify_request={"temperature": 0.9}))

        await pipeline.execute(
            LLMRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                temperature=0.3,
            ),
        )

        # The fake client should have received the modified temperature
        assert fake.received_kwargs["temperature"] == 0.9

    async def test_chaining_use(self):
        """``use()`` returns self for fluent chaining."""
        fake = FakeClient()
        pipeline = (
            MiddlewarePipeline(fake)
            .use(RecordingMiddleware("A"))
            .use(RecordingMiddleware("B"))
        )
        result = await pipeline.execute(
            LLMRequest(messages=[], model="m"),
        )
        assert result.content == "default response"

    async def test_request_passes_to_client(self):
        """Verify all request fields are forwarded to the client."""
        fake = FakeClient()
        pipeline = MiddlewarePipeline(fake)

        req = LLMRequest(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            max_tokens=2048,
            temperature=0.5,
            tools=[{"function": {"name": "shell"}}],
            tool_choice="auto",
            context_length=16384,
        )
        await pipeline.execute(req)

        assert fake.received_kwargs["model"] == "test-model"
        assert fake.received_kwargs["max_tokens"] == 2048
        assert fake.received_kwargs["temperature"] == 0.5
        assert fake.received_kwargs["tool_choice"] == "auto"
        assert fake.received_kwargs["context_length"] == 16384
