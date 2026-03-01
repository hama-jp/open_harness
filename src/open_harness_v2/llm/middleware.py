"""Middleware pipeline for composable LLM request/response processing.

The pipeline chains middleware functions around the core LLM client call,
allowing concerns like prompt optimization, error recovery, and logging
to be stacked declaratively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from open_harness_v2.types import LLMResponse

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request type
# ---------------------------------------------------------------------------

@dataclass
class LLMRequest:
    """Encapsulates everything needed for a single LLM call."""

    messages: list[dict[str, Any]]
    model: str
    max_tokens: int = 4096
    temperature: float = 0.3
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    context_length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Middleware protocol
# ---------------------------------------------------------------------------

# Type alias for the "next" function a middleware calls to continue the chain
NextFn = Callable[["LLMRequest"], Any]  # actually returns Awaitable[LLMResponse]


class Middleware(Protocol):
    """Protocol that all middleware must implement."""

    async def process(
        self,
        request: LLMRequest,
        next_fn: NextFn,
    ) -> LLMResponse:
        """Process the request, optionally modifying it, call *next_fn*, and
        optionally modify the response."""
        ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MiddlewarePipeline:
    """Ordered chain of middleware around an ``AsyncLLMClient``.

    Usage::

        from open_harness_v2.llm.client import AsyncLLMClient

        pipeline = MiddlewarePipeline(client)
        pipeline.use(PromptOptimizerMiddleware(...))
        pipeline.use(ErrorRecoveryMiddleware(...))
        response = await pipeline.execute(LLMRequest(...))

    Middleware is executed in the order registered (first added = outermost).
    """

    def __init__(self, client: Any) -> None:  # client: AsyncLLMClient
        self._client = client
        self._middlewares: list[Middleware] = []

    def use(self, middleware: Middleware) -> MiddlewarePipeline:
        """Register a middleware.  Returns ``self`` for chaining."""
        self._middlewares.append(middleware)
        return self

    async def execute(self, request: LLMRequest) -> LLMResponse:
        """Run the full middleware chain and return the final response."""

        async def _core(req: LLMRequest) -> LLMResponse:
            """The innermost call — delegates to the actual LLM client."""
            return await self._client.chat(
                messages=req.messages,
                model=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                tools=req.tools,
                tool_choice=req.tool_choice,
                context_length=req.context_length,
            )

        # Build the chain from inside out (last middleware wraps the core)
        chain = _core
        for mw in reversed(self._middlewares):
            chain = _wrap(mw, chain)

        return await chain(request)


def _wrap(
    middleware: Middleware,
    next_fn: Callable[[LLMRequest], Any],
) -> Callable[[LLMRequest], Any]:
    """Create a closure that calls ``middleware.process(req, next_fn)``."""

    async def _handler(request: LLMRequest) -> LLMResponse:
        return await middleware.process(request, next_fn)

    return _handler
