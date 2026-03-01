"""Error classification and recovery middleware.

Adapted from v1 ``open_harness.llm.compensator`` (ErrorClassifier + Compensator).
Implements the ``Middleware`` protocol so it can be plugged into the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from open_harness_v2.types import LLMResponse

from .middleware import LLMRequest, Middleware, NextFn

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classifier
# ---------------------------------------------------------------------------

class ErrorClassifier:
    """Classify LLM failures to select the optimal retry strategy.

    Error classes:
      empty_response  - no content at all (escalate model immediately)
      malformed_json  - JSON syntax error (repairable without LLM retry)
      wrong_tool_name - tool name not in registry (fuzzy match suggestion)
      missing_args    - required arguments missing
      prose_wrapped   - JSON wrapped in prose (parser handles it, skip)
      unknown         - unrecognized error pattern
    """

    def __init__(self, tool_names: list[str] | None = None) -> None:
        self._tool_names: set[str] = set(tool_names) if tool_names else set()

    def classify(self, error_context: str, failed_response: str) -> str:
        """Return the error class string."""
        if not failed_response or not failed_response.strip():
            return "empty_response"

        # Check for JSON syntax issues
        stripped = failed_response.strip()
        if stripped.startswith("{"):
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                return "malformed_json"

        # Check for wrong tool name
        if self._tool_names and "Unknown tool" in error_context:
            return "wrong_tool_name"

        # Check for missing args
        if "missing" in error_context.lower() and "arg" in error_context.lower():
            return "missing_args"

        # Check if it looks like prose-wrapped JSON
        if re.search(r'\{[^}]*"tool"', failed_response):
            return "prose_wrapped"

        return "unknown"

    def suggest_tool(self, wrong_name: str) -> str | None:
        """Fuzzy-match a wrong tool name to the closest registered tool."""
        if not self._tool_names:
            return None
        wrong_lower = wrong_name.lower().replace("-", "_").replace(" ", "_")
        best_match: str | None = None
        best_score = 0
        for name in self._tool_names:
            name_lower = name.lower()
            if wrong_lower in name_lower or name_lower in wrong_lower:
                score = len(name_lower)
                if score > best_score:
                    best_score = score
                    best_match = name
            elif len(wrong_lower) >= 4 and wrong_lower[:4] == name_lower[:4]:
                score = 1
                if score > best_score:
                    best_score = score
                    best_match = name
        return best_match


# ---------------------------------------------------------------------------
# Recovery strategy helpers
# ---------------------------------------------------------------------------

def _refine_prompt(
    messages: list[dict[str, Any]],
    failed_response: str,
    error_context: str,
) -> list[dict[str, Any]]:
    """Append a correction message after the failed response."""
    correction = (
        f"Your previous response could not be processed. Error: {error_context}\n\n"
        f"Please try again. To use a tool, respond with ONLY:\n"
        f'{{"tool": "tool_name", "args": {{"param": "value"}}}}\n'
        f"To respond normally, just write text."
    )
    refined = list(messages)
    refined.append({"role": "assistant", "content": failed_response})
    refined.append({"role": "user", "content": correction})
    return refined


def _add_examples(
    messages: list[dict[str, Any]],
    failed_response: str,
    error_context: str,
) -> list[dict[str, Any]]:
    """Append concrete tool-usage examples after the failed response."""
    example_msg = (
        f"Error: {error_context}\n\nExamples of correct tool usage:\n"
        f'{{"tool": "shell", "args": {{"command": "ls -la"}}}}\n'
        f'{{"tool": "read_file", "args": {{"path": "src/main.py"}}}}\n'
        f"Normal response (no tool): Just write text.\nTry again."
    )
    refined = list(messages)
    refined.append({"role": "assistant", "content": failed_response})
    refined.append({"role": "user", "content": example_msg})
    return refined


# ---------------------------------------------------------------------------
# Error recovery middleware
# ---------------------------------------------------------------------------

# Default strategy order — mirrors v1 compensation config
_DEFAULT_STRATEGIES = ["refine_prompt", "add_examples", "escalate_model"]


@dataclass
class ErrorRecoveryMiddleware:
    """Middleware that retries failed LLM calls with classified strategies.

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts.
    tool_names:
        Known tool names (for classification and fuzzy matching).
    strategies:
        Ordered list of retry strategies to try:
        ``"refine_prompt"``, ``"add_examples"``, ``"escalate_model"``.
    on_escalate:
        Optional callback ``(current_model, request) -> new_model``.
        If ``None``, escalation is skipped.
    """

    max_retries: int = 3
    tool_names: list[str] = field(default_factory=list)
    strategies: list[str] = field(
        default_factory=lambda: list(_DEFAULT_STRATEGIES),
    )
    on_escalate: Any = None  # Callable[[str, LLMRequest], str] | None

    def __post_init__(self) -> None:
        self._classifier = ErrorClassifier(self.tool_names)

    async def process(
        self,
        request: LLMRequest,
        next_fn: NextFn,
    ) -> LLMResponse:
        """Call *next_fn*, and if the response indicates an error, retry."""
        response = await next_fn(request)

        for attempt in range(self.max_retries):
            if not self._needs_recovery(response):
                return response

            error_context = self._build_error_context(response)
            error_class = self._classifier.classify(
                error_context, response.content,
            )
            _logger.info(
                "Error recovery attempt %d/%d — class=%s",
                attempt + 1, self.max_retries, error_class,
            )

            # prose_wrapped is handled by the parser — no retry needed
            if error_class == "prose_wrapped":
                return response

            # Choose strategy
            strategy = self._pick_strategy(attempt, error_class)
            if strategy is None:
                return response

            request = self._apply_strategy(
                strategy, request, response, error_context, error_class,
            )
            response = await next_fn(request)

        return response

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _needs_recovery(self, response: LLMResponse) -> bool:
        """Decide whether the response should trigger recovery."""
        if response.finish_reason == "error":
            return True
        if response.has_tool_calls:
            return False
        if not response.content or not response.content.strip():
            return True
        return False

    def _build_error_context(self, response: LLMResponse) -> str:
        """Build a human-readable error context string."""
        if response.finish_reason == "error":
            return f"LLM error response: {response.content}"
        if not response.content or not response.content.strip():
            return "Empty response from LLM"
        return "No tool call detected in response"

    def _pick_strategy(
        self, attempt: int, error_class: str,
    ) -> str | None:
        """Select a strategy for the given attempt and error class."""
        # empty_response => jump to escalation if available
        if error_class == "empty_response":
            if "escalate_model" in self.strategies and self.on_escalate:
                return "escalate_model"
            # Fall through to default order
        # wrong_tool_name / missing_args => always refine first
        if error_class in ("wrong_tool_name", "missing_args") and attempt == 0:
            return "refine_prompt"
        # Default: use strategy order
        if attempt < len(self.strategies):
            return self.strategies[attempt]
        return None

    def _apply_strategy(
        self,
        strategy: str,
        request: LLMRequest,
        response: LLMResponse,
        error_context: str,
        error_class: str,
    ) -> LLMRequest:
        """Apply a recovery strategy and return a modified request."""
        if strategy == "refine_prompt":
            extra = ""
            if error_class == "wrong_tool_name":
                # Try fuzzy match
                suggestion = self._classifier.suggest_tool(
                    error_context.split("Unknown tool:")[-1]
                    .strip()
                    .split(".")[0]
                    if "Unknown tool:" in error_context
                    else "",
                )
                if suggestion:
                    extra = f" Did you mean '{suggestion}'?"
                if self._classifier._tool_names:
                    extra += (
                        f"\nAvailable tools: "
                        f"{', '.join(sorted(self._classifier._tool_names))}"
                    )
            messages = _refine_prompt(
                request.messages,
                response.content,
                error_context + extra,
            )
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

        if strategy == "add_examples":
            messages = _add_examples(
                request.messages, response.content, error_context,
            )
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

        if strategy == "escalate_model" and self.on_escalate:
            new_model = self.on_escalate(request.model, request)
            _logger.info("Escalating model: %s -> %s", request.model, new_model)
            return LLMRequest(
                messages=request.messages,
                model=new_model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                tools=request.tools,
                tool_choice=request.tool_choice,
                context_length=request.context_length,
                metadata=request.metadata,
            )

        # No-op fallback
        return request
