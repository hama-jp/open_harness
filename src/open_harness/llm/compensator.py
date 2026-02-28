"""Weak LLM compensation engine.

Strategies to help local LLMs succeed at tool calling and complex tasks:
1. Parse fallback - extract tool calls from messy output
2. Prompt refinement - add examples, rephrase, add CoT
3. Model escalation - try larger models
4. Task decomposition - break complex tasks into steps
5. Self-verification - have LLM check its own output
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from open_harness.config import CompensationConfig

logger = logging.getLogger(__name__)


@dataclass
class CompensationResult:
    """Result of a compensation attempt."""
    strategy: str
    success: bool
    modified_messages: list[dict[str, Any]] | None = None
    escalated_tier: str | None = None
    notes: str = ""


class Compensator:
    """Engine that compensates for weak LLM responses."""

    def __init__(self, config: CompensationConfig):
        self.config = config
        self._attempt_count = 0

    def reset(self):
        self._attempt_count = 0

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.config.max_retries - self._attempt_count)

    def next_strategy(
        self,
        messages: list[dict[str, Any]],
        failed_response: str,
        error_context: str,
        current_tier: str,
    ) -> CompensationResult | None:
        """Determine the next compensation strategy to try.

        Returns None if all retries exhausted.
        """
        if self._attempt_count >= self.config.max_retries:
            return None

        self._attempt_count += 1
        strategies = self.config.retry_strategies

        if self._attempt_count <= len(strategies):
            strategy = strategies[self._attempt_count - 1]
        else:
            return None

        if strategy == "refine_prompt":
            return self._refine_prompt(messages, failed_response, error_context)
        elif strategy == "add_examples":
            return self._add_examples(messages, failed_response, error_context)
        elif strategy == "escalate_model":
            return self._escalate_model(current_tier)
        else:
            logger.warning(f"Unknown compensation strategy: {strategy}")
            return None

    def _refine_prompt(
        self,
        messages: list[dict[str, Any]],
        failed_response: str,
        error_context: str,
    ) -> CompensationResult:
        """Add explicit correction to the conversation."""
        correction = (
            f"Your previous response could not be processed. "
            f"Error: {error_context}\n\n"
            f"Please try again. Remember:\n"
            f"- To use a tool, respond with ONLY a JSON object: "
            f'{{"tool": "tool_name", "args": {{"param": "value"}}}}\n'
            f"- To respond normally, just write text without any JSON tool calls.\n"
            f"- Do NOT wrap tool calls in code blocks or add extra text around them."
        )

        refined = list(messages)
        refined.append({"role": "assistant", "content": failed_response})
        refined.append({"role": "user", "content": correction})

        return CompensationResult(
            strategy="refine_prompt",
            success=True,
            modified_messages=refined,
            notes="Added explicit correction message",
        )

    def _add_examples(
        self,
        messages: list[dict[str, Any]],
        failed_response: str,
        error_context: str,
    ) -> CompensationResult:
        """Add few-shot examples to help the model understand the format."""
        example_msg = (
            f"Your previous response could not be processed. "
            f"Error: {error_context}\n\n"
            f"Here are examples of correct tool usage:\n\n"
            f"Example 1 - Running a command:\n"
            f'{{"tool": "shell", "args": {{"command": "ls -la"}}}}\n\n'
            f"Example 2 - Reading a file:\n"
            f'{{"tool": "read_file", "args": {{"path": "/home/user/code.py"}}}}\n\n'
            f"Example 3 - Normal text response (no tool needed):\n"
            f"The function calculates the factorial of a number using recursion.\n\n"
            f"Now please try again with the correct format."
        )

        refined = list(messages)
        refined.append({"role": "assistant", "content": failed_response})
        refined.append({"role": "user", "content": example_msg})

        return CompensationResult(
            strategy="add_examples",
            success=True,
            modified_messages=refined,
            notes="Added few-shot examples",
        )

    def _escalate_model(self, current_tier: str) -> CompensationResult:
        """Suggest using a larger model."""
        tier_order = ["small", "medium", "large"]
        try:
            idx = tier_order.index(current_tier)
        except ValueError:
            idx = 0

        if idx < len(tier_order) - 1:
            next_tier = tier_order[idx + 1]
            return CompensationResult(
                strategy="escalate_model",
                success=True,
                escalated_tier=next_tier,
                notes=f"Escalating from {current_tier} to {next_tier}",
            )
        else:
            return CompensationResult(
                strategy="escalate_model",
                success=False,
                notes=f"Already at highest tier ({current_tier}), cannot escalate",
            )


def build_tool_prompt(tools_description: str, thinking_mode: str = "auto") -> str:
    """Build the system prompt for tool-calling with local LLMs."""
    thinking_instruction = ""
    if thinking_mode == "never":
        thinking_instruction = "/no_think\n"
    elif thinking_mode == "auto":
        thinking_instruction = (
            "Use <think>...</think> for complex reasoning. "
            "Skip thinking for simple, direct tasks.\n"
        )

    return f"""{thinking_instruction}You are a capable AI assistant with access to tools.

## Available Tools

{tools_description}

## How to Use Tools

When you need to use a tool, respond with EXACTLY this JSON format (nothing else):
{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}

IMPORTANT RULES:
- Output ONLY the JSON object when calling a tool, with NO other text
- Only ONE tool call per response
- When you want to respond to the user (not use a tool), just write your response as normal text
- After receiving a tool result, analyze it and either call another tool or respond to the user
- If a tool fails, try a different approach or explain the issue

## Working Style

- Break complex tasks into steps
- Verify your work when possible
- If something fails, try alternative approaches before giving up
- Be concise but thorough in your responses"""
