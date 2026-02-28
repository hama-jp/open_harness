"""Weak LLM compensation engine.

Strategies to help local LLMs succeed at tool calling and complex tasks:
1. Parse fallback - extract tool calls from messy output
2. Prompt refinement - add examples, rephrase, add CoT
3. Model escalation - try larger models
4. Output truncation - reduce tool output size for weak models
5. Step-limit escalation - auto-escalate when hitting step limits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from open_harness.config import CompensationConfig

logger = logging.getLogger(__name__)

TIER_ORDER = ["small", "medium", "large"]


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
        self._step_escalation_used = False

    def reset(self):
        self._attempt_count = 0
        self._step_escalation_used = False

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
        """Determine the next compensation strategy to try."""
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

    def on_step_limit(
        self,
        messages: list[dict[str, Any]],
        current_tier: str,
        step_count: int,
    ) -> CompensationResult | None:
        """Called when agent reaches step limit. Tries model escalation.

        Returns None if escalation is not possible.
        """
        if self._step_escalation_used:
            return None

        next_tier = _next_tier(current_tier)
        if next_tier is None:
            return None

        self._step_escalation_used = True

        # Summarize conversation so far to reduce context size
        summary_msg = (
            "The previous attempt used too many steps and could not complete. "
            "You are now using a more capable model. "
            "Please complete the original task efficiently, using fewer tool calls. "
            "Combine operations where possible (e.g., use shell pipes)."
        )
        condensed = _condense_messages(messages, summary_msg)

        return CompensationResult(
            strategy="step_limit_escalation",
            success=True,
            modified_messages=condensed,
            escalated_tier=next_tier,
            notes=f"Step limit reached ({step_count} steps). Escalating {current_tier} -> {next_tier}",
        )

    def _refine_prompt(
        self,
        messages: list[dict[str, Any]],
        failed_response: str,
        error_context: str,
    ) -> CompensationResult:
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
        next_tier = _next_tier(current_tier)
        if next_tier:
            return CompensationResult(
                strategy="escalate_model",
                success=True,
                escalated_tier=next_tier,
                notes=f"Escalating from {current_tier} to {next_tier}",
            )
        return CompensationResult(
            strategy="escalate_model",
            success=False,
            notes=f"Already at highest tier ({current_tier}), cannot escalate",
        )


def _next_tier(current: str) -> str | None:
    """Get the next tier up, or None if already at max."""
    try:
        idx = TIER_ORDER.index(current)
    except ValueError:
        return None
    if idx < len(TIER_ORDER) - 1:
        return TIER_ORDER[idx + 1]
    return None


def _condense_messages(
    messages: list[dict[str, Any]],
    summary_prefix: str,
) -> list[dict[str, Any]]:
    """Condense a long message history, keeping system prompt and user request.

    Truncates tool results to save context window for the larger model.
    """
    if not messages:
        return messages

    condensed: list[dict[str, Any]] = []

    # Keep system prompt
    if messages[0].get("role") == "system":
        condensed.append(messages[0])

    # Find original user request (first user message)
    original_request = ""
    for m in messages:
        if m.get("role") == "user" and not m["content"].startswith("[Tool Result"):
            original_request = m["content"]
            break

    # Collect tool results summary
    tool_summary_parts = []
    for m in messages:
        content = m.get("content", "")
        if content.startswith("[Tool Result for "):
            # Extract tool name and truncate result
            first_line = content.split("\n")[0]
            result_preview = content[len(first_line):].strip()
            if len(result_preview) > 200:
                result_preview = result_preview[:200] + "..."
            tool_summary_parts.append(f"{first_line}\n{result_preview}")

    # Build condensed user message
    parts = [summary_prefix, "", f"Original request: {original_request}"]
    if tool_summary_parts:
        parts.append("")
        parts.append("Previous tool results (summarized):")
        for ts in tool_summary_parts[-5:]:  # Keep last 5 tool results
            parts.append(ts)

    condensed.append({"role": "user", "content": "\n".join(parts)})
    return condensed


def truncate_tool_output(output: str, max_length: int = 5000) -> str:
    """Truncate tool output to fit within context constraints.

    Keeps the beginning and end for context.
    """
    if len(output) <= max_length:
        return output
    half = max_length // 2
    return output[:half] + f"\n\n... [{len(output) - max_length} chars truncated] ...\n\n" + output[-half:]


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
- Prefer combining operations (e.g., shell pipes like `find ... | wc -l`) over many small tool calls

## Working Style

- Break complex tasks into steps, but minimize the number of tool calls
- Use shell pipes and command chaining to do multiple things in one call
- Verify your work when possible
- If something fails, try alternative approaches before giving up
- Be concise but thorough in your responses"""
