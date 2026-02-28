"""Weak LLM compensation engine and prompt builders.

Strategies:
1. Parse fallback - extract tool calls from messy output
2. Prompt refinement - add examples, rephrase
3. Model escalation - try larger models
4. Output truncation - reduce tool output size
5. Step-limit escalation - auto-escalate on step limits
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
        self, messages, failed_response, error_context, current_tier,
    ) -> CompensationResult | None:
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
            return None

    def on_step_limit(self, messages, current_tier, step_count) -> CompensationResult | None:
        if self._step_escalation_used:
            return None
        next_tier = _next_tier(current_tier)
        if next_tier is None:
            return None
        self._step_escalation_used = True
        condensed = _condense_messages(messages,
            "Previous attempt hit the step limit. Use a more efficient approach.")
        return CompensationResult(
            strategy="step_limit_escalation", success=True,
            modified_messages=condensed, escalated_tier=next_tier,
            notes=f"Step limit ({step_count}). Escalating {current_tier} -> {next_tier}",
        )

    def _refine_prompt(self, messages, failed_response, error_context):
        correction = (
            f"Your previous response could not be processed. Error: {error_context}\n\n"
            f"Please try again. To use a tool, respond with ONLY:\n"
            f'{{"tool": "tool_name", "args": {{"param": "value"}}}}\n'
            f"To respond normally, just write text."
        )
        refined = list(messages)
        refined.append({"role": "assistant", "content": failed_response})
        refined.append({"role": "user", "content": correction})
        return CompensationResult(strategy="refine_prompt", success=True,
                                  modified_messages=refined, notes="Added correction")

    def _add_examples(self, messages, failed_response, error_context):
        example_msg = (
            f"Error: {error_context}\n\nExamples of correct tool usage:\n"
            f'{{"tool": "shell", "args": {{"command": "ls -la"}}}}\n'
            f'{{"tool": "read_file", "args": {{"path": "src/main.py"}}}}\n'
            f"Normal response (no tool): Just write text.\nTry again."
        )
        refined = list(messages)
        refined.append({"role": "assistant", "content": failed_response})
        refined.append({"role": "user", "content": example_msg})
        return CompensationResult(strategy="add_examples", success=True,
                                  modified_messages=refined, notes="Added examples")

    def _escalate_model(self, current_tier):
        nt = _next_tier(current_tier)
        if nt:
            return CompensationResult(strategy="escalate_model", success=True,
                                      escalated_tier=nt, notes=f"{current_tier} -> {nt}")
        return CompensationResult(strategy="escalate_model", success=False,
                                  notes=f"Already at {current_tier}")


def _next_tier(current: str) -> str | None:
    try:
        idx = TIER_ORDER.index(current)
    except ValueError:
        return None
    return TIER_ORDER[idx + 1] if idx < len(TIER_ORDER) - 1 else None


def _condense_messages(messages, summary_prefix):
    condensed = []
    if messages and messages[0].get("role") == "system":
        condensed.append(messages[0])
    original = ""
    for m in messages:
        if m.get("role") == "user" and not m["content"].startswith("[Tool Result"):
            original = m["content"]
            break
    tool_summaries = []
    for m in messages:
        c = m.get("content", "")
        if c.startswith("[Tool Result for "):
            first_line = c.split("\n")[0]
            preview = c[len(first_line):].strip()[:200]
            tool_summaries.append(f"{first_line}\n{preview}...")
    parts = [summary_prefix, f"\nOriginal request: {original}"]
    if tool_summaries:
        parts.append("\nPrevious results (summarized):")
        parts.extend(tool_summaries[-5:])
    condensed.append({"role": "user", "content": "\n".join(parts)})
    return condensed


def truncate_tool_output(output: str, max_length: int = 5000) -> str:
    if len(output) <= max_length:
        return output
    half = max_length // 2
    return output[:half] + f"\n\n... [{len(output) - max_length} chars truncated] ...\n\n" + output[-half:]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_TOOL_FORMAT = """\
When you need to use a tool, respond with EXACTLY this JSON (nothing else):
{"tool": "tool_name", "args": {"param1": "value1"}}

RULES:
- Output ONLY the JSON when calling a tool — no other text around it
- ONE tool call per response
- To respond to the user, just write normal text (no JSON)
- Prefer shell pipes and chaining to minimize tool calls"""


def build_tool_prompt(tools_description: str, thinking_mode: str = "auto") -> str:
    """System prompt for interactive (conversational) mode."""
    think = ""
    if thinking_mode == "never":
        think = "/no_think\n"
    elif thinking_mode == "auto":
        think = "Use <think>...</think> for complex reasoning. Skip for simple tasks.\n"

    return f"""{think}You are a capable AI assistant with access to tools.

## Available Tools

{tools_description}

## How to Use Tools

{_TOOL_FORMAT}

## Working Style

- Break complex tasks into steps, minimize tool calls
- Use shell pipes (e.g., `find ... | wc -l`) to combine operations
- Verify your work when possible
- If something fails, try a different approach
- Be concise but thorough"""


def build_autonomous_prompt(
    tools_description: str,
    project_context: str,
    thinking_mode: str = "auto",
) -> str:
    """System prompt for autonomous goal-driven mode.

    Key difference from interactive: the agent works until the goal is achieved
    without asking the user for permission or clarification.
    """
    think = ""
    if thinking_mode == "never":
        think = "/no_think\n"
    elif thinking_mode == "auto":
        think = "Use <think>...</think> for planning and complex reasoning.\n"

    return f"""{think}You are an autonomous AI coding agent. You work independently to achieve goals.

## Core Behavior

- Work step by step toward the goal WITHOUT asking for user confirmation
- After each tool result, evaluate progress and decide your next action
- If something fails, try a different approach automatically
- When the goal is fully achieved, write a clear summary of what you did
- DO NOT ask questions or request clarification — make reasonable decisions
- If you are unsure, choose the most reasonable option and proceed
- Keep working until the goal is FULLY COMPLETE

## Available Tools

{tools_description}

## How to Use Tools

{_TOOL_FORMAT}

## Project Context

{project_context}

## Autonomous Work Patterns

### Code Changes
1. Read the relevant files first to understand existing code
2. Make changes using write_file or edit_file
3. Run tests to verify (run_tests tool)
4. If tests fail, analyze errors, fix code, re-run tests
5. Repeat until all tests pass

### Test-Driven Loop
When fixing bugs or implementing features:
1. Run tests to see current state
2. Read failing test to understand expected behavior
3. Read and modify source code
4. Re-run tests
5. Iterate until all tests pass

### Git Workflow
- Use git_status to check current state
- Commit working changes with clear messages
- Create feature branches for larger changes

## Completion
When the goal is achieved, respond with a summary:
- What was done
- Files modified
- Test results (if applicable)
- Any important notes"""
