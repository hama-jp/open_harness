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
        c = m.get("content", "") or ""
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
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


# Built-in defaults: tool_name -> (description, [strengths])
# Users can override via external_agents.*.description / strengths in config.
_DEFAULT_AGENT_INFO: dict[str, tuple[str, list[str]]] = {
    "claude_code": (
        "Complex refactoring, multi-file changes, planning, Japanese/multilingual text, "
        "code review (logic/IDOR/XSS). Has Plan Mode and Extended Thinking.",
        ["refactoring", "planning", "japanese_text", "code_review_logic",
         "bug_fix", "architecture"],
    ),
    "codex": (
        "Fast coding, security-focused code review (path traversal/SSRF), "
        "CI/CD integration, sandbox execution. Fastest response time.",
        ["fast_coding", "code_review_security", "ci_cd", "sandbox"],
    ),
    "gemini_cli": (
        "Large codebase understanding (1M context), abstract reasoning, "
        "science tasks, spec-to-task decomposition. Free tier available.",
        ["large_codebase", "abstract_reasoning", "science",
         "spec_decomposition"],
    ),
}

# Task routing guidance embedded in the orchestrator prompt.
_ROUTING_GUIDE = """\
### Task Routing Guide

Choose the best agent for each task type:

| Task | Best Agent | Reason |
|------|-----------|--------|
| Complex refactoring / multi-file changes | claude_code | Plan Mode, SWE-bench top tier |
| Security review (path traversal, SSRF) | codex | Highest detection rate |
| Logic review (IDOR, XSS, auth) | claude_code | Context-dependent analysis |
| Japanese / multilingual docs | claude_code | WMT24 translation winner |
| Large codebase understanding | gemini_cli | 1M context, lower cost |
| Speed-critical simple tasks | codex | Fastest response |
| Abstract reasoning / science | gemini_cli | ARC-AGI-2 77%, GPQA 94% |
| Planning / architecture design | claude_code | Plan Mode, Extended Thinking |

If one agent fails, try a different agent for the same task."""


def _build_orchestrator_section(
    available_tools: list[str] | None,
    agent_configs: dict | None = None,
) -> str:
    """Build orchestrator role section only if external agents are available."""
    if not available_tools:
        return ""

    # Collect agent info: config overrides > built-in defaults
    agents: dict[str, str] = {}
    for name in available_tools:
        if name not in _DEFAULT_AGENT_INFO:
            continue
        default_desc, _default_strengths = _DEFAULT_AGENT_INFO[name]
        # Check if user provided a custom description in config
        if agent_configs:
            # agent_configs is dict[str, ExternalAgentConfig] — map tool name to config key
            cfg = _find_agent_config(name, agent_configs)
            if cfg and cfg.description:
                agents[name] = cfg.description
                continue
        agents[name] = default_desc

    if not agents:
        return ""

    lines = [
        "\n## Your Role\n",
        "You are a LOCAL LLM acting as an orchestrator. Your strengths are planning, judgment,",
        "and tool selection. For tasks requiring code generation, code analysis, complex reasoning,",
        "or debugging, delegate to the appropriate external agent tool.\n",
        "## External Agent Strengths\n",
    ]
    for name, desc in agents.items():
        lines.append(f"- **{name}**: {desc}")

    lines.append(f"\n{_ROUTING_GUIDE}")

    return "\n".join(lines)


def _find_agent_config(tool_name: str, agent_configs: dict):
    """Map a tool name (e.g. 'claude_code') to its ExternalAgentConfig."""
    # Config keys: codex, claude, gemini -> tool names: codex, claude_code, gemini_cli
    _KEY_TO_TOOL = {"claude": "claude_code", "gemini": "gemini_cli", "codex": "codex"}
    for key, cfg in agent_configs.items():
        if _KEY_TO_TOOL.get(key) == tool_name or key == tool_name:
            return cfg
    return None


def _current_datetime() -> str:
    """Return the current date and time as a human-readable string."""
    from datetime import datetime, timezone
    now = datetime.now()
    utc = datetime.now(timezone.utc)
    return (
        f"Current date/time: {now.strftime('%Y-%m-%d %H:%M')} (local), "
        f"{utc.strftime('%Y-%m-%d %H:%M')} (UTC)"
    )


def build_tool_prompt(
    tools_description: str,
    thinking_mode: str = "auto",
    available_tools: list[str] | None = None,
    agent_configs: dict | None = None,
) -> str:
    """System prompt for interactive (conversational) mode."""
    think = ""
    if thinking_mode == "never":
        think = "/no_think\n"
    elif thinking_mode == "auto":
        think = "Use <think>...</think> for complex reasoning. Skip for simple tasks.\n"

    orchestrator = _build_orchestrator_section(available_tools, agent_configs)
    if orchestrator:
        role = "You are an orchestrator — a planning and coordination AI with access to tools."
        style = (
            "- Plan first, then delegate complex work to external agents\n"
            "- Use file/shell tools for simple reads and checks\n"
            "- Delegate code generation, analysis, and debugging to external agents\n"
            "- Verify results after delegation (e.g., run tests)\n"
            "- If one agent fails, try a different agent\n"
            "- Be concise but thorough"
        )
    else:
        role = "You are a capable AI assistant with access to tools."
        style = (
            "- Break complex tasks into steps, minimize tool calls\n"
            "- Use shell pipes (e.g., `find ... | wc -l`) to combine operations\n"
            "- Verify your work when possible\n"
            "- If something fails, try a different approach\n"
            "- Be concise but thorough"
        )

    return f"""{think}{role}
{orchestrator}

{_current_datetime()}

## Available Tools

{tools_description}

## How to Use Tools

{_TOOL_FORMAT}

## Working Style

{style}"""


def build_autonomous_prompt(
    tools_description: str,
    project_context: str,
    thinking_mode: str = "auto",
    available_tools: list[str] | None = None,
    agent_configs: dict | None = None,
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

    has_external = bool(
        available_tools
        and any(t in _DEFAULT_AGENT_INFO for t in available_tools)
    )

    if has_external:
        # Build agent descriptions from config overrides or defaults
        agents: dict[str, str] = {}
        for name in available_tools:
            if name not in _DEFAULT_AGENT_INFO:
                continue
            default_desc, _ = _DEFAULT_AGENT_INFO[name]
            cfg = _find_agent_config(name, agent_configs) if agent_configs else None
            agents[name] = cfg.description if (cfg and cfg.description) else default_desc

        agent_lines = "\n".join(f"- **{k}**: {v}" for k, v in agents.items())
        role_section = f"""## Your Role

You are a LOCAL LLM acting as an orchestrator. Your job is PLANNING, COORDINATION,
and VERIFICATION. For tasks requiring code generation, code analysis, complex reasoning,
or debugging, delegate to external agent tools.

## External Agent Strengths

{agent_lines}

{_ROUTING_GUIDE}"""

        core_behavior = """\
- Work step by step toward the goal WITHOUT asking for user confirmation
- Choose the BEST external agent for each sub-task based on the routing guide
- Use file/shell tools for simple reads, checks, and verification
- After each tool result, evaluate progress and decide your next action
- If one external agent fails, try a different one for the same task
- When the goal is fully achieved, write a clear summary of what you did
- DO NOT ask questions or request clarification — make reasonable decisions
- Keep working until the goal is FULLY COMPLETE"""

        agent_names = ", ".join(agents.keys())
        work_patterns = f"""\
### Code Changes (Orchestrator Style)
1. Read the relevant files first to understand the context
2. Delegate code generation/modification to the best external agent ({agent_names})
   - Complex refactoring / multi-file → claude_code
   - Speed-critical simple changes → codex
   - Large codebase context needed → gemini_cli
3. Verify the changes by reading modified files
4. Run tests to verify (run_tests tool)
5. If tests fail, delegate the fix to an external agent with error context
6. Repeat until all tests pass

### Code Review
1. Read the relevant files
2. Delegate review to the best agent:
   - Security review → codex
   - Logic / architecture review → claude_code
   - Large codebase review → gemini_cli
3. Summarize findings

### Japanese / Multilingual Text
- Prefer claude_code for Japanese documentation, comments, and commit messages
- gemini_cli is a good alternative for multilingual tasks

### Test-Driven Loop
1. Run tests to see current state
2. Delegate bug fixes to an external agent with test output as context
3. Re-run tests to verify
4. Iterate until all tests pass"""

        role_intro = "You are an autonomous orchestrator agent. You coordinate tools and external agents to achieve goals."
    else:
        role_section = ""
        core_behavior = """\
- Work step by step toward the goal WITHOUT asking for user confirmation
- After each tool result, evaluate progress and decide your next action
- If something fails, try a different approach automatically
- When the goal is fully achieved, write a clear summary of what you did
- DO NOT ask questions or request clarification — make reasonable decisions
- If you are unsure, choose the most reasonable option and proceed
- Keep working until the goal is FULLY COMPLETE"""

        work_patterns = """\
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
5. Iterate until all tests pass"""

        role_intro = "You are an autonomous AI coding agent. You work independently to achieve goals."

    return f"""{think}{role_intro}

{_current_datetime()}

{role_section}

## Core Behavior

{core_behavior}

## Available Tools

{tools_description}

## How to Use Tools

{_TOOL_FORMAT}

## Project Context

{project_context}

## Autonomous Work Patterns

{work_patterns}

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
