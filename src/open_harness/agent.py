"""Core agent loop — interactive and autonomous goal-driven modes."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Generator

from open_harness.config import HarnessConfig
from open_harness.llm.client import LLMResponse, ToolCall
from open_harness.llm.compensator import (
    Compensator,
    build_autonomous_prompt,
    build_tool_prompt,
    truncate_tool_output,
)
from open_harness.llm.router import ModelRouter
from open_harness.memory.store import MemoryStore
from open_harness.project import ProjectContext
from open_harness.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

MAX_INTERACTIVE_STEPS = 15
MAX_GOAL_STEPS = 50  # Autonomous mode gets more room


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: str  # thinking, text, tool_call, tool_result, compensation, status, done
    data: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent:
    """ReAct agent with interactive and autonomous goal-driven modes."""

    def __init__(
        self,
        config: HarnessConfig,
        tools: ToolRegistry,
        memory: MemoryStore,
        project: ProjectContext | None = None,
    ):
        self.config = config
        self.tools = tools
        self.memory = memory
        self.project = project or ProjectContext()
        self.router = ModelRouter(config)
        self.compensator = Compensator(config.compensation)
        self._interactive_prompt: str | None = None
        self._autonomous_prompt: str | None = None

    @property
    def interactive_prompt(self) -> str:
        if self._interactive_prompt is None:
            self._interactive_prompt = build_tool_prompt(
                self.tools.get_prompt_description(),
                self.config.compensation.thinking_mode,
            )
        return self._interactive_prompt

    @property
    def autonomous_prompt(self) -> str:
        if self._autonomous_prompt is None:
            self._autonomous_prompt = build_autonomous_prompt(
                self.tools.get_prompt_description(),
                self.project.to_prompt(),
                self.config.compensation.thinking_mode,
            )
        return self._autonomous_prompt

    def invalidate_prompts(self):
        """Call after tools or project context change."""
        self._interactive_prompt = None
        self._autonomous_prompt = None

    # ------------------------------------------------------------------
    # Interactive mode (existing behavior)
    # ------------------------------------------------------------------

    def run_stream(self, user_message: str) -> Generator[AgentEvent, None, None]:
        """Interactive single-turn with streaming."""
        self.compensator.reset()
        self.memory.add_turn("user", user_message)
        messages = [
            {"role": "system", "content": self.interactive_prompt},
            *self.memory.get_messages(),
        ]
        yield from self._agent_loop(messages, MAX_INTERACTIVE_STEPS)

    def run(self, user_message: str) -> str:
        final = ""
        for ev in self.run_stream(user_message):
            if ev.type == "done":
                final = ev.data
        return final

    # ------------------------------------------------------------------
    # Autonomous goal mode — the core of "self-driving"
    # ------------------------------------------------------------------

    def run_goal(self, goal: str) -> Generator[AgentEvent, None, None]:
        """Run autonomously toward a goal.

        The agent keeps working (calling tools, reasoning, fixing errors)
        until the goal is achieved or the step budget is exhausted.
        """
        self.compensator.reset()

        yield AgentEvent("status", f"Goal: {goal}")
        yield AgentEvent("status", f"Project: {self.project.info['type']} @ {self.project.info['root']}")

        # Safety: create a checkpoint before autonomous work
        checkpoint = self._create_checkpoint()
        if checkpoint:
            yield AgentEvent("status", f"Checkpoint: {checkpoint}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.autonomous_prompt},
            {"role": "user", "content": f"GOAL: {goal}\n\nWork autonomously to achieve this goal. Do not ask me questions — just do it."},
        ]

        try:
            yield from self._agent_loop(messages, MAX_GOAL_STEPS)
        finally:
            # Restore stashed changes if we created a checkpoint
            if checkpoint == "stashed uncommitted changes":
                restored = self._restore_checkpoint()
                if restored:
                    yield AgentEvent("status", f"Restored: {restored}")

    # ------------------------------------------------------------------
    # Shared agent loop
    # ------------------------------------------------------------------

    def _agent_loop(
        self,
        messages: list[dict[str, Any]],
        max_steps: int,
        tier: str | None = None,
    ) -> Generator[AgentEvent, None, None]:
        """Core ReAct loop shared by interactive and goal modes."""
        tier = tier or self.router.current_tier
        step = 0

        while step < max_steps:
            step += 1
            yield AgentEvent("status", f"[{step}/{max_steps}] {tier}")

            response = yield from self._stream_llm(messages, tier)

            # API error
            if response.finish_reason == "error":
                comp = self.compensator.next_strategy(
                    messages, response.content, "API error", tier)
                if comp and comp.success:
                    yield AgentEvent("compensation", comp.notes)
                    if comp.modified_messages:
                        messages = comp.modified_messages
                    if comp.escalated_tier:
                        tier = comp.escalated_tier
                    continue
                self.memory.add_turn("assistant", response.content)
                yield AgentEvent("done", response.content)
                return

            # Tool call
            if response.has_tool_call:
                tc = response.tool_calls[0]
                yield AgentEvent("tool_call", tc.name, {"tool": tc.name, "args": tc.arguments})

                result = self.tools.execute(tc.name, tc.arguments)
                output = truncate_tool_output(result.to_message(), 8000)
                yield AgentEvent("tool_result", output,
                                 {"success": result.success, "tool": tc.name})

                messages.append({"role": "assistant",
                    "content": f'{{"tool": "{tc.name}", "args": {_safe_json(tc.arguments)}}}'})
                messages.append({"role": "user",
                    "content": f"[Tool Result for {tc.name}]\n{output}"})
                continue

            # Malformed tool call
            if self._looks_like_failed_tool_call(response.content):
                comp = self.compensator.next_strategy(
                    messages, response.content, "Malformed tool call", tier)
                if comp and comp.success:
                    yield AgentEvent("compensation", comp.notes)
                    if comp.modified_messages:
                        messages = comp.modified_messages
                    if comp.escalated_tier:
                        tier = comp.escalated_tier
                    continue

            # Text response — done
            self.memory.add_turn("assistant", response.content)
            yield AgentEvent("done", response.content,
                             {"latency_ms": response.latency_ms, "steps": step})
            return

        # Step limit — try escalation
        comp = self.compensator.on_step_limit(messages, tier, step)
        if comp and comp.success:
            yield AgentEvent("compensation", comp.notes)
            messages = comp.modified_messages or messages
            tier = comp.escalated_tier or tier
            yield from self._agent_loop(messages, max_steps, tier=tier)
            return

        yield AgentEvent("done",
            f"[Reached {max_steps} steps. Use /tier large or simplify the goal.]")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stream_llm(self, messages, tier) -> Generator[AgentEvent, None, LLMResponse]:
        gen = self.router.chat_stream(messages=messages, tier=tier, temperature=0.3)
        response: LLMResponse | None = None
        try:
            while True:
                etype, data = next(gen)
                if etype == "thinking":
                    yield AgentEvent("thinking", data)
                elif etype == "text":
                    yield AgentEvent("text", data)
        except StopIteration as e:
            response = e.value
        if response is None:
            response = LLMResponse(content="", finish_reason="error")
        return response

    @staticmethod
    def _looks_like_failed_tool_call(content: str) -> bool:
        indicators = [
            '"tool"' in content and '"args"' in content and "{" in content,
            "```json" in content and '"tool"' in content,
            content.strip().startswith("{") and '"tool_call"' in content,
        ]
        return any(indicators) and len(content) < 500

    def _create_checkpoint(self) -> str | None:
        """Create a git checkpoint before autonomous work."""
        if not self.project.info.get("has_git"):
            return None
        cwd = str(self.project.root)
        try:
            r = subprocess.run(
                "git status --porcelain", shell=True,
                capture_output=True, text=True, timeout=10, cwd=cwd,
            )
            if r.stdout.strip():
                stash_r = subprocess.run(
                    "git stash push -m 'open-harness: auto-checkpoint before goal'",
                    shell=True, capture_output=True, text=True, timeout=10, cwd=cwd,
                )
                if stash_r.returncode == 0 and "No local changes" not in stash_r.stdout:
                    return "stashed uncommitted changes"
            return "clean working tree"
        except Exception as e:
            logger.warning(f"Failed to create checkpoint: {e}")
            return None

    def _restore_checkpoint(self) -> str | None:
        """Restore stashed changes after goal completion."""
        if not self.project.info.get("has_git"):
            return None
        cwd = str(self.project.root)
        try:
            r = subprocess.run(
                "git stash list", shell=True,
                capture_output=True, text=True, timeout=10, cwd=cwd,
            )
            if "open-harness: auto-checkpoint" in r.stdout:
                pop_r = subprocess.run(
                    "git stash pop", shell=True,
                    capture_output=True, text=True, timeout=10, cwd=cwd,
                )
                if pop_r.returncode == 0:
                    return "restored stashed changes"
                return f"stash pop failed: {pop_r.stderr.strip()}"
            return None
        except Exception as e:
            logger.warning(f"Failed to restore checkpoint: {e}")
            return None


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)
