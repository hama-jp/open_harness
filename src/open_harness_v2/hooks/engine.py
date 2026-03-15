"""Hook engine — runs shell commands in response to agent events.

Enhanced with decision control inspired by Claude Code's PreToolUse hooks.
Pre-tool hooks can now return decisions (allow/deny/ask) by exit code:

- Exit 0: allow (hook passed, no objection)
- Exit 1: deny  (hook blocked the action)
- Exit 2: ask   (hook wants user confirmation)

Hook stdout on deny/ask is used as the reason message.

Post-tool hooks with ``on_failure: true`` only fire when the tool fails.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType

_logger = logging.getLogger(__name__)

# Maximum time (seconds) a hook command can run
_HOOK_TIMEOUT = 30


class HookDecision(enum.Enum):
    """Decision returned by a pre-tool hook."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class HookResult:
    """Result of running a hook."""

    decision: HookDecision = HookDecision.ALLOW
    message: str = ""
    exit_code: int = 0


@dataclass
class HookSpec:
    """Definition of a single hook."""

    command: str
    match_tools: list[str] = field(default_factory=list)
    timeout: int = _HOOK_TIMEOUT
    on_failure: bool = False  # Only fire on tool failure (post_tool)

    def matches_tool(self, tool_name: str) -> bool:
        """Return True if this hook should fire for the given tool."""
        if not self.match_tools:
            return True  # no filter = match all
        return tool_name in self.match_tools


@dataclass
class HooksConfig:
    """All hooks grouped by event type."""

    pre_goal: list[HookSpec] = field(default_factory=list)
    post_goal: list[HookSpec] = field(default_factory=list)
    pre_tool: list[HookSpec] = field(default_factory=list)
    post_tool: list[HookSpec] = field(default_factory=list)
    on_error: list[HookSpec] = field(default_factory=list)


class HookEngine:
    """Runs hooks in response to EventBus events.

    Hooks are shell commands that run asynchronously.  Pre-tool hooks
    support decision control (allow/deny/ask via exit codes).
    """

    def __init__(self, hooks_config: HooksConfig | None = None) -> None:
        self._config = hooks_config or HooksConfig()
        self._cwd = Path.cwd()
        # Callback for tool decisions (set by executor)
        self._on_pre_tool_decision: Any = None

    @property
    def has_hooks(self) -> bool:
        """Return True if any hooks are configured."""
        c = self._config
        return bool(c.pre_goal or c.post_goal or c.pre_tool or c.post_tool or c.on_error)

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to relevant events on the EventBus."""
        event_bus.subscribe(EventType.AGENT_STARTED, self._on_agent_started)
        event_bus.subscribe(EventType.AGENT_DONE, self._on_agent_done)
        event_bus.subscribe(EventType.TOOL_EXECUTING, self._on_tool_executing)
        event_bus.subscribe(EventType.TOOL_EXECUTED, self._on_tool_executed)
        event_bus.subscribe(EventType.TOOL_ERROR, self._on_tool_error)
        event_bus.subscribe(EventType.AGENT_ERROR, self._on_agent_error)

    async def check_pre_tool(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> HookResult:
        """Run pre-tool hooks and return the aggregate decision.

        If any hook returns DENY, the overall result is DENY.
        If any hook returns ASK (and none DENY), the result is ASK.
        Otherwise, the result is ALLOW.
        """
        hooks = [h for h in self._config.pre_tool if h.matches_tool(tool_name)]
        if not hooks:
            return HookResult(decision=HookDecision.ALLOW)

        variables = {
            "tool_name": tool_name,
            "tool_args": json.dumps(tool_args, default=str),
        }

        overall_decision = HookDecision.ALLOW
        messages: list[str] = []

        for hook in hooks:
            result = await self._exec_with_decision(
                hook.command, variables, timeout=hook.timeout
            )
            if result.decision == HookDecision.DENY:
                overall_decision = HookDecision.DENY
                if result.message:
                    messages.append(result.message)
            elif (
                result.decision == HookDecision.ASK
                and overall_decision != HookDecision.DENY
            ):
                overall_decision = HookDecision.ASK
                if result.message:
                    messages.append(result.message)

        return HookResult(
            decision=overall_decision,
            message="; ".join(messages) if messages else "",
        )

    async def _on_agent_started(self, event: AgentEvent) -> None:
        goal = event.data.get("goal", "")
        await self._run_hooks(self._config.pre_goal, {"goal": goal})

    async def _on_agent_done(self, event: AgentEvent) -> None:
        response = event.data.get("response", "")
        await self._run_hooks(self._config.post_goal, {"response": response})

    async def _on_tool_executing(self, event: AgentEvent) -> None:
        # Pre-tool hooks with decision control are now handled by
        # check_pre_tool() called directly from the executor.
        # This handler is kept for non-decision hooks (backward compat).
        pass

    async def _on_tool_executed(self, event: AgentEvent) -> None:
        tool_name = event.data.get("tool", "")
        success = event.data.get("success", True)
        hooks = [
            h for h in self._config.post_tool
            if h.matches_tool(tool_name) and not h.on_failure
        ]
        await self._run_hooks(hooks, {
            "tool_name": tool_name,
            "success": str(success),
            "output": str(event.data.get("output", ""))[:500],
        })

    async def _on_tool_error(self, event: AgentEvent) -> None:
        tool_name = event.data.get("tool", "")
        # Fire on_failure hooks
        hooks = [
            h for h in self._config.post_tool
            if h.matches_tool(tool_name) and h.on_failure
        ]
        await self._run_hooks(hooks, {
            "tool_name": tool_name,
            "success": "False",
            "error": str(event.data.get("error", "")),
        })

    async def _on_agent_error(self, event: AgentEvent) -> None:
        error = event.data.get("error", "")
        await self._run_hooks(self._config.on_error, {"error": error})

    async def _run_hooks(
        self,
        hooks: list[HookSpec],
        variables: dict[str, str],
    ) -> None:
        """Run a list of hooks, substituting template variables."""
        for hook in hooks:
            cmd = hook.command
            for key, value in variables.items():
                cmd = cmd.replace(f"{{{{{key}}}}}", str(value))
            await self._exec(cmd, timeout=hook.timeout)

    async def _exec_with_decision(
        self, cmd: str, variables: dict[str, str], timeout: int = _HOOK_TIMEOUT
    ) -> HookResult:
        """Execute a hook and interpret exit code as a decision."""
        for key, value in variables.items():
            cmd = cmd.replace(f"{{{{{key}}}}}", str(value))

        try:
            env = {**os.environ, "HARNESS_HOOK": "1"}
            # Pass tool info as environment variables
            for key, value in variables.items():
                env[f"HARNESS_{key.upper()}"] = str(value)

            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                stdout_text = stdout.decode(errors="replace").strip()
                returncode = proc.returncode or 0

                if returncode == 0:
                    return HookResult(
                        decision=HookDecision.ALLOW, message=stdout_text
                    )
                elif returncode == 1:
                    return HookResult(
                        decision=HookDecision.DENY,
                        message=stdout_text or "Hook denied this action",
                        exit_code=1,
                    )
                elif returncode == 2:
                    return HookResult(
                        decision=HookDecision.ASK,
                        message=stdout_text or "Hook requests user confirmation",
                        exit_code=2,
                    )
                else:
                    _logger.warning(
                        "Hook returned unexpected exit code %d: %s",
                        returncode, cmd,
                    )
                    return HookResult(decision=HookDecision.ALLOW)

            except asyncio.TimeoutError:
                proc.kill()
                _logger.warning("Hook timed out after %ds: %s", timeout, cmd)
                return HookResult(decision=HookDecision.ALLOW)

        except Exception:
            _logger.exception("Hook execution failed: %s", cmd)
            return HookResult(decision=HookDecision.ALLOW)

    async def _exec(self, cmd: str, timeout: int = _HOOK_TIMEOUT) -> None:
        """Execute a hook command asynchronously (fire-and-forget)."""
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                env={**os.environ, "HARNESS_HOOK": "1"},
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                if proc.returncode != 0:
                    stderr_text = stderr.decode(errors="replace").strip()
                    _logger.warning(
                        "Hook command failed (exit %d): %s — %s",
                        proc.returncode, cmd, stderr_text,
                    )
                else:
                    stdout_text = stdout.decode(errors="replace").strip()
                    if stdout_text:
                        _logger.debug("Hook output: %s", stdout_text[:200])
            except asyncio.TimeoutError:
                proc.kill()
                _logger.warning("Hook command timed out after %ds: %s", timeout, cmd)
        except Exception:
            _logger.exception("Hook execution failed: %s", cmd)


def parse_hooks_config(raw: dict[str, Any] | None) -> HooksConfig:
    """Parse hooks configuration from a raw YAML dict."""
    if not raw:
        return HooksConfig()

    config = HooksConfig()
    for hook_type in ("pre_goal", "post_goal", "pre_tool", "post_tool", "on_error"):
        raw_hooks = raw.get(hook_type, [])
        specs = []
        for h in raw_hooks:
            if isinstance(h, str):
                specs.append(HookSpec(command=h))
            elif isinstance(h, dict):
                specs.append(HookSpec(
                    command=h.get("command", ""),
                    match_tools=h.get("match_tools", []),
                    timeout=h.get("timeout", _HOOK_TIMEOUT),
                    on_failure=h.get("on_failure", False),
                ))
        setattr(config, hook_type, specs)

    return config


def load_hooks(project_root: Path | None = None) -> HooksConfig:
    """Load hooks from project-level .harness/hooks.yaml if it exists."""
    import yaml

    if project_root:
        hooks_file = project_root / ".harness" / "hooks.yaml"
        if hooks_file.is_file():
            with open(hooks_file) as f:
                raw = yaml.safe_load(f) or {}
            return parse_hooks_config(raw)

    return HooksConfig()
