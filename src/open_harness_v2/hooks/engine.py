"""Hook engine — runs shell commands in response to agent events."""

from __future__ import annotations

import asyncio
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


@dataclass
class HookSpec:
    """Definition of a single hook."""

    command: str
    match_tools: list[str] = field(default_factory=list)
    timeout: int = _HOOK_TIMEOUT

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

    Hooks are shell commands that run asynchronously.  Failures
    are logged but do not block the agent.
    """

    def __init__(self, hooks_config: HooksConfig | None = None) -> None:
        self._config = hooks_config or HooksConfig()
        self._cwd = Path.cwd()

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
        event_bus.subscribe(EventType.AGENT_ERROR, self._on_agent_error)

    async def _on_agent_started(self, event: AgentEvent) -> None:
        goal = event.data.get("goal", "")
        await self._run_hooks(self._config.pre_goal, {"goal": goal})

    async def _on_agent_done(self, event: AgentEvent) -> None:
        response = event.data.get("response", "")
        await self._run_hooks(self._config.post_goal, {"response": response})

    async def _on_tool_executing(self, event: AgentEvent) -> None:
        tool_name = event.data.get("tool", "")
        hooks = [h for h in self._config.pre_tool if h.matches_tool(tool_name)]
        await self._run_hooks(hooks, {"tool_name": tool_name})

    async def _on_tool_executed(self, event: AgentEvent) -> None:
        tool_name = event.data.get("tool", "")
        hooks = [h for h in self._config.post_tool if h.matches_tool(tool_name)]
        await self._run_hooks(hooks, {
            "tool_name": tool_name,
            "success": str(event.data.get("success", "")),
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
            # Simple template substitution: {{var_name}} → value
            for key, value in variables.items():
                cmd = cmd.replace(f"{{{{{key}}}}}", value)
            await self._exec(cmd, timeout=hook.timeout)

    async def _exec(self, cmd: str, timeout: int = _HOOK_TIMEOUT) -> None:
        """Execute a hook command asynchronously."""
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
