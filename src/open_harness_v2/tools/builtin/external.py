"""Async external agent tools — Codex CLI, Claude Code, Gemini CLI.

Each tool delegates a task to an external CLI agent via
asyncio.create_subprocess_exec, with timeout handling and
process-tree cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import shutil
from typing import Any

from open_harness_v2.tools.base import Tool
from open_harness_v2.types import ToolParameter, ToolResult

_logger = logging.getLogger(__name__)

# Default timeout (seconds) for external agent calls.
_DEFAULT_TIMEOUT = 600


async def _run_external(
    cmd: list[str],
    *,
    cwd: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    tool_label: str = "external agent",
) -> ToolResult:
    """Run an external CLI agent asynchronously with timeout handling.

    On timeout the entire process tree is killed and any partial
    output captured so far is returned.
    """
    try:
        # Start a new process group so the entire tree can be killed
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
    except FileNotFoundError:
        return ToolResult(
            success=False, output="",
            error=f"Command not found: {cmd[0]}",
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        _kill_process_tree(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        return ToolResult(
            success=False,
            output="",
            error=f"{tool_label} timed out after {timeout}s",
        )

    output = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    if stderr_text:
        output += f"\n[stderr]\n{stderr_text}" if output else stderr_text

    returncode = proc.returncode or 0
    return ToolResult(
        success=returncode == 0,
        output=output.strip(),
        error="" if returncode == 0 else f"Exit code: {returncode}",
        metadata={"returncode": returncode},
    )


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a process and its entire process group."""
    try:
        if os.name != "nt" and proc.pid is not None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, OSError):
        pass  # already dead


# ------------------------------------------------------------------
# Tool classes
# ------------------------------------------------------------------

class CodexTool(Tool):
    """Delegate tasks to OpenAI Codex CLI."""

    name = "codex"
    max_output = 5000
    description = (
        "Delegate a coding task to OpenAI Codex CLI agent. "
        "Best for complex code generation, refactoring, and debugging tasks. "
        "Codex has its own sandbox and can read/write files."
    )
    parameters = [
        ToolParameter(
            name="prompt",
            type="string",
            description="The task description to send to Codex",
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory for Codex",
            required=False,
        ),
    ]

    def __init__(self, command: str = "codex", timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.command = command
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.command) is not None
        return self._available

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self.available:
            return ToolResult(
                success=False, output="",
                error=f"'{self.command}' not found in PATH",
            )
        prompt = kwargs.get("prompt", "")
        cwd = kwargs.get("cwd")
        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        return await _run_external(
            [self.command, "exec", "--full-auto", prompt],
            cwd=cwd,
            timeout=self.timeout,
            tool_label="Codex",
        )


class ClaudeCodeTool(Tool):
    """Delegate tasks to Claude Code (Anthropic CLI)."""

    name = "claude_code"
    max_output = 5000
    description = (
        "Delegate a coding task to Claude Code (Anthropic CLI agent). "
        "Best for code generation, code analysis, complex reasoning, and refactoring. "
        "Claude Code has its own sandbox and can read/write files."
    )
    parameters = [
        ToolParameter(
            name="prompt",
            type="string",
            description="The task description to send to Claude Code",
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory for Claude Code",
            required=False,
        ),
    ]

    def __init__(self, command: str = "claude", timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.command = command
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.command) is not None
        return self._available

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self.available:
            return ToolResult(
                success=False, output="",
                error=f"'{self.command}' not found in PATH",
            )
        prompt = kwargs.get("prompt", "")
        cwd = kwargs.get("cwd")
        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        return await _run_external(
            [
                self.command, "-p", prompt,
                "--allowedTools", "Bash", "Read", "Write", "Edit",
                "Glob", "Grep",
            ],
            cwd=cwd,
            timeout=self.timeout,
            tool_label="Claude Code",
        )


class GeminiCliTool(Tool):
    """Delegate tasks to Google Gemini CLI."""

    name = "gemini_cli"
    max_output = 5000
    description = (
        "Delegate a task to Google Gemini CLI agent. "
        "Useful for tasks that benefit from Gemini's capabilities."
    )
    parameters = [
        ToolParameter(
            name="prompt",
            type="string",
            description="The task description to send to Gemini CLI",
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory",
            required=False,
        ),
    ]

    def __init__(self, command: str = "gemini", timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.command = command
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.command) is not None
        return self._available

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self.available:
            return ToolResult(
                success=False, output="",
                error=f"'{self.command}' not found in PATH",
            )
        prompt = kwargs.get("prompt", "")
        cwd = kwargs.get("cwd")
        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        return await _run_external(
            [self.command, "-p", prompt, "-y"],
            cwd=cwd,
            timeout=self.timeout,
            tool_label="Gemini CLI",
        )
