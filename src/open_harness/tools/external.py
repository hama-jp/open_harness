"""External agent tools - Codex CLI, Claude Code, Gemini CLI.

Uses subprocess.Popen for real-time output streaming instead of
subprocess.run, which would buffer all output until process completion.
"""

from __future__ import annotations

import logging
import os
import signal
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

from open_harness.tools.base import Tool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Default timeout (seconds) if not overridden by config.
_DEFAULT_TIMEOUT = 600

# Type alias for progress callback: receives a line of output.
ProgressCallback = Callable[[str], None]


def _run_streaming(
    cmd: list[str],
    *,
    cwd: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    progress_callback: ProgressCallback | None = None,
    tool_label: str = "external agent",
) -> ToolResult:
    """Run a CLI command with real-time stdout streaming.

    Instead of subprocess.run(capture_output=True) which buffers everything,
    this uses Popen with a reader thread so that:

    1. Progress can be shown to the user in real time via progress_callback.
    2. On timeout, all output collected so far is returned (not discarded).
    3. The child process tree is properly killed on timeout.

    Architecture:
    - A reader thread reads stdout line-by-line (readline blocks, but only
      in the reader thread — not the main thread).
    - The main thread waits for the process to finish with proc.wait(timeout=...).
    - On timeout, the main thread kills the process tree, which unblocks
      the reader thread (the pipe closes → readline returns "").
    """
    stdout_lines: list[str] = []
    stderr_data: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            # Start a new process group so we can kill the entire tree on timeout
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
    except FileNotFoundError:
        return ToolResult(
            success=False, output="",
            error=f"Command not found: {cmd[0]}",
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))

    def _read_stdout():
        """Read stdout line-by-line in a thread. Runs until pipe closes."""
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            stdout_lines.append(line)
            if progress_callback:
                try:
                    progress_callback(line.rstrip("\n"))
                except Exception:
                    pass  # callback errors must not kill the reader

    def _read_stderr():
        """Read stderr in a thread."""
        assert proc.stderr is not None
        stderr_data.append(proc.stderr.read())

    reader = threading.Thread(target=_read_stdout, daemon=True)
    stderr_reader = threading.Thread(target=_read_stderr, daemon=True)
    reader.start()
    stderr_reader.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)

    # Wait for reader threads to finish (pipe closes after kill)
    reader.join(timeout=5)
    stderr_reader.join(timeout=5)

    if timed_out:
        partial = "".join(stdout_lines).strip()
        n_lines = len(stdout_lines)
        return ToolResult(
            success=False,
            output=partial,
            error=(
                f"{tool_label} timed out after {timeout}s "
                f"({n_lines} lines of output captured)"
            ),
        )

    output = "".join(stdout_lines)
    stderr = "".join(stderr_data)
    if stderr:
        output += f"\n[stderr]\n{stderr}" if output else stderr

    return ToolResult(
        success=proc.returncode == 0,
        output=output.strip(),
        error="" if proc.returncode == 0 else f"Exit code: {proc.returncode}",
    )


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a process and its entire process group."""
    try:
        if os.name != "nt":
            # Kill the entire process group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, OSError):
        pass  # already dead
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass


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

    def __init__(self, command: str = "codex", timeout: int = _DEFAULT_TIMEOUT):
        self.command = command
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.command) is not None
        return self._available

    def execute(self, **kwargs: Any) -> ToolResult:
        if not self.available:
            return ToolResult(
                success=False, output="",
                error=f"'{self.command}' not found in PATH",
            )

        prompt = kwargs.get("prompt", "")
        cwd = kwargs.get("cwd")
        progress_callback = kwargs.get("progress_callback")

        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        return _run_streaming(
            [self.command, "exec", "--full-auto", prompt],
            cwd=cwd,
            timeout=self.timeout,
            progress_callback=progress_callback,
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

    def __init__(self, command: str = "claude", timeout: int = _DEFAULT_TIMEOUT):
        self.command = command
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.command) is not None
        return self._available

    def execute(self, **kwargs: Any) -> ToolResult:
        if not self.available:
            return ToolResult(
                success=False, output="",
                error=f"'{self.command}' not found in PATH",
            )

        prompt = kwargs.get("prompt", "")
        cwd = kwargs.get("cwd")
        progress_callback = kwargs.get("progress_callback")

        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        return _run_streaming(
            [
                self.command, "-p", prompt,
                "--allowedTools", "Bash", "Read", "Write", "Edit",
                "Glob", "Grep",
            ],
            cwd=cwd,
            timeout=self.timeout,
            progress_callback=progress_callback,
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

    def __init__(self, command: str = "gemini", timeout: int = _DEFAULT_TIMEOUT):
        self.command = command
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.command) is not None
        return self._available

    def execute(self, **kwargs: Any) -> ToolResult:
        if not self.available:
            return ToolResult(
                success=False, output="",
                error=f"'{self.command}' not found in PATH",
            )

        prompt = kwargs.get("prompt", "")
        cwd = kwargs.get("cwd")
        progress_callback = kwargs.get("progress_callback")

        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        return _run_streaming(
            [self.command, "-p", prompt, "-y"],
            cwd=cwd,
            timeout=self.timeout,
            progress_callback=progress_callback,
            tool_label="Gemini CLI",
        )
