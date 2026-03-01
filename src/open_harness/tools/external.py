"""External agent tools - Codex CLI, Gemini CLI, etc."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from open_harness.tools.base import Tool, ToolParameter, ToolResult


class CodexTool(Tool):
    """Delegate tasks to OpenAI Codex CLI."""

    name = "codex"
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

    def __init__(self, command: str = "codex"):
        self.command = command
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

        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        try:
            result = subprocess.run(
                [self.command, "--approval-mode", "full-auto", prompt],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=cwd,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}" if output else result.stderr
            return ToolResult(
                success=result.returncode == 0,
                output=output.strip(),
                error="" if result.returncode == 0 else f"Exit code: {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Codex timed out after 300s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ClaudeCodeTool(Tool):
    """Delegate tasks to Claude Code (Anthropic CLI)."""

    name = "claude_code"
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

    def __init__(self, command: str = "claude"):
        self.command = command
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

        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        try:
            result = subprocess.run(
                [
                    self.command, "-p", prompt,
                    "--allowedTools", "Bash", "Read", "Write", "Edit",
                    "Glob", "Grep",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=cwd,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}" if output else result.stderr
            return ToolResult(
                success=result.returncode == 0,
                output=output.strip(),
                error="" if result.returncode == 0 else f"Exit code: {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Claude Code timed out after 300s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class GeminiCliTool(Tool):
    """Delegate tasks to Google Gemini CLI."""

    name = "gemini_cli"
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

    def __init__(self, command: str = "gemini"):
        self.command = command
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

        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        try:
            result = subprocess.run(
                [self.command, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=cwd,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}" if output else result.stderr
            return ToolResult(
                success=result.returncode == 0,
                output=output.strip(),
                error="" if result.returncode == 0 else f"Exit code: {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Gemini CLI timed out after 300s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
