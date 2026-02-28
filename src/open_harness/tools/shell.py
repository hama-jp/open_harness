"""Shell command execution tool."""

from __future__ import annotations

import subprocess
from typing import Any

from open_harness.config import ShellToolConfig
from open_harness.tools.base import Tool, ToolParameter, ToolResult


class ShellTool(Tool):
    """Execute shell commands."""

    name = "shell"
    description = "Execute a shell command and return the output. Use for running programs, git operations, system commands, etc."
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="The shell command to execute",
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout in seconds",
            required=False,
            default=30,
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory for the command",
            required=False,
        ),
    ]

    def __init__(self, config: ShellToolConfig | None = None):
        self.config = config or ShellToolConfig()

    def _is_blocked(self, command: str) -> str | None:
        for blocked in self.config.blocked_commands:
            if blocked in command:
                return f"Command blocked for safety: contains '{blocked}'"
        return None

    def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", self.config.timeout)
        cwd = kwargs.get("cwd")

        if not command:
            return ToolResult(success=False, output="", error="No command provided")

        blocked = self._is_blocked(command)
        if blocked:
            return ToolResult(success=False, output="", error=blocked)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}" if output else result.stderr

            # Truncate very long outputs
            if len(output) > 50000:
                output = output[:25000] + "\n...[truncated]...\n" + output[-25000:]

            return ToolResult(
                success=result.returncode == 0,
                output=output.strip(),
                error="" if result.returncode == 0 else f"Exit code: {result.returncode}",
                metadata={"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to execute: {e}",
            )
