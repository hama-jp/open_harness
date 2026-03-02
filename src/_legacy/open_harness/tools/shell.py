"""Shell command execution tool."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from open_harness.config import ShellToolConfig
from open_harness.tools.base import Tool, ToolParameter, ToolResult

# Environment variable prefixes that should not leak to child processes
_SENSITIVE_PREFIXES = (
    "AWS_", "OPENAI_", "ANTHROPIC_", "GITHUB_", "GH_",
    "AZURE_", "GOOGLE_", "HF_", "HUGGING",
)

# Exact environment variable names to strip
_SENSITIVE_NAMES = frozenset({
    "API_KEY", "SECRET", "TOKEN", "PASSWORD", "DATABASE_URL",
    "SECRET_KEY", "PRIVATE_KEY",
})


class ShellTool(Tool):
    """Execute shell commands."""

    name = "shell"
    description = "Execute a shell command and return the output. Use for running programs, git operations, system commands, etc."
    max_output = 3000
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
        self._safe_env: dict[str, str] | None = None

    def _build_safe_env(self) -> dict[str, str]:
        """Build an environment dict with sensitive variables stripped.

        Preserves PATH, HOME, LANG, USER, SHELL, TERM and other basic
        variables while removing API keys, tokens, and cloud credentials.
        """
        if self._safe_env is None:
            self._safe_env = {
                k: v for k, v in os.environ.items()
                if not any(k.startswith(p) for p in _SENSITIVE_PREFIXES)
                and k not in _SENSITIVE_NAMES
            }
        return self._safe_env

    def _check_safety(self, command: str) -> str | None:
        """Check command against allowlist and blocklist.

        Returns an error message if the command is not allowed, None if OK.
        """
        # Blocklist takes priority
        for blocked in self.config.blocked_commands:
            if blocked in command:
                return f"Command blocked for safety: contains '{blocked}'"

        # If allowlist is configured, command must match at least one entry
        if self.config.allowed_commands:
            cmd_base = command.split()[0] if command.split() else ""
            if not any(cmd_base == allowed or command.startswith(allowed)
                       for allowed in self.config.allowed_commands):
                return (
                    f"Command not in allowlist. Allowed: {', '.join(self.config.allowed_commands)}"
                )

        return None

    def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", self.config.timeout)
        cwd = kwargs.get("cwd")

        if not command:
            return ToolResult(success=False, output="", error="No command provided")

        safety_error = self._check_safety(command)
        if safety_error:
            return ToolResult(success=False, output="", error=safety_error)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=self._build_safe_env(),
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
