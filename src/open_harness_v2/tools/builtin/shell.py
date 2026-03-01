"""Async shell command execution tool for Open Harness v2."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from open_harness_v2.tools.base import Tool
from open_harness_v2.types import ToolParameter, ToolResult

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


def _build_safe_env() -> dict[str, str]:
    """Build an environment dict with sensitive variables stripped."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(k.startswith(p) for p in _SENSITIVE_PREFIXES)
        and k not in _SENSITIVE_NAMES
    }


class ShellTool(Tool):
    """Execute shell commands asynchronously."""

    name = "shell"
    description = (
        "Execute a shell command and return the output. "
        "Use for running programs, git operations, system commands, etc."
    )
    max_output = 8000
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="The shell command to execute",
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory for the command",
            required=False,
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout in seconds",
            required=False,
            default=120,
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command", "")
        cwd = kwargs.get("cwd")
        timeout = kwargs.get("timeout", 120)

        if not command:
            return ToolResult(success=False, output="", error="No command provided")

        safe_env = _build_safe_env()

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=safe_env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout}s",
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
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to execute: {e}",
            )
