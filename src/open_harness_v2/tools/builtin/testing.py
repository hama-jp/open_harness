"""Async test runner tool for Open Harness v2."""

from __future__ import annotations

import asyncio
import shlex
from typing import Any

from open_harness_v2.tools.base import Tool
from open_harness_v2.types import ToolParameter, ToolResult


class TestRunnerTool(Tool):
    """Run project tests asynchronously and return results."""

    name = "run_tests"
    max_output = 4000
    description = (
        "Run the project's test suite (or a specific test file/pattern). "
        "Returns test output including pass/fail status. "
        "Use this to verify code changes work correctly."
    )
    parameters = [
        ToolParameter(
            name="target",
            type="string",
            description=(
                "Specific test file or pattern (e.g., 'tests/test_foo.py', 'tests/'). "
                "Leave empty to run all tests."
            ),
            required=False,
        ),
        ToolParameter(
            name="verbose",
            type="boolean",
            description="Show verbose test output",
            required=False,
            default=False,
        ),
    ]

    def __init__(
        self,
        test_command: str = "python -m pytest",
        cwd: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.test_command = test_command
        self.cwd = cwd
        self.timeout = timeout

    async def execute(self, **kwargs: Any) -> ToolResult:
        target = kwargs.get("target", "")
        verbose = kwargs.get("verbose", False)

        cmd_parts = shlex.split(self.test_command)
        if verbose and "pytest" in self.test_command:
            cmd_parts.append("-v")
        if target:
            for part in shlex.split(target):
                cmd_parts.append(part)

        # Add short traceback for pytest unless already specified
        if "pytest" in self.test_command and "--tb" not in " ".join(cmd_parts):
            cmd_parts.append("--tb=short")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Tests timed out after {self.timeout}s",
                )

            output = stdout.decode(errors="replace")
            stderr_text = stderr.decode(errors="replace")
            if stderr_text:
                output += f"\n{stderr_text}" if output else stderr_text

            # Truncate but keep the summary at the end
            if len(output) > 15000:
                lines = output.splitlines()
                head = "\n".join(lines[:50])
                tail = "\n".join(lines[-100:])
                output = (
                    f"{head}\n\n"
                    f"... [{len(lines) - 150} lines truncated] ...\n\n"
                    f"{tail}"
                )

            passed = (proc.returncode or 0) == 0
            return ToolResult(
                success=passed,
                output=output.strip(),
                error="" if passed else f"Tests failed (exit code {proc.returncode})",
                metadata={"returncode": proc.returncode or 0, "all_passed": passed},
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error=f"Test command not found: {cmd_parts[0]}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
