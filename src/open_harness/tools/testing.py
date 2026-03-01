"""Test runner tool — runs tests and parses results for the agent."""

from __future__ import annotations

import subprocess
from typing import Any

from open_harness.tools.base import Tool, ToolParameter, ToolResult


class TestRunnerTool(Tool):
    """Run project tests and return results."""

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
            description="Specific test file or pattern (e.g., 'tests/test_foo.py', 'tests/'). Leave empty to run all tests.",
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

    def __init__(self, test_command: str = "python -m pytest", cwd: str | None = None):
        self.test_command = test_command
        self.cwd = cwd

    def execute(self, **kwargs: Any) -> ToolResult:
        import shlex

        target = kwargs.get("target", "")
        verbose = kwargs.get("verbose", False)

        cmd_parts = shlex.split(self.test_command)
        if verbose and "pytest" in self.test_command:
            cmd_parts.append("-v")
        if target:
            # Sanitize: only allow path-like targets
            for part in shlex.split(target):
                cmd_parts.append(part)

        # Add short summary for pytest
        if "pytest" in self.test_command and "--tb" not in " ".join(cmd_parts):
            cmd_parts.append("--tb=short")

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.cwd,
            )
            output = result.stdout
            if result.stderr:
                # pytest puts progress to stderr
                output += f"\n{result.stderr}" if output else result.stderr

            # Truncate but keep the summary at the end
            if len(output) > 15000:
                lines = output.splitlines()
                # Keep first 50 and last 100 lines (summary is at the end)
                head = "\n".join(lines[:50])
                tail = "\n".join(lines[-100:])
                output = f"{head}\n\n... [{len(lines) - 150} lines truncated] ...\n\n{tail}"

            passed = result.returncode == 0
            return ToolResult(
                success=passed,
                output=output.strip(),
                error="" if passed else f"Tests failed (exit code {result.returncode})",
                metadata={"returncode": result.returncode, "all_passed": passed},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Tests timed out after 120s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
