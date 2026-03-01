"""Async git workflow tools for Open Harness v2."""

from __future__ import annotations

import asyncio
from typing import Any

from open_harness_v2.tools.base import Tool
from open_harness_v2.types import ToolParameter, ToolResult


async def _git(
    args: list[str], cwd: str | None = None, timeout: int = 30
) -> tuple[int, str, str]:
    """Run a git command asynchronously, returning (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 1, "", f"git {args[0]} timed out after {timeout}s"

    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


class GitStatusTool(Tool):
    """Show git status."""

    name = "git_status"
    description = "Show git status: modified, staged, and untracked files."
    parameters: list[ToolParameter] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        rc, out, err = await _git(["status", "--short"])
        rc2, diff_out, _ = await _git(["diff", "--stat"])
        output = out.strip()
        if diff_out.strip():
            output += f"\n\nDiff summary:\n{diff_out.strip()}"
        return ToolResult(
            success=rc == 0,
            output=output or "(clean)",
            error=err.strip() if rc != 0 else "",
        )


class GitDiffTool(Tool):
    """Show the diff of current changes."""

    name = "git_diff"
    description = "Show the diff of current changes (unstaged by default)."
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Specific file to diff",
            required=False,
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        cmd = ["diff"]
        if path:
            cmd += ["--", path]
        rc, out, err = await _git(cmd)
        output = out.strip() or "(no changes)"
        if len(output) > 15000:
            output = output[:15000] + "\n... [truncated]"
        return ToolResult(
            success=rc == 0,
            output=output,
            error=err.strip() if rc != 0 else "",
        )


class GitLogTool(Tool):
    """Show recent git log."""

    name = "git_log"
    description = "Show recent git log."
    parameters = [
        ToolParameter(
            name="count",
            type="integer",
            description="Number of commits to show",
            required=False,
            default=10,
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        count = kwargs.get("count", 10)
        rc, out, err = await _git(["log", "--oneline", f"-{count}"])
        return ToolResult(
            success=rc == 0,
            output=out.strip() or "(no commits)",
            error=err.strip() if rc != 0 else "",
        )


class GitCommitTool(Tool):
    """Stage files and create a git commit."""

    name = "git_commit"
    description = "Stage files and create a git commit."
    parameters = [
        ToolParameter(
            name="message",
            type="string",
            description="Commit message",
        ),
        ToolParameter(
            name="files",
            type="string",
            description="Files to stage (space-separated). Use '.' for all changed files.",
            required=False,
            default=".",
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        message = kwargs.get("message", "")
        files = kwargs.get("files", ".")

        if not message:
            return ToolResult(success=False, output="", error="No commit message provided")

        rc, out, err = await _git(["add"] + files.split())
        if rc != 0:
            return ToolResult(success=False, output="", error=f"git add failed: {err}")

        rc, out, err = await _git(["commit", "-m", message])
        if rc != 0:
            stderr = err.strip()
            if "nothing to commit" in stderr or "nothing to commit" in out:
                return ToolResult(success=True, output="Nothing to commit (working tree clean)")
            return ToolResult(success=False, output="", error=f"git commit failed: {stderr}")

        return ToolResult(success=True, output=out.strip())


class GitBranchTool(Tool):
    """Create a new branch or list branches."""

    name = "git_branch"
    description = "Create a new branch or list branches."
    parameters = [
        ToolParameter(
            name="name",
            type="string",
            description="Branch name to create (omit to list branches)",
            required=False,
        ),
        ToolParameter(
            name="checkout",
            type="boolean",
            description="Also checkout the new branch",
            required=False,
            default=True,
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        name = kwargs.get("name", "")
        checkout = kwargs.get("checkout", True)

        if not name:
            rc, out, err = await _git(["branch", "-a"])
            return ToolResult(
                success=rc == 0,
                output=out.strip(),
                error=err.strip() if rc != 0 else "",
            )

        cmd = ["checkout", "-b", name] if checkout else ["branch", name]
        rc, out, err = await _git(cmd)
        if rc != 0:
            return ToolResult(success=False, output="", error=err.strip())
        return ToolResult(
            success=True,
            output=f"Created branch: {name}" + (" (checked out)" if checkout else ""),
        )
