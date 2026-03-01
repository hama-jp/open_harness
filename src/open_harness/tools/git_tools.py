"""Git workflow tools for autonomous coding."""

from __future__ import annotations

import subprocess
from typing import Any

from open_harness.tools.base import Tool, ToolParameter, ToolResult


def _git(args: list[str], cwd: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show git status: modified, staged, and untracked files."
    parameters = [
        ToolParameter(name="cwd", type="string", description="Repository directory", required=False),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        cwd = kwargs.get("cwd")
        r = _git(["status", "--short"], cwd=cwd)
        diff = _git(["diff", "--stat"], cwd=cwd)
        output = r.stdout.strip()
        if diff.stdout.strip():
            output += f"\n\nDiff summary:\n{diff.stdout.strip()}"
        return ToolResult(success=r.returncode == 0, output=output or "(clean)")


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Show the diff of current changes (unstaged by default, or staged with staged=true)."
    parameters = [
        ToolParameter(name="staged", type="boolean", description="Show staged changes", required=False, default=False),
        ToolParameter(name="path", type="string", description="Specific file to diff", required=False),
        ToolParameter(name="cwd", type="string", description="Repository directory", required=False),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        staged = kwargs.get("staged", False)
        path = kwargs.get("path", "")
        cwd = kwargs.get("cwd")
        cmd = ["diff", "--staged"] if staged else ["diff"]
        if path:
            cmd += ["--", path]
        r = _git(cmd, cwd=cwd)
        output = r.stdout.strip() or "(no changes)"
        if len(output) > 15000:
            output = output[:15000] + "\n... [truncated]"
        return ToolResult(success=r.returncode == 0, output=output)


class GitCommitTool(Tool):
    name = "git_commit"
    description = "Stage files and create a git commit."
    parameters = [
        ToolParameter(name="message", type="string", description="Commit message"),
        ToolParameter(name="files", type="string", description="Files to stage (space-separated). Use '.' for all changed files.", required=False, default="."),
        ToolParameter(name="cwd", type="string", description="Repository directory", required=False),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        message = kwargs.get("message", "")
        files = kwargs.get("files", ".")
        cwd = kwargs.get("cwd")

        if not message:
            return ToolResult(success=False, output="", error="No commit message provided")

        add_r = _git(["add"] + files.split(), cwd=cwd)
        if add_r.returncode != 0:
            return ToolResult(success=False, output="", error=f"git add failed: {add_r.stderr}")

        commit_r = _git(["commit", "-m", message], cwd=cwd)
        if commit_r.returncode != 0:
            stderr = commit_r.stderr.strip()
            if "nothing to commit" in stderr or "nothing to commit" in commit_r.stdout:
                return ToolResult(success=True, output="Nothing to commit (working tree clean)")
            return ToolResult(success=False, output="", error=f"git commit failed: {stderr}")

        return ToolResult(success=True, output=commit_r.stdout.strip())


class GitBranchTool(Tool):
    name = "git_branch"
    description = "Create a new branch or list branches."
    parameters = [
        ToolParameter(name="name", type="string", description="Branch name to create (omit to list branches)", required=False),
        ToolParameter(name="checkout", type="boolean", description="Also checkout the new branch", required=False, default=True),
        ToolParameter(name="cwd", type="string", description="Repository directory", required=False),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        name = kwargs.get("name", "")
        checkout = kwargs.get("checkout", True)
        cwd = kwargs.get("cwd")

        if not name:
            r = _git(["branch", "-a"], cwd=cwd)
            return ToolResult(success=r.returncode == 0, output=r.stdout.strip())

        cmd = ["checkout", "-b", name] if checkout else ["branch", name]
        r = _git(cmd, cwd=cwd)
        if r.returncode != 0:
            return ToolResult(success=False, output="", error=r.stderr.strip())
        return ToolResult(success=True, output=f"Created branch: {name}" + (" (checked out)" if checkout else ""))


class GitLogTool(Tool):
    name = "git_log"
    description = "Show recent git log."
    parameters = [
        ToolParameter(name="count", type="integer", description="Number of commits to show", required=False, default=10),
        ToolParameter(name="cwd", type="string", description="Repository directory", required=False),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        count = kwargs.get("count", 10)
        cwd = kwargs.get("cwd")
        r = _git(["log", "--oneline", f"-{count}"], cwd=cwd)
        return ToolResult(success=r.returncode == 0, output=r.stdout.strip() or "(no commits)")
