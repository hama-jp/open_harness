"""Policy engine -- automatic guardrails for autonomous execution.

Instead of asking users for approval (which defeats the purpose of autonomy),
policies define what the agent CAN and CANNOT do. Policy violations are returned
as tool errors, letting the agent adapt automatically.

Tool categories:
  read     - read_file, list_dir, search_files, git_status, git_diff, git_log
  write    - write_file, edit_file
  execute  - shell, run_tests
  git      - git_commit, git_branch
  external - codex, gemini_cli, claude_code
"""

from __future__ import annotations

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from open_harness_v2.config import PolicySpec

logger = logging.getLogger(__name__)

# Tool name -> category mapping
TOOL_CATEGORIES: dict[str, str] = {
    "read_file": "read",
    "list_dir": "read",
    "search_files": "read",
    "git_status": "read",
    "git_diff": "read",
    "git_log": "read",
    "write_file": "write",
    "edit_file": "write",
    "shell": "execute",
    "run_tests": "execute",
    "git_commit": "git",
    "git_branch": "git",
    "codex": "external",
    "gemini_cli": "external",
    "claude_code": "external",
}


@dataclass
class PolicyViolation:
    """Describes why a tool call was blocked."""

    rule: str
    message: str
    tool: str
    category: str


@dataclass
class BudgetUsage:
    """Tracks resource usage within a single goal execution."""

    file_writes: int = 0
    shell_commands: int = 0
    git_commits: int = 0
    external_calls: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)

    def record(self, tool_name: str, category: str) -> None:
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
        if category == "write":
            self.file_writes += 1
        elif category == "execute":
            self.shell_commands += 1
        elif category == "git" and tool_name == "git_commit":
            self.git_commits += 1
        elif category == "external":
            self.external_calls += 1

    def summary(self) -> str:
        elapsed = time.monotonic() - self.start_time
        parts: list[str] = []
        if self.file_writes:
            parts.append(f"writes:{self.file_writes}")
        if self.shell_commands:
            parts.append(f"shell:{self.shell_commands}")
        if self.git_commits:
            parts.append(f"commits:{self.git_commits}")
        if self.external_calls:
            parts.append(f"external:{self.external_calls}")
        total = sum(self.tool_calls.values())
        return f"tools:{total} ({', '.join(parts)}) in {elapsed:.0f}s"


class PolicyEngine:
    """Evaluates tool calls against the active policy.

    Usage::

        engine = PolicyEngine(spec)
        engine.begin_goal()             # reset budgets
        violation = engine.check(tool_name, args)
        if violation:
            return error
        engine.record(tool_name)        # count toward budgets
    """

    def __init__(self, spec: PolicySpec | None = None) -> None:
        self.spec = spec or PolicySpec()
        self.budget = BudgetUsage()
        self._project_root: Path | None = None
        self._token_usage: int = 0
        # Pre-compiled denied path patterns: [(expanded_str, parent_str, raw_pattern)]
        self._compiled_denied: list[tuple[str, str, str]] = []
        self._denied_cache: dict[str, bool] = {}
        self._compile_denied_patterns()

    def _compile_denied_patterns(self) -> None:
        """Pre-expand denied_paths into (expanded, parent, raw) tuples."""
        self._compiled_denied = []
        for pattern in self.spec.denied_paths:
            expanded = str(Path(pattern).expanduser())
            parent = expanded.removesuffix("/*").removesuffix("*")
            self._compiled_denied.append((expanded, parent, pattern))
        self._denied_cache.clear()

    def set_project_root(self, root: str | Path) -> None:
        """Set the project root for path restriction checks."""
        self._project_root = Path(root).resolve()
        self._denied_cache.clear()

    def begin_goal(self) -> None:
        """Reset budgets for a new goal."""
        self.budget = BudgetUsage()
        self._token_usage = 0

    def record_usage(self, usage: dict[str, int]) -> None:
        """Accumulate token usage from an LLM response."""
        self._token_usage += usage.get("total_tokens", 0)

    def check_token_budget(self) -> str | None:
        """Return a reason string if the token budget is exceeded, else None."""
        limit = self.spec.max_tokens_per_goal
        if limit > 0 and self._token_usage >= limit:
            return (
                f"Token budget exceeded: {self._token_usage}/{limit} tokens used. "
                f"Goal terminated to prevent runaway costs."
            )
        return None

    def check(self, tool_name: str, args: dict[str, Any]) -> PolicyViolation | None:
        """Check if a tool call is allowed. Returns None if OK."""
        category = TOOL_CATEGORIES.get(tool_name, "unknown")

        # Disabled tools
        if tool_name in self.spec.disabled_tools:
            return PolicyViolation(
                rule="disabled_tool",
                message=f"Tool '{tool_name}' is disabled by policy.",
                tool=tool_name,
                category=category,
            )

        # Budget checks
        violation = self._check_budget(tool_name, category)
        if violation:
            return violation

        # Path checks for file-accessing tools
        if tool_name in (
            "read_file", "write_file", "edit_file", "list_dir", "search_files",
        ):
            path = args.get("path", "")
            if path:
                if tool_name in ("write_file", "edit_file"):
                    violation = self._check_write_path(path, tool_name, category)
                else:
                    violation = self._check_read_path(path, tool_name, category)
                if violation:
                    return violation

        # Shell command checks
        if tool_name == "shell":
            command = args.get("command", "")
            violation = self._check_shell(command, tool_name, category)
            if violation:
                return violation

        return None

    def record(self, tool_name: str) -> None:
        """Record a successful tool execution for budget tracking."""
        category = TOOL_CATEGORIES.get(tool_name, "unknown")
        self.budget.record(tool_name, category)

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_budget(
        self, tool_name: str, category: str
    ) -> PolicyViolation | None:
        spec = self.spec
        if category == "write" and spec.max_file_writes > 0:
            if self.budget.file_writes >= spec.max_file_writes:
                return PolicyViolation(
                    rule="budget_file_writes",
                    message=f"File write budget exhausted ({spec.max_file_writes}). "
                            f"Summarize what you've done so far.",
                    tool=tool_name,
                    category=category,
                )
        if category == "execute" and spec.max_shell_commands > 0:
            if self.budget.shell_commands >= spec.max_shell_commands:
                return PolicyViolation(
                    rule="budget_shell",
                    message=f"Shell command budget exhausted ({spec.max_shell_commands}).",
                    tool=tool_name,
                    category=category,
                )
        if tool_name == "git_commit" and spec.max_git_commits > 0:
            if self.budget.git_commits >= spec.max_git_commits:
                return PolicyViolation(
                    rule="budget_git_commits",
                    message=f"Git commit budget exhausted ({spec.max_git_commits}).",
                    tool=tool_name,
                    category=category,
                )
        if category == "external" and spec.max_external_calls > 0:
            if self.budget.external_calls >= spec.max_external_calls:
                return PolicyViolation(
                    rule="budget_external",
                    message=f"External agent call budget exhausted ({spec.max_external_calls}).",
                    tool=tool_name,
                    category=category,
                )
        return None

    def _check_denied(
        self, path_str: str, resolved: Path, path: str,
        tool_name: str, category: str,
    ) -> PolicyViolation | None:
        """Check if a path matches any denied pattern."""
        cached = self._denied_cache.get(path_str)
        if cached is not None:
            if not cached:
                return None
            # Cache says denied; fall through to find the pattern for the message.

        resolved_name = resolved.name
        for expanded, parent, raw_pattern in self._compiled_denied:
            if (
                fnmatch.fnmatch(path_str, expanded)
                or path_str == parent
                or path_str.startswith(parent + "/")
                or fnmatch.fnmatch(resolved_name, raw_pattern)
            ):
                if len(self._denied_cache) < 256:
                    self._denied_cache[path_str] = True
                return PolicyViolation(
                    rule="denied_path",
                    message=f"Access to '{path}' is denied by policy (matches '{raw_pattern}'). "
                            f"Use a different path or approach.",
                    tool=tool_name,
                    category=category,
                )
        if len(self._denied_cache) < 256:
            self._denied_cache[path_str] = False
        return None

    def _check_read_path(
        self, path: str, tool_name: str, category: str
    ) -> PolicyViolation | None:
        """Check read access against denied_paths."""
        resolved = Path(path).expanduser().resolve()
        path_str = str(resolved)

        return self._check_denied(path_str, resolved, path, tool_name, category)

    def _check_write_path(
        self, path: str, tool_name: str, category: str
    ) -> PolicyViolation | None:
        """Check write access: denied_paths, then restrict to project root + writable_paths."""
        resolved = Path(path).expanduser().resolve()
        path_str = str(resolved)

        # 1. Denied paths check
        violation = self._check_denied(path_str, resolved, path, tool_name, category)
        if violation:
            return violation

        # 2. Project root check -- always allowed
        if self._project_root:
            try:
                resolved.relative_to(self._project_root)
                return None  # inside project root
            except ValueError:
                pass

        # 3. writable_paths check
        for pattern in self.spec.writable_paths:
            expanded = str(Path(pattern).expanduser())
            if fnmatch.fnmatch(path_str, expanded):
                return None
            parent = expanded.removesuffix("/*").removesuffix("*")
            if path_str.startswith(parent + "/") or path_str == parent:
                return None

        # 4. Blocked -- outside project root
        hint = (
            "Add the path to 'writable_paths' in your policy config, "
            "or use 'policy.mode: full' to allow writes to the home directory."
        )
        return PolicyViolation(
            rule="write_outside_project",
            message=f"Write to '{path}' is denied: outside project root"
                    f"{f' ({self._project_root})' if self._project_root else ''}. {hint}",
            tool=tool_name,
            category=category,
        )

    def _check_shell(
        self, command: str, tool_name: str, category: str
    ) -> PolicyViolation | None:
        """Check shell command against blocked patterns."""
        cmd_lower = command.lower().strip()
        for pattern in self.spec.blocked_shell_patterns:
            pat_lower = pattern.lower()
            if "*" in pat_lower:
                if fnmatch.fnmatch(cmd_lower, pat_lower):
                    return PolicyViolation(
                        rule="blocked_shell_pattern",
                        message=f"Shell command blocked by policy: matches '{pattern}'. "
                                f"Try a safer alternative.",
                        tool=tool_name,
                        category=category,
                    )
            else:
                if pat_lower in cmd_lower:
                    return PolicyViolation(
                        rule="blocked_shell_pattern",
                        message=f"Shell command blocked by policy: contains '{pattern}'. "
                                f"Try a safer alternative.",
                        tool=tool_name,
                        category=category,
                    )
        return None
