"""Policy engine — automatic guardrails for autonomous execution.

Instead of asking users for approval (which defeats the purpose of autonomy),
policies define what the agent CAN and CANNOT do. Policy violations are returned
as tool errors, letting the agent adapt automatically.

Tool categories:
  read     - read_file, list_dir, search_files, git_status, git_diff, git_log
  write    - write_file, edit_file
  execute  - shell, run_tests
  git      - git_commit, git_branch
  external - codex, gemini_cli
"""

from __future__ import annotations

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
}


@dataclass
class PolicyConfig:
    """Policy configuration loaded from config.yaml."""

    mode: str = "balanced"  # safe, balanced, full

    # Per-goal budgets (0 = unlimited)
    max_file_writes: int = 0
    max_shell_commands: int = 0
    max_git_commits: int = 0
    max_external_calls: int = 5

    # Path restrictions for file operations
    allowed_paths: list[str] = field(default_factory=list)  # globs, empty = all
    denied_paths: list[str] = field(
        default_factory=lambda: [
            "/etc/*", "/usr/*", "/bin/*", "/sbin/*", "/boot/*",
            "~/.ssh/*", "~/.gnupg/*", "**/.env", "**/.env.*",
            "**/credentials*", "**/secrets*",
        ]
    )

    # Shell command additions (on top of ShellToolConfig)
    blocked_shell_patterns: list[str] = field(
        default_factory=lambda: [
            "curl * | *sh", "wget * | *sh",  # pipe-to-shell
            "chmod 777", "chmod -R 777",
            "> /dev/sd*",
            "git push --force", "git push -f",
            "git reset --hard",
        ]
    )

    # Tool-level disable
    disabled_tools: list[str] = field(default_factory=list)


# Preset policies
PRESETS: dict[str, dict[str, Any]] = {
    "safe": {
        "mode": "safe",
        "max_file_writes": 20,
        "max_shell_commands": 30,
        "max_git_commits": 3,
        "max_external_calls": 2,
    },
    "balanced": {
        "mode": "balanced",
        "max_file_writes": 0,  # unlimited
        "max_shell_commands": 0,
        "max_git_commits": 10,
        "max_external_calls": 5,
    },
    "full": {
        "mode": "full",
        "max_file_writes": 0,
        "max_shell_commands": 0,
        "max_git_commits": 0,
        "max_external_calls": 0,
    },
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

    def record(self, tool_name: str, category: str):
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
        parts = []
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

    Usage:
        engine = PolicyEngine(config)
        engine.begin_goal()  # reset budgets
        violation = engine.check(tool_name, args)
        if violation:
            return error
        engine.record(tool_name)  # count toward budgets
    """

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()
        self.budget = BudgetUsage()
        self._project_root: Path | None = None

    def set_project_root(self, root: str | Path):
        self._project_root = Path(root).resolve()

    def begin_goal(self):
        """Reset budgets for a new goal."""
        self.budget = BudgetUsage()

    def check(self, tool_name: str, args: dict[str, Any]) -> PolicyViolation | None:
        """Check if a tool call is allowed. Returns None if OK."""
        category = TOOL_CATEGORIES.get(tool_name, "unknown")

        # Disabled tools
        if tool_name in self.config.disabled_tools:
            return PolicyViolation(
                rule="disabled_tool",
                message=f"Tool '{tool_name}' is disabled by policy.",
                tool=tool_name, category=category,
            )

        # Budget checks
        violation = self._check_budget(tool_name, category)
        if violation:
            return violation

        # Path checks for file tools
        if category == "write" or tool_name == "read_file":
            path = args.get("path", "")
            if path:
                violation = self._check_path(path, tool_name, category)
                if violation:
                    return violation

        # Shell command checks
        if tool_name == "shell":
            command = args.get("command", "")
            violation = self._check_shell(command, tool_name, category)
            if violation:
                return violation

        return None

    def record(self, tool_name: str):
        """Record a successful tool execution for budget tracking."""
        category = TOOL_CATEGORIES.get(tool_name, "unknown")
        self.budget.record(tool_name, category)

    def _check_budget(self, tool_name: str, category: str) -> PolicyViolation | None:
        cfg = self.config
        if category == "write" and cfg.max_file_writes > 0:
            if self.budget.file_writes >= cfg.max_file_writes:
                return PolicyViolation(
                    rule="budget_file_writes",
                    message=f"File write budget exhausted ({cfg.max_file_writes}). "
                            f"Summarize what you've done so far.",
                    tool=tool_name, category=category,
                )
        if category == "execute" and cfg.max_shell_commands > 0:
            if self.budget.shell_commands >= cfg.max_shell_commands:
                return PolicyViolation(
                    rule="budget_shell",
                    message=f"Shell command budget exhausted ({cfg.max_shell_commands}).",
                    tool=tool_name, category=category,
                )
        if tool_name == "git_commit" and cfg.max_git_commits > 0:
            if self.budget.git_commits >= cfg.max_git_commits:
                return PolicyViolation(
                    rule="budget_git_commits",
                    message=f"Git commit budget exhausted ({cfg.max_git_commits}).",
                    tool=tool_name, category=category,
                )
        if category == "external" and cfg.max_external_calls > 0:
            if self.budget.external_calls >= cfg.max_external_calls:
                return PolicyViolation(
                    rule="budget_external",
                    message=f"External agent call budget exhausted ({cfg.max_external_calls}).",
                    tool=tool_name, category=category,
                )
        return None

    def _check_path(self, path: str, tool_name: str, category: str) -> PolicyViolation | None:
        resolved = Path(path).expanduser().resolve()
        path_str = str(resolved)

        # Denied paths
        for pattern in self.config.denied_paths:
            expanded = str(Path(pattern).expanduser())
            if fnmatch.fnmatch(path_str, expanded) or fnmatch.fnmatch(resolved.name, pattern):
                return PolicyViolation(
                    rule="denied_path",
                    message=f"Access to '{path}' is denied by policy (matches '{pattern}'). "
                            f"Use a different path or approach.",
                    tool=tool_name, category=category,
                )

        # Allowed paths (if configured)
        if self.config.allowed_paths:
            matched = False
            for pattern in self.config.allowed_paths:
                expanded = str(Path(pattern).expanduser())
                if fnmatch.fnmatch(path_str, expanded):
                    matched = True
                    break
            # Also allow if within project root
            if not matched and self._project_root:
                try:
                    resolved.relative_to(self._project_root)
                    matched = True
                except ValueError:
                    pass
            if not matched:
                return PolicyViolation(
                    rule="allowed_paths",
                    message=f"Path '{path}' is outside allowed paths. "
                            f"Only project files and allowed paths are permitted.",
                    tool=tool_name, category=category,
                )

        return None

    def _check_shell(self, command: str, tool_name: str, category: str) -> PolicyViolation | None:
        cmd_lower = command.lower().strip()
        for pattern in self.config.blocked_shell_patterns:
            # Simple substring match with wildcard support
            pat_lower = pattern.lower()
            if "*" in pat_lower:
                if fnmatch.fnmatch(cmd_lower, pat_lower):
                    return PolicyViolation(
                        rule="blocked_shell_pattern",
                        message=f"Shell command blocked by policy: matches '{pattern}'. "
                                f"Try a safer alternative.",
                        tool=tool_name, category=category,
                    )
            else:
                if pat_lower in cmd_lower:
                    return PolicyViolation(
                        rule="blocked_shell_pattern",
                        message=f"Shell command blocked by policy: contains '{pattern}'. "
                                f"Try a safer alternative.",
                        tool=tool_name, category=category,
                    )
        return None


def load_policy(raw: dict[str, Any] | None) -> PolicyConfig:
    """Load policy config from raw dict (from config.yaml)."""
    if not raw:
        return PolicyConfig()

    mode = raw.get("mode", "balanced")
    # Start with preset, then override with explicit values
    base = dict(PRESETS.get(mode, PRESETS["balanced"]))
    base.update({k: v for k, v in raw.items() if v is not None})
    return PolicyConfig(**base)
