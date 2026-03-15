"""Approval system — interactive human-in-the-loop for dangerous operations.

Inspired by Codex CLI's approval modes. Three levels:

- ``suggest``    — all tool calls require approval (except reads)
- ``auto-edit``  — reads and writes auto-approved; shell/git need approval
- ``full-auto``  — everything runs without approval

Usage::

    approver = ApprovalEngine(mode=ApprovalMode.AUTO_EDIT)
    decision = await approver.check(tool_name, tool_args)
    if decision == ApprovalDecision.DENIED:
        # user declined the action
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)

# Tool categories for approval classification
_READ_TOOLS = frozenset({
    "read_file", "list_dir", "search_files",
    "git_status", "git_diff", "git_log",
})

_WRITE_TOOLS = frozenset({
    "write_file", "edit_file",
})

_EXECUTE_TOOLS = frozenset({
    "shell", "run_tests",
})

_GIT_MUTATE_TOOLS = frozenset({
    "git_commit", "git_branch",
})

_EXTERNAL_TOOLS = frozenset({
    "codex", "gemini_cli", "claude_code",
})


class ApprovalMode(enum.Enum):
    """How much autonomy the agent has."""

    SUGGEST = "suggest"        # Ask for everything except reads
    AUTO_EDIT = "auto-edit"    # Auto-approve reads + writes; ask for shell/git
    FULL_AUTO = "full-auto"    # Never ask


class ApprovalDecision(enum.Enum):
    """Result of an approval check."""

    APPROVED = "approved"
    DENIED = "denied"
    SKIPPED = "skipped"  # No approval needed


@dataclass
class ApprovalRequest:
    """Details of an action requiring approval."""

    tool_name: str
    arguments: dict[str, Any]
    reason: str  # Why approval is needed
    category: str  # read/write/execute/git/external


class ApprovalEngine:
    """Interactive approval system for tool execution.

    Parameters
    ----------
    mode:
        Approval level (suggest/auto-edit/full-auto).
    console:
        Optional Rich Console for interactive prompts.
    """

    def __init__(
        self,
        mode: ApprovalMode = ApprovalMode.AUTO_EDIT,
        console: Any = None,
    ) -> None:
        self.mode = mode
        self._console = console
        # Track approvals in this session (tool_name → last decision)
        self._session_approvals: dict[str, ApprovalDecision] = {}
        # Tools the user has "always approved" for this session
        self._always_approved: set[str] = set()

    def needs_approval(self, tool_name: str, args: dict[str, Any]) -> ApprovalRequest | None:
        """Check if a tool call needs approval. Returns None if auto-approved."""
        if self.mode == ApprovalMode.FULL_AUTO:
            return None

        if tool_name in self._always_approved:
            return None

        if tool_name in _READ_TOOLS:
            return None  # Reads never need approval

        if self.mode == ApprovalMode.AUTO_EDIT:
            if tool_name in _WRITE_TOOLS:
                return None  # Writes auto-approved in auto-edit mode

        # Determine category and reason
        if tool_name in _WRITE_TOOLS:
            category = "write"
            path = args.get("path", "unknown")
            reason = f"Write to file: {path}"
        elif tool_name in _EXECUTE_TOOLS:
            category = "execute"
            command = args.get("command", "")
            reason = f"Execute shell command: {command[:100]}"
        elif tool_name in _GIT_MUTATE_TOOLS:
            category = "git"
            if tool_name == "git_commit":
                msg = args.get("message", "")
                reason = f"Git commit: {msg[:80]}"
            else:
                reason = f"Git operation: {tool_name}"
        elif tool_name in _EXTERNAL_TOOLS:
            category = "external"
            task = args.get("task", "")
            reason = f"Delegate to external agent: {task[:80]}"
        else:
            category = "unknown"
            reason = f"Unknown tool: {tool_name}"

        return ApprovalRequest(
            tool_name=tool_name,
            arguments=args,
            reason=reason,
            category=category,
        )

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask the user for approval interactively.

        Parameters
        ----------
        request:
            The approval request details.

        Returns
        -------
        ApprovalDecision
            Whether the action was approved or denied.
        """
        if not self._console:
            _logger.warning(
                "Approval required but no console available; denying: %s",
                request.reason,
            )
            return ApprovalDecision.DENIED

        self._console.print(
            f"\n[yellow bold]Approval required:[/yellow bold] {request.reason}"
        )
        self._console.print(
            "  [dim][y] approve  [n] deny  [a] always approve this tool  [q] deny all[/dim]"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: input("  Approve? [y/n/a/q]: ").strip().lower(),
            )
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENIED

        if response in ("y", "yes"):
            self._session_approvals[request.tool_name] = ApprovalDecision.APPROVED
            return ApprovalDecision.APPROVED
        elif response in ("a", "always"):
            self._always_approved.add(request.tool_name)
            self._session_approvals[request.tool_name] = ApprovalDecision.APPROVED
            self._console.print(
                f"  [green]Auto-approving '{request.tool_name}' for this session[/green]"
            )
            return ApprovalDecision.APPROVED
        elif response in ("q", "quit"):
            # Deny all future requests by switching to suggest mode
            self._console.print("  [red]Denying all remaining operations[/red]")
            return ApprovalDecision.DENIED
        else:
            return ApprovalDecision.DENIED

    @staticmethod
    def from_string(mode_str: str) -> ApprovalMode:
        """Parse an approval mode from a string."""
        mode_map = {
            "suggest": ApprovalMode.SUGGEST,
            "auto-edit": ApprovalMode.AUTO_EDIT,
            "auto_edit": ApprovalMode.AUTO_EDIT,
            "full-auto": ApprovalMode.FULL_AUTO,
            "full_auto": ApprovalMode.FULL_AUTO,
            "full": ApprovalMode.FULL_AUTO,
        }
        return mode_map.get(mode_str.lower(), ApprovalMode.AUTO_EDIT)
