"""Git-based transactional checkpoint engine.

Subscribes to the EventBus and autonomously creates snapshots after
tool executions.  The Orchestrator is unaware of checkpoints.

Policy modes control snapshot frequency:
  - safe:     every 5 writes + after every git/shell
  - balanced: every 10 writes + after git only
  - full:     no snapshots
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from open_harness_v2.config import PolicySpec
from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType

_logger = logging.getLogger(__name__)

_TIMEOUT = 15  # seconds for git commands

# Tool-to-category mapping
_TOOL_CATEGORIES: dict[str, str] = {
    "write_file": "write",
    "edit_file": "write",
    "shell": "execute",
    "git_commit": "git",
    "git_branch": "git",
}


@dataclass
class Snapshot:
    commit_hash: str
    description: str
    timestamp: float = field(default_factory=time.time)


class CheckpointEngine:
    """EventBus-driven git checkpoint manager.

    Parameters
    ----------
    project_root:
        Root of the git repository.
    policy:
        Policy spec — determines snapshot frequency.
    """

    def __init__(self, project_root: Path, policy: PolicySpec) -> None:
        self._root = project_root
        self._policy = policy
        self._snapshots: list[Snapshot] = []
        self._writes_since_snapshot = 0
        self._original_branch: str | None = None
        self._work_branch: str | None = None
        self._stashed = False
        self._active = False

    # ------------------------------------------------------------------
    # EventBus wiring
    # ------------------------------------------------------------------

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to agent lifecycle and tool events."""
        event_bus.subscribe(EventType.AGENT_STARTED, self._on_agent_started)
        event_bus.subscribe(EventType.TOOL_EXECUTED, self._on_tool_executed)
        event_bus.subscribe(EventType.AGENT_DONE, self._on_agent_done)
        event_bus.subscribe(EventType.AGENT_ERROR, self._on_agent_error)
        event_bus.subscribe(EventType.AGENT_CANCELLED, self._on_agent_cancelled)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_agent_started(self, event: AgentEvent) -> None:
        if self._policy.mode == "full":
            return  # no checkpoints in full mode
        self._begin()

    def _on_tool_executed(self, event: AgentEvent) -> None:
        if not self._active:
            return
        tool = event.data.get("tool", "")
        success = event.data.get("success", True)
        if not success:
            return

        category = _TOOL_CATEGORIES.get(tool, "other")
        interval = self._snapshot_interval()

        if category == "write":
            self._writes_since_snapshot += 1
            if self._writes_since_snapshot >= interval:
                self._snapshot(f"after {self._writes_since_snapshot} writes")
                self._writes_since_snapshot = 0
        elif category == "git":
            self._snapshot(f"after {tool}")
        elif category == "execute" and self._policy.mode == "safe":
            self._snapshot(f"after shell: {tool}")

    def _on_agent_done(self, event: AgentEvent) -> None:
        if self._active:
            self._finish(keep_changes=True)

    def _on_agent_error(self, event: AgentEvent) -> None:
        if self._active:
            self._finish(keep_changes=False)

    def _on_agent_cancelled(self, event: AgentEvent) -> None:
        if self._active:
            self._finish(keep_changes=False)

    # ------------------------------------------------------------------
    # Snapshot interval
    # ------------------------------------------------------------------

    def _snapshot_interval(self) -> int:
        return {"safe": 5, "balanced": 10, "full": 999999}.get(
            self._policy.mode, 10,
        )

    @staticmethod
    def _categorize(tool_name: str) -> str:
        return _TOOL_CATEGORIES.get(tool_name, "other")

    # ------------------------------------------------------------------
    # Git operations
    # ------------------------------------------------------------------

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a git command in the project root."""
        return subprocess.run(
            ["git", *args],
            cwd=self._root,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=check,
        )

    def _is_git_repo(self) -> bool:
        try:
            r = self._git("rev-parse", "--is-inside-work-tree", check=False)
            return r.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _begin(self) -> None:
        """Create work branch and stash uncommitted changes."""
        if not self._is_git_repo():
            _logger.debug("Not a git repo, skipping checkpoint")
            return

        try:
            # Save current branch
            r = self._git("rev-parse", "--abbrev-ref", "HEAD")
            self._original_branch = r.stdout.strip()

            # Stash uncommitted work
            status = self._git("status", "--porcelain", check=False)
            if status.stdout.strip():
                self._git(
                    "stash", "push", "--include-untracked",
                    "-m", "open-harness: pre-goal checkpoint",
                )
                self._stashed = True

            # Create work branch
            branch = f"harness/goal-{int(time.time())}"
            self._git("checkout", "-b", branch)
            self._work_branch = branch
            self._active = True
            self._snapshots = []
            self._writes_since_snapshot = 0
            _logger.debug("Checkpoint started: %s", branch)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            _logger.warning("Failed to start checkpoint: %s", e)
            self._active = False

    def _snapshot(self, description: str) -> Snapshot | None:
        """Commit current state as a checkpoint."""
        if not self._active:
            return None

        try:
            # Check for changes
            diff = self._git("status", "--porcelain", "-uno", check=False)
            untracked = self._git(
                "ls-files", "--others", "--exclude-standard", check=False,
            )
            if not diff.stdout.strip() and not untracked.stdout.strip():
                return None  # nothing to snapshot

            # Stage and commit
            self._git("add", "-A")
            commit_result = self._git(
                "commit", "-a", "-m",
                f"harness-snapshot: {description}",
                check=False,
            )

            if commit_result.returncode != 0:
                _logger.warning(
                    "Snapshot commit failed (rc=%d): %s",
                    commit_result.returncode,
                    commit_result.stderr.strip(),
                )
                return None

            # Get commit hash
            r = self._git("rev-parse", "--short", "HEAD")
            commit_hash = r.stdout.strip()

            snap = Snapshot(commit_hash=commit_hash, description=description)
            self._snapshots.append(snap)
            _logger.debug("Snapshot: %s — %s", commit_hash, description)
            return snap

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            _logger.warning("Snapshot failed: %s", e)
            return None

    def _rollback(self, to: Snapshot | None = None) -> None:
        """Roll back to a specific snapshot or pre-goal state."""
        if not self._active:
            return

        try:
            if to:
                self._git("reset", "--hard", to.commit_hash)
                # Prune snapshots after the target
                idx = next(
                    (i for i, s in enumerate(self._snapshots) if s.commit_hash == to.commit_hash),
                    -1,
                )
                if idx >= 0:
                    self._snapshots = self._snapshots[:idx + 1]
            elif self._snapshots:
                self._git("reset", "--hard", f"{self._snapshots[0].commit_hash}~1")
                self._snapshots.clear()
            else:
                self._git("reset", "--hard", "HEAD")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            _logger.warning("Rollback failed: %s", e)

    def _finish(self, keep_changes: bool) -> None:
        """Merge work into original branch or discard, restore stash."""
        if not self._active or not self._work_branch or not self._original_branch:
            self._active = False
            return

        try:
            if keep_changes:
                # Commit any remaining uncommitted changes
                status = self._git("status", "--porcelain", check=False)
                if status.stdout.strip():
                    self._git("add", "-A")
                    self._git(
                        "commit", "-m",
                        "harness-snapshot: uncommitted changes at finish",
                        check=False,
                    )

                # Switch back and squash-merge
                work_branch = self._work_branch
                self._git("checkout", self._original_branch)
                diff = self._git(
                    "diff", self._original_branch, work_branch, "--stat", check=False,
                )
                if diff.stdout.strip():
                    merge_result = self._git(
                        "merge", "--squash", self._work_branch, check=False,
                    )
                    if merge_result.returncode == 0:
                        n = len(self._snapshots)
                        self._git(
                            "commit", "-m",
                            f"harness: goal completed ({n} snapshots merged)",
                            check=False,
                        )
                    else:
                        _logger.warning("Merge conflict, aborting merge")
                        self._git("merge", "--abort", check=False)
            else:
                # Discard changes
                self._git("checkout", "-f", self._original_branch)

            # Delete work branch
            self._git("branch", "-D", self._work_branch, check=False)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            _logger.warning("Checkpoint finish failed: %s", e)
            # Best-effort return to original branch
            try:
                self._git("checkout", "-f", self._original_branch, check=False)
                self._git("branch", "-D", self._work_branch, check=False)
            except Exception:
                pass

        finally:
            # Restore stash
            if self._stashed:
                try:
                    self._git("stash", "pop", check=False)
                except Exception:
                    _logger.warning("Failed to pop stash")
                self._stashed = False

            self._active = False
            self._work_branch = None
            self._snapshots = []
            self._writes_since_snapshot = 0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    @property
    def active(self) -> bool:
        return self._active

    @staticmethod
    def cleanup_orphan_branches(root: Path) -> list[str]:
        """Delete stale harness/goal-* branches from interrupted sessions."""
        cleaned: list[str] = []
        try:
            r = subprocess.run(
                ["git", "branch", "--list", "harness/goal-*"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
            current = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            ).stdout.strip()

            for line in r.stdout.strip().splitlines():
                branch = line.strip().lstrip("* ")
                if branch and branch != current:
                    subprocess.run(
                        ["git", "branch", "-D", branch],
                        cwd=root,
                        capture_output=True,
                        timeout=_TIMEOUT,
                    )
                    cleaned.append(branch)

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return cleaned
