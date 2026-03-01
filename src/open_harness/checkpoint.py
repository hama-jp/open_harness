"""Checkpoint and rollback engine for autonomous goal execution.

Provides transactional safety for code changes:
1. Pre-goal checkpoint (stash uncommitted work)
2. Mid-goal snapshots (lightweight git commits on temp branch)
3. Auto-rollback on test failure
4. Post-goal restore (pop stash)

The agent can call rollback() to undo changes since the last snapshot,
then retry with a different approach.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _git(args: list[str], cwd: str, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        logger.warning("git %s timed out after %ds", " ".join(args), timeout)
        return subprocess.CompletedProcess(
            args=["git"] + args, returncode=1,
            stdout="", stderr=f"timeout after {timeout}s",
        )


@dataclass
class Snapshot:
    """A lightweight checkpoint within a goal."""
    commit_hash: str
    description: str
    timestamp: float = field(default_factory=time.time)


class CheckpointEngine:
    """Manages git-based checkpoints for autonomous execution."""

    def __init__(self, project_root: str | Path, has_git: bool = True):
        self.cwd = str(Path(project_root).resolve())
        self.has_git = has_git
        self._original_branch: str | None = None
        self._work_branch: str | None = None
        self._stashed: bool = False
        self._snapshots: list[Snapshot] = []
        self._active: bool = False
        self._auto_initialized: bool = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    def begin(self) -> str:
        """Start a checkpoint session. Call before autonomous work.

        If the project is not under git, initializes a repository first
        so that all changes can be tracked and reverted.

        Returns a status message.
        """
        if not self.has_git:
            init_msg = self._auto_git_init()
            if not self.has_git:
                # git init failed — proceed without checkpoint protection
                return f"git init failed: {init_msg}"

        if self._active:
            return "already active"

        self._active = True

        # Remember current branch
        r = _git(["rev-parse", "--abbrev-ref", "HEAD"], self.cwd)
        self._original_branch = r.stdout.strip() if r.returncode == 0 else "main"

        # Stash any uncommitted changes
        status = _git(["status", "--porcelain"], self.cwd)
        if status.stdout.strip():
            stash = _git(["stash", "push", "-m", "open-harness: pre-goal checkpoint"], self.cwd)
            if stash.returncode == 0 and "No local changes" not in stash.stdout:
                self._stashed = True

        # Create work branch for this goal
        ts = int(time.time())
        self._work_branch = f"harness/goal-{ts}"
        branch = _git(["checkout", "-b", self._work_branch], self.cwd)
        if branch.returncode != 0:
            # Branch might exist, try with suffix
            self._work_branch = f"harness/goal-{ts}-retry"
            retry = _git(["checkout", "-b", self._work_branch], self.cwd)
            if retry.returncode != 0:
                logger.warning("Failed to create work branch: %s", retry.stderr.strip())
                self._active = False
                self._work_branch = None
                return f"branch creation failed: {retry.stderr.strip()[:80]}"

        parts = []
        if self._stashed:
            parts.append("stashed uncommitted changes")
        parts.append(f"branch: {self._work_branch}")
        return ", ".join(parts)

    def snapshot(self, description: str = "auto-snapshot") -> Snapshot | None:
        """Create a lightweight snapshot of current state.

        Returns the snapshot, or None if nothing to snapshot.
        """
        if not self._active or not self.has_git:
            return None

        # Check if there are changes to snapshot
        status = _git(["status", "--porcelain"], self.cwd)
        if not status.stdout.strip():
            return None

        # Stage and commit
        _git(["add", "-A"], self.cwd)
        msg = f"harness-snapshot: {description}"
        commit = _git(["commit", "-m", msg, "--allow-empty"], self.cwd)
        if commit.returncode != 0:
            return None

        # Get commit hash
        rev = _git(["rev-parse", "--short", "HEAD"], self.cwd)
        commit_hash = rev.stdout.strip()

        snap = Snapshot(commit_hash=commit_hash, description=description)
        self._snapshots.append(snap)
        return snap

    def rollback(self, to_snapshot: Snapshot | None = None) -> str:
        """Rollback to the given snapshot (or to pre-goal state).

        Returns a status message.
        """
        if not self._active or not self.has_git:
            return "no active checkpoint"

        if to_snapshot:
            # Reset to specific snapshot
            r = _git(["reset", "--hard", to_snapshot.commit_hash], self.cwd)
            if r.returncode == 0:
                # Remove snapshots after this one
                idx = next(
                    (i for i, s in enumerate(self._snapshots)
                     if s.commit_hash == to_snapshot.commit_hash),
                    -1,
                )
                if idx >= 0:
                    self._snapshots = self._snapshots[:idx + 1]
                return f"rolled back to {to_snapshot.commit_hash}: {to_snapshot.description}"
        else:
            # Reset all changes on work branch
            if self._snapshots:
                first = self._snapshots[0]
                r = _git(["reset", "--hard", f"{first.commit_hash}~1"], self.cwd)
            else:
                r = _git(["reset", "--hard", "HEAD"], self.cwd)
            if r.returncode == 0:
                self._snapshots.clear()
                return "rolled back all goal changes"

        return "rollback failed"

    def finish(self, keep_changes: bool = True) -> str:
        """End the checkpoint session.

        If keep_changes=True, merges work into original branch.
        If keep_changes=False, discards work branch.

        Returns a status message.
        """
        if not self._active or not self.has_git:
            self._active = False
            return "no active checkpoint"

        self._active = False
        parts = []

        if keep_changes and self._snapshots:
            # Commit any uncommitted changes on work branch before switching
            status = _git(["status", "--porcelain"], self.cwd)
            if status.stdout.strip():
                _git(["add", "-A"], self.cwd)
                _git(["commit", "-m", "harness-snapshot: uncommitted changes at finish"], self.cwd)

            # Switch back to original branch
            checkout = _git(["checkout", self._original_branch], self.cwd)
            if checkout.returncode != 0:
                # Force checkout as last resort (changes are saved in work branch commits)
                checkout = _git(["checkout", "-f", self._original_branch], self.cwd)
                if checkout.returncode != 0:
                    parts.append(f"checkout failed: {checkout.stderr.strip()[:100]}")
                    # Still on work branch — skip merge, just clean up
                    self._cleanup_stash(parts)
                    self._snapshots.clear()
                    self._work_branch = None
                    return ", ".join(parts) if parts else "finish failed"

            # Squash-merge work into original branch
            merge = _git(["merge", "--squash", self._work_branch], self.cwd)
            if merge.returncode == 0:
                # Check if there's anything to commit
                status = _git(["status", "--porcelain"], self.cwd)
                if status.stdout.strip():
                    parts.append(f"merged {len(self._snapshots)} snapshots")
                else:
                    parts.append("no net changes to merge")
            else:
                # Merge conflict: abort to restore clean state
                _git(["merge", "--abort"], self.cwd)
                parts.append(f"merge conflict (aborted): {merge.stderr.strip()[:100]}")
            # Clean up work branch
            _git(["branch", "-D", self._work_branch], self.cwd)
        elif self._work_branch:
            # Discard work branch — force checkout to discard uncommitted changes
            _git(["checkout", "-f", self._original_branch], self.cwd)
            _git(["branch", "-D", self._work_branch], self.cwd)
            parts.append("discarded goal changes")

        self._cleanup_stash(parts)
        self._snapshots.clear()
        self._work_branch = None
        return ", ".join(parts) if parts else "clean finish"

    def _auto_git_init(self) -> str:
        """Initialize a git repository if none exists.

        Creates .gitignore, runs git init, and makes an initial commit
        so that all subsequent changes can be tracked and reverted.

        Returns a status message. Sets self.has_git = True on success.
        """
        cwd = self.cwd

        # Double-check: maybe .git exists but has_git flag was wrong
        if (Path(cwd) / ".git").is_dir():
            self.has_git = True
            return "git already exists"

        # Create .gitignore if missing
        gitignore = Path(cwd) / ".gitignore"
        if not gitignore.exists():
            try:
                gitignore.write_text(
                    "# Auto-generated by Open Harness\n"
                    "__pycache__/\n*.pyc\n.venv/\nvenv/\n"
                    "node_modules/\n.env\n*.egg-info/\ndist/\nbuild/\n"
                )
            except OSError as e:
                logger.warning("Failed to create .gitignore: %s", e)

        r = _git(["init"], cwd)
        if r.returncode != 0:
            logger.warning("git init failed: %s", r.stderr.strip())
            return r.stderr.strip()

        # Stage all existing files and create initial commit
        _git(["add", "-A"], cwd)
        r = _git(["commit", "-m", "Initial commit (auto-created by Open Harness)"], cwd)
        if r.returncode != 0:
            # Without a baseline commit, checkpoints cannot work safely.
            # Remove .git to avoid a broken state.
            import shutil
            git_dir = Path(cwd) / ".git"
            if git_dir.is_dir():
                shutil.rmtree(git_dir, ignore_errors=True)
            logger.warning("Initial commit failed: %s", r.stderr.strip())
            return f"initial commit failed: {r.stderr.strip()}"

        self.has_git = True
        self._auto_initialized = True
        logger.info("Auto-initialized git repository at %s", cwd)
        return "auto-initialized git"

    def _cleanup_stash(self, parts: list[str]):
        """Restore stashed changes if any."""
        if self._stashed:
            pop = _git(["stash", "pop"], self.cwd)
            if pop.returncode == 0:
                parts.append("restored stashed changes")
            else:
                parts.append(f"stash pop failed: {pop.stderr.strip()[:80]}")
            self._stashed = False

    def get_diff_since_start(self) -> str:
        """Get a summary of all changes since goal started."""
        if not self._active or not self.has_git:
            return ""
        r = _git(["diff", "--stat", f"HEAD~{len(self._snapshots)}", "HEAD"], self.cwd)
        return r.stdout.strip() if r.returncode == 0 else ""
