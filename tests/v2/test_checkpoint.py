"""Tests for the v2 checkpoint engine."""

from __future__ import annotations

import subprocess
import time

import pytest

from open_harness_v2.checkpoint.engine import (
    CheckpointEngine,
    Snapshot,
    _TOOL_CATEGORIES,
)
from open_harness_v2.config import PolicySpec
from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType


# ===================================================================
# Helpers
# ===================================================================

@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with an initial commit."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, capture_output=True,
    )
    (tmp_path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path, capture_output=True, check=True,
    )
    return tmp_path


def _branch_name(repo) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo, capture_output=True, text=True,
    )
    return r.stdout.strip()


def _commit_count(repo) -> int:
    r = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=repo, capture_output=True, text=True,
    )
    return int(r.stdout.strip())


def _list_branches(repo) -> list[str]:
    r = subprocess.run(
        ["git", "branch", "--list"],
        cwd=repo, capture_output=True, text=True,
    )
    return [b.strip().lstrip("* ") for b in r.stdout.strip().splitlines() if b.strip()]


# ===================================================================
# Snapshot interval logic
# ===================================================================

class TestSnapshotInterval:
    def test_safe_interval(self):
        engine = CheckpointEngine(tmp_path := None, PolicySpec(mode="safe"))
        engine._root = None  # not used for interval check
        assert engine._snapshot_interval() == 5

    def test_balanced_interval(self):
        engine = CheckpointEngine(None, PolicySpec(mode="balanced"))
        assert engine._snapshot_interval() == 10

    def test_full_interval(self):
        engine = CheckpointEngine(None, PolicySpec(mode="full"))
        assert engine._snapshot_interval() == 999999


# ===================================================================
# Tool categorization
# ===================================================================

class TestToolCategories:
    def test_write_tools(self):
        assert _TOOL_CATEGORIES.get("write_file") == "write"
        assert _TOOL_CATEGORIES.get("edit_file") == "write"

    def test_git_tools(self):
        assert _TOOL_CATEGORIES.get("git_commit") == "git"

    def test_execute_tools(self):
        assert _TOOL_CATEGORIES.get("shell") == "execute"

    def test_unknown_tool(self):
        assert _TOOL_CATEGORIES.get("read_file") is None


# ===================================================================
# Git operations
# ===================================================================

class TestCheckpointGitOps:
    def test_is_git_repo(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        assert engine._is_git_repo()

    def test_not_git_repo(self, tmp_path):
        engine = CheckpointEngine(tmp_path, PolicySpec(mode="safe"))
        assert not engine._is_git_repo()

    def test_begin_creates_work_branch(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        original = _branch_name(git_repo)
        engine._begin()

        assert engine._active
        assert engine._original_branch == original
        assert engine._work_branch is not None
        assert "harness/goal-" in _branch_name(git_repo)

    def test_begin_stashes_changes(self, git_repo):
        (git_repo / "dirty.txt").write_text("dirty")
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        engine._begin()

        assert engine._stashed
        # dirty file should be stashed away
        assert not (git_repo / "dirty.txt").exists()

    def test_begin_skips_non_git(self, tmp_path):
        engine = CheckpointEngine(tmp_path, PolicySpec(mode="safe"))
        engine._begin()
        assert not engine._active

    def test_snapshot_creates_commit(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        engine._begin()
        initial = _commit_count(git_repo)

        (git_repo / "new_file.txt").write_text("content")
        snap = engine._snapshot("test snapshot")

        assert snap is not None
        assert _commit_count(git_repo) == initial + 1
        assert len(engine._snapshots) == 1

    def test_snapshot_no_changes(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        engine._begin()

        snap = engine._snapshot("nothing to commit")
        assert snap is None

    def test_finish_keeps_changes(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        original = _branch_name(git_repo)
        engine._begin()

        (git_repo / "feature.txt").write_text("feature")
        engine._snapshot("add feature")
        engine._finish(keep_changes=True)

        assert _branch_name(git_repo) == original
        assert (git_repo / "feature.txt").exists()
        assert not engine._active

    def test_finish_discards_changes(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        original = _branch_name(git_repo)
        engine._begin()

        (git_repo / "bad.txt").write_text("bad")
        engine._snapshot("bad change")
        engine._finish(keep_changes=False)

        assert _branch_name(git_repo) == original
        assert not (git_repo / "bad.txt").exists()
        assert not engine._active

    def test_finish_restores_stash(self, git_repo):
        (git_repo / "pre_goal.txt").write_text("pre-goal work")
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        engine._begin()

        assert not (git_repo / "pre_goal.txt").exists()  # stashed

        engine._finish(keep_changes=True)
        assert (git_repo / "pre_goal.txt").exists()  # restored

    def test_rollback(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        engine._begin()

        (git_repo / "file1.txt").write_text("1")
        snap1 = engine._snapshot("first")

        (git_repo / "file2.txt").write_text("2")
        engine._snapshot("second")

        engine._rollback(to=snap1)
        assert not (git_repo / "file2.txt").exists()
        assert (git_repo / "file1.txt").exists()
        assert len(engine._snapshots) == 1

    def test_cleanup_orphan_branches(self, git_repo):
        # Create some orphan branches
        subprocess.run(
            ["git", "branch", "harness/goal-111"],
            cwd=git_repo, capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "harness/goal-222"],
            cwd=git_repo, capture_output=True,
        )

        cleaned = CheckpointEngine.cleanup_orphan_branches(git_repo)
        assert len(cleaned) == 2
        assert "harness/goal-111" in cleaned

        # Verify branches are gone
        branches = _list_branches(git_repo)
        assert not any("harness/goal-" in b for b in branches)


# ===================================================================
# EventBus integration
# ===================================================================

class TestCheckpointEventBus:
    def test_full_mode_skips_checkpoint(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="full"))
        bus = EventBus()
        engine.attach(bus)

        import asyncio
        asyncio.run(
            bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"}))
        )
        assert not engine._active

    async def test_safe_mode_activates(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        bus = EventBus()
        engine.attach(bus)

        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"}))
        assert engine._active

        await bus.emit(AgentEvent(type=EventType.AGENT_DONE, data={"response": "ok"}))
        assert not engine._active

    async def test_write_triggers_snapshot_at_interval(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        bus = EventBus()
        engine.attach(bus)

        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"}))

        # Create files and emit write events
        for i in range(5):
            (git_repo / f"file_{i}.txt").write_text(f"content {i}")
            await bus.emit(AgentEvent(
                type=EventType.TOOL_EXECUTED,
                data={"tool": "write_file", "success": True},
            ))

        # After 5 writes in safe mode, should have 1 snapshot
        assert len(engine._snapshots) == 1

        await bus.emit(AgentEvent(type=EventType.AGENT_DONE, data={"response": "ok"}))

    async def test_git_tool_always_snapshots(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="balanced"))
        bus = EventBus()
        engine.attach(bus)

        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"}))

        (git_repo / "change.txt").write_text("x")
        await bus.emit(AgentEvent(
            type=EventType.TOOL_EXECUTED,
            data={"tool": "git_commit", "success": True},
        ))

        assert len(engine._snapshots) == 1

        await bus.emit(AgentEvent(type=EventType.AGENT_DONE, data={"response": "ok"}))

    async def test_error_discards_changes(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        bus = EventBus()
        engine.attach(bus)

        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"}))
        original = engine._original_branch

        (git_repo / "oops.txt").write_text("oops")
        engine._snapshot("before error")

        await bus.emit(AgentEvent(type=EventType.AGENT_ERROR, data={"error": "boom"}))

        assert not engine._active
        assert _branch_name(git_repo) == original
        assert not (git_repo / "oops.txt").exists()

    async def test_failed_tool_doesnt_snapshot(self, git_repo):
        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        bus = EventBus()
        engine.attach(bus)

        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"}))

        await bus.emit(AgentEvent(
            type=EventType.TOOL_EXECUTED,
            data={"tool": "write_file", "success": False},
        ))

        assert engine._writes_since_snapshot == 0  # not counted

        await bus.emit(AgentEvent(type=EventType.AGENT_DONE, data={"response": "ok"}))
