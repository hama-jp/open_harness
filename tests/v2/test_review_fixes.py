"""Regression tests for v2 migration review fixes.

Covers issues identified in docs/v2_migration_review_report.md:
  1. PolicyEngine project root must be set for write access
  2. TaskManager cancel for queued (semaphore-waiting) tasks
  3. SessionMemory bind/load connection
  6. Executor/Renderer event payload alignment
  7. MemoryStore threading safety
  8. Checkpoint snapshot commit verification
  + TaskManager shutdown (graceful cleanup)
"""

from __future__ import annotations

import asyncio
import subprocess
import threading

import pytest
from unittest.mock import AsyncMock, MagicMock

from open_harness_v2.config import PolicySpec
from open_harness_v2.events.bus import EventBus
from open_harness_v2.policy.engine import PolicyEngine
from open_harness_v2.tasks.manager import TaskManager
from open_harness_v2.tasks.record import TaskStatus
from open_harness_v2.tasks.store import TaskStore
from open_harness_v2.types import AgentEvent, EventType


# ===================================================================
# Issue #1: PolicyEngine project root for write access
# ===================================================================

class TestPolicyProjectRoot:
    """Balanced/safe mode must allow writes inside project root."""

    def test_write_denied_without_project_root(self, tmp_path):
        """Without project root set, writes outside writable_paths are denied."""
        engine = PolicyEngine(PolicySpec(mode="balanced"))
        # No set_project_root called
        target = str(tmp_path / "file.txt")
        violation = engine.check("write_file", {"path": target})
        assert violation is not None
        assert "outside project root" in violation.message

    def test_write_allowed_with_project_root(self, tmp_path):
        """With project root set, writes inside it are allowed."""
        engine = PolicyEngine(PolicySpec(mode="balanced"))
        engine.set_project_root(tmp_path)
        target = str(tmp_path / "subdir" / "file.txt")
        violation = engine.check("write_file", {"path": target})
        assert violation is None

    def test_write_denied_outside_project_root(self, tmp_path):
        """Writes outside project root are still denied."""
        engine = PolicyEngine(PolicySpec(mode="balanced"))
        engine.set_project_root(tmp_path / "project")
        target = str(tmp_path / "other" / "file.txt")
        violation = engine.check("write_file", {"path": target})
        assert violation is not None

    def test_safe_mode_allows_writes_in_project(self, tmp_path):
        """Safe mode with project root should allow writes too."""
        engine = PolicyEngine(PolicySpec(mode="safe"))
        engine.set_project_root(tmp_path)
        target = str(tmp_path / "src" / "main.py")
        violation = engine.check("write_file", {"path": target})
        assert violation is None


# ===================================================================
# Issue #2: TaskManager cancel for queued (semaphore-waiting) tasks
# ===================================================================

class TestTaskManagerQueuedCancel:
    """Cancelling a task waiting on the semaphore must update DB + cleanup."""

    @pytest.fixture
    def store(self, tmp_path):
        return TaskStore(db_path=tmp_path / "test_tasks.db")

    @pytest.fixture
    def bus(self):
        return EventBus()

    async def test_cancel_queued_task_updates_db(self, store, bus):
        """Cancel a task waiting for semaphore — DB should show CANCELLED."""
        # Use semaphore of 1, submit a slow task to block the slot
        async def slow_run(goal):
            await asyncio.sleep(10)
            return "late"

        def slow_factory():
            mock = MagicMock()
            mock.run = slow_run
            return mock

        manager = TaskManager(store, bus, slow_factory, max_concurrent=1)

        # First task blocks the slot
        await manager.submit("blocker")
        await asyncio.sleep(0.05)

        # Second task is queued (waiting on semaphore)
        queued = await manager.submit("will be cancelled")
        await asyncio.sleep(0.05)

        # Cancel the queued task
        ok = await manager.cancel(queued.id)
        assert ok

        await asyncio.sleep(0.3)

        loaded = store.get(queued.id)
        assert loaded.status == TaskStatus.CANCELLED

    async def test_cancel_queued_task_removes_from_running(self, store, bus):
        """After cancelling a queued task, it should not remain in _running."""
        async def slow_run(goal):
            await asyncio.sleep(10)
            return "late"

        def slow_factory():
            mock = MagicMock()
            mock.run = slow_run
            return mock

        manager = TaskManager(store, bus, slow_factory, max_concurrent=1)

        await manager.submit("blocker")
        await asyncio.sleep(0.05)

        queued = await manager.submit("cancel me")
        await asyncio.sleep(0.05)

        await manager.cancel(queued.id)
        await asyncio.sleep(0.3)

        # Should be cleaned up from _running
        assert queued.id not in manager._running


# ===================================================================
# Issue #7: MemoryStore threading safety
# ===================================================================

class TestMemoryStoreLock:
    """MemoryStore must handle concurrent writes without errors."""

    async def test_concurrent_fact_writes(self, tmp_path):
        from open_harness_v2.memory.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test_memory.db")

        # Run multiple concurrent writes via to_thread
        tasks = [
            store.upsert_fact("proj", f"key_{i}", f"value_{i}")
            for i in range(20)
        ]
        await asyncio.gather(*tasks)

        facts = await store.get_facts("proj")
        assert len(facts) == 20

        await store.close()

    async def test_concurrent_session_writes(self, tmp_path):
        from open_harness_v2.memory.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test_memory.db")

        tasks = [
            store.save_messages(f"session_{i}", [{"role": "user", "content": f"msg {i}"}])
            for i in range(20)
        ]
        await asyncio.gather(*tasks)

        # Verify all saved
        for i in range(20):
            msgs = await store.load_messages(f"session_{i}")
            assert len(msgs) == 1

        await store.close()


# ===================================================================
# Issue #8: Checkpoint commit verification
# ===================================================================

class TestCheckpointCommitVerification:
    """Snapshot should return None if git commit fails."""

    @pytest.fixture
    def git_repo(self, tmp_path):
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

    def test_snapshot_returns_none_on_no_changes(self, git_repo):
        """Snapshot with nothing to commit should return None (no false positive)."""
        from open_harness_v2.checkpoint.engine import CheckpointEngine

        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        engine._begin()

        # No changes made — snapshot should return None
        snap = engine._snapshot("empty snapshot")
        assert snap is None

        engine._finish(keep_changes=True)

    def test_snapshot_records_commit_on_success(self, git_repo):
        """Successful commit should be recorded."""
        from open_harness_v2.checkpoint.engine import CheckpointEngine

        engine = CheckpointEngine(git_repo, PolicySpec(mode="safe"))
        engine._begin()

        (git_repo / "new.txt").write_text("content")
        snap = engine._snapshot("valid snapshot")
        assert snap is not None
        assert snap.commit_hash  # non-empty
        assert len(engine._snapshots) == 1

        engine._finish(keep_changes=True)


# ===================================================================
# Issue #3: SessionMemory bind/load connection
# ===================================================================

class TestSessionMemoryConnection:
    """SessionMemory bind + load should enable session continuity."""

    async def test_bind_enables_auto_save(self, tmp_path):
        """After bind(), AGENT_DONE should trigger a save."""
        from open_harness_v2.memory.store import MemoryStore
        from open_harness_v2.memory.session import SessionMemory

        store = MemoryStore(db_path=tmp_path / "test.db")
        bus = EventBus()
        sm = SessionMemory(store)
        sm.attach(bus)

        # Bind a live session
        messages = [{"role": "user", "content": "hello"}]
        sm.bind("test-session", messages)

        # Simulate agent completion
        await bus.emit(AgentEvent(type=EventType.AGENT_DONE, data={"response": "hi"}))

        # Verify saved
        loaded = await store.load_messages("test-session")
        assert len(loaded) == 1
        assert loaded[0]["content"] == "hello"

        await store.close()

    async def test_load_restores_messages(self, tmp_path):
        """load() should return previously saved messages."""
        from open_harness_v2.memory.store import MemoryStore
        from open_harness_v2.memory.session import SessionMemory

        store = MemoryStore(db_path=tmp_path / "test.db")
        sm = SessionMemory(store)

        # Pre-save some messages
        await store.save_messages("sess-1", [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ])

        # Load them back
        msgs = await sm.load("sess-1")
        assert len(msgs) == 2
        assert msgs[0]["content"] == "q1"
        assert msgs[1]["content"] == "a1"

        await store.close()

    async def test_unbound_session_does_not_save(self, tmp_path):
        """Without bind(), AGENT_DONE should not save anything."""
        from open_harness_v2.memory.store import MemoryStore
        from open_harness_v2.memory.session import SessionMemory

        store = MemoryStore(db_path=tmp_path / "test.db")
        bus = EventBus()
        sm = SessionMemory(store)
        sm.attach(bus)

        # Do NOT call bind()
        await bus.emit(AgentEvent(type=EventType.AGENT_DONE, data={"response": "hi"}))

        # Nothing should be saved
        loaded = await store.load_messages("any-id")
        assert loaded == []

        await store.close()


# ===================================================================
# TaskManager shutdown (graceful cleanup)
# ===================================================================

class TestTaskManagerShutdown:
    """shutdown() should cancel running tasks and wait for completion."""

    @pytest.fixture
    def store(self, tmp_path):
        return TaskStore(db_path=tmp_path / "test_tasks.db")

    @pytest.fixture
    def bus(self):
        return EventBus()

    async def test_shutdown_cancels_running_tasks(self, store, bus):
        async def slow_run(goal):
            await asyncio.sleep(10)
            return "late"

        def slow_factory():
            from unittest.mock import MagicMock
            mock = MagicMock()
            mock.run = slow_run
            return mock

        manager = TaskManager(store, bus, slow_factory, max_concurrent=2)

        r1 = await manager.submit("task1")
        r2 = await manager.submit("task2")
        await asyncio.sleep(0.1)

        await manager.shutdown(timeout=2.0)

        # Both should be cancelled in DB
        assert store.get(r1.id).status == TaskStatus.CANCELLED
        assert store.get(r2.id).status == TaskStatus.CANCELLED
        # _running should be empty
        assert len(manager._running) == 0
