"""Tests for the v2 task queue subsystem."""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from open_harness_v2.tasks.record import TaskRecord, TaskStatus
from open_harness_v2.tasks.store import TaskStore
from open_harness_v2.tasks.manager import TaskManager
from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType


# ===================================================================
# TaskRecord
# ===================================================================

class TestTaskRecord:
    def test_defaults(self):
        r = TaskRecord(goal="test")
        assert r.status == TaskStatus.QUEUED
        assert len(r.id) == 8
        assert r.is_terminal is False

    def test_terminal_states(self):
        for status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            r = TaskRecord(goal="x", status=status)
            assert r.is_terminal is True

    def test_elapsed_not_started(self):
        r = TaskRecord(goal="x")
        assert r.elapsed is None

    def test_elapsed_running(self):
        import time
        r = TaskRecord(goal="x", started_at=time.time() - 5)
        assert r.elapsed is not None
        assert r.elapsed >= 4.5

    def test_elapsed_finished(self):
        r = TaskRecord(goal="x", started_at=100.0, finished_at=110.0)
        assert r.elapsed == pytest.approx(10.0)


# ===================================================================
# TaskStore
# ===================================================================

class TestTaskStore:
    @pytest.fixture
    def store(self, tmp_path):
        return TaskStore(db_path=tmp_path / "test_tasks.db")

    def test_create_and_get(self, store):
        record = store.create("Fix the bug")
        assert record.goal == "Fix the bug"
        assert record.status == TaskStatus.QUEUED

        loaded = store.get(record.id)
        assert loaded is not None
        assert loaded.goal == "Fix the bug"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_update_status_running(self, store):
        record = store.create("task")
        store.update_status(record.id, TaskStatus.RUNNING)
        loaded = store.get(record.id)
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.started_at is not None

    def test_update_status_cancelled(self, store):
        record = store.create("task")
        store.update_status(record.id, TaskStatus.CANCELLED)
        loaded = store.get(record.id)
        assert loaded.status == TaskStatus.CANCELLED
        assert loaded.finished_at is not None

    def test_update_result_succeeded(self, store):
        record = store.create("task")
        store.update_result(record.id, TaskStatus.SUCCEEDED, result="done!")
        loaded = store.get(record.id)
        assert loaded.status == TaskStatus.SUCCEEDED
        assert loaded.result == "done!"
        assert loaded.finished_at is not None

    def test_update_result_failed(self, store):
        record = store.create("task")
        store.update_result(record.id, TaskStatus.FAILED, error="oops")
        loaded = store.get(record.id)
        assert loaded.status == TaskStatus.FAILED
        assert loaded.error == "oops"

    def test_list_recent(self, store):
        store.create("task1")
        store.create("task2")
        store.create("task3")
        tasks = store.list_recent(2)
        assert len(tasks) == 2
        # Most recent first
        assert tasks[0].goal == "task3"
        assert tasks[1].goal == "task2"

    def test_recover_stale(self, store):
        r1 = store.create("task1")
        r2 = store.create("task2")
        store.update_status(r1.id, TaskStatus.RUNNING)
        store.update_status(r2.id, TaskStatus.RUNNING)

        count = store.recover_stale()
        assert count == 2

        assert store.get(r1.id).status == TaskStatus.QUEUED
        assert store.get(r2.id).status == TaskStatus.QUEUED

    async def test_close(self, store):
        store.create("task")
        await store.close()
        # Re-connecting after close should work
        loaded = store.get("nonexistent")
        assert loaded is None


# ===================================================================
# TaskManager
# ===================================================================

class TestTaskManager:
    @pytest.fixture
    def store(self, tmp_path):
        return TaskStore(db_path=tmp_path / "test_tasks.db")

    @pytest.fixture
    def bus(self):
        return EventBus()

    def _make_factory(self, result: str = "ok"):
        """Create a factory that returns mock orchestrators."""
        def factory():
            mock = MagicMock()
            mock.run = AsyncMock(return_value=result)
            return mock
        return factory

    async def test_submit_and_complete(self, store, bus):
        manager = TaskManager(store, bus, self._make_factory("done"))
        record = await manager.submit("Fix it")
        assert record.status == TaskStatus.QUEUED
        assert record.goal == "Fix it"

        # Wait for background task to complete
        await asyncio.sleep(0.2)

        loaded = store.get(record.id)
        assert loaded.status == TaskStatus.SUCCEEDED
        assert loaded.result == "done"

    async def test_submit_emits_events(self, store, bus):
        events: list[AgentEvent] = []
        bus.subscribe(EventType.TASK_STARTED, lambda e: events.append(e))
        bus.subscribe(EventType.TASK_COMPLETED, lambda e: events.append(e))

        manager = TaskManager(store, bus, self._make_factory("ok"))
        record = await manager.submit("test goal")

        await asyncio.sleep(0.2)

        started = [e for e in events if e.type == EventType.TASK_STARTED]
        completed = [e for e in events if e.type == EventType.TASK_COMPLETED]
        assert len(started) == 1
        assert started[0].data["task_id"] == record.id
        assert len(completed) == 1
        assert completed[0].data["status"] == "succeeded"

    async def test_submit_failure(self, store, bus):
        def failing_factory():
            mock = MagicMock()
            mock.run = AsyncMock(side_effect=RuntimeError("boom"))
            return mock

        manager = TaskManager(store, bus, failing_factory)
        record = await manager.submit("will fail")

        await asyncio.sleep(0.2)

        loaded = store.get(record.id)
        assert loaded.status == TaskStatus.FAILED
        assert "boom" in loaded.error

    async def test_cancel_running(self, store, bus):
        async def slow_run(goal):
            await asyncio.sleep(10)
            return "late"

        def slow_factory():
            mock = MagicMock()
            mock.run = slow_run
            return mock

        manager = TaskManager(store, bus, slow_factory)
        record = await manager.submit("slow task")

        await asyncio.sleep(0.1)  # let it start
        ok = await manager.cancel(record.id)
        assert ok

        await asyncio.sleep(0.2)
        loaded = store.get(record.id)
        assert loaded.status == TaskStatus.CANCELLED

    async def test_list_tasks(self, store, bus):
        manager = TaskManager(store, bus, self._make_factory())
        await manager.submit("task1")
        await manager.submit("task2")

        await asyncio.sleep(0.2)

        tasks = manager.list_tasks()
        assert len(tasks) == 2

    async def test_get_task(self, store, bus):
        manager = TaskManager(store, bus, self._make_factory())
        record = await manager.submit("find me")

        await asyncio.sleep(0.2)

        found = manager.get_task(record.id)
        assert found is not None
        assert found.goal == "find me"

    async def test_semaphore_limits_concurrency(self, store, bus):
        """With max_concurrent=1, only one task runs at a time."""
        running_count = 0
        max_seen = 0

        async def counting_run(goal):
            nonlocal running_count, max_seen
            running_count += 1
            max_seen = max(max_seen, running_count)
            await asyncio.sleep(0.05)
            running_count -= 1
            return "ok"

        def counting_factory():
            mock = MagicMock()
            mock.run = counting_run
            return mock

        manager = TaskManager(store, bus, counting_factory, max_concurrent=1)
        await manager.submit("t1")
        await manager.submit("t2")
        await manager.submit("t3")

        await asyncio.sleep(0.5)

        assert max_seen == 1  # never more than 1 concurrent

    async def test_recover(self, store, bus):
        # Simulate crash: tasks stuck in RUNNING
        r1 = store.create("stale1")
        store.update_status(r1.id, TaskStatus.RUNNING)

        manager = TaskManager(store, bus, self._make_factory())
        recovered = await manager.recover()
        assert recovered == 1
        assert store.get(r1.id).status == TaskStatus.QUEUED
