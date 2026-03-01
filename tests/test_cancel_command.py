"""Tests for Issue 3: Background task cancellation (/cancel command)."""

import os
import tempfile
import threading
import time

from open_harness.tasks.queue import TaskQueueManager, TaskRecord, TaskStatus, TaskStore


class TestCancelCommand:
    def setup_method(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = TaskStore(self.db_path)

    def teardown_method(self):
        self.store.close()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_cancel_queued_task(self):
        """Should be able to cancel a queued task directly via store."""
        task_id = self.store.create_task("test goal", "/tmp/test.log")
        task = self.store.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.QUEUED

        self.store.mark_canceled(task_id)
        task = self.store.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.CANCELED
        assert task.is_terminal

    def test_cancel_no_running_task(self):
        """cancel() with no running task should return informative message."""
        def mock_factory():
            raise RuntimeError("should not be called")

        mgr = TaskQueueManager(self.store, mock_factory)
        result = mgr.cancel()
        assert "No task" in result

    def test_cancel_nonexistent_task(self):
        """cancel() with unknown task_id should return not found."""
        def mock_factory():
            raise RuntimeError("should not be called")

        mgr = TaskQueueManager(self.store, mock_factory)
        result = mgr.cancel("nonexistent")
        assert "not found" in result

    def test_cancel_already_finished_task(self):
        """cancel() on a finished task should indicate it's already done."""
        task_id = self.store.create_task("done goal", "/tmp/test.log")
        self.store.mark_running(task_id)
        self.store.mark_succeeded(task_id, "result")

        def mock_factory():
            raise RuntimeError("should not be called")

        mgr = TaskQueueManager(self.store, mock_factory)
        result = mgr.cancel(task_id)
        assert "already finished" in result

    def test_cancel_queued_via_manager(self):
        """cancel() should cancel a queued task via manager."""
        def mock_factory():
            raise RuntimeError("should not be called")

        mgr = TaskQueueManager(self.store, mock_factory)
        task_id = self.store.create_task("queued goal", "/tmp/test.log")

        result = mgr.cancel(task_id)
        assert "canceled" in result.lower()
        task = self.store.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.CANCELED
