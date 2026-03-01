"""Async task queue for background goal execution.

Goals are submitted to a FIFO queue and executed sequentially by a
background worker thread. Each task gets an isolated Agent instance
to prevent shared-state races.

Single-threaded LLM inference constraint: tasks run one at a time,
but the CLI remains responsive while tasks execute.
"""

from __future__ import annotations

import copy
import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class TaskRecord:
    """A submitted task with its lifecycle state."""
    id: str
    goal: str
    status: TaskStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result_text: str | None = None
    error_text: str | None = None
    log_path: str | None = None

    @property
    def elapsed(self) -> float | None:
        if self.started_at:
            end = self.finished_at or time.time()
            return end - self.started_at
        return None

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED)


# -------------------------------------------------------------------
# Task Store (SQLite persistence)
# -------------------------------------------------------------------

class TaskStore:
    """SQLite-backed task persistence. Thread-safe via internal lock."""

    def __init__(self, db_path: str = "~/.open_harness/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                result_text TEXT,
                error_text TEXT,
                log_path TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status, created_at DESC);
        """)
        self._conn.commit()

    def create_task(self, goal: str, log_path: str) -> str:
        task_id = uuid.uuid4().hex[:8]
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks (id, goal, status, created_at, log_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, goal, TaskStatus.QUEUED.value, now, log_path),
            )
            self._conn.commit()
        return task_id

    def mark_running(self, task_id: str):
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status = ?, started_at = ? WHERE id = ?",
                (TaskStatus.RUNNING.value, time.time(), task_id),
            )
            self._conn.commit()

    def mark_succeeded(self, task_id: str, result_text: str):
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status = ?, finished_at = ?, result_text = ? "
                "WHERE id = ?",
                (TaskStatus.SUCCEEDED.value, time.time(), result_text, task_id),
            )
            self._conn.commit()

    def mark_failed(self, task_id: str, error_text: str):
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status = ?, finished_at = ?, error_text = ? "
                "WHERE id = ?",
                (TaskStatus.FAILED.value, time.time(), error_text, task_id),
            )
            self._conn.commit()

    def mark_canceled(self, task_id: str):
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status = ?, finished_at = ? WHERE id = ?",
                (TaskStatus.CANCELED.value, time.time(), task_id),
            )
            self._conn.commit()

    def recover_stale_running(self):
        """Mark any 'running' tasks from a previous crash as failed."""
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status = ?, finished_at = ?, "
                "error_text = 'Process crashed during execution' "
                "WHERE status = ?",
                (TaskStatus.FAILED.value, time.time(), TaskStatus.RUNNING.value),
            )
            self._conn.commit()

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, goal, status, created_at, started_at, finished_at, "
                "result_text, error_text, log_path FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_tasks(self, limit: int = 20) -> list[TaskRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, goal, status, created_at, started_at, finished_at, "
                "result_text, error_text, log_path FROM tasks "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_queued_ids(self) -> list[str]:
        """Get IDs of queued tasks in FIFO order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM tasks WHERE status = ? ORDER BY created_at",
                (TaskStatus.QUEUED.value,),
            ).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _row_to_record(row: tuple) -> TaskRecord:
        return TaskRecord(
            id=row[0], goal=row[1], status=TaskStatus(row[2]),
            created_at=row[3], started_at=row[4], finished_at=row[5],
            result_text=row[6], error_text=row[7], log_path=row[8],
        )

    def close(self):
        self._conn.close()


# -------------------------------------------------------------------
# Task Queue Manager
# -------------------------------------------------------------------

class TaskQueueManager:
    """Manages background task execution with sequential processing.

    Spawns a single worker thread that processes tasks one at a time.
    Each task gets a fresh Agent instance to avoid shared-state issues.
    """

    def __init__(
        self,
        store: TaskStore,
        agent_factory: Any,  # callable() -> Agent
        on_complete: Any | None = None,  # callback(TaskRecord)
    ):
        self.store = store
        self._agent_factory = agent_factory
        self._on_complete = on_complete
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._current_task_id: str | None = None
        self._lock = threading.Lock()

    def start(self):
        """Start the background worker thread."""
        if self._worker and self._worker.is_alive():
            return
        # Recover: mark stale running tasks as failed, re-enqueue queued
        self.store.recover_stale_running()
        for task_id in self.store.get_queued_ids():
            self._queue.put(task_id)
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="task-worker")
        self._worker.start()

    def shutdown(self, timeout: float = 30) -> bool:
        """Stop the worker thread gracefully.

        Waits for the current task to finish (up to timeout seconds).
        Any remaining queued tasks stay in DB for next startup.

        Returns True if the worker stopped cleanly, False on timeout.
        """
        self._stop.set()
        self._queue.put("")  # wake up the worker
        if self._worker:
            self._worker.join(timeout=timeout)
            if self._worker.is_alive():
                logger.warning("Worker thread did not stop within %ds", timeout)
                return False
        return True

    def submit(self, goal: str) -> TaskRecord:
        """Submit a goal for background execution."""
        log_dir = Path.home() / ".open_harness" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(
            log_dir / f"task_{int(time.time())}_{uuid.uuid4().hex[:6]}.log")

        task_id = self.store.create_task(goal, log_path)
        self._queue.put(task_id)
        return self.store.get_task(task_id)  # type: ignore

    @property
    def current_task_id(self) -> str | None:
        with self._lock:
            return self._current_task_id

    def is_busy(self) -> bool:
        """True if a task is currently executing."""
        return self.current_task_id is not None

    def list_tasks(self, limit: int = 20) -> list[TaskRecord]:
        return self.store.list_tasks(limit)

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self.store.get_task(task_id)

    def _run(self):
        """Worker loop: process tasks sequentially."""
        while not self._stop.is_set():
            try:
                task_id = self._queue.get(timeout=2)
            except queue.Empty:
                continue

            if not task_id or self._stop.is_set():
                break

            self._execute_task(task_id)

    def _execute_task(self, task_id: str):
        """Execute a single task with an isolated agent."""
        task = self.store.get_task(task_id)
        if not task or task.is_terminal:
            return

        with self._lock:
            self._current_task_id = task_id

        self.store.mark_running(task_id)
        logger.info(f"Task {task_id}: starting goal: {task.goal}")
        agent = None

        try:
            agent = self._agent_factory()
            result_text = ""

            if not task.log_path:
                self.store.mark_failed(task_id, "No log path assigned")
                return

            with open(task.log_path, "w") as f:
                f.write(f"=== Task {task_id}: {task.goal} ===\n")
                f.write(f"=== Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

                for event in agent.run_goal(task.goal):
                    ts = time.strftime("%H:%M:%S")
                    if event.type == "status":
                        f.write(f"[{ts}] {event.data}\n")
                    elif event.type == "tool_call":
                        tool = event.metadata.get("tool", "?")
                        args = str(event.metadata.get("args", {}))[:200]
                        f.write(f"[{ts}] TOOL: {tool} {args}\n")
                    elif event.type == "tool_result":
                        ok = "OK" if event.metadata.get("success") else "FAIL"
                        f.write(f"[{ts}] RESULT ({ok}): {event.data[:500]}\n")
                    elif event.type == "thinking":
                        f.write(f"[{ts}] THINKING: {event.data[:200]}...\n")
                    elif event.type == "text":
                        f.write(event.data)
                    elif event.type == "compensation":
                        f.write(f"[{ts}] COMPENSATE: {event.data}\n")
                    elif event.type == "done":
                        f.write(f"\n\n=== DONE ===\n{event.data}\n")
                        result_text = event.data
                    f.flush()

                f.write(f"\n=== Finished: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

            self.store.mark_succeeded(task_id, result_text)
            logger.info(f"Task {task_id}: succeeded")

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self.store.mark_failed(task_id, error)
            logger.error(f"Task {task_id}: failed: {error}")
        finally:
            # Always clean up agent resources
            if agent:
                try:
                    agent.router.close()
                    agent.memory.close()
                    agent.project_memory_store.close()
                except (OSError, Exception) as e:
                    logger.debug("Error cleaning up agent for task %s: %s", task_id, e)

            with self._lock:
                self._current_task_id = None

            # Notify completion
            updated = self.store.get_task(task_id)
            if updated and self._on_complete:
                try:
                    self._on_complete(updated)
                except Exception as e:
                    logger.warning("on_complete callback failed for task %s: %s", task_id, e)
