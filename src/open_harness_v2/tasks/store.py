"""SQLite persistence for task records.

Shares the same DB directory as MemoryStore but uses a separate file
(``~/.open_harness/tasks.db``) to keep concerns separated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from open_harness_v2.tasks.record import TaskRecord, TaskStatus

_logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".open_harness"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "tasks.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    goal        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    result      TEXT,
    error       TEXT
);
"""


class TaskStore:
    """Synchronous SQLite store — wrapped with async helpers for the manager."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path), check_same_thread=False,
            )
            self._conn.executescript(_SCHEMA)
        return self._conn

    # ------------------------------------------------------------------
    # CRUD (sync, called via asyncio.to_thread in manager)
    # ------------------------------------------------------------------

    def create(self, goal: str) -> TaskRecord:
        record = TaskRecord(goal=goal, created_at=time.time())
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO tasks (id, goal, status, created_at) VALUES (?, ?, ?, ?)",
                (record.id, record.goal, record.status.value, record.created_at),
            )
            conn.commit()
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        with self._lock:
            conn = self._get_conn()
            now = time.time()
            if status == TaskStatus.RUNNING:
                conn.execute(
                    "UPDATE tasks SET status = ?, started_at = ? WHERE id = ?",
                    (status.value, now, task_id),
                )
            elif status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                conn.execute(
                    "UPDATE tasks SET status = ?, finished_at = ? WHERE id = ?",
                    (status.value, now, task_id),
                )
            else:
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ?",
                    (status.value, task_id),
                )
            conn.commit()

    def update_result(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE tasks SET status = ?, finished_at = ?, result = ?, error = ? WHERE id = ?",
                (status.value, time.time(), result, error, task_id),
            )
            conn.commit()

    def list_recent(self, limit: int = 20) -> list[TaskRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def recover_stale(self) -> int:
        """Move RUNNING tasks back to QUEUED (for crash recovery)."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "UPDATE tasks SET status = ? WHERE status = ?",
                (TaskStatus.QUEUED.value, TaskStatus.RUNNING.value),
            )
            conn.commit()
            return cursor.rowcount

    def _close_sync(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    @staticmethod
    def _row_to_record(row: tuple) -> TaskRecord:
        return TaskRecord(
            id=row[0],
            goal=row[1],
            status=TaskStatus(row[2]),
            created_at=row[3],
            started_at=row[4],
            finished_at=row[5],
            result=row[6],
            error=row[7],
        )
