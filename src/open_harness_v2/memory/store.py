"""SQLite backend for the memory subsystem.

Manages two tables:
  - ``sessions`` — serialized conversation history per session
  - ``facts``    — key/value project-specific facts

All public methods are async, wrapping synchronous sqlite3 calls via
``asyncio.to_thread`` so they integrate cleanly with the async orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".open_harness"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "memory.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    messages   TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    project_id TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (project_id, key)
);
"""


class MemoryStore:
    """Thin async wrapper around a SQLite database."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path), check_same_thread=False,
            )
            self._conn.executescript(_SCHEMA)
        return self._conn

    def _close_sync(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def _save_messages_sync(
        self, session_id: str, messages: list[dict[str, Any]], ts: float,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, messages, updated_at) "
                "VALUES (?, ?, ?)",
                (session_id, json.dumps(messages, ensure_ascii=False), ts),
            )
            conn.commit()

    async def save_messages(
        self, session_id: str, messages: list[dict[str, Any]],
    ) -> None:
        import time
        await asyncio.to_thread(self._save_messages_sync, session_id, messages, time.time())

    def _load_messages_sync(self, session_id: str) -> list[dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT messages FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return []
        return json.loads(row[0])

    async def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._load_messages_sync, session_id)

    def _delete_session_sync(self, session_id: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete_session_sync, session_id)

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def _upsert_fact_sync(
        self, project_id: str, key: str, value: str, ts: float,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO facts (project_id, key, value, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (project_id, key, value, ts),
            )
            conn.commit()

    async def upsert_fact(self, project_id: str, key: str, value: str) -> None:
        import time
        await asyncio.to_thread(self._upsert_fact_sync, project_id, key, value, time.time())

    def _delete_fact_sync(self, project_id: str, key: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM facts WHERE project_id = ? AND key = ?",
                (project_id, key),
            )
            conn.commit()

    async def delete_fact(self, project_id: str, key: str) -> None:
        await asyncio.to_thread(self._delete_fact_sync, project_id, key)

    def _get_facts_sync(self, project_id: str) -> list[tuple[str, str]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT key, value FROM facts WHERE project_id = ? ORDER BY key",
            (project_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    async def get_facts(self, project_id: str) -> list[tuple[str, str]]:
        return await asyncio.to_thread(self._get_facts_sync, project_id)
