"""Conversation memory store."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # user, assistant, system, tool
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class MemoryStore:
    """SQLite-backed conversation and memory store."""

    def __init__(self, db_path: str = "~/.open_harness/memory.db", max_turns: int = 50):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()
        self._conversation: list[ConversationTurn] = []
        self._max_turns = max_turns

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_mem_category ON memories(category);
        """)
        self._conn.commit()

    def add_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None):
        """Add a conversation turn."""
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._conversation.append(turn)

        # Trim conversation if too long
        if len(self._conversation) > self._max_turns:
            self._conversation = self._conversation[-self._max_turns:]

    def get_messages(self, include_system: bool = False) -> list[dict[str, str]]:
        """Get conversation history as message list."""
        msgs = []
        for turn in self._conversation:
            if turn.role == "system" and not include_system:
                continue
            msgs.append(turn.to_message())
        return msgs

    def clear_conversation(self):
        """Clear current conversation."""
        self._conversation.clear()

    def save_session(self, session_id: str):
        """Persist current conversation to database."""
        for turn in self._conversation:
            self._conn.execute(
                "INSERT INTO conversations (session_id, role, content, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, turn.role, turn.content,
                 json.dumps(turn.metadata), turn.timestamp),
            )
        self._conn.commit()

    def load_session(self, session_id: str):
        """Load a conversation from database."""
        self._conversation.clear()
        rows = self._conn.execute(
            "SELECT role, content, metadata, created_at FROM conversations "
            "WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        for role, content, meta_str, ts in rows:
            self._conversation.append(ConversationTurn(
                role=role,
                content=content,
                timestamp=ts,
                metadata=json.loads(meta_str),
            ))

    def remember(self, key: str, value: str, category: str = "general"):
        """Store a persistent memory."""
        now = time.time()
        self._conn.execute(
            "INSERT INTO memories (key, value, category, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
            (key, value, category, now, now, value, now),
        )
        self._conn.commit()

    def recall(self, key: str) -> str | None:
        """Retrieve a persistent memory."""
        row = self._conn.execute(
            "SELECT value FROM memories WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def search_memories(self, query: str, category: str | None = None) -> list[tuple[str, str]]:
        """Search memories by key pattern."""
        if category:
            rows = self._conn.execute(
                "SELECT key, value FROM memories WHERE key LIKE ? AND category = ?",
                (f"%{query}%", category),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, value FROM memories WHERE key LIKE ?",
                (f"%{query}%",),
            ).fetchall()
        return rows

    def close(self):
        self._conn.close()
