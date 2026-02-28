"""Persistent project memory — auto-learned knowledge per project.

Stores patterns, structure knowledge, error hints, and runbooks that
persist across sessions. Memories are automatically learned from tool
usage and injected into LLM prompts.

Categories:
  pattern   - project conventions (test commands, build tools, etc.)
  structure - file/directory layout knowledge
  error     - error patterns and fixes
  runbook   - reusable step sequences for common tasks
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Limits to prevent memory bloat
MAX_MEMORIES_PER_PROJECT = 200
MAX_MEMORY_BLOCK_CHARS = 1200
PROMOTION_THRESHOLD = 2  # seen N times before persisting
STALE_DAYS = 60
MIN_SCORE_TO_KEEP = 0.15


@dataclass
class MemoryItem:
    """A single piece of learned project knowledge."""
    key: str
    value: str
    kind: str  # pattern, structure, error, runbook
    score: float = 0.5
    seen_count: int = 1
    pinned: bool = False


@dataclass
class Runbook:
    """A reusable sequence of steps for a common task."""
    slug: str
    title: str
    trigger: str  # keywords that activate this runbook
    steps: list[str]
    usage_count: int = 0
    success_count: int = 0


class ProjectMemoryStore:
    """SQLite-backed persistent project memory."""

    def __init__(self, db_path: str = "~/.open_harness/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS project_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0.5,
                seen_count INTEGER NOT NULL DEFAULT 1,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(project_id, kind, key)
            );
            CREATE INDEX IF NOT EXISTS idx_pmem_project_kind
                ON project_memories(project_id, kind, score DESC);

            CREATE TABLE IF NOT EXISTS runbooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                slug TEXT NOT NULL,
                title TEXT NOT NULL,
                trigger_text TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                usage_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(project_id, slug)
            );
        """)
        self._conn.commit()

    def _project_id(self, project_root: str) -> str:
        """Stable identifier for a project directory."""
        return hashlib.sha256(
            str(Path(project_root).resolve()).encode()
        ).hexdigest()[:16]

    # ---------------------------------------------------------------
    # CRUD
    # ---------------------------------------------------------------

    def upsert(self, project_root: str, kind: str, key: str, value: str,
               score: float = 0.5):
        """Insert or update a memory item."""
        pid = self._project_id(project_root)
        now = time.time()
        self._conn.execute("""
            INSERT INTO project_memories
                (project_id, kind, key, value, score, seen_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(project_id, kind, key) DO UPDATE SET
                value = excluded.value,
                score = MAX(score, excluded.score),
                seen_count = seen_count + 1,
                updated_at = excluded.updated_at
        """, (pid, kind, key, value, score, now, now))
        self._conn.commit()

    def get_memories(self, project_root: str,
                     kind: str | None = None,
                     limit: int = 20) -> list[MemoryItem]:
        """Get top memories for a project, sorted by score."""
        pid = self._project_id(project_root)
        if kind:
            rows = self._conn.execute(
                "SELECT key, value, kind, score, seen_count, pinned "
                "FROM project_memories WHERE project_id = ? AND kind = ? "
                "ORDER BY pinned DESC, score DESC LIMIT ?",
                (pid, kind, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT key, value, kind, score, seen_count, pinned "
                "FROM project_memories WHERE project_id = ? "
                "ORDER BY pinned DESC, score DESC LIMIT ?",
                (pid, limit),
            ).fetchall()
        return [
            MemoryItem(key=r[0], value=r[1], kind=r[2],
                       score=r[3], seen_count=r[4], pinned=bool(r[5]))
            for r in rows
        ]

    def upsert_runbook(self, project_root: str, slug: str, title: str,
                       trigger: str, steps: list[str]):
        """Insert or update a runbook."""
        pid = self._project_id(project_root)
        now = time.time()
        self._conn.execute("""
            INSERT INTO runbooks
                (project_id, slug, title, trigger_text, steps_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, slug) DO UPDATE SET
                title = excluded.title,
                trigger_text = excluded.trigger_text,
                steps_json = excluded.steps_json,
                updated_at = excluded.updated_at
        """, (pid, slug, title, trigger, json.dumps(steps), now, now))
        self._conn.commit()

    def get_runbooks(self, project_root: str, limit: int = 10) -> list[Runbook]:
        """Get runbooks for a project."""
        pid = self._project_id(project_root)
        rows = self._conn.execute(
            "SELECT slug, title, trigger_text, steps_json, usage_count, success_count "
            "FROM runbooks WHERE project_id = ? ORDER BY usage_count DESC LIMIT ?",
            (pid, limit),
        ).fetchall()
        return [
            Runbook(slug=r[0], title=r[1], trigger=r[2],
                    steps=json.loads(r[3]), usage_count=r[4], success_count=r[5])
            for r in rows
        ]

    def record_runbook_usage(self, project_root: str, slug: str, success: bool):
        """Record that a runbook was used."""
        pid = self._project_id(project_root)
        if success:
            self._conn.execute(
                "UPDATE runbooks SET usage_count = usage_count + 1, "
                "success_count = success_count + 1, updated_at = ? "
                "WHERE project_id = ? AND slug = ?",
                (time.time(), pid, slug),
            )
        else:
            self._conn.execute(
                "UPDATE runbooks SET usage_count = usage_count + 1, updated_at = ? "
                "WHERE project_id = ? AND slug = ?",
                (time.time(), pid, slug),
            )
        self._conn.commit()

    def prune(self, project_root: str):
        """Remove stale, low-score memories to prevent bloat."""
        pid = self._project_id(project_root)
        cutoff = time.time() - (STALE_DAYS * 86400)

        # Delete stale, low-score, unpinned memories
        self._conn.execute(
            "DELETE FROM project_memories "
            "WHERE project_id = ? AND pinned = 0 AND score < ? AND updated_at < ?",
            (pid, MIN_SCORE_TO_KEEP, cutoff),
        )

        # Enforce per-project cap — keep top N by score
        self._conn.execute("""
            DELETE FROM project_memories WHERE id IN (
                SELECT id FROM project_memories
                WHERE project_id = ? AND pinned = 0
                ORDER BY score DESC
                LIMIT -1 OFFSET ?
            )
        """, (pid, MAX_MEMORIES_PER_PROJECT))
        self._conn.commit()

    def close(self):
        self._conn.close()


# -------------------------------------------------------------------
# Auto-learning engine
# -------------------------------------------------------------------

class ProjectMemoryEngine:
    """Automatically learns project knowledge from agent interactions."""

    def __init__(self, store: ProjectMemoryStore, project_root: str):
        self.store = store
        self.project_root = project_root
        self._pending: dict[str, dict[str, Any]] = {}  # track patterns before committing

    def on_tool_result(self, tool_name: str, args: dict[str, Any],
                       result_success: bool, result_output: str):
        """Learn from a tool call and its result."""
        # Learn test commands
        if tool_name == "run_tests" and result_success:
            cmd = args.get("command", "") or args.get("target", "")
            if cmd:
                self.store.upsert(
                    self.project_root, "pattern",
                    "test_command", f"Tests run with: {cmd}",
                    score=0.7,
                )

        # Learn shell patterns
        if tool_name == "shell" and result_success:
            cmd = args.get("command", "")
            if cmd and any(k in cmd for k in ["npm", "yarn", "pip", "cargo", "make", "gradle"]):
                key = f"build_tool:{cmd.split()[0]}"
                self.store.upsert(
                    self.project_root, "pattern",
                    key, f"Build/package command: {cmd[:100]}",
                    score=0.5,
                )

        # Learn file structure from read/write operations
        if tool_name in ("read_file", "write_file", "edit_file") and result_success:
            path = args.get("path", "")
            if path:
                dir_path = str(Path(path).parent)
                # Normalize to relative if within project
                try:
                    rel = str(Path(path).resolve().relative_to(
                        Path(self.project_root).resolve()))
                    dir_rel = str(Path(rel).parent)
                    if dir_rel and dir_rel != ".":
                        key = f"dir:{dir_rel}"
                        self.store.upsert(
                            self.project_root, "structure",
                            key, f"Active directory: {dir_rel}",
                            score=0.3,
                        )
                except ValueError:
                    pass

        # Learn from errors
        if not result_success and result_output:
            error_hint = _extract_error_pattern(result_output)
            if error_hint:
                key = f"error:{error_hint['type']}"
                self.store.upsert(
                    self.project_root, "error",
                    key, error_hint["hint"],
                    score=0.4,
                )

    def on_session_end(self):
        """Run cleanup at session end."""
        self.store.prune(self.project_root)


# -------------------------------------------------------------------
# Prompt injection
# -------------------------------------------------------------------

def build_memory_block(store: ProjectMemoryStore, project_root: str,
                       max_chars: int = MAX_MEMORY_BLOCK_CHARS) -> str:
    """Build a compact memory block for prompt injection."""
    memories = store.get_memories(project_root, limit=30)
    runbooks = store.get_runbooks(project_root, limit=5)

    if not memories and not runbooks:
        return ""

    sections: dict[str, list[str]] = {
        "pattern": [],
        "structure": [],
        "error": [],
    }

    for m in memories:
        if m.kind in sections:
            sections[m.kind].append(f"- {m.value}")

    parts = ["PROJECT MEMORY (auto-learned):"]
    if sections["pattern"]:
        parts.append("Patterns:")
        parts.extend(sections["pattern"][:5])
    if sections["structure"]:
        parts.append("Structure:")
        parts.extend(sections["structure"][:5])
    if sections["error"]:
        parts.append("Error hints:")
        parts.extend(sections["error"][:3])
    if runbooks:
        parts.append("Runbooks:")
        for rb in runbooks[:3]:
            steps_short = " -> ".join(rb.steps[:4])
            parts.append(f"- {rb.title}: {steps_short}")

    block = "\n".join(parts)
    if len(block) > max_chars:
        block = block[:max_chars] + "\n..."
    return block


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

_ERROR_PATTERNS = [
    ("ModuleNotFoundError", "ModuleNotFoundError — check imports, __init__.py, or install missing package"),
    ("ImportError", "ImportError — verify module path and package installation"),
    ("FileNotFoundError", "FileNotFoundError — check file path and working directory"),
    ("PermissionError", "PermissionError — check file permissions"),
    ("SyntaxError", "SyntaxError — check for typos, missing colons, or indentation"),
    ("TypeError", "TypeError — check argument types and None values"),
    ("KeyError", "KeyError — check dictionary keys exist before accessing"),
    ("ConnectionError", "ConnectionError — check network/service availability"),
    ("TimeoutError", "TimeoutError — increase timeout or check service responsiveness"),
]


def _extract_error_pattern(output: str) -> dict[str, str] | None:
    """Extract a recognizable error pattern from tool output."""
    for error_type, hint in _ERROR_PATTERNS:
        if error_type in output:
            return {"type": error_type, "hint": hint}
    return None
