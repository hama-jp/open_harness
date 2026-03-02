"""Project memory — persistent key/value fact store.

Users and LLM tools can explicitly ``remember`` and ``forget`` facts.
No automatic learning — only explicit actions.
"""

from __future__ import annotations

import logging
import re

from open_harness_v2.memory.store import MemoryStore

_logger = logging.getLogger(__name__)


def _sanitize(text: str, max_length: int = 500) -> str:
    """Strip control characters and limit length to prevent prompt injection."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return cleaned[:max_length]


class ProjectMemory:
    """Key/value fact store scoped to a project.

    Parameters
    ----------
    store:
        Shared ``MemoryStore`` backend.
    project_id:
        Identifier for the current project (e.g. hash of cwd).
    """

    def __init__(self, store: MemoryStore, project_id: str) -> None:
        self._store = store
        self._project_id = project_id
        self._cache: list[tuple[str, str]] | None = None

    async def remember(self, key: str, value: str) -> None:
        """Store or update a fact (upsert)."""
        key = _sanitize(key, max_length=100)
        value = _sanitize(value, max_length=500)
        await self._store.upsert_fact(self._project_id, key, value)
        self._cache = None  # invalidate

    async def forget(self, key: str) -> None:
        """Delete a fact."""
        await self._store.delete_fact(self._project_id, key)
        self._cache = None

    async def recall(self, key: str) -> str | None:
        """Get a single fact by key."""
        facts = await self.list_all()
        for k, v in facts:
            if k == key:
                return v
        return None

    async def list_all(self) -> list[tuple[str, str]]:
        """Return all facts as ``(key, value)`` pairs."""
        if self._cache is None:
            self._cache = await self._store.get_facts(self._project_id)
        return list(self._cache)

    async def build_context_block(self) -> str:
        """Build a string suitable for injection into the system prompt.

        Returns an empty string when there are no facts.  The result is
        cached until a ``remember`` or ``forget`` invalidates it.
        """
        facts = await self.list_all()
        if not facts:
            return ""
        lines = ["## Project Memory", ""]
        for key, value in facts:
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)
