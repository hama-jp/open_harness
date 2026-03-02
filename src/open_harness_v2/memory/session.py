"""Session memory — persist conversation history across sessions.

Subscribes to ``AGENT_DONE`` on the EventBus so that the conversation is
automatically saved whenever a goal completes.
"""

from __future__ import annotations

import logging
from typing import Any

from open_harness_v2.memory.store import MemoryStore
from open_harness_v2.types import AgentEvent, EventType

_logger = logging.getLogger(__name__)


class SessionMemory:
    """Persist and restore conversation messages across sessions.

    Parameters
    ----------
    store:
        Shared ``MemoryStore`` backend.
    max_turns:
        Maximum number of messages to keep per session.
    """

    def __init__(self, store: MemoryStore, max_turns: int = 50) -> None:
        self._store = store
        self._max_turns = max_turns
        self._session_id: str | None = None
        self._messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Save (replace) messages for *session_id*."""
        trimmed = messages[-self._max_turns :] if len(messages) > self._max_turns else messages
        await self._store.save_messages(session_id, trimmed)
        _logger.debug("Saved %d messages for session %s", len(trimmed), session_id)

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """Load saved messages for *session_id*."""
        msgs = await self._store.load_messages(session_id)
        _logger.debug("Loaded %d messages for session %s", len(msgs), session_id)
        return msgs

    async def clear(self, session_id: str) -> None:
        """Delete saved messages for *session_id*."""
        await self._store.delete_session(session_id)

    # ------------------------------------------------------------------
    # EventBus integration
    # ------------------------------------------------------------------

    def bind(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Bind a live session so that auto-save knows what to save."""
        self._session_id = session_id
        self._messages = messages

    def attach(self, event_bus: "open_harness_v2.events.bus.EventBus") -> None:  # noqa: F821
        """Subscribe to AGENT_DONE for automatic saves."""
        event_bus.subscribe(EventType.AGENT_DONE, self._on_agent_done)

    async def _on_agent_done(self, event: AgentEvent) -> None:
        if self._session_id and self._messages:
            await self.save(self._session_id, self._messages)
