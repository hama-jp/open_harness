"""Async pub/sub EventBus for decoupling agent internals from UI."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable

from open_harness_v2.types import AgentEvent, EventType

_logger = logging.getLogger(__name__)

# Sentinel used for wildcard subscriptions (receive all events)
_WILDCARD = "*"

# Type alias for handlers (sync or async callables taking an AgentEvent)
Handler = Callable[[AgentEvent], Any]


class EventBus:
    """Lightweight async pub/sub event bus.

    Features:
    - Subscribe to specific EventType or wildcard ``"*"`` for all events.
    - Handlers can be sync or async — sync handlers are auto-wrapped.
    - ``emit()`` fans out to matching handlers concurrently.
    - ``unsubscribe()`` removes a handler.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self._history: list[AgentEvent] = []
        self._max_history: int = 200

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: EventType | str,
        handler: Handler,
    ) -> None:
        """Register *handler* for *event_type* (or ``"*"`` for all)."""
        key = self._key(event_type)
        self._handlers.setdefault(key, []).append(handler)

    def unsubscribe(
        self,
        event_type: EventType | str,
        handler: Handler,
    ) -> None:
        """Remove *handler* from *event_type*."""
        key = self._key(event_type)
        handlers = self._handlers.get(key, [])
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    async def emit(self, event: AgentEvent) -> None:
        """Emit an event to all matching handlers.

        Handlers for the specific event type AND wildcard handlers are called.
        Exceptions in individual handlers are logged and do not propagate.
        """
        # Record in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        key = self._key(event.type)
        handlers = list(self._handlers.get(key, []))
        handlers.extend(self._handlers.get(_WILDCARD, []))

        if not handlers:
            return

        tasks = []
        for handler in handlers:
            tasks.append(self._call_handler(handler, event))
        await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def history(self) -> list[AgentEvent]:
        """Return a copy of the event history."""
        return list(self._history)

    def clear(self) -> None:
        """Remove all handlers and history."""
        self._handlers.clear()
        self._history.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _key(event_type: EventType | str) -> str:
        if isinstance(event_type, EventType):
            return event_type.value
        return str(event_type)

    @staticmethod
    async def _call_handler(handler: Handler, event: AgentEvent) -> None:
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            _logger.exception(
                "EventBus handler %s raised for event %s",
                getattr(handler, "__name__", handler),
                event.type,
            )
