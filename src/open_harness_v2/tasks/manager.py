"""Asyncio-native task manager for background goal execution.

Uses ``asyncio.Task`` + ``asyncio.Semaphore`` for concurrency control and
``EventBus`` for progress notifications.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, TYPE_CHECKING

from open_harness_v2.tasks.record import TaskRecord, TaskStatus
from open_harness_v2.tasks.store import TaskStore
from open_harness_v2.types import AgentEvent, EventType

if TYPE_CHECKING:
    from open_harness_v2.core.orchestrator import Orchestrator
    from open_harness_v2.events.bus import EventBus

_logger = logging.getLogger(__name__)


class TaskManager:
    """Asyncio-based background task queue.

    Parameters
    ----------
    store:
        Persistent task store.
    event_bus:
        EventBus for notifications.
    orchestrator_factory:
        Callable that creates a fresh ``Orchestrator`` for each task.
    max_concurrent:
        Maximum number of tasks running in parallel (1 recommended
        for local LLM backends).
    """

    def __init__(
        self,
        store: TaskStore,
        event_bus: EventBus,
        orchestrator_factory: Callable[[], Orchestrator],
        max_concurrent: int = 1,
    ) -> None:
        self._store = store
        self._event_bus = event_bus
        self._factory = orchestrator_factory
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running: dict[str, asyncio.Task[None]] = {}

    async def submit(self, goal: str) -> TaskRecord:
        """Add a task and start background execution."""
        record = await asyncio.to_thread(self._store.create, goal)
        task = asyncio.create_task(self._execute(record))
        self._running[record.id] = task
        return record

    async def _execute(self, record: TaskRecord) -> None:
        try:
            async with self._semaphore:
                await asyncio.to_thread(
                    self._store.update_status, record.id, TaskStatus.RUNNING,
                )
                await self._event_bus.emit(AgentEvent(
                    type=EventType.TASK_STARTED,
                    data={"task_id": record.id, "goal": record.goal},
                ))

                orchestrator = self._factory()
                result = await orchestrator.run(record.goal)
                await asyncio.to_thread(
                    self._store.update_result,
                    record.id,
                    TaskStatus.SUCCEEDED,
                    result=result,
                )
                await self._event_bus.emit(AgentEvent(
                    type=EventType.TASK_COMPLETED,
                    data={
                        "task_id": record.id,
                        "status": "succeeded",
                        "result": (result or "")[:200],
                    },
                ))
        except asyncio.CancelledError:
            await asyncio.to_thread(
                self._store.update_status, record.id, TaskStatus.CANCELLED,
            )
            await self._event_bus.emit(AgentEvent(
                type=EventType.TASK_COMPLETED,
                data={"task_id": record.id, "status": "cancelled"},
            ))
        except Exception as exc:
            _logger.exception("Task %s failed", record.id)
            await asyncio.to_thread(
                self._store.update_result,
                record.id,
                TaskStatus.FAILED,
                error=str(exc),
            )
            await self._event_bus.emit(AgentEvent(
                type=EventType.TASK_COMPLETED,
                data={
                    "task_id": record.id,
                    "status": "failed",
                    "error": str(exc),
                },
            ))
        finally:
            self._running.pop(record.id, None)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running or queued task. Returns True if cancelled."""
        task = self._running.get(task_id)
        if task and not task.done():
            task.cancel()
            return True
        # If queued but not yet picked up
        record = await asyncio.to_thread(self._store.get, task_id)
        if record and record.status == TaskStatus.QUEUED:
            await asyncio.to_thread(
                self._store.update_status, task_id, TaskStatus.CANCELLED,
            )
            return True
        return False

    def list_tasks(self, limit: int = 20) -> list[TaskRecord]:
        """List recent tasks (sync — safe because it's read-only)."""
        return self._store.list_recent(limit)

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._store.get(task_id)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel all running tasks and wait for them to finish."""
        for task_id, task in list(self._running.items()):
            if not task.done():
                task.cancel()
        if self._running:
            pending = [t for t in self._running.values() if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=timeout)

    async def recover(self) -> int:
        """Recover stale RUNNING tasks from a previous crash."""
        return await asyncio.to_thread(self._store.recover_stale)
