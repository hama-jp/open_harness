"""Task record and status types."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field


class TaskStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskRecord:
    """Immutable snapshot of a background task."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    goal: str = ""
    status: TaskStatus = TaskStatus.QUEUED
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    result: str | None = None
    error: str | None = None

    @property
    def elapsed(self) -> float | None:
        """Seconds from start to finish, or None if not started."""
        if self.started_at is None:
            return None
        end = self.finished_at or 0.0
        if end == 0.0:
            import time
            end = time.time()
        return end - self.started_at

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )
