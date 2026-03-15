"""Todo/task tracking for agent sessions, inspired by Claude Code's TodoWrite.

Provides in-session task tracking that the agent can use to plan
and track progress on multi-step goals.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


class TodoStatus(enum.Enum):
    """Status of a todo item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class TodoItem:
    """A single todo item."""

    id: int
    content: str
    status: TodoStatus = TodoStatus.PENDING

    def to_display(self) -> str:
        """Format for display."""
        icons = {
            TodoStatus.PENDING: "[ ]",
            TodoStatus.IN_PROGRESS: "[~]",
            TodoStatus.COMPLETED: "[x]",
        }
        return f"{icons[self.status]} {self.id}. {self.content}"


class TodoManager:
    """Manages a session-scoped todo list.

    The agent can add, update, and complete items during a goal.
    The user can view and manage todos via REPL commands.
    """

    def __init__(self) -> None:
        self._items: list[TodoItem] = []
        self._next_id = 1

    def add(self, content: str) -> TodoItem:
        """Add a new todo item."""
        item = TodoItem(id=self._next_id, content=content)
        self._items.append(item)
        self._next_id += 1
        return item

    def start(self, item_id: int) -> bool:
        """Mark a todo as in_progress."""
        item = self._get(item_id)
        if item:
            item.status = TodoStatus.IN_PROGRESS
            return True
        return False

    def complete(self, item_id: int) -> bool:
        """Mark a todo as completed."""
        item = self._get(item_id)
        if item:
            item.status = TodoStatus.COMPLETED
            return True
        return False

    def remove(self, item_id: int) -> bool:
        """Remove a todo item."""
        item = self._get(item_id)
        if item:
            self._items.remove(item)
            return True
        return False

    def clear(self) -> None:
        """Clear all todos."""
        self._items.clear()
        self._next_id = 1

    def list_all(self) -> list[TodoItem]:
        """Return all todo items."""
        return list(self._items)

    def list_pending(self) -> list[TodoItem]:
        """Return pending and in-progress items."""
        return [i for i in self._items if i.status != TodoStatus.COMPLETED]

    def summary(self) -> str:
        """Return a one-line summary."""
        total = len(self._items)
        done = sum(1 for i in self._items if i.status == TodoStatus.COMPLETED)
        active = sum(1 for i in self._items if i.status == TodoStatus.IN_PROGRESS)
        return f"{done}/{total} done, {active} active"

    def to_context_block(self) -> str:
        """Build a context block for injecting into system prompt."""
        if not self._items:
            return ""
        lines = ["## Current Tasks"]
        for item in self._items:
            lines.append(item.to_display())
        return "\n".join(lines)

    def _get(self, item_id: int) -> TodoItem | None:
        for item in self._items:
            if item.id == item_id:
                return item
        return None
