"""Tests for todo/task tracking."""

from __future__ import annotations

from open_harness_v2.todo import TodoItem, TodoManager, TodoStatus


class TestTodoManager:
    def test_add(self):
        tm = TodoManager()
        item = tm.add("Fix the bug")
        assert item.id == 1
        assert item.content == "Fix the bug"
        assert item.status == TodoStatus.PENDING

    def test_auto_increment_ids(self):
        tm = TodoManager()
        a = tm.add("task a")
        b = tm.add("task b")
        assert a.id == 1
        assert b.id == 2

    def test_start(self):
        tm = TodoManager()
        item = tm.add("task")
        assert tm.start(item.id) is True
        assert item.status == TodoStatus.IN_PROGRESS

    def test_complete(self):
        tm = TodoManager()
        item = tm.add("task")
        assert tm.complete(item.id) is True
        assert item.status == TodoStatus.COMPLETED

    def test_start_nonexistent(self):
        tm = TodoManager()
        assert tm.start(999) is False

    def test_complete_nonexistent(self):
        tm = TodoManager()
        assert tm.complete(999) is False

    def test_remove(self):
        tm = TodoManager()
        item = tm.add("task")
        assert tm.remove(item.id) is True
        assert len(tm.list_all()) == 0

    def test_remove_nonexistent(self):
        tm = TodoManager()
        assert tm.remove(999) is False

    def test_clear(self):
        tm = TodoManager()
        tm.add("a")
        tm.add("b")
        tm.clear()
        assert len(tm.list_all()) == 0

    def test_list_pending(self):
        tm = TodoManager()
        a = tm.add("pending")
        b = tm.add("done")
        tm.complete(b.id)
        pending = tm.list_pending()
        assert len(pending) == 1
        assert pending[0].id == a.id

    def test_summary(self):
        tm = TodoManager()
        tm.add("a")
        tm.add("b")
        b = tm.list_all()[1]
        tm.start(b.id)
        summary = tm.summary()
        assert "0/2 done" in summary
        assert "1 active" in summary

    def test_to_context_block_empty(self):
        tm = TodoManager()
        assert tm.to_context_block() == ""

    def test_to_context_block_with_items(self):
        tm = TodoManager()
        tm.add("Fix bug")
        tm.add("Write tests")
        block = tm.to_context_block()
        assert "Current Tasks" in block
        assert "Fix bug" in block
        assert "Write tests" in block


class TestTodoItem:
    def test_display_pending(self):
        item = TodoItem(id=1, content="task", status=TodoStatus.PENDING)
        assert "[ ]" in item.to_display()

    def test_display_in_progress(self):
        item = TodoItem(id=1, content="task", status=TodoStatus.IN_PROGRESS)
        assert "[~]" in item.to_display()

    def test_display_completed(self):
        item = TodoItem(id=1, content="task", status=TodoStatus.COMPLETED)
        assert "[x]" in item.to_display()
