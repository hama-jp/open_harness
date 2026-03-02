"""Tests for the v2 memory subsystem."""

from __future__ import annotations

import asyncio

import pytest

from open_harness_v2.memory.store import MemoryStore
from open_harness_v2.memory.session import SessionMemory
from open_harness_v2.memory.project import ProjectMemory, _sanitize
from open_harness_v2.tools.builtin.memory_tool import RememberTool, ForgetTool
from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType


# ===================================================================
# MemoryStore
# ===================================================================

class TestMemoryStore:
    """Low-level SQLite store operations."""

    @pytest.fixture
    def store(self, tmp_path):
        return MemoryStore(db_path=tmp_path / "test.db")

    async def test_save_and_load_messages(self, store):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        await store.save_messages("s1", msgs)
        loaded = await store.load_messages("s1")
        assert loaded == msgs

    async def test_load_nonexistent_session(self, store):
        loaded = await store.load_messages("nonexistent")
        assert loaded == []

    async def test_save_overwrites(self, store):
        await store.save_messages("s1", [{"role": "user", "content": "first"}])
        await store.save_messages("s1", [{"role": "user", "content": "second"}])
        loaded = await store.load_messages("s1")
        assert len(loaded) == 1
        assert loaded[0]["content"] == "second"

    async def test_delete_session(self, store):
        await store.save_messages("s1", [{"role": "user", "content": "hi"}])
        await store.delete_session("s1")
        assert await store.load_messages("s1") == []

    async def test_upsert_and_get_facts(self, store):
        await store.upsert_fact("proj1", "lang", "Python")
        await store.upsert_fact("proj1", "framework", "FastAPI")
        facts = await store.get_facts("proj1")
        assert ("framework", "FastAPI") in facts
        assert ("lang", "Python") in facts

    async def test_upsert_overwrites(self, store):
        await store.upsert_fact("proj1", "lang", "Python")
        await store.upsert_fact("proj1", "lang", "Rust")
        facts = await store.get_facts("proj1")
        assert facts == [("lang", "Rust")]

    async def test_delete_fact(self, store):
        await store.upsert_fact("proj1", "lang", "Python")
        await store.delete_fact("proj1", "lang")
        assert await store.get_facts("proj1") == []

    async def test_facts_scoped_by_project(self, store):
        await store.upsert_fact("proj1", "lang", "Python")
        await store.upsert_fact("proj2", "lang", "Rust")
        assert await store.get_facts("proj1") == [("lang", "Python")]
        assert await store.get_facts("proj2") == [("lang", "Rust")]

    async def test_close(self, store):
        await store.save_messages("s1", [{"role": "user", "content": "hi"}])
        await store.close()
        # After close, creating a new connection should still work
        loaded = await store.load_messages("s1")
        assert loaded == [{"role": "user", "content": "hi"}]


# ===================================================================
# SessionMemory
# ===================================================================

class TestSessionMemory:
    """Session persistence with EventBus auto-save."""

    @pytest.fixture
    def store(self, tmp_path):
        return MemoryStore(db_path=tmp_path / "test.db")

    @pytest.fixture
    def session(self, store):
        return SessionMemory(store, max_turns=5)

    async def test_save_and_load(self, session):
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(3)]
        await session.save("s1", msgs)
        loaded = await session.load("s1")
        assert loaded == msgs

    async def test_max_turns_trims(self, session):
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        await session.save("s1", msgs)
        loaded = await session.load("s1")
        assert len(loaded) == 5
        assert loaded[0]["content"] == "msg5"  # oldest trimmed

    async def test_clear(self, session):
        await session.save("s1", [{"role": "user", "content": "hi"}])
        await session.clear("s1")
        assert await session.load("s1") == []

    async def test_auto_save_on_agent_done(self, session):
        bus = EventBus()
        session.attach(bus)

        msgs = [{"role": "user", "content": "hello"}]
        session.bind("s1", msgs)

        await bus.emit(AgentEvent(type=EventType.AGENT_DONE, data={"response": "ok"}))

        loaded = await session.load("s1")
        assert loaded == msgs


# ===================================================================
# ProjectMemory
# ===================================================================

class TestProjectMemory:
    """Key-value fact store."""

    @pytest.fixture
    def store(self, tmp_path):
        return MemoryStore(db_path=tmp_path / "test.db")

    @pytest.fixture
    def memory(self, store):
        return ProjectMemory(store, project_id="test-proj")

    async def test_remember_and_recall(self, memory):
        await memory.remember("lang", "Python")
        assert await memory.recall("lang") == "Python"

    async def test_recall_nonexistent(self, memory):
        assert await memory.recall("nope") is None

    async def test_forget(self, memory):
        await memory.remember("lang", "Python")
        await memory.forget("lang")
        assert await memory.recall("lang") is None

    async def test_list_all(self, memory):
        await memory.remember("lang", "Python")
        await memory.remember("framework", "FastAPI")
        facts = await memory.list_all()
        assert len(facts) == 2
        assert ("lang", "Python") in facts

    async def test_upsert_updates_value(self, memory):
        await memory.remember("lang", "Python")
        await memory.remember("lang", "Rust")
        assert await memory.recall("lang") == "Rust"

    async def test_build_context_block_empty(self, memory):
        block = await memory.build_context_block()
        assert block == ""

    async def test_build_context_block_with_facts(self, memory):
        await memory.remember("lang", "Python")
        await memory.remember("test_runner", "pytest")
        block = await memory.build_context_block()
        assert "## Project Memory" in block
        assert "**lang**" in block
        assert "**test_runner**" in block

    async def test_cache_invalidation(self, memory):
        await memory.remember("x", "1")
        facts1 = await memory.list_all()
        await memory.remember("y", "2")
        facts2 = await memory.list_all()
        assert len(facts2) == len(facts1) + 1


# ===================================================================
# Sanitize helper
# ===================================================================

class TestSanitize:
    def test_strips_control_chars(self):
        assert _sanitize("hello\x00world") == "helloworld"

    def test_truncates_long_text(self):
        long_text = "a" * 1000
        assert len(_sanitize(long_text)) == 500

    def test_preserves_normal_text(self):
        assert _sanitize("normal text") == "normal text"


# ===================================================================
# Memory Tools
# ===================================================================

class TestRememberTool:
    @pytest.fixture
    def memory(self, tmp_path):
        store = MemoryStore(db_path=tmp_path / "test.db")
        return ProjectMemory(store, project_id="test-proj")

    @pytest.fixture
    def tool(self, memory):
        return RememberTool(memory)

    async def test_remember(self, tool, memory):
        result = await tool.execute(key="lang", value="Python")
        assert result.success
        assert "Remembered" in result.output
        assert await memory.recall("lang") == "Python"

    async def test_remember_missing_key(self, tool):
        result = await tool.execute(key="", value="val")
        assert not result.success

    async def test_remember_missing_value(self, tool):
        result = await tool.execute(key="key", value="")
        assert not result.success


class TestForgetTool:
    @pytest.fixture
    def memory(self, tmp_path):
        store = MemoryStore(db_path=tmp_path / "test.db")
        return ProjectMemory(store, project_id="test-proj")

    @pytest.fixture
    def tool(self, memory):
        return ForgetTool(memory)

    async def test_forget_existing(self, tool, memory):
        await memory.remember("lang", "Python")
        result = await tool.execute(key="lang")
        assert result.success
        assert await memory.recall("lang") is None

    async def test_forget_nonexistent(self, tool):
        result = await tool.execute(key="nope")
        assert not result.success
        assert "No fact found" in result.error

    async def test_forget_missing_key(self, tool):
        result = await tool.execute(key="")
        assert not result.success
