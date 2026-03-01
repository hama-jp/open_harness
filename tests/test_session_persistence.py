"""Tests for Issue 10: Session Persistence."""

import os
import tempfile

from open_harness.memory.store import MemoryStore


class TestSessionPersistence:
    def setup_method(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = MemoryStore(self.db_path, max_turns=50)

    def teardown_method(self):
        self.store.close()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_save_and_load(self):
        self.store.add_turn("user", "hello")
        self.store.add_turn("assistant", "hi there")
        self.store.save_session("test-session")

        # Create new store to simulate restart
        store2 = MemoryStore(self.db_path)
        store2.load_session("test-session")
        msgs = store2.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["content"] == "hello"
        assert msgs[1]["content"] == "hi there"
        store2.close()

    def test_load_nonexistent_session(self):
        self.store.load_session("nonexistent")
        msgs = self.store.get_messages()
        assert len(msgs) == 0

    def test_save_empty_session(self):
        self.store.save_session("empty")
        store2 = MemoryStore(self.db_path)
        store2.load_session("empty")
        assert len(store2.get_messages()) == 0
        store2.close()

    def test_overwrite_session(self):
        self.store.add_turn("user", "first")
        self.store.save_session("s1")
        self.store.add_turn("user", "second")
        self.store.save_session("s1")

        store2 = MemoryStore(self.db_path)
        store2.load_session("s1")
        # Should have all turns (save appends, doesn't overwrite)
        msgs = store2.get_messages()
        assert len(msgs) >= 2
        store2.close()

    def test_multiple_sessions(self):
        self.store.add_turn("user", "session A")
        self.store.save_session("A")

        self.store.clear_conversation()
        self.store.add_turn("user", "session B")
        self.store.save_session("B")

        store2 = MemoryStore(self.db_path)
        store2.load_session("A")
        msgs = store2.get_messages()
        assert any("session A" in m["content"] for m in msgs)
        store2.close()
