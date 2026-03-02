"""Tests for Issue 4: Session persistence deduplication."""

import os
import tempfile

from open_harness.memory.store import MemoryStore


class TestSessionDedup:
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

    def test_save_twice_no_duplicates(self):
        """Saving the same session twice should not duplicate rows."""
        self.store.add_turn("user", "hello")
        self.store.add_turn("assistant", "hi")
        self.store.save_session("s1")
        self.store.save_session("s1")

        store2 = MemoryStore(self.db_path)
        store2.load_session("s1")
        msgs = store2.get_messages()
        # Should have exactly 2 messages, not 4
        assert len(msgs) == 2
        store2.close()

    def test_save_replaces_content(self):
        """Saving again after adding more turns should replace old data."""
        self.store.add_turn("user", "first")
        self.store.save_session("s1")

        self.store.add_turn("user", "second")
        self.store.save_session("s1")

        store2 = MemoryStore(self.db_path)
        store2.load_session("s1")
        msgs = store2.get_messages()
        # Should have the updated conversation (2 turns), not duplicated (3)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"
        store2.close()

    def test_different_sessions_independent(self):
        """Different session IDs should not interfere."""
        self.store.add_turn("user", "session A")
        self.store.save_session("A")

        self.store.clear_conversation()
        self.store.add_turn("user", "session B")
        self.store.save_session("B")

        store2 = MemoryStore(self.db_path)
        store2.load_session("A")
        msgs_a = store2.get_messages()
        assert len(msgs_a) == 1
        assert msgs_a[0]["content"] == "session A"

        store2.clear_conversation()
        store2.load_session("B")
        msgs_b = store2.get_messages()
        assert len(msgs_b) == 1
        assert msgs_b[0]["content"] == "session B"
        store2.close()

    def test_multiple_saves_same_session(self):
        """Multiple saves should always result in current state only."""
        for i in range(5):
            self.store.add_turn("user", f"msg-{i}")
            self.store.save_session("repeated")

        store2 = MemoryStore(self.db_path)
        store2.load_session("repeated")
        msgs = store2.get_messages()
        # Should have exactly 5 messages (the current conversation), not 15
        assert len(msgs) == 5
        store2.close()
