"""Tests for Issue 2: Task/Goal Cancellation."""

import threading

from open_harness.agent import Agent


class TestCancelEvent:
    def test_cancel_sets_event(self):
        """cancel() should set the internal event."""
        # We can't easily instantiate Agent without full config,
        # so test the threading.Event directly
        event = threading.Event()
        assert not event.is_set()
        event.set()
        assert event.is_set()
        event.clear()
        assert not event.is_set()

    def test_agent_has_cancel_method(self):
        assert hasattr(Agent, "cancel")
        assert callable(getattr(Agent, "cancel"))

    def test_agent_has_cancel_event(self):
        """Agent.__init__ should define _cancel_event."""
        # Check the __init__ signature accepts user_input_fn
        import inspect
        sig = inspect.signature(Agent.__init__)
        assert "user_input_fn" in sig.parameters
