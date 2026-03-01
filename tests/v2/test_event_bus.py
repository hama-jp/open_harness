"""Tests for the async EventBus."""

import asyncio

import pytest

from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType


@pytest.fixture
def bus():
    return EventBus()


class TestSubscribeAndEmit:
    @pytest.mark.asyncio
    async def test_async_handler(self, bus: EventBus):
        received = []

        async def handler(event: AgentEvent):
            received.append(event)

        bus.subscribe(EventType.AGENT_STARTED, handler)
        ev = AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"})
        await bus.emit(ev)

        assert len(received) == 1
        assert received[0] is ev

    @pytest.mark.asyncio
    async def test_sync_handler(self, bus: EventBus):
        received = []

        def handler(event: AgentEvent):
            received.append(event)

        bus.subscribe(EventType.TOOL_EXECUTED, handler)
        ev = AgentEvent(type=EventType.TOOL_EXECUTED)
        await bus.emit(ev)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_no_cross_delivery(self, bus: EventBus):
        received = []

        async def handler(event: AgentEvent):
            received.append(event)

        bus.subscribe(EventType.AGENT_STARTED, handler)
        await bus.emit(AgentEvent(type=EventType.AGENT_DONE))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, bus: EventBus):
        counts = {"a": 0, "b": 0}

        async def handler_a(event: AgentEvent):
            counts["a"] += 1

        async def handler_b(event: AgentEvent):
            counts["b"] += 1

        bus.subscribe(EventType.LLM_RESPONSE, handler_a)
        bus.subscribe(EventType.LLM_RESPONSE, handler_b)
        await bus.emit(AgentEvent(type=EventType.LLM_RESPONSE))

        assert counts["a"] == 1
        assert counts["b"] == 1


class TestWildcard:
    @pytest.mark.asyncio
    async def test_wildcard_receives_all(self, bus: EventBus):
        received = []

        async def handler(event: AgentEvent):
            received.append(event.type)

        bus.subscribe("*", handler)
        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED))
        await bus.emit(AgentEvent(type=EventType.TOOL_EXECUTED))
        await bus.emit(AgentEvent(type=EventType.LLM_ERROR))

        assert len(received) == 3
        assert EventType.AGENT_STARTED in received
        assert EventType.TOOL_EXECUTED in received

    @pytest.mark.asyncio
    async def test_wildcard_plus_specific(self, bus: EventBus):
        calls = []

        async def specific(event: AgentEvent):
            calls.append("specific")

        async def wildcard(event: AgentEvent):
            calls.append("wildcard")

        bus.subscribe(EventType.AGENT_STARTED, specific)
        bus.subscribe("*", wildcard)
        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED))

        assert "specific" in calls
        assert "wildcard" in calls


class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus: EventBus):
        received = []

        async def handler(event: AgentEvent):
            received.append(event)

        bus.subscribe(EventType.AGENT_DONE, handler)
        await bus.emit(AgentEvent(type=EventType.AGENT_DONE))
        assert len(received) == 1

        bus.unsubscribe(EventType.AGENT_DONE, handler)
        await bus.emit(AgentEvent(type=EventType.AGENT_DONE))
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, bus: EventBus):
        async def handler(event: AgentEvent):
            pass

        # Should not raise
        bus.unsubscribe(EventType.AGENT_DONE, handler)


class TestHistory:
    @pytest.mark.asyncio
    async def test_history_recorded(self, bus: EventBus):
        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED))
        await bus.emit(AgentEvent(type=EventType.AGENT_DONE))

        assert len(bus.history) == 2
        assert bus.history[0].type == EventType.AGENT_STARTED
        assert bus.history[1].type == EventType.AGENT_DONE

    @pytest.mark.asyncio
    async def test_history_limit(self, bus: EventBus):
        bus._max_history = 5
        for i in range(10):
            await bus.emit(AgentEvent(type=EventType.LLM_STREAMING, data={"i": i}))

        assert len(bus.history) == 5

    @pytest.mark.asyncio
    async def test_clear(self, bus: EventBus):
        async def handler(event: AgentEvent):
            pass

        bus.subscribe(EventType.AGENT_STARTED, handler)
        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED))

        bus.clear()
        assert len(bus.history) == 0
        assert len(bus._handlers) == 0


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_handler_exception_does_not_propagate(self, bus: EventBus):
        async def bad_handler(event: AgentEvent):
            raise ValueError("boom")

        received = []

        async def good_handler(event: AgentEvent):
            received.append(event)

        bus.subscribe(EventType.AGENT_STARTED, bad_handler)
        bus.subscribe(EventType.AGENT_STARTED, good_handler)

        # Should not raise
        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED))
        assert len(received) == 1
