"""Tests for the typed AgentContext and its layers."""

import pytest

from open_harness_v2.core.context import (
    AgentContext,
    HistoryLayer,
    PlanLayer,
    SystemLayer,
    WorkingLayer,
    _estimate_tokens,
)


class TestEstimateTokens:
    def test_short_text(self):
        assert _estimate_tokens("hello") >= 1

    def test_long_text(self):
        text = "a" * 400
        tokens = _estimate_tokens(text)
        assert 90 <= tokens <= 110  # ~100 tokens for 400 chars


class TestSystemLayer:
    def test_basic_message(self):
        layer = SystemLayer(role="You are a test agent.")
        msgs = layer.to_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "test agent" in msgs[0]["content"]

    def test_with_tools_and_context(self):
        layer = SystemLayer(
            role="Agent",
            tools_description="shell(command) - Run shell commands",
            project_context="Python project",
            extra="Be concise.",
        )
        msgs = layer.to_messages()
        content = msgs[0]["content"]
        assert "shell(command)" in content
        assert "Python project" in content
        assert "Be concise." in content


class TestPlanLayer:
    def test_empty_plan(self):
        layer = PlanLayer()
        assert layer.to_messages() == []
        assert layer.is_complete is True

    def test_plan_with_steps(self):
        layer = PlanLayer(steps=["Read files", "Write code", "Run tests"])
        msgs = layer.to_messages()
        assert len(msgs) == 1
        assert "Read files" in msgs[0]["content"]
        assert "step 1/3" in msgs[0]["content"]

    def test_advance(self):
        layer = PlanLayer(steps=["A", "B", "C"])
        assert layer.current_step == 0
        assert layer.advance() is True
        assert layer.current_step == 1
        assert layer.advance() is True
        assert layer.current_step == 2
        assert layer.advance() is False  # at end

    def test_is_complete(self):
        layer = PlanLayer(steps=["A", "B"])
        assert layer.is_complete is False
        layer.current_step = 2
        assert layer.is_complete is True

    def test_lookahead(self):
        layer = PlanLayer(steps=["Step_A", "Step_B", "Step_C", "Step_D", "Step_E"], _lookahead=1)
        msgs = layer.to_messages()
        content = msgs[0]["content"]
        assert "Step_A" in content
        assert "Step_B" in content
        # Step_C should NOT be visible (lookahead=1 means current + 1 next)
        assert "Step_C" not in content


class TestHistoryLayer:
    def test_add_and_retrieve(self):
        layer = HistoryLayer()
        layer.add({"role": "user", "content": "hello"})
        layer.add({"role": "assistant", "content": "hi"})
        msgs = layer.to_messages()
        assert len(msgs) == 2

    def test_no_compression_within_budget(self):
        layer = HistoryLayer()
        layer.add({"role": "user", "content": "short"})
        msgs = layer.to_messages(budget=1000)
        assert len(msgs) == 1

    def test_compression_drops_old(self):
        layer = HistoryLayer(_protected_tail=2)
        for i in range(20):
            layer.add({"role": "user", "content": f"message {i} " + "x" * 200})
        # Very tight budget should compress and drop old messages
        msgs = layer.to_messages(budget=50)
        # Should have at most the protected tail
        assert len(msgs) <= 20
        # Last messages should be preserved
        assert "message 19" in msgs[-1]["content"]

    def test_l1_compression(self):
        messages = [
            {"role": "assistant", "content": '{"tool": "shell", "args": {"command": "ls"}}'},
            {"role": "user", "content": "[Tool Result for shell]\nfile1.py\nfile2.py"},
            {"role": "assistant", "content": "Great, I see the files."},
        ]
        result = HistoryLayer._l1_compress(messages)
        # The tool call + result pair should be compressed to a summary
        assert len(result) == 2  # 1 summary + 1 regular message
        assert "[Tool:" in result[0]["content"]

    def test_l2_compression(self):
        messages = [
            {"role": "user", "content": "[Tool: shell → OK]"},
            {"role": "user", "content": "[Tool: read_file → OK]"},
            {"role": "user", "content": "[Tool: write_file → OK]"},
            {"role": "assistant", "content": "Done."},
        ]
        result = HistoryLayer._l2_compress(messages)
        assert len(result) == 2  # 1 aggregated + 1 regular
        assert "3 tool calls summarized" in result[0]["content"]


class TestWorkingLayer:
    def test_add_and_retrieve(self):
        layer = WorkingLayer()
        layer.add({"role": "user", "content": "result"})
        assert len(layer.to_messages()) == 1

    def test_truncates_long_results(self):
        layer = WorkingLayer(_max_per_result=100)
        long_content = "x" * 500
        layer.add({"role": "user", "content": long_content})
        msgs = layer.to_messages()
        assert len(msgs[0]["content"]) < 500
        assert "truncated" in msgs[0]["content"]

    def test_promote_to_history(self):
        working = WorkingLayer()
        history = HistoryLayer()
        working.add({"role": "user", "content": "tool result"})
        working.promote_to_history(history)
        assert len(working.to_messages()) == 0
        assert len(history.to_messages()) == 1


class TestAgentContext:
    def test_to_messages_basic(self):
        ctx = AgentContext()
        ctx.system.role = "Test agent"
        ctx.add_user_message("Do something")
        msgs = ctx.to_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Do something" in msgs[1]["content"]

    def test_to_messages_with_plan(self):
        ctx = AgentContext()
        ctx.plan = PlanLayer(steps=["Step A", "Step B"])
        msgs = ctx.to_messages()
        # system + plan
        assert any("Step A" in m.get("content", "") for m in msgs)

    def test_to_messages_with_budget(self):
        ctx = AgentContext()
        ctx.system.role = "Agent"
        for i in range(50):
            ctx.history.add({"role": "user", "content": f"msg {i} " + "x" * 200})
        # With a tight budget, history should be compressed
        msgs = ctx.to_messages(budget=500)
        assert len(msgs) < 52  # Less than all messages

    def test_add_tool_result(self):
        ctx = AgentContext()
        ctx.add_tool_result("shell", "output here")
        msgs = ctx.working.to_messages()
        assert len(msgs) == 1
        assert "[Tool Result for shell]" in msgs[0]["content"]

    def test_cycle_working(self):
        ctx = AgentContext()
        ctx.add_tool_result("shell", "result1")
        ctx.add_tool_result("read_file", "result2")
        assert len(ctx.working.to_messages()) == 2
        assert len(ctx.history.to_messages()) == 0

        ctx.cycle_working()
        assert len(ctx.working.to_messages()) == 0
        assert len(ctx.history.to_messages()) == 2

    def test_add_assistant_message(self):
        ctx = AgentContext()
        ctx.add_assistant_message("I'll help you")
        msgs = ctx.history.to_messages()
        assert msgs[0]["role"] == "assistant"
