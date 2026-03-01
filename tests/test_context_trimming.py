"""Tests for Issue 1: Context Window Management."""

import json

from open_harness.agent import Agent


class TestEstimateTokens:
    def test_basic(self):
        msgs = [{"role": "user", "content": "a" * 400}]
        assert Agent._estimate_tokens(msgs) == 100

    def test_empty(self):
        assert Agent._estimate_tokens([]) == 0

    def test_none_content(self):
        """content=None should not raise TypeError."""
        msgs = [{"role": "assistant", "content": None}]
        assert Agent._estimate_tokens(msgs) == 0

    def test_missing_content_key(self):
        msgs = [{"role": "system"}]
        assert Agent._estimate_tokens(msgs) == 0


class TestTrimMessages:
    def _make_tool_pair(self, tool_name: str, output: str = "OK") -> list[dict]:
        return [
            {"role": "assistant", "content": json.dumps({"tool": tool_name, "args": {}})},
            {"role": "user", "content": f"[Tool Result for {tool_name}]\n{output}"},
        ]

    def test_no_trim_under_limit(self):
        msgs = [{"role": "system", "content": "sys"}]
        result = Agent._trim_messages(msgs, max_tokens=1000)
        # Under limit: should return same list (unmodified)
        assert len(result) == len(msgs)
        assert result[0]["content"] == "sys"

    def test_trims_large_context(self):
        msgs = [{"role": "system", "content": "system prompt"}]
        for i in range(30):
            msgs.extend(self._make_tool_pair("read_file", "x" * 2000))
        msgs.append({"role": "user", "content": "final"})

        trimmed = Agent._trim_messages(msgs, max_tokens=5000)
        assert len(trimmed) < len(msgs)
        # System prompt preserved
        assert trimmed[0]["role"] == "system"
        assert trimmed[0]["content"] == "system prompt"
        # Final message preserved
        assert trimmed[-1]["content"] == "final"

    def test_preserves_tail_messages(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.extend(self._make_tool_pair(f"tool_{i}", "x" * 2000))
        last_msg = {"role": "user", "content": "last question"}
        msgs.append(last_msg)

        trimmed = Agent._trim_messages(msgs, max_tokens=3000)
        # Final message must be preserved
        assert trimmed[-1]["content"] == "last question"
        # At least the last few tool pairs should be untouched (not compressed)
        tail_contents = [m.get("content", "") for m in trimmed[-6:]]
        assert not any("[Earlier:" in c for c in tail_contents)

    def test_compressed_format(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.extend(self._make_tool_pair("read_file", "x" * 2000))
        msgs.append({"role": "user", "content": "end"})

        trimmed = Agent._trim_messages(msgs, max_tokens=3000)
        compressed = [m for m in trimmed if "[Earlier:" in m.get("content", "")]
        assert len(compressed) > 0
        # Check format
        assert "read_file" in compressed[0]["content"]
        assert "success" in compressed[0]["content"] or "fail" in compressed[0]["content"]

    def test_short_list_not_trimmed(self):
        """Lists with <= protected_tail + 1 messages should not be trimmed."""
        msgs = [{"role": "system", "content": "x" * 100000}]
        for i in range(5):
            msgs.append({"role": "user", "content": "x" * 10000})
        result = Agent._trim_messages(msgs, max_tokens=100)
        assert len(result) == len(msgs)

    def test_failed_tool_compressed_as_fail(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.append({"role": "assistant", "content": json.dumps({"tool": "shell", "args": {}})})
            msgs.append({"role": "user", "content": f"[Tool Result for shell]\n[Tool Error] FAIL: command not found\n{'x' * 2000}"})
        msgs.append({"role": "user", "content": "end"})

        trimmed = Agent._trim_messages(msgs, max_tokens=3000)
        compressed = [m for m in trimmed if "[Earlier:" in m.get("content", "")]
        assert len(compressed) > 0
        assert any("fail" in c["content"] for c in compressed)
