"""Tests for Issue 1: Streaming native tool calls."""

from open_harness.llm.client import NativeToolCallAccumulator


class TestNativeToolCallAccumulator:
    def test_empty_accumulator(self):
        acc = NativeToolCallAccumulator()
        assert not acc.has_calls()
        assert acc.finalize() == []

    def test_single_call_single_chunk(self):
        """Complete tool call in one chunk."""
        acc = NativeToolCallAccumulator()
        acc.feed({
            "tool_calls": [{
                "index": 0,
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "test.py"}',
                },
            }],
        })
        assert acc.has_calls()
        calls = acc.finalize()
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "test.py"}

    def test_single_call_multiple_chunks(self):
        """Tool call arguments split across multiple chunks."""
        acc = NativeToolCallAccumulator()

        # First chunk: name + start of arguments
        acc.feed({
            "tool_calls": [{
                "index": 0,
                "function": {
                    "name": "shell",
                    "arguments": '{"comma',
                },
            }],
        })

        # Second chunk: rest of arguments
        acc.feed({
            "tool_calls": [{
                "index": 0,
                "function": {
                    "arguments": 'nd": "ls"}',
                },
            }],
        })

        calls = acc.finalize()
        assert len(calls) == 1
        assert calls[0].name == "shell"
        assert calls[0].arguments == {"command": "ls"}

    def test_multiple_parallel_calls(self):
        """Multiple tool calls with different indices."""
        acc = NativeToolCallAccumulator()

        acc.feed({
            "tool_calls": [
                {"index": 0, "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                {"index": 1, "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}},
            ],
        })

        calls = acc.finalize()
        assert len(calls) == 2
        assert calls[0].name == "read_file"
        assert calls[0].arguments["path"] == "a.py"
        assert calls[1].name == "read_file"
        assert calls[1].arguments["path"] == "b.py"

    def test_no_tool_calls_in_delta(self):
        """Deltas without tool_calls should be ignored."""
        acc = NativeToolCallAccumulator()
        acc.feed({"content": "hello"})
        acc.feed({})
        assert not acc.has_calls()

    def test_malformed_arguments_json(self):
        """Should handle malformed JSON arguments gracefully."""
        acc = NativeToolCallAccumulator()
        acc.feed({
            "tool_calls": [{
                "index": 0,
                "function": {
                    "name": "test_tool",
                    "arguments": "not valid json",
                },
            }],
        })
        calls = acc.finalize()
        assert len(calls) == 1
        assert calls[0].name == "test_tool"
        assert calls[0].arguments == {}

    def test_empty_arguments(self):
        """Should handle empty arguments string."""
        acc = NativeToolCallAccumulator()
        acc.feed({
            "tool_calls": [{
                "index": 0,
                "function": {
                    "name": "no_args_tool",
                    "arguments": "",
                },
            }],
        })
        calls = acc.finalize()
        assert len(calls) == 1
        assert calls[0].arguments == {}
