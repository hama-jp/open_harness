"""Tests for Issue 12: Tool Call Recovery."""

from open_harness.llm.client import (
    ToolCall,
    _extract_balanced_json,
    _parse_tool_calls_from_text,
)


class TestExtractBalancedJson:
    def test_simple_object(self):
        text = '{"key": "value"}'
        result = _extract_balanced_json(text, 0)
        assert result == '{"key": "value"}'

    def test_nested_object(self):
        text = '{"tool": "read", "args": {"path": "/tmp/x"}}'
        result = _extract_balanced_json(text, 0)
        assert result == text

    def test_with_prefix(self):
        text = 'Some text {"tool": "read"} after'
        idx = text.index("{")
        result = _extract_balanced_json(text, idx)
        assert result == '{"tool": "read"}'

    def test_escaped_braces(self):
        text = '{"content": "a \\"b\\" c"}'
        result = _extract_balanced_json(text, 0)
        assert result is not None

    def test_not_json(self):
        result = _extract_balanced_json("hello", 0)
        assert result is None

    def test_unbalanced(self):
        result = _extract_balanced_json('{"key": "value"', 0)
        assert result is None


class TestParseToolCallsFromText:
    def test_fenced_json(self):
        text = '```json\n{"tool": "read_file", "args": {"path": "/tmp/x"}}\n```'
        calls = _parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments["path"] == "/tmp/x"

    def test_bare_json(self):
        text = 'Let me read the file. {"tool": "read_file", "args": {"path": "/tmp/x"}}'
        calls = _parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"

    def test_whole_text_json(self):
        text = '{"tool": "shell", "args": {"command": "ls"}}'
        calls = _parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "shell"

    def test_tool_call_format(self):
        text = '{"tool_call": {"name": "write_file", "arguments": {"path": "/tmp/x", "content": "hi"}}}'
        calls = _parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "write_file"

    def test_no_tool_call(self):
        text = "I don't know how to help with that."
        calls = _parse_tool_calls_from_text(text)
        assert len(calls) == 0

    def test_string_args_converted(self):
        text = '{"tool": "shell", "args": "ls -la"}'
        calls = _parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert isinstance(calls[0].arguments, dict)

    def test_multiple_calls(self):
        text = (
            '{"tool": "read_file", "args": {"path": "/a"}}\n'
            '{"tool": "read_file", "args": {"path": "/b"}}'
        )
        calls = _parse_tool_calls_from_text(text)
        assert len(calls) == 2

    def test_malformed_json(self):
        text = '{"tool": broken json'
        calls = _parse_tool_calls_from_text(text)
        assert len(calls) == 0
