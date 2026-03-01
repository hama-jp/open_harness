"""Tests for the v2 tool registry and base Tool class."""

from __future__ import annotations

from typing import Any

from open_harness_v2.tools.base import Tool
from open_harness_v2.tools.registry import ToolRegistry, _smart_truncate
from open_harness_v2.types import ToolParameter, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class EchoTool(Tool):
    """Simple mock tool for testing."""

    name = "echo"
    description = "Echoes the input message."
    parameters = [
        ToolParameter(name="message", type="string", description="Message to echo"),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        msg = kwargs.get("message", "")
        return ToolResult(success=True, output=f"Echo: {msg}")


class FailTool(Tool):
    """Tool that always raises."""

    name = "fail"
    description = "Always fails."
    parameters: list[ToolParameter] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("intentional failure")


class BigOutputTool(Tool):
    """Tool that produces large output for truncation testing."""

    name = "big_output"
    description = "Produces a lot of output."
    max_output = 100
    parameters: list[ToolParameter] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="x" * 500)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSmartTruncate:
    def test_no_truncation_when_short(self):
        assert _smart_truncate("hello", 100) == "hello"

    def test_truncation_preserves_head_and_tail(self):
        text = "A" * 100 + "B" * 100  # 200 chars
        result = _smart_truncate(text, 100)
        assert len(result) > 90  # contains head + tail + separator
        assert result.startswith("A" * 25)  # head is 25% of 100
        assert result.endswith("B" * 75)    # tail is 75% of 100
        assert "truncated" in result

    def test_exact_length_not_truncated(self):
        text = "x" * 100
        assert _smart_truncate(text, 100) == text


class TestToolBase:
    def test_to_openai_schema(self):
        tool = EchoTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert "message" in schema["function"]["parameters"]["properties"]
        assert "message" in schema["function"]["parameters"]["required"]

    def test_to_prompt_description(self):
        tool = EchoTool()
        desc = tool.to_prompt_description()
        assert "### echo" in desc
        assert "message" in desc
        assert "required" in desc

    def test_to_compact_description(self):
        tool = EchoTool()
        desc = tool.to_compact_description()
        assert desc.startswith("echo(")
        assert "message: string" in desc

    def test_optional_param_marker(self):
        class OptTool(Tool):
            name = "opt"
            description = "Has optional param"
            parameters = [
                ToolParameter(name="x", type="string", description="required"),
                ToolParameter(name="y", type="integer", description="optional", required=False, default=5),
            ]
            async def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, output="")

        tool = OptTool()
        compact = tool.to_compact_description()
        assert "y: integer?" in compact
        schema = tool.to_openai_schema()
        assert "y" not in schema["function"]["parameters"]["required"]
        assert schema["function"]["parameters"]["properties"]["y"]["default"] == 5


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = EchoTool()
        reg.register(tool)
        assert reg.get("echo") is tool
        assert reg.get("nonexistent") is None

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        tools = reg.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_tool_names(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        assert reg.tool_names() == ["echo"]

    async def test_execute_success(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        result = await reg.execute("echo", {"message": "hello"})
        assert result.success
        assert result.output == "Echo: hello"

    async def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = await reg.execute("unknown", {})
        assert not result.success
        assert "Unknown tool" in result.error

    async def test_execute_catches_exception(self):
        reg = ToolRegistry()
        reg.register(FailTool())
        result = await reg.execute("fail", {})
        assert not result.success
        assert "intentional failure" in result.error

    async def test_execute_truncates_output(self):
        reg = ToolRegistry()
        reg.register(BigOutputTool())
        result = await reg.execute("big_output", {})
        assert result.success
        assert len(result.output) < 500  # was 500 chars, should be truncated
        assert "truncated" in result.output

    def test_get_openai_schemas(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        schemas = reg.get_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "echo"

    def test_get_prompt_description(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        desc = reg.get_prompt_description()
        assert "### echo" in desc

    def test_get_compact_prompt_description(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        desc = reg.get_compact_prompt_description()
        assert "echo(" in desc
