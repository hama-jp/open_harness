"""Tests for the async external agent tools (Codex, Claude Code, Gemini CLI)."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock

from open_harness_v2.tools.builtin.external import (
    CodexTool,
    ClaudeCodeTool,
    GeminiCliTool,
    _run_external,
)
from open_harness_v2.types import ToolResult


# ------------------------------------------------------------------
# _run_external helper
# ------------------------------------------------------------------

@pytest.mark.anyio
async def test_run_external_success():
    """A simple echo command should succeed."""
    result = await _run_external(["echo", "hello"], tool_label="test")
    assert result.success
    assert "hello" in result.output


@pytest.mark.anyio
async def test_run_external_failure():
    """A failing command should return success=False."""
    result = await _run_external(["false"], tool_label="test")
    assert not result.success


@pytest.mark.anyio
async def test_run_external_timeout():
    """A command exceeding timeout should be killed."""
    result = await _run_external(
        ["sleep", "60"], timeout=1, tool_label="test"
    )
    assert not result.success
    assert "timed out" in result.error.lower()


@pytest.mark.anyio
async def test_run_external_command_not_found():
    """A nonexistent command should return clear error."""
    result = await _run_external(
        ["nonexistent_cmd_xyz_999"], tool_label="test"
    )
    assert not result.success
    assert "not found" in result.error.lower()


@pytest.mark.anyio
async def test_run_external_stderr():
    """stderr should be captured."""
    result = await _run_external(
        ["python3", "-c", "import sys; sys.stderr.write('warn\\n')"],
        tool_label="test",
    )
    assert "warn" in result.output


# ------------------------------------------------------------------
# CodexTool
# ------------------------------------------------------------------

class TestCodexTool:
    def test_name(self):
        assert CodexTool().name == "codex"

    @pytest.mark.anyio
    async def test_not_available(self):
        tool = CodexTool(command="nonexistent_codex_xyz")
        result = await tool.execute(prompt="test")
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.anyio
    async def test_empty_prompt(self):
        tool = CodexTool()
        tool._available = True  # skip PATH check
        result = await tool.execute(prompt="")
        assert not result.success
        assert "no prompt" in result.error.lower()

    @pytest.mark.anyio
    async def test_available_property_caches(self):
        tool = CodexTool(command="nonexistent_codex_xyz")
        assert tool.available is False
        assert tool._available is False
        # Second access uses cache
        assert tool.available is False


# ------------------------------------------------------------------
# ClaudeCodeTool
# ------------------------------------------------------------------

class TestClaudeCodeTool:
    def test_name(self):
        assert ClaudeCodeTool().name == "claude_code"

    @pytest.mark.anyio
    async def test_not_available(self):
        tool = ClaudeCodeTool(command="nonexistent_claude_xyz")
        result = await tool.execute(prompt="test")
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.anyio
    async def test_empty_prompt(self):
        tool = ClaudeCodeTool()
        tool._available = True
        result = await tool.execute(prompt="")
        assert not result.success
        assert "no prompt" in result.error.lower()


# ------------------------------------------------------------------
# GeminiCliTool
# ------------------------------------------------------------------

class TestGeminiCliTool:
    def test_name(self):
        assert GeminiCliTool().name == "gemini_cli"

    @pytest.mark.anyio
    async def test_not_available(self):
        tool = GeminiCliTool(command="nonexistent_gemini_xyz")
        result = await tool.execute(prompt="test")
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.anyio
    async def test_empty_prompt(self):
        tool = GeminiCliTool()
        tool._available = True
        result = await tool.execute(prompt="")
        assert not result.success
        assert "no prompt" in result.error.lower()


# ------------------------------------------------------------------
# Integration: tool with real echo command
# ------------------------------------------------------------------

@pytest.mark.anyio
async def test_codex_with_echo():
    """Use echo as a stand-in for the codex command."""
    tool = CodexTool(command="echo")
    result = await tool.execute(prompt="test prompt")
    assert result.success
    # echo receives: exec --full-auto "test prompt"
    assert "exec" in result.output
    assert "test prompt" in result.output


@pytest.mark.anyio
async def test_gemini_with_echo():
    """Use echo as a stand-in for the gemini command."""
    tool = GeminiCliTool(command="echo")
    result = await tool.execute(prompt="test prompt")
    assert result.success
    assert "test prompt" in result.output
    assert "-y" in result.output
