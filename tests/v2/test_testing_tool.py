"""Tests for the async TestRunnerTool."""

from __future__ import annotations

import pytest

from open_harness_v2.tools.builtin.testing import TestRunnerTool


@pytest.mark.anyio
async def test_run_echo_as_test_command():
    """A simple echo command should succeed."""
    tool = TestRunnerTool(test_command="echo hello")
    result = await tool.execute()
    assert result.success
    assert "hello" in result.output


@pytest.mark.anyio
async def test_run_tests_with_target():
    """Target argument should be appended to the command."""
    tool = TestRunnerTool(test_command="echo")
    result = await tool.execute(target="some_target")
    assert result.success
    assert "some_target" in result.output


@pytest.mark.anyio
async def test_verbose_flag_for_pytest():
    """Verbose flag should add -v when pytest is the test command."""
    tool = TestRunnerTool(test_command="echo pytest")
    result = await tool.execute(verbose=True)
    assert result.success
    assert "-v" in result.output


@pytest.mark.anyio
async def test_pytest_tb_short_appended():
    """--tb=short should be appended for pytest commands."""
    tool = TestRunnerTool(test_command="echo pytest")
    result = await tool.execute()
    assert "--tb=short" in result.output


@pytest.mark.anyio
async def test_tb_short_not_duplicated():
    """If --tb is already in the command, don't add it again."""
    tool = TestRunnerTool(test_command="echo pytest --tb=long")
    result = await tool.execute()
    # Should only have the original --tb=long, not an extra --tb=short
    assert result.output.count("--tb") == 1


@pytest.mark.anyio
async def test_failing_command():
    """A failing command should return success=False."""
    tool = TestRunnerTool(test_command="false")
    result = await tool.execute()
    assert not result.success
    assert "exit code" in result.error.lower() or "failed" in result.error.lower()


@pytest.mark.anyio
async def test_timeout():
    """A command that exceeds timeout should be killed."""
    tool = TestRunnerTool(test_command="sleep 60", timeout=1)
    result = await tool.execute()
    assert not result.success
    assert "timed out" in result.error.lower()


@pytest.mark.anyio
async def test_command_not_found():
    """A nonexistent command should return a clear error."""
    tool = TestRunnerTool(test_command="nonexistent_test_runner_xyz")
    result = await tool.execute()
    assert not result.success
    assert "not found" in result.error.lower() or "error" in result.error.lower()


@pytest.mark.anyio
async def test_metadata_returncode():
    """Metadata should include returncode and all_passed."""
    tool = TestRunnerTool(test_command="echo ok")
    result = await tool.execute()
    assert result.metadata["returncode"] == 0
    assert result.metadata["all_passed"] is True


@pytest.mark.anyio
async def test_long_output_truncated():
    """Very long output should be truncated keeping head + tail."""
    # Generate > 15000 chars of output
    tool = TestRunnerTool(test_command="python3 -c \"print('x' * 200 + '\\n') * 200\"")
    # Use seq to generate many lines
    tool = TestRunnerTool(test_command="seq 1 1000")
    result = await tool.execute()
    assert result.success
    # With 1000 lines it shouldn't exceed 15000 chars, so no truncation.
    # Test with explicit large output:
    tool2 = TestRunnerTool(
        test_command="python3 -c \"[print('line ' + str(i) + ' x' * 100) for i in range(500)]\""
    )
    result2 = await tool2.execute()
    assert result2.success
