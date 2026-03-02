"""Tests for the v2 CLI entry point."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from click.testing import CliRunner

from open_harness_v2.cli import main
from open_harness_v2.config import HarnessConfig


@pytest.fixture
def runner():
    return CliRunner()


def _mock_orchestrator():
    """Create a mock orchestrator with all attributes the CLI expects."""
    mock = MagicMock()
    mock.run = AsyncMock(return_value="answer")
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    mock._router = MagicMock()
    mock._router.get_client.return_value = mock_client
    mock._registry = MagicMock()

    # Memory mocks
    mock_pmem = MagicMock()
    mock_pmem.build_context_block = AsyncMock(return_value="")
    mock._project_memory = mock_pmem

    mock_smem = MagicMock()
    mock_smem.load = AsyncMock(return_value=[])
    mock_smem.save = AsyncMock()
    mock._session_memory = mock_smem

    mock_store = MagicMock()
    mock_store.close = AsyncMock()
    mock._memory_store = mock_store

    # Task queue mocks
    mock_task_manager = MagicMock()
    mock_task_manager.shutdown = AsyncMock()
    mock._task_manager = mock_task_manager
    mock_task_store = MagicMock()
    mock_task_store.close = AsyncMock()
    mock._task_store = mock_task_store

    return mock


def test_help(runner):
    """--help should exit 0 and show usage."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Open Harness v2" in result.output
    assert "--config" in result.output
    assert "--profile" in result.output
    assert "--verbose" in result.output


def test_version(runner):
    """--version should show version string."""
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "harness2" in result.output


@patch("open_harness_v2.cli._build_components")
def test_one_shot_calls_orchestrator(mock_build, runner):
    """One-shot mode should call orchestrator.run() with the goal."""
    mock_orchestrator = _mock_orchestrator()

    from rich.console import Console
    from io import StringIO
    mock_console = Console(file=StringIO())

    mock_build.return_value = (
        mock_orchestrator,
        MagicMock(),  # event_bus
        MagicMock(),  # renderer
        mock_console,
        HarnessConfig(),
    )

    result = runner.invoke(main, ["What is 2+2?"])
    assert result.exit_code == 0
    mock_orchestrator.run.assert_called_once_with("What is 2+2?")


@patch("open_harness_v2.cli._build_components")
def test_config_and_profile_passed(mock_build, runner):
    """Config and profile options should be forwarded to _build_components."""
    mock_orchestrator = _mock_orchestrator()

    from rich.console import Console
    from io import StringIO
    mock_console = Console(file=StringIO())

    mock_build.return_value = (
        mock_orchestrator,
        MagicMock(),
        MagicMock(),
        mock_console,
        HarnessConfig(),
    )

    result = runner.invoke(main, [
        "--config", "/tmp/test.yaml",
        "--profile", "api",
        "hello",
    ])
    assert result.exit_code == 0
    mock_build.assert_called_once_with("/tmp/test.yaml", "api", False)


@patch("open_harness_v2.cli._build_components")
def test_verbose_flag(mock_build, runner):
    """Verbose flag should be passed through."""
    mock_orchestrator = _mock_orchestrator()

    from rich.console import Console
    from io import StringIO
    mock_console = Console(file=StringIO())

    mock_build.return_value = (
        mock_orchestrator,
        MagicMock(),
        MagicMock(),
        mock_console,
        HarnessConfig(),
    )

    result = runner.invoke(main, ["-v", "test"])
    assert result.exit_code == 0
    mock_build.assert_called_once_with(None, None, True)
