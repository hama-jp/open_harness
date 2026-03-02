"""Tests for v0.4.5: Mode simplification (chat→plan, submit removed from cycling)."""

from unittest.mock import MagicMock, patch


def test_cli_modes_are_plan_and_goal():
    """CLI mode list should only contain 'plan' and 'goal'."""
    # We test by importing and checking the mode list that gets set
    # in the main() function.  Since the modes are local variables,
    # we verify the constants indirectly through the help text.
    from open_harness.cli import handle_command

    agent = MagicMock()
    config = MagicMock()
    display = MagicMock()

    # Capture the help output
    with patch("open_harness.cli.console") as mock_console:
        handle_command("/help", agent, config, display)
        help_text = mock_console.print.call_args[0][0]

    # plan mode should be present, chat/submit modes should not be
    # listed as cycling modes
    assert "plan" in help_text
    assert "goal" in help_text
    # "chat" should NOT appear as a mode label
    assert "  chat    " not in help_text
    # submit should still be mentioned as a power user command
    assert "/submit" in help_text


def test_cli_mode_colors_and_labels():
    """Mode colors and labels should match the new plan/goal scheme."""
    # The mode definitions are local to main(), but we can verify
    # through the startup message pattern.
    from open_harness.cli import handle_command

    agent = MagicMock()
    config = MagicMock()
    display = MagicMock()

    with patch("open_harness.cli.console") as mock_console:
        handle_command("/help", agent, config, display)
        help_text = mock_console.print.call_args[0][0]

    # Verify plan mode description
    assert "plan" in help_text.lower()
    # Verify goal mode description
    assert "goal" in help_text.lower()


def test_submit_command_still_works():
    """/submit command should still be available even though submit mode is removed."""
    from open_harness.cli import handle_command

    agent = MagicMock()
    config = MagicMock()
    display = MagicMock()

    # /submit without args should print usage
    with patch("open_harness.cli.console") as mock_console:
        result = handle_command("/submit", agent, config, display)
    assert result is True  # handled


def test_tui_modes_are_plan_and_goal():
    """TUI mode list should only contain 'plan' and 'goal'."""
    try:
        from open_harness.tui.app import HarnessApp
    except ImportError:
        import pytest
        pytest.skip("textual not installed")

    config = MagicMock()
    project = MagicMock()
    project.info = {"type": "python", "root": "/tmp/test"}
    agent = MagicMock()
    tools = MagicMock()
    tools.list_tools.return_value = []
    memory = MagicMock()
    task_queue = MagicMock()
    task_store = MagicMock()

    app = HarnessApp(
        config=config,
        project=project,
        agent=agent,
        tools=tools,
        memory=memory,
        task_queue=task_queue,
        task_store=task_store,
        version="0.4.5",
    )

    assert app._modes == ["plan", "goal"]
    assert app.current_mode == "plan"

    # Cycling should go plan → goal → plan
    app._mode_index = 0
    app._mode_index = (app._mode_index + 1) % len(app._modes)
    assert app.current_mode == "goal"
    app._mode_index = (app._mode_index + 1) % len(app._modes)
    assert app.current_mode == "plan"


def test_tui_submit_mode_still_in_run_input():
    """TUI _run_input should still handle force_mode='submit' via task_queue."""
    try:
        from open_harness.tui.app import HarnessApp
    except ImportError:
        import pytest
        pytest.skip("textual not installed")

    config = MagicMock()
    project = MagicMock()
    project.info = {"type": "python", "root": "/tmp/test"}
    project.root = "/tmp/test"
    agent = MagicMock()
    tools = MagicMock()
    tools.list_tools.return_value = []
    memory = MagicMock()
    task_queue = MagicMock()
    task_store = MagicMock()

    app = HarnessApp(
        config=config,
        project=project,
        agent=agent,
        tools=tools,
        memory=memory,
        task_queue=task_queue,
        task_store=task_store,
        version="0.4.5",
    )

    # The submit path in _run_input calls task_queue.submit then
    # queries widgets.  We verify the code path exists by checking
    # that mode == "submit" still triggers task_queue.submit before
    # widget access.
    task_mock = MagicMock()
    task_mock.id = "test-1"
    task_mock.goal = "test goal"
    task_mock.log_path = "/tmp/log"
    app.task_queue.submit.return_value = task_mock

    # Patch query_one to avoid NoMatches error (app isn't mounted)
    with patch.object(app, "query_one"):
        app._run_input("test goal", force_mode="submit")
    app.task_queue.submit.assert_called_once_with("test goal")
