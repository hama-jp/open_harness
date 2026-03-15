"""Hooks system — event-driven shell commands inspired by Claude Code.

Hooks allow users to run shell commands in response to agent events.
They are configured in ``open_harness.yaml`` or ``.harness/hooks.yaml``.

Hook types:
  - ``pre_goal``: Before the agent starts processing a goal
  - ``post_goal``: After the agent finishes a goal
  - ``pre_tool``: Before a tool executes
  - ``post_tool``: After a tool executes
  - ``on_error``: When the agent encounters an error

Example configuration::

    hooks:
      pre_goal:
        - command: "echo 'Starting: {{goal}}'"
      post_tool:
        - command: "notify-send 'Tool done: {{tool_name}}'"
          match_tools: ["shell", "git_commit"]
      post_goal:
        - command: "./scripts/lint-check.sh"
"""

from open_harness_v2.hooks.engine import HookEngine, HookSpec

__all__ = ["HookEngine", "HookSpec"]
