"""Tests for hooks system."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import yaml

from open_harness_v2.hooks.engine import (
    HookEngine,
    HookSpec,
    HooksConfig,
    load_hooks,
    parse_hooks_config,
)
from open_harness_v2.events.bus import EventBus
from open_harness_v2.types import AgentEvent, EventType


class TestHookSpec:
    def test_matches_tool_no_filter(self):
        spec = HookSpec(command="echo hi")
        assert spec.matches_tool("shell") is True
        assert spec.matches_tool("anything") is True

    def test_matches_tool_with_filter(self):
        spec = HookSpec(command="echo hi", match_tools=["shell", "git_commit"])
        assert spec.matches_tool("shell") is True
        assert spec.matches_tool("git_commit") is True
        assert spec.matches_tool("read_file") is False


class TestParseHooksConfig:
    def test_parse_empty(self):
        config = parse_hooks_config(None)
        assert config.pre_goal == []
        assert config.post_goal == []

    def test_parse_string_hooks(self):
        raw = {
            "pre_goal": ["echo start"],
            "post_goal": ["echo done"],
        }
        config = parse_hooks_config(raw)
        assert len(config.pre_goal) == 1
        assert config.pre_goal[0].command == "echo start"

    def test_parse_dict_hooks(self):
        raw = {
            "post_tool": [
                {
                    "command": "notify-send done",
                    "match_tools": ["shell"],
                    "timeout": 10,
                }
            ],
        }
        config = parse_hooks_config(raw)
        assert len(config.post_tool) == 1
        assert config.post_tool[0].match_tools == ["shell"]
        assert config.post_tool[0].timeout == 10


class TestLoadHooks:
    def test_load_from_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            hooks_dir = Path(tmp) / ".harness"
            hooks_dir.mkdir()
            hooks_file = hooks_dir / "hooks.yaml"
            hooks_file.write_text(yaml.dump({
                "pre_goal": ["echo starting"],
            }))
            config = load_hooks(project_root=Path(tmp))
            assert len(config.pre_goal) == 1

    def test_load_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_hooks(project_root=Path(tmp))
            assert config.pre_goal == []


class TestHookEngine:
    def test_has_hooks(self):
        engine = HookEngine(HooksConfig())
        assert engine.has_hooks is False

        engine2 = HookEngine(HooksConfig(pre_goal=[HookSpec(command="echo hi")]))
        assert engine2.has_hooks is True

    @pytest.mark.asyncio
    async def test_hook_runs_on_event(self):
        """Hook should execute when the matching event is emitted."""
        config = HooksConfig(
            pre_goal=[HookSpec(command="echo hook_fired")],
        )
        engine = HookEngine(config)
        bus = EventBus()
        engine.attach(bus)

        # Emit agent.started — should trigger pre_goal hooks
        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"}))
        # No exception = success (hooks run fire-and-forget)

    @pytest.mark.asyncio
    async def test_hook_template_substitution(self):
        """Template variables like {{goal}} should be replaced."""
        marker_file = tempfile.mktemp()
        config = HooksConfig(
            pre_goal=[HookSpec(command=f"echo '{{{{goal}}}}' > {marker_file}")],
        )
        engine = HookEngine(config)
        bus = EventBus()
        engine.attach(bus)

        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "my_goal"}))
        # Give async subprocess time to complete
        await asyncio.sleep(0.5)

        content = Path(marker_file).read_text().strip()
        assert content == "my_goal"
        Path(marker_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_propagate(self):
        """A failing hook should not crash the system."""
        config = HooksConfig(
            pre_goal=[HookSpec(command="exit 1")],
        )
        engine = HookEngine(config)
        bus = EventBus()
        engine.attach(bus)

        # Should not raise
        await bus.emit(AgentEvent(type=EventType.AGENT_STARTED, data={"goal": "test"}))
