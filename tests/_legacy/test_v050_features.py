"""Tests for v0.5.0 performance improvements."""

import json

from open_harness.agent import ContextManager, _aggregate_l1_run
from open_harness.context_compactor import build_context_summary, _short_path
from open_harness.llm.client import ToolCallParser, _parse_tool_calls_from_text, _try_parse_tool_json
from open_harness.llm.compensator import ErrorClassifier
from open_harness.planner import GoalComplexityEstimator
from open_harness.policy import PolicyEngine, PolicyConfig
from open_harness.tools.base import ToolRegistry, ToolResult, _smart_truncate


# -----------------------------------------------------------------------
# 1.1 Tool output truncation
# -----------------------------------------------------------------------

class TestSmartTruncate:
    def test_short_text_unchanged(self):
        assert _smart_truncate("hello", 100) == "hello"

    def test_truncation_preserves_head_tail(self):
        text = "A" * 500 + "B" * 500
        result = _smart_truncate(text, 200)
        assert result.startswith("A")
        assert result.endswith("B")
        assert "truncated" in result

    def test_truncation_size_limit(self):
        text = "x" * 10000
        result = _smart_truncate(text, 2000)
        # Head (500) + truncation marker + tail (1500) should fit roughly
        assert len(result) < 2100  # some overhead from the marker


# -----------------------------------------------------------------------
# 1.2 Policy glob caching
# -----------------------------------------------------------------------

class TestPolicyCache:
    def test_denied_cache_populated(self):
        engine = PolicyEngine(PolicyConfig(denied_paths=["/etc/*", "/usr/*"]))
        engine.set_project_root("/home/test/project")
        # Check a denied path twice — second should use cache
        from pathlib import Path
        v1 = engine._check_denied("/etc/passwd", Path("/etc/passwd"), "/etc/passwd", "read_file", "read")
        assert v1 is not None
        assert "/etc/passwd" in engine._denied_cache
        # Second check (cached)
        v2 = engine._check_denied("/etc/passwd", Path("/etc/passwd"), "/etc/passwd", "read_file", "read")
        assert v2 is not None

    def test_compiled_denied_patterns(self):
        engine = PolicyEngine(PolicyConfig(denied_paths=["/etc/*", "~/.ssh/*"]))
        assert len(engine._compiled_denied) == 2
        # Each tuple should have (expanded, parent, raw_pattern)
        assert engine._compiled_denied[0][2] == "/etc/*"

    def test_cache_invalidated_on_set_project_root(self):
        engine = PolicyEngine()
        engine._denied_cache["foo"] = True
        engine.set_project_root("/home/test")
        assert len(engine._denied_cache) == 0


# -----------------------------------------------------------------------
# 2.1 Rolling token management
# -----------------------------------------------------------------------

class TestContextManager:
    def test_adaptive_threshold_default(self):
        assert ContextManager.adaptive_threshold(0) == 12000

    def test_adaptive_threshold_32k(self):
        threshold = ContextManager.adaptive_threshold(32768)
        assert threshold == int(32768 * 0.75 / 4)

    def test_adaptive_threshold_minimum(self):
        assert ContextManager.adaptive_threshold(1000) == 4000

    def test_l2_aggregation(self):
        run = [
            {"role": "user", "content": "[Earlier: used shell → success]", "_l1": True},
            {"role": "user", "content": "[Earlier: used read_file → success]", "_l1": True},
            {"role": "user", "content": "[Earlier: used shell → fail]", "_l1": True},
        ]
        result = _aggregate_l1_run(run)
        assert "3 tool calls" in result["content"]
        assert "2 succeeded" in result["content"]

    def test_l2_single_entry(self):
        run = [
            {"role": "user", "content": "[Earlier: used shell → success]", "_l1": True},
        ]
        result = _aggregate_l1_run(run)
        # Single entry should not be aggregated
        assert "used shell" in result["content"]


# -----------------------------------------------------------------------
# 2.2 Schema-first parser
# -----------------------------------------------------------------------

class TestToolCallParser:
    def test_known_tool_fast_path(self):
        parser = ToolCallParser(["shell", "read_file", "write_file"])
        text = '{"tool": "shell", "args": {"command": "ls"}}'
        calls = parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "shell"

    def test_unknown_tool_fallback(self):
        parser = ToolCallParser(["shell"])
        text = '{"tool": "unknown_tool", "args": {}}'
        calls = parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "unknown_tool"

    def test_empty_tool_names(self):
        parser = ToolCallParser()
        text = '{"tool": "shell", "args": {"command": "ls"}}'
        calls = parser.parse(text)
        assert len(calls) == 1

    def test_try_parse_tool_json_repair(self):
        raw = '```json\n{"tool": "shell", "args": {"command": "ls"}}\n```'
        call = _try_parse_tool_json(raw)
        assert call is not None
        assert call.name == "shell"


# -----------------------------------------------------------------------
# 2.3 Error classifier
# -----------------------------------------------------------------------

class TestErrorClassifier:
    def test_empty_response(self):
        ec = ErrorClassifier(["shell", "read_file"])
        assert ec.classify("error", "") == "empty_response"
        assert ec.classify("error", "   ") == "empty_response"

    def test_malformed_json(self):
        ec = ErrorClassifier(["shell"])
        assert ec.classify("parse error", '{"tool": "shell", "args": {') == "malformed_json"

    def test_wrong_tool_name(self):
        ec = ErrorClassifier(["shell", "read_file"])
        assert ec.classify("Unknown tool: shel", '{"tool": "shel"}') == "wrong_tool_name"

    def test_prose_wrapped(self):
        ec = ErrorClassifier(["shell"])
        text = 'I will run: {"tool": "shell", "args": {"command": "ls"}}'
        assert ec.classify("parse error", text) == "prose_wrapped"

    def test_suggest_tool_fuzzy(self):
        ec = ErrorClassifier(["shell", "read_file", "write_file"])
        assert ec.suggest_tool("read") == "read_file"
        assert ec.suggest_tool("write") == "write_file"


# -----------------------------------------------------------------------
# 2.5 System prompt compression
# -----------------------------------------------------------------------

class TestSystemPromptCompression:
    def test_small_tier_minimal_prompt(self):
        from open_harness.llm.compensator import build_autonomous_prompt
        prompt = build_autonomous_prompt(
            "tool descriptions",
            "project context",
            tier="small",
        )
        # Small tier should NOT include routing guide or work patterns
        assert "Routing Guide" not in prompt
        assert "Orchestrator Style" not in prompt
        assert "tool descriptions" in prompt

    def test_medium_tier_full_prompt(self):
        from open_harness.llm.compensator import build_autonomous_prompt
        prompt = build_autonomous_prompt(
            "tool descriptions",
            "project context",
            tier="medium",
        )
        assert "tool descriptions" in prompt


# -----------------------------------------------------------------------
# 3.1 Context summarization
# -----------------------------------------------------------------------

class TestContextCompactor:
    def test_small_batch_returns_none(self):
        msgs = [{"role": "user", "content": "small"}]
        assert build_context_summary(msgs) is None

    def test_extracts_file_modifications(self):
        msgs = [
            {"role": "assistant", "content": json.dumps({"tool": "write_file", "args": {"path": "src/main.py"}})},
            {"role": "user", "content": "[Tool Result for write_file]\n" + "x" * 20000},
        ] * 2
        summary = build_context_summary(msgs)
        assert summary is not None
        assert "main.py" in summary

    def test_short_path(self):
        assert _short_path("a/b/c/d/e.py") == "c/d/e.py"
        assert _short_path("a/b.py") == "a/b.py"


# -----------------------------------------------------------------------
# 3.2 Complexity-adaptive planning
# -----------------------------------------------------------------------

class TestGoalComplexity:
    def test_simple_goal_is_low(self):
        assert GoalComplexityEstimator.estimate("fix typo") == "low"

    def test_medium_goal(self):
        assert GoalComplexityEstimator.estimate(
            "implement a new feature to add user authentication") == "medium"

    def test_complex_goal_is_high(self):
        assert GoalComplexityEstimator.estimate(
            "refactor the entire database schema and migrate all data, "
            "then redesign the architecture for microservices") == "high"

    def test_long_goal_is_high(self):
        assert GoalComplexityEstimator.estimate("word " * 150) == "high"

    def test_profile_low(self):
        p = GoalComplexityEstimator.get_profile("low")
        assert p["max_steps"] == 3
        assert p["max_agent_steps"] == 8
        assert p["replan_depth"] == 0

    def test_profile_high(self):
        p = GoalComplexityEstimator.get_profile("high")
        assert p["max_steps"] == 8
        assert p["max_agent_steps"] == 15
        assert p["replan_depth"] == 2


# -----------------------------------------------------------------------
# 4.1 Checkpoint branch lifecycle guarantees
# -----------------------------------------------------------------------

import subprocess
import tempfile
import os

class TestCheckpointBranchLifecycle:
    """Verify that checkpoint branches are always properly merged or discarded."""

    def _git(self, args, cwd):
        return subprocess.run(
            ["git"] + args, capture_output=True, text=True, cwd=cwd,
        )

    def _init_repo(self, tmp):
        """Create a minimal git repo with one commit."""
        self._git(["init"], tmp)
        self._git(["config", "user.email", "test@test.com"], tmp)
        self._git(["config", "user.name", "Test"], tmp)
        f = os.path.join(tmp, "file.txt")
        with open(f, "w") as fh:
            fh.write("initial\n")
        self._git(["add", "-A"], tmp)
        self._git(["commit", "-m", "initial"], tmp)

    def test_finish_auto_commits_squash_merge(self):
        """After finish(keep_changes=True), changes must be committed, not just staged."""
        from open_harness.checkpoint import CheckpointEngine
        with tempfile.TemporaryDirectory() as tmp:
            self._init_repo(tmp)
            ckpt = CheckpointEngine(tmp, has_git=True)
            ckpt.begin()

            # Make a change and snapshot
            f = os.path.join(tmp, "file.txt")
            with open(f, "w") as fh:
                fh.write("changed\n")
            ckpt.snapshot("test change")

            ckpt.finish(keep_changes=True)

            # Working tree should be clean (no staged-but-uncommitted changes)
            status = self._git(["status", "--porcelain"], tmp)
            assert status.stdout.strip() == "", f"Dirty working tree: {status.stdout}"

            # The commit message should contain "harness: goal completed"
            log = self._git(["log", "-1", "--format=%s"], tmp)
            assert "harness" in log.stdout.lower()

    def test_finish_keep_changes_no_snapshots_with_diff(self):
        """keep_changes=True with no snapshots but actual changes should still merge."""
        from open_harness.checkpoint import CheckpointEngine
        with tempfile.TemporaryDirectory() as tmp:
            self._init_repo(tmp)
            ckpt = CheckpointEngine(tmp, has_git=True)
            ckpt.begin()

            # Make a change but DON'T snapshot
            f = os.path.join(tmp, "file.txt")
            with open(f, "w") as fh:
                fh.write("changed without snapshot\n")

            ckpt.finish(keep_changes=True)

            # The change should be merged (not discarded)
            with open(f) as fh:
                content = fh.read()
            assert "changed without snapshot" in content

    def test_finish_discard_removes_branch(self):
        """finish(keep_changes=False) must delete work branch entirely."""
        from open_harness.checkpoint import CheckpointEngine
        with tempfile.TemporaryDirectory() as tmp:
            self._init_repo(tmp)
            ckpt = CheckpointEngine(tmp, has_git=True)
            ckpt.begin()
            work_branch = ckpt._work_branch

            f = os.path.join(tmp, "file.txt")
            with open(f, "w") as fh:
                fh.write("should be discarded\n")
            ckpt.snapshot("will be discarded")

            ckpt.finish(keep_changes=False)

            # Work branch should not exist
            branches = self._git(["branch", "--list", "harness/goal-*"], tmp)
            assert work_branch not in branches.stdout

            # Changes should be reverted
            with open(f) as fh:
                assert fh.read() == "initial\n"

    def test_cleanup_orphan_branches(self):
        """cleanup_orphan_branches removes leftover harness/goal-* branches."""
        from open_harness.checkpoint import CheckpointEngine
        with tempfile.TemporaryDirectory() as tmp:
            self._init_repo(tmp)

            # Simulate orphan branches from crashed sessions
            self._git(["branch", "harness/goal-111111"], tmp)
            self._git(["branch", "harness/goal-222222"], tmp)

            cleaned = CheckpointEngine.cleanup_orphan_branches(tmp)
            assert len(cleaned) == 2

            branches = self._git(["branch", "--list", "harness/goal-*"], tmp)
            assert branches.stdout.strip() == ""

    def test_cleanup_orphan_skips_current_branch(self):
        """cleanup_orphan_branches must not delete the current branch."""
        from open_harness.checkpoint import CheckpointEngine
        with tempfile.TemporaryDirectory() as tmp:
            self._init_repo(tmp)

            # Create and switch to a harness branch (simulates being mid-crash)
            self._git(["checkout", "-b", "harness/goal-333333"], tmp)

            cleaned = CheckpointEngine.cleanup_orphan_branches(tmp)
            assert len(cleaned) == 0

            # Branch should still exist
            branches = self._git(["branch", "--list", "harness/goal-*"], tmp)
            assert "harness/goal-333333" in branches.stdout
