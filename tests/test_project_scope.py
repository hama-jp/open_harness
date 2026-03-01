"""Tests for Issue 5: Project-scope read enforcement."""

from pathlib import Path

from open_harness.policy import PolicyConfig, PolicyEngine


class TestProjectScope:
    def test_default_restricts_reads_to_project(self):
        """With project_scope_default=True, reads outside project should be blocked."""
        config = PolicyConfig(project_scope_default=True)
        engine = PolicyEngine(config)
        engine.set_project_root("/home/user/myproject")

        # Inside project should be allowed
        v = engine.check("read_file", {"path": "/home/user/myproject/src/main.py"})
        assert v is None

        # Outside project should be blocked
        v = engine.check("read_file", {"path": "/tmp/other_file.txt"})
        assert v is not None
        assert v.rule == "allowed_paths"

    def test_project_scope_false_allows_all_reads(self):
        """With project_scope_default=False, all reads are allowed."""
        config = PolicyConfig(project_scope_default=False)
        engine = PolicyEngine(config)
        engine.set_project_root("/home/user/myproject")

        # Outside project should still be allowed
        v = engine.check("read_file", {"path": "/tmp/other_file.txt"})
        # Should pass (not in denied_paths and no allowed_paths restriction)
        assert v is None

    def test_denied_paths_still_apply(self):
        """Denied paths should still block even with project_scope_default."""
        config = PolicyConfig(project_scope_default=True)
        engine = PolicyEngine(config)
        engine.set_project_root("/home/user/myproject")

        # /etc/* is in default denied_paths
        v = engine.check("read_file", {"path": "/etc/passwd"})
        assert v is not None
        assert v.rule == "denied_path"

    def test_explicit_allowed_paths_override_default(self):
        """If allowed_paths is already set, project_scope_default doesn't add more."""
        config = PolicyConfig(
            project_scope_default=True,
            allowed_paths=["/custom/path/*"],
        )
        engine = PolicyEngine(config)
        engine.set_project_root("/home/user/myproject")

        # allowed_paths was already set, so project root NOT auto-added
        # But project root is always checked in _check_read_path
        v = engine.check("read_file", {"path": "/home/user/myproject/file.py"})
        assert v is None  # project root check still passes

    def test_project_scope_auto_adds_glob(self):
        """set_project_root should add project root glob to allowed_paths."""
        config = PolicyConfig(project_scope_default=True)
        engine = PolicyEngine(config)
        assert config.allowed_paths == []

        engine.set_project_root("/home/user/myproject")
        assert len(config.allowed_paths) == 1
        assert "/home/user/myproject" in config.allowed_paths[0]
