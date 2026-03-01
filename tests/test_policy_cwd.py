"""Tests for Issue 7: Shell/Git/Test CWD Policy."""

from open_harness.policy import PolicyConfig, PolicyEngine


def _make_engine(project_root: str = "/home/user/project") -> PolicyEngine:
    pe = PolicyEngine(PolicyConfig())
    pe.set_project_root(project_root)
    return pe


class TestCwdPolicy:
    def test_cwd_inside_project(self):
        pe = _make_engine("/home/user/project")
        v = pe.check("shell", {"command": "ls", "cwd": "/home/user/project/src"})
        assert v is None

    def test_cwd_outside_project(self):
        pe = _make_engine("/home/user/project")
        v = pe.check("shell", {"command": "ls", "cwd": "/tmp"})
        assert v is not None
        assert v.rule == "cwd_outside_project"

    def test_cwd_exact_project_root(self):
        pe = _make_engine("/home/user/project")
        v = pe.check("shell", {"command": "ls", "cwd": "/home/user/project"})
        assert v is None

    def test_no_cwd_no_check(self):
        pe = _make_engine("/home/user/project")
        v = pe.check("shell", {"command": "ls"})
        # No cwd arg — should not trigger cwd check
        assert v is None or v.rule != "cwd_outside_project"

    def test_cwd_applies_to_git_tools(self):
        pe = _make_engine("/home/user/project")
        for tool in ("git_diff", "git_commit", "git_log", "run_tests"):
            v = pe.check(tool, {"cwd": "/etc"})
            assert v is not None, f"{tool} should reject cwd=/etc"
            assert "cwd" in v.rule

    def test_cwd_invalid_type(self):
        pe = _make_engine("/home/user/project")
        v = pe.check("shell", {"command": "ls", "cwd": 12345})
        assert v is not None
        assert v.rule == "cwd_invalid_type"

    def test_empty_cwd_string_no_check(self):
        pe = _make_engine("/home/user/project")
        v = pe.check("shell", {"command": "ls", "cwd": ""})
        # Empty string is falsy, should skip check
        assert v is None or v.rule != "cwd_outside_project"

    def test_no_project_root(self):
        pe = PolicyEngine(PolicyConfig())
        # No project root set — cwd check should pass
        v = pe.check("shell", {"command": "ls", "cwd": "/anywhere"})
        assert v is None or v.rule != "cwd_outside_project"


class TestProjectTreePolicy:
    def test_project_tree_inside_project(self):
        pe = _make_engine("/home/user/project")
        v = pe.check("project_tree", {"path": "/home/user/project/src"})
        assert v is None

    def test_project_tree_outside_project(self):
        pe = _make_engine("/home/user/project")
        v = pe.check("project_tree", {"path": "/etc"})
        assert v is not None
