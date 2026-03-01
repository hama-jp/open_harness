"""Tests for Issue 8: Shell environment isolation."""

import os

from open_harness.tools.shell import ShellTool, _SENSITIVE_PREFIXES, _SENSITIVE_NAMES


class TestShellEnvIsolation:
    def setup_method(self):
        self.tool = ShellTool()

    def test_safe_env_strips_aws_keys(self):
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret"
        os.environ["AWS_ACCESS_KEY_ID"] = "test-key-id"
        try:
            env = self.tool._build_safe_env()
            assert "AWS_SECRET_ACCESS_KEY" not in env
            assert "AWS_ACCESS_KEY_ID" not in env
        finally:
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
            os.environ.pop("AWS_ACCESS_KEY_ID", None)
            self.tool._safe_env = None  # reset cache

    def test_safe_env_strips_openai_key(self):
        os.environ["OPENAI_API_KEY"] = "sk-test123"
        try:
            self.tool._safe_env = None
            env = self.tool._build_safe_env()
            assert "OPENAI_API_KEY" not in env
        finally:
            os.environ.pop("OPENAI_API_KEY", None)
            self.tool._safe_env = None

    def test_safe_env_strips_github_token(self):
        os.environ["GITHUB_TOKEN"] = "ghp_test123"
        os.environ["GH_TOKEN"] = "ghp_test456"
        try:
            self.tool._safe_env = None
            env = self.tool._build_safe_env()
            assert "GITHUB_TOKEN" not in env
            assert "GH_TOKEN" not in env
        finally:
            os.environ.pop("GITHUB_TOKEN", None)
            os.environ.pop("GH_TOKEN", None)
            self.tool._safe_env = None

    def test_safe_env_preserves_path(self):
        self.tool._safe_env = None
        env = self.tool._build_safe_env()
        assert "PATH" in env
        assert "HOME" in env

    def test_safe_env_strips_exact_names(self):
        os.environ["API_KEY"] = "test"
        os.environ["SECRET"] = "test"
        os.environ["PASSWORD"] = "test"
        try:
            self.tool._safe_env = None
            env = self.tool._build_safe_env()
            assert "API_KEY" not in env
            assert "SECRET" not in env
            assert "PASSWORD" not in env
        finally:
            for k in ("API_KEY", "SECRET", "PASSWORD"):
                os.environ.pop(k, None)
            self.tool._safe_env = None

    def test_subprocess_uses_safe_env(self):
        """Shell commands should not see sensitive vars."""
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        try:
            self.tool._safe_env = None
            result = self.tool.execute(command="echo $ANTHROPIC_API_KEY", timeout=5)
            # The variable should be empty/unset in the child process
            assert "test-anthropic-key" not in result.output
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            self.tool._safe_env = None
