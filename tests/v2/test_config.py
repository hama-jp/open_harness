"""Tests for Open Harness v2 config."""

import tempfile
from pathlib import Path

import yaml

from open_harness_v2.config import (
    HarnessConfig,
    PolicySpec,
    ProfileSpec,
    load_config,
)


class TestProfileSpec:
    def test_defaults(self):
        p = ProfileSpec()
        assert p.provider == "ollama"
        assert p.models == ["qwen3-8b"]
        assert p.tier_count == 1

    def test_model_for_tier(self):
        p = ProfileSpec(models=["small-model", "medium-model", "large-model"])
        assert p.model_for_tier(0) == "small-model"
        assert p.model_for_tier(1) == "medium-model"
        assert p.model_for_tier(2) == "large-model"

    def test_model_for_tier_clamped(self):
        p = ProfileSpec(models=["only-model"])
        assert p.model_for_tier(0) == "only-model"
        assert p.model_for_tier(5) == "only-model"
        assert p.model_for_tier(-1) == "only-model"


class TestPolicySpec:
    def test_defaults(self):
        p = PolicySpec()
        assert p.mode == "balanced"
        assert p.max_file_writes == 0
        assert len(p.denied_paths) > 0
        assert len(p.blocked_shell_patterns) > 0

    def test_custom(self):
        p = PolicySpec(mode="safe", max_file_writes=10)
        assert p.mode == "safe"
        assert p.max_file_writes == 10


class TestHarnessConfig:
    def test_defaults(self):
        cfg = HarnessConfig()
        assert cfg.profile == "local"
        assert "local" in cfg.profiles
        assert cfg.max_retries == 3
        assert cfg.max_steps == 50

    def test_active_profile(self):
        cfg = HarnessConfig(
            profile="api",
            profiles={
                "local": ProfileSpec(),
                "api": ProfileSpec(provider="openai", models=["gpt-4o-mini", "gpt-4o"]),
            },
        )
        assert cfg.active_profile.provider == "openai"
        assert cfg.active_profile.models == ["gpt-4o-mini", "gpt-4o"]

    def test_active_profile_fallback(self):
        cfg = HarnessConfig(profile="nonexistent")
        # Should return default ProfileSpec when profile name doesn't match
        p = cfg.active_profile
        assert p.provider == "ollama"


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path):
        cfg = load_config(tmp_path / "does_not_exist.yaml")
        assert cfg.profile == "local"
        assert cfg.max_retries == 3

    def test_load_from_yaml(self, tmp_path):
        config = {
            "profile": "api",
            "profiles": {
                "local": {
                    "provider": "ollama",
                    "url": "http://localhost:11434/v1",
                    "models": ["qwen3-8b", "qwen3-30b"],
                },
                "api": {
                    "provider": "openai",
                    "url": "https://api.openai.com/v1",
                    "api_key": "sk-test",
                    "models": ["gpt-4o-mini", "gpt-4o"],
                },
            },
            "policy": {
                "mode": "safe",
                "max_file_writes": 15,
            },
            "max_retries": 5,
            "thinking_mode": "never",
            "max_steps": 100,
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        cfg = load_config(config_path)
        assert cfg.profile == "api"
        assert cfg.active_profile.provider == "openai"
        assert cfg.active_profile.models == ["gpt-4o-mini", "gpt-4o"]
        assert cfg.profiles["local"].models == ["qwen3-8b", "qwen3-30b"]
        assert cfg.policy.mode == "safe"
        assert cfg.policy.max_file_writes == 15
        assert cfg.max_retries == 5
        assert cfg.thinking_mode == "never"
        assert cfg.max_steps == 100

    def test_load_empty_yaml(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        cfg = load_config(config_path)
        assert cfg.profile == "local"

    def test_policy_preset_with_overrides(self, tmp_path):
        config = {
            "policy": {
                "mode": "safe",
                "max_external_calls": 20,
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        cfg = load_config(config_path)
        # From "safe" preset
        assert cfg.policy.max_file_writes == 20
        assert cfg.policy.max_shell_commands == 30
        # Overridden
        assert cfg.policy.max_external_calls == 20
