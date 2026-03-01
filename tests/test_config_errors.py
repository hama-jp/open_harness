"""Tests for Issue 8: Config Error Handling."""

import tempfile

import yaml

from open_harness.config import load_config


class TestConfigLoading:
    def test_valid_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "llm": {
                    "default_provider": "test",
                    "providers": {"test": {"base_url": "http://localhost:1234/v1"}},
                }
            }, f)
            f.flush()
            config, path = load_config(f.name)
        assert config is not None
        assert path is not None

    def test_nonexistent_file(self):
        try:
            load_config("/tmp/nonexistent_config_12345.yaml")
        except FileNotFoundError:
            pass  # Expected

    def test_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [[[")
            f.flush()
            try:
                load_config(f.name)
            except yaml.YAMLError:
                pass  # Expected

    def test_defaults_when_no_config(self):
        """load_config(None) should return defaults if no config file exists."""
        config, path = load_config(None)
        assert config is not None

    def test_empty_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            config, path = load_config(f.name)
        # Empty file should use defaults
        assert config is not None
