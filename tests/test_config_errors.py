"""Tests for Issue 8: Config Error Handling."""

import os
import tempfile

import pytest
import yaml

from open_harness.config import load_config


class TestConfigLoading:
    def test_valid_config(self):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump({
                    "llm": {
                        "default_provider": "test",
                        "providers": {"test": {"base_url": "http://localhost:1234/v1"}},
                    }
                }, f)
            config, cfg_path = load_config(path)
            assert config is not None
            assert cfg_path is not None
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/tmp/nonexistent_config_12345.yaml")

    def test_invalid_yaml(self):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("invalid: yaml: content: [[[")
            with pytest.raises((yaml.YAMLError, Exception)):
                load_config(path)
        finally:
            os.unlink(path)

    def test_defaults_when_no_config(self):
        """load_config(None) should return defaults if no config file exists."""
        # Temporarily rename any real config to avoid interference
        config, path = load_config(None)
        assert config is not None

    def test_empty_config(self):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("")
            config, cfg_path = load_config(path)
            # Empty file should use defaults
            assert config is not None
        finally:
            os.unlink(path)
