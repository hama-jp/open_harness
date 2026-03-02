"""Tests for Open Harness v2 setup wizard and Ollama auto-start."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

from open_harness_v2.setup import (
    _build_yaml,
    _fetch_models,
    _is_ollama_reachable,
    ensure_ollama,
    run_setup_wizard,
)


# ---------------------------------------------------------------------------
# _build_yaml
# ---------------------------------------------------------------------------

class TestBuildYaml:
    def test_ollama_yaml(self):
        out = _build_yaml(
            provider_key="ollama",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            api_type="ollama",
            model_name="qwen3-8b",
            policy_mode="balanced",
        )
        parsed = yaml.safe_load(out)
        assert parsed["profile"] == "local"
        assert parsed["profiles"]["local"]["provider"] == "ollama"
        assert parsed["profiles"]["local"]["api_type"] == "ollama"
        assert parsed["profiles"]["local"]["models"] == ["qwen3-8b"]
        assert parsed["policy"]["mode"] == "balanced"

    def test_openai_yaml(self):
        out = _build_yaml(
            provider_key="lm_studio",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            api_type="openai",
            model_name="llama3",
            policy_mode="safe",
        )
        parsed = yaml.safe_load(out)
        assert parsed["profiles"]["local"]["api_type"] == "openai"
        assert parsed["profiles"]["local"]["api_key"] == "lm-studio"
        assert parsed["policy"]["mode"] == "safe"

    def test_yaml_is_valid(self):
        """The generated YAML should be loadable by the v2 config loader."""
        out = _build_yaml(
            provider_key="ollama",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            api_type="ollama",
            model_name="test-model",
            policy_mode="full",
        )
        from open_harness_v2.config import load_config

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(out)
            f.flush()
            cfg = load_config(f.name)

        assert cfg.active_profile.provider == "ollama"
        assert cfg.active_profile.models == ["test-model"]
        assert cfg.policy.mode == "full"


# ---------------------------------------------------------------------------
# _is_ollama_reachable / ensure_ollama
# ---------------------------------------------------------------------------

class TestOllamaReachable:
    def test_reachable(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("open_harness_v2.setup.httpx.get", return_value=mock_resp):
            assert _is_ollama_reachable("http://localhost:11434") is True

    def test_unreachable_connect_error(self):
        with patch(
            "open_harness_v2.setup.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            assert _is_ollama_reachable("http://localhost:11434") is False

    def test_unreachable_timeout(self):
        with patch(
            "open_harness_v2.setup.httpx.get",
            side_effect=httpx.ReadTimeout("timeout"),
        ):
            assert _is_ollama_reachable("http://localhost:11434") is False


class TestEnsureOllama:
    def test_already_running(self):
        with patch("open_harness_v2.setup._is_ollama_reachable", return_value=True):
            assert ensure_ollama("http://localhost:11434") is True

    def test_not_installed(self):
        with (
            patch("open_harness_v2.setup._is_ollama_reachable", return_value=False),
            patch("open_harness_v2.setup.shutil.which", return_value=None),
        ):
            assert ensure_ollama("http://localhost:11434") is False

    def test_start_success(self):
        reachable_calls = [False, False, True]  # 3rd poll succeeds

        with (
            patch(
                "open_harness_v2.setup._is_ollama_reachable",
                side_effect=reachable_calls,
            ),
            patch("open_harness_v2.setup.shutil.which", return_value="/usr/bin/ollama"),
            patch("open_harness_v2.setup.subprocess.Popen") as mock_popen,
            patch("open_harness_v2.setup.time.sleep"),
        ):
            assert ensure_ollama("http://localhost:11434") is True
            mock_popen.assert_called_once()

    def test_start_timeout(self):
        with (
            patch("open_harness_v2.setup._is_ollama_reachable", return_value=False),
            patch("open_harness_v2.setup.shutil.which", return_value="/usr/bin/ollama"),
            patch("open_harness_v2.setup.subprocess.Popen"),
            patch("open_harness_v2.setup.time.sleep"),
        ):
            assert ensure_ollama("http://localhost:11434") is False

    def test_popen_oserror(self):
        with (
            patch("open_harness_v2.setup._is_ollama_reachable", return_value=False),
            patch("open_harness_v2.setup.shutil.which", return_value="/usr/bin/ollama"),
            patch(
                "open_harness_v2.setup.subprocess.Popen",
                side_effect=OSError("exec failed"),
            ),
        ):
            assert ensure_ollama("http://localhost:11434") is False


# ---------------------------------------------------------------------------
# _fetch_models
# ---------------------------------------------------------------------------

class TestFetchModels:
    def test_ollama_models(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "qwen3-8b"}, {"name": "llama3"}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("open_harness_v2.setup.httpx.get", return_value=mock_resp):
            models = _fetch_models(
                "http://localhost:11434/v1", "ollama", "ollama"
            )
        assert models == ["qwen3-8b", "llama3"]

    def test_openai_models(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("open_harness_v2.setup.httpx.get", return_value=mock_resp):
            models = _fetch_models(
                "http://localhost:1234/v1", "lm-studio", "openai"
            )
        assert models == ["gpt-4o", "gpt-4o-mini"]

    def test_connection_error_returns_empty(self):
        with patch(
            "open_harness_v2.setup.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            assert _fetch_models("http://localhost:11434/v1", "", "ollama") == []


# ---------------------------------------------------------------------------
# run_setup_wizard (full flow with mocked I/O)
# ---------------------------------------------------------------------------

class TestRunSetupWizard:
    def test_full_wizard_flow(self, tmp_path):
        """Simulate a full wizard run: Ollama → default URL → default key → model → balanced."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "qwen3-8b"}]}

        prompt_answers = iter([
            1,                                          # provider: Ollama
            "http://localhost:11434/v1",                 # server URL
            "ollama",                                    # api key
            1,                                          # model: qwen3-8b (from list)
            2,                                          # policy: balanced
        ])

        def fake_prompt(text, **kwargs):
            return next(prompt_answers)

        with (
            patch("open_harness_v2.setup.httpx.get", return_value=mock_resp),
            patch("open_harness_v2.setup.click.prompt", side_effect=fake_prompt),
            patch("open_harness_v2.setup.click.confirm", return_value=True),
        ):
            result = run_setup_wizard(config_dir=tmp_path)

        assert result == tmp_path / "open_harness.yaml"
        assert result.exists()

        parsed = yaml.safe_load(result.read_text())
        assert parsed["profile"] == "local"
        assert parsed["profiles"]["local"]["provider"] == "ollama"
        assert parsed["profiles"]["local"]["models"] == ["qwen3-8b"]
        assert parsed["policy"]["mode"] == "balanced"

    def test_wizard_abort(self, tmp_path):
        """User aborts at confirmation step."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "qwen3-8b"}]}

        prompt_answers = iter([1, "http://localhost:11434/v1", "ollama", 1, 2])

        with (
            patch("open_harness_v2.setup.httpx.get", return_value=mock_resp),
            patch("open_harness_v2.setup.click.prompt", side_effect=lambda *a, **kw: next(prompt_answers)),
            patch("open_harness_v2.setup.click.confirm", return_value=False),
            pytest.raises(SystemExit),
        ):
            run_setup_wizard(config_dir=tmp_path)

        assert not (tmp_path / "open_harness.yaml").exists()
