"""Integration tests — real Ollama and external agent execution.

These tests hit live services and are marked with ``pytest.mark.integration``
so they can be skipped in CI with ``pytest -m "not integration"``.

Requirements:
  - Ollama installed and a model pulled  (``ollama list`` should show >=1 model)
  - codex / claude / gemini on PATH       (skip individually if absent)
"""

from __future__ import annotations

import os
import shutil

import httpx
import pytest

# ---------------------------------------------------------------------------
# Markers & helpers
# ---------------------------------------------------------------------------

integration = pytest.mark.integration

OLLAMA_BASE = "http://localhost:11434"


def _ollama_reachable() -> bool:
    try:
        return httpx.get(OLLAMA_BASE, timeout=3).status_code < 500
    except (httpx.HTTPError, OSError):
        return False


def _ollama_has_models() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return bool(resp.json().get("models"))
    except Exception:
        return False


def _first_model() -> str:
    resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    return resp.json()["models"][0]["name"]


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(), reason="Ollama server not reachable"
)
requires_ollama_models = pytest.mark.skipif(
    not _ollama_has_models(), reason="Ollama has no models pulled"
)
requires_codex = pytest.mark.skipif(
    shutil.which("codex") is None, reason="codex not on PATH"
)
requires_claude = pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("CLAUDECODE"),
    reason="claude not on PATH or running inside Claude Code session",
)
requires_gemini = pytest.mark.skipif(
    shutil.which("gemini") is None, reason="gemini not on PATH"
)


# ===================================================================
# Ollama — setup helpers
# ===================================================================

@integration
@requires_ollama
class TestOllamaConnection:
    """Verify real Ollama connectivity."""

    def test_server_reachable(self):
        resp = httpx.get(OLLAMA_BASE, timeout=5)
        assert resp.status_code == 200
        assert "Ollama" in resp.text

    def test_fetch_models_returns_real_list(self):
        from open_harness_v2.setup import _fetch_models

        models = _fetch_models(
            f"{OLLAMA_BASE}/v1", "ollama", "ollama"
        )
        assert isinstance(models, list)
        if _ollama_has_models():
            assert len(models) >= 1
            assert all(isinstance(m, str) for m in models)

    def test_ensure_ollama_already_running(self):
        """ensure_ollama should return True immediately when server is up."""
        from open_harness_v2.setup import ensure_ollama

        assert ensure_ollama(OLLAMA_BASE) is True


# ===================================================================
# Ollama — real LLM call via v2 orchestrator
# ===================================================================

@integration
@requires_ollama
@requires_ollama_models
class TestOllamaLLMCall:
    """Test actual LLM inference through the v2 stack."""

    async def test_simple_chat(self):
        """Send a trivial prompt and verify we get a non-empty response."""
        from open_harness_v2.config import HarnessConfig, ProfileSpec
        from open_harness_v2.llm.router import ModelRouter

        config = HarnessConfig(
            profile="test",
            profiles={
                "test": ProfileSpec(
                    provider="ollama",
                    url=f"{OLLAMA_BASE}/v1",
                    api_key="ollama",
                    api_type="ollama",
                    models=[_first_model()],
                ),
            },
        )
        router = ModelRouter(config)
        client = router.get_client()

        messages = [{"role": "user", "content": "Reply with exactly: HELLO"}]
        resp = await client.chat(messages, model=router.current_model)
        assert resp.content, "LLM returned empty content"
        assert len(resp.content) > 0

    async def test_orchestrator_one_shot(self):
        """Full orchestrator run: prompt -> LLM -> text response."""
        from open_harness_v2.config import HarnessConfig, ProfileSpec, PolicySpec
        from open_harness_v2.core.orchestrator import Orchestrator
        from open_harness_v2.events.bus import EventBus
        from open_harness_v2.llm.router import ModelRouter
        from open_harness_v2.policy.engine import PolicyEngine
        from open_harness_v2.tools.registry import ToolRegistry

        config = HarnessConfig(
            profile="test",
            profiles={
                "test": ProfileSpec(
                    provider="ollama",
                    url=f"{OLLAMA_BASE}/v1",
                    api_key="ollama",
                    api_type="ollama",
                    models=[_first_model()],
                ),
            },
            policy=PolicySpec(mode="safe"),
        )
        router = ModelRouter(config)
        registry = ToolRegistry()
        policy = PolicyEngine(config.policy)
        bus = EventBus()

        orchestrator = Orchestrator(
            router=router,
            registry=registry,
            policy=policy,
            event_bus=bus,
            max_steps=5,
        )

        result = await orchestrator.run(
            "What is 2+2? Reply with just the number."
        )
        assert result, "Orchestrator returned empty result"
        assert "4" in result


# ===================================================================
# External agents — real execution
# ===================================================================

@integration
@requires_codex
class TestCodexReal:
    """Run real Codex CLI."""

    async def test_codex_simple_prompt(self):
        from open_harness_v2.tools.builtin.external import CodexTool

        tool = CodexTool(timeout=120)
        assert tool.available

        result = await tool.execute(
            prompt="Print 'hello from codex' to stdout"
        )
        assert result.success, f"Codex failed: {result.error}"
        assert result.output, "Codex returned empty output"


@integration
@requires_claude
class TestClaudeCodeReal:
    """Run real Claude Code CLI."""

    async def test_claude_simple_prompt(self):
        from open_harness_v2.tools.builtin.external import ClaudeCodeTool

        tool = ClaudeCodeTool(timeout=120)
        assert tool.available

        result = await tool.execute(
            prompt="What is 2+2? Answer with just the number."
        )
        assert result.success, f"Claude Code failed: {result.error}"
        assert result.output, "Claude Code returned empty output"


@integration
@requires_gemini
class TestGeminiCliReal:
    """Run real Gemini CLI."""

    async def test_gemini_simple_prompt(self):
        from open_harness_v2.tools.builtin.external import GeminiCliTool

        tool = GeminiCliTool(timeout=120)
        assert tool.available

        result = await tool.execute(
            prompt="What is 2+2? Answer with just the number."
        )
        assert result.success, f"Gemini CLI failed: {result.error}"
        assert result.output, "Gemini CLI returned empty output"


# ===================================================================
# Config round-trip — wizard output -> load_config
# ===================================================================

@integration
@requires_ollama
class TestWizardConfigRoundTrip:
    """Verify wizard-generated YAML works with real Ollama."""

    def test_generated_config_loads_and_connects(self, tmp_path):
        from open_harness_v2.setup import _build_yaml, _fetch_models

        models = _fetch_models(f"{OLLAMA_BASE}/v1", "ollama", "ollama")
        model = models[0] if models else "qwen3-8b"

        yaml_str = _build_yaml(
            provider_key="ollama",
            base_url=f"{OLLAMA_BASE}/v1",
            api_key="ollama",
            api_type="ollama",
            model_name=model,
            policy_mode="balanced",
        )

        config_path = tmp_path / "open_harness.yaml"
        config_path.write_text(yaml_str)

        from open_harness_v2.config import load_config

        cfg = load_config(str(config_path))
        assert cfg.active_profile.provider == "ollama"
        assert cfg.active_profile.models[0] == model
        assert cfg.policy.mode == "balanced"

        # Verify we can actually reach the server via this config
        native_url = cfg.active_profile.url.rstrip("/").removesuffix("/v1")
        resp = httpx.get(native_url, timeout=5)
        assert resp.status_code == 200
