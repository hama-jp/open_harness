"""Model router - routes requests to appropriate model tier."""

from __future__ import annotations

import logging
from typing import Any, Generator

from open_harness.config import HarnessConfig, ModelConfig
from open_harness.llm.client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes LLM requests to the appropriate model based on tier."""

    def __init__(self, config: HarnessConfig):
        self.config = config
        self._clients: dict[str, LLMClient] = {}
        self._current_tier: str = config.llm.default_tier

    def _get_client(self, provider_name: str) -> LLMClient:
        if provider_name not in self._clients:
            provider_cfg = self.config.llm.providers.get(provider_name)
            if provider_cfg is None:
                raise ValueError(f"Unknown provider: {provider_name}")
            self._clients[provider_name] = LLMClient(provider_cfg)
        return self._clients[provider_name]

    def get_model_config(self, tier: str | None = None) -> ModelConfig:
        tier = tier or self._current_tier
        cfg = self.config.llm.models.get(tier)
        if cfg is None:
            raise ValueError(f"Unknown model tier: {tier}")
        return cfg

    @property
    def current_tier(self) -> str:
        return self._current_tier

    @current_tier.setter
    def current_tier(self, tier: str):
        if tier in self.config.llm.models:
            self._current_tier = tier
        else:
            logger.warning(f"Unknown tier: {tier}, keeping {self._current_tier}")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tier: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        tier = tier or self._current_tier
        model_cfg = self.get_model_config(tier)
        client = self._get_client(model_cfg.provider)
        return client.chat(
            messages=messages,
            model=model_cfg.model,
            max_tokens=max_tokens or model_cfg.max_tokens,
            temperature=temperature,
            tools=tools,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tier: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> Generator[tuple[str, str], None, LLMResponse]:
        """Streaming chat via the specified tier's model.

        Yields (event_type, data) and returns LLMResponse at completion.
        """
        tier = tier or self._current_tier
        model_cfg = self.get_model_config(tier)
        client = self._get_client(model_cfg.provider)
        return client.chat_stream(
            messages=messages,
            model=model_cfg.model,
            max_tokens=max_tokens or model_cfg.max_tokens,
            temperature=temperature,
        )

    def list_tiers(self) -> dict[str, str]:
        return {
            name: cfg.description
            for name, cfg in self.config.llm.models.items()
        }

    def close(self):
        for name, client in self._clients.items():
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()
