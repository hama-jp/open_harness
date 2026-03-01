"""Model router — profile-based tier selection.

Uses ``ProfileSpec.model_for_tier()`` for tier-based model selection.
Tiers are integer indices (0 = smallest / fastest, last = largest).
"""

from __future__ import annotations

import logging

from open_harness_v2.config import HarnessConfig, ProfileSpec

from .client import AsyncLLMClient

_logger = logging.getLogger(__name__)


class ModelRouter:
    """Profile-based model tier selection and escalation.

    Parameters
    ----------
    config:
        The full ``HarnessConfig`` (used to read the active profile).
    client:
        An optional pre-built ``AsyncLLMClient``.  If ``None``, one will be
        created from the active profile.
    """

    def __init__(
        self,
        config: HarnessConfig,
        client: AsyncLLMClient | None = None,
    ) -> None:
        self._config = config
        self._profile: ProfileSpec = config.active_profile
        self._tier: int = 0
        self._client = client or AsyncLLMClient(self._profile)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_client(self) -> AsyncLLMClient:
        """Return the (shared) ``AsyncLLMClient``."""
        return self._client

    def model_for_tier(self, tier: int) -> str:
        """Return the model name for a given tier index."""
        return self._profile.model_for_tier(tier)

    @property
    def current_tier(self) -> int:
        """The currently active tier index."""
        return self._tier

    @property
    def current_model(self) -> str:
        """Convenience: model name for the current tier."""
        return self.model_for_tier(self._tier)

    def escalate(self) -> bool:
        """Move to the next larger tier.

        Returns ``True`` if escalation happened, ``False`` if already at max.
        """
        if self._tier >= self._profile.tier_count - 1:
            _logger.info(
                "Already at maximum tier %d (model=%s)",
                self._tier, self.current_model,
            )
            return False
        old = self._tier
        self._tier += 1
        _logger.info(
            "Escalated tier %d (%s) -> %d (%s)",
            old, self.model_for_tier(old),
            self._tier, self.current_model,
        )
        return True

    def reset_tier(self) -> None:
        """Reset to the smallest / cheapest tier."""
        self._tier = 0
