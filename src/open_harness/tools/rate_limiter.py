"""Rate limit tracking and fallback routing for external agents.

Detects when an external agent hits its usage quota, records the cooldown
period, and transparently re-routes tasks to a fallback agent until the
original agent recovers.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Patterns that indicate a rate/quota limit was hit.
# Each pattern is tried against the combined stdout+stderr of the agent.
_RATE_LIMIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"quota.?exceed", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"429", re.IGNORECASE),
    re.compile(r"usage.?limit", re.IGNORECASE),
    re.compile(r"capacity", re.IGNORECASE),
    re.compile(r"try again (in|after)", re.IGNORECASE),
    re.compile(r"please wait", re.IGNORECASE),
    re.compile(r"throttl", re.IGNORECASE),
]

# Try to extract a "retry after N minutes/seconds/hours" hint.
_RETRY_AFTER_PATTERN = re.compile(
    r"(?:retry|try\s+again|wait|available|resets?)[\s:]+(?:(?:in|after)\s+)?"
    r"(\d+)\s*(second|minute|hour|sec|min|hr|s|m|h)",
    re.IGNORECASE,
)

# Default cooldown when we can't parse a retry-after hint (15 minutes).
DEFAULT_COOLDOWN_SECONDS = 15 * 60

# Ordered fallback preferences: tool_name -> [fallback1, fallback2]
DEFAULT_FALLBACK_ORDER: dict[str, list[str]] = {
    "claude_code": ["codex", "gemini_cli"],
    "codex":       ["claude_code", "gemini_cli"],
    "gemini_cli":  ["claude_code", "codex"],
}


@dataclass
class CooldownEntry:
    """Tracks when an agent will be available again."""
    agent_name: str
    until: float  # time.time() epoch when cooldown expires
    reason: str = ""

    @property
    def remaining(self) -> float:
        return max(0.0, self.until - time.time())

    @property
    def expired(self) -> bool:
        return time.time() >= self.until

    def human_remaining(self) -> str:
        secs = self.remaining
        if secs <= 0:
            return "available now"
        if secs < 60:
            return f"{secs:.0f}s"
        if secs < 3600:
            return f"{secs / 60:.0f}m"
        return f"{secs / 3600:.1f}h"


class AgentRateLimiter:
    """Tracks rate-limited agents and provides fallback routing."""

    def __init__(
        self,
        fallback_order: dict[str, list[str]] | None = None,
        available_agents: list[str] | None = None,
    ):
        self._cooldowns: dict[str, CooldownEntry] = {}
        self._fallback_order = fallback_order or DEFAULT_FALLBACK_ORDER
        self._available = set(available_agents) if available_agents else None

    # ----- Query -----

    def is_available(self, agent_name: str) -> bool:
        """Check if an agent is available (not rate-limited)."""
        entry = self._cooldowns.get(agent_name)
        if entry is None:
            return True
        if entry.expired:
            del self._cooldowns[agent_name]
            logger.info("Agent %s cooldown expired — now available", agent_name)
            return True
        return False

    def get_cooldown(self, agent_name: str) -> CooldownEntry | None:
        entry = self._cooldowns.get(agent_name)
        if entry and entry.expired:
            del self._cooldowns[agent_name]
            return None
        return entry

    def get_all_cooldowns(self) -> dict[str, CooldownEntry]:
        """Return all active (non-expired) cooldowns."""
        self._cleanup()
        return dict(self._cooldowns)

    def get_fallback(self, agent_name: str) -> str | None:
        """Get the best available fallback for a rate-limited agent.

        Returns None if no fallback is available.
        """
        candidates = self._fallback_order.get(agent_name, [])
        for candidate in candidates:
            if self._available and candidate not in self._available:
                continue
            if self.is_available(candidate):
                return candidate
        return None

    def get_best_agent(self, preferred: str) -> tuple[str, str | None]:
        """Get the best agent to use, considering rate limits.

        Returns (agent_to_use, reason_or_None).
        If the preferred agent is available, returns (preferred, None).
        Otherwise returns (fallback, reason_string).
        """
        if self.is_available(preferred):
            return preferred, None
        entry = self._cooldowns.get(preferred)
        fallback = self.get_fallback(preferred)
        if fallback:
            reason = (
                f"{preferred} rate-limited (available in {entry.human_remaining()}), "
                f"using {fallback} instead"
            )
            return fallback, reason
        # No fallback available — still return the preferred agent and let it fail
        return preferred, None

    # ----- Record -----

    def record_rate_limit(
        self,
        agent_name: str,
        output: str,
        cooldown_seconds: float | None = None,
    ) -> CooldownEntry:
        """Record that an agent hit its rate limit.

        If cooldown_seconds is not provided, tries to parse a retry-after
        hint from the output. Falls back to DEFAULT_COOLDOWN_SECONDS.
        """
        if cooldown_seconds is None:
            cooldown_seconds = _parse_retry_after(output)

        until = time.time() + cooldown_seconds
        entry = CooldownEntry(
            agent_name=agent_name,
            until=until,
            reason=f"rate-limited at {time.strftime('%H:%M:%S')}",
        )
        self._cooldowns[agent_name] = entry
        logger.warning(
            "Agent %s rate-limited for %.0fs (until %s)",
            agent_name,
            cooldown_seconds,
            time.strftime("%H:%M:%S", time.localtime(until)),
        )
        return entry

    def clear(self, agent_name: str | None = None):
        """Clear cooldown for one or all agents."""
        if agent_name:
            self._cooldowns.pop(agent_name, None)
        else:
            self._cooldowns.clear()

    # ----- Detection -----

    @staticmethod
    def is_rate_limit_error(output: str) -> bool:
        """Detect whether the output indicates a rate limit error.

        Only scans the first 2000 chars — rate limit messages appear near
        the start of output, so scanning the full text is wasteful.
        """
        # Short-circuit: rate limit errors appear in the first portion of output
        output = output[:2000]
        for pattern in _RATE_LIMIT_PATTERNS:
            if pattern.search(output):
                return True
        return False

    # ----- Status -----

    def status_summary(self) -> str:
        """Human-readable summary of current cooldowns."""
        self._cleanup()
        if not self._cooldowns:
            return "All agents available"
        lines = []
        for name, entry in self._cooldowns.items():
            lines.append(f"  {name}: cooldown {entry.human_remaining()}")
        return "Rate-limited agents:\n" + "\n".join(lines)

    def _cleanup(self):
        expired = [k for k, v in self._cooldowns.items() if v.expired]
        for k in expired:
            del self._cooldowns[k]


def _parse_retry_after(output: str) -> float:
    """Try to extract a retry-after duration from agent output."""
    m = _RETRY_AFTER_PATTERN.search(output)
    if not m:
        return DEFAULT_COOLDOWN_SECONDS

    value = int(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("h"):
        return value * 3600
    if unit.startswith("m"):
        return value * 60
    return float(value)  # seconds
