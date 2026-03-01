"""Simplified configuration for Open Harness v2.

Config discovery (first match wins):
  1. ``--config`` flag
  2. ``./open_harness.yaml``
  3. ``~/.config/open-harness/config.yaml``
  4. Built-in defaults
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config data structures
# ---------------------------------------------------------------------------

@dataclass
class ProfileSpec:
    """A named LLM provider profile.

    ``models`` is an ordered list where position implies tier:
    index 0 = smallest/fastest, last = largest.
    """

    provider: str = "ollama"
    url: str = "http://localhost:11434/v1"
    api_key: str = "no-key"
    api_type: str = "openai"
    models: list[str] = field(default_factory=lambda: ["qwen3-8b"])
    extra_params: dict[str, Any] = field(default_factory=dict)

    @property
    def tier_count(self) -> int:
        return len(self.models)

    def model_for_tier(self, tier: int) -> str:
        """Return model name for a tier index (clamped to valid range)."""
        idx = max(0, min(tier, len(self.models) - 1))
        return self.models[idx]


@dataclass
class PolicySpec:
    """Policy configuration — controls guardrails for autonomous execution."""

    mode: str = "balanced"  # "safe" | "balanced" | "full"
    max_file_writes: int = 0
    max_shell_commands: int = 0
    max_git_commits: int = 10
    max_external_calls: int = 0
    denied_paths: list[str] = field(
        default_factory=lambda: [
            "/etc/*", "/usr/*", "/bin/*", "/sbin/*", "/boot/*",
            "~/.ssh/*", "~/.gnupg/*", "**/.env", "**/.env.*",
            "**/credentials*", "**/secrets*",
        ]
    )
    writable_paths: list[str] = field(default_factory=list)
    blocked_shell_patterns: list[str] = field(
        default_factory=lambda: [
            "curl * | *sh", "wget * | *sh",
            "chmod 777", "chmod -R 777",
            "> /dev/sd*",
            "git push --force", "git push -f",
            "git reset --hard",
        ]
    )
    disabled_tools: list[str] = field(default_factory=list)
    max_tokens_per_goal: int = 0


# Presets applied before user overrides
_PRESETS: dict[str, dict[str, Any]] = {
    "safe": {
        "max_file_writes": 20,
        "max_shell_commands": 30,
        "max_git_commits": 3,
        "max_external_calls": 10,
    },
    "balanced": {
        "max_file_writes": 0,
        "max_shell_commands": 0,
        "max_git_commits": 10,
        "max_external_calls": 0,
    },
    "full": {
        "max_file_writes": 0,
        "max_shell_commands": 0,
        "max_git_commits": 0,
        "max_external_calls": 0,
        "writable_paths": ["~/*"],
    },
}


@dataclass
class HarnessConfig:
    """Top-level config for Open Harness v2."""

    # Active profile name
    profile: str = "local"

    # Named profiles
    profiles: dict[str, ProfileSpec] = field(
        default_factory=lambda: {"local": ProfileSpec()}
    )

    # Policy spec
    policy: PolicySpec = field(default_factory=PolicySpec)

    # Compensation
    max_retries: int = 3
    thinking_mode: str = "auto"  # "auto" | "always" | "never"

    # Agent loop
    max_steps: int = 50

    @property
    def active_profile(self) -> ProfileSpec:
        return self.profiles.get(self.profile, ProfileSpec())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_SEARCH_PATHS = [
    Path("./open_harness.yaml"),
    Path.home() / ".config" / "open-harness" / "config.yaml",
]


def _parse_profile(raw: dict[str, Any]) -> ProfileSpec:
    return ProfileSpec(
        provider=raw.get("provider", "ollama"),
        url=raw.get("url", "http://localhost:11434/v1"),
        api_key=raw.get("api_key", "no-key"),
        api_type=raw.get("api_type", "openai"),
        models=raw.get("models", ["qwen3-8b"]),
        extra_params=raw.get("extra_params", {}),
    )


def _parse_policy(raw: dict[str, Any] | None) -> PolicySpec:
    if not raw:
        return PolicySpec()
    mode = raw.get("mode", "balanced")
    # Start from preset, overlay explicit values
    base = dict(_PRESETS.get(mode, _PRESETS["balanced"]))
    base["mode"] = mode
    for k, v in raw.items():
        if v is not None and k in PolicySpec.__dataclass_fields__:
            base[k] = v
    return PolicySpec(**base)


def load_config(path: str | Path | None = None) -> HarnessConfig:
    """Load configuration from YAML.

    Parameters
    ----------
    path:
        Explicit config path.  If *None*, search default locations.

    Returns
    -------
    HarnessConfig
    """
    config_path: Path | None = None

    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            _logger.warning("Config file not found: %s — using defaults", path)
            return HarnessConfig()
    else:
        for candidate in _SEARCH_PATHS:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None:
        _logger.info("No config file found — using defaults")
        return HarnessConfig()

    _logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Parse profiles
    profiles: dict[str, ProfileSpec] = {}
    for name, praw in raw.get("profiles", {}).items():
        profiles[name] = _parse_profile(praw)

    # If no profiles section, build one from legacy llm config
    if not profiles:
        profiles["local"] = ProfileSpec()

    return HarnessConfig(
        profile=raw.get("profile", "local"),
        profiles=profiles,
        policy=_parse_policy(raw.get("policy")),
        max_retries=raw.get("max_retries", 3),
        thinking_mode=raw.get("thinking_mode", "auto"),
        max_steps=raw.get("max_steps", 50),
    )
