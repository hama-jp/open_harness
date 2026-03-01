"""Configuration management for Open Harness."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    base_url: str
    api_key: str = "no-key"
    api_type: str = "openai"  # "openai" or "ollama" (native /api/chat)
    extra_params: dict[str, Any] = Field(default_factory=dict)  # merged into every request


class ModelConfig(BaseModel):
    provider: str
    model: str
    max_tokens: int = 4096
    context_length: int = 0  # num_ctx for Ollama (0 = provider default)
    description: str = ""


class CompensationConfig(BaseModel):
    max_retries: int = 3
    retry_strategies: list[str] = Field(
        default_factory=lambda: ["refine_prompt", "add_examples", "escalate_model"]
    )
    parse_fallback: bool = True
    thinking_mode: str = "auto"


class ShellToolConfig(BaseModel):
    timeout: int = 30
    allowed_commands: list[str] = Field(default_factory=list)
    blocked_commands: list[str] = Field(
        default_factory=lambda: ["rm -rf /", "mkfs", "dd if="]
    )


class FileToolConfig(BaseModel):
    max_read_size: int = 100_000


class ToolsConfig(BaseModel):
    shell: ShellToolConfig = Field(default_factory=ShellToolConfig)
    file: FileToolConfig = Field(default_factory=FileToolConfig)


class ExternalAgentConfig(BaseModel):
    enabled: bool = False
    command: str = ""
    description: str = ""  # Human-readable description for the orchestrator
    strengths: list[str] = Field(default_factory=list)  # Task categories this agent excels at


class MemoryConfig(BaseModel):
    backend: str = "sqlite"
    db_path: str = "~/.open_harness/memory.db"
    max_conversation_turns: int = 50


class LLMConfig(BaseModel):
    default_provider: str = "lm_studio"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    default_tier: str = "medium"


class HarnessConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    compensation: CompensationConfig = Field(default_factory=CompensationConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    external_agents: dict[str, ExternalAgentConfig] = Field(default_factory=dict)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    policy: dict[str, Any] = Field(default_factory=dict)  # raw dict, parsed by PolicyEngine


CONFIG_FILENAME = "open_harness.yaml"

# Also accept legacy name for backwards compatibility
_CONFIG_NAMES = [CONFIG_FILENAME, "config.yaml"]


def load_config(
    config_path: str | Path | None = None,
) -> tuple[HarnessConfig, Path | None]:
    """Load configuration from YAML file.

    Returns (config, resolved_path).  *resolved_path* is ``None`` when
    no file was found and built-in defaults are used.

    Search order (first match wins):
      1. Explicit ``--config`` path
      2. Current working directory: ``./open_harness.yaml`` (or legacy ``config.yaml``)
      3. User config dir: ``~/.open_harness/open_harness.yaml``
      4. Package directory (development): ``<repo>/open_harness.yaml``

    This allows placing ``open_harness.yaml`` in any project directory
    to customize settings per-project.
    """
    if config_path is None:
        search_dirs = [
            Path.cwd(),
            Path.home() / ".open_harness",
            Path(__file__).parent.parent.parent,  # repo root (dev mode)
        ]
        for d in search_dirs:
            for name in _CONFIG_NAMES:
                p = d / name
                if p.exists():
                    config_path = p
                    break
            if config_path:
                break

    resolved: Path | None = Path(config_path) if config_path else None
    if resolved and resolved.exists():
        with open(resolved) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return HarnessConfig.model_validate(raw), resolved.resolve()

    # Explicit path was given but file doesn't exist — report the error
    if config_path is not None and (resolved is None or not resolved.exists()):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return HarnessConfig(), None
