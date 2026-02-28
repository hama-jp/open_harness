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


class ModelConfig(BaseModel):
    provider: str
    model: str
    max_tokens: int = 4096
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


def load_config(config_path: str | Path | None = None) -> HarnessConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        candidates = [
            Path.cwd() / "config.yaml",
            Path.home() / ".open_harness" / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        for p in candidates:
            if p.exists():
                config_path = p
                break

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return HarnessConfig.model_validate(raw)

    return HarnessConfig()
