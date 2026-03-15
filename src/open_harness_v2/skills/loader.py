"""Skill loading and registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_logger = logging.getLogger(__name__)

# Skill discovery paths (later paths override earlier)
_BUILTIN_DIR = Path(__file__).parent / "builtin"
_USER_DIR = Path.home() / ".config" / "open-harness" / "skills"
_PROJECT_DIR_NAME = ".harness/skills"


@dataclass
class Skill:
    """A reusable slash command that expands into a prompt."""

    name: str
    description: str
    prompt: str
    args: str = "none"  # "none" | "optional" | "required"
    args_description: str = ""
    source: str = "builtin"  # "builtin" | "user" | "project"

    def expand(self, user_args: str = "") -> str:
        """Expand the skill prompt, optionally incorporating user arguments."""
        prompt = self.prompt
        if user_args:
            prompt += f"\n\nUser input: {user_args}"
        return prompt


class SkillRegistry:
    """Registry for discovering and managing skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill (later registrations override earlier ones)."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def skill_names(self) -> list[str]:
        """Return list of registered skill names."""
        return list(self._skills.keys())

    def discover(self, project_root: Path | None = None) -> None:
        """Load skills from builtin, user, and project directories.

        Later sources override earlier ones (builtin < user < project).
        """
        # 1. Built-in skills
        self._load_dir(_BUILTIN_DIR, source="builtin")

        # 2. User-level skills
        self._load_dir(_USER_DIR, source="user")

        # 3. Project-level skills
        if project_root:
            project_skills = project_root / _PROJECT_DIR_NAME
            self._load_dir(project_skills, source="project")

        _logger.info("Discovered %d skill(s)", len(self._skills))

    def _load_dir(self, directory: Path, source: str) -> None:
        """Load all YAML skill files from a directory."""
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.yaml")):
            try:
                skill = _parse_skill_file(path, source)
                self.register(skill)
                _logger.debug("Loaded skill: %s from %s", skill.name, path)
            except Exception:
                _logger.exception("Failed to load skill: %s", path)
        for path in sorted(directory.glob("*.yml")):
            try:
                skill = _parse_skill_file(path, source)
                self.register(skill)
                _logger.debug("Loaded skill: %s from %s", skill.name, path)
            except Exception:
                _logger.exception("Failed to load skill: %s", path)


def _parse_skill_file(path: Path, source: str) -> Skill:
    """Parse a YAML skill file into a Skill object."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    name = raw.get("name") or path.stem
    description = raw.get("description", "")
    prompt = raw.get("prompt", "")
    if not prompt:
        raise ValueError(f"Skill {name} has no prompt")

    return Skill(
        name=name,
        description=description,
        prompt=prompt,
        args=raw.get("args", "none"),
        args_description=raw.get("args_description", ""),
        source=source,
    )
