"""HARNESS.md — project-level instructions, inspired by Claude Code's CLAUDE.md.

Loads project instructions from HARNESS.md (or .harness/instructions.md)
and injects them into the system prompt context.

Discovery order (first match wins):
  1. ``HARNESS.md`` in project root
  2. ``.harness/instructions.md`` in project root
  3. ``~/.config/open-harness/instructions.md`` (global)
"""

from __future__ import annotations

import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

# File names to search for project instructions
_PROJECT_FILES = [
    "HARNESS.md",
    ".harness/instructions.md",
]
_GLOBAL_FILE = Path.home() / ".config" / "open-harness" / "instructions.md"

# Maximum size of instructions file to prevent accidental huge injections
_MAX_INSTRUCTIONS_SIZE = 10_000  # chars


def load_project_instructions(project_root: Path | None = None) -> str:
    """Load project instructions from HARNESS.md or fallbacks.

    Returns the instructions text, or empty string if none found.
    """
    instructions_parts: list[str] = []

    # 1. Global instructions (lowest priority)
    if _GLOBAL_FILE.is_file():
        text = _read_limited(_GLOBAL_FILE)
        if text:
            instructions_parts.append(f"# Global Instructions\n{text}")
            _logger.info("Loaded global instructions from %s", _GLOBAL_FILE)

    # 2. Project-level instructions (highest priority)
    if project_root:
        for filename in _PROJECT_FILES:
            path = project_root / filename
            if path.is_file():
                text = _read_limited(path)
                if text:
                    instructions_parts.append(f"# Project Instructions\n{text}")
                    _logger.info("Loaded project instructions from %s", path)
                break  # first match wins

    if not instructions_parts:
        return ""

    return "\n\n".join(instructions_parts)


def _read_limited(path: Path) -> str:
    """Read a file, truncating to _MAX_INSTRUCTIONS_SIZE."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > _MAX_INSTRUCTIONS_SIZE:
            _logger.warning(
                "Instructions file %s truncated from %d to %d chars",
                path, len(text), _MAX_INSTRUCTIONS_SIZE,
            )
            text = text[:_MAX_INSTRUCTIONS_SIZE] + "\n... (truncated)"
        return text.strip()
    except Exception:
        _logger.exception("Failed to read instructions: %s", path)
        return ""
