"""Sandbox module — OS-level isolation for shell command execution.

Inspired by Codex CLI's sandbox system. Provides three modes:

- ``read-only``   — commands can only read files (default for safe policy)
- ``workspace``   — writes limited to the project workspace
- ``full-access`` — no restrictions (for full policy mode)

On Linux, uses Landlock LSM when available for filesystem restriction.
Falls back to a minimal restriction approach on unsupported platforms.
"""

from open_harness_v2.sandbox.engine import SandboxEngine, SandboxMode

__all__ = ["SandboxEngine", "SandboxMode"]
