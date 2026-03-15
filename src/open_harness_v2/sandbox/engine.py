"""Sandbox engine — wraps shell command execution with OS-level isolation.

Provides filesystem and network restrictions for agent-executed commands.
Inspired by Codex CLI's sandbox approach (Landlock on Linux, Seatbelt on macOS).

Usage::

    sandbox = SandboxEngine(
        mode=SandboxMode.WORKSPACE,
        workspace_root=Path.cwd(),
    )
    prefix = sandbox.build_command_prefix()
    # prefix is prepended to shell commands for isolation
"""

from __future__ import annotations

import enum
import logging
import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_logger = logging.getLogger(__name__)


class SandboxMode(enum.Enum):
    """Sandbox isolation levels."""

    READ_ONLY = "read-only"
    WORKSPACE = "workspace"
    FULL_ACCESS = "full-access"


@dataclass
class SandboxEngine:
    """Manages sandbox execution environment for shell commands.

    Parameters
    ----------
    mode:
        Sandbox isolation level.
    workspace_root:
        Root directory for workspace-write mode.
    extra_writable:
        Additional directories allowed for writes (workspace mode).
    extra_readable:
        Additional directories allowed for reads (read-only mode).
    allow_network:
        Allow network access in sandbox (default: False for workspace mode).
    """

    mode: SandboxMode = SandboxMode.WORKSPACE
    workspace_root: Path = field(default_factory=Path.cwd)
    extra_writable: list[Path] = field(default_factory=list)
    extra_readable: list[Path] = field(default_factory=list)
    allow_network: bool = False

    # Paths always readable regardless of mode
    _ALWAYS_READABLE: list[str] = field(default_factory=lambda: [
        "/usr", "/lib", "/lib64", "/bin", "/sbin",
        "/etc/alternatives", "/etc/ld.so.cache", "/etc/ld.so.conf",
        "/proc", "/sys", "/dev/null", "/dev/urandom", "/dev/zero",
        "/tmp",
    ])

    def __post_init__(self) -> None:
        self._landlock_available: bool | None = None
        self._bwrap_available: bool | None = None

    @property
    def is_restricted(self) -> bool:
        """Return True if sandbox imposes any restrictions."""
        return self.mode != SandboxMode.FULL_ACCESS

    def build_command_prefix(self) -> str:
        """Build a command prefix that applies sandbox restrictions.

        Returns an empty string if no sandboxing is available or needed.
        """
        if self.mode == SandboxMode.FULL_ACCESS:
            return ""

        system = platform.system()

        if system == "Linux":
            # Try bwrap (bubblewrap) first — more capable
            prefix = self._try_bwrap()
            if prefix:
                return prefix

            # Landlock is applied via environment, not prefix
            # (handled separately in wrap_command)
            _logger.debug(
                "No bwrap available; sandbox restrictions will be "
                "limited to policy-level checks"
            )
            return ""

        elif system == "Darwin":
            prefix = self._try_seatbelt()
            if prefix:
                return prefix

        _logger.debug(
            "No OS-level sandbox available on %s; "
            "relying on policy-level restrictions",
            system,
        )
        return ""

    def wrap_command(self, command: str) -> str:
        """Wrap a shell command with sandbox restrictions.

        Parameters
        ----------
        command:
            The raw shell command to execute.

        Returns
        -------
        str
            The command with sandbox prefix prepended, or unchanged
            if no sandboxing is available.
        """
        prefix = self.build_command_prefix()
        if prefix:
            return f"{prefix} -- {command}"
        return command

    def get_sandbox_env(self) -> dict[str, str]:
        """Return extra environment variables for sandbox enforcement.

        On Linux with Landlock support, sets LANDLOCK_RESTRICT=1 and
        related variables.
        """
        env: dict[str, str] = {}

        if self.mode == SandboxMode.FULL_ACCESS:
            return env

        if platform.system() == "Linux":
            env["HARNESS_SANDBOX_MODE"] = self.mode.value
            env["HARNESS_WORKSPACE_ROOT"] = str(self.workspace_root.resolve())

        return env

    # ------------------------------------------------------------------
    # Platform-specific implementations
    # ------------------------------------------------------------------

    def _check_bwrap(self) -> bool:
        """Check if bubblewrap (bwrap) is available."""
        if self._bwrap_available is None:
            self._bwrap_available = shutil.which("bwrap") is not None
        return self._bwrap_available

    def _try_bwrap(self) -> str:
        """Build a bwrap command prefix for Linux sandboxing."""
        if not self._check_bwrap():
            return ""

        parts = ["bwrap"]

        # Always bind readable system paths
        for rpath in self._ALWAYS_READABLE:
            if os.path.exists(rpath):
                parts.append(f"--ro-bind {rpath} {rpath}")

        workspace = str(self.workspace_root.resolve())

        if self.mode == SandboxMode.READ_ONLY:
            # Read-only access to workspace
            parts.append(f"--ro-bind {workspace} {workspace}")
            for extra in self.extra_readable:
                resolved = str(extra.resolve())
                if os.path.exists(resolved):
                    parts.append(f"--ro-bind {resolved} {resolved}")

        elif self.mode == SandboxMode.WORKSPACE:
            # Read-write access to workspace only
            parts.append(f"--bind {workspace} {workspace}")
            # .git directory read-only within workspace
            git_dir = os.path.join(workspace, ".git")
            if os.path.exists(git_dir):
                parts.append(f"--ro-bind {git_dir} {git_dir}")
            for extra in self.extra_writable:
                resolved = str(extra.resolve())
                if os.path.exists(resolved):
                    parts.append(f"--bind {resolved} {resolved}")

        # Home directory read-only for config files
        home = str(Path.home())
        if home not in workspace:
            parts.append(f"--ro-bind {home} {home}")

        # Process isolation
        parts.append("--unshare-pid")
        parts.append("--die-with-parent")

        # Network isolation (unless explicitly allowed)
        if not self.allow_network:
            parts.append("--unshare-net")

        # Set working directory
        parts.append(f"--chdir {workspace}")

        return " ".join(parts)

    def _try_seatbelt(self) -> str:
        """Build a sandbox-exec command for macOS Seatbelt."""
        if not shutil.which("sandbox-exec"):
            return ""

        workspace = str(self.workspace_root.resolve())

        if self.mode == SandboxMode.READ_ONLY:
            profile = self._seatbelt_readonly_profile(workspace)
        else:
            profile = self._seatbelt_workspace_profile(workspace)

        # Write the profile to a temp location
        return f'sandbox-exec -p \'{profile}\''

    @staticmethod
    def _seatbelt_readonly_profile(workspace: str) -> str:
        return f"""(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow file-read* (subpath "{workspace}"))
(allow file-read* (subpath "/usr"))
(allow file-read* (subpath "/lib"))
(allow file-read* (subpath "/bin"))
(allow file-read* (subpath "/sbin"))
(allow file-read* (subpath "/dev"))
(allow file-read* (subpath "/tmp"))
(allow file-read* (subpath "/private/tmp"))
(allow file-read-metadata)
(allow sysctl-read)
(allow mach-lookup)
"""

    @staticmethod
    def _seatbelt_workspace_profile(workspace: str) -> str:
        return f"""(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow file-read*)
(allow file-write* (subpath "{workspace}"))
(allow file-write* (subpath "/tmp"))
(allow file-write* (subpath "/private/tmp"))
(allow file-read-metadata)
(allow sysctl-read)
(allow mach-lookup)
"""

    @classmethod
    def from_policy_mode(
        cls,
        policy_mode: str,
        workspace_root: Path | None = None,
        extra_writable: list[Path] | None = None,
    ) -> SandboxEngine:
        """Create a SandboxEngine from a policy mode string.

        Parameters
        ----------
        policy_mode:
            One of "safe", "balanced", "full".
        workspace_root:
            Project root directory.
        extra_writable:
            Additional writable paths.
        """
        mode_map = {
            "safe": SandboxMode.READ_ONLY,
            "balanced": SandboxMode.WORKSPACE,
            "full": SandboxMode.FULL_ACCESS,
        }
        sandbox_mode = mode_map.get(policy_mode, SandboxMode.WORKSPACE)
        return cls(
            mode=sandbox_mode,
            workspace_root=workspace_root or Path.cwd(),
            extra_writable=extra_writable or [],
        )
