"""Project context auto-detection and loading.

Scans the working directory to understand project type, structure, conventions,
and test configuration. This context is injected into the system prompt so the
LLM can work autonomously without the user having to explain the project.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProjectContext:
    """Auto-detects project context from the filesystem."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.getcwd()).resolve()
        self._info: dict[str, Any] | None = None

    @property
    def info(self) -> dict[str, Any]:
        if self._info is None:
            self._info = self._detect()
        return self._info

    def _detect(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "root": str(self.root),
            "type": "unknown",
            "languages": [],
            "test_command": None,
            "lint_command": None,
            "build_command": None,
            "has_git": (self.root / ".git").is_dir(),
            "key_files": [],
            "custom_context": "",
        }

        # Custom harness context file (highest priority)
        for name in [".harness/context.md", "AGENTS.md", "CLAUDE.md"]:
            p = self.root / name
            if p.exists():
                try:
                    text = p.read_text()[:4000]  # Cap size
                    ctx["custom_context"] = text
                    ctx["key_files"].append(name)
                except Exception:
                    pass
                break

        # Python
        if (self.root / "pyproject.toml").exists():
            ctx["type"] = "python"
            ctx["languages"].append("python")
            ctx["key_files"].append("pyproject.toml")
            ctx["test_command"] = self._detect_python_test()
            ctx["lint_command"] = "ruff check ."
            if (self.root / "setup.py").exists():
                ctx["key_files"].append("setup.py")

        elif (self.root / "setup.py").exists():
            ctx["type"] = "python"
            ctx["languages"].append("python")
            ctx["test_command"] = "python3 -m pytest"

        # JavaScript / TypeScript
        if (self.root / "package.json").exists():
            ctx["languages"].append("javascript")
            if ctx["type"] == "unknown":
                ctx["type"] = "javascript"
            ctx["key_files"].append("package.json")
            ctx["test_command"] = ctx["test_command"] or "npm test"
            if (self.root / "tsconfig.json").exists():
                ctx["languages"].append("typescript")
                ctx["key_files"].append("tsconfig.json")

        # Rust
        if (self.root / "Cargo.toml").exists():
            ctx["type"] = "rust"
            ctx["languages"].append("rust")
            ctx["key_files"].append("Cargo.toml")
            ctx["test_command"] = "cargo test"
            ctx["build_command"] = "cargo build"

        # Go
        if (self.root / "go.mod").exists():
            ctx["type"] = "go"
            ctx["languages"].append("go")
            ctx["key_files"].append("go.mod")
            ctx["test_command"] = "go test ./..."

        # Detect structure
        ctx["structure"] = self._scan_structure()

        return ctx

    def _detect_python_test(self) -> str:
        if (self.root / "pytest.ini").exists() or (self.root / "pyproject.toml").exists():
            return "python3 -m pytest"
        if (self.root / "tests").is_dir():
            return "python3 -m pytest tests/"
        return "python3 -m pytest"

    def _scan_structure(self, max_depth: int = 3, max_entries: int = 60) -> str:
        """Generate a compact directory tree."""
        lines: list[str] = []
        skip = {".git", ".venv", "venv", "node_modules", "__pycache__",
                ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build",
                ".eggs", ".tox", ".next", "target"}

        def _walk(path: Path, prefix: str, depth: int):
            if depth > max_depth or len(lines) >= max_entries:
                return
            try:
                entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
            except PermissionError:
                return
            dirs = [e for e in entries if e.is_dir() and e.name not in skip]
            files = [e for e in entries if e.is_file()]
            for f in files:
                if len(lines) >= max_entries:
                    return
                lines.append(f"{prefix}{f.name}")
            for d in dirs:
                if len(lines) >= max_entries:
                    return
                lines.append(f"{prefix}{d.name}/")
                _walk(d, prefix + "  ", depth + 1)

        _walk(self.root, "", 0)
        return "\n".join(lines)

    def ensure_git(self) -> str:
        """Ensure the project directory is under git management.

        If no .git directory exists, runs ``git init`` and creates an initial
        commit so that any file changes made by the agent can be reverted.

        Returns a status message.
        """
        if (self.root / ".git").is_dir():
            return "git already initialized"

        # Create .gitignore if missing
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            try:
                gitignore.write_text(
                    "# Auto-generated by Open Harness\n"
                    "__pycache__/\n*.pyc\n.venv/\nvenv/\n"
                    "node_modules/\n.env\n*.egg-info/\ndist/\nbuild/\n"
                )
            except OSError as e:
                logger.warning("Failed to create .gitignore: %s", e)

        cwd = str(self.root)
        r = subprocess.run(
            ["git", "init"], capture_output=True, text=True, timeout=15, cwd=cwd,
        )
        if r.returncode != 0:
            msg = f"git init failed: {r.stderr.strip()}"
            logger.warning(msg)
            return msg

        subprocess.run(
            ["git", "add", "-A"], capture_output=True, text=True, timeout=15, cwd=cwd,
        )
        commit = subprocess.run(
            ["git", "commit", "-m", "Initial commit (auto-created by Open Harness)"],
            capture_output=True, text=True, timeout=15, cwd=cwd,
        )
        if commit.returncode != 0:
            # Initial commit failed (e.g. no git identity configured).
            # Without a baseline commit, checkpoints cannot work safely,
            # so leave has_git=False and clean up the .git directory.
            import shutil
            git_dir = self.root / ".git"
            if git_dir.is_dir():
                shutil.rmtree(git_dir, ignore_errors=True)
            msg = f"git init succeeded but initial commit failed: {commit.stderr.strip()}"
            logger.warning(msg)
            return msg

        # Update cached info so has_git=True and git tools get registered
        if self._info is not None:
            self._info["has_git"] = True

        logger.info("Auto-initialized git repository at %s", self.root)
        return "auto-initialized git"

    def to_prompt(self) -> str:
        """Format context for LLM system prompt."""
        info = self.info
        parts = [f"Project root: {info['root']}"]
        parts.append(f"Type: {info['type']} ({', '.join(info['languages']) or 'unknown'})")

        if info["test_command"]:
            parts.append(f"Test command: {info['test_command']}")
        if info["lint_command"]:
            parts.append(f"Lint command: {info['lint_command']}")
        if info["build_command"]:
            parts.append(f"Build command: {info['build_command']}")
        if info["has_git"]:
            parts.append("Version control: git")

        if info["structure"]:
            parts.append(f"\nProject structure:\n{info['structure']}")

        if info["custom_context"]:
            parts.append(f"\nProject instructions:\n{info['custom_context']}")

        return "\n".join(parts)
