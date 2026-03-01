"""Tab-completion for @file references in the REPL."""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.completion import Completer, Completion

# Directories to skip during completion
_SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".eggs", "*.egg-info", "dist", "build", ".hg", ".svn",
})


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


class AtFileCompleter(Completer):
    """Completes file/directory paths after ``@`` in the input."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        at_pos = text.rfind("@")
        if at_pos < 0:
            return

        # Skip if preceded by alphanumeric (e.g. email addresses)
        if at_pos > 0 and text[at_pos - 1].isalnum():
            return

        partial = text[at_pos + 1:]

        # Split into directory part and name prefix
        if "/" in partial:
            dir_part, name_prefix = partial.rsplit("/", 1)
            search_dir = (self.project_root / dir_part).resolve()
        else:
            dir_part = ""
            name_prefix = partial
            search_dir = self.project_root

        if not search_dir.is_dir():
            return

        # Security: don't complete paths outside project root
        try:
            search_dir.relative_to(self.project_root)
        except ValueError:
            return

        # Calculate how far back to replace (from cursor)
        start_position = -(len(partial))

        try:
            entries = sorted(search_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            name = entry.name

            # Skip hidden/ignored directories
            if name in _SKIP_DIRS:
                continue
            if entry.is_dir() and name.startswith(".") and name not in (".github",):
                continue

            if not name.lower().startswith(name_prefix.lower()):
                continue

            if entry.is_dir():
                completion_text = f"{dir_part}/{name}/" if dir_part else f"{name}/"
                yield Completion(
                    completion_text,
                    start_position=start_position,
                    display=f"{name}/",
                    display_meta="directory",
                )
            else:
                completion_text = f"{dir_part}/{name}" if dir_part else name
                try:
                    size = _human_size(entry.stat().st_size)
                except OSError:
                    size = "?"
                yield Completion(
                    completion_text,
                    start_position=start_position,
                    display=name,
                    display_meta=size,
                )
