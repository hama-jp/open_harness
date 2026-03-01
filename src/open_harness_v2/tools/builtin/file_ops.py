"""Async file operation tools for Open Harness v2."""

from __future__ import annotations

import asyncio
import re as _re
from pathlib import Path
from typing import Any

from open_harness_v2.tools.base import Tool
from open_harness_v2.types import ToolParameter, ToolResult


class ReadFileTool(Tool):
    """Read file contents with optional offset and limit."""

    name = "read_file"
    description = "Read the contents of a file. Returns the file content as text."
    max_output = 8000
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to read",
        ),
        ToolParameter(
            name="offset",
            type="integer",
            description="Line number to start reading from (0-based)",
            required=False,
            default=0,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of lines to read",
            required=False,
            default=0,
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 0)

        if not path:
            return ToolResult(success=False, output="", error="No path provided")

        def _read() -> ToolResult:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return ToolResult(success=False, output="", error=f"File not found: {p}")
            if not p.is_file():
                return ToolResult(success=False, output="", error=f"Not a file: {p}")

            try:
                size = p.stat().st_size
                if size > 10_000_000:  # 10 MB
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"File too large ({size} bytes, max 10MB)",
                    )

                text = p.read_text(errors="replace")
                lines = text.splitlines(keepends=True)

                if offset > 0:
                    lines = lines[offset:]
                if limit > 0:
                    lines = lines[:limit]

                numbered = []
                for i, line in enumerate(lines, start=offset + 1):
                    numbered.append(f"{i:>5}\t{line.rstrip()}")

                return ToolResult(success=True, output="\n".join(numbered))
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))

        return await asyncio.to_thread(_read)


class WriteFileTool(Tool):
    """Write content to a file."""

    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to write",
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to the file",
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")

        if not path:
            return ToolResult(success=False, output="", error="No path provided")

        def _write() -> ToolResult:
            p = Path(path).expanduser().resolve()
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
                return ToolResult(
                    success=True,
                    output=f"Written {len(content)} bytes to {p}",
                )
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))

        return await asyncio.to_thread(_write)


class EditFileTool(Tool):
    """Edit a file by replacing text."""

    name = "edit_file"
    description = "Edit a file by replacing a specific string with a new string. The old_string must match exactly."
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to edit",
        ),
        ToolParameter(
            name="old_string",
            type="string",
            description="The exact text to find and replace",
        ),
        ToolParameter(
            name="new_string",
            type="string",
            description="The replacement text",
        ),
    ]

    @staticmethod
    def _normalize_ws(s: str) -> str:
        """Normalize whitespace: strip each line, collapse runs of spaces."""
        lines = [_re.sub(r"\s+", " ", line.strip()) for line in s.splitlines()]
        return "\n".join(lines)

    def _fuzzy_find(self, text: str, old: str) -> tuple[int, int] | int | None:
        """Find *old* in *text* using whitespace-normalized comparison.

        Returns ``(start, end)`` indices if exactly one match, the match count
        (int > 1) if multiple, or ``None`` if no match.
        """
        norm_old = self._normalize_ws(old)
        norm_old_lines = norm_old.splitlines()
        if not norm_old_lines:
            return None

        text_lines = text.splitlines(keepends=True)
        norm_text_lines = [_re.sub(r"\s+", " ", line.strip()) for line in text_lines]

        window = len(norm_old_lines)
        matches: list[tuple[int, int]] = []
        for i in range(len(norm_text_lines) - window + 1):
            candidate = "\n".join(norm_text_lines[i : i + window])
            if candidate == norm_old:
                start = sum(len(l) for l in text_lines[:i])
                end = sum(len(l) for l in text_lines[: i + window])
                matches.append((start, end))

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return len(matches)
        return None

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", "")
        old = kwargs.get("old_string", "")
        new = kwargs.get("new_string", "")

        if not path:
            return ToolResult(success=False, output="", error="No path provided")
        if not old:
            return ToolResult(success=False, output="", error="No old_string provided")

        def _edit() -> ToolResult:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return ToolResult(success=False, output="", error=f"File not found: {p}")

            try:
                text = p.read_text()
                count = text.count(old)
                if count == 1:
                    text = text.replace(old, new, 1)
                    p.write_text(text)
                    return ToolResult(success=True, output=f"Edit applied to {p}")
                if count > 1:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"old_string found {count} times. Provide more context to make it unique.",
                    )
                # count == 0 -- try whitespace-normalized fuzzy match
                span = self._fuzzy_find(text, old)
                if isinstance(span, tuple):
                    text = text[: span[0]] + new + text[span[1] :]
                    p.write_text(text)
                    return ToolResult(
                        success=True,
                        output=f"Edit applied to {p} (matched with whitespace normalization)",
                    )
                if isinstance(span, int):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"old_string found {span} times (with whitespace normalization). "
                              f"Provide more context to make it unique.",
                    )
                return ToolResult(success=False, output="", error="old_string not found in file")
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))

        return await asyncio.to_thread(_edit)


class ListDirectoryTool(Tool):
    """List directory contents."""

    name = "list_dir"
    description = "List files and directories in a given path."
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Directory path to list",
            required=False,
            default=".",
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path", ".")

        def _list() -> ToolResult:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return ToolResult(success=False, output="", error=f"Path not found: {p}")
            if not p.is_dir():
                return ToolResult(success=False, output="", error=f"Not a directory: {p}")

            try:
                items = sorted(p.iterdir())
                lines: list[str] = []
                for item in items[:500]:
                    prefix = "d " if item.is_dir() else "f "
                    size = ""
                    if item.is_file():
                        s = item.stat().st_size
                        if s < 1024:
                            size = f" ({s}B)"
                        elif s < 1024 * 1024:
                            size = f" ({s // 1024}KB)"
                        else:
                            size = f" ({s // (1024 * 1024)}MB)"
                    lines.append(f"{prefix}{item.name}{size}")

                return ToolResult(success=True, output="\n".join(lines))
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))

        return await asyncio.to_thread(_list)


class SearchFilesTool(Tool):
    """Search for text in files."""

    name = "search_files"
    description = (
        "Search for a text pattern in files within a directory. "
        "Returns matching lines with file paths and line numbers."
    )
    parameters = [
        ToolParameter(
            name="pattern",
            type="string",
            description="Text or regex pattern to search for",
        ),
        ToolParameter(
            name="path",
            type="string",
            description="Directory to search in",
            required=False,
            default=".",
        ),
    ]

    _SKIP_DIRS = {
        ".git", ".venv", "venv", "node_modules", "__pycache__",
        ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build",
        ".eggs", ".tox", ".next", "target", ".cache",
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        import re as re_mod

        pattern = kwargs.get("pattern", "")
        path = kwargs.get("path", ".")

        if not pattern:
            return ToolResult(success=False, output="", error="No pattern provided")

        def _search() -> ToolResult:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return ToolResult(success=False, output="", error=f"Path not found: {p}")

            try:
                regex = re_mod.compile(pattern, re_mod.IGNORECASE)
            except re_mod.error:
                regex = re_mod.compile(re_mod.escape(pattern), re_mod.IGNORECASE)

            matches: list[str] = []
            try:
                for fp in p.glob("**/*"):
                    if not fp.is_file():
                        continue
                    parts = fp.relative_to(p).parts
                    if any(part in self._SKIP_DIRS for part in parts):
                        continue
                    try:
                        if fp.stat().st_size > 1_000_000:
                            continue
                        text = fp.read_text(errors="replace")
                        for i, line in enumerate(text.splitlines(), 1):
                            if regex.search(line):
                                rel = fp.relative_to(p)
                                matches.append(f"{rel}:{i}: {line.strip()}")
                                if len(matches) >= 200:
                                    break
                    except (UnicodeDecodeError, PermissionError):
                        continue
                    if len(matches) >= 200:
                        break

                if not matches:
                    return ToolResult(success=True, output="No matches found.")
                return ToolResult(success=True, output="\n".join(matches))
            except Exception as e:
                return ToolResult(success=False, output="", error=str(e))

        return await asyncio.to_thread(_search)
