"""Tests for HARNESS.md project instructions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from open_harness_v2.project_instructions import load_project_instructions


class TestProjectInstructions:
    def test_no_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = load_project_instructions(Path(tmp))
            assert result == ""

    def test_load_harness_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness_md = Path(tmp) / "HARNESS.md"
            harness_md.write_text("# My Project\nAlways use pytest for tests.")
            result = load_project_instructions(Path(tmp))
            assert "My Project" in result
            assert "pytest" in result

    def test_load_dotharness_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            instr_dir = Path(tmp) / ".harness"
            instr_dir.mkdir()
            instr_file = instr_dir / "instructions.md"
            instr_file.write_text("Use ruff for linting.")
            result = load_project_instructions(Path(tmp))
            assert "ruff" in result

    def test_harness_md_takes_priority(self):
        """HARNESS.md should win over .harness/instructions.md."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create both files
            Path(tmp, "HARNESS.md").write_text("HARNESS.md content")
            instr_dir = Path(tmp) / ".harness"
            instr_dir.mkdir()
            Path(instr_dir, "instructions.md").write_text("dotharness content")

            result = load_project_instructions(Path(tmp))
            assert "HARNESS.md content" in result
            # Should NOT include the dotharness content (first match wins)
            assert "dotharness content" not in result

    def test_truncation(self):
        """Very large files should be truncated."""
        with tempfile.TemporaryDirectory() as tmp:
            harness_md = Path(tmp) / "HARNESS.md"
            harness_md.write_text("x" * 20_000)
            result = load_project_instructions(Path(tmp))
            assert "truncated" in result
            assert len(result) < 20_000
