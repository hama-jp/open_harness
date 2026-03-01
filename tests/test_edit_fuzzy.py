"""Tests for Issue 3: EditFileTool Fuzzy Matching."""

import os
import tempfile

from open_harness.tools.file_ops import EditFileTool


class TestEditFileFuzzy:
    def setup_method(self):
        self.tool = EditFileTool()
        self._tmp_files = []

    def teardown_method(self):
        for path in self._tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _write_tmp(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.write(fd, content.encode())
        os.close(fd)
        self._tmp_files.append(path)
        return path

    def test_exact_match(self):
        path = self._write_tmp("hello world\nfoo bar\n")
        result = self.tool.execute(path=path, old_string="hello world", new_string="replaced")
        assert result.success
        assert "whitespace" not in result.output
        content = open(path).read()
        assert "replaced" in content

    def test_fuzzy_whitespace_match(self):
        path = self._write_tmp("  hello   world  \n  foo bar  \n")
        result = self.tool.execute(
            path=path,
            old_string="hello world\nfoo bar",
            new_string="replaced",
        )
        assert result.success
        assert "whitespace normalization" in result.output
        content = open(path).read()
        assert "replaced" in content

    def test_no_match(self):
        path = self._write_tmp("hello world\n")
        result = self.tool.execute(
            path=path,
            old_string="completely different text",
            new_string="replaced",
        )
        assert not result.success
        assert "not found" in result.error

    def test_multiple_exact_matches(self):
        path = self._write_tmp("foo\nfoo\n")
        result = self.tool.execute(path=path, old_string="foo", new_string="bar")
        assert not result.success
        assert "2 times" in result.error

    def test_multiple_fuzzy_matches(self):
        path = self._write_tmp("  hello  \n  hello  \n")
        result = self.tool.execute(path=path, old_string="hello", new_string="bar")
        # Exact match finds 0 (because of whitespace), fuzzy finds 2
        # Should report multiple fuzzy matches
        assert not result.success
        assert "2 times" in result.error

    def test_empty_old_string(self):
        path = self._write_tmp("content\n")
        result = self.tool.execute(path=path, old_string="", new_string="x")
        assert not result.success
        assert "No old_string" in result.error

    def test_file_not_found(self):
        result = self.tool.execute(
            path="/tmp/nonexistent_file_12345.txt",
            old_string="x",
            new_string="y",
        )
        assert not result.success
        assert "not found" in result.error.lower() or "File not found" in result.error

    def test_fuzzy_tabs_vs_spaces(self):
        path = self._write_tmp("\thello\tworld\n")
        result = self.tool.execute(
            path=path,
            old_string="hello world",
            new_string="replaced",
        )
        assert result.success
        assert "whitespace normalization" in result.output

    def test_path_is_directory(self):
        tmpdir = tempfile.mkdtemp()
        self._tmp_files.append(tmpdir)  # will fail unlink but that's ok
        result = self.tool.execute(path=tmpdir, old_string="x", new_string="y")
        assert not result.success


class TestNormalizeWs:
    def test_strips_and_collapses(self):
        result = EditFileTool._normalize_ws("  hello   world  \n  foo   bar  ")
        assert result == "hello world\nfoo bar"

    def test_empty(self):
        assert EditFileTool._normalize_ws("") == ""

    def test_single_line(self):
        assert EditFileTool._normalize_ws("  a  b  ") == "a b"
