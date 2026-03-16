"""Tests for structured result extraction from external agent output."""

import pytest

from open_harness_v2.tools.builtin.external import (
    _extract_structured_result,
    _format_structured_output,
)


class TestExtractStructuredResult:
    """Test the structured result parser for external agent output."""

    def test_empty_input(self):
        result = _extract_structured_result("")
        assert result["summary"] == ""
        assert result["changed_files"] == []
        assert result["risks"] == []

    def test_plain_text_summary(self):
        output = "I fixed the bug in the login module.\nThe issue was a missing null check."
        result = _extract_structured_result(output)
        assert "fixed the bug" in result["summary"]

    def test_file_detection(self):
        output = "Modified src/auth/login.py and src/auth/utils.py to fix the issue."
        result = _extract_structured_result(output)
        assert "src/auth/login.py" in result["changed_files"]
        assert "src/auth/utils.py" in result["changed_files"]

    def test_file_deduplication(self):
        output = "Changed src/main.py and then also updated src/main.py."
        result = _extract_structured_result(output)
        assert result["changed_files"].count("src/main.py") == 1

    def test_test_results_passed(self):
        output = "All done. 15 tests passed, 0 failed."
        result = _extract_structured_result(output)
        assert result["tests"]["passed"] == 15
        assert result["tests"]["failed"] == 0

    def test_test_results_failed(self):
        output = "2 tests failed out of 10."
        result = _extract_structured_result(output)
        assert result["tests"]["failed"] == 2

    def test_risk_detection(self):
        output = "Warning: This change may break backwards compatibility."
        result = _extract_structured_result(output)
        assert len(result["risks"]) > 0

    def test_todo_detection(self):
        output = "TODO: Need to add error handling for edge cases."
        result = _extract_structured_result(output)
        assert len(result["risks"]) > 0
        assert "error handling" in result["risks"][0].lower()

    def test_json_block_extraction(self):
        output = """Here's what I did:
```json
{
    "summary": "Fixed auth bug",
    "changed_files": ["auth.py", "tests/test_auth.py"],
    "tests": {"passed": 5, "failed": 0},
    "risks": ["Breaking API change"]
}
```
Done!"""
        result = _extract_structured_result(output)
        assert result["summary"] == "Fixed auth bug"
        assert result["changed_files"] == ["auth.py", "tests/test_auth.py"]
        assert result["tests"]["passed"] == 5
        assert result["risks"] == ["Breaking API change"]

    def test_json_block_partial(self):
        output = """Result:
```json
{"summary": "Quick fix"}
```"""
        result = _extract_structured_result(output)
        assert result["summary"] == "Quick fix"
        # Other fields should remain defaults
        assert result["changed_files"] == []

    def test_raw_length_tracked(self):
        output = "a" * 1000
        result = _extract_structured_result(output)
        assert result["raw_length"] == 1000

    def test_no_url_in_changed_files(self):
        output = "See https://example.com/docs/guide.py for more info."
        result = _extract_structured_result(output)
        # URLs should not be treated as changed files
        assert not any("http" in f for f in result["changed_files"])


class TestFormatStructuredOutput:
    """Test the output formatter."""

    def test_appends_structured_section(self):
        raw = "Fixed the bug."
        structured = {
            "summary": "Fixed auth bug",
            "changed_files": ["auth.py"],
            "tests": {"passed": 5, "failed": 0},
            "risks": [],
        }
        formatted = _format_structured_output(raw, structured)
        assert "Fixed the bug." in formatted
        assert "--- Structured Result ---" in formatted
        assert "Summary: Fixed auth bug" in formatted
        assert "Changed files: auth.py" in formatted
        assert "Tests: 5 passed, 0 failed" in formatted

    def test_empty_structured(self):
        raw = "Output."
        structured = {
            "summary": "",
            "changed_files": [],
            "tests": {},
            "risks": [],
        }
        formatted = _format_structured_output(raw, structured)
        assert "--- Structured Result ---" not in formatted
        assert formatted == "Output."

    def test_risks_included(self):
        raw = "Done."
        structured = {
            "summary": "Completed",
            "changed_files": [],
            "tests": {},
            "risks": ["May break API"],
        }
        formatted = _format_structured_output(raw, structured)
        assert "Risks: May break API" in formatted
