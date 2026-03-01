"""Tests for SessionLogger diagnostics."""

import json
import tempfile
from pathlib import Path

import pytest

from open_harness.diagnostics import SessionLogger


@pytest.fixture
def tmp_log(tmp_path):
    """Create a SessionLogger writing to a temp file."""
    log_path = tmp_path / "test_session.jsonl"
    logger = SessionLogger(path=log_path)
    yield logger, log_path
    logger.close()


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestSessionLogger:
    def test_file_created(self, tmp_log):
        logger, path = tmp_log
        assert path.exists()

    def test_log_event(self, tmp_log):
        logger, path = tmp_log

        class FakeEvent:
            type = "tool_call"
            data = "shell"
            metadata = {"tool": "shell", "args": {"command": "ls"}}

        logger.log_event(FakeEvent())
        lines = _read_lines(path)
        assert len(lines) == 1
        assert lines[0]["kind"] == "event"
        assert lines[0]["type"] == "tool_call"
        assert lines[0]["metadata"]["tool"] == "shell"

    def test_sequence_numbers(self, tmp_log):
        logger, path = tmp_log

        class FakeEvent:
            type = "text"
            data = "hello"
            metadata = {}

        logger.log_event(FakeEvent())
        logger.log_event(FakeEvent())
        logger.log_event(FakeEvent())

        lines = _read_lines(path)
        assert [l["_seq"] for l in lines] == [0, 1, 2]

    def test_elapsed_ms_increases(self, tmp_log):
        logger, path = tmp_log

        class FakeEvent:
            type = "status"
            data = "test"
            metadata = {}

        logger.log_event(FakeEvent())
        logger.log_event(FakeEvent())

        lines = _read_lines(path)
        assert lines[0]["_elapsed_ms"] <= lines[1]["_elapsed_ms"]

    def test_log_system_prompt(self, tmp_log):
        logger, path = tmp_log
        logger.log_system_prompt("You are an assistant...", "plan")

        lines = _read_lines(path)
        assert len(lines) == 1
        assert lines[0]["kind"] == "system_prompt"
        assert lines[0]["mode"] == "plan"
        assert lines[0]["length"] == len("You are an assistant...")

    def test_log_llm_turn(self, tmp_log):
        logger, path = tmp_log
        logger.log_llm_turn(
            tier="medium",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=1234.5,
            messages_count=5,
            response_preview="Hello world",
        )

        lines = _read_lines(path)
        assert len(lines) == 1
        assert lines[0]["kind"] == "llm_turn"
        assert lines[0]["tier"] == "medium"
        assert lines[0]["usage"]["prompt_tokens"] == 100
        assert lines[0]["latency_ms"] == 1234.5

    def test_log_session_start_end(self, tmp_log):
        logger, path = tmp_log
        logger.log_session_start({"version": "0.4.5", "project": "/tmp/test"})
        logger.log_session_end({"total_events": 42, "elapsed_s": 10.5})

        lines = _read_lines(path)
        assert len(lines) == 2
        assert lines[0]["kind"] == "session_start"
        assert lines[0]["version"] == "0.4.5"
        assert lines[1]["kind"] == "session_end"
        assert lines[1]["total_events"] == 42

    def test_jsonl_format(self, tmp_log):
        """Each line must be valid JSON."""
        logger, path = tmp_log

        class FakeEvent:
            type = "text"
            data = "line1"
            metadata = {}

        logger.log_event(FakeEvent())
        logger.log_system_prompt("prompt", "goal")

        raw = path.read_text()
        for line in raw.strip().splitlines():
            parsed = json.loads(line)
            assert "_seq" in parsed
            assert "_ts" in parsed
            assert "_elapsed_ms" in parsed

    def test_close_idempotent(self, tmp_log):
        logger, path = tmp_log
        logger.close()
        logger.close()  # should not raise

    def test_default_path(self, tmp_path, monkeypatch):
        """Default path goes to ~/.open_harness/sessions/."""
        monkeypatch.setenv("HOME", str(tmp_path))
        # Also patch Path.home() for consistency
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        logger = SessionLogger()
        try:
            assert logger.path.parent == tmp_path / ".open_harness" / "sessions"
            assert logger.path.name.startswith("session_")
            assert logger.path.suffix == ".jsonl"
        finally:
            logger.close()

    def test_long_data_truncated(self, tmp_log):
        """Event data longer than 500 chars is truncated in the log."""
        logger, path = tmp_log

        class FakeEvent:
            type = "text"
            data = "x" * 1000
            metadata = {}

        logger.log_event(FakeEvent())
        lines = _read_lines(path)
        assert len(lines[0]["data"]) == 500
