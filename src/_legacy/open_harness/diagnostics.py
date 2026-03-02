"""Session diagnostics — structured JSONL logging for agent sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class SessionLogger:
    """Log all agent events and LLM turns to a timestamped JSONL file.

    File: ~/.open_harness/sessions/session_YYYYMMDD_HHMMSS.jsonl
    Each line: {"_seq": 0, "_elapsed_ms": 123.4, "_ts": "...", ...}
    """

    def __init__(self, path: Path | None = None):
        if path is None:
            sessions_dir = Path.home() / ".open_harness" / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = sessions_dir / f"session_{ts}.jsonl"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        self.path = path
        self._file = open(path, "a", encoding="utf-8")
        self._seq = 0
        self._start = time.monotonic()

    def _write(self, record: dict[str, Any]) -> None:
        record["_seq"] = self._seq
        record["_elapsed_ms"] = round((time.monotonic() - self._start) * 1000, 1)
        record["_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._seq += 1
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def log_event(self, event: Any) -> None:
        """Log an AgentEvent (or anything with .type, .data, .metadata)."""
        record: dict[str, Any] = {
            "kind": "event",
            "type": event.type,
            "data": event.data[:500] if event.data else "",
        }
        if event.metadata:
            record["metadata"] = event.metadata
        self._write(record)

    def log_system_prompt(self, prompt: str, mode: str) -> None:
        """Record the system prompt used for this session/turn."""
        self._write({
            "kind": "system_prompt",
            "mode": mode,
            "length": len(prompt),
            "preview": prompt[:200],
        })

    def log_llm_turn(
        self,
        tier: str,
        usage: dict[str, int] | None = None,
        latency_ms: float = 0,
        messages_count: int = 0,
        response_preview: str = "",
    ) -> None:
        """Record stats for a single LLM API call."""
        self._write({
            "kind": "llm_turn",
            "tier": tier,
            "usage": usage or {},
            "latency_ms": round(latency_ms, 1),
            "messages_count": messages_count,
            "response_preview": response_preview[:200],
        })

    def log_session_start(self, info: dict[str, Any]) -> None:
        """Record session start information."""
        self._write({"kind": "session_start", **info})

    def log_session_end(self, summary: dict[str, Any]) -> None:
        """Record session end summary."""
        self._write({"kind": "session_end", **summary})

    def close(self) -> None:
        """Flush and close the log file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()
