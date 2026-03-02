"""Context compactor — rule-based structured summaries for trimmed messages.

Extracts key information from tool results before they are compressed
by the ContextManager. This preserves important context (files modified,
test results, key findings) that would otherwise be lost during trimming.
"""

from __future__ import annotations

import re
from typing import Any


# Minimum estimated tokens in a compressed batch to apply summarization.
_MIN_TOKENS_FOR_SUMMARY = 5000

# Maximum chars in a single summary block.
_MAX_SUMMARY_CHARS = 500


def build_context_summary(messages: list[dict[str, Any]]) -> str | None:
    """Build a structured summary from a batch of tool result messages.

    Returns a compact summary string, or None if the batch is too small
    to warrant summarization.
    """
    total_chars = sum(len(m.get("content") or "") for m in messages)
    if total_chars // 4 < _MIN_TOKENS_FOR_SUMMARY:
        return None

    files_modified: list[str] = []
    files_read: list[str] = []
    test_results: list[str] = []
    key_errors: list[str] = []
    shell_commands: list[str] = []

    for msg in messages:
        content = msg.get("content", "") or ""
        role = msg.get("role", "")

        # Parse tool call messages (assistant role, JSON format)
        if role == "assistant" and content.startswith("{"):
            try:
                import json
                data = json.loads(content)
                tool = data.get("tool", "")
                args = data.get("args", {})
                if tool in ("write_file", "edit_file"):
                    path = args.get("path", "")
                    if path:
                        files_modified.append(_short_path(path))
                elif tool == "read_file":
                    path = args.get("path", "")
                    if path:
                        files_read.append(_short_path(path))
                elif tool == "shell":
                    cmd = args.get("command", "")
                    if cmd:
                        shell_commands.append(cmd[:60])
            except (json.JSONDecodeError, AttributeError):
                pass

        # Parse tool result messages
        if role == "user" and "[Tool Result" in content:
            # Extract test results
            if "passed" in content or "failed" in content:
                match = re.search(
                    r"(\d+)\s*passed.*?(\d+)\s*failed", content)
                if match:
                    test_results.append(
                        f"{match.group(1)} passed, {match.group(2)} failed")
                else:
                    # Simple pass/fail detection
                    if "FAIL" in content[:200]:
                        test_results.append("tests failed")
                    elif "passed" in content[:200]:
                        test_results.append("tests passed")

            # Extract error patterns
            for error_type in ("Error:", "Exception:", "Traceback"):
                if error_type in content:
                    # Grab the error line
                    for line in content.split("\n"):
                        if error_type in line:
                            key_errors.append(line.strip()[:100])
                            break
                    break

    # Build summary
    parts = ["[Context Summary]"]
    if files_modified:
        unique = sorted(set(files_modified))[:5]
        parts.append(f"Files modified: {', '.join(unique)}")
    if files_read:
        unique = sorted(set(files_read))[:5]
        parts.append(f"Files read: {', '.join(unique)}")
    if test_results:
        parts.append(f"Test results: {'; '.join(test_results[:3])}")
    if key_errors:
        parts.append(f"Errors: {'; '.join(key_errors[:3])}")
    if shell_commands:
        parts.append(f"Commands run: {len(shell_commands)}")

    if len(parts) <= 1:
        return None

    summary = "\n".join(parts)
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS] + "..."
    return summary


def _short_path(path: str) -> str:
    """Shorten a file path for summary display."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) > 3:
        return "/".join(parts[-3:])
    return path
