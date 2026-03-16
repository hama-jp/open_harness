"""Async external agent tools — Codex CLI, Claude Code, Gemini CLI.

Each tool delegates a task to an external CLI agent via
asyncio.create_subprocess_exec, with timeout handling and
process-tree cleanup.

Results are parsed into a structured format with standard fields:
summary, changed_files, tests, risks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import shutil
from typing import Any

from open_harness_v2.tools.base import Tool
from open_harness_v2.types import ToolParameter, ToolResult

_logger = logging.getLogger(__name__)

# Default timeout (seconds) for external agent calls.
_DEFAULT_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Structured result parsing
# ---------------------------------------------------------------------------

def _extract_structured_result(raw_output: str) -> dict[str, Any]:
    """Extract structured fields from external agent output.

    Attempts to find JSON blocks in the output first. If none found,
    uses heuristics to extract summary, changed files, test results,
    and potential risks.

    Returns a dict with standard keys: summary, changed_files, tests, risks.
    """
    result: dict[str, Any] = {
        "summary": "",
        "changed_files": [],
        "tests": {"passed": None, "failed": None, "details": ""},
        "risks": [],
        "raw_length": len(raw_output),
    }

    if not raw_output:
        return result

    # 1. Try to find an embedded JSON block
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", raw_output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, dict):
                for key in ("summary", "changed_files", "tests", "risks"):
                    if key in parsed:
                        result[key] = parsed[key]
                return result
        except json.JSONDecodeError:
            pass

    # 2. Heuristic extraction from plain text

    # Summary: first non-empty paragraph or first 3 sentences
    lines = raw_output.strip().split("\n")
    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if summary_lines:
                break
            continue
        summary_lines.append(stripped)
        if len(summary_lines) >= 3:
            break
    result["summary"] = " ".join(summary_lines)[:500]

    # Changed files: look for file path patterns
    file_patterns = re.findall(
        r"(?:^|\s)([\w./\-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|rb|yaml|yml|json|toml|md|txt|css|html|sh))\b",
        raw_output,
    )
    # Deduplicate preserving order
    seen: set[str] = set()
    for fp in file_patterns:
        if fp not in seen and not fp.startswith("http"):
            seen.add(fp)
            result["changed_files"].append(fp)

    # Test results: look for common test output patterns
    test_match = re.search(
        r"(\d+)\s+(?:tests?\s+)?passed", raw_output, re.I
    )
    if test_match:
        result["tests"]["passed"] = int(test_match.group(1))
    fail_match = re.search(
        r"(\d+)\s+(?:tests?\s+)?failed", raw_output, re.I
    )
    if fail_match:
        result["tests"]["failed"] = int(fail_match.group(1))

    # Risks: look for warning/risk indicators
    risk_patterns = [
        re.compile(r"(?:warning|caution|risk|breaking change|deprecated)[:\s]+(.*)", re.I),
        re.compile(r"(?:TODO|FIXME|HACK|XXX)[:\s]+(.*)", re.I),
    ]
    for pat in risk_patterns:
        for match in pat.finditer(raw_output):
            risk_text = match.group(1).strip()[:200]
            if risk_text:
                result["risks"].append(risk_text)

    return result


def _format_structured_output(raw_output: str, structured: dict[str, Any]) -> str:
    """Format the raw output with appended structured metadata."""
    parts = [raw_output.rstrip()]

    metadata_parts: list[str] = []
    if structured.get("summary"):
        metadata_parts.append(f"Summary: {structured['summary']}")
    if structured.get("changed_files"):
        files_str = ", ".join(structured["changed_files"][:20])
        metadata_parts.append(f"Changed files: {files_str}")
    tests = structured.get("tests", {})
    if tests.get("passed") is not None or tests.get("failed") is not None:
        test_parts = []
        if tests.get("passed") is not None:
            test_parts.append(f"{tests['passed']} passed")
        if tests.get("failed") is not None:
            test_parts.append(f"{tests['failed']} failed")
        metadata_parts.append(f"Tests: {', '.join(test_parts)}")
    if structured.get("risks"):
        metadata_parts.append(f"Risks: {'; '.join(structured['risks'][:5])}")

    if metadata_parts:
        parts.append("\n--- Structured Result ---")
        for mp in metadata_parts:
            parts.append(mp)

    return "\n".join(parts)


async def _run_external(
    cmd: list[str],
    *,
    cwd: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    tool_label: str = "external agent",
) -> ToolResult:
    """Run an external CLI agent asynchronously with timeout handling.

    On timeout the entire process tree is killed and any partial
    output captured so far is returned.
    """
    try:
        # Start a new process group so the entire tree can be killed
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
    except FileNotFoundError:
        return ToolResult(
            success=False, output="",
            error=f"Command not found: {cmd[0]}",
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        _kill_process_tree(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        return ToolResult(
            success=False,
            output="",
            error=f"{tool_label} timed out after {timeout}s",
        )

    output = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    if stderr_text:
        output += f"\n[stderr]\n{stderr_text}" if output else stderr_text

    returncode = proc.returncode or 0
    raw_output = output.strip()

    # Parse structured result from raw output
    structured = _extract_structured_result(raw_output)
    formatted_output = _format_structured_output(raw_output, structured)

    return ToolResult(
        success=returncode == 0,
        output=formatted_output,
        error="" if returncode == 0 else f"Exit code: {returncode}",
        metadata={
            "returncode": returncode,
            "structured": structured,
        },
    )


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a process and its entire process group."""
    try:
        if os.name != "nt" and proc.pid is not None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, OSError):
        pass  # already dead


# ------------------------------------------------------------------
# Tool classes
# ------------------------------------------------------------------

class CodexTool(Tool):
    """Delegate tasks to OpenAI Codex CLI."""

    name = "codex"
    max_output = 5000
    description = (
        "Delegate a coding task to OpenAI Codex CLI agent. "
        "Best for complex code generation, refactoring, and debugging tasks. "
        "Codex has its own sandbox and can read/write files."
    )
    parameters = [
        ToolParameter(
            name="prompt",
            type="string",
            description="The task description to send to Codex",
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory for Codex",
            required=False,
        ),
    ]

    def __init__(self, command: str = "codex", timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.command = command
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.command) is not None
        return self._available

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self.available:
            return ToolResult(
                success=False, output="",
                error=f"'{self.command}' not found in PATH",
            )
        prompt = kwargs.get("prompt", "")
        cwd = kwargs.get("cwd")
        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        return await _run_external(
            [self.command, "exec", "--full-auto", prompt],
            cwd=cwd,
            timeout=self.timeout,
            tool_label="Codex",
        )


class ClaudeCodeTool(Tool):
    """Delegate tasks to Claude Code (Anthropic CLI)."""

    name = "claude_code"
    max_output = 5000
    description = (
        "Delegate a coding task to Claude Code (Anthropic CLI agent). "
        "Best for code generation, code analysis, complex reasoning, and refactoring. "
        "Claude Code has its own sandbox and can read/write files."
    )
    parameters = [
        ToolParameter(
            name="prompt",
            type="string",
            description="The task description to send to Claude Code",
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory for Claude Code",
            required=False,
        ),
    ]

    def __init__(self, command: str = "claude", timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.command = command
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.command) is not None
        return self._available

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self.available:
            return ToolResult(
                success=False, output="",
                error=f"'{self.command}' not found in PATH",
            )
        prompt = kwargs.get("prompt", "")
        cwd = kwargs.get("cwd")
        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        return await _run_external(
            [
                self.command, "-p", prompt,
                "--allowedTools", "Bash", "Read", "Write", "Edit",
                "Glob", "Grep",
            ],
            cwd=cwd,
            timeout=self.timeout,
            tool_label="Claude Code",
        )


class GeminiCliTool(Tool):
    """Delegate tasks to Google Gemini CLI."""

    name = "gemini_cli"
    max_output = 5000
    description = (
        "Delegate a task to Google Gemini CLI agent. "
        "Useful for tasks that benefit from Gemini's capabilities."
    )
    parameters = [
        ToolParameter(
            name="prompt",
            type="string",
            description="The task description to send to Gemini CLI",
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory",
            required=False,
        ),
    ]

    def __init__(self, command: str = "gemini", timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.command = command
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = shutil.which(self.command) is not None
        return self._available

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self.available:
            return ToolResult(
                success=False, output="",
                error=f"'{self.command}' not found in PATH",
            )
        prompt = kwargs.get("prompt", "")
        cwd = kwargs.get("cwd")
        if not prompt:
            return ToolResult(success=False, output="", error="No prompt provided")

        return await _run_external(
            [self.command, "-p", prompt, "-y"],
            cwd=cwd,
            timeout=self.timeout,
            tool_label="Gemini CLI",
        )
