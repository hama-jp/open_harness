#!/usr/bin/env python3
"""Create a small Python project with intentional bugs for harness testing.

Usage:
    python tests/fixtures/create_test_project.py [target_dir]

Default target: /tmp/test_harness_project/

Bugs planted:
    1. core.py: divide() has no zero-division guard
    2. history.py: _entries defaults to None instead of []
    3. history.py: get_last() crashes on empty (None) history
    4. history.py: average_result() off-by-one (len + 1)
    5. formatter.py: format_table() crashes on empty list
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

DEFAULT_TARGET = "/tmp/test_harness_project"

# ──────────────────────────────────────────────
# File contents
# ──────────────────────────────────────────────

PYPROJECT_TOML = """\
[project]
name = "calculator"
version = "0.1.0"
requires-python = ">=3.10"

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
"""

INIT_PY = """\
\"\"\"Calculator package.\"\"\"
__version__ = "0.1.0"
"""

CORE_PY = """\
\"\"\"Core calculator operations.\"\"\"


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    # BUG 1: No zero-division guard
    return a / b
"""

HISTORY_PY = """\
\"\"\"Calculation history tracker.\"\"\"

from __future__ import annotations


class History:
    \"\"\"Records calculation results.\"\"\"

    def __init__(self):
        # BUG 2: Should be [] not None
        self._entries = None

    def record(self, expression: str, result: float) -> None:
        \"\"\"Add a calculation to history.\"\"\"
        if self._entries is None:
            self._entries = []
        self._entries.append({"expression": expression, "result": result})

    def get_all(self) -> list[dict]:
        \"\"\"Return all history entries.\"\"\"
        # BUG 2 (surface): returns None if record() was never called
        return self._entries

    def get_last(self, n: int = 1) -> list[dict]:
        \"\"\"Return the last N entries.\"\"\"
        # BUG 3: Crashes if _entries is None (no None check)
        return self._entries[-n:]

    def clear(self) -> None:
        \"\"\"Clear all history.\"\"\"
        self._entries = []

    def count(self) -> int:
        \"\"\"Return number of entries.\"\"\"
        if self._entries is None:
            return 0
        return len(self._entries)

    def average_result(self) -> float:
        \"\"\"Return average of all results.\"\"\"
        if not self._entries:
            return 0.0
        total = sum(e["result"] for e in self._entries)
        # BUG 4: Off-by-one — divides by len + 1 instead of len
        return total / (len(self._entries) + 1)
"""

FORMATTER_PY = """\
\"\"\"Output formatting utilities.\"\"\"

from __future__ import annotations


def format_result(expression: str, result: float) -> str:
    \"\"\"Format a single calculation result.\"\"\"
    return f"{expression} = {result}"


def format_table(entries: list[dict]) -> str:
    \"\"\"Format history entries as a text table.\"\"\"
    # BUG 5: No guard for empty list — header uses entries[0] keys
    headers = list(entries[0].keys())
    widths = [max(len(h), max(len(str(e[h])) for e in entries)) for h in headers]

    lines = []
    # Header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append("-+-".join("-" * w for w in widths))

    # Rows
    for entry in entries:
        row = " | ".join(str(entry[h]).ljust(w) for h, w in zip(headers, widths))
        lines.append(row)

    return "\\n".join(lines)
"""

TEST_CORE_PY = """\
\"\"\"Tests for core calculator operations.\"\"\"

from calculator.core import add, subtract, multiply, divide


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5


def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 100) == 0
    assert multiply(-2, 3) == -6


def test_divide():
    assert divide(10, 2) == 5.0
    assert divide(7, 2) == 3.5


def test_divide_by_zero():
    \"\"\"This test SHOULD pass but will FAIL due to BUG 1.\"\"\"
    try:
        divide(10, 0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # expected
"""

TEST_HISTORY_PY = """\
\"\"\"Tests for calculation history.\"\"\"

from calculator.history import History


def test_record_and_get():
    h = History()
    h.record("1+1", 2)
    h.record("2*3", 6)
    entries = h.get_all()
    assert len(entries) == 2
    assert entries[0]["result"] == 2


def test_count():
    h = History()
    assert h.count() == 0
    h.record("1+1", 2)
    assert h.count() == 1


def test_clear():
    h = History()
    h.record("1+1", 2)
    h.clear()
    assert h.count() == 0
"""


def create_project(target: str | Path) -> Path:
    """Create the test project at the given path. Returns the project root."""
    root = Path(target)

    # Clean existing
    if root.exists():
        import shutil
        shutil.rmtree(root)

    # Create directory structure
    (root / "src" / "calculator").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)

    # Write files
    files = {
        "pyproject.toml": PYPROJECT_TOML,
        "src/calculator/__init__.py": INIT_PY,
        "src/calculator/core.py": CORE_PY,
        "src/calculator/history.py": HISTORY_PY,
        "src/calculator/formatter.py": FORMATTER_PY,
        "tests/test_core.py": TEST_CORE_PY,
        "tests/test_history.py": TEST_HISTORY_PY,
    }

    for rel_path, content in files.items():
        p = root / rel_path
        p.write_text(textwrap.dedent(content))

    print(f"Created test project at: {root}")
    print(f"  Files: {len(files)}")
    print(f"  Bugs: 5 (see docstring for details)")
    print()
    print("To verify bugs:")
    print(f"  cd {root}")
    print(f"  PYTHONPATH=src python -m pytest tests/ -v")

    return root


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    create_project(target)
