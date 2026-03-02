#!/usr/bin/env python3
"""Create the taskflow-hub challenge project for comprehensive harness testing.

Usage:
    python tests/fixtures/create_challenge_project.py [target_dir]
    uv run python tests/fixtures/create_challenge_project.py [target_dir]

Default target: /tmp/taskflow-hub/

This creates a Python + TypeScript task management library with intentional
bugs and a security vulnerability. Designed to exercise all harness features
across four escalating goal levels.

Bugs planted:
    BUG1  models.py     tags=None → TypeError, description=None → AttributeError
    BUG2  engine.py     range(len-1) off-by-one — last task skipped
    BUG3  schema.ts     Regex rejects valid chars like ' . : in task titles
    SEC1  config.py     Hardcoded API key (sk-proj-...) and DB password

Expected test failures:
    pytest  → 4 FAIL  (2 in test_models.py, 2 in test_engine.py)
    vitest  → 1 FAIL  (schema.test.ts)
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

DEFAULT_TARGET = "/tmp/taskflow-hub"

# ──────────────────────────────────────────────
# Python project files
# ──────────────────────────────────────────────

PYPROJECT_TOML = """\
[project]
name = "taskflow-hub"
version = "0.1.0"
description = "A lightweight task management library"
requires-python = ">=3.10"

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
"""

PACKAGE_JSON = """\
{
  "name": "taskflow-hub-validators",
  "version": "0.1.0",
  "private": true,
  "description": "TypeScript validators for taskflow-hub",
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "vitest": "^1.6.0"
  }
}
"""

TSCONFIG_JSON = """\
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "outDir": "dist",
    "rootDir": "src",
    "declaration": true
  },
  "include": ["src/**/*.ts"],
  "exclude": ["node_modules", "dist"]
}
"""

VITEST_CONFIG_TS = """\
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
"""

GITIGNORE = """\
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/

# Node
node_modules/
dist/

# IDE
.vscode/
.idea/

# OS
.DS_Store
"""

# ──────────────────────────────────────────────
# Python source files
# ──────────────────────────────────────────────

INIT_PY = """\
\"\"\"taskflow-hub: A lightweight task management library.\"\"\"
__version__ = "0.1.0"
"""

MODELS_PY = """\
\"\"\"Task model definitions.\"\"\"

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum


class Priority(IntEnum):
    \"\"\"Task priority levels with numeric scores.\"\"\"
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    \"\"\"Represents a single task.\"\"\"
    title: str
    priority: Priority = Priority.MEDIUM
    tags: list[str] | None = None
    description: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False

    def tag_count(self) -> int:
        \"\"\"Return the number of tags on this task.\"\"\"
        # BUG1a: tags can be None — calling len(None) raises TypeError
        return len(self.tags)

    def summary(self) -> str:
        \"\"\"Return a one-line summary: title + first 30 chars of description.\"\"\"
        # BUG1b: description can be None — .strip() on None raises AttributeError
        short = self.description.strip()[:30]
        return f"[{self.priority.name}] {self.title}: {short}"

    def score(self) -> float:
        \"\"\"Calculate priority score (higher = more urgent).

        Score = priority_value * 10 + tag_bonus
        tag_bonus = 2 per tag (capped at 10)
        \"\"\"
        tag_bonus = min(len(self.tags or []) * 2, 10)
        return float(self.priority.value * 10 + tag_bonus)
"""

ENGINE_PY = """\
\"\"\"Task processing engine.\"\"\"

from __future__ import annotations

from .models import Task, Priority


class TaskEngine:
    \"\"\"Manages a collection of tasks.\"\"\"

    def __init__(self) -> None:
        self._tasks: list[Task] = []

    def add(self, task: Task) -> None:
        \"\"\"Add a task to the engine.\"\"\"
        self._tasks.append(task)

    def get_all(self) -> list[Task]:
        \"\"\"Return all tasks.\"\"\"
        return list(self._tasks)

    def count(self) -> int:
        \"\"\"Return number of tasks.\"\"\"
        return len(self._tasks)

    def get_by_priority(self, priority: Priority) -> list[Task]:
        \"\"\"Return tasks matching the given priority.\"\"\"
        return [t for t in self._tasks if t.priority == priority]

    def complete_all(self) -> int:
        \"\"\"Mark all tasks as completed. Returns count of newly completed.\"\"\"
        count = 0
        # BUG2: Off-by-one — range(len-1) skips the last task
        for i in range(len(self._tasks) - 1):
            if not self._tasks[i].completed:
                self._tasks[i].completed = True
                count += 1
        return count

    def complete_by_priority(self, min_priority: Priority) -> int:
        \"\"\"Complete tasks with priority >= min_priority. Returns count.\"\"\"
        count = 0
        # BUG2 (same pattern): Off-by-one — skips last task
        for i in range(len(self._tasks) - 1):
            if not self._tasks[i].completed and self._tasks[i].priority >= min_priority:
                self._tasks[i].completed = True
                count += 1
        return count

    def pending_tasks(self) -> list[Task]:
        \"\"\"Return tasks that are not yet completed.\"\"\"
        return [t for t in self._tasks if not t.completed]

    def sorted_by_score(self) -> list[Task]:
        \"\"\"Return tasks sorted by score (descending).\"\"\"
        return sorted(self._tasks, key=lambda t: t.score(), reverse=True)
"""

CONFIG_PY = """\
\"\"\"Application configuration.\"\"\"

from __future__ import annotations

import os

# SEC1: Hardcoded secrets — these should come from environment variables
API_KEY = "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234"
DB_PASSWORD = "password123"
DB_HOST = "db.internal.example.com"

# These are fine — not secrets
APP_NAME = "taskflow-hub"
MAX_TASKS = 10000
DEFAULT_PRIORITY = "MEDIUM"


def get_api_key() -> str:
    \"\"\"Return the API key.\"\"\"
    # SEC1: Should use os.environ.get() with the hardcoded value as fallback,
    #       or better yet, raise an error if not set.
    return API_KEY


def get_db_url() -> str:
    \"\"\"Build the database connection URL.\"\"\"
    # SEC1: Exposes password in connection string
    return f"postgresql://admin:{DB_PASSWORD}@{DB_HOST}:5432/taskflow"


def get_config() -> dict:
    \"\"\"Return full application config.\"\"\"
    return {
        "app_name": APP_NAME,
        "max_tasks": MAX_TASKS,
        "default_priority": DEFAULT_PRIORITY,
        "api_key": get_api_key(),
        "db_url": get_db_url(),
    }
"""

# ──────────────────────────────────────────────
# TypeScript source files
# ──────────────────────────────────────────────

SCHEMA_TS = """\
/**
 * Task schema validation for the TypeScript side.
 */

/** Valid priority levels — must match Python's Priority enum values. */
export const VALID_PRIORITIES = [1, 2, 3, 4] as const;
export type PriorityLevel = (typeof VALID_PRIORITIES)[number];

/** Validate a task title string. */
export function validateTitle(title: string): { valid: boolean; error?: string } {
  if (!title || title.trim().length === 0) {
    return { valid: false, error: "Title must not be empty" };
  }
  if (title.length > 200) {
    return { valid: false, error: "Title must be 200 characters or fewer" };
  }
  // BUG3: Regex is too strict — rejects apostrophes, periods, colons, etc.
  //       Should allow common punctuation in task titles.
  const validPattern = /^[a-zA-Z0-9\\s\\-_]+$/;
  if (!validPattern.test(title)) {
    return { valid: false, error: "Title contains invalid characters" };
  }
  return { valid: true };
}

/** Validate a priority value. */
export function validatePriority(
  priority: number
): { valid: boolean; error?: string } {
  if (!Number.isInteger(priority)) {
    return { valid: false, error: "Priority must be an integer" };
  }
  if (!(VALID_PRIORITIES as readonly number[]).includes(priority)) {
    return {
      valid: false,
      error: `Priority must be one of: ${VALID_PRIORITIES.join(", ")}`,
    };
  }
  return { valid: true };
}

/** Validate a list of tags. */
export function validateTags(
  tags: string[]
): { valid: boolean; error?: string } {
  if (!Array.isArray(tags)) {
    return { valid: false, error: "Tags must be an array" };
  }
  for (const tag of tags) {
    if (typeof tag !== "string" || tag.trim().length === 0) {
      return { valid: false, error: "Each tag must be a non-empty string" };
    }
    if (tag.length > 50) {
      return { valid: false, error: "Each tag must be 50 characters or fewer" };
    }
  }
  return { valid: true };
}
"""

PRIORITY_TS = """\
/**
 * Priority scoring — mirrors Python's Task.score() for cross-language consistency.
 */

export type PriorityLevel = 1 | 2 | 3 | 4;

export const PRIORITY_NAMES: Record<PriorityLevel, string> = {
  1: "LOW",
  2: "MEDIUM",
  3: "HIGH",
  4: "CRITICAL",
};

/**
 * Calculate priority score — must match Python's Task.score() exactly.
 *
 * Score = priority_value * 10 + tag_bonus
 * tag_bonus = 2 per tag (capped at 10)
 */
export function calculateScore(
  priority: PriorityLevel,
  tagCount: number
): number {
  const tagBonus = Math.min(tagCount * 2, 10);
  return priority * 10 + tagBonus;
}

/**
 * Compare two tasks by score (for sorting).
 * Returns negative if a should come first (higher score).
 */
export function compareByScore(
  a: { priority: PriorityLevel; tagCount: number },
  b: { priority: PriorityLevel; tagCount: number }
): number {
  return calculateScore(b.priority, b.tagCount) - calculateScore(a.priority, a.tagCount);
}
"""

# ──────────────────────────────────────────────
# Python test files
# ──────────────────────────────────────────────

TEST_MODELS_PY = """\
\"\"\"Tests for task models.\"\"\"

from taskflow.models import Task, Priority


def test_create_task_defaults():
    \"\"\"Basic task creation with defaults.\"\"\"
    task = Task(title="Write docs")
    assert task.title == "Write docs"
    assert task.priority == Priority.MEDIUM
    assert task.completed is False


def test_create_task_with_tags():
    \"\"\"Task with explicit tags.\"\"\"
    task = Task(title="Deploy", tags=["ops", "urgent"])
    assert task.tags == ["ops", "urgent"]
    assert task.tag_count() == 2


def test_tag_count_no_tags():
    \"\"\"tag_count() should return 0 when tags is None.

    BUG1a: This FAILS — tags=None causes TypeError in len(None).
    \"\"\"
    task = Task(title="Simple task")
    assert task.tag_count() == 0


def test_summary_no_description():
    \"\"\"summary() should handle None description gracefully.

    BUG1b: This FAILS — description=None causes AttributeError on .strip().
    \"\"\"
    task = Task(title="Quick fix", priority=Priority.HIGH)
    result = task.summary()
    assert "Quick fix" in result
    assert "HIGH" in result


def test_score_basic():
    \"\"\"Score calculation with tags.\"\"\"
    task = Task(title="Test", priority=Priority.HIGH, tags=["a", "b"])
    # HIGH=3 → 3*10 + min(2*2, 10) = 30 + 4 = 34
    assert task.score() == 34.0


def test_score_no_tags():
    \"\"\"Score calculation with no tags.\"\"\"
    task = Task(title="Test", priority=Priority.LOW)
    # LOW=1 → 1*10 + 0 = 10
    assert task.score() == 10.0
"""

TEST_ENGINE_PY = """\
\"\"\"Tests for the task engine.\"\"\"

from taskflow.models import Task, Priority
from taskflow.engine import TaskEngine


def test_add_and_count():
    \"\"\"Add tasks and check count.\"\"\"
    engine = TaskEngine()
    engine.add(Task(title="Task 1"))
    engine.add(Task(title="Task 2"))
    assert engine.count() == 2


def test_get_by_priority():
    \"\"\"Filter tasks by priority.\"\"\"
    engine = TaskEngine()
    engine.add(Task(title="Low task", priority=Priority.LOW))
    engine.add(Task(title="High task", priority=Priority.HIGH))
    engine.add(Task(title="High task 2", priority=Priority.HIGH))

    high = engine.get_by_priority(Priority.HIGH)
    assert len(high) == 2


def test_complete_all():
    \"\"\"complete_all() should mark ALL tasks as completed.

    BUG2: This FAILS — off-by-one skips the last task.
    \"\"\"
    engine = TaskEngine()
    engine.add(Task(title="Task A"))
    engine.add(Task(title="Task B"))
    engine.add(Task(title="Task C"))

    completed = engine.complete_all()
    assert completed == 3
    assert len(engine.pending_tasks()) == 0


def test_complete_by_priority():
    \"\"\"complete_by_priority() should complete matching tasks including the last one.

    BUG2: This FAILS — off-by-one skips the last task.
    \"\"\"
    engine = TaskEngine()
    engine.add(Task(title="Low", priority=Priority.LOW))
    engine.add(Task(title="High 1", priority=Priority.HIGH))
    engine.add(Task(title="High 2", priority=Priority.HIGH))

    completed = engine.complete_by_priority(Priority.HIGH)
    assert completed == 2
    # Both HIGH tasks should be completed, LOW should remain
    pending = engine.pending_tasks()
    assert len(pending) == 1
    assert pending[0].title == "Low"


def test_sorted_by_score():
    \"\"\"Tasks should be sorted by score descending.\"\"\"
    engine = TaskEngine()
    engine.add(Task(title="Low", priority=Priority.LOW))
    engine.add(Task(title="Critical", priority=Priority.CRITICAL, tags=["a"]))
    engine.add(Task(title="Medium", priority=Priority.MEDIUM))

    ordered = engine.sorted_by_score()
    assert ordered[0].title == "Critical"
    assert ordered[-1].title == "Low"
"""

# ──────────────────────────────────────────────
# TypeScript test files
# ──────────────────────────────────────────────

SCHEMA_TEST_TS = """\
import { describe, it, expect } from "vitest";
import { validateTitle, validatePriority, validateTags } from "../../src/validators/schema";

describe("validateTitle", () => {
  it("accepts simple alphanumeric titles", () => {
    expect(validateTitle("Fix login bug")).toEqual({ valid: true });
  });

  it("accepts titles with hyphens and underscores", () => {
    expect(validateTitle("task-123_update")).toEqual({ valid: true });
  });

  it("rejects empty titles", () => {
    const result = validateTitle("");
    expect(result.valid).toBe(false);
  });

  it("rejects titles over 200 chars", () => {
    const result = validateTitle("a".repeat(201));
    expect(result.valid).toBe(false);
  });

  it("accepts titles with common punctuation", () => {
    // BUG3: This FAILS — regex rejects apostrophes, periods, colons
    //       Real task titles often contain these characters.
    const result = validateTitle("Fix the user's login: v2.0 update");
    expect(result.valid).toBe(true);
  });
});

describe("validatePriority", () => {
  it("accepts valid priority values", () => {
    expect(validatePriority(1)).toEqual({ valid: true });
    expect(validatePriority(4)).toEqual({ valid: true });
  });

  it("rejects invalid priority values", () => {
    expect(validatePriority(0).valid).toBe(false);
    expect(validatePriority(5).valid).toBe(false);
  });

  it("rejects non-integer values", () => {
    expect(validatePriority(1.5).valid).toBe(false);
  });
});

describe("validateTags", () => {
  it("accepts valid tag arrays", () => {
    expect(validateTags(["bug", "urgent"])).toEqual({ valid: true });
  });

  it("rejects empty string tags", () => {
    expect(validateTags(["valid", ""]).valid).toBe(false);
  });

  it("rejects tags over 50 chars", () => {
    expect(validateTags(["a".repeat(51)]).valid).toBe(false);
  });
});
"""


def create_project(target: str | Path) -> Path:
    """Create the taskflow-hub challenge project. Returns the project root."""
    root = Path(target)

    # Clean existing
    if root.exists():
        import shutil
        shutil.rmtree(root)

    # Create directory structure
    (root / "src" / "taskflow").mkdir(parents=True)
    (root / "src" / "validators").mkdir(parents=True)
    (root / "tests" / "validators").mkdir(parents=True)

    # Define all files
    files = {
        "pyproject.toml": PYPROJECT_TOML,
        "package.json": PACKAGE_JSON,
        "tsconfig.json": TSCONFIG_JSON,
        "vitest.config.ts": VITEST_CONFIG_TS,
        ".gitignore": GITIGNORE,
        "src/taskflow/__init__.py": INIT_PY,
        "src/taskflow/models.py": MODELS_PY,
        "src/taskflow/engine.py": ENGINE_PY,
        "src/taskflow/config.py": CONFIG_PY,
        "src/validators/schema.ts": SCHEMA_TS,
        "src/validators/priority.ts": PRIORITY_TS,
        "tests/test_models.py": TEST_MODELS_PY,
        "tests/test_engine.py": TEST_ENGINE_PY,
        "tests/validators/schema.test.ts": SCHEMA_TEST_TS,
    }

    # Write all files
    for rel_path, content in files.items():
        p = root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(content))

    # Initialize git repo with initial commit (needed for checkpoint/snapshot)
    subprocess.run(["git", "init"], cwd=root, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit: taskflow-hub with known issues"],
        cwd=root,
        capture_output=True,
        env={**__import__("os").environ, "GIT_AUTHOR_NAME": "Challenge Bot",
             "GIT_AUTHOR_EMAIL": "bot@example.com",
             "GIT_COMMITTER_NAME": "Challenge Bot",
             "GIT_COMMITTER_EMAIL": "bot@example.com"},
    )

    print(f"Created challenge project at: {root}")
    print(f"  Files: {len(files)}")
    print(f"  Python bugs: 2 (BUG1 in models.py, BUG2 in engine.py)")
    print(f"  TypeScript bugs: 1 (BUG3 in schema.ts)")
    print(f"  Security issues: 1 (SEC1 in config.py)")
    print(f"  Git: initialized with initial commit")
    print()
    print("Expected test failures:")
    print(f"  cd {root}")
    print(f"  pytest tests/ -v          → 4 FAIL (test_models: 2, test_engine: 2)")
    print(f"  npm install && npm test   → 1 FAIL (schema.test.ts: punctuation)")
    print()
    print("Goal prompts (escalating difficulty):")
    print('  G1: "Fix all failing Python tests."')
    print('  G2: "Fix all failing tests (Python + TypeScript)."')
    print('  G3: "Fix all bugs and security vulnerabilities. Commit changes."')
    print('  G4: "Complete quality pass: fix tests, security, add edge-case tests, commit."')

    return root


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    create_project(target)
