# Open Harness Tutorial

A comprehensive guide to using Open Harness — a self-driving AI agent harness optimized for local LLMs.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Configuration](#2-configuration)
3. [Getting Started — Interactive Mode](#3-getting-started--interactive-mode)
4. [Autonomous Mode — /goal](#4-autonomous-mode--goal)
5. [Background Tasks — /submit](#5-background-tasks--submit)
6. [Tools Reference](#6-tools-reference)
7. [Model Tiers and Routing](#7-model-tiers-and-routing)
8. [Policy and Safety Guardrails](#8-policy-and-safety-guardrails)
9. [Planner-Critic-Executor Loop](#9-planner-critic-executor-loop)
10. [Checkpoint and Rollback](#10-checkpoint-and-rollback)
11. [Project Memory](#11-project-memory)
12. [External Agent Integration](#12-external-agent-integration)
13. [Per-Project Configuration](#13-per-project-configuration)
14. [Practical Examples](#14-practical-examples)
15. [Troubleshooting](#15-troubleshooting)
16. [@file References and Tab Completion](#16-file-references-and-tab-completion)
17. [Mode Switching](#17-mode-switching)
18. [Rate-Limit Fallback](#18-rate-limit-fallback)

---

## 1. Installation

### Requirements

- Python 3.11+
- A local LLM server (LM Studio, Ollama, or any OpenAI-compatible API)

### Install with uv (recommended)

```bash
git clone https://github.com/hama-jp/open_harness.git
cd open_harness
uv venv && source .venv/bin/activate
uv pip install -e .
```

### Install with pip

```bash
git clone https://github.com/hama-jp/open_harness.git
cd open_harness
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Verify installation

```bash
harness --help
```

---

## 2. Configuration

### Configuration file

Open Harness uses `open_harness.yaml` as its configuration file. The file is searched in this order:

1. Explicit path via `--config` option
2. Current working directory (`./open_harness.yaml`)
3. User config directory (`~/.open_harness/open_harness.yaml`)
4. Repository root (development mode)

> Legacy name `config.yaml` is also accepted for backward compatibility.

### Minimal configuration

```yaml
# open_harness.yaml
llm:
  default_provider: "lm_studio"
  providers:
    lm_studio:
      base_url: "http://localhost:1234/v1"
      api_key: "lm-studio"
  models:
    medium:
      provider: "lm_studio"
      model: "your-model-name"
      max_tokens: 8192
  default_tier: "medium"
```

### Full configuration reference

```yaml
# ── LLM ──
llm:
  default_provider: "lm_studio"    # or "ollama"

  providers:
    lm_studio:
      base_url: "http://192.168.11.3:1234/v1"
      api_key: "lm-studio"
    ollama:
      base_url: "http://localhost:11434/v1"
      api_key: "ollama"

  models:
    small:
      provider: "lm_studio"
      model: "unsloth/qwen3.5-35b-a3b@q4_k_m"
      max_tokens: 4096
      description: "Fast, simple tasks"
    medium:
      provider: "lm_studio"
      model: "unsloth/qwen3.5-35b-a3b@q4_k_m"
      max_tokens: 8192
      description: "Balanced performance"
    large:
      provider: "lm_studio"
      model: "unsloth/qwen3.5-35b-a3b@q4_k_m"
      max_tokens: 16384
      description: "Complex reasoning"

  default_tier: "medium"

# ── Compensation Engine ──
compensation:
  max_retries: 3
  retry_strategies:
    - "refine_prompt"       # Add correction hints and retry
    - "add_examples"        # Add tool-call examples to prompt
    - "escalate_model"      # Escalate to a larger model tier
  parse_fallback: true      # Extract tool calls from messy output
  thinking_mode: "auto"     # auto | always | never

# ── Tools ──
tools:
  shell:
    timeout: 30             # seconds
    allowed_commands: []    # empty = all allowed
    blocked_commands:
      - "rm -rf /"
      - "mkfs"
      - "dd if="
  file:
    max_read_size: 100000   # bytes

# ── External Agents ──
external_agents:
  claude:
    enabled: true
    command: "claude"
  codex:
    enabled: true
    command: "codex"
  gemini:
    enabled: true
    command: "gemini"

# ── Policy ──
policy:
  mode: "balanced"          # safe | balanced | full
  # writable_paths:         # extra dirs writable beyond project root
  #   - "/tmp/*"
  #   - "~/other-project/*"

# ── Memory ──
memory:
  backend: "sqlite"
  db_path: "~/.open_harness/memory.db"
  max_conversation_turns: 50
```

### Setting up Ollama

```yaml
llm:
  default_provider: "ollama"
  providers:
    ollama:
      base_url: "http://localhost:11434/v1"
      api_key: "ollama"
  models:
    medium:
      provider: "ollama"
      model: "qwen2.5:14b"
      max_tokens: 8192
```

### Setting up LM Studio (remote machine)

```yaml
llm:
  default_provider: "lm_studio"
  providers:
    lm_studio:
      base_url: "http://192.168.1.100:1234/v1"   # LAN IP of LM Studio host
      api_key: "lm-studio"
  models:
    medium:
      provider: "lm_studio"
      model: "unsloth/qwen3.5-35b-a3b@q4_k_m"
      max_tokens: 8192
```

---

## 3. Getting Started — Interactive Mode

### Launch

```bash
harness
```

On startup, Open Harness displays:

```
╭──────────────────────────────────────────────────╮
│ Open Harness v0.5.0                              │
│ Self-driving AI agent for local LLMs             │
│ Type /help for commands, /goal <task> for auto    │
╰──────────────────────────────────────────────────╯
Config: /home/you/project/open_harness.yaml
Project: python @ /home/you/project
Tests: python3 -m pytest
Model: unsloth/qwen3.5-35b-a3b@q4_k_m (medium)
Tools (14): shell, read_file, write_file, ...
Git: already initialized
Task queue: ready
```

> **Tip**: Press **Shift+Tab** to cycle between input modes (chat / goal / submit). See [Mode Switching](#17-mode-switching).

### Basic conversation

Simply type your message. The agent can use tools automatically:

```
> What files are in this project?

> list_dir .
OK list_dir
src/
tests/
pyproject.toml
README.md
...

This project contains a Python package with source code in src/,
tests in tests/, and standard project files.
```

### Ask about code

```
> Explain the main function in cli.py

> read_file src/open_harness/cli.py
OK read_file
(file contents shown)

The main function in cli.py sets up the CLI using Click...
```

### Request changes

```
> Add a docstring to the Agent class

> read_file src/open_harness/agent.py
> edit_file src/open_harness/agent.py ...
```

### Available REPL commands

Type `/help` to see all commands:

| Command | Description |
|---------|-------------|
| `/goal <task>` | Run autonomous mode on a goal |
| `/submit <task>` | Submit goal to background queue |
| `/tasks` | List all tasks and their status |
| `/result <id>` | Show detailed result of a task |
| `/model [tier]` | Show all model tiers with provider, host, and max_tokens; or switch tier |
| `/tier [name]` | Show or switch model tier |
| `/policy [mode]` | Show or set safety policy |
| `/tools` | List available tools |
| `/project` | Show detected project context |
| `/memory` | Show learned project memories |
| `/update` | Self-update Open Harness from git |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

---

## 4. Autonomous Mode — /goal

The `/goal` command activates full autonomous execution. The agent plans the task, executes step by step, uses checkpoints, and handles failures automatically.

### Basic usage

```
> /goal Add unit tests for the config module
```

The agent will:
1. Analyze the goal and create a multi-step plan
2. Execute each step sequentially
3. Take git snapshots after each successful step
4. Replan if a step fails
5. Fall back to direct execution if planning fails
6. Display a summary showing tool calls, compensations, rollbacks, and files modified

### Example: Bug fix

```
> /goal Fix the TypeError in utils.py line 42
```

Output (streaming):

```
thinking: Analyzing the error...
> read_file src/utils.py
OK read_file
(file contents)

thinking: The issue is that the function returns None when...
> edit_file src/utils.py ...
OK edit_file

> run_tests tests/test_utils.py
OK run_tests
3 passed

Goal completed in 28.3s
```

### Example: Feature implementation

```
> /goal Add a /version command that shows the current version from pyproject.toml
```

The agent will:
1. Plan: read pyproject.toml, modify cli.py, test
2. Read the current version string
3. Add the command handler
4. Verify it works
5. Report completion

### Limits and safety

- Max 50 agent steps per goal
- Max 3-8 plan steps (adaptive based on goal complexity)
- Checkpoint/rollback on failures (in git repos)
- Policy guardrails enforced throughout

---

## 5. Background Tasks — /submit

Submit long-running goals to execute in the background while you continue interacting with the agent.

### Submit a task

```
> /submit Refactor all test files to use pytest fixtures

Task a3f2e1b0 queued: Refactor all test files to use pytest fixtures
Log: ~/.open_harness/logs/task_1709312400_abc123.log
Check: /tasks | /result a3f2e1b0
```

### Check status

```
> /tasks

┌──────────┬──────────┬────────────────────────────────────────────┬────────┐
│ ID       │ Status   │ Goal                                       │ Time   │
├──────────┼──────────┼────────────────────────────────────────────┼────────┤
│ a3f2e1b0 │ running  │ Refactor all test files to use pytest...   │ 45s    │
│ 9c1d4e5f │ OK       │ Add logging to database module             │ 32s    │
└──────────┴──────────┴────────────────────────────────────────────┴────────┘
```

### View result

```
> /result a3f2e1b0

Task a3f2e1b0: Refactor all test files to use pytest fixtures
Status: succeeded
Time: 128.4s
╭── Result ──────────────────────────────────────────────╮
│ Refactored 5 test files to use pytest fixtures.        │
│ All 23 tests pass.                                     │
╰────────────────────────────────────────────────────────╯
Log: ~/.open_harness/logs/task_1709312400_abc123.log
```

### Key behaviors

- Tasks are processed one at a time (sequential FIFO queue)
- Each task gets an isolated Agent instance (no shared state)
- A terminal bell sounds when a task completes
- Completion notifications appear at the next prompt
- Tasks persist across restarts (crash recovery)
- You can continue chatting while background tasks run

---

## 6. Tools Reference

### File operations

| Tool | Description | Example |
|------|-------------|---------|
| `read_file` | Read file contents | `read_file("src/main.py")` |
| `write_file` | Create or overwrite a file | `write_file("new.py", "print('hello')")` |
| `edit_file` | Replace text in a file | `edit_file("main.py", "old_text", "new_text")` |
| `list_dir` | List directory contents | `list_dir("src/", "*.py")` |
| `search_files` | Search text in files (regex) | `search_files("TODO", "src/")` |

### Shell

| Tool | Description | Example |
|------|-------------|---------|
| `shell` | Execute shell commands | `shell("pip install requests")` |

Shell has safety defaults:
- 30-second timeout (configurable)
- Blocked: `rm -rf /`, `mkfs`, `dd if=`
- Policy-level blocking: `curl | sh`, `chmod 777`, etc.

### Git

| Tool | Description | Example |
|------|-------------|---------|
| `git_status` | Show modified/staged files | `git_status()` |
| `git_diff` | Show changes | `git_diff(staged=True)` |
| `git_commit` | Stage and commit | `git_commit("Fix bug", "src/main.py")` |
| `git_branch` | Create/list branches | `git_branch("feature/new")` |
| `git_log` | Show recent commits | `git_log(count=5)` |

### Testing

| Tool | Description | Example |
|------|-------------|---------|
| `run_tests` | Run project test suite | `run_tests("tests/test_config.py")` |

Test commands are auto-detected per project type:
- Python: `python3 -m pytest`
- Rust: `cargo test`
- JavaScript: `npm test`
- Go: `go test ./...`

### External agents

| Tool | Description | Example |
|------|-------------|---------|
| `claude_code` | Delegate to Claude Code (Anthropic) | `claude_code("Refactor this module")` |
| `codex` | Delegate to OpenAI Codex CLI | `codex("Generate a REST API client")` |
| `gemini_cli` | Delegate to Google Gemini CLI | `gemini_cli("Analyze this architecture")` |

> **Orchestrator architecture**: Open Harness uses the local LLM as an orchestrator for planning
> and coordination. Code generation, analysis, and debugging are delegated to external agents
> (Claude Code, Codex, Gemini CLI) which are far more capable at these tasks.

---

## 7. Model Tiers and Routing

Open Harness supports multiple model tiers to balance speed and capability.

### Tiers

| Tier | Typical use | Default max_tokens |
|------|------------|-------------------|
| `small` | Simple tasks, planning | 4096 |
| `medium` | General tasks (default) | 8192 |
| `large` | Complex reasoning | 16384 |

### View tier details at runtime

Use `/model` to see model name, provider, host, and max_tokens for every tier:

```
> /model
Model tiers:
  small:  qwen3.5-35b-a3b @ lm_studio (192.168.11.3) max_tokens=4096
  medium: qwen3.5-35b-a3b @ lm_studio (192.168.11.3) max_tokens=8192  *
  large:  qwen3.5-35b-a3b @ lm_studio (192.168.11.3) max_tokens=16384

> /model large
  large:  qwen3.5-35b-a3b @ lm_studio (192.168.11.3) max_tokens=16384  *
```

### Switch tiers at runtime

```
> /tier
  small: Fast, simple tasks
  medium: Balanced performance *
  large: Complex reasoning

> /tier large
Tier: large
```

### Launch with a specific tier

```bash
harness --tier large
```

### Automatic escalation

When the compensation engine detects repeated failures, it can automatically escalate to a larger tier:

```
small (fails) → medium (fails) → large
```

This is controlled by the `escalate_model` retry strategy.

### Using different models per tier

You can assign different models to each tier:

```yaml
models:
  small:
    provider: "ollama"
    model: "qwen2.5:7b"       # Fast, lightweight
    max_tokens: 4096
  medium:
    provider: "lm_studio"
    model: "qwen2.5:14b"      # Balanced
    max_tokens: 8192
  large:
    provider: "lm_studio"
    model: "qwen2.5:32b"      # Maximum capability
    max_tokens: 16384
```

---

## 8. Policy and Safety Guardrails

The policy engine provides automatic safety guardrails during autonomous execution.

### Policy presets

| Preset | File writes | Shell | Git commits | External calls |
|--------|------------|-------|-------------|----------------|
| `safe` | 20 | 30 | 3 | 10 |
| `balanced` | unlimited | unlimited | 10 | unlimited |
| `full` | unlimited | unlimited | unlimited | unlimited |

### Switch policy at runtime

```
> /policy
Policy mode: balanced
Budgets: git commits: 10, external calls: 5

> /policy safe
Policy switched to: safe
```

### View current policy

```
> /policy
Policy mode: safe
Budgets: file writes: 20, shell commands: 30, git commits: 3, external calls: 2
Denied paths: 8 patterns
Blocked shell: 7 patterns
```

### Write path restrictions

By default, `write_file` and `edit_file` are restricted to the **project root directory**. This prevents the agent from accidentally modifying files outside your project.

To allow writing to additional directories, use `writable_paths`:

```yaml
policy:
  writable_paths:
    - "/tmp/*"              # Allow writes to /tmp
    - "~/other-project/*"   # Allow writes to another project
```

The `full` preset allows writes to the entire home directory (`~/*`).

> **Note**: Read operations (`read_file`, `list_dir`, `search_files`) are **not** restricted by `writable_paths` — they can read any path not in `denied_paths`.

### Default denied paths

The following paths are blocked by default in all modes (both read and write):

- `/etc/*`, `/usr/*`, `/bin/*`, `/sbin/*`, `/boot/*`
- `~/.ssh/*`, `~/.gnupg/*`
- `**/.env`, `**/.env.*`
- `**/credentials*`, `**/secrets*`

### Default blocked shell patterns

- `curl | sh`, `wget | sh`
- `chmod 777`, `chmod -R 777`
- `> /dev/sd*`
- `git push --force`, `git reset --hard`

### Custom policy in config

```yaml
policy:
  mode: "balanced"
  max_file_writes: 50
  max_git_commits: 5
  writable_paths:
    - "/tmp/*"             # Allow writing to /tmp
  disabled_tools:
    - "shell"              # Disable shell entirely
  denied_paths:
    - "/home/user/secrets/*"
  blocked_shell_patterns:
    - "npm publish"
```

### How violations work

When the agent attempts a restricted action, it receives an error message (not a hard stop). The agent can then adapt its approach:

```
> shell rm -rf /tmp/build
VIOLATION: shell command matches blocked pattern: rm -rf /

thinking: That command is blocked. Let me use a safer approach...
> shell find /tmp/build -delete
OK shell
```

---

## 9. Planner-Critic-Executor Loop

For complex goals, Open Harness uses a plan-execute-verify loop.

### How it works

```
Goal
  │
  ▼
Planner ─── creates plan (up to 3-8 steps, adaptive)
  │
  ▼
Plan Critic ─── validates plan (rule-based)
  │
  ▼
Executor ─── executes each step
  │
  ├── Step 1 → snapshot ✓
  ├── Step 2 → snapshot ✓
  ├── Step 3 → FAIL
  │     │
  │     ▼
  │   Replanner ─── replan remaining steps (0-2 attempts)
  │     │
  │     ▼
  │   Continue execution...
  │
  ▼
Done (or fallback to direct mode)
```

### Plan structure

Each plan contains:
- **Goal**: Original task description
- **Steps**: Up to 3-8 sequential steps (adaptive), each with:
  - Title and detailed instruction
  - Success criteria (verifiable conditions)
  - Agent step budget (8-15 per step, adaptive)
- **Assumptions**: What the planner assumed about the environment

### Complexity-adaptive planning (v0.5.0)

The planner automatically estimates goal complexity (Low / Medium / High) and adjusts its behavior:

| Complexity | Max steps | Agent step budget | Replan attempts |
|------------|-----------|-------------------|-----------------|
| Low | 3 | 8 per step | 0 |
| Medium | 5 | 12 per step | 1 |
| High | 8 | 15 per step | 2 |

Simple goals like "fix typo" get fewer steps and a tighter budget, while complex goals like "refactor the database schema and migrate all data" get more room to work.

### Fallback behavior

The system is designed for graceful degradation:

1. If the planner can't create a plan → direct execution
2. If the critic rejects the plan → direct execution
3. If a step fails → replan (up to complexity-dependent limit), then fallback
4. If replanning fails → direct execution with completed context

This means `/goal` always attempts to complete the task, even with a weak LLM.

---

## 10. Checkpoint and Rollback

Checkpoints now work in **both interactive and goal modes**. In git repositories, Open Harness uses git-based checkpoints for safe, reversible execution.

### Git auto-initialization

If you open a project that is **not** a git repository, Open Harness automatically runs `git init` to enable checkpoint support. The startup display shows:

```
Git: already initialized     ← existing repo
Git: initialized (new)       ← auto-initialized by Open Harness
```

This ensures that checkpoints and rollback are always available, even in new projects.

### Interactive mode protection

In interactive (chat) mode, Open Harness creates a **session-level checkpoint** when the session starts. If the agent makes file changes during conversation that you want to undo, the session checkpoint provides a safe rollback point. This is lighter-weight than the full branch workflow used in goal mode.

### Automatic behavior

When you run `/goal`, the system automatically:

1. **Stashes** any uncommitted changes (safety net)
2. **Creates a work branch** (`harness/goal-<timestamp>`)
3. **Snapshots** after every 10 file writes and after each plan step
4. **Rolls back** if tests fail
5. **Squash-merges** all changes back to your branch on success
6. **Restores** your original stashed changes

### Example flow

```
> /goal Add input validation to the User model

[checkpoint] Stashed 2 uncommitted changes
[checkpoint] Created branch harness/goal-1709312400
...
[checkpoint] Snapshot: Added validation functions
...
[checkpoint] Snapshot: Updated model with validators
...
> run_tests
FAIL: test_user_email_validation
[checkpoint] Rolling back to: Added validation functions
...
(agent retries with different approach)
...
> run_tests
OK: 12 passed
[checkpoint] Squash-merged to main
[checkpoint] Restored stashed changes
```

### Safety guarantees

- Your uncommitted work is preserved via git stash
- Failed changes are automatically reverted
- Work branch is cleaned up after completion
- Original branch is never directly modified during execution

---

## 11. Project Memory

Open Harness automatically learns and remembers project-specific patterns across sessions.

### What it learns

| Kind | Example |
|------|---------|
| **pattern** | "Test command: pytest -x --tb=short" |
| **structure** | "Config files are in src/config/" |
| **error** | "ImportError → check virtual env activation" |
| **runbook** | Multi-step workflow recipes |

### View learned memories

```
> /memory

Learned memories:
  [P] test command: python -m pytest tests/ (score:0.8 seen:5)
  [S] source code lives in src/open_harness/ (score:0.6 seen:3)
  [E] ModuleNotFoundError: check .venv activation (score:0.7 seen:2)

Runbooks:
  Deploy to staging (3 uses, 2 ok)
    1. Run tests
    2. Build package
    3. Upload to PyPI
```

### How it works

1. **Observation**: Agent notices patterns during tool usage
2. **Promotion**: Pattern must be seen 2+ times before persisting (noise reduction)
3. **Scoring**: Memories gain score when useful, decay when stale
4. **Injection**: Top memories are included in the autonomous prompt
5. **Pruning**: Memories with score < 0.15 or older than 60 days are cleaned up

### Memory is per-project

Each project directory has its own set of learned memories. When you `cd` to a different project and run `harness`, it loads memories specific to that project.

---

## 12. External Agent Integration

Open Harness uses an **orchestrator architecture**: the local LLM handles planning, judgment, and tool selection, while complex tasks like code generation, analysis, and debugging are delegated to external AI agents.

### Why Orchestrator?

Local LLMs are good at following instructions and making simple decisions, but struggle with code generation and complex reasoning. By delegating these tasks to powerful external agents (Claude, Codex, Gemini), you get the best of both worlds:

- **Local LLM** → Fast planning, tool selection, coordination
- **External agents** → High-quality code generation, analysis, debugging

### Claude Code (Anthropic) — Recommended

Best for: code generation, code analysis, refactoring, complex reasoning.

```yaml
external_agents:
  claude:
    enabled: true
    command: "claude"
```

Usage in autonomous mode:

```
> /goal Refactor the authentication module to use JWT tokens
```

The agent will delegate to Claude Code:

```
> claude_code "Refactor src/auth.py to use JWT tokens instead of session-based auth. Keep backward compatibility."
```

### Codex (OpenAI)

Best for: code generation, debugging, autonomous coding tasks.

```yaml
external_agents:
  codex:
    enabled: true
    command: "codex"
```

### Gemini CLI (Google)

Best for: code review, analysis, alternative perspectives.

```yaml
external_agents:
  gemini:
    enabled: true
    command: "gemini"
```

### Configurable routing

You can customize agent descriptions and declare their strengths for smarter routing:

```yaml
external_agents:
  claude:
    enabled: true
    command: "claude"
    description: "Complex refactoring, Japanese text, planning"
    strengths: ["refactoring", "japanese_text", "planning"]
```

The orchestrator uses these hints to pick the best agent for each sub-task.

### Rate-limit fallback

When an external agent hits its usage quota, Open Harness detects the error (via patterns like `429`, `"rate limit"`, `"quota exceeded"`), records the cooldown time, and automatically routes the task to the next available agent. When the cooldown expires, it switches back to the original agent. See [Rate-Limit Fallback](#18-rate-limit-fallback) for full details.

### Policy limits

External calls are limited by policy:
- `safe`: 10 calls max
- `balanced`: unlimited (orchestrator delegates freely)
- `full`: unlimited

### Requirements

External agents must be installed and accessible in PATH. Open Harness checks availability on startup and only registers tools that are found.

---

## 13. Per-Project Configuration

Open Harness supports per-project configuration by placing `open_harness.yaml` in the project directory.

### Use case: Different LLM endpoints per project

```
~/projects/
├── web-app/
│   └── open_harness.yaml    ← Uses remote GPU server
├── scripts/
│   └── open_harness.yaml    ← Uses local Ollama
└── ~/.open_harness/
    └── open_harness.yaml    ← Global fallback
```

### Example: Restrict tools for a production project

```yaml
# ~/projects/production-api/open_harness.yaml
policy:
  mode: "safe"
  max_git_commits: 1
  disabled_tools:
    - "shell"
  denied_paths:
    - "**/migrations/*"
    - "**/.env*"
```

### Example: Custom test command

Place a `.harness/context.md` in your project root to provide additional context:

```markdown
# Project Context

This is a Django application. Run tests with:
  python manage.py test --parallel

The API lives in apps/api/. Do not modify apps/legacy/.
```

Open Harness reads this file automatically and includes it in agent prompts.

### Project detection

Open Harness auto-detects your project type and adapts:

| Detected file | Project type | Auto test command |
|---------------|-------------|-------------------|
| `pyproject.toml` / `setup.py` | Python | `python3 -m pytest` |
| `package.json` | JavaScript | `npm test` |
| `Cargo.toml` | Rust | `cargo test` |
| `go.mod` | Go | `go test ./...` |

---

## 14. Practical Examples

### Example 1: Quick code review

```
> Read src/auth.py and identify any security issues

> read_file src/auth.py
...

I found the following issues:
1. Line 23: SQL query uses string formatting instead of parameterized queries
2. Line 45: Password comparison is not constant-time
3. Line 67: Session token has insufficient entropy
```

### Example 2: Autonomous test generation

```
> /goal Generate comprehensive unit tests for src/utils.py

thinking: Let me analyze the module first...
> read_file src/utils.py

Plan:
  Step 1: Analyze all functions in utils.py
  Step 2: Create test file with test cases for each function
  Step 3: Run tests and fix any failures

> write_file tests/test_utils.py ...
> run_tests tests/test_utils.py
OK: 12 passed

Goal completed in 45.2s
```

### Example 3: Refactoring with safety

```
> /policy safe
> /goal Rename the 'data' variable to 'payload' throughout src/api/

[checkpoint] Stashed changes, created work branch
...
> search_files "data" src/api/
> edit_file src/api/handlers.py ...
> edit_file src/api/models.py ...
> run_tests
OK: 28 passed
[checkpoint] Squash-merged to main

Goal completed in 32.1s
```

### Example 4: Multi-step debugging

```
> /goal The /users endpoint returns 500. Find the bug and fix it.

Plan:
  Step 1: Read the endpoint handler and trace the error
  Step 2: Identify the root cause
  Step 3: Implement the fix
  Step 4: Add a regression test
  Step 5: Verify all tests pass

> read_file src/api/users.py
> shell python -c "import src.api.users"
FAIL: ImportError: cannot import 'UserSchema' from 'models'

thinking: The import path changed but users.py wasn't updated...

> edit_file src/api/users.py "from models import UserSchema" "from src.models import UserSchema"
> run_tests tests/test_users.py
OK: 5 passed

Goal completed in 38.7s
```

### Example 5: Background code generation

```
> /submit Generate API documentation for all endpoints in src/api/

Task b2c3d4e5 queued
Log: ~/.open_harness/logs/task_...

(continue working on other things)

> What does the config module do?
...

(notification bell rings)
OK Task b2c3d4e5 complete: Generate API documentation...

> /result b2c3d4e5
(shows generated documentation)
```

### Example 6: Using different tiers for different tasks

```
> /tier small
> Summarize this error log
(fast response using small model)

> /tier large
> /goal Redesign the database schema to support multi-tenancy
(complex task using large model with more tokens)
```

### Example 7: Delegating to external agents

```
> /goal Ask Codex to generate a comprehensive test suite, then review the results

> codex "Generate pytest tests for src/calculator.py with edge cases"
OK codex
(test code generated)

> write_file tests/test_calculator.py ...
> run_tests tests/test_calculator.py
OK: 15 passed

Goal completed in 120.5s
```

---

## 15. Troubleshooting

### Connection refused

```
Error: Connection refused at http://localhost:1234/v1
```

**Fix**: Ensure your LLM server is running.

```bash
# LM Studio: Start from the GUI, enable "Local Server"
# Ollama:
ollama serve
```

### No config file found

```
Config: defaults (no open_harness.yaml found)
```

**Fix**: Create `open_harness.yaml` in your project directory or `~/.open_harness/`.

### Model returns garbage / tool calls fail

This is expected with smaller LLMs. The compensation engine handles this automatically with error-class-specific strategies (v0.5.0):

| Error class | Strategy |
|-------------|----------|
| `malformed_json` | JSON repair (no LLM retry needed) |
| `wrong_tool_name` | Fuzzy match to suggest correct tool name |
| `missing_args` | Inject parameter schema into correction prompt |
| `empty_response` | Immediate model escalation |
| `prose_wrapped` | Extract JSON from prose (parser handles it) |

If classification-based recovery fails, the standard retry sequence applies:

1. `refine_prompt` — adds correction hints
2. `add_examples` — adds tool-call examples
3. `escalate_model` — tries a larger tier

If all retries fail, consider using a more capable model.

### Policy blocks my action

```
VIOLATION: file write denied by path restriction: .env
```

**Fix**: Adjust the policy or use a different approach.

```
> /policy full          # Remove all restrictions (use with caution)
> /policy balanced      # Default restrictions
```

### Tests not auto-detected

If your test command isn't detected:

1. Check `/project` output
2. Create `.harness/context.md` with explicit test command
3. Or specify in config:

```yaml
# This is handled via project detection, but you can override:
# Place a .harness/context.md in your project root with:
# Test command: python -m pytest -x tests/
```

### Background task hangs

```
> /tasks
│ a3f2e1b0 │ running │ ...  │ 600s │
```

The task may be stuck on an LLM call. On next restart, stale running tasks are automatically marked as failed (crash recovery).

### Memory database locked

If you see SQLite locking errors:

**Fix**: Only one `harness` process should run per machine. The database uses WAL mode for concurrency between the main thread and background tasks, but not between processes.

---

## 16. @file References and Tab Completion

You can attach file contents to your message using `@path/to/file`:

```
> Review this file @src/open_harness/cli.py
```

The file content is automatically expanded and attached to the message sent to the LLM.

### Tab completion

Press Tab after `@` to get file/directory completion:

```
> @src/[Tab]
  src/open_harness/     (directory)

> @src/open_harness/[Tab]
  __init__.py           (328B)
  cli.py                (19KB)
  agent.py              (12KB)
  ...
```

- Directories show `/` suffix for further navigation
- File sizes are shown as metadata
- `.git`, `__pycache__`, `node_modules`, `.venv` are excluded
- Path traversal outside the project root is blocked

---

## 17. Mode Switching

The REPL has three input modes, cycled with **Shift+Tab**:

| Mode | Indicator | Behavior |
|------|-----------|----------|
| chat | green `>` | Interactive conversation (default) |
| goal | yellow `>` | Autonomous execution -- input becomes a goal |
| submit | blue `>` | Background queue -- input is submitted as a task |

The current mode is shown in the bottom toolbar. Press Shift+Tab to cycle through modes, then type your message and press Enter.

---

## 18. Rate-Limit Fallback

When an external agent hits its usage quota, Open Harness automatically:

1. **Detects** the rate limit (patterns: "429", "rate limit", "quota exceeded", "too many requests", etc.)
2. **Records** the cooldown period (parsed from "try again in X minutes" hints, default 15 min)
3. **Routes** the task to the next available fallback agent
4. **Recovers** -- switches back to the original agent when its cooldown expires

### Fallback order

| Primary | Fallback 1 | Fallback 2 |
|---------|-----------|-----------|
| claude_code | codex | gemini_cli |
| codex | claude_code | gemini_cli |
| gemini_cli | claude_code | codex |

### Example

```
> /goal Refactor the auth module

[compensation] codex rate-limited (cooldown 15m)
[compensation] Retrying with claude_code
> claude_code "Refactor the auth module..."
OK claude_code
```

If all agents are rate-limited, the task proceeds with the original agent (which will likely fail and trigger the compensation engine's retry strategies).

---

## Appendix: Command-Line Reference

```
harness [OPTIONS]

Options:
  -c, --config PATH     Path to open_harness.yaml
  -t, --tier TIER       Model tier (small, medium, large)
  -g, --goal TEXT       Run goal non-interactively and exit
  -v, --verbose         Enable debug logging
  --help                Show help
```

### Non-interactive mode

```bash
# Run a goal and exit
harness --goal "Fix the failing tests in tests/test_auth.py"

# With specific tier
harness --tier large --goal "Refactor the entire API module"

# With custom config
harness --config ./my_config.yaml --goal "Add logging"

# Verbose output for debugging
harness -v --goal "Why are tests failing?"
```
