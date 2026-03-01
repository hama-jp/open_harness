# Open Harness

Self-driving AI agent harness optimized for local LLMs.

## Features

- **Local LLM First** — Designed for LM Studio, Ollama, and any OpenAI-compatible API
- **Weak Model Compensation** — Retry strategies, prompt refinement, and automatic model escalation
- **Multi-Model Routing** — Route tasks to small/medium/large tiers based on complexity
- **Autonomous Goal Execution** — Planner-Critic-Executor loop with automatic replanning
- **Checkpoint & Rollback** — Git-based transactional safety for autonomous operations
- **Background Task Queue** — Submit goals to a FIFO queue, keep working while they execute
- **Project Memory** — Learns project patterns across sessions (test commands, error fixes, workflows)
- **Policy Guardrails** — Configurable safety limits (safe / balanced / full presets)
- **Orchestrator Architecture** — Local LLM plans and coordinates; external agents (Claude Code, Codex, Gemini CLI) handle code generation and analysis
- **15 Built-in Tools** — File ops, shell, git, testing, and external agent delegation (Claude Code, Codex, Gemini CLI)
- **Per-Project Config** — Place `open_harness.yaml` in any directory to customize per-project

## Quick Start

```bash
git clone https://github.com/hama-jp/open_harness.git
cd open_harness
uv venv && source .venv/bin/activate
uv pip install -e .
```

Edit `open_harness.yaml` to point to your LLM server, then:

```bash
harness
```

## Usage

```bash
# Interactive mode
harness

# Run a goal non-interactively
harness --goal "Fix the failing tests"

# Use a specific model tier
harness --tier large

# Custom config path
harness --config ~/myproject/open_harness.yaml
```

### REPL commands

```
/goal <task>       Autonomous execution with planning and checkpoints
/submit <task>     Submit to background queue
/tasks             List background tasks
/result <id>       Show task result
/model [tier]      Show model details for all tiers, or switch tier
/tier [name]       Switch model tier (small/medium/large)
/policy [mode]     Switch safety policy (safe/balanced/full)
/tools             List available tools
/memory            Show learned project knowledge
/help              Show all commands
```

## Configuration

Create `open_harness.yaml` in your project directory or `~/.open_harness/`:

```yaml
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

## Documentation

- [Tutorial (English)](docs/tutorial.md)
- [チュートリアル（日本語）](docs/tutorial_ja.md)

## Requirements

- Python 3.11+
- A local LLM server (LM Studio, Ollama, or any OpenAI-compatible endpoint)

## License

MIT
