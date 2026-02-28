# Open Harness

AI agent harness optimized for local LLMs with weak-model compensation.

## Features

- **Local LLM First**: Designed for LM Studio, Ollama, and other local inference servers
- **Weak Model Compensation**: Retry strategies, prompt refinement, and model escalation
- **Multi-Model Routing**: Route tasks to small/medium/large models based on complexity
- **Tool System**: Shell, file operations, and external agent integration (Codex, Gemini CLI)
- **MCP Compatible**: Extensible tool format compatible with the broader ecosystem

## Quick Start

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
harness
```

## Configuration

Edit `config.yaml` to configure LLM endpoints, model tiers, and tools.

## License

MIT
