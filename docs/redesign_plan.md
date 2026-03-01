# Open Harness v2.0 — Redesign Plan

> If we could redesign the project from scratch, what would the architecture look like?

## Current State (v0.5.x)

- ~10K lines of Python across 35 files
- Synchronous throughout (blocking httpx, threading for external agents)
- `agent.py` acts as God Object (context, tools, LLM calls, compensation, planning)
- Hardcoded tool system (no plugin interface)
- CLI/TUI tightly coupled to agent loop
- 5-level config file discovery with Pydantic models

## What to Keep

These design decisions are correct and should survive a rewrite:

| Component | Why It Works |
|-----------|-------------|
| **Weak-model compensation (Compensator)** | Core value proposition. Error classification → class-specific recovery is the right approach |
| **Policy Engine returning tool errors** | Agent adapts instead of blocking user. Elegant |
| **Two-level context compression** | L1 (tool pair summaries) → L2 (aggregate counts) is practical |
| **Git-based checkpoints** | Using real filesystem for safety is correct |
| **SQLite for persistence** | Sufficient for task queue and memory. No external DB needed |
| **Textual TUI** | Good implementation. Just reconnect via event bus |
| **Local LLM optimization** | The project's reason to exist |

---

## Redesign: 7 Architectural Changes

### 1. Async-First Architecture

**Problem:** Everything is synchronous. httpx blocks, external agents use threading.

**Solution:** Use `asyncio` as the core runtime.

```python
# Current: all serial
result = llm_client.chat(messages)      # blocks
tool_result = tool.execute(**args)       # blocks
external = subprocess.Popen(...)        # thread

# Redesign: asyncio core
result = await llm_client.chat(messages)
tool_result = await tool.execute(**args)
external = await asyncio.create_subprocess_exec(...)
```

**Benefits:**
- **Parallel tool execution** — run independent tool calls concurrently
- **Clean cancellation** — `asyncio.Task.cancel()` for instant stop
- **Non-blocking streaming** — TUI updates smoothly
- Sync API compatibility via `asyncio.run()` wrapper

### 2. Decompose the God Object (agent.py)

**Problem:** `agent.py` owns too many responsibilities: context management, tool execution, LLM calls, compensation, plan execution.

**Solution:** Split into three focused components.

```
Orchestrator          # Controls overall flow only
  ├── Reasoner        # Interprets LLM output, decides next action
  ├── Executor        # Runs tools, collects results
  └── ContextStore    # Accumulates, compresses, retrieves context
```

The Orchestrator becomes a thin loop:

```python
class Orchestrator:
    async def run(self, goal: str):
        while not done:
            context = self.context_store.build_messages()
            response = await self.reasoner.think(context)
            if response.has_tool_calls:
                results = await self.executor.run(response.tool_calls)
                self.context_store.add(results)
            else:
                done = True
```

### 3. Middleware Pipeline for LLM Calls

**Problem:** Compensator is monolithic. Error classification, retry, model escalation are tightly coupled.

**Solution:** Composable middleware chain.

```
Request
  → PromptOptimizer    # Inject weak-model hints
  → ModelRouter        # Select tier
  → RateLimiter        # Rate limit
  → LLMClient          # Actual API call
  → ResponseParser     # JSON extraction, tool call parsing
  → ErrorRecovery      # Classify error → retry at appropriate stage
  → Result
```

Each middleware is independently testable. Order is swappable. The concept of "weak model compensation" survives but becomes **distributed across pipeline stages** instead of living in one class.

```python
class Middleware(Protocol):
    async def process(
        self, request: LLMRequest, next: Callable
    ) -> LLMResponse: ...

class PromptOptimizer:
    async def process(self, request, next):
        request.messages = self.inject_hints(request.messages)
        return await next(request)

class ErrorRecovery:
    async def process(self, request, next):
        for attempt in range(self.max_retries):
            try:
                return await next(request)
            except LLMError as e:
                strategy = self.classify(e)
                request = strategy.adjust(request)
        raise MaxRetriesExceeded()
```

### 4. Plugin-Based Tool System

**Problem:** Tools are hardcoded. Adding a new tool requires code changes.

**Solution:** Dynamic discovery via entry points.

```toml
# pyproject.toml
[project.entry-points."open_harness.tools"]
file_ops = "open_harness.tools.builtin.file_ops:register"
shell = "open_harness.tools.builtin.shell:register"

# External package can add tools:
# my_custom_tool = "my_package.tools:register"
```

```python
# tools/registry.py
class ToolRegistry:
    def discover(self):
        """Load tools from entry points"""
        for ep in importlib.metadata.entry_points(group="open_harness.tools"):
            register_fn = ep.load()
            register_fn(self)

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
```

Users install `pip install harness-tool-docker` and Docker tools become available automatically.

Future: MCP (Model Context Protocol) support for standardized tool interface.

### 5. Typed Context Object

**Problem:** Context is `list[dict]` (OpenAI messages format). Compression logic scattered through agent.py.

**Solution:** Structured context with layer-specific compression.

```python
@dataclass
class AgentContext:
    system: SystemLayer      # Always retained (project info, tools, memories)
    plan: PlanLayer          # Current plan if any
    history: HistoryLayer    # Compressible past turns
    working: WorkingLayer    # Recent tool results (protected)

    def to_messages(self, budget: int) -> list[dict]:
        """Convert to messages within token budget.
        Each layer owns its own compression strategy."""
        ...
```

Each layer knows **how to compress itself**. No compression logic leaks into agent.py.

- `SystemLayer`: Never compressed. Regenerated from config + project context.
- `PlanLayer`: Compressed to current step + next 2 steps.
- `HistoryLayer`: L1 → L2 compression (existing strategy, encapsulated).
- `WorkingLayer`: Head+tail truncation per tool output.

### 6. Event Bus for UI Decoupling

**Problem:** CLI/TUI are tightly coupled to the agent loop.

**Solution:** Pub/sub event bus.

```python
class EventBus:
    async def emit(self, event: AgentEvent): ...
    def subscribe(self, event_type: type, handler: Callable): ...

# Agent only emits
await self.events.emit(ToolExecuted(name="shell", result=result))
await self.events.emit(LLMStreaming(chunk="..."))
await self.events.emit(PlanStepCompleted(step=3, total=5))

# UI only subscribes
event_bus.subscribe(ToolExecuted, self.update_tool_panel)
event_bus.subscribe(LLMStreaming, self.append_output)
```

**Benefits:**
- CLI, TUI, Web UI, tests — all just different event consumers
- Adding a new UI doesn't touch `agent.py`
- Clean testing: subscribe and assert events
- Structured logging becomes another subscriber

### 7. Configuration Simplification

**Problem:** 5-level config discovery, complex Pydantic models.

**Solution:** 1 file + environment variables + profiles.

```yaml
# ~/.config/open-harness/config.yaml
profile: local

profiles:
  local:
    provider: lm-studio
    url: http://localhost:1234/v1
    models: [qwen3-8b, qwen3-30b]    # Order = tier (small→large)

  api:
    provider: openai
    models: [gpt-4o-mini, gpt-4o]

policy: balanced    # "safe" | "balanced" | "full"
```

Model tiers (small/medium/large) are **implicitly determined by array order**. No explicit tier definitions needed.

Config discovery: `--config` flag > `./open_harness.yaml` > `~/.config/open-harness/config.yaml` > built-in defaults. Three levels max.

---

## Proposed Directory Structure

```
src/open_harness/
├── core/
│   ├── orchestrator.py     # Thin main loop
│   ├── reasoner.py         # LLM output interpretation
│   ├── executor.py         # Tool execution
│   └── context.py          # Typed context (layer-based compression)
├── llm/
│   ├── client.py           # Async httpx client
│   ├── middleware.py        # Pipeline infrastructure
│   ├── prompt_optimizer.py # Weak-model prompt optimization
│   ├── response_parser.py  # JSON extraction, tool call parsing
│   ├── error_recovery.py   # Error classification, retry, escalation
│   └── router.py           # Model selection (array-order based)
├── tools/
│   ├── registry.py         # Plugin discovery and registration
│   ├── base.py             # Tool ABC
│   └── builtin/            # Built-in tools
│       ├── file_ops.py
│       ├── shell.py
│       ├── git_tools.py
│       ├── testing.py
│       └── external.py
├── policy/
│   └── engine.py           # Guardrails (keep current design)
├── events/
│   └── bus.py              # Pub/sub event bus
├── memory/                  # Keep as-is
├── tasks/                   # Keep as-is
├── ui/
│   ├── cli.py              # CLI event consumer
│   └── tui/                # TUI event consumer
└── config.py               # Simplified configuration
```

---

## Key Design Principle

| Current (v0.5.x) | Redesign (v2.0) |
|---|---|
| `agent.py` knows everything — "smart center" | Small components collaborate — "smart network" |

---

## Migration Strategy

Full rewrite is high risk. Incremental migration is preferred:

### Phase 1: Middleware Pipeline (Low Risk)
- Extract Compensator into middleware stages
- `PromptOptimizer`, `ResponseParser`, `ErrorRecovery` as separate classes
- Agent.py calls the pipeline instead of Compensator directly
- **No behavior change**, just structural refactor

### Phase 2: Event Bus (Medium Risk)
- Introduce `EventBus` class
- Agent emits events alongside current direct calls
- CLI/TUI subscribe to events (dual path: old + new)
- Gradually remove direct coupling

### Phase 3: Context Layers (Medium Risk)
- Replace `list[dict]` with `AgentContext`
- Move compression logic into layer classes
- `to_messages()` produces same output as before

### Phase 4: Async Core (High Risk)
- Convert `LLMClient` to async httpx
- Convert tool execution to async
- Orchestrator becomes `async def run()`
- CLI wraps with `asyncio.run()`

### Phase 5: Plugin Tools (Low Risk)
- Add entry_points discovery to ToolRegistry
- Move built-in tools into `builtin/` subdirectory
- Existing tools keep working, new tools can be external packages

---

*Created: 2026-03-02*
*Based on analysis of Open Harness v0.5.03 (~10K LOC, 35 Python files)*
