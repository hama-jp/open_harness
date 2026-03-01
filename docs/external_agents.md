# External Agent Routing Guide

> Reference for orchestrator task routing. Last updated: 2026-03-01.
> Benchmarks change rapidly -- verify against current leaderboards before relying on specific numbers.

## Models (as of 2026-03-01)

| Tool | Model | Context | Notes |
|------|-------|---------|-------|
| Claude Code | Claude Opus 4.6 / Sonnet 4.6 | 200K (1M Beta) | Extended Thinking, Plan Mode |
| Codex CLI | GPT-5.3-Codex | 200K | Sandbox, 25% faster than 5.2 |
| Gemini CLI | Gemini 3.1 Pro (Preview) | 1M | ARC-AGI-2 77.1% |

## Benchmark Summary

| Benchmark | Claude Opus 4.6 | GPT-5.3-Codex | Gemini 3.1 Pro |
|-----------|-----------------|---------------|----------------|
| SWE-bench Verified | 80.8% | -- | 80.6% |
| SWE-bench Pro (generic scaffold) | 45.9% | **56.8%** | -- |
| ARC-AGI-2 (abstract reasoning) | 68.8% | -- | **77.1%** |
| GPQA Diamond (PhD science) | 91.3% | -- | **94.3%** |
| HLE + Tools (tool-augmented) | **53.1%** | -- | 51.4% |
| Japanese MMLU | 93 | -- | **94** |

## Task Routing Recommendations

### Code Generation & Refactoring
- **Best**: `claude_code` -- SWE-bench top tier, Plan Mode for multi-file changes
- **Alternative**: `codex` -- strong at SWE-bench Pro with good scaffolding, fast

### Code Review
- **Security (path traversal, SSRF)**: `codex` -- 47% / 34% detection rate
- **Logic (IDOR, XSS)**: `claude_code` -- 22% / 16% detection rate
- **Large codebase review**: `gemini_cli` -- 1M context window

### Japanese / Multilingual Text
- **Best**: `claude_code` -- WMT24 translation winner (9/11 pairs), `language` setting
- **Alternative**: `gemini_cli` -- MMLU Japanese 94, strong multilingual

### Planning & Architecture
- **Best**: `claude_code` -- Plan Mode, Extended Thinking, 45% fewer architecture errors
- **Alternative**: `gemini_cli` -- spec-to-task decomposition

### Large Codebase Understanding
- **Best**: `gemini_cli` -- 1M context standard, half the price
- **Alternative**: `claude_code` -- 1M Beta available

### Speed-Critical Tasks
- **Best**: `codex` -- optimized infrastructure, fastest response
- **Alternative**: `gemini_cli` -- Flash model available

### Abstract Reasoning & Science
- **Best**: `gemini_cli` -- ARC-AGI-2 77.1%, GPQA 94.3%

## Sources

- [SWE-Bench Pro Leaderboard (Scale AI)](https://scale.com/leaderboard/swe_bench_pro_public)
- [Gemini 3.1 Pro Announcement (Google)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/)
- [GPT-5.3-Codex (OpenAI)](https://openai.com/index/introducing-gpt-5-3-codex/)
- [Gemini 3.1 Pro vs Claude Opus 4.6 Benchmarks](https://www.trendingtopics.eu/gemini-3-1-pro-leads-most-benchmarks-but-trails-claude-opus-4-6-in-some-tasks/)
- [Semgrep: Finding Vulnerabilities with Claude Code and Codex](https://www.semgrep.dev/blog/2025/finding-vulnerabilities-in-modern-web-apps-using-claude-code-and-openai-codex/)
- [Artificial Analysis: Japanese Multilingual Index](https://artificialanalysis.ai/models/multilingual/japanese)
