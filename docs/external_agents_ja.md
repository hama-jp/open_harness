# 外部エージェント ルーティングガイド

> オーケストレーターのタスク振り分け参考資料。最終更新: 2026-03-01。
> ベンチマークは急速に変化します。具体的な数値に依存する前に最新のリーダーボードを確認してください。

## 利用可能なモデル (2026-03-01 時点)

| ツール | モデル | コンテキスト | 備考 |
|--------|--------|-------------|------|
| Claude Code | Claude Opus 4.6 / Sonnet 4.6 | 200K (1M Beta) | Extended Thinking, Plan Mode |
| Codex CLI | GPT-5.3-Codex | 200K | サンドボックス実行、5.2比25%高速 |
| Gemini CLI | Gemini 3.1 Pro (Preview) | 1M | ARC-AGI-2 77.1% |

## ベンチマーク概要

| ベンチマーク | Claude Opus 4.6 | GPT-5.3-Codex | Gemini 3.1 Pro |
|-------------|-----------------|---------------|----------------|
| SWE-bench Verified | 80.8% | -- | 80.6% |
| SWE-bench Pro | 45.9% | **56.8%** | -- |
| ARC-AGI-2 (抽象推論) | 68.8% | -- | **77.1%** |
| GPQA Diamond (科学) | 91.3% | -- | **94.3%** |
| HLE + Tools (ツール活用) | **53.1%** | -- | 51.4% |
| 日本語 MMLU | 93 | -- | **94** |

## タスク別推奨ルーティング

### コード生成・リファクタリング
- **最適**: `claude_code` -- SWE-bench最上位圏、Plan Modeで複数ファイル変更に強い
- **代替**: `codex` -- SWE-bench Proに強い、高速レスポンス

### コードレビュー
- **セキュリティ (パストラバーサル, SSRF)**: `codex` -- 検出率 47% / 34%
- **ロジック (IDOR, XSS)**: `claude_code` -- 検出率 22% / 16%
- **大規模コードベースレビュー**: `gemini_cli` -- 1Mコンテキスト

### 日本語・多言語テキスト
- **最適**: `claude_code` -- WMT24翻訳で11ペア中9ペア1位、language設定あり
- **代替**: `gemini_cli` -- 日本語MMLU 94、多言語全般に強い

### 計画・アーキテクチャ設計
- **最適**: `claude_code` -- Plan Mode、Extended Thinking、アーキテクチャエラー45%削減
- **代替**: `gemini_cli` -- 仕様→タスク分解に強い

### 大規模コードベース理解
- **最適**: `gemini_cli` -- 1Mコンテキスト標準、半額
- **代替**: `claude_code` -- 1M Beta利用可

### 速度重視タスク
- **最適**: `codex` -- 最適化インフラ、最速レスポンス
- **代替**: `gemini_cli` -- Flashモデル利用可

### 抽象推論・科学タスク
- **最適**: `gemini_cli` -- ARC-AGI-2 77.1%、GPQA 94.3%

## 設定方法

`open_harness.yaml` の `external_agents` セクションで各エージェントの得意分野を指定できます:

```yaml
external_agents:
  codex:
    enabled: true
    command: "codex"
    strengths:
      - "code_review_security"
      - "fast_coding"
    description: "高速コーディングとセキュリティレビューに最適"
  claude:
    enabled: true
    command: "claude"
    strengths:
      - "refactoring"
      - "japanese_text"
      - "planning"
      - "code_review_logic"
    description: "複雑なリファクタリング、日本語、計画に最適"
  gemini:
    enabled: true
    command: "gemini"
    strengths:
      - "large_codebase"
      - "abstract_reasoning"
      - "science"
    description: "大規模コード理解と抽象推論に最適"
```

詳細は [設定リファレンス](../open_harness.yaml) を参照してください。

## ソース

- [SWE-Bench Pro リーダーボード (Scale AI)](https://scale.com/leaderboard/swe_bench_pro_public)
- [Gemini 3.1 Pro 発表 (Google)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/)
- [GPT-5.3-Codex (OpenAI)](https://openai.com/index/introducing-gpt-5-3-codex/)
- [Gemini 3.1 Pro vs Claude Opus 4.6 ベンチマーク](https://www.trendingtopics.eu/gemini-3-1-pro-leads-most-benchmarks-but-trails-claude-opus-4-6-in-some-tasks/)
- [Semgrep: セキュリティ脆弱性検出の比較](https://www.semgrep.dev/blog/2025/finding-vulnerabilities-in-modern-web-apps-using-claude-code-and-openai-codex/)
- [Artificial Analysis: 日本語多言語インデックス](https://artificialanalysis.ai/models/multilingual/japanese)
