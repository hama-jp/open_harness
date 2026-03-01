# Open Harness チュートリアル

ローカルLLM向けに最適化された自律駆動AIエージェントハーネス「Open Harness」の総合ガイドです。

---

## 目次

1. [インストール](#1-インストール)
2. [設定](#2-設定)
3. [はじめよう — 対話モード](#3-はじめよう--対話モード)
4. [自律モード — /goal](#4-自律モード--goal)
5. [バックグラウンドタスク — /submit](#5-バックグラウンドタスク--submit)
6. [ツールリファレンス](#6-ツールリファレンス)
7. [モデルティアとルーティング](#7-モデルティアとルーティング)
8. [ポリシーと安全ガードレール](#8-ポリシーと安全ガードレール)
9. [Planner-Critic-Executor ループ](#9-planner-critic-executor-ループ)
10. [チェックポイントとロールバック](#10-チェックポイントとロールバック)
11. [プロジェクトメモリ](#11-プロジェクトメモリ)
12. [外部エージェント連携](#12-外部エージェント連携)
13. [プロジェクトごとの設定](#13-プロジェクトごとの設定)
14. [実践例](#14-実践例)
15. [トラブルシューティング](#15-トラブルシューティング)
16. [@ファイル参照とTab補完](#16-ファイル参照とtab補完)
17. [モード切替](#17-モード切替)
18. [レート制限フォールバック](#18-レート制限フォールバック)

---

## 1. インストール

### 必要環境

- Python 3.11 以上
- ローカルLLMサーバー（LM Studio、Ollama、または任意のOpenAI互換API）

### uv でインストール（推奨）

```bash
git clone https://github.com/hama-jp/open_harness.git
cd open_harness
uv venv && source .venv/bin/activate
uv pip install -e .
```

### pip でインストール

```bash
git clone https://github.com/hama-jp/open_harness.git
cd open_harness
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### インストール確認

```bash
harness --help
```

---

## 2. 設定

### 設定ファイル

Open Harness は `open_harness.yaml` を設定ファイルとして使用します。以下の順序で検索されます：

1. `--config` オプションで指定されたパス
2. カレントディレクトリ（`./open_harness.yaml`）
3. ユーザー設定ディレクトリ（`~/.open_harness/open_harness.yaml`）
4. リポジトリルート（開発モード）

> 後方互換性のため、旧名称 `config.yaml` も認識されます。

### 最小構成

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

### 全設定リファレンス

```yaml
# ── LLM設定 ──
llm:
  default_provider: "lm_studio"    # または "ollama"

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
      description: "高速・簡単なタスク向け"
    medium:
      provider: "lm_studio"
      model: "unsloth/qwen3.5-35b-a3b@q4_k_m"
      max_tokens: 8192
      description: "バランス型"
    large:
      provider: "lm_studio"
      model: "unsloth/qwen3.5-35b-a3b@q4_k_m"
      max_tokens: 16384
      description: "複雑な推論向け"

  default_tier: "medium"

# ── 補償エンジン ──
compensation:
  max_retries: 3
  retry_strategies:
    - "refine_prompt"       # 修正ヒントを追加してリトライ
    - "add_examples"        # ツール呼び出しの例をプロンプトに追加
    - "escalate_model"      # より大きなモデルティアにエスカレーション
  parse_fallback: true      # 不整合な出力からツール呼び出しを抽出
  thinking_mode: "auto"     # auto | always | never

# ── ツール設定 ──
tools:
  shell:
    timeout: 30             # 秒
    allowed_commands: []    # 空 = すべて許可
    blocked_commands:
      - "rm -rf /"
      - "mkfs"
      - "dd if="
  file:
    max_read_size: 100000   # バイト

# ── 外部エージェント ──
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

# ── ポリシー ──
policy:
  mode: "balanced"          # safe | balanced | full
  # writable_paths:         # プロジェクトルート以外の書き込み可能ディレクトリ
  #   - "/tmp/*"
  #   - "~/other-project/*"

# ── メモリ ──
memory:
  backend: "sqlite"
  db_path: "~/.open_harness/memory.db"
  max_conversation_turns: 50
```

### Ollama の設定例

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

### LM Studio の設定例（リモートマシン）

```yaml
llm:
  default_provider: "lm_studio"
  providers:
    lm_studio:
      base_url: "http://192.168.1.100:1234/v1"   # LM Studioホストの LAN IP
      api_key: "lm-studio"
  models:
    medium:
      provider: "lm_studio"
      model: "unsloth/qwen3.5-35b-a3b@q4_k_m"
      max_tokens: 8192
```

---

## 3. はじめよう — 対話モード

### 起動

```bash
harness
```

起動時に以下の情報が表示されます：

```
╭──────────────────────────────────────────────────╮
│ Open Harness v0.3.3                              │
│ Self-driving AI agent for local LLMs             │
│ Type /help for commands, /goal <task> for auto    │
╰──────────────────────────────────────────────────╯
Config: /home/you/project/open_harness.yaml
Git: already initialized
Project: python @ /home/you/project
Tests: python3 -m pytest
Model: unsloth/qwen3.5-35b-a3b@q4_k_m (medium)
Tools (14): shell, read_file, write_file, ...
Task queue: ready
```

### 基本的な会話

メッセージを入力するだけです。エージェントは必要に応じて自動的にツールを使用します：

```
> このプロジェクトにはどんなファイルがある？

> list_dir .
OK list_dir
src/
tests/
pyproject.toml
README.md
...

このプロジェクトはPythonパッケージで、ソースコードは src/ に、
テストは tests/ にあります。
```

### コードについて質問する

```
> cli.pyのmain関数を説明して

> read_file src/open_harness/cli.py
OK read_file
(ファイル内容が表示)

cli.py の main 関数は Click を使って CLI をセットアップしています...
```

### コードの変更を依頼する

```
> Agent クラスにdocstringを追加して

> read_file src/open_harness/agent.py
> edit_file src/open_harness/agent.py ...
```

### REPL コマンド一覧

`/help` と入力するとすべてのコマンドが表示されます：

| コマンド | 説明 |
|---------|------|
| `/goal <タスク>` | 自律モードでゴールを実行 |
| `/submit <タスク>` | ゴールをバックグラウンドキューに送信 |
| `/tasks` | すべてのタスクとステータスを一覧表示 |
| `/result <id>` | タスクの詳細結果を表示 |
| `/model [ティア]` | 全ティアのモデル名・プロバイダー・ホスト・max_tokensを表示、またはティア切替 |
| `/tier [名前]` | モデルティアの表示・切替 |
| `/policy [モード]` | 安全ポリシーの表示・切替 |
| `/tools` | 利用可能なツールを一覧表示 |
| `/project` | 検出されたプロジェクト情報を表示 |
| `/memory` | 学習済みのプロジェクト記憶を表示 |
| `/clear` | 会話履歴をクリア |
| `/update` | Open Harness を git から自己更新 |
| `/quit` | 終了 |

---

## 4. 自律モード — /goal

`/goal` コマンドは完全な自律実行を起動します。エージェントがタスクを計画し、ステップごとに実行し、チェックポイントを取り、失敗時は自動的にリカバリーします。

### 基本的な使い方

```
> /goal config モジュールのユニットテストを追加する
```

エージェントは以下のように動作します：
1. ゴールを分析して複数ステップの計画を作成
2. 各ステップを順番に実行
3. 成功したステップごとに git スナップショットを取得
4. ステップ失敗時は再計画
5. 計画が失敗したら直接実行にフォールバック

### 例：バグ修正

```
> /goal utils.py の42行目の TypeError を修正する

thinking: エラーを分析中...
> read_file src/utils.py
OK read_file
(ファイル内容)

thinking: この関数が None を返す場合に...
> edit_file src/utils.py ...
OK edit_file

> run_tests tests/test_utils.py
OK run_tests
3 passed

Goal completed in 28.3s
```

### 例：機能実装

```
> /goal pyproject.toml からバージョンを表示する /version コマンドを追加する
```

エージェントの動作：
1. 計画：pyproject.toml を読む → cli.py を修正 → テスト
2. 現在のバージョン文字列を取得
3. コマンドハンドラを追加
4. 動作を検証
5. 完了を報告

### ゴールサマリー

ゴール完了時に実行サマリーが表示されます：

- ツール呼び出し回数
- 成功/失敗
- ファイル変更数
- ロールバック回数
- 補償回数

### 制限と安全性

- ゴールあたり最大50エージェントステップ
- 計画ステップは最大5つ（過剰計画を防止）
- git リポジトリではチェックポイント/ロールバックが有効
- ポリシーガードレールが全体を通じて適用

---

## 5. バックグラウンドタスク — /submit

長時間かかるゴールをバックグラウンドで実行し、その間もエージェントと対話を続けられます。

### タスクの送信

```
> /submit すべてのテストファイルを pytest fixtures を使うようにリファクタリングする

Task a3f2e1b0 queued: すべてのテストファイルを pytest fixtures を...
Log: ~/.open_harness/logs/task_1709312400_abc123.log
Check: /tasks | /result a3f2e1b0
```

### ステータス確認

```
> /tasks

┌──────────┬──────────┬────────────────────────────────────────────┬────────┐
│ ID       │ Status   │ Goal                                       │ Time   │
├──────────┼──────────┼────────────────────────────────────────────┼────────┤
│ a3f2e1b0 │ running  │ すべてのテストファイルを pytest...          │ 45s    │
│ 9c1d4e5f │ OK       │ database モジュールにログ追加               │ 32s    │
└──────────┴──────────┴────────────────────────────────────────────┴────────┘
```

### 結果の確認

```
> /result a3f2e1b0

Task a3f2e1b0: すべてのテストファイルを pytest fixtures を使うようにリファクタリングする
Status: succeeded
Time: 128.4s
╭── Result ──────────────────────────────────────────────╮
│ 5つのテストファイルを pytest fixtures に変換しました。  │
│ 23件のテストすべてがパスしました。                     │
╰────────────────────────────────────────────────────────╯
Log: ~/.open_harness/logs/task_1709312400_abc123.log
```

### 主な特徴

- タスクは1つずつ順番に処理される（FIFO キュー）
- 各タスクに独立した Agent インスタンスが割り当てられる（状態の共有なし）
- タスク完了時にターミナルベルが鳴る
- 完了通知は次のプロンプトで表示される
- タスクは再起動後も保持される（クラッシュリカバリー）
- バックグラウンドタスク実行中も対話を続行可能

---

## 6. ツールリファレンス

### ファイル操作

| ツール | 説明 | 使用例 |
|--------|------|--------|
| `read_file` | ファイルの読み取り | `read_file("src/main.py")` |
| `write_file` | ファイルの作成・上書き | `write_file("new.py", "print('hello')")` |
| `edit_file` | ファイル内のテキストを置換 | `edit_file("main.py", "旧テキスト", "新テキスト")` |
| `list_dir` | ディレクトリ内容の一覧 | `list_dir("src/", "*.py")` |
| `search_files` | ファイル内のテキスト検索（正規表現） | `search_files("TODO", "src/")` |

### シェル

| ツール | 説明 | 使用例 |
|--------|------|--------|
| `shell` | シェルコマンドの実行 | `shell("pip install requests")` |

シェルには安全デフォルトが設定されています：
- 30秒タイムアウト（設定変更可能）
- ブロック対象：`rm -rf /`、`mkfs`、`dd if=`
- ポリシーレベルのブロック：`curl | sh`、`chmod 777` など

### Git

| ツール | 説明 | 使用例 |
|--------|------|--------|
| `git_status` | 変更・ステージ済みファイルの表示 | `git_status()` |
| `git_diff` | 差分の表示 | `git_diff(staged=True)` |
| `git_commit` | ステージングとコミット | `git_commit("バグ修正", "src/main.py")` |
| `git_branch` | ブランチの作成・一覧表示 | `git_branch("feature/new")` |
| `git_log` | 最近のコミット履歴 | `git_log(count=5)` |

### テスト

| ツール | 説明 | 使用例 |
|--------|------|--------|
| `run_tests` | プロジェクトのテストスイートを実行 | `run_tests("tests/test_config.py")` |

テストコマンドはプロジェクトタイプに応じて自動検出されます：
- Python: `python3 -m pytest`
- Rust: `cargo test`
- JavaScript: `npm test`
- Go: `go test ./...`

### 外部エージェント

| ツール | 説明 | 使用例 |
|--------|------|--------|
| `claude_code` | Claude Code (Anthropic) に委譲 | `claude_code("このモジュールをリファクタリングして")` |
| `codex` | OpenAI Codex CLI に委譲 | `codex("REST APIクライアントを生成して")` |
| `gemini_cli` | Google Gemini CLI に委譲 | `gemini_cli("このアーキテクチャを分析して")` |

> **オーケストレーターアーキテクチャ**: Open Harness はローカルLLMを計画・調整のオーケストレーターとして使用します。
> コード生成・分析・デバッグは外部エージェント（Claude Code, Codex, Gemini CLI）に委譲されます。

---

## 7. モデルティアとルーティング

Open Harness は速度と能力のバランスを取るために複数のモデルティアをサポートしています。

### ティア一覧

| ティア | 典型的な用途 | デフォルト max_tokens |
|--------|-------------|----------------------|
| `small` | 簡単なタスク、計画作成 | 4096 |
| `medium` | 汎用タスク（デフォルト） | 8192 |
| `large` | 複雑な推論 | 16384 |

### 実行時のティア詳細表示

`/model` で各ティアのモデル名・プロバイダー・接続先ホスト・max_tokens を確認できます：

```
> /model
Model tiers:
  small:  qwen3.5-35b-a3b @ lm_studio (192.168.11.3) max_tokens=4096
  medium: qwen3.5-35b-a3b @ lm_studio (192.168.11.3) max_tokens=8192  *
  large:  qwen3.5-35b-a3b @ lm_studio (192.168.11.3) max_tokens=16384

> /model large
  large:  qwen3.5-35b-a3b @ lm_studio (192.168.11.3) max_tokens=16384  *
```

### 実行時のティア切替

```
> /tier
  small: Fast, simple tasks
  medium: Balanced performance *
  large: Complex reasoning

> /tier large
Tier: large
```

### 起動時にティアを指定

```bash
harness --tier large
```

### 自動エスカレーション

補償エンジンが繰り返しの失敗を検出すると、自動的により大きなティアにエスカレーションします：

```
small（失敗）→ medium（失敗）→ large
```

これは `escalate_model` リトライ戦略によって制御されます。

### ティアごとに異なるモデルを使用

各ティアに異なるモデルを割り当てることができます：

```yaml
models:
  small:
    provider: "ollama"
    model: "qwen2.5:7b"       # 高速・軽量
    max_tokens: 4096
  medium:
    provider: "lm_studio"
    model: "qwen2.5:14b"      # バランス型
    max_tokens: 8192
  large:
    provider: "lm_studio"
    model: "qwen2.5:32b"      # 最大能力
    max_tokens: 16384
```

---

## 8. ポリシーと安全ガードレール

ポリシーエンジンは自律実行中の自動安全ガードレールを提供します。

### ポリシープリセット

| プリセット | ファイル書き込み | シェル | Git コミット | 外部呼び出し |
|-----------|----------------|--------|-------------|-------------|
| `safe` | 20回 | 30回 | 3回 | 10回 |
| `balanced` | 無制限 | 無制限 | 10回 | 無制限 |
| `full` | 無制限 | 無制限 | 無制限 | 無制限 |

### 実行時のポリシー切替

```
> /policy
Policy mode: balanced
Budgets: git commits: 10, external calls: 5

> /policy safe
Policy switched to: safe
```

### 現在のポリシーを確認

```
> /policy
Policy mode: safe
Budgets: file writes: 20, shell commands: 30, git commits: 3, external calls: 2
Denied paths: 8 patterns
Blocked shell: 7 patterns
```

### 書き込みパス制限

デフォルトでは、`write_file` と `edit_file` は**プロジェクトルートディレクトリ**以下に制限されます。これにより、エージェントがプロジェクト外のファイルを誤って変更することを防ぎます。

追加のディレクトリへの書き込みを許可するには、`writable_paths` を使用します：

```yaml
policy:
  writable_paths:
    - "/tmp/*"              # /tmp への書き込みを許可
    - "~/other-project/*"   # 別のプロジェクトへの書き込みを許可
```

`full` プリセットではホームディレクトリ全体（`~/*`）への書き込みが許可されます。

> **注意**: 読み取り操作（`read_file`、`list_dir`、`search_files`）は `writable_paths` による制限を**受けません** — `denied_paths` に含まれないすべてのパスを読み取れます。

### デフォルトの拒否パス

以下のパスはすべてのモードでデフォルトでブロックされます（読み取り・書き込み両方）：

- `/etc/*`、`/usr/*`、`/bin/*`、`/sbin/*`、`/boot/*`
- `~/.ssh/*`、`~/.gnupg/*`
- `**/.env`、`**/.env.*`
- `**/credentials*`、`**/secrets*`

### デフォルトのブロック対象シェルパターン

- `curl | sh`、`wget | sh`
- `chmod 777`、`chmod -R 777`
- `> /dev/sd*`
- `git push --force`、`git reset --hard`

### 設定ファイルでのカスタムポリシー

```yaml
policy:
  mode: "balanced"
  max_file_writes: 50
  max_git_commits: 5
  writable_paths:
    - "/tmp/*"             # /tmp への書き込みを許可
  disabled_tools:
    - "shell"              # シェルを完全に無効化
  denied_paths:
    - "/home/user/secrets/*"
  blocked_shell_patterns:
    - "npm publish"
```

### 違反時の動作

エージェントが制限されたアクションを試みると、エラーメッセージを受け取ります（ハードストップではありません）。エージェントはアプローチを変更して対応できます：

```
> shell rm -rf /tmp/build
VIOLATION: shell command matches blocked pattern: rm -rf /

thinking: そのコマンドはブロックされている。より安全な方法を使おう...
> shell find /tmp/build -delete
OK shell
```

---

## 9. Planner-Critic-Executor ループ

複雑なゴールに対して、Open Harness は「計画→実行→検証」のループを使用します。

### 仕組み

```
ゴール
  │
  ▼
Planner ─── 計画を作成（最大5ステップ）
  │
  ▼
Plan Critic ─── 計画を検証（ルールベース）
  │
  ▼
Executor ─── 各ステップを実行
  │
  ├── ステップ 1 → スナップショット ✓
  ├── ステップ 2 → スナップショット ✓
  ├── ステップ 3 → 失敗
  │     │
  │     ▼
  │   Replanner ─── 残りのステップを再計画（1回のみ）
  │     │
  │     ▼
  │   実行を続行...
  │
  ▼
完了（または直接実行モードにフォールバック）
```

### 計画の構造

各計画には以下が含まれます：
- **ゴール**：元のタスクの説明
- **ステップ**：最大5つの順次ステップ、各ステップには：
  - タイトルと詳細な指示
  - 成功基準（検証可能な条件）
  - エージェントステップの予算（デフォルト各12ステップ）
- **前提条件**：プランナーが環境について想定した内容

### フォールバック動作

システムはグレースフルデグラデーション（緩やかな機能低下）を前提に設計されています：

1. プランナーが計画を作成できない場合 → 直接実行
2. クリティックが計画を却下した場合 → 直接実行
3. ステップが失敗した場合 → 1回再計画、その後フォールバック
4. 再計画も失敗した場合 → 完了済みのコンテキストを保持して直接実行

これにより、弱い LLM を使用していても `/goal` は常にタスクの完了を試みます。

---

## 10. チェックポイントとロールバック

git リポジトリでは、対話モードおよび自律モード（goal）が git ベースのチェックポイントを使用して安全で可逆的な実行を行います。

### 自動的な動作

`/goal` を実行すると、システムは自動的に：

1. コミットされていない変更を **スタッシュ**（安全網）
2. 作業ブランチを**作成**（`harness/goal-<タイムスタンプ>`）
3. ファイル書き込み5回ごと、および各計画ステップ後に**スナップショット**を取得
4. テストが失敗した場合に**ロールバック**
5. 成功時にすべての変更をブランチに**スカッシュマージ**
6. 元のスタッシュした変更を**復元**

### 実行例

```
> /goal User モデルに入力バリデーションを追加する

[checkpoint] コミットされていない変更を2件スタッシュしました
[checkpoint] ブランチ harness/goal-1709312400 を作成
...
[checkpoint] スナップショット: バリデーション関数を追加
...
[checkpoint] スナップショット: モデルにバリデータを追加
...
> run_tests
FAIL: test_user_email_validation
[checkpoint] ロールバック先: バリデーション関数を追加
...
（エージェントが別のアプローチでリトライ）
...
> run_tests
OK: 12 passed
[checkpoint] main にスカッシュマージしました
[checkpoint] スタッシュした変更を復元しました
```

### Git 自動初期化

git リポジトリではないプロジェクトでは、自動的に `git init` と初期コミットが実行され、すべての編集を元に戻せるようになります。初期コミットが失敗した場合は `.git` ディレクトリが削除され、チェックポイント機能はスキップされます。

### 対話モードの保護

対話モードでは、セッションレベルのチェックポイントが以下の保護を提供します：

- 最初のツール使用時に作業ブランチを作成
- ファイル書き込み5回ごとに自動スナップショット
- テスト失敗時に自動ロールバック
- セッション終了時に変更をスカッシュマージして元のブランチに統合

### 安全性の保証

- コミットされていない作業は git stash で保護される
- 失敗した変更は自動的に元に戻される
- 作業ブランチは完了後にクリーンアップされる
- 元のブランチは実行中に直接変更されることはない

---

## 11. プロジェクトメモリ

Open Harness はプロジェクト固有のパターンをセッションをまたいで自動的に学習・記憶します。

### 学習する内容

| 種類 | 例 |
|------|-----|
| **pattern** | 「テストコマンド: pytest -x --tb=short」 |
| **structure** | 「設定ファイルは src/config/ にある」 |
| **error** | 「ImportError → 仮想環境の有効化を確認」 |
| **runbook** | 複数ステップのワークフローレシピ |

### 学習済みメモリの表示

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

### 仕組み

1. **観察**：エージェントがツール使用中にパターンに気づく
2. **昇格**：パターンは2回以上観察されないと永続化されない（ノイズ低減）
3. **スコアリング**：有用なメモリのスコアが上がり、使われないメモリは減衰する
4. **注入**：上位のメモリが自律モードのプロンプトに含められる
5. **剪定**：スコアが 0.15 未満または60日以上前のメモリは自動削除

### メモリはプロジェクトごと

各プロジェクトディレクトリには独自の学習済みメモリがあります。別のプロジェクトに `cd` して `harness` を実行すると、そのプロジェクト固有のメモリが読み込まれます。

---

## 12. 外部エージェント連携

Open Harness は**オーケストレーターアーキテクチャ**を採用しています。ローカルLLMが計画・判断・ツール選択を担当し、コード生成・分析・デバッグなどの複雑なタスクは外部AIエージェントに委譲します。

### なぜオーケストレーター？

ローカルLLMは指示に従い簡単な判断を下すのは得意ですが、コード生成や複雑な推論は苦手です。強力な外部エージェント（Claude, Codex, Gemini）に委譲することで、両方の利点を活用できます：

- **ローカルLLM** → 高速な計画、ツール選択、調整
- **外部エージェント** → 高品質なコード生成、分析、デバッグ

### Claude Code（Anthropic）— 推奨

得意分野：コード生成、コード分析、リファクタリング、複雑な推論

```yaml
external_agents:
  claude:
    enabled: true
    command: "claude"
```

自律モードでの使用例：

```
> /goal 認証モジュールをJWTトークン方式にリファクタリングする
```

エージェントは Claude Code に委譲します：

```
> claude_code "src/auth.py をセッションベースからJWTトークン方式にリファクタリングしてください。後方互換性を維持してください。"
```

### Codex（OpenAI）

得意分野：コード生成、デバッグ、自律的なコーディングタスク

```yaml
external_agents:
  codex:
    enabled: true
    command: "codex"
```

### Gemini CLI（Google）

得意分野：コードレビュー、分析、別の視点からの提案

```yaml
external_agents:
  gemini:
    enabled: true
    command: "gemini"
```

### ルーティングのカスタマイズ

デフォルトのルーティング情報は内蔵されていますが、`description` と `strengths` で上書きしてオーケストレーターの委譲判断をカスタマイズできます。

```yaml
external_agents:
  claude:
    enabled: true
    command: "claude"
    description: "複雑なリファクタリング、日本語テキスト、計画"
    strengths: ["refactoring", "japanese_text", "planning"]
```

### レート制限フォールバック

外部エージェントの利用枠がいっぱいになると、自動的に検出してフォールバックエージェントに切り替えます。クールダウン時間が経過すると元のエージェントに戻ります。

### ポリシーによる制限

外部呼び出しはポリシーで制限されています：
- `safe`：最大10回
- `balanced`：無制限（オーケストレーターが自由に委譲）
- `full`：無制限

### 前提条件

外部エージェントは PATH 上にインストールされ、アクセス可能である必要があります。Open Harness は起動時に利用可能性をチェックし、見つかったツールのみを登録します。

---

## 13. プロジェクトごとの設定

プロジェクトディレクトリに `open_harness.yaml` を配置することで、プロジェクトごとの設定をカスタマイズできます。

### 使用例：プロジェクトごとに異なる LLM エンドポイント

```
~/projects/
├── web-app/
│   └── open_harness.yaml    ← リモートGPUサーバーを使用
├── scripts/
│   └── open_harness.yaml    ← ローカルOllamaを使用
└── ~/.open_harness/
    └── open_harness.yaml    ← グローバルフォールバック
```

### 例：本番プロジェクトのツールを制限する

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

### 例：カスタムテストコマンド

プロジェクトルートに `.harness/context.md` を配置して追加コンテキストを提供できます：

```markdown
# プロジェクトコンテキスト

これは Django アプリケーションです。テストの実行方法：
  python manage.py test --parallel

API は apps/api/ にあります。apps/legacy/ は変更しないでください。
```

Open Harness はこのファイルを自動的に読み込み、エージェントプロンプトに含めます。

### プロジェクト検出

Open Harness はプロジェクトタイプを自動検出して適応します：

| 検出されるファイル | プロジェクトタイプ | 自動テストコマンド |
|-------------------|-------------------|-------------------|
| `pyproject.toml` / `setup.py` | Python | `python3 -m pytest` |
| `package.json` | JavaScript | `npm test` |
| `Cargo.toml` | Rust | `cargo test` |
| `go.mod` | Go | `go test ./...` |

---

## 14. 実践例

### 例1：コードレビュー

```
> src/auth.py を読んでセキュリティ上の問題を指摘して

> read_file src/auth.py
...

以下の問題を発見しました：
1. 23行目: SQLクエリがパラメータ化クエリではなく文字列フォーマットを使用
2. 45行目: パスワード比較が定数時間ではない
3. 67行目: セッショントークンのエントロピーが不十分
```

### 例2：テストの自動生成

```
> /goal src/utils.py の包括的なユニットテストを生成する

thinking: まずモジュールを分析しよう...
> read_file src/utils.py

Plan:
  Step 1: utils.py のすべての関数を分析
  Step 2: 各関数のテストケースを含むテストファイルを作成
  Step 3: テストを実行して失敗があれば修正

> write_file tests/test_utils.py ...
> run_tests tests/test_utils.py
OK: 12 passed

Goal completed in 45.2s
```

### 例3：安全なリファクタリング

```
> /policy safe
> /goal src/api/ 全体で変数名 'data' を 'payload' にリネームする

[checkpoint] 変更をスタッシュし、作業ブランチを作成
...
> search_files "data" src/api/
> edit_file src/api/handlers.py ...
> edit_file src/api/models.py ...
> run_tests
OK: 28 passed
[checkpoint] main にスカッシュマージ

Goal completed in 32.1s
```

### 例4：マルチステップデバッグ

```
> /goal /users エンドポイントが500を返す。バグを見つけて修正して。

Plan:
  Step 1: エンドポイントハンドラを読んでエラーを追跡
  Step 2: 根本原因を特定
  Step 3: 修正を実装
  Step 4: 回帰テストを追加
  Step 5: すべてのテストがパスすることを確認

> read_file src/api/users.py
> shell python -c "import src.api.users"
FAIL: ImportError: cannot import 'UserSchema' from 'models'

thinking: import パスが変更されたが users.py が更新されていない...

> edit_file src/api/users.py "from models import UserSchema" "from src.models import UserSchema"
> run_tests tests/test_users.py
OK: 5 passed

Goal completed in 38.7s
```

### 例5：バックグラウンドでのコード生成

```
> /submit src/api/ のすべてのエンドポイントのAPIドキュメントを生成する

Task b2c3d4e5 queued
Log: ~/.open_harness/logs/task_...

（他の作業を続行）

> config モジュールは何をしている？
...

（通知ベルが鳴る）
OK Task b2c3d4e5 complete: src/api/ のすべてのエンドポイント...

> /result b2c3d4e5
（生成されたドキュメントが表示される）
```

### 例6：タスクに応じたティアの使い分け

```
> /tier small
> このエラーログを要約して
（small モデルで高速レスポンス）

> /tier large
> /goal マルチテナンシー対応にデータベーススキーマを再設計する
（large モデルでトークン数を増やして複雑なタスクに対応）
```

### 例7：外部エージェントへの委譲

```
> /goal Codex に包括的なテストスイートを生成させて、その結果をレビューする

> codex "src/calculator.py のエッジケースを含む pytest テストを生成してください"
OK codex
（テストコードが生成される）

> write_file tests/test_calculator.py ...
> run_tests tests/test_calculator.py
OK: 15 passed

Goal completed in 120.5s
```

---

## 15. トラブルシューティング

### 接続拒否エラー

```
Error: Connection refused at http://localhost:1234/v1
```

**対処法**：LLM サーバーが起動していることを確認してください。

```bash
# LM Studio: GUIから起動し、"Local Server" を有効にする
# Ollama:
ollama serve
```

### 設定ファイルが見つからない

```
Config: defaults (no open_harness.yaml found)
```

**対処法**：プロジェクトディレクトリまたは `~/.open_harness/` に `open_harness.yaml` を作成してください。

### モデルが不正な出力を返す / ツール呼び出しが失敗する

小さな LLM では想定される動作です。補償エンジンが自動的に対処します：

1. `refine_prompt` — 修正ヒントを追加
2. `add_examples` — ツール呼び出しの例を追加
3. `escalate_model` — より大きなティアを試す

すべてのリトライが失敗する場合は、より高性能なモデルの使用を検討してください。

### ポリシーによるブロック

```
VIOLATION: file write denied by path restriction: .env
```

**対処法**：ポリシーを調整するか、別のアプローチを使用してください。

```
> /policy full          # すべての制限を解除（注意して使用）
> /policy balanced      # デフォルトの制限
```

### テストが自動検出されない

テストコマンドが検出されない場合：

1. `/project` の出力を確認
2. プロジェクトルートに `.harness/context.md` を作成して明示的にテストコマンドを指定
3. または設定ファイルで指定：

```yaml
# プロジェクト検出で処理されますが、オーバーライドも可能：
# プロジェクトルートに .harness/context.md を配置し、以下を記述：
# Test command: python -m pytest -x tests/
```

### バックグラウンドタスクがハングする

```
> /tasks
│ a3f2e1b0 │ running │ ...  │ 600s │
```

タスクが LLM 呼び出しでスタックしている可能性があります。次回起動時に、停滞中の running タスクは自動的に failed としてマークされます（クラッシュリカバリー）。

### メモリデータベースのロック

SQLite のロックエラーが表示される場合：

**対処法**：1マシンあたり1つの `harness` プロセスのみ実行してください。データベースはメインスレッドとバックグラウンドタスク間の並行処理に WAL モードを使用していますが、プロセス間の並行処理には対応していません。

---

## 16. @ファイル参照とTab補完

REPL では `@path/to/file` の形式でファイル内容をメッセージに添付できます。

### 基本的な使い方

```
> このファイルをレビューして @src/open_harness/cli.py
```

### Tab 補完

`@` に続けて Tab を押すと、ファイルやディレクトリの補完候補が表示されます。

- ディレクトリは `/` 付きで表示されます
- ファイルサイズがメタデータとして表示されます

### 除外対象

以下のディレクトリは補完候補から自動的に除外されます：

- `.git`
- `__pycache__`
- `node_modules`
- `.venv`

### セキュリティ

プロジェクトルート外へのパストラバーサル（例：`@../../etc/passwd`）はブロックされます。

---

## 17. モード切替

REPL には3つの入力モードがあり、**Shift+Tab** で切り替えられます。

| モード | 色 | 説明 |
|--------|-----|------|
| `chat` | 緑 | 対話モード（デフォルト） |
| `goal` | 黄 | 自律実行モード — 入力がゴールになる |
| `submit` | 青 | バックグラウンドキュー — 入力がタスクとして送信される |

現在のモードはボトムツールバーに表示されます。

`chat` モードでは通常の対話が行われます。`goal` モードに切り替えると、入力がそのまま `/goal` コマンドとして実行されます。`submit` モードでは入力が `/submit` コマンドとしてバックグラウンドキューに送信されます。

---

## 18. レート制限フォールバック

外部エージェントの利用枠がいっぱいになった場合、Open Harness は自動的にフォールバックエージェントに切り替えます。

### 検出パターン

以下のパターンが外部エージェントの出力から検出されるとレート制限と判断されます：

- `429`
- `rate limit`
- `quota exceeded`
- `too many requests`

### クールダウン時間

レスポンスに含まれる `try again in X minutes` などの表現からクールダウン時間を解析します。解析できない場合はデフォルトで15分のクールダウンが適用されます。

### フォールバック順序

| レート制限されたエージェント | フォールバック先 |
|---------------------------|----------------|
| `claude_code` | `codex` → `gemini_cli` |
| `codex` | `claude_code` → `gemini_cli` |
| `gemini_cli` | `claude_code` → `codex` |

### 自動復帰

クールダウン期限が切れると、自動的に元のエージェントに復帰します。

### すべてのエージェントがレート制限された場合

すべてのエージェントがレート制限された場合は、元のエージェントでそのまま実行を試みます。失敗した場合は補償エンジンが対応します。

---

## 付録：コマンドラインリファレンス

```
harness [オプション]

オプション:
  -c, --config パス       open_harness.yaml のパス
  -t, --tier ティア       モデルティア（small, medium, large）
  -g, --goal テキスト     ゴールを非対話的に実行して終了
  -v, --verbose           デバッグログを有効化
  --help                  ヘルプを表示
```

### 非対話モード

```bash
# ゴールを実行して終了
harness --goal "tests/test_auth.py の失敗しているテストを修正する"

# 特定のティアを指定
harness --tier large --goal "APIモジュール全体をリファクタリングする"

# カスタム設定ファイルを指定
harness --config ./my_config.yaml --goal "ログ機能を追加する"

# デバッグ用の詳細出力
harness -v --goal "テストが失敗する原因は？"
```
