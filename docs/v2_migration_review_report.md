# Open Harness v2 マイグレーション実装レビュー

作成日: 2026-03-01  
対象:
- `open_harness/` フォルダ
- `src/open_harness_v2/` 実装（`tests/v2` の実質対象）
- `tests/v2/`
- 方針文書: `docs/redesign_plan.md`, `docs/v2_remaining_plan.md`

---

## 結論サマリ

v2 の骨格（`Orchestrator`/`EventBus`/`Memory`/`Task`/`Checkpoint`）は揃っていますが、**運用時に致命的になりうる未接続・契約不整合が複数残っています**。  
特に以下は優先修正が必要です。

1. `balanced/safe` で `write_file/edit_file` が実質全拒否になる初期化不備
2. タスクキューの「待機中タスク cancel」で状態不整合が起きる
3. SessionMemory が CLI で実質未接続（load/bind 未実装）
4. LLM パイプラインが未配線で、弱モデル補償が有効化されていない

---

## 重大指摘（High/Critical）

### 1. `PolicyEngine` の project root 未設定で write が常時拒否される（Critical）
- 根拠:
  - `PolicyEngine._check_write_path()` は `self._project_root` がない場合、`writable_paths` にも一致しないと拒否  
    `src/open_harness_v2/policy/engine.py:292-320`
  - CLI で `PolicyEngine` 生成後に `set_project_root()` を呼んでいない  
    `src/open_harness_v2/cli.py:155-173`
  - `balanced` 既定では `writable_paths` が空  
    `src/open_harness_v2/config.py:89-94`, `src/open_harness_v2/config.py:67`
- 影響:
  - デフォルト運用で `write_file/edit_file` が拒否され、エージェントが編集不能になる。
- 推奨:
  - `_build_components()` で `policy.set_project_root(Path.cwd())` を必ず実行。
  - 回帰防止テストを追加（`balanced` で project root 内 write が許可されること）。

### 2. 待機中タスクの `cancel()` で状態更新漏れ・`_running` リーク（Critical）
- 根拠:
  - `submit()` 時点で全タスクを `_running` に登録  
    `src/open_harness_v2/tasks/manager.py:53-58`
  - `_execute()` は `async with self._semaphore:` の外側で `CancelledError` を捕捉していない  
    `src/open_harness_v2/tasks/manager.py:60-113`
  - `cancel()` は `_running` にあれば即 `task.cancel()` して return。待機中タスクの DB 更新経路に入らない  
    `src/open_harness_v2/tasks/manager.py:114-127`
- 影響:
  - キュー待機中キャンセル時に `CANCELLED` へ遷移しない/`_running` に残留する可能性。
- 推奨:
  - `_execute()` 全体を `try/except/finally` で囲み、セマフォ待機中 cancel でも状態更新・`_running.pop()` する。
  - `tests/v2` に「待機中タスク cancel」ケースを追加。

### 3. SessionMemory が CLI 経路で実質機能していない（High）
- 根拠:
  - `SessionMemory` は `bind(session_id, messages)` 前提  
    `src/open_harness_v2/memory/session.py:59-70`
  - CLI は `attach()` のみで、`bind()`/`load()` が呼ばれていない  
    `src/open_harness_v2/cli.py:145-160`, `src/open_harness_v2/cli.py:462-493`
  - 方針文書では `load()` 結果を Context へ注入する設計  
    `docs/v2_remaining_plan.md:73-77`, `docs/v2_remaining_plan.md:136-151`
- 影響:
  - セッション継続（再起動後復元）が動作しない。
- 推奨:
  - CLI 起動時に `session_id` を決めて `load()`、`AgentContext.history` に注入。
  - 実行中メッセージ参照を `bind()` で接続。

### 4. LLM ミドルウェア（PromptOptimizer/ErrorRecovery）が未配線（High）
- 根拠:
  - Orchestrator 既定 pipeline は生 `MiddlewarePipeline(router.get_client())` のみ  
    `src/open_harness_v2/core/orchestrator.py:65`
  - CLI 側でも middleware を `use()` していない  
    `src/open_harness_v2/cli.py:167-173`
  - 方針文書では「弱モデル補償をパイプラインに分散」し、Phase1 で行う想定  
    `docs/redesign_plan.md:20`, `docs/redesign_plan.md:84-123`, `docs/redesign_plan.md:292-297`
- 影響:
  - v1 の価値だった補償挙動が実運用経路で有効にならない。
- 推奨:
  - CLI で `PromptOptimizerMiddleware` + `ErrorRecoveryMiddleware` を組み込み。
  - `ModelRouter.escalate()` と recovery の接続を追加。

### 5. LLM リクエストに `tools` スキーマが渡っていない（High）
- 根拠:
  - Orchestrator が `LLMRequest` 作成時に `tools/tool_choice` を設定していない  
    `src/open_harness_v2/core/orchestrator.py:114-117`
  - Registry には schema 生成 API があるが未使用  
    `src/open_harness_v2/tools/registry.py:119-121`
  - PromptOptimizer は `request.tools` がある場合のみツールヒントを追加  
    `src/open_harness_v2/llm/prompt_optimizer.py:78-85`
- 影響:
  - ネイティブ function calling の活用率低下。ツール呼び出し品質が不安定。
- 推奨:
  - `request.tools = registry.get_openai_schemas()` を標準化。
  - `tool_choice="auto"` のデフォルト検討。

### 6. Executor ↔ Renderer のイベント契約が不一致（High）
- 根拠:
  - Executor emits:
    - `TOOL_EXECUTING` with `arguments`  
      `src/open_harness_v2/core/executor.py:100-103`
    - `TOOL_EXECUTED` with `output_length` のみ  
      `src/open_harness_v2/core/executor.py:112-115`
    - `TOOL_ERROR` with `error`  
      `src/open_harness_v2/core/executor.py:117-120`
  - Renderer expects:
    - `TOOL_EXECUTING` の `args`  
      `src/open_harness_v2/ui/renderer.py:84-86`
    - `TOOL_EXECUTED` の `success` と `output`  
      `src/open_harness_v2/ui/renderer.py:88-105`
    - `TOOL_ERROR` も同一ハンドラに流す  
      `src/open_harness_v2/ui/renderer.py:52`
- 影響:
  - 実表示で引数/出力/失敗表示が崩れる。失敗が成功に見える可能性がある。
- 推奨:
  - Event payload を型で固定（例: `ToolExecutedEventData`）。
  - `TOOL_ERROR` 専用レンダラを分離。

### 7. `MemoryStore` の SQLite 接続が並行アクセス非同期化に対して無保護（High）
- 根拠:
  - `check_same_thread=False` の単一 connection を共有  
    `src/open_harness_v2/memory/store.py:56-59`
  - `TaskStore` は lock を持つが `MemoryStore` には lock がない  
    `src/open_harness_v2/tasks/store.py:45`（比較）
- 影響:
  - 複数 `asyncio.to_thread` 呼び出しが重なると race/`database is locked` のリスク。
- 推奨:
  - `threading.Lock` 追加、または接続を都度分離、または `aiosqlite` へ移行。

### 8. Checkpoint snapshot が commit 失敗時でも成功扱いされる（High）
- 根拠:
  - `git commit` を `check=False` で実行後、return code を確認せず `HEAD` を snapshot として保存  
    `src/open_harness_v2/checkpoint/engine.py:205-217`
- 影響:
  - snapshot の信頼性が崩れ、ロールバック期待が外れる。
- 推奨:
  - commit return code を検証し、失敗時は snapshot 追加しない。
  - 失敗理由を EventBus へ通知。

---

## 中程度の指摘（Medium）

### 9. CLI 終了時に LLM client close とバックグラウンド task shutdown がない
- 根拠:
  - cleanup は memory/task store の close のみ  
    `src/open_harness_v2/cli.py:489-493`
- 影響:
  - ソケット/バックグラウンド処理が残る可能性。

### 10. `open_harness/` とルート実装が二重化しており移行時の認知負荷が高い
- 根拠:
  - ルート `pyproject.toml` は `src/open_harness` と `src/open_harness_v2` をパッケージ化  
    `pyproject.toml:33-35`
  - 一方で `open_harness/pyproject.toml` に独立した `open-harness` 定義（version 0.1.0）が存在  
    `open_harness/pyproject.toml:1-25`
- 影響:
  - どの実装が正なのか曖昧化し、レビュー/修正/リリース事故の原因になる。

---

## `tests/v2` レビュー（品質・不足観点）

### 良い点
- レイヤ単位テストが広く揃っている（context/event_bus/policy/registry など）。
- `checkpoint`・`memory`・`tasks` の基本ケースが存在し、v2主要機能の最小保証はある。

### 不足・改善点

1. 重大バグを取り逃している
- `TaskManager` の「待機中 cancel」ケースが未テスト  
  `tests/v2/test_tasks.py`（`test_cancel_running` のみ: `194-214`）
- Policy の project root 初期化が CLI 経路で検証されていない  
  `tests/v2/test_cli.py:60-127`
- Executor/Renderer の実イベント契約差分を検出できていない  
  `tests/v2/test_renderer.py:50-82`, `tests/v2/test_executor.py:84-96`

2. 時間依存テストが多く flaky になりやすい
- 固定 `asyncio.sleep()` 依存  
  `tests/v2/test_tasks.py:156-171`, `188-212`, `258-260`

3. 統合テスト運用の分離が弱い
- `tests/v2/test_integration.py` は実環境依存が強く、ローカルCLI認証状態でも結果が揺れる。

---

## 実行ベース確認メモ

- `tests/v2` のうち多くは個別実行で pass を確認。
- この実行環境では `asyncio.to_thread()` を使うテスト群（file/memory/tasks 系）がハングし、完走確認に制約あり（環境依存）。

---

## 追加提案（優先順）

1. **即時修正**: `policy.set_project_root(Path.cwd())` を CLI 初期化に追加。
2. **即時修正**: TaskManager の cancel 経路を再設計（待機中キャンセルを正しく `CANCELLED` 化）。
3. **即時修正**: Executor/Renderer の event payload 契約を統一。
4. **短期**: SessionMemory の `load/bind/save` を CLI/Orchestrator に接続。
5. **短期**: middleware 実配線（PromptOptimizer + ErrorRecovery + Router escalate）。
6. **短期**: `tests/v2` に上記回帰テストを追加。

