# Open Harness v2 — 残り機能の実装計画

> v1 を踏まえつつ、ゼロから自由に再設計するなら？

## 対象機能

| 機能 | v1 の実装 | v2 での方針 |
|------|-----------|-------------|
| Memory | 2層 (会話 + プロジェクト学習) | 1層に統合、EventBus 駆動 |
| Task Queue | threading + SQLite | asyncio ネイティブ |
| Checkpoint | git branch ベース | EventBus subscriber、Policy 連携 |

---

## 1. Memory — `memory/`

### v1 の問題点

- **会話メモリとプロジェクトメモリが混在** — 2つの Store、2つの SQLite テーブル、API が重複
- **自動学習が中途半端** — 限られたツール名のハードコードパターンマッチ、runbook は未使用
- **ユーザー制御なし** — 記憶の追加・削除・編集ができない
- **毎回 DB クエリ** — prompt 生成のたびに全メモリを再取得

### v2 の再設計

#### 核心思想: **「2種類のメモリだけ」**

```
SessionMemory          — 会話履歴の永続化 (セッション間で引き継ぎ)
ProjectMemory          — プロジェクト固有の事実 (ユーザーが明示的に教える)
```

v1 の「自動学習」は削除。理由:
- パターンマッチの品質が低く、ノイズが多い
- LLM 自身が `/remember` コマンドで明示保存する方が正確
- 200件の上限管理やスコアリングの複雑さが不要になる

#### ファイル構成

```
src/open_harness_v2/memory/
├── __init__.py
├── store.py           # MemoryStore (SQLite、両方のテーブル管理)
├── session.py         # SessionMemory (会話永続化)
└── project.py         # ProjectMemory (KV ファクトストア)
```

#### SessionMemory — 会話の永続化

```python
class SessionMemory:
    """セッション間で会話履歴を保持する。

    EventBus の AGENT_DONE を購読し、ゴール完了時に自動保存。
    """

    def __init__(self, store: MemoryStore, max_turns: int = 50):
        ...

    async def save(self, session_id: str, messages: list[dict]) -> None:
        """会話を DB に保存 (atomic replace)。"""

    async def load(self, session_id: str) -> list[dict]:
        """保存済み会話を復元。"""

    async def clear(self, session_id: str) -> None:
        """セッションをクリア。"""

    def attach(self, event_bus: EventBus) -> None:
        """AGENT_DONE イベントで自動保存を設定。"""
```

変更点 (v1 比):
- **async API** — SQLite は `aiosqlite` or `run_in_executor`
- **EventBus 駆動** — 明示的な save/load 呼び出し不要、イベントで自動化
- **ContextStore 連携** — `load()` の結果を `HistoryLayer` に直接注入

#### ProjectMemory — プロジェクトの事実

```python
class ProjectMemory:
    """プロジェクト固有の永続的な事実を管理する。

    ユーザーまたは LLM が明示的に保存する KV ストア。
    自動学習は行わない。
    """

    def __init__(self, store: MemoryStore, project_id: str):
        ...

    async def remember(self, key: str, value: str) -> None:
        """事実を保存 (upsert)。"""

    async def forget(self, key: str) -> None:
        """事実を削除。"""

    async def recall(self, key: str) -> str | None:
        """事実を取得。"""

    async def list_all(self) -> list[tuple[str, str]]:
        """全事実を一覧。"""

    def build_context_block(self) -> str:
        """システムプロンプトに注入する文字列を生成。
        キャッシュ付き — DB クエリは変更時のみ。
        """
```

変更点 (v1 比):
- **自動学習を廃止** — `on_tool_result()` の複雑なパターンマッチを削除
- **ユーザー制御** — `/remember`, `/forget`, `/memories` REPL コマンド
- **キャッシュ** — `build_context_block()` は変更時のみ再構築
- **サニタイズ維持** — prompt injection 対策は v1 から継承

#### MemoryStore — 共通 SQLite バックエンド

```python
class MemoryStore:
    """SQLite バックエンド。2テーブルを管理。"""

    def __init__(self, db_path: str = "~/.open_harness/memory.db"):
        ...

    # sessions テーブル
    async def save_messages(self, session_id: str, messages: list[dict]) -> None: ...
    async def load_messages(self, session_id: str) -> list[dict]: ...

    # facts テーブル
    async def upsert_fact(self, project_id: str, key: str, value: str) -> None: ...
    async def delete_fact(self, project_id: str, key: str) -> None: ...
    async def get_facts(self, project_id: str) -> list[tuple[str, str]]: ...

    async def close(self) -> None: ...
```

#### Orchestrator / CLI 連携

```python
# cli.py — 起動時
session_memory = SessionMemory(store, max_turns=50)
project_memory = ProjectMemory(store, project_id=_project_hash())
session_memory.attach(event_bus)  # AGENT_DONE で自動保存

# Orchestrator — コンテキスト注入
context.system.extra = project_memory.build_context_block()

# REPL コマンド
"/remember <key> <value>"  → project_memory.remember(key, value)
"/forget <key>"            → project_memory.forget(key)
"/memories"                → project_memory.list_all()
```

#### remember ツール (LLM 用)

```python
class RememberTool(Tool):
    """LLM が実行中に事実を保存するためのツール。"""
    name = "remember"
    parameters = [
        ToolParameter(name="key", type="string", description="Fact identifier"),
        ToolParameter(name="value", type="string", description="Fact content"),
    ]

    async def execute(self, **kwargs) -> ToolResult:
        await self.project_memory.remember(kwargs["key"], kwargs["value"])
        return ToolResult(success=True, output=f"Remembered: {kwargs['key']}")
```

---

## 2. Task Queue — `tasks/`

### v1 の問題点

- **threading ベース** — v2 の async アーキテクチャと不整合
- **逐次実行のみ** — 優先度・依存関係なし
- **通知が原始的** — ベル音 (`\a`) とキュー経由のポーリング

### v2 の再設計

#### 核心思想: **「asyncio.Task のラッパー」**

v1 は threading.Thread + queue.Queue だったが、
v2 では asyncio.Task をそのまま活用し、EventBus で通知する。

#### ファイル構成

```
src/open_harness_v2/tasks/
├── __init__.py
├── store.py           # TaskStore (SQLite 永続化)
├── record.py          # TaskRecord, TaskStatus
└── manager.py         # TaskManager (asyncio ベース)
```

#### TaskRecord / TaskStatus

```python
class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskRecord:
    id: str                           # 8文字 hex UUID
    goal: str
    status: TaskStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: str | None = None
    error: str | None = None

    @property
    def elapsed(self) -> float | None: ...

    @property
    def is_terminal(self) -> bool: ...
```

v1 からの変更: `log_path` を削除。ログは EventBus 経由で別のサブスクライバーが処理する。

#### TaskManager — asyncio ネイティブ

```python
class TaskManager:
    """asyncio ベースのタスクキュー。

    submit() で asyncio.Task を生成し、EventBus で進捗を通知する。
    """

    def __init__(
        self,
        store: TaskStore,
        event_bus: EventBus,
        orchestrator_factory: Callable[[], Orchestrator],
        max_concurrent: int = 1,  # ローカル LLM は 1 推奨
    ):
        self._store = store
        self._event_bus = event_bus
        self._factory = orchestrator_factory
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running: dict[str, asyncio.Task] = {}

    async def submit(self, goal: str) -> TaskRecord:
        """タスクをキューに追加し、非同期実行を開始する。"""
        record = self._store.create(goal)
        task = asyncio.create_task(self._execute(record))
        self._running[record.id] = task
        return record

    async def _execute(self, record: TaskRecord) -> None:
        """セマフォで並行数を制限しつつタスクを実行する。"""
        async with self._semaphore:
            self._store.update_status(record.id, TaskStatus.RUNNING)
            await self._event_bus.emit(AgentEvent(
                EventType.TASK_STARTED, {"task_id": record.id, "goal": record.goal}
            ))

            orchestrator = self._factory()
            try:
                result = await orchestrator.run(record.goal)
                self._store.update_result(record.id, TaskStatus.SUCCEEDED, result=result)
                await self._event_bus.emit(AgentEvent(
                    EventType.TASK_COMPLETED,
                    {"task_id": record.id, "status": "succeeded", "result": result},
                ))
            except asyncio.CancelledError:
                self._store.update_status(record.id, TaskStatus.CANCELLED)
                await self._event_bus.emit(AgentEvent(
                    EventType.TASK_COMPLETED,
                    {"task_id": record.id, "status": "cancelled"},
                ))
            except Exception as exc:
                self._store.update_result(
                    record.id, TaskStatus.FAILED, error=str(exc)
                )
                await self._event_bus.emit(AgentEvent(
                    EventType.TASK_COMPLETED,
                    {"task_id": record.id, "status": "failed", "error": str(exc)},
                ))
            finally:
                self._running.pop(record.id, None)

    async def cancel(self, task_id: str) -> bool:
        """実行中のタスクをキャンセルする。"""
        task = self._running.get(task_id)
        if task and not task.done():
            task.cancel()
            return True
        # キュー待ちの場合は直接ステータス変更
        record = self._store.get(task_id)
        if record and record.status == TaskStatus.QUEUED:
            self._store.update_status(task_id, TaskStatus.CANCELLED)
            return True
        return False

    def list_tasks(self, limit: int = 20) -> list[TaskRecord]:
        return self._store.list_recent(limit)

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._store.get(task_id)

    async def recover(self) -> int:
        """起動時: RUNNING のまま残ったタスクを QUEUED に戻し再実行する。"""
        return self._store.recover_stale()
```

v1 からの変更:
- **threading → asyncio** — `asyncio.Task` + `Semaphore` で並行制御
- **EventBus 通知** — ベル音ではなく `TASK_STARTED` / `TASK_COMPLETED` イベント
- **Orchestrator 分離** — `agent_factory` ではなく `orchestrator_factory`
- **ログファイル廃止** — EventBus subscriber がログを担当
- **`max_concurrent`** — 将来のマルチ GPU / API 並行対応の拡張点

#### REPL コマンド

```
/submit <goal>        — バックグラウンドタスク追加
/tasks                — タスク一覧 (status, elapsed)
/result <id>          — タスク詳細
/cancel <id>          — キャンセル
```

#### 新しい EventType (types.py に追加)

```python
TASK_STARTED = "task.started"
TASK_COMPLETED = "task.completed"
```

---

## 3. Checkpoint — `checkpoint/`

### v1 の問題点

- **Policy との連携が薄い** — スナップショットタイミングがハードコード (10 writes)
- **テスト失敗時のみロールバック** — 予防的なロールバックなし
- **監査証跡なし** — スナップショットはマージ後に消える
- **stash pop 失敗** — ユーザーの変更が消える可能性

### v2 の再設計

#### 核心思想: **「EventBus Subscriber として実装」**

v1 では Orchestrator (agent.py) が直接 `ckpt.snapshot()` を呼んでいた。
v2 では Checkpoint エンジンが EventBus を購読し、**ツール実行イベントから自律的にスナップショットを取る**。

Orchestrator はチェックポイントの存在を知らない。

#### ファイル構成

```
src/open_harness_v2/checkpoint/
├── __init__.py
└── engine.py          # CheckpointEngine (EventBus subscriber)
```

#### CheckpointEngine

```python
@dataclass
class Snapshot:
    commit_hash: str
    description: str
    timestamp: float

class CheckpointEngine:
    """Git ベースのトランザクショナルチェックポイント。

    EventBus を購読し、ツール実行後に自律的にスナップショットを取る。
    Orchestrator は Checkpoint の存在を知らない (完全分離)。
    """

    def __init__(
        self,
        project_root: Path,
        policy: PolicySpec,
        event_bus: EventBus,
    ):
        self._root = project_root
        self._policy = policy
        self._snapshots: list[Snapshot] = []
        self._writes_since_snapshot = 0
        self._original_branch: str | None = None
        self._work_branch: str | None = None

    def attach(self, event_bus: EventBus) -> None:
        """EventBus に接続し、以下のイベントを購読する:
        - AGENT_STARTED  → begin() 呼び出し
        - TOOL_EXECUTED   → スナップショット判定
        - AGENT_DONE      → finish(keep=True)
        - AGENT_ERROR     → finish(keep=False) + ロールバック
        - AGENT_CANCELLED → finish(keep=False)
        """
        event_bus.subscribe(EventType.AGENT_STARTED, self._on_agent_started)
        event_bus.subscribe(EventType.TOOL_EXECUTED, self._on_tool_executed)
        event_bus.subscribe(EventType.AGENT_DONE, self._on_agent_done)
        event_bus.subscribe(EventType.AGENT_ERROR, self._on_agent_error)
        event_bus.subscribe(EventType.AGENT_CANCELLED, self._on_agent_cancelled)

    def _on_tool_executed(self, event: AgentEvent) -> None:
        """ツール実行後のスナップショット判定。

        Policy モードに応じてスナップショット頻度を変える:
        - safe:     write 5回ごと + git/shell 後は毎回
        - balanced: write 10回ごと + git 後のみ
        - full:     スナップショットなし
        """
        tool = event.data.get("tool", "")
        success = event.data.get("success", True)
        if not success:
            return

        category = self._categorize(tool)
        interval = self._snapshot_interval()

        if category == "write":
            self._writes_since_snapshot += 1
            if self._writes_since_snapshot >= interval:
                self._snapshot(f"after {self._writes_since_snapshot} writes")
                self._writes_since_snapshot = 0
        elif category == "git":
            self._snapshot(f"after {tool}")
        elif category == "execute" and self._policy.mode == "safe":
            self._snapshot(f"after shell: {tool}")

    def _snapshot_interval(self) -> int:
        """Policy モードに応じたスナップショット間隔。"""
        return {"safe": 5, "balanced": 10, "full": 999999}[self._policy.mode]

    # --- Git 操作 (v1 から継承) ---

    def _begin(self) -> None:
        """作業ブランチを作成し、既存変更を stash する。"""
        ...

    def _snapshot(self, description: str) -> Snapshot:
        """現在の状態をコミット。"""
        ...

    def _rollback(self, to: Snapshot | None = None) -> None:
        """指定スナップショットまでロールバック。"""
        ...

    def _finish(self, keep_changes: bool) -> None:
        """作業ブランチをマージまたは破棄。stash を復元。"""
        ...

    @staticmethod
    def cleanup_orphan_branches(root: Path) -> list[str]:
        """起動時: harness/ プレフィックスの孤立ブランチを削除。"""
        ...
```

v1 からの変更:
- **EventBus subscriber** — Orchestrator からの直接呼び出しを廃止
- **Policy 連携** — `mode` に応じてスナップショット頻度を自動調整
- **Orchestrator は無関知** — checkpoint を付けるかどうかは設定次第
- **全エージェントイベントをフック** — 開始/完了/エラー/キャンセル全てに対応
- **v1 の git 操作はそのまま継承** — branch/stash/commit/merge のロジックは実績あり

---

## 実装順序

### Phase 1: Memory (最小構成)

**見積: ファイル 4 つ + テスト 1 つ + CLI 変更**

1. `memory/store.py` — SQLite バックエンド (sessions + facts テーブル)
2. `memory/session.py` — SessionMemory (save/load/clear)
3. `memory/project.py` — ProjectMemory (remember/forget/recall/list)
4. `memory/__init__.py`
5. `cli.py` 変更 — 起動時 load、終了時 save、REPL コマンド追加
6. `tests/v2/test_memory.py`

先にこれを実装する理由: 他の機能への依存なし、ユーザー体験に直結

### Phase 2: Task Queue

**見積: ファイル 4 つ + テスト 1 つ + CLI 変更 + types.py 変更**

1. `tasks/record.py` — TaskRecord, TaskStatus
2. `tasks/store.py` — TaskStore (SQLite)
3. `tasks/manager.py` — TaskManager (asyncio)
4. `tasks/__init__.py`
5. `types.py` — TASK_STARTED / TASK_COMPLETED 追加
6. `cli.py` 変更 — /submit, /tasks, /result, /cancel
7. `tests/v2/test_tasks.py`

Phase 1 の MemoryStore と DB ファイルを共有できる

### Phase 3: Checkpoint

**見積: ファイル 2 つ + テスト 1 つ + CLI 変更**

1. `checkpoint/engine.py` — CheckpointEngine
2. `checkpoint/__init__.py`
3. `cli.py` 変更 — `_build_components()` で attach
4. `tests/v2/test_checkpoint.py`

Phase 2 の EventType を利用。git のある環境でのみ有効化。

---

## 設計原則まとめ

| 原則 | 適用 |
|------|------|
| **EventBus 駆動** | Checkpoint は subscriber。Memory は AGENT_DONE で自動保存。Task は TASK_* イベントで通知 |
| **Orchestrator は無関知** | チェックポイントもメモリも Orchestrator のコードに現れない |
| **明示 > 暗黙** | 自動学習を廃止、ユーザーと LLM が明示的に `/remember` |
| **async 一貫** | 全 API が async。SQLite は executor 経由 |
| **Policy 連携** | Checkpoint 頻度は policy mode で自動調整 |
| **v1 の実績を継承** | git 操作、SQLite スキーマ、サニタイズは v1 ベース |

---

*Created: 2026-03-02*
*Based on analysis of v1 Memory (~600 LOC), Task Queue (~400 LOC), Checkpoint (~370 LOC)*
