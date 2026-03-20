# Open Harness v2 パフォーマンス最適化計画

## 現状分析

### アーキテクチャ特性

Open Harness v2 は **I/Oバウンド** なアプリケーション。ボトルネックの大部分はLLM APIの応答待ち（数秒〜数十秒）であり、Python側のCPU処理時間（ミリ秒未満）は相対的に無視できるレベル。

```
典型的な1ステップの時間配分:
  LLM API応答待ち:  2,000〜30,000ms  (95%以上)
  レスポンスパース:       0.1〜5ms
  コンテキスト構築:       0.5〜10ms
  ツール実行:          10〜5,000ms  (shell等は外部プロセス依存)
  スタック検出/反省:      0.01〜1ms
```

### ボトルネック箇所の評価

| モジュール | 処理内容 | 現在の所要時間 | Rustで改善幅 | 優先度 |
|-----------|---------|--------------|-------------|-------|
| `llm/response_parser.py` | JSON抽出、ストリーム処理 | ~1ms | 10-50x → ~0.02ms | **高** |
| `core/context.py` | コンテキスト圧縮、トークン推定 | ~5ms | 5-20x → ~0.5ms | **高** |
| `llm/client.py` | SSEストリームパース | ~0.5ms/chunk | 5-10x | 中 |
| `core/stuck_detector.py` | MD5ハッシュ、パターン検出 | ~0.1ms | 10x → ~0.01ms | 低 |
| `memory/store.py` | SQLite I/O | ~1-10ms | 変わらない | 低 |

---

## Phase 1: Python レベル最適化（即効性・低リスク）

Rustに手を出す前に、Pythonのまま改善できる部分を先に対処する。
効果が薄ければ Phase 2 へ進む判断材料にもなる。

### 1.1 orjson 導入（JSON パース高速化）

**対象**: `response_parser.py`, `client.py`, `memory/store.py`

```python
# Before
import json
data = json.loads(raw)

# After
import orjson
data = orjson.loads(raw)  # 2-10x faster
```

- `orjson` はRust実装のJSON パーサー（C拡張経由）
- `json.loads` / `json.dumps` の全箇所を置換
- **効果**: JSON処理が2-10倍高速化。特にストリーミング中の大量の小さなJSONパースで効く
- **リスク**: 低（drop-in replacement）

### 1.2 トークン推定の改善

**対象**: `core/context.py` の `_estimate_tokens()`

```python
# 現状: 非常にラフな推定
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

# 改善案A: tiktoken（OpenAI互換トークナイザ、Rust実装）
import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")
def _estimate_tokens(text: str) -> int:
    return len(_enc.encode(text))

# 改善案B: 日本語対応の軽量推定（tiktoken不要）
def _estimate_tokens(text: str) -> int:
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + non_ascii)  # 非ASCII文字は1文字≒1トークン
```

- 案Aは正確だが遅い（1回 ~0.5ms）。コンテキスト構築で何度も呼ばれるためキャッシュ必要
- 案Bは十分な精度で高速。ローカルLLM用途なので正確なトークン数は不要
- **推奨**: 案Bをデフォルト、`tiktoken` を optional dependency として案Aも選べるように

### 1.3 正規表現のプリコンパイル

**対象**: `context.py:_extract_tool_name()`, `response_parser.py`

```python
# 現状: 関数呼び出しごとに re.search()
def _extract_tool_name(text: str) -> str:
    import re  # 関数内import
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', text_str)

# 改善: モジュールレベルでコンパイル
_RE_TOOL_NAME = re.compile(r'"tool"\s*:\s*"([^"]+)"')
_RE_TOOL_RESULT = re.compile(r'\[Tool Result for (\w+)\]')
_RE_FUNC_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')

def _extract_tool_name(text: str) -> str:
    m = _RE_TOOL_NAME.search(str(text))
    ...
```

- **効果**: 正規表現コンパイルのオーバーヘッド除去。呼び出し頻度が高い箇所で有効
- **リスク**: なし

### 1.4 aiosqlite 移行

**対象**: `memory/store.py`

```python
# 現状: asyncio.to_thread で同期sqlite3をラップ
await asyncio.to_thread(self._save_messages_sync, ...)

# 改善: aiosqlite で真のasync化
import aiosqlite
async with aiosqlite.connect(self._db_path) as db:
    await db.execute(...)
    await db.commit()
```

- `asyncio.to_thread` はスレッドプールを消費する
- `aiosqlite` は内部的にスレッドを使うが、接続管理がより効率的
- **効果**: 小。現在の使用頻度では差は微小
- **リスク**: 低

### 1.5 httpx 接続プール最適化

**対象**: `llm/client.py`

```python
# 現状: デフォルト設定
self._client = httpx.AsyncClient(...)

# 改善: ローカルLLM向けに最適化
self._client = httpx.AsyncClient(
    base_url=base_url,
    headers=headers,
    timeout=httpx.Timeout(timeout, connect=30, read=300),
    limits=httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=30,
    ),
    http2=True,  # HTTP/2 対応サーバーなら有効
)
```

- ローカルサーバーへの接続はkeep-alive再利用が効果的
- HTTP/2 はOllama/LM Studio が対応していれば有効
- **効果**: 接続確立オーバーヘッドの削減（数十ms/リクエスト）

---

## Phase 2: Rust ネイティブ拡張（PyO3 / maturin）

Phase 1 で対応しきれない、または将来的にデータ量が増えた際に効くCPUバウンド処理をRustで書き直す。

### 2.1 ビルド基盤の構築

```
open_harness/
├── rust/                          # NEW: Rust拡張
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                 # PyO3 モジュールエントリ
│       ├── json_extract.rs        # バランスJSON抽出
│       ├── stream_processor.rs    # ストリーム処理ステートマシン
│       ├── context_compress.rs    # コンテキスト圧縮
│       └── token_estimate.rs      # トークン推定
├── pyproject.toml                 # maturin ビルド追加
└── src/open_harness_v2/
    └── _native.py                 # Rustバインディング fallback ラッパー
```

```toml
# rust/Cargo.toml
[package]
name = "open_harness_native"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
serde_json = "1"
memchr = "2"           # 高速文字列検索
```

```toml
# pyproject.toml に追記
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
manifest-path = "rust/Cargo.toml"
python-source = "src"
features = ["pyo3/extension-module"]
```

**フォールバック戦略**: Rust拡張がビルドできない環境でもPure Python版で動くように。

```python
# src/open_harness_v2/_native.py
try:
    from open_harness_native import (
        extract_balanced_json,
        parse_tool_calls,
        compress_history,
        estimate_tokens,
    )
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    # Python fallback implementations
    from open_harness_v2.llm.response_parser import _extract_balanced_json as extract_balanced_json
    ...
```

### 2.2 JSON バランス抽出（最優先）

**対象**: `response_parser.py:_extract_balanced_json()`

現状は1文字ずつPythonでループ。LLMの出力が大きい場合に最もCPU時間を使う箇所。

```rust
// rust/src/json_extract.rs
use pyo3::prelude::*;

/// Extract a balanced JSON object starting at `start`.
#[pyfunction]
fn extract_balanced_json(text: &str, start: usize) -> Option<String> {
    let bytes = text.as_bytes();
    if start >= bytes.len() || bytes[start] != b'{' {
        return None;
    }

    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape = false;

    for i in start..bytes.len() {
        let ch = bytes[i];
        if escape {
            escape = false;
            continue;
        }
        if ch == b'\\' {
            escape = true;
            continue;
        }
        if ch == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}
```

- **効果**: 10-50倍高速化。大きなJSONレスポンス（10KB+）で顕著
- **リスク**: 低（ロジックは完全同一、テスト容易）

### 2.3 ストリームプロセッサ

**対象**: `response_parser.py:StreamProcessor`

ステートマシンによるストリーム処理。チャンクが小さく呼び出し頻度が高い。

```rust
// rust/src/stream_processor.rs
use pyo3::prelude::*;

#[pyclass]
struct StreamProcessor {
    buffer: String,
    state: State,
    thinking: String,
    content_start: usize,
    displayed_up_to: usize,
}

#[pymethods]
impl StreamProcessor {
    #[new]
    fn new() -> Self { ... }

    fn feed(&mut self, chunk: &str) -> Vec<(String, String)> { ... }

    fn finish(&mut self) -> (String, String, Vec<PyObject>) { ... }
}
```

- **効果**: ストリーミング処理のオーバーヘッドを最小化。Pythonジェネレータ→Rust Vecバッチで呼び出しオーバーヘッドも削減
- **リスク**: 中（Pythonインターフェースの設計が必要）

### 2.4 コンテキスト圧縮

**対象**: `context.py:HistoryLayer._compress()`

メッセージ履歴の L1/L2 圧縮。メッセージ数が多い場合にイテレーション回数が増える。

```rust
// rust/src/context_compress.rs
#[pyfunction]
fn compress_history(
    messages: Vec<HashMap<String, String>>,
    budget: usize,
    protected_tail: usize,
) -> Vec<HashMap<String, String>> { ... }
```

- **効果**: 長時間セッションでメッセージが数百に達した場合に有効
- **リスク**: 中（メッセージ構造のシリアライゼーション）

---

## Phase 3: アーキテクチャレベル最適化

### 3.1 ストリーミングパイプラインの非同期化強化

現状の `chat_stream()` は SSE チャンクごとに `json.loads()` を呼ぶ。

```python
# 改善: バッファリングによるバッチ処理
async for raw_line in resp.aiter_lines():
    # 現状: 1行ずつパース
    data = json.loads(data_str)

# 改善案: チャンクをバッファして一括パース
buffer = []
async for raw_line in resp.aiter_lines():
    buffer.append(raw_line)
    if len(buffer) >= 10 or ...:  # バッチサイズ or タイムアウト
        for line in buffer:
            ...
        buffer.clear()
```

### 3.2 コンテキストのインクリメンタル構築

`to_messages()` が毎回全レイヤーを再構築している。

```python
# 改善: 変更フラグによるキャッシュ
class AgentContext:
    def __init__(self):
        ...
        self._cached_messages: list | None = None
        self._dirty = True

    def to_messages(self, budget=0):
        if not self._dirty and self._cached_messages is not None:
            return self._cached_messages
        result = self._build_messages(budget)
        self._cached_messages = result
        self._dirty = False
        return result
```

### 3.3 プロファイリング基盤

最適化の効果を測定するためのベンチマーク基盤を整備。

```python
# benchmarks/bench_parser.py
import time
from open_harness_v2.llm.response_parser import _extract_balanced_json

SAMPLE = '{"tool": "read_file", "args": {"path": "/foo/bar.py"}}' * 100

def bench():
    start = time.perf_counter_ns()
    for _ in range(10000):
        _extract_balanced_json(SAMPLE, 0)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    print(f"extract_balanced_json: {elapsed:.1f}ms for 10k iterations")
```

---

## 実施ロードマップ

```
Week 1-2: Phase 1 (Python最適化)
  ├── orjson 導入
  ├── 正規表現プリコンパイル
  ├── トークン推定改善
  ├── httpx 接続プール最適化
  └── ベンチマーク基盤構築

Week 3-4: Phase 2 準備
  ├── maturin ビルド基盤セットアップ
  ├── PyO3 プロジェクト構造
  └── CI/CD でのRustビルド対応

Week 5-6: Phase 2 実装
  ├── extract_balanced_json Rust実装
  ├── StreamProcessor Rust実装
  ├── フォールバックラッパー
  └── ベンチマーク比較

Week 7-8: Phase 3 + 統合テスト
  ├── コンテキストキャッシュ
  ├── ストリーミング最適化
  ├── 全体パフォーマンステスト
  └── リリース準備
```

---

## 判断基準：Rustに書き直すべきか

以下の条件を **すべて** 満たす場合にRust化を検討:

1. **CPU バウンド** である（I/O待ちではない）
2. **呼び出し頻度が高い**（1セッションで数百〜数千回）
3. **Python実装で 1ms 以上** かかる
4. **ロジックが安定** している（頻繁に変更しない部分）
5. **テストが容易** である（入出力が明確）

現時点での正直な評価:
- **Phase 1 だけで実用上十分な可能性が高い**
- LLM API応答が律速なので、パース処理を100倍速くしても体感差はほぼない
- Rust化の真の価値は、将来的に **バッチ処理** や **並列エージェント実行** を行う場合に発揮される
- まずは Phase 1 + ベンチマークで定量的に判断し、Phase 2 の必要性を見極める

---

## 補足: Phase 1 で特に効果が大きい箇所

1. **`orjson`**: ストリーミング中に毎チャンク `json.loads()` → 即効性あり
2. **トークン推定改善**: コンテキスト予算計算の精度向上 → トークン無駄遣い防止
3. **接続プール**: ローカルLLMへの接続再利用 → レイテンシ削減
