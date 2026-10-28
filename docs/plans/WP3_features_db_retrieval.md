# WP3 — State features, SQLite seeding and hybrid retrieval (spec for a Sonnet implementer)

Author: Opus · 2026-10-28 · Implements D-007 (fast-loop retrieval), D-009 (embedded storage), D-013, D-015 and the retrieval rows of `docs/research/course_mapping.md` §1.
Depends on WP1 (package `clashbot`, `dataset/v1/states.jsonl`, `games.jsonl`, `provenance.jsonl`). Uses WP2 outputs (`dataset/v1/kg/*.jsonl`, `card_stats.jsonl`, `documents.jsonl`) **only if they exist**.

**You are implementing, not designing.** If anything is unclear or contradicts the data, stop and report the question.

## 0. Ground rules
- Python 3.11. Allowed third-party: `numpy` (required), `hnswlib` (optional, import inside the function that needs it). `pytest` for tests. Nothing else.
- Do not modify `knowledge_base/`, WP1/WP2 files, `tools/extract_states.py`, `cr_perception/`. Do not git commit/push.
- Scripts: `ROOT = Path(__file__).resolve().parents[2]`; `sys.path.insert(0, str(ROOT))`.

## 1. Files to create
```
clashbot/features.py
clashbot/resources/schema.sql
clashbot/db.py
clashbot/retrieval.py
tools/db/__init__.py          (empty)
tools/db/seed.py
tests/test_retrieval.py
```

## 2. `clashbot/features.py` — structured state vector (FEATURE_VERSION = 1)

```python
FEATURE_VERSION = 1
BLOCK_WEIGHTS = {"scalars": 1.0, "phase": 1.0, "hand": 1.0, "deck": 1.0, "opp_deck": 1.0,
                 "opp_recent": 1.0, "own_recent": 1.0, "towers": 1.0, "field": 1.0, "threat": 1.0}
RECENT_S = 5.0
TOWER_HP_SCALE = 5000.0
```
`vocab()` → sorted list of slugs `s` with `is_playable(s, load_cards())`; cached. `V = len(vocab())`.

Blocks, in this exact order, with dimensions:

| block | dim | values |
|---|---|---|
| scalars | 3 | `own.elixir/10` (None→0.0); `opponent.elixir_estimate/10` (None→0.5); `min(max(match_seconds/300, 0), 1.5)` (None→0.5) |
| phase | 5 | one-hot over `("single_elixir","double_elixir","triple_elixir","overtime","unknown")`; None or unknown value → "unknown" |
| hand | V | multi-hot of non-null `own.hand` entries that are in vocab |
| deck | V | multi-hot of `own.deck` |
| opp_deck | V | multi-hot of `opponent.deck_known` |
| opp_recent | V | multi-hot of `opponent.recent_plays` cards with `0 <= t - p.t <= RECENT_S` |
| own_recent | V | same for `own.recent_plays` |
| towers | 6 | own left, right, king, then enemy left, right, king: `None→1.0`, else `min(max(hp/TOWER_HP_SCALE,0),1)` |
| field | 24 | index = `side_i*12 + lane_i*4 + zone_i` with side order `("ally","enemy")`, lanes `LANES`, zones `ZONES`; count units whose `lane_of(tile)` and `zone_of(tile)` are not None; value `log1p(count)` |
| threat | 6 | for towers `("own_left","own_right","own_king")` the min `eta_tower.s` over **enemy** units whose `eta_tower.tower` equals it; then `("enemy_left","enemy_right","enemy_king")` over **ally** units; value `1/(1+s)`, 0.0 when none |

```python
def featurize(state: dict, t: float) -> np.ndarray   # float32, shape (dim(),)
```
Build each block as float64, L2-normalise it if its norm > 0, multiply by `BLOCK_WEIGHTS[name]`, concatenate, then L2-normalise the whole vector if its norm > 0, cast to float32.
Also: `dim() -> int`, `block_slices() -> dict[str, list[int]]` (`[start, stop]` per block), `feature_names() -> list[str]` (`"scalars:own_elixir"`, `"scalars:opp_elixir"`, `"scalars:match_progress"`, `"phase:<name>"`, `"hand:<slug>"`, ..., `"towers:own_left"` ..., `"field:<side>:<lane>:<zone>"`, `"threat:<tower>"`).

## 3. `clashbot/resources/schema.sql`
Exactly these statements (SQLite):
```sql
PRAGMA journal_mode = WAL;
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE games (game_id TEXT PRIMARY KEY, split TEXT, own_deck TEXT, deck_status TEXT, own_deck_key TEXT,
  own_archetype TEXT, opponent_deck TEXT, opponent_archetype_guess TEXT, result TEXT, flags TEXT);
CREATE TABLE provenance (game_id TEXT PRIMARY KEY, video_id TEXT, match_index TEXT, creator TEXT,
  video_title TEXT, video_url TEXT, video_time TEXT);
CREATE TABLE states (vec_row INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, game_id TEXT, split TEXT, kind TEXT,
  t REAL, phase TEXT, match_seconds REAL, own_elixir REAL, deck_status TEXT, own_deck_key TEXT,
  action_type TEXT, action_card TEXT, action_lane TEXT, action_zone TEXT, action_key TEXT,
  verdict TEXT, elixir_trade REAL, train_eligible INTEGER, flags TEXT, state_text TEXT, context_text TEXT, record TEXT);
CREATE INDEX states_game ON states(game_id);
CREATE INDEX states_split ON states(split);
CREATE INDEX states_deck ON states(own_deck_key);
CREATE INDEX states_phase ON states(phase);
CREATE INDEX states_action_card ON states(action_card);
CREATE VIRTUAL TABLE states_fts USING fts5(id UNINDEXED, context_text);
CREATE TABLE cards (slug TEXT PRIMARY KEY, name TEXT, card_type TEXT, rarity TEXT, elixir_cost REAL, targets TEXT,
  attributes TEXT, stats_by_level TEXT);
CREATE TABLE kg_nodes (id TEXT PRIMARY KEY, type TEXT, name TEXT, props TEXT);
CREATE TABLE kg_edges (edge_id INTEGER PRIMARY KEY, src TEXT, rel TEXT, dst TEXT, props TEXT);
CREATE INDEX kg_edges_src ON kg_edges(src, rel);
CREATE INDEX kg_edges_dst ON kg_edges(dst, rel);
CREATE TABLE documents (doc_id TEXT PRIMARY KEY, type TEXT, title TEXT, text TEXT, cards TEXT, deck_key TEXT,
  archetype TEXT, game_id TEXT, split TEXT);
CREATE VIRTUAL TABLE documents_fts USING fts5(doc_id UNINDEXED, title, text);
```
JSON-valued columns (`own_deck`, `opponent_deck`, `flags`, `record`, `attributes`, `stats_by_level`, `props`, `cards`) hold `json.dumps(...)` strings.

## 4. `clashbot/db.py`
```python
def connect(path: Path = DB_PATH, readonly: bool = False) -> sqlite3.Connection
```
`readonly=True` opens with URI `file:<path>?mode=ro`. Sets `row_factory = sqlite3.Row`.
```python
def create_schema(conn) -> None      # executescript(schema.sql)
def fts_query(text: str, max_terms: int = 64) -> str | None
```
`fts_query`: tokens = `re.findall(r"[a-z][a-z0-9]*", text.lower())`; drop tokens of length < 3 and the tokens `{"own","opp","elixir","hand","next","deck","field","threats","recent","towers","enemy","none","ago","eta","unknown"}`; keep first occurrences in order; take the first `max_terms`; return `" OR ".join(f'"{tok}"' for tok in tokens)`, or None if empty.

## 5. `tools/db/seed.py`
CLI: `python3 tools/db/seed.py [--dataset DIR] [--db PATH] [--hnsw]` (defaults from config).
Steps, in order:
1. Delete the db file (and `-wal`/`-shm` siblings) if present; `mkdir -p` its folder; `connect()`; `create_schema()`.
2. `games.jsonl` → `games` (list fields JSON-encoded). `provenance.jsonl` → `provenance`.
3. `states.jsonl` → `states`, in file order, `vec_row` = 0-based line index. Columns: `phase = state.phase`, `match_seconds = state.match_seconds`, `own_elixir = state.own.elixir`, `deck_status = state.own.deck_status`, `own_deck_key` from the game's row in games.jsonl, `action_*` from `action` (`action_key = action.key`), `verdict = outcome.verdict`, `elixir_trade = outcome.elixir_trade`, `train_eligible` as 0/1, `record = json.dumps(full record)`.
4. `states_fts`: `INSERT INTO states_fts(rowid, id, context_text) VALUES (vec_row + 1, id, context_text)`.
5. Features: `M = np.stack([featurize(r["state"], r["t"]) for r in records])` → save `db/state_features.npy` (path = db file's folder / `state_features.npy`) and `state_features.meta.json` = `{"feature_version", "dim", "n", "block_slices", "built_at"}`.
6. If `--hnsw`: `import hnswlib`; `index = hnswlib.Index(space="ip", dim=D)`; `init_index(max_elements=N, ef_construction=200, M=16)`; `add_items(M, np.arange(N))`; `save_index(<folder>/state_features.hnsw)`.
7. If `<dataset>/card_stats.jsonl` exists → `cards`. If `<dataset>/kg/nodes.jsonl` and `edges.jsonl` exist → `kg_nodes`, `kg_edges` (`edge_id` = 0-based line index). If `<dataset>/documents.jsonl` exists → `documents` + `documents_fts(rowid = line index + 1, doc_id, title, text)`. Each missing file → print `skip <name>: not built` and continue.
8. `meta` rows: `dataset_version`, `feature_version`, `built_at` (UTC ISO8601), `n_states`, `feature_dim`, `source_manifest_sha256` (sha256 of `<dataset>/MANIFEST.json` bytes, or "missing").
9. Commit, close, print table counts.

## 6. `clashbot/retrieval.py`

```python
@dataclass
class Hit:
    row: int
    id: str
    game_id: str
    kind: str
    action_key: str | None
    verdict: str | None
    elixir_trade: float | None
    score: float
    ranks: dict          # e.g. {"knn": 1, "bm25": 4}; a missing channel is absent
```

```python
class StateIndex:
    def __init__(self, db_path: Path = DB_PATH, use_hnsw: bool = False)
```
- Opens the db read-only. Loads into numpy arrays (index = vec_row) the columns `id, game_id, split, kind, own_deck_key, phase, action_key, verdict, elixir_trade, train_eligible`, ordered by vec_row.
- Loads `state_features.npy` from the db folder with `np.load(..., mmap_mode="r")`; checks `meta.json` `feature_version == FEATURE_VERSION` and dim matches, else raises `RuntimeError`.
- If `use_hnsw`: loads `state_features.hnsw` with hnswlib (`set_ef(64)`); if the file is missing raise `FileNotFoundError`.

```python
    def mask(self, *, splits=None, exclude_game=None, deck_key=None, phase=None, require_action=True) -> np.ndarray
```
Boolean array of length N, True = allowed: `split in splits` (if given), `game_id != exclude_game` (if given), `own_deck_key == deck_key` (if given), `phase == phase` (if given), `action_key is not None` (if require_action).

```python
    def knn(self, vec: np.ndarray, k: int = 20, **filters) -> list[tuple[int, float]]
```
Exact path (default): `scores = M @ vec`; set disallowed rows to `-inf`; take the top k by score descending (ties by row ascending) excluding `-inf`. HNSW path: query `k * 10` labels (capped at N), drop disallowed rows (post-filter), keep the first k; score = `1 - distance` (hnswlib "ip" returns `1 - ip`).

```python
    def bm25(self, text: str, k: int = 20, **filters) -> list[tuple[int, float]]
```
`q = fts_query(text)`; None → `[]`. SQL: `SELECT rowid, bm25(states_fts) FROM states_fts WHERE states_fts MATCH ? ORDER BY bm25(states_fts) LIMIT ?` with limit `k * 10`; row = rowid − 1; score = −bm25; drop disallowed rows; keep first k.

```python
    @staticmethod
    def rrf(lists: dict[str, list[tuple[int, float]]], k_rrf: int = 60) -> list[tuple[int, float, dict]]
```
For each channel name and its list, rank r = 1-based position; fused score `Σ 1/(k_rrf + r)`; returns `(row, score, {channel: rank})` sorted by score desc, ties by row asc.

```python
    def search(self, state: dict, t: float, context_text: str, k: int = 10, depth: int = 50, **filters) -> list[Hit]
```
`knn(featurize(state, t), depth, **filters)` and `bm25(context_text, depth, **filters)` → `rrf({"knn": ..., "bm25": ...})` → first k → `Hit` objects.

## 7. Tests — `tests/test_retrieval.py`
1. `featurize` on a minimal synthetic state: shape == `(dim(),)`, dtype float32, unit norm (±1e-5); `block_slices()` cover `[0, dim())` contiguously; a state with all-None fields still works.
2. `field` block: one ally unit at tile [3, 20] sets the `field:ally:left:enemy_half` coordinate (> 0 before global normalisation, test via the block slice being non-zero at that index).
3. `fts_query("hand: skeletons, the-log (next ?)")` → contains `"skeletons"` and `"log"` and not `"hand"` or `"next"`.
4. `rrf` on two hand-made lists gives the expected order and rank dicts.
5. Integration (skip if `dataset/v1/states.jsonl` is missing): run `tools/db/seed.py --db <tmp>/t.sqlite --hnsw` via subprocess; open `StateIndex(<tmp>/t.sqlite)`; take the first train-split state record from `states.jsonl`; `search(..., exclude_game=<its game>)` returns k hits, none from that game, all with non-null `action_key`; `knn` with `use_hnsw=True` returns k rows; `SELECT count(*) FROM states` equals the line count of states.jsonl.

## 8. Acceptance (run, paste outputs)
```
python3 -m pytest -q tests/test_retrieval.py tests/test_clean_dataset.py
python3 tools/db/seed.py --hnsw
python3 - <<'PY'
import json, time, sys
sys.path.insert(0, ".")
from clashbot.retrieval import StateIndex
idx = StateIndex()
rec = next(json.loads(l) for l in open("dataset/v1/states.jsonl") if '"split": "val"' in l and '"train_eligible": true' in l)
t0 = time.perf_counter(); hits = idx.search(rec["state"], rec["t"], rec["context_text"], k=5, splits=["train"]); dt = time.perf_counter() - t0
print(rec["id"], rec["action"]["key"]); [print(h) for h in hits]; print(f"search {dt*1000:.1f} ms")
PY
ls -la db/
git status --short
```

## 9. Report back (≤ 40 lines)
Files created; test results; seed table counts; the acceptance search output incl. latency; any deviation and why.
