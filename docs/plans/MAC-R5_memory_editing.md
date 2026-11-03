# MAC-R5 — memory stores + patch-versioned knowledge editing and its evaluation (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Course: AGT memory types §4.1, Mem0 [25], Reflexion [5], ACE [22]; editing criteria [46]; SERAC/GRACE vs ROME [53, 54, 49]; ClashEval [67]. Implements D-034. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
clashbot/memory.py          (new)
tools/eval/edit_eval.py     (new)
tests/test_memory.py        (new)
```
Stdlib `sqlite3` + numpy only. The edit route here is **non-parametric** (an external, versioned fact store consulted at run time — the SERAC/GRACE family), compared against "no edit".

## 1. `clashbot/memory.py`
```python
MEMORY_DB_PATH = DB_DIR / "memory.sqlite"
BASE_PATCH = "base"
EDITABLE_ATTRIBUTES = ("elixir_cost", "card_type", "rarity", "targets")   # keys of load_cards() entries
LESSON_OPS = ("ADD", "UPDATE", "DELETE", "NOOP")                         # Mem0-style write operations

class MemoryStore:
    def __init__(self, path: Path | str = MEMORY_DB_PATH, now=None): ...
    def close(self) -> None
    # patches
    def add_patch(self, patch: str, note: str = "") -> int
    def patch_seq(self, patch: str) -> int
    def patches(self) -> list[str]
    # semantic memory: patch-versioned card facts with provenance
    def apply_edit(self, card: str, attribute: str, value, patch: str, source: str, provenance: str) -> int
    def retract(self, fact_id: int) -> None
    def card_fact(self, card: str, attribute: str, patch: str | None = None, base_cards: dict | None = None)
    def edited_cards(self, base_cards: dict, patch: str | None = None) -> dict
    def edits_since(self, patch: str) -> list[dict]
    # episodic memory
    def add_episode(self, game_id: str, t: float, state: dict, action_key: str | None, outcome: dict | None,
                    source: str = "own", patch: str | None = None) -> int
    def episodes(self, game_id: str | None = None, source: str | None = None) -> list[dict]
    # procedural memory (playbook lessons)
    def write_lesson(self, op: str, scope: str, text: str = "", lesson_id: int | None = None,
                     source: str = "reflection") -> int | None
    def lessons(self, scope: str | None = None) -> list[dict]
    def log(self) -> list[dict]

def stale_cards(store: MemoryStore, sample_patch: str = BASE_PATCH, current_patch: str | None = None) -> set[str]
def conflict_weight(action_key: str | None, stale: set[str], weight: float = 0.5) -> float
def reweight_hits(hits: list, stale: set[str], weight: float = 0.5) -> list
```
Schema (created in `__init__` with `CREATE TABLE IF NOT EXISTS`; `path` may be `":memory:"`; `mkdir` the parent for file paths):
```sql
patches(seq INTEGER PRIMARY KEY, patch TEXT UNIQUE NOT NULL, note TEXT, created_at TEXT)
card_facts(fact_id INTEGER PRIMARY KEY, card TEXT NOT NULL, attribute TEXT NOT NULL, value TEXT NOT NULL,   -- value = json
           patch TEXT NOT NULL, valid_from INTEGER NOT NULL, valid_to INTEGER, source TEXT, provenance TEXT, created_at TEXT)
episodes(episode_id INTEGER PRIMARY KEY, game_id TEXT, t REAL, source TEXT, patch TEXT, state TEXT, action_key TEXT,
         outcome TEXT, created_at TEXT)
lessons(lesson_id INTEGER PRIMARY KEY, scope TEXT NOT NULL, text TEXT NOT NULL, source TEXT, support INTEGER DEFAULT 1,
        created_at TEXT, updated_at TEXT, deleted INTEGER DEFAULT 0)
memory_log(log_id INTEGER PRIMARY KEY, op TEXT, store TEXT, ref INTEGER, detail TEXT, created_at TEXT)
```
Semantics:
- `now` is a callable returning an ISO timestamp (default `datetime.now(timezone.utc).isoformat(timespec="seconds")`); tests inject a fixed one.
- Patches: `BASE_PATCH` is inserted with `seq 0` on creation. `add_patch` is idempotent (existing name → its seq); new patches get `max(seq) + 1`. `patches()` in seq order. `patch_seq` unknown → `KeyError`.
- `apply_edit`: `attribute` not in `EDITABLE_ATTRIBUTES` → `ValueError`; `s = add_patch(patch)`. The open fact for `(card, attribute)` (`valid_to IS NULL`): if its `valid_from < s` → set `valid_to = s`; if `valid_from == s` (re-edit in the same patch) → set `valid_to = valid_from` (never valid). Insert the new fact (`valid_from = s`, `valid_to NULL`, `value = json.dumps(value)`). Log `("ADD" if no previous fact else "UPDATE", "semantic", fact_id, json detail)`. Return `fact_id`.
- `retract(fact_id)`: set its `valid_to = valid_from`; the most recent earlier fact of the same `(card, attribute)` whose `valid_to` equals this fact's `valid_from` gets `valid_to = NULL` again. Log `("DELETE", "semantic", fact_id, ...)`.
- `card_fact(card, attribute, patch=None, base_cards=None)`: `s` = seq of `patch` (or the latest seq); the fact with `valid_from <= s < coalesce(valid_to, ∞)` and `valid_to != valid_from` → `json.loads(value)`; none → `(base_cards or load_cards())[card][attribute]`.
- `edited_cards(base_cards, patch=None)`: `copy.deepcopy(base_cards)` with every applicable fact (as of `patch`) applied to cards present in `base_cards`; the input is never mutated.
- `edits_since(patch)`: facts with `valid_from > patch_seq(patch)` and `valid_to IS NULL or valid_to > valid_from`, as dicts.
- Episodes: `state`/`outcome` stored as JSON; `patch` default = latest patch; `episodes()` returns dicts with decoded JSON, ordered by `episode_id`. Log `("ADD", "episodic", ...)`.
- `write_lesson(op, scope, text, lesson_id, source)`: `op` not in `LESSON_OPS` → `ValueError`.
  - `ADD`: if a non-deleted lesson with the same `scope` and the same `text.strip()` exists → bump its `support`, `updated_at`, log `NOOP`, return its id; else insert, log `ADD`, return the new id.
  - `UPDATE`: needs an existing non-deleted `lesson_id` (else `KeyError`); set `text`, `updated_at`, `support + 1`; log; return id.
  - `DELETE`: needs `lesson_id`; set `deleted = 1`; log; return id.
  - `NOOP`: log; return `lesson_id`.
- `lessons(scope=None)`: non-deleted, `ORDER BY support DESC, lesson_id`.
- `stale_cards`: cards with any fact whose `valid_from` is in `(patch_seq(sample_patch), patch_seq(current_patch or latest)]` and not retracted.
- `conflict_weight(action_key, stale, weight)`: `weight` if `action_key` is a play key whose card is in `stale`, else `1.0` (ClashEval-style down-weighting of pre-patch evidence about edited cards).
- `reweight_hits(hits, stale, weight)`: for `clashbot.retrieval.Hit` objects return new `Hit`s (`dataclasses.replace`) with `score = score * conflict_weight(hit.action_key, stale, weight)`, sorted by `(-score, row)`.

## 2. `tools/eval/edit_eval.py` (knowledge-editing criteria [46])
```
$PY tools/eval/edit_eval.py [--dataset DIR] [--db PATH] [--card hog-rider] [--attribute elixir_cost] [--value 5]
        [--patch P1] [--n-probe 300] [--out PATH]
```
Edit (D-034, **synthetic** balance change, say so in the output): `store = MemoryStore(":memory:")`; `base = load_cards()`; `old = base[card][attribute]`; `fid = store.apply_edit(card, attribute, value, patch, "synthetic_balance_change", "edit_eval / D-034")`; `edited = store.edited_cards(base)`. Probe records: val+test records of `states.jsonl` (first `--n-probe` per probe set). Compute every probe twice — with `edited` (the non-parametric route) and with `base` ("no edit") — except where noted.

| criterion | probe | pass rule |
|---|---|---|
| reliability | direct: `store.card_fact(card, attribute)`, `edited[card][attribute]`, `d1_format.card_description(card, edited)` contains `f"{value} elixir"` (for `elixir_cost`) | equals the new value |
| generalization | records whose `own.hand` contains `card` and whose elixir `e` makes legality flip (`old - 0.5 <= e < value - 0.5` for `elixir_cost`) | `card not in legal_cards(state, cards)` |
| locality (other cards) | every other card × every editable attribute | `edited[c][a] == base[c][a]` |
| locality (other states) | records whose hand and deck do not contain `card` | `legal_action_keys(state, edited) == legal_action_keys(state, base)` |
| portability | the generalization records: `CandidateGenerator(StateIndex(db), KnowledgeGraph(db), cards).generate(state, t, context_text, n=4)` (skip with `"skipped": "no db"` if the DB is missing) | no candidate key uses `card` |
| conflict handling | records with `card` in hand: `StateIndex(db).search(state, t, context_text, k=10, splits=["train"], exclude_game=g)`; `stale = stale_cards(store)` | report `share_of_hits_with_card` before vs after `reweight_hits` (top-10) |
| versioning | `store.card_fact(card, attribute, patch=BASE_PATCH)` | equals `old` |
| rollback | `store.retract(fid)`; `store.edited_cards(base) == base` and `card_fact(...) == old` (run last) | true |

For `attribute != "elixir_cost"`, generalization/portability are reported as `"n/a for this attribute"`.
Output JSON (`--out`, default `<dataset>/eval/edit_eval.json`): `{"edit": {card, attribute, old, new, patch, synthetic: true}, "criteria": {name: {"n", "edited", "no_edit"}}, "conflict": {"n", "before", "after"}, "versioning": bool, "rollback": bool}`; `<out>.md` with `| criterion | n | edited | no edit |`; print it.

## 3. Tests — `tests/test_memory.py` (fixed `now`, `":memory:"` or `tmp_path` files)
1. Patches: `base` is seq 0; `add_patch` idempotent; `patches()` order; unknown `patch_seq` → `KeyError`.
2. `apply_edit` + `card_fact`: latest value; as of `base` → base value (pass a small `base_cards` dict); a second edit in a later patch closes the first; re-edit in the same patch replaces (the first never valid); non-editable attribute → `ValueError`.
3. `retract` reopens the previous fact; `edits_since("base")` excludes retracted facts.
4. `edited_cards` deep-copies; base dict untouched; only the edited card differs.
5. Lessons: ADD; duplicate ADD → same id, `support == 2`, log shows `NOOP`; UPDATE; DELETE hides it from `lessons()`; bad op → `ValueError`; UPDATE of a deleted id → `KeyError`; `log()` ops in order.
6. Episodes: JSON round trip; filters by `game_id` and `source`.
7. `stale_cards` / `conflict_weight` / `reweight_hits` with `clashbot.retrieval.Hit` objects: a hit whose action uses the stale card drops below an equal-score hit that does not.
8. File persistence: write, `close()`, reopen the same path → facts, lessons and episodes still there.
9. `edit_eval.main` on a synthetic dataset dir (6 hand-made records, `--db` pointing to a missing file): JSON written; `reliability.edited == 1.0`, `reliability.no_edit == 0.0`, locality 1.0, portability and conflict `skipped`.
10. Real data (skip if `db/clashbot.sqlite` missing): `main(["--n-probe", "50", "--out", str(tmp_path / "e.json")])` → reliability 1.0, both locality criteria 1.0, `versioning` and `rollback` true.

## 4. Smoke (allowed once, after the symlinks)
`$PY tools/eval/edit_eval.py --n-probe 100 --out runs/smoke/R5/e.json` — paste the table.

## 5. Heavy run (Opus, at integration)
Full run (`--n-probe 300`) → `docs/data/edit_eval_v1.{json,md}`.

## 6. Report
Per MAC-00 §5.
