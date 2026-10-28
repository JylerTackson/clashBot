# WP5 — Runtime skeleton for the two-speed loop (spec for a Sonnet implementer)

Author: Opus · 2026-10-28 · Implements D-003 (read-only agent, code executes), D-004, D-007 (two-speed retrieve-and-rank), D-010, D-015; agent-tool rows of `docs/research/course_mapping.md` §4.
Depends on: WP1 (`clashbot.config`, `game_rules`, `state_text`), WP2 (KG rows in the DB), WP3 (`clashbot.db`, `clashbot.retrieval.StateIndex`, `clashbot.features`), WP4 (`clashbot.actions`, `dataset/v1/placement_prior.json`). The seeded DB `db/clashbot.sqlite` must exist for integration tests (skip them otherwise).

No model weights are available in this environment. Model adapters are stubs with a fixed interface; everything else must run and be tested here.

**You are implementing, not designing.** If anything is unclear, stop and report.

## 0. Ground rules
- Python 3.11, stdlib + numpy. No new dependencies. No commits. Do not modify WP1–WP4 files, `knowledge_base/`, `cr_perception/`.

## 1. Files to create
```
clashbot/kg.py
clashbot/candidates.py
clashbot/ranker.py
clashbot/tools.py
clashbot/planner.py
clashbot/executor.py
clashbot/loop.py
tests/test_runtime.py
```

## 2. `clashbot/kg.py`
```python
class KnowledgeGraph:
    def __init__(self, db_path: Path = DB_PATH)
```
Opens the DB read-only (`clashbot.db.connect(readonly=True)`), loads all `kg_edges` rows into memory: `self.out[(src, rel)] -> list[(dst, props_dict)]`, `self.inc[(dst, rel)] -> list[(src, props_dict)]`.

```python
    def counter_scores(self, threat_card: str) -> dict[str, float]
```
Returns `{own_card_slug: score}` for cards that answer `threat_card`:
- for each edge `card:<X> -COUNTERS-> card:<threat_card>` (i.e. `inc[("card:"+threat_card, "COUNTERS")]`): `score[X] += 1.0`;
- for each edge `card:<threat_card> -ANSWERED_BY-> card:<Y>`: `score[Y] += props["confidence"] * min(1.0, props["support"] / 10)`.
Slugs are returned without the `card:` prefix.

```python
    def neighbors(self, node_id: str, rel: str | None = None, direction: str = "out", limit: int = 50) -> list[dict]
```
Returns `[{"node": id, "rel": rel, "props": props}]` from `out` (direction "out") or `inc` ("in"); `rel=None` means all relations (iterate keys in sorted order); first `limit` items.

## 3. `clashbot/candidates.py`
```python
@dataclass
class Candidate:
    key: str
    score: float
    sources: dict          # e.g. {"knn": 0.031, "kg": 0.5}

class CandidateGenerator:
    def __init__(self, index: StateIndex, kg: KnowledgeGraph, cards: dict | None = None,
                 kg_weight: float = 0.5, hold_prior: float = 0.01)
    def generate(self, state: dict, t: float, context_text: str, n: int = 4, **filters) -> list[Candidate]
```
Algorithm (exact):
1. `legal = legal_action_keys(state, cards)`; `legal_set = set(legal)`.
2. **kNN channel**: `hits = index.search(state, t, context_text, k=20, depth=50, **filters)`; `knn[key] += hit.score` for each hit whose `action_key` is in `legal_set`.
3. **KG channel**: threats = (a) cards of `state.opponent.recent_plays` with a non-null card and `0 <= t - p.t <= 5.0`, each with its `p.tile` (may be None); (b) units with `side == "enemy"`, non-null `card`, and `eta_tower` a dict whose `tower` starts with `"own"` and `s <= 8.0`, each with the unit's `tile`. For each threat `(X, tile)`, for each `(C, s)` in `kg.counter_scores(X).items()` with `C in legal_cards(state, cards)`:
   - if `cards[C]["card_type"] == "Spell"`: requires `tile` not None; key = `action_key(C, lane_of(tile), zone_of(tile))`;
   - else: key = `action_key(C, lane_of(tile) or "middle", "mid")`;
   - if key in `legal_set`: `kg[key] += s`.
4. Normalise each channel by its max value (skip if empty). `score[key] = knn_norm.get(key, 0) + kg_weight * kg_norm.get(key, 0)` for every key in either channel. `score["hold"] = knn_norm.get("hold", 0) + hold_prior`.
5. Ranking: keys sorted by `(-score, key)`; take the first `n - 1` play keys (non-"hold"), then append `"hold"`. If there are fewer than `n - 1` play keys, fill from `legal` (in order, excluding hold and already chosen) with score 0.0 and sources `{"fallback": 1}`. Each Candidate's `sources` = `{"knn": knn_norm.get(key, 0.0), "kg": kg_norm.get(key, 0.0)}` (rounded to 4) unless fallback.
6. Return the list (length ≤ n).

## 4. `clashbot/ranker.py`
```python
@dataclass
class RankResult:
    index: int
    probs: list[float]
    model: str

def format_multiple_choice(context: str, options: list[str], plan: str | None = None) -> str
```
Exact format (this string is the training/serving prompt for Laya/d1):
```
<context>
[plan] <plan>            ← only this line when plan is a non-empty string
Options:
A) <options[0]>
B) <options[1]>
...
Answer:
```
Letters `A`… for up to 8 options; raise ValueError for more than 8 or zero.

```python
class Ranker(Protocol):
    name: str
    def choose(self, context: str, options: list[str], plan: str | None = None) -> RankResult: ...

class ScoreRanker:          # baseline: trusts the candidate generator order
    name = "score"
    def choose(...) -> RankResult      # index 0, probs = [1.0] + [0.0] * (len(options) - 1)

class _StubDecisionModel:
    name = "stub"
    model_id = ""
    def __init__(self, model_path: str | None = None): self.model_path = model_path
    def choose(...):
        raise NotImplementedError(f"{self.name}: load {self.model_id} on the Mac (see docs/plans/WP5_runtime_skeleton.md §4); "
                                  "input is format_multiple_choice(...), output is a probability per option letter")

class LayaRanker(_StubDecisionModel):  name = "laya";  model_id = "convaiinnovations/laya (≈421M, Apache-2.0)"
class D1Ranker(_StubDecisionModel):    name = "d1";    model_id = "LiquidAI/d1-omni-600M (LFM Open License)"
```
Note for the docstring of `_StubDecisionModel`: "Both models return probabilities over a fixed answer set; the answer set is the option letters A–H. Fine-tuning data: dataset/v1/ranker/*.jsonl rendered with format_multiple_choice."

## 5. `clashbot/tools.py` — read-only tools for the slow-loop LLM
```python
@dataclass
class ToolContext:
    state: dict | None = None
    t: float | None = None
    context_text: str | None = None
    plan_text: str | None = None
    index: StateIndex | None = None
    kg: KnowledgeGraph | None = None
    db_path: Path = DB_PATH
```
`TOOLS: list[dict]` — each `{"name", "description", "parameters"}` where `parameters` is a JSON Schema object. Exactly these six:

| name | parameters (JSON Schema properties; `required`) | returns |
|---|---|---|
| `get_game_state` | `{}` | `{"t", "context_text", "state"}` from ctx |
| `search_similar_states` | `k: integer 1..20` (default 5) | list of `{"id", "action_key", "verdict", "score"}` from `ctx.index.search(ctx.state, ctx.t, ctx.context_text, k=k)` |
| `kg_query` | `node: string` (required), `relation: string` (enum `COUNTERS, ANSWERED_BY, SYNERGY, SUBSTITUTE, IN_DECK, SPAWNED_BY, EVOLUTION_OF, HERO_OF`), `direction: string` enum `out,in` (default `out`) | `ctx.kg.neighbors(node, relation, direction, limit=25)` |
| `card_info` | `slug: string` (required) | the `cards` table row as a dict (JSON columns decoded) or `{"error": "unknown card"}` |
| `search_documents` | `query: string` (required), `k: integer 1..10` (default 3), `type: string` (optional) | `documents_fts` BM25 top-k (use `fts_query`) → `[{"doc_id","type","title","text"[:800]}]`, filtered by `type` when given |
| `get_current_plan` | `{}` | `{"plan": ctx.plan_text}` |

```python
def call_tool(name: str, args: dict, ctx: ToolContext) -> Any
```
Unknown name → `ValueError`. Validate: required keys present; integers within min/max; enums respected; apply defaults; reject unknown argument names with `ValueError`. Dispatch to the implementation. All tools are read-only: never write to the DB (open it read-only).

## 6. `clashbot/planner.py`
```python
@dataclass
class Plan:
    text: str
    source: str           # "rules" | "llm"

class RuleBasedPlanner:
    def __init__(self, db_path: Path = DB_PATH)
    def pregame(self, own_deck: list[str], deck_key: str | None) -> Plan
```
`pregame`: if `deck_key` and a `documents` row `deck_why:<deck_key>` exists → its text; else the `archetype_guide:<archetype>` text where archetype = `kg_nodes` props `archetype_primary` of `deck:<deck_key>` (if any); else `"Deck: " + ", ".join(own_deck)`. Cut to 600 characters. `source="rules"`.
```python
    def update(self, plan: Plan, state: dict, t: float) -> Plan     # returns plan unchanged (placeholder for the LLM planner)

def build_pregame_prompt(own_deck: list[str], deck_docs: list[str], tool_names: list[str]) -> str
```
Returns a fixed template string (put it as a module constant `PREGAME_TEMPLATE` with `{deck}`, `{docs}`, `{tools}` placeholders):
```
You are the strategist for a Clash Royale agent. You cannot act; you can only read via tools: {tools}.
Our deck: {deck}
Professional notes:
{docs}
Write a game plan in at most 5 short lines: win condition, what to save each spell for, preferred lane, how to defend the opponent's likely win condition, and when to stop spending.
```
```python
class LLMPlanner:
    def __init__(self, model_path: str | None = None)
    def pregame(...): raise NotImplementedError("Qwen planner runs on the Mac via MLX or llama.cpp; see docs/design/pipeline_review.md §4")
```

## 7. `clashbot/executor.py`
```python
@dataclass
class Calibration:
    src_tiles: list[list[float]]    # 4 arena tile coordinates [col, row]
    dst_pixels: list[list[float]]   # the same 4 points in screen pixels of the mirrored window

def homography(cal: Calibration) -> np.ndarray     # 3x3 via the standard 8-unknown linear solve (np.linalg.solve), h33 = 1
def tile_to_screen(tile: list[float], H: np.ndarray) -> tuple[float, float]

class DryRunExecutor:
    def __init__(self): self.log = []
    def execute(self, key: str, tile: list[float] | None, t: float) -> dict   # appends and returns {"t","key","tile"}

class MirroringExecutor:
    def __init__(self, calibration: Calibration, card_slot_pixels: list[list[float]])
    def execute(self, key, tile, t):
        if sys.platform != "darwin": raise NotImplementedError("tapping needs macOS + iPhone Mirroring (Quartz events)")
        raise NotImplementedError("TODO on the Mac: click card slot, then click tile_to_screen(tile)")
```

## 8. `clashbot/loop.py`
```python
@dataclass
class Decision:
    t: float
    candidates: list[Candidate]
    choice: str
    tile: list[float] | None
    timings_ms: dict          # {"candidates": x, "rank": y, "execute": z, "total": w}

def should_decide(state: dict, t: float, last_decision_t: float | None, cards: dict) -> bool
```
True iff `legal_cards(state, cards)` is non-empty AND at least one of: (a) any enemy unit with `eta_tower` dict, tower starting with `"own"`, `s <= 4.0`; (b) `own.elixir >= 9`; (c) `last_decision_t is None or t - last_decision_t >= 3.0`; (d) any opponent recent play with `0 <= t - p.t <= 2.0`.

```python
class FastLoop:
    def __init__(self, generator: CandidateGenerator, ranker: Ranker, executor, prior: dict, plan: Plan | None = None)
    def decide(self, state: dict, t: float, context_text: str, **filters) -> Decision
```
`decide`: time each stage with `time.perf_counter()`; candidates = `generator.generate(state, t, context_text, n=4, **filters)`; options = `[option_text(c.key) for c in candidates]`; `res = ranker.choose(context_text, options, plan.text if plan else None)`; `choice = candidates[res.index].key`; `tile = placement_tile(choice, prior)`; `executor.execute(choice, tile, t)`; return Decision.

Replay CLI (`python3 -m clashbot.loop --replay GAME_ID [--limit N]`):
- Load the game's records from `dataset/v1/states.jsonl` in `t` order (these stand in for live states).
- Build `StateIndex()`, `KnowledgeGraph()`, `CandidateGenerator`, `ScoreRanker`, `DryRunExecutor`, placement prior, `RuleBasedPlanner().pregame(...)` using the game's own deck and `own_deck_key` from games.jsonl.
- For each record: if `should_decide(...)` → `decide(..., exclude_game=GAME_ID)` (so the game cannot retrieve itself) and print one line: `t`, choice, pro `action.key`, match yes/no, total ms. Stop after `--limit` decisions.
- Summary: decisions, exact-key agreement with the pro (on records whose `action.key` is not null), card agreement, latency p50/p95 for each timing field.

## 9. Tests — `tests/test_runtime.py`
1. `format_multiple_choice` exact output for 3 options with and without plan; ValueError for 0 and 9 options.
2. `homography` + `tile_to_screen`: with 4 points mapping an axis-aligned rectangle (scale 10, offset 100), tile [1,2] → (110, 120) within 1e-6.
3. `call_tool`: unknown tool → ValueError; unknown arg → ValueError; `k=99` → ValueError; no tool name contains `write`, `play`, `tap`, `set`, `delete`.
4. `should_decide` cases (a)–(d) with synthetic states; returns False when no legal card.
5. Integration (skip if `db/clashbot.sqlite` missing): `KnowledgeGraph().counter_scores("hog-rider")` non-empty; `CandidateGenerator.generate` on the first val record returns ≤ 4 candidates, last is `"hold"`, all keys legal or hold; `call_tool("search_documents", {"query": "graveyard"}, ctx)` returns ≥ 1 doc; replay CLI via subprocess with `--limit 5` exits 0 and prints a summary.

## 10. Acceptance (run, paste outputs)
```
python3 -m pytest -q tests/test_runtime.py
python3 -m clashbot.loop --replay LGb0mz8Sb8w-m0.0 --limit 15
python3 -m pytest -q
git status --short
```

## 11. Report back (≤ 40 lines)
Files; tests; replay summary (agreement and latency); deviations and why.

---

## Amendment D (Opus review, 2026-10-28)

Reconciles WP5 with D-018 (separate act/hold head; the ranker chooses among plays only).

**D1. `CandidateGenerator.generate`** returns play keys only. Delete the `"hold"` score in step 4 and the hold append in step 5: take the first `n` play keys by `(-score, key)`, fill from `legal` (excluding hold) when fewer than `n`. The kNN channel ignores hits whose `action_key == "hold"`. Default `n = 4` (the ranker training files have 4 options).
**D2. `FastLoop.decide`**: if `generate` returns an empty list, return a `Decision` with `choice = "hold"`, `tile = None`, `candidates = []`, without calling the ranker or executor. Otherwise unchanged. (When to act stays with `should_decide` until the act/hold head exists, D-020.)
**D3. Replay CLI ids starting with "-"**: before `argparse`, rewrite `sys.argv` so that a value following `--replay` becomes `--replay=<value>`; same for `--limit` is not needed. Add a subprocess test with `--replay -V4H_YeMGGk-m0.0 --limit 2`.
**D4. Stub messages**: `_StubDecisionModel` should cite `docs/design/pipeline_review.md §5` instead of `docs/plans/WP5_runtime_skeleton.md §4`.
**D5. Tests**: replace the "last candidate is hold" assertion with "no candidate is hold and every key is in `legal_action_keys`"; add a D2 test with a state that has no legal cards (`own.elixir = 0`, hand of 5-elixir cards).
Accepted as implemented: `LLMPlanner.pregame(own_deck, deck_key=None)`, `ValueError` when ctx.index / ctx.kg is missing, SQL-side `type` filter, direct `props["confidence"]`/`props["support"]` access (all ANSWERED_BY edges carry both).
Re-run §10 with `python3 -m pytest -q --ignore-glob='tests/test_*synthetic*.py' tests/test_runtime.py tests/test_ranker_dataset.py tests/test_retrieval.py tests/test_kg_documents.py tests/test_clean_dataset.py tests/test_states.py` (the cv2-based perception tests need opencv, which this environment lacks).
