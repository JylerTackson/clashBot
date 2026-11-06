# MAC-AK — knowledge graph v2: stats, mechanics, pro answers with placement, opponent-deck model (specs for Sonnet implementers)

Author: Opus · 2026-11-06 · Implements D-048. Read `docs/plans/MAC-00_common.md` first.

**Packages.**
- **K1** (store + query API + builder), **K2** (pro-answer mining) and **K3** (opponent-deck model) run **in parallel now**.
- **K4** (mechanics edges + use in engine / policy / LLM context) runs after MAC-AH and MAC-AI merge.
- All packages are CPU only: never load a model, never touch the screen.

**Running at the same time.**
- MAC-AH owns `clashbot/unit_stats.py`, `clashbot/resources/unit_stats.json`, `tactics.py` and `policy.py`.
- MAC-AI (hero ability) owns perception, executor, candidates and `d1_format`.
- Do not touch any of their files.

## Shared contracts (fixed — all packages code against these)

**Node ids:**
- `card:<slug>`, `unit:<class>`, `hero:<slug>`, `evolution:<slug>`, `ability:<hero-slug>`;
- `deck:<deck_key>`, `archetype:<name>`, `game:<game_id>`;
- `situation:<key>` (K2);
- stat blocks are **not** nodes; they live in the `stats` table.

**Store:** SQLite `db/kg2.sqlite` (db/ is gitignored), built by `tools/data/build_kg_v2.py`.
```sql
nodes(id TEXT PRIMARY KEY, type TEXT, name TEXT, props TEXT)                -- props JSON
edges(src TEXT, rel TEXT, dst TEXT, weight REAL, support INTEGER, source TEXT, props TEXT)
stats(card TEXT, level INTEGER, unit TEXT, props TEXT, PRIMARY KEY(card, level, unit))   -- props JSON: hp, damage, dps, …
meta(key TEXT PRIMARY KEY, value TEXT)                                       -- build info, versions, counts
```
- Indices on `edges(src, rel)`, `edges(dst, rel)` and `stats(card)`.
- `source` is one of `wiki`, `pro_mined`, `deck_cooc`, `role_transfer`, `sim` (K4) or `kb_table`.

**Relations:**
- From v1, kept: `COUNTERS` (wiki), `SYNERGY`, `SUBSTITUTE`, `IN_DECK`, `HAS_ARCHETYPE`, `SPAWNED_BY`, `EVOLUTION_OF`, `HERO_OF`, `GAME_OWN_CARD`, `GAME_OPP_CARD`, `GAME_DECK`.
- New:
  - `ANSWERED_BY` (K2, replaces the v1 edges);
  - `PLAYED_AFTER` (K2);
  - `CO_OCCURS` (K3);
  - `SIMILAR_ROLE` (K1);
  - `HAS_ABILITY` (K1);
  - `KILLS`, `STOPS`, `PULLS` (K4).

**K2 mined files** (inputs to K1), in `dataset/v1/kg2/`:
- `answer_events.jsonl`: one line per pro answer event (fields below);
- `answers.jsonl`: aggregated `{threat, answer, support, held, held_rate, mean_trade, mean_dt_s, offset: {dx_med, dy_med, dx_iqr, dy_iqr}, eta_at_answer_med, dist_at_answer_med}`;
- `situations.jsonl`: `{key, threat_cards, lane_rel, eta_band, elixir_band, n, answers: {card: {n, held_rate}}}`;
- `sequences.jsonl`: `{who: own|opp, first, then, support, lag_med_s}`.

**K3 file:** `dataset/v1/kg2/deck_cooc.json`, containing `{"cards": [...], "pair_counts": {a: {b: n}}, "card_counts": {...}, "decks": [[...8 cards...], ...]}`.

**Query API** (`clashbot/kg2.py`, K1):
```python
class KG2:
    def __init__(self, path=DB_DIR/"kg2.sqlite"): ...
    def stats(self, card: str, level: int) -> dict | None
    def answers_for(self, threat: str, our_cards: list[str], k: int = 5) -> list[Answer]
    def predict_opp_deck(self, seen: list[str], top: int = 8) -> list[tuple[str, float]]   # delegates to K3's model
    def situation(self, threat_cards: list[str], lane_rel: str, eta: float | None, elixir: float | None) -> dict | None
    def facts_lines(self, state: dict, our_deck: list[str], max_lines: int = 3) -> list[str]
@dataclass
class Answer:
    card: str; score: float; sources: dict   # {"pro": x, "role_transfer": y, "wiki": z, "sim": w}
    support: int; held_rate: float | None; offset: dict | None; eta_at_answer: float | None; via: str | None  # via = pro card for role transfer
```

---
## K1 — store, builder and query API

### Owns
```
clashbot/kg2.py   tools/data/build_kg_v2.py   tests/test_kg2.py
clashbot/config.py (only a KG2_PATH constant, if needed)
```

### Builder (`tools/data/build_kg_v2.py [--dataset DIR] [--out db/kg2.sqlite]`)
1. **v1.** Load nodes and edges from `dataset/v1/kg/{nodes,edges}.jsonl` and copy the v1 relations listed above. Drop the v1 `ANSWERED_BY`; K2 supersedes it.
2. **Stats.**
   - If `clashbot/resources/unit_stats.json` exists (MAC-AH), load every card and unit at levels 1–16 into `stats`, through `clashbot.unit_stats.stat_at` if importable.
   - Otherwise, parse `knowledge_base/cards/*.md` "Stats by level" tables directly, using a minimal parser in K1 for the columns Hitpoints / Damage / Damage per second / Death Damage / Area Damage / Crown Tower Damage, with case and plural variants.
   - Record which path was used in `meta`.
3. **Heroes and abilities.** For each `knowledge_base/heroes/*-hero.md` with `ability_name`, add an `ability:<hero>` node with name / cost / description, plus `HAS_ABILITY` from `hero:<slug>`. Example: musketeer → Trusty Turret, cost 3.
4. **SIMILAR_ROLE.** Build a role vector per card from `clashbot.card_roles.load_roles()`: targets, flying, hp_class, dps_class, range, splash, building, and the roles set. Add edges for the top 3 most similar cards (cosine ≥ 0.6) with weight = similarity. These drive role transfer.
5. **K2 / K3 inputs.** If K2's files exist, add `ANSWERED_BY` edges from `answers.jsonl`: weight `held_rate × log1p(support)`, props = the whole aggregate. Add situation nodes from `situations.jsonl`. Add `PLAYED_AFTER` from `sequences.jsonl`. If K3's file exists, add `CO_OCCURS` edges (weight = P(b|a), props: count, lift). **Missing files are fine:** log and continue, so tests use fixtures.
6. **`meta`.** Counts per relation and source, and the build time.

### Query semantics
- **`answers_for(threat, our_cards)`.** Candidate answers are the cards in `our_cards`. Score each from four sources:
  - **pro:** the `ANSWERED_BY` edge `threat → card`: held-rate weighted, shrunk by support (`support / (support + 5)`);
  - **role_transfer:** for pro answers `threat → Y` with Y not in our cards, credit each of our cards Z with `SIMILAR_ROLE(Y, Z) × pro_score(Y) × 0.7`. Set `via = Y`;
  - **wiki:** v1 `COUNTERS`;
  - **sim:** K4 `STOPS`, 0 until K4.

  `score = 0.45 pro + 0.30 role_transfer + 0.10 wiki + 0.15 sim`, renormalised over the sources that exist. Return the top k with their evidence (`support`, `held_rate`, `offset` from the pro or `via` card).
- **`predict_opp_deck`.** Delegate to `clashbot.opp_model` (K3) if importable, else co-occurrence counting from `CO_OCCURS` edges.
- **`situation`.** The nearest situation key (the same key function as K2; import it from `tools.data.mine_pro_answers` if present, else a local copy kept identical).
- **`facts_lines(state, our_deck)`.** At most 3 lines, each ≤ 160 characters, for the LLM:
  - **line 1:** the most important enemy threat on the board, its key stats at level 15 and the top 2 answers with evidence. Example: `KG: balloon L15 2458hp 937/drop; answers musketeer (pros via musketeer 31% n=48 held 70%), tesla (role of cannon, n=22)`;
  - **line 2:** the second threat, if any;
  - **line 3:** the predicted opponent cards not yet seen. Example: `KG: opp likely also has night-witch 0.62, lightning 0.41`.

### Tests (fixture KB and fixture K2 / K3 files in tmp)
- The builder creates the tables and copies v1 relations.
- Stats come from the KB tables (balloon L15 HP 2458 in the real KB; test with a fixture table).
- The ability node exists.
- `SIMILAR_ROLE` is symmetric-ish and sensible: tesla ↔ cannon / inferno-tower high; knight ↔ valkyrie medium.
- `answers_for` scores the pro, role-transfer and wiki sources, with renormalisation when a source is missing.
- `facts_lines` respects the length limits.
- Running against the real `db/kg2.sqlite` after a real build: report the counts.

---
## K2 — mine pro answers, placement offsets, timing, outcomes, sequences, situations

### Owns
```
tools/data/mine_pro_answers.py   tests/test_mine_answers.py
outputs: dataset/v1/kg2/{answer_events,answers,situations,sequences}.jsonl
```

### Source
- `dataset/v1/states.jsonl`: per record, `state` (own / opponent towers, `recent_plays` with `t`, card, tile; units with side, tile, `eta_tower`, heading) and `action`. Read the format; `t` is game seconds.
- Also `dataset/v1/act_dense/states_{split}.jsonl`: 1 Hz sidecar states from the live builder, to follow units and tower HP between plays.
- **Train split only** for the aggregates. Also write val / test events with their split, so K4 can evaluate.

### Answer events
- **Threat.** For each opponent play P (card, tile, time) — from `recent_plays` deduplicated across records, or an enemy unit first seen crossing to our half — the threat is P's card.
- **Answer.** Take the pro's first own play A within `[P.t, P.t + 8 s]` that is defensive:
  - A's tile is on own half or the river (row ≤ 16);
  - or A is a spell landing within 4 tiles of the threat's tracked position.
- **Event fields:**
  - `threat`, `answer`, `dt_s = A.t − P.t`;
  - `threat_tile_at_answer`: the tracked tile of the threat unit nearest A.t, from the dense states if available, else P's tile;
  - `answer_tile`;
  - `offset = answer_tile − threat_tile_at_answer`, **mirrored to a canonical lane**: if the threat is in the right half (x ≥ 9), set `dx = −dx`, so offsets toward the centre are positive;
  - `eta_at_answer` (the threat's `eta_tower` if present), `dist_at_answer` (tiles to the nearest own princess tower);
  - `own_elixir`, `opp_elixir_est`;
  - **`held`**: the own princess tower in the threat's lane loses < 300 HP within the next 12 s (tower HP from the dense states; unknown → null);
  - `trade = threat_cost − answer_cost`;
  - `split`, `game_id`.
- **Aggregates per (threat, answer)** (train only):
  - `support`, `held_rate` (null-safe), `mean_trade`, `mean_dt_s`;
  - the offset median and IQR, `eta_at_answer_med`, `dist_at_answer_med`.

### Situations
- **Key** `(top-2 threat cards by elixir on our half / within ETA 10 s, sorted)`, `lane_rel ∈ {same, split, centre}`, `eta_band ∈ {0-3, 3-6, 6-10, >10}`, `elixir_band ∈ {0-3, 4-6, 7-10}`. Put the key function in the module as `situation_key(...)` (K1 imports it).
- **Value:** n plus the per-answer-card counts and held rates.

### Sequences
- Own play → own next play within 5 s (pushes, supports); opponent → opponent likewise.
- Keep `support` and `lag_med_s`, with support ≥ 3.

### Sanity report (print it, and put it in the report)
- Number of answer events per split.
- For the 10 most common threats: the top 3 answers with support and held rate.
- For building answers vs building-targeter threats (giant, hog-rider, balloon, royal-giant, ram-rider, battle-ram): the offset distribution. Expected: buildings mostly toward the centre, `dy` negative (in front of the threat).
- The held-rate distribution.
- **Data checks:** events dropped for missing tiles, unknown cards, or duplicate plays. Duplicate opponent plays across records must be deduplicated: same card within 1.5 s and 3 tiles = the same play.

### Tests
On a synthetic dataset:
- dedup;
- lane mirroring of offsets;
- the defensive filter;
- the `held` computation from tower HP;
- aggregation;
- situation keys;
- sequences.

---
## K3 — opponent-deck model

### Owns
```
clashbot/opp_model.py   tools/data/build_deck_cooc.py   tools/eval/eval_opp_model.py   tests/test_opp_model.py
output: dataset/v1/kg2/deck_cooc.json
```

### Data
- Every full 8-card deck seen: `dataset/v1/games.jsonl` (`own_deck` and `opponent_deck` when 7–8 cards are known) plus the 68 `knowledge_base/decks/*.md` deck lists (parse the card list; see `tools/data/kb_markdown.py::load_deck_registry`).
- Deduplicate identical decks, but count them.

### Model (`clashbot/opp_model.py`)
- `OppDeckModel.load(path)`; `predict(seen: list[str], top=8) -> list[(card, prob)]`, excluding seen cards.
- **Score** = a mix of:
  - (a) **nearest decks:** Jaccard(seen, deck) over known decks, softmax with temperature;
  - (b) **naive co-occurrence:** `Σ_s log P(c | s)` with add-one smoothing.

  Combine 0.6 (a) + 0.4 (b), with probabilities normalised to sum to (8 − |seen|).
- **Archetype hint:** `likely_archetype(seen)` from deck → archetype (`HAS_ARCHETYPE` in v1 nodes / edges, or the deck files).

### Evaluation (`tools/eval/eval_opp_model.py`)
- **Leave-one-game-out** on `games.jsonl` opponent decks: for k = 1..5 cards revealed in order of first appearance in the game (from `recent_plays` times), report recall@(8 − k) of the hidden cards, and precision@3.
- **Baselines:** global card popularity, and random.
- Write `dataset/v1/eval/opp_model.{json,md}`.

### Tests
- Fixtures with 4 decks:
  - seen ⊂ deck A → A's other cards rank top;
  - seen cards are never predicted;
  - probabilities sum as specified.
- The builder parses a fixture KB deck file.

---
## K4 — mechanics edges and using the graph in decisions (after MAC-AH and MAC-AI; Opus amends with exact details at dispatch)
- **`STOPS`, `KILLS`, `PULLS`** edges from MAC-AH's simulation, for all (our card × opponent card) pairs at our / opponent levels:
  - `STOPS` weight = damage prevented per elixir;
  - `KILLS` for spells by HP;
  - `PULLS` for building vs building-targeter.
- **Engine.** Defend candidates and their ordering come from `KG2.answers_for`, merged with the simulation adequacy. Placement uses the pro offset (median, mirrored back to the lane) when `support ≥ 5`, else the geometric rule.
- **Policy.** Reserve elixir for the predicted opponent cards: keep an air answer when `P(balloon / minions / lava) ≥ 0.5`.
- **LLM context.** Add `KG2.facts_lines`, then rebuild `joint_tac`, retrain d1 overnight and evaluate.

## Implementation checks (Opus, for every package)
- Read the report against the spec.
- Run the package tests plus the full suite on the merged result.
- Spot-check outputs:
  - K2: the top answers for balloon / giant / hog; the building offsets;
  - K3: recall@k vs popularity;
  - K1: `answers_for("balloon", our deck)` and `facts_lines` on a recorded live state.
- Any failure → a FIX_LIST row plus a fix agent before the next package depends on it.
