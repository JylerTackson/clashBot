# MAC-Z — tactical engine, with d1 deciding over lookup and tactics (specs for Sonnet implementers)

Author: Opus · 2026-11-05 · Implements D-042. Read `docs/plans/MAC-00_common.md` first.

There are three packages:
- **Z1** (card roles) and **Z2** (engine) run in parallel. Z2 codes against the Z1 schema below, and its tests use inline role fixtures.
- **Z3** (wiring into candidates, d1 and the live loop, plus offline eval and the d1 dataset) runs after Z1 and Z2 are merged.

## Why
In live match 2, 17 of the 22 chosen moves came from the KG counter table at fixed pro-median tiles, and 3 came from kNN. Nothing reasoned about what a move does. The engine adds that reasoning:
- which enemy push threatens which tower, and what it needs (air defence, a tank answer, splash);
- which hand card answers it and where to put it;
- the elixir trade;
- when there is an offence window.

Its answer is shown to d1 next to the lookup candidates, and d1 picks.

## Coordinates (from `cr_perception/geometry.py`, `cr_perception/tracking.py`)
- The arena is 18 columns × 32 rows.
- Row 0 is behind our king. Rows 0–14 are our half, rows 15–16 are the river, and rows 17–31 are the enemy half. Row 0 is deployable only in columns 6–11.
- Tower tiles: `own_left (3.5, 4.3)`, `own_right (14.5, 4.3)`, `own_king (9.0, 0.5)`, `enemy_left (3.5, 23.3)`, `enemy_right (14.5, 23.3)`, `enemy_king (9.0, 26.3)`.
- `lane_of` / `zone_of` come from `clashbot/game_rules.py` (col ≤ 5 left, ≥ 12 right; rows ≤ 4 back, ≤ 11 mid, ≤ 16 bridge, else enemy_half).

---
## Z1 — card role table

### Owns
```
clashbot/card_roles.py                 (new: load_roles(), constants, validate_roles())
clashbot/resources/card_roles.json     (new, committed)
tools/data/build_card_roles.py         (new)
tests/test_card_roles.py               (new)
```

### Schema (fixed — Z2 depends on it)
```json
{"version": 1, "cards": {
  "knight":   {"type": "troop", "targets": "ground", "flying": false, "hp_class": "medium", "dps_class": "medium",
               "splash": false, "range": "melee", "speed": "medium", "count": 1, "roles": ["mini_tank"],
               "source": "stats+manual"},
  "fireball": {"type": "spell", "spell_class": "medium", "radius": 2.5, "kills": ["swarm", "light"],
               "roles": ["spell_medium"], "source": "manual"},
  "tesla":    {"type": "building", "targets": "air_ground", "flying": false, "hp_class": "medium",
               "dps_class": "medium", "splash": false, "range": "long", "speed": null, "count": 1,
               "roles": ["building", "air_defense"], "source": "stats+manual"}}}
```
- **Vocabularies** (constants in `card_roles.py`):
  - `TYPES = {troop, building, spell}`
  - `TARGETS = {ground, air_ground, buildings}`
  - `HP_CLASSES = {swarm, light, medium, tank}`
  - `DPS_CLASSES = {low, medium, high}`
  - `RANGES = {melee, short, long}`
  - `SPEEDS = {slow, medium, fast, very_fast}`; `null` for buildings
  - `SPELL_CLASSES = {small, medium, large, utility}`
  - `ROLES = {win_condition, tank, mini_tank, tank_killer, splash_defense, air_defense, swarm, building, kamikaze, support, cycle, spell_small, spell_medium, spell_large, bait}`
- **Meanings:**
  - `hp_class`: `swarm` = dies to a small spell (zap, log); `light` = dies to a medium spell (fireball, poison); `medium` = survives a medium spell; `tank` = ≥ about 3,000 HP at tournament level.
  - `range`: `melee` ≤ 1.5 tiles; `short` < 5; `long` ≥ 5.
  - `count`: the number of units the card spawns.
  - `radius`: in tiles, for spells.
  - `kills`: the HP classes the spell kills outright.
  - `utility`: spells that do no real damage (rage, clone, mirror, freeze, tornado).
  - `win_condition`: cards that target buildings or are played mainly to hit towers (hog rider, ram rider, goblin barrel, royal hogs, wall breakers, balloon, graveyard, miner, battle ram, giant, ...).

### `tools/data/build_card_roles.py`
```
$PY tools/data/build_card_roles.py [--stats-dir data/external/cr-api-data] [--out clashbot/resources/card_roles.json] [--fetch]
```
1. With `--fetch`, download `https://raw.githubusercontent.com/RoyaleAPI/cr-api-data/master/docs/json/cards_stats.json` and `.../cards.json` into `--stats-dir` (gitignored `data/`). Without it, use the files already there.
2. **Stats-derived fields**, for cards whose name matches a `characters` entry. Map our slug to cr-api with `cards.json` `key` (same slugs, e.g. `knight`), then to the character via `summon_character` in `cards_stats.json` (`troop`/`building` sections).
   - `flying` = `flying_height > 0`;
   - `targets` = `buildings` if `target_only_buildings`, else `air_ground` if `attacks_air`, else `ground`;
   - `splash` = `area_damage_radius > 0`;
   - `range` from `range` (game units, 1,000 per tile): ≤ 1,500 melee, < 5,000 short, else long;
   - `speed` from `speed`: ≤ 45 slow, ≤ 60 medium, ≤ 90 fast, else very_fast.
3. **Manual fields** (`hp_class`, `dps_class`, `roles`, `count`, and every spell field) come from a `MANUAL` dict in the script, written by you for **every** playable card in `clashbot.game_rules.load_cards()` (126 cards). Fill in all stats fields manually as well for the 19 cards with no stats: berserker, boss-bandit, cannoneer, dagger-duchess, goblin-curse, goblin-demolisher, goblin-machine, goblinstein, little-prince, mini-p-e-k-k-a, p-e-k-k-a, ronin, royal-chef, rune-giant, spirit-empress, suspicious-bush, tower-princess, vines, void. Use your best game knowledge and mark them `"source": "manual"`.
   - **Where stats and the manual value disagree, stats win.** Print every disagreement, so the manual table can be corrected.
4. Write the JSON sorted by slug. Print a coverage table: cards, stats-derived count, manual-only count, disagreements.

### `clashbot/card_roles.py`
- `load_roles(path=RESOURCE) -> dict[str, dict]`, cached.
- `validate_roles(roles, cards) -> list[str]` returns the problems:
  - a missing playable card;
  - an unknown vocabulary value;
  - a troop or building without `targets` / `hp_class`;
  - a spell without `spell_class` / `radius`.
- Helpers:
  - `hits_air(role) -> bool`: spells → True except the-log / barbarian-barrel / earthquake / giant-snowball (ground only); otherwise `targets == "air_ground"`.
  - `is_tank(role)`, `is_win_condition(role)`.

### Tests (`tests/test_card_roles.py`)
1. `validate_roles(load_roles(), load_cards()) == []`.
2. Spot checks:
   - knight melee/ground/medium;
   - tesla building air_ground;
   - giant tank, targets buildings, win_condition;
   - bats flying swarm air_ground;
   - zap small, kills swarm;
   - fireball medium, kills swarm and light;
   - the-log does not hit air;
   - magic-archer long range;
   - goblin-barrel win_condition;
   - skeletons swarm, count 3.
3. The builder on a tiny fake stats dir (2 characters) derives `flying`, `targets`, `splash`, `range` and `speed` exactly by the mapping rules, and reports a disagreement with a manual value.

### Smoke
Run `build_card_roles.py --fetch` once (network, public data), then the tests. Paste the coverage table and the disagreement list.

---
## Z2 — tactical engine

### Owns
```
clashbot/tactics.py      (new)
tests/test_tactics.py    (new)
```
Only read `clashbot/policy.py`: use `ReactivePolicy(cards).spell_target(card, lane, zone, state)` for spell aim and value (MAC-Y2). Load roles with `clashbot.card_roles.load_roles` when no roles are passed. Tests pass a roles dict built inline in the Z1 schema, so Z2 does not wait for Z1.

### API (fixed — Z3 depends on it)
```python
@dataclass
class Push:
    lane: str                    # "left" | "right" (middle units go to the lane of the nearer x: col < 9 left)
    units: list[dict]            # the enemy unit dicts of the state
    elixir: float                # sum of elixir_cost over distinct known cards
    eta_s: float                 # most urgent: min eta to an own tower; units on our half without eta -> 0.0 if row <= 8 else 2.0
    needs: list[str]             # subset of ["air", "tank", "swarm", "building_targeter"], in that order
    summary: str                 # "giant, musketeer (9 elixir), tower in 6.2s"

@dataclass
class TacticalMove:
    card: str
    key: str                     # card|lane|zone of `tile` (lane_of / zone_of)
    tile: list[float]            # concrete deploy tile, rounded to 0.5
    intent: str                  # "defend" | "counter_push" | "punish" | "leak"
    reason: str                  # <= 60 chars, e.g. "pulls giant to the centre, trade +5"
    score: float                 # engine preference, higher is better
    trade: float | None          # push elixir - card cost (defend), else None

@dataclass
class TacticalReport:
    pushes: list[Push]           # most urgent first
    moves: list[TacticalMove]    # best first, distinct keys, at most 6
    best: TacticalMove | None
    hold_reason: str | None      # set when best is None
    lines: list[str]             # 1..4 lines, each starting with "tactics: "

class TacticalEngine:
    def __init__(self, cards: dict | None = None, roles: dict | None = None, kg=None): ...
    def analyse(self, state: dict, t: float | None = None) -> TacticalReport: ...
    def score_key(self, key: str, state: dict, report: TacticalReport | None = None) -> float | None: ...
```
`kg`: optional `clashbot.kg.KnowledgeGraph`. Use `kg.counter_scores(threat_card)` only if it is given.

### Rules
1. **Hand.**
   - `hand_cards` = the known hand slots with `elixir_cost <= own elixir + 0.5`.
   - If the whole hand is unknown, use `legal_cards(state, cards)` instead (its deck fallback, for dataset states). The live loop's `require_hand` still guards the taps.
   - The engine never proposes a card outside `hand_cards`.
2. **Pushes.** Enemy units (`side == "enemy"`) with an own-tower ETA ≤ 8 s or a tile on our half (row ≤ 14). Group them by lane; middle units go to the nearer x.
   - `needs`:
     - `air` if any unit is flying (role `flying`);
     - `tank` if any `hp_class == tank`;
     - `swarm` if the units of swarm-class cards number ≥ 3, or a card with `count >= 3` and `hp_class == swarm` is present;
     - `building_targeter` if any unit has `targets == buildings`.
   - Units whose card has no role entry count as `medium` ground units.
3. **Defence scores**, for each card `c` in `hand_cards` against the most urgent push `P` (and the second push, if any):
   - **air:** if `P.needs` has `air` and `c` cannot hit air (`hits_air`), skip `c` when P is only air units, else −0.5.
   - **tank:**
     - +0.6 if `c` has `tank_killer` or `dps_class == high`;
     - +0.5 if `c` is a building and P has `building_targeter`;
     - +0.3 if `c` has `swarm` (distraction);
     - −0.6 for spells, unless the spell target (step 4) holds non-tank support worth its cost.
   - **swarm:** +0.6 if `c.splash`, or `c` is a spell whose `kills` includes `swarm`; −0.4 for single-target melee troops.
   - **building_targeter:** +0.5 building, +0.3 `swarm` or `kamikaze`, +0.3 `mini_tank`.
   - **KG:** +0.2 × the normalised KG counter score of the most urgent unit's card (if `kg` is given).
   - **Trade:**
     - `trade = P.elixir − cost(c)`;
     - −0.15 × max(0, −trade − 1), so over-paying by more than 1 elixir is penalised;
     - +0.1 if trade ≥ 1.
   - −0.4 if `c` is a `win_condition`, which is kept for offence.
   - The base score is 0.5.
4. **Defence placement** relative to `pos`, the most urgent unit's tile led by 1.0 s along `predicted_tile_2s` (as `_lead_tile` in policy). `tower` = the own princess tower of `P.lane`. `toward_centre` = +1 for lane left, −1 for lane right.
   - **building:**
     - if P has a `building_targeter`: x = 7.5 for lane left and 10.5 for lane right, with y = clamp(pos.y − 4, 6, 10). This pulls the unit to the centre between the towers.
     - otherwise: `(tower.x + 2 × toward_centre, 6.5)`.
   - **mini_tank / tank_killer / tank (melee):** `(pos.x, max(pos.y − 2.5, 1))` when pos.y > 8 (meet it in front, on our side); else `(pos.x, pos.y − 1)`.
   - **Ranged troop** (`range` short or long, not a building): behind the tower, at `(tower.x + 1.5 × toward_centre, 2.5)`.
   - **swarm troops:** on the threat, at `pos`.
   - **kamikaze:** in front of the threat, at `(pos.x, pos.y − 2.5)`.
   - **Spells:** the tile from `ReactivePolicy(cards).spell_target(c, lane_of(pos), zone_of(pos), state)`. If it returns None, the spell is not proposed.
   - **Clamp** every troop or building tile to x ∈ [0.5, 17.5] and y ∈ [1, 14], then round to 0.5.
   - **Reason** examples: "pulls giant to the centre", "tanks for the tower vs mini-p-e-k-k-a", "kills bats from behind the tower", "fireball musketeer + bats (6 elixir)", each followed by `, trade +N`.
5. **Offence**, when no push has ETA ≤ 8 s. Win conditions in `hand_cards`:
   - **counter_push:**
     - when an ally unit stands in lane L with tile row ≥ 9, own elixir ≥ cost + 1: the win condition at the bridge of L, `(3.5 or 14.5, 14)`. Goblin barrel goes on the enemy princess tower of L, and other spell win conditions (graveyard, miner) go on that tower as well. Score 0.6.
   - **punish:**
     - opponent `elixir_estimate` ≤ 3 and own ≥ cost + 1;
     - the lane opposite the opponent's newest play with a tile; else the lane with the lower enemy tower HP; else left.
     - Score 0.55.
   - **leak:**
     - own elixir ≥ 9.5;
     - a win condition at the bridge of the lane with the lower enemy tower HP (score 0.5); else the cheapest non-spell card, behind the king at `(9.0, 1)` for a tank, else `(tower.x + 1.5 × toward_centre, 2.5)`.
     - Score 0.4.
6. **Best and hold.**
   - `moves` = the defence moves of all pushes plus the offence moves, sorted by score (ties: lower cost, then the card name), keeping the first move per key, at most 6.
   - `best` = the top move. If there are no moves, `best = None` and `hold_reason` is:
     - "no threat, saving elixir (own E, opp ~O)"; or
     - "no card in hand answers the push"; or
     - "not enough elixir".
7. **Lines** (≤ 4), each starting with `tactics: `:
   - **one per push** (max 2): `push {lane}: {summary}; needs {needs or 'nothing special'}`;
   - **best move:** `best {card} @[x,y] {intent}: {reason}`, or `hold: {hold_reason}`;
   - **alternatives:** `also {card} @[x,y] ({reason}); ...` for moves 2–3, if any.
8. **`score_key(key, state, report)`**: the score of the first move whose key equals `key`. Otherwise the defence score (step 3) of the key's card against the most urgent push, if the card is in `hand_cards`. Otherwise None. Z3's tactics ranker uses this for the 4-way grid.
9. Deterministic, with no I/O except the role and KG loading in `__init__`. It must take < 5 ms on a state with 15 units; time it in a test.

### Tests (`tests/test_tactics.py`)
- Hand-made states as in `tests/test_policy.py`, with inline roles and the real `load_cards()`.
1. A giant in the right lane at row 13 with ETA 8 s, and a hand of tesla, knight, zap, musketeer at 7 elixir:
   - `best` is tesla near `(10.5, 9)`, with intent defend and trade +1;
   - the lines contain `push right` and `best tesla`.
2. Bats (3 units) at the left bridge, with a hand of zap, knight, musketeer, fireball:
   - zap is the best answer (spell kills swarm, trade 0);
   - the knight cannot hit air → it is not proposed when the push is air only.
3. Air-only push → no ground-only card is proposed.
4. Over-pay: a lone skeleton push (1 elixir) → fireball is not best, and a cheap troop wins.
5. No push, opponent at 2 elixir, own 7, ram-rider in hand:
   - punish ram-rider at the bridge of the lane opposite the opponent's last play;
   - goblin-barrel → aimed on the enemy princess tower tile.
6. No push, own 6, opponent 6 → `best` None and `hold_reason` starts with "no threat".
7. An all-unknown hand at elixir 10 → the deck fallback is used.
8. `score_key` returns the move score for a proposed key, the defence score for an unproposed hand card, and None for a card not in the hand.
9. Every tile is clamped and lies on our half for defence. `key` matches `lane_of` / `zone_of` of `tile`.
10. A timing test of < 5 ms on 15 units (generous bound, mark `slow` if flaky).

### Smoke
Unit tests only. Do not load models.

---
## Z3 — wiring: d1 decides over lookup and tactics (after Z1 + Z2; amended by Opus at dispatch)

### Owns
```
clashbot/candidates.py   clashbot/d1_format.py   clashbot/ranker.py   clashbot/policy.py (filter exemption only)
clashbot/live.py         (not _build_perception — MAC-PF owns that function)
tools/eval/eval_tactics.py (new)   tools/data/build_d1_tac.py (new)
tests/test_tactics_integration.py (new) + tests/test_candidates*.py, tests/test_d1_format.py, tests/test_live_loop.py,
tests/test_policy.py (update only as needed)
```
- Callers that pass no tactics must see **identical behaviour and output**. Existing tests must pass unchanged unless a change is listed here.

### Z3.1 Candidates (`clashbot/candidates.py`)
- `Candidate` gains `tile: list[float] | None = None` and `note: str | None = None`, as trailing defaulted fields.
- `CandidateGenerator.generate(state, t, context_text, n=4, report=None, tac_n=2, **filters)`.
- **With `report is None`, nothing changes:** no notes, no tiles.
- **With a report:**
  1. **Tactical candidates.** Walk `report.moves` in order and keep the moves whose `key` is in `legal_action_keys(state, cards)`, up to `tac_n` distinct keys. Each becomes `Candidate(key, score=move.score, sources={"tac": round(move.score, 4)}, tile=move.tile, note=f"tactics: {move.intent}, {move.reason}")`.
  2. **Lookup candidates** come next: today's kNN/KG ranking, skipping keys already taken.
     - A lookup key equal to a tactical key merges its `knn`/`kg` sources into the tactical candidate.
     - If `knn > 0`, append `; pros played this too` to the note.
  3. **Lookup notes.** All lookup notes come from one shared helper, `_lookup_note(key, knn_n, kg_n, kg_threat)`:
     - `knn_n > 0` → `pros played this in similar states (support {knn_n:.2f})`;
     - else `kg_n > 0` → `known counter of {threat card}`. Remember which threat produced each KG key in `generate`.
     - else, for fallback candidates → `filler: no pro or counter support`.
  4. The total is still `n`; fill from `legal` as today if short.
- **`CandidateGenerator.option_notes(state, t, context_text, keys, report=None, **filters) -> dict[str, str]`** computes the notes the live path would attach to arbitrary `keys`, with the same kNN/KG normalisation as `generate`:
  - a key that is one of the report's moves → its tactics note;
  - otherwise → the lookup note (or none).

  `build_d1_tac.py` uses it, so training and live notes come from the same code.

### Z3.2 d1 request with notes (`clashbot/d1_format.py`, `clashbot/ranker.py`)
- `joint_request(context_text, candidate_keys, plan=None, notes=None)`: when `notes` has a non-empty entry for a key, the criterion becomes `option_description(key) + " — " + note`.
  - Clip the note to 120 characters.
  - Reject `"<|"`, as the state is rejected.
  - Without notes the output is byte-identical.
- `D1Ranker.build_request(..., notes=None)` and `choose_keys(context, keys, plan=None, notes=None)` pass the notes through.
- `clashbot/live.py::rank_keys(ranker, context, keys, plan, notes=None)` forwards `notes` only when the ranker's `choose_keys` accepts it (inspect the signature), so fakes keep working.
- **`TacticsRanker`** (in `ranker.py`), name `"tactics"`. `choose_candidates(cands) -> RankResult` returns:
  - the index of the highest `sources.get("tac")` (ties → first);
  - else 0;
  - one-hot probs.

### Z3.3 Policy exemption (`clashbot/policy.py`)
- In `ReactivePolicy.filter_candidates`, candidates whose `sources` contain `"tac"` skip the lane/zone constraints of defend, leak and punish. The engine placed them on purpose: a Tesla pull tile is in the middle lane while the threat is in the right lane.
- They **still** go through the spell rule (`_unblind`, including the defend-threat check).
- Order is unchanged.
- Test: a tactical tesla at `middle|mid` survives a defend decision for lane right; a tactical spell with no worthwhile target is dropped.

### Z3.4 Live loop (`clashbot/live.py`)
- `LiveLoop(..., tactics=None)` takes a `TacticalEngine` or None.
- In `step`, after `verdict.act` is True:
  - `report = self.tactics.analyse(state, t)` if engine;
  - `rank_ctx = ctx + "\n" + "\n".join(report.lines)` (else `ctx`);
  - `generate(..., report=report)`;
  - rank with `rank_ctx` and `notes={c.key: c.note for c in cands if c.note}`.
- If the ranker has `choose_candidates`, use it instead of `rank_keys`.
- The act policy keeps receiving `ctx` (its head was trained without tactics).
- **Tile precedence** for the chosen key:
  1. policy `target_tile` (spells);
  2. else the chosen candidate's `tile`;
  3. else `placement_tile`.
- **Logging:**
  - each decision record gains `"tactics_lines"` (list, `[]` without an engine);
  - each logged candidate gains `"tile"` and `"note"`.
  - `summary()` gains `"decisions_by_source"`, a Counter of the chosen candidate's top source: `tac` / `knn` / `kg` / `fallback` / `hold`.
- **CLI:**
  - `--ranker score|d1|tactics`;
  - `--no-tactics` (the default is tactics on);
  - `main` builds `TacticalEngine(cards, kg=<the KnowledgeGraph given to the generator>)`;
  - `--ranker tactics` → `TacticsRanker()`.
- **Tests (fakes):**
  - the ranker context contains the lines, and the policy context does not;
  - notes reach the ranker;
  - tile precedence (all three levels);
  - `--ranker tactics` picks the highest `tac` candidate;
  - `--no-tactics` → identical to today;
  - log fields present.

### Z3.5 `tools/eval/eval_tactics.py` (offline, CPU, no models)
```
$PY tools/eval/eval_tactics.py [--dataset DIR] [--splits val,test] [--out PATH]
```
1. **Per play record.** For each record in `<dataset>/states.jsonl` of the split with `action.type == "play"` and a known card: `report = engine.analyse(rec["state"], rec["t"])`. Report:
   - **coverage**: `best` not None;
   - **top-1 card agreement**: `best.card == pro card`;
   - **top-3 card agreement**: the pro card is among `moves[:3]`;
   - when the card matches: lane and zone agreement (`lane_of` / `zone_of` of the move tile vs the action's lane / zone), and the mean tile distance to `action.tile`.

   Report each overall and split by situation: `push` (report has pushes) vs `no push`.
   The engine uses `kg=KnowledgeGraph()` if it loads, else no KG.
2. **4-way grid.** For each line of `<dataset>/ranker/<split>.jsonl` (fields `id`, `options[].key`, `options[].source == "pos"` marks the answer), with the state from `states.jsonl` by `id`:
   - the tactics ranker picks the option with the highest `engine.score_key(key, state, report)`; None counts as −inf, ties → first;
   - report its accuracy next to random 0.25, placement prior 0.493 / 0.384 and d1 fine-tuned 0.524 / 0.483 (val / test; constants from `docs/data/d1_finetune_joint_eval_v1.json`).
3. Write JSON and `.md` to `--out` (default `<dataset>/eval/tactics.json`) and print the tables.

### Z3.6 `tools/data/build_d1_tac.py` (CPU, no models)
```
$PY tools/data/build_d1_tac.py [--dataset DIR] [--out <dataset>/d1/joint_tac]
```
For each line of `<dataset>/d1/joint/{train,val,test}.jsonl`:
- `rec = states[id]`, `report = engine.analyse(rec["state"], rec["t"])`.
- `context` = `render_context(rec["state"], rec["t"])`; it must equal the line's state text. Count mismatches; if there is one, use the line's state text.
- `notes = generator.option_notes(rec["state"], rec["t"], context, keys, report, splits=("train",), exclude_game=rec["game_id"])`. This avoids retrieving the line's own game.
- The new line has state = `context + "\n" + "\n".join(report.lines)` and the same questions, with notes applied through `joint_request(context_with_lines, keys, notes=notes)`.
- Answers and metadata are unchanged.
- Validate every line with `d1_format.validate`.
- `stats.json`, per split:
  - lines;
  - the share of lines where the answer key has a tactics note;
  - the share where the tactics top move is the answer;
  - the share with any lookup note;
  - mean state characters.
- Build the generator and engine exactly as `clashbot/live.py::main` does: `StateIndex()`, `KnowledgeGraph()`, `load_cards()`.

### Z3.7 Tests (`tests/test_tactics_integration.py` + updates)
1. `generate` with a fake report:
   - tactical candidates come first, with tile and note;
   - illegal tactical keys are skipped;
   - a duplicate lookup key is merged into the tactical candidate;
   - lookup notes are present.
2. Without a report, the result is identical to before. Compare to a call on the same fakes.
3. `option_notes` gives the same notes as `generate` for the keys `generate` returned.
4. `joint_request` with notes: the criteria are suffixed, and `validate` passes. Without notes, the output is byte-identical.
5. The policy exemption (Z3.3).
6. The live-loop items of Z3.4.
7. `eval_tactics.main` and `build_d1_tac.main` on a tiny synthetic dataset dir, with a fake `StateIndex`/`KnowledgeGraph` (monkeypatched): files written, metrics present, validate passes.

### Z3.8 Smoke (CPU only; a GPU training job may be running; never load d1, never touch the screen)
- `$PY tools/eval/eval_tactics.py --out runs/smoke/Z3/tactics.json`: paste both tables.
- `$PY tools/data/build_d1_tac.py --out runs/smoke/Z3/joint_tac`: paste the stats and one example val line (state tail + criteria).
- Then the full suite.

### Z3.9 Heavy runs (Opus)
- `build_d1_tac.py` → `dataset/v1/d1/joint_tac`.
- `train_d1.py --variant joint --data dataset/v1/d1/joint_tac --init db/models/d1-ft-joint --epochs 2 --out db/models/d1-ft-joint-tac`.
- Evaluate it on `joint_tac` val/test against `d1-ft-joint` on `joint`, and against the tactics ranker.
- Then a live match with `--ranker d1 --d1-model db/models/d1-ft-joint-tac`.
