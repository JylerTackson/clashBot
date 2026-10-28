# WP1 — Clean dataset builder + shared rules package (spec for a Sonnet implementer)

Author: Opus · 2026-10-28 · Decisions this implements: D-006, D-008, D-010, D-011, D-012, D-013, D-014, D-015 (`docs/decisions/DECISIONS.md`).

**You are implementing, not designing.** Every behaviour is specified below. If something is not covered, or the spec seems wrong when you hit real data, STOP and report the question instead of choosing. Do not add features, options or files that are not listed.

## 0. Ground rules

- Work in `/home/user/clashBot`. Python 3.11. Allowed third-party imports: none in WP1 (stdlib only). `pytest` is installed for tests.
- Do **not** modify anything under `knowledge_base/`, `tools/extract_states.py`, `cr_perception/`, or existing tests.
- Do **not** `git commit` or `git push`. Leave changes in the working tree.
- Do **not** print or open `data/cookies.txt` (it does not exist here; never create it).
- Keep code style like `tools/extract_states.py`: type hints, small functions, module docstring, `ROOT = Path(__file__).resolve().parents[N]` style paths.

## 1. Files to create

```
clashbot/__init__.py
clashbot/config.py
clashbot/game_rules.py
clashbot/units.py
clashbot/state_text.py
clashbot/resources/unit_map.json      (copy verbatim from docs/plans/WP1_unit_map.json)
tools/data/__init__.py                 (empty)
tools/data/build_clean_dataset.py
tests/test_clean_dataset.py
```
Also append to `.gitignore` exactly these two lines (each on its own line, at the end):
```
/dataset/
/db/
```

## 2. `clashbot/__init__.py`
Only a docstring: `"""clashbot runtime package: shared game rules, state rendering, retrieval and the agent loop."""`

## 3. `clashbot/config.py`
```python
"""Paths and constants shared by the dataset builders and the live loop."""
from __future__ import annotations
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
KB = ROOT / "knowledge_base"
CARD_INDEX = KB / "meta" / "card_index.json"
STATES_DIR = KB / "states"
MATCHES_DIR = KB / "matches"
DECKS_DIR = KB / "decks"
RESOURCES = PKG / "resources"
UNIT_MAP_PATH = RESOURCES / "unit_map.json"
DATASET_VERSION = "v1"
DATASET_DIR = ROOT / "dataset" / DATASET_VERSION
DB_DIR = ROOT / "db"
DB_PATH = DB_DIR / "clashbot.sqlite"

LANES = ("left", "middle", "right")
ZONES = ("back", "mid", "bridge", "enemy_half")
PHASES = ("single_elixir", "double_elixir", "triple_elixir", "overtime")
HOLD_KEY = "hold"
ANYWHERE_CARDS = frozenset({"miner", "goblin-drill"})
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train, val, test — by video (D-006)
```

## 4. `clashbot/game_rules.py`
Module docstring: "Lane/zone geometry, card table and action keys (D-010). Single source of truth for builders and the live loop."

Functions (exact behaviour):

```python
def lane_of(tile) -> str | None      # copy of tools/extract_states.lane_of: None if not tile; col=tile[0]; col<=5 'left'; col>=12 'right'; else 'middle'
def zone_of(tile) -> str | None      # copy of tools/extract_states.zone_of: None if not tile; row=tile[1]; <=4 'back'; <=11 'mid'; <=16 'bridge'; else 'enemy_half'
```
`tile` values may be floats; compare the raw numbers exactly as the originals do (no rounding).

```python
def load_cards(path: Path = CARD_INDEX) -> dict[str, dict]:
```
Reads `card_index.json` → `{"cards": [...]}`. Returns `{slug: {"slug", "name", "card_type", "rarity", "targets", "elixir_cost", "has_evolution", "has_hero_variant"}}` where `elixir_cost` is converted: int/float → float; a string that `float()` accepts → float; anything else (e.g. `"n/a"`, None) → None. Cache with `functools.lru_cache(maxsize=1)` on a no-arg helper `_cards_cached()`; `load_cards()` with the default path returns the cached dict.

```python
def is_playable(slug: str, cards: dict) -> bool
```
True iff slug in cards and `card_type != "Tower Troop"` and `elixir_cost is not None`.

```python
def action_key(card: str | None, lane: str | None, zone: str | None) -> str | None
```
Returns `f"{card}|{lane}|{zone}"` if all three are non-empty strings, else None.

```python
def parse_action_key(key: str) -> dict
```
`"hold"` → `{"type": "hold", "card": None, "lane": None, "zone": None}`; otherwise split on `|` into exactly 3 parts → `{"type": "play", "card", "lane", "zone"}`; raise `ValueError` for anything else.

```python
def legal_cards(state: dict, cards: dict, elixir_tolerance: float = 0.5) -> list[str]
```
- `hand = [h for h in state["own"]["hand"] if h]`. If `hand` is non-empty, `pool = hand` else `pool = list(state["own"]["deck"])`.
- Keep `c` if `is_playable(c, cards)` and `cards[c]["elixir_cost"] <= (state["own"]["elixir"] or 0) + elixir_tolerance`.
- Return in pool order with duplicates removed (first occurrence wins).

```python
def legal_zones(card: str, state: dict, cards: dict) -> list[str]
```
- If `cards[card]["card_type"] == "Spell"` or `card in ANYWHERE_CARDS` → `list(ZONES)`.
- Else → `["back", "mid", "bridge"]`, plus `"enemy_half"` appended when `state["opponent"]["towers"]["left"] == 0` or `state["opponent"]["towers"]["right"] == 0` (use `== 0`; None does not count).

```python
def legal_action_keys(state: dict, cards: dict, elixir_tolerance: float = 0.5) -> list[str]
```
For each card in `legal_cards(...)` (in order), for each lane in `LANES`, for each zone in `legal_zones(card, ...)`: `action_key(card, lane, zone)`. Append `HOLD_KEY` last.

## 5. `clashbot/units.py`
Docstring: "Resolve YOLO unit classes to cards (D-014)."

```python
def load_unit_map(path: Path = UNIT_MAP_PATH) -> dict[str, dict]
```
Load JSON, drop keys starting with `_`. Cache like `load_cards`.

```python
def resolve(unit_class: str, cards: dict | None = None, unit_map: dict | None = None) -> dict
```
Defaults: `cards = load_cards()`, `unit_map = load_unit_map()`.
- If `unit_class in unit_map`: `e = unit_map[unit_class]` → `{"card": e["primary"], "is_evo": e["is_evo"], "kind": e["kind"], "candidates": [c for c in e["candidates"] if c in cards]}`.
- Elif `unit_class in cards` → `{"card": unit_class, "is_evo": False, "kind": "card", "candidates": [unit_class]}`.
- Else → `{"card": None, "is_evo": False, "kind": "unknown", "candidates": []}`.

## 6. `clashbot/state_text.py`
Docstring: "Deterministic text views of a clean game state (D-015). Same code for dataset building and the live loop."

Copy `fmt_num` verbatim from `tools/extract_states.py`.

```python
def unit_token(u: dict) -> str
```
If `u.get("card")`: `u["card"] + ("*" if u.get("is_evo") else "")`; else `u["unit_class"]`.

```python
def render_threats(units: list[dict]) -> list[str]
```
Same logic as `tools/extract_states.render_threats` but the name is `unit_token(u)` instead of `u['unit']`:
skip units whose `side != "enemy"` or whose `eta_tower` is not a dict; skip if `str(eta.get("tower","")).startswith("own")` is False or `eta.get("s") is None` or `eta["s"] > 8.0`; else append `f"{unit_token(u)}(e) {u.get('heading') or 'approaching'}, tower in {eta['s']}s"`.

```python
def render_context(state: dict, t: float | None = None) -> str
```
Exactly lines 1–5 of `tools/extract_states.state_text`, with these differences only:
- field entries use `unit_token(u)` instead of `u['unit']`;
- line 1 is exactly:
  ```python
  status = "" if own.get("deck_status", "valid") == "valid" else f" [deck {own['deck_status']}]"
  opp_deck = ", ".join(opp["deck_known"]) if opp.get("deck_known") else "unknown"
  line1 = (f"[{state['phase'] or 'unknown'}|{state['clock'] or '?'}] "
           f"elixir {fmt_num(own['elixir'])} (opp ~{fmt_num(opp['elixir_estimate'])}). "
           f"hand: {hand} (next {own.get('next_card') or '?'}). "
           f"deck: {', '.join(own['deck']) if own['deck'] else 'unknown'}{status}. "
           f"opp deck: {opp_deck}")
  ```
Returns the 5 lines joined with `"\n"`. It must never contain the substring `"action:"`.

```python
def render_action(action: dict) -> str
```
Exactly line 6 of `tools/extract_states.state_text` (the `action: ...` line), same formatting.

```python
def render_state_text(state: dict, action: dict, t: float | None = None) -> str
```
`render_context(state, t) + "\n" + render_action(action)`.

## 7. `clashbot/resources/unit_map.json`
`cp docs/plans/WP1_unit_map.json clashbot/resources/unit_map.json` — byte-identical.

## 8. `tools/data/build_clean_dataset.py`

Docstring: what it builds, the CLI, and "Deterministic: same inputs → byte-identical outputs except MANIFEST.json's built_at."

CLI:
```
python3 tools/data/build_clean_dataset.py [--out DIR] [--games ID1,ID2,...]
```
`--out` default `clashbot.config.DATASET_DIR`. `--games` restricts to those game ids (file stems in `knowledge_base/states/`), used by tests. Make `sys.path` include ROOT so `import clashbot` works when run as a script (`sys.path.insert(0, str(ROOT))`).

Inputs: every `knowledge_base/states/<game_id>.jsonl` (sorted by file name), the matching `knowledge_base/matches/<game_id>.md` frontmatter, the card table, the unit map.

### 8.1 Match frontmatter
Parse the YAML-ish frontmatter between the first two `---` lines using a tiny local parser: for each line `key: value` at column 0, store `value.strip()`. For list values written as `[a, b, c]`, split into a list of stripped strings. Strip surrounding double quotes from scalars. You need only: `own_deck_key`, `own_archetype`, `opponent_archetype_guess`, `result`, `video_time`, `match_index`, `video_id`. Missing → None.

### 8.2 Splits (by video)
- `videos` = sorted set of `sample["source"]["video_id"]` over all games being built **when `--games` is not given**; when `--games` is given, still compute splits from ALL state files (read only the first line of each file for its `source.video_id`) so a game's split never depends on `--games`.
- Order videos by `hashlib.sha1(video_id.encode()).hexdigest()` ascending.
- `n = len(videos)`, `n_train = round(0.70 * n)`, `n_val = round(0.15 * n)`, test = the rest. First `n_train` → train, next `n_val` → val, rest → test.

### 8.3 Per-game cleaning
For each game file (samples sorted by `t`):

1. **Own deck** — `raw = most_common(tuple(s["state"]["own"]["deck"]) for s in samples)` (Counter.most_common(1); ties: the first one Counter returns). `own_deck = sorted({c for c in raw if is_playable(c, cards)})`. `deck_status = "valid" if len == 8 else "incomplete" if len == 7 else "invalid"`.
2. **Opponent play counts** — set `seen = set()`; for every sample, for `p` in `state.opponent.recent_plays + outcome.opponent_plays` (treat missing lists as empty): if `p.get("card")` is playable, add `(round(p["t"], 1), p["card"])`. `play_count[card] = number of entries in seen with that card`.
3. **Game opponent deck** — `union = {c for s in samples for c in (s["state"]["opponent"]["deck_known"] or []) if is_playable(c, cards)} | set(play_count)`. If `len(union) <= 8`: `opp_deck = sorted(union)`, `opp_dropped = []`. Else: `opp_deck = sorted(sorted(union, key=lambda c: (-play_count.get(c, 0), c))[:8])`, `opp_dropped = sorted(union - set(opp_deck))`, game flag `"opp_deck_capped"`.
4. **Each sample** → clean record (§8.4).

### 8.4 Clean sample record (field order as listed)

```
id                 original id
game_id            file stem
split              train | val | test
kind               key | play | periodic
t                  original t (seconds on the source timeline; only differences are meaningful)
state:
  clock, phase, match_seconds          copied
  own:
    elixir, hand_confidence, hero_cards, towers   copied
    hand       [h if (h and is_playable(h, cards)) else None for h in padded] where padded = (list(original hand or [])[:4] + [None]*4)[:4]
    next_card  original if playable else None
    deck       own_deck (game level, §8.3.1)
    deck_status
    recent_plays   original list with each entry's card replaced by None if not playable
  opponent:
    elixir_estimate, towers                       copied
    deck_known  [c for c in original deck_known if is_playable(c) and (len(union) <= 8 or c in opp_deck)] (original order, duplicates removed)
    deck_complete  len(deck_known) == 8
    recent_plays   as for own
  units: list of {unit_class, card, is_evo, side, side_repaired, tile, heading, speed_tiles_s, predicted_tile_2s, eta_tower}
  threats: render_threats(units)        (recomputed after repairs)
action:
  type        copied
  card        original if playable else None
  tile        copied
  lane        lane_of(tile)
  zone        zone_of(tile)
  key         action_key(card, lane, zone) if type == "play" else ("hold" if type == "hold" else None)
  delay_s, elixir_before, elixir_after, detect_source, confidence, window_s    copied (None if missing)
  following_plays   original list, cards not playable → None
outcome:      copied, with opponent_plays cards not playable → None
key_moment:   original value; if it is a dict with a "text" string, apply scrub() to that text
enrichment:   original dict with scrub() applied to every string value and to every string inside list values
quality:      original dict without the key "state_source"
flags:        list of strings (§8.5), sorted
train_eligible: bool (§8.5)
exclusion_reasons: list of the blocking flags present, sorted
state_text:   render_state_text(state, action, t)
context_text: render_context(state, t)
```
Dropped entirely: `schema_version`, `source`, `commentary`, `context_refs`.

**Units** — for each original unit `u`: `r = resolve(u["unit"])`; start `{unit_class: u["unit"], card: r["card"], is_evo: r["is_evo"], side: u["side"], side_repaired: False, tile, heading, speed_tiles_s, predicted_tile_2s, eta_tower}`.
Side repair runs only when `deck_status != "invalid"`:
`own_ok = bool(set(r["candidates"]) & set(own_deck))`, `opp_ok = bool(set(r["candidates"]) & set(opp_deck))`.
If `side == "ally" and not own_ok and opp_ok` → `side = "enemy"`; if `side == "enemy" and not opp_ok and own_ok` → `side = "ally"`. When flipped: `side_repaired = True` and `eta_tower = None`.

**scrub(text)** — apply in this order:
1. `re.sub(r"\(\s*t\s*[=≈~][^)]*\)", "", text)`
2. `re.sub(r"\bt\s*[=≈~]\s*\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*(?:—|–|:)?\s*", "", text)`
3. `re.sub(r"https?://\S+", "", text)`
4. `re.sub(r"\[([^\]]+)\]\((?:\.\./)*[^)]*?\.md\)", r"\1", text)`
5. `re.sub(r"[ \t]{2,}", " ", text)`; then strip leading/trailing whitespace.

### 8.5 Flags and training eligibility
Sample flags (exact strings):

| flag | condition | blocking |
|---|---|---|
| `deck_invalid` | deck_status == "invalid" | yes |
| `deck_incomplete` | deck_status == "incomplete" | no |
| `action_unknown` | action.type == "unknown" | yes |
| `action_card_unknown` | type == "play" and card is None | yes |
| `placement_unknown` | type == "play" and tile is None | yes |
| `action_card_off_deck` | type == "play" and card is not None and deck_status == "valid" and card not in own_deck | yes |
| `phase_unknown` | state.phase is None | no |
| `hand_unknown` | all 4 cleaned hand slots are None | no |
| `units_side_repaired` | any unit side_repaired | no |
| `units_unverified` | any unit whose resolve() candidates list is empty | no |
| `opp_deck_capped` | game was capped (§8.3.3) | no |

`train_eligible = not any(blocking flag present)`.

### 8.6 Outputs (in `--out`, created with parents)

1. `states.jsonl` — all clean records, games in file-name order, samples by `t`. `json.dumps(rec, ensure_ascii=False)` one per line.
2. `games.jsonl` — one line per game:
   `{game_id, split, own_deck, deck_status, own_deck_key, own_archetype, opponent_deck, opponent_dropped, opponent_archetype_guess, result, n_samples: {key, play, periodic}, n_train_eligible, side_repairs, flags}`
   - `own_deck_key` = frontmatter value if `deck_status == "valid"` and `knowledge_base/decks/<value>.md` exists, else None.
   - `result` = frontmatter `result` (string) or None.
   - `side_repairs` = number of units flipped in the game; `flags` = sorted game flags (`opp_deck_capped`, and `deck_invalid`/`deck_incomplete` when applicable).
3. `provenance.jsonl` — `{game_id, video_id, match_index, creator, video_title, video_url, video_time}` from the first sample's `source` and the frontmatter `video_time`.
4. `splits.json` — `{"method": "sha1(video_id) order, 70/15/15 by video", "train": [...], "val": [...], "test": [...]}` (video ids, in hash order).
5. `CLEANING_REPORT.md` — sections, in order:
   - Totals: games, samples by kind, train-eligible by split × kind (table).
   - Flag counts (table: flag, samples, blocking yes/no).
   - Deck status per game: list of non-valid games with own_deck length.
   - Opponent decks capped: game id → dropped cards.
   - Side repairs: total flipped units; top 20 unit classes flipped (class, count).
   - Unresolved unit classes: classes whose resolve() kind is "unknown", with counts.
   - Action keys: top 30 keys among train-eligible play samples with counts; count of hold.
6. `MANIFEST.json` — `{"dataset_version": "v1", "built_at": <UTC ISO8601>, "source_commit": <git rev-parse HEAD or null>, "builder": "tools/data/build_clean_dataset.py", "files": {name: {"sha256", "lines"}}}` for files 1–4.

Print a 6-line summary to stdout: games, samples, train-eligible, side repairs, capped games, output dir.

## 9. `tests/test_clean_dataset.py`
Use `pytest`. Tests (each small, synthetic dicts unless stated):
1. `lane_of`/`zone_of` boundaries: cols 5/6/11/12, rows 4/5/11/12/16/17, None tile.
2. `action_key` / `parse_action_key` round trip; `parse_action_key("x|y")` raises ValueError.
3. `legal_cards`: known hand filters by elixir (+0.5 tolerance); empty hand falls back to deck; tower troops excluded.
4. `legal_zones`: spell → 4 zones; troop → 3; troop with `opponent.towers.left == 0` → 4; `"miner"` → 4.
5. `resolve`: `"skeleton-evolution"` → card skeletons, is_evo True; `"knight"` → card knight; `"nonsense"` → kind unknown; every candidate in unit_map.json exists in the card index.
6. `render_context` never contains `"action:"`; `render_state_text` ends with the action line; `unit_token` adds `*` for evolutions.
7. Integration (real data): run the builder via `subprocess` with `--games LGb0mz8Sb8w-m0.0,vp7jP_vjxcU-m0.1 --out <tmp_path>`; assert: both games present in games.jsonl; `vp7jP_vjxcU-m0.1` has `deck_status == "invalid"` and every sample of it has `train_eligible is False`; no record in states.jsonl has keys `source`, `commentary`, `context_refs`; every `opponent.deck_known` has length ≤ 8; every `context_text` lacks `"action:"`; for each game, look up its `video_id` in provenance.jsonl, find which list of splits.json contains that video id, and assert it equals the game's `split` in games.jsonl.
8. Determinism: build the same two games twice into two tmp dirs; `states.jsonl` files are byte-identical.

## 10. Acceptance (run all, paste outputs in your report)

```
python3 -m pytest -q tests/test_clean_dataset.py tests/test_states.py
python3 tools/data/build_clean_dataset.py
head -c 1500 dataset/v1/states.jsonl; echo
cat dataset/v1/CLEANING_REPORT.md
python3 -c "import json;r=[json.loads(l) for l in open('dataset/v1/states.jsonl')];print(len(r), sum(x['train_eligible'] for x in r))"
git status --short
```

## 11. Report back (≤ 40 lines)
Files created/changed; test results; the six summary lines; the flag table from the report; anything in the spec you could not follow exactly and why (do not silently deviate).

---

## Amendment A (Opus review, 2026-10-28) — apply on top of the spec above

Review found that the play-count opponent deck agrees with the curated match-file deck on only 56% of cards (e.g. `LGb0mz8Sb8w-m0.0` puts the player's own Royal Hogs in the opponent deck and drops the Giant). Decision D-016 replaces §8.3 step 3 and the `deck_known` rule in §8.4.

**A1. Game opponent deck (replaces §8.3.3).** Read frontmatter `opponent_deck_seen` (list; missing → []). `fm = [c for c in it if is_playable(c, cards)]`, duplicates removed, original order kept.
- If `1 <= len(fm) <= 8`: `opp_deck = sorted(fm)`, `opponent_deck_source = "match_file"`, `opp_dropped = []`.
- If `len(fm) > 8`: rank `fm` by `(-play_count.get(c, 0), fm.index(c))`, keep the first 8, `opp_deck = sorted(those)`, `opp_dropped = sorted(rest)`, `opponent_deck_source = "match_file_capped"`, game flag `opp_deck_capped`.
- If `len(fm) == 0`: use the original §8.3.3 rule unchanged; `opponent_deck_source = "perception_capped"` when it capped, else `"perception"`.
`play_count` is still computed as in §8.3.2.

**A2. First-seen times.** For each card `c` in `opp_deck`: `first_seen[c]` = the minimum over (a) every `p.t` of opponent plays of `c` collected in §8.3.2 (from `recent_plays` and `outcome.opponent_plays`) and (b) every sample `t` whose original `opponent.deck_known` contains `c`. Cards never observed get no entry.

**A3. Per-sample `opponent.deck_known` (replaces the §8.4 rule).** `[c for c in opp_deck if c in first_seen and first_seen[c] <= t]`, ordered by `(first_seen[c], c)`. `deck_complete = len(deck_known) == len(opp_deck) == 8`.

**A4. games.jsonl.** Add the field `opponent_deck_source` after `opponent_dropped`.

**A5. CLEANING_REPORT.md.** Add a line under Totals: counts of games by `opponent_deck_source`. In every markdown table cell, escape `|` as `\|`.

**A6. `.gitignore`.** Line 5 is `data/`, which also ignores `tools/data/` and `docs/data/`. Replace that exact line with `/data/` (root-only; it still covers the cookies and videos folder). Do not change any other line.

**A7. Tests.** Add: (a) in the integration test, `LGb0mz8Sb8w-m0.0` has `opponent_deck_source == "match_file"` and its `opponent_deck` contains `"giant"` and not `"royal-hogs"`; (b) for every sample of that game, `deck_known` is a subset of the game's `opponent_deck` and is non-decreasing in length as `t` increases; (c) `git check-ignore tools/data/build_clean_dataset.py` exits with status 1 (not ignored).

Re-run all §10 acceptance commands after applying A1–A7, plus `git check-ignore -v tools/data/build_clean_dataset.py; echo $?`.
