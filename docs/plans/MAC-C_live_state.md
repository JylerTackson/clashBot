# MAC-C — live state builder: perception → clean state (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Implements D-015 (same code path for training and serving), pipeline review §3.5, D-029. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
clashbot/live_state.py           (new)
tools/eval/live_state_parity.py  (new)
tests/test_live_state.py         (new)
```

## Goal
Turn the perception output (`cr_perception.state.GameState` + `PlayEvent`s) into a `state` dict with **exactly the shape of `state` in `dataset/v1/states.jsonl`**, plus its `context_text`, so retrieval, features, candidates and d1 requests see live states the way they saw training states. The dataset path was: perception `states.jsonl` row → `tools/extract_states.py::Game.build_state` (the `states.jsonl` branch) → `tools/data/build_clean_dataset.py::clean_sample`. You re-implement that path for one live instant. Read both functions before you start.

## Contract (package G codes against this — keep it exact)
```python
# clashbot/live_state.py
RECENT_S = 12.0                     # same as tools/extract_states.py
SCHEMA_PHASES = ("single_elixir", "double_elixir", "triple_elixir", "overtime")

@dataclass
class LiveConfig:
    own_deck: list[str]                       # the 8 card slugs the user plays
    hero_cards: list[str] = field(default_factory=list)
    recent_s: float = RECENT_S

class LiveStateBuilder:
    def __init__(self, config: LiveConfig, cards: dict | None = None, unit_map: dict | None = None): ...
    def reset(self) -> None
    def observe_events(self, events: list) -> None          # PlayEvent objects or their to_json() dicts
    def build(self, gs) -> tuple[dict, str] | None          # GameState object or its to_json() dict
```
`build` returns `(state, context_text)` or `None` when `gs.readiness != "match"` or `gs.own` is empty.

## 1. Normalisation of inputs
- `gs`: if it has `to_json`, use `gs.to_json()`; else it is already that dict (keys `t, readiness, match_id, match_clock, match_seconds, phase, own, opponent, units`). Units in the dict form are `{"class","side","tile","conf","bbox","track_id","category","motion"?}`.
- Events: if an item has `to_json`, use it; dict keys `timestamp, player ("own"|"opponent"), card, tile, detect_source, confidence`. Keep events whose `timestamp` is not `None`. Dedupe on `(round(timestamp, 1), player, card)`. Keep them sorted by timestamp.
- `match_id`: when `gs["match_id"]` differs from the last seen value (initially `None`), call `reset()` first (new match: forget events and first-seen times). `reset()` clears events, opponent first-seen dict and the last `match_id`.

## 2. `build(gs)` — field by field (t = `gs["t"]`)
`cards = load_cards()` and `unit_map = load_unit_map()` by default; `deck = sorted({c for c in config.own_deck if is_playable(c, cards)})`; `deck_status = "valid" if len(deck) == 8 else "incomplete" if len(deck) == 7 else "invalid"`; `deck_set = set(config.own_deck) if len(set(config.own_deck)) >= 8 else None` (exactly `Game.deck_set`).

| output | rule (source code it mirrors) |
|---|---|
| `clock` | `gs["match_clock"]` |
| `phase` | `norm_phase`: in `SCHEMA_PHASES` → itself; endswith `"_overtime"` or `== "overtime"` → `"overtime"`; else `None` |
| `match_seconds` | `gs["match_seconds"]` |
| `own.elixir` | `own["elixir"]`; if not `None` → `max(0.0, min(10.0, float(x)))` |
| `own.hand_confidence` | mean of the numeric entries of `own["hand_conf"]`, `round(..., 3)`, else `None` |
| `own.hero_cards` | `list(config.hero_cards)` |
| `own.towers` | `{"king","left","right"}` from `own["towers"]` (missing → `None`) |
| `own.hand` | `pad4(own["hand"])`; if `deck_set`: a slot whose card is not in `deck_set` → `None`; then a slot whose card is not playable → `None` |
| `own.next_card` | `own["next_card"]`; not in `deck_set` (when known) → `None`; not playable → `None` |
| `own.deck` / `own.deck_status` | `deck` / `deck_status` above |
| `own.recent_plays` | own events with `t - recent_s <= ts <= t`, dropping HUD reads of foreign cards (`detect_source == "hud"` and `deck_set` and card not in `deck_set`), each as `play_record` = `{"t": round(ts, 3), "player", "card", "tile", "elixir_cost": cards[card]["elixir_cost"] if card in cards else None, "detect_source", "confidence"}`, then `clean_plays` (card not playable → `None`) |
| `opponent.elixir_estimate` | `opp.get("elixir_est")` |
| `opponent.towers` | as own |
| `opponent.deck_known` | first-seen times per playable opponent card: from opponent events (`ts`) and from `opp.get("deck_known")` (first `t` the card appears there); keep cards first seen at or before `t`; the 8 earliest by `(first_seen, slug)`; output sorted by `(first_seen, slug)` |
| `opponent.deck_complete` | `len(deck_known) == 8` |
| `opponent.recent_plays` | opponent events with `t - recent_s <= ts <= t`, `play_record` then `clean_plays` |
| `units` | from `gs["units"]` excluding `category == "tower"` and `class` in `TOWER_CLASSES` (import from `cr_perception.detect` inside a `try`, fallback to the literal set used in `tools/extract_states.py`); each through `unit_from_state` (copy of `tools/extract_states.py::unit_from_state`) then `clean_units(raw, deck_status, deck, deck_known, cards, unit_map)` (copy of `build_clean_dataset.clean_units`, same output keys and order) |
| `threats` | `clashbot.state_text.render_threats(units)` |
| `context_text` | `clashbot.state_text.render_context(state, t)` |

Key order of the dicts must equal `clean_sample`'s: `state = {clock, phase, match_seconds, own{elixir, hand_confidence, hero_cards, towers, hand, next_card, deck, deck_status, recent_plays}, opponent{elixir_estimate, towers, deck_known, deck_complete, recent_plays}, units, threats}`.

Do not import from `tools/` inside `clashbot/live_state.py` (runtime package must not depend on build scripts); copy the small helpers (`pad4`, `towers_of`, `norm_phase`, `play_record`, `clean_plays`, `unit_from_state`, `clean_units`) as private functions. The tests prove they match the originals.

## 3. `tools/eval/live_state_parity.py`
```
$PY tools/eval/live_state_parity.py --perception PATH/states.jsonl --game GAME_ID [--dataset DIR] [--tol 0.6] [--out PATH]
```
1. Read the perception jsonl (`cr_perception.recorder.read_jsonl`): rows with `type == "state"` → states (keep `readiness == "match"`), rows with `type == "play_event"` → events.
2. Game deck: `own_deck` from `<dataset>/games.jsonl` for `--game`. Records: `<dataset>/states.jsonl` rows with that `game_id`, sorted by `t`.
3. For each record: nearest perception state with `|t_row - t| <= tol` (skip the record and count `no_state` if none); fresh `LiveStateBuilder(LiveConfig(own_deck))`; `observe_events([e for e in events if e["timestamp"] <= rec_t])`; `build(row)` (override `row["t"] = rec_t` so recent-play windows line up).
4. Per-record agreement vs `rec["state"]`: `phase`, `clock` (equal), `elixir` (`abs diff <= 0.5`, both `None` = agree), `hand_slots` (fraction of 4 slots equal), `next_card`, `towers` (fraction of the 6 tower values equal within 1% or both `None`), `units_count_diff` (abs), `units_jaccard` (multiset Jaccard of `(card or unit_class, side)`), `deck_known_jaccard`, `own_recent_jaccard`, `opp_recent_jaccard` (multisets of card), `context_exact` (live `context_text == rec["context_text"]`), `feature_cosine` (`float(featurize(live_state, t) @ featurize(rec["state"], t))`).
5. Output JSON (`--out`, default `<dataset>/eval/live_state_parity_<game>.json`, mkdir parents): `{"game_id", "n_records", "n_compared", "no_state", "mean": {field: value}, "median_feature_cosine", "examples": first 10 per-record dicts (id, t, each field)}`; print the `mean` table as markdown.

## 4. Tests — `tests/test_live_state.py`
Build fixtures in the test with `cr_perception.state.GameState/UnitObs/PlayEvent` (pure dataclasses; importing `cr_perception.state` is cheap). Use a real deck: `DECK = ["archers","berserker","bomb-tower","goblinstein","magic-archer","royal-delivery","tornado","wall-breakers"]` and real cards (`load_cards()`).
1. `readiness="menu"` → `None`; empty `own` → `None`.
2. Shape: built `state` has exactly the key order above at every level; `context_text == render_context(state, t)`.
3. Hand sanitising: hand `["tornado", "mega-minion", None, "berserker"]` → `["tornado", None, None, "berserker"]`; next card off-deck → `None`; elixir `12` → `10.0`; phase `"double_elixir_overtime"` → `"overtime"`, `"weird"` → `None`.
4. Recent plays: own events at `t-13` (out), `t-5` (in), HUD read of an off-deck card at `t-2` (dropped); opponent events at `t-12` (in, boundary) and `t+1` (out); record shapes equal `play_record` from `tools/extract_states.py` followed by `clean_plays` from `tools/data/build_clean_dataset.py` (import both in the test).
5. Opponent deck_known: 10 distinct opponent cards over time → only 8 earliest, sorted by first-seen; card only in `opponent["deck_known"]` counts from the first `t` it appears; `deck_complete` true at 8.
6. Units: a tower `UnitObs` (category tower) is dropped; a unit with `motion={"heading": "advancing left lane", "speed": 1.2, "pred_2s": [3, 10], "eta_tower": {"tower": "own_left", "s": 3.0}, "tile": [3, 12]}` → output unit equals `build_clean_dataset.clean_units([extract_states.unit_from_state(u_json)], ...)` exactly; `threats` contains that unit's line.
7. Match reset: events from `match_id=1` do not appear after a state with `match_id=2`.
8. Duplicate events (same rounded timestamp/player/card) are counted once.
9. Real-data round trip (skip if `dataset/v1/states.jsonl` missing): for 50 records (first 50 of the val split), turn `rec["state"]` back into a perception-like `gs` dict (units as `{"class": u["unit_class"], "side": u["side"], "tile": u["tile"], "category": "unit", "motion": {"heading": u["heading"], "speed": u["speed_tiles_s"], "pred_2s": u["predicted_tile_2s"], "eta_tower": u["eta_tower"]}}`, own `hand`/`next_card`/`elixir`/`towers`, `hand_conf: None`, opponent `elixir_est`, `towers`, `deck_known`; events from `own.recent_plays` + `opponent.recent_plays` with `timestamp = p["t"]`); build with `LiveConfig(own_deck=rec["state"]["own"]["deck"], hero_cards=rec["state"]["own"]["hero_cards"] or [])`; assert `featurize(live, t) @ featurize(rec_state, t) >= 0.99` for at least 45 of 50 and hand/elixir/phase equal for all 50. (The opponent deck uses first-seen order, so allow `deck_known` to differ.)
10. Parity tool: write a tiny perception jsonl in `tmp_path` (2 states + 1 event) and a tiny dataset dir (`games.jsonl`, `states.jsonl` with 2 records built by your builder); run `main([...])` → JSON written, `n_compared == 2`, `mean["phase"] == 1.0`.

## 5. Smoke (allowed once, after the symlinks in MAC-00 §1)
`$PY -m pytest -q tests/test_live_state.py` with the real dataset present (test 9 runs).

## 6. Heavy run (Opus, at integration)
Download one val/test video (D-029) into `data/videos/`, `tools/calibrate.py` on it, `tools/run_video.py --start/--end` over one game, then `live_state_parity.py`. Field agreement + median feature cosine go into STATUS.md.

## 7. Report
Per MAC-00 §5.
