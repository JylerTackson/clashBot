# MAC-PF — perception fixes: opponent plays, elixir, deck; phantom units; evo/hero hand art (spec for a Sonnet implementer)

Author: Opus · 2026-11-05 · Implements D-043 (user allowed perception fixes) for FIX_LIST #30, plus the unread hand slot of #26. Read `docs/plans/MAC-00_common.md` first. The causes below come from a read-only investigation of `runs/poc/20261105-172506` and the code (file:line as of `45fea45`).

## Owns
```
cr_perception/events.py  cr_perception/elixir_sim.py  cr_perception/decktracker.py
cr_perception/perception.py  cr_perception/detect.py  cr_perception/hud.py  cr_perception/tracking.py (only if needed)
clashbot/live_state.py   (only _clean_units and the opponent first-seen logic in observe_events/build)
clashbot/live.py         (only _build_perception: pass the deck)
tests/test_perception_fixes.py (new) + existing perception/live-state tests (update only as needed, say which)
```
- **Video and offline processing must keep working.** Every new restriction that needs the own deck is opt-in, and is used only when a deck is given: the live loop passes it, video batch processing does not.
- The dense dataset and `dataset/v1` are not rebuilt (D-043).

## 1. Opponent plays counted several times → opponent elixir ~0 (`events.py`, `perception.py`, `elixir_sim.py`)
**Evidence.** Ten opponent plays (36 elixir) were reported between t = 7.8 and t = 24.9, when at most about 14 elixir was possible. The estimate equals the simulation exactly.

**Fixes:**
- a. **Label after arena.**
  - `PlayDetector.update_labels` currently checks only own plays (`events.py` ~303).
  - Fix: skip an opponent label event when an opponent play of the same card was reported within ±3 s and within 4 tiles (when both tiles are known; time only otherwise).
- b. **Spawned units.**
  - Add `SPAWNED_BY: dict[str, set[str]]` (spawned card → spawner cards). At least:
    - bats ← night-witch;
    - skeletons ← graveyard, witch, tombstone, skeleton-barrel, skeleton-king;
    - goblins ← goblin-hut, goblin-barrel, goblin-drill, goblin-machine;
    - spear-goblins ← goblin-hut, goblin-giant;
    - fire-spirit ← furnace;
    - barbarians ← battle-ram, barbarian-barrel;
    - royal-recruits ← royal-delivery.
    - Use the slugs of `clashbot/resources/unit_map.json` / `knowledge_base` and add the ones you know.
  - In the arena path (`update_units` ~170–227), suppress a play of card X while a track of a spawner of X (same side) is alive, or ended < 2 s ago, within 5 tiles.
  - Also suppress it when a play of a spawner of X was reported in the last 4 s within 5 tiles (spell spawners such as graveyard or barrel).
- c. **Class flicker.**
  - The problem: a new arena track that starts within 1.5 tiles of another enemy track's current position (alive, or ended < 1 s ago) is a continuation of that track (e.g. `unknown_unit` → giant, giant ↔ royal-giant). It is not a play.
  - Fix: suppress the play in that case.
- d. **Predicted positions are not hits.**
  - `units` includes predicted-only Kalman tracks for up to 2.5 s (`perception.py` ~375, `tracking.py` ~294), and every detect frame adds a hit (`events.py` ~186).
  - Fix: pass `since_seen` (or a `measured` flag) with each unit and count a hit only on frames where the track was measured.
  - Keep the rule of 2 hits over 0.8 s.
- e. **Late join.**
  - On the first clock read of a match, start the opponent simulation at `min(10, 5 + elapsed / 2.8)`, with `elapsed = 180 − seconds left` in single elixir (use `clock.py` helpers).
  - Mark `elixir_conf` low until the first opponent play after that.

**Tests** (synthetic `UnitObs` / label events into `PlayDetector`):
- a night-witch(e) track at [13, 24], then bats tracks at t + 5 and t + 10 → zero bats plays;
- an arena giant, then a giant label 1 s later at the same tile → one play;
- a track `unknown_unit` → giant at the same position → one play;
- a unit seen in 1 measured frame plus predictions → no play;
- a first clock read of 2:43 → the opponent simulation starts at 10.

## 2. Opponent deck admitted on one weak observation (`decktracker.py`, `clashbot/live_state.py`)
**Evidence.** 8 cards were admitted in 17 s, giant and royal-giant together.

**Fix:**
- `observe_play(card, weight)` sums evidence per card:
  - label score ≥ 0.9 → 1.0;
  - label 0.85–0.9 → 0.6;
  - arena play → 0.5.
- A card is **known** once its sum reaches 1.0.
- While fewer than 8 cards are known, admit as before. When 8 are known, a new card that reaches 1.5 replaces the known card with the lowest evidence, if that card's evidence is below the newcomer's.
- `deck_known` lists only known cards, in first-known order.
- `LiveStateBuilder.observe_events` currently admits every opponent event's card itself (`live_state.py` ~256 and ~329). Make it apply the same weights, from the event's `detect_source` and `score`/`conf` fields (check what the events carry). A card counts as first seen at the time it becomes known.
  - Keep the old behaviour behind `LiveConfig.opp_evidence: bool = False`. The live loop sets it True, so the dataset builders keep their output.

**Tests:**
- one arena play → the card is not known;
- two arena plays 5 s apart → known;
- one label at 0.92 → known;
- a 9th card with strong evidence replaces the weakest.

## 3. Our towers detected as ally units (`detect.py`, `perception.py`, `clashbot/live_state.py`)
**Evidence.** "Stationary" allies (`skeleton-dragons`, `axe`, `dark-prince`) sat at [3, 4]–[3, 6] and [14, 4]–[14, 6], next to our princess towers at (3.5, 4.3) and (14.5, 4.3), from t = 9 to t = 41. `unknown_unit(a)` sat at the enemy king tower.

**Fixes:**
- a. `detect.py` ~184–186: a weak detection whose class is a tower class keeps category `tower`; it does not become `unknown_unit` of category unit.
- b. `perception.py` ~360–367: drop a unit detection when **both** of these hold:
  - its tile is within 1.0 tile of a standing tower's centre (`tracking.TOWER_TILES`);
  - its class is `unknown_unit` or a class whose card is in neither known deck. With no deck given, this means `unknown_unit` only.

  Real units hitting a tower stand ≥ 1.5 tiles from its centre, so they are kept.
- c. `clashbot/live_state.py::_clean_units`: when `deck_status == "valid"`, drop ally units whose resolved candidate cards do not overlap our deck. Also drop ally `unknown_unit`s.
  - Keep enemy `unknown_unit`s: an unknown enemy can still be a threat.
  - Behind `LiveConfig.strict_units: bool = False`, set True by the live loop.

**Tests:**
- `_clean_units` with an ally `skeleton-dragons` at [14, 4] and a valid deck without it → dropped (strict), kept (not strict);
- synthetic detections at a tower centre (unknown_unit) → not tracked, while a `hog-rider` 1.8 tiles from the tower is kept.

## 4. Evolution / hero hand art and the deck restriction (`hud.py`, `perception.py`, `clashbot/live.py`)
**Evidence.** One slot was never read at 7–10 elixir (likely ram-rider: wiki art only). The evolution art in `knowledge_base/evolutions/images/*-evolution.png` is never loaded: `extra_dirs` is not passed, and `hud.py` ~280 only strips `_ev\d+`. The matcher scores all 126 cards, so a best match outside the deck is discarded later.

**Fixes:**
- a. Load `knowledge_base/evolutions/images` by default (as an extra template set), mapping `<slug>-evolution` → `<slug>` (strip `-evolution` as well as `_ev\d+`).
- b. `Perception(..., own_deck: list[str] | None = None)` → `CardMatcher.match(..., allowed=set)`. When a deck is given, scores of cards outside the deck become −9, so the best in-deck card wins. Pass the deck from `clashbot/live.py::_build_perception` (`args.deck`). Video processing passes nothing (unchanged).
- c. Learned templates: the per-card cap of 12 is already full with crops from other videos, so live crops never enter. Make live crops (`live_*.png`) load in addition to the cap (the last 12 live + the last 12 others).

**Tests** (synthetic hand ROIs, like `tests/test_hud_synthetic.py`):
- `knight-evolution.png` pasted in a slot → `knight`;
- `ram-rider.png` with deck restriction → `ram-rider`;
- a card outside the deck with the restriction → not returned.

## 5. Smoke (CPU only — a GPU training job is running)
- Run the live loop in **dry run** on a recorded pro video segment of about 90 s, with `--device cpu`. Choose any `data/videos/<id>/<id>.mp4` and a match window from `dataset/v1/provenance.jsonl`, deck = that game's `own_deck`, the cheapest ranker, and `--out runs/smoke/PF/after`. Read `clashbot/live.py --help` for the flags.
- Run it again with your changes stashed (`--out runs/smoke/PF/before`).
- Compare:
  - the number of opponent plays per minute;
  - opponent `deck_known` vs the game's `opponent_deck` in `games.jsonl` (precision);
  - the opponent elixir estimate range;
  - ally units not in the deck;
  - the fraction of hand slots read.
- Never touch the screen.

## 6. Report
Per MAC-00 §5, plus the before/after smoke table and every existing test you changed (with the reason).
