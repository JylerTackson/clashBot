# MAC-Y — live fixes from Training Camp match 2: hand memory, no plays from an unread hand, aimed and worthwhile spells (spec for two Sonnet implementers)

Author: Opus · 2026-11-05 · FIX_LIST #26–#29. Read `docs/plans/MAC-00_common.md` first. Two packages, run in parallel: **Y1** (hand) and **Y2** (spells). The only shared surface is the `target_tile` method (§Y2.4) that Y1 calls through `getattr`.

## Evidence (run `runs/poc/20261105-172506`, fine-tuned d1 + reactive policy, lost)
- 27 decisions, only 9 taps sent: **15 were `skipped_card_not_in_hand`**. From t ≈ 13 s on, the hand read `?, ?, ?, ?` in 17 of 20 decisions. In this run, every one of the 24 cards the hand reader recognized was affordable at that moment, so greyed-out (unaffordable) cards are not recognized. One slot was never read even at 7–10 elixir. `legal_cards` then falls back to the whole deck, so the bot picked cards it could not tap.
- Spells were wasted:
  - a fireball on a lone giant 10.9 s from our tower;
  - two zaps aimed at `[8, 26]` in the enemy half while the threat sat at our tower.

  The reason: the spell tile is the pro median tile of the `card|lane|zone` key (`placement_tile`), not where the enemy units are. The no-blind-spell rule only checks that some enemy unit is somewhere in that lane and zone.
- The result: 10 elixir spent in the first 15 s, then 2–4 elixir for the rest of the match while a giant, bats, balloon, mini P.E.K.K.A and night witch pushed the right lane. The right tower fell, then the king tower.

---
## Y1 — positional hand memory (#26) and no plays from an unread hand (#27)

### Owns
```
clashbot/live_state.py
clashbot/live.py
tests/test_live_state.py
tests/test_live_loop.py
```
Do **not** change the default behaviour of `LiveStateBuilder`. `tools/data/build_dense_act.py` and the parity tests use it: the tracker is opt-in.

### Y1.1 `HandTracker` (in `clashbot/live_state.py`)
```python
PENDING_S = 1.5

class HandTracker:
    """Positional hand memory for the live loop. A slot keeps its last read card until we play from it
    (executor-confirmed taps); greyed-out cards are therefore remembered instead of lost. With the 8-card deck
    known, the cycle (last 4 own plays = the queue, front = next card) fills a single unknown slot."""
    def __init__(self, deck: list[str] | None, pending_s: float = PENDING_S): ...
    def reset(self) -> None: ...                 # slots [None]*4, next None, log [], pending {}
    def note_play(self, card: str, slot: int, t: float) -> None: ...
    def update(self, read: list, read_next: str | None, t: float) -> list: ...   # returns the 4 tracked slots
    filled: list[bool]                           # per slot: value NOT read in this frame (memory / inference)
```
- `deck` is used only when it has 8 distinct cards, else `None` (no cycle inference).
- **`note_play(card, slot, t)`**:
  1. `incoming = self.next if self.next and self.next != card else (self.log[-4] if len(self.log) >= 4 else None)`. Before the play, the queue is `log[-4:]` and its front is `log[-4]`.
  2. `self.log.append(card)`; `self.slots[slot] = incoming`; `self.pending[slot] = (card, t)`.
  3. `self.next = self.log[-4] if len(self.log) >= 4 else None`.
- **`update(read, read_next, t)`**, in this order:
  1. `read_next` not None → `self.next = read_next`.
  2. For each slot `i` with read `c = read[i]`:
     - **slot `i` has `pending = (pc, pt)`:**
       - `c is None`: if `t - pt >= pending_s`, drop the pending entry (the play landed). Either way, keep the slot.
       - `c == pc` and `t - pt < pending_s`: ignore the read (the screen lags behind the tap).
       - `c == pc` and `t - pt >= pending_s`: the play did not land. Set `slots[i] = pc`, remove the **last** occurrence of `pc` from `log`, and drop the pending entry. If `self.next` was derived from the log, recompute it the same way as in `note_play` step 3.
       - `c != pc`: drop the pending entry and treat `c` as a normal read.
     - **normal read, `c` not None:** set `slots[i] = c` and clear every other slot `j` holding `c`, because a card is in at most one slot. If `c` is in `log[-4:]` (we think it is in the queue), remove its last occurrence from `log`: that play did not really happen.
  3. **Cycle inference**, only when the deck is known and `len(log) >= 4`:
     - `queue = set(log[-4:])`;
     - set every slot holding a card in `queue` to None (stale);
     - `rest = [c for c in deck if c not in queue and c not in slots]`;
     - if exactly one slot is None and `len(rest) == 1`, fill that slot with `rest[0]`;
     - if `read_next` is None, `self.next = log[-4]`.
  4. `filled[i] = read[i] is None and slots[i] is not None` (after the steps above). Return `list(self.slots)`.
- Memory never times out by itself: a slot only changes on a read, a play, or inference.

### Y1.2 Builder integration
- `LiveConfig` gains `hand_tracker: bool = False`.
- `LiveStateBuilder.__init__`: when the flag is set, `self.tracker = HandTracker(sorted(set(config.own_deck)) if len(set(config.own_deck)) == 8 else None)`, else `self.tracker = None`. `reset()` also resets the tracker (new match).
- `build()`: when `self.tracker` is set, replace `_hold_hand` with `hand = self.tracker.update(hand, nxt, t)` (the hand after the existing deck / playable filtering) and `nxt = self.tracker.next`; `self.last_hand_reused = any(self.tracker.filled)`. Keep the existing `hand_confidence` rule (None when reused). Without the tracker, nothing changes.
- New `LiveStateBuilder.note_own_play(card, slot, t)`: forwards to the tracker, or does nothing when there is no tracker.

### Y1.3 `clashbot/live.py`
- `main` builds `LiveStateBuilder(LiveConfig(own_deck=deck, hero_cards=heroes, hand_tracker=True))`.
- After `self.executor.execute(...)`: if `entry.get("sent")` and `entry.get("slot") is not None` and the builder has `note_own_play`, call `self.builder.note_own_play(key.split("|")[0], entry["slot"], t)`. A dry run never calls it, because `sent` is False there.
- **No plays from an unread hand (#27).** `LiveLoop.__init__(..., require_hand: bool = True)`.
  - After `filter_candidates`, when `require_hand` is set, keep only the candidates whose card is in `state["own"]["hand"]`, keeping their order.
  - If this empties a non-empty list, record a decision with `key "hold"` and executor entry `{"t", "key": "hold", "tile": None, "status": "hold_hand_unknown", "clicks": [], "sent": False}`. Do not call the executor. Set `last_decision_t`, as for `hold_policy`. Log `cands` unfiltered, as `hold_policy` does.
  - Count these holds in `summary()["hand_unknown_holds"]`.
- **Spell aim.** Where the loop computes `tile = placement_tile(key, self.prior)`: first `aim = getattr(self.policy, "target_tile", None)`. If it is callable and returns a tile for `(key, state)`, use that tile; otherwise use `placement_tile`.

### Y1.4 Tests
- **`tests/test_live_state.py`, tracker unit tests:**
  1. A full read followed by all-None reads for 10 s keeps all four cards, with `filled` all True.
  2. `note_play(X, 1, t)` with `next` read as Y → slot 1 = Y.
  3. A read of X in slot 1 at `t + 0.5` is ignored. A read of X at `t + 2.0` restores X and removes it from `log`.
  4. Reading slot 2 = c while memory has c in slot 0 → slot 0 = None.
  5. After 4 logged plays with an 8-card deck: `next == log[-4]`; a single unknown slot is filled by the one remaining card; a memory slot holding a queued card is cleared.
  6. A new `match_id` resets the tracker.
- **`tests/test_live_state.py`, builder:** with `hand_tracker=False`, the output is identical to before (all existing tests pass unchanged). With `True`, greyed frames keep the hand and `last_hand_reused` is True.
- **`tests/test_live_loop.py`** (fakes):
  1. A sent play calls `note_own_play(card, slot, t)`; a dry run does not.
  2. A candidate card that is not in the hand → `hold_hand_unknown`, the executor is not called, and it is counted in the summary.
  3. A fake policy whose `target_tile` returns `[3.0, 20.0]` → the executor receives that tile.
  4. A policy without `target_tile` → `placement_tile` is used.

### Y1.5 Smoke
Unit tests only: `pytest tests/test_live_state.py tests/test_live_loop.py -q`, then the full suite. A GPU training job is running, so do not load models and do not touch the screen.

---
## Y2 — aimed and worthwhile spells (#28, #29)

### Owns
```
clashbot/policy.py
tests/test_policy.py
tests/test_act_head.py        (update only if needed)
```

### Y2.1 Constants (top of `clashbot/policy.py`)
```python
SPELL_RADIUS = {"zap": 2.5, "fireball": 2.5, "arrows": 4.0, "poison": 3.5, "lightning": 3.5, "rocket": 2.0,
                "the-log": 2.0, "barbarian-barrel": 2.0, "giant-snowball": 2.5, "earthquake": 3.5, "tornado": 5.5,
                "freeze": 3.0, "void": 2.5, "goblin-curse": 3.0, "royal-delivery": 3.0, "vines": 2.5}
DEFAULT_SPELL_RADIUS = 2.5
SPELL_LEAD_S = {"fireball": 1.0, "rocket": 1.5, "arrows": 0.8, "poison": 0.5, "lightning": 0.6}   # travel time
DEFAULT_SPELL_LEAD_S = 0.3
TANK_CARDS = {"giant", "royal-giant", "golem", "p-e-k-k-a", "mega-knight", "lava-hound", "electro-giant",
              "goblin-giant", "giant-skeleton", "elixir-golem", "goblinstein", "rune-giant"}
NON_DAMAGE_SPELLS = {"rage", "clone", "mirror", "goblin-barrel", "graveyard"}   # keep the old lane+zone rule
SWARM_MIN_UNITS = 3
ARENA_COLS, ARENA_ROWS = 18, 32
```

### Y2.2 Targets
- `_lead_tile(u, lead_s)`: the unit's tile moved toward its predicted position:
  - `tile + (pred - tile) * (lead_s / 2.0)` when `predicted_tile_2s` is a 2-list;
  - else `tile`;
  - else None.
- **Counted units:** enemy units with a tile whose `card` is not in `TANK_CARDS` and not of card type `Spell`. Unknown cards (`card` None) count, at value 0.
- **`spell_target(card, lane, zone, state) -> tuple[list[float], float, int] | None`**, for damage spells (spells not in `NON_DAMAGE_SPELLS`):
  1. `r = SPELL_RADIUS.get(card, DEFAULT_SPELL_RADIUS)`, `lead = SPELL_LEAD_S.get(card, DEFAULT_SPELL_LEAD_S)`.
  2. `E` = the counted units whose **current** tile has `lane_of == lane` and `zone_of == zone`. If `E` is empty, return None.
  3. For each `e` in `E`, take `c = lead tile of e` and `members` = the counted units (any lane) whose lead tile is within `r` (Euclidean, in tiles) of `c`.
     - `value` = the sum of `elixir_cost` over the **distinct** known cards among `members`;
     - `count = len(members)`.
     - Keep the best `(value, count)`. On a tie, prefer the centre with the smaller row (closer to our side), then the first.
  4. Worth it if `value >= cost(card)` or `count >= SWARM_MIN_UNITS`; else return None.
  5. Return `(tile, value, count)`. `tile` = the mean of the members' lead tiles, clamped to `[0, ARENA_COLS - 1] × [0, ARENA_ROWS - 1]` and rounded to 0.1.

### Y2.3 Filtering
- Replace `_unblind`:
  - **damage spell candidate:** kept iff `spell_target(card, lane, zone, state)` is not None;
  - **non-damage spell:** the old rule (an enemy unit in its lane and zone);
  - **non-spell:** always kept.
- `no_blind_spells` and every mode use the new rule.
- **defend:** a damage spell is kept only if, additionally, the threat unit (`decision.threat`) is a counted unit and its lead tile lies within `r` of the spell's target tile. Spells on a tank threat, or on a cluster away from the threat, are dropped.
- Keep the order of the generator's output among survivors, and keep the existing fallbacks of each mode (KG, then the spell-filtered list). Those fallbacks now use the new spell rule.

### Y2.4 `target_tile`
- `ReactivePolicy.target_tile(key, state) -> list[float] | None`:
  - a damage-spell `card|lane|zone` key → the tile from `spell_target`, or None when there is no target;
  - every other key → None (the loop then uses the pro placement prior).
- `LearnedActPolicy.target_tile` delegates to its rules.

### Y2.5 Tests (`tests/test_policy.py`; hand-made states, real `load_cards()`)
1. Fireball vs. a lone giant in its lane and zone → dropped. Giant plus a musketeer 1 tile apart → kept, and `target_tile` is at the musketeer (the giant is not a member).
2. Zap vs. two bats (card `bats`, value 2 ≥ 2) → kept, aimed at their mean. A single skeleton (value 1 < 2, count 1) → dropped. Three unknown units → kept (swarm).
3. Lead: a fireball target unit with `predicted_tile_2s` 2 tiles ahead → the tile is shifted by 1.0 tile. For zap (lead 0.3), it is shifted by 0.3.
4. defend: threat at our tower plus a cluster in the enemy half → `zap|middle|enemy_half` is dropped. Threat cluster at the tower → the zap in the threat's lane and zone is kept, aimed at the threat.
5. A non-damage spell (rage) keeps the old lane+zone rule.
6. `LearnedActPolicy.target_tile` delegates. In learned mode, filtering uses the new rule.
7. Update existing tests that relied on the old "any enemy unit in lane + zone" spell rule only where the new rule rightly changes the outcome. Say which ones in the report.

### Y2.6 Smoke
`$PY tools/eval/eval_act_policy.py --out runs/smoke/Y2/act_policy.json` (offline; `decide` is unchanged, so the table should match the MAC-W numbers), then the full suite. Do not load models, and do not touch the screen.

## Report
Per MAC-00 §5, per package.
