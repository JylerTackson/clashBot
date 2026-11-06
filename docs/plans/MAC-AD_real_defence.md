# MAC-AD — real defence: deploy windows, adequate = stops the push, count our own answers (spec for one Sonnet implementer)

Author: Opus · 2026-11-05 · From live run `runs/poc/20261105-205934` (1½ Training Camp games, all MAC-AB/AC fixes in). User: *"The bot is not defending the giant or the balloon at all, it is just letting them get to the tower."*
- Read `docs/plans/MAC-00_common.md` first.
- D-045 holds: the LLM picks every play; this fixes **when** it is asked and **which cards** it may choose from.
- One package, because the three parts depend on each other.
- CPU only; never load a model; never touch the screen.
- A GPU training job is running (`train_d1.py`); do not use MPS.

## Evidence (game 1, t = 154–181 s)
- Bomb tower vs the giant was placed correctly at t = 154 ("pulls giant to the centre"). In the next 27 s the bot then tapped 11 ice/fire spirits ("slows giant", "slows balloon").
- The rule found the spirit "adequate": `best_any` was also weak, since magic archer has low dps vs a tank.
- Our own fresh bomb tower did not count as answering the push. It sits at the pull tile, more than 3 tiles from the giant, and our taps reach `own.recent_plays` only once perception sees them.
- Threats count only from ETA ≤ 4 s, so a building is placed too late to pull: the giant is already at the tower. The balloon reached the tower.
- Taps cost 35 elixir with about 27 available, because of repeated spirit taps.

## Owns
```
clashbot/policy.py   clashbot/tactics.py   clashbot/live_state.py
clashbot/live.py     (one line: pass the tile to note_own_play)
tests/test_policy.py   tests/test_tactics.py   tests/test_live_state.py   tests/test_live_loop.py (update only as needed)
```
- The public APIs stay compatible. New dataclass fields get defaults.
- `tools/data/build_d1_tac.py` output may change slightly because engine lines change. That is fine; Opus rebuilds it.

## 1. Our own taps are known at once (`live_state.py`, tracker mode only)
- `note_own_play(card, slot, t, tile=None)` also inserts a synthetic own play event: `{"player": "own", "card": card, "tile": tile, "timestamp": t, "detect_source": "executor"}`. It then appears in `state.own.recent_plays` immediately.
- **Dedupe.** In `observe_events`, an own event of the same card within ±3 s of an executor event is merged into it, not added: fill a missing tile, keep the executor timestamp.
- **Failed tap.** When C2 marks a tap as failed (`mark_failed`), remove its synthetic event.
- `live.py`: `note_play(card, entry["slot"], t, tile=tile)`.
- The default (non-tracker) builder output stays byte-identical.

## 2. Engine: stopping class on moves; a push answered by our recent plays and buildings (`tactics.py`)
- `TacticalMove` gains `stop: str = "n/a"`, one of `"high" | "medium" | "low" | "spell" | "n/a"`. This is the card's stopping class against that push, the same class the A2 stopping term uses, including the air-defense rule. Set it on every defend move.
- `Push` gains `answered: bool = False` and `answered_by: list[str] = []`.
- A push is **answered** when the distinct-card elixir of the following is ≥ `w.answered_frac` × the push's elixir:
  - (a) ally units within 3 tiles of the lead (as today);
  - (b) ally **buildings** (card type Building or role `building`) on our half within 7 tiles of the lead, if the push has a `building_targeter`;
  - (c) **own recent plays** (`own.recent_plays` with `t ≥ now − 4.0` s) whose tile is on our half, and either in the push's lane or within 7 tiles of the lead. Cards of type Spell count only if their tile is within 3 tiles of the lead.
- Answered pushes keep today's score reduction. The push line keeps `(being answered by …)`.

## 3. Policy: per-push deploy windows and adequacy (`policy.py`, replaces the C1 defend core when the engine works)
- **Pushes.** Use the engine's pushes, not only units with ETA ≤ 4 s. For each push P (most urgent first), take `full` (as in C1: the state with elixir 10) and `now`.
- **Adequate moves.** A defend move M for P is adequate if:
  - P has `medium` or `tank` units → `M.stop ∈ {high, medium}`, or `M.stop == "spell"` and the spell's value ≥ its cost (Y2 rule);
  - P has only swarm or light units → any stop.
- **Deploy window** for an adequate M (seconds of `P.eta_s`), by role:
  - building vs a push with `building_targeter`: ≤ `building_window_s` = 9.0, **or** the push's lead unit is on our half (row ≤ 14);
  - ranged troop, or any adequate card vs a flying push: ≤ 6.0;
  - melee / mini_tank / tank_killer: ≤ 4.5;
  - kamikaze or spell: ≤ 2.5.
- **Rules per push P**, first that applies:
  1. `P.answered` and `P.eta_s > 1.0` → hold, mode `save`, reason `push {lane} answered by {answered_by}`. Then look at the next push.
  2. An affordable adequate M in `now.moves` whose window is open → **act**, `defend`.
     - `allowed_cards` = the cards of the affordable adequate moves for P within `adequate_margin` of the best.
     - Reason `defend {threat} {lane}: {card} ({stop})`.
  3. The best adequate M in `full.moves` is not affordable and `need_s(M) + save_margin_s ≤ P.eta_s` → **hold**, `save`, reason `save for {card} vs {threat}`.
  4. `P.eta_s ≤ emergency_eta_s` (1.5) → act with the best affordable move of any stop (allowed = that card), reason `emergency`.
  5. Otherwise → hold, `save`, reason `waiting for the {card} window` or `no adequate answer in hand`.
- **Low-stop moves** ("slows …") are allowed only in rule 4, or when the hand holds no adequate card for P even at elixir 10 and `P.eta_s ≤ 2.5`.
- **Commitment.** After an act against P (keyed by lane + lead card), remember `(t, card)`. Within 4 s, rule 1 also counts that card as answering P, even before it appears in `recent_plays`.
- **Fallback.** Keep the C1 fallback when the engine fails. Leak and punish stay as they are, after the pushes (a push in its window takes precedence).

## Tests
**Policy** (real engine and real roles where possible; pin the weights to `DEFAULT_WEIGHTS` or the tuned file explicitly):
1. A giant on the bridge in the right lane (row 15, ETA 9 s), elixir 5, hand bomb-tower / ice-spirit / magic-archer / berserker → act `defend`, allowed ⊇ {bomb-tower}, ice-spirit **not** allowed. The engine tile is at the pull position.
2. The same, 1 s after our bomb-tower tap (`note_own_play` with a tile at the pull position) → hold, "answered".
3. The same giant at ETA 9 s, elixir 2, bomb-tower in hand → save for bomb-tower. No spirit is tapped.
4. A balloon at ETA 6 s in the left lane, magic-archer affordable → defend with magic-archer. Fire/ice spirit is not allowed, and bomb-tower is not proposed (it cannot hit air).
5. A balloon at ETA 1.2 s with only ice-spirit affordable → emergency, ice-spirit.
6. Bats (swarm) at ETA 2 s with zap or fire-spirit affordable → defend (a low stop is fine vs swarm).
7. A push answered by an ally bomb-tower 5 tiles from the giant → answered.

**Live state:** an executor play appears in `recent_plays` at once; a later HUD event of the same card is merged; a failed tap removes it.

**Engine:** `stop` is set on defend moves; rules (b) and (c) of "answered".

**Offline:** `tools/eval/eval_act_policy.py --out runs/smoke/AD/act_policy.json`. Compare the modes and act rates against `docs/data/act_policy_v1.json` and the C1 numbers (test/all act rate 0.200).

Run the full suite with `--deselect tests/test_perception_latency.py`.

## Report
Per MAC-00 §5, with the eval table and the reasons produced for test situations 1–5.
