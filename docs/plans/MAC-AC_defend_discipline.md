# MAC-AC — defend discipline and own-elixir accounting (FIX_LIST #32–#34; specs for two Sonnet implementers)

Author: Opus · 2026-11-05 · From live match A (`runs/poc/20261105-193555`, LLM + lookup, lost in about 1.5 min). Read `docs/plans/MAC-00_common.md` first.
- D-045 holds: the LLM still makes every play. These packages fix **when** it is consulted and **which options** it gets, and the elixir it believes it has.
- Two packages run in parallel: **C1** (`clashbot/policy.py`) and **C2** (`clashbot/live_state.py`).
- Four other agents are running (MAC-AB A1–A4). Do not touch `clashbot/live.py`, `clashbot/tactics.py`, `clashbot/act_model.py`, `clashbot/arbiter.py` or `cr_perception/`.
- CPU only; never load a model; never touch the screen.

## Evidence
- **53 of 56 decisions were `defend`.** Any enemy unit on our half was a threat: a battle-healer with ETA 20.6 s, a retreating giant, an `unknown_unit` with ETA 8.6 s.
- **Defend played the cheapest affordable card about every second.** That was 26 ice-spirit taps, so elixir never rose above about 3, and bomb tower (vs giant) and magic archer (vs balloon) were never affordable.
- **42 taps were sent, worth 62 elixir, but at most about 38 was available.** Many taps did not land: the elixir reading lags our own spend (4.3 fps under CPU load) and was never corrected.

---
## C1 — defend discipline (`clashbot/policy.py`, `tests/test_policy.py`)

### Owns
```
clashbot/policy.py   tests/test_policy.py
```
Keep `decide(state, t, context_text=None)`, `filter_candidates`, `no_blind_spells`, `target_tile` and `ActDecision` compatible. `ActDecision` may gain fields with defaults; `LearnedActPolicy` and A1's `FastActPolicy` delegate to these rules.

### 1. Threats (`_threats`)
- Drop units whose `heading` contains `"retreat"`.
- Drop `unknown_unit` (no card) unless ETA ≤ 4 s.
- Drop units whose card is a Spell, except `graveyard`.
- A unit on our half without an ETA counts only if row ≤ 8. Otherwise a unit counts only by ETA ≤ `defend_eta_s`.
- The ETA rule (≤ `defend_eta_s` = 4.0) is unchanged.
- **Add a watch list.** Units with an ETA ≤ 10 s that are not yet threats are returned separately by `_watch(state)`. Rule 2b uses them.

### 2. Defend uses the tactical engine to save elixir for an adequate answer
- `ReactivePolicy(..., engine=None, save_margin_s=1.0, emergency_eta_s=1.5, adequate_margin=0.15)`. With `engine=None`, build a `clashbot.tactics.TacticalEngine()` lazily on first use, and pass the same `cards` to it.
- **2a. Threat present.** When `decide` finds threats (rule 1), compute:
  - `now = engine.analyse(state)`;
  - `full = engine.analyse(<state copy with own.elixir = 10>)`, which scores every known hand card as if affordable;
  - for the push P containing the most urgent threat (`push.units`, matched by lane): `best_now` = the best defend move for P in `now.moves`, and `best_any` = the same in `full.moves`.
  - `need_s(card)` = max(0, cost − elixir) × `seconds_per_elixir(phase)`, where single is 2.8, double 1.4, triple/overtime 0.93, and unknown counts as single.

  Rules, first that applies:
  1. `best_now` and `best_now.score ≥ best_any.score − adequate_margin` → **act** with mode `defend`, reason `defend {threat} with an adequate answer`.
     - Set `decision.allowed_cards` = the cards of the moves in `now.moves` for P with `score ≥ best_now.score − adequate_margin`.
  2. `best_any`, and `need_s(best_any.card) + save_margin_s ≤ P.eta_s` → **hold**, mode `save`, reason `save for {card} vs {threat}: {need:.1f}s, eta {eta:.1f}s`.
  3. `P.eta_s ≤ emergency_eta_s` and `best_now` → **act**, `defend` (emergency), with `allowed_cards = {best_now.card}`.
  4. `best_now` → act, `defend`, with `allowed_cards = {best_now.card}`.
  5. Otherwise → **hold**, mode `save`, reason `no answer in hand for {threat}`.

  If the engine raises an error or returns no pushes, fall back to today's defend rule, so behaviour never gets worse than now.
- **2b. No threat, but watch-list units.** The leak and punish rules still apply. A punish additionally requires that `own elixir − cost of the cheapest adequate answer to the watched push ≥ 0`, i.e. it never spends the elixir needed for the next defence.
- `ActDecision` gains `allowed_cards: frozenset[str] | None = None`, and the mode vocabulary gains `"save"`.

### 3. Candidates
- `filter_candidates` in defend mode: when `decision.allowed_cards` is set, keep only candidates of those cards, and apply that before the existing lane/zone rules.
  - Tactical candidates (`tac` in sources) are still exempt from lane and zone, but not from `allowed_cards`.
  - If nothing survives, return `[]`; the loop then records a `hold_policy`.
- **The LLM still picks among the survivors (D-045).**

### Tests (fake engine + one real-engine case)
1. A retreating giant, an unknown_unit with ETA 8 s, and a battle-healer with ETA 20 s on our half → no threats.
2. Giant, ETA 9.5 s, elixir 2, hand ice-spirit / bomb-tower / ... (fake engine: best_any = bomb-tower 0.9, best_now = ice-spirit 0.5) → hold, mode `save`, with reason "save for bomb-tower". The same case at ETA 1.2 s → defend (emergency) with allowed {ice-spirit}.
3. Elixir 5, bomb-tower affordable → defend, allowed {bomb-tower} (plus anything within the margin); `filter_candidates` drops ice-spirit candidates.
4. Bats at ETA 2 s, zap affordable and adequate → defend.
5. The engine raises → today's behaviour.
6. A punish blocked by the reserve rule.
7. **Real engine:** giant at row 13 heading right, ETA 9 s, elixir 2, hand ice-spirit, bomb-tower, fire-spirit, magic-archer → save (not act).

Run `tests/test_policy.py`, `tests/test_act_head.py`, `tests/test_live_loop.py`, `tests/test_tactics*.py`, then the full suite (`--deselect tests/test_perception_latency.py`).

---
## C2 — own-elixir accounting and failed-tap detection (`clashbot/live_state.py`, `tests/test_live_state.py`)

### Owns
```
clashbot/live_state.py   tests/test_live_state.py
```
This is tracker mode only (`LiveConfig.hand_tracker=True`); the default output must stay byte-identical.

### Behaviour
1. `note_own_play(card, slot, t)` also records `(t, card, slot, cost, elixir_at_play)`. `elixir_at_play` is the last elixir value `build()` returned.
2. **Elixir.** In `build()`, for each recorded play younger than `confirm_s = 2.5` s:
   - `expected = elixir_at_play − cost + regen(phase) × (t − t_play)`, clamped to [0, 10];
   - `regen` = 1 / `seconds_per_elixir` (single 2.8, double 1.4, triple 0.93);
   - while `t − t_play < stale_s = 1.2`, the returned own elixir is `min(read, expected)`, since the HUD reading lags our tap.
3. **Failed tap.** Run this check once per play. If at `t − t_play ≥ stale_s` the reading is still `≥ elixir_at_play − 0.5` (and the reading is not None), the tap did not land:
   - call a new `HandTracker.mark_failed(slot, card)`, which puts `card` back in `slot`, removes its last log entry and clears pending;
   - put `card` on a **cooldown** of `cooldown_s = 2.0` s, during which it is shown as `None` in the returned hand, so the loop does not re-tap it at once;
   - count it in `self.failed_taps`.
4. A play is confirmed (removed from the pending list) once the reading drops by ≥ `cost − 0.5`, or the slot reads a different card.
5. Expose `failed_taps` and `confirmed_taps` counters. The loop may log them later; no `live.py` change now.

### Tests
- **Stale reading:** a play at elixir 6 (cost 4), with reads of 6 at +0.3 s and +0.8 s → own elixir is about 2.1 / 2.3, not 6.
- **Fresh reading:** a read of 2 at +0.6 s → confirmed, no further correction.
- **Failed tap:** reads stay at 6 at +1.3 s → `mark_failed`; the card is back in its slot, but hidden for 2 s; `failed_taps == 1`.
- **Phases:** in double elixir the regen is 1/1.4.
- **No tracker:** the output is byte-identical.

Run `tests/test_live_state.py` and `tests/test_live_loop.py`, then the full suite.

---
## Report
Per MAC-00 §5, per package.
