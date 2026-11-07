# MAC-AL — timing (honest ETAs, big-threat budget), fireball by clusters, balloon defence, hero ability used sparingly (specs for two Sonnet implementers)

Author: Opus · 2026-11-07 · From the recorded run `runs/poc/20261106-233907` (4 full Training Camp games and part of a fifth; d1-ft-joint-tac, policy fast, all K4b code). Read `docs/plans/MAC-00_common.md` first. D-045 holds: the LLM picks every move; this changes **when** it is asked, **which** cards it may choose from, and the engine's numbers. D-049 (hero ability used sparingly) is new.

User, after the run:
1. *"A lack of fireball … several enemy units clumped together for a perfect fireball but it will place a unit rather than fireball them."*
2. *"Not understanding how to counter balloon."*
3. *"Not understanding how to properly use the musketeer hero ability, honestly this should be used very sparingly."*
4. *"I had a hunch about 'panic plays' and I feel this could be also a bigger issue."*

## Evidence (all from `decisions.jsonl` / `shots/` of the run)
**Timing (panic plays).**
- 72 defend acts:
  - 39 % came when the engine's push ETA was ≤ 1.5 s; 22 % at 0.0 s (the push was already at the tower);
  - 15 % were labelled `emergency`; 11 % `nothing adequate in hand`;
  - the median act came 5.1 s after the opponent played a card of that push.
- **Giant / balloon:** the opponent played one 16 times. Our answer came 7–11 s after the play, i.e. when it was at the tower. The plays in between went to bats, archers or e-wizard. Two were full-bar leaks right after a giant (t = 123.4: ram-rider) or a balloon (t = 367.5: knight at the middle bridge).
- **The tracker ETA is biased late** (`cr_perception/tracking.py` `eta_to_nearest_enemy_tower`):
  - melee units and flyers: the tracker's speed is 0.72× the card's speed (median; p25 0.50, p75 0.85), so the ETA is +1.3 s late;
  - ranged units: it measures to the tower edge, not to attack range, so archers / e-wizard / firecracker are +5.3 s late;
  - units at the tower that read `stationary` / `retreating` get no ETA (`speed < 0.1 → None`). Example: at t = 580.8 the balloon is at [14, 7], on our right tower, while the threats line says "tower in 6.2 s".
- **The engine prefers the tracker ETA.** `TacticalEngine` sets `push.eta_s = min(tracker ETAs)`. The simulation's ETA (`SimResult.eta_s`: real speed and range, "first unit in reach of the tower") is used only for units without one.
- **Latency:** decision-to-tap is 336 / 506 ms (p50 / p95), of which 183 ms is the executor drag. Positions update every ~0.4 s (detection every 3rd frame, 7.95 fps).

**Fireball.**
- In hand in 104 of 116 decisions. Mentioned by the engine 12 times, among the ranker's candidates 14 times, played 3 times.
- The causes in the code:
  - `policy._window_s`: a spell's deploy window is `BURST_WINDOW_S` = 2.5 s, and the ETA it compares against is late, so a clump is spread or at the tower before fireball is allowed;
  - `tactics._spell_aim` looks only in the cell of the push's lead unit, so support troops behind a tank are never a fireball centre;
  - `policy._aims_at_threat` (the defend filter in `_unblind`) needs the push's lead to be a counted unit. Giant-led pushes can never be fireballed;
  - the allowed fireball never reaches the LLM. At t = 261.5 the reason is `defend bats left: fireball (spell)` and the engine's best move is `fireball @[4,18] … saves ~500`, but 0 fireball candidates remain after `filter_candidates`. The LLM picked electro-dragon. The same at t = 750.8 / 751.9: the engine's best is fireball (9.4 elixir value), the policy allowed knight only (the spell window was closed).

**Balloon.**
- t = 367.5: balloon + firecracker at our left bridge, elixir 10, musketeer in hand. The engine's best is `musketeer @[5,2.5] … saves ~1,600`. The policy said `leak` (the ETA 6.7 s was outside the 6 s window). The LLM took `knight|middle|bridge` (0.55) over the musketeer (0.45).
- t = 492.0: `defend balloon left: knight (emergency)`. A knight cannot hit air: the emergency rule takes the best move of any kind.
- t = 574.5 → 580.8: the balloon was played at the enemy bridge [16, 17]. The bot fireballed an e-wizard in the left lane (4 elixir), then had no elixir when the balloon reached the right tower. Tesla then came in `emergency` ("slows balloon": locked on, no pull).
- Four games-start `learned` plays put a tesla in the middle with no enemy unit on the field (t = 95.2, 109.2, 437.3, 833.1). 4 elixir each, expired unused.

**Hero ability.**
- 6 taps. 4 of them (t = 134.7, 596.2, 715.1, 779.1) were not proposed by the engine, and the policy's reason named another card, e.g. `defend giant left: knight (emergency)`.
- The cause: `live.py` adds `hero_ability_move(state)` as an option whenever the button is usable and our Musketeer is on the field. Ability candidates bypass `filter_candidates`. `policy._judge` also adds the ability to `allowed` "always". The LLM gives the option 0.39–0.85.
- The taps do fire. `review_shots` marks 4 of 6 `failed` because at 2×/3× elixir the bar refills before +0.5 s. The screenshots show the bar at 0 and the button greyed. The good one: t = 407.9, turret vs balloon right, "saves ~1,600".

---
## Package AL-T — honest ETAs in the tracker (one Sonnet implementer)

### Owns
```
cr_perception/tracking.py   cr_perception/perception.py (only: pass range priors to UnitTracker)
tools/eval/eval_eta.py (new)   tools/eval/review_shots.py (ability verdict only)
tests/test_tracking.py   tests/test_eta_eval.py (new)   tests for review_shots (new file if none exists)
```
Do not touch `clashbot/`. Package AL-P owns it and runs at the same time.

### T1. ETA to attack range at the card's real speed (`tracking.py`)
- **Range priors.** `load_range_priors(path=<repo>/clashbot/resources/unit_stats.json) -> dict[str, float]`: per card slug, the attack range in tiles of its first unit (`units[0].range_tiles`, 0.0 when missing). Key the dict exactly like `load_speed_priors` so a track's `cls` finds both. `UnitTracker(range_priors=...)` passes them to its tracks. `Perception` loads them next to the speed priors; a missing file gives `{}` (old behaviour).
- **`KalmanTrack.eta_to_nearest_enemy_tower()`**:
  - `d = hypot(...) - 1.5 - range_tiles(cls)` (`TOWER_EDGE = 1.5`, unchanged), floored at 0;
  - **`d == 0` → ETA 0.0 whatever the speed** (a unit in reach of a tower is at it, even when it reads stationary);
  - speed: if the track moves toward that tower (the velocity's cosine to the tower direction ≥ 0.5) and a speed prior exists, use `max(kalman_speed, 0.9 × prior)`. Units walk at their card's speed in this game; the filter lags. Otherwise use the Kalman speed, as today;
  - `speed < 0.1` and `d > 0` → None, as today.
- `summary()` keeps its keys. The `eta_tower.s` value changes. Add `eta_tower.range` (the range used) for debugging.
- Also expose a pure function `eta_to_tower(pos, vel, side, speed_prior, range_prior) -> (tower, s) | None` and use it inside the method. The training data builder can recompute ETAs with the same rule later (parity, FIX_LIST #50).

### T2. Measure it (`tools/eval/eval_eta.py`)
- **Command:** `eval_eta.py --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60 --seconds 240 [--device mps] --out runs/smoke/AL/eta.json`.
- **What it does:** runs `cr_perception` on the video exactly as `clashbot.live --source video` does (same fps / detect-every defaults; reuse the live video source and Perception classes, no new perception code). Per frame, record each enemy track's `(t, cls, pos, eta_tower)` with the old and the new rule.
- **Ground truth:** the first time a track is within `1.5 + range` of an own tower (or its ETA is 0 by the new rule).
- **Error:** for every earlier frame of that track with an ETA, `predicted − (t_arrival − t)`. Report the median / p25 / p75 / |err| ≤ 1 s share, old vs new, split into melee, ranged and flyer.
- **Acceptance:** new median |error| ≤ 1.0 s for melee/flyer and ≤ 1.5 s for ranged, each better than old. Report the numbers either way.
- The YOLO model may run on MPS; no other GPU job is scheduled.

### T3. `review_shots.py`: ability verdict
- An `ability:*` tap is `landed` when the elixir drop by +0.5 s or +1.5 s is ≥ `cost − 1.5`. At 2× / 3× elixir the bar refills ~0.7–1.1 elixir/s.
- Or, if the shot index records the ability button state after the tap, when it reads not-ready.
- Re-run on `runs/poc/20261106-233907` and report the new ability verdicts. The screenshots of t = 134.7 and 407.9 show both fired.

### Tests
- **ETA:**
  - a ranged track (range 5) 7 tiles from the tower edge moving at the prior → ETA = 2/speed;
  - a melee track whose Kalman speed is 0.6× the prior, moving toward the tower → ETA from 0.9 × prior;
  - a stationary track at the tower → 0.0;
  - a track moving away → the old rule;
  - no priors → identical to the old output.
- **eval_eta:** on a synthetic track list with a known arrival, the errors are computed right.
- **review_shots:** 3 → 1 at +0.5 s for an ability tap is `landed`; 3 → 3 is `failed`.

Run the full suite (`--deselect tests/test_perception_latency.py`).

---
## Package AL-P — engine and policy: timing, big-threat budget, fireball, balloon, ability (one Sonnet implementer)

### Owns
```
clashbot/tactics.py   clashbot/policy.py   clashbot/act_model.py (mode pass-through only)
clashbot/live.py (only: the ability offering, ABILITY_* timing, the hold_fallback mode list)   clashbot/actions.py (ability constants)
tests/test_tactics.py   tests/test_policy.py   tests/test_live_loop.py   tests/test_act_model.py
```
- Do not touch `cr_perception/`. AL-T changes the tracker ETAs at the same time; your tests feed `eta_tower` values directly.
- Pin engine weights in tests (`DEFAULT_WEIGHTS`).
- Keep public APIs compatible; new dataclass fields get defaults.

### P1. Timing
1. **Push ETA from the simulation.** In `TacticalEngine` (the block that sets `push.eta_s`):
   - when `push.sim_eta_s` is not None, `push.eta_s = min(push.sim_eta_s, min(tracker ETAs))`;
   - otherwise keep today's rule.

   The simulation uses the cards' real speeds and ranges, and the tracker's ETA is late (Evidence). Add `push.eta_source: "sim" | "tracker"` (default `"tracker"`). The push line keeps `tower in Xs` with the value used.
2. **Latency.** In `policy._judge`, compare the windows against `push.eta_s − DECISION_LATENCY_S`, with `DECISION_LATENCY_S = 0.5` (measured p50–p95 decision-to-tap: 0.34–0.51 s). The geometric building window is unchanged.
3. **Big threats are budgeted from the moment they are played.**
   - **Early pushes.** A building targeter or a flying unit of the opponent anywhere on the field, moving toward our side or played in the last 3 s (`opponent.recent_plays`), is a push. If it is beyond today's rows, mark it `early = True`. Its ETA is the simulation's. This extends today's river/bridge rule.
   - **The big push** is the push with the largest `damage_if_unanswered`, if ≥ `BIG_PUSH_DAMAGE = 1500`. Compute it in `_plan` **before** the per-push loop, with its reserve: the cost of the cheapest adequate answer within `adequate_margin` of the best in its `full` moves (`_reserve`).
   - **Every other push P** that would act, unless P alone destroys a tower (`destroys_with_allies`, else `destroys_tower`), goes through `_keep_reserve` against the big push. This applies whatever the ranking order, while the big push's ETA ≤ `RESERVE_WINDOW_S`, which rises 10 → 14 s (a balloon from the enemy back row needs ~13 s).
   - The hold reason: `keep {X} elixir for the {card} ({damage} if unanswered), not for {P}`.
4. **Full bar: pre-place, don't leak.** In `decide`, before the leak branch, when `leak_due`:
   - if an unanswered, not `tower_ok` push P has an affordable adequate non-building move whose window is not open yet, act `defend`;
   - `allowed` = those moves within `adequate_margin` of the best; engine tiles / sources as `_judge` sets them;
   - reason `full bar: pre-place {card} vs {threat} {lane}`;
   - the biggest such push first. Only when there is none, today's leak.
5. **The window for air answers.** Against a push led by a flyer (or with a flying building targeter) whose `damage_if_unanswered ≥ 1000`, a ranged troop that hits air has window `AIR_WINDOW_S = 9.0` (was 6.0). It deploys behind the tower and must stand there when the balloon arrives. Buildings keep the geometric rule.

### P2. Balloon specifics
1. **Emergency / nothing-adequate.** In `_judge`, `best_now` for the `emergency` and `nothing adequate in hand` rules is taken from the moves that are **not** `_air_blocked(card, push)`. If none is left, hold with `save`, reason `no air answer at hand vs {what}`. Never a knight vs a balloon (t = 492.0).
2. **No blind buildings.** A defensive building (card type Building, not a win condition) with no enemy push on the field is dropped:
   - `learned`: `no_blind_spells`;
   - `leak` (non-win-condition) and `punish`: `filter_candidates`.

   Tesla at game start expires unused (t = 95.2, 109.2, 437.3, 833.1).
3. **The air reserve** (`air_reserve` / `_keeps_air_reserve`, K4b) also applies to `counter` decisions and to the generic (non-win-condition) leak. A win-condition leak at 10 elixir is allowed if the elixir left after it covers the reserve.

### P3. Fireball (and zap) by clusters
1. **Engine: the best cluster anywhere** (`tactics.py`, replacing `_spell_aim`'s lead-cell search for damage spells):
   - for each damage spell in hand (Spell type, not in `NON_DAMAGE_SPELLS`), the centres are every enemy unit's lead tile (`SPELL_LEAD_S`), on any half and in any lane, except spells;
   - members are the enemy units within `SPELL_RADIUS` of the centre at their lead tiles;
   - the value is the existing kill-based helper (MAC-AH 3f) over all members. Tanks count as damaged survivors but are never a centre alone (`_counted` stays the centre rule);
   - **+1 elixir** when the circle overlaps an enemy princess tower (crown-tower damage), with reason clause `+ tower`;
   - keep the best centre per spell (highest value; ties → more kills).

   Add `policy.spell_best(card, state) -> (tile, value, members, kills) | None` next to `spell_target`. Same value code; no second value function.
2. **Spell moves.**
   - A spell's best cluster that contains **any** unit of push P becomes P's defend move, with the simulation run on P minus the killed members → `prevented` / `adequate` like other moves.
   - Otherwise it is a move with the new intent `spell_value`.
   - Reason, ≤ 60 chars: `fireball night-witch + e-wizard + bats x3: kills 4, ~9 elixir`.
3. **Policy windows.** A damage spell whose engine move has `value ≥ cost` is in window whenever its target exists (`_window_s` returns `inf` for it). The aim already leads the units by the spell's flight time. Kamikaze cards keep `BURST_WINDOW_S`.
4. **Defend filter.** `ActDecision` gains `push_units: tuple = ()`, which `_judge` sets from `push.units`. `_aims_at_threat` accepts a spell whose target circle contains any unit of `push_units`. With an empty tuple, today's rule applies.
5. **Great spell first.** In `_judge`, if an open adequate damage-spell move has `value ≥ GREAT_SPELL_RATIO × cost` (`GREAT_SPELL_RATIO = 2.0`), then:
   - `allowed = {that spell}`;
   - reason `defend {what}: {spell} hits {n} units (~{value:g} elixir)`;
   - the LLM picks among that spell's placements.
6. **Spell window outside pushes.** In `decide`, after the push plan acted or held, and before the counter:
   - if the engine's best `spell_value` move has `value ≥ GREAT_SPELL_RATIO × cost`, the elixir covers cost + any reserve (P1.3), and it keeps the air reserve;
   - then act with mode **`spell`**: `allowed = {spell}`, engine tile, `tile_sources = {spell: "spell"}`, reason `spell: {engine reason}`.
   - `LearnedActPolicy.decide` and act_model `FastActPolicy` pass `spell` through like `counter`.
   - `filter_candidates` treats `spell` like `counter`.
   - In live, `hold_fallback` exempts `spell` like the other act modes.
7. **The allowed card always reaches the LLM.** In `filter_candidates`, for `defend` / `counter` / `leak` / `spell`: every allowed card with an engine tile but no candidate left gets the engine candidate (`_engine_candidates`, restricted to the missing cards, appended after the kept ones). This fixes t = 261.5.

### P4. Hero ability used sparingly (D-049)
1. **Live** (`live.py`):
   - remove the part-1 offering (`hero_ability_move(state)` when the engine proposed nothing). The ability is an option **only** when the engine proposed it **and** `ABILITY_KEY` is in `verdict.allowed_cards`;
   - `ABILITY_MIN_GAP_S = 30.0` game seconds since the last ability tap; this replaces `ABILITY_RETRY_S` = 10 as the guard (keep the cooling state for 10 s).
2. **Engine.** `ability_min_prevent` 400 → **1000**. Remove the "makes the push adequate" path. A proposal needs `prevented ≥ ability_min_prevent`.
3. **Policy.** Replace "the ability is always on offer". `ABILITY_KEY` joins `allowed` only if:
   - no adequate affordable **card** move for that push has its window open; or
   - the ability's `prevented` ≥ 1.5 × the best open card move's `prevented`.

   The emergency and nothing-adequate rules never add it.
4. **The option note** says the cost and the cooldown: `hero ability: {engine reason}; 3 elixir, use rarely`.
5. Expected effect on this run: ≤ 1 use per game, as at t = 407.9 (balloon at the right tower, Musketeer behind it, no air card affordable).

### Tests
Real engine, `DEFAULT_WEIGHTS`, deck D-047 (electro-dragon, musketeer, tesla, fireball, knight, zap, ram-rider, mini-p-e-k-k-a).
1. **Timing.**
   - A push with tracker ETA 6 s and simulation ETA 3 s → `eta_s` 3, `eta_source` sim;
   - a ranged-troop window opens at `eta_s − 0.5 ≤ 6`.
2. **Big threat.**
   - Balloon at [16, 22] (enemy half, moving down, just played); left lane bats ×2 at ETA 2 s (tower handles or not). Elixir 5, hand musketeer, zap, knight, fireball:
     - the balloon is an early push;
     - no card that leaves < 4 elixir is spent on the bats;
     - reason `keep 4 elixir for the balloon …`.
   - The same with our left tower at 300 HP (the bats destroy it) → the bats may be answered.
3. **Full-bar pre-place (t = 367.5).** Balloon [1.5, 16] + firecracker [2.5, 16.5] at the left bridge, ETA ~7 s; elixir 10; hand zap, fireball, musketeer, knight → act `defend`, allowed ∋ musketeer at the engine tile behind the left tower; knight not allowed.
4. **Air window.** Balloon at the river, ETA 8.5 s, musketeer affordable → defend musketeer.
5. **Air-blocked emergency (t = 492.0).** Balloon at [3, 8] (ETA 0), giant [4, 7]; hand fireball, ram-rider, knight; elixir 3 → no knight; hold `no air answer at hand`, or fireball only if its value rule allows it.
6. **Blind building.** No enemy units; `learned` act; hand tesla, knight, … → no tesla candidate.
7. **Fireball cluster.**
   - Giant lead at [4, 12] ETA 5 s; night-witch [4, 15.5], e-wizard [5, 16], bats ×3 around [4, 15] →
     - the engine has a fireball defend move whose members include night-witch, e-wizard and bats;
     - the policy allows fireball (great spell → allowed = {fireball});
     - a fireball candidate is in `filter_candidates` output, with its tile within 1 tile of the cluster.
   - The same cluster on their side ([4, 19–21], no push yet) → mode `spell`.
   - **Regression:** zap vs a lone L15 balloon and fireball vs a lone giant are not proposed.
8. **Injection (t = 261.5).** Allowed {fireball, electro-dragon} and the generator's candidates have no fireball key → the output has the engine's fireball candidate.
9. **Ability.**
   - Not offered when the engine proposed nothing (Musketeer on the field, button ready, a bats push);
   - offered when the engine proposes it (balloon at the right tower, Musketeer behind it, no air card affordable);
   - not offered again within 30 s;
   - never in an emergency for another card.
10. **Pass-through.** `LearnedActPolicy` / `FastActPolicy` pass `spell`; live does not `hold_fallback` a `spell` decision.
11. The existing MAC-AD / AG / AH / AK tests pass, or are updated with a stated reason (window values, the ability rule).

Offline:
- `tools/eval/eval_act_policy.py --out runs/smoke/AL/act_policy.json`: the mode counts incl. `spell` and the act rate vs `docs/data/act_policy_v1.json`;
- `tools/eval/eval_tactics.py`: no regression in its headline numbers. Report them.

Run the full suite (`--deselect tests/test_perception_latency.py`). CPU only; never touch the screen.

### Smoke (after the suite passes)
A 120 s video dry run (no `--live`):
```
.venv/bin/python -m clashbot.live --dry-run --source video --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60 \
  --deck electro-dragon,musketeer,tesla,fireball,knight,zap,ram-rider,mini-p-e-k-k-a --hero-cards musketeer \
  --ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac --policy fast --act-model db/models/act-gbdt \
  --record-shots --seconds 120 --out runs/smoke/AL_dry
```
Report the decision modes, `spell` count, ability offers / taps, shot errors.

## Report
Per MAC-00 §5, per package, with:
- AL-T: the ETA error table, old vs new;
- AL-P: the reasons produced for tests 2–9 and the eval tables.
