# MAC-AM — spells lead their targets, are cast only for value, and Tesla goes in the centre; log why the bot holds (spec for one Sonnet implementer)

Author: Opus · 2026-11-07 · From the recorded run `runs/poc/20261107-013441`: 3 Training Camp games with all MAC-AL code. **The first win**: game 3 in overtime. Game 1 took a tower; game 2 was a 3-crown loss. Read `docs/plans/MAC-00_common.md` first. D-045 holds: the LLM picks every move.

User, after the run:
1. *"It is not leading the fireballs correctly but it seems to be placing them correctly, like it needs to place them a few tiles in front to account for travel time."*
2. *"Only like 1 occasion … it launched at a single enemy that wasn't a good use of a fireball, that enemy was a fire cracker."*
3. *"The zaps it was using were not very good."*
4. *"The teslas it's placing are not very good, they should mainly be placed in the center of the arena to kite and they are placed really close to the towers."*

## Evidence
**Fireball lead.**
- `policy.SPELL_LEAD_S["fireball"] = 1.0` s, applied by `_lead_tile` as a fraction of the tracker's 2 s prediction. That prediction uses the Kalman velocity, about 0.72× the card's speed (MAC-AL evidence).
- The real delay is the decision-to-tap latency (0.5 s, `policy.DECISION_LATENCY_S`) plus the projectile's flight from our King Tower. The KB gives "Projectile Speed": fireball 600, rocket 350, arrows 1100, on the same scale as troop speed (60 = 1 tile/s), so fireball travels at 10 tiles/s. From the king tower to the bridge (~13 tiles) that is 1.3 s, and to their tower (~22 tiles) 2.2 s: 1.8–2.7 s in total, not 1.0.
- Shots `d3243` (t = 404.4): aimed at [15, 15] on night-witch + bats; at +1.5 s they had walked ~1.5–2 tiles further.

**Spell value.**
- t = 325.5: `fireball firecracker: kills 1, ~4 elixir + tower`, a lone firecracker at their back. It passed `value ≥ cost` only through the +1 tower bonus.
- 16 zaps in 3 games. Most were valued 2–2.4 elixir for a 2-elixir spell; three killed nothing (t = 164.6, 514.8, 518.2: `kills 0`); 8 were `emergency`.
- t = 514.8 (shots `d4099`): a zap at a lone firecracker; the game answered "Not enough Elixir!".

**Tesla.**
- 18 teslas, 72 elixir, the most-played card. Tiles: [12.5, 6.5] ×4, [5.5, 6.5] ×2, [7, 4] ×2, [8, 4], [10, 6.5], [11, 6.5], [7.5, 6], [5, 2.5] (a full-bar leak), [8, 9] (`learned`, no threat on the field), [7, 11] ×3, [11, 9.5].
- **The cause is `tactics._pull_tile`.** It ranks pull tiles over x 6..12 / y 4..11 by "the longest walk inside both the princess and the king tower range", which favours deep tiles near the king tower.
- **The non-targeter fallback** is `_clamp_tile(tower_x + 2·toward_centre, 6.5)` (`blocks X at the tower`), next to the tower.
- **When the pull happens is a second cause.** It is judged once the targeter has crossed the river, and the giant is often already deep, so the best tile is deep too.

**Holds are not logged.**
- The policy said `save` on 1,916 frames (previous run) and the reasons are lost: only acts and a few holds reach `decisions.jsonl`. We cannot see why the bot waited.
- When iPhone Mirroring closed, the run crashed (`window capture failed`) and wrote no `summary.json`.

## Owns
```
clashbot/policy.py   clashbot/tactics.py   clashbot/unit_stats.py (spell value only, if M2.3 needs it)
clashbot/live.py (only: holds log, graceful source failure)
tests/test_policy.py   tests/test_tactics.py   tests/test_unit_stats.py   tests/test_live_loop.py
```
Pin engine weights in tests (`DEFAULT_WEIGHTS`). Keep public APIs compatible; new fields get defaults.

## M1. Spells lead their targets by the real delay (`policy.py`; the engine uses the same helpers)
1. **`spell_delay_s(card, target_tile) -> float`** = `DECISION_LATENCY_S` + flight.
   - **Projectile spells:** read the KB card table's "Projectile Speed" once (as `tracking.load_speed_priors` does for troops: `knowledge_base/cards/<slug>.md`). Flight = distance from our King Tower tile to the target / (speed / 60) tiles/s. Use the King Tower tile the policy already uses (`OWN_TOWER_TILES` or the tracker's `own_king`; the same coordinate frame as unit tiles).
   - **Instant spells** (zap, and any spell with no projectile speed): flight 0.
2. **`lead_tile_s(unit, s, state)`**: the unit's tile moved `s` seconds along its velocity direction.
   - Speed: `max(kalman_speed, 0.9 × card_speed)` when it moves toward our side, as MAC-AL T1 does. Take the card speed from the same KB speed table, through a small loader in `policy.py` or `unit_stats` (`speed_tiles_s`), not from `cr_perception`.
   - A unit that reads `stationary`, or is within its attack range of one of our units or towers, does not move.
   - Clamp to the arena.
3. **Solve the circularity** (flight depends on the target, the target on the flight) with 2 iterations. Centres and members in `spell_best` / `spell_target` and the defend check `_aims_at_threat` use the lead tiles at `spell_delay_s(card, centre)`. `SPELL_LEAD_S` stays only as the fallback when no speed is known.
4. **The executed tile** (`target_tile`, `tile_source = "spell"`) is that centre.

## M2. A spell is cast only for value (`policy.py`, `tactics.py`)
1. **One rule.** `spell_worth(card, value, kills) -> bool`, used everywhere `value ≥ cost` is used today: `_adequate`, `_window_s`, `_unblind`, `spell_target` / `spell_best`, the engine's spell moves.
   - **Zap** (and any spell of cost ≤ 2): `kills ≥ 2` and `value ≥ cost + 0.5`.
   - **Other damage spells:** `value ≥ cost + 1.0` and `kills ≥ 1`; or `kills == 0` with `value ≥ cost + 2.0`.
   - The `+ tower` bonus counts in `value`.
   - The same bar in `emergency` and `nothing adequate` (no exception).
   - `GREAT_SPELL_RATIO` (2.0) is unchanged.
2. **Expected:**
   - t = 325.5 (lone firecracker + tower: 4) → not cast;
   - fireball on archers + firecracker (t = 519.2, `kills 2, ~7.5`) → cast;
   - zap `kills 0` → never;
   - zap on 2+ bats → yes.
3. **Check swarm valuation** (`unit_stats.spell_value` / the policy's members):
   - killing every detected unit of one played swarm card (bats ×5, skeletons, minions ×3) must not exceed that card's elixir;
   - one bat of five is ≈ 0.4;
   - the reason `zap bats: kills 1, ~2 elixir` (t = 508.3) suggests a single detected bat is priced as the whole card.

   Fix it if so, with a test. Report what you found.

## M3. Tesla (and any defensive building) in the centre (`tactics.py`, `policy.py`)
1. **The centre box.**
   - `CENTER_BOX = x 7.0..11.0, y 8.5..11.5` (0.5 grid);
   - anchor per lane: left (8.0, 10.0), right (10.0, 10.0), middle (9.0, 10.0).
2. **Pull tile** (`_pull_tile`):
   - consider only `CENTER_BOX` tiles that meet today's pull condition (`PULL_MIN_DIST ≤ d ≤ nearest_own_tower_dist − PULL_CLOSER_TILES`), with the targeter's lead position at `DECISION_LATENCY_S + 1.0 s` deploy, using M1.2's speed rule;
   - rank by distance to the lane's anchor (ties: the higher row);
   - none qualifies → no pull (the building is not proposed against that targeter; today's deep tiles are gone).
3. **The pull window.** A building against a building-targeter may be placed from when the targeter is at row ≤ 20 heading to our side (the bridge approach), not only after it crossed. Keep the "locked on → lost" rule. The anchor tile pulls earliest.
4. **Non-targeters** (today's `blocks X at the tower`):
   - the building goes to the lane's anchor tile with reason `kites {threat} to the centre`, if the lead unit is ≥ 3 tiles from the anchor and still above it (row > the anchor's + 1);
   - otherwise it is not proposed.
   - **Exception:** a flying push with no other affordable card that hits air may use the old tower tile, reason `air dps at the tower`.
5. **Candidates from lookup / prior / leak / learned.** A Building candidate whose tile is outside `CENTER_BOX` is dropped in `filter_candidates` / `no_blind_spells` unless its tile is the engine's (`engine_tiles`). The engine's own building tiles follow M3.2–M3.4. This removes the full-bar leak tesla at [5, 2.5].

## M4. Log why the bot holds (`live.py`)
- `holds.jsonl` in the run dir: one line whenever the policy verdict's `(mode, act, reason)` changes, and at most once per second while unchanged. Fields:
  - `t`, `mode`, `act`, `reason`, `elixir`, `hand`;
  - `pushes`: up to 3 of `{lane, lead, eta_s, eta_source, damage, answered}` from the engine report the loop already has, or `[]`.
- `summary.json` gains `holds_by_reason`: counts by the reason's first 3 words.
- Measure the cost; it must stay < 1 ms p95 per frame.

## M5. A closed window ends the run cleanly (`live.py`)
When the frame source raises (window closed / capture failed / end of stream), stop the loop and finalise exactly as at the time limit:
- `summary.json` and the shots index flushed;
- `holds.jsonl` closed;
- a one-line message `source ended: <error>`;
- exit code 0.

## Tests
1. **Lead.**
   - A unit at [4, 16] moving down at card speed 1.0 (Kalman 0.7): fireball delay ≈ 0.5 + dist(king, aim) / 10 ≈ 2.0 s, aim ≈ [4, 14.0] (old ≈ [4, 15.3]);
   - zap delay 0.5 s, aim ≈ [4, 15.5];
   - a stationary unit at their tower → no lead.
2. **Worth.**
   - lone firecracker + tower (value 4) → no fireball, in every mode incl. emergency;
   - archers + firecracker (kills 2, 7.5) → fireball;
   - zap kills 0 → no;
   - zap on 3 bats → yes;
   - zap on 1 bat of 5 → no;
   - swarm valuation per M2.3.
3. **Tesla.**
   - giant at the left bridge [3.5, 15], heading down → tesla tile in `CENTER_BOX`, nearest (8, 10), reason `pulls giant to the centre`;
   - giant at [4, 8] (deep, not locked) → no `CENTER_BOX` tile qualifies → no tesla move;
   - firecracker at the right bridge → tesla at (10, 10) `kites firecracker to the centre`;
   - balloon at our tower and no other air card affordable → `air dps at the tower` allowed;
   - a leak candidate `tesla|left|back` at [5, 2.5] → dropped.
4. **Holds log:** a scripted loop with a hold → act → hold sequence writes the expected lines; rate limit 1 Hz; `holds_by_reason` in the summary.
5. **Graceful end:** a fake source that raises after N frames → the summary is written, exit 0.
6. The MAC-AD / AG / AH / AK / AL tests pass, or are updated with a stated reason (pull tiles move to the centre box; the spell bar is higher).

Offline:
- `tools/eval/eval_act_policy.py --out runs/smoke/AM/act_policy.json`;
- `tools/eval/eval_tactics.py`: compare with the MAC-AL numbers in `docs/mac/STATUS.md` (act rate 0.232 test/all; tactics top-1/top-3).

Run the full suite (`--deselect tests/test_perception_latency.py`). CPU only; never touch the screen; never use `--live`.

Smoke: the MAC-AL 120 s video dry run (`--calib runs/poc/mirror_calib.json` is needed), `--out runs/smoke/AM_dry`. Report the tesla tiles, the spells cast with their reasons, the holds.jsonl line count and the `holds_by_reason` table.

## Report
Per MAC-00 §5, with:
- the lead numbers of test 1;
- the swarm-valuation finding;
- the tesla tiles of test 3;
- the eval tables vs MAC-AL;
- the smoke tables.
