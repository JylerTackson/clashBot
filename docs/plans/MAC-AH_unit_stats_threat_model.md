# MAC-AH — real unit stats and a threat model: rank pushes by tower damage, judge answers and spells by what they kill (spec for one Sonnet implementer)

Author: Opus · 2026-11-06 · From the recorded run `runs/poc/20261106-170722` (2 games). User: *"it is just not understanding the magnitude or the stats of the enemies … a witch and 5 bats in one lane and a balloon in the other and it spawns 2 enemies in response to the 5 bats and ignores the balloon. I feel this is because it is not properly fetching card unit data from the KB?"*

Read `docs/plans/MAC-00_common.md` first. D-045 holds: the LLM picks every move. This changes the engine's analysis, the policy's priorities, and the numbers in the tactics lines the LLM reads.

## Evidence
- **The balloon was not a push.** At t = 240.7 there were two threats:
  - left: bats ×2, night witch, e-wizard (10 elixir, ETA 2 s);
  - right: a balloon at the river ([16, 15]).

  The balloon was not a push: pushes need ETA ≤ 10 s or row ≤ 14, and its ETA was unknown. The bot spent on the left lane (a knight) and had nothing left when the balloon crossed.
- **Pushes are ranked by ETA only.** There is no notion that a balloon does ~1,000+ per drop on a tower, while bats do ~40 per hit and die to the tower in seconds.
- **Spell value counts elixir, not kills.** Zap was cast 3× on a lone balloon ("zap balloon (5 elixir)"), because value 5 ≥ cost 2. Zap does not kill or meaningfully hurt a level-15 balloon.
- **The knowledge base has no numeric unit stats.** `card_roles.json` holds coarse classes only (`hp_class`, `dps_class`).

## Owns
```
clashbot/unit_stats.py (new)   clashbot/resources/unit_stats.json (new, generated, committed)
clashbot/resources/levels.json (new)   tools/data/build_unit_stats.py (new)
clashbot/tactics.py   clashbot/policy.py   tests/test_unit_stats.py (new)   tests/test_tactics.py   tests/test_policy.py
```
`clashbot/live.py` is not touched. Levels come from `levels.json`, with a `--opp-level` flag only if trivially needed. If so, report it rather than editing `live.py`.

## 1. Stats table (`tools/data/build_unit_stats.py` → `clashbot/resources/unit_stats.json`)
**Source.** RoyaleAPI cr-api-data `cards_stats.json` + `cards.json`. Download with `--fetch` into `data/external/cr-api-data/` (gitignored), or reuse an existing copy there. Map cards to units via `summon_character` / `summon_number`, plus the second unit where present (goblin gang, rascals, …), the spawn and death-spawn characters, and buildings from the `building` section.

**Per card** (our slugs):
- `type`;
- `units`: a list of `{name, count, hp, damage, hit_speed_s, dps, range_tiles, speed_tiles_s, targets: ground|air_ground|buildings, flying, splash_radius_tiles, death_damage, death_spawn}`;
- building `lifetime_s`;
- for spells: `damage`, `crown_tower_damage`, `radius_tiles`;
- `elixir`;
- `rarity`.

Values are stored at the cr-api base level, with the rarity's level offset.

**Units.**
- Game units: 1,000 per tile for range and radius.
- Speed: 60 ≈ 1 tile/s per the usual mapping. Document the mapping you use.
- `hit_speed` is in ms.

**Level scaling** (`unit_stats.stat_at(card, level)`):
- `value × 1.1 ** (level − start_level(rarity))`, with start levels common 1, rare 3, epic 6, legendary 9, champion 11 (current card-level numbering).
- Round as the game does (to integers).
- **Calibrate and test against observed values:** the princess tower at level 15 shows **4,424 HP** in our screenshots, so the tower base must reproduce that. Add the princess and king tower entries (HP, damage, hit speed, range) from cr-api's tower data if present, else from known values. Report how close the scaling gets for a few cards you know.

**Cards missing from cr-api** (e.g. berserker, goblinstein, ronin): a `MANUAL_STATS` block with your best values at a stated level, marked `"source": "manual"`. Priority:
- the user's deck: electro-dragon, musketeer, tesla (incl. the evo note), fireball, knight, zap, ram-rider, mini-p-e-k-k-a;
- the Training Camp opponents seen: giant, balloon, bats, night-witch, minions, graveyard (skeletons), executioner, electro-wizard, dark-prince, mini-p-e-k-k-a, firecracker, archers, fire-spirit, golem, little-prince, bomber, royal-giant.

**`clashbot/resources/levels.json`:**
```
{"opponent_default": 15,
 "own": {"knight":16, "electro-dragon":16, "ram-rider":16, "mini-p-e-k-k-a":15, "tesla":15, "musketeer":15, "fireball":15,
         "zap":15, "princess":15, "goblin-barrel":14, "bowler":14, "baby-dragon":14, "electro-spirit":14, "bomber":14,
         "freeze":14, "goblins":14},
 "own_default": 14, "tower_level": 15}
```

## 2. Threat model (`clashbot/unit_stats.py`, used by `tactics.py`)
`simulate_lane(enemy_units, own_defenders, tower, horizon_s=20) -> SimResult`. This is a small deterministic 1-D simulation per lane, and must take < 2 ms.

**Enemy units** (with their units' stats at the opponent level):
- Each walks toward the lane's princess tower at its speed, starting from its tile; the distance is to `TOWER_TILES` minus the attack range.
- Flyers fly straight. Building-targeters go to the nearest building; others go to our nearest defender, else the tower.

**Our side:**
- **The tower** (level 15 stats) shoots one target at a time: the nearest unit within its range, air or ground.
- **Defenders** (our placed or proposed units at our levels) attack the units they can target once those are in range. A defender takes damage from units that target troops and are in range.

**Outputs:**
- `tower_damage` (sum of damage dealt to the princess tower, plus death damage that lands in range);
- `tower_destroyed` (bool) and its time;
- `units_alive_at_end`;
- `time_to_clear`.

Approximations are fine (no collisions, no retargeting except building pulls handled by MAC-AG). Document them.

## 3. Engine changes (`tactics.py`)
**a. Pushes include river/bridge threats.** Every enemy unit on our half, **or** at rows 15–18 heading toward our side, counts:
- toward our side means `heading` advancing, or a velocity with negative row speed;
- also any building-targeter or flyer at rows ≤ 20 in a lane.

ETA comes from the simulation when the tracker gave none.

**b. Push magnitude.** `Push.damage_if_unanswered = simulate_lane(units, [], tower).tower_damage`, plus `Push.destroys_tower`. The push line shows it: `tactics: push right: balloon (5 elixir), ~3,100 tower damage if unanswered, tower in 6s; needs air`.

**c. Rank pushes by magnitude.** Order by `destroys_tower`, then `damage_if_unanswered`, then ETA.

**d. The tower handles it.** A push with `damage_if_unanswered < tower_ok_damage` (default 300) gets no defend moves, and its line says `tower handles it`. Example: two bats.

**e. Answer value by simulation.**
- For each defend move (card C at its engine tile), run the simulation with C as a defender:
  - `prevented = damage_if_unanswered − damage_with_C`;
  - `efficiency = prevented / cost`.
- Adequate means `damage_with_C ≤ max(tower_ok_damage, 0.35 × damage_if_unanswered)`, or no unit reaches the tower.
- The move score gets `w.prevent × normalised prevented`. Keep the role terms at a lower weight as a tie-break.
- Reasons quote it: `tesla pulls balloon … saves ~2,400 tower HP`.

**f. Spells by kills and damage.** The spell's damage at our level vs each unit's HP at the opponent level, in the radius:
- killed units count their share of card elixir (`elixir × killed / count`);
- damaged survivors count `0.5 × elixir × damage / hp` (capped at 0.5 × elixir).

The value must be ≥ the spell's cost. So zap vs a lone L15 balloon is not worth it; zap vs bats kills them all (full value). Replace the elixir-only value in `policy.spell_target` through a helper both use. Keep the policy API.

**g. Two pushes, limited elixir.** The policy answers the push with the larger `damage_if_unanswered` first. It keeps a reserve for a bigger push that arrives within 10 s: never spend the elixir needed for the bigger push's best adequate answer on a smaller one, unless the smaller push alone would destroy a tower.

## 4. Policy (`policy.py`)
- Use the engine's push order and magnitudes (3c, 3d, 3g) in the MAC-AD/AG defend rules.
- A `tower handles it` push leads to hold, mode `save`, reason `tower handles {push}`.
- Adequacy comes from the engine's simulation-based `adequate` flag on moves. Fall back to the role rule when stats are missing for a card.

## Tests
- **Stats:**
  - the princess tower at level 15 has 4,424 HP;
  - knight / giant / balloon / bats at levels 15–16 have plausible HP (state the expected ranges in the test);
  - zap damage at 15 kills L15 bats but not a balloon;
  - fireball at 15 kills L15 archers / musketeer-class but not a giant.
- **Simulation:**
  - 2 bats vs the L15 tower → tower damage < 300;
  - a lone L15 balloon unanswered → large damage (≥ 1,500);
  - a balloon with a musketeer behind the tower → much less damage.
- **The user's scenario:** left lane bats ×2 + night witch + e-wizard (ETA 2), right lane balloon at [16, 15] heading down. Deck electro-dragon / musketeer / tesla / fireball / knight / zap / ram-rider / mini-p-e-k-k-a, elixir 7, hand musketeer, tesla, knight, zap:
  - the balloon push is ranked first;
  - the defence goes to the balloon (tesla pull or musketeer);
  - the left lane gets zap or nothing (tower/e-dragon), never a knight that spends the balloon's reserve.
- **Zap vs a lone balloon** is not proposed, and the policy filter drops it.
- Each existing MAC-AD/AG test passes, or is updated with a stated reason.

Run the full suite (`--deselect tests/test_perception_latency.py`). CPU only, never load a model, never touch the screen.

## Report
Per MAC-00 §5, with:
- the stats calibration table;
- the simulation outputs for the test scenarios;
- the new tactics lines for the user's scenario;
- the spell values (zap / fireball vs bats, balloon, archers).
