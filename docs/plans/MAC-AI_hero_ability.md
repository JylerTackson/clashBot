# MAC-AI — hero ability (Hero Musketeer's Trusty Turret): see the button, tap it, offer it to the LLM (spec; part 1 now, part 2 after MAC-AH)

Author: Opus · 2026-11-06. Read `docs/plans/MAC-00_common.md` first.

User: *"the engine is not using the hero ability it has access to. It doesn't need to if it does not think it is the right move but we need to make sure it has the ability to."* D-045 holds: the LLM decides whether to use it.

## Facts (`knowledge_base/heroes/musketeer-hero.md`)
**Trusty Turret:**
- costs 3 elixir;
- after a 1 s delay, places a turret **3 tiles forward of the Musketeer**;
- turret range 4.0, targets air and ground, lifetime 10 s;
- serves as a distraction for troops and as a building that attracts win conditions;
- per-level turret HP / DPS / spawn damage are in the file.

**The button:**
- It sits "just above the player's card slots", on the side of the screen where the hero was placed.
- Recorded example: `runs/poc/20261106-170722/shots/d1518_t191.4_a.jpg` (50 % scale, 316×696). The button is centred at about (278, 542) px, i.e. **content fractions (0.879, 0.779)**: a round blue button with the turret icon and a purple "3" elixir badge at its top-left.
- The left-side position is presumably mirrored, at about (0.121, 0.779).

Our deck (D-047) has the Hero Musketeer only, so any ally `musketeer` on the field is the hero. Evo Tesla and Evo Electro Dragon have no button.

## Part 1 (now) — perception, calibration, executor, option plumbing

### Owns
```
cr_perception/hud.py (an AbilityButtonReader)   cr_perception/perception.py (wire it; state.own.hero_ability)
cr_perception/assets/hero_ability/ (new templates)   clashbot/executor.py (tap_ability)
clashbot/live_state.py (pass hero_ability into the clean state; tracker elixir accounting for ability taps)
clashbot/candidates.py   clashbot/d1_format.py   clashbot/actions.py   clashbot/live.py (ability key routing only)
clashbot/state_text.py (one context token)   tests/test_hero_ability.py (new) + updates as needed
```
- Perception edits are allowed under D-043.
- `clashbot/tactics.py` and `clashbot/policy.py` belong to MAC-AH, which is running. Do not touch them.

### 1. Detect the button (`AbilityButtonReader`)
- **ROIs.** Two, left and right, around (0.121, 0.779) and (0.879, 0.779), each about 0.16 × 0.09 of the content. Make them calibratable as `rois["hero_ability_left" | "hero_ability_right"]`, with these defaults.
- **Template.** Extract it from the recorded shots. Scan `runs/poc/20261106-170722/shots/*_a.jpg` and `*_b.jpg` (50 % scale; upscale ×2 to content scale) for the button. Save one or more crops to `cr_perception/assets/hero_ability/musketeer*.png`, including a greyed (unaffordable) one if any frame has it. Match with the same lighting-normalised approach as `CardMatcher`, plus a check that the circle is present.
- **Output.** `state.own["hero_ability"] = {"hero": "musketeer", "visible": bool, "side": "left"|"right"|None, "affordable": bool, "cost": 3, "conf": float}`.
  - `affordable` comes from saturation (greyed vs coloured), else from elixir ≥ cost.
  - Hysteresis: the button must be visible in 2 of the last 3 reads.
- **Report.** How many recorded frames show the button, and the detection precision / recall on frames you label by eye (at least 20 positives and 20 negatives from the shots).

### 2. Executor: `MirroringExecutor.tap_ability(side, t) -> entry`
- One tap at the button centre (`content_to_screen` of the ROI centre).
- The same double live guard, bounds check and logging as card taps.
- Dry-run executors log `planned_ability`.

### 3. The option for the LLM (plumbing only)
- **Key format.** `ability:musketeer|{lane}|{zone}`. Lane and zone come from the turret spawn tile: the ally Musketeer's tile + 3 tiles toward the enemy (row + 3), clamped.
- **`d1_format.option_description`.** For an ability key: `activate Hero Musketeer's Trusty Turret (3 elixir): a turret 3 tiles in front of her, range 4, pulls win conditions, 10 s`. `validate` must accept these keys.
- **Generator.** `CandidateGenerator.generate(..., ability_moves=None)` appends ability candidates passed in by the caller, with source `ability` and a note. **Part 2 will produce them**; for now the live loop passes one when:
  - `hero_ability.visible` and `affordable`;
  - an ally musketeer is on the field;
  - the policy says act.

  Its note is `hero ability: turret in front of the Musketeer`.
- **Live loop routing.**
  - When the chosen key starts with `ability:`, call `executor.tap_ability(side, t)` instead of a card tap.
  - Skip `require_hand`.
  - Record `note_own_play("ability:musketeer", None, t, cost=3)`, so C2's elixir accounting includes the 3 elixir.
  - Log `mode` / `tile_source = "ability"`.
- **Context text.** Add the token `hero: musketeer turret ready (3)` to the hand line when the button is visible, so the LLM knows. Dataset builders are unaffected: the key is absent in recorded states.

### 4. Tests
- The reader detects the template in synthetic frames on the right and left, rejects frames without it, and reports `affordable` False for a greyed crop.
- `tap_ability` maps to the right screen point and is guarded.
- An ability key passes `d1_format.validate`, and its description is as specified.
- The live loop routes an ability key to `tap_ability`, skips the hand check, and accounts for 3 elixir.
- Recorded-frame check: the detector finds the button on `d1518_t191.4_a.jpg`.

### 5. Smoke
A 30 s **video** dry run with `--record-shots`. Report that nothing crashed and whether the button was ever seen (it may not be, in pro videos). Check `pgrep -f clashbot.live` first, and never use `--live`.

## Part 2 (after MAC-AH merges) — when it is worth it (engine + policy)
- **Engine intent `ability`**, scored with the MAC-AH simulation: the turret as a defender at the spawn tile, with KB stats at musketeer level 15. Typical good moments:
  - a building-targeter is heading for the Musketeer's lane (pull);
  - enemy troops are attacking the Musketeer (distraction);
  - a swarm is in turret range.
- **Policy.** The ability counts as an adequate answer when the simulation says so. It goes into `allowed_cards` as `ability:musketeer`, and the LLM picks.
