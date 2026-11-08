# Backlog after the showcase (2026-11-09): from the 10-game Training Camp trial

Author: Opus · 2026-11-07 · Source: the user's notes from the trial and the session log `runs/demo/20261107-213709`. The trial had 9 Training Camp battles logged and 10 played. Result per the user: **0 wins, 10 losses; crowns 3 for, 18 against; 2 overtime losses.** Work starts after the code freeze (D-051) lifts. Specs go in `docs/plans/` as usual. Priority = the user's order ("we will work on these items first"), then the findings from the logs.

Battle ↔ game mapping (user's numbering), from the end-of-battle screenshots:
- B8 = game 7: OT, 1–3; its frame shows "Overtime 1:29".
- B9 = game 8: 0–3.
- B11 = game 10: the last battle.

The other games map in order with one gap. B1, B4, B5, B6, B7 and B10 are games 1–6 and 9, but which game went unlogged is not certain. B2 and B3 are two short screens that read banner "2618" and were correctly gated as not Training Camp (D-050).

## P1 — the user's items

### W1. Tower HP reading is unstable (user: "not consistently attacking the same towers … issues with reading tower health")
**Evidence** (`context_text` "towers own L/R/K, enemy L/R/K"):
- B1, t 41–70: own left flips 4424 ↔ 424 seven times. A digit is dropped.
- B1 156–188: own right reads 3145 → 8145 → 145 → 3145.
- B4 618–667: enemy left reads 1665 → 3991 → 1663 → 3124.
- B7 1589: enemy left 553 → 1515.

**Fix.** Filter HP per tower and per battle:
- HP may only go down, except at a battle reset.
- Reject reads that are a digit drop or insertion of the last good value.
- Take the median of the last 3 consistent reads.
- Mark a tower destroyed when its sprite or HP bar is gone for ≥ 1 s. Then it reads 0, not None or "?".

**Test:** replay these runs' frames; zero flips; destroyed towers become 0.

### W2. Commit to a target tower; never attack the king while a princess tower stands (user: "attacked the king tower even though there was still a queen tower left")
**Evidence.**
- Counter-pushes follow the lane of the defended push (`tactics._offence_moves`: counter_push in the ally's lane; split uses the opposite lane; punish goes away from the opponent's last play). Nothing keeps one target.
- B4: the first ram rider went left (enemy left fell to 1663), then 3 counter ram riders went right. It ended with enemy L 1663, R 1824, and no tower taken.
- B6: left, then right ×3. It ended at L 656–959, R 1544.
- B8 (game 7): after the enemy right tower fell (468 → 125 at t 1905), ram riders kept going right (t 1925, overtime), and the right lane leads to the king.

**Fix.**
- Keep a per-battle target tower: the weaker princess tower on filtered HP (W1), with hysteresis.
- Offence moves (leak, punish, split, ram rider) go to the target lane unless a counter-push in the other lane is clearly better. Compare them with the lane simulation's expected tower damage.
- After a princess tower falls, the target becomes the other princess tower.
- The king is a target only when both princess towers are down, or in a late-overtime race where the sim says the king falls first.
- The ranker sees the target in its context line.

### W3. Use the bigger deploy area after taking a tower (user: "not taking advantage of the increased spawn room")
**Evidence.** `game_rules.legal_zones` adds `enemy_half` only when an enemy princess tower reads exactly 0. Perception never reports 0 (destroyed towers read None, "?" or flicker), so the pocket is never legal.

**Fix.**
- Use W1's destroyed flag.
- The pocket is only the destroyed tower's side, up to its row, not the whole enemy half.
- Tiles: ram rider and support in the pocket, behind the king tower's range where it matters.
- The engine offers pocket placements for offence.

### W4. Back up counter-pushes, with spare elixir only (user: "doesn't back up the counter push with a musketeer … lets them die to a witch"; ram rider: "backing up the ram riders with a ranged troop … or spawning a knight in front of the ram rider … like a shield")
**Evidence.** `_offence_moves` counter_push adds only win-condition cards ("backs knight in the left lane"). No support card is ever offered behind a surviving defence (knight + mini P.E.K.K.A).

**Fix.**
- New intents:
  - `support` (ranged troop such as musketeer or e-dragon behind an ally push across the river);
  - `shield` (knight placed so it leads the ram rider: same lane, about 1–2 tiles ahead at the bridge);
  - `pair` (ram rider + support).
- Each is offered only when elixir ≥ card cost + the defence reserve (MAC-AL regen-aware reserve), and no unanswered push threatens within the reserve window.
- The simulation scores the expected tower damage with vs without the support.

### W5. Fireball for value on grouped win conditions, and spell + troop combos (user: "it doesn't understand how to use the fireball effectively when units are clumped … fireball when their win conditions are grouped")
**Evidence.**
- B9 = game 8, t 2075–2087. Night witch + bats ×3, then giant + balloon + bats came down the right lane.
- The bot played musketeer as a full-bar *leak* (t 2077), mini P.E.K.K.A vs skeletons (2078) and e-dragon in an emergency (2087). The engine never proposed a fireball.
- The units were spread over about 10 tiles at first. When they bunched up, the spell bar valued a fireball by elixir destroyed only: giant + balloon + bat ≈ 3.3 < 5.
- Good fireballs for reference:
  - B8 t 1810 (game 7, ~1:55): e-wizard + night witch.
  - B11 t 2599 (game 10, 0:29): wizard + e-wizard.

**Fix.**
- Value spells by **simulated tower damage prevented**, as the zap group rule already does. This applies to fireball on win conditions such as balloon, giant + witch, or a hog pack.
- Add spell + troop combo moves, e.g. fireball balloon + e-dragon on the giant. The pair is scored by the lane sim.
- Predict clumping: aim at the tile where the push converges within the fireball's flight time.
- No full-bar leak on the far side while a 10+ elixir push is building.

### W6. Zap only on real groups (user: "Zap is used against a giant and 1–2 bats; it needs 3–4 bats or a firecracker and an electro wizard grouped")
**Evidence.**
- "defend giant middle: zap … zap bats x2: tower finishes them, saves ~700 HP" (B8 t 1951).
- "zap bats x2: … saves ~700 HP" (B10 t 2381).
- The MAC-AM group rule counts every member in the circle, the giant included.

**Fix.**
- The group rule counts only units the zap meaningfully hurts: HP ≤ 2× zap damage, or support units (e-wizard, firecracker, archers, spear goblins…).
- Require ≥ 3 such units, e.g. 3–4 bats, or e-wizard + firecracker, plus another unit.
- Tanks never count.
- Keep the kill-based rule (a).

### W7. Game 8 review: recognise a big push forming beyond the river (user: "Game 8 altogether was just very poorly played")
**Evidence (B9).**
- Knight leak at the back at 2:49 at full elixir.
- Musketeer leak at 2:37.
- Mini P.E.K.K.A emergency at 6 elixir.
- E-dragon emergency at 5.
- The tactics engine credited a ram rider of ours, possibly a phantom detection, with answering the right push ("being answered by ram-rider").

**Fix.**
- Count the opponent's committed elixir beyond the river and hold the reserve for it.
- Treat a push as answered only when an ally unit is confirmed and close.
- Covered partly by W5 and W8.

## P2 — from the logs

### W8. Over-playing and elixir starvation
**Evidence.**
- The trial had 243 taps in 9 battles: 27 per battle, 40 decisions per battle.
- 40 emergency defends at 2–5 elixir.
- Tesla was played 43 times (18% of taps); knight 56.
- Bot vs pro on video (24 test games): the bot taps 22.8 times per minute vs the pro's 17.4 (`docs/data/vs_pro_v1.md`).

**Fix.** An elixir budget per minute, set from the pro rate. Tesla only against a building-targeter that will cross, or with spare elixir. The "act model p below threshold" holds should be trusted more.

### W9. Tap spam and taps that fail
**Evidence.**
- B1 t 44–62: 10 leak taps in 18 s, at [5, 2.5] each. The elixir was probably misread at the start.
- 30 tap pairs were < 1.6 s apart in the trial.

**Fix.**
- After a tap, require the elixir drop or the card leaving the hand before the next tap of that card.
- Read the "Not enough Elixir!" banner.
- Cap leak taps to one per 3 s.

### W10. Phantom threats ("graveyard" at our side)
**Evidence.** "defend graveyard middle" drove zaps and teslas at t 667, 1560, 1885, 1946, 2134 and 2189. Earlier logs show graveyard(e)@[8,5] "stationary" on our half, and Trainer Cheddar's deck has no graveyard.

**Fix.**
- Find which sprite KataCR labels graveyard; check the shots.
- Ignore stationary enemy spells outside a 1-s cast window.
- Require deck evidence for a card the opponent has never shown.

### W11. Automatic result logging
**Evidence.** The end-of-battle frames missed the result screen in 5 of 9 battles:
- B1 black;
- B4 and B10 show the next battle's VS screen;
- B5 and B11 show menus.

**Fix.** Detect the result screen (the "Winner!" banner and the crown rows) during the 10 s after the battle, save that frame, and OCR crowns for / against and the winner into `battles.jsonl`. Then live trials need no manual notes.

### W12. Battle start detection gap
9 battles were logged against 10 played. Find the missed one: the menu → match gate, and the short "2618" screens.

## Also open (FIX_LIST)
- #58: the LLM's chosen tile loses to the verdict's engine tile (D-045).
- #57: enemy HP bars.
- #49 and #50: KG-leak retrain and dataset ETA parity.
- #62: banner tolerance.
- #63 and #64: clean-checkout tests and demo calib fallback.
- #65: the stale tuning guard.

## Suggested order after the showcase
1. W1, then W2 and W3, which depend on W1.
2. W6 (small).
3. W5.
4. W4.
5. W9 and W8.
6. W10.
7. W11, which is needed for the next live trials.
8. W12 and W7 fold into the others.

Re-run the 10-battle trial after W1–W6 with automatic result logging (W11), and compare against this 0/10 baseline.
