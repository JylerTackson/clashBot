# MAC-AQ — the bot vs the pros on their own videos: run the full live stack open-loop on test-split games and compare its decisions with what the pro did (spec for one Sonnet implementer)

Author: Opus · 2026-11-07 · Showcase Monday 2026-11-09. Code freeze D-051: decision logic frozen at `66917e7`. Read `docs/plans/MAC-00_common.md` first.

User: *"I also want to run the bot against the videos we downloaded and show how its decisions compare against the pros."*

## Owns
```
tools/eval/eval_vs_pro.py (new)        tests/test_eval_vs_pro.py (new)
clashbot/viz/data.py, clashbot/viz/static/* (only: show the pro's plays when a run dir has pro_plays.jsonl — Q4)
tests/test_viz.py (only for Q4)
docs/data/vs_pro_v1.{json,md} (new)    runs/eval/vs_pro/<game_id>/ (run dirs, gitignored)
```
No change to `clashbot/live.py`, `demo.py`, policy, tactics, candidates or ranker. Use `clashbot.live.build_stack` (MAC-AN) and `LiveLoop` with `VideoFrameSource` and `DryRunExecutor` — the exact live configuration: `--ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac --policy fast --act-model db/models/act-gbdt`, tactics and kg2 on, `--record-shots`, fps 10, detect-every 3.

## Facts
- `dataset/v1/games.jsonl`: game_id (`<video_id>-m<k>`), split, own_deck (the pro's 8 cards), result, flags. `dataset/v1/provenance.jsonl` maps games to the video file and time window (find the fields). `dataset/v1/states.jsonl`: one row per sample (`kind` play / key / periodic) with `t` and, for plays, `action` {card, tile, lane, zone, key, delay_s, confidence}. Videos: `data/videos/<video_id>/<video_id>.mp4`. Check how `t` maps to video time (match time vs video time) on one game before anything else and write the mapping into the report.
- The pro's POV is the bottom player, like ours, so perception and the calibration (`runs/poc/mirror_calib.json` worked for video dry runs on `Z8-4VhLjrGU`) apply; verify the game rect on each video (the dry runs print rect changes).
- The bot's actions do not change the video (open loop): this measures what the bot would do in the pro's situation.

## Q1. Runner
`python -m tools.eval.eval_vs_pro --split test [--games N] [--stride 3] [--out runs/eval/vs_pro]`
- For each game: deck = the pro's `own_deck` (hero / evo flags from the dataset if present, else none); run the stack over the game's video window; write a normal run dir (decisions.jsonl, holds.jsonl, shots, summary.json, config.json) plus `pro_plays.jsonl` (the pro's play rows for that game: t in run time, card, tile, lane, zone, key, confidence).
- Budget: time one game first; choose N so the whole run fits in ≤ 3 h wall; at least 8 test games, all 28 if they fit; prefer `deck_status == "valid"` games with known result; record which games and why.
- Every run under the heavy lock `/usr/bin/lockf /private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/heavy.lock` and pause between games while `/private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/HOLD_HEAVY` exists (the user is playing live). Resumable: skip games whose run dir has a summary.json.

## Q2. Metrics (`--score` over the run dirs; no model needed)
Match bot acts (decisions whose executor status is planned/dry_run, i.e. it would tap) with pro plays by time:
1. **Timing**: pro-play recall (a bot act within ±2 s of the pro play; also ±1 s, ±3 s), bot-act precision (a pro play within ±2 s), acts per minute bot vs pro, median |Δt| of matched pairs, share of bot acts earlier vs later.
2. **Choice on matched pairs**: same card; same lane; same zone; same key; tile distance (tiles) when both have tiles; elixir spent per minute bot vs pro.
3. **When the bot holds**: for unmatched pro plays, the bot's hold reason at that moment (from holds.jsonl: bucketed reason) — the top reasons table.
4. **Baselines** at the pro's play times on the same games: random card from the perceived hand (expected agreement = 1/hand size), the placement prior's key, the card the KG v1 `counter_scores` ranks first among the perceived hand (if a threat exists).
5. Breakdowns by the bot's mode (defend / punish / counter / leak / spell / learned), by game, by pro play confidence.
6. Bootstrap CIs (resample games) for recall, precision, same-card rate.
Outputs: `docs/data/vs_pro_v1.json` + `.md` (tables; one paragraph on caveats: open loop, perception noise, the pro's hidden information).

## Q3. Examples for the slides
Pick 6 matched decisions: 3 where the bot agreed with the pro (same card, ≤ 2 tiles), 3 where it disagreed in an interesting way; for each write game, t, pro play, bot play, bot reason, and the shot file. Put them in the .md.

## Q4. Dashboard: show the pro (only `clashbot/viz`)
When a run dir has `pro_plays.jsonl`: the timeline draws the pro's plays as a second row of marks ("pro"), and the decision panel shows the nearest pro play within ±3 s ("pro played X at t, Δt = …") with a match badge (same card / same lane / different). No change for other runs. Test with a fixture.

## Tests
`test_eval_vs_pro.py`: the matching and every metric on a hand-made toy (known answers), the time mapping, resumability (skip finished games). Full suite green under the lock.

## Report
Branch, commit, files; the t→video mapping; games run (ids, minutes, wall time); the headline table (recall, precision, same card / lane / key with CIs vs baselines); top hold reasons on missed pro plays; the 6 examples; deviations; notes for Opus.
