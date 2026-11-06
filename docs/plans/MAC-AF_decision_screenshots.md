# MAC-AF — record what the bot sees and taps: decision screenshots + review sheets (spec for one Sonnet implementer)

Author: Opus · 2026-11-06 · After 4 live losses (`runs/poc/20261106-133529`). Read `docs/plans/MAC-00_common.md` first.

## Why
Live analysis so far is inferred from logs only:
- hand slots read different cards second to second;
- taps land about 60 % of the time (magic archer 9/19, berserker 14/26), estimated from elixir;
- we cannot tell whether a failed tap hit the wrong slot, missed the arena, or met an unaffordable or still-selected card.

We need ground truth on screen.

## Owns
```
clashbot/live.py (recording hooks + CLI flags)   clashbot/recorder_shots.py (new)
tools/eval/review_shots.py (new)   tests/test_shots.py (new)
```
`clashbot/executor.py` is read-only. Get the tap screen points from the executor entry (`clicks`) and map them back to image pixels with `executor.screen_to_image` (exists).

## 1. Recording (`--record-shots`, off by default; live and dry run)
- **Per decision** that reaches the executor or a hold record, save `shots/d{idx:04d}_t{t:.1f}_a.jpg`: the frame the decision used, JPEG quality 80, scaled to 50 %. Draw on it:
  - the 4 hand slot ROIs with the tracked hand labels;
  - for a sent tap: the card-slot tap point (cyan) and the field tap point (magenta), plus the target tile number;
  - the elixir reading.
- **After a sent tap**, also save the frames that arrive ≥ 0.5 s and ≥ 1.5 s later as `…_b.jpg` and `…_c.jpg`, unannotated except a small header (`+0.5s`, `+1.5s`, elixir read).
  - Write them from a small background thread with a bounded queue, so the loop never blocks. If the queue is full, drop and count `shots_dropped`.
  - The frames are the source's frames; keep references, and convert or encode in the thread.
- **Index:** `shots/index.jsonl`, one line per decision, with:
  - `{idx, t, key, mode, reason, status, slot, clicks_screen, clicks_image, tile, elixir_before, hand_before, hand_conf, files: [a,b,c], elixir_after_0.5, elixir_after_1.5, hand_after_1.5}`.
  - Fill the `_after` fields from the builder's readings when those frames are processed.
- `--shots-max 400`: stop after N decisions.
- Disk use is about 60 KB per image.

## 2. `tools/eval/review_shots.py RUN_DIR [--only failed|all] [--out RUN_DIR/shots/sheets]`
- **Landing verdict** per sent tap:
  - `landed`: the elixir dropped by ≥ cost − 0.6 within 1.5 s, or the tapped card left its slot;
  - `failed`: neither;
  - `unknown`: no reading.
- **Contact sheets:** PNG, 3 images per row (a / b / c) per decision, 6 decisions per sheet. Each row has a caption with key, mode, reason, verdict, elixir before → after, and the hand read before / after.
- **Summary table:** landing rate by card, by slot, by target row band, and by mode. Also how often the tapped slot's card at +1.5 s equals the tapped card (still in hand). Write it to `review.md`.

## 3. Tests (synthetic frames, fake executor)
- Recording is off by default: no files.
- With `--record-shots`:
  - a/b/c are written with annotations;
  - the index has the fields;
  - a full queue drops without blocking (step time is not affected).
- `review_shots` on a tiny synthetic run gives the right verdicts and writes the sheets and `review.md`.

## 4. Smoke (no live)
A 30 s **video** dry run with `--record-shots`. Use the full stack:
```
--ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac --policy fast --act-model db/models/act-gbdt
```
Then run `review_shots.py` and paste `review.md` and the path of one sheet. Report the decision-latency p95 with vs without recording; recording must not raise it by more than 10 %. Check `pgrep -f clashbot.live` first, and never use `--live`.

## Report
Per MAC-00 §5.
