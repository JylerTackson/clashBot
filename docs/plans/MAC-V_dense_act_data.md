# MAC-V — dense wait/act dataset from re-run perception (spec for a Sonnet implementer)

Author: Opus · 2026-11-03 · Implements D-040. Read `docs/plans/MAC-00_common.md` first.

## Context
Opus re-downloads the 33 pro videos (`scripts/download_videos.py --ids …` → `data/videos/<id>/<id>.mp4`) and runs `tools/batch_process.py --device mps` overnight, which writes `runs/videos/<video_id>/match_<n>/[game_<k>/]states.jsonl` (one perception `GameState` row per processed frame, `type == "state"`, plus `type == "play_event"` rows; times `t` are **video seconds**). You build the dataset from those files; you do not run perception.

## Owns (only these files)
```
tools/data/build_dense_act.py   (new)
tests/test_dense_act.py         (new)
```

## 1. `tools/data/build_dense_act.py`
```
$PY tools/data/build_dense_act.py [--dataset DIR] [--runs runs/videos] [--step 1.0] [--horizon 1.0] [--tol 0.6]
        [--games ID,...] [--out DIR]
```
Defaults: `--out <dataset>/act_dense`.
1. **Games**: `<dataset>/games.jsonl` (game_id, split, own_deck) + `<dataset>/provenance.jsonl` (video_id, video_time = [start, end] video seconds as strings or numbers). Skip games without both (count `skipped_no_time`).
2. **Perception rows** for a game: every `states.jsonl` under `<runs>/<video_id>/` (recursive glob), read with `cr_perception.recorder.read_jsonl`; keep `type == "state"` rows with `readiness == "match"` and `start - 1 <= t <= end + 1`, and `type == "play_event"` rows with `timestamp` in that range. Sort by time; drop exact-duplicate times. No rows → count `skipped_no_perception` and continue.
3. **Pro deploy times** (labels) from `<dataset>/states.jsonl` records of that game: own plays from `state.own.recent_plays`, `action.following_plays` and the record's own action when `action.type == "play"` and `delay_s` is a number (time `rec.t + delay_s`); keep entries with a playable card; dedupe on `(round(t, 1), card)`; collapse the same card within 1.0 s. Function `pro_deploy_times(records, cards) -> list[tuple[float, str]]`.
4. **Samples**: `t` from `start` to `end` every `--step`; nearest state row within `--tol` (else skip, count `no_state`). Build the state with a **fresh** `LiveStateBuilder(LiveConfig(own_deck=game["own_deck"]))` per game, fed incrementally: before each sample, `observe_events` every play_event with `timestamp <= t` not yet fed; then `build(row with "t" overridden to t)`; `None` → skip. (Incremental feeding keeps first-seen times and the hand hold exactly like the live loop.)
5. **Label**: `act = any(t < d <= t + horizon for d, _ in deploys)`. Also `meta`: `{"since_opp_play": seconds since the latest opponent play in `state.opponent.recent_plays` (None if none), "threat": bool(state["threats"]), "own_elixir": state.own.elixir, "opp_elixir": state.opponent.elixir_estimate, "next_deploy_in": seconds to the next deploy (None)}`.
6. **Line** (d1 act format, validated with `clashbot.d1_format.validate`): `{"id": f"{game_id}#{t:.1f}", "game_id", "split", "kind": "dense", "variant": "act", "state": s, "questions": q, "answers": {"act": "true" if act else "false"}, "meta": {...}}` with `s, q = d1_format.act_request(context_text)`. Write `<out>/{train,val,test}.jsonl` (by the game's split, file order = game order then time).
7. **Stats** `<out>/stats.json`: per split `lines`, `act_rate`, `games`, `skipped` counters, `act_rate_when_threat`, `act_rate_within_3s_of_opp_play`; print a table. `<out>/README.md` (generated): format, labels, horizon, that states come from the live builder (D-040).

## 2. Tests — `tests/test_dense_act.py`
Synthetic: a tiny `games.jsonl` / `provenance.jsonl` / `states.jsonl` (2 games, train + val) in `tmp_path`, and a perception `states.jsonl` per video under a fake `runs/` (state rows every 0.2 s with `readiness: match`, own/opponent fields like `tests/test_live_state.py` builds, one opponent `play_event`).
1. `pro_deploy_times` dedupes overlapping `recent_plays` windows and collapses a same-card re-read 0.4 s later.
2. Labels: a deploy at `t + 0.5` → `"true"` for the sample at `t`; at `t + 1.5` → `"false"` (horizon 1.0).
3. Sampling: `--step 1.0` over a 10 s game → 10 or 11 samples; a gap without rows longer than `tol` → counted `no_state`.
4. Incremental events: the opponent play appears in `state.opponent.recent_plays` only for samples after its timestamp.
5. Every line passes `d1_format.validate`; splits follow `games.jsonl`.
6. `main([...])` writes the three files, `stats.json` and `README.md`.

## 3. Smoke
Unit tests only (no perception output exists yet). Do not run `batch_process.py` or download anything.

## 4. Heavy run (Opus, after the overnight perception run)
`build_dense_act.py` → `dataset/v1/act_dense`; then `train_d1.py --variant act --data dataset/v1/act_dense`.

## 5. Report
Per MAC-00 §5.

---
## Amendment 1 (Opus, 2026-11-03) — FIX_LIST #24: one perception match per game
`LiveStateBuilder` resets its events when `match_id` changes, and a game's `[start − 1, end + 1]` window can contain rows of a neighbouring match; `match_id` values also restart in every `states.jsonl` (one perception run per match folder). In step 2, tag every row with `(source file path, match_id)`; keep only the group with the most `state` rows inside `[start, end]` (ties → the group whose first row is closest to `start`); use only that group's `play_event` rows. Count games where more than one group was present (`multi_match_groups`) in the stats. Test: a window with rows from two groups (different files, same `match_id` value) → only the dominant group's rows and events are used, and the opponent deck of the other match never appears.
