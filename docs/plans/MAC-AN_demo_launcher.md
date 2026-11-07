# MAC-AN — one-command demo launcher: asks the deck, loads, says READY, watches iPhone Mirroring and plays Training Camp battles until stopped (spec for one Sonnet implementer)

Author: Opus · 2026-11-07 · For the class showcase. Read `docs/plans/MAC-00_common.md` first. No decision logic changes: policy, tactics, candidates, ranker and the d1 context text stay as they are (D-045 holds).

User: *"package the startup command in an easy-to-use command for me to demonstrate in the presentation, it should run, ask me my cards, evos, heros, then load weights, then let me know when its ready to start. Then it should persistently run till I turn it off. It should wait and just monitor the iphone mirror until I turn it off and if it sees the training camp game then it starts playing."*

## Owns
```
clashbot/demo.py (new)            scripts/demo.sh (new)
cr_perception/trainer_banner.py (new)
clashbot/live.py   (only: a reusable stack builder split out of `main`, a tap gate, per-battle reset hook, an optional
                    `match` field on rows; `main` must behave exactly as before)
clashbot/live_state.py (only: `LiveConfig.evo_cards`, carried into the state like `hero_cards`; and a per-battle reset if
                    one is missing)
tests/test_demo.py (new)   tests/test_trainer_banner.py (new)   tests/test_live_loop.py
```
Another implementer (MAC-AO, dashboard) edits `clashbot/candidates.py` and adds fields to the `decisions.jsonl` row in
`LiveLoop.step`. Do not touch those lines; keep your `live.py` edits away from the decision row.

## Facts you need
- Today's live command (keep these as the launcher's fixed defaults):
  `CLASHBOT_ALLOW_LIVE=1 HF_HUB_OFFLINE=1 ../.venv/bin/python -m clashbot.live --live --source mirroring --calib runs/poc/mirror_calib.json --deck electro-dragon,musketeer,tesla,fireball,knight,zap,ram-rider,mini-p-e-k-k-a --hero-cards musketeer --ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac --policy fast --act-model db/models/act-gbdt --record-shots --seconds 900`
- `live.main` builds source → executor → perception → builder → cards / KG / kg2 / StateIndex / CandidateGenerator /
  tactics / ranker / policy, then `LiveLoop(...).run(max_seconds=...)`. `run` already survives several battles in one
  process (900 s runs played 3–4 battles) and since MAC-AM ends cleanly when the source raises.
- `cr_perception.sources.MirroringSource` raises `RuntimeError("iPhone Mirroring window not found ...")` when the window is
  absent and `quartz_window_grab` raises `RuntimeError("window capture failed ...")` when it closes.
- Perception's screen gate gives `GameState.readiness` in {"menu", "match", None}.
- In Training Camp the opponent banner at the top-left of the arena reads **"Trainer <Name>"** (e.g. "Trainer Cheddar II"
  in pink, above the clan line). See `runs/poc/20261107-111554/shots/d0217_t22.5_a.jpg`; every `*_a.jpg` in
  `runs/poc/2026110*/shots/` is a Training Camp frame. Ladder / friendly battles show a player name instead.
- The double live guard (D-031) is unchanged: taps need `--live` AND `CLASHBOT_ALLOW_LIVE=1`; only `clashbot/executor.py`
  sends input.
- D-050 (this spec): the launcher taps only in a battle whose opponent banner reads "Trainer"; any other battle is watched
  and logged, never tapped. Unknown = not Training Camp (fail closed).

## N1. Prompts (`clashbot/demo.py`)
`python -m clashbot.demo [--live] [--deck ...] [--evo-cards ...] [--hero-cards ...] [--yes] [--dashboard] [--out DIR]`
1. Print `TOS_NOTE`. If `runs/demo/last_setup.json` exists, show it and accept Enter to reuse it.
2. "Your 8 cards (comma-separated):" — resolve each name against `load_cards()` slugs and display names: lower-case, drop
   spaces / dots / hyphens / apostrophes, then exact match, else `difflib.get_close_matches(cutoff=0.75)` with one clear
   winner. Accept e.g. "mini pekka", "Mini P.E.K.K.A", "e dragon"? (only if unambiguous), "electro dragon". Re-ask on an
   unknown or ambiguous name (list the 3 closest), on duplicates, or when the count is not 8. Echo the resolved slugs.
3. "Which are evolutions? (Enter for none):" — a subset of the deck; only cards whose card index has `has_evolution`.
4. "Which are heroes? (Enter for none):" — a subset with `has_hero_variant`; a card cannot be both evo and hero.
5. Summary + "Start with this deck? [Y/n]". Save to `runs/demo/last_setup.json`.
6. The flags skip the matching prompt (`--yes` skips the confirmation) so tests and scripts run without stdin.

Evolutions: add `evo_cards: list[str]` to `LiveConfig` (default empty) and carry it into the state's own dict next to
`hero_cards`. Find every consumer of the own deck's variants (e.g. `unit_stats` variant lookups, `card_roles`) and report
which ones could use it; wire it only where a variant is already supported for an own unit and does not change the d1
context text (the text format is what the ranker was fine-tuned on). Record evo/hero in `config.json`.

## N2. Load, warm up, READY
1. Split `live.main`'s build block into `build_stack(args, source=None) -> Stack` (dataclass with the built parts) used by
   both `main` and the launcher. `main`'s behaviour, flags, prints and exit codes must not change (existing tests).
2. Load every model before looking for the window: KataCR detector, d1 ranker, act-gbdt, KG/kg2, StateIndex, tactics. If
   `Perception` needs the source (window rect), build the detector first and the `Perception` object when the window
   appears; rebuild only the source-bound parts when the window comes back.
3. Warm-up: one detector pass on a blank 720×1280 frame and one d1 ranking of a synthetic 4-candidate prompt; print each
   load and warm-up time.
4. Then print a clear banner:
   `READY — open iPhone Mirroring and start a Training Camp battle. Taps: ON|OFF (reason). Ctrl-C to stop.`
   Taps OFF reasons: no `--live`, or `--live` without `CLASHBOT_ALLOW_LIVE=1` (then print the D-031 message and continue
   watch-only; do not exit).

## N3. Watch loop (runs until Ctrl-C)
States and the one line printed on each change (no per-frame output):
- `WAITING_WINDOW` → "waiting for the iPhone Mirroring window…": try to open the source every 2 s.
- `WATCHING` → "watching (no battle)": frames flow through perception; nothing is decided while readiness ≠ "match".
- `CHECKING` (readiness turned "match") → read the banner (N4) on up to 10 frames over ≤ 5 s.
- `PLAYING` → "Training Camp battle vs Trainer <name> — playing" (or "— watch only (taps OFF)").
- `NOT_TRAINING_CAMP` → "this battle is not Training Camp — watching only (Supercell ToS)".
- battle end (readiness ≠ "match" for ≥ 3 s) → "battle over: <duration>, <decisions> decisions, <taps> taps" → `WATCHING`.
- capture error / window closed → `WAITING_WINDOW`; reopen and continue in the same session directory.
- Ctrl-C → finish like the time limit (summary written, shots closed, files closed), print the session totals; a second
  Ctrl-C exits at once.

Implementation:
- `LiveLoop` gets `tap_gate: Callable[[], bool] | None = None`. When it returns False the loop still decides and logs, but
  the executor is not called: the entry is `{"sent": False, "status": "gated", "gate": <reason>}`. No gate = today's
  behaviour. Prefer driving `LiveLoop.step` from the launcher's own frame loop over changing `run`; if you change `run`,
  `main` must behave as before.
- Per-battle reset at the start of each battle: opponent deck evidence, hand tracker, elixir estimate, Kalman tracks,
  recent plays, the ability timer, the policy's per-battle state. First check what already resets on the menu→match
  transition (perception gate `entered`, `cr_perception/context.py`); add an explicit `reset_battle()` only for what does
  not, and list in the report what you found and changed.
- Session dir `runs/demo/<YYYYmmdd-HHMMSS>/` (or `--out`): the usual files appended across battles, every row of
  `decisions.jsonl` / `holds.jsonl` / `frames.jsonl` gets `"battle": <n>` (None outside a battle), plus `battles.jsonl`
  (one row per battle: n, t_start, t_end, banner text, verdict, decisions, taps).
- Performance: the watch loop adds < 1 ms per frame outside `CHECKING`; banner OCR runs only in `CHECKING`.
- `--dashboard`: if `clashbot.viz` is importable (MAC-AO), start `python -m clashbot.viz --follow runs/demo --open` as a
  subprocess after READY and stop it on exit; otherwise print "dashboard not installed". Never import it in-process.
- `scripts/demo.sh`: `cd` to the repo, `export HF_HUB_OFFLINE=1`, `exec <repo>/../.venv/bin/python -m clashbot.demo "$@"`
  (fall back to `python3` when that venv is absent). The presenter types: `CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live`.

## N4. Trainer banner (`cr_perception/trainer_banner.py`)
- `banner_box(frame_shape, calib) -> (x0, y0, x1, y1)`: the opponent name line at the top-left of the arena, as fractions of
  the game rect from the calibration (measure it on the shots listed above; the shots are the game rect plus a text
  header — account for that offset).
- `read_banner(frame, calib, ocr) -> str`: crop, upscale ×2, OCR with the OCR engine the HUD already uses.
- `is_trainer(text) -> bool`: true when a token is within edit distance 2 of "trainer" (OCR noise: "Traner", "TRAINER",
  "Trainer:"). Never true on empty text.
- Verdict: Training Camp when ≥ 2 of the checked frames are `is_trainer`; else not.
- Measure it: run `read_banner` on every `*_a.jpg` in `runs/poc/2026110*/shots/` (cropped to the game rect) and report the
  hit rate (target ≥ 95% of frames, 100% of battles), the OCR time per call, and 5 example texts.

## Tests
- `test_trainer_banner.py`: `is_trainer` table (true: "Trainer Cheddar II", "TRAINER", "Traner Cheddar"; false: "", "36ono
  scope", "Tyler", "Training"? — decide and document); crop on 3 real shots (skip when `runs/` is absent).
- `test_demo.py`: name resolution (fuzzy, ambiguous, duplicate, count ≠ 8), evo/hero validation, reuse of the last setup,
  the guard (no env → taps OFF and continues), and the state machine with a fake source and fake perception: window
  missing → appears → match with "Trainer X" → taps go to the executor → match ends → match with "Tyler" → gated, no
  executor call → window raises → waits → returns → Ctrl-C → summary written, exit 0. Battle numbers on rows.
- `test_live_loop.py`: `tap_gate` False → no executor call, status "gated"; no gate → unchanged.
- Full suite green: `../.venv/bin/python -m pytest -q --deselect tests/test_perception_latency.py`.

## Verification
- Video dry run through the launcher's code path: `python -m clashbot.demo --deck <the deck> --hero-cards musketeer
  --evo-cards tesla,electro-dragon --yes --source video --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60 --seconds 180 --out runs/smoke/AN_demo`
  (add `--source/--video/--start/--seconds` as hidden test flags). That video is a pro ladder game, so the banner check
  must say NOT_TRAINING_CAMP and no executor call happens — report it. Then the same with a test-only
  `--assume-training-camp` flag (hidden, refused together with `--live`) to show decisions flow.
- Do NOT run anything with `--live` or `CLASHBOT_ALLOW_LIVE=1`.

## Report
Branch, commit, files; the reset audit; banner hit rate and OCR ms; load and warm-up times; the two verification runs
(states printed, decisions, gated count); deviations and why; anything for Opus.
