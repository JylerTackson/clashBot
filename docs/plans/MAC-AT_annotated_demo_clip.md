# MAC-AT — annotated demo clip: the agent's decisions drawn over a held-out pro video (spec for one Sonnet implementer)

Author: Opus · 2026-11-08 · For the submission: graders should be able to watch the agent run with no app, phone, emulator or models. The clip is rendered from finished MAC-AQ run dirs, so no model is needed. Tooling only; code freeze D-051 holds (no decision code changes). Read `docs/plans/MAC-00_common.md` first.

User: *"so they can view the agent being ran on the videos without downloading the app and running the emulator … can you show me the annotated demo clip"*

## Owns
```
tools/eval/render_clip.py (new)    tests/test_render_clip.py (new)
dist/media/ (outputs; gitignored)
```

## Inputs
- A MAC-AQ run dir `runs/eval/vs_pro/<game_id>/` with:
  - `config.json`: video path, start, stride, calib;
  - `decisions.jsonl`: t, key, tile, mode, reason, candidates with sources and notes, probs, executor status, tactics_lines, kg_lines, retrieval;
  - `holds.jsonl`;
  - `pro_plays.jsonl`.
- The source video in `data/videos/<video_id>/`. The calibration is the one the run used (see its config: per-video calib under `runs/videos/<video>/match_0/calib.json`).
- Run time equals video seconds (MAC-AQ report).

## Output
`python -m tools.eval.render_clip --run runs/eval/vs_pro/<game_id> [--start S --seconds 90 | --auto 90] --out dist/media/<name>.mp4`
- `--auto N` picks the N-second window with the most bot acts matched to pro plays. Ties go to windows containing one of the MAC-AQ example moments in `docs/data/vs_pro_v1.md`.
- Frame layout, 1920×1080, 30 fps (or the source fps), H.264 so it plays in QuickTime and browsers:
  - **Left:** the game area cropped from the source frame with the calibration's game rect and scaled to full height.
    - The bot's tap is drawn at its tile: a cyan ring that fades over 1.5 s, labelled with the card.
    - The pro's play is drawn at its tile: a magenta ring labelled "pro: card".
  - **Right panel, black with large white text:**
    - game clock, and the bot's mode (defend / punish / counter / leak / spell / learned / hold);
    - the latest decision: card · lane · zone, tile, the policy reason;
    - up to 4 candidates, each with its d1 probability as a bar and its source chips (tac / knn / kg);
    - the chosen candidate's note;
    - one tactics line, and the first KG line if present;
    - the top 3 retrieved pro states (pro's play, fused score).
  - **Bottom strip:** "pro played X at t (Δt = …)" with a badge: green = same card, amber = different card, red = the bot did not act. While the bot holds, show its hold reason from holds.jsonl.
- First 4 s: a title card. "Clashbot on a held-out pro game (open loop): what the bot would do vs what the pro did", then game id, deck and model. Add a legend for the rings and badges.
- Last 3 s: the game's headline from vs_pro_v1 (recall ±2 s, same-card rate).
- No audio. Use cv2 only (VideoWriter with fourcc `avc1`). If `avc1` is unavailable, write `mp4v` and say so.
- Verify:
  - the file reopens with cv2 with the right frame count;
  - `mdls -name kMDItemCodecs -name kMDItemDurationSeconds` reports H.264 and the duration;
  - `qlmanage -t` makes a thumbnail. Run it only if `pgrep -f clashbot.demo` finds nothing, because Quick Look takes window focus.

## Clips to render
1. `dist/media/clashbot_vs_pro_<game>_90s.mp4`, from the game with the most "agree" examples in vs_pro_v1.md, using `--auto 90`.
2. `dist/media/clashbot_vs_pro_<game2>_90s.mp4`, another test game that includes one "disagree" example.

Keep each under 2 minutes (third-party footage: short excerpts only) and under about 40 MB.

## Tests
- tile → pixel mapping round trip with a calib fixture;
- window picking on a toy run;
- panel text built from a fixture decision row;
- a 2-second render of a synthetic video, which reopens with the right frame count.

Run the full suite under the heavy lock (`/usr/bin/lockf <scratchpad>/heavy.lock ...`).

## Report
Branch, commit, files; the two clip paths, durations, sizes, codecs; 3 still frames saved as PNG (for Opus to look at), paths only; deviations.
