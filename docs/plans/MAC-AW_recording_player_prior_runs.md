# MAC-AW — watch the agent on a recording in real time, and browse every prior game in the grader build (spec for two Sonnet implementers working in parallel)

Author: Opus · 2026-11-09 · Code due on Canvas in about 20 h. Read `docs/plans/MAC-00_common.md` first. Code freeze D-051 holds. Tooling only: the viz dashboard, the video frame source (behind a flag), packaging. No decision-logic changes. Mac behaviour must not change unless a new flag is passed.

User:
- *"in the zip, we need to include all prior runs so they can view all prior games in the dashboard."*
- *"lets say they are viewing a recording, we need to pop up the recording and show it in a window in real time and show the bots moves in real time (it doesnt have to be a popup it is just to show that it is working)"*

## Part R — real-time recording player (implementer 1)
Owns:
- `cr_perception/sources.py` (VideoFrameSource: a `realtime` option);
- `clashbot/live.py` (only: a `--realtime` CLI flag passed to the video source, and a `live_meta.json` written at run start);
- `clashbot/viz/` (server, data, static);
- `docker/entrypoint.sh` (add `--realtime`);
- tests: `tests/test_viz.py` (append), `tests/test_video_realtime.py` (new).

1. **Real-time pacing** (`VideoFrameSource(realtime=True)`, `--realtime` on `clashbot.live --source video`):
   - Frame k (video time t_k) is delivered no earlier than wall start + (t_k − t_0).
   - When the loop falls behind, skip to the newest frame whose time has passed. Drop frames, never queue them: the agent sees the recording like a live stream, as on the iPhone. `--stride` still caps the rate.
   - Record dropped-frame counts in the summary.
   - Default off. The Mac's non-realtime behaviour and every existing test stay unchanged.
2. **`live_meta.json`** in the run dir at the first frame:
   - wall-clock start (epoch s);
   - video path and the video time of the first frame;
   - fps, stride and realtime flag;
   - the calibration's arena rect and grid. Use the same mapping the shot recorder / executor use for tile ↔ pixel, so the browser can map a tile to a video pixel.
3. **Server:**
   - `GET /api/run/<id>/video`: the run's source video, with HTTP Range support (206, `Content-Range`, `Accept-Ranges`) so `<video>` can play and seek. Serve only the path named in that run's config, and only `.mp4`.
   - `GET /api/run/<id>/livemeta`.
   - Path safety as for shots.
4. **Page, a "Recording" panel** at the top, full width beside the screen panel or replacing it when a video exists:
   - A `<video>` of the run's clip with a `<canvas>` overlay.
   - **Live run** (follow mode, run not finished): keep `video.currentTime` at (now − wall start) + first-frame video time, so it plays in real time. Each new decision appears as an overlay the moment it arrives:
     - a cyan ring at its tile with the card label and a fading trail;
     - a banner with mode, card, lane and reason.
     - Holds show the hold reason in a small strip.
     - Show a "LIVE — the agent is watching this recording now" badge and the agent's lag: video time of the latest processed frame vs the playhead.
   - **Finished run** (replay): normal video controls. Overlays are drawn by decision time as it plays. Clicking a decision in the timeline seeks there. Pro plays (`pro_plays.jsonl`) appear as magenta rings when present.
   - **Run without a video** (live iPhone runs, which only have decision screenshots): the panel becomes a **screenshot player**. Play / pause / speed 1×–4× steps through the decision frames in game time: hold each until the next decision's time, scaled by speed. The tap is drawn on the screenshot by the recorder, and the decision banner shows above it.
   - No external JS. It must work in Chrome, Safari, Firefox and Edge.
5. **Docker:** the entrypoint adds `--realtime`. The default `DEMO_SECONDS` stays 60. Measure on this Mac in Docker: wall time ≈ 60 s + load, decisions, dropped frames. Update `docs/DOCKER.md` ("open http://localhost:8765: the Recording panel plays the clip in real time and the agent's taps appear as it makes them").
6. **Tests:**
   - pacing with a fake clock (frames delivered on time; frames dropped when the loop is slow; never ahead);
   - the Range endpoint (206, correct bytes, 416 out of range, rejecting a path not in the config);
   - livemeta;
   - screenshot-player data (decision frames ordered by time).
7. **Verify:** a 30 s Docker run with the dashboard. Fetch the endpoints. Take headless screenshots (as MAC-AO did; do not use the user's Chrome) of the Recording panel mid-run and after. Save 2 PNGs to `/private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/aw_shots/`.

## Part P — prior runs in the grader build (implementer 2)
Owns:
- `scripts/pack_prior_runs.py` (new);
- `scripts/make_docker_submission.sh`;
- `scripts/docker_artifacts.py` (only if needed);
- `docker/Dockerfile` (only the COPY of the prior-runs pack and the env);
- `clashbot/viz/__main__.py` and `clashbot/viz/data.py` (only: extra run parents from `--parents` / env `CLASHBOT_VIZ_PARENTS`; nothing else in viz — Part R owns the rest);
- `docs/DOCKER.md` (a "Browse every prior game" section);
- `tests/test_pack_prior_runs.py` (new).

1. **Which runs:**
   - **Every live game:** each `runs/poc/2026*` with a non-empty `decisions.jsonl`, plus `runs/demo/20261107-213709` (the 10-battle trial). Exclude empty or test sessions, and say which.
   - **All 24 bot-vs-pro runs:** `runs/eval/vs_pro/*`.
   - **Smoke runs:** only `runs/smoke/AO_demo` and `runs/smoke/AM_merged` (dry runs on the test clip).
   - Keep the parent grouping (`poc`, `demo`, `vs_pro`, `smoke`) so the dashboard's run picker groups them.
2. **What each run keeps:**
   - `config.json`, `summary.json`, `decisions.jsonl`, `holds.jsonl`, `battles.jsonl` / `matches` files, `pro_plays.jsonl`;
   - `shots/` with each JPEG re-encoded at 60% scale, quality 70;
   - the shots index file.
   - Drop `frames.jsonl` (the dashboard does not need it; check `clashbot/viz/data.py`) and any `.mp4`.
   - Rewrite absolute paths inside `config.json` only if the dashboard reads them; never rewrite decision data.
   - Report the size before and after, per group.
3. **Size budget:** the packed runs should come to ≤ about 250 MB on disk, so the zip stays near or under ~450 MB. If it is over, lower the JPEG quality or scale for the vs-pro shots first, then keep shots for only the 6 example games and drop the rest's shots (decisions stay). Say what you chose.
4. **Placement:**
   - The pack goes in the zip at `prior_runs/<group>/<run>/`.
   - The Dockerfile copies it to `/app/prior_runs`, read-only.
   - The entrypoint starts viz with `--parents /data/runs,/app/prior_runs/poc,/app/prior_runs/demo,/app/prior_runs/vs_pro,/app/prior_runs/smoke`, or the env var. Coordinate the entrypoint change with Part R: Part R owns `docker/entrypoint.sh`, so add only the `--parents` argument line and say so in your report.
   - The live demo run stays the default selected run; the picker lists the others with readable names, e.g. "Live trial 2026-11-07 (10 battles)", "Live 2026-11-07 01:34 (first win)", "Bot vs pro: 6qYQNJ1Uaeg m0.3".
5. **Native Mac:** `scripts/setup_mac.sh` copies or links `prior_runs/` into `runs/` so `python -m clashbot.viz` on the Mac lists them too.
6. **Tests:**
   - the pack selection and the file filter;
   - JPEG re-encode keeps the count;
   - viz lists runs from multiple parents;
   - a fixture pack loads in the dashboard API.
7. **Verify:** rebuild the zip with Part R's changes once both are merged. Opus merges; just make `make_docker_submission.sh` include `prior_runs/`. Measure the zip size. Load the dashboard in Docker and list the runs.

## Both parts
- Heavy work (docker build/run, the full pytest suite) goes under `/usr/bin/lockf /private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/heavy.lock`.
- No `--live`. No uploads or registry pushes. Never commit run data, images or zips.
- Full suite green.
- Commit on your worktree branch with the attribution lines; do not push.
- Report under 400 words.
