# MAC-AS — submission package: README for graders, quickstart, requirements, a zip builder, and end-of-battle frames in the launcher (spec for one Sonnet implementer)

Author: Opus · 2026-11-07 · Showcase Monday 2026-11-09. Grading: difficulty 30%, code runnability 30%, demo/usable system/benchmarks 20%, course learning 20%. User: *"make sure the code can be zipped and submitted"*; *"everything will be run and hosted on my mac"* (no uploads: weights stay on the Mac). Code freeze D-051: no decision-logic changes. Read `docs/plans/MAC-00_common.md` and `docs/research/packaging_and_benchmarks_audit.md` Part 1 first.

## Owns
```
README.md (rewrite)    docs/QUICKSTART.md (new)    requirements-mac.txt (add missing direct deps only)
configs/mirror_calib.example.json (new: a copy of runs/poc/mirror_calib.json, checked for secrets/personal data)
scripts/make_submission.sh (new)    scripts/check_submission.sh (new)
clashbot/demo.py (only: S4)    tests/test_demo.py (only: S4)
```

## S1. README.md (rewrite; keep it short and true)
Sections: what it is (one paragraph + the pipeline diagram in text); the course map (one table: lecture block → component → file → eval file; take it from `docs/research/course_mapping_audit.md` A/B, built items only); install (Python 3.12, `requirements-mac.txt`, libomp, Xcode CLT, KataCR clone + weights paths/env vars, the model dirs that must exist under `db/models/`); run the demo (`CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live --dashboard`, D-031, D-050, Training Camp only); replay a run in the dashboard; run tests; run the evals (one line each, pointing to `docs/data/BENCHMARKS.md`); repository map (top-level dirs, one line each); data provenance and ToS note. Every command in it must be run once by you and work (CPU-only where possible). Fix the stale statements listed in the audit §1.7.

## S2. docs/QUICKSTART.md — a grader path that runs in ≤ 10 min on this Mac without the iPhone
1. env check script output; 2. `pytest` (the deselect flag); 3. one CPU eval (`tools/eval/eval_tactics.py --out /tmp/...`); 4. a 60 s video dry run with the live configuration on `data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60` into `runs/smoke/quickstart`; 5. open the dashboard on it. Time each step on this Mac and write the times.

## S3. Zip builder
- `scripts/make_submission.sh [OUT=dist/clashbot_submission.zip]`: `git archive --format=zip HEAD` (never zip the working tree: `.claude/` is 10 GB) plus `configs/mirror_calib.example.json` (tracked anyway) and nothing gitignored. Print the size and the file count. `dist/` must be gitignored (add the line).
- `scripts/check_submission.sh ZIP`: unzip into a temp dir, run `python -m compileall -q`, check that README/QUICKSTART/requirements exist, that no file > 50 MB is inside, and grep for secrets patterns (API keys, tokens, `.env`) — print the findings, exit 1 on any.
- Add `huggingface_hub` (direct import) to requirements-mac.txt with the installed version; list `imageio-ffmpeg` as optional in the README. No other dependency changes.

## S4. Launcher: save the end of each battle (`clashbot/demo.py`)
For the live trials' results: when a battle ends (the existing battle-over transition), save the last 3 frames seen after readiness left "match" (or the last match frame if none) as `<session>/battles/<n>_end_{0,1,2}.jpg` (JPEG quality 85), and add `"end_frames": [...]` to that battle's row in `battles.jsonl`. Opus reads the result (win/loss, crowns) from these images. Bounded memory (keep only the last 3 frames), < 1 ms per frame, never blocks the loop (write on the shot recorder's thread or a small thread). Test with the fake source.

## Tests and checks
`tests/test_demo.py` additions for S4; run `scripts/make_submission.sh` and `scripts/check_submission.sh` on the result and paste the output in the report; full suite green under `/usr/bin/lockf /private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/heavy.lock` (wait while `.../scratchpad/HOLD_HEAVY` exists). The 60 s quickstart dry run is heavy: same lock and hold rule.

## Report
Branch, commit, files; quickstart step times; zip size / file count / check output; README commands you ran; deviations.
