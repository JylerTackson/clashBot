# MAC-AP — clean benchmarks for the report: re-run the stale evals on the frozen code, promote current ones, add the statistics audit (E1) and a latency table (spec for one Sonnet implementer)

Author: Opus · 2026-11-07 · Showcase Monday 2026-11-09. Code freeze D-051: decision logic is frozen at `66917e7`; this package only runs evals and adds eval tooling. Read `docs/plans/MAC-00_common.md` first.

User: *"We have to report clean benchmarks on the report so we will need to redo them so if they are recent then dont clear them."*

## Owns
```
tools/eval/eval_stats.py (new)        tools/eval/latency_table.py (new)
tools/eval/feature_ranker.py (only: a --dump-preds option; defaults unchanged)
docs/data/*_v2.* and new docs/data files listed below (never overwrite or delete an existing docs/data file)
docs/data/BENCHMARKS.md (new: the single table the report and slides quote)
tests/test_eval_stats.py (new)   tests/test_latency_table.py (new)
```
Do not touch `clashbot/` decision code, `db/models/*`, `clashbot/resources/tactics_weights.json`, `dataset/v1/*` existing files.

## P1. Re-run the stale evals (exact commands)
Run the plan in `docs/research/packaging_and_benchmarks_audit.md` §2.3, in its order: C1–C8 (CPU), H0 (joint_tac_v2), H1, H2 (d1 on the HEAD data), H3 (act head; back up and restore `act_threshold.json` exactly as written there), H6 (3 × whole-stack latency dry runs), H7 (promote AL/AM artefacts: copy, never move). Skip H4 (parity), H5 (coach), H8 (2B) unless everything else is done; if you run H8, record the fix but do not change model files.
- Every heavy step (H*, anything loading d1 / KataCR / torch MPS) runs under
  `/usr/bin/lockf /private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/heavy.lock <cmd>`.
- Before each heavy step check the hold flag: if
  `/private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/HOLD_HEAVY` exists, wait (poll every 60 s) — the user is playing live games.
- Record the git HEAD hash and wall time of every run in BENCHMARKS.md.
- If a command fails, fix only the eval script's bug (report it), never the decision code; if it needs decision code, stop that step and report.

## P2. E1 statistics audit (`tools/eval/eval_stats.py`)
As planned in `docs/research/course_gap_analysis.md` §7.3 "E1". Inputs: the per-example outputs of the d1 evals (the `_v2` ones from P1 and the existing `dataset/v1/eval/d1_eval_transformers_*.json`), the placement prior, and the feature rankers' per-line scores (`feature_ranker.py --dump-preds <path>`, same seeds/config as `docs/data/feature_ranker_v1.json`; check its accuracies reproduce v1 within 0.5 pt and say so).
Report on val and test (thresholds chosen on val only):
1. accuracy with 95% CI by cluster bootstrap over `game_id` (10,000 resamples, fixed seed) for: random, popularity, placement prior, logreg, LightGBM, DCN-V2, d1 zero-shot, d1-ft-joint, d1-ft-joint-tac, d1-ft-joint-kg;
2. paired differences with CI and exact McNemar p for: d1-tac vs LightGBM, d1-tac vs d1-joint (tactics notes), d1-tac vs placement prior, d1-kg vs d1-tac;
3. calibration of d1-tac: ECE (15 bins), Brier, a reliability table;
4. selective prediction: risk-coverage curve and the CRAG-style utility (+1 right, 0 abstain, −1 wrong) at the val-chosen threshold;
5. accuracy by popularity tercile of the gold key (train counts);
6. ensemble d1-tac + LightGBM (z-scored per line, weight chosen on val): test accuracy with paired CI vs each member.
Outputs: `docs/data/stats_audit_v1.{json,md}` (+ a CSV of the risk-coverage curve for a plot).

## P3. Latency table (`tools/eval/latency_table.py`)
Aggregate per-stage p50/p95 (capture, perception, live_state, trigger, candidates, rank, execute, total) over: the 3 H6 runs (mean ± sd across runs), and every live run under `runs/poc/2026110*` with a `summary.json` (one row each). Also isolated micro-benchmarks, 200 calls each after warm-up: KataCR detect on MPS vs CPU (one frame), d1-ft-joint-tac 4-option ranking on MPS (transformers), act-gbdt predict, exact kNN vs HNSW ef 40 query, `CandidateGenerator.generate` on 50 real states. Output `docs/data/latency_v2.{json,md}`.

## P4. `docs/data/BENCHMARKS.md`
One page, tables only, every number with its source file and the HEAD hash it was produced at, marked `re-run 2026-11-07` or `kept (v1, still current: <why>)`:
ranker 4-way (with CIs), act timing, retrieval (ablation + HNSW + filters, kept), KGE (kept), next card (kept), knowledge editing, opponent-deck model, tactics eval, arbiter, ETA, tap landing, latency, live history (runs, battles, decisions, taps). Leave a placeholder section "Bot vs pro on video (MAC-AQ)", "Rule mining / context conflict (MAC-AR)" and "Live trials" for Opus to fill.

## Tests
`test_eval_stats.py` (bootstrap on a toy set with known answer, McNemar exact on a 2×2, ECE of a perfectly calibrated toy = 0, ensemble weight picked on val only); `test_latency_table.py` (aggregation on two fake summaries). Full suite green under the lock: `../.venv/bin/python -m pytest -q --deselect tests/test_perception_latency.py -p no:cacheprovider`.

## Report
Branch, commit, files; a table of every step (command, wall time, output, verdict vs the old number); the E1 headline (which differences are significant); latency table headline; anything that failed and why.
