# Report placeholders and checks still open

`clashbot_report.tex` marks every missing result with `\pending{TBD: ...}` (red). 38 occurrences, grouped below by the file they will come from. Build: `cd docs/report && tectonic -X compile clashbot_report.tex` (local, XeTeX with TeX Gyre Times-compatible fonts) or pdflatex + bibtex on Overleaf. State on 2026-11-07: compiles with 0 errors, 0 overfull boxes and 0 BibTeX errors; 9 PDF pages, of which the content runs to about 1/3 of page 8 (about 7.3 pages) and the references start on page 8.

## 1. Placeholders (fill, then delete the `\pending`)

| # | Where in the .tex | Placeholder | Source to fill it from |
|---|---|---|---|
| 1 | Abstract, last sentence | `trial of [TBD: 10] battles it won [TBD: k]` | Training Camp trials tonight (D-052): the user's ~10 battles through `scripts/demo.sh --live`; the outcomes go into `docs/data/BENCHMARKS.md` sec. 16 (or a `live_trials_v1.md`) |
| 2 | Introduction, contribution 4 | `trial of [TBD: 10] battles` | same as 1 |
| 3 | Table II (course), Agents row | `win rate [TBD: k/10]` | same as 1 |
| 4 | Table II (course), Knowledge-conflicts row | `E2 follow-rate [TBD]` | `docs/data/context_conflict_v1.md` (MAC-AR R2) |
| 5 | Sec. V-E text | `Table VI: [TBD from context_conflict_v1 ...]` (one sentence: which conditions hurt, by how much) | `docs/data/context_conflict_v1.md` |
| 6 | Table VI (`tab:conflict`), 12 cells | C0-C4 accuracy [CI], delta vs C0 (McNemar p), follow-rate | `docs/data/context_conflict_v1.md`; if it reports val too, keep test only in the table |
| 7 | Sec. V-H text | `Table VIII: [TBD from vs_pro_v1 ...]` | `docs/data/vs_pro_v1.md` (MAC-AQ): games run, minutes, headline |
| 8 | Table VIII (`tab:vspro`), caption `[TBD: n]` + 8 cells | recall, precision (+-2 s), acts/min bot vs pro, same card, two baselines, same lane, top hold reason | `docs/data/vs_pro_v1.md` |
| 9 | Table IX(a) (`tab:live`), last row | `total, 3 video repetitions (mean +- sd) [TBD: latency_v2]` | `docs/data/latency_v2.md` (MAC-AP P3, `runs/rerun/lat_{1,2,3}`); optionally replace the single dry-run column by the latency_v2 means |
| 10 | Table IX(b), 4 rows | trial battles / wins; win rate with exact 95% CI; crowns for / against per battle; decisions / taps per battle | trials (as 1); CI = Clopper-Pearson; decisions and taps from each trial run's `decisions.jsonl`; also add tap landing via `tools/eval/review_shots.py` |
| 11 | Sec. V-I, "Trial." paragraph | n, k, exact CI, crowns, tap landing, one-sided Fisher p vs KataCR 1/20 | trials (as 1). Fisher p for n = 10 is in `docs/research/prior_bots_comparison.md` sec. 5 (k = 3: 0.095, k = 4: 0.031, k = 5: 0.009); recompute if n is not 10 |
| 12 | Table X (`tab:prior`), ours column | `[TBD: k/10]`, `[TBD: lo--hi]` | trials (as 1) |
| 13 | Conclusion, 2nd sentence | `[TBD: and k of 10 in the trial]` | trials (as 1) |

When the trial number lands, also re-read the abstract and conclusion wording ("won its first Training Camp battle") so the claim matches k (do not claim superiority over KataCR with fewer than 4 wins of 10).

## 2. Numbers read from files that were not yet merged into `docs/data`

These were copied from the MAC-AP worktree `.claude/worktrees/agent-aad7f4da136201045/docs/data/` (files dated 2026-11-07 13:44-17:31) and the MAC-AR worktree `.claude/worktrees/agent-a362518d25f251287/docs/data/` (13:59). The .tex comments mark them `[pre-merge]`. After the merge, diff each against the merged copy:

- `stats_audit_v1.md/.json/_risk_coverage.csv`: Table III (all CI rows), paired deltas in Sec. V-B, calibration, selective prediction, ensemble, popularity strata, Figs. 3 and 4b.
- `feature_ranker_v2.md` (3-seed mean, gain shares), `act_gbdt_v2.md` (Table VII), `act_policy_v2.md`, `tactics_eval_v2.md`, `arbiter_v2.md`, `edit_eval_v2.md`, `opp_model_v1.md`, `eta_eval_v1.md`, `tap_landing_20261107-013441_v2.md`, `tap_landing_20261107-111554_v2.md`, `BENCHMARKS.md` sec. 11/13.
- `answer_rules_v1.md` (MAC-AR): Table V(b) and the E6 paragraph.
- Then re-run `../.venv/bin/python docs/report/make_figures.py` **without** `--extra-data` (it was run with `--extra-data .claude/worktrees/agent-aad7f4da136201045/docs/data` because `stats_audit_v1*` was not in `docs/data` yet).

## 3. Facts to double-check

| Item | Issue |
|---|---|
| Sec. VI, "14 of 27 decisions were skipped because the hand read as unknown" | `BENCHMARKS.md` sec. 13 and `packaging_and_benchmarks_audit.md` 2.2 say 14 of 27; `course_mapping_audit.md` A3 says 15 of 27. Used 14 (two sources); check `runs/poc/20261105-172506/decisions.jsonl` |
| Live-state parity numbers (Sec. VI) | `live_state_parity_v1.json` is stale (PBA row 18; H4 re-run was not done). The text says "measured before later perception fixes". Replace if a v2 lands |
| Table IX(a) video dry-run column | from `runs/smoke/AM_merged/summary.json` via PBA row 26 (one sample); latency_v2 should supersede it |
| d1 described as "a 16-layer transformer trunk with a head that emits one logit per option marker" | from `docs/plans/MAC-T_d1_finetune.md` sec. 0 (the remote code calls it `model.encoder`); confirm against the model card |
| Number of tests | not cited in the paper (STATUS: 1417 passed, 1 isolation failure #61 at the last merge). Add to the conclusion if wanted once #61 is fixed |

## 4. Bibliography

- Appended 14 entries to `refs.bib` (304 -> 318, no existing key changed): `wu2025nonembedded`, `wu2024katacr`, `houseworth2025claudeRoyale`, `liquidai2026d1omni`, `ke2017lightgbm`, `jocher2023yolov8`, `rapidai2021rapidocr`, `burges2010lambdamart`, `kalman1960`, `vinyals2019alphastar`, `berner2019dota`, `chen2021dt`, `qwen2025qwen3`, `xiao2023cpack`.
- To verify: `liquidai2026d1omni` (year 2026 is the access year; release date unknown), `rapidai2021rapidocr` (year 2021 and title wording are mine; check the GitHub repo), `wu2024katacr` (title is descriptive; the repo has no formal title).
- Edited one pre-existing comment line in the `refs.bib` header (`(BibTeX @misc, allowed)` -> `(BibTeX misc entries, allowed)`): BibTeX treats any `@` as an entry start, so that comment raised a BibTeX error. No entry changed.
- `arXiv` ids for the new `@misc` preprints are in `howpublished` because IEEEtran.bst ignores `eprint`.
