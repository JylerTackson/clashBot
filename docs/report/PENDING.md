# Report placeholders and checks still open

`clashbot_report.tex` marks every missing result with `\pending{TBD: ...}` (red). After revision pass 2 (2026-11-07) there are **13 occurrences**, all from the Training Camp trial. Build: `cd docs/report && tectonic -X compile clashbot_report.tex` (local, XeTeX with TeX Gyre Times-compatible fonts) or pdflatex + bibtex on Overleaf. State after pass 2: compiles with 0 errors, 0 overfull boxes and a clean BibTeX run; 10 PDF pages; the content ends near the bottom of the right column of page 8 (about 7.9 content pages; PDFKit puts the References heading on page 8 at y = 96 pt of 792), so the trial text has room for only about 2 extra lines.

## 1. Placeholders (fill, then delete the `\pending`)

Source for all of them: the user's ~10 Training Camp battles tonight (D-052) through `CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live`, on the frozen code; write the outcomes into `docs/data/BENCHMARKS.md` sec. 16 (or a `live_trials_v1.md`) first, then copy them here with a `% src:` comment.

| # | Where in the .tex | Placeholder | How to fill |
|---|---|---|---|
| 1 | Abstract, last sentence | `trial of [TBD: 10] battles it won [TBD: k]` | n and k |
| 2 | Introduction, contribution 4 | `trial of [TBD: 10] battles` | n |
| 3 | Table II (course), Agents row | `win rate [TBD: k/10]` | k/n |
| 4 | Table IX(b) (`tab:live`), 4 rows | trial battles / wins; win rate with exact 95% CI; crowns for / against per battle; decisions / taps per battle | CI = Clopper-Pearson; crowns from the user's notes or end-of-battle frames (MAC-AS); decisions and taps from each trial run's `decisions.jsonl` (taps = `executor.sent`) |
| 5 | Sec. V-I, "Trial." sentence | n, k, exact CI, crowns, tap landing, one-sided Fisher p vs KataCR 1/20 | tap landing via `tools/eval/review_shots.py runs/...`; Fisher p for n = 10 is in `docs/research/prior_bots_comparison.md` sec. 5 (k = 3: 0.095, k = 4: 0.031, k = 5: 0.009); recompute if n is not 10 |
| 6 | Table X (`tab:prior`), ours column | `[TBD: k/10]`, `[TBD: lo--hi]` | as 4 |
| 7 | Conclusion, 2nd sentence | `[TBD: and k of 10 in the trial]` | k, n |

When the trial number lands, re-read the abstract and conclusion wording ("won its first Training Camp battle") so the claim matches k, and do not claim superiority over KataCR with fewer than 4 wins of 10. Recount pages afterwards: the content must stay within page 8.

## 2. Preliminary vs-pro numbers (`% VS_PRO_PRELIM`)

`docs/data/vs_pro_v1.md` covers 8 of the 24 usable test games (516 pro plays, 717 bot taps, 31 min); the remaining games run overnight and `eval_vs_pro.py --score` rewrites the file. Every number taken from it carries a `% VS_PRO_PRELIM` comment (18 lines: abstract, Table II Agents row and its source comment, Sec. V-H setup and results paragraphs, Table VIII caption and all 8 rows, Discussion "Perception caps agreement", Limitations). When the file is rewritten:

- `grep -n VS_PRO_PRELIM docs/report/clashbot_report.tex` and replace each number from the new Headline, Timing and "When the bot held" tables; the sentences are written so only numbers change (game and play counts are stated as "N of the 24 usable test games", "N test games, P pro plays").
- Re-check the few directional words that could flip with more games: the bot's recall above the shifted-tap null (Sec. V-H, Table II Agents row), "the ceiling explains most of the gap", and the two hold reasons named as most frequent.
- Remove the `VS_PRO_PRELIM` comments once all usable test games are in.

## 3. Facts to double-check

| Item | Issue |
|---|---|
| Sec. VI, "14 of 27 decisions were skipped because the hand read as unknown" | `BENCHMARKS.md` sec. 13 and `packaging_and_benchmarks_audit.md` 2.2 say 14 of 27; `course_mapping_audit.md` A3 says 15 of 27. Used 14 (two sources); check `runs/poc/20261105-172506/decisions.jsonl` |
| Live-state parity numbers (Sec. VI, "older parity check", hand slots 0.57) | `live_state_parity_v1.json` is stale (PBA row 18; the H4 re-run was not done, BENCHMARKS appendix A). Replace if a v2 lands |
| "Over all 931 decisions the total is 347/613 ms" (Sec. V-I) | from `course_mapping_audit.md` C7 line 216; `latency_v2.md` has per-run rows only, not the pooled figure |
| d1 described as "a 16-layer transformer trunk with a head that emits one logit per option marker" | from `docs/plans/MAC-T_d1_finetune.md` sec. 0 (the remote code calls it `model.encoder`); confirm against the model card |
| Number of tests | not cited in the paper (STATUS: 1417 passed, 1 isolation failure #61 at the last merge). Add to the conclusion if wanted once #61 is fixed |

## 4. Bibliography

- `refs.bib` was not changed in pass 2; the new citations (`shi2024cad`, `wang2025adacad`, `ru2024ragchecker`, `huang2025esca`, `liu2022tlogic`) were already in it.
- Still to verify from pass 1: `liquidai2026d1omni` (year 2026 is the access year; release date unknown), `rapidai2021rapidocr` (year 2021 and title wording are mine; check the GitHub repo), `wu2024katacr` (title is descriptive; the repo has no formal title).
- `arXiv` ids for `@misc` preprints are in `howpublished` because IEEEtran.bst ignores `eprint`.

## 5. Done in pass 2 (for the record)

- Every number tagged `[pre-merge]` was checked against the merged `docs/data` files (`stats_audit_v1`, `feature_ranker_v2`, `act_gbdt_v2`, `act_head_v2`, `act_policy_v2`, `tactics_eval_v2`, `arbiter_v2`, `edit_eval_v2`, `opp_model_v1`, `eta_eval_v1`, `answer_rules_v1`, the tap-landing files, `BENCHMARKS.md` sec. 11/13): no mismatch; tags removed.
- Filled: E2 (Table VI and Sec. V-E, from the `joint_tac_v2` replicate, which matches Table III; the 11-06 export is quoted as a replicate), E6 framing, latency (Table IX(a) dry-run column = `latency_v2` 3-run means, micro-benchmarks in Sec. V-I), tap landing (75/84 and 78/95 for the two newest runs, 95/100 named as the older run 233907), vs-pro (Table VIII, Sec. V-H). The ensemble gain is not claimed.
- Figures rebuilt with `../.venv/bin/python docs/report/make_figures.py` (no `--extra-data`). No vs-pro figure: Table VIII carries the comparison and the page budget has about 2 lines left.
