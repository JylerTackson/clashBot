# Benchmarks for the report (MAC-AP)

Produced 2026-11-07 on the frozen decision code `66917e7` (D-051); the repository HEAD while the re-runs executed was `54243d7` (`66917e7` plus documentation only); the whole-stack dry runs and the latency step ran at `54243d7` plus the new eval tooling and data files (a work-in-progress commit, `9ca39a7`, recorded in `latency_v2.json` and squashed into the final MAC-AP commit; no decision code differs from `66917e7`). Every number below has its source file in `docs/data/` and one of two marks: **re-run 2026-11-07** (re-produced on the frozen code, HEAD `54243d7`) or **kept (v1, still current: reason)**. Nothing current was overwritten: re-runs are `_v2` files next to the `_v1` ones. For kept rows the hash is the commit that added the file to the repository. Hardware: Apple M5, 32 GB, MPS; the Mac was shared with three other evaluation jobs (heavy jobs serialised by one lock, CPU-only jobs not), so wall times and latencies are upper bounds.

## 1. Ranker 4-way multiple choice (does the model pick the pro's play out of 4 options)

n val 761 / test 766 lines, 26 / 27 games. Accuracy with a 95% CI by cluster bootstrap over `game_id` (10,000 resamples, seed 20261107); full audit (paired tests, calibration, abstention, popularity strata, ensemble): `stats_audit_v1.md` / `.json`. Chance = 0.25.

| model | val acc [95% CI] | test acc [95% CI] | source | status |
|---|---|---|---|---|
| random (one seeded pick; expected 0.25) | 0.230 [0.207, 0.254] | 0.252 [0.218, 0.284] | `stats_audit_v1.json` (recomputed) | kept (v1, still current: baseline, data frozen; file committed at `cd67e95`) |
| popularity (train key counts) | 0.468 [0.412, 0.516] | 0.325 [0.280, 0.374] | `ranker_baselines_v1.json`; recomputed in `stats_audit_v1.json` | kept (v1, still current: ranker files unchanged since 11-03; file committed at `cd67e95`) |
| placement prior | 0.493 [0.430, 0.547] | 0.384 [0.337, 0.431] | `ranker_baselines_v1.json`; recomputed in `stats_audit_v1.json` | kept (v1, still current: ranker files unchanged since 11-03; file committed at `cd67e95`) |
| logreg (seed 0) | 0.543 [0.492, 0.586] | 0.470 [0.423, 0.512] | `feature_ranker_v2.json`, per-line scores `feature_ranker_preds_v2.json` | re-run 2026-11-07 (HEAD `54243d7`); byte-identical to `feature_ranker_v1.json` |
| LightGBM LambdaRank (seed 0) | 0.565 [0.520, 0.604] | 0.482 [0.448, 0.515] | `feature_ranker_v2.json` | re-run 2026-11-07 (HEAD `54243d7`); byte-identical to `feature_ranker_v1.json` |
| DCN-V2 (seed 0) | 0.537 [0.493, 0.576] | 0.403 [0.366, 0.439] | `feature_ranker_v2.json` | re-run 2026-11-07 (HEAD `54243d7`); byte-identical to `feature_ranker_v1.json` |
| d1-omni-600M zero-shot | 0.258 [0.226, 0.287] | 0.221 [0.191, 0.252] | `d1_zero_shot_v1.json` (mlx fp16) | kept (v1, still current: frozen model and data; file committed at `6dbedb8`) |
| d1-ft-joint (no tactics notes) | 0.524 [0.483, 0.561] | 0.483 [0.442, 0.522] | `d1_finetune_joint_eval_v1.json` | kept (v1, still current: model dir 11-03 and `joint` data unchanged; file committed at `6dbedb8`) |
| **d1-ft-joint-tac (live model), HEAD notes** | 0.547 [0.503, 0.584] | 0.507 [0.472, 0.540] | `d1_finetune_joint_tac_eval_v2.json` | re-run 2026-11-07 (HEAD `54243d7`): `joint_tac_v2` built from the HEAD engine |
| d1-ft-joint-kg, HEAD notes | 0.530 [0.480, 0.572] | 0.484 [0.450, 0.518] | `d1_joint_kg_on_tac_v2.json` | re-run 2026-11-07 (HEAD `54243d7`) |

Extra rows (same audit): d1-ft-joint scored on the HEAD tactics lines 0.477 [0.427, 0.523] / 0.461 [0.418, 0.506] (`d1_joint_on_tac_v2.json`, re-run 2026-11-07 (HEAD `54243d7`)); d1-ft-joint-tac on the as-trained 11-05 lines 0.540 [0.498, 0.577] / 0.497 [0.462, 0.529] (`d1_finetune_joint_tac_eval_v1.json`, kept (v1, still current: the as-trained record of the live model; file committed at `78ec6c8`)).

Paired differences on test (accuracy points, a minus b; cluster-bootstrap CI; exact McNemar p / cluster-bootstrap p):

| a vs b | delta [95% CI] | McNemar p | p boot | reading |
|---|---|---|---|---|
| d1-tac vs LightGBM | +2.5 [-1.4, +5.9] | 0.263 | 0.212 | not separable |
| d1-tac vs d1-ft-joint | +2.3 [-0.9, +5.3] | 0.159 | 0.156 | not separable |
| d1-tac vs placement prior | +12.3 [+8.0, +16.7] | <0.001 | <0.001 | significant |
| d1-kg vs d1-tac | -2.2 [-4.7, +0.5] | 0.155 | 0.119 | not separable |

d1-tac calibration (test): ECE 0.048 (15 bins, CI 0.039-0.089), Brier 0.616. Holding when unsure (threshold 0.483 picked on val): coverage 0.40, accuracy on answered 0.630, utility (+1/0/-1) 0.104 [0.060, 0.147] vs 0.013 [-0.056, 0.079] when answering everything. Ensemble d1-tac + LightGBM (d1 weight 0.35 picked on val): test 0.505 [0.472, 0.535], -0.1 [-3.0, +2.9] points vs d1-tac, +2.3 [-0.4, +5.1] vs LightGBM.

## 2. Act timing (play / wait)

| measure | value | source | status |
|---|---|---|---|
| reactive rules, offline on pro states: balanced acc / act rate (test, all) | 0.545 / 0.220 (pro play rate 0.941; v1 stale value was 0.598) | `act_policy_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`) |
| trigger / always baselines (same split) | balanced acc 0.518 / 0.500 | `act_policy_v2.md` | re-run 2026-11-07 (HEAD `54243d7`) |
| LightGBM act model, test balanced acc at horizon 1 / 2 / 3 s | 0.594 / 0.607 / 0.613 (AUC 0.632 / 0.659 / 0.672); 0.05 ms per decision | `act_gbdt_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); model files re-trained to a scratch dir are byte-identical to `db/models/act-gbdt` |
| rules comparator, same table (h = 1 / 2 / 3 s) | 0.506 / 0.509 / 0.510 (v1 stale: 0.529 / 0.533 / 0.530) | `act_gbdt_v2.md` | re-run 2026-11-07 (HEAD `54243d7`) |
| rules + GBDT (`fast` policy), same table | 0.571 / 0.584 / 0.593 (v1 stale: 0.562 / 0.577 / 0.583) | `act_gbdt_v2.md` | re-run 2026-11-07 (HEAD `54243d7`) |
| d1 act head, test balanced acc: learned / rules / combined | 0.558 / 0.513 / 0.512 (v1: 0.558 / 0.535 / 0.545) | `act_head_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); `act_threshold.json` restored byte-identical |

## 3. Retrieval (kNN / BM25 / text, exact vs HNSW, filters) - kept

| measure | value | source | status |
|---|---|---|---|
| hit@10 exact key, val / test, struct (kNN) channel | 0.166 / 0.136 (BM25 0.142 / 0.159, text 0.132 / 0.143, all three fused 0.156 / 0.170: channels within noise) | `retrieval_ablation_v1.md` | kept (v1, still current: `retrieval.py` unchanged since 10-28, db built 11-02; file committed at `cd67e95`) |
| card-level hit@10, all three fused, val / test | 0.461 / 0.514 | `retrieval_ablation_v1.md` | kept (v1, still current: `retrieval.py` unchanged since 10-28, db built 11-02; file committed at `cd67e95`) |
| HNSW ef 40 vs exact (10.9 k vectors, k = 10) | recall@10 0.978; 0.31 ms vs 1.51 ms p50 (ef 160: 0.993, 0.91 ms) | `retrieval_ablation_v1.md` | kept (v1, still current: `retrieval.py` unchanged since 10-28, db built 11-02; file committed at `cd67e95`); re-measured in section 12 |
| pre- vs post-filtering | deck pre-filter pool empty for 88.9% of queries; post@10 returns nothing for 50% (deck) / 87% (phase) | `retrieval_ablation_v1.md` | kept (v1, still current: `retrieval.py` unchanged since 10-28, db built 11-02; file committed at `cd67e95`) |

## 4. Knowledge-graph embeddings (held-out COUNTERS link prediction, 3 seeds) - kept

| method | MRR | H@10 | source | status |
|---|---|---|---|---|
| random | 0.022 | 0.113 | `kge_link_prediction_v1.md` | kept (v1, still current: script 11-02, KG built 11-02 22:11; file committed at `cd67e95`) |
| popularity | 0.110 | 0.239 | `kge_link_prediction_v1.md` | kept (v1, still current: script 11-02, KG built 11-02 22:11; file committed at `cd67e95`) |
| TransE | 0.082 | 0.238 | `kge_link_prediction_v1.md` | kept (v1, still current: script 11-02, KG built 11-02 22:11; file committed at `cd67e95`) |
| DistMult | 0.091 | 0.199 | `kge_link_prediction_v1.md` | kept (v1, still current: script 11-02, KG built 11-02 22:11; file committed at `cd67e95`) |
| ComplEx | 0.116 | 0.262 | `kge_link_prediction_v1.md` | kept (v1, still current: script 11-02, KG built 11-02 22:11; file committed at `cd67e95`) |
| RotatE | **0.159** | **0.326** | `kge_link_prediction_v1.md` | kept (v1, still current: script 11-02, KG built 11-02 22:11; file committed at `cd67e95`) |

## 5. Opponent next card (acc@1, test) - kept

| method | val acc@1 | test acc@1 | source | status |
|---|---|---|---|---|
| popularity | 0.000 | 0.002 | `next_card_v1.md` | kept (v1, still current: script and data 11-02; file committed at `cd67e95`) |
| copy-frequency (= cycle, w = 0) | 0.207 | 0.243 | `next_card_v1.md` | kept (v1, still current: script and data 11-02; file committed at `cd67e95`) |
| SASRec (3 seeds) | 0.129 +/- 0.014 | 0.222 +/- 0.012 | `next_card_v1.md` | kept (v1, still current: script and data 11-02; file committed at `cd67e95`) |

## 6. Knowledge editing (non-parametric, D-034) - re-run

| criterion | edited | no edit | source | status |
|---|---|---|---|---|
| reliability (n 15) | 1.000 | 0.000 | `edit_eval_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); identical to `edit_eval_v1` |
| generalization (n 465) | 1.000 | 0.000 | `edit_eval_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); identical to `edit_eval_v1` |
| locality, cards (n 2500) / states (n 1500) | 1.000 / 1.000 | 1.000 / 1.000 | `edit_eval_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); identical to `edit_eval_v1` |
| portability (n 465) | 1.000 | 0.067 | `edit_eval_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); identical to `edit_eval_v1` |
| conflict: share of top-10 hits using the edited card | 0.002 | 0.124 | `edit_eval_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); identical to `edit_eval_v1` |
| versioning / rollback | pass / pass | - | `edit_eval_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); identical to `edit_eval_v1` |

## 7. Opponent-deck model (leave-one-game-out, 155 games) - re-run

| k cards seen | recall@(8-k) model | popularity | random | source | status |
|---|---|---|---|---|---|
| 1 | 0.234 | 0.130 | 0.059 | `opp_model_v1.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); identical to the 11-06 17:53 evaluation (the file was not in `docs/data` before) |
| 2 | 0.291 | 0.124 | 0.051 | `opp_model_v1.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); identical to the 11-06 17:53 evaluation (the file was not in `docs/data` before) |
| 3 | 0.327 | 0.118 | 0.043 | `opp_model_v1.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`); identical to the 11-06 17:53 evaluation (the file was not in `docs/data` before) |

## 8. Tactics engine vs the pro plays - re-run

| measure | v2 (HEAD engine) | v1 (11-05 engine, stale) | source |
|---|---|---|---|
| top-1 card, val / test (all play records) | 13.1% / 10.0% | 10.4% / 9.0% | `tactics_eval_v2.md` vs `tactics_eval_v1.md` |
| coverage (an engine move exists), val / test | 52.1% / 41.7% | 45.8% / 41.2% | same |
| top-1 card on push states, val / test | 15.9% / 11.7% | 19.1% / 18.0% | same |
| 4-way grid, tactics ranker val / test (random 25.0%, placement prior 49.3% / 38.4%, d1 52.4% / 48.3%) | 29.3% / 26.5% | 27.5% / 25.5% | same |
| weight re-tuning on the HEAD engine, val objective defaults -> tuned (not shipped) | 0.4924 -> 0.5125; test 0.3631 -> 0.3940; top-1 test 10.0% -> 10.8% | 0.5129 -> 0.5321 (old engine) | `tactics_tuning_v2.{json,md}` |

Status of every row: re-run 2026-11-07 (HEAD `54243d7`). The re-tune wrote its weights to a scratch file (`clashbot/resources/tactics_weights.json` untouched). `tools/train/tune_tactics.py` refuses to start on the HEAD engine because its design guard `guard_ice_spirit_still_answers_bats` fails (the engine now says "tower handles it" for 3 bats, a deliberate MAC-AM change); the run used a driver that drops that one stale guard, the other three hold.

## 9. Arbiter (restrict d1's options with kNN / tactics support) - re-run

| strategy | val acc | test acc | source | status |
|---|---|---|---|---|
| d1-ft-joint-tac alone | 54.7% | 50.6% | `arbiter_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`): probabilities of the live `-tac` model on `joint_tac_v2` (v1 used `d1-ft-joint`: 52.4% / 48.3%) |
| arbiter, default config | 52.0% | 50.3% | `arbiter_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`): probabilities of the live `-tac` model on `joint_tac_v2` (v1 used `d1-ft-joint`: 52.4% / 48.3%) |
| arbiter, grid chosen on val | 54.5% | 50.9% | `arbiter_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`): probabilities of the live `-tac` model on `joint_tac_v2` (v1 used `d1-ft-joint`: 52.4% / 48.3%) |
| tactics-first (non-LLM) | 53.9% | 49.9% | `arbiter_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`): probabilities of the live `-tac` model on `joint_tac_v2` (v1 used `d1-ft-joint`: 52.4% / 48.3%) |
| knn-first (non-LLM) | 47.4% | 47.3% | `arbiter_v2.{json,md}` | re-run 2026-11-07 (HEAD `54243d7`): probabilities of the live `-tac` model on `joint_tac_v2` (v1 used `d1-ft-joint`: 52.4% / 48.3%) |

Verdict unchanged: no gain from the arbiter (the arbiter is off in the live stack).

## 10. ETA model (tracker arrival time) - promoted from the MAC-AL smoke run

| measure | old rule | new rule | source | status |
|---|---|---|---|---|
| median abs error, all frames (n 384) | 1.49 s | 0.30 s | `eta_eval_v1.{json,md}` | kept (promoted copy of `runs/smoke/AL/eta.json`, run 2026-11-07 00:23 on `tracking.py` from `af10598`, unchanged on the live path since) |
| melee (n 175) | 1.30 s | 0.30 s | `eta_eval_v1.{json,md}` | kept (promoted copy of `runs/smoke/AL/eta.json`, run 2026-11-07 00:23 on `tracking.py` from `af10598`, unchanged on the live path since) |
| ranged (n 192) | 2.19 s | 0.40 s | `eta_eval_v1.{json,md}` | kept (promoted copy of `runs/smoke/AL/eta.json`, run 2026-11-07 00:23 on `tracking.py` from `af10598`, unchanged on the live path since) |
| flyer (n 17) | 2.30 s | 0.20 s | `eta_eval_v1.{json,md}` | kept (promoted copy of `runs/smoke/AL/eta.json`, run 2026-11-07 00:23 on `tracking.py` from `af10598`, unchanged on the live path since) |
| share of frames within 1 s | 34% | 80% | `eta_eval_v1.{json,md}` | kept (promoted copy of `runs/smoke/AL/eta.json`, run 2026-11-07 00:23 on `tracking.py` from `af10598`, unchanged on the live path since) |

Small sample: one video (`Z8-4VhLjrGU`, 60-300 s), 70 enemy tracks, 11 arrivals.

## 11. Tap landing (did the tapped card leave the hand / elixir fall)

| run | taps sent | landed | landing rate | source | status |
|---|---|---|---|---|---|
| 20261106-233907 (MAC-AL code, 4+ games) | 100 | 95 | 95% | `tap_landing_233907_v1.md` | kept (promoted copy of `runs/smoke/AL/review/review.md`, after the ability-verdict fix) |
| 20261107-013441 (MAC-AL code; game 3 won) | 84 | 75 | 89% | `tap_landing_20261107-013441_v2.md` | re-run 2026-11-07 (HEAD `54243d7`) (`review_shots.py` on the recorded shots; first review of this run) |
| 20261107-111554 (MAC-AM code) | 95 | 78 | 82% | `tap_landing_20261107-111554_v2.md` | re-run 2026-11-07 (HEAD `54243d7`) (first review of this run; failures concentrate on tesla and fireball) |

## 12. Latency

Source: `latency_v2.{json,md}` (re-run 2026-11-07, `54243d7` + eval tooling, decision code unchanged). Whole stack = `clashbot.live --dry-run --source video` on `Z8-4VhLjrGU` 60-180 s, d1-ft-joint-tac + `fast` policy, MPS, 3 repetitions (`runs/rerun/lat_{1,2,3}`); times in ms, mean +- sd over the 3 runs of each run's p50 / p95. Target p95 < 300 ms.

| stage | p50 | p95 |
|---|---|---|
| capture | 0.8 +- 0.1 | 4.3 +- 1.4 |
| perception | 179.2 +- 7.6 | 234.2 +- 29.2 |
| live_state | 0.2 +- 0.0 | 0.5 +- 0.1 |
| trigger | 2.7 +- 0.1 | 9.7 +- 3.7 |
| candidates | 46.7 +- 2.4 | 86.1 +- 12.4 |
| rank | 87.1 +- 10.6 | 191.0 +- 21.2 |
| execute | 0.0 +- 0.0 | 0.1 +- 0.0 |
| total | 271.7 +- 20.1 | 455.2 +- 41.2 |

Isolated micro-benchmarks, 200 calls after warm-up (ms):

| benchmark | p50 | p95 |
|---|---|---|
| katacr / mps | 106.08 | 110.22 |
| katacr / cpu | 465.64 | 498.10 |
| d1 | 68.13 | 110.29 |
| act / p_act | 0.03 | 0.03 |
| act / features_only | 0.01 | 0.02 |
| knn / StateIndex.knn exact (k=10) | 1.53 | 2.04 |
| knn / StateIndex.knn hnsw (k=10) | 0.67 | 0.83 |
| knn / raw exact (matmul + top-10) | 1.54 | 2.37 |
| knn / raw hnswlib knn_query (k=10) | 0.33 | 0.41 |
| candidates | 29.76 | 40.90 |

Live runs (recorded `summary.json` of every run under `runs/poc/2026110*`): total p50 about 340 ms, p95 about 500-570 ms in the 900 s runs, including the 183 ms two-click `execute` stage; per-run rows in `latency_v2.md`. The p95 < 300 ms target was missed in every live run.

## 13. Live history (what the bot did against the game)

Counted from `runs/poc/2026110*/decisions.jsonl` (taps = decisions with `executor.sent`), 54243d7 tree: **12 live runs, 931 decisions, 746 taps**, 109 min of configured live time. Per-run latency rows: `latency_v2.md`.

| run | ranker / policy | configured s | decisions | taps | what is on record |
|---|---|---|---|---|---|
| 20261103-192225 | score / n/a | 60 | 28 | 26 | first live run; macOS blocked clicks for about 30 s |
| 20261103-192709 | score / n/a | 200 | 25 | 13 | match lost quickly |
| 20261105-172506 | d1 / reactive | 200 | 27 | 8 | 14 of 27 decisions skipped (hand `?`); lost in about 1 min |
| 20261105-193555 | d1 / reactive | 200 | 56 | 42 | `--no-tactics` (match A); lost |
| 20261105-194101 | d1 / reactive | 200 | 80 | 80 | tactics on (match B); lost |
| 20261105-205934 | d1 / reactive | 300 | 61 | 53 | about 2 matches |
| 20261106-133529 | d1 / fast | 900 | 181 | 161 | coach planner, about 4 matches |
| 20261106-135724 | d1 / fast | 900 | 61 | 41 | about 2 matches |
| 20261106-170722 | d1 / fast | 900 | 56 | 43 | tap landing 86% (37/43) |
| 20261106-233907 | d1 / fast | 900 | 116 | 100 | 4 full games + part of a 5th; tap landing 95% |
| 20261107-013441 | d1 / fast | 900 | 115 | 84 | 3 games; game 3 won in overtime (the first win) |
| 20261107-111554 | d1 / fast | 900 | 125 | 95 | MAC-AM code; outcomes not recorded |

Battle outcomes (win / loss / crowns) are not logged by the bot; the only wins on record are in `docs/mac/STATUS.md` (first win: run 20261107-013441, game 3, overtime). Win rate comes from the planned Training Camp trials (section 16), not from this table.

## 14. Bot vs pro on video (MAC-AQ)

These are open-loop runs of the live stack on the pro's own POV video, with the pro's deck. Source: `docs/data/vs_pro_v1.{json,md}`.
- Coverage: 24 usable test games, 90 min of video, 1,561 pro plays, 2,042 bot taps.
- Retrieval is filtered to the train and val splits.
- CIs are 95%, from a bootstrap over games.

| metric | bot | baseline |
|---|---|---|
| recall ±1 / ±2 / ±3 s | 58.9% / **77.1%** [71.8, 83.0] / 84.5% | ±2 s: 68.9% (the same taps shifted 20–60 s) |
| precision ±2 s | 72.6% [69.3, 76.2] | |
| same card, matched pairs (n 725) | **17.1%** [14.6, 20.2] | 12.8% [11.6, 14.1] (random card of the perceived hand) |
| same card, pairs with a KG threat (n 126) | 19.8% [13.3, 26.9] | 15.6% [8.8, 23.1] (KG v1 counter first in hand) |
| taps per minute | 22.8 | 17.4 (pro) |

The pro's card is in the bot's perceived hand for about 40% of plays, which caps card agreement (see the md).

## 15. Rule mining / context conflict (MAC-AR)

Source: `docs/data/answer_rules_v1.{json,md}` and `docs/data/context_conflict_v1.{json,md}`, at HEAD `54243d7`.

**E6: held-out mined answer rules.**
- Setup: rules come from train events only; each event ranks the pro's own deck; 1,042 val+test events in 53 games.
- hit@1 pooled: full rules .170 vs random .127. That is +.044 [+.020, +.068].
- Against train popularity of the answer: −.006 [−.035, +.020], so no gain.
- Test alone: hit@1 .154 and MRR .374, against random .128 / .345 and popularity .169 / .384.
- Sweeping support (n ≥ 1/3/5/10) and the answer window (2/4/8 s) gives test MRR .368–.393 in every cell.

**E2: context conflict, d1-ft-joint-tac on `joint_tac`, 1,527 lines.**

| condition | pooled accuracy | McNemar p vs C0 | follow-rate |
|---|---|---|---|
| C0 as exported | .521 [.494, .548] | – | – |
| C1 notes, tactics and KG lines removed | .522 | 1.00 | – |
| C2 notes shuffled | .532 | .07 | 9.2% |
| C3 poisoned | .521 | 1.00 | 2.8% |
| C4 transplanted block | .511 | .34 | 9.1% |

CAD and AdaCAD give no gain. The replicate on `joint_tac_v2` shows the same picture: C0 .527, C1 .522, C3 .522.

## 16. Live trials

**Trial 2026-11-07** (D-052, frozen code D-051).
- Setup: 10 Training Camp battles through `scripts/demo.sh --live`. The user recorded the results. Session log: `runs/demo/20261107-213709`, with 9 Training Camp battles logged (one battle start was missed).
- **Wins: 0 / 10**, 95% Clopper–Pearson CI [0, 30.8%].
- **Against KataCR's 1/20** in Training Camp: Fisher exact p = 1.0.
- Scores (us–them): 0-1, 0-1 (OT), 0-1, 1-2, 0-2, 1-3, 1-3 (OT), 0-3, 0-1, 0-1.
  - Crowns: 3 for, 18 against.
  - A tower was taken in 3 battles.
  - 2 battles went to overtime.
- Per logged battle: 40.3 decisions and 27.0 taps; median length 213 s.
- Decision latency (session): total p50 323 ms, p95 602 ms. Perception 108/266, candidates 27/58, rank 0/141, execute 181/189.
- The Training Camp gate rejected two non-battle screens (banner "2618").

**Earlier sessions:** one recorded win, in run 20261107-013441 (game 3, overtime).

**Failure analysis:** `docs/plans/BACKLOG_post_showcase.md` (W1–W12).

## 17. Decision models (MAC-AU)

Source: `docs/data/decision_models_v1.{json,md}` (generated 2026-11-09T06:26:17+00:00 by `tools/eval/score_decision_models.py` from `runs/bench/au/preds/*.jsonl`). Protocol: `docs/plans/MAC-AU_decision_models_retrieval_benchmark.md` (Amendment 1: all test lines, thinking off). Report: Sec. 5.2 and Table 3 of `docs/clashbot_report_cvpr`.
- Lines: all 766 4-way lines of `dataset/v1/d1/joint_tac_v2/test.jsonl`, 27 test games. Accuracy = the picked option is the pro's play; card accuracy = it has the pro's card. A null prediction (unparseable reply or error) counts as wrong.
- CIs: cluster bootstrap over the 27 games (10,000 resamples, seed 0). `diff` is paired against d1-ft-joint-tac; McNemar is exact and treats lines as independent.
- Latency is wall time per decision. Local models ran on the M5 under the heavy lock after a warm-up (d1 with transformers on MPS; Qwen with MLX 4-bit). API latency includes the network round trip.
- All LLMs share one prompt (`tools/eval/decision_prompt.py`): d1's state text with its `tactics:` / `KG:` lines and the options A-D with their notes.

| model | size / type | n | accuracy [95% CI] | card accuracy [95% CI] | diff vs d1-ft-joint-tac [95% CI] | McNemar p | latency p50 / p95 (ms) | cost / decision | parse failures |
|---|---|---|---|---|---|---|---|---|---|
| d1-ft-joint-tac (live) | 600M, fine-tuned | 766 | 0.506 [0.473, 0.538] | 0.692 [0.649, 0.730] | (reference) | (reference) | 75 / 117 | 0 (local) | 0/766 (0.0%) |
| d1-omni-600M zero-shot | 600M, zero-shot | 766 | 0.260 [0.235, 0.283] | 0.513 [0.484, 0.544] | -0.247 [-0.289, -0.202] | <0.001 | 73 / 113 | 0 (local) | 0/766 (0.0%) |
| d1-2B fine-tuned | 1.9B, fine-tuned | 766 | 0.429 [0.394, 0.466] | 0.614 [0.571, 0.653] | -0.077 [-0.122, -0.034] | <0.001 | 295 / 465 | 0 (local) | 0/766 (0.0%) |
| Qwen3-4B-Instruct-2507 zero-shot | 4B LLM, 4-bit | 766 | 0.286 [0.244, 0.331] | 0.527 [0.496, 0.563] | -0.221 [-0.272, -0.166] | <0.001 | 802 / 1119 | 0 (local) | 5/766 (0.7%) |
| Qwen3-4B + 3 retrieved pro states | 4B LLM, 4-bit, RAG few-shot | 766 | 0.283 [0.251, 0.316] | 0.516 [0.479, 0.554] | -0.223 [-0.266, -0.178] | <0.001 | 1535 / 1958 | 0 (local) | 1/766 (0.1%) |
| Qwen3-14B zero-shot | 14B LLM, 4-bit | 766 | 0.304 [0.264, 0.345] | 0.523 [0.486, 0.568] | -0.202 [-0.251, -0.155] | <0.001 | 2721 / 3866 | 0 (local) | 0/766 (0.0%) |
| Claude Haiku 5.5 | API | 766 | 0.279 [0.250, 0.310] | 0.499 [0.462, 0.532] | -0.227 [-0.269, -0.181] | <0.001 | 676 / 823 | $0.00009 | 29/766 (3.8%) |
| Claude Sonnet 5.5 (thinking between_tools, max 2,000 output tokens, answer parsed from the end of the reply) | API | 766 | 0.364 [0.325, 0.403] | 0.585 [0.543, 0.628] | -0.142 [-0.191, -0.093] | <0.001 | 2467 / 4824 | $0.00369 | 1/766 (0.1%) |
| random (one seeded draw) | baseline | 766 | 0.269 [0.237, 0.301] | 0.539 [0.504, 0.572] | -0.238 [-0.282, -0.191] | <0.001 | 0.001 / 0.001 | 0 (local) | 0/766 (0.0%) |
| popularity | baseline | 766 | 0.325 [0.280, 0.373] | 0.453 [0.398, 0.511] | -0.181 [-0.239, -0.119] | <0.001 | 0.001 / 0.002 | 0 (local) | 0/766 (0.0%) |
| placement prior | baseline | 766 | 0.384 [0.336, 0.431] | 0.614 [0.568, 0.657] | -0.123 [-0.167, -0.080] | <0.001 | 0.003 / 0.003 | 0 (local) | 0/766 (0.0%) |
| LightGBM LambdaRank (seed 0) | baseline, feature ranker | 766 | 0.482 [0.448, 0.514] | 0.677 [0.648, 0.708] | -0.025 [-0.059, +0.014] | 0.263 | 0.4 / 0.5 | 0 (local) | 0/766 (0.0%) |
| random (expected) | baseline, analytic | 766 | 0.250 | 0.500 | - | - | - | - | - |

- **Sonnet row.** The Sonnet row above is `claude-sonnet-5-5-long` (settings in `tools/eval/bench_claude_rankers.py`). With the 16-token cap of the other API runs (row `Claude Sonnet 5.5` in the md), 481 of 766 Sonnet replies (62.8%) were cut off (`stop_reason` max_tokens), for accuracy 0.144 [0.112, 0.175]. Haiku with the same cap had 29 of 766 replies (3.8%) cut off; they count as wrong.
- d1-ft-joint-tac is 0.5065 here (shown as 0.506); section 1 and `stats_audit_v1` show the same lines at 0.507 (bootstrap seed 20261107).
- d1 zero-shot scores 0.260 on these note-carrying lines, against 0.221 on the plain `joint` lines (section 1).
- d1-2B fine-tune (top 8 layers trained, val 0.4192, `docs/research/course_mapping_audit.md` C4). It loads after a loader fix (AutoConfig -> native `Lfm3VlConfig`, in a mirror at `runs/bench/au/d1-2b-loadable`). This closes H8 of Appendix A.

## 18. Retrieval methods for the KB and the KG (MAC-AU)

Source: `docs/data/retrieval_methods_v1.{json,md}` (run at HEAD `5bb8048ccb`). Report: Sec. 5.5 and Table 6 of `docs/clashbot_report_cvpr`.
- Lines and events: the same 766 test lines and 27 games as section 17 (5 videos). CIs are a cluster bootstrap over games (10,000 resamples, seed 0).
- **Train-only pool.** KB retrieval is restricted to train games: 4,997 of 10,923 indexed states. 125,599 hits were checked and none violated this. KG rules and popularity come from train events only.
- **Decision rule.** KB: an option's score is the sum of its top-20 hits with the same action key. KG: the option card's score against the line's recent threats. Unsupported lines tie (0.25).
- **Card-level cap.** The 2x2 grid gives two options per card, so a card-level KG score is capped at 0.5 4-way accuracy.
- Latency is measured under the heavy lock; read it as an upper bound.

**KB: the pro-state database (top-20 hits per query).**

| method | hit@10 | card_hit@10 | MRR | 4-way acc | card acc | coverage | acc if supported | empty | recall@20 vs exact | ms p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| dense exact (644-d structured, dot) | 0.144 [0.093, 0.200] | 0.423 [0.347, 0.513] | 0.067 [0.038, 0.102] | 0.284 [0.243, 0.331] | 0.476 [0.442, 0.513] | 0.53 [0.46, 0.60] | 0.314 [0.235, 0.391] | 0.000 | - | 1.16 / 1.22 |
| BM25 (FTS5 over context_text) | 0.166 [0.126, 0.205] | 0.491 [0.399, 0.580] | 0.060 [0.045, 0.075] | 0.318 [0.283, 0.351] | 0.526 [0.495, 0.556] | 0.53 [0.46, 0.59] | 0.379 [0.313, 0.440] | 0.000 | - | 22.14 / 31.97 |
| RRF dense + BM25 (live bot) | 0.175 [0.125, 0.229] | 0.501 [0.425, 0.586] | 0.074 [0.053, 0.097] | 0.305 [0.268, 0.347] | 0.495 [0.462, 0.529] | 0.58 [0.51, 0.64] | 0.346 [0.282, 0.412] | 0.000 | - | 24.19 / 33.84 |
| bge-small text embeddings (cosine) | 0.155 [0.103, 0.214] | 0.450 [0.382, 0.521] | 0.067 [0.041, 0.097] | 0.307 [0.265, 0.351] | 0.504 [0.464, 0.544] | 0.46 [0.39, 0.53] | 0.373 [0.285, 0.455] | 0.000 | - | 13.64 / 16.34 |
| RRF dense + BM25 + bge | 0.184 [0.137, 0.234] | 0.530 [0.455, 0.609] | 0.082 [0.056, 0.112] | 0.314 [0.279, 0.351] | 0.505 [0.473, 0.537] | 0.54 [0.47, 0.61] | 0.367 [0.306, 0.427] | 0.000 | - | 45.16 / 63.27 |
| HNSW ef 10 (dense, train-filtered) | 0.141 [0.089, 0.198] | 0.418 [0.341, 0.510] | 0.067 [0.038, 0.102] | 0.287 [0.245, 0.333] | 0.479 [0.446, 0.516] | 0.52 [0.46, 0.59] | 0.321 [0.241, 0.398] | 0.000 | 0.977 | 0.84 / 1.31 |
| HNSW ef 40 (dense, train-filtered) | 0.141 [0.089, 0.198] | 0.419 [0.342, 0.511] | 0.067 [0.038, 0.102] | 0.286 [0.244, 0.332] | 0.476 [0.443, 0.513] | 0.53 [0.46, 0.60] | 0.317 [0.239, 0.393] | 0.000 | 0.990 | 1.11 / 1.65 |
| HNSW ef 160 (dense, train-filtered) | 0.142 [0.091, 0.199] | 0.422 [0.346, 0.512] | 0.067 [0.038, 0.102] | 0.286 [0.244, 0.332] | 0.477 [0.444, 0.514] | 0.53 [0.46, 0.60] | 0.317 [0.237, 0.394] | 0.000 | 0.998 | 2.24 / 3.07 |
| deck pre-filter, then dense | 0.076 [0.016, 0.144] | 0.157 [0.035, 0.300] | 0.041 [0.007, 0.080] | 0.286 [0.253, 0.326] | 0.516 [0.494, 0.542] | 0.16 [0.04, 0.29] | 0.475 [0.312, 0.587] | 0.783 | - | 0.82 / 0.87 |

**KG: threat to answer.** 433 events, ranked over the pro's deck. The decision columns use all 766 lines; 374 of them have a threat.

| method | hit@1 | hit@3 | MRR | event coverage | 4-way acc | card acc | line coverage | query ms p50 / p95 | decision ms p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|
| KG v1 counter_scores (wiki COUNTERS + mined ANSWERED_BY) | 0.143 [0.122, 0.167] | 0.409 [0.372, 0.447] | 0.360 [0.339, 0.381] | 0.97 [0.94, 0.99] | 0.249 [0.241, 0.257] | 0.499 [0.482, 0.515] | 0.31 [0.26, 0.37] | 0.01 / 0.02 | 0.02 / 0.05 |
| KG v1 wiki COUNTERS only | 0.141 [0.121, 0.163] | 0.406 [0.369, 0.444] | 0.359 [0.339, 0.379] | 0.97 [0.93, 0.99] | 0.250 [0.242, 0.258] | 0.501 [0.485, 0.516] | 0.29 [0.24, 0.34] | 0.01 / 0.01 | 0.02 / 0.04 |
| KG v2 answers_for (full) | 0.154 [0.117, 0.196] | 0.422 [0.372, 0.480] | 0.374 [0.343, 0.410] | 0.99 [0.97, 1.00] | 0.239 [0.229, 0.249] | 0.477 [0.457, 0.498] | 0.45 [0.39, 0.51] | 0.05 / 0.07 | 0.02 / 0.13 |
| KG v2 pro source only | 0.129 [0.096, 0.168] | 0.399 [0.367, 0.438] | 0.350 [0.325, 0.381] | 0.63 [0.50, 0.75] | 0.236 [0.228, 0.244] | 0.472 [0.456, 0.487] | 0.20 [0.15, 0.26] | 0.04 / 0.06 | 0.02 / 0.12 |
| KG v2 role transfer only | 0.145 [0.118, 0.172] | 0.398 [0.366, 0.430] | 0.356 [0.335, 0.376] | 0.67 [0.55, 0.77] | 0.241 [0.231, 0.252] | 0.482 [0.462, 0.505] | 0.24 [0.18, 0.28] | 0.04 / 0.06 | 0.02 / 0.12 |
| KG v2 wiki only | 0.147 [0.123, 0.173] | 0.407 [0.371, 0.442] | 0.362 [0.341, 0.383] | 0.97 [0.93, 0.99] | 0.249 [0.241, 0.256] | 0.498 [0.482, 0.512] | 0.29 [0.24, 0.34] | 0.04 / 0.06 | 0.02 / 0.12 |
| KG v2 simulation (STOPS) only | 0.157 [0.140, 0.176] | 0.440 [0.410, 0.473] | 0.376 [0.359, 0.395] | 0.77 [0.66, 0.86] | 0.258 [0.250, 0.267] | 0.517 [0.501, 0.534] | 0.31 [0.25, 0.37] | 0.04 / 0.06 | 0.02 / 0.12 |
| KGE RotatE (card COUNTERS threat) | 0.125 [0.095, 0.158] | 0.400 [0.352, 0.448] | 0.340 [0.315, 0.367] | 1.00 [1.00, 1.00] | 0.253 [0.239, 0.265] | 0.505 [0.477, 0.530] | 0.49 [0.43, 0.54] | 0.18 / 0.18 | 0.02 / 0.39 |
| KGE TransE (card COUNTERS threat) | 0.118 [0.081, 0.152] | 0.353 [0.318, 0.389] | 0.335 [0.309, 0.360] | 1.00 [1.00, 1.00] | 0.245 [0.231, 0.257] | 0.491 [0.462, 0.515] | 0.49 [0.43, 0.54] | 0.13 / 0.14 | 0.02 / 0.30 |
| train popularity of the answer card | 0.169 [0.124, 0.223] | 0.433 [0.374, 0.492] | 0.384 [0.349, 0.427] | 1.00 [1.00, 1.00] | 0.217 [0.205, 0.231] | 0.435 [0.410, 0.462] | 0.49 [0.43, 0.54] | 0.00 / 0.00 | 0.02 / 0.03 |
| random (expected) | 0.128 [0.125, 0.132] | 0.384 [0.375, 0.395] | 0.345 [0.340, 0.351] | - | 0.250 [0.250, 0.250] | 0.500 [0.500, 0.500] | - | - | - |

- **As decision models**, every method is far below d1-ft-joint-tac (0.507 [0.473, 0.539] on these lines). The best KB method, BM25, reaches 0.318 [0.283, 0.351]. Deltas vs d1 are -0.189 to -0.222 for the KB and -0.248 to -0.289 for the KG. Every McNemar p is ≤ 2.28e-13. The per-method table is in the md, under "Retrieval as a decision model".
- The deck pre-filter returns nothing for 78.3% of lines, whose decks never occur in train. On the lines it supports it is the most accurate KB rule (0.475).
- HNSW keeps 0.977-0.998 of the exact top 20. Exact dense search over the train pool takes only 1.16 ms p50.

## Appendix A. Re-run ledger (command, wall time, verdict)

All commands from the worktree root, `PY=.venv/bin/python`, HEAD `54243d7`, `HF_HUB_OFFLINE=1`. Heavy steps ran one at a time under the shared lock; the wall time of a heavy step is the time with the lock held. Not run: H4 (live-state parity), H5 (coach bench), H8 (d1-2B fine-tune evaluation: still blocked by the checkpoint loader `KeyError: 'lfm3'`).

| step | what | wall | output | verdict vs the old number |
|---|---|---|---|---|
| C1 | act policy: `tools/eval/eval_act_policy.py --out docs/data/act_policy_v2.json` | 2s | `act_policy_v2.{json,md}` | reactive balanced acc (test, all) 0.598 -> 0.545 (AM value reproduced); act rate 0.220 |
| C2 | tactics eval: `eval_tactics.py --out docs/data/tactics_eval_v2.json` | 2s | `tactics_eval_v2.{json,md}` | top-1 val 10.4% -> 13.1%, test 9.0% -> 10.0%; 4-way 29.3% / 26.5% |
| C3 | act GBDT comparators: `train_act_gbdt.py --out runs/rerun/act-gbdt --report docs/data/act_gbdt_v2.json --bal dataset/v1/act_dense_bal --d1-json docs/data/act_head_v1.json` | 11s | `act_gbdt_v2.{json,md}` | GBDT columns identical; model files and `meta.json` byte-identical to `db/models/act-gbdt`; rules comparator 0.533 -> 0.509 (h = 2 s) |
| C4 | tactics tuning: driver around `tune_tactics.py --out runs/rerun/tactics_weights_rerun.json --report docs/data/tactics_tuning_v2.json` (stale guard dropped) | 134s | `tactics_tuning_v2.{json,md}` | first attempt failed (design guard); with the driver: val 0.4924 -> 0.5125 |
| C5 | opp model: `eval_opp_model.py --out docs/data/opp_model_v1.json` | 3s | `opp_model_v1.{json,md}` | identical to the 11-06 17:53 numbers (0.234 / 0.291 / 0.327) |
| C6 | edit eval: `edit_eval.py --out docs/data/edit_eval_v2.json` | 61s | `edit_eval_v2.{json,md}` | identical to `edit_eval_v1` |
| C7 | tap landing: `review_shots.py runs/poc/20261107-111554` and `...013441` | 20s + 18s | `tap_landing_*_v2.md` | new: 111554 82% (78/95), 013441 89% (75/84) |
| H0 | `tools/data/build_d1_tac.py --out dataset/v1/d1/joint_tac_v2` (HEAD engine) | 167s | `dataset/v1/d1/joint_tac_v2/` (not tracked) | built; `dataset/v1/d1/stats.json` untouched |
| H1 | d1-ft-joint-tac on `joint_tac_v2`: `eval_d1_zero_shot.py --dataset runs/eval_tac_v2 --variant joint --backend transformers --model db/models/d1-ft-joint-tac --out docs/data/d1_finetune_joint_tac_eval_v2.json` | 129s (lock held) | `d1_finetune_joint_tac_eval_v2.json` | 0.540 / 0.497 (as trained) and 0.536 / 0.507 (11-06 rebuild) -> 0.547 / 0.507 |
| H2a | same, `--model db/models/d1-ft-joint --out docs/data/d1_joint_on_tac_v2.json` | 133s (lock held) | `d1_joint_on_tac_v2.json` | no-tactics model given the HEAD notes: 0.477 / 0.461 |
| H2b | same, `--model db/models/d1-ft-joint-kg --out docs/data/d1_joint_kg_on_tac_v2.json` | 124s (lock held) | `d1_joint_kg_on_tac_v2.json` | see section 1 |
| FR | `feature_ranker.py --out docs/data/feature_ranker_v2.json --dump-preds docs/data/feature_ranker_preds_v2.json` | 35s (lock held) | `feature_ranker_v2.{json,md}`, `feature_ranker_preds_v2.json` | byte-identical to `feature_ranker_v1` (all accuracies reproduce exactly, within 0.5 pt required) |
| C8 | arbiter: `eval_arbiter.py --d1-eval docs/data/d1_finetune_joint_tac_eval_v2.json --lines dataset/v1/d1/joint_tac_v2 --out docs/data/arbiter_v2.json` | 3s | `arbiter_v2.{json,md}` | d1 alone 52.4% / 48.3% -> 54.7% / 50.6% (live model); arbiter still no gain |
| H3 | act head: `eval_act_head.py --model db/models/d1-ft-act --out docs/data/act_head_v2.json` (threshold file backed up and restored) | 11172s (lock held) (killed) / 346s (lock held) | `act_head_v2.{json,md}` | learned head unchanged (test balanced acc 0.558); rules comparator 0.535 -> 0.513, combined 0.545 -> 0.512; `act_threshold.json` restored byte-identical. The first attempt was killed after 1500 of 3008 val lines (it crawled at about 4 s per line for 11,172 s with the lock held while the Mac was busy or asleep); the retry ran at normal speed (65 ms per line) |
| H6 | 3 whole-stack dry runs (`clashbot.live --dry-run --source video ... --out runs/rerun/lat_N`) | 531s (lock held) / 403s (lock held) / 383s (lock held) | `runs/rerun/lat_{1,2,3}/summary.json` | see section 12 |
| H7 | `cp` of `runs/smoke/AL/eta.{json,md}` and `review/review.md` to `docs/data` | seconds | `eta_eval_v1.{json,md}`, `tap_landing_233907_v1.md` | promoted, not re-run |
| P2 | `tools/eval/eval_stats.py` | 1s | `stats_audit_v1.{json,md}`, `stats_audit_v1_risk_coverage.csv` | new |
| P3 | `tools/eval/latency_table.py` | 151s (lock held) | `latency_v2.{json,md}` | new |

