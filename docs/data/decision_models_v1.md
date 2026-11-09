# Decision models vs the pro: accuracy and latency on the held-out test games (MAC-AU)

Generated 2026-11-09T05:07:33+00:00 by `tools/eval/score_decision_models.py` from `runs/bench/au/preds/*.jsonl`; re-run it when predictions change.

**Lines:** all 766 4-way lines of `dataset/v1/d1/joint_tac_v2/test.jsonl`, 27 games (Amendment 1; the first plan was the 5 largest games).

**Accuracy** = the picked option is the pro's play. **Card accuracy** = the picked option has the pro's card. A null prediction (unparseable reply or error) counts as wrong. 95% CIs are cluster bootstraps over the 27 games (10,000 resamples, seed 0, as in `stats_audit_v1`). `diff` is the paired accuracy difference to d1-ft-joint-tac (same lines, same game resamples); McNemar is the exact two-sided test on the lines where the two disagree (lines treated as independent). Latency is wall time per decision (local models: validate, tokenise, forward, parse, measured under the heavy lock after a warm-up; API models: includes the network round trip).

## Accuracy and latency

| model | size / type | n | accuracy [95% CI] | card accuracy [95% CI] | diff vs d1-ft-joint-tac [95% CI] | McNemar p | latency p50 / p95 (ms) | cost / decision | parse failures |
|---|---|---|---|---|---|---|---|---|---|
| d1-ft-joint-tac (live) | 600M, fine-tuned | 766 | 0.506 [0.473, 0.538] | 0.692 [0.649, 0.730] | (reference) | (reference) | 75 / 117 | 0 (local) | 0/766 (0.0%) |
| d1-omni-600M zero-shot | 600M, zero-shot | 766 | 0.260 [0.235, 0.283] | 0.513 [0.484, 0.544] | -0.247 [-0.289, -0.202] | <0.001 | 73 / 113 | 0 (local) | 0/766 (0.0%) |
| d1-2B fine-tuned | 1.9B, fine-tuned | 766 | 0.429 [0.394, 0.466] | 0.614 [0.571, 0.653] | -0.077 [-0.122, -0.034] | <0.001 | 295 / 465 | 0 (local) | 0/766 (0.0%) |
| Qwen3-4B-Instruct-2507 zero-shot | 4B LLM, 4-bit | 766 | 0.286 [0.244, 0.331] | 0.527 [0.496, 0.563] | -0.221 [-0.272, -0.166] | <0.001 | 802 / 1119 | 0 (local) | 5/766 (0.7%) |
| Claude Haiku 5.5 | API | 766 | 0.279 [0.250, 0.310] | 0.499 [0.462, 0.532] | -0.227 [-0.269, -0.181] | <0.001 | 676 / 823 | $0.00009 | 29/766 (3.8%) |
| random (one seeded draw) | baseline | 766 | 0.269 [0.237, 0.301] | 0.539 [0.504, 0.572] | -0.238 [-0.282, -0.191] | <0.001 | 0.001 / 0.001 | 0 (local) | 0/766 (0.0%) |
| popularity | baseline | 766 | 0.325 [0.280, 0.373] | 0.453 [0.398, 0.511] | -0.181 [-0.239, -0.119] | <0.001 | 0.001 / 0.002 | 0 (local) | 0/766 (0.0%) |
| placement prior | baseline | 766 | 0.384 [0.336, 0.431] | 0.614 [0.568, 0.657] | -0.123 [-0.167, -0.080] | <0.001 | 0.003 / 0.003 | 0 (local) | 0/766 (0.0%) |
| LightGBM LambdaRank (seed 0) | baseline, feature ranker | 766 | 0.482 [0.448, 0.514] | 0.677 [0.648, 0.708] | -0.025 [-0.059, +0.014] | 0.263 | 0.4 / 0.5 | 0 (local) | 0/766 (0.0%) |
| random (expected) | baseline, analytic | 766 | 0.250 | 0.500 | - | - | - | - | - |

## Pending

No predictions file yet for: `qwen3-4b-rag3`, `qwen3-14b`, `claude-sonnet-5-5`.

## How the local runs were made

- `d1-2b-ft-joint-tac`: backend transformers; dtype auto; mps True; torch 2.15.0; transformers 5.22.1; load_and_warmup_s 5.48; loader_note loader fix: AutoConfig -> native_lfm3_vl.Lfm3VlConfig in a mirror at runs/bench/au/d1-2b-loadable; finished 2026-11-09T04:50:25+00:00
- `d1-ft-joint-tac`: backend transformers; dtype auto; mps True; torch 2.15.0; transformers 5.22.1; load_and_warmup_s 4.0; finished 2026-11-09T04:51:56+00:00
- `d1-zero-shot`: backend transformers; dtype auto; mps True; torch 2.15.0; transformers 5.22.1; load_and_warmup_s 3.95; finished 2026-11-09T04:53:36+00:00
- `lightgbm`: note LightGBM LambdaRank seed 0, predictions from docs/data/feature_ranker_preds_v2.json; latency = build_features + predict per line of a model retrained with the same config and seed (its picks match the dump on 766/766 lines)
- `placement-prior`: note argmax of the share of the card's placements in the key's zone, first maximum on ties
- `popularity`: note argmax of the key's train-split play count, first maximum on ties
- `random`: note one seeded draw (numpy default_rng(0)); expected accuracy 0.25
