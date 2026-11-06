## Accuracy of the strategies (d1 decides; parameters chosen on val, then applied to test)

| strategy | parameters | val acc | test acc | test changed d1 | test fixed / broken |
|---|---|---|---|---|---|
| d1 alone | - | 52.4% (399/761) | 48.3% (370/766) | 0 | 0 / 0 |
| d1 restricted to tac + best lookup when unsure | conf=0.4 | 52.6% (400/761) | 48.0% (368/766) | 12 | 3 / 5 |
| d1 restricted to kNN + tac when unsure | conf=0.3, knn_min=0.4 | 52.4% (399/761) | 48.3% (370/766) | 3 | 0 / 0 |
| arbiter, default config | - | 50.2% (382/761) | 47.6% (365/766) | 101 | 32 / 37 |
| arbiter, grid chosen on val | conf=0.3, knn_min=0.4, urgent_eta_s=4 | 52.6% (400/761) | 48.3% (370/766) | 3 | 0 / 0 |
| tactics-first (non-LLM baseline) | - | 51.2% (390/761) | 46.9% (359/766) | 38 | 7 / 18 |
| knn-first (non-LLM baseline) | - | 46.8% (356/761) | 45.6% (349/766) | 162 | 45 / 66 |

Chosen arbiter grid point (best val accuracy): conf=0.3, knn_min=0.4, urgent_eta_s=4 - val 52.6%, test 48.3%; d1 alone: val 52.4%, test 48.3%.

## Rule firing: arbiter, grid chosen on val (conf=0.3, knn_min=0.4, urgent_eta_s=4)

| split | rule | fired | share | acc when fired | d1 acc on the same | gold in subset | d1 pick in subset | mean subset size | pick != d1 |
|---|---|---|---|---|---|---|---|---|---|
| val | d1 confident | 735 | 96.6% | 53.5% | 53.5% | 100.0% | 100.0% | 4.0 | 0 |
| val | d1 | 12 | 1.6% | 33.3% | 33.3% | 100.0% | 100.0% | 4.0 | 0 |
| val | d1 among pro-supported + tactics | 12 | 1.6% | 16.7% | 16.7% | 16.7% | 25.0% | 1.17 | 9 |
| val | urgent push: d1 among tactics+lookup | 2 | 0.3% | 50.0% | 0.0% | 50.0% | 50.0% | 1.5 | 1 |
| test | d1 confident | 744 | 97.1% | 49.2% | 49.2% | 100.0% | 100.0% | 4.0 | 0 |
| test | d1 | 15 | 2.0% | 20.0% | 20.0% | 100.0% | 100.0% | 4.0 | 0 |
| test | d1 among pro-supported + tactics | 7 | 0.9% | 14.3% | 14.3% | 28.6% | 57.1% | 1.57 | 3 |

## Rule firing: arbiter, default config (conf=0.55, knn_min=0.5, urgent_eta_s=4)

| split | rule | fired | share | acc when fired | d1 acc on the same | gold in subset | d1 pick in subset | mean subset size | pick != d1 |
|---|---|---|---|---|---|---|---|---|---|
| val | d1 | 298 | 39.2% | 46.0% | 46.0% | 100.0% | 100.0% | 4.0 | 0 |
| val | d1 among pro-supported + tactics | 157 | 20.6% | 28.0% | 39.5% | 33.1% | 45.2% | 1.29 | 86 |
| val | d1 confident | 290 | 38.1% | 65.2% | 65.2% | 100.0% | 100.0% | 4.0 | 0 |
| val | urgent push: d1 among tactics+lookup | 16 | 2.1% | 75.0% | 68.8% | 75.0% | 68.8% | 1.44 | 5 |
| test | d1 | 376 | 49.1% | 43.1% | 43.1% | 100.0% | 100.0% | 4.0 | 0 |
| test | d1 confident | 207 | 27.0% | 65.2% | 65.2% | 100.0% | 100.0% | 4.0 | 0 |
| test | d1 among pro-supported + tactics | 166 | 21.7% | 36.8% | 40.4% | 39.2% | 44.6% | 1.22 | 92 |
| test | urgent push: d1 among tactics+lookup | 17 | 2.2% | 41.2% | 35.3% | 47.1% | 47.1% | 1.65 | 9 |

## Arbiter grid: top 8 points on val (with their test accuracy)

| parameters | val acc | test acc |
|---|---|---|
| conf=0.3, knn_min=0.4, urgent_eta_s=4 | 52.6% | 48.3% |
| conf=0.3, knn_min=0.6, urgent_eta_s=4 | 52.6% | 48.3% |
| conf=0.3, knn_min=0.2, urgent_eta_s=4 | 52.6% | 48.3% |
| conf=0.3, knn_min=0.8, urgent_eta_s=4 | 52.6% | 48.3% |
| conf=0.3, knn_min=0.4, urgent_eta_s=6 | 52.6% | 48.3% |
| conf=0.3, knn_min=0.6, urgent_eta_s=6 | 52.6% | 48.3% |
| conf=0.3, knn_min=0.2, urgent_eta_s=6 | 52.6% | 48.3% |
| conf=0.3, knn_min=0.8, urgent_eta_s=6 | 52.6% | 48.3% |

All 165 grid points: val 48.5% .. 52.6%, test 46.0% .. 49.7%.

## d1 calibration: accuracy when its top probability is >= conf (confident) vs below (unsure)

| conf | val confident | val acc | val unsure | val acc | test confident | test acc | test unsure | test acc |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 735 | 53.5% | 26 | 23.1% | 744 | 49.2% | 22 | 18.2% |
| 0.35 | 646 | 55.6% | 115 | 34.8% | 646 | 51.5% | 120 | 30.8% |
| 0.4 | 557 | 58.7% | 204 | 35.3% | 528 | 52.6% | 238 | 38.7% |
| 0.45 | 454 | 63.4% | 307 | 36.2% | 382 | 58.1% | 384 | 38.5% |
| 0.5 | 373 | 63.8% | 388 | 41.5% | 293 | 61.4% | 473 | 40.2% |
| 0.55 | 290 | 65.2% | 471 | 44.6% | 207 | 65.2% | 559 | 42.0% |
| 0.6 | 202 | 71.8% | 559 | 45.4% | 148 | 73.0% | 618 | 42.4% |
| 0.65 | 161 | 75.8% | 600 | 46.2% | 103 | 79.6% | 663 | 43.4% |
| 0.7 | 124 | 83.1% | 637 | 46.5% | 65 | 84.6% | 701 | 44.9% |
| 0.75 | 95 | 87.4% | 666 | 47.4% | 43 | 86.1% | 723 | 46.1% |
| 0.8 | 67 | 94.0% | 694 | 48.4% | 27 | 96.3% | 739 | 46.6% |

## Source coverage (share of lines)

| measure | val | test |
|---|---|---|
| mean_options | 4.0 | 4.0 |
| with_tac | 9.7% | 7.6% |
| with_knn | 36.7% | 37.1% |
| with_knn_ge_0.5 | 32.9% | 30.6% |
| with_kg | 5.3% | 3.9% |
| gold_has_tac | 4.6% | 2.6% |
| gold_has_knn | 15.6% | 17.1% |
| gold_has_knn_ge_0.5 | 13.8% | 12.9% |
| gold_has_kg | 1.5% | 1.8% |
| with_push | 42.8% | 39.2% |
| urgent_2s | 14.4% | 11.0% |
| urgent_2s_with_tac | 1.8% | 1.2% |
| urgent_4s | 22.1% | 18.0% |
| urgent_4s_with_tac | 3.3% | 2.6% |
| urgent_6s | 32.5% | 25.2% |
| urgent_6s_with_tac | 5.7% | 4.3% |

## Input checks

| split | lines | joined | missing d1 | size mismatch | pred mismatch | gold mismatch |
|---|---|---|---|---|---|---|
| val | 761 | 761 | 0 | 0 | 0 | 0 |
| test | 766 | 766 | 0 | 0 | 0 | 0 |

pred / gold mismatch: the eval's `pred` is not the line's option at the argmax of the probs / its `gold` is not the line's answer (0 = the probs follow the criteria order). Rules not listed never fired. `gold in subset` is the ceiling of the rule; `fixed / broken` count lines where the pick differs from d1's own argmax and is right / wrong where d1 was wrong / right.
