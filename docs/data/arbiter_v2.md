## Accuracy of the strategies (d1 decides; parameters chosen on val, then applied to test)

| strategy | parameters | val acc | test acc | test changed d1 | test fixed / broken |
|---|---|---|---|---|---|
| d1 alone | - | 54.7% (416/761) | 50.6% (388/766) | 0 | 0 / 0 |
| d1 restricted to tac + best lookup when unsure | conf=0.55 | 54.8% (417/761) | 50.8% (389/766) | 16 | 7 / 6 |
| d1 restricted to kNN + tac when unsure | conf=0.35, knn_min=0.4 | 54.5% (415/761) | 50.9% (390/766) | 30 | 8 / 6 |
| arbiter, default config | - | 52.0% (396/761) | 50.3% (385/766) | 94 | 29 / 32 |
| arbiter, grid chosen on val | conf=0.35, knn_min=0.4, urgent_eta_s=4 | 54.5% (415/761) | 50.9% (390/766) | 30 | 8 / 6 |
| tactics-first (non-LLM baseline) | - | 53.9% (410/761) | 49.9% (382/766) | 29 | 6 / 12 |
| knn-first (non-LLM baseline) | - | 47.4% (361/761) | 47.3% (362/766) | 151 | 36 / 62 |

Chosen arbiter grid point (best val accuracy): conf=0.35, knn_min=0.4, urgent_eta_s=4 - val 54.5%, test 50.9%; d1 alone: val 54.7%, test 50.6%.

## Rule firing: arbiter, grid chosen on val (conf=0.35, knn_min=0.4, urgent_eta_s=4)

| split | rule | fired | share | acc when fired | d1 acc on the same | gold in subset | d1 pick in subset | mean subset size | pick != d1 |
|---|---|---|---|---|---|---|---|---|---|
| val | d1 confident | 651 | 85.5% | 59.8% | 59.8% | 100.0% | 100.0% | 4.0 | 0 |
| val | d1 | 62 | 8.2% | 24.2% | 24.2% | 100.0% | 100.0% | 4.0 | 0 |
| val | d1 among pro-supported + tactics | 44 | 5.8% | 22.7% | 25.0% | 34.1% | 45.5% | 1.43 | 24 |
| val | urgent push: d1 among tactics+lookup | 4 | 0.5% | 25.0% | 25.0% | 25.0% | 75.0% | 1.0 | 1 |
| test | d1 confident | 624 | 81.5% | 53.7% | 53.7% | 100.0% | 100.0% | 4.0 | 0 |
| test | d1 | 84 | 11.0% | 41.7% | 41.7% | 100.0% | 100.0% | 4.0 | 0 |
| test | d1 among pro-supported + tactics | 50 | 6.5% | 32.0% | 30.0% | 34.0% | 44.0% | 1.2 | 28 |
| test | urgent push: d1 among tactics+lookup | 8 | 1.0% | 50.0% | 37.5% | 75.0% | 75.0% | 1.62 | 2 |

## Rule firing: arbiter, default config (conf=0.55, knn_min=0.5, urgent_eta_s=4)

| split | rule | fired | share | acc when fired | d1 acc on the same | gold in subset | d1 pick in subset | mean subset size | pick != d1 |
|---|---|---|---|---|---|---|---|---|---|
| val | d1 | 308 | 40.5% | 45.1% | 45.1% | 100.0% | 100.0% | 4.0 | 0 |
| val | d1 confident | 282 | 37.1% | 71.3% | 71.3% | 100.0% | 100.0% | 4.0 | 0 |
| val | d1 among pro-supported + tactics | 157 | 20.6% | 33.8% | 44.0% | 39.5% | 57.3% | 1.26 | 67 |
| val | urgent push: d1 among tactics+lookup | 14 | 1.8% | 21.4% | 50.0% | 21.4% | 64.3% | 1.36 | 5 |
| test | d1 | 375 | 49.0% | 46.7% | 46.7% | 100.0% | 100.0% | 4.0 | 0 |
| test | d1 among pro-supported + tactics | 167 | 21.8% | 38.3% | 39.5% | 39.5% | 46.1% | 1.22 | 90 |
| test | d1 confident | 207 | 27.0% | 68.1% | 68.1% | 100.0% | 100.0% | 4.0 | 0 |
| test | urgent push: d1 among tactics+lookup | 17 | 2.2% | 29.4% | 35.3% | 52.9% | 76.5% | 1.71 | 4 |

## Arbiter grid: top 8 points on val (with their test accuracy)

| parameters | val acc | test acc |
|---|---|---|
| conf=0.35, knn_min=0.4, urgent_eta_s=4 | 54.5% | 50.9% |
| conf=0.35, knn_min=0.6, urgent_eta_s=4 | 54.5% | 50.6% |
| conf=0.35, knn_min=0.2, urgent_eta_s=4 | 54.5% | 50.8% |
| conf=0.35, knn_min=0.8, urgent_eta_s=4 | 54.5% | 50.6% |
| conf=0.35, knn_min=0.4, urgent_eta_s=2 | 54.5% | 50.9% |
| conf=0.35, knn_min=0.4, urgent_eta_s=6 | 54.5% | 50.9% |
| conf=0.35, knn_min=0.6, urgent_eta_s=2 | 54.5% | 50.6% |
| conf=0.35, knn_min=0.6, urgent_eta_s=6 | 54.5% | 50.6% |

All 165 grid points: val 49.8% .. 54.5%, test 47.8% .. 50.9%.

## d1 calibration: accuracy when its top probability is >= conf (confident) vs below (unsure)

| conf | val confident | val acc | val unsure | val acc | test confident | test acc | test unsure | test acc |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 738 | 55.7% | 23 | 21.7% | 734 | 50.9% | 32 | 43.8% |
| 0.35 | 651 | 59.8% | 110 | 24.6% | 624 | 53.7% | 142 | 37.3% |
| 0.4 | 546 | 61.7% | 215 | 36.7% | 494 | 56.3% | 272 | 40.4% |
| 0.45 | 457 | 64.3% | 304 | 40.1% | 382 | 60.0% | 384 | 41.4% |
| 0.5 | 360 | 68.6% | 401 | 42.1% | 269 | 65.8% | 497 | 42.4% |
| 0.55 | 282 | 71.3% | 479 | 44.9% | 207 | 68.1% | 559 | 44.2% |
| 0.6 | 214 | 75.2% | 547 | 46.6% | 155 | 73.6% | 611 | 44.8% |
| 0.65 | 167 | 80.8% | 594 | 47.3% | 107 | 81.3% | 659 | 45.7% |
| 0.7 | 122 | 86.1% | 639 | 48.7% | 74 | 86.5% | 692 | 46.8% |
| 0.75 | 97 | 88.7% | 664 | 49.7% | 42 | 88.1% | 724 | 48.5% |
| 0.8 | 78 | 93.6% | 683 | 50.2% | 25 | 88.0% | 741 | 49.4% |

## Source coverage (share of lines)

| measure | val | test |
|---|---|---|
| mean_options | 4.0 | 4.0 |
| with_tac | 12.3% | 7.4% |
| with_knn | 36.7% | 37.1% |
| with_knn_ge_0.5 | 33.4% | 30.7% |
| with_kg | 6.8% | 4.7% |
| gold_has_tac | 6.4% | 3.1% |
| gold_has_knn | 15.6% | 17.1% |
| gold_has_knn_ge_0.5 | 14.3% | 13.1% |
| gold_has_kg | 2.0% | 2.2% |
| with_push | 52.0% | 47.0% |
| urgent_2s | 20.8% | 19.2% |
| urgent_2s_with_tac | 2.4% | 1.2% |
| urgent_4s | 30.9% | 29.0% |
| urgent_4s_with_tac | 3.9% | 2.5% |
| urgent_6s | 40.9% | 35.4% |
| urgent_6s_with_tac | 5.4% | 3.5% |

## Input checks

| split | lines | joined | missing d1 | size mismatch | pred mismatch | gold mismatch |
|---|---|---|---|---|---|---|
| val | 761 | 761 | 0 | 0 | 0 | 0 |
| test | 766 | 766 | 0 | 0 | 0 | 0 |

pred / gold mismatch: the eval's `pred` is not the line's option at the argmax of the probs / its `gold` is not the line's answer (0 = the probs follow the criteria order). Rules not listed never fired. `gold in subset` is the ceiling of the rule; `fixed / broken` count lines where the pick differs from d1's own argmax and is right / wrong where d1 was wrong / right.
