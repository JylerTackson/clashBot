# Fast act model (LightGBM on structured features, MAC-AB A1)

data `dataset/v1/act_dense`, features 29, seed 0, default horizon 2 s; lines train 29036, val 5560, test 5279; `answers.act` vs rounded `meta.next_deploy_in <= 1` disagree on train 9, val 1, test 2 lines

## Overview

| horizon | threshold | rounds | val pro act | val AUC | val bacc | test pro act | test AUC | test bacc | test rules bacc | test fast bacc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 s | 0.26 | 4 | 0.233 | 0.621 | 0.585 | 0.217 | 0.632 | 0.594 | 0.506 | 0.571 |
| 2 s | 0.47 | 21 | 0.404 | 0.653 | 0.613 | 0.385 | 0.659 | 0.607 | 0.509 | 0.584 |
| 3 s | 0.56 | 75 | 0.534 | 0.664 | 0.619 | 0.507 | 0.672 | 0.613 | 0.510 | 0.593 |

## Horizon 1 s  (threshold 0.26, best round 4)

### val  (n=5560, pro act rate 0.233, AUC 0.621)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| gbdt | 0.604 | 0.585 | 0.418 | 0.306 | 0.548 | 0.819 | 0.621 |
| rules | 0.624 | 0.487 | 0.250 | 0.215 | 0.230 | 0.760 | 0.744 |
| fast | 0.558 | 0.551 | 0.459 | 0.273 | 0.537 | 0.800 | 0.564 |
| always | 0.233 | 0.500 | 1.000 | 0.233 | 1.000 | n/a | 0.000 |
| never | 0.767 | 0.500 | 0.000 | n/a | 0.000 | 0.767 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=3017, pro act rate 0.259

| policy | act rate | act recall | act prec |
|---|---|---|---|
| gbdt | 0.526 | 0.626 | 0.308 |
| rules | 0.269 | 0.243 | 0.234 |
| fast | 0.530 | 0.588 | 0.287 |
| always | 1.000 | 1.000 | 0.259 |
| never | 0.000 | 0.000 | n/a |

### test  (n=5279, pro act rate 0.217, AUC 0.632)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| gbdt | 0.623 | 0.594 | 0.395 | 0.298 | 0.542 | 0.836 | 0.646 |
| rules | 0.667 | 0.506 | 0.212 | 0.227 | 0.221 | 0.785 | 0.791 |
| fast | 0.592 | 0.571 | 0.422 | 0.274 | 0.533 | 0.824 | 0.608 |
| always | 0.217 | 0.500 | 1.000 | 0.217 | 1.000 | n/a | 0.000 |
| never | 0.783 | 0.500 | 0.000 | n/a | 0.000 | 0.783 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=2887, pro act rate 0.234

| policy | act rate | act recall | act prec |
|---|---|---|---|
| gbdt | 0.488 | 0.609 | 0.291 |
| rules | 0.227 | 0.243 | 0.251 |
| fast | 0.488 | 0.582 | 0.279 |
| always | 1.000 | 1.000 | 0.234 |
| never | 0.000 | 0.000 | n/a |

### Feature importance (gain, top 15)

| feature | gain | share |
|---|---|---|
| phase_idx | 3815.5 | 0.252 |
| own_elixir | 2757.2 | 0.182 |
| since_own_play | 2191.1 | 0.145 |
| enemy_min_eta_own | 1369.6 | 0.090 |
| hand_cost_mean | 1130.3 | 0.075 |
| since_opp_play | 728.7 | 0.048 |
| opp_princess_hp_min | 606.4 | 0.040 |
| enemy_count_own_half | 508.0 | 0.034 |
| opp_elixir_est | 348.2 | 0.023 |
| elixir_diff | 331.8 | 0.022 |
| match_seconds | 315.6 | 0.021 |
| enemy_count | 173.6 | 0.011 |
| enemy_elixir_field | 156.3 | 0.010 |
| own_princess_hp_min | 149.1 | 0.010 |
| own_spent_5s | 134.7 | 0.009 |

## Horizon 2 s  (threshold 0.47, best round 21)

### val  (n=5560, pro act rate 0.404, AUC 0.653)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| gbdt | 0.622 | 0.613 | 0.432 | 0.531 | 0.567 | 0.692 | 0.660 |
| rules | 0.557 | 0.510 | 0.250 | 0.423 | 0.262 | 0.602 | 0.758 |
| fast | 0.593 | 0.592 | 0.476 | 0.498 | 0.586 | 0.680 | 0.598 |
| always | 0.404 | 0.500 | 1.000 | 0.404 | 1.000 | n/a | 0.000 |
| never | 0.596 | 0.500 | 0.000 | n/a | 0.000 | 0.596 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=3017, pro act rate 0.430

| policy | act rate | act recall | act prec |
|---|---|---|---|
| gbdt | 0.524 | 0.640 | 0.525 |
| rules | 0.269 | 0.282 | 0.449 |
| fast | 0.532 | 0.632 | 0.510 |
| always | 1.000 | 1.000 | 0.430 |
| never | 0.000 | 0.000 | n/a |

### test  (n=5279, pro act rate 0.385, AUC 0.659)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| gbdt | 0.622 | 0.607 | 0.411 | 0.508 | 0.543 | 0.701 | 0.671 |
| rules | 0.575 | 0.509 | 0.212 | 0.405 | 0.223 | 0.621 | 0.795 |
| fast | 0.590 | 0.584 | 0.451 | 0.472 | 0.554 | 0.687 | 0.613 |
| always | 0.385 | 0.500 | 1.000 | 0.385 | 1.000 | n/a | 0.000 |
| never | 0.615 | 0.500 | 0.000 | n/a | 0.000 | 0.615 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=2887, pro act rate 0.411

| policy | act rate | act recall | act prec |
|---|---|---|---|
| gbdt | 0.494 | 0.605 | 0.504 |
| rules | 0.227 | 0.243 | 0.440 |
| fast | 0.506 | 0.598 | 0.485 |
| always | 1.000 | 1.000 | 0.411 |
| never | 0.000 | 0.000 | n/a |

### Feature importance (gain, top 15)

| feature | gain | share |
|---|---|---|
| own_elixir | 13350.1 | 0.380 |
| phase_idx | 7885.3 | 0.225 |
| enemy_min_eta_own | 2489.2 | 0.071 |
| since_own_play | 2109.5 | 0.060 |
| opp_princess_hp_min | 1712.3 | 0.049 |
| match_seconds | 960.9 | 0.027 |
| since_opp_play | 879.2 | 0.025 |
| enemy_elixir_own_half | 733.0 | 0.021 |
| elixir_diff | 728.7 | 0.021 |
| hand_cost_mean | 717.8 | 0.020 |
| own_princess_hp_min | 605.7 | 0.017 |
| opp_elixir_est | 535.3 | 0.015 |
| hand_known | 372.9 | 0.011 |
| ally_count_enemy_half | 303.8 | 0.009 |
| enemy_elixir_field | 293.6 | 0.008 |

## Horizon 3 s  (threshold 0.56, best round 75)

### val  (n=5560, pro act rate 0.534, AUC 0.664)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| gbdt | 0.612 | 0.619 | 0.408 | 0.679 | 0.519 | 0.566 | 0.719 |
| rules | 0.500 | 0.517 | 0.250 | 0.567 | 0.265 | 0.477 | 0.768 |
| fast | 0.600 | 0.602 | 0.466 | 0.643 | 0.562 | 0.562 | 0.643 |
| always | 0.534 | 0.500 | 1.000 | 0.534 | 1.000 | n/a | 0.000 |
| never | 0.466 | 0.500 | 0.000 | n/a | 0.000 | 0.466 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=3017, pro act rate 0.558

| policy | act rate | act recall | act prec |
|---|---|---|---|
| gbdt | 0.474 | 0.569 | 0.670 |
| rules | 0.269 | 0.284 | 0.589 |
| fast | 0.506 | 0.591 | 0.652 |
| always | 1.000 | 1.000 | 0.558 |
| never | 0.000 | 0.000 | n/a |

### test  (n=5279, pro act rate 0.507, AUC 0.672)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| gbdt | 0.611 | 0.613 | 0.383 | 0.654 | 0.494 | 0.585 | 0.731 |
| rules | 0.506 | 0.510 | 0.212 | 0.530 | 0.222 | 0.499 | 0.798 |
| fast | 0.592 | 0.593 | 0.441 | 0.612 | 0.533 | 0.576 | 0.653 |
| always | 0.507 | 0.500 | 1.000 | 0.507 | 1.000 | n/a | 0.000 |
| never | 0.493 | 0.500 | 0.000 | n/a | 0.000 | 0.493 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=2887, pro act rate 0.532

| policy | act rate | act recall | act prec |
|---|---|---|---|
| gbdt | 0.436 | 0.531 | 0.647 |
| rules | 0.227 | 0.238 | 0.560 |
| fast | 0.475 | 0.550 | 0.615 |
| always | 1.000 | 1.000 | 0.532 |
| never | 0.000 | 0.000 | n/a |

### Feature importance (gain, top 15)

| feature | gain | share |
|---|---|---|
| own_elixir | 17463.0 | 0.300 |
| phase_idx | 8900.7 | 0.153 |
| opp_princess_hp_min | 5320.8 | 0.091 |
| own_princess_hp_min | 3254.5 | 0.056 |
| enemy_min_eta_own | 3140.1 | 0.054 |
| match_seconds | 2910.2 | 0.050 |
| since_own_play | 2610.3 | 0.045 |
| hand_cost_mean | 2095.2 | 0.036 |
| elixir_diff | 1805.3 | 0.031 |
| enemy_elixir_own_half | 1214.5 | 0.021 |
| enemy_elixir_field | 1071.4 | 0.018 |
| since_opp_play | 1039.2 | 0.018 |
| own_spent_5s | 1027.3 | 0.018 |
| opp_elixir_est | 818.0 | 0.014 |
| ally_count_enemy_half | 769.2 | 0.013 |

## Apples to apples with the d1 head (horizon 1 s, ids of `act_dense_bal`)

d1 numbers from `docs/data/act_head_v1.json` (threshold 0.5); `gbdt` is the h=1 model at the threshold chosen on the full val (0.26), `gbdt@bal` at the threshold chosen on the balanced val (0.26), `fast` = rules' none / defend / leak + gbdt.

### val  (n=3008, pro act rate 0.233, gbdt AUC 0.638)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| d1 learned | 0.554 | 0.573 | 0.496 | 0.286 | 0.608 | 0.818 | 0.538 |
| d1 rules | 0.506 | 0.504 | 0.493 | 0.236 | 0.499 | 0.769 | 0.509 |
| d1 combined | 0.469 | 0.528 | 0.595 | 0.250 | 0.638 | 0.791 | 0.418 |
| rules (recomputed) | 0.626 | 0.490 | 0.250 | 0.219 | 0.235 | 0.762 | 0.745 |
| gbdt | 0.616 | 0.594 | 0.408 | 0.316 | 0.553 | 0.824 | 0.636 |
| fast (rules + gbdt) | 0.567 | 0.556 | 0.450 | 0.278 | 0.536 | 0.803 | 0.576 |
| gbdt@bal | 0.616 | 0.594 | 0.408 | 0.316 | 0.553 | 0.824 | 0.636 |
| always | 0.233 | 0.500 | 1.000 | 0.233 | 1.000 | n/a | 0.000 |
| never | 0.767 | 0.500 | 0.000 | n/a | 0.000 | 0.767 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=1629

| policy | act rate | act recall | act prec |
|---|---|---|---|
| d1 learned | 0.579 | 0.675 | 0.301 |
| d1 rules | 0.566 | 0.563 | 0.257 |
| d1 combined | 0.656 | 0.682 | 0.268 |
| rules (recomputed) | 0.277 | 0.257 | 0.239 |
| gbdt | 0.509 | 0.629 | 0.320 |
| fast (rules + gbdt) | 0.513 | 0.582 | 0.293 |
| gbdt@bal | 0.509 | 0.629 | 0.320 |
| always | 1.000 | 1.000 | 0.258 |
| never | 0.000 | 0.000 | n/a |

### test  (n=3221, pro act rate 0.206, gbdt AUC 0.639)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| d1 learned | 0.580 | 0.558 | 0.429 | 0.250 | 0.522 | 0.828 | 0.595 |
| d1 rules | 0.552 | 0.535 | 0.450 | 0.232 | 0.507 | 0.815 | 0.564 |
| d1 combined | 0.506 | 0.545 | 0.540 | 0.233 | 0.611 | 0.826 | 0.478 |
| rules (recomputed) | 0.675 | 0.515 | 0.219 | 0.228 | 0.243 | 0.800 | 0.787 |
| gbdt | 0.635 | 0.598 | 0.379 | 0.291 | 0.535 | 0.846 | 0.661 |
| fast (rules + gbdt) | 0.600 | 0.568 | 0.406 | 0.261 | 0.514 | 0.832 | 0.622 |
| gbdt@bal | 0.635 | 0.598 | 0.379 | 0.291 | 0.535 | 0.846 | 0.661 |
| always | 0.206 | 0.500 | 1.000 | 0.206 | 1.000 | n/a | 0.000 |
| never | 0.794 | 0.500 | 0.000 | n/a | 0.000 | 0.794 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=1745

| policy | act rate | act recall | act prec |
|---|---|---|---|
| d1 learned | 0.493 | 0.577 | 0.266 |
| d1 rules | 0.513 | 0.554 | 0.246 |
| d1 combined | 0.598 | 0.665 | 0.253 |
| rules (recomputed) | 0.232 | 0.277 | 0.272 |
| gbdt | 0.470 | 0.584 | 0.283 |
| fast (rules + gbdt) | 0.467 | 0.552 | 0.269 |
| gbdt@bal | 0.470 | 0.584 | 0.283 |
| always | 1.000 | 1.000 | 0.228 |
| never | 0.000 | 0.000 | n/a |

## Predict latency (horizon 2 s, 500 val rows, single row, 1 thread)

| step | mean ms | p50 ms | p95 ms | max ms |
|---|---|---|---|---|
| features | 0.015 | 0.014 | 0.021 | 0.040 |
| predict | 0.009 | 0.009 | 0.011 | 0.016 |
| p_act | 0.025 | 0.025 | 0.032 | 0.040 |

p_act mean < 1 ms: True

