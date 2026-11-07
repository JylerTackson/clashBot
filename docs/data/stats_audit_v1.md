# Statistics audit of the 4-way table (E1, MAC-AP)

- git HEAD `54243d7` at run time; frozen decision code `66917e7` (D-051). Lines: val 761 (26 games), test 766 (27 games).
- CIs: percentile, 10,000 cluster-bootstrap resamples of whole `game_id`s (seed 20261107); the same resampled games are used for every statistic of a split, so differences are paired.
- McNemar p is exact and treats lines as independent (lines of one game are correlated, so it is the optimistic one); `p boot` is the two-sided cluster-bootstrap p of the difference.
- Thresholds and the ensemble weight are chosen on val only. LightGBM / DCN-V2 / logreg configs were selected on val accuracy (feature_ranker.py), so the val column is optimistic for them; test is the clean column.


Reproduction: recomputed vs recorded accuracies, largest gap in points: rankers vs `feature_ranker_v1.json` 0.0 (within 0.5 pt: True), baselines vs `ranker/baselines.json` 0.003, d1 vs the eval JSON `acc` 0.005.

## 1. Accuracy with 95% CI

| model | val | test |
|---|---|---|
| random (one seeded pick per line) | 0.230 [0.207, 0.254] | 0.252 [0.218, 0.284] |
| popularity (train key counts) | 0.468 [0.412, 0.516] | 0.325 [0.280, 0.374] |
| placement prior | 0.493 [0.430, 0.547] | 0.384 [0.337, 0.431] |
| logreg (seed 0) | 0.543 [0.492, 0.586] | 0.470 [0.423, 0.512] |
| LightGBM LambdaRank (seed 0) | 0.565 [0.520, 0.604] | 0.482 [0.448, 0.515] |
| DCN-V2 (seed 0) | 0.537 [0.493, 0.576] | 0.403 [0.366, 0.439] |
| d1 zero-shot | 0.258 [0.226, 0.287] | 0.221 [0.191, 0.252] |
| d1-ft-joint (plain lines) | 0.524 [0.483, 0.561] | 0.483 [0.442, 0.522] |
| d1-ft-joint-tac (HEAD lines) | 0.547 [0.503, 0.584] | 0.507 [0.472, 0.540] |
| d1-ft-joint-kg (HEAD lines) | 0.530 [0.480, 0.572] | 0.484 [0.450, 0.518] |
| d1-ft-joint on HEAD tactics lines (extra) | 0.477 [0.427, 0.523] | 0.461 [0.418, 0.506] |
| d1-ft-joint-tac (as-trained lines) (extra) | 0.540 [0.498, 0.577] | 0.497 [0.462, 0.529] |

## 2. Paired differences (accuracy points, a minus b)

| a vs b | split | delta [95% CI] | a only right | b only right | McNemar p | p boot |
|---|---|---|---|---|---|---|
| d1_tac vs lgbm (spec) | test | +2.5 [-1.4, +5.9] | 139 | 120 | 0.263 | 0.212 |
| d1_tac vs d1_joint (spec) | test | +2.3 [-0.9, +5.3] | 82 | 64 | 0.159 | 0.156 |
| d1_tac vs placement_prior (spec) | test | +12.3 [+8.0, +16.7] | 196 | 102 | <0.001 | <0.001 |
| d1_kg vs d1_tac (spec) | test | -2.2 [-4.7, +0.5] | 55 | 72 | 0.155 | 0.119 |
| d1_tac vs d1_joint_on_tac (extra) | test | +4.6 [+1.0, +8.4] | 96 | 61 | 0.006 | 0.013 |
| d1_tac vs d1_tac_v1 (extra) | test | +0.9 [-1.1, +3.0] | 49 | 42 | 0.530 | 0.428 |
| d1_tac vs logreg (extra) | test | +3.7 [-0.7, +8.5] | 125 | 97 | 0.070 | 0.113 |
| d1_tac vs dcn_v2 (extra) | test | +10.3 [+6.6, +14.3] | 170 | 91 | <0.001 | <0.001 |
| d1_tac vs popularity (extra) | test | +18.1 [+11.7, +23.9] | 227 | 88 | <0.001 | <0.001 |
| d1_joint vs lgbm (extra) | test | +0.1 [-4.0, +3.8] | 133 | 132 | 1.000 | 0.950 |
| d1_tac vs lgbm_s1 (robustness) | test | +2.3 [-3.9, +8.1] | 157 | 139 | 0.323 | 0.474 |
| d1_tac vs lgbm_s2 (robustness) | test | -0.3 [-4.9, +4.0] | 131 | 133 | 0.951 | 0.943 |
| d1_tac vs lgbm (spec) | val | -1.8 [-4.9, +1.4] | 104 | 118 | 0.383 | 0.285 |
| d1_tac vs d1_joint (spec) | val | +2.2 [-0.9, +5.7] | 76 | 59 | 0.168 | 0.193 |
| d1_tac vs placement_prior (spec) | val | +5.4 [+1.2, +10.1] | 152 | 111 | 0.013 | 0.012 |
| d1_kg vs d1_tac (spec) | val | -1.7 [-4.7, +1.7] | 51 | 64 | 0.263 | 0.323 |
| d1_tac vs d1_joint_on_tac (extra) | val | +7.0 [+2.5, +11.1] | 103 | 50 | <0.001 | 0.002 |
| d1_tac vs d1_tac_v1 (extra) | val | +0.7 [-1.3, +2.5] | 41 | 36 | 0.649 | 0.551 |
| d1_tac vs logreg (extra) | val | +0.4 [-2.4, +3.6] | 105 | 102 | 0.889 | 0.832 |
| d1_tac vs dcn_v2 (extra) | val | +0.9 [-2.1, +4.0] | 113 | 106 | 0.685 | 0.570 |
| d1_tac vs popularity (extra) | val | +7.9 [+4.0, +12.0] | 157 | 97 | <0.001 | <0.001 |
| d1_joint vs lgbm (extra) | val | -4.1 [-7.1, -1.1] | 97 | 128 | 0.045 | 0.009 |
| d1_tac vs lgbm_s1 (robustness) | val | +1.6 [-1.5, +4.7] | 131 | 119 | 0.487 | 0.338 |
| d1_tac vs lgbm_s2 (robustness) | val | -0.3 [-3.4, +2.4] | 108 | 110 | 0.946 | 0.908 |

## 3. Calibration of d1-tac

| split | ECE (15 bins) [CI] | Brier [CI] | mean confidence | accuracy |
|---|---|---|---|---|
| val | 0.040 [0.031, 0.083] | 0.575 [0.540, 0.614] | 0.529 | 0.547 |
| test | 0.048 [0.039, 0.089] | 0.616 [0.594, 0.641] | 0.480 | 0.507 |

The ECE interval sits above the point estimate: ECE is biased upward on resamples (every resample re-bins noisy bins), so read the point estimate with the width of the interval, not the interval's centre.

Reliability, val (empty bins omitted)

| confidence bin | n | mean conf | accuracy | gap |
|---|---|---|---|---|
| 0.27-0.33 | 79 | 0.310 | 0.266 | -0.044 |
| 0.33-0.40 | 136 | 0.369 | 0.426 | +0.058 |
| 0.40-0.47 | 120 | 0.431 | 0.475 | +0.044 |
| 0.47-0.53 | 121 | 0.497 | 0.545 | +0.048 |
| 0.53-0.60 | 91 | 0.566 | 0.582 | +0.017 |
| 0.60-0.67 | 63 | 0.633 | 0.587 | -0.046 |
| 0.67-0.73 | 43 | 0.694 | 0.674 | -0.020 |
| 0.73-0.80 | 30 | 0.762 | 0.733 | -0.029 |
| 0.80-0.87 | 18 | 0.831 | 0.889 | +0.057 |
| 0.87-0.93 | 15 | 0.902 | 0.933 | +0.032 |
| 0.93-1.00 | 45 | 0.965 | 0.956 | -0.010 |

Reliability, test (empty bins omitted)

| confidence bin | n | mean conf | accuracy | gap |
|---|---|---|---|---|
| 0.20-0.27 | 1 | 0.260 | 0.000 | -0.260 |
| 0.27-0.33 | 92 | 0.310 | 0.304 | -0.006 |
| 0.33-0.40 | 179 | 0.365 | 0.458 | +0.093 |
| 0.40-0.47 | 161 | 0.435 | 0.447 | +0.012 |
| 0.47-0.53 | 109 | 0.498 | 0.514 | +0.016 |
| 0.53-0.60 | 69 | 0.567 | 0.522 | -0.045 |
| 0.60-0.67 | 61 | 0.629 | 0.574 | -0.055 |
| 0.67-0.73 | 41 | 0.700 | 0.805 | +0.105 |
| 0.73-0.80 | 28 | 0.760 | 0.857 | +0.097 |
| 0.80-0.87 | 13 | 0.821 | 0.923 | +0.102 |
| 0.87-0.93 | 9 | 0.900 | 0.778 | -0.122 |
| 0.93-1.00 | 3 | 0.953 | 1.000 | +0.047 |

Other d1 models (ECE / Brier, test): d1_zero_shot 0.175 / 0.823; d1_joint 0.051 / 0.627; d1_kg 0.020 / 0.630; d1_joint_on_tac 0.048 / 0.642; d1_tac_v1 0.037 / 0.619

## 4. Selective prediction (hold when unsure), d1-tac

Rule: answer when max probability >= threshold; threshold = best +1/0/-1 utility on val. Val-chosen threshold **0.4830**. Utility = (right - wrong) / N, abstain = 0.

| split | coverage | accuracy answered | utility at threshold | utility answering everything | AURC (risk) |
|---|---|---|---|---|---|
| val | 0.510 [0.451, 0.560] | 0.683 [0.630, 0.737] | 0.187 [0.130, 0.242] | 0.093 [0.005, 0.169] | 0.288 |
| test | 0.402 [0.359, 0.443] | 0.630 [0.577, 0.680] | 0.104 [0.060, 0.147] | 0.013 [-0.056, 0.079] | 0.363 |

Risk-coverage curve: `stats_audit_v1_risk_coverage.csv`.

## 5. Accuracy by popularity tercile of the gold key (train counts)

Terciles pooled over val + test lines; low: count <= 9; mid: <= 29; high: above. Share of gold keys never seen in train: val 0.04, test 0.14.

| model | split | low | mid | high |
|---|---|---|---|---|
| n lines | test | 348 | 201 | 217 |
| random (one seeded pick per line) | test | 0.239 [0.194, 0.283] | 0.254 [0.178, 0.318] | 0.272 [0.212, 0.337] |
| popularity (train key counts) | test | 0.103 [0.068, 0.149] | 0.308 [0.234, 0.400] | 0.696 [0.624, 0.771] |
| placement prior | test | 0.310 [0.243, 0.378] | 0.398 [0.293, 0.504] | 0.488 [0.443, 0.546] |
| logreg (seed 0) | test | 0.428 [0.359, 0.494] | 0.383 [0.317, 0.449] | 0.618 [0.544, 0.684] |
| LightGBM LambdaRank (seed 0) | test | 0.457 [0.400, 0.511] | 0.458 [0.377, 0.541] | 0.544 [0.471, 0.606] |
| DCN-V2 (seed 0) | test | 0.365 [0.304, 0.428] | 0.338 [0.264, 0.404] | 0.525 [0.438, 0.621] |
| d1 zero-shot | test | 0.201 [0.160, 0.245] | 0.264 [0.201, 0.338] | 0.212 [0.165, 0.260] |
| d1-ft-joint (plain lines) | test | 0.425 [0.367, 0.484] | 0.423 [0.344, 0.500] | 0.631 [0.564, 0.692] |
| d1-ft-joint-tac (HEAD lines) | test | 0.468 [0.418, 0.517] | 0.473 [0.396, 0.547] | 0.599 [0.535, 0.658] |
| d1-ft-joint-kg (HEAD lines) | test | 0.443 [0.393, 0.495] | 0.413 [0.347, 0.477] | 0.618 [0.555, 0.678] |
| d1-ft-joint on HEAD tactics lines | test | 0.425 [0.360, 0.487] | 0.403 [0.315, 0.497] | 0.571 [0.510, 0.624] |
| d1-ft-joint-tac (as-trained lines) | test | 0.434 [0.374, 0.495] | 0.443 [0.374, 0.515] | 0.650 [0.577, 0.714] |
| n lines | val | 170 | 323 | 268 |
| random (one seeded pick per line) | val | 0.241 [0.182, 0.302] | 0.251 [0.217, 0.284] | 0.198 [0.154, 0.242] |
| popularity (train key counts) | val | 0.135 [0.066, 0.215] | 0.375 [0.325, 0.426] | 0.791 [0.729, 0.838] |
| placement prior | val | 0.276 [0.187, 0.376] | 0.517 [0.448, 0.580] | 0.601 [0.531, 0.659] |
| logreg (seed 0) | val | 0.247 [0.166, 0.318] | 0.529 [0.480, 0.575] | 0.746 [0.659, 0.817] |
| LightGBM LambdaRank (seed 0) | val | 0.335 [0.257, 0.402] | 0.601 [0.555, 0.642] | 0.668 [0.572, 0.743] |
| DCN-V2 (seed 0) | val | 0.218 [0.115, 0.325] | 0.542 [0.489, 0.588] | 0.735 [0.684, 0.779] |
| d1 zero-shot | val | 0.224 [0.161, 0.299] | 0.344 [0.288, 0.395] | 0.175 [0.132, 0.218] |
| d1-ft-joint (plain lines) | val | 0.271 [0.204, 0.339] | 0.545 [0.495, 0.589] | 0.660 [0.612, 0.707] |
| d1-ft-joint-tac (HEAD lines) | val | 0.353 [0.250, 0.465] | 0.548 [0.500, 0.592] | 0.668 [0.611, 0.716] |
| d1-ft-joint-kg (HEAD lines) | val | 0.282 [0.180, 0.392] | 0.567 [0.522, 0.609] | 0.642 [0.592, 0.684] |
| d1-ft-joint on HEAD tactics lines | val | 0.282 [0.214, 0.360] | 0.489 [0.439, 0.533] | 0.586 [0.509, 0.652] |
| d1-ft-joint-tac (as-trained lines) | val | 0.324 [0.230, 0.426] | 0.560 [0.506, 0.608] | 0.653 [0.599, 0.699] |

## 6. Ensemble d1-tac + LightGBM

Score = w * z(log p of d1-tac) + (1 - w) * z(LightGBM score), each z-scored across the 4 options of a line; w (the d1 weight) picked on val from {0, 0.05, ..., 1}, ties to the most even blend.

| LightGBM seed | w (d1) | split | ensemble acc [CI] | vs d1-tac (pts) | p McNemar | vs LightGBM (pts) | p McNemar |
|---|---|---|---|---|---|---|---|
| lgbm | 0.35 | test | 0.505 [0.472, 0.535] | -0.1 [-3.0, +2.9] | 1.000 | +2.3 [-0.4, +5.1] | 0.098 |
| lgbm | 0.35 | val | 0.576 [0.534, 0.611] | +2.9 [+0.6, +5.0] | 0.084 | +1.1 [-0.9, +3.3] | 0.445 |
| lgbm_s1 | 0.55 | test | 0.546 [0.511, 0.579] | +3.9 [+0.9, +7.1] | 0.008 | +6.3 [+1.0, +11.2] | <0.001 |
| lgbm_s1 | 0.55 | val | 0.574 [0.529, 0.613] | +2.8 [+0.4, +5.0] | 0.031 | +4.3 [+1.0, +7.5] | 0.015 |
| lgbm_s2 | 0.25 | test | 0.527 [0.493, 0.562] | +2.1 [-1.8, +6.2] | 0.286 | +1.8 [-0.8, +4.6] | 0.151 |
| lgbm_s2 | 0.25 | val | 0.569 [0.530, 0.605] | +2.2 [-0.6, +5.3] | 0.229 | +2.0 [+0.1, +3.8] | 0.063 |
