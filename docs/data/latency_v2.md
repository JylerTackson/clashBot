# Latency (MAC-AP)

git HEAD `9ca39a7` at run time; decision code frozen at `66917e7` (D-051). Times in ms. `decision_ms` stages are per decision, from each run's `summary.json`; the live runs include the 183 ms two-click `execute` stage (dry runs do not click, execute = 0).

## Whole stack, 3 repeated dry runs (video `Z8-4VhLjrGU` 60-180 s, d1-ft-joint-tac + fast policy, MPS)

Mean ± sd across runs (n-1) of each run's p50 and p95.

| stage | p50 mean ± sd | p95 mean ± sd | p50 per run | p95 per run |
|---|---|---|---|---|
| capture | 0.8 ± 0.1 | 4.3 ± 1.4 | 0.8, 0.9, 0.7 | 4.2, 3.0, 5.8 |
| perception | 179.2 ± 7.6 | 234.2 ± 29.2 | 170.7, 181.1, 185.6 | 207.3, 229.9, 265.3 |
| live_state | 0.2 ± 0.0 | 0.5 ± 0.1 | 0.2, 0.2, 0.2 | 0.6, 0.5, 0.4 |
| trigger | 2.7 ± 0.1 | 9.7 ± 3.7 | 2.8, 2.7, 2.5 | 10.2, 13.2, 5.8 |
| candidates | 46.7 ± 2.4 | 86.1 ± 12.4 | 45.4, 49.5, 45.3 | 85.0, 99.1, 74.3 |
| rank | 87.1 ± 10.6 | 191.0 ± 21.2 | 76.1, 97.2, 87.8 | 170.7, 212.9, 189.2 |
| execute | 0.0 ± 0.0 | 0.1 ± 0.0 | 0.0, 0.0, 0.0 | 0.1, 0.1, 0.1 |
| total | 271.7 ± 20.1 | 455.2 ± 41.2 | 260.6, 294.9, 259.5 | 410.2, 491.0, 464.3 |

Runs: lat_1, lat_2, lat_3; decisions per run: [45, 45, 45]; target p95 < 300 ms met in 0 of 3 runs.

## Every recorded run under `runs/poc` (p50 / p95 per stage)

| run | mode | decisions | capture | perception | live_state | trigger | candidates | rank | execute | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261103-192225 | live | 28 | 21 / 25 | 94 / 545 | 0 / 0 | 0 / 0 | 16 / 30 | 0 / 0 | 393 / 400 | 527 / 850 |
| 20261103-192709 | live | 25 | 22 / 32 | 93 / 216 | 0 / 0 | 0 / 0 | 24 / 50 | 0 / 0 | 383 / 398 | 441 / 624 |
| 20261105-194101 | live | 80 | 21 / 44 | 107 / 223 | 0 / 0 | 0 / 0 | 30 / 45 | 56 / 146 | 183 / 189 | 393 / 577 |
| 20261105-205720 | dry | 0 | - | - | - | - | - | - | - | - |
| 20261105-205934 | live | 61 | 21 / 46 | 96 / 202 | 0 / 0 | 0 / 0 | 31 / 39 | 0 / 77 | 183 / 189 | 331 / 492 |
| 20261106-133529 | live | 181 | 22 / 62 | 107 / 312 | 0 / 0 | 0 / 1 | 31 / 60 | 65 / 315 | 183 / 192 | 371 / 791 |
| 20261106-135724 | live | 61 | 20 / 33 | 45 / 225 | 0 / 0 | 0 / 1 | 25 / 49 | 0 / 142 | 182 / 186 | 263 / 572 |
| 20261106-170722 | live | 56 | 21 / 47 | 94 / 198 | 0 / 0 | 0 / 1 | 23 / 52 | 0 / 74 | 183 / 186 | 303 / 486 |
| 20261106-233907 | live | 116 | 21 / 38 | 97 / 225 | 0 / 0 | 1 / 4 | 25 / 44 | 0 / 120 | 184 / 186 | 336 / 506 |
| 20261107-111554 | live | 125 | 21 / 32 | 102 / 230 | 0 / 0 | 2 / 9 | 26 / 57 | 0 / 137 | 183 / 188 | 339 / 572 |
| median of live runs | live | - | 21 / 38 | 96 / 225 | 0 / 0 | 0 / 1 | 25 / 49 | 0 / 120 | 183 / 189 | 339 / 572 |

## Isolated micro-benchmarks (200 calls each after warm-up)

| benchmark | n | mean | p50 | p95 | min | max |
|---|---|---|---|---|---|---|
| katacr / mps | 200 | 105.31 | 106.08 | 110.22 | 95.71 | 119.46 |
| katacr / cpu | 200 | 471.41 | 465.64 | 498.10 | 440.01 | 549.41 |
| d1 | 200 | 75.11 | 68.13 | 110.29 | 49.16 | 178.15 |
| act / p_act | 200 | 0.03 | 0.03 | 0.03 | 0.02 | 0.04 |
| act / features_only | 200 | 0.01 | 0.01 | 0.02 | 0.01 | 0.03 |
| knn / StateIndex.knn exact (k=10) | 200 | 1.50 | 1.53 | 2.04 | 1.06 | 2.33 |
| knn / StateIndex.knn hnsw (k=10) | 200 | 0.67 | 0.67 | 0.83 | 0.41 | 1.08 |
| knn / raw exact (matmul + top-10) | 200 | 1.71 | 1.54 | 2.37 | 1.36 | 2.69 |
| knn / raw hnswlib knn_query (k=10) | 200 | 0.33 | 0.33 | 0.41 | 0.21 | 0.64 |
| candidates | 200 | 28.79 | 29.76 | 40.90 | 13.38 | 54.67 |

- katacr: frame = data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 t=90 s, content 566x1195, arena crop 566x839, imgsz 640
- d1: cold_first_call_ms = 3017.4; note = each call = ranker.probabilities on one joint_tac_v2 line (4 options); 50 distinct lines cycled
- act: horizon_s = 2.0
- knn: n_vectors = 10923; dim = 644; ef = 40; hnsw_recall_at_10_vs_exact = 0.986
- candidates: note = 50 distinct val states cycled 4x; no tactics report, no exclude_game filter (as in the live stack)
