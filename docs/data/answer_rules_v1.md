# E6 - held-out evaluation of the mined answer rules (MAC-AR R1)

Run at HEAD `54243d78bc`; bootstrap 10000 resamples over games (seed 0); source weights {'pro': 0.45, 'role_transfer': 0.3, 'wiki': 0.05, 'sim': 0.15}.

## Setup and leakage check

Rules are the train-split ANSWERED_BY edges (support >= 3, answer window 8 s). Each val / test event is one opponent play and the first defensive pro play after it; the task is to rank the pro's own deck (7-8 cards) for that threat. Ties are scored in expectation, so a method with no evidence equals random.

- events per split: {'test': 436, 'train': 2575, 'val': 609}; held-out games: {'val': 26, 'test': 27}
- rules over all events equal rules over train events only: True; equal the shipped answers.jsonl (287 vs 287 rules): True; every rule's support is within its train events: True
- events left out because the pro's answer is not in the recorded deck: {'answer_not_in_deck': 3}

## Headline - val (609 events, 26 games, mean deck 7.96)

| method | hit@1 | hit@3 | MRR | coverage | hit@1 if covered |
|---|---|---|---|---|---|
| full KG2.answers_for | 0.182 [0.156, 0.209] | 0.466 [0.429, 0.504] | 0.402 [0.383, 0.423] | 100% [99, 100] | 0.182 [0.156, 0.210] |
| pro only | 0.191 [0.161, 0.220] | 0.443 [0.424, 0.464] | 0.396 [0.377, 0.415] | 68% [59, 76] | 0.220 [0.177, 0.271] |
| role_transfer only | 0.146 [0.125, 0.170] | 0.429 [0.401, 0.455] | 0.368 [0.350, 0.386] | 70% [61, 79] | 0.155 [0.124, 0.192] |
| wiki only | 0.117 [0.096, 0.139] | 0.372 [0.329, 0.413] | 0.334 [0.313, 0.355] | 93% [89, 96] | 0.116 [0.094, 0.141] |
| sim only | 0.165 [0.140, 0.191] | 0.437 [0.401, 0.472] | 0.383 [0.363, 0.405] | 77% [69, 84] | 0.176 [0.144, 0.209] |
| all but pro | 0.158 [0.134, 0.183] | 0.440 [0.389, 0.492] | 0.380 [0.355, 0.406] | 100% [99, 100] | 0.158 [0.134, 0.183] |
| all but role_transfer | 0.187 [0.154, 0.224] | 0.452 [0.415, 0.490] | 0.401 [0.377, 0.429] | 99% [96, 100] | 0.188 [0.155, 0.226] |
| all but wiki | 0.200 [0.173, 0.230] | 0.493 [0.452, 0.533] | 0.417 [0.395, 0.442] | 98% [95, 100] | 0.202 [0.174, 0.234] |
| all but sim | 0.168 [0.148, 0.189] | 0.469 [0.442, 0.499] | 0.390 [0.377, 0.405] | 99% [98, 100] | 0.168 [0.148, 0.190] |
| wiki COUNTERS only (KnowledgeGraph) | 0.117 [0.093, 0.144] | 0.367 [0.323, 0.410] | 0.333 [0.311, 0.358] | 93% [89, 96] | 0.116 [0.092, 0.146] |
| KnowledgeGraph.counter_scores (COUNTERS + v1 train-mined ANSWERED_BY) | 0.114 [0.091, 0.140] | 0.359 [0.317, 0.402] | 0.330 [0.308, 0.353] | 96% [93, 98] | 0.113 [0.090, 0.141] |
| train popularity of the answer card | 0.182 [0.147, 0.220] | 0.517 [0.456, 0.582] | 0.420 [0.392, 0.450] | 100% [100, 100] | 0.182 [0.147, 0.220] |
| random in deck (expected) | 0.126 [0.125, 0.127] | 0.377 [0.375, 0.381] | 0.341 [0.340, 0.343] | - | - |

Paired differences (val; CI over games, p = bootstrap two-sided):

| a - b | hit@1 | hit@3 | MRR |
|---|---|---|---|
| full - popularity | -0.001 [-0.038, +0.034] p=1.000 | -0.051 [-0.123, +0.025] p=0.178 | -0.018 [-0.048, +0.010] p=0.224 |
| full - random | +0.056 [+0.030, +0.084] p=0.000 | +0.089 [+0.052, +0.126] p=0.000 | +0.061 [+0.042, +0.082] p=0.000 |
| full - wiki_counters | +0.065 [+0.023, +0.103] p=0.003 | +0.099 [+0.043, +0.153] p=0.000 | +0.069 [+0.035, +0.099] p=0.000 |
| only:pro - popularity | +0.008 [-0.039, +0.053] p=0.750 | -0.074 [-0.139, -0.008] p=0.031 | -0.024 [-0.059, +0.008] p=0.137 |
| popularity - random | +0.057 [+0.022, +0.095] p=0.001 | +0.140 [+0.078, +0.205] p=0.000 | +0.079 [+0.051, +0.109] p=0.000 |
| full - only:pro | -0.009 [-0.044, +0.025] p=0.665 | +0.023 [-0.008, +0.054] p=0.137 | +0.006 [-0.015, +0.027] p=0.528 |
| without:pro - random | +0.032 [+0.008, +0.057] p=0.007 | +0.063 [+0.012, +0.114] p=0.015 | +0.039 [+0.014, +0.065] p=0.002 |

Held by the rank the full score gave the pro's answer (val):

| rank of the answer | events | share held |
|---|---|---|
| rank 1 | 112 | 88% [79, 95] |
| rank 2 | 96 | 87% [79, 93] |
| rank 3 | 76 | 81% [72, 89] |
| rank 4+ | 205 | 82% [74, 89] |
| no evidence | 120 | 81% [68, 93] |

Held by the full score of the pro's answer (val); AUROC of the score for held vs tower-hit = 0.560 (498 held, 98 not):

| score | events | share held |
|---|---|---|
| 0 (none) | 120 | 81% [68, 93] |
| (0, 0.2] | 271 | 82% [74, 88] |
| (0.2, 0.4] | 173 | 86% [77, 93] |
| (0.4, 0.6] | 41 | 90% [81, 100] |
| (0.6, 1] | 4 | 100% [100, 100] |

By the strongest pro rule available in the deck (val; hit@1 / MRR):

| stratum | events | full KG2.answers_for | pro only | role_transfer only | wiki only | sim only | train popularity of the answer card | random in deck (expected) |
|---|---|---|---|---|---|---|---|---|
| no pro rule for this threat in the deck | 194 | 0.137 / 0.369 | 0.126 / 0.342 | 0.144 / 0.357 | 0.138 / 0.361 | 0.167 / 0.379 | 0.206 / 0.444 | 0.126 / 0.342 |
| best pro rule support 3-4 | 151 | 0.205 / 0.429 | 0.225 / 0.427 | 0.169 / 0.388 | 0.145 / 0.353 | 0.181 / 0.393 | 0.199 / 0.440 | 0.126 / 0.341 |
| best pro rule support 5-9 | 228 | 0.215 / 0.427 | 0.235 / 0.432 | 0.135 / 0.365 | 0.085 / 0.307 | 0.157 / 0.390 | 0.162 / 0.395 | 0.125 / 0.340 |
| best pro rule support 10+ | 36 | 0.111 / 0.306 | 0.111 / 0.321 | 0.139 / 0.366 | 0.079 / 0.285 | 0.135 / 0.325 | 0.111 / 0.366 | 0.125 / 0.340 |

By how the threat was found (val; play = the opponent was seen playing the card, unit = a unit crossed with no play seen):

| threat source | method | hit@1 | MRR | coverage |
|---|---|---|---|---|
| play | full KG2.answers_for | 0.181 [0.142, 0.225] | 0.406 [0.381, 0.435] | 100% [99, 100] |
| play | train popularity of the answer card | 0.171 [0.133, 0.207] | 0.412 [0.381, 0.443] | 100% [100, 100] |
| play | random in deck (expected) | 0.126 [0.125, 0.127] | 0.341 [0.340, 0.344] | 0% [0, 0] |
| unit | full KG2.answers_for | 0.182 [0.131, 0.227] | 0.394 [0.352, 0.431] | 100% [100, 100] |
| unit | train popularity of the answer card | 0.203 [0.148, 0.262] | 0.434 [0.390, 0.478] | 100% [100, 100] |
| unit | random in deck (expected) | 0.125 [0.125, 0.127] | 0.340 [0.340, 0.342] | 0% [0, 0] |

## Headline - test (433 events, 27 games, mean deck 7.83)

| method | hit@1 | hit@3 | MRR | coverage | hit@1 if covered |
|---|---|---|---|---|---|
| full KG2.answers_for | 0.154 [0.117, 0.196] | 0.422 [0.372, 0.480] | 0.374 [0.343, 0.410] | 99% [97, 100] | 0.155 [0.117, 0.197] |
| pro only | 0.129 [0.096, 0.168] | 0.399 [0.367, 0.438] | 0.350 [0.325, 0.381] | 63% [50, 75] | 0.130 [0.080, 0.195] |
| role_transfer only | 0.145 [0.118, 0.172] | 0.398 [0.366, 0.430] | 0.356 [0.335, 0.376] | 67% [55, 77] | 0.152 [0.111, 0.194] |
| wiki only | 0.147 [0.123, 0.173] | 0.407 [0.371, 0.442] | 0.362 [0.341, 0.383] | 97% [93, 99] | 0.148 [0.123, 0.175] |
| sim only | 0.157 [0.140, 0.176] | 0.440 [0.410, 0.473] | 0.376 [0.359, 0.395] | 77% [66, 86] | 0.165 [0.145, 0.192] |
| all but pro | 0.169 [0.140, 0.200] | 0.422 [0.379, 0.465] | 0.382 [0.356, 0.408] | 99% [97, 100] | 0.170 [0.140, 0.200] |
| all but role_transfer | 0.148 [0.108, 0.193] | 0.432 [0.385, 0.488] | 0.372 [0.340, 0.410] | 99% [97, 100] | 0.148 [0.108, 0.193] |
| all but wiki | 0.153 [0.114, 0.198] | 0.433 [0.389, 0.484] | 0.374 [0.343, 0.411] | 95% [91, 98] | 0.154 [0.113, 0.200] |
| all but sim | 0.158 [0.121, 0.198] | 0.415 [0.370, 0.466] | 0.371 [0.341, 0.404] | 99% [97, 100] | 0.159 [0.121, 0.198] |
| wiki COUNTERS only (KnowledgeGraph) | 0.141 [0.121, 0.163] | 0.406 [0.369, 0.444] | 0.359 [0.339, 0.379] | 97% [93, 99] | 0.142 [0.121, 0.164] |
| KnowledgeGraph.counter_scores (COUNTERS + v1 train-mined ANSWERED_BY) | 0.143 [0.122, 0.167] | 0.409 [0.372, 0.447] | 0.360 [0.339, 0.381] | 97% [94, 99] | 0.144 [0.122, 0.168] |
| train popularity of the answer card | 0.169 [0.124, 0.223] | 0.433 [0.374, 0.492] | 0.384 [0.349, 0.427] | 100% [100, 100] | 0.169 [0.124, 0.223] |
| random in deck (expected) | 0.128 [0.125, 0.132] | 0.384 [0.375, 0.395] | 0.345 [0.340, 0.351] | - | - |

Paired differences (test; CI over games, p = bootstrap two-sided):

| a - b | hit@1 | hit@3 | MRR |
|---|---|---|---|
| full - popularity | -0.014 [-0.062, +0.024] p=0.509 | -0.011 [-0.064, +0.051] p=0.698 | -0.010 [-0.043, +0.020] p=0.527 |
| full - random | +0.026 [-0.013, +0.068] p=0.186 | +0.038 [-0.010, +0.095] p=0.128 | +0.029 [-0.003, +0.065] p=0.082 |
| full - wiki_counters | +0.013 [-0.029, +0.059] p=0.570 | +0.016 [-0.036, +0.077] p=0.570 | +0.015 [-0.019, +0.054] p=0.416 |
| only:pro - popularity | -0.040 [-0.083, -0.006] p=0.022 | -0.034 [-0.079, +0.017] p=0.180 | -0.034 [-0.065, -0.005] p=0.017 |
| popularity - random | +0.040 [-0.005, +0.095] p=0.084 | +0.049 [-0.007, +0.105] p=0.088 | +0.039 [+0.005, +0.080] p=0.023 |
| full - only:pro | +0.025 [+0.004, +0.047] p=0.019 | +0.023 [-0.011, +0.061] p=0.186 | +0.023 [+0.005, +0.043] p=0.013 |
| without:pro - random | +0.041 [+0.012, +0.072] p=0.006 | +0.037 [-0.004, +0.080] p=0.074 | +0.036 [+0.011, +0.063] p=0.004 |

Held by the rank the full score gave the pro's answer (test):

| rank of the answer | events | share held |
|---|---|---|
| rank 1 | 70 | 87% [78, 95] |
| rank 2 | 70 | 86% [77, 94] |
| rank 3 | 41 | 87% [73, 97] |
| rank 4+ | 164 | 88% [83, 93] |
| no evidence | 88 | 83% [72, 91] |

Held by the full score of the pro's answer (test); AUROC of the score for held vs tower-hit = 0.523 (364 held, 58 not):

| score | events | share held |
|---|---|---|
| 0 (none) | 88 | 83% [72, 91] |
| (0, 0.2] | 225 | 87% [82, 92] |
| (0.2, 0.4] | 82 | 84% [75, 92] |
| (0.4, 0.6] | 25 | 92% [83, 100] |
| (0.6, 1] | 13 | 100% [100, 100] |

By the strongest pro rule available in the deck (test; hit@1 / MRR):

| stratum | events | full KG2.answers_for | pro only | role_transfer only | wiki only | sim only | train popularity of the answer card | random in deck (expected) |
|---|---|---|---|---|---|---|---|---|
| no pro rule for this threat in the deck | 159 | 0.188 / 0.401 | 0.128 / 0.345 | 0.131 / 0.347 | 0.205 / 0.408 | 0.176 / 0.393 | 0.201 / 0.416 | 0.128 / 0.345 |
| best pro rule support 3-4 | 154 | 0.136 / 0.368 | 0.153 / 0.367 | 0.129 / 0.345 | 0.134 / 0.357 | 0.160 / 0.378 | 0.143 / 0.363 | 0.130 / 0.348 |
| best pro rule support 5-9 | 93 | 0.129 / 0.349 | 0.118 / 0.353 | 0.160 / 0.367 | 0.099 / 0.322 | 0.133 / 0.355 | 0.183 / 0.390 | 0.126 / 0.342 |
| best pro rule support 10+ | 27 | 0.148 / 0.331 | 0.037 / 0.277 | 0.259 / 0.434 | 0.046 / 0.250 | 0.106 / 0.328 | 0.074 / 0.295 | 0.125 / 0.340 |

By how the threat was found (test; play = the opponent was seen playing the card, unit = a unit crossed with no play seen):

| threat source | method | hit@1 | MRR | coverage |
|---|---|---|---|---|
| play | full KG2.answers_for | 0.138 [0.091, 0.192] | 0.364 [0.323, 0.410] | 99% [96, 100] |
| play | train popularity of the answer card | 0.154 [0.096, 0.225] | 0.370 [0.324, 0.425] | 100% [100, 100] |
| play | random in deck (expected) | 0.129 [0.125, 0.133] | 0.347 [0.340, 0.354] | 0% [0, 0] |
| unit | full KG2.answers_for | 0.185 [0.146, 0.228] | 0.391 [0.360, 0.428] | 99% [98, 100] |
| unit | train popularity of the answer card | 0.195 [0.121, 0.286] | 0.409 [0.348, 0.483] | 100% [100, 100] |
| unit | random in deck (expected) | 0.126 [0.125, 0.128] | 0.342 [0.340, 0.345] | 0% [0, 0] |

## Headline - val+test (1042 events, 53 games, mean deck 7.90)

| method | hit@1 | hit@3 | MRR | coverage | hit@1 if covered |
|---|---|---|---|---|---|
| full KG2.answers_for | 0.170 [0.148, 0.194] | 0.448 [0.416, 0.482] | 0.390 [0.372, 0.410] | 100% [99, 100] | 0.171 [0.148, 0.195] |
| pro only | 0.165 [0.140, 0.191] | 0.425 [0.405, 0.446] | 0.377 [0.359, 0.395] | 66% [59, 73] | 0.184 [0.146, 0.226] |
| role_transfer only | 0.146 [0.129, 0.164] | 0.416 [0.394, 0.436] | 0.363 [0.349, 0.376] | 69% [61, 75] | 0.154 [0.129, 0.182] |
| wiki only | 0.129 [0.113, 0.147] | 0.386 [0.356, 0.415] | 0.346 [0.329, 0.362] | 95% [92, 97] | 0.130 [0.112, 0.148] |
| sim only | 0.161 [0.145, 0.179] | 0.438 [0.414, 0.463] | 0.380 [0.366, 0.396] | 77% [71, 83] | 0.172 [0.151, 0.194] |
| all but pro | 0.163 [0.144, 0.182] | 0.432 [0.397, 0.467] | 0.381 [0.362, 0.400] | 100% [99, 100] | 0.163 [0.144, 0.182] |
| all but role_transfer | 0.171 [0.145, 0.199] | 0.444 [0.413, 0.476] | 0.389 [0.369, 0.413] | 99% [97, 100] | 0.171 [0.145, 0.200] |
| all but wiki | 0.181 [0.156, 0.207] | 0.468 [0.436, 0.501] | 0.399 [0.379, 0.421] | 97% [94, 99] | 0.182 [0.157, 0.209] |
| all but sim | 0.164 [0.144, 0.184] | 0.446 [0.420, 0.475] | 0.382 [0.367, 0.399] | 99% [98, 99] | 0.164 [0.145, 0.185] |
| wiki COUNTERS only (KnowledgeGraph) | 0.127 [0.110, 0.145] | 0.383 [0.351, 0.414] | 0.344 [0.327, 0.361] | 95% [92, 97] | 0.127 [0.109, 0.147] |
| KnowledgeGraph.counter_scores (COUNTERS + v1 train-mined ANSWERED_BY) | 0.126 [0.109, 0.145] | 0.380 [0.348, 0.411] | 0.342 [0.326, 0.359] | 96% [94, 98] | 0.126 [0.108, 0.145] |
| train popularity of the answer card | 0.177 [0.148, 0.209] | 0.482 [0.436, 0.529] | 0.405 [0.382, 0.430] | 100% [100, 100] | 0.177 [0.148, 0.209] |
| random in deck (expected) | 0.127 [0.125, 0.128] | 0.380 [0.376, 0.385] | 0.343 [0.340, 0.346] | - | - |

Paired differences (val+test; CI over games, p = bootstrap two-sided):

| a - b | hit@1 | hit@3 | MRR |
|---|---|---|---|
| full - popularity | -0.006 [-0.035, +0.020] p=0.651 | -0.035 [-0.083, +0.017] p=0.181 | -0.015 [-0.037, +0.007] p=0.178 |
| full - random | +0.044 [+0.020, +0.068] p=0.000 | +0.068 [+0.036, +0.101] p=0.000 | +0.048 [+0.029, +0.068] p=0.000 |
| full - wiki_counters | +0.043 [+0.011, +0.074] p=0.010 | +0.064 [+0.023, +0.108] p=0.002 | +0.046 [+0.020, +0.072] p=0.001 |
| only:pro - popularity | -0.012 [-0.046, +0.020] p=0.467 | -0.058 [-0.101, -0.013] p=0.015 | -0.028 [-0.052, -0.006] p=0.011 |
| popularity - random | +0.050 [+0.021, +0.082] p=0.001 | +0.102 [+0.056, +0.148] p=0.000 | +0.062 [+0.039, +0.087] p=0.000 |
| full - only:pro | +0.005 [-0.019, +0.027] p=0.635 | +0.023 [+0.000, +0.047] p=0.048 | +0.013 [-0.002, +0.029] p=0.084 |
| without:pro - random | +0.036 [+0.018, +0.055] p=0.000 | +0.052 [+0.018, +0.087] p=0.003 | +0.038 [+0.020, +0.057] p=0.000 |

Held by the rank the full score gave the pro's answer (val+test):

| rank of the answer | events | share held |
|---|---|---|
| rank 1 | 182 | 88% [81, 93] |
| rank 2 | 166 | 87% [81, 92] |
| rank 3 | 117 | 83% [76, 89] |
| rank 4+ | 369 | 85% [79, 89] |
| no evidence | 208 | 82% [73, 90] |

Held by the full score of the pro's answer (val+test); AUROC of the score for held vs tower-hit = 0.543 (862 held, 156 not):

| score | events | share held |
|---|---|---|
| 0 (none) | 208 | 82% [73, 90] |
| (0, 0.2] | 496 | 84% [79, 88] |
| (0.2, 0.4] | 255 | 85% [78, 91] |
| (0.4, 0.6] | 66 | 91% [85, 98] |
| (0.6, 1] | 17 | 100% [100, 100] |

By the strongest pro rule available in the deck (val+test; hit@1 / MRR):

| stratum | events | full KG2.answers_for | pro only | role_transfer only | wiki only | sim only | train popularity of the answer card | random in deck (expected) |
|---|---|---|---|---|---|---|---|---|
| no pro rule for this threat in the deck | 353 | 0.160 / 0.384 | 0.127 / 0.343 | 0.138 / 0.352 | 0.168 / 0.382 | 0.171 / 0.385 | 0.204 / 0.431 | 0.127 / 0.343 |
| best pro rule support 3-4 | 305 | 0.170 / 0.398 | 0.189 / 0.397 | 0.149 / 0.366 | 0.140 / 0.355 | 0.170 / 0.386 | 0.170 / 0.401 | 0.128 / 0.344 |
| best pro rule support 5-9 | 321 | 0.190 / 0.405 | 0.201 / 0.409 | 0.142 / 0.365 | 0.089 / 0.311 | 0.150 / 0.380 | 0.168 / 0.394 | 0.126 / 0.341 |
| best pro rule support 10+ | 63 | 0.127 / 0.317 | 0.079 / 0.302 | 0.190 / 0.395 | 0.065 / 0.270 | 0.123 / 0.326 | 0.095 / 0.335 | 0.125 / 0.340 |

By how the threat was found (val+test; play = the opponent was seen playing the card, unit = a unit crossed with no play seen):

| threat source | method | hit@1 | MRR | coverage |
|---|---|---|---|---|
| play | full KG2.answers_for | 0.163 [0.132, 0.197] | 0.389 [0.365, 0.415] | 99% [98, 100] |
| play | train popularity of the answer card | 0.164 [0.130, 0.200] | 0.395 [0.366, 0.424] | 100% [100, 100] |
| play | random in deck (expected) | 0.127 [0.125, 0.129] | 0.344 [0.341, 0.347] | 0% [0, 0] |
| unit | full KG2.answers_for | 0.183 [0.150, 0.215] | 0.393 [0.366, 0.419] | 100% [99, 100] |
| unit | train popularity of the answer card | 0.199 [0.154, 0.249] | 0.424 [0.386, 0.464] | 100% [100, 100] |
| unit | random in deck (expected) | 0.126 [0.125, 0.127] | 0.341 [0.340, 0.342] | 0% [0, 0] |

## Sweep - support threshold n x answer window (rules from train events of that window)

'fixed' evaluates on the canonical 8 s val / test events, so the rows are comparable; 'matched' re-mines the val / test events with the same window as the rules. Cells: MRR [95% CI] / hit@1.

### fixed, full score

| window | n >= | rules | val events | val MRR | val hit@1 | test events | test MRR | test hit@1 |
|---|---|---|---|---|---|---|---|---|
| 2 s | 1 | 721 | 609 | 0.398 [0.375, 0.423] | 0.182 | 433 | 0.393 [0.365, 0.425] | 0.177 |
| 2 s | 3 | 80 | 609 | 0.377 [0.357, 0.401] | 0.153 | 433 | 0.373 [0.346, 0.403] | 0.158 |
| 2 s | 5 | 16 | 609 | 0.359 [0.336, 0.384] | 0.135 | 433 | 0.373 [0.347, 0.400] | 0.157 |
| 2 s | 10 | 1 | 609 | 0.359 [0.333, 0.387] | 0.137 | 433 | 0.370 [0.346, 0.395] | 0.152 |
| 4 s | 1 | 1021 | 609 | 0.406 [0.382, 0.433] | 0.187 | 433 | 0.385 [0.356, 0.418] | 0.161 |
| 4 s | 3 | 163 | 609 | 0.390 [0.368, 0.413] | 0.171 | 433 | 0.374 [0.342, 0.412] | 0.150 |
| 4 s | 5 | 51 | 609 | 0.373 [0.356, 0.392] | 0.152 | 433 | 0.370 [0.341, 0.402] | 0.156 |
| 4 s | 10 | 3 | 609 | 0.361 [0.335, 0.388] | 0.141 | 433 | 0.369 [0.346, 0.395] | 0.152 |
| 8 s | 1 | 1274 | 609 | 0.409 [0.394, 0.426] | 0.184 | 433 | 0.377 [0.345, 0.414] | 0.158 |
| 8 s | 3 | 287 | 609 | 0.402 [0.383, 0.423] | 0.182 | 433 | 0.374 [0.343, 0.410] | 0.154 |
| 8 s | 5 | 100 | 609 | 0.397 [0.374, 0.421] | 0.180 | 433 | 0.376 [0.348, 0.407] | 0.160 |
| 8 s | 10 | 12 | 609 | 0.362 [0.339, 0.389] | 0.143 | 433 | 0.368 [0.345, 0.393] | 0.150 |

### fixed, pro source only

| window | n >= | rules | val events | val MRR | val hit@1 | test events | test MRR | test hit@1 |
|---|---|---|---|---|---|---|---|---|
| 2 s | 1 | 721 | 609 | 0.371 [0.353, 0.390] | 0.157 | 433 | 0.370 [0.345, 0.401] | 0.149 |
| 2 s | 3 | 80 | 609 | 0.352 [0.342, 0.363] | 0.139 | 433 | 0.346 [0.331, 0.362] | 0.126 |
| 2 s | 5 | 16 | 609 | 0.346 [0.339, 0.355] | 0.133 | 433 | 0.345 [0.337, 0.355] | 0.129 |
| 2 s | 10 | 1 | 609 | 0.341 [0.340, 0.343] | 0.126 | 433 | 0.345 [0.340, 0.351] | 0.128 |
| 4 s | 1 | 1021 | 609 | 0.373 [0.355, 0.392] | 0.157 | 433 | 0.365 [0.340, 0.395] | 0.134 |
| 4 s | 3 | 163 | 609 | 0.365 [0.355, 0.375] | 0.147 | 433 | 0.349 [0.327, 0.376] | 0.129 |
| 4 s | 5 | 51 | 609 | 0.350 [0.340, 0.360] | 0.137 | 433 | 0.343 [0.330, 0.357] | 0.125 |
| 4 s | 10 | 3 | 609 | 0.341 [0.340, 0.343] | 0.126 | 433 | 0.344 [0.339, 0.351] | 0.127 |
| 8 s | 1 | 1274 | 609 | 0.399 [0.380, 0.417] | 0.186 | 433 | 0.355 [0.327, 0.387] | 0.120 |
| 8 s | 3 | 287 | 609 | 0.396 [0.377, 0.415] | 0.191 | 433 | 0.350 [0.325, 0.381] | 0.129 |
| 8 s | 5 | 100 | 609 | 0.375 [0.356, 0.396] | 0.166 | 433 | 0.346 [0.331, 0.363] | 0.126 |
| 8 s | 10 | 12 | 609 | 0.340 [0.337, 0.344] | 0.125 | 433 | 0.341 [0.334, 0.349] | 0.123 |

### matched, full score

| window | n >= | rules | val events | val MRR | val hit@1 | test events | test MRR | test hit@1 |
|---|---|---|---|---|---|---|---|---|
| 2 s | 1 | 721 | 259 | 0.384 [0.354, 0.418] | 0.162 | 199 | 0.409 [0.368, 0.459] | 0.196 |
| 2 s | 3 | 80 | 259 | 0.380 [0.341, 0.426] | 0.155 | 199 | 0.385 [0.352, 0.422] | 0.173 |
| 2 s | 5 | 16 | 259 | 0.363 [0.328, 0.407] | 0.135 | 199 | 0.374 [0.344, 0.403] | 0.159 |
| 2 s | 10 | 1 | 259 | 0.382 [0.342, 0.432] | 0.167 | 199 | 0.370 [0.342, 0.398] | 0.154 |
| 4 s | 1 | 1021 | 431 | 0.403 [0.377, 0.431] | 0.181 | 305 | 0.391 [0.360, 0.429] | 0.164 |
| 4 s | 3 | 163 | 431 | 0.393 [0.369, 0.420] | 0.170 | 305 | 0.384 [0.351, 0.426] | 0.155 |
| 4 s | 5 | 51 | 431 | 0.372 [0.352, 0.393] | 0.147 | 305 | 0.371 [0.340, 0.404] | 0.153 |
| 4 s | 10 | 3 | 431 | 0.374 [0.345, 0.407] | 0.157 | 305 | 0.371 [0.345, 0.397] | 0.153 |
| 8 s | 1 | 1274 | 609 | 0.409 [0.394, 0.426] | 0.184 | 433 | 0.377 [0.345, 0.414] | 0.158 |
| 8 s | 3 | 287 | 609 | 0.402 [0.383, 0.423] | 0.182 | 433 | 0.374 [0.343, 0.410] | 0.154 |
| 8 s | 5 | 100 | 609 | 0.397 [0.374, 0.421] | 0.180 | 433 | 0.376 [0.348, 0.407] | 0.160 |
| 8 s | 10 | 12 | 609 | 0.362 [0.339, 0.389] | 0.143 | 433 | 0.368 [0.345, 0.393] | 0.150 |

### matched, pro source only

| window | n >= | rules | val events | val MRR | val hit@1 | test events | test MRR | test hit@1 |
|---|---|---|---|---|---|---|---|---|
| 2 s | 1 | 721 | 259 | 0.359 [0.334, 0.388] | 0.137 | 199 | 0.383 [0.346, 0.434] | 0.165 |
| 2 s | 3 | 80 | 259 | 0.347 [0.331, 0.365] | 0.133 | 199 | 0.356 [0.335, 0.383] | 0.139 |
| 2 s | 5 | 16 | 259 | 0.340 [0.332, 0.353] | 0.125 | 199 | 0.347 [0.338, 0.357] | 0.130 |
| 2 s | 10 | 1 | 259 | 0.341 [0.340, 0.343] | 0.126 | 199 | 0.345 [0.340, 0.353] | 0.128 |
| 4 s | 1 | 1021 | 431 | 0.369 [0.348, 0.391] | 0.151 | 305 | 0.369 [0.343, 0.399] | 0.141 |
| 4 s | 3 | 163 | 431 | 0.365 [0.350, 0.383] | 0.144 | 305 | 0.352 [0.328, 0.383] | 0.132 |
| 4 s | 5 | 51 | 431 | 0.347 [0.336, 0.358] | 0.133 | 305 | 0.343 [0.328, 0.359] | 0.123 |
| 4 s | 10 | 3 | 431 | 0.341 [0.340, 0.343] | 0.126 | 305 | 0.344 [0.338, 0.351] | 0.126 |
| 8 s | 1 | 1274 | 609 | 0.399 [0.380, 0.417] | 0.186 | 433 | 0.355 [0.327, 0.387] | 0.120 |
| 8 s | 3 | 287 | 609 | 0.396 [0.377, 0.415] | 0.191 | 433 | 0.350 [0.325, 0.381] | 0.129 |
| 8 s | 5 | 100 | 609 | 0.375 [0.356, 0.396] | 0.166 | 433 | 0.346 [0.331, 0.363] | 0.126 |
| 8 s | 10 | 12 | 609 | 0.340 [0.337, 0.344] | 0.125 | 433 | 0.341 [0.334, 0.349] | 0.123 |

Best on val (full score, fixed events): window 8 s, n >= 1 (val MRR 0.409); its test MRR 0.377 [0.345, 0.414], hit@1 0.158 [0.112, 0.209].
Events per window (all splits): {'2.0': {'test': 201, 'train': 1072, 'val': 259}, '4.0': {'test': 307, 'train': 1768, 'val': 431}, '8.0': {'test': 436, 'train': 2575, 'val': 609}}; the re-mined 8 s events equal the shipped answer_events.jsonl: True.

## Notes

- Coverage = some deck card has evidence under the method; 'hit@1 if covered' is the accuracy on those events. Ties are expectations, so the uncovered events count as random.
- Standard vs PCA-style confidence (AMIE) is not reported: a threat's candidate answers share one denominator, so the ranking inside a threat is the same under both, and answer_events.jsonl holds answered threats only (the unanswered ones are counted in the miner's stats, not stored).
- 'KnowledgeGraph.counter_scores' adds the v1 ANSWERED_BY edges (mined from train states, window 4 s) to the wiki COUNTERS; the 'wiki COUNTERS only' row drops them.
- The 'pro only' score is the support-shrunk held rate of the mined edge; the other three sources never see an answer event.

Runtime 12.1 s.
