# E2 - context-conflict test of d1-ft-joint-tac (MAC-AR R2)

Variant `joint_tac`; run at HEAD `54243d78bc`; 10000 bootstrap resamples over games (seed 0); lines {'val': {'lines': 761, 'games': 26, 'conditions_run': ['C0', 'C1', 'C2', 'C3', 'C4']}, 'test': {'lines': 766, 'games': 27, 'conditions_run': ['C0', 'C1', 'C2', 'C3', 'C4']}}.

Conditions: C0 = as exported; C1 = notes and tactics / KG lines removed; C2 = notes shuffled; C3 = poisoned (gold note swapped, wrong tactics card); C4 = tactics / KG block transplanted.

## Accuracy by condition - val

| condition | n | accuracy [95% CI] | vs C0 | McNemar p (right only here / right only in C0) | picks changed | mean P(gold) | ECE | mean confidence |
|---|---|---|---|---|---|---|---|---|
| C0 | 761 | 0.536 [0.489, 0.576] | - | - | - | 0.422 | 0.029 | 0.529 |
| C1 | 761 | 0.536 [0.487, 0.579] | +0.000 [-0.025, +0.026] | 1.0000 (61/61) | 23% | 0.415 | 0.041 | 0.530 |
| C2 | 761 | 0.547 [0.500, 0.587] | +0.011 [-0.007, +0.028] | 0.2682 (24/16) | 8% | 0.422 | 0.035 | 0.527 |
| C3 | 761 | 0.539 [0.491, 0.579] | +0.003 [-0.009, +0.013] | 0.8388 (13/11) | 5% | 0.421 | 0.031 | 0.524 |
| C4 | 761 | 0.532 [0.489, 0.571] | -0.004 [-0.026, +0.023] | 0.8439 (50/53) | 17% | 0.418 | 0.038 | 0.527 |

Follow-rate - val: among lines where the corruption points at options other than the C0 notes and the C0 pick is not already one of them, the share whose pick moves to a favoured option.

| condition | active lines | movable | moved | follow-rate [95% CI] | pick on favoured | C0 pick on favoured |
|---|---|---|---|---|---|---|
| C2 | 389 | 247 | 23 | 0.093 [0.068, 0.121] | 0.380 | 0.365 |
| C3 | 357 | 324 | 5 | 0.015 [0.003, 0.031] | 0.090 | 0.092 |
| C4 | 49 | 25 | 3 | 0.120 [0.000, 0.259] | 0.490 | 0.490 |

Accuracy by gold option's note (C0): right = informative, wrong = filler while a distractor is informative, none = all filler - val:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| none | 322 | 0.568 [0.507, 0.621] | 0.556 [0.483, 0.619] | 0.568 [0.507, 0.621] | 0.559 [0.496, 0.610] | 0.550 [0.479, 0.616] |
| right | 243 | 0.601 [0.548, 0.652] | 0.621 [0.566, 0.678] | 0.609 [0.549, 0.662] | 0.626 [0.574, 0.671] | 0.601 [0.541, 0.654] |
| wrong | 196 | 0.403 [0.324, 0.474] | 0.398 [0.320, 0.473] | 0.434 [0.340, 0.521] | 0.398 [0.321, 0.465] | 0.418 [0.335, 0.502] |

Accuracy by state's `tactics: best` card vs the gold card (C0) - val:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| names another card | 286 | 0.521 [0.464, 0.568] | 0.538 [0.472, 0.599] | 0.535 [0.476, 0.592] | 0.517 [0.458, 0.568] | 0.503 [0.451, 0.559] |
| names the gold card | 128 | 0.586 [0.500, 0.680] | 0.625 [0.536, 0.718] | 0.602 [0.511, 0.701] | 0.602 [0.515, 0.694] | 0.562 [0.444, 0.680] |
| no best line | 347 | 0.530 [0.466, 0.584] | 0.501 [0.441, 0.556] | 0.536 [0.468, 0.591] | 0.533 [0.469, 0.587] | 0.545 [0.479, 0.603] |

C0 context - val: note status {'none': 322, 'right': 243, 'wrong': 196}; tactics best line {'names another card': 286, 'names the gold card': 128, 'no best line': 347}; lines C3 changed 371.

## Accuracy by condition - test

| condition | n | accuracy [95% CI] | vs C0 | McNemar p (right only here / right only in C0) | picks changed | mean P(gold) | ECE | mean confidence |
|---|---|---|---|---|---|---|---|---|
| C0 | 766 | 0.507 [0.474, 0.537] | - | - | - | 0.375 | 0.055 | 0.480 |
| C1 | 766 | 0.508 [0.476, 0.541] | +0.001 [-0.025, +0.031] | 1.0000 (60/59) | 22% | 0.370 | 0.054 | 0.477 |
| C2 | 766 | 0.518 [0.484, 0.550] | +0.012 [-0.003, +0.027] | 0.1996 (24/15) | 7% | 0.373 | 0.062 | 0.476 |
| C3 | 766 | 0.503 [0.468, 0.536] | -0.004 [-0.021, +0.012] | 0.7608 (20/23) | 7% | 0.373 | 0.045 | 0.477 |
| C4 | 766 | 0.491 [0.457, 0.524] | -0.016 [-0.046, +0.012] | 0.2986 (50/62) | 19% | 0.377 | 0.043 | 0.484 |

Follow-rate - test: among lines where the corruption points at options other than the C0 notes and the C0 pick is not already one of them, the share whose pick moves to a favoured option.

| condition | active lines | movable | moved | follow-rate [95% CI] | pick on favoured | C0 pick on favoured |
|---|---|---|---|---|---|---|
| C2 | 349 | 219 | 20 | 0.091 [0.056, 0.132] | 0.398 | 0.372 |
| C3 | 319 | 277 | 12 | 0.043 [0.022, 0.068] | 0.138 | 0.132 |
| C4 | 19 | 8 | 0 | 0.000 [0.000, 0.000] | 0.526 | 0.579 |

Accuracy by gold option's note (C0): right = informative, wrong = filler while a distractor is informative, none = all filler - test:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| none | 367 | 0.526 [0.474, 0.580] | 0.499 [0.452, 0.548] | 0.526 [0.474, 0.580] | 0.515 [0.466, 0.566] | 0.480 [0.431, 0.529] |
| right | 231 | 0.567 [0.515, 0.614] | 0.602 [0.537, 0.662] | 0.597 [0.538, 0.654] | 0.563 [0.502, 0.623] | 0.580 [0.522, 0.638] |
| wrong | 168 | 0.381 [0.327, 0.436] | 0.399 [0.342, 0.457] | 0.393 [0.333, 0.450] | 0.393 [0.335, 0.452] | 0.393 [0.312, 0.476] |

Accuracy by state's `tactics: best` card vs the gold card (C0) - test:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| names another card | 239 | 0.506 [0.434, 0.564] | 0.519 [0.464, 0.572] | 0.510 [0.442, 0.565] | 0.490 [0.423, 0.548] | 0.481 [0.420, 0.541] |
| names the gold card | 104 | 0.529 [0.429, 0.617] | 0.615 [0.527, 0.690] | 0.615 [0.511, 0.719] | 0.548 [0.441, 0.650] | 0.596 [0.479, 0.699] |
| no best line | 423 | 0.501 [0.470, 0.533] | 0.475 [0.441, 0.514] | 0.499 [0.470, 0.529] | 0.499 [0.470, 0.529] | 0.470 [0.430, 0.513] |

C0 context - test: note status {'none': 367, 'right': 231, 'wrong': 168}; tactics best line {'names another card': 239, 'names the gold card': 104, 'no best line': 423}; lines C3 changed 333.

## Accuracy by condition - val+test

| condition | n | accuracy [95% CI] | vs C0 | McNemar p (right only here / right only in C0) | picks changed | mean P(gold) | ECE | mean confidence |
|---|---|---|---|---|---|---|---|---|
| C0 | 1527 | 0.521 [0.494, 0.548] | - | - | - | 0.399 | 0.041 | 0.504 |
| C1 | 1527 | 0.522 [0.494, 0.550] | +0.001 [-0.018, +0.020] | 1.0000 (121/120) | 23% | 0.393 | 0.035 | 0.504 |
| C2 | 1527 | 0.532 [0.505, 0.560] | +0.011 [+0.000, +0.023] | 0.0712 (48/31) | 7% | 0.397 | 0.042 | 0.502 |
| C3 | 1527 | 0.521 [0.492, 0.549] | -0.001 [-0.011, +0.009] | 1.0000 (33/34) | 6% | 0.397 | 0.031 | 0.500 |
| C4 | 1527 | 0.511 [0.484, 0.538] | -0.010 [-0.028, +0.009] | 0.3397 (100/115) | 18% | 0.397 | 0.033 | 0.506 |

Follow-rate - val+test: among lines where the corruption points at options other than the C0 notes and the C0 pick is not already one of them, the share whose pick moves to a favoured option.

| condition | active lines | movable | moved | follow-rate [95% CI] | pick on favoured | C0 pick on favoured |
|---|---|---|---|---|---|---|
| C2 | 738 | 466 | 43 | 0.092 [0.070, 0.116] | 0.389 | 0.369 |
| C3 | 676 | 601 | 17 | 0.028 [0.015, 0.043] | 0.112 | 0.111 |
| C4 | 68 | 33 | 3 | 0.091 [0.000, 0.194] | 0.500 | 0.515 |

Accuracy by gold option's note (C0): right = informative, wrong = filler while a distractor is informative, none = all filler - val+test:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| none | 689 | 0.546 [0.506, 0.586] | 0.525 [0.482, 0.569] | 0.546 [0.506, 0.586] | 0.536 [0.497, 0.574] | 0.512 [0.470, 0.557] |
| right | 474 | 0.584 [0.550, 0.619] | 0.612 [0.570, 0.653] | 0.603 [0.562, 0.642] | 0.595 [0.555, 0.632] | 0.591 [0.550, 0.630] |
| wrong | 364 | 0.393 [0.344, 0.440] | 0.398 [0.349, 0.447] | 0.415 [0.357, 0.473] | 0.396 [0.347, 0.441] | 0.407 [0.347, 0.464] |

Accuracy by state's `tactics: best` card vs the gold card (C0) - val+test:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| names another card | 525 | 0.514 [0.473, 0.552] | 0.530 [0.487, 0.571] | 0.524 [0.482, 0.564] | 0.505 [0.462, 0.544] | 0.493 [0.454, 0.534] |
| names the gold card | 232 | 0.560 [0.496, 0.624] | 0.621 [0.559, 0.680] | 0.608 [0.538, 0.679] | 0.578 [0.512, 0.644] | 0.578 [0.494, 0.658] |
| no best line | 770 | 0.514 [0.482, 0.547] | 0.487 [0.454, 0.521] | 0.516 [0.482, 0.549] | 0.514 [0.482, 0.547] | 0.504 [0.465, 0.544] |

## Contrastive decoding (CAD) and AdaCAD against C1

score = (1 + a) log p(context) - a log p(no context); a from the grid, tuned on val; AdaCAD a = Jensen-Shannon divergence (bits) of the two distributions. Delta columns are paired bootstrap differences (CI over games).

### context C0 (a tuned on val: a=0; mean AdaCAD a 0.017 val / 0.014 test)

| decoding | val acc | test acc | test vs plain | test vs C0 |
|---|---|---|---|---|
| a=0 | 0.536 [0.489, 0.576] | 0.507 [0.474, 0.537] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| a=0.25 | 0.528 [0.487, 0.562] | 0.495 [0.457, 0.530] | -0.012 [-0.025, +0.002] | -0.012 [-0.025, +0.002] |
| a=0.5 | 0.519 [0.475, 0.557] | 0.477 [0.440, 0.513] | -0.030 [-0.051, -0.011] | -0.030 [-0.051, -0.011] |
| a=1 | 0.485 [0.447, 0.519] | 0.475 [0.440, 0.512] | -0.031 [-0.060, -0.002] | -0.031 [-0.060, -0.002] |
| adacad | 0.534 [0.486, 0.573] | 0.507 [0.475, 0.537] | +0.000 [-0.004, +0.004] | +0.000 [-0.004, +0.004] |

### context C2 (a tuned on val: a=0; mean AdaCAD a 0.017 val / 0.014 test)

| decoding | val acc | test acc | test vs plain | test vs C0 |
|---|---|---|---|---|
| a=0 | 0.547 [0.500, 0.587] | 0.518 [0.484, 0.550] | +0.000 [+0.000, +0.000] | +0.012 [-0.003, +0.027] |
| a=0.25 | 0.531 [0.486, 0.569] | 0.495 [0.461, 0.528] | -0.023 [-0.036, -0.011] | -0.012 [-0.031, +0.007] |
| a=0.5 | 0.512 [0.470, 0.551] | 0.474 [0.442, 0.504] | -0.044 [-0.063, -0.025] | -0.033 [-0.054, -0.012] |
| a=1 | 0.491 [0.451, 0.527] | 0.462 [0.427, 0.497] | -0.056 [-0.082, -0.029] | -0.044 [-0.072, -0.017] |
| adacad | 0.540 [0.493, 0.582] | 0.518 [0.485, 0.550] | +0.000 [-0.006, +0.006] | +0.012 [-0.006, +0.030] |

### context C3 (a tuned on val: a=0; mean AdaCAD a 0.018 val / 0.015 test)

| decoding | val acc | test acc | test vs plain | test vs C0 |
|---|---|---|---|---|
| a=0 | 0.539 [0.491, 0.579] | 0.503 [0.468, 0.536] | +0.000 [+0.000, +0.000] | -0.004 [-0.021, +0.012] |
| a=0.25 | 0.535 [0.495, 0.568] | 0.491 [0.452, 0.527] | -0.012 [-0.029, +0.004] | -0.016 [-0.037, +0.006] |
| a=0.5 | 0.523 [0.479, 0.561] | 0.469 [0.434, 0.503] | -0.034 [-0.057, -0.015] | -0.038 [-0.060, -0.018] |
| a=1 | 0.493 [0.445, 0.534] | 0.470 [0.434, 0.507] | -0.033 [-0.061, -0.006] | -0.037 [-0.065, -0.010] |
| adacad | 0.536 [0.488, 0.577] | 0.504 [0.468, 0.538] | +0.001 [-0.003, +0.006] | -0.003 [-0.020, +0.014] |

### context C4 (a tuned on val: a=0; mean AdaCAD a 0.016 val / 0.013 test)

| decoding | val acc | test acc | test vs plain | test vs C0 |
|---|---|---|---|---|
| a=0 | 0.532 [0.489, 0.571] | 0.491 [0.457, 0.524] | +0.000 [+0.000, +0.000] | -0.016 [-0.046, +0.012] |
| a=0.25 | 0.511 [0.463, 0.553] | 0.488 [0.453, 0.522] | -0.003 [-0.019, +0.013] | -0.018 [-0.051, +0.010] |
| a=0.5 | 0.499 [0.449, 0.543] | 0.483 [0.445, 0.520] | -0.008 [-0.027, +0.012] | -0.023 [-0.057, +0.006] |
| a=1 | 0.482 [0.433, 0.525] | 0.474 [0.437, 0.510] | -0.017 [-0.036, +0.001] | -0.033 [-0.067, +0.000] |
| adacad | 0.530 [0.483, 0.570] | 0.492 [0.458, 0.525] | +0.001 [+0.000, +0.004] | -0.014 [-0.044, +0.012] |

## Reproduction check

C0 against the stored eval file: {"file": "/Users/tylerjackson/Projects/clashbot/clashBot/.claude/worktrees/agent-a362518d25f251287/dataset/v1/eval/d1_eval_transformers_d1-ft-joint-tac_on_kg.json", "val": {"lines_matched": 761, "pick_agreement": 1.0, "stored_acc": 0.5361366622864652, "rerun_acc": 0.5361366622864652}, "test": {"lines_matched": 766, "pick_agreement": 1.0, "stored_acc": 0.5065274151436031, "rerun_acc": 0.5065274151436031}}

## How to read the tables

- A model that follows its context loses accuracy in C3 and C4 and gains it in the 'right' note stratum only while the notes are present (C0), so a gap between strata that is also there in C1 (no notes, no tactics / KG lines) comes from the lines, not from reading the notes.
- Follow-rate is compared with the C0 column ('C0 pick on favoured'): the same share of picks lands on the favoured options when the context is not corrupted, so only the difference between the two is movement caused by the context.
- C3 is applied where it can bite: the gold note moves only when it is informative and a distractor's is not; lines with nothing to corrupt are left as exported (`changed lines` above).
- CAD uses the C1 probabilities as the context-free distribution; no extra d1 call.

## E2 - context-conflict test of d1-ft-joint-tac (MAC-AR R2)

Variant `joint_tac_v2`; run at HEAD `54243d78bc`; 10000 bootstrap resamples over games (seed 0); lines {'val': {'lines': 761, 'games': 26, 'conditions_run': ['C0', 'C1', 'C2', 'C3', 'C4']}, 'test': {'lines': 766, 'games': 27, 'conditions_run': ['C0', 'C1', 'C2', 'C3', 'C4']}}.

Conditions: C0 = as exported; C1 = notes and tactics / KG lines removed; C2 = notes shuffled; C3 = poisoned (gold note swapped, wrong tactics card); C4 = tactics / KG block transplanted.

### Accuracy by condition - val

| condition | n | accuracy [95% CI] | vs C0 | McNemar p (right only here / right only in C0) | picks changed | mean P(gold) | ECE | mean confidence |
|---|---|---|---|---|---|---|---|---|
| C0 | 761 | 0.547 [0.501, 0.585] | - | - | - | 0.422 | 0.040 | 0.529 |
| C1 | 761 | 0.536 [0.487, 0.579] | -0.011 [-0.033, +0.013] | 0.5298 (58/66) | 24% | 0.415 | 0.041 | 0.530 |
| C2 | 761 | 0.547 [0.499, 0.588] | +0.000 [-0.019, +0.019] | 1.0000 (21/21) | 8% | 0.421 | 0.042 | 0.526 |
| C3 | 761 | 0.541 [0.495, 0.580] | -0.005 [-0.015, +0.004] | 0.5413 (10/14) | 4% | 0.421 | 0.025 | 0.525 |
| C4 | 761 | 0.526 [0.478, 0.566] | -0.021 [-0.046, +0.007] | 0.1448 (45/61) | 18% | 0.416 | 0.045 | 0.527 |

Follow-rate - val: among lines where the corruption points at options other than the C0 notes and the C0 pick is not already one of them, the share whose pick moves to a favoured option.

| condition | active lines | movable | moved | follow-rate [95% CI] | pick on favoured | C0 pick on favoured |
|---|---|---|---|---|---|---|
| C2 | 396 | 257 | 25 | 0.097 [0.070, 0.130] | 0.374 | 0.351 |
| C3 | 341 | 312 | 6 | 0.019 [0.003, 0.037] | 0.091 | 0.085 |
| C4 | 50 | 23 | 3 | 0.130 [0.000, 0.294] | 0.520 | 0.540 |

Accuracy by gold option's note (C0): right = informative, wrong = filler while a distractor is informative, none = all filler - val:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| none | 325 | 0.572 [0.510, 0.624] | 0.560 [0.487, 0.622] | 0.572 [0.510, 0.624] | 0.563 [0.500, 0.614] | 0.548 [0.477, 0.612] |
| right | 234 | 0.611 [0.553, 0.668] | 0.611 [0.548, 0.674] | 0.585 [0.521, 0.643] | 0.611 [0.560, 0.661] | 0.585 [0.523, 0.642] |
| wrong | 202 | 0.431 [0.348, 0.505] | 0.411 [0.333, 0.485] | 0.460 [0.364, 0.549] | 0.426 [0.344, 0.497] | 0.421 [0.341, 0.494] |

Accuracy by state's `tactics: best` card vs the gold card (C0) - val:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| names another card | 274 | 0.551 [0.487, 0.600] | 0.551 [0.475, 0.615] | 0.547 [0.482, 0.606] | 0.529 [0.466, 0.579] | 0.507 [0.446, 0.561] |
| names the gold card | 126 | 0.571 [0.471, 0.672] | 0.595 [0.504, 0.689] | 0.563 [0.469, 0.664] | 0.571 [0.479, 0.669] | 0.532 [0.420, 0.640] |
| no best line | 361 | 0.535 [0.475, 0.585] | 0.504 [0.446, 0.558] | 0.540 [0.477, 0.591] | 0.540 [0.482, 0.589] | 0.537 [0.474, 0.596] |

C0 context - val: note status {'none': 325, 'right': 234, 'wrong': 202}; tactics best line {'names another card': 274, 'names the gold card': 126, 'no best line': 361}; lines C3 changed 354.

### Accuracy by condition - test

| condition | n | accuracy [95% CI] | vs C0 | McNemar p (right only here / right only in C0) | picks changed | mean P(gold) | ECE | mean confidence |
|---|---|---|---|---|---|---|---|---|
| C0 | 766 | 0.507 [0.473, 0.539] | - | - | - | 0.375 | 0.048 | 0.480 |
| C1 | 766 | 0.508 [0.476, 0.541] | +0.001 [-0.025, +0.031] | 1.0000 (57/56) | 21% | 0.370 | 0.054 | 0.477 |
| C2 | 766 | 0.517 [0.482, 0.550] | +0.010 [-0.005, +0.026] | 0.2682 (24/16) | 7% | 0.373 | 0.064 | 0.476 |
| C3 | 766 | 0.503 [0.469, 0.535] | -0.004 [-0.021, +0.013] | 0.7552 (19/22) | 6% | 0.373 | 0.052 | 0.477 |
| C4 | 766 | 0.495 [0.459, 0.530] | -0.012 [-0.040, +0.015] | 0.4519 (52/61) | 19% | 0.376 | 0.048 | 0.484 |

Follow-rate - test: among lines where the corruption points at options other than the C0 notes and the C0 pick is not already one of them, the share whose pick moves to a favoured option.

| condition | active lines | movable | moved | follow-rate [95% CI] | pick on favoured | C0 pick on favoured |
|---|---|---|---|---|---|---|
| C2 | 347 | 224 | 21 | 0.094 [0.060, 0.132] | 0.389 | 0.354 |
| C3 | 311 | 273 | 12 | 0.044 [0.021, 0.068] | 0.132 | 0.122 |
| C4 | 19 | 12 | 1 | 0.083 [0.000, 0.211] | 0.421 | 0.368 |

Accuracy by gold option's note (C0): right = informative, wrong = filler while a distractor is informative, none = all filler - test:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| none | 372 | 0.532 [0.478, 0.586] | 0.500 [0.452, 0.551] | 0.532 [0.478, 0.586] | 0.522 [0.472, 0.573] | 0.478 [0.427, 0.531] |
| right | 227 | 0.564 [0.514, 0.613] | 0.599 [0.533, 0.663] | 0.590 [0.535, 0.645] | 0.559 [0.505, 0.613] | 0.590 [0.526, 0.653] |
| wrong | 167 | 0.371 [0.314, 0.428] | 0.401 [0.342, 0.460] | 0.383 [0.320, 0.442] | 0.383 [0.321, 0.446] | 0.401 [0.319, 0.483] |

Accuracy by state's `tactics: best` card vs the gold card (C0) - test:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| names another card | 227 | 0.507 [0.436, 0.565] | 0.520 [0.465, 0.572] | 0.515 [0.450, 0.568] | 0.485 [0.424, 0.540] | 0.489 [0.423, 0.553] |
| names the gold card | 102 | 0.510 [0.402, 0.595] | 0.608 [0.507, 0.691] | 0.588 [0.492, 0.682] | 0.549 [0.427, 0.656] | 0.618 [0.483, 0.731] |
| no best line | 437 | 0.506 [0.476, 0.537] | 0.478 [0.444, 0.519] | 0.501 [0.470, 0.534] | 0.501 [0.470, 0.533] | 0.469 [0.431, 0.509] |

C0 context - test: note status {'none': 372, 'right': 227, 'wrong': 167}; tactics best line {'names another card': 227, 'names the gold card': 102, 'no best line': 437}; lines C3 changed 324.

### Accuracy by condition - val+test

| condition | n | accuracy [95% CI] | vs C0 | McNemar p (right only here / right only in C0) | picks changed | mean P(gold) | ECE | mean confidence |
|---|---|---|---|---|---|---|---|---|
| C0 | 1527 | 0.527 [0.499, 0.554] | - | - | - | 0.398 | 0.039 | 0.504 |
| C1 | 1527 | 0.522 [0.494, 0.550] | -0.005 [-0.022, +0.014] | 0.6968 (115/122) | 22% | 0.393 | 0.035 | 0.504 |
| C2 | 1527 | 0.532 [0.504, 0.559] | +0.005 [-0.007, +0.018] | 0.4397 (45/37) | 7% | 0.397 | 0.043 | 0.501 |
| C3 | 1527 | 0.522 [0.494, 0.550] | -0.005 [-0.015, +0.005] | 0.4570 (29/36) | 5% | 0.397 | 0.033 | 0.501 |
| C4 | 1527 | 0.510 [0.481, 0.537] | -0.016 [-0.035, +0.003] | 0.1046 (97/122) | 18% | 0.396 | 0.030 | 0.505 |

Follow-rate - val+test: among lines where the corruption points at options other than the C0 notes and the C0 pick is not already one of them, the share whose pick moves to a favoured option.

| condition | active lines | movable | moved | follow-rate [95% CI] | pick on favoured | C0 pick on favoured |
|---|---|---|---|---|---|---|
| C2 | 743 | 481 | 46 | 0.096 [0.074, 0.119] | 0.381 | 0.353 |
| C3 | 652 | 585 | 18 | 0.031 [0.016, 0.046] | 0.110 | 0.103 |
| C4 | 69 | 35 | 4 | 0.114 [0.000, 0.233] | 0.493 | 0.493 |

Accuracy by gold option's note (C0): right = informative, wrong = filler while a distractor is informative, none = all filler - val+test:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| none | 697 | 0.551 [0.511, 0.591] | 0.528 [0.484, 0.573] | 0.551 [0.511, 0.591] | 0.541 [0.502, 0.579] | 0.511 [0.467, 0.556] |
| right | 461 | 0.588 [0.551, 0.625] | 0.605 [0.561, 0.650] | 0.588 [0.546, 0.628] | 0.586 [0.548, 0.622] | 0.588 [0.544, 0.630] |
| wrong | 369 | 0.404 [0.353, 0.454] | 0.407 [0.356, 0.456] | 0.425 [0.365, 0.486] | 0.407 [0.354, 0.456] | 0.412 [0.355, 0.466] |

Accuracy by state's `tactics: best` card vs the gold card (C0) - val+test:

| stratum | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| names another card | 501 | 0.531 [0.487, 0.570] | 0.537 [0.491, 0.581] | 0.533 [0.489, 0.572] | 0.509 [0.467, 0.549] | 0.499 [0.456, 0.541] |
| names the gold card | 228 | 0.544 [0.474, 0.609] | 0.601 [0.534, 0.665] | 0.575 [0.505, 0.643] | 0.561 [0.489, 0.633] | 0.570 [0.484, 0.652] |
| no best line | 798 | 0.519 [0.488, 0.550] | 0.490 [0.457, 0.524] | 0.519 [0.486, 0.551] | 0.519 [0.488, 0.550] | 0.500 [0.463, 0.538] |

### Contrastive decoding (CAD) and AdaCAD against C1

score = (1 + a) log p(context) - a log p(no context); a from the grid, tuned on val; AdaCAD a = Jensen-Shannon divergence (bits) of the two distributions. Delta columns are paired bootstrap differences (CI over games).

#### context C0 (a tuned on val: a=0; mean AdaCAD a 0.017 val / 0.014 test)

| decoding | val acc | test acc | test vs plain | test vs C0 |
|---|---|---|---|---|
| a=0 | 0.547 [0.501, 0.585] | 0.507 [0.473, 0.539] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| a=0.25 | 0.530 [0.489, 0.564] | 0.490 [0.452, 0.524] | -0.017 [-0.030, -0.004] | -0.017 [-0.030, -0.004] |
| a=0.5 | 0.514 [0.471, 0.550] | 0.478 [0.444, 0.511] | -0.029 [-0.047, -0.012] | -0.029 [-0.047, -0.012] |
| a=1 | 0.490 [0.452, 0.524] | 0.479 [0.447, 0.513] | -0.027 [-0.054, -0.001] | -0.027 [-0.054, -0.001] |
| adacad | 0.547 [0.501, 0.585] | 0.505 [0.472, 0.537] | -0.001 [-0.004, +0.000] | -0.001 [-0.004, +0.000] |

#### context C2 (a tuned on val: a=0; mean AdaCAD a 0.018 val / 0.014 test)

| decoding | val acc | test acc | test vs plain | test vs C0 |
|---|---|---|---|---|
| a=0 | 0.547 [0.499, 0.588] | 0.517 [0.482, 0.550] | +0.000 [+0.000, +0.000] | +0.010 [-0.005, +0.026] |
| a=0.25 | 0.534 [0.490, 0.571] | 0.493 [0.459, 0.527] | -0.023 [-0.037, -0.009] | -0.013 [-0.032, +0.006] |
| a=0.5 | 0.506 [0.462, 0.544] | 0.478 [0.448, 0.507] | -0.039 [-0.058, -0.019] | -0.029 [-0.050, -0.008] |
| a=1 | 0.489 [0.447, 0.528] | 0.465 [0.428, 0.500] | -0.052 [-0.078, -0.025] | -0.042 [-0.069, -0.016] |
| adacad | 0.543 [0.496, 0.584] | 0.514 [0.481, 0.546] | -0.003 [-0.008, +0.004] | +0.008 [-0.009, +0.025] |

#### context C3 (a tuned on val: a=0; mean AdaCAD a 0.018 val / 0.015 test)

| decoding | val acc | test acc | test vs plain | test vs C0 |
|---|---|---|---|---|
| a=0 | 0.541 [0.495, 0.580] | 0.503 [0.469, 0.535] | +0.000 [+0.000, +0.000] | -0.004 [-0.021, +0.013] |
| a=0.25 | 0.534 [0.492, 0.568] | 0.486 [0.449, 0.521] | -0.017 [-0.034, -0.001] | -0.021 [-0.041, +0.001] |
| a=0.5 | 0.518 [0.471, 0.558] | 0.473 [0.438, 0.508] | -0.030 [-0.053, -0.009] | -0.034 [-0.056, -0.011] |
| a=1 | 0.493 [0.445, 0.534] | 0.471 [0.437, 0.507] | -0.031 [-0.058, -0.008] | -0.035 [-0.061, -0.011] |
| adacad | 0.537 [0.492, 0.576] | 0.501 [0.467, 0.534] | -0.001 [-0.004, +0.000] | -0.005 [-0.023, +0.012] |

#### context C4 (a tuned on val: a=0; mean AdaCAD a 0.016 val / 0.014 test)

| decoding | val acc | test acc | test vs plain | test vs C0 |
|---|---|---|---|---|
| a=0 | 0.526 [0.478, 0.566] | 0.495 [0.459, 0.530] | +0.000 [+0.000, +0.000] | -0.012 [-0.040, +0.015] |
| a=0.25 | 0.516 [0.466, 0.559] | 0.484 [0.447, 0.520] | -0.010 [-0.025, +0.003] | -0.022 [-0.055, +0.008] |
| a=0.5 | 0.495 [0.443, 0.541] | 0.482 [0.441, 0.520] | -0.013 [-0.034, +0.007] | -0.025 [-0.062, +0.007] |
| a=1 | 0.476 [0.425, 0.519] | 0.466 [0.430, 0.500] | -0.029 [-0.051, -0.008] | -0.040 [-0.076, -0.007] |
| adacad | 0.524 [0.476, 0.566] | 0.495 [0.459, 0.529] | +0.000 [-0.004, +0.004] | -0.012 [-0.039, +0.014] |


### How to read the tables

- A model that follows its context loses accuracy in C3 and C4 and gains it in the 'right' note stratum only while the notes are present (C0), so a gap between strata that is also there in C1 (no notes, no tactics / KG lines) comes from the lines, not from reading the notes.
- Follow-rate is compared with the C0 column ('C0 pick on favoured'): the same share of picks lands on the favoured options when the context is not corrupted, so only the difference between the two is movement caused by the context.
- C3 is applied where it can bite: the gold note moves only when it is informative and a distractor's is not; lines with nothing to corrupt are left as exported (`changed lines` above).
- CAD uses the C1 probabilities as the context-free distribution; no extra d1 call.
