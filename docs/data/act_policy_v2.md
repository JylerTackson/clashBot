# Act/hold policy: reactive vs trigger (offline, D-041)

Labels: 1 = the pro played a card, 0 = hold (unknown skipped). The labels are biased toward plays (D-020).

## test / all  (n=1369, pro play rate 0.941)

| policy | accuracy | balanced acc | act rate | tpr | tnr | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|
| reactive | 0.263 | 0.545 | 0.220 | 0.225 | 0.864 | 290 | 11 | 998 | 70 |
| trigger | 0.898 | 0.518 | 0.947 | 0.949 | 0.086 | 1222 | 74 | 66 | 7 |
| always | 0.941 | 0.500 | 1.000 | 1.000 | 0.000 | 1288 | 81 | 0 | 0 |

reactive modes: defend 105, counter 98, spell 1, leak 17, punish 80, save 310, hold 685, none 73

## test / periodic  (n=60, pro play rate 0.233)

| policy | accuracy | balanced acc | act rate | tpr | tnr | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|
| reactive | 0.650 | 0.474 | 0.183 | 0.143 | 0.804 | 2 | 9 | 12 | 37 |
| trigger | 0.317 | 0.554 | 0.917 | 1.000 | 0.109 | 14 | 41 | 0 | 5 |
| always | 0.233 | 0.500 | 1.000 | 1.000 | 0.000 | 14 | 46 | 0 | 0 |

reactive modes: defend 1, counter 5, leak 2, punish 3, save 8, hold 36, none 5

## val / all  (n=1366, pro play rate 0.947)

| policy | accuracy | balanced acc | act rate | tpr | tnr | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|
| reactive | 0.302 | 0.547 | 0.267 | 0.272 | 0.822 | 352 | 13 | 941 | 60 |
| trigger | 0.916 | 0.555 | 0.953 | 0.959 | 0.151 | 1240 | 62 | 53 | 11 |
| always | 0.947 | 0.500 | 1.000 | 1.000 | 0.000 | 1293 | 73 | 0 | 0 |

reactive modes: defend 138, counter 166, leak 11, punish 50, save 340, hold 597, none 64

## val / periodic  (n=50, pro play rate 0.240)

| policy | accuracy | balanced acc | act rate | tpr | tnr | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|
| reactive | 0.760 | 0.643 | 0.200 | 0.417 | 0.868 | 5 | 5 | 7 | 33 |
| trigger | 0.440 | 0.632 | 0.800 | 1.000 | 0.263 | 12 | 28 | 0 | 10 |
| always | 0.240 | 0.500 | 1.000 | 1.000 | 0.000 | 12 | 38 | 0 | 0 |

reactive modes: defend 4, counter 4, punish 2, save 13, hold 17, none 10

