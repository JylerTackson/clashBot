## Play records: engine vs the pro play

| split | situation | n | coverage | top-1 card | top-3 card | lane (top-1) | zone (top-1) | tile dist (top-1) | n top-1 | lane (top-3) | zone (top-3) | tile dist (top-3) | n top-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | all | 1216 | 45.8% | 10.4% | 15.8% | 30.9% | 32.5% | 7.37 | 126 | 30.7% | 37.5% | 7.29 | 192 |
| val | push | 514 | 86.0% | 19.1% | 31.7% | 30.6% | 25.5% | 7.27 | 98 | 30.1% | 33.7% | 7.27 | 163 |
| val | no push | 702 | 16.4% | 4.0% | 4.1% | 32.1% | 57.1% | 7.66 | 28 | 34.5% | 58.6% | 7.38 | 29 |
| test | all | 1194 | 41.2% | 9.0% | 14.9% | 26.9% | 35.2% | 8.33 | 108 | 27.0% | 34.3% | 8.64 | 178 |
| test | push | 477 | 87.2% | 18.0% | 32.5% | 27.9% | 26.7% | 8.22 | 86 | 27.1% | 29.0% | 8.68 | 155 |
| test | no push | 717 | 10.6% | 3.1% | 3.2% | 22.7% | 68.2% | 8.66 | 22 | 26.1% | 69.6% | 8.41 | 23 |

## 4-way grid: tactics ranker vs the other rankers

| split | n | tactics ranker | random | placement prior | d1 fine-tuned | scored share | acc when scored | acc push | acc no push |
|---|---|---|---|---|---|---|---|---|---|
| val | 761 | 27.5% | 25.0% | 49.3% | 52.4% | 28.5% | 27.7% | 29.8% | 25.8% |
| test | 766 | 25.5% | 25.0% | 38.4% | 48.3% | 26.8% | 23.9% | 24.0% | 26.4% |

lane / zone / tile dist (top-1) are measured on the best move when its card is the pro card; (top-3) on the first of the top-3 moves with the pro card. Tile distance is in tiles. Scored share = grid lines where at least one option got an engine score (else the ranker picks the first option).
