## Play records: engine vs the pro play

| split | situation | n | coverage | top-1 card | top-3 card | lane (top-1) | zone (top-1) | tile dist (top-1) | n top-1 | lane (top-3) | zone (top-3) | tile dist (top-3) | n top-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | all | 1216 | 52.1% | 13.1% | 16.7% | 35.9% | 51.6% | 6.5 | 159 | 36.4% | 47.8% | 6.5 | 203 |
| val | push | 630 | 60.8% | 15.9% | 22.2% | 37.0% | 46.0% | 6.41 | 100 | 37.1% | 42.1% | 6.37 | 140 |
| val | no push | 586 | 42.7% | 10.1% | 10.8% | 33.9% | 61.0% | 6.64 | 59 | 34.9% | 60.3% | 6.76 | 63 |
| test | all | 1194 | 41.7% | 10.0% | 12.6% | 29.4% | 43.7% | 8.13 | 119 | 30.0% | 40.7% | 8.16 | 150 |
| test | push | 572 | 49.3% | 11.7% | 17.0% | 29.8% | 41.8% | 8.05 | 67 | 29.9% | 37.1% | 8.18 | 97 |
| test | no push | 622 | 34.7% | 8.4% | 8.5% | 28.8% | 46.2% | 8.24 | 52 | 30.2% | 47.2% | 8.13 | 53 |

## By intent of the best move (play records)

| split | intent | n best | share of plays | card precision | lane (card matched) | in top-3 | top-3 card precision |
|---|---|---|---|---|---|---|---|
| val | bait | 52 | 4.3% | 28.8% | 46.7% | 66 | 28.8% |
| val | counter_push | 142 | 11.7% | 21.8% | 41.9% | 195 | 20.5% |
| val | cycle | 102 | 8.4% | 20.6% | 19.1% | 106 | 21.7% |
| val | defend | 288 | 23.7% | 26.0% | 36.0% | 522 | 21.8% |
| val | leak | 5 | 0.4% | 80.0% | 50.0% | 5 | 80.0% |
| val | pressure | 13 | 1.1% | 30.8% | 25.0% | 14 | 28.6% |
| val | punish | 2 | 0.2% | 0.0% | - | 4 | 25.0% |
| val | spell_value | 3 | 0.2% | 0.0% | - | 4 | 0.0% |
| val | split | 7 | 0.6% | 42.9% | 0.0% | 20 | 30.0% |
| val | support | 19 | 1.6% | 31.6% | 50.0% | 24 | 25.0% |
| test | bait | 18 | 1.5% | 27.8% | 60.0% | 29 | 34.5% |
| test | counter_push | 77 | 6.5% | 28.6% | 22.7% | 108 | 27.8% |
| test | cycle | 125 | 10.5% | 19.2% | 25.0% | 128 | 20.3% |
| test | defend | 227 | 19.0% | 25.1% | 33.3% | 407 | 23.3% |
| test | leak | 8 | 0.7% | 12.5% | 0.0% | 9 | 11.1% |
| test | pressure | 7 | 0.6% | 42.9% | 0.0% | 7 | 42.9% |
| test | punish | 2 | 0.2% | 0.0% | - | 3 | 0.0% |
| test | spell_value | 1 | 0.1% | 0.0% | - | 2 | 0.0% |
| test | split | 4 | 0.3% | 25.0% | 0.0% | 5 | 20.0% |
| test | support | 29 | 2.4% | 20.7% | 33.3% | 34 | 17.6% |

## 4-way grid: tactics ranker vs the other rankers

| split | n | tactics ranker | random | placement prior | d1 fine-tuned | scored share | acc when scored | acc push | acc no push |
|---|---|---|---|---|---|---|---|---|---|
| val | 761 | 29.3% | 25.0% | 49.3% | 52.4% | 27.9% | 34.4% | 30.3% | 28.2% |
| test | 766 | 26.5% | 25.0% | 38.4% | 48.3% | 20.9% | 28.1% | 25.0% | 27.8% |

lane / zone / tile dist (top-1) are measured on the best move when its card is the pro card; (top-3) on the first of the top-3 moves with the pro card. Tile distance is in tiles. Scored share = grid lines where at least one option got an engine score (else the ranker picks the first option).
