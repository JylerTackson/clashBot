# Shots review: runs/poc/20261106-233907

- decisions recorded: 116 (sent taps 100, dry-run taps 0, holds 16, skipped 0)
- verdicts over sent taps (100): landed 95, failed 5, unknown 0; landing rate 95% (unknown left out)
- landed = elixir fell by >= cost - 0.6 at +0.5 s or +1.5 s (hero ability: cost - 1.5), or the tapped slot reads another card; elixir and hand are the raw perception readings when recorded
- sheets: 20 written (all taps) in runs/smoke/AL/review

### Landing rate by card

| card | taps | landed | failed | unknown | landing rate |
|---|---|---|---|---|---|
| ability:musketeer | 6 | 6 | 0 | 0 | 100% |
| electro-dragon | 6 | 5 | 1 | 0 | 83% |
| fireball | 3 | 2 | 1 | 0 | 67% |
| knight | 21 | 21 | 0 | 0 | 100% |
| mini-p-e-k-k-a | 3 | 3 | 0 | 0 | 100% |
| musketeer | 15 | 15 | 0 | 0 | 100% |
| ram-rider | 11 | 11 | 0 | 0 | 100% |
| tesla | 21 | 19 | 2 | 0 | 90% |
| zap | 14 | 13 | 1 | 0 | 93% |

### Landing rate by slot

| slot | taps | landed | failed | unknown | landing rate |
|---|---|---|---|---|---|
| 0 | 19 | 17 | 2 | 0 | 89% |
| 1 | 11 | 9 | 2 | 0 | 82% |
| 2 | 38 | 37 | 1 | 0 | 97% |
| 3 | 26 | 26 | 0 | 0 | 100% |
| ? | 6 | 6 | 0 | 0 | 100% |

### Landing rate by target row band (tile row, own side = low rows)

| band | taps | landed | failed | unknown | landing rate |
|---|---|---|---|---|---|
| rows 0-7 (own back) | 59 | 56 | 3 | 0 | 95% |
| rows 16-23 (enemy front) | 4 | 4 | 0 | 0 | 100% |
| rows 8-15 (own front) | 37 | 35 | 2 | 0 | 95% |

### Landing rate by mode

| mode | taps | landed | failed | unknown | landing rate |
|---|---|---|---|---|---|
| counter | 8 | 8 | 0 | 0 | 100% |
| defend | 72 | 67 | 5 | 0 | 93% |
| leak | 8 | 8 | 0 | 0 | 100% |
| learned | 11 | 11 | 0 | 0 | 100% |
| punish | 1 | 1 | 0 | 0 | 100% |

### Tapped card still in its slot at +1.5 s

- all: 7/94 (7%)
- landed: 2/89 (2%)
- failed: 5/5 (100%)
- unknown: 0/0

### Taps

| idx | t | card | slot | tile | mode | verdict | elixir before > after | drop / cost | slot read +1.5s |
|---|---|---|---|---|---|---|---|---|---|
| 234 | 24.7 | knight | 1 | 5,2.5 | leak | landed | 10.0 > 7.0 | 10.0 / 3.0 | fireball |
| 324 | 34.7 | zap | 0 | 4,14.8 | leak | landed | 10.0 > 9.0 | 6.0 / 2.0 | tesla |
| 333 | 35.8 | electro-dragon | 2 | 5,2.5 | defend | failed | 8.0 > 4.0 | 4.0 / 5.0 | electro-dragon |
| 334 | 39.4 | tesla | 1 | 5.5,6.5 | defend | landed | 9.0 > 1.0 | 8.0 / 4.0 | mini-p-e-k-k-a |
| 469 | 55.9 | ram-rider | 2 | 3.5,14 | counter | landed | 6.0 > 2.0 | 4.0 / 5.0 | mini-p-e-k-k-a |
| 574 | 68.7 | musketeer | 0 | 5,2.5 | defend | landed | 5.0 > 2.0 | 5.0 / 4.0 | tesla |
| 598 | 71.9 | zap | 0 | 2.7,15 | defend | landed | 2.0 > 2.0 | 1.0 / 2.0 | tesla |
| 798 | 95.2 | tesla | 0 | 8,9 | learned | landed | 9.0 > 4.0 | 5.0 / 4.0 | tesla |
| 923 | 109.2 | tesla | 0 | 8,9 | learned | landed | 9.0 > 5.0 | 4.0 / 4.0 | tesla |
| 1047 | 123.4 | ram-rider | 0 | 3.5,14 | leak | landed | 10.0 > 5.0 | 5.0 / 5.0 | tesla |
| 1075 | 126.9 | musketeer | 0 | 5,2.5 | defend | landed | 5.0 > 2.0 | 3.0 / 4.0 | tesla |
| 1086 | 128.7 | zap | 0 | 4.9,9.5 | defend | landed | 2.0 > 1.0 | 1.0 / 2.0 | tesla |
| 1129 | 134.7 | ability:musketeer | None | 5,7 | defend | landed | 3.0 > 2.0 | 2.0 / 3.0 | ? |
| 1162 | 139.0 | knight | 2 | 3.5,1 | defend | landed | 3.0 > 2.0 | 1.0 / 3.0 | mini-p-e-k-k-a |
| 1199 | 144.4 | knight | 3 | 2,3.5 | defend | landed | 3.0 > 2.0 | 1.0 / 3.0 | mini-p-e-k-k-a |
| 1239 | 150.0 | tesla | 2 | 5.5,6.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | mini-p-e-k-k-a |
| 1297 | 158.1 | electro-dragon | 0 | 5,2.5 | defend | landed | 5.0 > 2.0 | 5.0 / 5.0 | tesla |
| 1329 | 162.6 | musketeer | 2 | 5,2.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | mini-p-e-k-k-a |
| 1352 | 166.7 | zap | 0 | 7.2,7.3 | defend | landed | 2.0 > 2.0 | 0.0 / 2.0 | tesla |
| 1449 | 179.0 | knight | 2 | 3,8 | learned | landed | 9.0 > 7.0 | 2.0 / 3.0 | mini-p-e-k-k-a |
| 1473 | 182.3 | ram-rider | 3 | 3.5,14 | counter | landed | 8.0 > 4.0 | 8.0 / 5.0 | fireball |
| 1546 | 191.9 | tesla | 3 | 5.5,6.5 | defend | landed | 10.0 > 5.0 | 10.0 / 4.0 | fireball |
| 2002 | 243.4 | ram-rider | 0 | 3,2 | learned | landed | 9.0 > 4.0 | 5.0 / 5.0 | tesla |
| 2151 | 260.5 | zap | 2 | 2.6,20.2 | leak | landed | 10.0 > 3.0 | 7.0 / 2.0 | mini-p-e-k-k-a |
| 2158 | 261.5 | electro-dragon | 3 | 5,2.5 | defend | landed | 8.0 > 3.0 | 5.0 / 5.0 | fireball |
| 2179 | 264.9 | tesla | 0 | 7,11 | defend | failed | 4.0 > 1.0 | 3.0 / 4.0 | tesla |
| 2242 | 273.3 | knight | 0 | 4.5,5.5 | defend | landed | 3.0 > 1.0 | 2.0 / 3.0 | tesla |
| 2376 | 290.2 | ram-rider | 0 | 3.5,14 | counter | landed | 6.0 > 2.0 | 6.0 / 5.0 | tesla |
| 2472 | 301.8 | musketeer | 2 | 5,2.5 | defend | landed | 5.0 > 2.0 | 3.0 / 4.0 | mini-p-e-k-k-a |
| 2535 | 309.2 | zap | 0 | 2.8,10.6 | defend | landed | 3.0 > 2.0 | 3.0 / 2.0 | tesla |
| 2649 | 323.4 | tesla | 2 | 5.5,6.5 | defend | landed | 6.0 > 2.0 | 4.0 / 4.0 | mini-p-e-k-k-a |
| 2679 | 327.8 | knight | 2 | 7.5,5 | defend | landed | 3.0 > 1.0 | 2.0 / 3.0 | mini-p-e-k-k-a |
| 2804 | 343.6 | ram-rider | 2 | 3.5,14 | counter | landed | 6.0 > 2.0 | 4.0 / 5.0 | mini-p-e-k-k-a |
| 2872 | 352.1 | tesla | 0 | 7,7 | defend | failed | 4.0 > 1.0 | 3.0 / 4.0 | tesla |
| 2997 | 367.5 | knight | 3 | 7.5,12.5 | leak | landed | 10.0 > 4.0 | 6.0 / 3.0 | fireball |
| 3004 | 368.5 | musketeer | 2 | 5,2.5 | defend | landed | 6.0 > 2.0 | 4.0 / 4.0 | mini-p-e-k-k-a |
| 3161 | 386.7 | knight | 2 | 5,2.5 | leak | landed | 10.0 > 7.0 | 10.0 / 3.0 | mini-p-e-k-k-a |
| 3175 | 389.0 | ram-rider | 2 | 14.5,14 | counter | landed | 7.0 > 3.0 | 4.0 / 5.0 | mini-p-e-k-k-a |
| 3187 | 391.1 | tesla | 2 | 5.5,6.5 | defend | landed | 4.0 > 2.0 | 3.0 / 4.0 | mini-p-e-k-k-a |
| 3222 | 396.3 | knight | 2 | 4,4.5 | defend | landed | 3.0 > 2.0 | 1.0 / 3.0 | mini-p-e-k-k-a |
| 3260 | 402.2 | musketeer | 2 | 13,2.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | mini-p-e-k-k-a |
| 3297 | 407.9 | ability:musketeer | None | 13,7 | defend | landed | 4.0 > 2.0 | 2.0 / 3.0 | ? |
| 3333 | 413.0 | knight | 2 | 14,8.5 | learned | landed | 4.0 > 2.0 | 2.0 / 3.0 | mini-p-e-k-k-a |
| 3535 | 437.3 | tesla | 2 | 8,9 | learned | landed | 9.0 > 5.0 | 4.0 / 4.0 | mini-p-e-k-k-a |
| 3632 | 448.6 | ram-rider | 0 | 3,2 | learned | landed | 9.0 > 4.0 | 5.0 / 5.0 | tesla |
| 3754 | 462.6 | musketeer | 3 | 1,8 | learned | landed | 9.0 > 5.0 | 9.0 / 4.0 | fireball |
| 3874 | 476.6 | zap | 1 | 2.7,19.1 | leak | landed | 10.0 > 8.0 | 5.0 / 2.0 | fireball |
| 3900 | 479.9 | electro-dragon | 3 | 5,2.5 | punish | landed | 8.0 > 4.0 | 8.0 / 5.0 | fireball |
| 3928 | 483.7 | tesla | 1 | 7,10.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | fireball |
| 3993 | 492.0 | knight | 3 | 3,7 | defend | landed | 3.0 > 1.0 | 2.0 / 3.0 | fireball |
| 4133 | 509.0 | ram-rider | 1 | 3.5,14 | counter | landed | 6.0 > 2.0 | 4.0 / 5.0 | fireball |
| 4248 | 522.8 | musketeer | 3 | 5,2.5 | defend | landed | 5.0 > 2.0 | 3.0 / 4.0 | fireball |
| 4277 | 527.0 | knight | 2 | 7.5,4 | defend | landed | 3.0 > 1.0 | 2.0 / 3.0 | mini-p-e-k-k-a |
| 4460 | 548.9 | tesla | 3 | 5.5,6.5 | defend | landed | 8.0 > 4.0 | 8.0 / 4.0 | fireball |
| 4488 | 552.9 | zap | 1 | 6,5.4 | defend | landed | 6.0 > 2.0 | 4.0 / 2.0 | fireball |
| 4496 | 553.9 | tesla | 2 | 5.5,6.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | mini-p-e-k-k-a |
| 4523 | 558.4 | musketeer | 2 | 5,2.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | mini-p-e-k-k-a |
| 4548 | 562.7 | knight | 2 | 9,9 | learned | landed | 3.0 > 2.0 | 1.0 / 3.0 | mini-p-e-k-k-a |
| 4587 | 568.6 | knight | 3 | 7.5,9 | defend | landed | 4.0 > 2.0 | 2.0 / 3.0 | fireball |
| 4625 | 574.5 | fireball | 2 | 3.4,15.3 | defend | landed | 4.0 > 1.0 | 3.0 / 4.0 | mini-p-e-k-k-a |
| 4669 | 580.8 | tesla | 2 | 12.5,6.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | mini-p-e-k-k-a |
| 4737 | 590.6 | musketeer | 2 | 5,2.5 | defend | landed | 6.0 > 4.0 | 6.0 / 4.0 | mini-p-e-k-k-a |
| 4768 | 594.6 | fireball | 0 | 4,6.9 | defend | landed | 5.0 > 3.0 | 3.0 / 4.0 | tesla |
| 4776 | 596.2 | ability:musketeer | None | 4,8 | defend | landed | 3.0 > 2.0 | 2.0 / 3.0 | ? |
| 4803 | 600.6 | zap | 3 | 13.1,6.4 | defend | landed | 3.0 > 2.0 | 1.0 / 2.0 | fireball |
| 4841 | 606.2 | tesla | 3 | 11,11 | defend | landed | 5.0 > 2.0 | 3.0 / 4.0 | fireball |
| 5131 | 640.9 | mini-p-e-k-k-a | 3 | 5,2.5 | leak | landed | 10.0 > 6.0 | 4.0 / 4.0 | fireball |
| 5204 | 649.5 | musketeer | 2 | 5,2.5 | defend | landed | 8.0 > 5.0 | 4.0 / 4.0 | mini-p-e-k-k-a |
| 5214 | 650.8 | mini-p-e-k-k-a | 2 | 7.5,1 | defend | landed | 5.0 > 5.0 | 0.0 / 4.0 | electro-dragon |
| 5268 | 657.5 | tesla | 3 | 5.5,6.5 | defend | landed | 7.0 > 3.0 | 4.0 / 4.0 | fireball |
| 5323 | 665.0 | ram-rider | 3 | 3.5,14 | counter | landed | 6.0 > 2.0 | 6.0 / 5.0 | fireball |
| 5415 | 676.7 | electro-dragon | 2 | 5,2.5 | defend | landed | 5.0 > 1.0 | 4.0 / 5.0 | mini-p-e-k-k-a |
| 5434 | 679.8 | zap | 1 | 3.6,7.4 | defend | failed | 2.0 > 2.0 | 0.0 / 2.0 | zap |
| 5461 | 683.2 | zap | 1 | 4.6,7.5 | defend | landed | 2.0 > 2.0 | 1.0 / 2.0 | fireball |
| 5538 | 692.8 | knight | 3 | 7.5,10.5 | defend | landed | 3.0 > 2.0 | 3.0 / 3.0 | fireball |
| 5607 | 701.7 | musketeer | 1 | 5,2.5 | defend | landed | 4.0 > 1.0 | 3.0 / 4.0 | fireball |
| 5720 | 715.1 | ability:musketeer | None | 5,6 | defend | landed | 4.0 > 2.0 | 2.0 / 3.0 | ? |
| 5857 | 731.7 | tesla | 3 | 5.5,6.5 | defend | landed | 7.0 > 4.0 | 3.0 / 4.0 | fireball |
| 5863 | 732.9 | knight | 2 | 5,1 | defend | landed | 4.0 > 1.0 | 3.0 / 3.0 | mini-p-e-k-k-a |
| 5890 | 737.0 | zap | 2 | 5.7,8.4 | defend | landed | 2.0 > 1.0 | 1.0 / 2.0 | mini-p-e-k-k-a |
| 6014 | 751.9 | knight | 2 | 7.5,8 | defend | landed | 5.0 > 4.0 | 2.0 / 3.0 | mini-p-e-k-k-a |
| 6036 | 755.3 | tesla | 3 | 10.5,4 | defend | landed | 5.0 > 2.0 | 3.0 / 4.0 | fireball |
| 6072 | 760.6 | musketeer | 2 | 13,2.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | mini-p-e-k-k-a |
| 6106 | 765.9 | knight | 2 | 7.5,1 | defend | landed | 3.0 > 1.0 | 2.0 / 3.0 | mini-p-e-k-k-a |
| 6154 | 772.8 | tesla | 3 | 5.5,6.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | fireball |
| 6198 | 779.1 | ability:musketeer | None | 13,16 | defend | landed | 5.0 > 0.0 | 5.0 / 3.0 | ? |
| 6205 | 780.1 | zap | 2 | 13.6,13.8 | defend | landed | 3.0 > 2.0 | 1.0 / 2.0 | mini-p-e-k-k-a |
| 6228 | 784.0 | knight | 3 | 10.5,14 | defend | landed | 3.0 > 2.0 | 1.0 / 3.0 | fireball |
| 6264 | 789.3 | musketeer | 3 | 5,2.5 | defend | landed | 4.0 > 2.0 | 4.0 / 4.0 | fireball |
| 6297 | 794.0 | knight | 3 | 3.5,9.5 | defend | landed | 3.0 > 2.0 | 1.0 / 3.0 | fireball |
| 6335 | 799.8 | tesla | 3 | 12.5,6.5 | defend | landed | 4.0 > 2.0 | 2.0 / 4.0 | fireball |
| 6381 | 806.7 | ability:musketeer | None | 9,6 | defend | landed | 5.0 > 3.0 | 3.0 / 3.0 | ? |
| 6397 | 809.3 | zap | 3 | 3.9,17 | defend | landed | 3.0 > 3.0 | 0.0 / 2.0 | fireball |
| 6597 | 833.1 | tesla | 1 | 8,9 | learned | landed | 9.0 > 5.0 | 4.0 / 4.0 | fireball |
| 6697 | 844.5 | electro-dragon | 2 | 8,9 | learned | landed | 9.0 > 4.0 | 5.0 / 5.0 | mini-p-e-k-k-a |
| 6807 | 857.4 | mini-p-e-k-k-a | 0 | 5,10.5 | defend | landed | 8.0 > 5.0 | 4.0 / 4.0 | tesla |
| 6887 | 866.8 | musketeer | 2 | 5,2.5 | defend | landed | 8.0 > 4.0 | 4.0 / 4.0 | mini-p-e-k-k-a |
| 6898 | 868.1 | fireball | 1 | 3.8,10.4 | defend | failed | 4.0 > 1.0 | 3.0 / 4.0 | fireball |
| 6952 | 875.6 | knight | 3 | 7.5,4 | defend | landed | 3.0 > 1.0 | 2.0 / 3.0 | fireball |
| 7091 | 892.2 | ram-rider | 2 | 3.5,14 | counter | landed | 6.0 > 2.0 | 4.0 / 5.0 | mini-p-e-k-k-a |
