# Tactics weight tuning (MAC-A2)

- dataset: `/Users/tylerjackson/Projects/clashbot/clashBot/.claude/worktrees/agent-aad7f4da136201045/dataset/v1`, KG: True, passes: 3
- objective: `[best.card == pro] + 0.5 * [pro card in moves[:3]] + 0.25 * [lane_of(best.tile) == pro lane, when the card matches]`, mean over play records whose pro card is in the affordable known pool (`eligible`); n, top-1, top-3 and coverage are over all known-card play records, like `eval_tactics.py`; random top-1 = a uniform pick from the pool, same base
- decision: **tuned weights kept** - pass 2 kept: val objective 0.4924 -> 0.5125

## Objective, top-1, top-3 and coverage: before -> after

| split | situation | n | n eligible | objective | top-1 card | top-3 card | coverage | random top-1 |
|---|---|---|---|---|---|---|---|---|
| train | all | 5404 | 2544 | 0.4406 -> 0.4830 | 11.8% -> 12.9% | 16.0% -> 17.4% | 49.1% -> 51.8% | 19.9% |
| train | push | 2832 | 1304 | 0.5721 -> 0.6511 | 14.2% -> 16.3% | 21.9% -> 24.5% | 61.0% -> 66.2% | 19.6% |
| train | no push | 2572 | 1240 | 0.3024 -> 0.3063 | 9.1% -> 9.2% | 9.6% -> 9.6% | 36.0% -> 36.0% | 20.2% |
| val | all | 1216 | 560 | 0.4924 -> 0.5125 | 13.2% -> 13.5% | 16.8% -> 17.8% | 52.1% -> 54.6% | 17.6% |
| val | push | 630 | 307 | 0.5839 -> 0.6319 | 15.9% -> 17.0% | 22.4% -> 24.4% | 60.8% -> 65.7% | 19.0% |
| val | no push | 586 | 253 | 0.3814 -> 0.3676 | 10.2% -> 9.7% | 10.8% -> 10.8% | 42.7% -> 42.7% | 16.0% |
| test | all | 1194 | 559 | 0.3631 -> 0.3940 | 10.0% -> 10.8% | 12.6% -> 13.7% | 41.7% -> 45.9% | 17.4% |
| test | push | 572 | 258 | 0.4680 -> 0.5291 | 11.7% -> 13.3% | 17.0% -> 19.2% | 49.3% -> 58.0% | 17.3% |
| test | no push | 622 | 301 | 0.2733 -> 0.2782 | 8.4% -> 8.5% | 8.5% -> 8.5% | 34.7% -> 34.7% | 17.6% |

## By intent of the best move: before -> after

| split | intent | n best | share of play records | card precision |
|---|---|---|---|---|
| train | bait | 103 -> 192 | 1.9% -> 3.5% | 32.0% -> 26.6% |
| train | counter_push | 426 -> 242 | 7.9% -> 4.5% | 19.0% -> 21.1% |
| train | cycle | 408 -> 391 | 7.5% -> 7.2% | 25.5% -> 26.1% |
| train | defend | 1429 -> 1675 | 26.4% -> 31.0% | 23.9% -> 24.4% |
| train | leak | 58 -> 61 | 1.1% -> 1.1% | 43.1% -> 42.6% |
| train | pressure | 15 -> 32 | 0.3% -> 0.6% | 20.0% -> 21.9% |
| train | punish | 14 -> 5 | 0.3% -> 0.1% | 28.6% -> 60.0% |
| train | spell_value | 8 -> 6 | 0.1% -> 0.1% | 12.5% -> 16.7% |
| train | split | 59 -> 60 | 1.1% -> 1.1% | 13.6% -> 21.7% |
| train | support | 131 -> 135 | 2.4% -> 2.5% | 26.7% -> 27.4% |
| val | bait | 52 -> 88 | 4.3% -> 7.2% | 28.8% -> 22.7% |
| val | counter_push | 132 -> 63 | 10.9% -> 5.2% | 24.2% -> 25.4% |
| val | cycle | 101 -> 99 | 8.3% -> 8.1% | 20.8% -> 20.2% |
| val | defend | 294 -> 340 | 24.2% -> 28.0% | 25.5% -> 25.0% |
| val | leak | 6 -> 7 | 0.5% -> 0.6% | 66.7% -> 71.4% |
| val | pressure | 4 -> 11 | 0.3% -> 0.9% | 25.0% -> 27.3% |
| val | punish | 4 -> 2 | 0.3% -> 0.2% | 0.0% -> 0.0% |
| val | spell_value | 3 -> 3 | 0.2% -> 0.2% | 0.0% -> 0.0% |
| val | split | 26 -> 36 | 2.1% -> 3.0% | 26.9% -> 25.0% |
| val | support | 11 -> 15 | 0.9% -> 1.2% | 45.5% -> 40.0% |
| test | bait | 18 -> 41 | 1.5% -> 3.4% | 27.8% -> 36.6% |
| test | counter_push | 78 -> 35 | 6.5% -> 2.9% | 32.0% -> 31.4% |
| test | cycle | 125 -> 119 | 10.5% -> 10.0% | 19.2% -> 18.5% |
| test | defend | 234 -> 303 | 19.6% -> 25.4% | 24.8% -> 23.8% |
| test | leak | 8 -> 10 | 0.7% -> 0.8% | 12.5% -> 10.0% |
| test | pressure | 2 -> 7 | 0.2% -> 0.6% | 0.0% -> 42.9% |
| test | punish | 3 -> 2 | 0.2% -> 0.2% | 0.0% -> 0.0% |
| test | spell_value | 1 -> 0 | 0.1% -> - | 0.0% -> - |
| test | split | 6 -> 5 | 0.5% -> 0.4% | 16.7% -> 0.0% |
| test | support | 23 -> 26 | 1.9% -> 2.2% | 21.7% -> 19.2% |

## Records

| split | known-card plays | eligible (pro card in the pool) |
|---|---|---|
| train | 5404 | 2544 |
| val | 1216 | 560 |
| test | 1194 | 559 |

## Search

| pass | train objective | val objective | weights changed |
|---|---|---|---|
| 0 | 0.4406 | 0.4924 | 0 |
| 1 | 0.4739 | 0.5045 | 18 |
| 2 | 0.4830 | 0.5125 | 6 |
| 3 | 0.4834 | 0.5125 | 2 |

Candidates vetoed by a design guard (improved train, broke a guard): cycle_block_score x9, answered_factor x1

## Weights that changed

| weight | default | tuned |
|---|---|---|
| base | 0.5 | 1.0 |
| air_cannot_hit_penalty | 0.5 | 1.0 |
| tank_swarm_bonus | 0.3 | 0.375 |
| swarm_splash_bonus | 0.6 | 1.2 |
| targeter_building_bonus | 0.5 | 1.0 |
| targeter_swarm_bonus | 0.3 | 0.45 |
| targeter_mini_tank_bonus | 0.3 | 0.15 |
| kg_factor | 0.2 | 0.15 |
| trade_overpay_per_elixir | 0.15 | 0.05 |
| trade_good_bonus | 0.1 | 0.15 |
| stop_high | 0.3 | 0.15 |
| answered_frac | 0.6 | 0.15 |
| answered_factor | 0.3 | 0.6 |
| tower_ok_damage | 300.0 | 112.5 |
| adequate_frac | 0.35 | 1.05 |
| spell_value | 0.55 | 0.4125 |
| counter_push | 0.6 | 0.45 |
| punish | 0.55 | 0.4125 |
| leak_wc | 0.5 | 0.625 |
| bait_unknown | 0.45 | 0.5625 |
| pressure | 0.45 | 0.5625 |
