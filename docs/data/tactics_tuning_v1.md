# Tactics weight tuning (MAC-A2)

- dataset: `/Users/tylerjackson/Projects/clashbot/clashBot/.claude/worktrees/agent-ab1e136742a752e0c/dataset/v1`, KG: True, passes: 3
- objective: `[best.card == pro] + 0.5 * [pro card in moves[:3]] + 0.25 * [lane_of(best.tile) == pro lane, when the card matches]`, mean over play records whose pro card is in the affordable known pool (`eligible`); n, top-1, top-3 and coverage are over all known-card play records, like `eval_tactics.py`; random top-1 = a uniform pick from the pool, same base
- decision: **tuned weights kept** - pass 2 kept: val objective 0.5129 -> 0.5321

## Objective, top-1, top-3 and coverage: before -> after

| split | situation | n | n eligible | objective | top-1 card | top-3 card | coverage | random top-1 |
|---|---|---|---|---|---|---|---|---|
| train | all | 5404 | 2544 | 0.5121 -> 0.5291 | 13.3% -> 14.0% | 19.5% -> 19.6% | 56.8% -> 56.8% | 19.9% |
| train | push | 2233 | 1029 | 0.8083 -> 0.8474 | 19.0% -> 20.6% | 33.2% -> 33.3% | 85.4% -> 85.4% | 20.0% |
| train | no push | 3171 | 1515 | 0.3109 -> 0.3129 | 9.3% -> 9.3% | 9.8% -> 9.9% | 36.7% -> 36.7% | 19.8% |
| val | all | 1216 | 560 | 0.5129 -> 0.5321 | 12.8% -> 13.5% | 19.7% -> 19.8% | 60.2% -> 60.2% | 17.6% |
| val | push | 514 | 246 | 0.6961 -> 0.7439 | 16.2% -> 18.1% | 31.9% -> 32.1% | 84.6% -> 84.6% | 18.6% |
| val | no push | 702 | 314 | 0.3694 -> 0.3662 | 10.3% -> 10.1% | 10.8% -> 10.8% | 42.3% -> 42.3% | 16.8% |
| test | all | 1194 | 559 | 0.4606 -> 0.4758 | 11.6% -> 12.3% | 18.2% -> 18.3% | 56.1% -> 56.1% | 17.4% |
| test | push | 477 | 218 | 0.7408 -> 0.7683 | 16.6% -> 17.8% | 32.1% -> 32.3% | 86.6% -> 86.6% | 17.6% |
| test | no push | 717 | 341 | 0.2815 -> 0.2889 | 8.4% -> 8.6% | 8.9% -> 9.1% | 35.8% -> 35.8% | 17.3% |

## By intent of the best move: before -> after

| split | intent | n best | share of play records | card precision |
|---|---|---|---|---|
| train | bait | 111 -> 118 | 2.1% -> 2.2% | 30.6% -> 29.7% |
| train | counter_push | 378 -> 353 | 7.0% -> 6.5% | 19.6% -> 19.8% |
| train | cycle | 456 -> 465 | 8.4% -> 8.6% | 25.4% -> 25.4% |
| train | defend | 1879 -> 1841 | 34.8% -> 34.1% | 22.0% -> 24.1% |
| train | leak | 63 -> 63 | 1.2% -> 1.2% | 39.7% -> 39.7% |
| train | pressure | 15 -> 40 | 0.3% -> 0.7% | 20.0% -> 17.5% |
| train | punish | 13 -> 10 | 0.2% -> 0.2% | 30.8% -> 40.0% |
| train | support | 155 -> 180 | 2.9% -> 3.3% | 31.0% -> 30.0% |
| val | bait | 52 -> 52 | 4.3% -> 4.3% | 28.8% -> 28.8% |
| val | counter_push | 112 -> 101 | 9.2% -> 8.3% | 21.4% -> 18.8% |
| val | cycle | 112 -> 113 | 9.2% -> 9.3% | 20.5% -> 21.2% |
| val | defend | 426 -> 417 | 35.0% -> 34.3% | 19.5% -> 21.6% |
| val | leak | 7 -> 6 | 0.6% -> 0.5% | 57.1% -> 66.7% |
| val | pressure | 4 -> 17 | 0.3% -> 1.4% | 25.0% -> 29.4% |
| val | punish | 4 -> 2 | 0.3% -> 0.2% | 0.0% -> 0.0% |
| val | support | 15 -> 24 | 1.2% -> 2.0% | 33.3% -> 29.2% |
| test | bait | 19 -> 20 | 1.6% -> 1.7% | 31.6% -> 30.0% |
| test | counter_push | 67 -> 59 | 5.6% -> 4.9% | 31.3% -> 30.5% |
| test | cycle | 132 -> 132 | 11.1% -> 11.1% | 19.7% -> 19.7% |
| test | defend | 408 -> 401 | 34.2% -> 33.6% | 19.1% -> 20.9% |
| test | leak | 7 -> 7 | 0.6% -> 0.6% | 14.3% -> 14.3% |
| test | pressure | 2 -> 7 | 0.2% -> 0.6% | 0.0% -> 42.9% |
| test | punish | 3 -> 2 | 0.2% -> 0.2% | 0.0% -> 0.0% |
| test | support | 32 -> 42 | 2.7% -> 3.5% | 21.9% -> 21.4% |

## Records

| split | known-card plays | eligible (pro card in the pool) |
|---|---|---|
| train | 5404 | 2544 |
| val | 1216 | 560 |
| test | 1194 | 559 |

## Search

| pass | train objective | val objective | weights changed |
|---|---|---|---|
| 0 | 0.5121 | 0.5129 | 0 |
| 1 | 0.5269 | 0.5277 | 18 |
| 2 | 0.5291 | 0.5321 | 5 |
| 3 | 0.5297 | 0.5299 | 2 |

Candidates vetoed by a design guard (improved train, broke a guard): targeter_swarm_bonus x6, support x2, cycle x5, cycle_block_score x9, stop_high x2, stop_low x2, bait_unknown x3

## Weights that changed

| weight | default | tuned |
|---|---|---|
| base | 0.5 | 0.1875 |
| air_cannot_hit_penalty | 0.5 | 0.5625 |
| tank_swarm_bonus | 0.3 | 0.1125 |
| swarm_splash_bonus | 0.6 | 0.9 |
| swarm_melee_penalty | 0.4 | 0.3 |
| targeter_building_bonus | 0.5 | 0.375 |
| targeter_swarm_bonus | 0.3 | 0.375 |
| targeter_mini_tank_bonus | 0.3 | 0.2812 |
| kg_factor | 0.2 | 0.1 |
| trade_overpay_per_elixir | 0.15 | 0.3 |
| trade_overpay_slack | 1.0 | 0.75 |
| trade_good_bonus | 0.1 | 0.15 |
| stop_high | 0.3 | 0.15 |
| stop_medium | 0.15 | 0.1125 |
| stop_low | -0.35 | -0.175 |
| counter_push | 0.6 | 0.75 |
| pressure | 0.45 | 0.9 |
| support | 0.5 | 1.0 |
