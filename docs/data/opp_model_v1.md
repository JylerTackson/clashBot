# Opponent-deck model: leave-one-game-out evaluation (MAC-AK K3)

Model: temperature 0.05, count power 0.0, mix 0.6 nearest decks + 0.4 co-occurrence. Training pool: 407 deck occurrences, 216 unique decks, 119 cards. Evaluated games: 155 (106 with a full 8-card deck).

Cards are revealed in order of first opponent play; recall is over the hidden cards at top (8-k), precision over the top 3. `random` is the exact expectation. `logo` drops only this game's deck from training; `lodo` drops every identical deck.

## logo: all 7-8 card opponent decks (155 games)

| k | n | recall@(8-k) model | popularity | random | precision@3 model | popularity | random |
|---|---|---|---|---|---|---|---|
| 1 | 155 | 0.234 | 0.130 | 0.059 | 0.267 | 0.170 | 0.056 |
| 2 | 152 | 0.291 | 0.124 | 0.051 | 0.309 | 0.145 | 0.048 |
| 3 | 138 | 0.327 | 0.118 | 0.043 | 0.343 | 0.126 | 0.041 |
| 4 | 111 | 0.264 | 0.087 | 0.035 | 0.237 | 0.099 | 0.033 |
| 5 | 76 | 0.250 | 0.094 | 0.026 | 0.233 | 0.088 | 0.024 |

## logo: 8-card opponent decks only (106 games)

| k | n | recall@(8-k) model | popularity | random | precision@3 model | popularity | random |
|---|---|---|---|---|---|---|---|
| 1 | 106 | 0.236 | 0.138 | 0.059 | 0.264 | 0.164 | 0.059 |
| 2 | 104 | 0.276 | 0.125 | 0.051 | 0.314 | 0.138 | 0.051 |
| 3 | 100 | 0.316 | 0.118 | 0.043 | 0.353 | 0.123 | 0.043 |
| 4 | 87 | 0.264 | 0.092 | 0.035 | 0.257 | 0.107 | 0.035 |
| 5 | 60 | 0.250 | 0.094 | 0.026 | 0.250 | 0.094 | 0.026 |

## logo: test split only (20 games)

| k | n | recall@(8-k) model | popularity | random | precision@3 model | popularity | random |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 0.318 | 0.129 | 0.059 | 0.383 | 0.150 | 0.056 |
| 2 | 20 | 0.350 | 0.117 | 0.051 | 0.400 | 0.133 | 0.048 |
| 3 | 19 | 0.305 | 0.100 | 0.043 | 0.368 | 0.088 | 0.040 |
| 4 | 13 | 0.231 | 0.064 | 0.035 | 0.205 | 0.077 | 0.032 |
| 5 | 7 | 0.095 | 0.095 | 0.026 | 0.095 | 0.095 | 0.025 |

## lodo: all 7-8 card opponent decks (155 games)

| k | n | recall@(8-k) model | popularity | random | precision@3 model | popularity | random |
|---|---|---|---|---|---|---|---|
| 1 | 155 | 0.231 | 0.130 | 0.059 | 0.254 | 0.170 | 0.056 |
| 2 | 152 | 0.282 | 0.124 | 0.051 | 0.289 | 0.145 | 0.048 |
| 3 | 138 | 0.309 | 0.112 | 0.043 | 0.324 | 0.126 | 0.041 |
| 4 | 111 | 0.260 | 0.085 | 0.035 | 0.231 | 0.099 | 0.033 |
| 5 | 76 | 0.246 | 0.094 | 0.026 | 0.228 | 0.088 | 0.024 |

## lodo: 8-card opponent decks only (106 games)

| k | n | recall@(8-k) model | popularity | random | precision@3 model | popularity | random |
|---|---|---|---|---|---|---|---|
| 1 | 106 | 0.231 | 0.138 | 0.059 | 0.245 | 0.164 | 0.059 |
| 2 | 104 | 0.263 | 0.125 | 0.051 | 0.285 | 0.138 | 0.051 |
| 3 | 100 | 0.292 | 0.110 | 0.043 | 0.327 | 0.123 | 0.043 |
| 4 | 87 | 0.259 | 0.089 | 0.035 | 0.249 | 0.107 | 0.035 |
| 5 | 60 | 0.244 | 0.094 | 0.026 | 0.244 | 0.094 | 0.026 |

## lodo: test split only (20 games)

| k | n | recall@(8-k) model | popularity | random | precision@3 model | popularity | random |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 0.318 | 0.129 | 0.059 | 0.367 | 0.150 | 0.056 |
| 2 | 20 | 0.342 | 0.117 | 0.051 | 0.383 | 0.133 | 0.048 |
| 3 | 19 | 0.295 | 0.100 | 0.043 | 0.368 | 0.088 | 0.040 |
| 4 | 13 | 0.231 | 0.064 | 0.035 | 0.205 | 0.077 | 0.032 |
| 5 | 7 | 0.095 | 0.095 | 0.026 | 0.095 | 0.095 | 0.025 |

## Example predictions (model trained on all decks)

- seen ['giant', 'balloon']; likely archetype beatdown (0.43): bowler 0.27, miner 0.21, tornado 0.17, zap 0.16, barbarians 0.16, minions 0.16, graveyard 0.14, skeletons 0.14
- seen ['bats', 'night-witch']; likely archetype beatdown (0.39): arrows 0.42, miner 0.35, the-log 0.28, spear-goblins 0.24, elite-barbarians 0.20, barbarian-barrel 0.19, goblins 0.19, mortar 0.19
