# Cleaning report

## Totals

- games: 197
- samples: 10923
- samples by kind: key=3539, play=6901, periodic=483
- games by opponent_deck_source: match_file=169, match_file_capped=28

Train-eligible by split x kind:

| split | key | play | periodic | total |
|---|---|---|---|---|
| train | 2024 | 2367 | 298 | 4689 |
| val | 351 | 591 | 47 | 989 |
| test | 375 | 547 | 58 | 980 |

## Flag counts

| flag | samples | blocking |
|---|---|---|
| deck_invalid | 447 | yes |
| deck_incomplete | 370 | no |
| action_unknown | 1668 | yes |
| action_card_unknown | 741 | yes |
| placement_unknown | 2165 | yes |
| action_card_off_deck | 165 | yes |
| phase_unknown | 2017 | no |
| hand_unknown | 623 | no |
| units_side_repaired | 1916 | no |
| units_unverified | 3897 | no |
| opp_deck_capped | 1732 | no |

## Deck status per game (non-valid)

- 9tx2iYsSmMw-m0.0: incomplete (own_deck length 7)
- 9tx2iYsSmMw-m0.4: incomplete (own_deck length 7)
- 9tx2iYsSmMw-m0.7: incomplete (own_deck length 7)
- CInrqMTlVkg-m0.0: incomplete (own_deck length 7)
- CInrqMTlVkg-m0.1: incomplete (own_deck length 7)
- CInrqMTlVkg-m0.2: invalid (own_deck length 5)
- CInrqMTlVkg-m0.3: incomplete (own_deck length 7)
- CdlYCOppdXQ-m0: invalid (own_deck length 9)
- Z8-4VhLjrGU-m0.2: invalid (own_deck length 6)
- vp7jP_vjxcU-m0.0: invalid (own_deck length 29)
- vp7jP_vjxcU-m0.1: invalid (own_deck length 35)
- yOGPFqpzNu4-m2: incomplete (own_deck length 7)
- yOGPFqpzNu4-m3: incomplete (own_deck length 7)
- yOGPFqpzNu4-m4: invalid (own_deck length 9)

## Opponent decks capped

- 5bU83eIW8Yg-m0.3: dropped barbarian-barrel
- 5nOaTBlYLlg-m0.1: dropped archers, arrows, barbarian-barrel, berserker, bomber, elite-barbarians, rocket, rune-giant, the-log, vines, wizard
- 6qYQNJ1Uaeg-m0.3: dropped bats
- 7R5FP2IYWsw-m0.5: dropped berserker
- 9tx2iYsSmMw-m0.1: dropped fire-spirit
- 9tx2iYsSmMw-m0.9: dropped fireball
- A5qMlVwLS1M-m0.1: dropped the-log
- B7SW-i94v8s-m0.0: dropped fireball
- B7SW-i94v8s-m0.1: dropped tornado
- CdlYCOppdXQ-m0: dropped miner, royal-delivery
- O97T50_dNGc-m1.0: dropped tombstone
- OKA_QNEkJIU-m0.1: dropped arrows, skeleton-king
- PuJls1qTZsU-m0.1: dropped bowler
- PuJls1qTZsU-m0.4: dropped fireball
- X2zRpx5TN2U-m0.4: dropped elite-barbarians, fireball, mega-minion
- X2zRpx5TN2U-m0.7: dropped goblinstein
- eEwTOkVPoFM-m0.2: dropped magic-archer
- eEwTOkVPoFM-m0.5: dropped freeze, tornado
- eEwTOkVPoFM-m0.6: dropped cannon-cart, goblins
- kiN_D3bRu34-m0.4: dropped wall-breakers
- klMJBAH2Zx8-m0.0: dropped arrows
- klMJBAH2Zx8-m0.2: dropped arrows, lightning
- nSXIs16M7Ag-m0.1: dropped little-prince, tombstone
- qHWw4rM-N2I-m0.1: dropped fireball
- qHWw4rM-N2I-m0.3: dropped barbarian-barrel
- vp7jP_vjxcU-m0.0: dropped clone, electro-wizard, firecracker, giant, lightning, mini-p-e-k-k-a, mortar, p-e-k-k-a, rocket, royal-ghost, valkyrie, wall-breakers, zap, zappies
- vp7jP_vjxcU-m0.1: dropped bandit, elite-barbarians, fireball, giant-skeleton, goblin-cage, goblin-gang, hog-rider, inferno-dragon, mega-minion, minions, musketeer, phoenix, skeleton-king, skeletons, tesla, the-log, three-musketeers, valkyrie
- ynafcLtQWDQ-m0.3: dropped cannon-cart

## Side repairs

- total flipped units: 2371

Top 20 unit classes flipped:

| unit_class | count |
|---|---|
| bomb-tower | 300 |
| skeleton | 285 |
| zappy | 207 |
| skeleton-evolution | 107 |
| skeleton-dragon | 93 |
| spear-goblin | 83 |
| bomb | 70 |
| hog-rider | 60 |
| minion | 58 |
| inferno-tower | 56 |
| bat-evolution | 55 |
| barbarian | 50 |
| the-log | 45 |
| bat | 44 |
| bomber-evolution | 41 |
| lava-pup | 35 |
| ice-spirit | 32 |
| barbarian-barrel | 31 |
| tesla-evolution | 29 |
| knight | 27 |

## Unresolved unit classes

| unit_class | count |
|---|---|
| unknown_unit | 5097 |
| hog | 41 |
| goblin-ball | 12 |
| royal-guardian | 11 |

## Action keys

Top 30 keys among train-eligible play samples:

| key | count |
|---|---|
| goblin-barrel\|left\|enemy_half | 173 |
| cannon\|middle\|mid | 157 |
| bomb-tower\|middle\|mid | 131 |
| goblin-barrel\|right\|enemy_half | 129 |
| skeletons\|middle\|bridge | 119 |
| skeletons\|middle\|mid | 107 |
| wall-breakers\|right\|bridge | 98 |
| berserker\|left\|bridge | 93 |
| wall-breakers\|left\|bridge | 92 |
| berserker\|right\|bridge | 86 |
| the-log\|left\|bridge | 82 |
| dart-goblin\|middle\|mid | 78 |
| ice-spirit\|middle\|bridge | 70 |
| electro-spirit\|middle\|mid | 68 |
| magic-archer\|middle\|mid | 63 |
| skeletons\|left\|bridge | 63 |
| skeletons\|right\|bridge | 60 |
| berserker\|middle\|mid | 57 |
| ice-spirit\|left\|bridge | 56 |
| electro-spirit\|middle\|bridge | 55 |
| the-log\|right\|bridge | 52 |
| berserker\|middle\|bridge | 51 |
| furnace\|middle\|mid | 50 |
| tesla\|middle\|mid | 50 |
| fire-spirit\|left\|bridge | 49 |
| fire-spirit\|right\|bridge | 49 |
| ice-spirit\|right\|bridge | 49 |
| skeletons\|right\|mid | 49 |
| cannon\|middle\|bridge | 47 |
| berserker\|left\|mid | 46 |

- hold (train-eligible): 645
