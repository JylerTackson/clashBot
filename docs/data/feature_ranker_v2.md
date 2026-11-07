Feature-interaction rankers on the 4-way task (n val 761, n test 766, 783 features; mean ± std over seeds)

| model | val acc | test acc |
|---|---|---|
| random_expected (reference) | 0.2500 | 0.2500 |
| placement_prior (reference) | 0.4928 | 0.3838 |
| popularity (reference) | 0.4678 | 0.3251 |
| option_set_shortcut (NOT A VALID MODEL: shows the option-set leak) | 0.2500 | 0.2500 |
| logreg | 0.5458 ± 0.0034 | 0.4761 ± 0.0043 |
| lgbm_lambdarank | 0.5484 ± 0.0140 | 0.4913 ± 0.0126 |
| dcn_v2 | 0.5261 ± 0.0134 | 0.4121 ± 0.0132 |

Test accuracy by kind

| model | key | periodic | play |
|---|---|---|---|
| logreg | 0.5273 ± 0.0057 (n=299) | 0.4545 ± 0.0000 (n=11) | 0.4430 ± 0.0054 (n=456) |
| lgbm_lambdarank | 0.5217 ± 0.0224 (n=299) | 0.6667 ± 0.0429 (n=11) | 0.4671 ± 0.0090 (n=456) |
| dcn_v2 | 0.4437 ± 0.0150 (n=299) | 0.3939 ± 0.1545 (n=11) | 0.3918 ± 0.0126 (n=456) |

LightGBM ablation (chosen config, seed 0)

| variant | features | val acc | test acc |
|---|---|---|---|
| full | 783 | 0.5650 | 0.4817 |
| no_kg_counter | 782 | 0.5335 | 0.4804 |
| no_popularity | 780 | 0.5506 | 0.5039 |
| no_state_block | 139 | 0.5388 | 0.4817 |
| no_train_stats | 779 | 0.5388 | 0.5209 |

LightGBM top-20 features by gain (chosen config, seed 0)

| feature | gain share |
|---|---|
| opt:log_pop_key | 0.1782 |
| opt:p_zone_given_card | 0.1240 |
| opt:log_pop_card | 0.0873 |
| opt:log_pop_key_phase | 0.0507 |
| state:scalars:own_elixir | 0.0248 |
| state:towers:enemy_right | 0.0229 |
| opt:elixir_after | 0.0223 |
| state:towers:own_left | 0.0215 |
| state:towers:own_king | 0.0179 |
| state:scalars:match_progress | 0.0178 |
| state:towers:enemy_left | 0.0168 |
| state:towers:own_right | 0.0157 |
| state:scalars:opp_elixir | 0.0152 |
| state:towers:enemy_king | 0.0131 |
| opt:lane:middle | 0.0115 |
| opt:in_hand | 0.0110 |
| opt:lane:right | 0.0097 |
| opt:cost | 0.0093 |
| opt:zone:mid | 0.0084 |
| opt:lane:left | 0.0082 |

Config selection and DCN-V2 early stopping use val accuracy, so the val column is optimistic; test is untouched.
