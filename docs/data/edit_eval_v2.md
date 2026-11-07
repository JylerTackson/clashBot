SYNTHETIC balance change (D-034); no real patch is involved: +1 elixir on each of 5 cards (skeletons 1 -> 2, berserker 2 -> 3, goblin-barrel 3 -> 4, electro-spirit 1 -> 2, cannon 3 -> 4) in patch P1; one fresh MemoryStore per edit; non-parametric route (versioned external fact store). Pooled = sum of passes / sum of n.

| criterion | n | edited | no edit |
|---|---|---|---|
| reliability | 15 | 1.000 | 0.000 |
| generalization (in hand 152, hand-or-possible 465) | 465 | 1.000 | 0.000 |
| locality_cards | 2500 | 1.000 | 1.000 |
| locality_states | 1500 | 1.000 | 1.000 |
| portability (in hand 152, hand-or-possible 465) | 465 | 1.000 | 0.067 |
| conflict: share of top-10 hits using the edited card (no edit = before, edited = after re-weighting) | 1245 | 0.002 | 0.124 |
| versioning (value as of base is the old one; all edits) | 5 | True | - |
| rollback (retract restores the base table; all edits) | 5 | True | - |

Per card (n / edited score):

| card | edit | reliability | generalization | locality_cards | locality_states | portability | conflict | versioning | rollback |
|---|---|---|---|---|---|---|---|---|---|
| skeletons | 1 -> 2 | 3 / 1.000 | 26 (8 in hand) / 1.000 | 500 / 1.000 | 300 / 1.000 | 26 (8 in hand) / 1.000 | 300 / 0.002 | True | True |
| berserker | 2 -> 3 | 3 / 1.000 | 259 (60 in hand) / 1.000 | 500 / 1.000 | 300 / 1.000 | 259 (60 in hand) / 1.000 | 300 / 0.000 | True | True |
| goblin-barrel | 3 -> 4 | 3 / 1.000 | 128 (58 in hand) / 1.000 | 500 / 1.000 | 300 / 1.000 | 128 (58 in hand) / 1.000 | 300 / 0.005 | True | True |
| electro-spirit | 1 -> 2 | 3 / 1.000 | 26 (8 in hand) / 1.000 | 500 / 1.000 | 300 / 1.000 | 26 (8 in hand) / 1.000 | 193 / 0.001 | True | True |
| cannon | 3 -> 4 | 3 / 1.000 | 26 (18 in hand) / 1.000 | 500 / 1.000 | 300 / 1.000 | 26 (18 in hand) / 1.000 | 152 / 0.001 | True | True |
