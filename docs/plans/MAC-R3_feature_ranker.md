# MAC-R3 — feature-interaction ranker baselines on the 4-way task (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Course: RAG FM/DLRM/DCN-V2 [56–58]; "tuned simple baselines" [5–7]; pipeline review §6 baseline (3). Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
tools/eval/feature_ranker.py     (new)
tests/test_feature_ranker.py     (new)
```
LightGBM 4.7 and torch are installed. CPU is enough.

## 0. Leakage rules (read twice — WP4 Amendment C is why)
1. **Pointwise features only.** Each option's features are computed from the state and that option alone. Never use anything that looks at the other options of the same example (counts of the option's card/lane/zone among the options, option index, etc.). Opus found that the option set alone gives the positive away ≈85% of the time (the WP4 negatives N2 = same card/other zone and N3 = other card/same lane+zone make the positive the "intersection" option). A pointwise scorer cannot exploit that; keep it that way.
2. Never use the ranker-file fields `source`, `label` (except as the target), `positive_key`, `pos_in_legal`.
3. No legality feature (`legal_action_keys`): positives are perceived-legal only 55% of the time (D-019), so legality is a perception artefact here; runtime candidates are all legal anyway.
4. Train-split statistics only, and **leave-one-game-out** for train examples: popularity/zone counts used as features for a train example exclude that example's own game (same value for every option of the example, so no label leak).
5. No kNN-vote feature (the N1 negatives are mined from kNN neighbours, see the NOTE in `tools/eval/eval_ranker_baselines.py`).

## 1. Data
`<dataset>/ranker/{train,val,test}.jsonl` (lines: `id, game_id, split, kind, context, options[{key, text, source}], label, positive_key, pos_in_legal`) and `<dataset>/states.jsonl` (by `id`: `state`, `t`, `game_id`, `action`). `cards = load_cards()`, `vocab = clashbot.features.vocab()`.

## 2. Features per (example, option key `card|lane|zone`)
- **state block** (644-d): `featurize(state, t)` (same vector as retrieval).
- **option block**: card one-hot over `vocab` (`len(vocab)`), lane one-hot (3), zone one-hot (4), `cost / 10`, card-type one-hot (`Troop`, `Spell`, `Building`, other), `in_hand` (card in `state.own.hand`), `elixir_after = ((own elixir or 0) - cost) / 10`, `p_zone_given_card` (train zone counts of that card, leave-one-game-out, add-one smoothing over the 4 zones), `log1p(pop_key)`, `log1p(pop_card)`, `log1p(pop_key_in_phase)` (train play counts, leave-one-game-out), `kg_counter` = max over current threats of `KnowledgeGraph.counter_scores(threat).get(card, 0)` where threats = opponent `recent_plays` within 5 s and enemy units with `eta_tower` to an own tower ≤ 8 s (copy the rule of `CandidateGenerator._threats`; needs `db/`; if the DB is missing use 0.0 and say so in the output).
Put feature building in `build_features(example, record, stats, kg) -> np.ndarray` (shape `(4, F)`) and `train_stats(records, exclude_game=None)`.

## 3. Models (equal small tuning budget, selection on val accuracy)
1. `logreg`: torch linear scorer on the concatenated features; listwise softmax cross-entropy over the 4 options; Adam, `lr ∈ {1e-2, 1e-3}`, weight decay 1e-4, 40 epochs, batch 128 examples.
2. `lgbm_lambdarank`: `lightgbm.LGBMRanker(objective="lambdarank", n_estimators ∈ {200, 500}, learning_rate=0.05, num_leaves ∈ {15, 31}, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)`, `group = [4] * n_examples`, label 1 for the positive option.
3. `dcn_v2`: torch DCN-V2-style scorer: `x0` = features (standardised with train mean/std); 3 cross layers `x_{l+1} = x0 * (W_l x_l + b_l) + x_l` (full-rank `W_l`); a parallel deep MLP `[256, 128]` with ReLU and dropout 0.2; concat cross output and deep output → `Linear → 1` score; listwise softmax cross-entropy over the 4 options; Adam `lr ∈ {1e-3, 3e-4}`, weight decay 1e-5, up to 40 epochs, early stopping on val accuracy (patience 5), batch 128 examples.
Each model: choose the config on val; then seeds `{0, 1, 2}` for the chosen config; report val and test accuracy mean ± std, plus test accuracy by `kind`. Torch: `torch.manual_seed(seed)`, CPU.
4. Ablation for `lgbm_lambdarank` (chosen config, seed 0): drop the popularity features; drop the state block; drop `kg_counter`.
5. Reference rows (copied, not recomputed): `random_expected`, `placement_prior`, `popularity` from `<dataset>/ranker/baselines.json`, plus `option_set_shortcut` computed here (for each example: `score(k) = #options with k's card + #options with k's (lane, zone)`; expected accuracy with random tie-break) — clearly labelled "not a valid model, shows the option-set leak".
6. LightGBM top-20 feature importances (gain) with feature names (`state:<feature_names()>`, `opt:card:<slug>`, …).

## 4. CLI and output
```
$PY tools/eval/feature_ranker.py [--dataset DIR] [--models logreg,lgbm_lambdarank,dcn_v2] [--seeds 0,1,2]
        [--limit N] [--quick] [--out PATH]
```
`--quick`: one config per model, one seed, 5 epochs (tests/smoke). `--limit N`: first N examples per split. Output JSON (`--out`, default `<dataset>/eval/feature_ranker.json`): `{"n": {split: n}, "features": F, "models": {name: {"grid": [...], "best": {...}, "val_acc_mean", "val_acc_std", "test_acc_mean", "test_acc_std", "test_by_kind"}}, "ablation": {...}, "reference": {...}, "importance": [...], "kg_counter_available": bool}` and `<out>.md` with `| model | val acc | test acc |` (mean ± std) including the reference rows; print it.

## 5. Tests — `tests/test_feature_ranker.py`
Synthetic examples in the test (3 cards from the real card table, hand-made states via a `_state()` helper like `tests/test_runtime.py`).
1. `build_features` returns shape `(4, F)` with a fixed `F`; changing the *other* options of an example does not change an option's feature row (pointwise guarantee).
2. Leave-one-game-out: `train_stats(records, exclude_game=g)` counts exclude game `g`; for a train example the popularity features of all 4 options are computed with its game excluded.
3. Forbidden fields: features are identical when `source`, `positive_key` and `pos_in_legal` are changed or deleted.
4. `option_set_shortcut` = 1.0 on the intersection example and 0.25 on an all-distinct example.
5. Listwise loss: a scorer that puts a large score on the label gives loss < 0.01; uniform scores give `log(4)`.
6. `--quick` end-to-end on a synthetic dataset directory in `tmp_path` (ranker files with 40/10/10 examples, matching `states.jsonl`, a `baselines.json`; `kg_counter` falls back to 0): `main([...])` writes the JSON with all three models.
7. Real data (skip if `dataset/v1/ranker/val.jsonl` missing): `main(["--quick", "--limit", "200", "--out", str(tmp_path / "f.json")])` finishes; accuracies in [0, 1].

## 6. Smoke (allowed once, after the symlinks)
`$PY tools/eval/feature_ranker.py --quick --out runs/smoke/R3/f.json` — paste the table.

## 7. Heavy run (Opus, at integration)
Full grid + seeds → `docs/data/feature_ranker_v1.{json,md}`.

## 8. Report
Per MAC-00 §5.
