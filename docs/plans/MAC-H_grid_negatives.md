# MAC-H — 2×2 card × placement negatives for the 4-way task; tests build into temp dirs (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Implements D-037 (decided by the user: 2×2 grid). Read `docs/plans/MAC-00_common.md` first.

## Why
With the WP4 negatives (N2 = same card/other zone, N3 = other card/same lane+zone, N1 = neighbour keys) the positive is the "intersection" option: picking the option that shares its card with another option and its lane+zone with another scores **0.85** with no state at all. In a 2×2 grid every option shares its card with exactly one other option and its placement with exactly one other, so this rule is at chance (0.25) by construction, while the negatives stay hard.

## Owns (only these files)
```
tools/data/build_ranker_dataset.py     (replace the negative selection; keep the act export and placement prior unchanged)
tools/eval/eval_ranker_baselines.py    (add the option_set_shortcut baseline)
tests/test_ranker_dataset.py           (update)
tests/test_d1_format.py                (only its integration test/fixture, to use the temp build)
tests/conftest.py                      (new: shared temp-build fixture)
```

## 1. Grid construction (`tools/data/build_ranker_dataset.py`)
Delete `select_negatives`, `build_pools` and `TARGET_NEGATIVES`; add:
```python
GRID_SOURCES = ("G_card", "G_place", "G_diag")   # same card/other placement, other card/same placement, both differ

def placement_candidates(c1: str, l1: str, z1: str, zone_freq: dict, variant: int) -> list[tuple[str, str]]
def grid_options(rec: dict, pos: str, possible_keys: list[str], neighbour_keys: list[str], cards: dict,
                 zone_freq: dict, seed: int) -> list[tuple[str, str]] | None
```
- `placement_candidates`: three ordered blocks, all `(lane, zone) != (l1, z1)`, no duplicates:
  - `Z` = `[(l1, z) for z in sorted(ZONES, key=lambda z: (-zone_freq.get(c1, {}).get(z, 0), z)) if z != z1]` (same lane, other zone; most frequent zone for that card first)
  - `L` = `[(l, z1) for l in LANES if l != l1]` (other lane, same zone)
  - `B` = `[(l, z) for l in LANES for z in ZONES if l != l1 and z != z1]` (both differ)
  - `variant = seed % 3`: 0 → `Z + L + B`, 1 → `L + Z + B`, 2 → `B + Z + L`. (Rotating the order keeps lane and zone both varying across the training set; otherwise every option of most examples would share the pro's lane.)
- `grid_options`: `c1, l1, z1 = pos.split("|")`; `possible = set(possible_keys)`;
  - second-card candidates in order: the cards of `neighbour_keys` (in order, first occurrence, skip `"hold"`), then `possible_cards(rec["state"], cards)` order; skip `c1` and repeats; keep `c2` only if `action_key(c2, l1, z1) in possible`.
  - for each `c2` in that order, for each `(l2, z2)` in `placement_candidates(c1, l1, z1, zone_freq, seed % 3)`: if `action_key(c1, l2, z2)` and `action_key(c2, l2, z2)` are both in `possible` → return `[("G_card", f"{c1}|{l2}|{z2}"), ("G_place", f"{c2}|{l1}|{z1}"), ("G_diag", f"{c2}|{l2}|{z2}")]`.
  - nothing found → `None`.
- In `build`: replace the `build_pools`/`select_negatives` part with `seed = int(hashlib.sha1(rec["id"].encode()).hexdigest()[:8], 16)` (same seed as now); `negatives = grid_options(rec, pos, possible, neighbour_keys, cards, zone_freq, seed)`; `None` → count `skipped_no_grid` and `continue`; `options = [("pos", pos)] + negatives`; shuffle with `random.Random(seed)` as now. Every example therefore has exactly 4 options.
- Stats per split: rename `skipped_no_negatives` → `skipped_no_grid`; keep the other fields; `negative_source_counts` now counts `G_card/G_place/G_diag`; add `placement_variant_counts` (`{"Z": n, "L": n, "B": n}` = which block the chosen `(l2, z2)` came from) and `option_set_shortcut` (expected accuracy of the rule in §2 on that split — must be 0.25). `print_stats` header/rows follow the renamed fields.
- Update the module docstring: "plays-only 4-way options in a 2×2 card × placement grid (D-037)".
- The act export, the placement prior and `top_neighbours`/`feature_matrix`/`train_row_indices`/`load_records`/`build_placement_prior` stay exactly as they are (other tools import them).

## 2. `tools/eval/eval_ranker_baselines.py`
Add `option_set_shortcut` to `RANKER_BASELINES` and compute it per example: `score(k) = #options with k's card + #options with k's (lane, zone)` (each count includes `k`); value = `1/|argmax set|` if the label is in the argmax set else 0 (expected accuracy under random tie-break) — record it via the existing `hits[...]` mechanism as a float, like `random_expected`. Nothing else changes.

## 3. Tests build into temp dirs (`tests/conftest.py`)
The current fixtures write into the real `dataset/v1/` (the `built` fixture in `tests/test_ranker_dataset.py` and the integration test in `tests/test_d1_format.py`). Replace that with one session fixture:
```python
@pytest.fixture(scope="session")
def built_dataset(tmp_path_factory):
    """Symlink the real states/games into a temp dir and build ranker, act, baselines and d1 there."""
```
- Skip (`pytest.skip`) if `ROOT/"dataset"/"v1"/"states.jsonl"` is missing.
- `d = tmp_path_factory.mktemp("ds")`; symlink `states.jsonl` and `games.jsonl` from the real dataset into `d`; run (subprocess, `sys.executable`, `check=True`, `capture_output=True`): `tools/data/build_ranker_dataset.py --dataset d`, `tools/eval/eval_ranker_baselines.py --dataset d`, `tools/data/build_d1_dataset.py --dataset d` (add `--tokenizer db/hf/d1_tokenizer.json` only if that file exists). Return `d`.
- `tests/test_ranker_dataset.py`: `built` uses `built_dataset` and reads from it; `test_baselines` reads `built_dataset/"ranker"/"baselines.json"` (no extra subprocess). `tests/test_d1_format.py`: its integration test reads the d1 files and stats from `built_dataset` instead of building into `dataset/v1`.
- After this change, `grep -n "DATASET /" tests/test_ranker_dataset.py tests/test_d1_format.py` shows reads only (no builder writes into the real dataset).

## 4. Tests (update `tests/test_ranker_dataset.py`)
Remove the `select_negatives` tests; add:
1. `placement_candidates("knight", "left", "mid", {"knight": {"bridge": 5, "back": 1}}, 0)` starts with `("left", "bridge")`, then `("left", "back")`, `("left", "enemy_half")`, then other-lane blocks; variant 1 starts with `("middle", "mid")`; variant 2 starts with a both-differ pair; no candidate equals `("left", "mid")`; no duplicates.
2. `grid_options` on a hand-made state (deck of 8 real cards, hand of 4, elixir 10): returns 3 negatives; the 4 options form a grid (2 cards × 2 placements, every combination once); a neighbour card is preferred as `c2` over the first possible card; returns `None` when only one card is possible.
3. Shortcut neutrality on that grid: the option-set rule gives every option the same score (4).
4. Built data (fixture): every example has exactly 4 options, one `pos`, keys unique and all in `possible_action_keys`; the 4 keys use exactly 2 cards and exactly 2 `(lane, zone)` placements, each combination once; `stats[split]["option_set_shortcut"] == 0.25` for every split; no `next_card`-blocked card offered (keep the existing next-card tests); train/val/test games disjoint.
5. Baselines: `res["ranker"][split]` has the 5 old names plus `option_set_shortcut`, whose `acc` is 0.25 ± 1e-9.
6. Keep the act-file and next-card tests.

## 5. Smoke (allowed once)
`$PY -m pytest -q tests/test_ranker_dataset.py tests/test_d1_format.py` with the dataset symlink from MAC-00 §1 (the fixture builds in a temp dir — nothing is written into `dataset/v1/`). Paste the stats table printed by the builder (run it once by hand into a temp dir: `$PY tools/data/build_ranker_dataset.py --dataset <tmp with symlinked states.jsonl>`).

## 6. Heavy run (Opus, at integration, after Wave 1)
Rebuild `dataset/v1/ranker`, `act`, `placement_prior.json`, baselines and `d1/` in the main checkout; re-run the d1 zero-shot and R3 evaluations on the new files; copy reviewed numbers to `docs/data/`.

## 7. Report
Per MAC-00 §5, with the per-split counts (`examples`, `skipped_no_grid`, `skipped_pos_impossible`), `placement_variant_counts`, and the baseline table.

---
## Amendment 1 (Opus review, 2026-11-02) — FIX_LIST #7, #10, #11
Owns additionally: `tests/test_kg_documents.py`.
1. (#7) `tests/conftest.py`: `def pytest_configure(config): config.addinivalue_line("markers", "slow: real-model tests, run only with CLASHBOT_SLOW_TESTS=1")`.
2. (#10) `tests/test_kg_documents.py` must not write into `dataset/v1`. Read `tools/data/build_kg.py` and `tools/data/build_documents.py`: both take `--dataset DIR`. Add a session fixture `built_kg(tmp_path_factory)` in `tests/conftest.py` (skip if the real `states.jsonl` is missing) that symlinks `states.jsonl` and `games.jsonl` (and anything else these builders read from the dataset dir — check their code) into a temp dir and runs both builders there; the tests read from it. Afterwards `grep -n "build_kg\|build_documents" tests/*.py` shows only temp-dir invocations.
3. (#11) `tools/eval/eval_ranker_baselines.py` `NOTE`: `"knn_vote is not reported on the 4-way files: the second card of each 2x2 grid is mined from kNN neighbours. It is evaluated fairly in candidate_recall over all possible action keys."`
Acceptance: `$PY -m pytest -q tests` (whole suite, with the symlinks) leaves the mtimes of every file under the real `dataset/v1/` unchanged — check with `find dataset/v1 -newer <marker file> -type f` printing nothing.
