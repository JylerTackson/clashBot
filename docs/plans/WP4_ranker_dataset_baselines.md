# WP4 — Ranker training set, placement prior and baseline evaluation (spec for a Sonnet implementer)

Author: Opus · 2026-10-28 · Implements D-004 (Laya / d1 fixed-answer decision models), D-006 (video splits), D-007 (candidates → ranker), D-010 (card × lane × zone) and the RecSys rows of `docs/research/course_mapping.md` §2 (hard negatives, RAFT-style distractors, "tuned simple baselines").
Depends on WP1 (`dataset/v1/states.jsonl`, `clashbot.game_rules`, `clashbot.config`) and WP3 (`clashbot.features.featurize`). It does **not** need the SQLite database.

**You are implementing, not designing.** If anything is unclear or contradicts the data, stop and report.

## 0. Ground rules
- Python 3.11, stdlib + numpy. pytest for tests. No commits. Do not modify files from WP1–WP3 or `knowledge_base/`.
- Scripts start with `ROOT = Path(__file__).resolve().parents[2]` and `sys.path.insert(0, str(ROOT))`.

## 1. Files to create
```
clashbot/actions.py                      option text + placement prior lookup (used later by the live loop)
tools/data/build_ranker_dataset.py       writes dataset/v1/ranker/{train,val,test}.jsonl, stats.json and dataset/v1/placement_prior.json
tools/eval/__init__.py                   (empty)
tools/eval/eval_ranker_baselines.py      writes dataset/v1/ranker/baselines.json and prints a table
tests/test_ranker_dataset.py
```

## 2. `clashbot/actions.py`
```python
ZONE_TEXT = {"back": "back, behind the king tower", "mid": "mid, on our side",
             "bridge": "at the bridge", "enemy_half": "on the enemy side"}

def option_text(key: str) -> str
```
`"hold"` → `"hold elixir and play nothing yet"`; play key `card|lane|zone` → `f"play {card} in the {lane} lane, {ZONE_TEXT[zone]}"`.

```python
def load_placement_prior(path: Path = DATASET_DIR / "placement_prior.json") -> dict
def placement_tile(key: str, prior: dict) -> list[float] | None
```
`placement_tile` returns `prior["by_key"][key]["median_tile"]` if present, else `prior["by_lane_zone"][f"{lane}|{zone}"]["median_tile"]` if present, else None (`"hold"` → None).

## 3. `tools/data/build_ranker_dataset.py`
CLI: `python3 tools/data/build_ranker_dataset.py [--dataset DIR] [--k 20]`.

### 3.1 Load
All records from `<dataset>/states.jsonl`. `cards = load_cards()`. Feature matrix `F = featurize(r["state"], r["t"])` for every record (float32, rows in file order).
`train_rows` = indices with `split == "train"` and `action.key` not null.

### 3.2 Placement prior → `<dataset>/placement_prior.json`
From records with `split == "train"`, `train_eligible`, `action.type == "play"`:
- `by_key[key]` = `{"count": n, "median_tile": [median of tile[0], median of tile[1]] (statistics.median, rounded to 1 decimal)}`.
- `by_lane_zone["lane|zone"]` = same over all cards.
- `zone_freq[card]` = `{zone: count}` over those records.
File: `{"source": "train split, train_eligible plays", "by_key": ..., "by_lane_zone": ..., "zone_freq": ...}` with keys sorted.

### 3.3 Examples — for every record with `train_eligible == True` (all splits)
`pos = action.key`. `legal = legal_action_keys(state, cards)`.

Candidate pools (each a list in priority order, no duplicates, never containing `pos`):
- **N1 neighbours**: `scores = F[train_rows] @ F[i]`; drop rows whose `game_id` equals this record's game; take the top `k` by score (ties: lower row index first); in that order, each neighbour's `action.key` if it is in `legal` and not already in N1.
- **N2 same card, other zone** (only if pos is a play): zones of `card` ordered by `(-zone_freq[card].get(z, 0), z)` over `ZONES`; first `z != pos_zone` with `z in legal_zones(card, state, cards)` → `action_key(card, pos_lane, z)`. At most one item.
- **N3 other card, same spot** (only if pos is a play): first `c` in `legal_cards(state, cards)` with `c != card` and `pos_zone in legal_zones(c, state, cards)` → `action_key(c, pos_lane, pos_zone)`. At most one item.
- **N4 hold**: `["hold"]` if `pos != "hold"`.
- **N5 legal fallback**: `legal` in order, excluding `"hold"`.

Selection (target 3 negatives, stop when 3 are chosen, skip duplicates):
1. first item of N1;
2. first item of N2;
3. first item of N4;
4. then remaining N1 items in order;
5. then N3;
6. then N5 items in order.
Record the pool name (`"N1"`..`"N5"`) of each chosen negative. If no negative was found, skip the record and count it as `skipped_no_negatives`.

Options = `[pos] + negatives`; shuffle with `random.Random(int(hashlib.sha1(id.encode()).hexdigest()[:8], 16)).shuffle(options_with_sources)`; `label` = index of pos after shuffling.

Output line (`<dataset>/ranker/<split>.jsonl`, records in states.jsonl order):
```
{"id", "game_id", "split", "kind", "context": record["context_text"],
 "options": [{"key", "text": option_text(key), "source": "pos"|"N1".."N5"}, ...],
 "label": int, "positive_key": pos, "pos_in_legal": pos in legal}
```

### 3.4 `stats.json`
Per split: `examples`, `skipped_no_negatives`, `mean_options` (3 decimals), `label_position_counts` (`{"0": n, ...}`), `negative_source_counts`, `pos_in_legal_rate` (3 decimals), `kind_counts`, `hold_positive_count`. Print the same as a short table.

## 4. `tools/eval/eval_ranker_baselines.py`
CLI: `python3 tools/eval/eval_ranker_baselines.py [--dataset DIR] [--k 20]`. Evaluates on `val` and `test` ranker files. Every baseline picks one option per example; accuracy = share of examples where the pick == label. Tie-break everywhere: lowest option index.

| baseline | score of an option |
|---|---|
| `random_expected` | not a pick: report the mean of `1 / len(options)` |
| `popularity` | count of the option key among train-split `train_eligible` records' `action.key` |
| `popularity_by_phase` | same count restricted to train records with the same `state.phase` as the example (None matches None) |
| `knn_vote` | with `F` and `train_rows` as in §3.1 (needs the example's state: look it up by `id` in states.jsonl), take the top `k` train neighbours from other games; score = sum of neighbour similarity where the neighbour's `action.key` equals the option key |
| `knn_vote_card` | same as knn_vote but match on the card only (`hold` matches `hold`) |

Also report, per split, `n` and the accuracy of each baseline split by `kind`.
Write `<dataset>/ranker/baselines.json` = `{split: {baseline: {"acc": x, "n": n, "by_kind": {...}}}}` (accuracies rounded to 4) and print a markdown table: rows = baselines, columns = val acc, test acc.

## 5. Tests — `tests/test_ranker_dataset.py`
1. `option_text` for a hold and a play key.
2. `placement_tile` falls back from `by_key` to `by_lane_zone`, returns None for hold.
3. Selection order: unit-test a helper `select_negatives(pools: dict[str, list[str]], pos: str) -> list[tuple[str, str]]` (implement it as a module-level function in the build script, exactly per §3.3) on hand-made pools, covering: duplicates across pools, fewer than 3 available, pos == "hold".
4. Integration (skip if `dataset/v1/states.jsonl` is missing): run the build via subprocess on the real dataset; every example has 2–4 options, exactly one with `source == "pos"`, `options[label].key == positive_key`, no `"action:"` in `context`; no val/test example has the same `game_id` as any train example; label positions are not all equal.
5. Integration: run the baselines script; `baselines.json` has val and test with all five baselines; every acc is in [0, 1].

## 6. Acceptance (run, paste outputs)
```
python3 -m pytest -q tests/test_ranker_dataset.py
python3 tools/data/build_ranker_dataset.py
cat dataset/v1/ranker/stats.json
python3 tools/eval/eval_ranker_baselines.py
head -c 1200 dataset/v1/ranker/val.jsonl; echo
git status --short
```

## 7. Report back (≤ 40 lines)
Files created; tests; stats per split; the baseline table; anything you could not follow exactly and why.
