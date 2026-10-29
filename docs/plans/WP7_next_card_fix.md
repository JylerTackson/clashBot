# WP7 — Exclude the "next" card from possible/legal cards (spec for a Sonnet implementer)

Author: Opus · 2026-10-28 · Approved by the user ("yes, fix the next card issue"). Implements D-024.

The card shown in the "next" slot cannot be played until another card is played. `possible_cards` currently adds every deck card not in the known hand when a hand slot is unread, which wrongly includes the next card (example: Wall Breakers offered as a factorized `card` option while shown as next). Audit (Opus): 2,138 of 6,013 eligible plays have a next card outside the known hand; in 263 of them the pro's play equals that next card (a perception contradiction), and 689 samples show the next card also inside the known hand (also a contradiction).

**You are implementing, not designing.** If anything is unclear, stop and report.

## 1. Code changes (only these)

`clashbot/game_rules.py`:

1. Add
```python
def next_card_blocked(state: dict) -> str | None
```
Returns `state["own"].get("next_card")` if it is a non-empty string **and** it is not among the known hand cards (`[h for h in state["own"]["hand"] if h]`); otherwise None. (When the next card also appears in the known hand, the reading is contradictory; the hand wins and nothing is blocked.)

2. `possible_cards(state, cards)`: after building the result exactly as today, remove `next_card_blocked(state)` if present. Nothing else changes.

3. `legal_cards(state, cards, elixir_tolerance)`: in the fallback branch (`hand` empty → `pool = deck`), remove `next_card_blocked(state)` from `pool` before filtering. The known-hand branch is unchanged.

No other file changes in code. Do not modify the dataset builders, d1 format, runtime modules, or other tests except as listed in §2.

## 2. Tests — add to `tests/test_ranker_dataset.py` (new test functions only)
1. `possible_cards`: hand `["knight", None, None, None]`, deck of 8 incl. `"archers"`, `next_card = "archers"` → `"archers"` not in result, `"knight"` in result.
2. Contradiction: hand `["archers", None, None, None]`, `next_card = "archers"` → `"archers"` in result.
3. `legal_cards` fallback: hand all None, deck incl. `"archers"`, `next_card = "archers"`, enough elixir → `"archers"` not in result.
4. Integration (skip if `dataset/v1/ranker/val.jsonl` missing): for every line of `dataset/v1/d1/factorized/*.jsonl`, look up the state record by `id` in `dataset/v1/states.jsonl` and assert `next_card_blocked(state)` is not a key of `questions["card"]["criteria"]`; for every ranker example, no option card (key prefix before `|`) equals `next_card_blocked(state)`.

## 3. Rebuild, in this order
```
python3 tools/data/build_ranker_dataset.py
python3 tools/eval/eval_ranker_baselines.py
python3 -m pytest -q tests/test_ranker_dataset.py tests/test_d1_format.py tests/test_runtime.py tests/test_retrieval.py tests/test_clean_dataset.py
python3 tools/data/build_d1_dataset.py --tokenizer db/hf/d1_tokenizer.json
```
(The d1 builder runs last because the pytest integration test rewrites `d1/stats.json` without token fields.)

## 4. Report back (≤ 35 lines)
- Ranker stats per split: examples and `skipped_pos_impossible`, before → after (before: train 3865/330, val 807/109, test 797/105).
- The three baseline tables (candidate recall, ranker baselines, act).
- d1 counts per variant/split and `options_over_budget`.
- Test results; anything you could not follow exactly.
Do not git commit or push.
