# MAC-X — learned play/wait decision: act-head data sampler, learned policy, offline evaluation (spec for a Sonnet implementer)

Author: Opus · 2026-11-04 · Implements the learned half of D-041 with the dense data of D-040; trains with MAC-T (D-039). Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
tools/data/build_dense_act.py      (add the state sidecar only)
tools/data/sample_act_train.py     (new)
clashbot/policy.py                 (add LearnedActPolicy; ReactivePolicy.decide gains an ignored context_text kwarg)
clashbot/live.py                   (CLI + pass context_text to the policy)
tools/eval/eval_act_head.py        (new)
tests/test_act_head.py             (new)
tests/test_dense_act.py, tests/test_live_loop.py, tests/test_policy.py   (update only as needed)
```
A dense perception run is in progress and will call `tools/data/build_dense_act.py` this evening: keep its CLI and outputs backward compatible.

## 1. `build_dense_act.py` — state sidecar
Also write `<out>/states_<split>.jsonl`: one line per written act line, same order, `{"id", "t", "state"}` (the clean state dict used to render it). Flag `--no-states` to skip. Test: sidecar ids equal the act-file ids.

## 2. `tools/data/sample_act_train.py`
```
$PY tools/data/sample_act_train.py [--src <dataset>/act_dense] [--out <dataset>/act_dense_bal]
        [--pos-per-game 20] [--neg-per-game 40] [--eval-per-game 120] [--seed 0]
```
Train: per game at most `pos-per-game` positives and `neg-per-game` negatives (`random.Random(seed)` sample without replacement), then shuffle all train lines. Val/test: per game at most `eval-per-game` lines sampled uniformly (natural act rate kept), file order kept. Copy the matching sidecar lines into `<out>/states_<split>.jsonl`. `stats.json` per split: lines, act rate, games. (Full dense data is ≈ 40 K lines — too many for a d1 fine-tune at ~1 row/s; this gives ≈ 8.5 K train lines.)

## 3. `clashbot/policy.py` — `LearnedActPolicy`
```python
class LearnedActPolicy:
    def __init__(self, scorer, cards: dict | None = None, threshold: float = 0.5,
                 rules: ReactivePolicy | None = None, min_check_gap_s: float = 0.5): ...
    def p_act(self, context_text: str) -> float
    def decide(self, state: dict, t: float, context_text: str | None = None) -> ActDecision
    def filter_candidates(self, candidates, decision, state) -> list
```
- `scorer` has `probabilities(state_str, questions) -> {name: [p_false, p_true]}` (a `D1Ranker` built on the act checkpoint). `p_act(ctx)`: `s, q = d1_format.act_request(ctx)`; `return scorer.probabilities(s, q)["act"][1]`.
- `decide`: `r = rules.decide(state, t)` (rules default `ReactivePolicy(cards)`).
  1. `r.mode == "none"` → return `r` (no affordable card).
  2. `r.act and r.mode in ("defend", "leak")` → return `r` (safety: always answer a threat; never sit at full elixir).
  3. `context_text is None` → return `r`.
  4. `p = p_act(context_text)`, cached: reuse the last `p` if the previous query was less than `min_check_gap_s` ago (by `t`).
  5. `p >= threshold` → `ActDecision(True, r.mode if r.act else "learned", r.lane if r.act else None, f"learned act p={p:.2f}")`; else `ActDecision(False, "hold", None, f"learned wait p={p:.2f}")`.
- `filter_candidates`: modes defend/leak/punish → delegate to the rules; mode `learned` → only the no-blind-spell rule of `ReactivePolicy` (factor that rule into a helper both use).
- `ReactivePolicy.decide(self, state, t, context_text=None)` — the new argument is accepted and ignored.

## 4. `clashbot/live.py`
- `LiveLoop.step` calls `self.policy.decide(state, t, context_text=ctx)`.
- CLI: `--policy reactive|learned|trigger`; `--act-model PATH` (required for `learned`), `--act-threshold` (float, default: read `threshold` from `<act-model>/act_threshold.json` if present, else 0.5), `--act-backend mlx|transformers` (default transformers). `learned` → `LearnedActPolicy(D1Ranker(model_path=act_model, backend=act_backend), cards, threshold)`.

## 5. `tools/eval/eval_act_head.py`
```
$PY tools/eval/eval_act_head.py --model PATH [--data <dataset>/act_dense_bal] [--backend transformers]
        [--splits val,test] [--limit N] [--out PATH]
```
1. For every line of each split: `p = D1Ranker(model_path, backend).probabilities(state, questions)["act"][1]`; label = answers act. State dicts from the sidecar (by id) for the rules.
2. Threshold: sweep `0.05..0.95` step 0.01 on **val**, pick the max balanced accuracy (ties → threshold closest to 0.5). Write `{"threshold": thr, "val_balanced_acc": ...}` to `<model>/act_threshold.json`.
3. Per split, for `learned` (p ≥ thr), `rules` (`ReactivePolicy.decide(...).act`), `combined` (`LearnedActPolicy` logic with that threshold, computed offline from p and the rules), `always` and `never`: accuracy, balanced accuracy, act rate vs pro act rate, precision/recall for act and for wait.
4. **Reaction** slice: lines with `meta.since_opp_play <= 3.0`: act rate of the pro and of each policy, and recall of the pro's acts there.
5. JSON (`--out`, default `<dataset>/eval/act_head.json`) + `.md`; print the tables.

## 6. Tests — `tests/test_act_head.py` (+ updates)
Fake scorer returning a fixed `p` per context string. 1) `p_act` builds a valid act request (`d1_format.validate`). 2) decide: no card → none; defend/leak pass through regardless of p; p ≥ thr → act (mode learned with lane None when rules hold; keeps the rule's mode when the rules say punish); p < thr → hold; cache reuses p within `min_check_gap_s` (scorer called once). 3) learned-mode filtering drops blind spells only. 4) sampler caps per game and keeps eval files at the natural rate; sidecars aligned. 5) eval script on a tiny synthetic data dir with a monkeypatched `D1Ranker`: threshold chosen on val, file written, all five policies reported, reaction slice present. 6) live CLI: `--policy learned` without `--act-model` → exit 2; with a model dir containing `act_threshold.json` → that threshold used (monkeypatched `D1Ranker`). Never load a real model.

## 7. Smoke
Unit tests only (no dense data yet; a GPU job is running — do not load models).

## 8. Heavy runs (Opus)
`sample_act_train.py` → `train_d1.py --variant act --data <dataset>/act_dense_bal --init db/models/d1-ft-joint --train-layers 8 --epochs 2` → `eval_act_head.py --model db/models/d1-ft-act` → live with `--policy learned --act-model db/models/d1-ft-act`.

## 9. Report
Per MAC-00 §5.
