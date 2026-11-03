# MAC-D — d1 zero-shot ranker + evaluation (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Implements D-022, D-023, D-026. **No fine-tuning** (D-022). Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
clashbot/ranker.py                 (implement D1Ranker; keep everything else as is)
tools/eval/eval_d1_zero_shot.py    (new)
tests/test_d1_ranker.py            (new)
```

## Contract (package G and loop.py code against this — keep it exact)
```python
# clashbot/ranker.py
D1_MLX_ID = "LiquidAI/d1-omni-600M-MLX-fp16"
D1_TORCH_ID = "LiquidAI/d1-omni-600M"

class D1Ranker:
    name = "d1"
    def __init__(self, model_path: str | None = None, backend: str = "mlx", device: str | None = None,
                 predict_fn=None): ...
    def build_request(self, context: str, candidate_keys: list[str], plan: str | None = None) -> tuple[str, dict]
    def probabilities(self, state: str, questions: dict) -> dict[str, list[float]]   # {question name: probs in criteria order}
    def choose_keys(self, context: str, candidate_keys: list[str], plan: str | None = None) -> RankResult
    def choose(self, context: str, options: list[str], plan: str | None = None) -> RankResult
    last_ms: float                   # wall time of the last model call
```
`RankResult(index, probs, model)` is unchanged (`model = f"d1-{backend}"`).

## 1. Backends (verified by Opus from the model cards on 2026-11-02)
**mlx** (default, D-026): `mlx-vlm` 0.7.11 is installed.
```python
from mlx_vlm import load, predict
model, processor = load(model_path or D1_MLX_ID)
result = predict(model, processor, state, questions)     # {"answers": {name: answer}, "usage": {...}}
```
A choice answer has `"value"` (chosen label) and `"probabilities"`; a noul answer has `"probability"` (P(yes)).
**transformers** (reference): `transformers` 5.22 installed.
```python
import torch
from transformers import AutoModel
dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.float16
model = AutoModel.from_pretrained(model_path or D1_TORCH_ID, trust_remote_code=True, dtype=dtype).to(dev).eval()
result = model.system_one(state, questions)              # {"answers": {name: answer}, ...}
```
A choice answer has `"choice"`, `"confidence"`, `"probabilities"` (dict name→p); a noul answer has `"noul"` (P(yes)).

Load lazily on the first `probabilities` call (imports inside the method), cache the model on the instance. Wrap the backend call in `torch.no_grad()` for transformers.

## 2. `probabilities(state, questions)`
1. `d1_format.validate(state, questions)` first.
2. `raw = self._predict(state, questions)` — `predict_fn(state, questions)` if given (tests), else the backend call above; returns the backend's `result` dict.
3. Normalise each question `name` in `questions` (iterate in `questions` order):
   - `ans = raw["answers"][name]`; `p = ans.get("probabilities")`.
   - choice: `keys = list(questions[name]["criteria"])`; if `p` is a dict → `[float(p[k]) for k in keys]`; if a list/tuple → `[float(x) for x in p]` (must have `len(keys)` entries, else `ValueError`). 
   - noul: `y = ans.get("probability", ans.get("noul"))`; if `y is None` and `p` is a dict: `y = p.get("true", p.get("yes"))`; output `[1.0 - float(y), float(y)]` (order: false, true).
   - score: list in level order (dict → sorted by int key).
4. Record `self.last_ms` (perf_counter around step 2). Return `{name: probs}`.

## 3. `choose_keys` / `choose`
- `choose_keys(context, keys, plan)`: `state, q = self.build_request(context, keys, plan)`; `probs = self.probabilities(state, q)["play"]`; `index = argmax_first(probs)` (first maximum wins — add a small module-level `argmax_first` helper); `RankResult(index, probs, f"d1-{backend}")`.
- `choose(context, options, plan)`: if every option is a play key (`len(o.split("|")) == 3`) delegate to `choose_keys`; otherwise raise `ValueError("D1Ranker.choose needs action keys (card|lane|zone); use choose_keys")`.
- `D1Ranker` no longer subclasses `_StubDecisionModel`; keep `LayaRanker` and `_StubDecisionModel` unchanged. Keep `model_id` attribute = `model_path or default id`.
- Invalid `backend` → `ValueError` in `__init__`. Do not import mlx/torch/transformers at module import time.

## 4. `tools/eval/eval_d1_zero_shot.py`
```
$PY tools/eval/eval_d1_zero_shot.py [--dataset DIR] [--backend mlx|transformers] [--variant joint|factorized|both]
        [--splits val,test] [--limit N] [--out PATH] [--compare PATH_A PATH_B]
```
- Reads `<dataset>/d1/<variant>/<split>.jsonl` (lines: `id, game_id, split, kind, variant, state, questions, answers`). Uses each line's `state` and `questions` **as exported** (do not rebuild requests).
- One `D1Ranker(backend=...)`; warm-up: one call on the first line, not timed. Then per line: `probs = ranker.probabilities(state, questions)`, `ms = ranker.last_ms`.
- **joint** metrics per split: `acc` (argmax_first == index of `answers["play"]` in criteria order), `acc_by_kind`, `mean_gold_prob`, `pred_position_counts` (histogram of predicted index 0..3), `gold_position_counts`, `ms_p50`, `ms_p95`, `n`.
- **factorized** per split: per question (`card`, `lane`, `zone`) accuracy, `all_three` accuracy, `key_acc` (predicted `card|lane|zone` == gold), `ms_p50/p95`, `n`.
- Baselines next to it: read `<dataset>/ranker/baselines.json` → `ranker[split]` accuracies (`random_expected`, `popularity`, `placement_prior`, …) copied into the output for the joint table.
- **Option-set shortcut baseline** (Opus finding, 2026-11-02: the WP4 negatives N2 = same card/other zone and N3 = other card/same lane+zone make the positive identifiable from the option keys alone, ≈0.85 accuracy). Compute it per split on the joint lines and print it as a column: for each line, `keys = list(criteria)`; `score(k) = #options with the same card as k + #options with the same (lane, zone) as k` (each count includes k itself); expected accuracy with random tie-break = `1/|argmax set|` if the gold key is in the argmax set, else 0; mean over lines → `option_set_shortcut`. Also report `d1_agrees_with_shortcut` = fraction of lines where d1's prediction is in the shortcut's argmax set.
- Output (`--out`, default `<dataset>/eval/d1_zero_shot_<backend>.json`, mkdir parents): `{"backend", "model_id", "variants": {variant: {split: metrics}}, "baselines": {split: {...}}, "limit", "per_example": [{"id", "variant", "split", "pred", "gold", "probs"}]}` (per_example keeps every evaluated line — needed for `--compare`).
- Print markdown: `| split | n | d1 acc | option_set_shortcut | placement_prior | popularity | random | ms p50 | ms p95 |` for joint, and `| split | n | card | lane | zone | all three | key | ms p50 |` for factorized.
- `--compare A B`: no model; load two output files, match `per_example` by `(id, variant)`, print and return `{"n_common", "top_agreement", "max_abs_prob_diff", "mean_abs_prob_diff"}` (used to compare mlx vs transformers).

## 5. Tests — `tests/test_d1_ranker.py` (fakes only; no model loads)
A fake backend: `predict_fn` returning answers built from the questions (e.g. the last criteria key gets 0.7, others share 0.3), in **both** output styles (mlx: `{"value", "probabilities": {..}}`; transformers: `{"choice", "probabilities": {..}}`; also a list-style `probabilities`).
1. `choose_keys` with 4 keys returns the index of the boosted key, `probs` in key order summing to ~1, `model == "d1-mlx"`.
2. dict-style, list-style and transformers-style answers all normalise to the same probs.
3. noul normalisation: `{"probability": 0.8}` → `[0.2, 0.8]`; `{"noul": 0.3}` → `[0.7, 0.3]`.
4. `choose` with option texts (not keys) raises `ValueError`; with keys equals `choose_keys`.
5. `D1Ranker(backend="onnx")` raises `ValueError`; constructing `D1Ranker()` imports neither `mlx_vlm` nor `transformers` (check `sys.modules` before/after, with those names removed first if present).
6. Ties: probs `[0.25]*4` → index 0.
7. Request passes `d1_format.validate`; `"<|"` in context raises before the backend is called (fake must not be called — count calls).
8. Eval script on a tiny fake dataset in `tmp_path` (`d1/joint/val.jsonl` with 3 lines, `d1/factorized/val.jsonl` with 2 lines, `ranker/baselines.json`): call its `main([...])` with a monkeypatched `D1Ranker` whose `predict_fn` is the fake → output JSON exists, joint `acc` matches the hand-computed value, factorized has `card`, `lane`, `zone`, `all_three`, `key_acc`.
9. `--compare` on two hand-written output files → `top_agreement` correct.
9b. Shortcut baseline: options `a|left|mid` (gold), `a|left|back`, `b|left|mid`, `c|right|bridge` → shortcut argmax = {gold} → 1.0; an all-distinct option set → 0.25 for any gold.
10. `@pytest.mark.slow` + skip unless `CLASHBOT_SLOW_TESTS=1`: real mlx backend on one joint val line returns 4 probabilities summing to 1 ± 1e-3.

## 6. Smoke (allowed once each, after the symlinks in MAC-00 §1; models are cached)
```bash
$PY tools/eval/eval_d1_zero_shot.py --backend mlx --variant both --splits val --limit 5 --out runs/smoke/D/mlx.json
$PY tools/eval/eval_d1_zero_shot.py --backend transformers --variant joint --splits val --limit 5 --out runs/smoke/D/torch.json
$PY tools/eval/eval_d1_zero_shot.py --compare runs/smoke/D/mlx.json runs/smoke/D/torch.json
```
If a backend's real output shape differs from §1, adapt the normaliser (that is the purpose of the smoke run) and say so under deviations, pasting one raw answer dict.

## 7. Heavy run (Opus, at integration)
Full val+test, both variants, mlx; joint val+test transformers; `--compare`. Results → `docs/data/d1_zero_shot_v1.json` + a table in STATUS.md.

## 8. Report
Per MAC-00 §5, plus the smoke tables.
