# MAC-R2 — KG embeddings for missing COUNTERS links (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Course: KG TransE [2], DistMult [42], ComplEx [38], RotatE [36]; evaluation audits — equal tuning [34], random tie-breaking [37]. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
tools/eval/kge_link_prediction.py   (new)
tests/test_kge.py                   (new)
```
PyKEEN 1.11.1 is installed. CPU only (`device="cpu"`): the graph is small.

## 1. Data
Input `<dataset>/kg/edges.jsonl` (`{"src","rel","dst","props"}`) and `<dataset>/kg/nodes.jsonl` (`{"id","type",...}`).
- Training relations: `COUNTERS, SYNERGY, SUBSTITUTE, ANSWERED_BY, IN_DECK, SPAWNED_BY, EVOLUTION_OF, HERO_OF, HAS_ARCHETYPE` (drop the `GAME_*` relations). Deduplicate `(src, rel, dst)` triples (COUNTERS comes from both "strong against" and "weak against" pages, so the same triple can appear twice).
- Target: the deduplicated `COUNTERS` triples, sorted, shuffled with `random.Random(0)`, split 80/10/10 → `train_c`, `val_c`, `test_c`. Training triples = all non-COUNTERS triples + `train_c`. Drop val/test triples whose head or tail never occurs in training (count them as `dropped_unseen`).
- Two settings (report both):
  - `full`: as above.
  - `no_inverse`: also drop every `ANSWERED_BY` triple from training. Reason: `(X ANSWERED_BY Y)` is close to the inverse of `(Y COUNTERS X)` — the inverse-relation leakage the KG audit papers warn about.
- Candidates for ranking: all `card:` entities that occur in training (heads and tails are cards for COUNTERS).

## 2. Models and equal budget
Models: `TransE`, `DistMult`, `ComplEx`, `RotatE` (PyKEEN names), each with its PyKEEN default loss. **Same budget for every model**: grid `embedding_dim ∈ {32, 64}` × `lr ∈ {1e-2, 1e-3}` (4 configs), `num_epochs = --epochs` (default 150), `batch_size = 256`, optimizer `Adam`, `negative_sampler="basic"` with `num_negs_per_pos = 8`, `random_seed = 0`, training loop `sLCWA`. Train with
```python
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
tf = TriplesFactory.from_labeled_triples(train_array, create_inverse_triples=False)
res = pipeline(training=tf, testing=tf, model=name, model_kwargs={"embedding_dim": d},
               optimizer="Adam", optimizer_kwargs={"lr": lr}, training_loop="sLCWA",
               negative_sampler="basic", negative_sampler_kwargs={"num_negs_per_pos": 8},
               training_kwargs={"num_epochs": epochs, "batch_size": 256, "use_tqdm": False},
               random_seed=seed, device="cpu", evaluator=None)
model = res.model
```
If `evaluator=None` is rejected by this PyKEEN version, train with `pykeen.training.SLCWATrainingLoop` directly; **we do our own evaluation** (§3) so every model and baseline uses one protocol. Select the config by val MRR (§3), then retrain the selected config with seeds `{0, 1, 2}` and report test mean ± std.

## 3. One evaluation protocol for all methods (filtered, realistic rank)
For each triple `(h, COUNTERS, t)` in the evaluated split, rank in both directions over the card candidates `C`:
- tail: scores `s(h, COUNTERS, c)` for `c ∈ C`; **filter** = every other `c` with `(h, COUNTERS, c)` in train ∪ val ∪ test → removed; rank of `t` = `1 + #(s > s_t) + 0.5 * #(s == s_t, c ≠ t)` (expected rank under random tie-breaking, [37]).
- head: same with `s(c, COUNTERS, t)`.
- Metrics: `MRR` = mean of `1/rank` over both directions; `hits@1/3/10` = fraction with `rank <= k`; `n`.
Scores from a PyKEEN model: `model.score_t(hr_batch)` / `model.score_h(rt_batch)` (LongTensor of mapped ids, `torch.no_grad()`, `model.eval()`), indexed by `tf.entity_to_id` for `C`. Put the ranking in a pure function `realistic_rank(scores: np.ndarray, true_idx: int, filter_idx: list[int]) -> float`.

Baselines (same protocol):
- `random`: scores all equal → realistic rank = middle of the filtered list.
- `popularity`: tail score = in-degree of `c` in training COUNTERS; head score = out-degree of `c` in training COUNTERS.

Analysis for the best config of each model on `test_c`: `asymmetry` = fraction of test triples with `s(h, r, t) > s(t, r, h)` (DistMult is symmetric, so it should be ≈ 0 with ties); `reverse_in_graph` = fraction of test triples whose reverse `(t, COUNTERS, h)` is in training (context for the numbers).

## 4. CLI and output
```
$PY tools/eval/kge_link_prediction.py [--dataset DIR] [--epochs 150] [--models TransE,DistMult,ComplEx,RotatE]
        [--settings full,no_inverse] [--grid full|small] [--seeds 0,1,2] [--out PATH]
```
`--grid small` = only `(dim 32, lr 1e-2)` (for tests/smoke). Output JSON (`--out`, default `<dataset>/eval/kge_link_prediction.json`): `{"counts": {train/val/test triples, dropped_unseen, relations}, "settings": {setting: {"baselines": {...}, "models": {name: {"grid": [{dim, lr, val_mrr}], "best": {dim, lr}, "test": {"mrr_mean", "mrr_std", "hits@1", "hits@3", "hits@10" (means)}, "asymmetry", "train_s"}}}}, "novel_predictions": top 20 (h, t, score) pairs not in any split, from the model with the best `full` test MRR, with card names}`; also `<out>.md` with `| setting | method | MRR | H@1 | H@3 | H@10 |` and print it.

## 5. Tests — `tests/test_kge.py`
1. `realistic_rank`: scores `[0.9, 0.5, 0.5, 0.1]`, true 1, no filter → 2.5; filter `[0]` → 1.5; all equal (4) → 2.5.
2. Split determinism: two runs of the split function on a synthetic triple list give identical splits; no COUNTERS triple in two splits; duplicates removed; `no_inverse` has no ANSWERED_BY.
3. Popularity baseline on a synthetic star graph (one card countered by everything) ranks that card first for tail prediction.
4. Tiny PyKEEN run: synthetic KG of 12 card entities, 60 triples; each of the 4 models with `dim 8`, 2 epochs trains and returns finite MRR in (0, 1]. Keep it under ~30 s total.
5. Real data (skip if `dataset/v1/kg/edges.jsonl` missing): split counts sum to the number of deduplicated COUNTERS triples minus dropped; candidates are all `card:`.

## 6. Smoke (allowed once, after the symlinks)
`$PY tools/eval/kge_link_prediction.py --epochs 5 --models TransE,DistMult --grid small --seeds 0 --out runs/smoke/R2/kge.json` — paste the table and the run time.

## 7. Heavy run (Opus, at integration)
Full grid, 150 epochs, both settings → `docs/data/kge_link_prediction_v1.{json,md}`.

## 8. Report
Per MAC-00 §5.

---
## Amendment 1 (Opus review, 2026-11-02) — FIX_LIST #5
Hits@k must be the expectation under random tie-breaking, not Hits@k of the expected rank (a full tie over 126 candidates currently gives Hits@10 = 0 instead of ≈ 0.079).
Add `realistic_hits(scores, true_idx, filter_idx, k) -> float`: with `better = #(s > s_t)` and `tied = #(s == s_t)` including the true item (filtered set), return `max(0.0, min(1.0, (k - better) / tied))`. Use it for every hits@k (models and baselines); keep `realistic_rank` for MRR. Tests: 4 all-tied candidates → hits@1 0.25, hits@3 0.75; no ties → 0/1 as before; scores `[0.9, 0.5, 0.5, 0.1]`, true 1 → hits@1 0.0, hits@2 0.5.
