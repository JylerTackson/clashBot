# MAC-R1 — retrieval ablation: BM25 vs structured vs text embedding vs RRF; exact vs HNSW; pre- vs post-filter (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Course: RET BM25 [32], bi-encoder limits [14], RRF [3], HNSW [20], filtering §2.2, BEIR [40]. Implements D-033. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
clashbot/text_embed.py           (new)
tools/eval/eval_retrieval.py     (new)
tests/test_eval_retrieval.py     (new)
```
Do not change `clashbot/retrieval.py`; reuse `StateIndex.knn`, `StateIndex.bm25`, `StateIndex.rrf`, `StateIndex.mask` so the ablation measures the runtime code.

## 1. `clashbot/text_embed.py`
```python
DEFAULT_TEXT_MODEL = "BAAI/bge-small-en-v1.5"      # 384-d (D-033)

class TextEmbedder:
    def __init__(self, model_id: str = DEFAULT_TEXT_MODEL, device: str | None = None, encode_fn=None): ...
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray     # (n, d) float32, rows L2-normalised

class HashEmbedder:                                   # deterministic, model-free (tests, --embedder hash)
    def __init__(self, dim: int = 384): ...
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray

def build_state_text_matrix(db_path: Path = DB_PATH, embedder=None, cache_dir: Path | None = None,
                            force: bool = False) -> np.ndarray
def text_knn(E: np.ndarray, q: np.ndarray, allowed: np.ndarray, k: int) -> list[tuple[int, float]]
```
- `TextEmbedder`: lazy `from sentence_transformers import SentenceTransformer` on first `encode`; `device` default `"mps"` if `torch.backends.mps.is_available()` else `"cpu"`; `model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)`. With `encode_fn`, call `encode_fn(texts)` and L2-normalise rows yourself.
- `HashEmbedder.encode`: tokens = `re.findall(r"[a-z][a-z0-9\-]*", text.lower())`; each token adds 1.0 to index `int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim`; L2-normalise (zero rows stay zero).
- `build_state_text_matrix`: texts = `SELECT context_text FROM states ORDER BY vec_row` (read-only connection); `cache_dir` default `DB_DIR`; cache files `state_text_emb.npy` + `state_text_emb.meta.json` = `{"model": embedder id (TextEmbedder.model_id or "hash-<dim>"), "n", "dim", "texts_sha1": sha1 of "\n".join(texts)}`; reuse the cache when meta matches and not `force`; else encode and write both files.
- `text_knn`: `scores = E @ q` (float64), `scores[~allowed] = -inf`, then exactly the ordering of `StateIndex.knn` (lexsort by `(row, -score)`), top `k`, finite only.

## 2. `tools/eval/eval_retrieval.py`
```
$PY tools/eval/eval_retrieval.py [--dataset DIR] [--db PATH] [--embedder bge|hash] [--cache-dir DIR]
        [--splits val,test] [--limit N] [--depth 50] [--out PATH] [--skip-hnsw] [--skip-filter]
```
**Queries**: `states.jsonl` records with `split` in `--splits`, `train_eligible`, `action.type == "play"`, `action.key` not None (first `--limit` per split if given). **Pool**: rows of `StateIndex(db)` with `split == "train"` and an action (`mask(splits=["train"], exclude_game=query_game, require_action=True)`). Ground truth: the pro's `action.key` and card.

### 2a. Channel ablation (depth `D = --depth`, default 50)
Per query: `bm25 = index.bm25(context_text, D, splits=["train"], exclude_game=g)`; `struct = index.knn(featurize(state, t), D, ...same filters)` (exact; `StateIndex(db, use_hnsw=False)`); `text = text_knn(E, e_q, allowed, D)` with `E = build_state_text_matrix(...)` and `e_q` the query's embedding (precompute all query embeddings in one batch). Fusions with `StateIndex.rrf(..., k_rrf=60)`: `struct+bm25` (= the runtime default), `struct+text`, `bm25+text`, `all3`. Seven ranked lists per query, each truncated to `D`.

Metrics per list and split, using `index.action_key[row]` of the hits (`card = key.split("|")[0]`, `"hold"` never matches):
- `hit@k` for k ∈ {1, 5, 10, 20}: exact key among the top-k hits; `card_hit@k` likewise for the card.
- `vote_recall@k` for k ∈ {1, 3, 10}: weight each hit at rank r (1-based) by `1/(60 + r)`, sum weights per action key (non-hold), rank keys by `(-weight, key)`; pro key within the top-k keys. (This is what `CandidateGenerator` does.)
- `ndcg@10`: relevance 2 = exact key, 1 = same card other key, 0 otherwise; `DCG = Σ rel_r / log2(r + 1)`; ideal DCG from the pool counts: `n_exact` = pool rows (same mask) with that key, `n_card` = pool rows with the same card and another key → ideal list `[2]*min(10, n_exact) + [1]*…` up to 10; `ndcg = DCG / IDCG` (0 when IDCG = 0).
- Latency: on the first 200 queries per split time each channel separately (BM25 call, `featurize` + exact kNN, single-query `encode` + `text_knn`), fusions = sum of their channels + the `rrf` call; report `ms_p50`, `ms_p95`.

### 2b. Exact vs HNSW (struct channel, unfiltered; skip with `--skip-hnsw`)
`hnswlib.Index(space="ip", dim)` loaded from `<db dir>/state_features.hnsw` (built with M=16, ef_construction=200). For `ef ∈ {10, 20, 40, 80, 160, 320}`: `set_ef(ef)`; per query (val+test) `recall@10 = |HNSW top-10 ∩ exact top-10| / 10` over **all** rows (no mask, k=10) and `ms` per query; exact baseline `ms` = numpy `M @ q` + top-10. Table: `| ef | recall@10 | ms p50 | ms p95 |` plus an `exact` row.

### 2c. Pre- vs post-filtering (struct channel; skip with `--skip-filter`)
Filters: `deck` (pool rows with `own_deck_key == query's own_deck_key`; queries whose own deck key is None are skipped and counted) and `phase` (`phase == query phase`). Strategies, each returning up to 10 hits:
- `pre`: `index.knn(vec, 10, splits=["train"], exclude_game=g, deck_key=… | phase=…)` (mask first, exact).
- `post@K` for K ∈ {10, 50, 200}: HNSW (`ef = max(64, K)`) unfiltered top-K over all rows, then keep rows that pass `train & not same game & filter`, first 10.
Metrics: `pool_empty_rate` (no pool row passes the filter — same for all strategies), `empty_rate` (0 hits returned, over queries with a non-empty pool), `mean_returned`, `hit@10` (exact key), `ms_p50`.

### 2d. Output
JSON (`--out`, default `<dataset>/eval/retrieval_ablation.json`, mkdir parents): `{"embedder", "depth", "n_queries": {split: n}, "channels": {split: {list_name: metrics}}, "hnsw": [...], "filtering": {filter: {strategy: metrics}}}`; also write the same path with `.md` holding three markdown tables (channels per split: `hit@1 hit@10 card_hit@10 vote_recall@10 ndcg@10 ms_p50`; HNSW; filtering) and print them.

## 3. Tests — `tests/test_eval_retrieval.py`
Factor the metric code into importable pure functions in `eval_retrieval.py` (`hit_at_k`, `vote_recall_at_k`, `ndcg_at_k`, `recall_overlap`) and test them on hand-made lists.
1. `HashEmbedder`: deterministic, unit-norm rows, identical texts → identical rows, empty text → zero row.
2. `TextEmbedder(encode_fn=...)` normalises rows; constructing it does not import `sentence_transformers`.
3. `text_knn`: masked rows never returned; ties broken by lower row; `k` larger than allowed rows returns only allowed rows.
4. `hit_at_k`, `vote_recall_at_k` (hold ignored, weights `1/(60+r)`), `ndcg_at_k` (perfect list → 1.0; no relevant → 0.0; a hand-computed mixed case), `recall_overlap`.
5. `build_state_text_matrix` on a tiny DB in `tmp_path` (`create_schema` + 3 `states` rows with `context_text`) with `HashEmbedder`: shape `(3, 384)`, cache files written, second call reuses the cache (monkeypatch the embedder's `encode` to raise).
6. Real data (skip if `db/clashbot.sqlite` missing): `main(["--embedder", "hash", "--limit", "20", "--skip-hnsw", "--skip-filter", "--cache-dir", str(tmp_path), "--out", str(tmp_path / "r.json")])` → JSON has the seven list names for val and test, all metrics in [0, 1].

## 4. Smoke (allowed once each, after the symlinks; writes only to runs/smoke)
```bash
$PY tools/eval/eval_retrieval.py --embedder hash --limit 50 --cache-dir runs/smoke/R1 --out runs/smoke/R1/hash.json
$PY tools/eval/eval_retrieval.py --embedder bge --limit 30 --skip-hnsw --skip-filter --cache-dir runs/smoke/R1 --out runs/smoke/R1/bge.json
```
(The second embeds all 10,923 states once on MPS, ≈1–2 min; the cache goes to `runs/smoke/R1`, not `db/`.)

## 5. Heavy run (Opus, at integration)
Full run with `--embedder bge` (cache in `db/`) → `docs/data/retrieval_ablation_v1.{json,md}`.

## 6. Report
Per MAC-00 §5, plus the smoke tables.
