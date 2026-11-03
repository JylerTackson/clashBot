"""Retrieval ablation (MAC-R1, D-033): BM25 vs structured vs text embedding vs RRF fusions, exact vs HNSW,
pre- vs post-filtering. Reuses StateIndex so the numbers measure the runtime code.

  python3 tools/eval/eval_retrieval.py [--dataset DIR] [--db PATH] [--embedder bge|hash] [--cache-dir DIR]
          [--splits val,test] [--limit N] [--depth 50] [--out PATH] [--skip-hnsw] [--skip-filter]

Queries are val/test play states; the pool is the train split without the query's game. Writes
<dataset>/eval/retrieval_ablation.json (+ .md with three tables) and prints them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.config import DATASET_DIR, DB_PATH, HOLD_KEY  # noqa: E402
from clashbot.features import featurize  # noqa: E402
from clashbot.retrieval import StateIndex  # noqa: E402
from clashbot.text_embed import (HashEmbedder, TextEmbedder, build_state_text_matrix,  # noqa: E402
                                 text_knn)

HIT_KS = (1, 5, 10, 20)
VOTE_KS = (1, 3, 10)
NDCG_K = 10
K_RRF = 60
LAT_N = 200                      # queries per split used for latency
EF_SWEEP = (10, 20, 40, 80, 160, 320)
POST_KS = (10, 50, 200)
TOP = 10                         # hits returned by the filtering strategies
CHANNELS = ("bm25", "struct", "text")
FUSIONS = {"struct+bm25": ("struct", "bm25"), "struct+text": ("struct", "text"),
           "bm25+text": ("bm25", "text"), "all3": ("struct", "bm25", "text")}
LIST_NAMES = CHANNELS + tuple(FUSIONS)


# ---- pure metric functions ------------------------------------------------------------------------------

def card_of(key: str | None) -> str | None:
    if not key or key == HOLD_KEY:
        return None
    return key.split("|")[0]


def hit_at_k(hit_keys: Sequence[str | None], target: str | None, k: int) -> float:
    """1.0 if the exact action key is among the top-k hits ("hold" never matches)."""
    if not target or target == HOLD_KEY:
        return 0.0
    return 1.0 if target in hit_keys[:k] else 0.0


def card_hit_at_k(hit_keys: Sequence[str | None], target: str | None, k: int) -> float:
    card = card_of(target)
    if card is None:
        return 0.0
    return 1.0 if any(card_of(h) == card for h in hit_keys[:k]) else 0.0


def vote_ranking(hit_keys: Sequence[str | None], k_rrf: int = K_RRF) -> list[str]:
    """Action keys ranked by summed 1/(k_rrf + rank) over their hits, hold ignored (CandidateGenerator vote)."""
    weight: dict[str, float] = {}
    for r, key in enumerate(hit_keys, start=1):
        if key is None or key == HOLD_KEY:
            continue
        weight[key] = weight.get(key, 0.0) + 1.0 / (k_rrf + r)
    return sorted(weight, key=lambda key: (-weight[key], key))


def vote_recall_at_k(hit_keys: Sequence[str | None], target: str | None, k: int, k_rrf: int = K_RRF) -> float:
    if not target or target == HOLD_KEY:
        return 0.0
    return 1.0 if target in vote_ranking(hit_keys, k_rrf)[:k] else 0.0


def ndcg_at_k(hit_keys: Sequence[str | None], target: str | None, n_exact: int, n_card: int,
              k: int = NDCG_K) -> float:
    """Graded nDCG: 2 = exact key, 1 = same card other key, 0 otherwise. The ideal list comes from the
    pool counts (n_exact rows with the key, n_card rows with the same card and another key)."""
    card = card_of(target)
    if card is None:
        return 0.0
    dcg = 0.0
    for r, key in enumerate(hit_keys[:k], start=1):
        if key == target:
            rel = 2
        elif card_of(key) == card:
            rel = 1
        else:
            rel = 0
        dcg += rel / math.log2(r + 1)
    n2 = min(k, max(n_exact, 0))
    n1 = min(k - n2, max(n_card, 0))
    ideal = [2] * n2 + [1] * n1
    idcg = sum(rel / math.log2(r + 1) for r, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_overlap(approx: Sequence[int], exact: Sequence[int], k: int = 10) -> float:
    """|approx[:k] ∩ exact[:k]| / |exact[:k]|."""
    ex = set(exact[:k])
    if not ex:
        return 0.0
    return len(set(approx[:k]) & ex) / len(ex)


def query_metrics(hit_keys: Sequence[str | None], target: str, n_exact: int, n_card: int) -> dict[str, float]:
    m: dict[str, float] = {}
    for k in HIT_KS:
        m[f"hit@{k}"] = hit_at_k(hit_keys, target, k)
    for k in HIT_KS:
        m[f"card_hit@{k}"] = card_hit_at_k(hit_keys, target, k)
    for k in VOTE_KS:
        m[f"vote_recall@{k}"] = vote_recall_at_k(hit_keys, target, k)
    m[f"ndcg@{NDCG_K}"] = ndcg_at_k(hit_keys, target, n_exact, n_card, NDCG_K)
    return m


def latency_stats(ms: list[float]) -> dict:
    if not ms:
        return {"ms_p50": None, "ms_p95": None}
    return {"ms_p50": round(float(np.percentile(ms, 50)), 3), "ms_p95": round(float(np.percentile(ms, 95)), 3)}


def mean_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    return {name: round(float(np.mean([m[name] for m in per_query])), 4) for name in per_query[0]}


# ---- data -----------------------------------------------------------------------------------------------

def load_queries(dataset: Path, splits: Sequence[str], limit: int | None) -> list[dict]:
    """Eligible play states of the given splits, in file order, at most `limit` per split."""
    out: list[dict] = []
    count: dict[str, int] = defaultdict(int)
    with open(Path(dataset) / "states.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            split = r.get("split")
            action = r.get("action") or {}
            if (split not in splits or not r.get("train_eligible") or action.get("type") != "play"
                    or action.get("key") is None):
                continue
            if limit is not None and count[split] >= limit:
                continue
            count[split] += 1
            out.append({"id": r["id"], "game_id": r["game_id"], "split": split, "t": r["t"],
                        "state": r["state"], "context_text": r.get("context_text") or "",
                        "key": action["key"], "card": card_of(action["key"])})
    return out


def make_embedder(name: str):
    if name == "hash":
        return HashEmbedder()
    return TextEmbedder()


def load_hnsw(index: StateIndex):
    path = index.db_path.parent / "state_features.hnsw"
    if not path.exists():
        return None
    import hnswlib
    h = hnswlib.Index(space="ip", dim=int(index.M.shape[1]))
    h.load_index(str(path), max_elements=index.n)
    return h


# ---- 2a channel ablation --------------------------------------------------------------------------------

def run_channels(index: StateIndex, E: np.ndarray, embedder, queries: list[dict], q_emb: np.ndarray,
                 depth: int, lat_n: int = LAT_N) -> dict:
    card_arr = np.array([card_of(k) for k in index.action_key], dtype=object)
    acc: dict[str, dict[str, list]] = defaultdict(lambda: {name: [] for name in LIST_NAMES})
    lat: dict[str, dict[str, list]] = defaultdict(lambda: {name: [] for name in LIST_NAMES})
    seen: dict[str, int] = defaultdict(int)
    pc = time.perf_counter
    for q, e_q in zip(queries, q_emb):
        split, target = q["split"], q["key"]
        local_idx = seen[split]
        seen[split] += 1
        flt = {"splits": ["train"], "exclude_game": q["game_id"]}

        t0 = pc()
        bm = index.bm25(q["context_text"], depth, **flt)
        t_bm = pc() - t0
        t0 = pc()
        q["vec"] = featurize(q["state"], q["t"])
        st = index.knn(q["vec"], depth, **flt)
        t_st = pc() - t0
        t0 = pc()
        allowed = index.mask(**flt)
        tx = text_knn(E, e_q, allowed, depth)
        t_tx = pc() - t0
        timed = local_idx < lat_n
        if timed:                                   # single-query encode, as the live loop would do
            t0 = pc()
            embedder.encode([q["context_text"]])
            t_tx += pc() - t0

        lists = {"bm25": bm, "struct": st, "text": tx}
        chan_t = {"bm25": t_bm, "struct": t_st, "text": t_tx}
        for name, members in FUSIONS.items():
            t0 = pc()
            fused = index.rrf({m: lists[m] for m in members}, k_rrf=K_RRF)
            t_rrf = pc() - t0
            lists[name] = [(row, score) for row, score, _ranks in fused[:depth]]
            chan_t[name] = sum(chan_t[m] for m in members) + t_rrf

        pool_keys = index.action_key[allowed]
        n_exact = int(np.count_nonzero(pool_keys == target))
        n_card = int(np.count_nonzero(card_arr[allowed] == q["card"])) - n_exact
        for name in LIST_NAMES:
            hit_keys = [index.action_key[row] for row, _ in lists[name]]
            acc[split][name].append(query_metrics(hit_keys, target, n_exact, n_card))
            if timed:
                lat[split][name].append(chan_t[name] * 1000.0)

    out: dict[str, dict] = {}
    for split in acc:
        out[split] = {name: {**mean_metrics(acc[split][name]), **latency_stats(lat[split][name])}
                      for name in LIST_NAMES}
    return out


# ---- 2b exact vs HNSW -----------------------------------------------------------------------------------

def run_hnsw(index: StateIndex, hnsw, vecs: list[np.ndarray], ef_sweep: Sequence[int] = EF_SWEEP) -> list[dict]:
    if hnsw is None or not vecs:
        return []
    pc = time.perf_counter
    M = np.asarray(index.M)
    rows = np.arange(index.n)
    exact_top: list[list[int]] = []
    exact_ms: list[float] = []
    for v in vecs:
        t0 = pc()
        scores = np.asarray(M @ v, dtype=np.float64)
        top = np.lexsort((rows, -scores))[:TOP]
        exact_ms.append((pc() - t0) * 1000.0)
        exact_top.append([int(x) for x in top])
    out: list[dict] = []
    for ef in ef_sweep:
        hnsw.set_ef(ef)
        rec: list[float] = []
        ms: list[float] = []
        for v, ex in zip(vecs, exact_top):
            t0 = pc()
            labels, _ = hnsw.knn_query(v.reshape(1, -1), k=TOP)
            ms.append((pc() - t0) * 1000.0)
            rec.append(recall_overlap([int(x) for x in labels[0]], ex, TOP))
        out.append({"ef": ef, "recall@10": round(float(np.mean(rec)), 4), **latency_stats(ms)})
    out.append({"ef": "exact", "recall@10": 1.0, **latency_stats(exact_ms)})
    return out


# ---- 2c pre- vs post-filtering --------------------------------------------------------------------------

def run_filtering(index: StateIndex, hnsw, queries: list[dict]) -> dict:
    pc = time.perf_counter
    out: dict[str, dict] = {}
    for fname, column, kw in (("deck", index.own_deck_key, "deck_key"), ("phase", index.phase, "phase")):
        strategies = ["pre"] + ([f"post@{K}" for K in POST_KS] if hnsw is not None else [])
        acc = {s: {"returned": [], "hit": [], "ms": []} for s in strategies}
        n_skipped = n_pool_empty = n_total = 0
        for q in queries:
            value = column[q["row"]]
            if value is None:
                n_skipped += 1
                continue
            n_total += 1
            base_kw = {"splits": ["train"], "exclude_game": q["game_id"], kw: value}
            if not index.mask(**base_kw).any():
                n_pool_empty += 1
                continue
            vec = q["vec"]
            for s in strategies:
                t0 = pc()
                if s == "pre":
                    rows = [r for r, _ in index.knn(vec, TOP, **base_kw)]
                else:
                    K = int(s.split("@")[1])
                    hnsw.set_ef(max(64, K))
                    allowed = index.mask(**base_kw)
                    labels, _ = hnsw.knn_query(vec.reshape(1, -1), k=min(K, index.n))
                    rows = [int(lab) for lab in labels[0] if allowed[int(lab)]][:TOP]
                ms = (pc() - t0) * 1000.0
                acc[s]["returned"].append(len(rows))
                acc[s]["hit"].append(1.0 if any(index.action_key[r] == q["key"] for r in rows) else 0.0)
                acc[s]["ms"].append(ms)
        n_eval = n_total - n_pool_empty
        out[fname] = {}
        for s in strategies:
            a = acc[s]
            out[fname][s] = {
                "pool_empty_rate": round(n_pool_empty / n_total, 4) if n_total else None,
                "empty_rate": round(float(np.mean([n == 0 for n in a["returned"]])), 4) if n_eval else None,
                "mean_returned": round(float(np.mean(a["returned"])), 3) if n_eval else None,
                "hit@10": round(float(np.mean(a["hit"])), 4) if n_eval else None,
                "ms_p50": latency_stats(a["ms"])["ms_p50"],
                "n_eval": n_eval, "n_skipped": n_skipped}
    return out


# ---- output ---------------------------------------------------------------------------------------------

def _f(v, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(result: dict) -> str:
    lines: list[str] = []
    cols = ("hit@1", "hit@10", "card_hit@10", "vote_recall@10", "ndcg@10", "ms_p50")
    for split, lists in result["channels"].items():
        lines += [f"### Channels, {split} split (n={result['n_queries'].get(split, 0)}, "
                  f"embedder {result['embedder']}, depth {result['depth']})", "",
                  "| list | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for name in LIST_NAMES:
            m = lists.get(name, {})
            lines.append(f"| {name} | " + " | ".join(_f(m.get(c)) for c in cols) + " |")
        lines.append("")
    if result["hnsw"]:
        lines += ["### Exact vs HNSW (struct channel, unfiltered, k=10)", "",
                  "| ef | recall@10 | ms p50 | ms p95 |", "|---|---|---|---|"]
        for r in result["hnsw"]:
            lines.append(f"| {r['ef']} | {_f(r['recall@10'], 4)} | {_f(r['ms_p50'])} | {_f(r['ms_p95'])} |")
        lines.append("")
    if result["filtering"]:
        lines += ["### Pre- vs post-filtering (struct channel; rates over eligible queries, "
                  "returned/hit over queries with a non-empty pool)", "",
                  "| filter | strategy | pool_empty_rate | empty_rate | mean_returned | hit@10 | ms p50 |",
                  "|---|---|---|---|---|---|---|"]
        for fname, strategies in result["filtering"].items():
            for s, m in strategies.items():
                lines.append(f"| {fname} | {s} | {_f(m['pool_empty_rate'])} | {_f(m['empty_rate'])} | "
                             f"{_f(m['mean_returned'])} | {_f(m['hit@10'])} | {_f(m['ms_p50'])} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--embedder", choices=("bge", "hash"), default="bge")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-hnsw", action="store_true")
    ap.add_argument("--skip-filter", action="store_true")
    args = ap.parse_args(argv)
    splits = [s for s in args.splits.split(",") if s]
    out_path = args.out if args.out is not None else args.dataset / "eval" / "retrieval_ablation.json"

    index = StateIndex(args.db, use_hnsw=False)
    embedder = make_embedder(args.embedder)
    E = build_state_text_matrix(args.db, embedder, args.cache_dir)

    id_to_row = {sid: i for i, sid in enumerate(index.id)}
    queries = []
    for q in load_queries(args.dataset, splits, args.limit):
        if q["id"] in id_to_row:
            q["row"] = id_to_row[q["id"]]
            queries.append(q)
    n_queries = {s: sum(1 for q in queries if q["split"] == s) for s in splits}
    print(f"queries: {n_queries}  pool: {int(index.mask(splits=['train']).sum())} train rows with an action")
    q_emb = embedder.encode([q["context_text"] for q in queries], batch_size=64)

    result: dict = {"embedder": getattr(embedder, "model_id", args.embedder), "depth": args.depth,
                    "n_queries": n_queries, "channels": {}, "hnsw": [], "filtering": {}}
    result["channels"] = run_channels(index, E, embedder, queries, q_emb, args.depth)
    need_hnsw = not (args.skip_hnsw and args.skip_filter)
    hnsw = load_hnsw(index) if need_hnsw else None
    if need_hnsw and hnsw is None:
        print("warning: state_features.hnsw not found; skipping HNSW-based results")
    if not args.skip_hnsw:
        result["hnsw"] = run_hnsw(index, hnsw, [q["vec"] for q in queries])
    if not args.skip_filter:
        result["filtering"] = run_filtering(index, hnsw, queries)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    md = render_markdown(result)
    out_path.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out_path} and {out_path.with_suffix('.md')}")
    return result


if __name__ == "__main__":
    main()
