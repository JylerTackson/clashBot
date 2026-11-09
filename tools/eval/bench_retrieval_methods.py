"""MAC-AU Part C: every retrieval method we have for the KB (pro-state database) and the KG, on the held-out test lines.

  python3 tools/eval/bench_retrieval_methods.py lines                    # verify / write runs/bench/au/lines.jsonl
  python3 tools/eval/bench_retrieval_methods.py embed                    # bge query embeddings        (heavy lock)
  python3 tools/eval/bench_retrieval_methods.py kge --models RotatE      # retrain one KGE             (heavy lock)
  python3 tools/eval/bench_retrieval_methods.py accuracy                 # untimed accuracy metrics    (CPU)
  python3 tools/eval/bench_retrieval_methods.py latency                  # timed query passes          (heavy lock)
  python3 tools/eval/bench_retrieval_methods.py report                   # docs/data/retrieval_methods_v1.{json,md}

Lines are the rows of `dataset/v1/d1/joint_tac_v2/test.jsonl` (Amendment 1: all 766, 27 test games). Retrieval is restricted
to the TRAIN split (the index holds every split); KG rules come from train events only. Every number is reproducible from
`runs/bench/au/retrieval/` by `report`, which is cheap and rerunnable once Part B's d1 predictions exist. Read-only use of
clashbot/retrieval.py, features.py, text_embed.py, kg.py, kg2.py, candidates.py and the tools/eval modules named below
(D-051 code freeze); nothing is written to dataset/ or db/.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.candidates import CandidateGenerator  # noqa: E402
from clashbot.config import DATASET_DIR, DB_PATH, HOLD_KEY, KG2_PATH  # noqa: E402
from clashbot.features import featurize  # noqa: E402
from clashbot.kg import KnowledgeGraph  # noqa: E402
from clashbot.kg2 import KG2  # noqa: E402
from clashbot.retrieval import StateIndex  # noqa: E402
from clashbot.text_embed import (DEFAULT_TEXT_MODEL, TextEmbedder, build_state_text_matrix,  # noqa: E402
                                 text_knn)
from tools.eval.eval_answer_rules import (SOURCES, GameBootstrap, combine_sources, counters_only,  # noqa: E402
                                          decks_of, eval_rows, git_head, leak_check, popularity_counts,
                                          read_jsonl, tie_metrics)
from tools.eval.eval_retrieval import (K_RRF, card_hit_at_k, card_of, hit_at_k, latency_stats,  # noqa: E402
                                       load_hnsw, recall_overlap)
from tools.eval.kge_link_prediction import (CARD_PREFIX, PykeenScorer, build_splits, load_triples,  # noqa: E402
                                            make_factory, train_model)

AU_DIR = ROOT / "runs" / "bench" / "au"
LINES_PATH = AU_DIR / "lines.jsonl"
PREDS_DIR = AU_DIR / "preds"
OUT_DIR = AU_DIR / "retrieval"
KGE_CONFIG = ROOT / "docs" / "data" / "kge_link_prediction_v1.json"
D1_EVAL = ROOT / "docs" / "data" / "d1_finetune_joint_tac_eval_v2.json"
DOC_OUT = ROOT / "docs" / "data" / "retrieval_methods_v1.json"
D1_ID = "d1-ft-joint-tac"

TOP = 20                         # hits per query
DEPTH = 50                       # per-channel depth fed to RRF (the live bot's CandidateGenerator uses 50)
POOL = {"splits": ["train"]}     # leakage filter: the index holds all splits, retrieval sees train only
EF_SWEEP = (10, 40, 160)
WARMUP = 10
BOOT = 10_000
SEED = 0
KGE_MODELS = ("RotatE", "TransE")
KGE_SETTING = "full"
KGE_SEED = 0
DECIMALS = 9                     # scores equal to this many decimals are tied

KB_METHODS = ("dense", "bm25", "rrf_dense_bm25", "bge", "rrf_all3", "hnsw_ef10", "hnsw_ef40", "hnsw_ef160",
              "deck_dense")
KB_LABELS = {
    "dense": "dense exact (644-d structured, dot)",
    "bm25": "BM25 (FTS5 over context_text)",
    "rrf_dense_bm25": "RRF dense + BM25 (live bot)",
    "bge": "bge-small text embeddings (cosine)",
    "rrf_all3": "RRF dense + BM25 + bge",
    "hnsw_ef10": "HNSW ef 10 (dense, train-filtered)",
    "hnsw_ef40": "HNSW ef 40 (dense, train-filtered)",
    "hnsw_ef160": "HNSW ef 160 (dense, train-filtered)",
    "deck_dense": "deck pre-filter, then dense"}
KG_METHODS = ("counter_scores_v1", "wiki_counters", "full", "only:pro", "only:role_transfer", "only:wiki", "only:sim",
              "kge_rotate", "kge_transe", "popularity", "random")
KG_LABELS = {
    "counter_scores_v1": "KG v1 counter_scores (wiki COUNTERS + mined ANSWERED_BY)",
    "wiki_counters": "KG v1 wiki COUNTERS only",
    "full": "KG v2 answers_for (full)",
    "only:pro": "KG v2 pro source only",
    "only:role_transfer": "KG v2 role transfer only",
    "only:wiki": "KG v2 wiki only",
    "only:sim": "KG v2 simulation (STOPS) only",
    "kge_rotate": "KGE RotatE (card COUNTERS threat)",
    "kge_transe": "KGE TransE (card COUNTERS threat)",
    "popularity": "train popularity of the answer card",
    "random": "random (expected)"}
KGE_FOR = {"kge_rotate": "RotatE", "kge_transe": "TransE"}


# ------------------------------------------------------------------------------------------------ lines
def build_lines(dataset: Path) -> list[dict]:
    """The decision lines from `<dataset>/d1/joint_tac_v2/test.jsonl`, in file order (the same rule as Part B's file):
    line_id = state_id = the row id, `keys` = the 4 option keys in the order d1 sees them, `gold` = index of the pro's."""
    out = []
    for r in read_jsonl(Path(dataset) / "d1" / "joint_tac_v2" / "test.jsonl"):
        keys = list(r["questions"]["play"]["criteria"])
        out.append({"line_id": r["id"], "game_id": r["game_id"], "state_id": r["id"], "kind": r["kind"],
                    "gold": keys.index(r["answers"]["play"]), "keys": keys})
    return out


def verify_lines(have: list[dict], built: list[dict]) -> list[str]:
    """Differences between Part B's line file and the one built here (empty = identical on the shared fields)."""
    diffs = []
    if len(have) != len(built):
        diffs.append(f"line count {len(have)} vs {len(built)}")
    for i, (a, b) in enumerate(zip(have, built)):
        for f in ("line_id", "game_id", "state_id", "gold", "keys"):
            if a.get(f) != b.get(f):
                diffs.append(f"row {i} {f}: {a.get(f)!r} vs {b.get(f)!r}")
    return diffs[:20]


def load_lines(path: Path, dataset: Path) -> tuple[list[dict], dict]:
    """Part B's `lines.jsonl` when present (checked against the dataset), else built from the dataset and written."""
    built = build_lines(dataset)
    if len({l["line_id"] for l in built}) != len(built):
        raise ValueError("line ids are not unique")
    path = Path(path)
    if path.exists():
        have = read_jsonl(path)
        diffs = verify_lines(have, built)
        if diffs:
            raise ValueError("lines.jsonl differs from the dataset rows: " + "; ".join(diffs))
        return have, {"source": str(path), "identical_to_dataset_rows": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(l) + "\n" for l in built))
    return built, {"source": str(path), "identical_to_dataset_rows": True, "written_by": "bench_retrieval_methods"}


def load_queries(lines: list[dict], dataset: Path, index: StateIndex) -> list[dict]:
    """Each line plus its dataset state (dict), t, retrieval text, index row and deck key."""
    wanted = {l["state_id"] for l in lines}
    found: dict[str, dict] = {}
    with open(Path(dataset) / "states.jsonl", encoding="utf-8") as f:
        for raw in f:
            if raw.strip() and len(found) < len(wanted):
                r = json.loads(raw)
                if r["id"] in wanted:
                    found[r["id"]] = r
    row_of = {sid: i for i, sid in enumerate(index.id)}
    out = []
    for l in lines:
        r = found[l["state_id"]]
        row = row_of[l["state_id"]]
        out.append({**l, "state": r["state"], "t": r["t"], "context_text": r.get("context_text") or "",
                    "row": row, "deck_key": index.own_deck_key[row],
                    "own_deck": list((r["state"].get("own") or {}).get("deck") or [])})
    return out


# ------------------------------------------------------------------------------------------------ leakage
class LeakageError(RuntimeError):
    pass


def check_leakage(index: StateIndex, rows: Sequence[int], allowed_splits: Sequence[str],
                  forbidden_games: set[str]) -> None:
    """Every retrieved row must be in an allowed split and from a game that is not one of the evaluated (test) games."""
    for r in rows:
        if index.split[r] not in allowed_splits:
            raise LeakageError(f"row {r} ({index.id[r]}) is in split {index.split[r]!r}, allowed {list(allowed_splits)}")
        if index.game_id[r] in forbidden_games:
            raise LeakageError(f"row {r} ({index.id[r]}) is from an evaluated game")


def drop_nontrain_decks(triples: list[tuple], games: list[dict]) -> list[tuple]:
    """KGE training graph, train only: drop the triples of deck nodes (IN_DECK, HAS_ARCHETYPE) whose deck was only played
    in val / test games. Decks that also occur in a train game stay. COUNTERS / SYNERGY / wiki edges are not game events."""
    train_keys = {g["own_deck_key"] for g in games if g["split"] == "train"}
    held_keys = {g["own_deck_key"] for g in games if g["split"] != "train"} - train_keys
    bad = {"deck:" + k for k in held_keys}
    return [t for t in triples if t[0] not in bad and t[2] not in bad]


# ------------------------------------------------------------------------------------------------ metrics
def mrr_exact(hit_keys: Sequence[str | None], target: str | None) -> float:
    """Reciprocal rank of the first hit whose action key equals the target ("hold" never matches); 0 when absent."""
    if not target or target == HOLD_KEY:
        return 0.0
    for r, key in enumerate(hit_keys, start=1):
        if key == target:
            return 1.0 / r
    return 0.0


def option_scores(options: Sequence[str], hit_keys: Sequence[str | None], hit_scores: Sequence[float]) -> list[float]:
    """Score of each option key = the summed retrieval scores of the hits whose action key equals it."""
    total: dict[str, float] = {}
    for key, s in zip(hit_keys, hit_scores):
        if key is not None:
            total[key] = total.get(key, 0.0) + float(s)
    return [total.get(o, 0.0) for o in options]


def decide(scores: Sequence[float], keys: Sequence[str], gold: int, seed_text: str) -> dict:
    """Pick the best-scoring option, ties in expectation (uniform among the tied top options).

    credit = P(pick is the gold option), card_credit = P(pick has the gold card); supported = some option has a positive
    score; pred = one seeded draw among the tied options (for the exact McNemar test)."""
    sc = [round(float(s), DECIMALS) for s in scores]
    top = max(sc)
    tied = [i for i, s in enumerate(sc) if s == top]
    gold_card = card_of(keys[gold])
    return {"credit": (1.0 if gold in tied else 0.0) / len(tied),
            "card_credit": sum(card_of(keys[i]) == gold_card for i in tied) / len(tied),
            "supported": bool(top > 0), "n_tied": len(tied), "pred": random.Random(seed_text).choice(tied)}


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test (binomial on the discordant pairs) of a against b on the same lines."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int(np.sum(a & ~b))                 # only a right
    n10 = int(np.sum(~a & b))                 # only b right
    n = n01 + n10
    if n == 0:
        return {"a_only": n01, "b_only": n10, "p": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return {"a_only": n01, "b_only": n10, "p": float(min(1.0, p))}


def latency_row(ms: Sequence[float]) -> dict:
    out = latency_stats(list(ms))
    out["ms_mean"] = round(float(np.mean(ms)), 3) if len(ms) else None
    out["n"] = len(ms)
    return out


# ------------------------------------------------------------------------------------------------ KB methods
def hnsw_filtered(h, vec: np.ndarray, allowed: np.ndarray, k: int, ef: int) -> list[tuple[int, float]]:
    """HNSW top-k over the allowed rows, the filter applied inside the graph search (so ef controls the quality of the
    train-only answer; hnswlib raises ef to k when ef < k, so ef 10 behaves like ef 20)."""
    h.set_ef(ef)
    k = min(k, int(np.count_nonzero(allowed)))
    while k > 0:
        try:
            labels, dists = h.knn_query(np.asarray(vec, dtype=np.float32).reshape(1, -1), k=k,
                                        filter=lambda i: bool(allowed[i]))
        except RuntimeError:                   # hnswlib could not reach k allowed neighbours: ask for fewer
            k //= 2
            continue
        return [(int(lab), float(1.0 - d)) for lab, d in zip(labels[0], dists[0])]
    return []


def kb_methods(index: StateIndex, hnsw, E: np.ndarray, embed: Callable[[str], np.ndarray] | None = None
               ) -> dict[str, Callable[[dict], list[tuple[int, float]]]]:
    """name -> query function(q) -> top-TOP [(row, score)], each doing everything the live bot would per query
    (featurize / encode included). `embed` None = use the precomputed q["emb"] (accuracy pass); else encode per query."""
    def emb(q):
        return embed(q["context_text"]) if embed is not None else q["emb"]

    def dense(q, **extra):
        return index.knn(featurize(q["state"], q["t"]), TOP, **POOL, **extra)

    def rrf3(q):
        lists = {"knn": index.knn(featurize(q["state"], q["t"]), DEPTH, **POOL),
                 "bm25": index.bm25(q["context_text"], DEPTH, **POOL),
                 "text": text_knn(E, emb(q), index.mask(**POOL), DEPTH)}
        return [(row, score) for row, score, _ranks in index.rrf(lists, k_rrf=K_RRF)[:TOP]]

    def live(q):
        return [(h.row, h.score) for h in index.search(q["state"], q["t"], q["context_text"], k=TOP, depth=DEPTH, **POOL)]

    def hnsw_fn(ef):
        return lambda q: hnsw_filtered(hnsw, featurize(q["state"], q["t"]), index.mask(**POOL), TOP, ef)

    fns = {"dense": dense,
           # the FTS query takes the best 10 x k rows over ALL splits before the train filter, so ask at the live channel's
           # depth (50 -> 500 rows) and keep the top 20; k = 20 alone leaves 28% of the lines with no train hit
           "bm25": lambda q: index.bm25(q["context_text"], DEPTH, **POOL)[:TOP],
           "rrf_dense_bm25": live,
           "bge": lambda q: text_knn(E, emb(q), index.mask(**POOL), TOP),
           "rrf_all3": rrf3,
           "deck_dense": lambda q: dense(q, deck_key=q["deck_key"]) if q["deck_key"] is not None else []}
    if hnsw is not None:
        for ef in EF_SWEEP:
            fns[f"hnsw_ef{ef}"] = hnsw_fn(ef)
    return {m: fns[m] for m in KB_METHODS if m in fns}


def kb_line_record(method: str, q: dict, hits: list[tuple[int, float]], index: StateIndex,
                   forbidden: set[str], exact_rows: list[int] | None) -> dict:
    rows = [r for r, _ in hits]
    check_leakage(index, rows, POOL["splits"], forbidden)
    keys = [index.action_key[r] for r in rows]
    gold_key = q["keys"][q["gold"]]
    rec = {"method": method, "line_id": q["line_id"], "game_id": q["game_id"], "gold": q["gold"],
           "hit@1": hit_at_k(keys, gold_key, 1), "hit@10": hit_at_k(keys, gold_key, 10),
           "card_hit@10": card_hit_at_k(keys, gold_key, 10), "mrr": mrr_exact(keys, gold_key), "n_hits": len(rows),
           **decide(option_scores(q["keys"], keys, [s for _, s in hits]), q["keys"], q["gold"],
                    f"{method}|{q['line_id']}")}
    if exact_rows is not None:
        rec["recall@20"] = recall_overlap(rows, exact_rows, TOP)
    return rec


# ------------------------------------------------------------------------------------------------ KG methods
def line_threats(state: dict, t: float) -> list[str]:
    """The line's recent enemy threats, as the live bot sees them: opponent plays of the last 5 s and enemy units heading
    to our towers within 8 s (`CandidateGenerator._threats`, which does not use `self`), distinct cards, in order."""
    return list(dict.fromkeys(card for card, _tile in CandidateGenerator._threats(None, state, t)))


def shift_scores(raw: dict[str, float]) -> dict[str, float]:
    """Order-preserving positive scores for a dense (possibly negative) scorer: raw - min + 1 over the scored cards;
    a card the scorer does not know keeps the implicit 0 and ranks last."""
    if not raw:
        return {}
    lo = min(raw.values())
    return {c: v - lo + 1.0 for c, v in raw.items()}


class KGEScorer:
    """A trained PyKEEN model: scores (card, COUNTERS, threat) for a deck, shifted positive (see `shift_scores`)."""

    def __init__(self, scorer: PykeenScorer, candidates: Sequence[str]):
        self.scorer = scorer
        self.cidx = {c: i for i, c in enumerate(candidates)}

    def __call__(self, threat: str, deck: Sequence[str]) -> dict[str, float]:
        node = CARD_PREFIX + threat
        if node not in self.scorer.ent:
            return {}
        s = self.scorer.score_heads([node])[0]
        raw = {c: float(s[self.cidx[CARD_PREFIX + c]]) for c in dict.fromkeys(deck) if CARD_PREFIX + c in self.cidx}
        return shift_scores(raw)


def kg_methods(kg2: KG2, kg1: KnowledgeGraph, pop: Counter, kge: dict[str, KGEScorer]
               ) -> dict[str, Callable[[str, list[str]], dict[str, float]]]:
    """name -> score(threat, deck) -> {card: score >= 0} (higher is better, a card with no evidence is absent)."""
    def v2(weights_of):
        def fn(threat, deck):
            answers = kg2.answers_for(threat, deck, k=max(len(deck), 1))
            return combine_sources(answers, weights_of())
        return fn

    full_w = lambda: {s: kg2.source_weights[s] for s in SOURCES}                     # noqa: E731
    fns: dict[str, Callable] = {
        "counter_scores_v1": lambda threat, deck: kg1.counter_scores(threat),
        "wiki_counters": lambda threat, deck: counters_only(kg1, threat),
        "full": v2(full_w),
        "popularity": lambda threat, deck: {c: float(pop.get(c, 0)) for c in deck},
        "random": lambda threat, deck: {}}
    for s in SOURCES:
        fns[f"only:{s}"] = v2(lambda s=s: {s: kg2.source_weights[s]})
    for name in KGE_FOR:
        if name in kge:
            fns[name] = kge[name]
    return {m: fns[m] for m in KG_METHODS if m in fns}


def kg_line_record(method: str, fn, q: dict, threats: list[str]) -> dict:
    """4-way decision: each option's score is the summed method score of its card against every threat of the line. A
    line with no threat leaves all four options tied (expected 0.25)."""
    deck = list(dict.fromkeys(list(q["own_deck"]) + [card_of(k) for k in q["keys"]]))
    option_cards = [card_of(k) for k in q["keys"]]
    totals = [0.0] * len(q["keys"])
    for threat in threats:
        sc = fn(threat, deck)
        for i, c in enumerate(option_cards):
            totals[i] += sc.get(c, 0.0)
    return {"method": method, "line_id": q["line_id"], "game_id": q["game_id"], "gold": q["gold"],
            "n_threats": len(threats), **decide(totals, q["keys"], q["gold"], f"{method}|{q['line_id']}")}


# ------------------------------------------------------------------------------------------------ stages
def paths(args) -> dict[str, Path]:
    out = Path(args.out_dir)
    return {"out": out, "emb": out / "query_emb.npy", "emb_meta": out / "query_emb.meta.json",
            "kb": out / "kb_per_line.jsonl", "kg_line": out / "kg_per_line.jsonl",
            "kg_event": out / "kg_per_event.jsonl", "acc_meta": out / "accuracy_meta.json",
            "latency": out / "latency.json"}


def lines_for(args) -> tuple[list[dict], dict]:
    lines, info = load_lines(Path(args.lines), Path(args.dataset))
    if args.limit:
        lines = lines[:args.limit]
    info["n_lines"] = len(lines)
    info["n_games"] = len({l["game_id"] for l in lines})
    return lines, info


def read_text_matrix(db_path: Path) -> np.ndarray:
    """db/state_text_emb.npy through `build_state_text_matrix`, but never re-encoding or writing into db/."""
    class CacheOnly:
        model_id = DEFAULT_TEXT_MODEL

        def encode(self, texts, batch_size=64):
            raise RuntimeError("db/state_text_emb.npy is stale: refusing to re-encode or write into db/")

    return build_state_text_matrix(db_path, CacheOnly())


def stage_lines(args) -> None:
    lines, info = lines_for(args)
    print(json.dumps(info), f"games={info['n_games']}")


def stage_embed(args) -> None:
    """bge embeddings of the lines' retrieval texts (heavy lock). Sanity: they match the db rows of the same states."""
    P = paths(args)
    P["out"].mkdir(parents=True, exist_ok=True)
    index = StateIndex(args.db)
    lines, _ = lines_for(args)
    queries = load_queries(lines, args.dataset, index)
    E = read_text_matrix(args.db)
    emb = TextEmbedder().encode([q["context_text"] for q in queries], batch_size=64)
    cos = np.einsum("ij,ij->i", emb, E[[q["row"] for q in queries]])
    np.save(P["emb"], emb)
    P["emb_meta"].write_text(json.dumps({
        "model": DEFAULT_TEXT_MODEL, "n": len(queries), "line_ids": [q["line_id"] for q in queries],
        "min_cosine_vs_db_row": float(cos.min()), "mean_cosine_vs_db_row": float(cos.mean())}))
    print(f"embedded {len(queries)} texts; cosine vs db rows min {cos.min():.5f} mean {cos.mean():.5f}")


def kge_graph(dataset: Path):
    """(training triples, candidates, factory) of the final KGE: the 'full' setting's graph, train decks only, trained on
    every COUNTERS edge (no held-out part: these edges are wiki facts, not game events)."""
    triples = drop_nontrain_decks(load_triples(dataset), read_jsonl(dataset / "games.jsonl"))
    sp = build_splits(triples, KGE_SETTING)
    train = sorted(set(sp.train) | sp.all_counters)
    candidates = sorted({x for h, _, t in train for x in (h, t) if x.startswith(CARD_PREFIX)})
    return train, candidates, make_factory(dataclasses.replace(sp, train=train, candidates=candidates))


def stage_kge(args) -> None:
    """Retrain the named KGE models with the best config of kge_link_prediction_v1.json (heavy lock, one per call)."""
    import torch
    P = paths(args)
    P["out"].mkdir(parents=True, exist_ok=True)
    cfg = json.loads(KGE_CONFIG.read_text())
    epochs = cfg["counts"]["epochs"]
    train, candidates, tf = kge_graph(Path(args.dataset))
    print(f"KGE graph: {len(train)} train triples, {len(candidates)} card candidates")
    for name in [m for m in args.models.split(",") if m]:
        best = cfg["settings"][KGE_SETTING]["models"][name]["best"]
        model, secs = train_model(name, tf, best["dim"], best["lr"], epochs, KGE_SEED)
        torch.save(model, P["out"] / f"kge_{name}.pt")
        (P["out"] / f"kge_{name}.meta.json").write_text(json.dumps({
            "model": name, "setting": KGE_SETTING, "dim": best["dim"], "lr": best["lr"], "epochs": epochs,
            "seed": KGE_SEED, "train_s": round(secs, 1), "n_train_triples": len(train), "candidates": candidates,
            "entity_to_id": tf.entity_to_id, "relation_to_id": tf.relation_to_id}))
        print(f"{name}: dim {best['dim']} lr {best['lr']} epochs {epochs}, trained in {secs:.1f}s")


def load_kge(out_dir: Path) -> dict[str, KGEScorer]:
    import torch
    from types import SimpleNamespace
    found = {}
    for key, name in KGE_FOR.items():
        meta_path, model_path = out_dir / f"kge_{name}.meta.json", out_dir / f"kge_{name}.pt"
        if not (meta_path.exists() and model_path.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        tf = SimpleNamespace(entity_to_id=meta["entity_to_id"], relation_to_id=meta["relation_to_id"])
        model = torch.load(model_path, weights_only=False, map_location="cpu")
        found[key] = KGEScorer(PykeenScorer(model, tf, meta["candidates"]), meta["candidates"])
    return found


def load_kg_inputs(args):
    ds = Path(args.dataset)
    games = read_jsonl(ds / "games.jsonl")
    events_all = read_jsonl(ds / "kg2" / "answer_events.jsonl")
    shipped = read_jsonl(ds / "kg2" / "answers.jsonl")
    return games, events_all, shipped


def stage_accuracy(args) -> None:
    P = paths(args)
    P["out"].mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    index = StateIndex(args.db)
    hnsw = load_hnsw(index)
    E = read_text_matrix(args.db)
    lines, line_info = lines_for(args)
    queries = load_queries(lines, args.dataset, index)
    meta = json.loads(P["emb_meta"].read_text())
    emb = np.load(P["emb"])
    pos = {lid: i for i, lid in enumerate(meta["line_ids"])}
    for q in queries:
        q["emb"] = emb[pos[q["line_id"]]]
    forbidden = {l["game_id"] for l in lines}

    # ---- KB
    fns = kb_methods(index, hnsw, E)
    kb_rows = []
    for q in queries:
        exact_rows = None
        for m, fn in fns.items():
            hits = fn(q)
            if m == "dense":
                exact_rows = [r for r, _ in hits]
            kb_rows.append(kb_line_record(m, q, hits, index, forbidden,
                                          exact_rows if m.startswith("hnsw") else None))
    P["kb"].write_text("".join(json.dumps(r) + "\n" for r in kb_rows))

    # ---- KG
    games, events_all, shipped = load_kg_inputs(args)
    leak = leak_check(events_all, shipped)
    if not (leak["rules_all_events_equal_train_only"] and leak["rules_equal_shipped_answers_jsonl"]):
        raise LeakageError(f"KG rules are not train-only: {leak}")
    decks = decks_of(games)
    pop = popularity_counts(events_all)                            # train events only
    test_games = {l["game_id"] for l in lines}
    events = [e for e in events_all if e["split"] == "test" and e["game_id"] in test_games]
    kg2 = KG2(args.kg2)
    kg1 = KnowledgeGraph(args.db)
    kge = load_kge(P["out"])
    kfns = kg_methods(kg2, kg1, pop, kge)
    event_rows, skipped, wiring = [], Counter(), 0.0
    for e in events:
        res = eval_rows(kg2, [e], decks, pop, kg1)
        if not res["rows"]:
            skipped.update(res["skipped"])
            continue
        row = res["rows"][0]
        deck = list(dict.fromkeys(decks[e["game_id"]]))
        for name in ("kge_rotate", "kge_transe"):
            if name in kfns:
                row["m"][name] = tie_metrics(kfns[name](e["threat"], deck), deck, e["answer"])
        for name, fn in kfns.items():                              # wiring check: our callables == eval_rows
            if name in row["m"] and not name.startswith("kge") and name != "random":
                mine = tie_metrics(fn(e["threat"], deck), deck, e["answer"])
                wiring = max(wiring, abs(mine["hit1"] - row["m"][name]["hit1"]), abs(mine["rr"] - row["m"][name]["rr"]))
        event_rows.append({"game_id": e["game_id"], "threat": e["threat"], "answer": e["answer"],
                           "m": {n: {k: row["m"][n][k] for k in ("hit1", "hit3", "rr", "covered")}
                                 for n in KG_METHODS if n in row["m"]}})
    P["kg_event"].write_text("".join(json.dumps(r) + "\n" for r in event_rows))
    kg_rows = []
    for q in queries:
        threats = line_threats(q["state"], q["t"])
        for m, fn in kfns.items():
            kg_rows.append(kg_line_record(m, fn, q, threats))
    P["kg_line"].write_text("".join(json.dumps(r) + "\n" for r in kg_rows))

    train_pool = int(index.mask(**POOL).sum())
    P["acc_meta"].write_text(json.dumps({
        "lines": line_info, "pool": {"splits": POOL["splits"], "states_with_action": train_pool,
                                     "index_states": int(index.n), "split_counts": dict(Counter(index.split))},
        "leak_check": leak, "kg_events": {"n_events": len(events), "n_scored": len(event_rows), "skipped": dict(skipped),
                                          "games": len({r["game_id"] for r in event_rows}),
                                          "wiring_max_abs_diff_vs_eval_rows": wiring},
        "kg_lines": {"with_threat": sum(1 for q in queries if line_threats(q["state"], q["t"])), "n": len(queries)},
        "methods": {"kb": list(fns), "kg": list(kfns)}, "seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"accuracy done in {time.time() - t0:.1f}s: {len(kb_rows)} KB rows, {len(kg_rows)} KG rows, "
          f"{len(event_rows)} events, wiring diff {wiring:.2e}")


def time_calls(fn, items: list, warmup: int = WARMUP) -> list[float]:
    for it in items[:warmup]:
        fn(it)
    out = []
    for it in items:
        t0 = time.perf_counter()
        fn(it)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def stage_latency(args) -> None:
    """Timed query passes (heavy lock held by the caller): wall time per query after a warm-up, one method at a time."""
    P = paths(args)
    t0 = time.time()
    index = StateIndex(args.db)
    hnsw = load_hnsw(index)
    E = read_text_matrix(args.db)
    lines, _ = lines_for(args)
    queries = load_queries(lines, args.dataset, index)
    embedder = TextEmbedder()
    embedder.encode(["warm up"] * 3)
    embed = lambda text: embedder.encode([text])[0]                # noqa: E731  (a single-query encode, as the live loop)
    fns = kb_methods(index, hnsw, E, embed)
    result: dict = {"kb": {}, "kg": {}, "env": {"python": platform.python_version(), "machine": platform.machine(),
                                                "platform": platform.platform(), "warmup": WARMUP,
                                                "note": "other CPU jobs (API clients) may have run; latencies are upper bounds"}}
    for m, fn in fns.items():
        query, decision = [], []
        for q in queries[:WARMUP]:
            fn(q)
        for q in queries:
            a = time.perf_counter()
            hits = fn(q)
            b = time.perf_counter()
            option_scores(q["keys"], [index.action_key[r] for r, _ in hits], [s for _, s in hits])
            c = time.perf_counter()
            query.append((b - a) * 1000.0)
            decision.append((c - a) * 1000.0)
        result["kb"][m] = {"query": latency_row(query), "decision": latency_row(decision)}
        print(f"KB {m}: p50 {result['kb'][m]['query']['ms_p50']} ms", flush=True)

    games, events_all, shipped = load_kg_inputs(args)
    decks = decks_of(games)
    pop = popularity_counts(events_all)
    test_games = {l["game_id"] for l in lines}
    events = [e for e in events_all if e["split"] == "test" and e["game_id"] in test_games
              and e["answer"] in decks.get(e["game_id"], [])]
    kg2, kg1 = KG2(args.kg2), KnowledgeGraph(args.db)
    kfns = kg_methods(kg2, kg1, pop, load_kge(P["out"]))
    ev_items = [(e["threat"], list(dict.fromkeys(decks[e["game_id"]]))) for e in events]
    for m, fn in kfns.items():
        if m == "random":
            continue
        query = time_calls(lambda it: fn(*it), ev_items)
        decision = time_calls(lambda q: kg_line_record(m, fn, q, line_threats(q["state"], q["t"])), queries)
        result["kg"][m] = {"query": latency_row(query), "decision": latency_row(decision)}
        print(f"KG {m}: query p50 {result['kg'][m]['query']['ms_p50']} ms, decision p50 "
              f"{result['kg'][m]['decision']['ms_p50']} ms", flush=True)
    result["seconds"] = round(time.time() - t0, 1)
    P["latency"].write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------------------------------------ report
def d1_correct(line_ids: list[str], golds: dict[str, int], keys: dict[str, list[str]]) -> tuple[dict | None, str]:
    """Part B's d1-ft-joint-tac predictions per line (+ latency), else (provisional) the per-example record of the
    d1_finetune_joint_tac_eval_v2.json run (accuracy only). Returns ({line_id: {pred, latency_ms}}, source)."""
    path = PREDS_DIR / f"{D1_ID}.jsonl"
    if path.exists():
        rows = {r["line_id"]: r for r in read_jsonl(path)}
        if all(l in rows for l in line_ids):
            return {l: {"pred": rows[l]["pred"], "latency_ms": rows[l].get("latency_ms")} for l in line_ids}, \
                f"runs/bench/au/preds/{D1_ID}.jsonl (Part B)"
    if D1_EVAL.exists():
        per = {r["id"]: r for r in json.loads(D1_EVAL.read_text())["per_example"] if r.get("split") == "test"}
        if all(l in per for l in line_ids):
            return ({l: {"pred": keys[l].index(per[l]["pred"]) if per[l]["pred"] in keys[l] else None,
                         "latency_ms": None} for l in line_ids},
                    "docs/data/d1_finetune_joint_tac_eval_v2.json per_example (provisional: Part B's preds not found; "
                    "no latency)")
    return None, "missing"


def decision_summary(rows: list[dict], boot: GameBootstrap, d1: dict | None, golds: dict[str, int]) -> dict:
    cred = np.array([r["credit"] for r in rows], float)
    card = np.array([r["card_credit"] for r in rows], float)
    sup = np.array([r["supported"] for r in rows], float)
    out = {"accuracy": boot.ci(cred), "card_accuracy": boot.ci(card), "coverage": boot.ci(sup),
           "accuracy_supported": boot.ci(cred, mask=sup), "accuracy_unsupported": boot.ci(cred, mask=1 - sup)}
    if d1 is not None:
        mine = [r["pred"] == r["gold"] for r in rows]
        theirs = [d1[r["line_id"]]["pred"] == r["gold"] for r in rows]
        out["mcnemar_vs_d1"] = mcnemar_exact(mine, theirs)
        out["delta_vs_d1"] = boot.diff(cred, np.array(theirs, float))
    return out


def render_markdown(res: dict) -> str:
    def ci(d, nd=3):
        if not d or d.get("n", 1) == 0 or d["mean"] != d["mean"]:
            return "-"
        return f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"

    def lat(d):
        return "-" if not d else f"{d['ms_p50']:.2f} / {d['ms_p95']:.2f}"

    def mc(m):
        return "-" if not m else f"{m['p']:.3g}"

    L = ["# Retrieval methods for the KB and the KG, benchmarked as retrievers and as decision models (MAC-AU part C)", ""]
    g = res["git"]
    L += [f"Run at HEAD `{(g.get('head') or '?')[:10]}`{' (+ uncommitted changes)' if g.get('dirty') else ''}. "
          f"{res['lines']['n_lines']} decision lines of `dataset/v1/d1/joint_tac_v2/test.jsonl` from "
          f"{res['lines']['n_games']} test games ({res['lines']['n_videos']} videos); CIs are a cluster bootstrap over games "
          f"({res['bootstrap']} resamples, seed {res['seed']}). Chance on a 4-way line is 0.25.", ""]
    best_kb = max(res["kb"], key=lambda m: res["kb"][m]["decision"]["accuracy"]["mean"])
    best_kg = max((m for m in res["kg"] if m != "random"), key=lambda m: res["kg"][m]["decision"]["accuracy"]["mean"])
    best_ev = max((m for m in res["kg"] if m != "random"), key=lambda m: res["kg"][m]["mrr"]["mean"])
    d1a = res["d1"].get("accuracy")
    L += ["## Headline", "",
          f"- Best KB method as a decision model: {KB_LABELS[best_kb]}, accuracy {ci(res['kb'][best_kb]['decision']['accuracy'])} "
          f"(coverage {res['kb'][best_kb]['decision']['coverage']['mean']:.2f}, query ms p50 / p95 {lat(res['kb'][best_kb]['latency'].get('query'))})"
          + (f"; d1-ft-joint-tac on the same lines {ci(d1a)}." if d1a else "."),
          f"- Best KG method as a decision model: {KG_LABELS[best_kg]}, accuracy {ci(res['kg'][best_kg]['decision']['accuracy'])}, "
          f"card accuracy {ci(res['kg'][best_kg]['decision']['card_accuracy'])} (random 0.250 / 0.500; the KG cannot choose a placement).",
          f"- As a threat-to-answer ranker the best KG method is {KG_LABELS[best_ev]} (MRR {ci(res['kg'][best_ev]['mrr'])}; "
          f"random {res['kg']['random']['mrr']['mean']:.3f}).", ""]
    kge = res.get("kge") or {}
    kge_text = "; ".join(f"{n} dim {v['dim']} lr {v['lr']}" for n, v in kge.items())
    L += ["## Setup and leakage", "",
          f"- KB retrieval is restricted to the **train** split (`splits=['train']`; {res['pool']['states_with_action']} of "
          f"{res['pool']['index_states']} indexed states are train states with an action). Every retrieved row was checked: "
          f"split in train and game not one of the {res['lines']['n_games']} evaluated games "
          f"({res['leakage']['kb_hits_checked']} hits checked, none violated). Splits are by video, so no other match of an "
          "evaluated video is in the pool.",
          f"- KG rules are the train-split answer rules: rules over all events equal rules over train events only "
          f"({res['leak_check']['rules_all_events_equal_train_only']}) and equal the shipped answers.jsonl "
          f"({res['leak_check']['rules_equal_shipped_answers_jsonl']}); popularity counts train events only. The v1 "
          "ANSWERED_BY edges are mined from train states. KGE: the graph of `kge_link_prediction_v1` (setting `full`) with the "
          "IN_DECK / HAS_ARCHETYPE edges of decks played only in val / test games removed, trained on every COUNTERS edge "
          f"(wiki facts), best config per model from that file ({kge_text}), seed {KGE_SEED}.",
          "- **Decision rule.** KB: an option's score is the sum of the retrieval scores of the top-20 hits whose action key equals "
          "the option key. KG: an option's score is the sum, over the line's recent enemy threats (opponent plays of the last 5 s "
          "and enemy units reaching our towers within 8 s), of the method's score for the option's card against the threat, "
          "taken over the own deck. Ties split uniformly (expected credit); a line where no option has a positive score is "
          "unsupported (all four tied, 0.25). McNemar uses one seeded tie-break draw per line.",
          "- **The KG cannot place.** The four options are a 2x2 grid (the pro's play, the same card elsewhere, another card at the "
          "pro's placement, another card elsewhere), so two options share each card. A card-level KG score can at best "
          "separate the two cards and then splits 50/50: expected accuracy is capped at 0.5 and random card accuracy is 0.5.",
          "- Latency is wall time per query after a 10-query warm-up, measured with the heavy lock held (so no local model runs "
          "alongside); other CPU jobs may still have run, so read the figures as upper bounds. KB queries include featurising "
          "and, for bge, one single-query encode on MPS.", ""]
    # KB table
    L += ["## KB: the pro-state database (top-20 hits per query)", "",
          "| method | hit@10 | card_hit@10 | MRR | 4-way acc | card acc | coverage | acc if supported | empty | recall@20 vs exact | ms p50 / p95 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in res["kb"]:
        r = res["kb"][m]
        L.append(f"| {KB_LABELS[m]} | {ci(r['hit@10'])} | {ci(r['card_hit@10'])} | {ci(r['mrr'])} | {ci(r['decision']['accuracy'])} | "
                 f"{ci(r['decision']['card_accuracy'])} | {ci(r['decision']['coverage'], 2)} | "
                 f"{ci(r['decision']['accuracy_supported'])} | {r['empty_rate']:.3f} | "
                 f"{'-' if r.get('recall@20') is None else format(r['recall@20'], '.3f')} | {lat(r['latency']['query'])} |")
    L += ["", "hit@10 / card_hit@10 / MRR: a retrieved state's pro play equals the line's gold play key (or has the gold card); "
          "empty = share of lines with no hit at all; recall@20 = overlap of the HNSW top-20 with the exact dense top-20 over the "
          "same train pool. HNSW applies the train filter inside the graph search; hnswlib raises ef to k = 20, so ef 10 equals ef 20.", ""]
    # KG table
    L += ["## KG: threat to answer (events: the pro's answer to a threat, ranked over the pro's deck)", "",
          f"Events: {res['kg_events']['n_scored']} threat to answer events of the {res['kg_events']['games']} test games "
          f"({res['kg_events']['skipped'] or 'none'} skipped because the answer is not in the recorded deck). The decision columns "
          f"use all {res['lines']['n_lines']} lines ({res['kg_lines']['with_threat']} have at least one threat).", "",
          "| method | hit@1 | hit@3 | MRR | event coverage | 4-way acc | card acc | line coverage | query ms p50 / p95 | decision ms p50 / p95 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m in res["kg"]:
        r = res["kg"][m]
        L.append(f"| {KG_LABELS[m]} | {ci(r['hit@1'])} | {ci(r['hit@3'])} | {ci(r['mrr'])} | "
                 f"{'-' if m == 'random' else ci(r['coverage'], 2)} | {ci(r['decision']['accuracy'])} | "
                 f"{ci(r['decision']['card_accuracy'])} | {'-' if m == 'random' else ci(r['decision']['coverage'], 2)} | "
                 f"{lat(r['latency'].get('query'))} | {lat(r['latency'].get('decision'))} |")
    L += ["", "Ties are expectations, so a method without evidence scores exactly like random (hit@1 about 1/8 on a deck of 8). "
          "Query = one threat scored over the deck; decision = threat extraction plus all threats for one line "
          f"({res['kg_lines']['n'] - res['kg_lines']['with_threat']} of {res['kg_lines']['n']} lines have no threat, so the "
          "decision p50 is mostly the cost of finding none; the p95 is a line with threats).", ""]
    # combined
    d1 = res["d1"]
    L += ["## Retrieval as a decision model, next to the d1 reference", "",
          f"d1-ft-joint-tac numbers from {d1['source']}.", "",
          "| method | 4-way acc [95% CI] | card acc | coverage | decision ms p50 / p95 | McNemar p vs d1 | delta vs d1 [CI] |",
          "|---|---|---|---|---|---|---|"]
    if d1.get("accuracy"):
        L.append(f"| **d1-ft-joint-tac (reference)** | {ci(d1['accuracy'])} | {ci(d1['card_accuracy'])} | 1.00 | "
                 f"{lat(d1.get('latency'))} | - | - |")
    else:
        L.append("| **d1-ft-joint-tac (reference)** | pending Part B preds | | | | | |")
    for kind, table in (("KB", "kb"), ("KG", "kg")):
        for m, r in res[table].items():
            dm = r["decision"]
            labels = KB_LABELS if kind == "KB" else KG_LABELS
            dl = r["latency"].get("decision")
            delta = dm.get("delta_vs_d1")
            L.append(f"| {kind}: {labels[m]} | {ci(dm['accuracy'])} | {ci(dm['card_accuracy'])} | "
                     f"{'-' if m == 'random' else format(dm['coverage']['mean'], '.2f')} | {lat(dl)} | "
                     f"{mc(dm.get('mcnemar_vs_d1'))} | {'-' if not delta else ci(delta)} |")
    L += ["", "## Notes", ""]
    L += [f"- {n}" for n in res["notes"]]
    L.append(f"\nRuntime of the accuracy pass {res['seconds']['accuracy']} s, latency pass {res['seconds']['latency']} s.")
    return "\n".join(L) + "\n"


def stage_report(args) -> dict:
    P = paths(args)
    lines, line_info = lines_for(args)
    acc_meta = json.loads(P["acc_meta"].read_text())
    lat_all = json.loads(P["latency"].read_text()) if P["latency"].exists() else {"kb": {}, "kg": {}, "env": {}}
    ids = [l["line_id"] for l in lines]
    gold = {l["line_id"]: l["gold"] for l in lines}
    keys = {l["line_id"]: l["keys"] for l in lines}
    game_of = {l["line_id"]: l["game_id"] for l in lines}
    boot = GameBootstrap([game_of[i] for i in ids], BOOT, SEED)
    d1_preds, d1_src = d1_correct(ids, gold, keys)
    d1 = {"source": d1_src}
    if d1_preds is not None:
        correct = np.array([d1_preds[i]["pred"] == gold[i] for i in ids], float)
        card_ok = np.array([d1_preds[i]["pred"] is not None and
                            card_of(keys[i][d1_preds[i]["pred"]]) == card_of(keys[i][gold[i]]) for i in ids], float)
        lats = [d1_preds[i]["latency_ms"] for i in ids if d1_preds[i]["latency_ms"] is not None]
        d1.update({"accuracy": boot.ci(correct), "card_accuracy": boot.ci(card_ok), "latency": latency_row(lats) if lats else None})

    def by_method(path: Path) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in read_jsonl(path):
            out.setdefault(r["method"], []).append(r)
        return out

    kb_rows = by_method(P["kb"])
    kb = {}
    for m, rows in kb_rows.items():
        assert [r["line_id"] for r in rows] == ids, "KB rows out of line order"
        b = GameBootstrap([r["game_id"] for r in rows], BOOT, SEED)
        r = {n: b.ci([x[n] for x in rows]) for n in ("hit@1", "hit@10", "card_hit@10", "mrr")}
        r["empty_rate"] = float(np.mean([x["n_hits"] == 0 for x in rows]))
        r["mean_hits"] = float(np.mean([x["n_hits"] for x in rows]))
        r["recall@20"] = float(np.mean([x["recall@20"] for x in rows])) if "recall@20" in rows[0] else None
        r["decision"] = decision_summary(rows, b, d1_preds, gold)
        r["latency"] = {k: lat_all["kb"].get(m, {}).get(k) for k in ("query", "decision")}
        kb[m] = r

    events = read_jsonl(P["kg_event"])
    eb = GameBootstrap([e["game_id"] for e in events], BOOT, SEED)
    kg_rows = by_method(P["kg_line"])
    kg = {}
    for m, rows in kg_rows.items():
        assert [r["line_id"] for r in rows] == ids, "KG rows out of line order"
        r = {"hit@1": eb.ci([e["m"][m]["hit1"] for e in events]), "hit@3": eb.ci([e["m"][m]["hit3"] for e in events]),
             "mrr": eb.ci([e["m"][m]["rr"] for e in events]), "coverage": eb.ci([e["m"][m]["covered"] for e in events])}
        cov = [e["m"][m]["covered"] for e in events]
        r["hit@1_covered"] = eb.ci([e["m"][m]["hit1"] for e in events], mask=cov)
        r["decision"] = decision_summary(rows, GameBootstrap([x["game_id"] for x in rows], BOOT, SEED), d1_preds, gold)
        r["latency"] = {k: lat_all["kg"].get(m, {}).get(k) for k in ("query", "decision")}
        kg[m] = r

    kge = {n: json.loads((P["out"] / f"kge_{n}.meta.json").read_text()) for n in KGE_MODELS
           if (P["out"] / f"kge_{n}.meta.json").exists()}
    for v in kge.values():
        v.pop("candidates", None), v.pop("entity_to_id", None), v.pop("relation_to_id", None)
    n_hits = sum(r["n_hits"] for rows in kb_rows.values() for r in rows)
    ev_meta = acc_meta["kg_events"]
    videos = {g.rsplit("-m", 1)[0] for g in game_of.values()}
    res = {
        "version": "retrieval_methods_v1", "experiment": "MAC-AU part C: retrieval methods as retrievers and as decision models",
        "git": git_head(), "bootstrap": BOOT, "seed": SEED,
        "lines": {**line_info, "n_videos": len(videos), "videos": sorted(videos)},
        "pool": acc_meta["pool"], "leak_check": acc_meta["leak_check"],
        "leakage": {"kb_hits_checked": n_hits, "violations": 0}, "kge": kge,
        "kg_events": ev_meta, "kg_lines": acc_meta["kg_lines"], "d1": d1, "kb": kb, "kg": kg,
        "latency_env": lat_all.get("env"), "seconds": {"accuracy": acc_meta["seconds"], "latency": lat_all.get("seconds")},
        "methods": {"kb": KB_LABELS, "kg": KG_LABELS},
        "notes": [
            "Only 5 distinct videos stand behind the 27 games, so game-level resampling still understates the uncertainty "
            "between videos; read the CIs as optimistic. McNemar is an exact test on lines and ignores the clustering.",
            "The KB 4-way rule needs the exact (card, lane, zone) key of an option among the 20 hits, so many lines have no "
            "supported option (coverage column) and count 0.25: exact hit@10 is far below card_hit@10.",
            "The deck pre-filter only returns hits when the line's own deck was played in a train game (its 'empty' share "
            "is the lines whose deck never appears in train); where it does return, the hits are plays from train games with that deck.",
            "The d1 options' notes are produced by the same train-only retrieval and KG, so d1 sees these signals in its prompt; "
            "the methods here use them without a language model.",
            "KGE scores are shifted to positives per threat (order preserving) so the tie logic and the event metrics treat "
            "them like the other methods; a deck card the graph does not know ranks last."]}
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")
    md = render_markdown(res)
    DOC_OUT.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {DOC_OUT} and {DOC_OUT.with_suffix('.md')}")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("lines", "embed", "kge", "accuracy", "latency", "report"))
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--kg2", type=Path, default=KG2_PATH)
    ap.add_argument("--lines", type=Path, default=LINES_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--models", default=",".join(KGE_MODELS), help="kge stage: comma list")
    ap.add_argument("--limit", type=int, default=None, help="first N lines (smoke runs; use another --out-dir)")
    args = ap.parse_args(argv)
    {"lines": stage_lines, "embed": stage_embed, "kge": stage_kge, "accuracy": stage_accuracy,
     "latency": stage_latency, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
