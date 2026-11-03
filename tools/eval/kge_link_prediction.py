"""KG-embedding link prediction for held-out COUNTERS edges (MAC-R2).

  python3 tools/eval/kge_link_prediction.py [--dataset DIR] [--epochs 150] [--models TransE,DistMult,ComplEx,RotatE]
         [--settings full,no_inverse] [--grid full|small] [--seeds 0,1,2] [--out PATH]

Splits the deduplicated COUNTERS triples 80/10/10, trains PyKEEN TransE/DistMult/ComplEx/RotatE on everything else
plus the COUNTERS train part (equal tuning budget per model) and scores every method - models, a popularity and a
random baseline - with ONE filtered evaluator written here: MRR from the realistic rank (expected rank under random tie-breaking) and
Hits@k as the expectation of the hit under random tie-breaking (`realistic_hits`; not Hits@k of the expected rank).
Setting `no_inverse` also drops ANSWERED_BY from training (it is close to the inverse of COUNTERS).
Writes the JSON (default <dataset>/eval/kge_link_prediction.json), a markdown table next to it, and prints the table.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.config import DATASET_DIR  # noqa: E402

Triple = tuple[str, str, str]

TARGET = "COUNTERS"
TRAIN_RELATIONS = ("COUNTERS", "SYNERGY", "SUBSTITUTE", "ANSWERED_BY", "IN_DECK", "SPAWNED_BY", "EVOLUTION_OF",
                   "HERO_OF", "HAS_ARCHETYPE")
SETTINGS = ("full", "no_inverse")
MODELS = ("TransE", "DistMult", "ComplEx", "RotatE")
GRIDS = {"full": [(d, lr) for d in (32, 64) for lr in (1e-2, 1e-3)], "small": [(32, 1e-2)]}
GRID_SEED = 0
BATCH_SIZE = 256
NUM_NEGS = 8
HITS_KS = (1, 3, 10)
CARD_PREFIX = "card:"


# ---------------------------------------------------------------- data & splits

def load_triples(dataset: Path) -> list[Triple]:
    """Deduplicated, sorted (src, rel, dst) triples of the training relations (GAME_* relations dropped)."""
    keep = set(TRAIN_RELATIONS)
    seen: set[Triple] = set()
    with (dataset / "kg" / "edges.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e["rel"] in keep:
                seen.add((e["src"], e["rel"], e["dst"]))
    return sorted(seen)


def load_names(dataset: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with (dataset / "kg" / "nodes.jsonl").open() as f:
        for line in f:
            if line.strip():
                n = json.loads(line)
                names[n["id"]] = n.get("name", n["id"])
    return names


@dataclass
class Splits:
    setting: str
    train: list[Triple]          # every training triple: non-COUNTERS (minus ANSWERED_BY if no_inverse) + train_c
    train_c: list[Triple]
    val_c: list[Triple]
    test_c: list[Triple]
    dropped_unseen: int
    candidates: list[str]        # card: entities that occur in training (sorted)
    all_counters: set[Triple]    # every deduplicated COUNTERS triple (train+val+test+dropped): the evaluation filter


def split_counters(counters: Iterable[Triple], seed: int = 0) -> tuple[list[Triple], list[Triple], list[Triple]]:
    """Deduplicate, sort, shuffle with Random(seed), cut 80/10/10 -> (train_c, val_c, test_c)."""
    items = sorted(set(counters))
    random.Random(seed).shuffle(items)
    n = len(items)
    n_tr, n_va = int(round(0.8 * n)), int(round(0.1 * n))
    return items[:n_tr], items[n_tr:n_tr + n_va], items[n_tr + n_va:]


def build_splits(triples: Iterable[Triple], setting: str = "full", seed: int = 0) -> Splits:
    """COUNTERS held-out split + training set for a setting ('full' or 'no_inverse')."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}")
    uniq = sorted(set(triples))
    counters = [t for t in uniq if t[1] == TARGET]
    others = [t for t in uniq if t[1] != TARGET and not (setting == "no_inverse" and t[1] == "ANSWERED_BY")]
    train_c, val_c, test_c = split_counters(counters, seed)
    train = sorted(set(others) | set(train_c))
    seen = {x for h, _, t in train for x in (h, t)}
    n_before = len(val_c) + len(test_c)
    val_c = [t for t in val_c if t[0] in seen and t[2] in seen]
    test_c = [t for t in test_c if t[0] in seen and t[2] in seen]
    return Splits(setting=setting, train=train, train_c=sorted(train_c), val_c=sorted(val_c), test_c=sorted(test_c),
                  dropped_unseen=n_before - len(val_c) - len(test_c),
                  candidates=sorted(e for e in seen if e.startswith(CARD_PREFIX)), all_counters=set(counters))


# ---------------------------------------------------------------- evaluation protocol

def _better_and_tied(scores: np.ndarray, true_idx: int, filter_idx: Sequence[int]) -> tuple[int, int]:
    """Over the filtered candidate set: (#(s > s_t), #(s == s_t) including the true item). The true index is never filtered."""
    keep = np.ones(len(scores), dtype=bool)
    keep[np.asarray(filter_idx, dtype=int)] = False
    keep[true_idx] = True
    s = scores[keep]
    st = scores[true_idx]
    return int(np.sum(s > st)), int(np.sum(s == st))


def realistic_rank(scores: np.ndarray, true_idx: int, filter_idx: Sequence[int]) -> float:
    """Filtered expected rank of `true_idx` under random tie-breaking: 1 + #(s > s_t) + 0.5 * #(s == s_t, c != t)."""
    better, tied = _better_and_tied(scores, true_idx, filter_idx)
    return float(1 + better + 0.5 * (tied - 1))


def realistic_hits(scores: np.ndarray, true_idx: int, filter_idx: Sequence[int], k: int) -> float:
    """Filtered Hits@k as the expectation under random tie-breaking: the chance the true item lands in the top k when the
    `tied` items sharing its score (itself included) are ordered at random after the `better` ones,
    clip((k - better) / tied, 0, 1). Not `realistic_rank(...) <= k`: a full tie over 126 candidates gives Hits@10 = 10/126, not 0."""
    better, tied = _better_and_tied(scores, true_idx, filter_idx)
    return float(max(0.0, min(1.0, (k - better) / tied)))


class RandomScorer:
    """All scores equal -> realistic rank is the middle of the filtered list."""

    def __init__(self, n_cand: int):
        self.n = n_cand

    def score_tails(self, heads: Sequence[str]) -> np.ndarray:
        return np.zeros((len(heads), self.n))

    def score_heads(self, tails: Sequence[str]) -> np.ndarray:
        return np.zeros((len(tails), self.n))


class PopularityScorer:
    """Tail score = in-degree of c in the training COUNTERS edges; head score = out-degree of c."""

    def __init__(self, train_counters: Iterable[Triple], candidates: Sequence[str]):
        indeg, outdeg = Counter(), Counter()
        for h, _, t in train_counters:
            outdeg[h] += 1
            indeg[t] += 1
        self.tail_scores = np.array([indeg[c] for c in candidates], dtype=float)
        self.head_scores = np.array([outdeg[c] for c in candidates], dtype=float)

    def score_tails(self, heads: Sequence[str]) -> np.ndarray:
        return np.tile(self.tail_scores, (len(heads), 1))

    def score_heads(self, tails: Sequence[str]) -> np.ndarray:
        return np.tile(self.head_scores, (len(tails), 1))


def rank_and_hits(scorer, triples: Sequence[Triple], candidates: Sequence[str], known: Iterable[Triple],
                  chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Filtered realistic ranks, shape (n, 2), and expected hits, shape (n, 2, len(HITS_KS)).

    Column / axis-1 index 0 = tail prediction, 1 = head prediction; hits[:, :, j] is `realistic_hits` at k = HITS_KS[j].
    `known` = every COUNTERS triple of train+val+test; other known answers for the query are filtered out."""
    cidx = {c: i for i, c in enumerate(candidates)}
    tails_of: dict[str, set[str]] = defaultdict(set)
    heads_of: dict[str, set[str]] = defaultdict(set)
    for h, _, t in known:
        tails_of[h].add(t)
        heads_of[t].add(h)
    triples = [t for t in triples if t[0] in cidx and t[2] in cidx]
    ranks = np.zeros((len(triples), 2))
    hits = np.zeros((len(triples), 2, len(HITS_KS)))
    for lo in range(0, len(triples), chunk):
        part = triples[lo:lo + chunk]
        s_tail = scorer.score_tails([h for h, _, _ in part])
        s_head = scorer.score_heads([t for _, _, t in part])
        for i, (h, _, t) in enumerate(part):
            queries = ((s_tail[i], cidx[t], [cidx[c] for c in tails_of[h] if c in cidx]),
                       (s_head[i], cidx[h], [cidx[c] for c in heads_of[t] if c in cidx]))
            for col, (scores, true, flt) in enumerate(queries):
                ranks[lo + i, col] = realistic_rank(scores, true, flt)
                hits[lo + i, col] = [realistic_hits(scores, true, flt, k) for k in HITS_KS]
    return ranks, hits


def rank_triples(scorer, triples: Sequence[Triple], candidates: Sequence[str], known: Iterable[Triple],
                 chunk: int = 512) -> np.ndarray:
    """Filtered realistic ranks only, shape (n, 2) (see `rank_and_hits`)."""
    return rank_and_hits(scorer, triples, candidates, known, chunk)[0]


def metrics_from_ranks(ranks: np.ndarray, hits: np.ndarray) -> dict:
    """MRR from the realistic ranks, hits@k from the expected hits (`realistic_hits`), pooled over both prediction
    directions (n = 2 * number of triples). `hits` has shape (n, 2, len(HITS_KS)) as returned by `rank_and_hits`."""
    r = np.asarray(ranks, dtype=float).reshape(-1)
    if r.size == 0:
        return {"mrr": float("nan"), **{f"hits@{k}": float("nan") for k in HITS_KS}, "n": 0}
    h = np.asarray(hits, dtype=float).reshape(r.size, len(HITS_KS))
    return {"mrr": float(np.mean(1.0 / r)), **{f"hits@{k}": float(np.mean(h[:, j])) for j, k in enumerate(HITS_KS)},
            "n": int(r.size)}


def evaluate(scorer, triples: Sequence[Triple], sp: Splits) -> dict:
    return metrics_from_ranks(*rank_and_hits(scorer, triples, sp.candidates, sp.all_counters))


# ---------------------------------------------------------------- PyKEEN models

class PykeenScorer:
    """Adapter from a trained PyKEEN model to the score_tails / score_heads interface over the card candidates."""

    def __init__(self, model, tf, candidates: Sequence[str]):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.ent = tf.entity_to_id
        self.rel = tf.relation_to_id[TARGET]
        self.cand = torch.as_tensor([self.ent[c] for c in candidates], dtype=torch.long)

    def _scores(self, fn, batch: list[list[int]]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for lo in range(0, len(batch), 256):
                s = fn(self.torch.as_tensor(batch[lo:lo + 256], dtype=self.torch.long))
                out.append(s[:, self.cand].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.cand)))

    def score_tails(self, heads: Sequence[str]) -> np.ndarray:
        return self._scores(self.model.score_t, [[self.ent[h], self.rel] for h in heads])

    def score_heads(self, tails: Sequence[str]) -> np.ndarray:
        return self._scores(self.model.score_h, [[self.rel, self.ent[t]] for t in tails])

    def score_triples(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        batch = self.torch.as_tensor([[self.ent[h], self.rel, self.ent[t]] for h, t in pairs], dtype=self.torch.long)
        with self.torch.no_grad():
            return self.model.score_hrt(batch).reshape(-1).cpu().numpy()


def asymmetry(scorer: PykeenScorer, triples: Sequence[Triple]) -> float:
    """Fraction of triples with s(h, r, t) > s(t, r, h), ties (float noise) not counted. DistMult should give ~0."""
    if not triples:
        return float("nan")
    fwd = scorer.score_triples([(h, t) for h, _, t in triples])
    rev = scorer.score_triples([(t, h) for h, _, t in triples])
    tol = 1e-5 * np.maximum(1.0, np.maximum(np.abs(fwd), np.abs(rev)))
    return float(np.mean(fwd > rev + tol))


def make_factory(sp: Splits):
    from pykeen.triples import TriplesFactory
    return TriplesFactory.from_labeled_triples(np.array(sp.train, dtype=str), create_inverse_triples=False)


def train_model(name: str, tf, dim: int, lr: float, epochs: int, seed: int):
    """Train one PyKEEN model (default loss, Adam, sLCWA, basic negative sampler) on CPU. Returns (model, seconds).

    Uses the training loop directly instead of `pipeline(evaluator=None)`: the pipeline would still run its default
    rank-based evaluation over the whole training set, and we do our own evaluation."""
    import torch
    from pykeen.models import model_resolver
    from pykeen.training import SLCWATrainingLoop
    from pykeen.utils import set_random_seed
    set_random_seed(seed)
    model = model_resolver.make(name, triples_factory=tf, embedding_dim=dim, random_seed=seed).to("cpu")
    opt = torch.optim.Adam(model.get_grad_params(), lr=lr)
    loop = SLCWATrainingLoop(model=model, triples_factory=tf, optimizer=opt, negative_sampler="basic",
                             negative_sampler_kwargs={"num_negs_per_pos": NUM_NEGS})
    t0 = time.time()
    loop.train(triples_factory=tf, num_epochs=epochs, batch_size=BATCH_SIZE, use_tqdm=False, use_tqdm_batch=False,
                pin_memory=False)
    return model, time.time() - t0


def novel_predictions(scorer: PykeenScorer, sp: Splits, names: dict[str, str], top: int = 20) -> list[dict]:
    """Highest-scoring (h, t) card pairs that are in no COUNTERS split (h != t)."""
    cands = sp.candidates
    s = scorer.score_tails(cands).astype(float)
    cidx = {c: i for i, c in enumerate(cands)}
    s[np.arange(len(cands)), np.arange(len(cands))] = -np.inf
    for h, _, t in sp.all_counters:
        if h in cidx and t in cidx:
            s[cidx[h], cidx[t]] = -np.inf
    order = np.argsort(-s, axis=None, kind="stable")[:top]
    out = []
    for k in order:
        i, j = divmod(int(k), len(cands))
        if not np.isfinite(s[i, j]):
            break
        out.append({"h": cands[i], "t": cands[j], "h_name": names.get(cands[i], cands[i]),
                    "t_name": names.get(cands[j], cands[j]), "score": float(s[i, j])})
    return out


def run_model(name: str, sp: Splits, tf, grid: list[tuple[int, float]], epochs: int, seeds: Sequence[int]):
    """Grid search on val MRR (seed GRID_SEED), then retrain the best config for every seed; report test."""
    cache: dict[tuple[int, float, int], tuple[PykeenScorer, float]] = {}

    def fit(dim: int, lr: float, seed: int):
        if (dim, lr, seed) not in cache:
            model, secs = train_model(name, tf, dim, lr, epochs, seed)
            cache[(dim, lr, seed)] = (PykeenScorer(model, tf, sp.candidates), secs)
        return cache[(dim, lr, seed)]

    grid_rows = []
    for dim, lr in grid:
        scorer, _ = fit(dim, lr, GRID_SEED)
        grid_rows.append({"dim": dim, "lr": lr, "val_mrr": evaluate(scorer, sp.val_c, sp)["mrr"]})
    best = max(grid_rows, key=lambda r: (r["val_mrr"] if np.isfinite(r["val_mrr"]) else -1.0))
    runs = []
    for seed in seeds:
        scorer, secs = fit(best["dim"], best["lr"], seed)
        runs.append({"seed": seed, "metrics": evaluate(scorer, sp.test_c, sp), "asymmetry": asymmetry(scorer, sp.test_c),
                     "train_s": secs, "scorer": scorer})
    mrrs = np.array([r["metrics"]["mrr"] for r in runs])
    result = {"grid": grid_rows, "best": {"dim": best["dim"], "lr": best["lr"]},
              "test": {"mrr_mean": float(mrrs.mean()), "mrr_std": float(mrrs.std()),
                       **{f"hits@{k}": float(np.mean([r["metrics"][f"hits@{k}"] for r in runs])) for k in HITS_KS},
                       "n": runs[0]["metrics"]["n"], "seeds": list(seeds)},
              "asymmetry": float(np.mean([r["asymmetry"] for r in runs])),
              "train_s": float(np.mean([r["train_s"] for r in runs]))}
    return result, runs[0]["scorer"]


def run_setting(sp: Splits, models: Sequence[str], grid_name: str, epochs: int, seeds: Sequence[int]):
    """Baselines + models for one setting. Returns (result dict, best model's name, its scorer)."""
    train_c = set(sp.train_c)
    reverse = [t for t in sp.test_c if (t[2], TARGET, t[0]) in train_c]
    out: dict = {"baselines": {}, "models": {}, "reverse_in_graph": len(reverse) / max(1, len(sp.test_c))}
    for bname, scorer in (("random", RandomScorer(len(sp.candidates))),
                          ("popularity", PopularityScorer(sp.train_c, sp.candidates))):
        out["baselines"][bname] = {"val": evaluate(scorer, sp.val_c, sp), "test": evaluate(scorer, sp.test_c, sp)}
    tf = make_factory(sp)
    best_name, best_scorer, best_mrr = None, None, -1.0
    for name in models:
        res, scorer = run_model(name, sp, tf, GRIDS[grid_name], epochs, seeds)
        out["models"][name] = res
        print(f"[{sp.setting}] {name}: best={res['best']} test MRR={res['test']['mrr_mean']:.4f}"
              f" ± {res['test']['mrr_std']:.4f} ({res['train_s']:.1f}s/run)", flush=True)
        if res["test"]["mrr_mean"] > best_mrr:
            best_name, best_scorer, best_mrr = name, scorer, res["test"]["mrr_mean"]
    return out, best_name, best_scorer


def markdown_table(settings: dict) -> str:
    rows = ["| setting | method | MRR | H@1 | H@3 | H@10 |", "|---|---|---|---|---|---|"]
    for setting, res in settings.items():
        for bname, b in res["baselines"].items():
            t = b["test"]
            rows.append(f"| {setting} | {bname} | {t['mrr']:.4f} | {t['hits@1']:.4f} | {t['hits@3']:.4f} | {t['hits@10']:.4f} |")
        for mname, m in res["models"].items():
            t = m["test"]
            rows.append(f"| {setting} | {mname} | {t['mrr_mean']:.4f} ± {t['mrr_std']:.4f} | {t['hits@1']:.4f} |"
                        f" {t['hits@3']:.4f} | {t['hits@10']:.4f} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--grid", choices=sorted(GRIDS), default="full")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    settings = [s for s in args.settings.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    out = args.out or args.dataset / "eval" / "kge_link_prediction.json"

    t0 = time.time()
    triples = load_triples(args.dataset)
    names = load_names(args.dataset)
    result: dict = {"counts": {"relations": dict(Counter(t[1] for t in triples)), "models": models, "epochs": args.epochs,
                               "grid": args.grid, "seeds": seeds},
                    "settings": {}}
    best_by_setting: dict[str, tuple[str, float, object, Splits]] = {}
    for setting in settings:
        sp = build_splits(triples, setting)
        result["counts"][setting] = {"train_triples": len(sp.train), "train_counters": len(sp.train_c),
                                     "val_triples": len(sp.val_c), "test_triples": len(sp.test_c),
                                     "dropped_unseen": sp.dropped_unseen, "candidates": len(sp.candidates)}
        print(f"[{setting}] {result['counts'][setting]}", flush=True)
        res, best_name, best_scorer = run_setting(sp, models, args.grid, args.epochs, seeds)
        result["settings"][setting] = res
        if best_name is not None:
            best_by_setting[setting] = (best_name, res["models"][best_name]["test"]["mrr_mean"], best_scorer, sp)
    if best_by_setting:
        setting = "full" if "full" in best_by_setting else next(iter(best_by_setting))
        name, _, scorer, sp = best_by_setting[setting]
        result["novel_predictions"] = {"setting": setting, "model": name,
                                       "best": result["settings"][setting]["models"][name]["best"],
                                       "pairs": novel_predictions(scorer, sp, names)}
    result["total_s"] = time.time() - t0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    table = markdown_table(result["settings"])
    out.with_suffix(".md").write_text(table + "\n")
    print(table)
    print(f"total {result['total_s']:.1f}s -> {out}")


if __name__ == "__main__":
    main()
