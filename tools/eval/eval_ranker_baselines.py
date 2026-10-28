"""Evaluate simple baselines on the ranker val/test sets (WP4).

  python3 tools/eval/eval_ranker_baselines.py [--dataset DIR] [--k 20]

Writes <dataset>/ranker/baselines.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.config import DATASET_DIR, HOLD_KEY  # noqa: E402
from tools.data.build_ranker_dataset import (feature_matrix, load_records, top_neighbours,  # noqa: E402
                                             train_row_indices)

EVAL_SPLITS = ("val", "test")
BASELINES = ("random_expected", "popularity", "popularity_by_phase", "knn_vote", "knn_vote_card")


def card_of(key: str) -> str:
    return HOLD_KEY if key == HOLD_KEY else key.split("|")[0]


def argmax_first(scores: list[float]) -> int:
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    dataset = args.dataset

    records = load_records(dataset)
    by_id = {r["id"]: i for i, r in enumerate(records)}
    F = feature_matrix(records)
    train_rows = np.array(train_row_indices(records), dtype=np.int64)
    games = np.array([r["game_id"] for r in records])
    train_games = games[train_rows]
    Ftr = F[train_rows]

    pop = Counter()
    pop_phase: dict = defaultdict(Counter)
    for r in records:
        if r["split"] == "train" and r["train_eligible"] and r["action"]["key"] is not None:
            pop[r["action"]["key"]] += 1
            pop_phase[r["state"]["phase"]][r["action"]["key"]] += 1

    result: dict = {}
    for split in EVAL_SPLITS:
        path = dataset / "ranker" / f"{split}.jsonl"
        examples = [json.loads(line) for line in open(path) if line.strip()]
        hits = {b: [] for b in BASELINES}   # list of (kind, value)
        for ex in examples:
            opts = [o["key"] for o in ex["options"]]
            label = ex["label"]
            rec_i = by_id[ex["id"]]
            rec = records[rec_i]
            kind = ex["kind"]
            hits["random_expected"].append((kind, 1.0 / len(opts)))
            picks = {}
            picks["popularity"] = argmax_first([float(pop[k]) for k in opts])
            pp = pop_phase[rec["state"]["phase"]]
            picks["popularity_by_phase"] = argmax_first([float(pp[k]) for k in opts])
            nbrs = top_neighbours(Ftr, train_rows, train_games, F[rec_i], rec["game_id"], args.k)
            nbr_keys = [(records[j]["action"]["key"], sim) for j, sim in nbrs]
            vote = [sum(sim for nk, sim in nbr_keys if nk == k) for k in opts]
            vote_card = [sum(sim for nk, sim in nbr_keys if card_of(nk) == card_of(k)) for k in opts]
            picks["knn_vote"] = argmax_first(vote)
            picks["knn_vote_card"] = argmax_first(vote_card)
            for b, p in picks.items():
                hits[b].append((kind, float(p == label)))
        result[split] = {}
        for b in BASELINES:
            vals = hits[b]
            n = len(vals)
            by_kind: dict = {}
            for kd in sorted({k for k, _ in vals}):
                kv = [v for k, v in vals if k == kd]
                by_kind[kd] = {"acc": round(sum(kv) / len(kv), 4), "n": len(kv)}
            result[split][b] = {"acc": round(sum(v for _, v in vals) / n, 4) if n else 0.0,
                                "n": n, "by_kind": by_kind}

    (dataset / "ranker" / "baselines.json").write_text(json.dumps(result, indent=2) + "\n")
    print("| baseline | val acc | test acc |")
    print("|---|---|---|")
    for b in BASELINES:
        print(f"| {b} | {result['val'][b]['acc']} | {result['test'][b]['acc']} |")
    for split in EVAL_SPLITS:
        kinds = sorted({kd for b in BASELINES for kd in result[split][b]["by_kind"]})
        print(f"\n{split} by kind (n = {', '.join(kd + ':' + str(result[split]['popularity']['by_kind'][kd]['n']) for kd in kinds)})")
        print("| baseline | " + " | ".join(kinds) + " |")
        print("|---" * (len(kinds) + 1) + "|")
        for b in BASELINES:
            print(f"| {b} | " + " | ".join(str(result[split][b]["by_kind"][kd]["acc"]) for kd in kinds) + " |")


if __name__ == "__main__":
    main()
