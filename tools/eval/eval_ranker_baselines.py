"""Evaluate baselines for the ranker, candidate generation and the act/hold head (WP4, Amendment C).

  python3 tools/eval/eval_ranker_baselines.py [--dataset DIR] [--k 20]

Writes <dataset>/ranker/baselines.json and prints three markdown tables.
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

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.game_rules import load_cards, possible_action_keys  # noqa: E402
from tools.data.build_ranker_dataset import (build_placement_prior, feature_matrix,  # noqa: E402
                                             load_records, top_neighbours, train_row_indices)

EVAL_SPLITS = ("val", "test")
RECALL_KS = (1, 3, 5, 10)
RANKER_BASELINES = ("random_expected", "popularity", "popularity_by_phase", "card_popularity_in_deck",
                    "placement_prior")
ACT_BASELINES = ("always_play", "phase_majority")
NOTE = ("knn_vote is not reported on the multiple-choice files: their N1 negatives are mined from the "
        "same kNN neighbours. It is evaluated fairly in candidate_recall over all possible action keys.")


def card_of(key: str) -> str:
    return key.split("|")[0]


def argmax_first(scores: list[float]) -> int:
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def by_kind_summary(vals: list[tuple[str, float]]) -> dict:
    out = {}
    for kd in sorted({k for k, _ in vals}):
        kv = [v for k, v in vals if k == kd]
        out[kd] = {"acc": round(sum(kv) / len(kv), 4), "n": len(kv)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    dataset = args.dataset

    cards = load_cards()
    records = load_records(dataset)
    by_id = {r["id"]: i for i, r in enumerate(records)}
    F = feature_matrix(records)
    train_rows = np.array(train_row_indices(records), dtype=np.int64)
    games = np.array([r["game_id"] for r in records])
    train_games = games[train_rows]
    Ftr = F[train_rows]
    zone_freq = build_placement_prior(records)["zone_freq"]

    pop = Counter()
    pop_phase: dict = defaultdict(Counter)
    deck_records = Counter()      # card -> train_eligible records whose own deck contains it
    deck_plays = Counter()        # card -> those records where that card was played
    act_by_phase: dict = defaultdict(Counter)
    act_all = Counter()
    for r in records:
        if r["split"] != "train" or not r["train_eligible"]:
            continue
        a = r["action"]
        if a["key"] is not None:
            pop[a["key"]] += 1
            pop_phase[r["state"]["phase"]][a["key"]] += 1
        deck = set(r["state"]["own"]["deck"])
        for c in deck:
            deck_records[c] += 1
        if a["type"] == "play" and a["card"] in deck:
            deck_plays[a["card"]] += 1
        label = 1 if a["type"] == "play" else 0
        act_by_phase[r["state"]["phase"]][label] += 1
        act_all[label] += 1

    def majority(counter: Counter) -> int:
        return 1 if counter[1] >= counter[0] else 0

    phase_major = {ph: majority(c) for ph, c in act_by_phase.items()}
    global_major = majority(act_all)

    def card_pop(c: str) -> float:
        return deck_plays[c] / deck_records[c] if deck_records[c] else 0.0

    def placement(key: str) -> float:
        card, _, zone = key.split("|")
        zf = zone_freq.get(card, {})
        tot = sum(zf.values())
        return zf.get(zone, 0) / tot if tot else 0.0

    result: dict = {"note": NOTE, "candidate_recall": {}, "ranker": {}, "act": {}}
    for split in EVAL_SPLITS:
        examples = read_jsonl(dataset / "ranker" / f"{split}.jsonl")

        # --- candidate generation recall over all possible action keys
        rec_hits = {m: {f"{p}@{kk}": 0 for p in ("recall", "card_recall") for kk in RECALL_KS}
                    for m in ("knn_vote", "popularity")}
        for ex in examples:
            rec_i = by_id[ex["id"]]
            rec = records[rec_i]
            cand = possible_action_keys(rec["state"], cards)
            nbrs = top_neighbours(Ftr, train_rows, train_games, F[rec_i], rec["game_id"], args.k)
            vote: Counter = Counter()
            for j, sim in nbrs:
                vote[records[j]["action"]["key"]] += sim
            ranked = {"knn_vote": sorted(cand, key=lambda key: (-vote[key], key)),
                      "popularity": sorted(cand, key=lambda key: (-pop[key], key))}
            pos = ex["positive_key"]
            for m, order in ranked.items():
                for kk in RECALL_KS:
                    top = order[:kk]
                    rec_hits[m][f"recall@{kk}"] += int(pos in top)
                    rec_hits[m][f"card_recall@{kk}"] += int(card_of(pos) in {card_of(x) for x in top})
        n = len(examples)
        result["candidate_recall"][split] = {
            m: {"n": n, **{name: round(v / n, 4) if n else 0.0 for name, v in hits.items()}}
            for m, hits in rec_hits.items()}

        # --- ranker baselines on the multiple-choice files
        hits = {b: [] for b in RANKER_BASELINES}
        for ex in examples:
            opts = [o["key"] for o in ex["options"]]
            label = ex["label"]
            kind = ex["kind"]
            rec = records[by_id[ex["id"]]]
            hits["random_expected"].append((kind, 1.0 / len(opts)))
            pp = pop_phase[rec["state"]["phase"]]
            picks = {
                "popularity": argmax_first([float(pop[k]) for k in opts]),
                "popularity_by_phase": argmax_first([float(pp[k]) for k in opts]),
                "card_popularity_in_deck": argmax_first([card_pop(card_of(k)) for k in opts]),
                "placement_prior": argmax_first([placement(k) for k in opts]),
            }
            for b, p in picks.items():
                hits[b].append((kind, float(p == label)))
        result["ranker"][split] = {}
        for b in RANKER_BASELINES:
            vals = hits[b]
            result["ranker"][split][b] = {
                "acc": round(sum(v for _, v in vals) / len(vals), 4) if vals else 0.0,
                "n": len(vals), "by_kind": by_kind_summary(vals)}

        # --- act/hold baselines
        act_examples = read_jsonl(dataset / "act" / f"{split}.jsonl")
        ahits = {b: [] for b in ACT_BASELINES}
        for ex in act_examples:
            ph = records[by_id[ex["id"]]]["state"]["phase"]
            ahits["always_play"].append((ex["kind"], float(ex["label"] == 1)))
            ahits["phase_majority"].append((ex["kind"], float(ex["label"] == phase_major.get(ph, global_major))))
        result["act"][split] = {}
        for b in ACT_BASELINES:
            vals = ahits[b]
            result["act"][split][b] = {
                "acc": round(sum(v for _, v in vals) / len(vals), 4) if vals else 0.0,
                "n": len(vals), "by_kind": by_kind_summary(vals)}

    (dataset / "ranker" / "baselines.json").write_text(json.dumps(result, indent=2) + "\n")

    print("Candidate generation recall over all possible action keys")
    cols = [f"{p}@{kk}" for p in ("recall", "card_recall") for kk in RECALL_KS]
    print("| split | method | n | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 3) + "|")
    for split in EVAL_SPLITS:
        for m, v in result["candidate_recall"][split].items():
            print(f"| {split} | {m} | {v['n']} | " + " | ".join(str(v[c]) for c in cols) + " |")

    print("\nRanker baselines on the multiple-choice files (accuracy)")
    print("| baseline | val acc | test acc |")
    print("|---|---|---|")
    for b in RANKER_BASELINES:
        print(f"| {b} | {result['ranker']['val'][b]['acc']} | {result['ranker']['test'][b]['acc']} |")

    print("\nAct/hold baselines (accuracy)")
    print("| baseline | val acc | test acc |")
    print("|---|---|---|")
    for b in ACT_BASELINES:
        print(f"| {b} | {result['act']['val'][b]['acc']} | {result['act']['test'][b]['acc']} |")


if __name__ == "__main__":
    main()
