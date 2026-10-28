"""Build the ranker training set (WP4): plays-only multiple-choice options with hard negatives, the act/hold export and the placement prior.

  python3 tools/data/build_ranker_dataset.py [--dataset DIR] [--k 20]

Writes <dataset>/ranker/{train,val,test}.jsonl, <dataset>/ranker/stats.json,
<dataset>/act/{train,val,test}.jsonl and <dataset>/placement_prior.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.actions import option_text  # noqa: E402
from clashbot.config import DATASET_DIR, HOLD_KEY, ZONES  # noqa: E402
from clashbot.features import featurize  # noqa: E402
from clashbot.game_rules import (action_key, legal_action_keys, legal_zones, load_cards,  # noqa: E402
                                 possible_action_keys, possible_cards)

SPLITS = ("train", "val", "test")
TARGET_NEGATIVES = 3


def load_records(dataset: Path) -> list[dict]:
    with open(Path(dataset) / "states.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def feature_matrix(records: list[dict]) -> np.ndarray:
    return np.stack([featurize(r["state"], r["t"]) for r in records]).astype(np.float32)


def train_row_indices(records: list[dict]) -> list[int]:
    return [i for i, r in enumerate(records) if r["split"] == "train" and r["action"]["key"] is not None]


def top_neighbours(Ftr: np.ndarray, train_rows: np.ndarray, train_games: np.ndarray,
                   vec: np.ndarray, game_id: str, k: int) -> list[tuple[int, float]]:
    """Top-k train neighbours (record index, similarity) from other games; ties -> lower row index.

    Ftr = F[train_rows]; similarity is the dot product."""
    scores = (Ftr @ vec).astype(np.float64)
    scores[train_games == game_id] = -np.inf
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(train_rows[j]), float(scores[j])) for j in order if np.isfinite(scores[j])]


def build_placement_prior(records: list[dict]) -> dict:
    by_key: dict[str, list] = defaultdict(list)
    by_lz: dict[str, list] = defaultdict(list)
    zone_freq: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        a = r["action"]
        if r["split"] != "train" or not r["train_eligible"] or a["type"] != "play":
            continue
        tile = a["tile"]
        by_key[a["key"]].append(tile)
        by_lz[f"{a['lane']}|{a['zone']}"].append(tile)
        zone_freq[a["card"]][a["zone"]] += 1

    def summarize(groups):
        return {k: {"count": len(v),
                    "median_tile": [round(statistics.median(t[0] for t in v), 1),
                                    round(statistics.median(t[1] for t in v), 1)]}
                for k, v in sorted(groups.items())}

    return {"source": "train split, train_eligible plays",
            "by_key": summarize(by_key),
            "by_lane_zone": summarize(by_lz),
            "zone_freq": {c: dict(sorted(z.items())) for c, z in sorted(zone_freq.items())}}


def select_negatives(pools: dict[str, list[str]], pos: str) -> list[tuple[str, str]]:
    """Pick up to 3 negatives as (pool_name, key): first N1, first N2, first N3, rest of N1, then N5."""
    chosen: list[tuple[str, str]] = []
    seen = {pos}

    def take(name: str, key: str) -> None:
        if len(chosen) < TARGET_NEGATIVES and key not in seen:
            seen.add(key)
            chosen.append((name, key))

    n1 = pools.get("N1", [])
    for name in ("N1", "N2", "N3"):
        items = pools.get(name, [])
        if items:
            take(name, items[0])
    for key in n1:
        take("N1", key)
    for key in pools.get("N5", []):
        take("N5", key)
    return chosen


def build_pools(rec: dict, pos: str, possible: list[str], neighbour_keys: list[str], cards: dict,
                zone_freq: dict) -> dict[str, list[str]]:
    state = rec["state"]
    possible_set = set(possible)
    pools: dict[str, list[str]] = {"N1": [], "N2": [], "N3": [], "N5": []}
    for key in neighbour_keys:
        if key in possible_set and key not in pools["N1"] and key != pos:
            pools["N1"].append(key)
    a = rec["action"]
    card, lane, zone = a["card"], a["lane"], a["zone"]
    freq = zone_freq.get(card, {})
    lz = legal_zones(card, state, cards)
    for z in sorted(ZONES, key=lambda z: (-freq.get(z, 0), z)):
        if z != zone and z in lz:
            pools["N2"].append(action_key(card, lane, z))
            break
    for c in possible_cards(state, cards):
        if c != card and zone in legal_zones(c, state, cards):
            pools["N3"].append(action_key(c, lane, zone))
            break
    pools["N5"] = [k for k in possible if k != pos]
    return pools


def build(dataset: Path, k: int) -> dict:
    dataset = Path(dataset)
    records = load_records(dataset)
    cards = load_cards()
    F = feature_matrix(records)
    train_rows = np.array(train_row_indices(records), dtype=np.int64)
    games = np.array([r["game_id"] for r in records])
    train_games = games[train_rows]
    Ftr = F[train_rows]

    prior = build_placement_prior(records)
    (dataset / "placement_prior.json").write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n")
    zone_freq = prior["zone_freq"]

    out_dir = dataset / "ranker"
    act_dir = dataset / "act"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: open(out_dir / f"{s}.jsonl", "w") for s in SPLITS}
    act_handles = {s: open(act_dir / f"{s}.jsonl", "w") for s in SPLITS}
    acc = {s: {"examples": 0, "skipped_no_negatives": 0, "skipped_pos_impossible": 0, "options_total": 0,
               "label_position_counts": Counter(), "negative_source_counts": Counter(),
               "pos_in_legal": 0, "kind_counts": Counter(),
               "act_positives": Counter(), "act_negatives": Counter()} for s in SPLITS}
    try:
        for i, rec in enumerate(records):
            if not rec["train_eligible"]:
                continue
            split = rec["split"]
            st = acc[split]
            is_play = rec["action"]["type"] == "play"
            act_handles[split].write(json.dumps({
                "id": rec["id"], "game_id": rec["game_id"], "split": split, "kind": rec["kind"],
                "context": rec["context_text"], "label": 1 if is_play else 0}) + "\n")
            (st["act_positives"] if is_play else st["act_negatives"])[rec["kind"]] += 1
            if not is_play:
                continue
            pos = rec["action"]["key"]
            possible = possible_action_keys(rec["state"], cards)
            if pos not in set(possible):
                st["skipped_pos_impossible"] += 1
                continue
            nbrs = top_neighbours(Ftr, train_rows, train_games, F[i], rec["game_id"], k)
            neighbour_keys = [records[j]["action"]["key"] for j, _ in nbrs]
            pools = build_pools(rec, pos, possible, neighbour_keys, cards, zone_freq)
            negatives = select_negatives(pools, pos)
            if not negatives:
                st["skipped_no_negatives"] += 1
                continue
            options = [("pos", pos)] + [(name, key) for name, key in negatives]
            seed = int(hashlib.sha1(rec["id"].encode()).hexdigest()[:8], 16)
            random.Random(seed).shuffle(options)
            label = next(j for j, (src, _) in enumerate(options) if src == "pos")
            in_legal = pos in set(legal_action_keys(rec["state"], cards))
            line = {"id": rec["id"], "game_id": rec["game_id"], "split": split, "kind": rec["kind"],
                    "context": rec["context_text"],
                    "options": [{"key": key, "text": option_text(key), "source": src} for src, key in options],
                    "label": label, "positive_key": pos, "pos_in_legal": in_legal}
            handles[split].write(json.dumps(line) + "\n")
            st["examples"] += 1
            st["options_total"] += len(options)
            st["label_position_counts"][str(label)] += 1
            for src, _ in options:
                if src != "pos":
                    st["negative_source_counts"][src] += 1
            st["pos_in_legal"] += int(in_legal)
            st["kind_counts"][rec["kind"]] += 1
    finally:
        for h in list(handles.values()) + list(act_handles.values()):
            h.close()

    stats = {}
    for s in SPLITS:
        a = acc[s]
        n = a["examples"]
        kinds = sorted(set(a["act_positives"]) | set(a["act_negatives"]))
        stats[s] = {"examples": n, "skipped_no_negatives": a["skipped_no_negatives"],
                    "skipped_pos_impossible": a["skipped_pos_impossible"],
                    "mean_options": round(a["options_total"] / n, 3) if n else 0.0,
                    "label_position_counts": dict(sorted(a["label_position_counts"].items())),
                    "negative_source_counts": dict(sorted(a["negative_source_counts"].items())),
                    "pos_in_legal_rate": round(a["pos_in_legal"] / n, 3) if n else 0.0,
                    "kind_counts": dict(sorted(a["kind_counts"].items())),
                    "act_counts": {kd: {"positives": a["act_positives"][kd], "negatives": a["act_negatives"][kd]}
                                   for kd in kinds}}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def print_stats(stats: dict) -> None:
    print("| split | examples | skipped_no_neg | skipped_pos_impossible | mean_opts | pos_in_legal | label_pos | neg_sources | kinds | act (pos/neg by kind) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s, v in stats.items():
        act = ", ".join(f"{kd}:{c['positives']}/{c['negatives']}" for kd, c in v["act_counts"].items())
        print(f"| {s} | {v['examples']} | {v['skipped_no_negatives']} | {v['skipped_pos_impossible']} | "
              f"{v['mean_options']} | {v['pos_in_legal_rate']} | {v['label_position_counts']} | "
              f"{v['negative_source_counts']} | {v['kind_counts']} | {act} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    print_stats(build(args.dataset, args.k))


if __name__ == "__main__":
    main()
