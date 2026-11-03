"""Build the ranker training set (WP4): plays-only 4-way options in a 2x2 card x placement grid (D-037), the act/hold export and the placement prior.

  python3 tools/data/build_ranker_dataset.py [--dataset DIR] [--k 20]

Writes <dataset>/ranker/{train,val,test}.jsonl, <dataset>/ranker/stats.json,
<dataset>/act/{train,val,test}.jsonl and <dataset>/placement_prior.json.

Every example has exactly 4 options: the pro's play (c1 at l1/z1), the same card elsewhere (G_card), another card at the
pro's placement (G_place) and another card elsewhere (G_diag). Each option shares its card with exactly one other option
and its placement with exactly one other, so "pick the option that shares both" is at chance (0.25) by construction.
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
from clashbot.config import DATASET_DIR, HOLD_KEY, LANES, ZONES  # noqa: E402
from clashbot.features import featurize  # noqa: E402
from clashbot.game_rules import (action_key, legal_action_keys, load_cards,  # noqa: E402
                                 possible_action_keys, possible_cards)

SPLITS = ("train", "val", "test")
GRID_SOURCES = ("G_card", "G_place", "G_diag")   # same card/other placement, other card/same placement, both differ
PLACEMENT_BLOCKS = ("Z", "L", "B")               # same lane/other zone, other lane/same zone, both differ


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


def placement_candidates(c1: str, l1: str, z1: str, zone_freq: dict, variant: int) -> list[tuple[str, str]]:
    """Ordered (lane, zone) alternatives to (l1, z1): Z same lane/other zone, L other lane/same zone, B both differ.

    variant = seed % 3 rotates the block order (0: Z L B, 1: L Z B, 2: B Z L) so lane and zone both vary across the set."""
    freq = zone_freq.get(c1, {})
    z_block = [(l1, z) for z in sorted(ZONES, key=lambda z: (-freq.get(z, 0), z)) if z != z1]
    l_block = [(lane, z1) for lane in LANES if lane != l1]
    b_block = [(lane, z) for lane in LANES for z in ZONES if lane != l1 and z != z1]
    return {0: z_block + l_block + b_block, 1: l_block + z_block + b_block, 2: b_block + z_block + l_block}[variant]


def grid_options(rec: dict, pos: str, possible_keys: list[str], neighbour_keys: list[str], cards: dict,
                 zone_freq: dict, seed: int) -> list[tuple[str, str]] | None:
    """The three grid negatives [(G_card, key), (G_place, key), (G_diag, key)] for the positive `pos`, or None if no grid fits.

    c2 = first card (neighbour cards, then possible cards) playable at the pro's placement; (l2, z2) = first
    placement candidate where both c1 and c2 are possible. All four options are then in possible_keys."""
    c1, l1, z1 = pos.split("|")
    possible = set(possible_keys)
    candidates = [k.split("|")[0] for k in neighbour_keys if k != HOLD_KEY] + possible_cards(rec["state"], cards)
    second: list[str] = []
    for c in candidates:
        if c != c1 and c not in second and action_key(c, l1, z1) in possible:
            second.append(c)
    placements = placement_candidates(c1, l1, z1, zone_freq, seed % 3)
    for c2 in second:
        for l2, z2 in placements:
            if action_key(c1, l2, z2) in possible and action_key(c2, l2, z2) in possible:
                return [("G_card", f"{c1}|{l2}|{z2}"), ("G_place", f"{c2}|{l1}|{z1}"),
                        ("G_diag", f"{c2}|{l2}|{z2}")]
    return None


def placement_block(pos: str, key: str) -> str:
    """Which placement_candidates block (Z, L or B) the (lane, zone) of `key` belongs to, relative to the positive `pos`."""
    _, l1, z1 = pos.split("|")
    _, l2, z2 = key.split("|")
    return "Z" if l2 == l1 else "L" if z2 == z1 else "B"


def shortcut_value(keys: list[str], label: int) -> float:
    """Expected accuracy (random tie-break) of the option-set rule: score(k) = #options with k's card + #options with k's (lane, zone), k included."""
    cards = [key.split("|")[0] for key in keys]
    places = [tuple(key.split("|")[1:]) for key in keys]
    scores = [cards.count(c) + places.count(p) for c, p in zip(cards, places)]
    top = [j for j, sc in enumerate(scores) if sc == max(scores)]
    return 1.0 / len(top) if label in top else 0.0


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
    acc = {s: {"examples": 0, "skipped_no_grid": 0, "skipped_pos_impossible": 0, "options_total": 0,
               "label_position_counts": Counter(), "negative_source_counts": Counter(),
               "placement_variant_counts": Counter(), "shortcut_total": 0.0, "pos_in_legal": 0, "kind_counts": Counter(),
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
            seed = int(hashlib.sha1(rec["id"].encode()).hexdigest()[:8], 16)
            negatives = grid_options(rec, pos, possible, neighbour_keys, cards, zone_freq, seed)
            if negatives is None:
                st["skipped_no_grid"] += 1
                continue
            options = [("pos", pos)] + negatives
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
            st["placement_variant_counts"][placement_block(pos, negatives[0][1])] += 1
            st["shortcut_total"] += shortcut_value([key for _, key in options], label)
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
        stats[s] = {"examples": n, "skipped_no_grid": a["skipped_no_grid"],
                    "skipped_pos_impossible": a["skipped_pos_impossible"],
                    "mean_options": round(a["options_total"] / n, 3) if n else 0.0,
                    "label_position_counts": dict(sorted(a["label_position_counts"].items())),
                    "negative_source_counts": dict(sorted(a["negative_source_counts"].items())),
                    "placement_variant_counts": {b: a["placement_variant_counts"][b] for b in PLACEMENT_BLOCKS},
                    "option_set_shortcut": round(a["shortcut_total"] / n, 4) if n else 0.0,
                    "pos_in_legal_rate": round(a["pos_in_legal"] / n, 3) if n else 0.0,
                    "kind_counts": dict(sorted(a["kind_counts"].items())),
                    "act_counts": {kd: {"positives": a["act_positives"][kd], "negatives": a["act_negatives"][kd]}
                                   for kd in kinds}}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def print_stats(stats: dict) -> None:
    print("| split | examples | skipped_no_grid | skipped_pos_impossible | mean_opts | pos_in_legal | label_pos | neg_sources | placement_variants | option_set_shortcut | kinds | act (pos/neg by kind) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, v in stats.items():
        act = ", ".join(f"{kd}:{c['positives']}/{c['negatives']}" for kd, c in v["act_counts"].items())
        print(f"| {s} | {v['examples']} | {v['skipped_no_grid']} | {v['skipped_pos_impossible']} | "
              f"{v['mean_options']} | {v['pos_in_legal_rate']} | {v['label_position_counts']} | "
              f"{v['negative_source_counts']} | {v['placement_variant_counts']} | {v['option_set_shortcut']} | "
              f"{v['kind_counts']} | {act} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    print_stats(build(args.dataset, args.k))


if __name__ == "__main__":
    main()
