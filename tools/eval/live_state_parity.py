"""Parity of the live state builder against the training dataset (D-015, D-029).

  python3 tools/eval/live_state_parity.py --perception PATH/states.jsonl --game GAME_ID
         [--dataset DIR] [--tol 0.6] [--out PATH]

Replays one game's perception output (`state` + `play_event` rows from tools/run_video.py) through
clashbot.live_state.LiveStateBuilder at the instants of the dataset records of that game and
compares the live state with `rec["state"]`, field by field. Writes
<dataset>/eval/live_state_parity_<game>.json and prints the mean agreement table.
"""
from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.features import featurize  # noqa: E402
from clashbot.live_state import LiveConfig, LiveStateBuilder  # noqa: E402
from cr_perception.recorder import read_jsonl  # noqa: E402

FIELDS = ("phase", "clock", "elixir", "hand_slots", "next_card", "towers", "units_count_diff",
          "units_jaccard", "deck_known_jaccard", "own_recent_jaccard", "opp_recent_jaccard",
          "context_exact", "feature_cosine")
ELIXIR_TOL = 0.5
TOWER_REL_TOL = 0.01
N_EXAMPLES = 10


# --------------------------------------------------------------------------- field comparisons

def multiset_jaccard(a, b) -> float:
    ca, cb = Counter(a), Counter(b)
    union = sum((ca | cb).values())
    return 1.0 if union == 0 else sum((ca & cb).values()) / union


def elixir_agree(a, b) -> float:
    if a is None or b is None:
        return 1.0 if a is None and b is None else 0.0
    return 1.0 if abs(a - b) <= ELIXIR_TOL else 0.0


def tower_agree(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOWER_REL_TOL * max(abs(a), abs(b))


def towers_fraction(live: dict, ref: dict) -> float:
    pairs = [(((live[side].get("towers")) or {}).get(k), ((ref[side].get("towers")) or {}).get(k))
             for side in ("own", "opponent") for k in ("king", "left", "right")]
    return sum(tower_agree(a, b) for a, b in pairs) / len(pairs)


def unit_keys(units) -> list:
    return [(u.get("card") or u.get("unit_class"), u.get("side")) for u in units or []]


def play_cards(plays) -> list:
    return [p.get("card") for p in plays or []]


def compare(live: dict, live_text: str, rec: dict) -> dict:
    ref, t = rec["state"], rec["t"]
    lh, rh = list(live["own"]["hand"]), list(ref["own"]["hand"])
    return {
        "phase": float(live["phase"] == ref["phase"]),
        "clock": float(live["clock"] == ref["clock"]),
        "elixir": elixir_agree(live["own"]["elixir"], ref["own"]["elixir"]),
        "hand_slots": sum(a == b for a, b in zip(lh, rh)) / 4,
        "next_card": float(live["own"]["next_card"] == ref["own"]["next_card"]),
        "towers": towers_fraction(live, ref),
        "units_count_diff": float(abs(len(live["units"]) - len(ref["units"]))),
        "units_jaccard": multiset_jaccard(unit_keys(live["units"]), unit_keys(ref["units"])),
        "deck_known_jaccard": multiset_jaccard(live["opponent"]["deck_known"],
                                               ref["opponent"].get("deck_known") or []),
        "own_recent_jaccard": multiset_jaccard(play_cards(live["own"]["recent_plays"]),
                                               play_cards(ref["own"]["recent_plays"])),
        "opp_recent_jaccard": multiset_jaccard(play_cards(live["opponent"]["recent_plays"]),
                                               play_cards(ref["opponent"]["recent_plays"])),
        "context_exact": float(live_text == rec["context_text"]),
        "feature_cosine": float(featurize(live, t) @ featurize(ref, t)),
    }


# --------------------------------------------------------------------------- main

def load_perception(path: Path) -> tuple[list[dict], list[dict]]:
    """(match-readiness states sorted by t, play events sorted by timestamp)."""
    states, events = [], []
    for row in read_jsonl(path):
        kind = row.get("type")
        if kind == "state" and row.get("readiness") == "match":
            states.append(row)
        elif kind == "play_event" and row.get("timestamp") is not None:
            events.append(row)
    states.sort(key=lambda r: r["t"])
    events.sort(key=lambda r: r["timestamp"])
    return states, events


def read_game(dataset: Path, game_id: str) -> tuple[list[str], list[dict]]:
    game = None
    for line in (dataset / "games.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            g = json.loads(line)
            if g["game_id"] == game_id:
                game = g
                break
    if game is None:
        raise SystemExit(f"game {game_id} not in {dataset / 'games.jsonl'}")
    recs = []
    with (dataset / "states.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if f'"{game_id}"' in line:
                r = json.loads(line)
                if r.get("game_id") == game_id:
                    recs.append(r)
    recs.sort(key=lambda r: r["t"])
    return game["own_deck"], recs


def nearest(ts: list[float], t: float, tol: float) -> int | None:
    i = bisect.bisect_left(ts, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(ts) and abs(ts[j] - t) <= tol and (best is None or abs(ts[j] - t) < abs(ts[best] - t)):
            best = j
    return best


def mean_table(mean: dict) -> str:
    rows = ["| field | mean |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in mean.items()]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perception", type=Path, required=True, help="perception states.jsonl of the game")
    ap.add_argument("--game", required=True, help="game_id in the dataset")
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--tol", type=float, default=0.6, help="max |t_state - t_record| in seconds")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    own_deck, recs = read_game(a.dataset, a.game)
    states, events = load_perception(a.perception)
    s_ts = [s["t"] for s in states]
    # events of the match the nearest state belongs to (a perception file can hold several matches)
    match_t0: dict = {}
    for s in states:
        match_t0.setdefault(s.get("match_id"), s["t"])

    results, no_state = [], 0
    for rec in recs:
        t = rec["t"]
        i = nearest(s_ts, t, a.tol)
        if i is None:
            no_state += 1
            continue
        row = dict(states[i])
        row["t"] = t                                  # line the recent-play windows up
        t0 = match_t0.get(row.get("match_id"), 0.0)
        builder = LiveStateBuilder(LiveConfig(own_deck=list(own_deck)))
        builder.observe_events([e for e in events if t0 <= e["timestamp"] <= t])
        out = builder.build(row)
        if out is None:
            no_state += 1
            continue
        live, text = out
        results.append({"id": rec["id"], "t": t, **compare(live, text, rec)})

    mean = {f: (round(statistics.fmean(r[f] for r in results), 4) if results else None) for f in FIELDS}
    cosines = [r["feature_cosine"] for r in results]
    summary = {"game_id": a.game, "n_records": len(recs), "n_compared": len(results),
               "no_state": no_state, "mean": mean,
               "median_feature_cosine": round(statistics.median(cosines), 4) if cosines else None,
               "examples": results[:N_EXAMPLES]}
    out_path = a.out or a.dataset / "eval" / f"live_state_parity_{a.game}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"game {a.game}: {len(results)}/{len(recs)} records compared, {no_state} without a perception state")
    print(mean_table({k: v for k, v in mean.items() if v is not None}))
    print(f"median feature cosine: {summary['median_feature_cosine']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
