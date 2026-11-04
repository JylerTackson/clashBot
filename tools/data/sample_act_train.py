"""Balanced sample of the dense wait/act dataset for a d1 fine-tune (MAC-X, D-040).

  python3 tools/data/sample_act_train.py [--src <dataset>/act_dense] [--out <dataset>/act_dense_bal]
         [--pos-per-game 20] [--neg-per-game 40] [--eval-per-game 120] [--seed 0]

The full dense data (build_dense_act.py) is ~40 K lines, too many for a d1 fine-tune at ~1 row/s. Train: per game at
most `--pos-per-game` positives (act = true) and `--neg-per-game` negatives, sampled without replacement, then all
train lines shuffled. Val / test: per game at most `--eval-per-game` lines sampled uniformly (the natural act rate
is kept), file order kept. The matching `states_<split>.jsonl` sidecar lines are copied next to each output file
(same order). Writes <out>/{train,val,test}.jsonl, <out>/states_<split>.jsonl and <out>/stats.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402

SPLITS = ("train", "val", "test")


def read_lines(path: Path) -> list[tuple[dict, str]]:
    """(parsed, raw text) of every non-empty line; [] when the file is missing."""
    if not path.exists():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            out.append((json.loads(raw), raw))
    return out


def by_game(rows: list[tuple[dict, str]]) -> dict[str, list[tuple[dict, str]]]:
    """Rows grouped by game_id, games in order of first appearance, rows in file order."""
    games: dict[str, list[tuple[dict, str]]] = {}
    for row in rows:
        games.setdefault(row[0]["game_id"], []).append(row)
    return games


def is_act(row: dict) -> bool:
    return row["answers"]["act"] == "true"


def sample_train(rows: list[tuple[dict, str]], pos_per_game: int, neg_per_game: int,
                 rng: random.Random) -> list[tuple[dict, str]]:
    picked: list[tuple[dict, str]] = []
    for game_rows in by_game(rows).values():
        pos = [r for r in game_rows if is_act(r[0])]
        neg = [r for r in game_rows if not is_act(r[0])]
        picked += rng.sample(pos, min(len(pos), pos_per_game))
        picked += rng.sample(neg, min(len(neg), neg_per_game))
    rng.shuffle(picked)
    return picked


def sample_eval(rows: list[tuple[dict, str]], per_game: int, rng: random.Random) -> list[tuple[dict, str]]:
    picked: list[tuple[dict, str]] = []
    for game_rows in by_game(rows).values():
        idx = sorted(rng.sample(range(len(game_rows)), min(len(game_rows), per_game)))
        picked += [game_rows[i] for i in idx]
    return picked


def split_stats(rows: list[tuple[dict, str]]) -> dict:
    acts = sum(is_act(r[0]) for r in rows)
    return {"lines": len(rows), "act_rate": round(acts / len(rows), 4) if rows else None,
            "games": len({r[0]["game_id"] for r in rows})}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=DATASET_DIR / "act_dense", help="build_dense_act.py output")
    ap.add_argument("--out", type=Path, default=DATASET_DIR / "act_dense_bal")
    ap.add_argument("--pos-per-game", type=int, default=20)
    ap.add_argument("--neg-per-game", type=int, default=40)
    ap.add_argument("--eval-per-game", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if min(a.pos_per_game, a.neg_per_game, a.eval_per_game) < 0:
        ap.error("--pos-per-game, --neg-per-game and --eval-per-game must be >= 0")
    if a.src.resolve() == a.out.resolve():
        ap.error("--out must differ from --src")

    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"args": {"src": str(a.src), "pos_per_game": a.pos_per_game, "neg_per_game": a.neg_per_game,
                              "eval_per_game": a.eval_per_game, "seed": a.seed}, "splits": {}}
    for split in SPLITS:
        rows = read_lines(a.src / f"{split}.jsonl")
        picked = (sample_train(rows, a.pos_per_game, a.neg_per_game, rng) if split == "train"
                  else sample_eval(rows, a.eval_per_game, rng))
        (a.out / f"{split}.jsonl").write_text("".join(raw + "\n" for _, raw in picked), encoding="utf-8")
        stats = split_stats(picked)
        sidecar = a.src / f"states_{split}.jsonl"
        if sidecar.exists():
            state_rows = {row["id"]: raw for row, raw in read_lines(sidecar)}
            missing = [r["id"] for r, _ in picked if r["id"] not in state_rows]
            (a.out / f"states_{split}.jsonl").write_text(
                "".join(state_rows[r["id"]] + "\n" for r, _ in picked if r["id"] in state_rows), encoding="utf-8")
            stats["states_missing"] = len(missing)
        else:
            stats["states_missing"] = None            # no sidecar in the source: none written
        summary["splits"][split] = stats
        print(f"{split}: {stats['lines']} lines, act rate {stats['act_rate']}, {stats['games']} games")
    (a.out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
