"""Write coach plans for the pro games, as training context for d1 (D-046, MAC-AE E1).

  python3 tools/data/build_coach_plans.py [--dataset DIR] [--period 10] [--limit-games N] [--out FILE]
                                          [--db PATH] [--model ID]

For every game of <dataset>/games.jsonl the game's records of <dataset>/states.jsonl are walked in time. A plan is
written at the first record, every `--period` game seconds after the last one, and at a record that shows a new
opponent card (`planner.should_update`, the cadence of the live CoachWorker). Each plan is one `CoachPlanner.update`
(Qwen3-14B, no tools) on the record's state, the tactics engine's lines and what the train-split pros played in similar
states (the same game excluded). The chain starts from the rules pre-game plan of the game's deck and every update sees
the plan before it. Lines: {game_id, t, plan, latency_s}, appended and flushed one by one to <dataset>/coach_plans.jsonl.
The run is resumable: a (game_id, t) pair already in the file is not planned again (its plan still seeds the chain).
Heavy (hours); needs the GPU. `build_d1_tac.py --plans FILE` turns the plans into d1 training context.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR, DB_PATH  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.kg import KnowledgeGraph  # noqa: E402
from clashbot.planner import (COACH_MODEL_ID, CoachPlanner, Plan, RuleBasedPlanner, lookup_summary,  # noqa: E402
                              should_update)
from clashbot.retrieval import StateIndex  # noqa: E402
from clashbot.state_text import render_context  # noqa: E402
from clashbot.tactics import TacticalEngine  # noqa: E402


PLAN_PERIOD_S = 10.0         # training plans every 10 game seconds (the live coach runs every 20 s, in quiet time)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: Path) -> dict[str, dict[float, str]]:
    """game_id -> {t: plan} of the plans already written."""
    done: dict[str, dict[float, str]] = defaultdict(dict)
    for row in read_jsonl(path):
        done[row["game_id"]][row["t"]] = row["plan"]
    return done


def seed_plan(game: dict, rules: RuleBasedPlanner) -> Plan:
    """The rules pre-game plan of the game's deck: what the first update revises."""
    deck = game.get("own_deck") or []
    try:
        return rules.pregame(deck, game.get("own_deck_key"))
    except (sqlite3.Error, FileNotFoundError):
        return Plan("Deck: " + ", ".join(deck), "rules")


def plan_game(game_id: str, records: list[dict], planner, engine, index, seed: Plan, period: float = PLAN_PERIOD_S,
              done: dict[float, str] | None = None, clock=time.perf_counter):
    """Yield one row per new plan of a game. The cadence is a function of the records only, so a resumed run lands on
    the same times; a time already in `done` is not planned again but its plan stays the current one."""
    done = done or {}
    plan, last_t, seen = seed, None, set()
    for rec in sorted(records, key=lambda r: r["t"]):      # stable: equal times keep file order
        t, state = rec["t"], rec["state"]
        known = (state.get("opponent") or {}).get("deck_known") or []
        if not should_update(last_t, t, known, seen, period):
            continue
        last_t = t
        seen.update(known)
        if t in done:
            plan = Plan(done[t], "coach")
            continue
        context = rec.get("context_text") or render_context(state, t)
        lines = engine.analyse(state, t).lines
        hits = index.search(state, t, context, k=10, splits=("train",), exclude_game=game_id)
        t0 = clock()
        new = planner.update(plan, state, t, tactics_lines=lines, lookup=lookup_summary(hits), context_text=context)
        latency = round(clock() - t0, 2)
        if new is plan:                 # the model wrote nothing usable: no row, so a rerun tries again
            continue
        plan = new
        yield {"game_id": game_id, "t": t, "plan": new.text, "latency_s": latency}


def group_records(path: Path) -> dict[str, list[dict]]:
    by_game: dict[str, list[dict]] = defaultdict(list)
    for rec in read_jsonl(path):
        by_game[rec["game_id"]].append(rec)
    return by_game


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--period", type=float, default=PLAN_PERIOD_S, help="game seconds between plans")
    ap.add_argument("--limit-games", type=int, default=None, help="plan only the first N games")
    ap.add_argument("--out", type=Path, default=None, help="plans file (default <dataset>/coach_plans.jsonl)")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--model", default=COACH_MODEL_ID, help="mlx-lm coach model")
    args = ap.parse_args(argv)
    dataset = Path(args.dataset)
    out = Path(args.out) if args.out else dataset / "coach_plans.jsonl"

    games = read_jsonl(dataset / "games.jsonl")
    by_game = group_records(dataset / "states.jsonl")
    games = [g for g in games if g["game_id"] in by_game]
    if args.limit_games is not None:
        games = games[:args.limit_games]
    done = load_done(out)

    cards = load_cards()
    kg = KnowledgeGraph(args.db)
    index = StateIndex(args.db)
    engine = TacticalEngine(cards, kg=kg)
    planner = CoachPlanner(model_path=args.model, db_path=args.db)
    rules = RuleBasedPlanner(args.db)

    out.parent.mkdir(parents=True, exist_ok=True)
    total, t_start = 0, time.perf_counter()
    with open(out, "a") as fh:
        for i, game in enumerate(games, 1):
            gid = game["game_id"]
            new = 0
            for row in plan_game(gid, by_game[gid], planner, engine, index, seed_plan(game, rules), args.period,
                                 done.get(gid)):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                new += 1
            total += new
            elapsed = time.perf_counter() - t_start
            print(f"game {i}/{len(games)} {gid}: {new} new plans ({len(done.get(gid, {}))} already written); "
                  f"{total} total, {elapsed / 60:.1f} min", flush=True)
    print(f"{total} plans -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
