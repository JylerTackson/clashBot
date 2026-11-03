"""Benchmark the Qwen slow-loop planner on a few val/test decks (MAC-E, D-027).

  python3 tools/eval/bench_planner.py [--dataset DIR] [--db PATH] [--decks 5] [--model ID] [--out PATH]

Per deck: one pre-game plan and one in-game update (first state of that game with a phase, else its first
state). Writes
<dataset>/eval/planner_bench.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR, DB_PATH  # noqa: E402
from clashbot.kg import KnowledgeGraph  # noqa: E402
from clashbot.planner import DEFAULT_QWEN_ID, LLMPlanner, MLXChat  # noqa: E402
from clashbot.retrieval import StateIndex  # noqa: E402
from clashbot.tools import ToolContext  # noqa: E402

EVAL_SPLITS = ("val", "test")


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pick_decks(games_path: Path, n: int, has_phase: set[str]) -> list[dict]:
    """One game per distinct own_deck_key among val/test games, sorted by key. Within a key the first game that
    has a phase-bearing state wins (so the update can be timed), else the first game."""
    by_key: dict[str, dict] = {}
    for g in read_jsonl(games_path):
        key = g.get("own_deck_key")
        if g.get("split") not in EVAL_SPLITS or not key:
            continue
        if key not in by_key or (by_key[key]["game_id"] not in has_phase and g["game_id"] in has_phase):
            by_key[key] = g
    return [by_key[k] for k in sorted(by_key)[:n]]


def first_phase_states(states_path: Path) -> dict[str, dict]:
    """Per game, the first record whose state has a phase (some games never read the clock: their first record)."""
    out: dict[str, dict] = {}
    for rec in read_jsonl(states_path):
        cur = out.get(rec["game_id"])
        if cur is None or (cur["state"].get("phase") is None and rec["state"].get("phase") is not None):
            out[rec["game_id"]] = rec
    return out


def pct50(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 2) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--decks", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_QWEN_ID)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or args.dataset / "eval" / "planner_bench.json"

    first = first_phase_states(args.dataset / "states.jsonl")
    has_phase = {gid for gid, rec in first.items() if rec["state"].get("phase") is not None}
    games = pick_decks(args.dataset / "games.jsonl", args.decks, has_phase)

    chat = MLXChat(args.model)
    t0 = time.perf_counter()
    chat.chat([{"role": "user", "content": "Say ok."}])        # warm-up: loads the model, not counted per deck
    load_s = round(time.perf_counter() - t0, 2)

    base = ToolContext(index=StateIndex(args.db), kg=KnowledgeGraph(args.db), db_path=args.db)
    rows = []
    for g in games:
        key = g["own_deck_key"]
        planner = LLMPlanner(chat, db_path=args.db)
        ctx = dataclasses.replace(base)
        t0 = time.perf_counter()
        plan = planner.pregame(g["own_deck"], key, ctx)
        pregame_s = time.perf_counter() - t0
        trace = plan.trace
        row = {"deck_key": key, "game_id": g["game_id"], "pregame_s": round(pregame_s, 2),
               "steps": max((e["step"] for e in trace), default=-1) + 1, "tool_calls": len(trace),
               "tools_used": dict(Counter(e["tool"] for e in trace)),
               "tool_errors": sum(1 for e in trace if not e["ok"]), "plan_chars": len(plan.text),
               "source": plan.source, "plan": plan.text, "update_s": None, "update": None}
        rec = first.get(g["game_id"])
        if rec is not None:
            uctx = dataclasses.replace(base, state=rec["state"], t=rec["t"], context_text=rec["context_text"],
                                       plan_text=plan.text)
            t0 = time.perf_counter()
            new = planner.update(plan, rec["state"], rec["t"], uctx)
            row["update_s"] = round(time.perf_counter() - t0, 2)
            row["update"] = {"t": rec["t"], "source": new.source, "tool_calls": len(new.trace),
                             "changed": new is not plan, "plan": new.text}
        rows.append(row)

    calls = sum(r["tool_calls"] for r in rows)
    summary = {
        "pregame_s_p50": pct50([r["pregame_s"] for r in rows]),
        "pregame_s_max": max((r["pregame_s"] for r in rows), default=None),
        "update_s_p50": pct50([r["update_s"] for r in rows if r["update_s"] is not None]),
        "mean_tool_calls": round(calls / len(rows), 2) if rows else None,
        "tool_error_rate": round(sum(r["tool_errors"] for r in rows) / calls, 3) if calls else 0.0,
        "fallbacks": sum(1 for r in rows if r["source"] == "rules_fallback"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.model, "load_s": load_s, "decks": rows, "summary": summary},
                                   indent=2, ensure_ascii=False))

    print(f"model {args.model}  load+warmup {load_s}s  -> {out_path}\n")
    print("| deck | pregame s | steps | tool calls | errors | update s | plan chars |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        upd = "-" if r["update_s"] is None else f"{r['update_s']:.2f}"
        print(f"| {r['deck_key'][:40]} | {r['pregame_s']:.2f} | {r['steps']} | {r['tool_calls']} | "
              f"{r['tool_errors']} | {upd} | {r['plan_chars']} |")
    print("\nsummary:", json.dumps(summary))
    for r in rows:
        print(f"\nplan ({r['deck_key'][:40]}, {r['source']}):\n{r['plan']}")
        if r["update"]:
            print(f"update at t={r['update']['t']}s ({r['update']['source']}):\n{r['update']['plan']}")


if __name__ == "__main__":
    main()
