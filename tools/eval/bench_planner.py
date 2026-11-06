"""Benchmark the Qwen slow-loop planner on a few val/test decks (MAC-E, D-027), or the Qwen3-14B coach (D-046).

  python3 tools/eval/bench_planner.py [--dataset DIR] [--db PATH] [--decks 5] [--model ID] [--out PATH]
                                      [--coach] [--updates N] [--period S]

Per deck: one pre-game plan and one in-game update (first state of that game with a phase, else its first state).
Writes <dataset>/eval/planner_bench.json and prints a markdown table. Model load and warm-up are timed apart; the
tokens/s are mlx-lm's own (prompt prefill and decode), the peak memory is MLX's.

`--coach` benches `CoachPlanner` (default model COACH_MODEL_ID): the same pre-game plan, then `--updates` coach
updates per deck (default 3, at least `--period` game seconds apart, each fed the tactics engine's lines and the
lookup summary, each revising the plan before it) as `update_s` p50 / p95, with the prefill time of each update.
The coach reuses the KV cache of the prompt prefix (`--no-prefix-cache` turns that off to compare), `--prefill-step N`
sets the prompt tokens per prefill chunk, `--yield-ms` a sleep after every chunk and `--token-yield-ms` one after every
token.
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
from clashbot.planner import (COACH_MODEL_ID, COACH_PREFILL_STEP, COACH_TOKEN_YIELD_S, COACH_YIELD_S,  # noqa: E402
                              DEFAULT_QWEN_ID, CoachPlanner, LLMPlanner, MLXChat, coach_chat, lookup_summary)
from clashbot.retrieval import StateIndex  # noqa: E402
from clashbot.tools import ToolContext  # noqa: E402

EVAL_SPLITS = ("val", "test")
UPDATE_PERIOD_S = 10.0       # game seconds between the benched coach updates


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


def game_records(states_path: Path, game_ids: set[str]) -> dict[str, list[dict]]:
    """Per game id, its records sorted by time."""
    out: dict[str, list[dict]] = {g: [] for g in game_ids}
    for rec in read_jsonl(states_path):
        if rec["game_id"] in out:
            out[rec["game_id"]].append(rec)
    return {g: sorted(recs, key=lambda r: r["t"]) for g, recs in out.items()}


def pick_update_records(recs: list[dict], first: dict | None, n: int, period: float) -> list[dict]:
    """`first` (the first phase-bearing record) and then up to n-1 later records at least `period` game seconds apart."""
    if first is None or n <= 0:
        return []
    picked = [first]
    for rec in recs:
        if len(picked) >= n:
            break
        if rec["t"] >= picked[-1]["t"] + period:
            picked.append(rec)
    return picked


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 2)


def pct50(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 2) if xs else None


class StatChat:
    """Wraps a chat object and keeps `last_stats` of every call, for tokens/s over a whole pregame or update."""

    def __init__(self, chat):
        self.chat_obj = chat
        self.log: list[dict] = []

    def chat(self, messages, tools=None, **kwargs):
        out = self.chat_obj.chat(messages, tools, **kwargs)
        self.log.append(dict(getattr(self.chat_obj, "last_stats", {}) or {}))
        return out

    def __getattr__(self, name):               # prefix_cache etc.: what the wrapped chat has
        return getattr(self.chat_obj, name)

    def take(self) -> dict:
        """Totals of the calls since the last `take`, then reset."""
        log, self.log = self.log, []
        gen = sum(s.get("gen_tokens", 0) for s in log)
        pre = sum(s.get("prompt_tokens", 0) for s in log)
        ms = sum(s.get("gen_ms", 0.0) for s in log)
        prefill_ms = sum(s.get("prefill_ms", 0.0) for s in log)
        cached = sum(s.get("cached_tokens", 0) for s in log)
        decode_s = sum(s.get("gen_tokens", 0) / s["gen_tps"] for s in log if s.get("gen_tps"))
        return {"calls": len(log), "prompt_tokens": pre, "cached_tokens": cached, "prefill_ms": round(prefill_ms, 1),
                "gen_tokens": gen,
                "tokens_per_s": round(gen / (ms / 1000), 1) if ms else None,        # incl. prefill
                "decode_tps": round(gen / decode_s, 1) if decode_s else None}


def peak_gb() -> float | None:
    try:
        import mlx.core as mx
        return round(mx.get_peak_memory() / 1e9, 2)
    except (ImportError, AttributeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--decks", type=int, default=5)
    ap.add_argument("--model", default=None, help="default: the Qwen3-4B planner, or the coach model with --coach")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--coach", action="store_true", help="bench CoachPlanner (Qwen3-14B, thinking off)")
    ap.add_argument("--updates", type=int, default=None, help="updates per deck (default 1; 3 with --coach)")
    ap.add_argument("--period", type=float, default=UPDATE_PERIOD_S, help="game seconds between the coach updates")
    ap.add_argument("--no-prefix-cache", action="store_true", help="coach: prefill the whole prompt every time")
    ap.add_argument("--prefill-step", type=int, default=COACH_PREFILL_STEP, help="coach: prompt tokens per prefill chunk")
    ap.add_argument("--yield-ms", type=float, default=COACH_YIELD_S * 1000, help="coach: sleep per prefill chunk, ms")
    ap.add_argument("--token-yield-ms", type=float, default=COACH_TOKEN_YIELD_S * 1000,
                    help="coach: sleep per generated token, ms")
    args = ap.parse_args()
    model = args.model or (COACH_MODEL_ID if args.coach else DEFAULT_QWEN_ID)
    n_updates = args.updates if args.updates is not None else (3 if args.coach else 1)
    out_path = args.out or args.dataset / "eval" / "planner_bench.json"

    first = first_phase_states(args.dataset / "states.jsonl")
    has_phase = {gid for gid, rec in first.items() if rec["state"].get("phase") is not None}
    games = pick_decks(args.dataset / "games.jsonl", args.decks, has_phase)
    recs_by_game = game_records(args.dataset / "states.jsonl", {g["game_id"] for g in games}) if args.coach else {}

    try:
        import mlx.core as mx
        mx.reset_peak_memory()
    except (ImportError, AttributeError):
        pass
    raw = (coach_chat(model, prefix_cache=not args.no_prefix_cache, prefill_step=args.prefill_step,
                      yield_s=args.yield_ms / 1000.0, token_yield_s=args.token_yield_ms / 1000.0)
           if args.coach else MLXChat(model))
    t0 = time.perf_counter()
    raw.load()
    load_s = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    raw.chat([{"role": "user", "content": "Say ok."}])          # warm-up: kernels compile, not counted per deck
    warmup_s = round(time.perf_counter() - t0, 2)
    chat = StatChat(raw)

    base = ToolContext(index=StateIndex(args.db), kg=KnowledgeGraph(args.db), db_path=args.db)
    engine = None
    if args.coach:
        from clashbot.game_rules import load_cards
        from clashbot.tactics import TacticalEngine
        engine = TacticalEngine(load_cards(), kg=base.kg)
    rows = []
    for g in games:
        key = g["own_deck_key"]
        planner = (CoachPlanner if args.coach else LLMPlanner)(chat, db_path=args.db)
        ctx = dataclasses.replace(base)
        t0 = time.perf_counter()
        plan = planner.pregame(g["own_deck"], key, ctx)
        pregame_s = time.perf_counter() - t0
        pre_stats = chat.take()
        if args.coach and not args.no_prefix_cache:
            planner.prime(g["own_deck"])               # as the live worker does after the pregame plan
            chat.take()
        trace = plan.trace
        row = {"deck_key": key, "game_id": g["game_id"], "pregame_s": round(pregame_s, 2),
               "steps": max((e["step"] for e in trace), default=-1) + 1, "tool_calls": len(trace),
               "tools_used": dict(Counter(e["tool"] for e in trace)),
               "tool_errors": sum(1 for e in trace if not e["ok"]), "plan_chars": len(plan.text),
               "source": plan.source, "plan": plan.text, "pregame_stats": pre_stats,
               "update_s": None, "update": None, "updates": []}
        rec = first.get(g["game_id"])
        if rec is not None and not args.coach:
            uctx = dataclasses.replace(base, state=rec["state"], t=rec["t"], context_text=rec["context_text"],
                                       plan_text=plan.text)
            t0 = time.perf_counter()
            new = planner.update(plan, rec["state"], rec["t"], uctx)
            row["update_s"] = round(time.perf_counter() - t0, 2)
            row["update"] = {"t": rec["t"], "source": new.source, "tool_calls": len(new.trace),
                             "changed": new is not plan, "plan": new.text}
            row["updates"] = [{"t": rec["t"], "s": row["update_s"], "plan": new.text, **chat.take()}]
        elif args.coach:
            current = plan
            for urec in pick_update_records(recs_by_game.get(g["game_id"], []), rec, n_updates, args.period):
                state, t = urec["state"], urec["t"]
                lines = engine.analyse(state, t).lines
                hits = base.index.search(state, t, urec["context_text"], k=10, splits=("train",),
                                         exclude_game=g["game_id"])
                t0 = time.perf_counter()
                new = planner.update(current, state, t, tactics_lines=lines, lookup=lookup_summary(hits),
                                     context_text=urec["context_text"])
                s = round(time.perf_counter() - t0, 2)
                row["updates"].append({"t": t, "s": s, "plan": new.text, "changed": new is not current,
                                       **chat.take()})
                current = new
            if row["updates"]:
                row["update_s"] = row["updates"][0]["s"]
                row["update"] = {"t": row["updates"][0]["t"], "source": "coach", "tool_calls": 0,
                                 "changed": row["updates"][0]["changed"], "plan": row["updates"][0]["plan"]}
        rows.append(row)

    calls = sum(r["tool_calls"] for r in rows)
    upd = [u["s"] for r in rows for u in r["updates"]]
    dec = [u["decode_tps"] for r in rows for u in r["updates"] if u.get("decode_tps")]
    dec += [r["pregame_stats"]["decode_tps"] for r in rows if r["pregame_stats"].get("decode_tps")]
    summary = {
        "pregame_s_p50": pct50([r["pregame_s"] for r in rows]),
        "pregame_s_max": max((r["pregame_s"] for r in rows), default=None),
        "update_s_p50": pct50(upd),
        "update_s_p95": pct(upd, 0.95),
        "update_s_max": max(upd, default=None),
        "update_prefill_ms_p50": pct50([u["prefill_ms"] for r in rows for u in r["updates"] if "prefill_ms" in u]),
        "update_prefill_ms_warm_p50": pct50([u["prefill_ms"] for r in rows for u in r["updates"][1:]
                                             if "prefill_ms" in u]),     # after the first update of a deck
        "update_cached_tokens_mean": pct50([u["cached_tokens"] for r in rows for u in r["updates"][1:]
                                            if "cached_tokens" in u]),
        "decode_tps_p50": pct50(dec),
        "tokens_per_s_p50": pct50([u["tokens_per_s"] for r in rows for u in r["updates"] if u.get("tokens_per_s")]),
        "peak_gb": peak_gb(),
        "mean_tool_calls": round(calls / len(rows), 2) if rows else None,
        "tool_error_rate": round(sum(r["tool_errors"] for r in rows) / calls, 3) if calls else 0.0,
        "fallbacks": sum(1 for r in rows if r["source"] == "rules_fallback"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": model, "coach": args.coach, "load_s": load_s, "warmup_s": warmup_s,
                                    "decks": rows, "summary": summary}, indent=2, ensure_ascii=False))

    print(f"model {model}  load {load_s}s  warm-up {warmup_s}s  peak {summary['peak_gb']} GB  -> {out_path}\n")
    print("| deck | pregame s | steps | tool calls | errors | update s | plan chars |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        upd_s = "-" if not r["updates"] else "/".join(f"{u['s']:.2f}" for u in r["updates"])
        print(f"| {r['deck_key'][:40]} | {r['pregame_s']:.2f} | {r['steps']} | {r['tool_calls']} | "
              f"{r['tool_errors']} | {upd_s} | {r['plan_chars']} |")
    print("\nsummary:", json.dumps(summary))
    for r in rows:
        print(f"\nplan ({r['deck_key'][:40]}, {r['source']}):\n{r['plan']}")
        for u in r["updates"]:
            print(f"update at t={u['t']}s ({u['s']}s, prefill {u.get('prefill_ms')} ms of {u.get('prompt_tokens')} "
                  f"tokens ({u.get('cached_tokens')} cached), {u.get('gen_tokens')} tokens, {u.get('decode_tps')} tok/s)"
                  f":\n{u['plan']}")


if __name__ == "__main__":
    main()
