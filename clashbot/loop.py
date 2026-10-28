"""Two-speed loop, fast path: should_decide -> candidates -> ranker -> executor; plus the replay CLI (D-007)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

import numpy as np

from clashbot.actions import load_placement_prior, option_text, placement_tile
from clashbot.candidates import Candidate, CandidateGenerator
from clashbot.config import DATASET_DIR, HOLD_KEY
from clashbot.game_rules import legal_cards, load_cards
from clashbot.kg import KnowledgeGraph
from clashbot.planner import Plan, RuleBasedPlanner
from clashbot.ranker import Ranker, ScoreRanker
from clashbot.retrieval import StateIndex
from clashbot.executor import DryRunExecutor


@dataclass
class Decision:
    t: float
    candidates: list[Candidate]
    choice: str
    tile: list[float] | None
    timings_ms: dict          # {"candidates": x, "rank": y, "execute": z, "total": w}


def should_decide(state: dict, t: float, last_decision_t: float | None, cards: dict) -> bool:
    if not legal_cards(state, cards):
        return False
    for u in state.get("units", []):
        eta = u.get("eta_tower")
        if (u.get("side") == "enemy" and isinstance(eta, dict)
                and str(eta.get("tower", "")).startswith("own")
                and eta.get("s") is not None and eta["s"] <= 4.0):
            return True
    if (state["own"].get("elixir") or 0) >= 9:
        return True
    if last_decision_t is None or t - last_decision_t >= 3.0:
        return True
    for p in state["opponent"].get("recent_plays", []):
        if p.get("t") is not None and 0 <= t - p["t"] <= 2.0:
            return True
    return False


class FastLoop:
    def __init__(self, generator: CandidateGenerator, ranker: Ranker, executor, prior: dict,
                 plan: Plan | None = None):
        self.generator = generator
        self.ranker = ranker
        self.executor = executor
        self.prior = prior
        self.plan = plan

    def decide(self, state: dict, t: float, context_text: str, **filters) -> Decision:
        t0 = time.perf_counter()
        candidates = self.generator.generate(state, t, context_text, n=4, **filters)
        t1 = time.perf_counter()
        if not candidates:
            t1 = time.perf_counter()
            timings = {"candidates": (t1 - t0) * 1000, "rank": 0.0, "execute": 0.0, "total": (t1 - t0) * 1000}
            return Decision(t=t, candidates=[], choice=HOLD_KEY, tile=None, timings_ms=timings)
        options = [option_text(c.key) for c in candidates]
        res = self.ranker.choose(context_text, options, self.plan.text if self.plan else None)
        choice = candidates[res.index].key
        tile = placement_tile(choice, self.prior)
        t2 = time.perf_counter()
        self.executor.execute(choice, tile, t)
        t3 = time.perf_counter()
        timings = {"candidates": (t1 - t0) * 1000, "rank": (t2 - t1) * 1000,
                   "execute": (t3 - t2) * 1000, "total": (t3 - t0) * 1000}
        return Decision(t=t, candidates=candidates, choice=choice, tile=tile, timings_ms=timings)


def _load_game(game_id: str) -> tuple[dict, list[dict]]:
    game = None
    for line in (DATASET_DIR / "games.jsonl").read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            if g["game_id"] == game_id:
                game = g
                break
    if game is None:
        raise SystemExit(f"unknown game id: {game_id}")
    records = []
    for line in (DATASET_DIR / "states.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["game_id"] == game_id:
                records.append(r)
    records.sort(key=lambda r: r["t"])
    return game, records


def replay(game_id: str, limit: int | None = None) -> int:
    game, records = _load_game(game_id)
    cards = load_cards()
    index = StateIndex()
    kg = KnowledgeGraph()
    generator = CandidateGenerator(index, kg, cards)
    executor = DryRunExecutor()
    plan = RuleBasedPlanner().pregame(game["own_deck"], game.get("own_deck_key"))
    loop = FastLoop(generator, ScoreRanker(), executor, load_placement_prior(), plan)

    last_t: float | None = None
    decisions: list[Decision] = []
    exact = card_hits = labelled = 0
    for rec in records:
        if limit is not None and len(decisions) >= limit:
            break
        state, t = rec["state"], rec["t"]
        if not should_decide(state, t, last_t, cards):
            continue
        d = loop.decide(state, t, rec["context_text"], exclude_game=game_id)
        last_t = t
        decisions.append(d)
        pro = (rec.get("action") or {}).get("key")
        pro_card = (rec.get("action") or {}).get("card")
        match = "n/a"
        if pro is not None:
            labelled += 1
            match = "yes" if d.choice == pro else "no"
            exact += d.choice == pro
            card_hits += d.choice.split("|")[0] == pro_card
        print(f"t={t:7.1f}  choice={d.choice:<40} pro={pro}  match={match}  {d.timings_ms['total']:.1f} ms")

    print(f"\nsummary: decisions={len(decisions)} labelled={labelled}")
    if labelled:
        print(f"exact-key agreement: {exact}/{labelled} = {exact / labelled:.3f}")
        print(f"card agreement:      {card_hits}/{labelled} = {card_hits / labelled:.3f}")
    else:
        print("exact-key agreement: n/a (no labelled decisions)")
        print("card agreement:      n/a (no labelled decisions)")
    for field in ("candidates", "rank", "execute", "total"):
        vals = [d.timings_ms[field] for d in decisions]
        if vals:
            print(f"latency {field:<10} p50={np.percentile(vals, 50):.2f} ms  p95={np.percentile(vals, 95):.2f} ms")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m clashbot.loop")
    ap.add_argument("--replay", metavar="GAME_ID", required=True)
    ap.add_argument("--limit", type=int, default=None)
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(argv[:-1]):      # game ids may start with "-": argparse would read them as options
        if a == "--replay":
            argv[i:i + 2] = [f"--replay={argv[i + 1]}"]
            break
    args = ap.parse_args(argv)
    return replay(args.replay, args.limit)


if __name__ == "__main__":
    sys.exit(main())
