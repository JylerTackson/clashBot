"""Tune the tactical engine's scoring weights on the pro play records (MAC-A2): CPU only, no models.

  python3 tools/train/tune_tactics.py [--dataset DIR] [--passes 3] [--out clashbot/resources/tactics_weights.json]
                                      [--report PATH]

Objective, over the `<dataset>/states.jsonl` play records whose pro card is in the engine's affordable known pool
(`legal_cards`; the others are skipped), one `analyse` per record:
    mean of [best.card == pro] + 0.5 * [pro card among the cards of moves[:3]] + 0.25 * [lane_of(best.tile) == pro lane,
    when the card matches]
Search: coordinate search over `DEFAULT_WEIGHTS` on the train split. For each weight in turn try x{0.5, 0.75, 1.25, 1.5,
2.0} (and, for weights near 0, +-0.1 without changing sign), keep the best strict improvement, repeat for --passes
passes. Only the records that read a weight are re-analysed for it (the engine reads its weights as `engine.w[key]`;
a tracing dict records which keys each record read, which is exact because the engine is deterministic).
Guard against overfitting: the pass with the best val objective is kept, and only if it beats the defaults on val;
otherwise the defaults are written and the report says so. Test is only reported.
Design guards: the pro data does not know our deck or the live-match fixes, so a change is also vetoed when it breaks
a designed behaviour (`default_guards`): the weight orderings (bait used >= unknown >= ready, stop high >= medium >=
low, cycle stays the lowest offence move), and three hand-made situations (a balloon is not answered by an ice-spirit,
a push our units fight is scaled down, an ice-spirit still answers bats).
Writes the weights JSON (committed) and `<dataset>/eval/tactics_tuning.{json,md}`: train / val / test before vs after.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.game_rules import lane_of, legal_cards, load_cards  # noqa: E402
from clashbot.kg import KnowledgeGraph  # noqa: E402
from clashbot.tactics import DEFAULT_WEIGHTS, WEIGHTS_PATH, TacticalEngine  # noqa: E402

SPLITS = ("train", "val", "test")
SITUATIONS = ("all", "push", "no_push")
MULTIPLIERS = (0.5, 0.75, 1.25, 1.5, 2.0)
NEAR_ZERO = 0.15                     # |weight| at or below this also gets the +-STEP moves
STEP = 0.1
MIN_GAIN = 1e-9                      # an objective sum must improve by more than this to accept a change
OBJECTIVE_FORMULA = ("[best.card == pro] + 0.5 * [pro card in moves[:3]] "
                     "+ 0.25 * [lane_of(best.tile) == pro lane, when the card matches]")


# -- records and the objective -----------------------------------------------------------------------------------------
@dataclass
class Rec:
    id: str
    split: str
    state: dict
    t: float
    card: str
    lane: str | None
    pool: int                        # size of the affordable known card pool
    eligible: bool                   # the pro card is in that pool


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_records(dataset: Path, cards: dict, splits=SPLITS) -> dict[str, list[Rec]]:
    """Every known-card play record per split; `eligible` marks the ones the objective counts."""
    out: dict[str, list[Rec]] = {s: [] for s in splits}
    for rec in read_jsonl(Path(dataset) / "states.jsonl"):
        action = rec.get("action") or {}
        if rec.get("split") not in out or action.get("type") != "play" or not action.get("card"):
            continue
        pool = legal_cards(rec["state"], cards)
        out[rec["split"]].append(Rec(rec["id"], rec["split"], rec["state"], rec["t"], action["card"],
                                     action.get("lane"), len(pool), action["card"] in pool))
    return out


def record_score(report, rec: Rec) -> float:
    """The objective of one record: top-1 card + 0.5 x top-3 card + 0.25 x lane (when the top-1 card matches)."""
    best = report.best
    score = 0.0
    if best is not None and best.card == rec.card:
        score += 1.0
        if rec.lane is not None and lane_of(best.tile) == rec.lane:
            score += 0.25
    if any(m.card == rec.card for m in report.moves[:3]):
        score += 0.5
    return score


class Tracer(dict):
    """The engine's weights dict; remembers which keys were read since `reads` was last replaced."""

    def __init__(self, data):
        super().__init__(data)
        self.reads: set[str] = set()

    def __getitem__(self, key):
        self.reads.add(key)
        return super().__getitem__(key)


class Objective:
    """Sum of the record scores for the engine's current weights, with per-weight incremental updates."""

    def __init__(self, engine: TacticalEngine, recs: list[Rec], weights: dict[str, float]):
        self.engine, self.recs = engine, recs
        self.tracer = Tracer(weights)
        engine.w = self.tracer
        self.scores: list[float] = []
        self.touch: dict[str, set[int]] = {k: set() for k in weights}      # weight -> records that read it
        for i, rec in enumerate(recs):
            score, reads = self._eval(rec)
            self.scores.append(score)
            for k in reads:
                self.touch[k].add(i)
        self.total = sum(self.scores)

    def _eval(self, rec: Rec) -> tuple[float, set[str]]:
        self.tracer.reads = set()
        score = record_score(self.engine.analyse(rec.state, rec.t), rec)
        return score, self.tracer.reads

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.tracer)

    def propose(self, key: str, value: float):
        """(total, updates) with weights[key] = value; only the records that read `key` are analysed again."""
        old = dict.__getitem__(self.tracer, key)
        dict.__setitem__(self.tracer, key, value)
        delta, updates = 0.0, []
        for i in sorted(self.touch[key]):
            score, reads = self._eval(self.recs[i])
            delta += score - self.scores[i]
            updates.append((i, score, reads))
        dict.__setitem__(self.tracer, key, old)
        return self.total + delta, updates

    def accept(self, key: str, value: float, total: float, updates) -> None:
        dict.__setitem__(self.tracer, key, value)
        for i, score, reads in updates:
            self.scores[i] = score
            for k in reads:
                self.touch[k].add(i)                   # a changed path may read weights it did not read before
        self.total = total


def candidates(value: float) -> list[float]:
    """Trial values for one weight, the smallest step first: x{0.5, 0.75, 1.25, 1.5, 2.0}, and +-0.1 (sign kept,
    zero allowed) when the weight is near 0."""
    out = {round(value * m, 4) for m in MULTIPLIERS}
    if abs(value) <= NEAR_ZERO:
        for d in (STEP, -STEP):
            v = round(value + d, 4)
            if value == 0 or v * value >= 0:
                out.add(v)
    out.discard(round(value, 4))
    return sorted(out, key=lambda v: (round(abs(v - value), 6), v))


def mean_objective(engine: TacticalEngine, recs: list[Rec], weights: dict[str, float]) -> float:
    """Mean objective of `recs` under `weights`; the engine's own weights are left as they were."""
    saved, engine.w = engine.w, dict(weights)
    try:
        return sum(record_score(engine.analyse(r.state, r.t), r) for r in recs) / len(recs) if recs else 0.0
    finally:
        engine.w = saved


# -- design guards -----------------------------------------------------------------------------------------------------
def _guard_state(hand: list[str], elixir: float, units: list[dict]) -> dict:
    return {"phase": "double_elixir", "match_seconds": 150,
            "own": {"elixir": elixir, "hand": hand, "next_card": None, "deck": list(hand),
                    "towers": {"king": None, "left": 4000, "right": 4000}, "recent_plays": []},
            "opponent": {"elixir_estimate": 4.0, "towers": {"king": None, "left": 4000, "right": 4000},
                         "deck_known": [], "recent_plays": []},
            "units": units, "threats": []}


def _guard_unit(card: str, side: str, tile: list[float], eta: float | None = None, tower: str = "own_right") -> dict:
    return {"unit_class": card, "card": card, "side": side, "tile": tile,
            "eta_tower": None if eta is None else {"tower": tower, "s": eta}}


def guard_weight_order(engine: TacticalEngine) -> bool:
    """The orderings the intents are designed around: a bait is best when their counter is cycling and worst when it is
    ready; stopping power rises from low to high, with low a penalty; an answered push is scaled down, never up; and the
    cycle stays the lowest-priority offence move (any other intent move blocks it)."""
    w = engine.w
    return (w["bait_counter_used"] >= w["bait_unknown"] >= w["bait_counter_ready"]
            and w["stop_high"] >= w["stop_medium"] >= w["stop_low"] and w["stop_low"] <= 0
            and 0 <= w["answered_factor"] <= 1
            and max(w["cycle"], w["cycle_block_score"]) <= min(
                w[k] for k in ("bait_unknown", "bait_counter_used", "pressure", "support", "counter_push", "punish",
                               "leak", "leak_wc")))


def guard_balloon_not_ice_spirit(engine: TacticalEngine) -> bool:
    """Live match B: a balloon push with an ice-spirit, magic-archer, bomb-tower and berserker in hand."""
    state = _guard_state(["ice-spirit", "magic-archer", "bomb-tower", "berserker"], 7.0,
                         [_guard_unit("balloon", "enemy", [14.5, 12], 5.0)])
    best = engine.analyse(state, 10.0).best
    return best is not None and best.card != "ice-spirit"


def guard_answered_push_is_scaled_down(engine: TacticalEngine) -> bool:
    """A giant that a bomb-tower and a berserker are fighting scores lower than the same giant alone."""
    hand = ["tesla", "knight", "zap", "musketeer"]
    giant = _guard_unit("giant", "enemy", [15, 13], 8.0)
    alone = engine.analyse(_guard_state(hand, 9.0, [giant]), 10.0).best
    fought = engine.analyse(_guard_state(hand, 9.0, [giant, _guard_unit("bomb-tower", "ally", [15, 11]),
                                                     _guard_unit("berserker", "ally", [14, 12])]), 10.0).best
    return alone is not None and fought is not None and fought.score < alone.score


def guard_ice_spirit_still_answers_bats(engine: TacticalEngine) -> bool:
    """An ice-spirit against a swarm of bats keeps its trade (it is proposed and claims it)."""
    bats = [_guard_unit("bats", "enemy", t, 6.0, "own_left") for t in ([3, 14], [4, 14], [3.5, 15])]
    rep = engine.analyse(_guard_state(["ice-spirit", "knight", "musketeer", "tesla"], 7.0, bats), 10.0)
    return any(m.card == "ice-spirit" and "trade" in m.reason for m in rep.moves[:3])


def default_guards() -> list:
    return [guard_weight_order, guard_balloon_not_ice_spirit, guard_answered_push_is_scaled_down,
            guard_ice_spirit_still_answers_bats]


def guards_hold(engine: TacticalEngine, weights: dict[str, float], guards) -> bool:
    """True when every guard holds for `engine` run with `weights`; the engine's own weights are left as they were."""
    saved, engine.w = engine.w, dict(weights)
    try:
        return all(g(engine) for g in guards)
    finally:
        engine.w = saved


def tune(engine: TacticalEngine, train: list[Rec], val: list[Rec], passes: int = 3, log=print, guards=None) -> dict:
    """Coordinate search on `train` from DEFAULT_WEIGHTS; the pass with the best val objective is selected, and only
    when it beats the defaults. A change that breaks a guard (callables on the engine, see `default_guards`) is never
    accepted. Returns {"weights", "tuned", "history", "reason", "picked_pass", "vetoed"}."""
    saved = engine.w
    try:
        return _tune(engine, [r for r in train if r.eligible], [r for r in val if r.eligible], passes, log,
                     list(guards or []))
    finally:
        engine.w = saved


def _tune(engine: TacticalEngine, train: list[Rec], val: list[Rec], passes: int, log, guards: list) -> dict:
    if guards and not guards_hold(engine, DEFAULT_WEIGHTS, guards):
        raise RuntimeError("the default weights break a design guard; fix the guard or the defaults first")
    vetoed: dict[str, int] = {}
    obj = Objective(engine, train, dict(DEFAULT_WEIGHTS))
    n = max(len(train), 1)
    history = [{"pass": 0, "train": obj.total / n, "val": mean_objective(engine, val, obj.weights), "changes": 0}]
    snapshots = [dict(DEFAULT_WEIGHTS)]
    log(f"pass 0 (defaults): train {history[0]['train']:.4f} val {history[0]['val']:.4f}")
    for p in range(1, passes + 1):
        changes = 0
        for key in DEFAULT_WEIGHTS:
            current = dict.__getitem__(obj.tracer, key)
            best = None
            for value in candidates(current):
                total, updates = obj.propose(key, value)
                if total > obj.total + MIN_GAIN and (best is None or total > best[0]):
                    if guards and not guards_hold(engine, {**obj.weights, key: value}, guards):
                        vetoed[key] = vetoed.get(key, 0) + 1
                        continue
                    best = (total, value, updates)
            if best is not None:
                total, value, updates = best
                log(f"  pass {p} {key}: {current} -> {value} (train {obj.total / n:.4f} -> {total / n:.4f})")
                obj.accept(key, value, total, updates)
                changes += 1
        weights = obj.weights
        full = mean_objective(engine, train, weights)                  # integrity check of the incremental updates
        if abs(full - obj.total / n) > 1e-9:
            raise RuntimeError(f"incremental objective {obj.total / n:.6f} != full recompute {full:.6f}")
        history.append({"pass": p, "train": obj.total / n, "val": mean_objective(engine, val, weights),
                        "changes": changes})
        snapshots.append(weights)
        log(f"pass {p}: train {history[-1]['train']:.4f} val {history[-1]['val']:.4f} ({changes} changes)")
        if changes == 0:
            break
    # the first pass with the best val objective; the defaults (pass 0) win ties and anything that does not beat them
    pick = 0
    for i in range(1, len(history)):
        if history[i]["val"] > history[pick]["val"] + MIN_GAIN:
            pick = i
    tuned = pick > 0
    if tuned:
        reason = (f"pass {pick} kept: val objective {history[0]['val']:.4f} -> {history[pick]['val']:.4f}")
    else:
        reason = "val objective did not improve over the defaults; the defaults are kept"
    return {"weights": snapshots[pick] if tuned else dict(DEFAULT_WEIGHTS), "tuned": tuned, "history": history,
            "reason": reason, "picked_pass": pick if tuned else 0, "vetoed": vetoed}


# -- reports -----------------------------------------------------------------------------------------------------------
def _rate(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def summarise(engine: TacticalEngine, recs: list[Rec], weights: dict[str, float]) -> dict:
    """Metrics of the engine under `weights` on one split, all / push / no push. n, top-1, top-3 and coverage are over
    every known-card play record (as `eval_tactics.py`; a pro card outside the pool can never match), the objective
    over the eligible ones; `random_top1` is the expected top-1 of a uniform pick from the pool, on the same base.
    `by_intent`: how often the best move has each intent and how often its card is the pro card."""
    saved, engine.w = engine.w, dict(weights)
    rows = {sit: {"n": 0, "n_elig": 0, "obj": 0.0, "top1": 0, "top3": 0, "covered": 0, "inv_pool": 0.0,
                  "lane_hit": 0} for sit in SITUATIONS}
    by_intent: dict[str, list[int]] = {}
    for rec in recs:
        report = engine.analyse(rec.state, rec.t)
        best = report.best
        hit = best is not None and best.card == rec.card
        if best is not None:
            cell = by_intent.setdefault(best.intent, [0, 0])
            cell[0] += 1
            cell[1] += hit
        for sit in ("all", "push" if report.pushes else "no_push"):
            row = rows[sit]
            row["n"] += 1
            row["top1"] += hit
            row["top3"] += any(m.card == rec.card for m in report.moves[:3])
            row["covered"] += best is not None
            row["lane_hit"] += hit and rec.lane is not None and lane_of(best.tile) == rec.lane
            if rec.eligible:
                row["n_elig"] += 1
                row["obj"] += record_score(report, rec)
                row["inv_pool"] += 1.0 / max(rec.pool, 1)
    engine.w = saved
    out = {"n_play": len(recs), "n_eligible": sum(r.eligible for r in recs), "situations": {}, "by_intent": {}}
    for sit, row in rows.items():
        n = row["n"]
        out["situations"][sit] = {"n": n, "n_eligible": row["n_elig"], "objective": _rate(row["obj"], row["n_elig"]),
                                  "top1": _rate(row["top1"], n), "top3": _rate(row["top3"], n),
                                  "coverage": _rate(row["covered"], n),
                                  "lane_when_top1": _rate(row["lane_hit"], row["top1"]),
                                  "random_top1": _rate(row["inv_pool"], n)}
    for intent, (cnt, hits) in sorted(by_intent.items()):
        out["by_intent"][intent] = {"n": cnt, "share": _rate(cnt, len(recs)), "precision": _rate(hits, cnt)}
    return out


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def _pair(before, after, pct=True) -> str:
    f = _pct if pct else (lambda v: "-" if v is None else f"{v:.4f}")
    return f"{f(before)} -> {f(after)}"


def render_markdown(result: dict) -> str:
    lines = ["# Tactics weight tuning (MAC-A2)", "",
             f"- dataset: `{result['dataset']}`, KG: {result['kg']}, passes: {result['passes']}",
             f"- objective: `{OBJECTIVE_FORMULA}`, mean over play records whose pro card is in the affordable known pool "
             "(`eligible`); n, top-1, top-3 and coverage are over all known-card play records, like `eval_tactics.py`; "
             "random top-1 = a uniform pick from the pool, same base",
             f"- decision: **{'tuned weights kept' if result['tuned'] else 'defaults kept'}** - {result['reason']}", "",
             "## Objective, top-1, top-3 and coverage: before -> after", "",
             "| split | situation | n | n eligible | objective | top-1 card | top-3 card | coverage | random top-1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for split in SPLITS:
        b, a = result["before"][split], result["after"][split]
        for sit in SITUATIONS:
            rb, ra = b["situations"][sit], a["situations"][sit]
            lines.append(f"| {split} | {sit.replace('_', ' ')} | {rb['n']} | {rb['n_eligible']} "
                         f"| {_pair(rb['objective'], ra['objective'], False)} "
                         f"| {_pair(rb['top1'], ra['top1'])} | {_pair(rb['top3'], ra['top3'])} "
                         f"| {_pair(rb['coverage'], ra['coverage'])} | {_pct(rb['random_top1'])} |")
    lines += ["", "## By intent of the best move: before -> after", "",
              "| split | intent | n best | share of play records | card precision |", "|---|---|---|---|---|"]
    for split in SPLITS:
        b, a = result["before"][split]["by_intent"], result["after"][split]["by_intent"]
        for intent in sorted(set(b) | set(a)):
            rb, ra = b.get(intent, {}), a.get(intent, {})
            lines.append(f"| {split} | {intent} | {rb.get('n', 0)} -> {ra.get('n', 0)} "
                         f"| {_pair(rb.get('share'), ra.get('share'))} "
                         f"| {_pair(rb.get('precision'), ra.get('precision'))} |")
    lines += ["", "## Records", "", "| split | known-card plays | eligible (pro card in the pool) |", "|---|---|---|"]
    for split in SPLITS:
        lines.append(f"| {split} | {result['before'][split]['n_play']} | {result['before'][split]['n_eligible']} |")
    lines += ["", "## Search", "", "| pass | train objective | val objective | weights changed |", "|---|---|---|---|"]
    for h in result["history"]:
        lines.append(f"| {h['pass']} | {h['train']:.4f} | {h['val']:.4f} | {h['changes']} |")
    vetoed = result.get("vetoed") or {}
    lines += ["", "Candidates vetoed by a design guard (improved train, broke a guard): "
              + (", ".join(f"{k} x{n}" for k, n in vetoed.items()) if vetoed else "none")]
    lines += ["", "## Weights that changed", "", "| weight | default | tuned |", "|---|---|---|"]
    if result["changed"]:
        lines += [f"| {k} | {v[0]} | {v[1]} |" for k, v in result["changed"].items()]
    else:
        lines.append("| (none) | | |")
    lines.append("")
    return "\n".join(lines)


def load_kg():
    """The knowledge graph if the db loads, else None (the engine then skips its KG bonus)."""
    try:
        return KnowledgeGraph()
    except Exception as exc:                      # noqa: BLE001 - any load failure just means "no KG"
        print(f"note: no knowledge graph ({type(exc).__name__}: {exc}); tuning without KG scores")
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH, help="weights JSON (default clashbot/resources/...)")
    ap.add_argument("--report", type=Path, default=None, help="report JSON (default <dataset>/eval/tactics_tuning.json)")
    args = ap.parse_args(argv)
    dataset = Path(args.dataset)
    report_path = Path(args.report) if args.report else dataset / "eval" / "tactics_tuning.json"

    started = time.time()
    cards = load_cards()
    kg = load_kg()
    engine = TacticalEngine(cards, kg=kg, weights=dict(DEFAULT_WEIGHTS))        # ignore any existing weights file
    recs = load_records(dataset, cards)
    print({s: (len(v), sum(r.eligible for r in v)) for s, v in recs.items()}, "(known-card plays, eligible)")

    before = {s: summarise(engine, recs[s], DEFAULT_WEIGHTS) for s in SPLITS}
    outcome = tune(engine, recs["train"], recs["val"], args.passes, guards=default_guards())
    weights = outcome["weights"]
    after = {s: summarise(engine, recs[s], weights) for s in SPLITS}
    changed = {k: [DEFAULT_WEIGHTS[k], weights[k]] for k in DEFAULT_WEIGHTS if weights[k] != DEFAULT_WEIGHTS[k]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": 1, "tuned": outcome["tuned"], "note": outcome["reason"],
                               "weights": weights}, indent=2) + "\n")
    result = {"dataset": str(dataset), "kg": kg is not None, "passes": args.passes, "objective": OBJECTIVE_FORMULA,
              "tuned": outcome["tuned"], "reason": outcome["reason"], "picked_pass": outcome["picked_pass"],
              "history": outcome["history"], "vetoed": outcome["vetoed"], "changed": changed, "weights_before": dict(DEFAULT_WEIGHTS),
              "weights_after": weights, "before": before, "after": after,
              "seconds": round(time.time() - started, 1)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    md = render_markdown(result)
    report_path.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out}, {report_path} and {report_path.with_suffix('.md')} ({result['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
