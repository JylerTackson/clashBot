"""Live loop CLI (D-007, D-031, D-041): capture -> perception -> live state -> act policy -> candidates -> ranker -> executor.

Dry run by default: the executor only logs the clicks it would make. Packages A-F (detector, mirroring source, live
state, ranker, planner, executor) are imported lazily inside the builders so this module stays light and testable
with fakes. The act policy is reactive by default (clashbot.policy: defend / leak / punish, else hold and save
elixir); `--policy learned --act-model PATH` adds the d1 act-head (MAC-X); `--policy trigger` restores the
fixed-interval `should_decide`. The loop plays only cards it can see in the hand (`require_hand`) and asks the
policy for a spell's aim point (`target_tile`) when it has one. With a `TacticalEngine` (default; `--no-tactics` turns
it off, MAC-Z3) the candidates start with the engine's concrete moves, every candidate carries a note, and the ranker
sees the engine's lines after the context; `--ranker tactics` plays the engine's own pick without any model.
Usage: python3 -m clashbot.live --calib CALIB --deck c1,...,c8 [--source video --video PATH]
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from clashbot.actions import load_placement_prior, option_text, placement_tile
from clashbot.candidates import CandidateGenerator
from clashbot.config import HOLD_KEY
from clashbot.game_rules import load_cards
from clashbot.kg import KnowledgeGraph
from clashbot.loop import should_decide
from clashbot.policy import ActDecision, ReactivePolicy
from clashbot.retrieval import StateIndex
from clashbot.tactics import TacticalEngine

STAGES = ("capture", "perception", "live_state", "trigger", "candidates", "rank", "execute", "total")
FRAME_STAGES = ("capture", "perception", "live_state")
TARGET_P95_MS = 300.0
FALLBACK_MIN_ELIXIR = 9.5     # a "fallback" candidate (no knn/kg support) is only played from a (nearly) full bar
TOS_NOTE = ("Live runs only in Training Camp or friendly battles "
            "(Supercell terms of service prohibit bots in normal play).")


@dataclass
class LiveDecision:
    t: float
    idx: int
    key: str
    tile: list[float] | None
    candidates: list[dict]        # [{"key", "score", "sources", "tile", "note"}]
    probs: list[float]
    executor: dict                # the executor's returned entry
    timings_ms: dict              # every STAGES name
    mode: str = "trigger"         # policy mode: defend | leak | punish (reactive) or trigger
    reason: str = ""              # the policy's human-readable reason
    tactics_lines: list = field(default_factory=list)      # the engine's lines shown to the ranker ([] without one)


def _accepts_notes(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "notes" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def rank_keys(ranker, context: str, keys: list[str], plan: str | None,
              notes: dict[str, str] | None = None) -> tuple[int, list[float]]:
    """Pick one of `keys`; returns (index, probs). A single key needs no model call (d1 wants >= 2 options).
    `notes` (key -> one line, MAC-Z3) go to rankers whose `choose_keys` accepts them; others are called as before."""
    if not keys:
        raise ValueError("rank_keys needs at least one key")
    if len(keys) == 1:
        return 0, [1.0]
    if hasattr(ranker, "choose_keys"):
        if notes and _accepts_notes(ranker.choose_keys):
            res = ranker.choose_keys(context, keys, plan, notes=notes)
        else:
            res = ranker.choose_keys(context, keys, plan)
    else:
        res = ranker.choose(context, [option_text(k) for k in keys], plan)
    return res.index, list(res.probs)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def top_source(sources: dict) -> str:
    """Which channel a candidate came from: tac (the tactics engine), knn / kg (the larger), else fallback."""
    if "tac" in sources:
        return "tac"
    knn, kg = sources.get("knn") or 0.0, sources.get("kg") or 0.0
    if knn <= 0 and kg <= 0:
        return "fallback"
    return "knn" if knn >= kg else "kg"


def decision_source(d: "LiveDecision") -> str:
    """`top_source` of the chosen candidate, or "hold" when the decision held."""
    if d.key == HOLD_KEY:
        return "hold"
    for c in d.candidates:
        if c["key"] == d.key:
            return top_source(c.get("sources") or {})
    return "fallback"


def _pcts(vals: list[float]) -> dict:
    if not vals:
        return {"p50": None, "p95": None}
    return {"p50": round(float(np.percentile(vals, 50)), 2), "p95": round(float(np.percentile(vals, 95)), 2)}


class LiveLoop:
    def __init__(self, source, perception, builder, generator, ranker, executor, cards: dict, prior: dict,
                 plan_text: str | None = None, out_dir: Path | None = None, filters: dict | None = None,
                 clock=time.perf_counter, min_gap_s: float = 1.0, policy=None, require_hand: bool = True,
                 tactics=None):
        self.source = source
        self.perception = perception
        self.builder = builder
        self.generator = generator
        self.ranker = ranker
        self.executor = executor
        self.cards = cards
        self.prior = prior
        self.plan_text = plan_text
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.filters = dict(filters or {})
        self.clock = clock
        self.min_gap_s = min_gap_s
        # None / "reactive" -> ReactivePolicy(cards); "trigger" -> the old should_decide path (self.policy stays None)
        if policy is None or policy == "reactive":
            policy = ReactivePolicy(cards)
        self.policy = None if policy == "trigger" else policy
        self.require_hand = require_hand             # never tap a card the hand read does not show (FIX_LIST #27)
        self.tactics = tactics                       # a TacticalEngine (MAC-Z3) or None = lookup candidates only
        self.hand_unknown_holds = 0                  # decisions held because no candidate card was in the read hand
        self.policy_holds: Counter = Counter()       # reactive hold decisions by mode (one per evaluated frame)
        self.last_decision_t: float | None = None
        self._handled_opp_t = float("-inf")       # newest opponent play already seen at a decision
        self.skipped_min_gap = 0
        self._rect_changes_seen = 0
        self.frames: list[dict] = []
        self.decisions: list[LiveDecision] = []
        self._fh: dict[str, object] = {}
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            for name in ("frames.jsonl", "decisions.jsonl"):
                (self.out_dir / name).write_text("")

    def _write(self, name: str, obj: dict) -> None:
        if self.out_dir is None:
            return
        fh = self._fh.get(name)
        if fh is None or fh.closed:
            fh = self._fh[name] = open(self.out_dir / name, "a")
        fh.write(json.dumps(obj, default=_json_default) + "\n")
        fh.flush()

    def _close_files(self) -> None:
        for fh in self._fh.values():
            fh.close()

    def _follow_rect(self) -> bool:
        """The capture source follows a moved window (`rect_changes`); tell the executor so its screen map follows too."""
        rc = getattr(self.source, "rect_changes", 0)
        if rc <= self._rect_changes_seen:
            return False
        self._rect_changes_seen = rc
        update_rect = getattr(self.executor, "update_rect", None)
        if callable(update_rect):
            update_rect(self.source.rect)
        return True

    def _fresh_opponent_state(self, state: dict) -> dict:
        """Copy of `state` whose opponent `recent_plays` drops plays a decision already saw (each play triggers once)."""
        opp = state.get("opponent") or {}
        plays = [p for p in opp.get("recent_plays", []) if p.get("t") is None or p["t"] > self._handled_opp_t]
        return {**state, "opponent": {**opp, "recent_plays": plays}}

    def step(self, frame, t: float, idx: int, capture_ms: float) -> LiveDecision | None:
        clock = self.clock
        rect_changed = self._follow_rect()
        a = clock()
        gs = self.perception.process(frame, t, idx)
        events = self.perception.drain_events()
        b = clock()
        self.builder.observe_events(events)
        built = self.builder.build(gs)
        c = clock()
        perception_ms = (b - a) * 1000.0
        live_state_ms = (c - b) * 1000.0
        row = {"t": t, "idx": idx, "readiness": getattr(gs, "readiness", None), "capture": capture_ms,
               "perception": perception_ms, "live_state": live_state_ms, "rect_changed": rect_changed}
        self.frames.append(row)
        self._write("frames.jsonl", row)
        if built is None:
            return None

        state, ctx = built
        d = clock()
        if self.last_decision_t is not None and t - self.last_decision_t < self.min_gap_s:
            self.skipped_min_gap += 1           # `should_decide` stays true for seconds; never tap twice this close
            return None
        if self.policy is None:
            go = should_decide(self._fresh_opponent_state(state), t, self.last_decision_t, self.cards)
            verdict = ActDecision(go, "trigger", None, "should_decide")
        else:
            verdict = self.policy.decide(state, t, context_text=ctx)
        e = clock()
        trigger_ms = (e - d) * 1000.0
        if not verdict.act:
            if self.policy is not None:
                self.policy_holds[verdict.mode] += 1
            return None

        report = self.tactics.analyse(state, t) if self.tactics is not None else None
        rank_ctx = ctx if report is None else ctx + "\n" + "\n".join(report.lines)   # the act policy keeps `ctx`
        if report is None:
            cands = generated = self.generator.generate(state, t, ctx, n=4, **self.filters)
        else:
            cands = generated = self.generator.generate(state, t, ctx, n=4, report=report, **self.filters)
        if self.policy is not None:
            cands = self.policy.filter_candidates(cands, verdict, state)
        policy_hold = self.policy is not None and not cands      # wants to act, but no candidate fits (blind spells...)
        hand_hold = False
        if self.require_hand and cands:
            in_hand = {c for c in state["own"]["hand"] if c}
            cands = [x for x in cands if x.key.split("|")[0] in in_hand]
            hand_hold = not cands                                # every candidate is a card we cannot see in the hand
        f = clock()
        candidates_ms = (f - e) * 1000.0
        chosen = None
        if not cands:
            key, probs, rank_ms, g = HOLD_KEY, [], 0.0, f
        else:
            if len(cands) == 1:
                i, probs = 0, [1.0]
            elif hasattr(self.ranker, "choose_candidates"):         # e.g. the tactics ranker: reads the candidates
                res = self.ranker.choose_candidates(cands)
                i, probs = res.index, list(res.probs)
            else:
                notes = {x.key: x.note for x in cands if getattr(x, "note", None)}
                i, probs = rank_keys(self.ranker, rank_ctx, [x.key for x in cands], self.plan_text,
                                     notes=notes or None)
            chosen = cands[i]
            key = chosen.key
            g = clock()
            rank_ms = (g - f) * 1000.0

        elixir = state["own"]["elixir"]
        if policy_hold:
            # the policy wanted to play but no candidate fits it: record a hold, never call the executor
            key, tile = HOLD_KEY, None
            entry = {"t": t, "key": HOLD_KEY, "tile": None, "status": "hold_policy", "clicks": [], "sent": False}
            cands = generated          # logged unfiltered: the record shows what the policy turned down
        elif hand_hold:
            # the hand is unread (greyed cards, a lost slot): tapping would only be skipped, so hold instead
            key, tile = HOLD_KEY, None
            entry = {"t": t, "key": HOLD_KEY, "tile": None, "status": "hold_hand_unknown", "clicks": [],
                     "sent": False}
            cands = generated
            self.hand_unknown_holds += 1
        elif cands and "fallback" in cands[i].sources and (elixir is None or elixir < FALLBACK_MIN_ELIXIR):
            # a filler candidate (no knn/kg support) is only worth playing from a full bar; otherwise wait
            key, tile = HOLD_KEY, None
            entry = {"t": t, "key": HOLD_KEY, "tile": None, "status": "hold_fallback", "clicks": [], "sent": False}
        else:
            aim = getattr(self.policy, "target_tile", None)      # spells aim at the enemy units, not the prior
            tile = aim(key, state) if callable(aim) else None
            if tile is None and getattr(chosen, "tile", None) is not None:
                tile = list(chosen.tile)                            # a tactical candidate carries its own tile
            if tile is None:
                tile = placement_tile(key, self.prior)
            entry = self.executor.execute(key, tile, t, hand=state["own"]["hand"])
            note_play = getattr(self.builder, "note_own_play", None)
            if entry.get("sent") and entry.get("slot") is not None and callable(note_play):
                note_play(key.split("|")[0], entry["slot"], t)    # the hand memory follows confirmed taps only
        h = clock()
        execute_ms = (h - g) * 1000.0

        timings = {"capture": capture_ms, "perception": perception_ms, "live_state": live_state_ms,
                   "trigger": trigger_ms, "candidates": candidates_ms, "rank": rank_ms, "execute": execute_ms}
        timings["total"] = sum(timings.values())
        self.last_decision_t = t
        opp_ts = [p["t"] for p in state.get("opponent", {}).get("recent_plays", []) if p.get("t") is not None]
        if opp_ts:
            self._handled_opp_t = max(self._handled_opp_t, max(opp_ts))
        decision = LiveDecision(
            t=t, idx=idx, key=key, tile=tile,
            candidates=[{"key": x.key, "score": x.score, "sources": x.sources, "tile": getattr(x, "tile", None),
                         "note": getattr(x, "note", None)} for x in cands],
            probs=list(probs), executor=entry, timings_ms=timings, mode=verdict.mode, reason=verdict.reason,
            tactics_lines=list(report.lines) if report is not None else [])
        self.decisions.append(decision)
        self._write("decisions.jsonl", {**asdict(decision), "elixir": state["own"]["elixir"],
                                        "hand": state["own"]["hand"], "context_text": ctx})
        return decision

    def run(self, max_seconds: float | None = None, max_frames: int | None = None,
            max_decisions: int | None = None) -> dict:
        n_frames = 0
        t_first: float | None = None
        it = iter(self.source.frames())
        try:
            while True:
                if max_frames is not None and n_frames >= max_frames:
                    break
                if max_decisions is not None and len(self.decisions) >= max_decisions:
                    break
                g0 = self.clock()
                try:
                    frame, t, idx = next(it)
                except StopIteration:
                    break
                measured_ms = (self.clock() - g0) * 1000.0
                grab_ms = getattr(self.source, "last_grab_ms", None)
                capture_ms = grab_ms if grab_ms is not None else measured_ms
                if t_first is None:
                    t_first = t
                if max_seconds is not None and t - t_first >= max_seconds:
                    break
                n_frames += 1
                self.step(frame, t, idx, capture_ms)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                close_it = getattr(it, "close", None)
                if close_it is not None:
                    close_it()
                self.source.close()
            finally:
                try:
                    close_perception = getattr(self.perception, "close", None)      # drains the async OCR worker
                    if callable(close_perception):
                        close_perception()
                finally:
                    self._close_files()
        summary = self.summary()
        if self.out_dir is not None:
            (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
        return summary

    def summary(self) -> dict:
        readiness = Counter("none" if r["readiness"] is None else str(r["readiness"]) for r in self.frames)
        timings = [d.timings_ms for d in self.decisions]
        total_p95 = _pcts([tm["total"] for tm in timings])["p95"]
        stats = getattr(self.source, "stats", None)
        timing_report = getattr(self.perception, "timing_report", None)
        return {
            "frames": len(self.frames),
            "decisions": len(self.decisions),
            "readiness_counts": dict(readiness),
            "frame_ms": {s: _pcts([r[s] for r in self.frames if r[s] is not None]) for s in FRAME_STAGES},
            "decision_ms": {s: _pcts([tm[s] for tm in timings]) for s in STAGES},
            "target_p95_ms": TARGET_P95_MS,
            "target_met": None if total_p95 is None else bool(total_p95 < TARGET_P95_MS),
            "decisions_by_status": dict(Counter(d.executor.get("status", "dry_run") for d in self.decisions)),
            "decisions_by_mode": dict(Counter(d.mode for d in self.decisions)),
            "decisions_by_source": dict(Counter(decision_source(d) for d in self.decisions)),
            "policy_holds": dict(self.policy_holds),
            "hand_unknown_holds": self.hand_unknown_holds,
            "rect_updates": sum(1 for r in self.frames if r.get("rect_changed")),
            "skipped_min_gap": self.skipped_min_gap,
            "source_stats": stats() if callable(stats) else None,
            "perception_timing": timing_report() if callable(timing_report) else None,
        }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python3 -m clashbot.live", description=TOS_NOTE)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="log the clicks that would be made (default)")
    mode.add_argument("--live", action="store_true",
                      help="send real taps (needs CLASHBOT_ALLOW_LIVE=1; Training Camp / friendly battles only)")
    ap.set_defaults(dry_run=True)
    ap.add_argument("--calib", required=True, help="calibration JSON (tools/calibrate.py)")
    ap.add_argument("--deck", required=True, help="own deck: 8 comma-separated card slugs")
    ap.add_argument("--hero-cards", default="", help="comma-separated slugs of hero-variant cards in the deck")
    ap.add_argument("--deck-key", default=None)
    ap.add_argument("--source", choices=("mirroring", "video"), default="mirroring")
    ap.add_argument("--video", default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--rect", default=None, help="manual window rect in points: l,t,w,h")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--detector", choices=("katacr", "none"), default="katacr")
    ap.add_argument("--detect-every", type=int, default=3)
    ap.add_argument("--sync-ocr", action="store_true", help="run OCR inline instead of on the background worker")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", choices=("cpu", "mps"), default=None)
    ap.add_argument("--ranker", choices=("score", "d1", "tactics"), default="score",
                    help="tactics = the tactics engine's own pick (needs the engine, i.e. no --no-tactics)")
    ap.add_argument("--no-tactics", action="store_true",
                    help="no tactics engine: lookup candidates only, no notes (the pre-MAC-Z3 behaviour)")
    ap.add_argument("--d1-backend", choices=("mlx", "transformers"), default="mlx")
    ap.add_argument("--d1-model", default=None, help="d1 model path / repo id (e.g. a fine-tuned checkpoint)")
    ap.add_argument("--planner", choices=("rules", "llm", "none"), default="rules")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--min-gap", type=float, default=1.0, help="minimum seconds between two decisions (0 = off)")
    ap.add_argument("--policy", choices=("reactive", "learned", "trigger"), default="reactive",
                    help="when to play: reactive = defend / leak / punish, else hold (D-041); learned = the rules' "
                         "defend / leak plus a d1 act-head (needs --act-model); trigger = every 3 s etc.")
    ap.add_argument("--act-model", default=None, help="d1 act-head checkpoint for --policy learned")
    ap.add_argument("--act-threshold", type=float, default=None,
                    help="p(act) needed to play (default: `threshold` of <act-model>/act_threshold.json, else 0.5)")
    ap.add_argument("--act-backend", choices=("mlx", "transformers"), default="transformers",
                    help="backend of the act-head model")
    ap.add_argument("--out", default=None, help="output dir (default runs/poc/<timestamp>)")
    return ap


def _split(text: str | None) -> list[str]:
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def _build_source(args):
    if args.source == "video":
        from cr_perception.sources import VideoFrameSource
        return VideoFrameSource(args.video, start=args.start, end=args.end, stride=args.stride)
    from cr_perception.sources import MirroringSource, WindowRect
    rect = None
    if args.rect:
        left, top, width, height = (int(round(float(p))) for p in _split(args.rect))
        rect = WindowRect(left, top, width, height, "manual")
    return MirroringSource(rect=rect, target_fps=args.fps)


def _build_perception(args, source):
    from cr_perception import Perception
    detector = None
    if args.detector == "katacr":
        from cr_perception.detect import default_katacr_detector
        detector = default_katacr_detector(imgsz=args.imgsz, device=args.device)
    return Perception(config=args.calib, source=source, detector=detector, detect_every=args.detect_every,
                      async_ocr=not args.sync_ocr)


def _build_ranker(args):
    if args.ranker == "d1":
        from clashbot.ranker import D1Ranker
        return D1Ranker(model_path=args.d1_model, backend=args.d1_backend)
    if args.ranker == "tactics":
        from clashbot.ranker import TacticsRanker
        return TacticsRanker()
    from clashbot.ranker import ScoreRanker
    return ScoreRanker()


def resolve_act_threshold(args) -> float:
    """`--act-threshold`, else `threshold` of <act-model>/act_threshold.json (written by eval_act_head), else 0.5."""
    if args.act_threshold is not None:
        return float(args.act_threshold)
    path = Path(args.act_model) / "act_threshold.json"
    if path.is_file():
        return float(json.loads(path.read_text())["threshold"])
    return 0.5


def _build_policy(args, cards: dict):
    """The policy argument of `LiveLoop`: a name (reactive / trigger) or a `LearnedActPolicy` on the act-head model."""
    if args.policy != "learned":
        return args.policy
    from clashbot.policy import LearnedActPolicy
    from clashbot.ranker import D1Ranker
    scorer = D1Ranker(model_path=args.act_model, backend=args.act_backend)
    return LearnedActPolicy(scorer, cards, args.act_threshold)


def _build_plan_text(args, deck: list[str]) -> str | None:
    if args.planner == "rules":
        from clashbot.planner import RuleBasedPlanner
        return RuleBasedPlanner().pregame(deck, args.deck_key).text
    if args.planner == "llm":
        from clashbot.planner import LLMPlanner
        return LLMPlanner().pregame(deck, args.deck_key).text
    return None


def format_table(summary: dict) -> str:
    lines = ["| stage | p50 ms | p95 ms |", "|---|---|---|"]
    for stage in STAGES:
        p = summary["decision_ms"][stage]
        lines.append(f"| {stage} | {p['p50']} | {p['p95']} |")
    lines.append(f"target_met (total p95 < {summary['target_p95_ms']:.0f} ms): {summary['target_met']}")
    return "\n".join(lines)


def resolve_device(device: str | None) -> str:
    """`--device` unset -> mps when torch reports it, else cpu (torch imported here so `import clashbot.live` stays light)."""
    if device is not None:
        return device
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv: list[str] | None = None) -> int:
    print(TOS_NOTE)
    args = build_parser().parse_args(argv)
    args.dry_run = not args.live
    deck = _split(args.deck)
    if len(deck) != 8:
        print(f"error: --deck needs 8 comma-separated card slugs, got {len(deck)}", file=sys.stderr)
        return 2
    heroes = _split(args.hero_cards)
    if args.source == "video" and not args.video:
        print("error: --source video needs --video PATH", file=sys.stderr)
        return 2
    if args.live and args.source != "mirroring":
        print("error: --live needs --source mirroring", file=sys.stderr)
        return 2
    if args.rect and len(_split(args.rect)) != 4:
        print("error: --rect needs l,t,w,h", file=sys.stderr)
        return 2
    if args.ranker == "tactics" and args.no_tactics:
        print("error: --ranker tactics needs the tactics engine (drop --no-tactics)", file=sys.stderr)
        return 2
    if args.policy == "learned":
        if not args.act_model:
            print("error: --policy learned needs --act-model PATH", file=sys.stderr)
            return 2
        try:
            args.act_threshold = resolve_act_threshold(args)         # recorded in config.json below
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"error: bad act_threshold.json in {args.act_model}: {exc}", file=sys.stderr)
            return 2
    args.device = resolve_device(args.device)         # recorded in config.json below

    source = None
    perception = None
    try:
        source = _build_source(args)
        try:
            if args.source == "mirroring":
                from clashbot.executor import MirroringExecutor
                executor = MirroringExecutor.from_calib(args.calib, source.rect, live=args.live)
            else:
                from clashbot.executor import DryRunExecutor
                executor = DryRunExecutor()
        except PermissionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            source.close()
            return 5
        perception = _build_perception(args, source)
        from clashbot.live_state import LiveConfig, LiveStateBuilder
        builder = LiveStateBuilder(LiveConfig(own_deck=deck, hero_cards=heroes, hand_tracker=True))
        cards = load_cards()
        kg = KnowledgeGraph()
        generator = CandidateGenerator(StateIndex(), kg, cards)
        tactics = None if args.no_tactics else TacticalEngine(cards, kg=kg)
        ranker = _build_ranker(args)
        policy = _build_policy(args, cards)
        plan_text = _build_plan_text(args, deck)
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        close_perception = getattr(perception, "close", None)
        if callable(close_perception):
            close_perception()
        if source is not None:
            source.close()
        return 3

    out_dir = Path(args.out) if args.out else Path("runs") / "poc" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    loop = LiveLoop(source, perception, builder, generator, ranker, executor, cards, load_placement_prior(),
                    plan_text=plan_text, out_dir=out_dir, min_gap_s=args.min_gap, policy=policy, tactics=tactics)
    mode = "LIVE" if args.live else "dry run"
    print(f"{mode}: source={args.source} ranker={args.ranker} planner={args.planner} policy={args.policy} "
          f"tactics={'off' if args.no_tactics else 'on'} out={out_dir}")
    summary = loop.run(max_seconds=args.seconds, max_decisions=args.max_decisions)
    print(f"frames={summary['frames']} decisions={summary['decisions']} "
          f"readiness={summary['readiness_counts']} status={summary['decisions_by_status']}")
    print(format_table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
