"""Live loop CLI (D-007, D-031): capture -> perception -> live state -> trigger -> candidates -> ranker -> executor.

Dry run by default: the executor only logs the clicks it would make. Packages A-F (detector, mirroring source, live
state, ranker, planner, executor) are imported lazily inside the builders so this module stays light and testable
with fakes. Usage: python3 -m clashbot.live --calib CALIB --deck c1,...,c8 [--source video --video PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from clashbot.actions import load_placement_prior, option_text, placement_tile
from clashbot.candidates import CandidateGenerator
from clashbot.config import HOLD_KEY
from clashbot.game_rules import load_cards
from clashbot.kg import KnowledgeGraph
from clashbot.loop import should_decide
from clashbot.retrieval import StateIndex

STAGES = ("capture", "perception", "live_state", "trigger", "candidates", "rank", "execute", "total")
FRAME_STAGES = ("capture", "perception", "live_state")
TARGET_P95_MS = 300.0
TOS_NOTE = ("Live runs only in Training Camp or friendly battles "
            "(Supercell terms of service prohibit bots in normal play).")


@dataclass
class LiveDecision:
    t: float
    idx: int
    key: str
    tile: list[float] | None
    candidates: list[dict]        # [{"key", "score", "sources"}]
    probs: list[float]
    executor: dict                # the executor's returned entry
    timings_ms: dict              # every STAGES name


def rank_keys(ranker, context: str, keys: list[str], plan: str | None) -> tuple[int, list[float]]:
    """Pick one of `keys`; returns (index, probs). A single key needs no model call (d1 wants >= 2 options)."""
    if not keys:
        raise ValueError("rank_keys needs at least one key")
    if len(keys) == 1:
        return 0, [1.0]
    if hasattr(ranker, "choose_keys"):
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


def _pcts(vals: list[float]) -> dict:
    if not vals:
        return {"p50": None, "p95": None}
    return {"p50": round(float(np.percentile(vals, 50)), 2), "p95": round(float(np.percentile(vals, 95)), 2)}


class LiveLoop:
    def __init__(self, source, perception, builder, generator, ranker, executor, cards: dict, prior: dict,
                 plan_text: str | None = None, out_dir: Path | None = None, filters: dict | None = None,
                 clock=time.perf_counter):
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
        self.last_decision_t: float | None = None
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
        go = should_decide(state, t, self.last_decision_t, self.cards)
        e = clock()
        trigger_ms = (e - d) * 1000.0
        if not go:
            return None

        cands = self.generator.generate(state, t, ctx, n=4, **self.filters)
        f = clock()
        candidates_ms = (f - e) * 1000.0
        if not cands:
            key, probs, rank_ms, g = HOLD_KEY, [], 0.0, f
        else:
            i, probs = rank_keys(self.ranker, ctx, [x.key for x in cands], self.plan_text)
            key = cands[i].key
            g = clock()
            rank_ms = (g - f) * 1000.0

        tile = placement_tile(key, self.prior)
        entry = self.executor.execute(key, tile, t, hand=state["own"]["hand"])
        h = clock()
        execute_ms = (h - g) * 1000.0

        timings = {"capture": capture_ms, "perception": perception_ms, "live_state": live_state_ms,
                   "trigger": trigger_ms, "candidates": candidates_ms, "rank": rank_ms, "execute": execute_ms}
        timings["total"] = sum(timings.values())
        self.last_decision_t = t
        decision = LiveDecision(
            t=t, idx=idx, key=key, tile=tile,
            candidates=[{"key": x.key, "score": x.score, "sources": x.sources} for x in cands],
            probs=list(probs), executor=entry, timings_ms=timings)
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
            "rect_updates": sum(1 for r in self.frames if r.get("rect_changed")),
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
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", choices=("cpu", "mps"), default=None)
    ap.add_argument("--ranker", choices=("score", "d1"), default="score")
    ap.add_argument("--d1-backend", choices=("mlx", "transformers"), default="mlx")
    ap.add_argument("--planner", choices=("rules", "llm", "none"), default="rules")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--max-decisions", type=int, default=None)
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
    return Perception(config=args.calib, source=source, detector=detector, detect_every=args.detect_every)


def _build_ranker(args):
    if args.ranker == "d1":
        from clashbot.ranker import D1Ranker
        return D1Ranker(backend=args.d1_backend)
    from clashbot.ranker import ScoreRanker
    return ScoreRanker()


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

    source = None
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
        builder = LiveStateBuilder(LiveConfig(own_deck=deck, hero_cards=heroes))
        cards = load_cards()
        generator = CandidateGenerator(StateIndex(), KnowledgeGraph(), cards)
        ranker = _build_ranker(args)
        plan_text = _build_plan_text(args, deck)
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if source is not None:
            source.close()
        return 3

    out_dir = Path(args.out) if args.out else Path("runs") / "poc" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    loop = LiveLoop(source, perception, builder, generator, ranker, executor, cards, load_placement_prior(),
                    plan_text=plan_text, out_dir=out_dir)
    mode = "LIVE" if args.live else "dry run"
    print(f"{mode}: source={args.source} ranker={args.ranker} planner={args.planner} out={out_dir}")
    summary = loop.run(max_seconds=args.seconds, max_decisions=args.max_decisions)
    print(f"frames={summary['frames']} decisions={summary['decisions']} "
          f"readiness={summary['readiness_counts']} status={summary['decisions_by_status']}")
    print(format_table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
