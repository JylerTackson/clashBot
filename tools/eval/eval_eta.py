"""How good is the tracker's ETA to our towers? (MAC-AL T2): old rule vs new rule on a recorded video.

  python3 tools/eval/eval_eta.py --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60 --seconds 240 \
      [--device mps] [--calib CALIB] [--fps 10] [--stride N] [--detect-every 3] --out runs/smoke/AL/eta.json

Runs `cr_perception.Perception` on the video the way `clashbot.live --source video` does (same VideoFrameSource, the
same detector, `--fps` 10 and `--detect-every` 3 defaults; the stride that turns the video's frame rate into `--fps`
unless `--stride` is given). After every processed frame it records, for each enemy unit track, the position, the ETA
of the old rule (straight line to the tower edge at the filter's speed) and of the new rule (to attack range, at the
card's speed; `KalmanTrack.eta_to_nearest_enemy_tower`), both from the same filter state.
Ground truth: the first frame at which the track is within 1.5 + its attack range of one of our towers (the moment the
new rule says 0.0). Error: for every earlier frame of that track with an ETA, `predicted - (t_arrival - t)`; a track
that never arrives (killed, pulled away) has no truth and is left out. Reported per kind of unit (melee, ranged,
flyer; `unknown` = a class without a unit_stats row) and for the old and the new rule: median / p25 / p75 of the signed
error (positive = ETA too late), the share of |error| <= 1 s, the median |error|, n. A second table keeps only the
frames within 10 s of the arrival (the window in which the engine acts).
Writes the JSON (summary + per-class coverage + every track's samples) and prints the tables.
Acceptance (spec): new median |error| <= 1.0 s for melee and flyer, <= 1.5 s for ranged, each better than old.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cr_perception.detect import to_card_slug  # noqa: E402
from cr_perception.tracking import TOWER_EDGE, TOWER_TILES, UNIT_STATS_PATH  # noqa: E402

KINDS = ("melee", "ranged", "flyer", "unknown")
RULES = ("old", "new")
RANGED_MIN_TILES = 2.5          # attack range from which a ground unit is "ranged"
OK_ERR_S = 1.0                  # |error| within this counts as right
NEAR_HORIZON_S = 10.0           # the second table: frames this close to the arrival


# ----------------------------------------------------------------------------- pure pieces

def legacy_eta(pos, vel, side: str) -> tuple[str, float] | None:
    """The tracker's ETA before MAC-AL, verbatim: straight line to the tower edge at the filter's speed."""
    sp = math.hypot(vel[0], vel[1])
    if sp < 0.1:
        return None
    names = ("enemy_left", "enemy_right", "enemy_king") if side == "ally" else ("own_left", "own_right", "own_king")
    best = None
    for nm in names:
        tc, tr = TOWER_TILES[nm]
        d = math.hypot(tc - pos[0], tr - pos[1]) - 1.5
        if best is None or d < best[1]:
            best = (nm, max(0.0, d))
    return (best[0], round(best[1] / sp, 1)) if best else None


def in_reach(pos, range_tiles: float, side: str = "enemy") -> bool:
    """The unit is within TOWER_EDGE + its attack range of one of the towers it attacks (ours, for an enemy)."""
    names = ("own_left", "own_right", "own_king") if side == "enemy" else ("enemy_left", "enemy_right", "enemy_king")
    return any(math.hypot(TOWER_TILES[n][0] - pos[0], TOWER_TILES[n][1] - pos[1]) <= TOWER_EDGE + range_tiles
               for n in names)


def load_cards_stats(path: Path | None = None) -> dict:
    try:
        return json.loads(Path(path or UNIT_STATS_PATH).read_text(encoding="utf-8")).get("cards") or {}
    except (OSError, ValueError):
        return {}


def unit_kind(cls: str, cards: dict) -> str:
    """melee | ranged | flyer from the first unit's row of unit_stats (`unknown` without a row)."""
    units = (cards.get(to_card_slug(cls)) or {}).get("units") or []
    if not units or not isinstance(units[0], dict):
        return "unknown"
    u = units[0]
    if u.get("flying"):
        return "flyer"
    return "ranged" if float(u.get("range_tiles") or 0.0) >= RANGED_MIN_TILES else "melee"


def make_sample(t: float, pos, vel, side: str, range_tiles: float, new_eta) -> list:
    """[t, col, row, old ETA | None, new ETA | None, at a tower] for one track at one frame."""
    old = legacy_eta(pos, vel, side)
    return [round(float(t), 3), round(float(pos[0]), 2), round(float(pos[1]), 2),
            None if old is None else old[1], None if new_eta is None else new_eta[1],
            in_reach(pos, range_tiles, side)]


def arrival_t(track: dict) -> float | None:
    """Time of the first sample at a tower, None when the track never got there."""
    for s in track["samples"]:
        if s[5]:
            return s[0]
    return None


def track_errors(track: dict, max_horizon_s: float | None = None) -> dict[str, list[float]]:
    """{rule: [predicted - (t_arrival - t) for the samples before the arrival that have an ETA]}; {} without an arrival."""
    t_arr = arrival_t(track)
    out: dict[str, list[float]] = {r: [] for r in RULES}
    if t_arr is None:
        return {}
    for t, _c, _r, old, new, _at in track["samples"]:
        if t >= t_arr or (max_horizon_s is not None and t_arr - t > max_horizon_s):
            continue
        for rule, v in zip(RULES, (old, new)):
            if v is not None:
                out[rule].append(float(v) - (t_arr - t))
    return out


def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0, "median": None, "p25": None, "p75": None, "within_1s": None, "median_abs": None}
    return {"n": len(errs), "median": round(statistics.median(errs), 2), "p25": round(quantile(errs, 0.25), 2),
            "p75": round(quantile(errs, 0.75), 2),
            "within_1s": round(sum(abs(e) <= OK_ERR_S for e in errs) / len(errs), 3),
            "median_abs": round(statistics.median(abs(e) for e in errs), 2)}


def evaluate(tracks: list[dict], max_horizon_s: float | None = None) -> dict:
    """{kind: {rule: summary}} over every track with an arrival; `all` pools the kinds."""
    pools = {k: {r: [] for r in RULES} for k in KINDS + ("all",)}
    for tr in tracks:
        for rule, errs in track_errors(tr, max_horizon_s).items():
            pools[tr["kind"]][rule] += errs
            pools["all"][rule] += errs
    return {k: {r: summarize(pools[k][r]) for r in RULES} for k in pools}


def coverage(tracks: list[dict]) -> dict:
    """Per detector class: its kind, priors and how many tracks / arrivals it has."""
    out: dict = {}
    for tr in tracks:
        c = out.setdefault(tr["cls"], {"kind": tr["kind"], "speed_prior": tr["speed_prior"], "range": tr["range"],
                                       "tracks": 0, "arrived": 0})
        c["tracks"] += 1
        c["arrived"] += arrival_t(tr) is not None
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["tracks"]))


def tracks_by_kind(tracks: list[dict]) -> dict:
    """How many tracks of each kind there are and how many of them reached a tower (the evidence base of the table)."""
    out = {k: {"tracks": 0, "arrived": 0} for k in KINDS}
    for tr in tracks:
        out[tr["kind"]]["tracks"] += 1
        out[tr["kind"]]["arrived"] += arrival_t(tr) is not None
    return out


def acceptance(table: dict) -> dict:
    """Spec acceptance per kind: new median |error| <= 1.0 s (melee, flyer) / 1.5 s (ranged) and better than old."""
    out = {}
    for kind, limit in (("melee", 1.0), ("flyer", 1.0), ("ranged", 1.5)):
        old, new = table[kind]["old"]["median_abs"], table[kind]["new"]["median_abs"]
        out[kind] = {"limit": limit, "old": old, "new": new,
                     "pass": bool(new is not None and old is not None and new <= limit and new < old)}
    return out


def format_table(table: dict, title: str) -> list[str]:
    def f(v, nd=2):
        return "-" if v is None else f"{v:.{nd}f}"
    lines = [f"### {title}", "", "| kind | rule | median | p25 | p75 | share abs err <= 1 s | median abs err | n |",
             "|---|---|---|---|---|---|---|---|"]
    for kind in KINDS + ("all",):
        for rule in RULES:
            s = table[kind][rule]
            share = "-" if s["within_1s"] is None else f"{100 * s['within_1s']:.0f}%"
            lines.append(f"| {kind} | {rule} | {f(s['median'])} | {f(s['p25'])} | {f(s['p75'])} | {share} | "
                         f"{f(s['median_abs'])} | {s['n']} |")
    return lines + [""]


# ----------------------------------------------------------------------------- recording

class EtaRecorder:
    """Collects, per enemy unit track, a sample per processed frame (see `make_sample`)."""

    def __init__(self, cards: dict):
        self.cards = cards
        self.tracks: dict[tuple, dict] = {}

    def observe(self, t: float, match_id: int, tracks) -> None:
        for tr in tracks:
            if tr.side != "enemy" or tr.static:
                continue
            key = (match_id, tr.id)
            rec = self.tracks.get(key)
            if rec is None:
                rec = self.tracks[key] = {"key": f"m{match_id}#{tr.id}", "cls": tr.cls, "kind": unit_kind(tr.cls, self.cards),
                                          "speed_prior": tr.speed_prior, "range": tr.range_prior, "samples": []}
            if rec["cls"] != tr.cls:                       # an unknown_unit read upgraded to a class
                rec.update(cls=tr.cls, kind=unit_kind(tr.cls, self.cards), speed_prior=tr.speed_prior, range=tr.range_prior)
            rec["samples"].append(make_sample(t, tr.position(), tr.velocity(), tr.side, tr.range_prior,
                                              tr.eta_to_nearest_enemy_tower()))

    def result(self) -> list[dict]:
        return list(self.tracks.values())


def run_video(args) -> tuple[list[dict], dict]:
    from cr_perception import Perception
    from cr_perception.detect import default_katacr_detector
    from cr_perception.sources import VideoFrameSource

    video = Path(args.video)
    calib = Path(args.calib) if args.calib else Path("runs/videos") / video.stem / "match_0" / "calib.json"
    probe = VideoFrameSource(video, start=args.start)
    fps_video = probe.fps
    probe.close()
    stride = args.stride or max(1, round(fps_video / args.fps))
    source = VideoFrameSource(video, start=args.start, end=args.start + args.seconds, stride=stride)
    detector = default_katacr_detector(imgsz=args.imgsz, device=args.device)
    perception = Perception(config=calib, source=source, detector=detector, detect_every=args.detect_every,
                            async_ocr=True, opp_evidence=True)
    rec = EtaRecorder(load_cards_stats())
    n_frames, n_match, t0 = 0, 0, time.time()
    try:
        for frame, t, idx in source.frames():
            perception.process(frame, t, idx)
            perception.drain_events()
            n_frames += 1
            if perception.match_started:
                n_match += 1
                rec.observe(t, perception.match_id, perception.tracker.confirmed())
    finally:
        source.close()
        perception.close()
    meta = {"video": str(video), "calib": str(calib), "start": args.start, "seconds": args.seconds,
            "video_fps": round(fps_video, 3), "stride": stride, "detect_every": args.detect_every, "imgsz": args.imgsz,
            "device": args.device, "frames": n_frames, "match_frames": n_match, "wall_s": round(time.time() - t0, 1)}
    return rec.result(), meta


def build_report(tracks: list[dict], meta: dict) -> dict:
    table = evaluate(tracks)
    return {"meta": meta, "tracks": len(tracks), "arrived": sum(arrival_t(t) is not None for t in tracks),
            "tracks_by_kind": tracks_by_kind(tracks), "by_kind": table, "near_arrival_10s": evaluate(tracks, NEAR_HORIZON_S), "acceptance": acceptance(table),
            "coverage": coverage(tracks),
            "track_samples": {t["key"]: {"cls": t["cls"], "kind": t["kind"], "range": t["range"],
                                         "speed_prior": t["speed_prior"], "arrival_t": arrival_t(t),
                                         "samples": t["samples"]} for t in tracks}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--calib", default=None, help="calibration JSON (default runs/videos/<video stem>/match_0/calib.json)")
    ap.add_argument("--fps", type=float, default=10.0, help="live's --fps default: the stride is video fps / this")
    ap.add_argument("--stride", type=int, default=None, help="frame stride (default: round(video fps / --fps))")
    ap.add_argument("--detect-every", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", choices=("cpu", "mps"), default=None)
    ap.add_argument("--out", required=True, help="JSON output path")
    args = ap.parse_args(argv)
    tracks, meta = run_video(args)
    report = build_report(tracks, meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    lines = ["# ETA error (predicted - actual seconds to be in reach of our tower, + = too late)", "",
             f"{meta['video']} start {meta['start']} s, {meta['seconds']} s, stride {meta['stride']}, "
             f"{meta['frames']} frames in {meta['wall_s']} s; {report['tracks']} enemy unit tracks, "
             f"{report['arrived']} reached a tower; tracks / arrived by kind: "
             + ", ".join(f"{k} {v['tracks']}/{v['arrived']}" for k, v in report["tracks_by_kind"].items()), ""]
    lines += format_table(report["by_kind"], "All frames before the arrival")
    lines += format_table(report["near_arrival_10s"], f"Frames within {NEAR_HORIZON_S:g} s of the arrival")
    lines += ["### Acceptance (new median abs error <= limit and better than old)", ""]
    lines += [f"- {k}: old {v['old']} -> new {v['new']} (limit {v['limit']}): {'PASS' if v['pass'] else 'FAIL'}"
              for k, v in report["acceptance"].items()]
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
