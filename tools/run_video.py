"""Run the perception pipeline over a recorded match.

  python3 tools/run_video.py --video match.mp4 --calib calib.json --out runs/<name> \
      [--detector katacr|buildabot|none] [--start 0 --end 300 --stride 1] [--overlay]

Writes: <out>/states.jsonl (states + events), <out>/overlay.mp4 (optional),
<out>/summary.json (timing, fps, events, elixir drift, deck tracker result).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr_perception import Perception, VideoFrameSource  # noqa: E402
from cr_perception import geometry as g  # noqa: E402
from cr_perception.overlay import OverlayVideoWriter, render  # noqa: E402
from cr_perception.recorder import JsonlRecorder  # noqa: E402

SCRATCH = os.environ.get("CR_SCRATCH", "/tmp/claude-0/-home-user-clashBot/b1ee76b3-087c-551c-820f-ad044281a081/scratchpad")


def build_detector(name: str, a):
    if name == "katacr":
        from cr_perception.detect import KataCRDetector
        return KataCRDetector(a.katacr_weights or [f"{SCRATCH}/katacr_d1.pt", f"{SCRATCH}/katacr_d2.pt"], a.katacr_root, imgsz=a.imgsz)
    if name == "buildabot":
        from cr_perception.detect import BuildABotDetector
        return BuildABotDetector(a.buildabot_root)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector", default="katacr")
    ap.add_argument("--katacr-root", default="/home/user/wty-yy/katacr")
    ap.add_argument("--katacr-weights", nargs="*")
    ap.add_argument("--buildabot-root", default="/home/user/pbatch/clashroyalebuildabot")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--detect-every", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--frames-every", type=int, default=0, help="dump every Nth frame as jpg for labelling")
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = VideoFrameSource(a.video, a.start, a.end, a.stride)
    det = build_detector(a.detector, a)
    p = Perception(a.calib, src, detector=det, detect_every=a.detect_every, use_ocr=not a.no_ocr)
    rec = JsonlRecorder(out / "states.jsonl", out / "frames" if a.frames_every else None, a.frames_every)
    writer = None
    n = 0
    events = []
    ready_counts = {}
    t_wall = time.perf_counter()
    for frame, t, idx in src.frames():
        state = p.process(frame, t, idx)
        ready_counts[state.readiness] = ready_counts.get(state.readiness, 0) + 1
        rec.write(state)
        rec.frame(idx, frame)
        for ev in p.drain_events():
            rec.write(ev)
            events.append(ev.to_json())
        if a.overlay:
            cx, cy, cw, ch = p.calib.content_rect(frame.shape[1], frame.shape[0])
            img = render(frame[cy:cy + ch, cx:cx + cw], p.H if state.readiness == "match" else None, state,
                         getattr(p, "_mask", None) if state.readiness == "match" else None, p.calib.rois, show_rois=True)
            if writer is None:
                writer = OverlayVideoWriter(str(out / "overlay.mp4"), src.fps / a.stride, (img.shape[1], img.shape[0]))
            writer.write(img)
        n += 1
        if n % 200 == 0:
            print(f"{n} frames, t={t:.1f}s, {state.readiness}, elixir={state.own.get('elixir')}, "
                  f"hand={state.own.get('hand')}, units={len(state.units)}, events={len(events)}", flush=True)
    wall = time.perf_counter() - t_wall
    rec.close()
    if writer:
        writer.close()
    summary = {"video": a.video, "frames": n, "wall_seconds": round(wall, 1), "fps_processed": round(n / wall, 2),
               "readiness_counts": ready_counts, "timing": p.timing_report(), "events": len(events),
               "own_events": sum(e["player"] == "own" for e in events),
               "opponent_events": sum(e["player"] == "opponent" for e in events),
               "unidentified_events": sum(e["card"] is None for e in events),
               "own_elixir_drift": p.own_sim.drift_stats(), "own_sim_resyncs": p.own_sim.resyncs,
               "opponent_deck": p.deck.summary()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
