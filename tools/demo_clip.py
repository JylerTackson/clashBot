"""One-command demo on a downloaded video: find the first readable match,
calibrate from it, run the pipeline for N seconds with the debug overlay,
and write a browser-playable clip plus a plain summary of what came back.

  python3 tools/demo_clip.py --video data/videos/<id>/<id>.mp4 --out runs/demo --seconds 30 [--detector katacr|buildabot|none]

Outputs in --out:
  calib.json, calib.verify.png   calibration + grid drawn over the frame
  overlay.mp4                    the 30 s overlay clip (H.264 when ffmpeg is available)
  states.jsonl                   every state + event
  summary.json / summary.md      counts, timings, events, sampled states
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr_perception import Perception, VideoFrameSource  # noqa: E402
from cr_perception.overlay import OverlayVideoWriter, render  # noqa: E402
from cr_perception.recorder import JsonlRecorder  # noqa: E402
from cr_perception.screen import MatchGate, assess, detect_content_rect, detect_game_panel  # noqa: E402

SCRATCH = os.environ.get("CR_SCRATCH", "/tmp/claude-0/-home-user-clashBot/b1ee76b3-087c-551c-820f-ad044281a081/scratchpad")


def find_first_match(video: str, step: float = 2.0, needed: int = 3, max_t: float | None = None) -> tuple[float | None, dict]:
    """Scan the video every `step` seconds; return the time where a match has
    been readable for `needed` consecutive samples (plus 10 s margin so the
    HUD is settled), and the per-sample log."""
    cap = cv2.VideoCapture(video)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    log = []
    streak, t = 0, 0.0
    while t < (max_t or dur):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        rect = detect_game_panel(fr) or detect_content_rect(fr)
        a = assess(fr, rect)
        log.append({"t": t, "state": a.state, "conf": a.conf, "rect": rect.to_json()})
        streak = streak + 1 if a.state == "match" else 0
        if streak >= needed:
            cap.release()
            return t - (needed - 1) * step, {"samples": log, "duration": dur}
        t += step
    cap.release()
    return None, {"samples": log, "duration": dur}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--start", type=float, default=None, help="override the auto-found match start")
    ap.add_argument("--detector", default="katacr")
    ap.add_argument("--detect-every", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=896, help="KataCR inference size (640 is ~2x faster, less precise)")
    ap.add_argument("--katacr-root", default="/home/user/wty-yy/katacr")
    ap.add_argument("--buildabot-root", default="/home/user/pbatch/clashroyalebuildabot")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. locate a match
    start = a.start
    scan = {}
    if start is None:
        start, scan = find_first_match(a.video)
        (out / "scan.json").write_text(json.dumps(scan, indent=1))
        if start is None:
            print("no readable match found in the video (see scan.json)")
            return 2
        dur = scan.get("duration", start + a.seconds + 10)
        start = min(start + 10.0, max(0.0, dur - a.seconds - 1.0))
    print(f"match starts around t={start:.1f}s")
    cap = cv2.VideoCapture(a.video)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    calib_t = min(start + 5.0, max(0.0, duration - 0.5))

    # 2. calibrate from a frame inside the match (auto arena from towers when a detector is available)
    calib_path = out / "calib.json"
    cmd = [sys.executable, str(Path(__file__).parent / "calibrate.py"), "--video", a.video, "--time", str(calib_t),
           "--out", str(calib_path)]
    if a.detector == "katacr":
        cmd += ["--detector", "katacr", "--katacr-root", a.katacr_root, "--katacr-weights",
                f"{SCRATCH}/katacr_d1.pt", f"{SCRATCH}/katacr_d2.pt"]
    elif a.detector == "buildabot":
        cmd += ["--detector", "buildabot", "--buildabot-root", a.buildabot_root]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-800:] if r.returncode else "")
    if r.returncode:
        return 3

    # 3. run the pipeline for N seconds with overlay
    src = VideoFrameSource(a.video, start, start + a.seconds)
    det = None
    if a.detector == "katacr":
        from cr_perception.detect import KataCRDetector
        det = KataCRDetector([f"{SCRATCH}/katacr_d1.pt", f"{SCRATCH}/katacr_d2.pt"], a.katacr_root, imgsz=a.imgsz)
    elif a.detector == "buildabot":
        from cr_perception.detect import BuildABotDetector
        det = BuildABotDetector(a.buildabot_root)
    p = Perception(calib_path, src, detector=det, detect_every=a.detect_every)
    rec = JsonlRecorder(out / "states.jsonl")
    writer = None
    samples, events, n = [], [], 0
    for frame, t, idx in src.frames():
        st = p.process(frame, t, idx)
        rec.write(st)
        for ev in p.drain_events():
            rec.write(ev)
            events.append(ev.to_json())
        cx, cy, cw, ch = p.calib.content_rect(frame.shape[1], frame.shape[0])
        img = render(frame[cy:cy + ch, cx:cx + cw], p.H if st.readiness == "match" else None, st,
                     getattr(p, "_mask", None) if st.readiness == "match" else None, p.calib.rois, show_rois=True)
        if writer is None:
            writer = OverlayVideoWriter(str(out / "overlay_raw.mp4"), src.fps, (img.shape[1], img.shape[0]))
        writer.write(img)
        if n % int(src.fps) == 0:
            js = st.to_json()
            samples.append({k: js[k] for k in ("t", "readiness", "match_clock", "phase", "own", "field_confidence", "stale")}
                           | {"opponent": {k: js["opponent"].get(k) for k in ("elixir_est", "elixir_conf", "deck_known", "deck_complete", "predicted_hand")},
                              "units": [(u["class"], u["side"], u["tile"], u["conf"]) for u in js["units"]]})
        n += 1
    rec.close()
    if writer:
        writer.close()
    # 4. browser-playable H.264
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = None
    if ff:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(out / "overlay_raw.mp4"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", str(out / "overlay.mp4")], check=False)
        if (out / "overlay.mp4").exists():
            (out / "overlay_raw.mp4").unlink()
    else:
        (out / "overlay_raw.mp4").rename(out / "overlay.mp4")

    summary = {"video": a.video, "start": start, "seconds": a.seconds, "frames": n, "detector": a.detector,
               "timing": p.timing_report(), "events": events, "own_elixir_drift": p.own_sim.drift_stats(),
               "opponent_deck": p.deck.summary(), "calibration": json.loads(calib_path.read_text()).get("notes"),
               "samples_1hz": samples}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    md = [f"# Demo: {Path(a.video).name} from t={start:.0f}s for {a.seconds:.0f}s", "",
          f"- frames processed: {n}; loop fps: {summary['timing'].get('loop_fps')}; detector: {a.detector} "
          f"(every {a.detect_every} frames, {summary['timing'].get('detect', {}).get('mean_ms')} ms each)",
          f"- events: {len(events)} ({sum(e['player']=='own' for e in events)} own, {sum(e['player']=='opponent' for e in events)} opponent, "
          f"{sum(e['card'] is None for e in events)} unidentified)",
          f"- own elixir drift (sim - HUD): {summary['own_elixir_drift']}",
          f"- opponent deck so far: {summary['opponent_deck']['deck_known']}", "", "## Events", ""]
    for e in events:
        md.append(f"- t={e['timestamp']:.1f} clock={e['match_clock']} {e['player']} {e['card']} tile={e['tile']} "
                  f"elixir {e['elixir_before']}->{e['elixir_after']} [{e['detect_source']}/{e['confidence']}] {e['detail']}")
    md += ["", "## States (1 per second)", ""]
    for s in samples:
        md.append(f"- t={s['t']:.0f} {s['readiness']} clock={s['match_clock']} phase={s['phase']} elixir={s['own'].get('elixir')} "
                  f"hand={s['own'].get('hand')} next={s['own'].get('next_card')} opp~{s['opponent']['elixir_est']} "
                  f"units={[(u[0], u[1], u[2]) for u in s['units']][:8]} conf={ {k: v for k, v in s['field_confidence'].items() if k in ('elixir','hand','clock','units')} }")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
