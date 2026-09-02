"""Process every downloaded Clash Royale video: find matches, calibrate,
run perception, build context packs for the agents. Resumable.

  python3 tools/batch_process.py [--videos data/videos] [--out runs/videos] [--only <id> ...]
                                 [--stride 3 --detect-every 10 --label-every 5 --imgsz 640]

Per video: runs/videos/<id>/scan.json, match_<n>/{calib.json, calib.verify.png,
states.jsonl, summary.json, context.json, context.md}; runs/videos/manifest.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cr_perception import Perception, VideoFrameSource  # noqa: E402
from cr_perception.context import build_context, render_context_md, split_matches  # noqa: E402
from cr_perception.recorder import JsonlRecorder  # noqa: E402
from cr_perception.screen import assess, detect_content_rect, detect_game_panel  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-clashBot/b1ee76b3-087c-551c-820f-ad044281a081/scratchpad"
from cr_perception.decktracker import load_kb_decks  # noqa: E402
KB_DECKS = load_kb_decks(ROOT / "knowledge_base")
WEIGHTS = [f"{SCRATCH}/katacr_d1.pt", f"{SCRATCH}/katacr_d2.pt"]


def scan_matches(video: str, step: float = 2.0, gap: float = 8.0, min_len: float = 45.0) -> tuple[list[tuple[float, float]], list[dict]]:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    log, periods, cur = [], [], None
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        rect = detect_game_panel(fr) or detect_content_rect(fr)
        a = assess(fr, rect)
        log.append({"t": t, "state": a.state, "conf": a.conf})
        is_match = a.state == "match" or (a.state == "match_weak" and cur is not None)
        if is_match:
            if cur is None:
                cur = [t, t]
            elif t - cur[1] <= gap:
                cur[1] = t
            else:
                periods.append(tuple(cur))
                cur = [t, t]
        t += step
    if cur:
        periods.append(tuple(cur))
    cap.release()
    return [p for p in periods if p[1] - p[0] >= min_len], log


def calibrate(video: str, t: float, out: Path) -> bool:
    cmd = [sys.executable, str(ROOT / "tools" / "calibrate.py"), "--video", video, "--time", str(t), "--out", str(out),
           "--detector", "katacr", "--katacr-weights", *WEIGHTS]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and out.exists() and "towers(" in r.stdout
    (out.parent / "calibrate.log").write_text(r.stdout[-3000:] + r.stderr[-1500:])
    return ok


def process_video(vid: str, vdir: Path, out_dir: Path, a, card_names: dict, det) -> dict:
    mp4 = next((f for f in vdir.iterdir() if f.suffix in (".mp4", ".mkv", ".webm")), None)
    info = json.loads(next(vdir.glob("*.info.json")).read_text()) if list(vdir.glob("*.info.json")) else {}
    title = info.get("title", vid)
    vtt = next(iter(sorted(vdir.glob("*.en.vtt"))), None) or next(iter(sorted(vdir.glob("*.vtt"))), None)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_p = out_dir / "scan.json"
    if scan_p.exists():
        periods = [tuple(p) for p in json.loads(scan_p.read_text())["periods"]]
    else:
        periods, log = scan_matches(str(mp4))
        scan_p.write_text(json.dumps({"periods": periods, "samples": log}, indent=1))
    result = {"id": vid, "title": title, "matches": [], "periods": periods}
    last_calib = None
    for i, (p0, p1) in enumerate(periods):
        mdir = out_dir / f"match_{i}"
        mdir.mkdir(exist_ok=True)
        if (mdir / "context.json").exists():
            result["matches"].append({"index": i, "status": "cached"})
            continue
        t_start = time.perf_counter()
        calib = mdir / "calib.json"
        ok = calibrate(str(mp4), p0 + 8.0, calib) or calibrate(str(mp4), p0 + 25.0, calib)
        method = "towers"
        if not ok:
            if last_calib and last_calib.exists():
                calib.write_bytes(last_calib.read_bytes())
                method = "reused_previous_match"
            elif not calib.exists():
                result["matches"].append({"index": i, "status": "failed", "reason": "calibration failed"})
                continue
            else:
                method = "fallback_default_corners"
        last_calib = calib
        try:
            src = VideoFrameSource(str(mp4), p0, p1, a.stride)
            p = Perception(calib, src, detector=det, detect_every=a.detect_every, label_every=a.label_every,
                           ocr_every=a.label_every)
            rec = JsonlRecorder(mdir / "states.jsonl")
            n = 0
            for frame, t, idx in src.frames():
                st = p.process(frame, t, idx)
                rec.write(st)
                for ev in p.drain_events():
                    rec.write(ev)
                n += 1
            rec.close()
            summary = {"frames": n, "seconds": round(time.perf_counter() - t_start, 1), "timing": p.timing_report(),
                       "own_elixir_drift": p.own_sim.drift_stats(), "opponent_deck": p.deck.summary()}
            (mdir / "summary.json").write_text(json.dumps(summary, indent=1))
            # one context per actual game: a readiness period can span several
            # back-to-back games, the clock reset splits them
            segs = split_matches(mdir / "states.jsonl") or [(None, p0, p1)]
            written = []
            for k, (mid, s0, s1) in enumerate(segs):
                sub = f"{i}" if len(segs) == 1 else f"{i}.{k}"
                ctx = build_context(mdir / "states.jsonl", vtt, {"video_id": vid, "title": title, "match_index": sub,
                                                                    "period": [s0, s1], "calibration_method": method,
                                                                    "url": f"https://www.youtube.com/watch?v={vid}"}, card_names,
                                    window=(s0, s1), kb_decks=KB_DECKS)
                cdir = mdir if len(segs) == 1 else mdir / f"game_{k}"
                cdir.mkdir(exist_ok=True)
                (cdir / "context.json").write_text(json.dumps(ctx, indent=1))
                (cdir / "context.md").write_text(render_context_md(ctx, card_names))
                written.append({"match": sub, "seconds": round(s1 - s0, 1), "events": len(ctx.get("events", [])), "own_deck": ctx.get("own_deck_observed")})
            if len(segs) > 1:
                (mdir / "context.json").write_text(json.dumps({"video_id": vid, "match_index": i, "split_into": written}, indent=1))
            result["matches"].append({"index": i, "status": "done", "frames": n, "seconds": summary["seconds"], "games": written})
            print(f"  {vid} period {i}: {n} frames in {summary['seconds']}s -> {len(written)} game(s): {written}", flush=True)
        except Exception as e:  # noqa: BLE001
            (mdir / "error.log").write_text(traceback.format_exc())
            result["matches"].append({"index": i, "status": "failed", "reason": str(e)[:300]})
            print(f"  {vid} match {i}: FAILED {e}", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "data" / "videos"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "videos"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--stride", type=int, default=5)        # 6 fps HUD at 30 fps source
    ap.add_argument("--detect-every", type=int, default=6)  # ~1 Hz unit detection
    ap.add_argument("--label-every", type=int, default=3)   # 2 Hz deploy-label OCR
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-detector", action="store_true")
    a = ap.parse_args()
    vroot, out = Path(a.videos), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = json.loads((ROOT / "knowledge_base" / "meta" / "card_index.json").read_text())
    card_names = {c["slug"]: c["name"] for c in idx["cards"]}
    det = None
    if not a.no_detector:
        from cr_perception.detect import KataCRDetector
        det = KataCRDetector(WEIGHTS, "/home/user/wty-yy/katacr", imgsz=a.imgsz)
    manifest_p = out / "manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {"videos": {}}
    dl = json.loads((vroot / "manifest.json").read_text()) if (vroot / "manifest.json").exists() else {"videos": []}
    ids = [v["id"] for v in dl["videos"] if v.get("status") in ("ok", "cached")]
    if not ids:  # manifest not written yet: take whatever has an mp4
        ids = [d.name for d in sorted(vroot.iterdir()) if d.is_dir() and any(f.suffix == ".mp4" for f in d.iterdir())]
    if a.only:
        ids = [i for i in ids if i in set(a.only)]
    print(f"{len(ids)} videos", flush=True)
    for vid in ids:
        vdir = vroot / vid
        if not any(f.suffix in (".mp4", ".mkv", ".webm") for f in vdir.iterdir()):
            continue
        if manifest["videos"].get(vid, {}).get("complete"):
            continue
        print(f"== {vid}", flush=True)
        try:
            r = process_video(vid, vdir, out / vid, a, card_names, det)
            r["complete"] = all(m["status"] in ("done", "cached") for m in r["matches"]) and bool(r["matches"])
        except Exception as e:  # noqa: BLE001
            r = {"id": vid, "complete": False, "error": traceback.format_exc()[-1500:]}
            print(f"  {vid}: FAILED {e}", flush=True)
        manifest["videos"][vid] = r
        manifest_p.write_text(json.dumps(manifest, indent=1))
    done = sum(1 for v in manifest["videos"].values() if v.get("complete"))
    print(f"done: {done}/{len(ids)} videos complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
