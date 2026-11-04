"""macOS: locate the iPhone Mirroring window (or an emulator window), verify Screen
Recording permission and report sustained capture rate. Read-only: never sends input.

  python3 tools/benchmark_capture.py [--source mirroring|emulator] [--backend quartz|mss]
                                     [--owner NAME] [--rect l,t,w,h]
                                     [--fps 0] [--sweep 10,20,30,60] [--seconds 5]
                                     [--save-frame PATH] [--list-windows]

--backend (mirroring only): quartz (default) captures the window itself by id at Retina
resolution; mss grabs the screen region (1x, other windows can show through). A manual
--rect has no window id, so it always uses mss.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr_perception.sources import (MirroringSource, ScreenSource, WindowRect,  # noqa: E402
                                   find_emulator_window, list_windows)


def _parse_rect(text: str) -> WindowRect:
    l, t, w, h = (int(v) for v in text.split(","))
    return WindowRect(l, t, w, h, "manual")


def _print_windows() -> int:
    wins = list_windows()
    if not wins:
        print("no windows listed (needs macOS with pyobjc Quartz)", file=sys.stderr)
    for w in wins:
        r = w["rect"]
        print(f"{w['owner']} | {w['title']} | {r.left},{r.top},{r.width},{r.height} | layer {w['layer']} | "
              f"id {w['window_id']}")
    return 0


def _make_source(a, rect: WindowRect | None, fps: float):
    if a.source == "mirroring":
        return MirroringSource(rect, target_fps=fps, owner_hint=a.owner, backend=a.backend)
    return ScreenSource(rect, target_fps=fps)


def _run(source, seconds: float, save_frame: str | None) -> dict:
    """Pull frames for `seconds`; return the source's stats (or basic counters)."""
    import cv2
    n = 0
    saved = save_frame is None
    t0 = time.perf_counter()
    try:
        for frame, _t, _i in source.frames():
            if not saved:
                cv2.imwrite(save_frame, frame)
                saved = True
            n += 1
            if time.perf_counter() - t0 >= seconds:
                break
        el = time.perf_counter() - t0
        stats = source.stats() if hasattr(source, "stats") else {
            "frames": n, "elapsed_s": round(el, 3), "fps": round(n / el, 2) if el else 0.0,
            "target_fps": source.fps}
        return {"rect": dict(source.rect.__dict__), **stats}
    finally:
        source.close()


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("mirroring", "emulator"), default="mirroring")
    ap.add_argument("--backend", choices=("quartz", "mss"), default="quartz",
                    help="mirroring capture backend: quartz = window by id, Retina (default); mss = screen region")
    ap.add_argument("--owner", help="window owner name hint (default: iPhone Mirroring / known emulators)")
    ap.add_argument("--rect", help="left,top,width,height in screen points (manual fallback)")
    ap.add_argument("--fps", type=float, default=0.0, help="target fps; 0 = unthrottled")
    ap.add_argument("--sweep", help="comma-separated target fps values, e.g. 10,20,30,60")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--save-frame", help="write the first captured frame to this PNG path")
    ap.add_argument("--list-windows", action="store_true", help="print on-screen windows and exit")
    a = ap.parse_args()

    if a.list_windows:
        return _print_windows()

    rect = _parse_rect(a.rect) if a.rect else None
    if a.source == "emulator" and rect is None:
        rect = find_emulator_window(a.owner)
        print("window:", rect)
        if rect is None:
            print("no emulator window found via Quartz; pass --rect")
            return 2

    targets = [float(v) for v in a.sweep.split(",") if v.strip()] if a.sweep else [a.fps]
    results: list[dict] = []
    save_path = a.save_frame
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for fps in targets:
        try:
            source = _make_source(a, rect, fps)
            res = _run(source, a.seconds, save_path)
        except (RuntimeError, PermissionError) as e:
            print("CAPTURE CHECK FAILED:", e)
            return 3
        save_path = None  # only the first run writes the frame
        results.append(res)
        print(json.dumps(res))

    if a.sweep:
        print()
        print("| target fps | achieved fps | dropped | grab p50 ms | grab p95 ms |")
        print("|---|---|---|---|---|")
        for r in results:
            print(f"| {_fmt(r.get('target_fps'))} | {_fmt(r.get('fps'))} | {_fmt(r.get('dropped'))} | "
                  f"{_fmt(r.get('grab_ms_p50'))} | {_fmt(r.get('grab_ms_p95'))} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
