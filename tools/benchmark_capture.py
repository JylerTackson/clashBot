"""macOS: locate the emulator window, verify Screen Recording permission and
report sustained mss capture fps. Read-only.

  python3 tools/benchmark_capture.py [--owner BlueStacks] [--rect left,top,w,h] [--seconds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr_perception.sources import ScreenSource, WindowRect, benchmark_capture, find_emulator_window  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner")
    ap.add_argument("--rect", help="left,top,width,height (manual fallback)")
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    rect = None
    if a.rect:
        l, t, w, h = (int(v) for v in a.rect.split(","))
        rect = WindowRect(l, t, w, h, "manual")
    else:
        rect = find_emulator_window(a.owner)
        print("window:", rect)
        if rect is None:
            print("no emulator window found via Quartz; pass --rect")
            return 2
    try:
        src = ScreenSource(rect, target_fps=0)
    except (PermissionError, RuntimeError) as e:
        print("CAPTURE CHECK FAILED:", e)
        return 3
    print(json.dumps({"rect": rect.__dict__, **benchmark_capture(src, a.seconds)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
