"""Frame sources. Every source yields (frame_bgr: np.ndarray, t_seconds: float,
frame_index: int). Perception is READ-ONLY: nothing here can send input.

  VideoFrameSource   - a recorded match (mp4); the primary source right now.
  ImageDirSource     - a folder of frames (for labelled evaluation).
  ScreenSource       - live capture of an emulator window on macOS via mss,
                       locating the window through Quartz; includes the
                       Screen-Recording-permission sanity check.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class FrameSource:
    fps: float = 0.0

    def frames(self) -> Iterator[tuple[np.ndarray, float, int]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VideoFrameSource(FrameSource):
    def __init__(self, path: str | Path, start: float = 0.0, end: float | None = None, stride: int = 1):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.start, self.end, self.stride = start, end, max(1, stride)
        if start > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    def frames(self):
        while True:
            idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            ok, frame = self.cap.read()
            if not ok:
                return
            t = idx / self.fps
            if self.end is not None and t > self.end:
                return
            yield frame, t, idx
            if self.stride > 1:
                for _ in range(self.stride - 1):
                    if not self.cap.grab():
                        return

    def frame_at(self, t: float) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class ImageDirSource(FrameSource):
    def __init__(self, folder: str | Path, fps: float = 1.0):
        self.files = sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.png"))
        self.fps = fps

    def frames(self):
        for i, f in enumerate(self.files):
            img = cv2.imread(str(f))
            if img is not None:
                yield img, i / self.fps, i


@dataclass
class WindowRect:
    left: int
    top: int
    width: int
    height: int
    owner: str = ""
    title: str = ""


EMULATOR_OWNERS = ("BlueStacks", "MuMu", "MuMuPlayer", "LDPlayer", "Nox", "Android Emulator", "qemu-system", "Genymotion")


def find_emulator_window(owner_hint: str | None = None) -> WindowRect | None:
    """macOS only: enumerate on-screen windows with Quartz and pick the
    emulator by owner name. Returns None off-macOS or when not found."""
    if sys.platform != "darwin":
        return None
    try:
        import Quartz  # type: ignore
    except ImportError:
        return None
    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    wins = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID)
    hints = (owner_hint,) if owner_hint else EMULATOR_OWNERS
    best = None
    for w in wins:
        owner = str(w.get("kCGWindowOwnerName", ""))
        if not any(h.lower() in owner.lower() for h in hints):
            continue
        b = w.get("kCGWindowBounds", {})
        r = WindowRect(int(b.get("X", 0)), int(b.get("Y", 0)), int(b.get("Width", 0)), int(b.get("Height", 0)),
                       owner, str(w.get("kCGWindowName", "")))
        if r.width * r.height > (best.width * best.height if best else 0):
            best = r
    return best


class ScreenSource(FrameSource):
    """Live capture via mss of a fixed window rect (macOS emulator)."""

    def __init__(self, rect: WindowRect | None = None, owner_hint: str | None = None, target_fps: float = 30.0):
        import mss  # local import so the module loads without mss on CI
        self.rect = rect or find_emulator_window(owner_hint)
        if self.rect is None:
            raise RuntimeError("emulator window not found; pass a manual rect in the config "
                               "(capture.rect = {left, top, width, height})")
        self.sct = mss.mss()
        self.mon = {"left": self.rect.left, "top": self.rect.top, "width": self.rect.width, "height": self.rect.height}
        self.fps = target_fps
        self._check_permission()

    def grab(self) -> np.ndarray:
        raw = self.sct.grab(self.mon)
        return np.asarray(raw)[:, :, :3].copy()  # BGRA -> BGR

    def _check_permission(self) -> None:
        """macOS returns the desktop or a black frame instead of an error when
        Screen Recording permission is missing. Detect that and fail loudly."""
        frame = self.grab()
        if frame.size == 0:
            raise RuntimeError("empty capture")
        std = float(frame.std())
        if std < 2.0:
            raise PermissionError(
                "Captured region is uniformly blank. On macOS this means Screen Recording permission is "
                "not granted to the process running this script. Grant it in System Settings > Privacy & "
                "Security > Screen Recording for your terminal/IDE, then restart it.")
        from .screen import looks_like_game
        if not looks_like_game(frame):
            raise RuntimeError(
                "Captured region does not look like the emulator (no game content detected). Either Screen "
                "Recording permission is missing (macOS silently captures the desktop instead) or the window "
                "rect is wrong. Check System Settings > Privacy & Security > Screen Recording and capture.rect.")

    def frames(self):
        i = 0
        t0 = time.perf_counter()
        period = 1.0 / self.fps if self.fps else 0.0
        while True:
            t = time.perf_counter() - t0
            yield self.grab(), t, i
            i += 1
            if period:
                dt = (time.perf_counter() - t0) - t
                if dt < period:
                    time.sleep(period - dt)


def benchmark_capture(source: FrameSource, seconds: float = 5.0) -> dict:
    """Sustained fps of raw frame acquisition from a source."""
    n = 0
    t0 = time.perf_counter()
    for _ in source.frames():
        n += 1
        if time.perf_counter() - t0 >= seconds:
            break
    el = time.perf_counter() - t0
    return {"frames": n, "seconds": round(el, 2), "fps": round(n / el, 1) if el else None}
