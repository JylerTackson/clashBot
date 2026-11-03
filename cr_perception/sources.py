"""Frame sources. Every source yields (frame_bgr: np.ndarray, t_seconds: float,
frame_index: int). Perception is READ-ONLY: nothing here can send input.

  VideoFrameSource   - a recorded match (mp4); the primary source right now.
  ImageDirSource     - a folder of frames (for labelled evaluation).
  ScreenSource       - live capture of an emulator window on macOS via mss,
                       locating the window through Quartz; includes the
                       Screen-Recording-permission sanity check.
  MirroringSource    - live capture of the macOS "iPhone Mirroring" window at a steady rate (D-001).
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


MIRRORING_OWNERS = ("iPhone Mirroring",)


def list_windows() -> list[dict]:
    """macOS only: every on-screen window as {owner, title, window_id, layer, rect}.
    Returns [] off-macOS or when Quartz is unavailable."""
    if sys.platform != "darwin":
        return []
    try:
        import Quartz  # type: ignore
    except ImportError:
        return []
    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    out: list[dict] = []
    for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []:
        owner = str(w.get("kCGWindowOwnerName", "") or "")
        title = str(w.get("kCGWindowName", "") or "")
        b = w.get("kCGWindowBounds", {}) or {}
        rect = WindowRect(int(b.get("X", 0)), int(b.get("Y", 0)), int(b.get("Width", 0)), int(b.get("Height", 0)),
                          owner, title)
        out.append({"owner": owner, "title": title, "window_id": int(w.get("kCGWindowNumber", 0) or 0),
                    "layer": int(w.get("kCGWindowLayer", 0) or 0), "rect": rect})
    return out


def find_mirroring_window(owner_hint: str | None = None, windows: list[dict] | None = None) -> WindowRect | None:
    """Largest normal-layer window (>= 100x100) whose owner matches the hint
    (default: the iPhone Mirroring app). None when there is no such window."""
    windows = list_windows() if windows is None else windows
    hints = (owner_hint,) if owner_hint else MIRRORING_OWNERS
    best = None
    for w in windows:
        owner = str(w.get("owner", ""))
        if not any(h.lower() in owner.lower() for h in hints):
            continue
        if int(w.get("layer", 0)) != 0:
            continue
        r = w["rect"]
        if r.width < 100 or r.height < 100:
            continue
        if best is None or r.width * r.height > best.width * best.height:
            best = r
    return best


def _mon_of(r: WindowRect) -> dict:
    return {"left": r.left, "top": r.top, "width": r.width, "height": r.height}


class MirroringSource(FrameSource):
    """Live capture of the macOS iPhone Mirroring window at a steady rate.
    The window rect is in global screen points; with an auto-found rect it is
    re-checked every `refind_every_s` so a moved window is followed. `grab_fn`
    (mon -> BGR frame) and `clock`/`sleep` are injectable for tests."""

    def __init__(self, rect: WindowRect | None = None, target_fps: float = 10.0, owner_hint: str | None = None,
                 grab_fn=None, clock=time.perf_counter, sleep=time.sleep, check_permission: bool = True,
                 refind_every_s: float = 2.0):
        self._auto = rect is None
        if rect is None:
            rect = find_mirroring_window(owner_hint)
        if rect is None:
            raise RuntimeError("iPhone Mirroring window not found: open iPhone Mirroring, keep the window "
                               "visible (not minimised), or pass a manual rect")
        self.rect = rect
        self.mon = _mon_of(rect)
        self._sct = None
        if grab_fn is None:
            import mss  # local import so the module loads without mss on CI
            self._sct = mss.mss()
            grab_fn = lambda mon: np.asarray(self._sct.grab(mon))[:, :, :3].copy()  # BGRA -> BGR  # noqa: E731
        self._grab_fn = grab_fn
        self._clock = clock
        self._sleep = sleep
        self.fps = target_fps
        self._refind_every_s = refind_every_s
        self.frames_out = 0
        self.dropped = 0
        self.rect_changes = 0
        self.last_grab_ms = 0.0
        self._grab_ms: list[float] = []
        self._t0: float | None = None
        self._last_shape: tuple | None = None
        self.looks_like_game: bool | None = None
        if check_permission:
            try:
                self._check_permission()
            except Exception:
                self.close()
                raise

    def grab(self) -> np.ndarray:
        frame = self._grab_fn(self.mon)
        self._last_shape = frame.shape
        return frame

    def _check_permission(self) -> None:
        """Same failure mode as ScreenSource: without Screen Recording permission
        macOS returns the desktop or a blank frame. Whether the frame looks like
        a match is informational only (the phone may be on its home screen)."""
        frame = self.grab()
        if frame.size == 0:
            raise RuntimeError("empty capture")
        if float(frame.std()) < 2.0:
            raise PermissionError(
                "Captured region is uniformly blank. On macOS this means Screen Recording permission is "
                "not granted to the process running this script. Grant it in System Settings > Privacy & "
                "Security > Screen Recording for your terminal/IDE, then restart it.")
        from .screen import looks_like_game
        self.looks_like_game = bool(looks_like_game(frame))

    def frames(self):
        clock, sleep = self._clock, self._sleep
        period = 1.0 / self.fps if self.fps and self.fps > 0 else 0.0
        t0 = clock()
        self._t0 = t0
        next_due = t0
        last_refind = t0
        i = 0
        while True:
            now = clock()
            if period and now < next_due:
                sleep(next_due - now)
                now = clock()
            if period:
                late = int((now - next_due) // period)  # whole periods we are behind
                if late > 0:
                    self.dropped += late
                    next_due += late * period
            if self._auto and now - last_refind >= self._refind_every_s:
                r = find_mirroring_window()
                last_refind = now
                if r is not None and (r.left, r.top, r.width, r.height) != (
                        self.rect.left, self.rect.top, self.rect.width, self.rect.height):
                    self.rect = r
                    self.mon = _mon_of(r)
                    self.rect_changes += 1
            g0 = clock()
            frame = self.grab()
            g1 = clock()
            self._grab_ms.append((g1 - g0) * 1000.0)
            self.last_grab_ms = self._grab_ms[-1]
            yield frame, g0 - t0, i
            i += 1
            self.frames_out += 1
            if period:
                next_due += period

    def stats(self) -> dict:
        elapsed = round(self._clock() - self._t0, 3) if self._t0 is not None else 0.0
        shape = self._last_shape
        p50 = p95 = None
        if self._grab_ms:
            p50 = round(float(np.percentile(self._grab_ms, 50)), 2)
            p95 = round(float(np.percentile(self._grab_ms, 95)), 2)
        return {
            "frames": self.frames_out,
            "dropped": self.dropped,
            "elapsed_s": elapsed,
            "fps": round(self.frames_out / elapsed, 2) if elapsed else 0.0,
            "target_fps": self.fps,
            "grab_ms_p50": p50,
            "grab_ms_p95": p95,
            "frame_shape": list(shape[:2]) if shape is not None else None,
            "scale": round(shape[1] / self.rect.width, 3) if shape is not None and self.rect.width else None,
            "rect": dict(self.rect.__dict__),
            "rect_changes": self.rect_changes,
            "looks_like_game": self.looks_like_game,
        }

    def close(self) -> None:
        sct = getattr(self, "_sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass


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
