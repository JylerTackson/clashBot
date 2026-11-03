"""Action executors: dry-run, and the macOS iPhone-Mirroring tapper (D-003, D-031, D-035).

Geometry: calib.json (perception, fractions of the content rect) + the captured window
rect in global screen POINTS -> a 4-point tile -> screen homography; card slots come from
the hand ROIs. Live taps need two independent switches (the executor's `live=True` AND
env CLASHBOT_ALLOW_LIVE=1, checked at construction and again before every tap) and are
only ever posted by `quartz_click`, the single function in the repo that sends input.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

LIVE_ENV = "CLASHBOT_ALLOW_LIVE"
GRID_COLS, GRID_ROWS = 18, 32
ARENA_TILE_CORNERS = [[0.0, 0.0], [18.0, 0.0], [18.0, 32.0], [0.0, 32.0]]   # bl, br, tr, tl (row 0 = own side)


@dataclass
class Calibration:
    src_tiles: list[list[float]]    # 4 arena tile coordinates [col, row]
    dst_pixels: list[list[float]]   # the same 4 points in screen pixels of the mirrored window


def homography(cal: Calibration) -> np.ndarray:
    if len(cal.src_tiles) != 4 or len(cal.dst_pixels) != 4:
        raise ValueError("calibration needs exactly 4 point pairs")
    A = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((x, y), (u, v)) in enumerate(zip(cal.src_tiles, cal.dst_pixels)):
        A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        b[2 * i] = u
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i + 1] = v
    h = np.linalg.solve(A, b)
    return np.append(h, 1.0).reshape(3, 3)


def tile_to_screen(tile: list[float], H: np.ndarray) -> tuple[float, float]:
    p = H @ np.array([float(tile[0]), float(tile[1]), 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


# ---------------------------------------------------------------- screen mapping

@dataclass
class ScreenMap:
    rect: list[float]                     # [left, top, width, height] of the captured window, global screen POINTS
    content_rect_frac: list[float]        # [x, y, w, h] as fractions of the raw frame (calib.json)
    arena_corners_frac: list[list[float]] # 4 x [x, y] content fractions: bl, br, tr, tl (calib.json)
    hand_rois: list[list[float]]          # rois hand_0..hand_3, [x, y, w, h] content fractions (calib.json)


def _rect_list(rect) -> list[float]:
    """A WindowRect-like object (left/top/width/height) or a 4-sequence -> [l, t, w, h]."""
    if hasattr(rect, "left"):
        return [float(rect.left), float(rect.top), float(rect.width), float(rect.height)]
    vals = [float(v) for v in rect]
    if len(vals) != 4:
        raise ValueError("rect needs 4 values: left, top, width, height")
    return vals


def screen_map_from_calib(calib: dict | str | Path, rect) -> ScreenMap:
    if not isinstance(calib, dict):
        calib = json.loads(Path(calib).read_text())
    corners = calib.get("arena_corners_frac") or []
    if len(corners) != 4:
        raise ValueError("calib.json has no arena_corners_frac; run tools/calibrate.py")
    from cr_perception.hud import DEFAULT_ROIS   # lazy: keep this module import-light
    rois = calib.get("rois") or {}
    hand = [list(rois.get(f"hand_{i}") or DEFAULT_ROIS[f"hand_{i}"]) for i in range(4)]
    return ScreenMap(
        rect=_rect_list(rect),
        content_rect_frac=list(calib.get("content_rect_frac") or [0.0, 0.0, 1.0, 1.0]),
        arena_corners_frac=[list(c) for c in corners],
        hand_rois=hand,
    )


def content_to_screen(fx: float, fy: float, smap: ScreenMap) -> tuple[float, float]:
    """Content fraction -> global screen point (through the content rect inside the window)."""
    cx, cy, cw, ch = smap.content_rect_frac
    rx, ry = cx + fx * cw, cy + fy * ch
    L, T, W, H = smap.rect
    return L + rx * W, T + ry * H


def screen_to_image(x: float, y: float, smap: ScreenMap, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Screen point -> pixel of the captured image (Retina frames are 2x the points)."""
    L, T, W, H = smap.rect
    return (x - L) / W * frame_w, (y - T) / H * frame_h


def calibration_from_screen_map(smap: ScreenMap) -> Calibration:
    # The arena corners in calib.json are the images of exactly the tile corners
    # (tools/calibrate.py), so a 4-point homography reproduces the perception mapping.
    return Calibration(
        src_tiles=[list(c) for c in ARENA_TILE_CORNERS],
        dst_pixels=[list(content_to_screen(x, y, smap)) for x, y in smap.arena_corners_frac],
    )


def card_slot_points(smap: ScreenMap) -> list[list[float]]:
    """Screen point of the centre of each hand slot, slots 0..3."""
    pts = []
    for x, y, w, h in smap.hand_rois:
        sx, sy = content_to_screen(x + w / 2, y + h / 2, smap)
        pts.append([sx, sy])
    return pts


def tile_target(tile: list[float]) -> list[float]:
    """Tap the centre of the tile cell, clamped inside the arena."""
    return [min(max(tile[0] + 0.5, 0.5), GRID_COLS - 0.5), min(max(tile[1] + 0.5, 0.5), GRID_ROWS - 0.5)]


# ---------------------------------------------------------------- live guard + Quartz

def check_live_allowed() -> None:
    if sys.platform != "darwin":
        raise PermissionError("live taps need macOS + iPhone Mirroring")
    if os.environ.get(LIVE_ENV) != "1":
        raise PermissionError(
            f"live taps are disabled; the user enables them with --live and {LIVE_ENV}=1 "
            "(Training Camp / friendly battles only)")


def quartz_click(x: float, y: float) -> None:
    """Move to (x, y) and left-click there. The ONLY function that posts input."""
    import Quartz

    pos = (float(x), float(y))
    for kind in (Quartz.kCGEventMouseMoved, Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
        ev = Quartz.CGEventCreateMouseEvent(None, kind, pos, Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.02)


# ---------------------------------------------------------------- executors

class DryRunExecutor:
    def __init__(self):
        self.log: list[dict] = []

    def execute(self, key: str, tile: list[float] | None, t: float, hand: list | None = None) -> dict:
        entry = {"t": t, "key": key, "tile": tile}
        self.log.append(entry)
        return entry


class MirroringExecutor:
    """Plans a two-click play (card slot, then target tile) and, only when live, sends it."""

    def __init__(self, calibration: Calibration, card_slot_pixels: list[list[float]], live: bool = False,
                 poster=None, bounds: list[float] | None = None, click_gap_s: float = 0.12, sleep=time.sleep):
        self.calibration = calibration
        self.H = homography(calibration)
        self.card_slot_pixels = card_slot_pixels
        self.live = live
        if live:
            check_live_allowed()   # raises at construction unless the env switch is on
        self.poster = poster or quartz_click
        self.bounds = bounds
        self.click_gap_s = click_gap_s
        self.sleep = sleep
        self.screen_map: ScreenMap | None = None
        self.log: list[dict] = []

    @classmethod
    def from_calib(cls, calib: dict | str | Path, rect, live: bool = False, **kw) -> "MirroringExecutor":
        smap = screen_map_from_calib(calib, rect)
        kw = {"bounds": list(smap.rect), **kw}
        ex = cls(calibration_from_screen_map(smap), card_slot_points(smap), live=live, **kw)
        ex.screen_map = smap
        return ex

    def update_rect(self, rect) -> None:
        """Follow a moved/resized iPhone Mirroring window: rebuild the screen map, homography,
        card slot points and bounds from the new rect. `live` and `poster` are left unchanged."""
        if self.screen_map is None:
            raise ValueError("update_rect needs an executor built with from_calib")
        smap = replace(self.screen_map, rect=_rect_list(rect))
        self.calibration = calibration_from_screen_map(smap)
        self.H = homography(self.calibration)
        self.card_slot_pixels = card_slot_points(smap)
        self.bounds = list(smap.rect)
        self.screen_map = smap
        self.log.append({"event": "rect_update", "rect": list(smap.rect)})

    def plan_clicks(self, key: str, tile: list[float] | None, hand: list | None) -> dict:
        if key == "hold" or tile is None:
            return {"status": "hold", "slot": None, "clicks": []}
        card = key.split("|")[0]
        if hand is None or card not in hand:
            return {"status": "skipped_card_not_in_hand", "slot": None, "clicks": []}
        slot = list(hand).index(card)
        target = tile_to_screen(tile_target(tile), self.H)
        clicks = [list(self.card_slot_pixels[slot]), [target[0], target[1]]]
        if self.bounds:
            L, T, W, Hh = self.bounds
            if any(not (L <= x <= L + W and T <= y <= T + Hh) for x, y in clicks):
                return {"status": "skipped_out_of_window", "slot": slot, "clicks": clicks}
        return {"status": "planned", "slot": slot, "clicks": clicks}

    def execute(self, key: str, tile: list[float] | None, t: float, hand: list | None = None) -> dict:
        plan = self.plan_clicks(key, tile, hand)
        entry = {"t": t, "key": key, "tile": tile, **plan, "sent": False}
        if self.live and plan["status"] == "planned":
            check_live_allowed()   # the env switch may have been removed since construction
            for x, y in plan["clicks"]:
                self.poster(x, y)
                self.sleep(self.click_gap_s)
            entry["sent"] = True
        self.log.append(entry)
        return entry
