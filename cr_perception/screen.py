"""Frame-level readiness: find the game content inside a frame (black bars,
letterboxing), and decide whether the frame is a readable match, a menu, or
unreadable. Anything not "match" is skipped downstream.

The match test uses cheap colour statistics in HUD regions that are fixed
fractions of the content rect:
  * the elixir bar strip at the bottom is predominantly purple/magenta while
    in a match;
  * the four hand-card slots hold saturated, high-variance card art;
  * the arena (middle) is mostly green/brown/blue field, not menu UI.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ContentRect:
    x: int
    y: int
    w: int
    h: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]

    def to_json(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @property
    def aspect(self) -> float:
        return self.h / max(1, self.w)


def detect_content_rect(frame: np.ndarray, black_thresh: int = 24, min_fill: float = 0.15) -> ContentRect:
    """Strip black side/top bars: columns and rows whose mean intensity is below
    `black_thresh` for almost their whole length are treated as bars."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    col_nonblack = (gray > black_thresh).mean(axis=0)
    row_nonblack = (gray > black_thresh).mean(axis=1)
    xs = np.where(col_nonblack > min_fill)[0]
    ys = np.where(row_nonblack > min_fill)[0]
    if len(xs) < w * 0.1 or len(ys) < h * 0.1:
        return ContentRect(0, 0, w, h)
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    return ContentRect(x0, y0, x1 - x0, y1 - y0)


def _roi(img: np.ndarray, fx: float, fy: float, fw: float, fh: float) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(fy * h):int((fy + fh) * h), int(fx * w):int((fx + fw) * w)]


def purple_fraction(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= 125) & (h <= 165) & (s > 90) & (v > 80)
    return float(m.mean()) if m.size else 0.0


def field_fraction(img: np.ndarray) -> float:
    """Green/blue-ish arena floor and river."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= 30) & (h <= 90) & (s > 60) & (v > 60)
    return float(green.mean()) if green.size else 0.0


@dataclass
class Readiness:
    state: str          # "match" | "menu" | "unreadable"
    conf: float
    scores: dict


def looks_like_game(frame: np.ndarray) -> bool:
    return assess(frame).state in ("match", "menu")


def assess(frame: np.ndarray, rect: ContentRect | None = None, rois: dict | None = None) -> Readiness:
    """Classify a frame. `rois` may carry calibrated fractional HUD boxes
    (elixir_bar, hand, arena); defaults are the portrait-layout fractions."""
    rect = rect or detect_content_rect(frame)
    img = rect.crop(frame)
    if img.size == 0 or rect.w < 64 or rect.h < 64:
        return Readiness("unreadable", 1.0, {"reason": "no content"})
    r = rois or {}
    eb = r.get("elixir_bar", (0.28, 0.955, 0.70, 0.025))
    hand = r.get("hand", (0.18, 0.80, 0.80, 0.15))
    arena = r.get("arena", (0.05, 0.12, 0.90, 0.60))
    s_purple = purple_fraction(_roi(img, *eb))
    s_field = field_fraction(_roi(img, *arena))
    hand_img = _roi(img, *hand)
    s_hand = float(cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY).std()) / 128.0 if hand_img.size else 0.0
    dark = float((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 20).mean())
    scores = {"purple_bar": round(s_purple, 3), "field": round(s_field, 3), "hand_var": round(s_hand, 3),
              "dark": round(dark, 3), "aspect": round(rect.aspect, 3)}
    if dark > 0.85 or s_hand < 0.05:
        return Readiness("unreadable", 0.9, scores)
    match_score = min(1.0, s_purple / 0.35) * 0.5 + min(1.0, s_field / 0.25) * 0.35 + min(1.0, s_hand / 0.3) * 0.15
    if s_purple > 0.2 and s_field > 0.08:
        return Readiness("match", round(match_score, 2), scores)
    return Readiness("menu", round(1.0 - match_score, 2), scores)
