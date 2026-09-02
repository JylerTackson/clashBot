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


# Standard portrait layout (1080x2316 phone recording): the elixir bar spans
# x 0.29-0.98 of the panel width and sits at y 0.95-0.978 of the panel
# height; panel aspect (h/w) is ~2.14. Used to recover the game panel from
# the bar alone (streaming layouts put the game beside a facecam).
BAR_X0, BAR_X1, BAR_Y1, PANEL_ASPECT = 0.29, 0.98, 0.978, 2.144


def detect_game_panel(frame: np.ndarray, min_bar_frac: float = 0.06) -> ContentRect | None:
    """Find the game panel from the purple elixir bar: the longest horizontal
    purple run in the lower part of the frame. Returns None when no bar is
    visible (not in a match), in which case fall back to detect_content_rect."""
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 125) & (h <= 170) & (s > 80) & (v > 90)).astype(np.uint8)
    m[: int(0.5 * H)] = 0
    # close small gaps (the bar has segment separators), then find components
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    best = None
    min_bar_px = max(24, int(min_bar_frac * W))
    for x, y, w, hh, area in stats[1:]:
        if w < min_bar_px or hh > 0.05 * H or w < 5 * hh:
            continue
        if best is None or w > best[2]:
            best = (x, y, w, hh)
    if best is None:
        return None
    x, y, w, hh = best
    # the visible fill may be shorter than the full bar (elixir < 10): the
    # LEFT edge is reliable, the width only when the bar is full. Use the
    # panel height from the bar's vertical position instead, and the left
    # edge for x. Width comes from the aspect ratio.
    y1 = y + hh
    panel_h = y1 / BAR_Y1
    panel_w = panel_h / PANEL_ASPECT
    x0 = x - BAR_X0 * panel_w
    # if the bar looks full, trust its width for the panel width instead
    full_w = w / (BAR_X1 - BAR_X0)
    if abs(full_w - panel_w) / panel_w < 0.08:
        panel_w = full_w
        x0 = x - BAR_X0 * panel_w
    y0 = y1 - panel_h
    x0i, y0i = int(round(max(0, x0))), int(round(max(0, y0)))
    x1i, y1i = int(round(min(W, x0 + panel_w))), int(round(min(H, y1 / BAR_Y1)))
    if x1i - x0i < 40 or y1i - y0i < 80:
        return None
    return ContentRect(x0i, y0i, x1i - x0i, y1i - y0i)


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
    return assess(frame).state in ("match", "match_weak", "menu")


def assess(frame: np.ndarray, rect: ContentRect | None = None, rois: dict | None = None) -> Readiness:
    """Classify a frame without assuming an arena colour (arenas differ):
      * elixir bar: a purple fill in the thin band at ~95-98% height (its
        length is the elixir level, so a nearly empty bar is weak evidence);
      * hand row: four saturated card thumbnails at ~80-93% height;
      * arena: textured (edges) and not dominated by a flat UI colour.
    Temporal hysteresis is applied by the caller (Perception)."""
    rect = rect or detect_content_rect(frame)
    img = rect.crop(frame)
    if img.size == 0 or rect.w < 64 or rect.h < 64:
        return Readiness("unreadable", 1.0, {"reason": "no content"})
    r = rois or {}
    H, W = img.shape[:2]
    eb = r.get("elixir_bar", (0.29, 0.955, 0.69, 0.028))
    hand = r.get("hand", (0.18, 0.80, 0.80, 0.14))
    arena = r.get("arena", (0.05, 0.12, 0.90, 0.60))
    bar_img = _roi(img, *eb)
    s_purple = purple_fraction(bar_img)
    # a live bar is a thin band: the strip just above it must not be purple
    above = _roi(img, eb[0], max(0.0, eb[1] - 2.5 * eb[3]), eb[2], 2.0 * eb[3])
    s_above = purple_fraction(above)
    hand_img = _roi(img, *hand)
    hand_hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV) if hand_img.size else None
    s_hand_sat = float(hand_hsv[..., 1].mean() / 255.0) if hand_hsv is not None else 0.0
    s_hand_var = float(cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY).std()) / 128.0 if hand_img.size else 0.0
    arena_img = _roi(img, *arena)
    gray_a = cv2.cvtColor(arena_img, cv2.COLOR_BGR2GRAY) if arena_img.size else np.zeros((2, 2), np.uint8)
    s_edges = float(cv2.Canny(gray_a, 80, 160).mean() / 255.0)
    dark = float((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 20).mean())
    scores = {"purple_bar": round(s_purple, 3), "purple_above": round(s_above, 3), "hand_sat": round(s_hand_sat, 3),
              "hand_var": round(s_hand_var, 3), "arena_edges": round(s_edges, 3), "dark": round(dark, 3),
              "field": round(field_fraction(arena_img), 3) if arena_img.size else 0.0, "aspect": round(rect.aspect, 3)}
    if dark > 0.85 or s_hand_var < 0.05:
        return Readiness("unreadable", 0.9, scores)
    bar_ok = s_purple >= 0.10 and s_above < max(0.25, 0.7 * s_purple)
    hand_ok = s_hand_sat >= 0.18 and s_hand_var >= 0.15
    arena_ok = s_edges >= 0.04
    conf = 0.5 * min(1.0, s_purple / 0.3) + 0.3 * min(1.0, s_hand_sat / 0.35) + 0.2 * min(1.0, s_edges / 0.15)
    if bar_ok and hand_ok and arena_ok:
        return Readiness("match", round(conf, 2), scores)
    if hand_ok and arena_ok and s_purple < 0.10:
        # bar empty (0 elixir) or hidden: weak match evidence; the caller's
        # hysteresis keeps a running match alive through this
        return Readiness("match_weak", round(0.4 * conf + 0.2, 2), scores)
    return Readiness("menu", round(1.0 - conf, 2), scores)


class MatchGate:
    """Temporal hysteresis over per-frame Readiness: enter "match" only after
    `enter_frames` consecutive strong reads; a weak read (empty elixir bar)
    only extends a running match; leave after `exit_seconds` with no
    evidence. Weak reads never start a match (menus and other games can look
    "weak")."""

    def __init__(self, enter_frames: int = 3, exit_seconds: float = 2.0):
        self.enter_frames, self.exit_seconds = enter_frames, exit_seconds
        self.in_match = False
        self.streak = 0
        self.last_evidence_t = -1e9
        self.entered_t: float | None = None

    def update(self, raw_state: str, t: float) -> tuple[str, bool]:
        """Returns (gated readiness, entered_now)."""
        entered = False
        if raw_state == "match":
            self.streak += 1
            self.last_evidence_t = t
        elif raw_state == "match_weak" and self.in_match:
            self.last_evidence_t = t
        else:
            self.streak = 0
        if not self.in_match and self.streak >= self.enter_frames:
            self.in_match, entered, self.entered_t = True, True, t
        elif self.in_match and (t - self.last_evidence_t) > self.exit_seconds:
            self.in_match = False
        if self.in_match:
            return "match", entered
        return ("menu" if raw_state in ("match", "match_weak") else raw_state), entered


def match_periods_from_sequence(states: list[str], times: list[float], enter_frames: int = 2, exit_seconds: float = 25.0,
                                min_len: float = 20.0) -> list[tuple[float, float]]:
    """Offline helper (e.g. storyboard frames every ~10 s) -> [(start, end)]."""
    gate = MatchGate(enter_frames, exit_seconds)
    periods, start = [], None
    for st, t in zip(states, times):
        g, entered = gate.update(st, t)
        if g == "match" and start is None:
            start = t
        if g != "match" and start is not None:
            periods.append((start, t))
            start = None
    if start is not None:
        periods.append((start, times[-1]))
    return [p for p in periods if p[1] - p[0] >= min_len]
