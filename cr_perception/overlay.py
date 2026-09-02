"""Debug overlay: tile grid, detections with assigned tiles, legal placement
mask and HUD reads drawn over the live frame. Renders to an image; the
caller shows it (cv2.imshow) or writes a video."""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry as g


def draw_grid(img: np.ndarray, h: g.Homography, color=(0, 255, 255), thick: int = 1) -> None:
    for p0, p1 in g.grid_lines(h):
        cv2.line(img, p0, p1, color, thick, cv2.LINE_AA)
    # river rows in blue, own/enemy halves' borders in white
    for r in (15, 17):
        p0 = h.tile_to_pixel(0, r, centre=False)
        p1 = h.tile_to_pixel(g.COLS, r, centre=False)
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 200, 0), 2, cv2.LINE_AA)


def draw_mask(img: np.ndarray, h: g.Homography, mask: np.ndarray, alpha: float = 0.18) -> None:
    ov = img.copy()
    for c in range(g.COLS):
        for r in range(g.ROWS):
            if mask[c, r]:
                pts = np.array([h.tile_to_pixel(c + dc, r + dr, centre=False) for dc, dr in ((0, 0), (1, 0), (1, 1), (0, 1))],
                               np.int32)
                cv2.fillPoly(ov, [pts], (0, 255, 0))
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def draw_units(img: np.ndarray, h: g.Homography, units, show_bbox: bool = True) -> None:
    for u in units:
        x1, y1, x2, y2 = [int(v) for v in u.bbox]
        col = (255, 120, 0) if u.side == "ally" else (0, 0, 255) if u.side == "enemy" else (200, 200, 200)
        if u.cls == "unknown_unit":
            col = (0, 255, 255)
        if show_bbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        bx, by = g.bbox_bottom_centre(u.bbox)
        cv2.circle(img, (int(bx), int(by)), 3, col, -1)
        if u.tile is not None:
            c, r = u.tile
            pts = np.array([h.tile_to_pixel(c + dc, r + dr, centre=False) for dc, dr in ((0, 0), (1, 0), (1, 1), (0, 1))], np.int32)
            cv2.polylines(img, [pts], True, col, 2)
            label = f"{u.cls} {u.conf:.2f} ({c},{r})"
        else:
            label = f"{u.cls} {u.conf:.2f}"
        cv2.putText(img, label, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


def draw_rois(img: np.ndarray, rois: dict, color=(0, 200, 255)) -> None:
    H, W = img.shape[:2]
    for name, box in rois.items():
        if len(box) != 4:
            continue                       # e.g. elixir_bar_full = [x0, x1]
        x, y, w, h = box
        p0 = (int(x * W), int(y * H))
        p1 = (int((x + w) * W), int((y + h) * H))
        cv2.rectangle(img, p0, p1, color, 1)
        cv2.putText(img, name, (p0[0], max(8, p0[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)


def draw_hud_text(img: np.ndarray, lines: list[str], origin=(6, 16), color=(255, 255, 255)) -> None:
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def render(content: np.ndarray, h: g.Homography | None, state, mask: np.ndarray | None, rois: dict | None = None,
           show_rois: bool = False) -> np.ndarray:
    img = content.copy()
    if h is not None:
        if mask is not None:
            draw_mask(img, h, mask)
        draw_grid(img, h)
        draw_units(img, h, state.units)
    if show_rois and rois:
        draw_rois(img, rois)
    own, opp = state.own, state.opponent
    lines = [f"t={state.t:.2f} {state.readiness} clock={state.match_clock} phase={state.phase}",
             f"elixir={own.get('elixir')} ({state.field_confidence.get('elixir')}) hand={own.get('hand')} next={own.get('next_card')}",
             f"opp elixir~{opp.get('elixir_est')} conf={opp.get('elixir_conf')} deck={len(opp.get('deck_known', []))}/8 "
             f"{'complete' if opp.get('deck_complete') else ''}",
             f"units={len(state.units)} conf={state.field_confidence.get('units')} stale={ {k: v for k, v in state.stale.items() if v} }"]
    draw_hud_text(img, lines)
    return img


class OverlayVideoWriter:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, img: np.ndarray) -> None:
        self.w.write(img)

    def close(self) -> None:
        self.w.release()
