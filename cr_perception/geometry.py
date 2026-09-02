"""Tile coordinate system for the Clash Royale arena.

Conventions (asserted in tests/test_geometry.py):
  * 18 columns x 32 rows. Column 0 is the left edge, row 0 is the bottom row
    of the OWN half (behind the own king tower), row 31 the top row of the
    ENEMY half. A tile index addresses the tile's centre; tile (c, r) covers
    continuous coordinates [c, c+1) x [r, r+1).
  * Own half rows 0-14, river rows 15-16, enemy half rows 17-31. Row 0 is
    restricted to the centre columns behind the king tower; rows 1-14 are fully
    deployable. The enemy mirror of row y is 31 - y.
  * Pixel <-> tile mapping is a homography (perspective), never a linear
    scale: the arena is drawn with vertical compression that varies with depth.
  * A unit's tile is derived from the BOTTOM-CENTRE of its bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

COLS = 18
ROWS = 32
OWN_ROWS = range(0, 15)
RIVER_ROWS = (15, 16)
ENEMY_ROWS = range(17, 32)
ROW0_COLS = range(COLS // 3, 2 * COLS // 3)  # 6..11 inclusive: behind the king tower


def mirror_row(row: int) -> int:
    return ROWS - 1 - row


@dataclass
class Homography:
    """Maps pixel coordinates (in the calibrated frame) to continuous tile
    coordinates (col, row) with row increasing UPWARD on screen."""
    H: np.ndarray  # 3x3, pixel -> tile
    Hinv: np.ndarray

    @classmethod
    def from_corners(cls, corners_px: Sequence[Sequence[float]]) -> "Homography":
        """corners_px: [bottom-left, bottom-right, top-right, top-left] of the
        18x32 playable grid in pixels. Bottom = own side (bottom of screen)."""
        src = np.array(corners_px, dtype=np.float32)
        dst = np.array([[0, 0], [COLS, 0], [COLS, ROWS], [0, ROWS]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return cls(H, np.linalg.inv(H))

    @classmethod
    def from_correspondences(cls, px: Sequence[Sequence[float]], tiles: Sequence[Sequence[float]]) -> "Homography":
        """Least-squares homography from >= 4 (pixel, tile) pairs, e.g. the
        six tower bases whose grid positions are known."""
        src = np.array(px, dtype=np.float32)
        dst = np.array(tiles, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            raise ValueError("degenerate correspondences")
        return cls(H.astype(np.float64), np.linalg.inv(H))

    def pixel_to_tile_f(self, px: float, py: float) -> tuple[float, float]:
        v = self.H @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def pixel_to_tile(self, px: float, py: float) -> tuple[int, int]:
        c, r = self.pixel_to_tile_f(px, py)
        return int(np.floor(c)), int(np.floor(r))

    def tile_to_pixel(self, col: float, row: float, centre: bool = True) -> tuple[float, float]:
        """Pixel of a tile. With centre=True the tile's centre point, otherwise
        the continuous grid coordinate (col, row) itself (e.g. a corner)."""
        if centre:
            col, row = col + 0.5, row + 0.5
        v = self.Hinv @ np.array([col, row, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def in_arena(self, px: float, py: float, margin: float = 0.0) -> bool:
        c, r = self.pixel_to_tile_f(px, py)
        return -margin <= c < COLS + margin and -margin <= r < ROWS + margin

    def to_json(self) -> list[list[float]]:
        return self.H.tolist()

    @classmethod
    def from_json(cls, H: list[list[float]]) -> "Homography":
        H = np.array(H, dtype=np.float64)
        return cls(H, np.linalg.inv(H))


def bbox_bottom_centre(bbox: Sequence[float]) -> tuple[float, float]:
    """(x1, y1, x2, y2) -> the point where the unit stands on the ground."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def bbox_to_tile(h: Homography, bbox: Sequence[float]) -> tuple[int, int]:
    return h.pixel_to_tile(*bbox_bottom_centre(bbox))


def clamp_tile(col: int, row: int) -> tuple[int, int]:
    return int(min(max(col, 0), COLS - 1)), int(min(max(row, 0), ROWS - 1))


@dataclass
class TowerState:
    """Alive flags for the six towers."""
    own_king: bool = True
    own_left: bool = True
    own_right: bool = True
    enemy_king: bool = True
    enemy_left: bool = True
    enemy_right: bool = True


# Tiles a destroyed enemy princess tower opens up (mirrors BuildABot's
# LEFT/RIGHT_PRINCESS_TILES): the river-bank tile in that lane plus the four
# enemy rows in front of the tower, on that half of the field.
def _princess_zone(left: bool) -> set[tuple[int, int]]:
    cols = range(0, COLS // 2) if left else range(COLS // 2, COLS)
    bridge_col = 3 if left else 14
    zone = {(bridge_col, 15), (bridge_col, 16)}
    zone |= {(c, r) for c in cols for r in range(17, 21)}
    return zone


def legal_placement_mask(towers: TowerState) -> np.ndarray:
    """Boolean [COLS, ROWS] array: tiles the player may deploy on right now.
    Indexed mask[col, row]."""
    mask = np.zeros((COLS, ROWS), dtype=bool)
    for c in ROW0_COLS:
        mask[c, 0] = True
    mask[:, 1:15] = True
    if not towers.enemy_left:
        for c, r in _princess_zone(left=True):
            mask[c, r] = True
    if not towers.enemy_right:
        for c, r in _princess_zone(left=False):
            mask[c, r] = True
    return mask


def mask_hash(mask: np.ndarray) -> str:
    import hashlib
    return hashlib.sha1(np.packbits(mask.astype(np.uint8)).tobytes()).hexdigest()[:12]


def grid_lines(h: Homography) -> Iterable[tuple[tuple[int, int], tuple[int, int]]]:
    """Pixel segments for every grid line (for the overlay)."""
    for c in range(COLS + 1):
        p0 = h.tile_to_pixel(c, 0, centre=False)
        p1 = h.tile_to_pixel(c, ROWS, centre=False)
        yield (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))
    for r in range(ROWS + 1):
        p0 = h.tile_to_pixel(0, r, centre=False)
        p1 = h.tile_to_pixel(COLS, r, centre=False)
        yield (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))
