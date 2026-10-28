"""Action executors: dry-run (testable here) and a macOS mirroring stub (D-003)."""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np


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


class DryRunExecutor:
    def __init__(self):
        self.log: list[dict] = []

    def execute(self, key: str, tile: list[float] | None, t: float) -> dict:
        entry = {"t": t, "key": key, "tile": tile}
        self.log.append(entry)
        return entry


class MirroringExecutor:
    def __init__(self, calibration: Calibration, card_slot_pixels: list[list[float]]):
        self.calibration = calibration
        self.card_slot_pixels = card_slot_pixels

    def execute(self, key: str, tile: list[float] | None, t: float):
        if sys.platform != "darwin":
            raise NotImplementedError("tapping needs macOS + iPhone Mirroring (Quartz events)")
        raise NotImplementedError("TODO on the Mac: click card slot, then click tile_to_screen(tile)")
