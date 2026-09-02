"""calib.json: everything geometric is stored as FRACTIONS of the content
rect (frame minus black bars), so a window move, a Retina scale change or a
different video resolution does not silently break the readers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Homography
from .hud import DEFAULT_ROIS


@dataclass
class Calibration:
    source: dict = field(default_factory=dict)       # {"type": "video"|"screen", "path"/"rect", "frame_size": [w, h]}
    content_rect_frac: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])  # of the raw frame
    arena_corners_frac: list[list[float]] = field(default_factory=list)  # 4 x [x, y] of content: bl, br, tr, tl
    homography_frac: list[list[float]] | None = None  # content-fraction pixel -> tile
    rois: dict = field(default_factory=lambda: dict(DEFAULT_ROIS))
    tower_anchors: list[dict] = field(default_factory=list)  # [{"name", "px_frac": [x,y], "tile": [c,r]}]
    notes: dict = field(default_factory=dict)

    # ---- persistence ----
    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(**{k: d.get(k, v) for k, v in cls().__dict__.items()})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    # ---- geometry helpers ----
    def content_rect(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        x, y, w, h = self.content_rect_frac
        return int(round(x * frame_w)), int(round(y * frame_h)), int(round(w * frame_w)), int(round(h * frame_h))

    def homography_for(self, content_w: int, content_h: int) -> Homography | None:
        """Homography in CONTENT PIXELS for this content size."""
        if self.homography_frac is None:
            if len(self.arena_corners_frac) == 4:
                px = [(x * content_w, y * content_h) for x, y in self.arena_corners_frac]
                return Homography.from_corners(px)
            return None
        Hf = np.array(self.homography_frac, dtype=np.float64)
        S = np.diag([1.0 / content_w, 1.0 / content_h, 1.0])  # pixel -> fraction
        H = Hf @ S
        return Homography(H, np.linalg.inv(H))

    def set_homography_from_pixels(self, H_px: np.ndarray, content_w: int, content_h: int) -> None:
        Sinv = np.diag([content_w, content_h, 1.0])  # fraction -> pixel
        self.homography_frac = (H_px @ Sinv).tolist()

    def arena_crop(self, content_w: int, content_h: int, pad: float = 0.03) -> tuple[int, int, int, int]:
        """Pixel box around the arena (for the detector), from the corners."""
        if len(self.arena_corners_frac) == 4:
            xs = [c[0] for c in self.arena_corners_frac]
            ys = [c[1] for c in self.arena_corners_frac]
            x0, x1 = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
            y0, y1 = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
        else:
            x0, y0, w, h = self.rois["arena"]
            x1, y1 = x0 + w, y0 + h
        return int(x0 * content_w), int(y0 * content_h), int((x1 - x0) * content_w), int((y1 - y0) * content_h)
