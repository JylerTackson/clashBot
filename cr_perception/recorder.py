"""JSONL recorder: one line per GameState and per PlayEvent, plus optional
frame dumps for later hand-labelling / training."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .state import dumps_any


class JsonlRecorder:
    def __init__(self, path: str | Path, frames_dir: str | Path | None = None, frame_every: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a")
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.frame_every = frame_every
        if self.frames_dir:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.write({"type": "session_start", "wall_time": time.time()})

    def write(self, obj) -> None:
        self.f.write(dumps_any(obj) + "\n")
        self.n += 1
        if self.n % 50 == 0:
            self.f.flush()

    def frame(self, frame_index: int, image: np.ndarray, force: bool = False) -> str | None:
        if not self.frames_dir:
            return None
        if not force and (self.frame_every <= 0 or frame_index % self.frame_every):
            return None
        p = self.frames_dir / f"{frame_index:07d}.jpg"
        cv2.imwrite(str(p), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(p)

    def close(self) -> None:
        self.write({"type": "session_end", "wall_time": time.time()})
        self.f.close()


def read_jsonl(path: str | Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
