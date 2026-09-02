"""Deployment labels: the game draws "<Card Name>" + "lvl N" in the arena at
the point where a card is deployed, for BOTH players, for about a second.
OCR on the arena crop therefore yields card identity and position that do
not depend on the unit detector's (frozen, mid-2024) class list. The same
OCR pass returns the tower HP numbers, which is a second, independent
tower-HP source.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import cv2
import numpy as np

LVL_RE = re.compile(r"^\D*(\d{1,2})\D*$")


@dataclass
class DeployLabel:
    card: str            # Phase 1 slug
    text: str            # raw OCR text
    match_score: float   # fuzzy similarity to the card name
    ocr_conf: float
    bbox: tuple[int, int, int, int]   # in CONTENT pixels
    level: int | None = None


@dataclass
class NumberLabel:
    value: int
    bbox: tuple[int, int, int, int]
    conf: float


class DeployLabelReader:
    def __init__(self, card_names: dict[str, str], ocr, scale: float = 1.0, min_similarity: float = 0.72):
        """card_names: slug -> display name."""
        self.ocr = ocr
        self.scale = scale
        self.min_sim = min_similarity
        self.names = {slug: self._norm(name) for slug, name in card_names.items()}
        self.norm_to_slug = {v: k for k, v in self.names.items()}
        self.keys = list(self.norm_to_slug.keys())

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z]", "", s.lower())

    def read(self, content: np.ndarray, arena_crop: tuple[int, int, int, int]) -> tuple[list[DeployLabel], list[NumberLabel]]:
        x0, y0, w, h = arena_crop
        img = content[y0:y0 + h, x0:x0 + w]
        if img.size == 0:
            return [], []
        s = self.scale
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        res, _ = self.ocr.ocr(img) if hasattr(self.ocr, "ocr") else self.ocr(img)
        labels, numbers = [], []
        boxes = []
        for r in res or []:
            pts, text, conf = r
            xs = [p[0] / s + x0 for p in pts]
            ys = [p[1] / s + y0 for p in pts]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            boxes.append((text, float(conf), bbox))
        for text, conf, bbox in boxes:
            t = text.strip()
            if conf < 0.5:
                continue
            if t.isdigit() and 1 <= len(t) <= 5:
                numbers.append(NumberLabel(int(t), bbox, conf))
                continue
            n = self._norm(t)
            if len(n) < 3:
                continue
            best = difflib.get_close_matches(n, self.keys, n=1, cutoff=self.min_sim)
            if not best:
                continue
            sim = difflib.SequenceMatcher(None, n, best[0]).ratio()
            slug = self.norm_to_slug[best[0]]
            # level from the line just below ("lvl 16")
            level = None
            for t2, c2, b2 in boxes:
                if b2[1] >= bbox[3] - 4 and b2[1] - bbox[3] < (bbox[3] - bbox[1]) * 1.5 and abs((b2[0] + b2[2]) / 2 - (bbox[0] + bbox[2]) / 2) < (bbox[2] - bbox[0]):
                    m = LVL_RE.match(t2.strip())
                    if m:
                        level = int(m.group(1))
                        bbox = (bbox[0], bbox[1], bbox[2], max(bbox[3], b2[3]))
                        break
            labels.append(DeployLabel(slug, t, round(sim, 3), conf, bbox, level))
        return labels, numbers


def label_ground_point(lbl: DeployLabel) -> tuple[float, float]:
    """The label block is drawn above the deployed unit; the unit stands a
    little below the block's bottom edge. Use the block's bottom-centre plus
    ~40% of its height as the ground point."""
    x1, y1, x2, y2 = lbl.bbox
    return (x1 + x2) / 2.0, y2 + 0.4 * (y2 - y1)
