"""HUD readers: fixed fractional ROIs, no learned models.

Every reader returns (value, confidence). Below a threshold the caller keeps
the previous value and marks the field stale; a wrong elixir is worse than a
missing one.

ROIs are fractions of the CONTENT rect (frame minus black bars), stored in
calib.json under "rois" as [x, y, w, h].
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Default fractional ROIs for the portrait 9:16-ish game layout. Calibration
# overrides these; they are only starting points.
DEFAULT_ROIS = {
    # measured on a 596x1280 phone recording (aspect 2.148); calib.json overrides
    "arena":       [0.02, 0.12, 0.96, 0.66],
    "clock":       [0.83, 0.083, 0.16, 0.035],     # digits under the "Time left:" label
    "elixir_bar":  [0.29, 0.958, 0.69, 0.024],     # part of the bar not covered by the badge
    "elixir_bar_full": [0.25, 0.972],              # true x-extent of the bar (fill is measured against this)
    "elixir_num":  [0.262, 0.947, 0.065, 0.042],   # the numeral right of the droplet icon
    "hand_0":      [0.235, 0.848, 0.160, 0.095],
    "hand_1":      [0.418, 0.848, 0.160, 0.095],
    "hand_2":      [0.598, 0.848, 0.160, 0.095],
    "hand_3":      [0.790, 0.848, 0.160, 0.095],
    "next_card":   [0.040, 0.928, 0.095, 0.048],
    "own_king_hp":    [0.40, 0.700, 0.20, 0.025],
    "own_left_hp":    [0.155, 0.640, 0.12, 0.020],
    "own_right_hp":   [0.735, 0.640, 0.12, 0.020],
    "enemy_king_hp":  [0.40, 0.108, 0.20, 0.025],
    "enemy_left_hp":  [0.155, 0.190, 0.12, 0.020],
    "enemy_right_hp": [0.735, 0.190, 0.12, 0.020],
}


def crop_roi(img: np.ndarray, roi) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = int(round(x * w)), int(round(y * h))
    x1, y1 = int(round((x + rw) * w)), int(round((y + rh) * h))
    return img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]


# ---------------------------------------------------------------------------
# Elixir
# ---------------------------------------------------------------------------

def _purple_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 125) & (h <= 170) & (s > 80) & (v > 90)


class ElixirBarReader:
    """Elixir from the fill length of the purple bar (0-10). The bar has 10
    segments; fill fraction * 10, rounded down, is the integer elixir. The
    digit read (ElixirDigitReader) is cross-checked when available."""

    def __init__(self, roi, full_extent: list[float] | None = None):
        self.roi = roi
        self.full = full_extent   # [x0, x1] fractions of the content width for the whole bar

    def read(self, content: np.ndarray) -> tuple[int | None, float]:
        img = crop_roi(content, self.roi)
        if img.size == 0:
            return None, 0.0
        m = _purple_mask(img)
        col = m.mean(axis=0)              # per-column purple fraction
        filled = col > 0.35
        W = content.shape[1]
        if self.full:
            x0_px = self.roi[0] * W
            full_px = (self.full[1] - self.full[0]) * W
            lead_px = x0_px - self.full[0] * W        # bar length hidden under the badge
        else:
            full_px, lead_px = float(len(col)), 0.0
        if not filled.any():
            return 0, 0.4
        # fill end = rightmost purple column while the cumulative purple density
        # from the left stays high (tolerates separator lines and flashes)
        n = len(col)
        cum = np.cumsum(filled) / np.arange(1, n + 1)
        ok_idx = np.where(filled & (cum >= 0.7))[0]
        run_end = int(ok_idx[-1]) if len(ok_idx) else int(np.where(filled)[0][0])
        frac = (lead_px + run_end + 1) / full_px
        elixir = int(np.clip(round(frac * 10), 0, 10))   # the fill is quantised in tenths
        elixir = max(0, min(10, elixir))
        # confidence: how clean the edge is (few purple columns after the run)
        noise = float(filled[run_end + 3:].mean()) if run_end + 3 < len(filled) else 0.0
        conf = max(0.0, 1.0 - 3.0 * noise) * (0.9 if m.mean() > 0.02 else 0.3)
        return elixir, round(conf, 2)


class DigitTemplates:
    """Digit templates for the elixir numeral and the clock: rendered from a
    bold font (works because the game uses a heavy, high-contrast font) and
    matched after binarisation. Templates can be replaced with real crops via
    tools/calibrate.py --digits to improve accuracy."""

    def __init__(self, height: int = 24, path: Path | None = None):
        self.height = height
        self.tpl: dict[str, np.ndarray] = {}
        if path and Path(path).exists():
            for p in Path(path).glob("*.png"):
                self.tpl[p.stem] = self._prep(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        if not self.tpl:
            for d in "0123456789":
                canvas = np.zeros((40, 30), np.uint8)
                cv2.putText(canvas, d, (2, 34), cv2.FONT_HERSHEY_DUPLEX, 1.3, 255, 3)
                self.tpl[d] = self._prep(canvas)

    def _prep(self, g: np.ndarray) -> np.ndarray:
        ys, xs = np.where(g > 127)
        if len(xs) == 0:
            return np.zeros((self.height, self.height // 2), np.uint8)
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        w = max(4, int(g.shape[1] * self.height / g.shape[0]))
        return (cv2.resize(g, (w, self.height), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)

    def match(self, glyph: np.ndarray) -> tuple[str, float]:
        g = self._prep(glyph)
        best, best_s = "?", -1.0
        for d, t in self.tpl.items():
            tw = max(g.shape[1], t.shape[1])
            a = cv2.resize(g, (tw, self.height), interpolation=cv2.INTER_NEAREST)
            b = cv2.resize(t, (tw, self.height), interpolation=cv2.INTER_NEAREST)
            s = float((a == b).mean())
            if s > best_s:
                best, best_s = d, s
        return best, best_s


def binarize_text(img: np.ndarray, white: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not white:
        b = 255 - b
    return b


def segment_glyphs(binary: np.ndarray, min_w: int = 3) -> list[tuple[int, int, int, int]]:
    """Connected components -> glyph boxes sorted left to right (colon and
    noise filtered by height)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = [(x, y, w, h) for x, y, w, h, a in stats[1:] if w >= min_w and h >= 4]
    if not comps:
        return []
    max_h = max(h for _, _, _, h in comps)
    # keep glyph-sized components (relative to the tallest one, not the crop)
    boxes = [b for b in comps if b[3] >= 0.35 * max_h]
    return sorted(boxes)


class DigitReader:
    """Reads an integer or a m:ss clock from a ROI with template digits."""

    def __init__(self, roi, templates: DigitTemplates, white_text: bool = True):
        self.roi, self.tpl, self.white = roi, templates, white_text

    def read_string(self, content: np.ndarray) -> tuple[str, float]:
        img = crop_roi(content, self.roi)
        if img.size == 0:
            return "", 0.0
        b = binarize_text(img, self.white)
        boxes = segment_glyphs(b)
        if not boxes:
            return "", 0.0
        out, confs = [], []
        prev_x1 = None
        H = max(h for _, _, _, h in boxes)
        for x, y, w, h in boxes:
            if prev_x1 is not None and x - prev_x1 > 0.5 * H and out and ":" not in out:
                out.append(":")
            if h < 0.6 * H and w < 0.5 * H:   # small blob: colon / noise
                prev_x1 = x + w
                continue
            d, s = self.tpl.match(b[y:y + h, x:x + w])
            out.append(d)
            confs.append(s)
            prev_x1 = x + w
        s = "".join(out).strip(":")
        return s, (float(min(confs)) if confs else 0.0)


class OcrReader:
    """RapidOCR (ONNX, no torch) for the clock / HP labels / phase banner.
    Slower (~0.1-0.8 s) so run it at a reduced rate."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr = RapidOCR()

    def read(self, img: np.ndarray, allow: str = "0123456789:") -> tuple[str, float]:
        if img.size == 0:
            return "", 0.0
        scale = max(1.0, 48.0 / img.shape[0])
        if scale > 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res, _ = self.ocr(img)
        if not res:
            return "", 0.0
        text = "".join(r[1] for r in res)
        conf = float(np.mean([r[2] for r in res]))
        cleaned = "".join(ch for ch in text if ch in allow)
        return cleaned, conf if cleaned else 0.0


CLOCK_RE = re.compile(r"^(\d):(\d\d)$")


def parse_clock(s: str) -> int | None:
    m = CLOCK_RE.match(s)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


# ---------------------------------------------------------------------------
# Hand / next card: template matching against the Phase 1 card art
# ---------------------------------------------------------------------------

@dataclass
class CardTemplate:
    slug: str
    feat: np.ndarray   # small normalised grayscale + colour descriptor


class CardMatcher:
    """Identifies a card crop by nearest neighbour over the Phase 1 card art.
    The descriptor is a 24x28 grayscale thumbnail (lighting-normalised) plus a
    coarse HSV histogram, compared with a correlation score. Deterministic,
    scale-invariant (both sides resized), no training."""

    SIZE = (24, 28)

    BUILDABOT_ALIAS = {"pekka": "p-e-k-k-a", "minipekka": "mini-p-e-k-k-a", "mini_pekka": "mini-p-e-k-k-a", "x_bow": "x-bow",
                       "log": "the-log", "fire_spirits": "fire-spirit"}

    def __init__(self, images_dir: Path, extra_dirs: list[Path] | None = None, valid_slugs: set[str] | None = None):
        """images_dir: Phase 1 wiki art (slug.png). extra_dirs: additional
        template sets (e.g. BuildABot's in-game hand thumbnails, *.jpg with
        underscore names and _ev1 evolution variants); their names are mapped
        onto Phase 1 slugs and unknown ones are skipped."""
        self.templates: list[CardTemplate] = []
        base = Path(images_dir)
        for p in sorted(base.glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                self.templates.append(CardTemplate(p.stem, self.describe(img)))
        known = valid_slugs or {t.slug for t in self.templates}
        for d in [Path(x) for x in (extra_dirs or [])]:
            for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                stem = re.sub(r"_ev\d+$", "", p.stem)
                slug = self.BUILDABOT_ALIAS.get(stem, stem.replace("_", "-"))
                if slug not in known:
                    continue
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    self.templates.append(CardTemplate(slug, self.describe(img)))
        if not self.templates:
            raise FileNotFoundError(f"no card art in {images_dir}")
        self.F = np.stack([t.feat for t in self.templates])
        self.slugs = [t.slug for t in self.templates]

    @classmethod
    def describe(cls, img: np.ndarray) -> np.ndarray:
        # ignore the elixir-cost badge (top-left) and the bottom name strip by
        # cropping the central art region
        h, w = img.shape[:2]
        art = img[int(0.12 * h):int(0.88 * h), int(0.08 * w):int(0.92 * w)]
        g = cv2.cvtColor(cv2.resize(art, cls.SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        hsv = cv2.cvtColor(cv2.resize(art, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-6)
        return np.concatenate([g.ravel() / np.sqrt(g.size), hist * 3.0]).astype(np.float32)

    def match(self, crop: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
            return []
        f = self.describe(crop)
        n = self.F.shape[1] - 48
        gcorr = self.F[:, :n] @ f[:n]                      # ~ [-1, 1]
        hdist = np.abs(self.F[:, n:] - f[n:]).sum(axis=1) / 3.0  # [0, 2]
        score = 0.75 * gcorr + 0.25 * (1.0 - hdist / 2.0)
        order = np.argsort(-score)[:top]
        return [(self.slugs[i], float(score[i])) for i in order]


class HandReader:
    """Four hand slots + next card. Confidence = best score, discounted when
    the runner-up is close (ambiguous). Empty/greyed slots (card just played)
    are reported as None with the reason."""

    def __init__(self, matcher: CardMatcher, rois: dict, threshold: float = 0.45, margin: float = 0.04):
        self.m, self.rois, self.thr, self.margin = matcher, rois, threshold, margin

    def _slot(self, content: np.ndarray, roi) -> tuple[str | None, float, list]:
        # a selected card is drawn raised: try the nominal box and one shifted up
        best = (None, 0.0, [])
        for dy in (0.0, -0.018):
            r = self._slot_at(content, [roi[0], roi[1] + dy, roi[2], roi[3]])
            if r[1] > best[1]:
                best = r
        return best

    def _slot_at(self, content: np.ndarray, roi) -> tuple[str | None, float, list]:
        crop = crop_roi(content, roi)
        if crop.size == 0:
            return None, 0.0, []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.std() < 12:                      # blank / greyed-out slot
            return None, 0.0, []
        cands = self.m.match(crop)
        if not cands:
            return None, 0.0, []
        slug, s = cands[0]
        conf = s
        if len(cands) > 1 and (s - cands[1][1]) < self.margin:
            conf *= 0.6
        if conf < self.thr:
            return None, round(conf, 3), cands
        return slug, round(conf, 3), cands

    def read(self, content: np.ndarray) -> dict:
        out = {"hand": [], "hand_conf": [], "candidates": []}
        for i in range(4):
            slug, conf, cands = self._slot(content, self.rois[f"hand_{i}"])
            out["hand"].append(slug)
            out["hand_conf"].append(conf)
            out["candidates"].append(cands[:3])
        slug, conf, cands = self._slot(content, self.rois["next_card"])
        out["next_card"], out["next_conf"] = slug, conf
        return out


# ---------------------------------------------------------------------------
# Tower HP labels (small numbers above each tower) + destroyed detection
# ---------------------------------------------------------------------------

class TowerHpReader:
    TOWERS = ("own_king", "own_left", "own_right", "enemy_king", "enemy_left", "enemy_right")

    def __init__(self, rois: dict, ocr: OcrReader | None):
        self.rois, self.ocr = rois, ocr

    def read(self, content: np.ndarray) -> dict:
        out = {}
        for name in self.TOWERS:
            roi = self.rois.get(f"{name}_hp")
            if roi is None:
                out[name] = (None, 0.0)
                continue
            img = crop_roi(content, roi)
            if img.size == 0:
                out[name] = (None, 0.0)
                continue
            # a destroyed tower shows no HP label: little bright text in the ROI
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bright = float((gray > 200).mean())
            if bright < 0.01:
                out[name] = (0, 0.6)          # 0 HP = destroyed (moderate confidence)
                continue
            if self.ocr is None:
                out[name] = (None, 0.0)
                continue
            s, c = self.ocr.read(img, allow="0123456789")
            out[name] = (int(s) if s.isdigit() else None, round(c, 2) if s.isdigit() else 0.0)
        return out
