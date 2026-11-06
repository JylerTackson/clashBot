"""HUD readers: fixed fractional ROIs, no learned models.

Every reader returns (value, confidence). Below a threshold the caller keeps
the previous value and marks the field stale; a wrong elixir is worse than a
missing one.

ROIs are fractions of the CONTENT rect (frame minus black bars), stored in
calib.json under "rois" as [x, y, w, h].
"""
from __future__ import annotations

import re
from collections import deque
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
    # hero ability button (MAC-AI): round, above the hand, on the side the hero was placed; each box is CENTRED on the
    # button (0.900 / 0.100, 0.776 of the content on the iPhone-mirroring layout: the executor taps the box centre)
    "hero_ability_left":  [0.00, 0.716, 0.20, 0.12],
    "hero_ability_right": [0.80, 0.716, 0.20, 0.12],
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
        # confidence: the fill must be dense up to the edge and sparse after it
        dense = float(filled[:run_end + 1].mean())
        after = float(filled[run_end + 3:].mean()) if run_end + 3 < len(filled) else 0.0
        conf = 0.9 if (dense >= 0.7 and after <= 0.35) else 0.6 if dense >= 0.5 else 0.3
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

    def __init__(self, images_dir: Path, extra_dirs: list[Path] | None = None, valid_slugs: set[str] | None = None,
                 learned_dir: Path | None = None, hero_images_dir: Path | None = None, costs: dict[str, int] | None = None):
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
        self.costs = costs or {}
        # hero variants use different card art but occupy the base card's slot
        if hero_images_dir and Path(hero_images_dir).exists():
            for p in sorted(Path(hero_images_dir).glob("*-hero.png")):
                slug = p.stem[:-5]
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None and slug in known:
                    self.templates.append(CardTemplate(slug, self.describe(img)))
        # self-labelled in-game templates: <learned_dir>/<slug>/*.png (written by
        # Perception when a deploy label confirms which card left a hand slot)
        self.learned_dir = learned_dir
        if learned_dir and Path(learned_dir).exists():
            for d in sorted(Path(learned_dir).iterdir()):
                if d.is_dir() and d.name in known:
                    for p in self.learned_files(d):
                        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                        if img is not None:
                            self.templates.append(CardTemplate(d.name, self.describe(img)))
        for d in [Path(x) for x in (extra_dirs or [])]:
            for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                # BuildABot "<name>_ev1" and the wiki's "<slug>-evolution": the evolved card occupies the base card's slot
                stem = re.sub(r"-evolution$", "", re.sub(r"_ev\d+$", "", p.stem))
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

    LEARNED_CAP = 12                 # per card: the last 12 crops of other videos + the last 12 live crops
    LIVE_PREFIX = "live_"            # crops written by the live loop (no video path -> stem "live")

    @classmethod
    def learned_files(cls, card_dir: Path) -> list[Path]:
        """The learned crops of one card to load: the last LEARNED_CAP of the video crops plus, in addition to
        that cap, the last LEARNED_CAP live ones (otherwise a card whose cap the pro videos filled never learns
        from the live screen)."""
        files = sorted(Path(card_dir).glob("*.png"))
        live = [p for p in files if p.name.startswith(cls.LIVE_PREFIX)]
        other = [p for p in files if not p.name.startswith(cls.LIVE_PREFIX)]
        return other[-cls.LEARNED_CAP:] + live[-cls.LEARNED_CAP:]

    def add_learned(self, slug: str, crop: np.ndarray, path: Path | None = None) -> None:
        """Add an in-game crop as a template for `slug` (and persist it)."""
        if crop is None or crop.size == 0:
            return
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), crop)
        self.templates.append(CardTemplate(slug, self.describe(crop)))
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

    @staticmethod
    def is_greyed(crop: np.ndarray) -> bool:
        """An unaffordable card is drawn desaturated: its colour histogram is
        meaningless, only the grayscale shape survives."""
        h, w = crop.shape[:2]
        art = crop[int(0.15 * h):int(0.85 * h), int(0.1 * w):int(0.9 * w)]
        sat = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)[..., 1]
        return float(np.mean(sat)) < 45

    def match(self, crop: np.ndarray, top: int = 3, cost: int | None = None,
              allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """cost: elixir cost read from the slot badge; when given, cards of a
        different cost are dropped (only if any template has that cost).
        allowed: the cards that can be in the hand (the own deck); every other
        card scores -9, so the best card of the deck wins (only if any template
        belongs to the set)."""
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
            return []
        f = self.describe(crop)
        n = self.F.shape[1] - 48
        gcorr = self.F[:, :n] @ f[:n]                      # ~ [-1, 1]
        hdist = np.abs(self.F[:, n:] - f[n:]).sum(axis=1) / 3.0  # [0, 2]
        if self.is_greyed(crop):
            score = gcorr
        else:
            score = 0.75 * gcorr + 0.25 * (1.0 - hdist / 2.0)
        if cost is not None and self.costs:
            ok = np.array([self.costs.get(sl) == cost for sl in self.slugs])
            if ok.any():
                score = np.where(ok, score, -9.0)
        if allowed:
            ok = np.array([sl in allowed for sl in self.slugs])
            if ok.any():
                score = np.where(ok, score, -9.0)
        order = np.argsort(-score)
        out: list[tuple[str, float]] = []
        for i in order:
            if all(self.slugs[i] != s for s, _ in out):
                out.append((self.slugs[i], float(score[i])))
            if len(out) >= top:
                break
        return out


class HandReader:
    """Four hand slots + next card. Confidence = best score, discounted when
    the runner-up is close (ambiguous). Empty/greyed slots (card just played)
    are reported as None with the reason."""

    # the elixir-cost badge sits over the slot's bottom centre (fractions of the slot box)
    BADGE = (0.28, 0.70, 0.44, 0.32)
    # alignment search: the calibrated box can be off by a few % (different
    # phones/streaming layouts), and a selected card is drawn raised and larger
    SHIFTS = [(dx, dy, sc) for sc in (1.0, 1.12) for dy in (0.0, -0.018, -0.032) for dx in (-0.02, 0.0, 0.02, 0.04)]

    def __init__(self, matcher: CardMatcher, rois: dict, threshold: float = 0.45, margin: float = 0.04,
                 digits: "DigitTemplates | None" = None, allowed: set[str] | None = None):
        self.m, self.rois, self.thr, self.margin = matcher, rois, threshold, margin
        self.digits = digits
        self.allowed = set(allowed) if allowed else None     # the own deck (live loop); None = all cards

    def _badge_cost(self, content: np.ndarray, roi) -> int | None:
        if self.digits is None:
            return None
        bx, by, bw, bh = self.BADGE
        sub = [roi[0] + bx * roi[2], roi[1] + by * roi[3], bw * roi[2], bh * roi[3]]
        txt, conf = DigitReader(sub, self.digits, white_text=True).read_string(content)
        if len(txt) == 1 and txt.isdigit() and conf >= 0.75:
            return int(txt)
        return None

    def _slot(self, content: np.ndarray, roi) -> tuple[str | None, float, list]:
        best = (None, 0.0, [])
        for dx, dy, sc in self.SHIFTS:
            w, h = roi[2] * sc, roi[3] * sc
            r = self._slot_at(content, [roi[0] + dx, roi[1] + dy - (h - roi[3]), w, h])
            if r[1] > best[1]:
                best = r
        return best

    def _slot_at(self, content: np.ndarray, roi) -> tuple[str | None, float, list]:
        crop = crop_roi(content, roi)
        if crop.size == 0:
            return None, 0.0, []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.std() < 12:                      # blank slot
            return None, 0.0, []
        cands = self.m.match(crop, cost=self._badge_cost(content, roi), allowed=self.allowed)
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
# Hero ability button (Hero Musketeer's Trusty Turret, MAC-AI)
# ---------------------------------------------------------------------------

ABILITY_DIR = Path(__file__).resolve().parent / "assets" / "hero_ability"
ABILITY_REF_W = 600.0            # content width (px) the templates were cut at (a 632 px mirrored frame minus its margins)
ABILITY_WORK = 0.5               # matching runs at this fraction of that scale (the templates come from 50 % shots)
ABILITY_DISC = (30.0, 26.0, 23.5)   # disc centre x, y and outer radius inside a template (work-scale px)
ABILITY_ICON_R = 0.58            # radius of the icon region (the disc without its ring and badge), in disc radii


@dataclass
class _AbilityTemplate:
    state: str              # "ready" (coloured, affordable) | "greyed" (unaffordable) | "flash" (white, just became affordable)
    gray: np.ndarray        # work-scale grayscale, float32
    mask: np.ndarray        # work-scale {0, 1} float32: the disc and the cost badge, not the background around them


@dataclass
class _Prepared:
    """A template at one (mirror, scale): the zero-mean masked form the correlation needs, and the icon region."""
    state: str
    gray: np.ndarray
    t0: np.ndarray          # gray minus its mean under `mask`, zero outside it
    mask: np.ndarray        # shared between templates with equal masks (its window sums are computed once)
    n: float
    std: float
    icon: np.ndarray        # {0, 1} float32: the inner disc, where the hero's icon is


class AbilityButtonReader:
    """Finds the hero ability button and reads whether it is usable. No learned model.

    The button is a round disc with the hero's icon and a purple elixir badge ("3") on its top-left; it is drawn
    greyed when the elixir is short and flashes white for a moment when it becomes affordable. Templates live in
    `assets/hero_ability/<hero>[_grey|_flash].png` (BGRA: the alpha channel is the disc + badge mask), cut from the
    recorded live shots `runs/poc/20261106-170722/shots/` (`d1999_t249.9_b`, `d1175_t144.7_b`, `d2416_t300.6_c`; box
    x 248..304, y 517..569 of the 316 px shot, upscaled x2).

    Matching is lighting-normalised like `CardMatcher`: a masked zero-mean normalised cross-correlation of the grayscale
    ROI against each template (3 scales around the reference). Every hero's button has the same disc and badge, so the
    score alone cannot tell the Musketeer's from another hero's (they score 0.6-0.75): the same correlation over the
    inner disc only (`icon`) has to match too. A circle check follows (the colour just inside the disc differs from
    the colour just outside it, all around). The state comes from the saturation of the disc's ring: coloured ->
    affordable, grey -> not, the white flash or anything in between -> the elixir (>= cost). The left box also tries
    the mirrored template (the left-hand layout has not been recorded). `read` adds hysteresis: visible in `need` of
    the last `window` reads."""

    SCALES = (0.92, 1.0, 1.08)
    REFINE_FROM = 0.45            # a best score this high at the reference scale is also tried at the other scales
    PAD = 6                       # work-scale px of replicated border around a ROI (the box may be a few px off)
    SAT_READY, SAT_GREY, V_FLASH = 70.0, 40.0, 200.0
    SIDES = ("left", "right")

    def __init__(self, rois: dict | None = None, directory: Path | str = ABILITY_DIR, hero: str = "musketeer",
                 cost: int = 3, threshold: float = 0.62, icon_min: float = 0.7, ring_min: float = 0.6,
                 window: int = 3, need: int = 2):
        self.rois = dict(rois or {})
        self.hero, self.cost = hero, cost
        self.threshold, self.icon_min, self.ring_min = threshold, icon_min, ring_min
        self.need = need
        self.hist: deque[dict] = deque(maxlen=window)
        self.templates: list[_AbilityTemplate] = []
        for p in sorted(Path(directory).glob(f"{hero}*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                continue
            k = ABILITY_WORK
            bgr = cv2.resize(img[..., :3], None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = (cv2.resize(img[..., 3], None, fx=k, fy=k, interpolation=cv2.INTER_AREA) > 127).astype(np.float32)
            state = "greyed" if p.stem.endswith("_grey") else "flash" if p.stem.endswith("_flash") else "ready"
            self.templates.append(_AbilityTemplate(state, gray, mask))
        self._cache: dict[tuple, tuple] = {}      # (flip, scale) -> tuple[_Prepared]

    # -- geometry --------------------------------------------------------------
    def roi(self, side: str) -> list[float]:
        key = f"hero_ability_{side}"
        return list(self.rois.get(key) or DEFAULT_ROIS[key])

    def center(self, side: str) -> tuple[float, float]:
        """Content fractions of the button centre = the centre of its ROI (where the executor taps)."""
        x, y, w, h = self.roi(side)
        return x + w / 2, y + h / 2

    # -- matching --------------------------------------------------------------
    def _variant(self, flip: bool, scale: float) -> tuple[_Prepared, ...]:
        """The templates resized by `scale` (and mirrored for `flip`), built once per (flip, scale)."""
        key = (flip, scale)
        if key not in self._cache:
            prep: list[_Prepared] = []
            for t in self.templates:
                g, m = t.gray, t.mask
                if flip:
                    g, m = g[:, ::-1], m[:, ::-1]
                g = cv2.resize(np.ascontiguousarray(g), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                m = (cv2.resize(np.ascontiguousarray(m), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
                     ).astype(np.float32)
                n = float(m.sum())
                if n < 50:
                    continue
                t0 = (g - float((g * m).sum() / n)) * m
                std = float(np.sqrt((t0 * t0).sum() / n))
                if std < 1e-3:
                    continue
                same = next((p.mask for p in prep if p.mask.shape == m.shape and np.array_equal(p.mask, m)), None)
                icon = np.zeros(g.shape, np.uint8)
                cx = ((t.gray.shape[1] - ABILITY_DISC[0]) if flip else ABILITY_DISC[0]) * scale
                cv2.circle(icon, (int(round(cx)), int(round(ABILITY_DISC[1] * scale))),
                           int(round(ABILITY_ICON_R * ABILITY_DISC[2] * scale)), 1, -1)
                prep.append(_Prepared(t.state, g, t0, m if same is None else same, n, std, icon.astype(np.float32)))
            self._cache[key] = tuple(prep)
        return self._cache[key]

    @staticmethod
    def _zncc(img: np.ndarray, sums: dict, p: _Prepared) -> np.ndarray:
        """Masked zero-mean normalised cross-correlation map of `img` with the template; `sums` memoises the image's
        window standard deviation under each distinct mask."""
        if img.shape[0] < p.t0.shape[0] or img.shape[1] < p.t0.shape[1]:
            return np.empty((0, 0), np.float32)
        if id(p.mask) not in sums:
            s1 = cv2.matchTemplate(img, p.mask, cv2.TM_CCORR) / p.n
            s2 = cv2.matchTemplate(img * img, p.mask, cv2.TM_CCORR) / p.n
            sums[id(p.mask)] = np.sqrt(np.maximum(s2 - s1 * s1, 1e-6))
        return cv2.matchTemplate(img, p.t0, cv2.TM_CCORR) / (p.n * sums[id(p.mask)] * p.std + 1e-6)

    @staticmethod
    def _patch_zncc(window: np.ndarray, p: _Prepared) -> float:
        """The same correlation over the icon region only, for one window of the image."""
        m = p.icon
        n = float(m.sum())
        if n < 20 or window.shape != p.gray.shape:
            return 0.0
        a = (p.gray - float((p.gray * m).sum() / n)) * m
        b = (window - float((window * m).sum() / n)) * m
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / den) if den > 1e-3 else 0.0

    def _work_roi(self, content: np.ndarray, side: str) -> np.ndarray | None:
        """The ROI at work scale (templates' scale), padded with replicated borders; None when it is empty."""
        sub = crop_roi(content, self.roi(side))
        if sub.size == 0 or min(sub.shape[:2]) < 8:
            return None
        k = ABILITY_WORK * ABILITY_REF_W / content.shape[1]
        sub = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        return cv2.copyMakeBorder(sub, self.PAD, self.PAD, self.PAD, self.PAD, cv2.BORDER_REPLICATE)

    def _best(self, gray: np.ndarray, flips: tuple[bool, ...]):
        """Best (score, prepared template, flip, scale, (x, y) of the template's top-left) over templates, mirrored
        variants and scales: the reference scale first, the others only when that is promising."""
        sums: dict = {}
        best: tuple = (-1.0, None, False, 1.0, (0, 0))
        for scales in ((1.0,), tuple(s for s in self.SCALES if s != 1.0)):
            if scales != (1.0,) and best[0] < self.REFINE_FROM:
                break
            for flip in flips:
                for s in scales:
                    for p in self._variant(flip, s):
                        r = self._zncc(gray, sums, p)
                        if r.size == 0:
                            continue
                        _, mx, _, loc = cv2.minMaxLoc(r)
                        if mx > best[0]:
                            best = (float(mx), p, flip, s, (int(loc[0]), int(loc[1])))
        return best

    def _ring(self, sub: np.ndarray, cx: float, cy: float, s: float, n: int = 36) -> float:
        """Circle check: the fraction of directions in which the colour at 0.72 R (the ring of the disc) differs from
        the colour at 1.23 R (the background); 0 when too much of the circle is outside the image."""
        r = ABILITY_DISC[2] * s
        a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        h, w = sub.shape[:2]
        xi = np.rint(cx + 0.72 * r * np.cos(a)).astype(int)
        yi = np.rint(cy + 0.72 * r * np.sin(a)).astype(int)
        xo = np.rint(cx + 1.23 * r * np.cos(a)).astype(int)
        yo = np.rint(cy + 1.23 * r * np.sin(a)).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (xo >= 0) & (xo < w) & (yo >= 0) & (yo < h)
        if ok.sum() < n // 2:
            return 0.0
        d = np.abs(sub[yi[ok], xi[ok]].astype(np.float32) - sub[yo[ok], xo[ok]].astype(np.float32)).mean(axis=1)
        return float((d > 28.0).mean())

    def _ring_hsv(self, sub: np.ndarray, cx: float, cy: float, s: float) -> tuple[float, float]:
        """Mean saturation and value of the disc's ring (0.55 R .. 0.9 R around the centre)."""
        r = ABILITY_DISC[2] * s
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        yy, xx = np.mgrid[0:hsv.shape[0], 0:hsv.shape[1]]
        d = np.hypot(xx - cx, yy - cy)
        ring = (d >= 0.55 * r) & (d <= 0.9 * r)
        if not ring.any():
            return 0.0, 0.0
        return float(hsv[..., 1][ring].mean()), float(hsv[..., 2][ring].mean())

    def detect(self, content: np.ndarray) -> dict | None:
        """The best candidate over both sides, whether or not it passes the thresholds: {"side", "score", "icon",
        "ring", "state" (best-matching template), "sat", "val", "visible"}; None without templates / usable ROIs."""
        if not self.templates or content is None or content.ndim != 3:
            return None
        out = None
        for side in self.SIDES:
            sub = self._work_roi(content, side)
            if sub is None:
                continue
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
            score, p, flip, s, (px, py) = self._best(gray, (False, True) if side == "left" else (False,))
            if p is None:
                continue
            th, tw = p.gray.shape
            icon = self._patch_zncc(gray[py:py + th, px:px + tw], p)
            cx = px + ((self.templates[0].gray.shape[1] - ABILITY_DISC[0]) if flip else ABILITY_DISC[0]) * s
            cy = py + ABILITY_DISC[1] * s                       # a mirrored box has its disc centre mirrored
            ring = self._ring(sub, cx, cy, s)
            sat, val = self._ring_hsv(sub, cx, cy, s)
            cand = {"side": side, "score": score, "icon": icon, "ring": ring, "state": p.state, "sat": sat, "val": val,
                    "visible": bool(score >= self.threshold and icon >= self.icon_min and ring >= self.ring_min)}
            if out is None or cand["score"] > out["score"]:
                out = cand
        return out

    def _affordable(self, d: dict, elixir) -> bool:
        if d["sat"] >= self.SAT_READY:
            return True                       # coloured
        if d["sat"] <= self.SAT_GREY and d["val"] < self.V_FLASH:
            return False                      # greyed out
        return bool(elixir is not None and elixir >= self.cost)    # the white flash / anything in between

    def read_raw(self, content: np.ndarray, elixir=None) -> dict:
        """One frame, no hysteresis: {"hero", "visible", "side", "affordable", "cost", "conf"}."""
        d = self.detect(content)
        base = {"hero": self.hero, "visible": False, "side": None, "affordable": False, "cost": self.cost,
                "conf": 0.0 if d is None else round(float(np.clip(min(d["score"], d["icon"]), 0.0, 1.0)), 3)}
        if d is None or not d["visible"]:
            return base
        return {**base, "visible": True, "side": d["side"], "affordable": self._affordable(d, elixir)}

    def read(self, content: np.ndarray, elixir=None) -> dict:
        """`read_raw` with hysteresis: visible when `need` of the last `window` reads saw the button; the side,
        affordability and confidence then come from those reads (the latest one decides side and affordability)."""
        raw = self.read_raw(content, elixir)
        self.hist.append(raw)
        seen = [r for r in self.hist if r["visible"]]
        if len(seen) < self.need:
            return {**raw, "visible": False, "side": None, "affordable": False,
                    "conf": max(r["conf"] for r in self.hist)}
        last = seen[-1]
        return {**last, "conf": round(float(np.mean([r["conf"] for r in seen])), 3)}

    def reset(self) -> None:
        self.hist.clear()


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
