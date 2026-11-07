"""Training Camp detector (MAC-AN, D-050): the opponent banner at the top-left of the arena reads "Trainer <Name>"
(pink, above the clan line) in Training Camp; ladder and friendly battles show a player name there. Read-only.

    box = banner_box(frame.shape, calib)            # the name line, in frame pixels
    text = read_banner(frame, calib, ocr)           # crop, upscale, recognise -> "TraiNeR CHeddaR I"
    is_trainer(text)                                # a token within edit distance 2 of "trainer"
    verdict(checks)                                 # Training Camp when >= 2 checked frames are trainer banners

Unknown (no text, no OCR, too few checks) is "not Training Camp": the launcher fails closed.

OCR: RapidOCR, the engine the HUD uses, in recognition-only mode on the single-line crop. Its text detector split the
stylised "TRAINER" (small capitals) into pieces and read "TR8 NeRO" on half-size frames (0 of 412 shots, ~450 ms a call);
the recogniser alone reads "TRaiNeR CHeddaR I" (412 of 412, ~10 ms).
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from .config import Calibration

# The name line as fractions of the content rect (calib.json `content_rect_frac`), measured on the half-size shots of
# runs/poc/2026110*/shots (a shot is the whole window frame at half size, its text header sits over the black bezel)
# and on the full-size frame runs/poc/mirror_frame_quartz.png. The clan badge ends at x 0.077, the name starts at 0.093
# and "Trainer Cheddar II" ends near 0.41, so the box runs to 0.483 for longer names; the clan line below starts at
# y 0.088. The box is one text line: the recogniser reads whatever is in it.
BANNER_FRAC = (0.083, 0.056, 0.40, 0.028)           # x, y, w, h of the content rect
MIN_SCALE = 2.0                                     # upscale of the crop ...
TARGET_H = 64.0                                     # ... or more, so that it is about this many pixels high
MAX_SCALE = 6.0
WORD = "trainer"
MAX_EDITS = 2
MIN_TOKEN = 5                                       # shorter fragments ("tra", "er") are noise, not the word
MIN_TRAINER_FRAMES = 2                              # Training Camp when at least this many checked frames read "Trainer"


def _as_calib(calib) -> Calibration:
    if isinstance(calib, Calibration):
        return calib
    if isinstance(calib, dict):
        return Calibration(**{k: calib.get(k, v) for k, v in Calibration().__dict__.items()})
    return Calibration.load(calib)


def banner_box(frame_shape, calib: Calibration | dict | str) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the opponent name line in the pixels of a frame of `frame_shape` (h, w[, c]); `calib` is a
    `Calibration`, its dict or a calib.json path. The box follows the content rect, so a Retina or half-size frame works."""
    calib = _as_calib(calib)
    h, w = int(frame_shape[0]), int(frame_shape[1])
    cx, cy, cw, ch = calib.content_rect(w, h)
    fx, fy, fw, fh = BANNER_FRAC
    x0, y0 = cx + int(round(fx * cw)), cy + int(round(fy * ch))
    x1, y1 = x0 + int(round(fw * cw)), y0 + int(round(fh * ch))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _engine(ocr):
    """The callable behind an OCR object: an `OcrReader` (its RapidOCR is `.ocr`) or a RapidOCR / fake callable."""
    return getattr(ocr, "ocr", ocr)


def _texts(res) -> list[str]:
    """The strings of one OCR result: recognition-only rows are [text, conf], detection rows [box, text, conf] (these
    are joined left to right)."""
    rows = []
    for r in res or []:
        try:
            if isinstance(r[0], str):
                rows.append((0.0, r[0]))
            else:
                rows.append((min(float(p[0]) for p in r[0]), str(r[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return [t.strip() for _, t in sorted(rows, key=lambda x: x[0]) if t.strip()]


def read_banner(frame: np.ndarray, calib: Calibration | dict | str, ocr) -> str:
    """The text of the opponent name line: crop `banner_box`, upscale (at least x2, to about 64 px high), recognise with
    RapidOCR (recognition only: the crop is one line). "" when the crop is empty or the OCR finds nothing or fails."""
    x0, y0, x1, y1 = banner_box(frame.shape, calib)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return ""
    if crop.ndim == 3 and crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    scale = min(MAX_SCALE, max(MIN_SCALE, TARGET_H / crop.shape[0]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    fn = _engine(ocr)
    try:
        try:
            res, _ = fn(crop, use_det=False, use_cls=False, use_rec=True)
        except TypeError:                            # an engine without the stage switches: detect + recognise
            res, _ = fn(crop)
    except Exception:                                # noqa: BLE001  a failed OCR pass is "no text" (fail closed)
        return ""
    return " ".join(_texts(res))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert / delete / substitute, cost 1 each)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_trainer(text: str | None) -> bool:
    """True when a token of `text` is within edit distance 2 of "trainer" (OCR noise: "Traner", "TRAINER", "Trainer:",
    "Tralner"); a token glued to the name ("TrainerCheddar") counts when its first 7 letters are within 1. Never true on
    empty text. "Training" (distance 3) and "Tyler" are not trainers; a player called "Trainee" or "Rainer" would be (the
    spec's tolerance), which only matters in a battle that also shows such a name for at least 2 checked frames."""
    for token in re.findall(r"[a-z]+", (text or "").lower()):
        if len(token) < MIN_TOKEN:
            continue
        if edit_distance(token, WORD) <= MAX_EDITS:
            return True
        if len(token) > len(WORD) and edit_distance(token[:len(WORD)], WORD) <= 1:
            return True
    return False


def verdict(checks: list[bool]) -> bool:
    """Training Camp when at least MIN_TRAINER_FRAMES of the checked frames read a trainer banner."""
    return sum(bool(c) for c in checks) >= MIN_TRAINER_FRAMES
