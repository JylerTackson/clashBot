#!/usr/bin/env python3
"""Build the Clashbot class-showcase deck (docs/report/slides/clashbot_showcase.pptx).

Re-runnable: every number comes from docs/data/*.json (preferring *_v2 when present) or, if the
experiment has not finished yet, from the PENDING dict below and is drawn in RED so it is easy to
spot. When a result file lands in docs/data/ the matching loader fills the number (black) on the
next run. Final hand-typed numbers go into FINAL (black) / TRIALS.

    python build_slides.py                      # build + verify + write NOTES.md
    python build_slides.py --stats-audit PATH   # use another stats_audit_v1.json (tests)

Needs python-pptx, matplotlib, Pillow (no project venv needed; see NOTES.md). Fonts: Arial.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageFont
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# ======================================================================================
# 1. NUMBERS NOT YET AVAILABLE.  Red in the deck.  Replace by editing FINAL / TRIALS, or
#    just drop the result file into docs/data/ (loaders below pick it up on the next run).
# ======================================================================================
PENDING = {
    # live Training Camp trials (user is playing them): set TRIALS below when done
    "trials_score": "n/10",
    "trials_n": "10",
    "trials_ci": "95% CI pending",
    # bot vs pros on their own videos  <- docs/data/vs_pro_v1.json (MAC-AQ)
    "vs_pro_games": "N",
    "vs_pro_recall": "xx%",
    "vs_pro_precision": "xx%",
    "vs_pro_card_bot": "xx%",
    "vs_pro_card_random": "xx%",
    "vs_pro_card_placement": "xx%",
    "vs_pro_card_kg": "xx%",
    "vs_pro_lane_bot": "xx%",
    "vs_pro_lane_random": "xx%",
    "vs_pro_lane_placement": "xx%",
    "vs_pro_lane_kg": "n/a",
    # held-out mined answer rules  <- docs/data/answer_rules_v1.json (MAC-AR, E6)
    "rules_hit3": "0.xx",
    "rules_hit3_wiki": "0.xx",
    "rules_hit3_pop": "0.xx",
    # context-conflict test of d1  <- docs/data/context_conflict_v1.json (MAC-AR, E2)
    "cc_c0_acc": "0.xx",
    "cc_c2_acc": "0.xx",
    "cc_c3_acc": "0.xx",
    "cc_c2_follow": "xx%",
    "cc_c3_follow": "xx%",
    # CIs on the ranker bar chart  <- docs/data/stats_audit_v1.json (MAC-AP, E1)
    "ci_note": "95% CI whiskers pending (stats_audit_v1)",
    # per-stage latency, mean over 3 runs  <- docs/data/latency_v2.json (MAC-AP)
    "latency_note": "3-run mean (latency_v2) pending",
}
# Numbers confirmed by hand (rendered black, win over PENDING, lose to data files):
FINAL: dict[str, str] = {}
# Training Camp trials: set wins and n when the trials are done -> "k/n (95% CI a-b%)" everywhere.
TRIALS = {"wins": None, "n": None}

DECK_DATE = "2026-11-09"
PRESENTER = "David Tyler Jackson"
COURSE = "CAP 6938 AI Knowledge Systems"
SCHOOL = "University of Central Florida"
INSTRUCTOR = "Dr. Ser Nam Lim"

# ======================================================================================
# 2. Paths, palette, geometry
# ======================================================================================
HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1]
DATA = DOCS / "data"
ASSETS = HERE / "assets"
LOGOS = ASSETS / "logos"
CHARTS = ASSETS / "charts"
SHOTS = ASSETS / "screenshots"
OUT_PPTX = HERE / "clashbot_showcase.pptx"
OUT_NOTES = HERE / "NOTES.md"
SCREENSHOT = SHOTS / "dashboard_viz6.png"
LOGO_UCF_SRC = LOGOS / "ucf_UCF-LOGO-Stacked-Alt-SingleBlack-330x280-1.png"
LOGO_CECS_SRC = LOGOS / "cecs_College-of-Engineering-and-Computer-Science-RGB-Bold-Gold-White-Text.png"
# the deck embeds RGBA copies (the UCF original is an 8-bit colormap PNG); originals stay untouched
LOGO_UCF = LOGOS / "ucf_logo_black_rgba.png"
LOGO_CECS = LOGOS / "cecs_logo_rgba.png"
LOGO_SOURCES = [
    "https://www.ucf.edu/wp-content/blogs.dir/34/files/2026/08/UCF-LOGO-Stacked-Alt-SingleBlack-330x280-1.png"
    "  (UCF brand site, https://www.ucf.edu/brand/logo-and-identity/)",
    "https://www.cecs.ucf.edu/web/wp-content/uploads/2026/10/College-of-Engineering-and-Computer-Science-RGB-Bold-Gold-White-Text.png"
    "  (CECS site header, https://www.cecs.ucf.edu/)",
]

GOLD = "FFC904"
BLACK = "000000"
INK = "111111"
GRAY = "5F5F5F"
MID = "9A9A9A"
LIGHT = "F2F2F2"
RULE = "D9D9D9"
PALEGOLD = "FFE680"
RED = "D00000"
WHITE = "FFFFFF"

SW, SH = 13.333, 7.5
FOOT_Y, FOOT_H = 6.75, 0.75
BODY_TOP, BODY_BOT = 1.28, 6.65
FONT = "Arial"
MIN_BODY, MIN_TITLE = 14, 28

NUM: dict[str, str] = {}  # headline accuracies, filled from the data files in main() (used in slide text and notes)
USED_PENDING: dict[str, set[int]] = {}
WARNINGS: list[str] = []
VALUES: dict[str, str] = {}  # data-file driven values (black)
_CUR_SLIDE = [0]


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print("WARNING:", msg, file=sys.stderr)


# ======================================================================================
# 3. Data loaders (docs/data)
# ======================================================================================
def jload(name: str):
    p = DATA / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:  # pragma: no cover
        warn(f"could not parse {p.name}: {e}")
        return None


def jpick(base: str):
    """Prefer docs/data/<base>_v2.json, else _v1. Returns (data, filename or None)."""
    for suffix in ("_v2", "_v1"):
        d = jload(base + suffix + ".json")
        if d is not None:
            return d, base + suffix + ".json"
    warn(f"no docs/data/{base}_v[12].json")
    return None, None


def flatten(o, prefix=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = o
    return out


def flat_find(flat: dict, *patterns: str):
    """Return the single leaf whose path matches ALL regexes; None (with a warning) if 0 or >1."""
    hits = [k for k in flat if all(re.search(p, k, re.I) for p in patterns)]
    if len(hits) == 1 and isinstance(flat[hits[0]], (int, float)):
        return flat[hits[0]]
    if len(hits) != 1:
        warn(f"flat_find{patterns}: {len(hits)} matches -> left PENDING" + (f" e.g. {hits[:3]}" if hits else ""))
    return None


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}%"


def load_value_overrides(stats_path: Path | None):
    """Fill VALUES from result files that are still pending. Strict: ambiguous -> stays red."""
    # --- vs pro (MAC-AQ). Paths are guesses against the spec; adjust the regexes when the file lands.
    d = jload("vs_pro_v1.json")
    if d:
        f = flatten(d)
        n = flat_find(f, r"n_games$") or flat_find(f, r"^games$")
        if n is not None:
            VALUES["vs_pro_games"] = str(int(n))
        for key, pats in {
            "vs_pro_recall": [r"recall", r"(2s|pm2|2\.0)", r"(est|mean|value)$"],
            "vs_pro_precision": [r"precision", r"(2s|pm2|2\.0)", r"(est|mean|value)$"],
            "vs_pro_card_bot": [r"same_card", r"bot", r"(est|mean|value)$"],
            "vs_pro_card_random": [r"same_card", r"random", r"(est|mean|value)$"],
            "vs_pro_card_placement": [r"same_card", r"placement", r"(est|mean|value)$"],
            "vs_pro_card_kg": [r"same_card", r"(kg|counter)", r"(est|mean|value)$"],
            "vs_pro_lane_bot": [r"same_lane", r"bot", r"(est|mean|value)$"],
            "vs_pro_lane_random": [r"same_lane", r"random", r"(est|mean|value)$"],
            "vs_pro_lane_placement": [r"same_lane", r"placement", r"(est|mean|value)$"],
            "vs_pro_lane_kg": [r"same_lane", r"(kg|counter)", r"(est|mean|value)$"],
        }.items():
            v = flat_find(f, *pats)
            if v is not None:
                VALUES[key] = pct(v) if v <= 1.0 else f"{v:.0f}%"
    # --- held-out answer rules (MAC-AR E6)
    d = jload("answer_rules_v1.json")
    if d:
        f = flatten(d)
        for key, pats in {
            "rules_hit3": [r"test", r"full", r"hit@?3", r"(est|mean|value)$"],
            "rules_hit3_wiki": [r"test", r"wiki", r"hit@?3", r"(est|mean|value)$"],
            "rules_hit3_pop": [r"test", r"pop", r"hit@?3", r"(est|mean|value)$"],
        }.items():
            v = flat_find(f, *pats)
            if v is not None:
                VALUES[key] = f"{v:.2f}"
    # --- context conflict (MAC-AR E2)
    d = jload("context_conflict_v1.json")
    if d:
        f = flatten(d)
        for key, pats in {
            "cc_c0_acc": [r"test", r"c0", r"acc", r"(est|mean|value)$"],
            "cc_c2_acc": [r"test", r"c2", r"acc", r"(est|mean|value)$"],
            "cc_c3_acc": [r"test", r"c3", r"acc", r"(est|mean|value)$"],
            "cc_c2_follow": [r"test", r"c2", r"follow", r"(est|mean|value)$"],
            "cc_c3_follow": [r"test", r"c3", r"follow", r"(est|mean|value)$"],
        }.items():
            v = flat_find(f, *pats)
            if v is not None:
                VALUES[key] = f"{v:.2f}" if key.endswith("acc") else pct(v)


def load_stats_audit(path: Path | None) -> dict | None:
    """accuracy[split][model] = {est, lo, hi}  (schema of tools/eval/eval_stats.py)."""
    p = path or (DATA / "stats_audit_v1.json")
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        acc = d["accuracy"]["test"]
        for k in ("random", "popularity", "placement_prior", "lgbm", "d1_zero_shot", "d1_joint", "d1_tac", "d1_kg"):
            acc[k]["est"], acc[k]["lo"], acc[k]["hi"]
        return d
    except Exception as e:
        warn(f"{p.name} found but schema not recognised ({e!r}); using v1 numbers without CIs")
        return None


def latency_stages() -> tuple[list[tuple[str, float, float]], str, bool]:
    """Per-stage (p50, p95) ms. Prefers docs/data/latency_v2.json, else a recorded run's summary.json."""
    d = jload("latency_v2.json")
    if d:
        try:  # expected: stages -> name -> {p50, p95} (strict; adjust if the real schema differs)
            st = d["stages"]
            rows = [(k, float(st[k]["p50"]), float(st[k]["p95"])) for k in
                    ("capture", "perception", "live_state", "trigger", "candidates", "rank", "execute", "total")]
            return rows, "docs/data/latency_v2.json", True
        except Exception as e:
            warn(f"latency_v2.json schema not recognised ({e!r}); using run 20261107-111554")
    run = DOCS.parent / "runs" / "poc" / "20261107-111554" / "summary.json"
    if run.exists():
        s = json.loads(run.read_text())["decision_ms"]
        rows = [(k, s[k]["p50"], s[k]["p95"]) for k in
                ("capture", "perception", "live_state", "trigger", "candidates", "rank", "execute", "total")]
        return rows, "run 20261107-111554 summary.json", False
    # copied from docs/research/course_mapping_audit.md section C7 (same run)
    rows = [("capture", 21.2, 32.1), ("perception", 102.4, 230.0), ("live_state", 0.13, 0.33),
            ("trigger", 1.5, 8.7), ("candidates", 26.1, 56.9), ("rank", 0.0, 136.7),
            ("execute", 183.3, 188.1), ("total", 338.9, 571.8)]
    return rows, "course_mapping_audit.md C7 (run 20261107-111554)", False


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    def ge(p):  # P(X >= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def le(p):  # P(X <= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))

    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            (b, a) = (m, a) if ge(m) > alpha / 2 else (b, m)
        lo = (a + b) / 2
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            (a, b) = (m, b) if le(m) > alpha / 2 else (a, m)
        hi = (a + b) / 2
    return lo, hi


def trials_text() -> tuple[str, bool]:
    """('k/n (95% CI a-b%)', pending?)"""
    k, n = TRIALS["wins"], TRIALS["n"]
    if k is None or n is None:
        return PENDING["trials_score"], True
    lo, hi = clopper_pearson(int(k), int(n))
    return f"{k}/{n} (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)", False


def resolve(key: str) -> tuple[str, bool]:
    """-> (text, is_pending). Order: data file VALUES, FINAL/TRIALS, PENDING (red)."""
    if key == "trials_score":
        txt, pend = trials_text()
    elif key == "trials_n":
        txt, pend = (str(TRIALS["n"]), False) if TRIALS["n"] is not None else (PENDING["trials_n"], True)
    elif key == "trials_ci":
        if TRIALS["wins"] is not None and TRIALS["n"] is not None:
            lo, hi = clopper_pearson(int(TRIALS["wins"]), int(TRIALS["n"]))
            txt, pend = f"95% CI {100 * lo:.0f}-{100 * hi:.0f}%", False
        else:
            txt, pend = PENDING[key], True
    elif key in VALUES:
        txt, pend = VALUES[key], False
    elif key in FINAL:
        txt, pend = FINAL[key], False
    elif key in PENDING:
        txt, pend = PENDING[key], True
    else:
        raise KeyError(f"unknown token {{p:{key}}}")
    if pend:
        USED_PENDING.setdefault(key, set()).add(_CUR_SLIDE[0])
    return txt, pend


# ======================================================================================
# 4. Text measuring (PIL) used both to size table rows and to verify nothing overflows
# ======================================================================================
_FONT_FILES = {
    False: ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    True: ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
           "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
}
_font_cache: dict = {}
LINE_FACTOR = 1.2


def _font(bold: bool):
    if bold not in _font_cache:
        f = None
        for path in _FONT_FILES[bold]:
            if Path(path).exists():
                f = path
                break
        _font_cache[bold] = f
    return _font_cache[bold]


def text_width_pt(text: str, size: float, bold: bool) -> float:
    path = _font(bold)
    if path is None:
        return len(text) * size * 0.55
    key = (path, 100)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, 100)
    return _font_cache[key].getlength(text) * size / 100.0


def plain(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return re.sub(r"\{p:([A-Za-z0-9_]+)\}", lambda m: _peek(m.group(1)), text)


def _peek(key: str) -> str:
    if key == "trials_score":
        return trials_text()[0]
    if key == "trials_n":
        return str(TRIALS["n"]) if TRIALS["n"] is not None else PENDING["trials_n"]
    if key == "trials_ci":
        return PENDING[key] if TRIALS["wins"] is None else "95% CI 00-00%"
    return VALUES.get(key) or FINAL.get(key) or PENDING.get(key, "?")


def count_lines(text: str, size: float, bold: bool, width_in: float) -> int:
    width_pt = max(width_in, 0.2) * 72.0
    total = 0
    for para in text.split("\n"):
        words = para.split(" ")
        line, n = "", 1
        for w in words:
            trial = (line + " " + w) if line else w
            if text_width_pt(trial, size, bold) <= width_pt:
                line = trial
            else:
                if line:
                    n += 1
                line = w
        total += n
    return total


# ======================================================================================
# 5. python-pptx helpers
# ======================================================================================
def I(v: float) -> Emu:  # inches -> EMU
    return Emu(int(round(v * 914400)))


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def _strip_style(shape) -> None:
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def box(slide, x, y, w, h, fill=None, line=None, lw=1.0, shape=MSO_SHAPE.RECTANGLE, name=None, radius=None, dash=False):
    s = slide.shapes.add_shape(shape, I(x), I(y), I(w), I(h))
    _strip_style(s)
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = MSO_LINE.DASH
    else:
        s.line.fill.background()
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if name:
        s.name = name
    return s


MIN_EXT = 12700 / 914400.0  # 1 pt: never emit a zero-extent line


def arrow(slide, x1, y1, x2, y2, color=BLACK, lw=2.5, head=True, dash=False, name=None):
    if abs(x2 - x1) < MIN_EXT:  # keep the line centred on the requested x, give it a 1 pt bounding box
        x1, x2 = x1 - MIN_EXT / 2, x2 + MIN_EXT / 2
    if abs(y2 - y1) < MIN_EXT:
        y1, y2 = y1 - MIN_EXT / 2, y2 + MIN_EXT / 2
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1), I(x2), I(y2))
    _strip_style(c)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(lw)
    if dash:
        c.line.dash_style = MSO_LINE.DASH
    if head:
        ln = c.line._get_or_add_ln()
        t = etree.SubElement(ln, qn("a:tailEnd"))
        t.set("type", "triangle")
        t.set("w", "med")
        t.set("len", "med")
    if name:
        c.name = name
    return c


TOKEN = re.compile(r"(\*\*.+?\*\*|\{p:[A-Za-z0-9_]+\})")


def _set_run(r, size, color, bold, italic=False):
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = rgb(color)


def fill_par(p, text: str, size: float, color=INK, bold=False, italic=False):
    if "\n" in text:  # explicit line breaks -> <a:br/>
        for k, piece in enumerate(text.split("\n")):
            if k:
                p.add_line_break()
            fill_par(p, piece, size, color, bold, italic)
        return
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run()
            r.text = part[2:-2]
            _set_run(r, size, color, True, italic)
        elif part.startswith("{p:"):
            val, pend = resolve(part[3:-1])
            r = p.add_run()
            r.text = val
            _set_run(r, size, RED if pend else color, True if pend else bold, italic)
        else:
            r = p.add_run()
            r.text = part
            _set_run(r, size, color, bold, italic)


def _bullet(p, level_in=0.25, color=GOLD, char="■"):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(level_in * 914400)))
    pPr.set("indent", str(-int(level_in * 914400)))
    for tag in ("a:buClr", "a:buFont", "a:buChar", "a:buNone"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    bc = etree.SubElement(pPr, qn("a:buClr"))
    c = etree.SubElement(bc, qn("a:srgbClr"))
    c.set("val", color)
    bs = etree.SubElement(pPr, qn("a:buSzPct"))
    bs.set("val", "70000")
    bf = etree.SubElement(pPr, qn("a:buFont"))
    bf.set("typeface", "Arial")
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", char)


def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         name=None, inset=(0.05, 0.03), shape=None):
    """Text box (or write into `shape`). paras: str | dict(t, size, bold, color, bullet, after, align, italic)."""
    if shape is None:
        shape = slide.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = I(inset[0])
    tf.margin_top = tf.margin_bottom = I(inset[1])
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    first = True
    for item in paras:
        d = {"t": item} if isinstance(item, str) else dict(item)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = d.get("align", align)
        p.space_after = Pt(d.get("after", 4))
        p.space_before = Pt(0)
        p.line_spacing = 1.0
        if d.get("bullet"):
            _bullet(p, color=d.get("bcolor", GOLD))
        fill_par(p, d["t"], d.get("size", size), d.get("color", color), d.get("bold", bold), d.get("italic", False))
    if name:
        shape.name = name
    return shape


WITH_NOTES = False  # --with-notes: python-pptx's notes master makes macOS Quick Look / Keynote refuse the file


def add_notes(slide, txt: str):
    if WITH_NOTES:
        slide.notes_slide.notes_text_frame.text = txt


# ---- tables -----------------------------------------------------------------------
def _style_cell(cell, fill, border_color=RULE, border_w=0.75, margins=(0.08, 0.04)):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for ch in list(tcPr):
        tcPr.remove(ch)
    tcPr.set("marL", str(int(margins[0] * 914400)))
    tcPr.set("marR", str(int(margins[0] * 914400)))
    tcPr.set("marT", str(int(margins[1] * 914400)))
    tcPr.set("marB", str(int(margins[1] * 914400)))
    tcPr.set("anchor", "ctr")
    for tag in ("a:lnL", "a:lnR", "a:lnT"):
        ln = etree.SubElement(tcPr, qn(tag))
        ln.set("w", "0")
        etree.SubElement(ln, qn("a:noFill"))
    lnB = etree.SubElement(tcPr, qn("a:lnB"))
    lnB.set("w", str(int(border_w * 12700)))
    sf = etree.SubElement(lnB, qn("a:solidFill"))
    c = etree.SubElement(sf, qn("a:srgbClr"))
    c.set("val", border_color)
    if fill:
        sf2 = etree.SubElement(tcPr, qn("a:solidFill"))
        c2 = etree.SubElement(sf2, qn("a:srgbClr"))
        c2.set("val", fill)
    else:
        etree.SubElement(tcPr, qn("a:noFill"))


def table(slide, x, y, col_w, rows, size=14, header=True, row_min=0.38, zebra=True, name=None, aligns=None,
          first_col_bold=False, hl_rows=(), hdr_fill=GOLD, hdr_color=BLACK, pad=0.04):
    """rows: list of list[str] (markup ok, \\n for line breaks). Returns (shape, bottom_y)."""
    nr, nc = len(rows), len(col_w)
    gs = slide.shapes.add_table(nr, nc, I(x), I(y), I(sum(col_w)), I(row_min * nr))
    tbl = gs.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow", "firstCol", "lastRow", "lastCol", "bandCol"):
        tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:  # python-pptx adds the blue "Medium Style 2" id; drop it (all styling is explicit)
        tblPr.remove(sid)
    for j, w in enumerate(col_w):
        tbl.columns[j].width = I(w)
    total = 0.0
    for i, row in enumerate(rows):
        is_hdr = header and i == 0
        need = row_min
        for j, txt in enumerate(row):
            cell = tbl.cell(i, j)
            bold = is_hdr or (first_col_bold and j == 0)
            fill = hdr_fill if is_hdr else (LIGHT if (zebra and i % 2 == 0) else None)
            if i in hl_rows:
                fill = PALEGOLD
            _style_cell(cell, fill, border_color=(BLACK if is_hdr else RULE), border_w=(1.5 if is_hdr else 0.75),
                        margins=(0.08, pad))
            tf = cell.text_frame
            tf.word_wrap = True
            lines = txt.split("\n")
            for k, ln in enumerate(lines):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                p.space_after = Pt(0)
                p.space_before = Pt(0)
                p.line_spacing = 1.0
                al = (aligns[j] if aligns else PP_ALIGN.LEFT)
                p.alignment = al
                fill_par(p, ln, size, hdr_color if is_hdr else INK, bold)
            nl = count_lines(plain(txt), size, bold, col_w[j] - 0.16 - 0.02)
            need = max(need, nl * size * LINE_FACTOR / 72.0 + 2 * pad + 0.02)
        tbl.rows[i].height = I(need)
        total += need
    gs.height = I(total)
    if name:
        gs.name = name
    return gs, y + total


# ---- slide chrome -----------------------------------------------------------------
def add_footer(slide, n: int):
    box(slide, 0, FOOT_Y, SW, FOOT_H, fill=GOLD, name="Footer bar")
    # UCF logo (black mark on the gold bar)
    lh = 0.62
    if LOGO_UCF.exists():
        lw = lh * 330 / 280
        pic = slide.shapes.add_picture(str(LOGO_UCF), I(0.3), I(FOOT_Y + (FOOT_H - lh) / 2), height=I(lh))
        pic.name = "Logo UCF"
    else:
        ph = box(slide, 0.3, FOOT_Y + 0.07, 0.9, 0.6, fill=None, line=BLACK, lw=1.5, dash=True, name="Logo UCF placeholder")
        text(slide, 0, 0, 0, 0, ["UCF LOGO"], size=14, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=ph)
    text(slide, 1.35, FOOT_Y, 6.6, FOOT_H, [f"Clashbot  |  {COURSE}"], size=14, color=BLACK, bold=True,
         anchor=MSO_ANCHOR.MIDDLE, name="Footer text")
    # slide number (real slide-number field) in a black disc
    d = 0.56
    disc = box(slide, 9.95, FOOT_Y + (FOOT_H - d) / 2, d, d, fill=BLACK, shape=MSO_SHAPE.OVAL, name="Slide number disc")
    nb = slide.shapes.add_textbox(I(9.95), I(FOOT_Y + (FOOT_H - d) / 2), I(d), I(d))
    nb.name = "Slide number"
    tf = nb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    fld = etree.SubElement(p._p, qn("a:fld"))
    fld.set("id", "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}")
    fld.set("type", "slidenum")
    rPr = etree.SubElement(fld, qn("a:rPr"))
    rPr.set("lang", "en-US")
    rPr.set("sz", "1600")
    rPr.set("b", "1")
    sf = etree.SubElement(rPr, qn("a:solidFill"))
    etree.SubElement(sf, qn("a:srgbClr")).set("val", GOLD)
    etree.SubElement(rPr, qn("a:latin")).set("typeface", FONT)
    etree.SubElement(fld, qn("a:t")).text = str(n)
    # CECS logo: white-text logo on a black chip so it reads on the gold bar
    ch, pich = 0.62, 0.50
    pw = pich * 1704 / 389
    cw = pw + 0.30
    cx = SW - 0.3 - cw
    cy = FOOT_Y + (FOOT_H - ch) / 2
    chip = box(slide, cx, cy, cw, ch, fill=BLACK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12, name="Logo CECS chip")
    if LOGO_CECS.exists():
        pic = slide.shapes.add_picture(str(LOGO_CECS), I(cx + 0.15), I(cy + (ch - pich) / 2), height=I(pich))
        pic.name = "Logo CECS"
    else:
        text(slide, 0, 0, 0, 0, ["CECS LOGO (placeholder)"], size=14, color=GOLD, bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, shape=chip)
        chip.name = "Logo CECS placeholder"


def new_slide(prs, n: int, title: str | None, notes: str = ""):
    _CUR_SLIDE[0] = n
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(WHITE)
    if title is not None:
        box(s, 0, 0.32, 0.26, 0.78, fill=GOLD, name="Title accent")
        text(s, 0.5, 0.27, 12.4, 0.88, [title], size=30, bold=True, anchor=MSO_ANCHOR.MIDDLE, name="Title")
    add_footer(s, n)
    add_notes(s, script_text(n))
    return s


def picture(slide, path: Path, x, y, w=None, h=None, name=None, border=None):
    kw = {}
    if w is not None:
        kw["width"] = I(w)
    if h is not None:
        kw["height"] = I(h)
    pic = slide.shapes.add_picture(str(path), I(x), I(y), **kw)
    if border:
        pic.line.color.rgb = rgb(border)
        pic.line.width = Pt(1.25)
    if name:
        pic.name = name
    return pic


def tile(slide, x, y, w, h, big, small, big_size=26, small_size=14, fill=LIGHT, name=None, big_color=BLACK):
    box(slide, x, y, 0.09, h, fill=GOLD, name=(name or "Tile") + " accent")
    bx = box(slide, x + 0.09, y, w - 0.09, h, fill=fill, name=name or "Tile")
    text(slide, 0, 0, 0, 0, [{"t": big, "size": big_size, "bold": True, "color": big_color, "after": 2},
                             {"t": small, "size": small_size, "color": GRAY, "after": 0}],
         anchor=MSO_ANCHOR.MIDDLE, shape=bx, inset=(0.15, 0.04))
    return bx


def badge(slide, x, y, num: int, d=0.36, size=16):
    c = box(slide, x, y, d, d, fill=GOLD, line=BLACK, lw=1.5, shape=MSO_SHAPE.OVAL, name=f"Badge {num}")
    text(slide, 0, 0, 0, 0, [str(num)], size=size, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
         shape=c, inset=(0, 0))
    return c


def caption(slide, x, y, w, t, size=14, color=GRAY, h=0.34, bold=False):
    return text(slide, x, y, w, h, [{"t": t, "size": size, "color": color, "bold": bold, "after": 0}], name="Caption")


# ======================================================================================
# 6. Charts (matplotlib, 200 dpi, figure size = placement size so 14 pt means 14 pt)
# ======================================================================================
def make_rgba_logos():
    """Normalise the downloaded logos to plain 8-bit RGBA PNGs (no palette) for maximum viewer compatibility."""
    for src, dst in ((LOGO_UCF_SRC, LOGO_UCF), (LOGO_CECS_SRC, LOGO_CECS)):
        if src.exists():
            Image.open(src).convert("RGBA").save(dst, format="PNG")


def setup_mpl():
    for path in _FONT_FILES[False] + _FONT_FILES[True]:
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass
    have = {f.name for f in fm.fontManager.ttflist}
    family = next((f for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans") if f in have), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": [family],
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
        "axes.linewidth": 1.2, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "xtick.color": "#222222", "ytick.color": "#222222", "text.color": "#111111", "axes.labelcolor": "#222222",
    })


C_GOLD, C_BLACK, C_GRAY, C_LGRAY, C_PGOLD = "#FFC904", "#000000", "#6B6B6B", "#C4C4C4", "#FFE680"


def _save(fig, name: str) -> Path:
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / name
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def ranker_rows(stats: dict | None) -> list[dict]:
    """Test-split 4-way accuracy rows. Prefer stats_audit (est + CI), else the v1/v2 eval JSONs."""
    order = [("random", "Random"), ("popularity", "Popularity"), ("placement_prior", "Placement prior"),
             ("lgbm", "LightGBM (tuned)"), ("d1_zero_shot", "d1 zero-shot"), ("d1_joint", "d1 fine-tuned"),
             ("d1_tac", "d1 + tactics (live)"), ("d1_kg", "d1 + tactics + KG lines")]
    kinds = {"random": "base", "popularity": "base", "placement_prior": "base", "lgbm": "lgbm",
             "d1_zero_shot": "zero", "d1_joint": "ft", "d1_tac": "live", "d1_kg": "ft"}
    rows = []
    if stats:
        acc = stats["accuracy"]["test"]
        for k, lab in order:
            a = acc[k]
            rows.append(dict(key=k, label=lab, est=a["est"], lo=a["lo"], hi=a["hi"], kind=kinds[k], src="stats_audit_v1"))
        return rows
    base, _ = jpick("ranker_baselines")
    fr, _ = jpick("feature_ranker")
    zs, _ = jpick("d1_zero_shot")
    j0, _ = jpick("d1_finetune_joint_eval")
    jt, _ = jpick("d1_finetune_joint_tac_eval")
    jk, _ = jpick("d1_finetune_joint_kg_eval")
    vals = {
        "random": base["ranker"]["test"]["random_expected"]["acc"],
        "popularity": base["ranker"]["test"]["popularity"]["acc"],
        "placement_prior": base["ranker"]["test"]["placement_prior"]["acc"],
        "lgbm": fr["models"]["lgbm_lambdarank"]["test_acc_mean"],
        "d1_zero_shot": zs["variants"]["joint"]["test"]["acc"],
        "d1_joint": j0["variants"]["joint"]["test"]["acc"],
        "d1_tac": jt["variants"]["joint"]["test"]["acc"],
        "d1_kg": jk["variants"]["joint"]["test"]["acc"],
    }
    for k, lab in order:
        rows.append(dict(key=k, label=lab, est=vals[k], lo=None, hi=None, kind=kinds[k], src="docs/data v1/v2 json"))
    return rows


def chart_ranker(rows, w, h) -> Path:
    colors = {"base": C_LGRAY, "lgbm": "#505050", "zero": C_PGOLD, "ft": C_GOLD, "live": C_GOLD}
    fig, ax = plt.subplots(figsize=(w, h))
    n = len(rows)
    ys = list(range(n))[::-1]
    for y, r in zip(ys, rows):
        ax.barh(y, r["est"], color=colors[r["kind"]], edgecolor=C_BLACK, linewidth=2.6 if r["kind"] == "live" else 1.2, height=0.68)
        end = r["est"]
        if r["lo"] is not None:
            ax.errorbar(r["est"], y, xerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], color=C_BLACK, capsize=5, lw=1.8, capthick=1.8)
            end = r["hi"]
        ax.text(end + 0.012, y, f"{r['est']:.2f}", va="center", ha="left", fontsize=14, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1.2), zorder=5)
    ax.axvline(0.25, color=C_GRAY, ls="--", lw=1.5)
    ax.text(0.253, n - 0.45, "chance 0.25", fontsize=13, color=C_GRAY, va="bottom", ha="left")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=14)
    ax.set_xlim(0, 0.72)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_xlabel("4-way accuracy, test split (n = 766)", fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    fig.tight_layout(pad=0.4)
    return _save(fig, "ranker_accuracy.png")


def chart_hnsw(hnsw_all: list[dict], w, h) -> Path:
    hnsw = [r for r in hnsw_all if isinstance(r["ef"], int)]  # the file's last row is the exact search
    fig, ax = plt.subplots(figsize=(w, h))
    xs = [r["ms_p50"] for r in hnsw]
    ys = [r["recall@10"] for r in hnsw]
    ax.plot(xs, ys, "-o", color=C_BLACK, lw=2.2, ms=8, mfc=C_GOLD, mec=C_BLACK, mew=1.8, zorder=3)
    for r in hnsw:
        dy = 0.0075 if r["ef"] not in (320,) else -0.012
        ha = "center"
        ax.annotate(f"ef {r['ef']}", (r["ms_p50"], r["recall@10"]), textcoords="offset points",
                    xytext=(0, 9 if r["ef"] != 320 else -20), ha=ha, fontsize=12, color="#222222")
    ex = [r for r in hnsw_all if not isinstance(r["ef"], int)]
    ex_ms = ex[0]["ms_p50"] if ex else 1.506
    ax.plot([ex_ms], [1.0], marker="*", ms=20, color=C_GOLD, mec=C_BLACK, mew=1.6, zorder=4, ls="none")
    ax.annotate("exact", (ex_ms, 1.0), textcoords="offset points", xytext=(12, -4), ha="left", va="center", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 2.5)
    ax.set_ylim(0.87, 1.015)
    ax.minorticks_off()
    ax.set_xticks([0.1, 0.3, 1, 2])
    ax.set_xticklabels(["0.1", "0.3", "1", "2"])
    ax.set_xlabel("ms per query (p50, log scale)", fontsize=14)
    ax.set_ylabel("recall@10 vs exact", fontsize=14)
    ax.grid(axis="y", color="#E3E3E3", lw=1)
    fig.tight_layout(pad=0.4)
    return _save(fig, "hnsw_recall_latency.png")


def chart_filters(filt: dict, w, h) -> Path:
    import numpy as np
    groups = [("deck filter", "deck"), ("phase filter", "phase")]
    ks = ["post@10", "post@50", "post@200"]
    cols = [C_BLACK, C_GOLD, C_PGOLD]
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(groups))
    bw = 0.25
    for i, k in enumerate(ks):
        vals = [filt[g][k]["empty_rate"] for _, g in groups]
        bars = ax.bar(x + (i - 1) * bw, vals, bw * 0.92, color=cols[i], edgecolor=C_BLACK, lw=1.2, label=f"top-K = {k.split('@')[1]}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{100 * v:.0f}%" if v >= 0.01 else f"{100 * v:.1f}%",
                    ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("queries with no result", fontsize=14)
    ax.legend(frameon=False, fontsize=13, loc="upper left", title="post-filter, ANN depth", title_fontsize=13)
    ax.grid(axis="y", color="#E3E3E3", lw=1)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    return _save(fig, "filter_emptiness.png")


def chart_lanesim(w, h) -> Path:
    # Source: docs/mac/STATUS.md 2026-11-06 (MAC-AH): lone L15 balloon, tower HP damage by answer.
    items = [("No answer", 3162, C_BLACK), ("Musketeer behind the tower", 351, C_GRAY), ("Tesla pull", 0, C_GOLD)]
    fig, ax = plt.subplots(figsize=(w, h))
    for i, (lab, v, c) in enumerate(items[::-1]):
        ax.barh(i, max(v, 8), color=c, edgecolor=C_BLACK, lw=1.2, height=0.62)
        ax.text(max(v, 8) + 70, i, f"{v:,}", va="center", fontsize=15, fontweight="bold")
    ax.set_yticks(range(3))
    ax.set_yticklabels([x[0] for x in items[::-1]], fontsize=14)
    ax.set_xlim(0, 3900)
    ax.set_xlabel("tower HP lost to a lone L15 balloon", fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    fig.tight_layout(pad=0.4)
    return _save(fig, "lane_simulation.png")


KG2_RELATIONS = [  # relation, edges, new_or_changed_in_v2   (docs/research/course_mapping_audit.md C2)
    ("STOPS", 8464, True), ("CO_OCCURS", 5142, True), ("COUNTERS", 4562, False), ("SYNERGY", 3136, False),
    ("GAME_OWN_CARD", 1613, False), ("GAME_OPP_CARD", 1401, False), ("IN_DECK", 544, False),
    ("PLAYED_AFTER", 450, True), ("SIMILAR_ROLE", 365, True), ("ANSWERED_BY", 287, True), ("KILLS", 283, True),
    ("PULLS", 204, True), ("GAME_DECK", 181, False), ("SPAWNED_BY", 154, False), ("SUBSTITUTE", 141, False),
    ("HAS_ARCHETYPE", 111, False), ("EVOLUTION_OF", 42, False), ("HERO_OF", 16, False), ("HAS_ABILITY", 16, True),
]
assert sum(r[1] for r in KG2_RELATIONS) == 27112 and len(KG2_RELATIONS) == 19


def chart_kg(w, h) -> Path:
    fig, ax = plt.subplots(figsize=(w, h))
    n = len(KG2_RELATIONS)
    for i, (rel, cnt, new) in enumerate(KG2_RELATIONS):
        y = n - 1 - i
        ax.barh(y, cnt, color=C_GOLD if new else C_LGRAY, edgecolor=C_BLACK, lw=1.0, height=0.72)
        ax.text(cnt + 110, y, f"{cnt:,}", va="center", fontsize=12)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in KG2_RELATIONS][::-1], fontsize=12)
    ax.set_xlim(0, 10300)
    ax.set_xticks([0, 2500, 5000, 7500])
    ax.set_xticklabels(["0", "2.5K", "5K", "7.5K"], fontsize=12)
    ax.set_xlabel("edges per relation (KG v2, 27,112 in total)", fontsize=13)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C_GOLD, ec=C_BLACK, label="new or extended in v2"), Patch(fc=C_LGRAY, ec=C_BLACK, label="v1 relations")],
              frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout(pad=0.4)
    return _save(fig, "kg_relations.png")


def chart_latency(rows, w, h, note: str) -> Path:
    names = {"capture": "Capture", "perception": "Perception", "live_state": "Live state", "trigger": "When to act",
             "candidates": "Candidates", "rank": "d1 ranker", "execute": "Taps (2 clicks)", "total": "Total"}
    fig, ax = plt.subplots(figsize=(w, h))
    n = len(rows)
    bh = 0.36
    for i, (k, p50, p95) in enumerate(rows):
        y = n - 1 - i - (0.35 if k == "total" else 0)
        ax.barh(y + bh / 2, p50, height=bh, color=C_GOLD, edgecolor=C_BLACK, lw=1.0)
        ax.barh(y - bh / 2, p95, height=bh, color=C_BLACK, edgecolor=C_BLACK, lw=1.0)
        ax.text(p50 + 8, y + bh / 2, f"{p50:.0f}" if p50 >= 10 else f"{p50:.1f}", va="center", fontsize=12)
        ax.text(p95 + 8, y - bh / 2, f"{p95:.0f}" if p95 >= 10 else f"{p95:.1f}", va="center", fontsize=12, fontweight="bold")
    ys = [n - 1 - i - (0.35 if r[0] == "total" else 0) for i, r in enumerate(rows)]
    ax.set_yticks(ys)
    ax.set_yticklabels([names[r[0]] for r in rows], fontsize=14)
    ax.axvline(300, color=C_GRAY, ls="--", lw=1.5)
    ax.text(305, n - 0.25, "p95 target 300 ms", fontsize=12, color=C_GRAY, va="top")
    ax.set_xlim(0, 680)
    ax.set_xlabel("ms per decision", fontsize=14)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C_GOLD, ec=C_BLACK, label="p50"), Patch(fc=C_BLACK, ec=C_BLACK, label="p95")],
              frameon=False, fontsize=13, loc="center right", bbox_to_anchor=(1.0, 0.45))
    ax.tick_params(axis="x", labelsize=13)
    fig.tight_layout(pad=0.4)
    return _save(fig, "latency_by_stage.png")


# ---- screenshot crops -----------------------------------------------------------------
CROPS = {
    # name: (left, top, right, bottom) in the 1700x2000 dashboard screenshot
    "dash_top": (0, 395, 1700, 1122),       # screen + decision + retrieval panels
    "dash_kg": (25, 1135, 1035, 1905),      # knowledge-graph panel
    "dash_phone": (25, 449, 360, 968),      # the mirrored game screen
}


def make_crops() -> dict[str, tuple[Path, tuple[int, int]]]:
    out = {}
    if not SCREENSHOT.exists():
        warn(f"{SCREENSHOT} missing: demo slide gets a placeholder")
        return out
    im = Image.open(SCREENSHOT).convert("RGB")
    for name, box_ in CROPS.items():
        c = im.crop(box_)
        p = SHOTS / f"{name}.png"
        c.save(p)
        out[name] = (p, c.size)
    return out


# ======================================================================================
# 7. Talk script (seconds per slide, sum = 840 s = 14:00) -> speaker notes + NOTES.md
# ======================================================================================
SCRIPT: dict[int, tuple[int, str]] = {
    1: (30, "Good afternoon, I am Tyler Jackson. Clashbot is an agent that plays Clash Royale in real time, from pixels. "
            "The talk is about how the course's knowledge-system ideas (retrieval, knowledge graphs, RAG and recommendation, "
            "agents) fit together in one working system. I will show the architecture, the numbers, including the ones "
            "where a simple baseline matches the LLM, and a short demo."),
    2: (60, "The setting is deliberately hard. There is no game API: the iPhone is mirrored to a Mac, the bot sees only that "
            "window and acts only by clicking in it. A match is three minutes, elixir regenerates all the time, every play is "
            "irreversible, so decisions must come in well under a second; we are at 0.34 seconds median from frame to click. "
            "Perception is noisy: greyed cards read as unknown, and our live state agrees with the dataset builder at a feature "
            "cosine of only 0.67. The opponent's deck and hand are hidden, so we need prior knowledge, and that is where 197 "
            "pro games come in. A general LLM does not have that knowledge: the d1 model scores <<d1_zs>> on a four-way pick, below "
            "chance. Scope: Training Camp against the built-in trainer only, never ranked, in line with the game's terms of "
            "service; the executor needs two explicit switches and checks that the banner reads Trainer."),
    3: (100, "Here is the whole system. Top row is the fast loop, running on every frame at about ten frames per second. "
             "Capture grabs the mirrored window; perception is two YOLOv8 detectors from KataCR plus OCR and a Kalman tracker; "
             "that becomes a live state, the same representation we built the dataset with. A small gate decides whether to act "
             "at all: reactive rules plus a LightGBM model, because pros wait about 80 percent of the seconds. If we act, a "
             "tactical engine, which runs a lane simulation to see how much tower damage each answer saves, and the knowledge "
             "graph propose and justify moves. The candidate generator merges those with retrieval from the pro database: kNN "
             "plus BM25 fused with reciprocal rank fusion, plus KG counters, to at most four options. The fine-tuned d1 language "
             "model picks one of the four, and the executor taps, behind a double guard. The tap changes the game and the loop "
             "closes. The dashed box is the slow loop, an optional Qwen coach that re-plans every twenty seconds; it is off in "
             "the showcase configuration, so everything you will see is the fast loop. Gold boxes are where the course "
             "material lives."),
    4: (50, "The knowledge comes from 197 pro games in 33 videos by one creator, about twelve match-hours, 10,923 sampled "
            "states. We split by video, not by game, so no match leaks across train and test; that is the contamination "
            "lesson from the RAG lectures. The knowledge graph has two versions. Version 1 came from the wiki and knowledge "
            "base: 590 nodes. Version 2, which the live bot uses, has 840 nodes and 27,112 edges in 19 relations; the gold "
            "bars are what v2 added: lane-simulation edges such as STOPS and KILLS, deck co-occurrence, role transfer, and "
            "mined answer events, 3,620 cases where the opponent played X and a pro answered with Y."),
    5: (65, "Retrieval, lectures 8 to 10. The live path is hybrid: exact kNN over a 644-dimensional structured state "
            "vector plus BM25 over the state text, fused by reciprocal rank fusion with k equals 60. On held-out pro states the "
            "channels are within noise of each other, hit at ten between 0.14 and 0.17, so a learned text embedding that "
            "costs twenty extra milliseconds is not worth it and is off. Left chart: HNSW at ef 40 reaches recall 0.978 in "
            "0.31 ms against 1.5 ms exact, but at 10.9 thousand vectors exact search is already fast enough, so we do not "
            "use HNSW live. Right chart: filtering. A deck filter applied before the search leaves an empty pool for 89 "
            "percent of queries, and filtering after the ANN search at top-10 returns nothing for half of the deck queries "
            "and 87 percent of the phase queries; that is why the live path filters by legality, not by deck."),
    6: (60, "Knowledge graph, lectures 5 to 7. For link prediction I held out COUNTERS triples and compared four "
            "embedding models with equal tuning and three seeds. RotatE is the only one clearly above the popularity "
            "baseline: MRR 0.159 against 0.110; random is 0.022. DistMult is symmetric by construction, so it cannot "
            "represent a directed counter. Neuro-symbolic part: the lane simulation, right, gives the engine a number to "
            "reason with: a lone level-15 balloon deals 3,162 tower hit points if unanswered, 351 with a musketeer behind "
            "the tower, zero with a Tesla pulling it. Hard constraints filter options: card in hand, elixir, legal tile, "
            "no blind spells. And the mined answer rules, held out: " "[read the red numbers from the slide when the "
            "E6 result lands]. Perceived legality is noisy: the pros' own move passes our legality check only 54 to 60 "
            "percent of the time."),
    7: (100, "RAG plus recommendation, lectures 11 to 14, is the heart of the decision. It is a two-stage recommender: "
             "stage one retrieves at most four candidate plays, stage two is the language model picking one. To train and "
             "test it, each example is the pro's real play plus three hard negatives in a two-by-two grid: same card other "
             "placement, other card same placement, other card other placement. An earlier version of the negatives leaked "
             "the answer: a rule that looked only at the option keys scored 0.85. We found it, rebuilt the grid, and that rule "
             "is now 0.25 by construction. The chart is four-way accuracy on the test split, chance 0.25. Zero-shot d1 is at "
             "chance, <<d1_zs>>. Fine-tuned it reaches <<d1_ft>>, with the tactics notes <<d1_tac>>. A tuned LightGBM feature ranker reaches <<lgbm>>: "
             "the RecSys lesson that tuned simple baselines are competitive, measured on our data. With 766 test lines the "
             "standard error is 1.8 points, so these three are not separable. [If the CI whiskers are on the chart, say "
             "so.] The d1 variant with KG lines trained in was worse, it overfit: its KG lines on the train split came "
             "from the same pro games."),
    8: (65, "Agents, lectures 15 to 18. The language model is the decision maker, and the things around it act like tools "
            "that have already been called: the engine's tactics lines, a few KG facts, and notes from retrieved pro states, "
            "all rendered into d1's context. I built typed read-only tools and a ReAct planner too, but the Qwen planner made "
            "zero tool calls, so in the final system the tool results are precomputed. Execution is separate and guarded: two "
            "switches, a command-line flag plus an environment variable, the banner must read Trainer, the hand must be read, "
            "and failed taps are detected with a cooldown. Right top: knowledge editing, with a synthetic plus-one-elixir change "
            "on five cards: the versioned fact store updates answers and leaves unrelated ones alone; it shows the mechanism "
            "works, not that it generalises to a real patch. Right bottom: the context-conflict test, does d1 follow a wrong "
            "note? [Read the red numbers when E2 lands.]"),
    9: (50, "To measure decisions without playing, I run the full live stack on test-split pro videos, open loop, with a "
            "dry-run executor, and match the bot's would-be plays to what the pro did within plus or minus two seconds. "
            "Timing: how many pro plays did the bot also play near that moment, and how many of the bot's plays had a pro "
            "play nearby. Choice: on matched pairs, same card, same lane, against three baselines at the pros' play times: "
            "a random card from the perceived hand, the placement prior, and the KG's top counter. [Fill the red numbers "
            "from the slide.] Caveat: open loop and the pros' hidden information, so this is agreement, not quality."),
    10: (65, "Live results, Training Camp. The latency chart is one recorded run: perception is the biggest software stage, "
             "about 100 ms median; the two clicks take 183 ms by themselves; the decision totals 339 ms median and 572 ms at "
             "the 95th percentile. We miss our 300 ms p95 target once the click is counted, and I would rather say so. The "
             "bot has landed taps reliably, 95 of 100 in the best-documented run. Trial results: [Training Camp record "
             "from the red tile]. The first win came in overtime in game three of one run, and I will not call that a win "
             "rate: with ten games the 95 percent interval is very wide."),
    11: (50, "Two prior systems. KataCR, Wu et al., is an offline-RL transformer imitating one human, reporting five percent, "
             "1 of 20, against the built-in AI for its headline model and zero to ten percent across nine models; "
             "we reuse its detector weights. Claude Royale is a Claude agent that plays screenshots through a vision model on "
             "the ladder, about seven seconds per decision and no built-in-AI win rate. Ours differs by design: it carries "
             "a pro-play database, a knowledge graph and a simulation engine, and takes 0.34 seconds. The protocols differ in "
             "deck, opponent version and sample size, so read the win-rate row as context, not as a benchmark."),
    12: (105, "[Play the 60-second demo clip or switch to the live screen share.] While it plays, point at the dashboard "
              "numbers. One: the ten pro states most like this situation. Two: up to four candidate plays with d1's probability "
              "as the bar, here 88 against 12 percent. Three: the reason, from the tactics engine, here that a Tesla pull "
              "to the centre saves about 500 tower hit points. Four: the knowledge graph, the counters the candidate "
              "generator used and the facts given to d1. If anything breaks live, the recorded clip is the fallback."),
    13: (40, "What worked: knowledge plus fine-tuning moved a chance-level LLM to about <<d1_tac>>, and the guarded executor lands "
             "its taps. What I would be honest about: simple baselines match the LLM; perception noise caps everything, the pro's "
             "own move passes our legality check only 54 to 60 percent of the time and the hand is fully read in about a third "
             "of moments; and n is small, wins are counted in tens. Next: elixir management, since in the last run about a "
             "third of match frames had no affordable card; retrain with the KG-leak fixed; more Training Camp games with "
             "intervals. Thank you; questions."),
    14: (0, "Backup: concept map from the course to the system, with the numbers. Not presented."),
}
assert sum(v[0] for v in SCRIPT.values()) == 840


def script_text(n: int) -> str:
    """Talk text with <<key>> replaced by the current headline numbers (so the notes follow the data)."""
    return re.sub(r"<<(\w+)>>", lambda m: NUM.get(m.group(1), "?"), SCRIPT[n][1])


# ======================================================================================
# 8. Slides
# ======================================================================================
def make_presentation():
    """python-pptx's default template is 4:3 (type="screen4x3"); make it a plain 13.333 x 7.5 in deck."""
    prs = Presentation()
    old_w = prs.slide_width
    prs.slide_width = Emu(12192000)
    prs.slide_height = Emu(6858000)
    sz = prs.part._element.find(qn("p:sldSz"))
    if sz is not None and "type" in sz.attrib:
        del sz.attrib["type"]
    k = 12192000 / old_w  # scale the master/layout placeholders horizontally so new slides line up
    for holder in [prs.slide_master] + list(prs.slide_layouts):
        for sh_ in holder.shapes:
            if sh_.left is not None and sh_.width is not None:
                sh_.left, sh_.width = Emu(int(sh_.left * k)), Emu(int(sh_.width * k))
    return prs


def slide_title(prs, n, crops):
    s = new_slide(prs, n, None, SCRIPT[n][1])
    box(s, 0, 0, 0.42, FOOT_Y, fill=GOLD, name="Title stripe")
    text(s, 0.95, 1.1, 8.0, 2.65, [{"t": "Clashbot: a knowledge-grounded real-time agent for Clash Royale", "size": 36, "bold": True, "after": 0}],
         anchor=MSO_ANCHOR.TOP, name="Cover title")
    box(s, 1.0, 3.3, 2.2, 0.1, fill=GOLD, name="Title rule")
    text(s, 0.95, 3.6, 7.7, 2.6, [
        {"t": PRESENTER, "size": 24, "bold": True, "after": 6},
        {"t": f"{COURSE}", "size": 18, "after": 2},
        {"t": f"{SCHOOL}", "size": 18, "after": 2},
        {"t": f"Instructor: {INSTRUCTOR}", "size": 18, "color": GRAY, "after": 2},
        {"t": f"Class showcase  |  {DECK_DATE}", "size": 18, "color": GRAY, "after": 0},
    ], name="Presenter block")
    if "dash_phone" in crops:
        pth, (cw_, ch_) = crops["dash_phone"]
        h = 5.0
        picture(s, pth, 9.45, 0.8, h=h, name="Title screenshot", border=BLACK)
        caption(s, 9.0, 5.85, 4.2, "Live capture of the mirrored game, tap overlay drawn", size=14, h=0.6)
    return s


def slide_problem(prs, n):
    s = new_slide(prs, n, "The problem: real-time play from pixels", SCRIPT[n][1])
    text(s, 0.5, 1.35, 6.9, 5.2, [
        {"t": "**Non-embedded.** No game API: the only input is a mirrored iPhone screen, the only output is a mouse click.", "bullet": True, "after": 9},
        {"t": "**Real time.** 3-minute match, elixir regenerates, every play is irreversible: frame to click in 0.34 s (p50).", "bullet": True, "after": 9},
        {"t": "**Noisy perception.** Greyed hand cards read as unknown; live state vs dataset builder: feature cosine 0.67.", "bullet": True, "after": 9},
        {"t": "**Hidden information.** The opponent's deck and hand are unknown: fill the gap with 197 pro games.", "bullet": True, "after": 9},
        {"t": f"**LLM alone is not enough.** Zero-shot d1 scores {NUM['d1_zs']} on a 4-way pick (chance 0.25).", "bullet": True, "after": 0},
    ], size=20, name="Problem bullets")
    # rules box
    box(s, 7.75, 1.4, 5.1, 2.75, fill=LIGHT, line=BLACK, lw=2, name="Rules box")
    box(s, 7.75, 1.4, 5.1, 0.5, fill=BLACK, name="Rules header")
    text(s, 7.75, 1.4, 5.1, 0.5, [{"t": "Scope and rules of play", "size": 16, "bold": True, "color": GOLD}],
         anchor=MSO_ANCHOR.MIDDLE, inset=(0.15, 0.03), name="Rules title")
    text(s, 7.85, 2.0, 4.9, 2.1, [
        {"t": "Training Camp (built-in Trainer AI) only; never ranked or ladder.", "bullet": True, "after": 5},
        {"t": "Terms of Service: friendly / Training Camp use only.", "bullet": True, "after": 5},
        {"t": "Taps need the --live flag plus an environment variable, and a banner reading \"Trainer\".", "bullet": True, "after": 0},
    ], size=15, name="Rules text")
    # setup mini diagram
    y = 4.65
    items = [("iPhone", 7.75, 1.3), ("Mac: iPhone\nMirroring window", 9.35, 1.75), ("clashbot", 11.4, 1.45)]
    for lab, x, w in items:
        b = box(s, x, y, w, 0.95, fill=(BLACK if lab == "clashbot" else WHITE), line=BLACK, lw=2, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
                radius=0.15, name=f"Setup {lab.split(':')[0]}")
        text(s, 0, 0, 0, 0, [{"t": l, "size": 14, "bold": True, "color": (GOLD if lab == "clashbot" else INK), "after": 0}
                             for l in lab.split("\n")], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=b, inset=(0.03, 0.03))
    arrow(s, 9.05, y + 0.32, 9.35, y + 0.32)
    arrow(s, 11.1, y + 0.32, 11.4, y + 0.32)
    arrow(s, 12.1, y + 0.95, 12.1, y + 1.3, color=GRAY)
    arrow(s, 12.1, y + 1.3, 10.2, y + 1.3, color=GRAY, head=False)
    arrow(s, 10.2, y + 1.3, 10.2, y + 0.95, color=GRAY)
    text(s, 7.75, y + 1.15, 2.4, 0.4, [{"t": "frames in, clicks out", "size": 14, "color": GRAY, "italic": True}], name="Setup caption")
    return s


def slide_system(prs, n):
    s = new_slide(prs, n, "System overview: perceive, retrieve, reason, tap", SCRIPT[n][1])
    bw, gap, x0 = 2.75, 0.45, 0.5
    xs = [x0 + i * (bw + gap) for i in range(4)]
    y1, h1 = 1.7, 1.3
    y2, h2 = 3.95, 1.45

    def node(x, y, w, h, title, detail, kind):
        fill, line, tc, dc = {"gray": (LIGHT, BLACK, INK, INK), "gold": (GOLD, BLACK, BLACK, BLACK), "black": (BLACK, BLACK, GOLD, WHITE)}[kind]
        b = box(s, x, y, w, h, fill=fill, line=line, lw=2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=f"Node {title}")
        text(s, 0, 0, 0, 0, [{"t": title, "size": 16, "bold": True, "color": tc, "after": 3},
                             {"t": detail, "size": 14, "color": dc, "after": 0}], anchor=MSO_ANCHOR.MIDDLE, shape=b, inset=(0.12, 0.05))
        return b

    # band labels
    tag = box(s, 0.5, 1.26, 5.6, 0.34, fill=BLACK, name="Fast loop tag")
    text(s, 0.5, 1.26, 5.6, 0.34, [{"t": "FAST LOOP  |  every frame, ~10 fps", "size": 14, "bold": True, "color": GOLD, "after": 0}],
         anchor=MSO_ANCHOR.MIDDLE, inset=(0.12, 0.02), name="Fast loop label")
    text(s, 6.3, 1.26, 6.5, 0.34, [{"t": "capture to click: 339 ms p50, 572 ms p95", "size": 14, "color": GRAY, "after": 0}],
         anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.RIGHT, name="Fast loop latency")
    node(xs[0], y1, bw, h1, "Capture", "iPhone Mirroring window, Quartz grab", "gray")
    node(xs[1], y1, bw, h1, "Perception", "YOLOv8 x2 (KataCR), OCR, Kalman tracker", "gray")
    node(xs[2], y1, bw, h1, "Live state", "hand, elixir, clock, towers, units", "gray")
    node(xs[3], y1, bw, h1, "When to act?", "rules + LightGBM gate (pros wait ~80%)", "gray")
    for i in range(3):
        arrow(s, xs[i] + bw + 0.03, y1 + h1 / 2, xs[i + 1] - 0.03, y1 + h1 / 2)
    # snake down
    arrow(s, xs[3] + bw / 2, y1 + h1 + 0.02, xs[3] + bw / 2, y2 - 0.03)
    text(s, xs[3] + bw / 2 + 0.1, y1 + h1 + 0.15, 1.3, 0.35, [{"t": "act", "size": 14, "bold": True, "after": 0}], name="Act label")
    # row 2 right-to-left
    node(xs[3], y2, bw, h2, "Engine + KG  (L5-7)", "lane simulation: tower HP saved; KG counters, answers", "gold")
    node(xs[2], y2, bw, h2, "Candidates  (L8-10)", "kNN + BM25 + RRF, KG counters, engine moves: up to 4", "gold")
    node(xs[1], y2, bw, h2, "d1 LLM ranker  (L11-14)", "fine-tuned 600M model picks 1 of 4", "gold")
    node(xs[0], y2, bw, h2, "Executor  (L15-18)", "Quartz taps; double guard + Trainer-banner check", "black")
    for i in (3, 2, 1):
        arrow(s, xs[i] - 0.03, y2 + h2 / 2, xs[i - 1] + bw + 0.03, y2 + h2 / 2)
    # loop closes: taps -> game -> capture
    arrow(s, xs[0] + bw / 2, y2 - 0.03, xs[0] + bw / 2, y1 + h1 + 0.03, color=GRAY, dash=True)
    text(s, xs[0] + bw / 2 + 0.1, y1 + h1 + 0.2, 2.5, 0.6, [{"t": "taps change the game", "size": 14, "color": GRAY, "italic": True, "after": 0}],
         name="Loop label")
    # slow loop
    sl = box(s, xs[1], 5.68, 12.85 - xs[1], 0.95, fill=WHITE, line=GRAY, lw=2, dash=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08,
             name="Slow loop box")
    text(s, 0, 0, 0, 0, [
        {"t": "SLOW LOOP (optional, off in the demo)", "size": 14, "bold": True, "after": 2},
        {"t": "Qwen3-14B coach: pre-game plan, re-plans every ~20 s in quiet moments; its [plan] line goes into d1's context.", "size": 14, "color": GRAY, "after": 0},
    ], anchor=MSO_ANCHOR.MIDDLE, shape=sl, inset=(0.15, 0.04))
    arrow(s, xs[1] + bw / 2, 5.66, xs[1] + bw / 2, y2 + h2 + 0.03, color=GRAY, dash=True)
    text(s, 0.5, 5.7, xs[1] - 0.65, 0.9, [{"t": "Gold = where the course material lives", "size": 14, "color": GRAY, "italic": True, "after": 0}],
         anchor=MSO_ANCHOR.MIDDLE, name="Legend")
    return s


def slide_data(prs, n, charts):
    s = new_slide(prs, n, "Data and knowledge: 197 pro games, one graph", SCRIPT[n][1])
    tw, th, gx, gy = 2.8, 1.35, 0.15, 0.15
    x0, y0 = 0.5, 1.4
    tile(s, x0, y0, tw, th, "197", "pro games, 33 videos, ~12 match-hours", big_size=34, name="Tile games")
    tile(s, x0 + tw + gx, y0, tw, th, "10,923", "states (6,658 train-eligible)", big_size=34, name="Tile states")
    tile(s, x0, y0 + th + gy, tw, th, "23 / 5 / 5", "videos train / val / test\n(142 / 27 / 28 games)", big_size=34, name="Tile split")
    tile(s, x0 + tw + gx, y0 + th + gy, tw, th, "3,620", "mined answer events (X played, Y answered)", big_size=34, name="Tile answers")
    text(s, 0.5, 4.5, 5.75, 2.15, [
        {"t": "Cleaned and flagged; 447 states with invalid decks are excluded from training.", "bullet": True, "after": 6},
        {"t": "Split by video, so no match leaks from train to test.", "bullet": True, "after": 6},
        {"t": "KG v1: 590 nodes, 12,030 edges (wiki, KB, mined). KG v2: **840 nodes, 27,112 edges, 19 relations**.", "bullet": True, "after": 0},
    ], size=16, name="Data bullets")
    caption(s, 6.65, 1.28, 6.2, "Knowledge graph v2: edges per relation", bold=True, color=INK)
    picture(s, charts["kg"], 6.65, 1.62, w=6.2, name="Chart KG relations")
    return s


def slide_retrieval(prs, n, charts):
    s = new_slide(prs, n, "Retrieval (L8-10): hybrid search over pro states", SCRIPT[n][1])
    caption(s, 0.5, 1.28, 6.1, "HNSW vs exact search (10.9 K vectors)", bold=True, color=INK)
    picture(s, charts["hnsw"], 0.5, 1.62, w=6.1, name="Chart HNSW")
    caption(s, 6.85, 1.28, 6.0, "Filtering after the ANN search", bold=True, color=INK)
    picture(s, charts["filters"], 6.85, 1.62, w=6.0, name="Chart filters")
    text(s, 0.5, 5.15, 12.35, 1.5, [
        {"t": "**Hybrid:** exact kNN over a 644-d state vector + BM25, fused by RRF (k = 60); channels are within noise (hit@10 0.14-0.17), the text-embedding channel adds ~20 ms for no gain.", "bullet": True, "after": 5},
        {"t": "**Scale:** HNSW ef 40 reaches recall@10 0.978 in 0.31 ms vs 1.51 ms exact; at 10.9 K vectors exact is fast enough.", "bullet": True, "after": 5},
        {"t": "**Filters:** a deck pre-filter leaves an empty pool for 89% of queries, so the live path filters by legality only.", "bullet": True, "after": 0},
    ], size=15, name="Retrieval bullets")
    return s


def slide_kg(prs, n, kge_rows, charts):
    s = new_slide(prs, n, "Knowledge graph and rules (L5-7)", SCRIPT[n][1])
    caption(s, 0.5, 1.28, 6.4, "Held-out COUNTERS link prediction (test, filtered)", bold=True, color=INK)
    rows = [["Model", "MRR", "Hits@10"]] + kge_rows
    table(s, 0.5, 1.68, [3.2, 1.5, 1.7], rows, size=15, row_min=0.42, name="KGE table",
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT], hl_rows=(1,))
    caption(s, 7.2, 1.28, 5.65, "Lane simulation: what an answer saves", bold=True, color=INK)
    picture(s, charts["lane"], 7.2, 1.66, w=5.65, name="Chart lane simulation")
    text(s, 0.5, 4.85, 12.35, 1.8, [
        {"t": "**Mined rules** (opponent plays X, pro answers Y), held-out hit@3: {p:rules_hit3} full vs {p:rules_hit3_wiki} wiki COUNTERS vs {p:rules_hit3_pop} popularity.", "bullet": True, "after": 5},
        {"t": "**Hard constraints** filter the LLM's options (card in hand, elixir, legal tile, no blind spells); perceived legality is noisy: the pros' own move passes only 54-60%.", "bullet": True, "after": 5},
        {"t": "RotatE is the only embedding clearly above popularity; DistMult is symmetric by construction (asymmetry 0.0).", "bullet": True, "after": 0},
    ], size=16, name="KG bullets")
    return s


def slide_ranker(prs, n, ranker_chart, stats):
    s = new_slide(prs, n, "The ranker: RAG + recommendation (L11-14)", SCRIPT[n][1])
    picture(s, ranker_chart, 0.5, 1.3, w=7.55, name="Chart ranker accuracy")
    note = "Whiskers: 95% game-cluster bootstrap CI (stats_audit_v1)." if stats else "{p:ci_note}"
    text(s, 0.5, 5.3, 7.55, 1.3, [
        {"t": f"Fine-tuned d1 ({NUM['d1_tac']}) is not separable from a tuned LightGBM ({NUM['lgbm']}): n = 766, standard error 1.8 points.", "size": 15, "after": 4},
        {"t": note, "size": 14, "color": GRAY, "after": 0},
    ], name="Ranker takeaway")
    # right column
    x = 8.35
    w = 4.5
    b1 = box(s, x, 1.35, 2.0, 0.8, fill=LIGHT, line=BLACK, lw=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1, name="Stage 1")
    text(s, 0, 0, 0, 0, [{"t": "Stage 1: up to 4 candidates", "size": 14, "bold": True, "after": 0}], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=b1, inset=(0.05, 0.02))
    arrow(s, x + 2.04, 1.75, x + 2.46, 1.75)
    b2 = box(s, x + 2.5, 1.35, 2.0, 0.8, fill=GOLD, line=BLACK, lw=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1, name="Stage 2")
    text(s, 0, 0, 0, 0, [{"t": "Stage 2: d1 picks one", "size": 14, "bold": True, "after": 0}], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=b2, inset=(0.05, 0.02))
    caption(s, x, 2.32, w, "Hard negatives: a 2x2 grid", bold=True, color=INK)
    grid = [["", "same spot", "other spot"],
            ["same card", "**pro play**", "negative"],
            ["other card", "negative", "negative"]]
    table(s, x, 2.7, [1.3, 1.6, 1.6], grid, size=14, row_min=0.42, zebra=False, name="Grid table", hl_rows=(1,),
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
    cb = box(s, x, 4.45, w, 2.15, fill=BLACK, name="Shortcut callout")
    text(s, 0, 0, 0, 0, [
        {"t": "Finding: the option-set shortcut", "size": 16, "bold": True, "color": GOLD, "after": 4},
        {"t": "The first negatives leaked the answer: a rule that only read the option keys scored **0.85**. The 2x2 grid makes it 0.25 by construction.", "size": 15, "color": WHITE, "after": 0},
    ], anchor=MSO_ANCHOR.MIDDLE, shape=cb, inset=(0.18, 0.08))
    return s


def slide_agents(prs, n, edit):
    s = new_slide(prs, n, "Agents (L15-18): an LLM over tools, behind guards", SCRIPT[n][1])
    # left: flow
    cw, cg = 2.0, 0.1
    ctx = [("Tactics lines", "engine, lane sim"), ("KG facts", "up to 3 lines"), ("Retrieval notes", "similar pro plays")]
    for i, (t, d) in enumerate(ctx):
        b = box(s, 0.5 + i * (cw + cg), 1.35, cw, 1.0, fill=LIGHT, line=BLACK, lw=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1, name=f"Context {t}")
        text(s, 0, 0, 0, 0, [{"t": t, "size": 15, "bold": True, "after": 1}, {"t": d, "size": 14, "color": GRAY, "after": 0}],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=b, inset=(0.04, 0.03))
        arrow(s, 0.5 + i * (cw + cg) + cw / 2, 2.37, 0.5 + i * (cw + cg) + cw / 2, 2.78)
    caption(s, 1.9, 2.4, 1.6, "tool-like context", color=GRAY, size=14)
    b = box(s, 0.5, 2.8, 6.4, 1.1, fill=GOLD, line=BLACK, lw=2, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name="d1 decision maker")
    text(s, 0, 0, 0, 0, [{"t": "d1 LLM = the decision maker", "size": 17, "bold": True, "after": 2},
                         {"t": "reads the state + context, picks 1 of up to 4 plays", "size": 15, "after": 0}],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=b, inset=(0.1, 0.04))
    arrow(s, 3.7, 3.92, 3.7, 4.3)
    ex = box(s, 0.5, 4.32, 6.4, 2.3, fill=BLACK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, name="Executor box")
    text(s, 0, 0, 0, 0, [
        {"t": "Executor: separate, guarded", "size": 16, "bold": True, "color": GOLD, "after": 4},
        {"t": "--live + CLASHBOT_ALLOW_LIVE=1, checked before each tap", "size": 14, "color": WHITE, "bullet": True, "after": 2},
        {"t": "taps only when the banner reads \"Trainer\"", "size": 14, "color": WHITE, "bullet": True, "after": 2},
        {"t": "no tap from an unread hand; failed taps get a 2 s cooldown", "size": 14, "color": WHITE, "bullet": True, "after": 2},
        {"t": "95/100 taps landed (best-documented run)", "size": 14, "color": WHITE, "bullet": True, "after": 0},
    ], anchor=MSO_ANCHOR.MIDDLE, shape=ex, inset=(0.2, 0.06))
    # right: tables
    x = 7.2
    caption(s, x, 1.28, 5.65, "Knowledge editing (synthetic +1 elixir, 5 cards)", bold=True, color=INK)
    rows = [["Criterion", "Edited", "No edit"]] + edit
    _, bot = table(s, x, 1.66, [3.05, 1.3, 1.3], rows, size=14, row_min=0.36, name="Edit table",
                   aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
    caption(s, x, bot + 0.1, 5.65, "Context conflict: does d1 follow a wrong note?", bold=True, color=INK)
    cc = [["Condition", "4-way acc.", "Follows"],
          ["C0 as exported", "{p:cc_c0_acc}", "-"],
          ["C2 notes shuffled", "{p:cc_c2_acc}", "{p:cc_c2_follow}"],
          ["C3 gold note poisoned", "{p:cc_c3_acc}", "{p:cc_c3_follow}"]]
    _, bot2 = table(s, x, bot + 0.48, [3.05, 1.3, 1.3], cc, size=14, row_min=0.36, name="Context conflict table",
                    aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
    text(s, x, bot2 + 0.12, 5.65, 6.62 - bot2 - 0.12, [
        {"t": "Synthetic edits show the mechanism, not a real patch.", "bullet": True, "size": 14, "after": 3},
        {"t": "Qwen planner made 0 tool calls: context is precomputed.", "bullet": True, "size": 14, "after": 0},
    ], name="Agents notes")
    return s


def slide_vs_pro(prs, n):
    s = new_slide(prs, n, "Bot vs pros on their own videos", SCRIPT[n][1])
    steps = [("Pro video", "test split, N = {p:vs_pro_games} games"), ("Full live stack", "open loop, dry-run executor"),
             ("Match by time", "bot act vs pro play, +/- 2 s"), ("Score", "timing + choice, vs baselines")]
    bw, gap = 2.85, 0.32
    for i, (t, d) in enumerate(steps):
        x = 0.5 + i * (bw + gap)
        b = box(s, x, 1.4, bw, 1.05, fill=(GOLD if i in (1,) else LIGHT), line=BLACK, lw=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1, name=f"Step {t}")
        text(s, 0, 0, 0, 0, [{"t": t, "size": 16, "bold": True, "after": 2}, {"t": d, "size": 14, "after": 0}], align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, shape=b, inset=(0.05, 0.03))
        if i < 3:
            arrow(s, x + bw + 0.02, 1.92, x + bw + gap - 0.02, 1.92)
    rows = [["Metric", "Clashbot", "Random (hand)", "Placement prior", "KG counter"],
            ["Pro-play recall (+/- 2 s)", "{p:vs_pro_recall}", "n/a", "n/a", "n/a"],
            ["Bot-act precision (+/- 2 s)", "{p:vs_pro_precision}", "n/a", "n/a", "n/a"],
            ["Same card (matched pairs)", "{p:vs_pro_card_bot}", "{p:vs_pro_card_random}", "{p:vs_pro_card_placement}", "{p:vs_pro_card_kg}"],
            ["Same lane (matched pairs)", "{p:vs_pro_lane_bot}", "{p:vs_pro_lane_random}", "{p:vs_pro_lane_placement}", "{p:vs_pro_lane_kg}"]]
    table(s, 0.5, 2.85, [3.75, 2.0, 2.2, 2.2, 2.2], rows, size=16, row_min=0.5, name="Vs pro table",
          aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)
    text(s, 0.5, 5.55, 12.35, 1.1, [
        {"t": "Open loop: the bot's plays do not change the video, so this measures what it would do in the pro's situation (agreement, not quality).", "bullet": True, "after": 5},
        {"t": "Baselines are scored at the pros' play times on the same games; CIs resample whole games.", "bullet": True, "after": 0},
    ], size=15, name="Vs pro bullets")
    return s


def slide_live(prs, n, lat_rows, lat_note, lat_final, charts):
    s = new_slide(prs, n, "Live results: Training Camp", SCRIPT[n][1])
    picture(s, charts["latency"], 0.5, 1.62, w=7.2, name="Chart latency")
    caption(s, 0.5, 1.28, 7.2, "Decision latency by stage (" + ("3-run mean" if lat_final else "run 20261107-111554") + ")", bold=True, color=INK)
    p50 = [r for r in lat_rows if r[0] == "total"][0]
    x, w, th = 8.0, 4.85, 1.12
    tile(s, x, 1.4, w, th, "{p:trials_score}", "Training Camp games won ({p:trials_ci})", big_size=28, name="Tile trials")
    tile(s, x, 1.4 + (th + 0.12), w, th, "First win", "game 3 of run 20261107-013441, in overtime", big_size=28, name="Tile first win")
    tile(s, x, 1.4 + 2 * (th + 0.12), w, th, "95 / 100 taps", "landed in run 20261106-233907 (6/6 ability taps)", big_size=28, name="Tile taps")
    tile(s, x, 1.4 + 3 * (th + 0.12), w, th, f"{p50[1]:.0f} / {p50[2]:.0f} ms", "capture to click, p50 / p95 (183 ms of it = 2 clicks)", big_size=28, name="Tile latency")
    if not lat_final:
        text(s, 0.5, 6.1, 7.2, 0.5, [{"t": "{p:latency_note}", "size": 14, "after": 0}], name="Latency note")
    else:
        text(s, 0.5, 6.1, 7.2, 0.5, [{"t": "p95 target 300 ms is missed once the click time counts.", "size": 14, "color": GRAY, "after": 0}], name="Latency note")
    return s


def slide_compare(prs, n):
    s = new_slide(prs, n, "Comparison with prior bots", SCRIPT[n][1])
    rows = [
        ["", "KataCR\n(Wu et al. 2025)", "Claude Royale\n(Houseworth 2025)", "Clashbot (ours)"],
        ["Setting", "phone video + ADB taps; Training Camp built-in AI", "BlueStacks on a Mac; ladder vs humans", "iPhone Mirroring on a Mac; Training Camp only"],
        ["Perception", "YOLOv8-l x2 (mAP 67.4), OCR, ResNets", "Claude vision reads screenshots", "same KataCR YOLOv8 weights, OCR, Kalman tracker"],
        ["Decision", "offline-RL transformer imitating 1 human", "Claude Haiku + hand-written prompts", "rules + LightGBM gate, engine, KG, retrieval, d1 picks 1 of 4"],
        ["Latency", "120 ms decide + 240 ms fusion", "~7 s per agent", "339 ms p50 / 572 ms p95, capture to click"],
        ["Win rate vs built-in AI", "5% (1/20); 0-10% over 9 models", "not reported (plays humans)", "{p:trials_score} (n = {p:trials_n}); one win in game 3 of one run"],
    ]
    _, bot = table(s, 0.5, 1.4, [2.15, 3.35, 3.35, 3.5], rows, size=16, row_min=0.6, name="Comparison table", first_col_bold=True, hl_rows=(5,))
    text(s, 0.5, bot + 0.18, 12.35, 6.65 - bot - 0.18, [
        {"t": "KataCR 1/20 has a 95% CI of 0.1-24.9%. Decks, opponent versions and n differ: read win rates as context, not a benchmark.", "size": 14, "color": GRAY, "after": 0},
    ], name="Comparison footnote")
    return s


def slide_demo(prs, n, crops):
    s = new_slide(prs, n, "Demo: what the decision dashboard shows", SCRIPT[n][1])
    if "dash_top" in crops:
        p, (cw_, ch_) = crops["dash_top"]
        W = 7.7
        H = W * ch_ / cw_
        picture(s, p, 0.5, 1.3, w=W, name="Dashboard crop", border=BLACK)
        fx, fy = W / cw_, H / ch_

        def mark(num, px, py):
            badge(s, 0.5 + px * fx - 0.18, 1.3 + py * fy - 0.18, num)

        mark(1, 1668, 40)    # retrieval panel, top-right corner (heading stays readable)
        mark(2, 820, 255)    # candidates heading row, right of its caption
        mark(3, 962, 392)    # reasons: bottom-right corner of the top candidate card
        ytxt = 1.3 + H + 0.12
    else:
        box(s, 0.5, 1.3, 7.7, 3.3, fill=None, line=BLACK, lw=2, dash=True, name="Dashboard placeholder")
        ytxt = 4.7
    if "dash_kg" in crops:
        p, (cw_, ch_) = crops["dash_kg"]
        h = 2.55
        w = h * cw_ / ch_
        picture(s, p, 8.55 + (4.3 - w) / 2, 4.1, h=h, name="KG crop", border=BLACK)
        badge(s, 8.55 + (4.3 - w) / 2 + w - 0.4, 4.1 + 0.06, 4)
    # video placeholder
    vb = box(s, 8.55, 1.3, 4.3, 2.55, fill=LIGHT, line=BLACK, lw=2.5, dash=True, name="Demo video placeholder")
    text(s, 0, 0, 0, 0, [{"t": "DEMO VIDEO", "size": 22, "bold": True, "after": 2},
                         {"t": "or live screen share", "size": 16, "color": GRAY, "after": 2},
                         {"t": "recorded clip, ~60 s", "size": 14, "color": GRAY, "after": 0}],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=vb, inset=(0.1, 0.05))
    items = [("Retrieval: the 10 pro states most like this one (kNN + BM25, RRF)"),
             ("Candidates: up to 4 plays, bar = d1 probability (here 88% vs 12%)"),
             ("Reasons: tactics note, e.g. pulls lava-hound to the centre, saves ~500 HP"),
             ("Knowledge graph: counters the candidate generator used, facts given to d1")]
    for i, t in enumerate(items):
        yy = ytxt + i * 0.44
        badge(s, 0.5, yy + 0.03, i + 1, d=0.34, size=14)
        text(s, 0.95, yy, 7.3, 0.4, [{"t": t, "size": 14, "after": 0}], anchor=MSO_ANCHOR.MIDDLE, name=f"Callout {i + 1}")
    return s


def slide_lessons(prs, n):
    s = new_slide(prs, n, "Lessons, limitations, future work", SCRIPT[n][1])
    cols = [
        ("What worked", GOLD, BLACK, [
            f"Knowledge + fine-tuning lift a chance-level LLM ({NUM['d1_zs']}) to {NUM['d1_tac']} on pick-the-pro-move.",
            "A separate, guarded executor: 95/100 taps landed, first Training Camp win in overtime."]),
        ("What to be honest about", BLACK, GOLD, [
            f"Simple baselines match the LLM: tuned LightGBM {NUM['lgbm']} vs d1 {NUM['d1_tac']}, inside the noise (n = 766).",
            "Perception caps everything (pros' own move passes our legality check only 54-60%); n is small, wins are counted in tens."]),
        ("Next", LIGHT, BLACK, [
            "Elixir management: no affordable card on ~1/3 of match frames in the last run; save for the answer.",
            "Retrain with the KG-lines leak fixed, own detector, 10+ Training Camp games with intervals."]),
    ]
    cw, gap = 4.0, 0.175
    for i, (hd, hfill, hcol, bl) in enumerate(cols):
        x = 0.5 + i * (cw + gap)
        box(s, x, 1.4, cw, 0.6, fill=hfill, line=BLACK, lw=1.5, name=f"Lessons head {i + 1}")
        text(s, x, 1.4, cw, 0.6, [{"t": hd, "size": 18, "bold": True, "color": hcol, "after": 0}], anchor=MSO_ANCHOR.MIDDLE,
             inset=(0.15, 0.03), name=f"Lessons title {i + 1}")
        box(s, x, 2.0, cw, 3.45, fill=WHITE, line=BLACK, lw=1.5, name=f"Lessons card {i + 1}")
        text(s, x + 0.05, 2.1, cw - 0.1, 3.3, [{"t": b, "size": 18, "bullet": True, "after": 12} for b in bl], name=f"Lessons text {i + 1}",
             inset=(0.12, 0.05))
    bn = box(s, 0.5, 5.75, 12.35, 0.85, fill=BLACK, name="Takeaway banner")
    text(s, 0, 0, 0, 0, [{"t": "Take-away: knowledge made a chance-level LLM useful; honest baselines show how much of the gain is the LLM.",
                          "size": 18, "bold": True, "color": GOLD, "after": 0}], anchor=MSO_ANCHOR.MIDDLE, shape=bn, inset=(0.2, 0.05))
    return s


def slide_backup(prs, n):
    s = new_slide(prs, n, "Backup: course-concept map", SCRIPT[n][1])
    rows = [
        ["Block", "Concept", "In clashbot", "Evidence"],
        ["L8-10", "BM25 + dense + RRF", "kNN over 644-d vector + FTS5, k = 60", "hit@10 0.16 (test), 26 ms"],
        ["L8-10", "HNSW, pre/post filter", "exact live; HNSW + filters offline", "recall 0.978 @ 0.31 ms; 89% empty"],
        ["L5-7", "KG + KGE link prediction", "KG v2 840 / 27,112; RotatE", "MRR 0.159 vs 0.110 popularity"],
        ["L5-7", "Rule mining, neuro-symbolic", "mined answers; legality filter", "held-out: {p:rules_hit3} hit@3"],
        ["L11-14", "Two-stage RAG / recsys", "candidates, then d1 picks 1 of 4", f"{NUM['d1_tac']} vs {NUM['prior']} placement prior"],
        ["L11-14", "Hard negatives, baselines", "2x2 grid; LightGBM", f"shortcut 0.85 to 0.25; LGBM {NUM['lgbm']}"],
        ["L15-18", "Typed tools, executor split", "engine + KG context; guarded taps", "95/100 taps; 0 planner tool calls"],
        ["L15-18", "Knowledge editing, conflicts", "versioned fact store; re-weighting", "reliability 1.00 vs 0.00"],
        ["L2-4", "Dense representation", "644-d vector; bge-small text channel", "text channel: no gain, off"],
    ]
    table(s, 0.5, 1.35, [1.05, 3.35, 4.45, 3.5], rows, size=16, row_min=0.5, name="Concept map table", first_col_bold=True)
    return s


# ======================================================================================
# 9. Verify (reopen with python-pptx): 14 slides, footer + logos, fonts, overflow
# ======================================================================================
def verify(path: Path) -> list[str]:
    prs = Presentation(str(path))
    problems: list[str] = []
    slides = list(prs.slides)
    if len(slides) != 14:
        problems.append(f"expected 14 slides, found {len(slides)}")
    sw, sh = prs.slide_width / 914400, prs.slide_height / 914400
    chrome = ("Footer", "Logo", "Slide number", "Title accent", "Title stripe")
    for i, sl in enumerate(slides, 1):
        names = {sh_.name: sh_ for sh_ in sl.shapes}
        for need in ("Footer bar", "Slide number"):
            if need not in names:
                problems.append(f"slide {i}: missing '{need}'")
        for logo in ("Logo UCF", "Logo CECS"):
            if logo not in names and f"{logo} placeholder" not in names:
                problems.append(f"slide {i}: missing {logo} (or placeholder)")
        if "Title" not in names and "Cover title" not in names:
            problems.append(f"slide {i}: no Title")
        if WITH_NOTES and (not sl.has_notes_slide or not sl.notes_slide.notes_text_frame.text.strip()):
            problems.append(f"slide {i}: no speaker notes")
        for sh_ in sl.shapes:
            x, y, w, h = (sh_.left or 0) / 914400, (sh_.top or 0) / 914400, (sh_.width or 0) / 914400, (sh_.height or 0) / 914400
            is_chrome = sh_.name.startswith(chrome)
            if x < -0.01 or y < -0.01 or x + w > sw + 0.01 or y + h > sh + 0.01:
                problems.append(f"slide {i}: '{sh_.name}' outside the slide")
            if not is_chrome and y + h > FOOT_Y - 0.04 and w > 0.05 and h > 0.05:
                problems.append(f"slide {i}: '{sh_.name}' reaches into the footer (bottom {y + h:.2f} in)")
            if sh_.has_text_frame and sh_.name != "Slide number":
                tf = sh_.text_frame
                total_h = 0.0
                for p in tf.paragraphs:
                    runs = p.runs
                    if not runs:
                        continue
                    sizes = [r.font.size.pt if r.font.size else None for r in runs]
                    if None in sizes:
                        problems.append(f"slide {i}: '{sh_.name}' run without explicit font size")
                        continue
                    size = max(sizes)
                    lim = MIN_TITLE if sh_.name in ("Title", "Cover title") else MIN_BODY
                    if size < lim and sh_.name != "Slide number":
                        problems.append(f"slide {i}: '{sh_.name}' font {size:.0f} pt < {lim}")
                    pPr = p._p.pPr
                    marL = int(pPr.get("marL", "0")) / 914400 if pPr is not None else 0
                    bold = any(r.font.bold for r in runs)
                    txt = "".join(r.text for r in runs)
                    wi = w - (tf.margin_left + tf.margin_right) / 914400 - marL
                    nl = count_lines(txt, size, bold, wi)
                    after = p.space_after.pt if p.space_after is not None else 0
                    total_h += nl * size * LINE_FACTOR / 72.0 + after / 72.0
                    if sh_.name == "Title" and nl > 1:
                        problems.append(f"slide {i}: Title wraps to {nl} lines")
                avail = h - (tf.margin_top + tf.margin_bottom) / 914400
                if w > 0.05 and total_h > avail + 0.04:
                    problems.append(f"slide {i}: '{sh_.name}' text needs {total_h:.2f} in, box has {avail:.2f} in (overflow)")
            if sh_.has_table if hasattr(sh_, "has_table") else False:
                tb = sh_.table
                ht = sum(r.height for r in tb.rows) / 914400
                if y + ht > FOOT_Y - 0.04:
                    problems.append(f"slide {i}: table '{sh_.name}' bottom {y + ht:.2f} in overlaps footer")
                for r_ in tb.rows:
                    for c_ in r_.cells:
                        for p in c_.text_frame.paragraphs:
                            for r in p.runs:
                                if r.font.size is None or r.font.size.pt < MIN_BODY:
                                    problems.append(f"slide {i}: table '{sh_.name}' font < {MIN_BODY}")
                                    break
    return problems


# ======================================================================================
# 10. NOTES.md
# ======================================================================================
def write_notes(used_slides: dict[str, set[int]], stats_found: bool, lat_final: bool):
    L = []
    L.append("# Clashbot showcase: talk script, timing and pending numbers\n")
    L.append("Generated by `build_slides.py` (edit the `SCRIPT` dict there, not this file). Course: CAP 6938 AI Knowledge Systems, "
             f"presenter {PRESENTER}, instructor {INSTRUCTOR}, {DECK_DATE}. Slide format 16:9, UCF white / gold #FFC904 / black.\n")
    tot = sum(v[0] for v in SCRIPT.values())
    L.append(f"## Timing (target 15 min Zoom slot, talk {tot // 60}:{tot % 60:02d} + about 1 min slack for questions)\n")
    L.append("| # | slide | time | cumulative |\n|---|---|---|---|")
    titles = {1: "Title", 2: "The problem", 3: "System overview", 4: "Data and knowledge", 5: "Retrieval (L8-10)",
              6: "Knowledge graph and rules (L5-7)", 7: "The ranker (L11-14)", 8: "Agents (L15-18)", 9: "Bot vs pros on videos",
              10: "Live results", 11: "Comparison with prior bots", 12: "Demo (clip or screen share)", 13: "Lessons and future work",
              14: "Backup: concept map (not presented)"}
    cum = 0
    for n_, (sec, _) in SCRIPT.items():
        cum += sec
        L.append(f"| {n_} | {titles[n_]} | {sec // 60}:{sec % 60:02d} | {cum // 60}:{cum % 60:02d} |")
    L.append("\nDemo: slide 12 has a placeholder where the video goes (or switch to a live screen share of the dashboard, "
             "`python -m clashbot.viz --follow runs/demo --open`). The 105 s budget includes a ~60 s clip.\n")
    L.append("## Script\n")
    for n_, (sec, _t) in SCRIPT.items():
        L.append(f"### {n_}. {titles[n_]} ({sec} s)\n\n{script_text(n_)}\n")
    L.append("## Pending numbers (red in the deck)\n")
    if used_slides:
        L.append("| key | placeholder | slides | where the number comes from |\n|---|---|---|---|")
        src = {
            "trials_score": "Training Camp trials: set `TRIALS = {'wins': k, 'n': n}` in build_slides.py (CI computed, Clopper-Pearson)",
            "trials_n": "same", "trials_ci": "same (computed)",
            "ci_note": "`docs/data/stats_audit_v1.json` (MAC-AP, E1)",
            "latency_note": "`docs/data/latency_v2.json` (MAC-AP)",
        }
        for k in sorted(used_slides):
            origin = src.get(k) or ("`docs/data/vs_pro_v1.json` (MAC-AQ)" if k.startswith("vs_pro") else
                                    "`docs/data/answer_rules_v1.json` (MAC-AR, E6)" if k.startswith("rules") else
                                    "`docs/data/context_conflict_v1.json` (MAC-AR, E2)" if k.startswith("cc_") else "?")
            L.append(f"| `{k}` | {PENDING.get(k)} | {', '.join(str(x) for x in sorted(used_slides[k]))} | {origin} |")
        L.append("")
    else:
        L.append("None: every number is final.\n")
    L.append("Data-file loaders in `build_slides.py` (`load_value_overrides`, `load_stats_audit`, `latency_stages`) fill these automatically "
             "when the file exists in `docs/data/`; the vs_pro / answer_rules / context_conflict / latency_v2 loaders were written against the "
             "specs (`docs/plans/MAC-AP..AR`) before those files existed, so check the printed WARNING lines on the first run and adjust the "
             "regexes if a number stays red although the file is there.\n")
    L.append(f"Status at build time: stats_audit_v1.json {'FOUND (CIs on the chart)' if stats_found else 'not found (chart without CIs, red note)'}; "
             f"latency_v2.json {'FOUND' if lat_final else 'not found (chart uses run 20261107-111554 summary.json)'}.\n")
    L.append("## Logos\n")
    for u in LOGO_SOURCES:
        L.append(f"- {u}")
    L.append("\nDownloaded with curl into `assets/logos/`, verified with `file` as PNG (UCF mark 330x280, CECS 1704x389 RGBA). "
             "The UCF black mark sits on the gold bar; the CECS logo is the official white-text/gold-mark version, so it is placed on a "
             "black chip to stay legible on the gold bar (logo not recolored).\n")
    L.append("## Rebuild\n")
    L.append("```bash\n"
             "# once (isolated venv; do not use the project .venv)\n"
             "python3 -m venv /tmp/pptxenv && /tmp/pptxenv/bin/pip install python-pptx matplotlib\n"
             "# every time numbers change\n"
             "cd docs/report/slides && /tmp/pptxenv/bin/python build_slides.py\n"
             "# preview (macOS): Quick Look gives the first slide only; Keynote shows all\n"
             "qlmanage -t -s 1600 -o /tmp/preview clashbot_showcase.pptx\n"
             "```\n")
    L.append("Edit `PENDING` / `FINAL` / `TRIALS` at the top of `build_slides.py`; the script prints which red keys remain.\n")
    L.append("## File format notes\n")
    L.append("- The deck is plain 13.333 x 7.5 in (no `type=` on `p:sldSz`), text boxes, rectangles, connectors, tables and PNG pictures only "
             "(Keynote, Google Slides and PowerPoint friendly). No zero-extent shapes, no palette PNGs (logos are converted to RGBA copies "
             "`assets/logos/*_rgba.png`; the downloaded originals are kept).\n"
             "- Speaker notes are NOT embedded by default: python-pptx's notes master makes macOS Quick Look hang and Keynote refuse the file. "
             "This file is the script. `python build_slides.py --with-notes` embeds it anyway (for PowerPoint users).\n"
             "- The footer (gold bar, UCF logo left, CECS logo right, slide-number field) is drawn on every slide, not in the master, "
             "so Google Slides import keeps it; the slide number is a real slide-number field.\n"
             "- The script writes to `clashbot_showcase.pptx.tmp` and renames it into place after the checks.\n")
    OUT_NOTES.write_text("\n".join(L))


# ======================================================================================
# 11. Main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-audit", type=Path, default=None, help="alternative stats_audit_v1.json (tests only)")
    ap.add_argument("--out", type=Path, default=OUT_PPTX)
    ap.add_argument("--with-notes", action="store_true",
                    help="embed the talk script as speaker notes (off by default: breaks Quick Look / Keynote import)")
    args = ap.parse_args()
    global WITH_NOTES
    WITH_NOTES = args.with_notes

    for d in (CHARTS, SHOTS, LOGOS):
        d.mkdir(parents=True, exist_ok=True)
    make_rgba_logos()
    for lg in (LOGO_UCF, LOGO_CECS):
        if not lg.exists():
            warn(f"logo missing: {lg.name} -> placeholder box in the footer")
    setup_mpl()
    load_value_overrides(args.stats_audit)
    stats = load_stats_audit(args.stats_audit)

    # --- data for charts / tables
    ret, _ = jpick("retrieval_ablation")
    kge, _ = jpick("kge_link_prediction")
    edit, _ = jpick("edit_eval")
    kge_rows = []
    full = kge["settings"]["full"]
    entries = {"random": ("Random", full["baselines"]["random"]["test"]["mrr"], full["baselines"]["random"]["test"]["hits@10"]),
               "popularity": ("Popularity", full["baselines"]["popularity"]["test"]["mrr"], full["baselines"]["popularity"]["test"]["hits@10"])}
    for m, v in full["models"].items():
        entries[m] = (m, v["test"]["mrr_mean"], v["test"]["hits@10"])
    for k in ("RotatE", "ComplEx", "popularity", "DistMult", "TransE", "random"):
        lab, mrr, h10 = entries[k]
        kge_rows.append([lab + (" (best)" if k == "RotatE" else ""), f"{mrr:.3f}", f"{h10:.3f}"])
    p = edit["pooled"]
    cf = edit["conflict_pooled"]
    edit_rows = [
        ["Reliability (n = %d)" % p["reliability"]["n"], f"{p['reliability']['edited']:.2f}", f"{p['reliability']['no_edit']:.2f}"],
        ["Generalization (n = %d)" % p["generalization"]["n"], f"{p['generalization']['edited']:.2f}", f"{p['generalization']['no_edit']:.2f}"],
        ["Locality (n = %d)" % p["locality_cards"]["n"], f"{p['locality_cards']['edited']:.2f}", f"{p['locality_cards']['no_edit']:.2f}"],
        ["Portability (n = %d)" % p["portability"]["n"], f"{p['portability']['edited']:.2f}", f"{p['portability']['no_edit']:.2f}"],
        ["Stale-card hit share", f"{cf['after']:.3f}", f"{cf['before']:.3f}"],
    ]
    lat_rows, lat_src, lat_final = latency_stages()

    # --- charts
    charts = {
        "hnsw": chart_hnsw(ret["hnsw"], 6.1, 3.45),
        "filters": chart_filters(ret["filtering"], 6.0, 3.45),
        "lane": chart_lanesim(5.65, 3.0),
        "kg": chart_kg(6.2, 5.0),
        "latency": chart_latency(lat_rows, 7.2, 4.4, lat_src),
    }
    rrows = ranker_rows(stats)
    byk = {r["key"]: r["est"] for r in rrows}
    NUM.update({"d1_zs": f"{byk['d1_zero_shot']:.2f}", "d1_ft": f"{byk['d1_joint']:.2f}", "d1_tac": f"{byk['d1_tac']:.2f}",
                "lgbm": f"{byk['lgbm']:.2f}", "prior": f"{byk['placement_prior']:.2f}", "d1_kg": f"{byk['d1_kg']:.2f}"})
    charts["ranker"] = chart_ranker(rrows, 7.55, 3.95)
    crops = make_crops()

    # --- deck
    prs = make_presentation()
    slide_title(prs, 1, crops)
    slide_problem(prs, 2)
    slide_system(prs, 3)
    slide_data(prs, 4, charts)
    slide_retrieval(prs, 5, charts)
    slide_kg(prs, 6, kge_rows, charts)
    slide_ranker(prs, 7, charts["ranker"], stats)
    slide_agents(prs, 8, edit_rows)
    slide_vs_pro(prs, 9)
    slide_live(prs, 10, lat_rows, lat_src, lat_final, charts)
    slide_compare(prs, 11)
    slide_demo(prs, 12, crops)
    slide_lessons(prs, 13)
    slide_backup(prs, 14)
    cp = prs.core_properties
    cp.title = "Clashbot: a knowledge-grounded real-time agent for Clash Royale"
    cp.author = PRESENTER
    cp.last_modified_by = PRESENTER
    cp.subject = f"{COURSE}, {SCHOOL}"
    cp.keywords = "Clash Royale, knowledge graph, retrieval, RAG, agents"
    cp.comments = f"Generated by docs/report/slides/build_slides.py; showcase {DECK_DATE}"
    tmp = args.out.with_name(args.out.name + ".tmp")
    prs.save(str(tmp))
    problems = verify(tmp)
    tmp.replace(args.out)  # never leave a half-written deck in place
    print(f"saved {args.out}")
    write_notes(USED_PENDING, stats is not None, lat_final)
    print(f"wrote {OUT_NOTES}")
    if USED_PENDING:
        print("still PENDING (red): " + ", ".join(f"{k}[s{','.join(map(str, sorted(v)))}]" for k, v in sorted(USED_PENDING.items())))
    for pr in problems:
        print("CHECK FAIL:", pr, file=sys.stderr)
    if problems:
        sys.exit(1)
    print("verify: 14 slides, footer + logos on every slide, fonts >= 14 pt (titles >= 28), no overflow detected")


if __name__ == "__main__":
    main()
