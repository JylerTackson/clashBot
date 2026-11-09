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
    # live Training Camp trials: set TRIALS below when done
    "trials_score": "n/10",
    "trials_n": "10",
    "trials_ci": "95% CI pending",
    # bot vs pros on their own videos  <- docs/data/vs_pro_v1.json (MAC-AQ); FINAL on 24 test games
    "vs_pro_games": "N",
    "vs_pro_plays": "N",
    "vs_pro_recall": "xx% (xx-xx)",
    "vs_pro_recall_null": "xx% (xx-xx)",
    "vs_pro_precision": "xx% (xx-xx)",
    "vs_pro_precision_null": "xx% (xx-xx)",
    "vs_pro_card_bot": "xx% (xx-xx)",
    "vs_pro_card_random": "xx% (xx-xx)",
    "vs_pro_card_n": "n",
    "vs_pro_taps_bot": "xx.x",
    "vs_pro_taps_pro": "xx.x",
    "vs_pro_ceiling": "xx%",
    "vs_pro_ceiling_ci": "xx-xx%",
    "vs_pro_ex_agree_head": "AGREE: example pending",
    "vs_pro_ex_agree_pro": "pro: pending",
    "vs_pro_ex_agree_bot": "bot: pending",
    "vs_pro_ex_agree_why": "bot's reason: pending",
    "vs_pro_ex_disagree_head": "DISAGREE: example pending",
    "vs_pro_ex_disagree_pro": "pro: pending",
    "vs_pro_ex_disagree_bot": "bot: pending",
    "vs_pro_ex_disagree_why": "bot's reason: pending",
    # held-out mined answer rules  <- docs/data/answer_rules_v1.json (MAC-AR, E6)
    "rules_t_full": "0.xx [0.xx, 0.xx]",
    "rules_t_pop": "0.xx [0.xx, 0.xx]",
    "rules_t_rand": "0.xx [0.xx, 0.xx]",
    "rules_p_full": "0.xx [0.xx, 0.xx]",
    "rules_p_pop": "0.xx [0.xx, 0.xx]",
    "rules_p_rand": "0.xx [0.xx, 0.xx]",
    "rules_t_n": "n",
    "rules_p_n": "n",
    "rules_p_full_pt": "0.xx",
    "rules_p_rand_pt": "0.xx",
    "rules_p_vs_rand": "+x.x pts, p = x",
    "rules_p_vs_pop": "-x.x pts, p = x",
    "rules_t_vs_rand": "+x.x pts, p = x",
    # context-conflict test of d1  <- docs/data/context_conflict_v1.json (MAC-AR, E2)
    "cc_c0_acc": "0.xx",
    "cc_c1_acc": "0.xx",
    "cc_c2_acc": "0.xx",
    "cc_c3_acc": "0.xx",
    "cc_c2_follow": "xx%",
    "cc_c3_follow": "xx%",
    # CIs on the ranker bar chart  <- docs/data/stats_audit_v1.json (MAC-AP, E1)
    "ci_note": "95% CI whiskers pending (stats_audit_v1)",
    "ci_vs_lgbm": "p = x",
    "ci_vs_prior": "+x.x pts, p < x",
    # per-stage latency, mean over 3 runs  <- docs/data/latency_v2.json (MAC-AP)
    "latency_note": "3-run mean (latency_v2) pending",
    # tap landing, two newest live runs  <- docs/data/tap_landing_*.md
    "tap_runs": "k/n = xx% and k/n = xx%",
    "tap_rates": "xx% / xx%",
    # decision-model benchmark  <- docs/data/decision_models_v1.json (MAC-AU part B)
    "dm_note": "decision_models_v1.json pending",
    # retrieval-method benchmark  <- docs/data/retrieval_methods_v1.json (MAC-AU part C)
    "rm_note": "retrieval_methods_v1.json pending",
}
# Numbers confirmed by hand (rendered black, win over PENDING, lose to data files):
FINAL: dict[str, str] = {}
# Training Camp trial (user-recorded, 2026-11-07, demo launcher, frozen code) -> "k/n (95% CI a-b%)" everywhere.
TRIALS = {"wins": 0, "n": 10}
# Per-battle scores us-them and overtime flag, in the order played (user-recorded).
TRIAL = {
    "date": "2026-11-07",
    "scores": [(0, 1, False), (0, 1, True), (0, 1, False), (1, 2, False), (0, 2, False),
               (1, 3, False), (1, 3, True), (0, 3, False), (0, 1, False), (0, 1, False)],
    "decisions_per_battle": 40, "taps_per_battle": 27,
    "lat_p50": 323, "lat_p95": 602,           # capture to click over the whole session (ms)
    "earlier_wins": 1,                         # one earlier recorded win (2026-11-07, overtime)
    "katacr": (1, 20),
}
TRIAL["crowns_for"] = sum(a for a, _, _ in TRIAL["scores"])
TRIAL["crowns_against"] = sum(b for _, b, _ in TRIAL["scores"])
TRIAL["tower_battles"] = sum(1 for a, _, _ in TRIAL["scores"] if a > 0)
TRIAL["overtime"] = sum(1 for *_, o in TRIAL["scores"] if o)
assert (TRIAL["crowns_for"], TRIAL["crowns_against"], TRIAL["tower_battles"], TRIAL["overtime"]) == (3, 18, 3, 2)
assert TRIALS["n"] == len(TRIAL["scores"]) and TRIALS["wins"] == sum(1 for a, b, _ in TRIAL["scores"] if a > b)

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


DATA_ROWS: dict = {}  # structured data for charts / tables (context conflict, vs-pro examples, tap landing, latency)


def dget(d, *path):
    for k in path:
        d = d[k]
    return d


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.0005 else f"p = {p:.2f}"


def ci_text(v, lo, hi, nd=0) -> str:
    return f"{100 * v:.{nd}f}% ({100 * lo:.{nd}f}-{100 * hi:.{nd}f}%)"


def _ex_line(ex: dict) -> tuple[str, str, str, str]:
    head = f"video {ex['game_id'].split('-m')[0]}, t = {ex['pro_t']:.0f} s"
    pro = "pro: " + str(ex.get("pro_key") or f"{ex.get('pro_card')} (placement not read)").replace("|", ", ")
    bot = "bot: " + str(ex.get("bot_key") or ex.get("bot_card")).replace("|", ", ")
    why = str(ex.get("reason", ""))
    mode = str(ex.get("mode", "?"))
    return head, pro, bot, "bot's reason: " + (why if why.startswith(mode) else f"{mode}: {why}")


def load_vs_pro():
    """docs/data/vs_pro_v1.json (aggregate.metrics.<name>.{value,ci95,n}); final on 24 test games (1561 pro plays)."""
    d = jload("vs_pro_v1.json")
    if not d:
        return
    try:
        a = d["aggregate"]
        m = a["metrics"]
        VALUES["vs_pro_games"] = str(int(a["n_games"]))
        VALUES["vs_pro_plays"] = f"{int(a['totals']['n_pro']):,}"
        for key, mk in (("vs_pro_recall", "recall_2s"), ("vs_pro_recall_null", "null_recall_2s"),
                        ("vs_pro_precision", "precision_2s"), ("vs_pro_precision_null", "null_precision_2s"),
                        ("vs_pro_card_bot", "same_card")):
            x = m[mk]
            VALUES[key] = ci_text(x["value"], *x["ci95"])
        VALUES["vs_pro_card_n"] = str(int(m["same_card"]["n"]))
        r = a["baselines"]["pair"]["rand_hand"]
        VALUES["vs_pro_card_random"] = ci_text(r["value"], *r["ci95"])
        VALUES["vs_pro_taps_bot"] = f"{m['bot_acts_per_min']['value']:.1f}"
        VALUES["vs_pro_taps_pro"] = f"{m['pro_plays_per_min']['value']:.1f}"
        c = m["pro_in_bot_hand"]
        VALUES["vs_pro_ceiling"] = f"{100 * c['value']:.0f}%"
        VALUES["vs_pro_ceiling_ci"] = f"{100 * c['ci95'][0]:.0f}-{100 * c['ci95'][1]:.0f}%"
        exs = d.get("examples", [])
        for verdict in ("agree", "disagree"):
            ofv = [e for e in exs if e.get("verdict") == verdict]
            pick = next((e for e in ofv if e.get("pro_key") and e.get("bot_key")), ofv[0] if ofv else None)
            if pick is None:
                warn(f"vs_pro_v1.json: no '{verdict}' example")
                continue
            head, pro, bot, why = _ex_line(pick)
            VALUES[f"vs_pro_ex_{verdict}_head"] = f"{verdict.upper()}: {head}"
            VALUES[f"vs_pro_ex_{verdict}_pro"] = pro
            VALUES[f"vs_pro_ex_{verdict}_bot"] = bot
            VALUES[f"vs_pro_ex_{verdict}_why"] = why
    except (KeyError, TypeError, IndexError) as e:
        warn(f"vs_pro_v1.json schema not recognised ({e!r}); vs-pro numbers stay PENDING")


def load_answer_rules():
    """docs/data/answer_rules_v1.json: headline.splits.<split>.methods.<m>.hit@1.{mean,lo,hi}, paired['a - b']."""
    d = jload("answer_rules_v1.json")
    if not d:
        return
    try:
        sp = d["headline"]["splits"]
        for tag, split in (("t", "test"), ("p", "val+test")):
            S = sp[split]
            for key, mk in (("full", "full"), ("pop", "popularity"), ("rand", "random")):
                h = S["methods"][mk]["hit@1"]
                VALUES[f"rules_{tag}_{key}"] = f"{h['mean']:.2f} [{h['lo']:.2f}, {h['hi']:.2f}]"
                if tag == "p" and key in ("full", "rand"):
                    VALUES[f"rules_p_{key}_pt"] = f"{h['mean']:.2f}"
            VALUES[f"rules_{tag}_n"] = f"{int(S['methods']['full']['hit@1']['n']):,}"
        for tag, split, pair, key in (("p", "val+test", "full - random", "rules_p_vs_rand"),
                                      ("p", "val+test", "full - popularity", "rules_p_vs_pop"),
                                      ("t", "test", "full - random", "rules_t_vs_rand")):
            h = sp[split]["paired"][pair]["hit@1"]
            VALUES[key] = f"{100 * h['mean']:+.1f} pts, {fmt_p(h['p'])}"
    except (KeyError, TypeError) as e:
        warn(f"answer_rules_v1.json schema not recognised ({e!r}); rule numbers stay PENDING")


def load_context_conflict():
    """docs/data/context_conflict_v1.json: splits.test.conditions.C0..C4.acc.{mean,lo,hi}, splits.test.follow.C2/C3."""
    d = jload("context_conflict_v1.json")
    if not d:
        return
    try:
        T = d["splits"]["test"]
        rows = []
        for c in ("C0", "C1", "C2", "C3", "C4"):
            a = T["conditions"][c]["acc"]
            rows.append((c, a["mean"], a["lo"], a["hi"]))
            if c in ("C0", "C1", "C2", "C3"):
                VALUES[f"cc_{c.lower()}_acc"] = f"{a['mean']:.2f}"
        DATA_ROWS["cc"] = rows
        DATA_ROWS["cc_n"] = int(T["conditions"]["C0"]["n"])
        for c in ("C2", "C3"):
            VALUES[f"cc_{c.lower()}_follow"] = pct(T["follow"][c]["follow_rate"])
    except (KeyError, TypeError) as e:
        warn(f"context_conflict_v1.json schema not recognised ({e!r}); context-conflict numbers stay PENDING")


def load_tap_landing():
    """docs/data/tap_landing_<run>_v{1,2}.md: 'verdicts over sent taps (N): landed K'. The two newest runs, oldest first."""
    runs: dict[str, tuple[int, int, str]] = {}
    for f in sorted(DATA.glob("tap_landing_*.md")):
        t = f.read_text()
        m1 = re.search(r"^# Shots review: runs/\w+/([0-9-]+)", t, re.M)
        m2 = re.search(r"verdicts over sent taps \((\d+)\): landed (\d+)", t)
        if not (m1 and m2):
            warn(f"{f.name}: tap-landing line not recognised")
            continue
        run, n, k = m1.group(1), int(m2.group(1)), int(m2.group(2))
        if run not in runs or f.name.endswith("_v2.md"):
            runs[run] = (k, n, f.name)
    if len(runs) < 2:
        return
    order = sorted(runs)  # run ids are timestamps
    newest = order[-2:]
    VALUES["tap_rates"] = " / ".join(f"{round(100 * runs[r][0] / runs[r][1])}%" for r in newest)
    VALUES["tap_runs"] = " and ".join(f"{runs[r][0]}/{runs[r][1]} = {round(100 * runs[r][0] / runs[r][1])}%" for r in newest)
    DATA_ROWS["tap_new"] = [(r,) + runs[r] for r in newest]
    if len(order) >= 3:
        r = order[-3]
        DATA_ROWS["tap_old"] = (r, runs[r][0], runs[r][1])


# ---- MAC-AU: decision models (part B) and retrieval methods (part C) -----------------------------
# chart/table label and colour kind for each model in decision_models_v1.json (the long-output Sonnet run is THE Sonnet row:
# the 16-token run, key claude-sonnet-5-5, had 62.8% cut-off replies and is only footnoted)
DM_MODELS = [
    ("d1-ft-joint-tac", "d1 fine-tuned (live)", "live"),
    ("lightgbm", "LightGBM", "lgbm"),
    ("d1-2b-ft-joint-tac", "d1-2B fine-tuned", "ft"),
    ("placement-prior", "Placement prior", "base"),
    ("claude-sonnet-5-5-long", "Claude Sonnet 5.5", "zero"),
    ("qwen3-14b", "Qwen3-14B", "zero"),
    ("qwen3-4b", "Qwen3-4B", "zero"),
    ("qwen3-4b-rag3", "Qwen3-4B + RAG", "zero"),
    ("claude-haiku-5-5", "Claude Haiku 5.5", "zero"),
    ("d1-zero-shot", "d1 zero-shot", "zero"),
]


def pct1(x: float) -> str:
    """0.5065 -> '50.6%' (rounds like the md table: 3 decimals first)."""
    return "{:.1f}%".format(float(format(x, ".3f")) * 100)


def ms_text(v: float) -> str:
    return f"{v:.2g}" if v < 10 else f"{v:,.0f}"


def load_decision_models():
    """docs/data/decision_models_v1.json: models.<key>.{acc,card_acc}.{est,lo,hi}, latency_ms.{p50,p95}, vs_reference.diff."""
    d = jload("decision_models_v1.json")
    if not d:
        warn("decision_models_v1.json missing: decision-model slide stays PENDING")
        return
    try:
        M = d["models"]
        rows = []
        for key, label, kind in DM_MODELS:
            m = M[key]
            rows.append(dict(key=key, label=label, kind=kind, est=m["acc"]["est"], lo=m["acc"]["lo"], hi=m["acc"]["hi"],
                             p50=m["latency_ms"]["p50"], p95=m["latency_ms"]["p95"], cost=m["cost_usd_per_decision"],
                             parse=m["parse_failures"]["rate"]))
        rows.sort(key=lambda r: -r["est"])
        R = {r["key"]: r for r in rows}
        d1, son = R["d1-ft-joint-tac"], R["claude-sonnet-5-5-long"]
        s16 = M["claude-sonnet-5-5"]
        diff = M["claude-sonnet-5-5-long"]["vs_reference"]["diff"]
        qw = [R[k]["est"] for k in ("qwen3-4b", "qwen3-4b-rag3", "qwen3-14b")]
        DATA_ROWS["dm"] = rows
        DATA_ROWS["dm_n"] = (int(d["lines"]["n"]), int(d["lines"]["n_games"]))
        def ci(r):
            return "[{:.1f}, {:.1f}]".format(100 * float(format(r["lo"], ".3f")), 100 * float(format(r["hi"], ".3f")))

        VALUES.update({
            "dm_n": f"{int(d['lines']['n'])}", "dm_games": f"{int(d['lines']['n_games'])}",
            "dm_d1_pct": pct1(d1["est"]), "dm_d1_ci": ci(d1), "dm_d1_p50": ms_text(d1["p50"]), "dm_d1_p95": ms_text(d1["p95"]),
            "dm_sonnet_pct": pct1(son["est"]), "dm_sonnet_ci": ci(son), "dm_sonnet_p50": ms_text(son["p50"]),
            "dm_sonnet_p95": ms_text(son["p95"]), "dm_sonnet_cost": f"${son['cost']:.4f}",
            "dm_gap_pts": f"{round(100 * (d1['est'] - son['est']))}",
            "dm_gap_ci": "[{:.1f}, {:.1f}]".format(100 * diff["lo"], 100 * diff["hi"]),
            "dm_gap_p": fmt_p(M["claude-sonnet-5-5-long"]["vs_reference"]["mcnemar_p"]),
            "dm_lat_ratio": f"{round(son['p50'] / d1['p50'])}",
            "dm_qwen_min": f"{min(qw):.2f}", "dm_qwen_max": f"{max(qw):.2f}",
            "dm_haiku_acc": f"{R['claude-haiku-5-5']['est']:.2f}",
            "dm_sonnet16_parse": f"{100 * s16['parse_failures']['rate']:.1f}%",
            "dm_sonnet16_acc": pct1(s16["acc"]["est"]),
            "dm_haiku_parse": f"{100 * R['claude-haiku-5-5']['parse']:.1f}%",
            "dm_zs_acc": f"{R['d1-zero-shot']['est']:.2f}",
        })
    except (KeyError, TypeError, IndexError, ValueError) as e:
        DATA_ROWS.pop("dm", None)
        warn(f"decision_models_v1.json schema not recognised ({e!r}); decision-model slide stays PENDING")


# method short names for the slide table / KG backup table
RM_KB_ROWS = [("dense", "Dense exact (644-d)"), ("bm25", "BM25 (FTS5)"), ("rrf_dense_bm25", "RRF dense + BM25 (live)"),
              ("bge", "bge-small text embeddings"), ("rrf_all3", "RRF dense + BM25 + bge"), ("hnsw_ef10", "HNSW ef 10"),
              ("hnsw_ef40", "HNSW ef 40"), ("hnsw_ef160", "HNSW ef 160"), ("deck_dense", "Deck pre-filter, then dense")]
RM_KG_ROWS = [("counter_scores_v1", "KG v1 counters (wiki + mined)"), ("wiki_counters", "KG v1 wiki COUNTERS only"),
              ("full", "KG v2 answers_for (full)"), ("only:pro", "KG v2 pro source only"),
              ("only:role_transfer", "KG v2 role transfer only"), ("only:wiki", "KG v2 wiki only"),
              ("only:sim", "KG v2 simulation (STOPS) only"), ("kge_rotate", "RotatE (KGE)"), ("kge_transe", "TransE (KGE)"),
              ("popularity", "Train popularity of the answer"), ("random", "Random (expected)")]


def load_retrieval_methods():
    """docs/data/retrieval_methods_v1.json: kb.<m>.{hit@10,mrr,decision.accuracy,latency.query,recall@20}, kg.<m>.{hit@1,mrr,decision}."""
    d = jload("retrieval_methods_v1.json")
    if not d:
        warn("retrieval_methods_v1.json missing: retrieval slide table stays PENDING")
        return
    try:
        kb, kg = d["kb"], d["kg"]
        krows = []
        for key, label in RM_KB_ROWS:
            m = kb[key]
            krows.append(dict(key=key, label=label, hit10=m["hit@10"]["mean"], acc=m["decision"]["accuracy"]["mean"],
                              ms=m["latency"]["query"]["ms_p50"], recall=m["recall@20"], empty=m["empty_rate"],
                              cov=m["decision"]["coverage"]["mean"]))
        grows = []
        for key, label in RM_KG_ROWS:
            m = kg[key]
            lat = m.get("latency", {}).get("decision")
            grows.append(dict(key=key, label=label, hit1=m["hit@1"]["mean"], mrr=m["mrr"]["mean"],
                              acc=m["decision"]["accuracy"]["mean"], card=m["decision"]["card_accuracy"]["mean"],
                              p50=(lat or {}).get("ms_p50"), p95=(lat or {}).get("ms_p95")))
        K = {r["key"]: r for r in krows}
        free = [r for r in krows if r["key"] != "deck_dense"]
        ka = [r["acc"] for r in krows]
        ga = [r["acc"] for r in grows if r["key"] != "random"]
        gm = [r["mrr"] for r in grows if r["key"] != "random"]
        hn = {x["ef"]: x for x in (jload("retrieval_ablation_v1.json") or {}).get("hnsw", [])}
        DATA_ROWS["rm_kb"], DATA_ROWS["rm_kg"] = krows, grows
        DATA_ROWS["rm_meta"] = dict(n=int(d["lines"]["n_lines"]), games=int(d["lines"]["n_games"]),
                                    events=int(d["kg_events"]["n_scored"]), with_threat=int(d["kg_lines"]["with_threat"]))
        rnd = next(r for r in grows if r["key"] == "random")
        VALUES.update({
            "rm_n": str(int(d["lines"]["n_lines"])),
            "rm_hit_min": f"{min(r['hit10'] for r in free):.2f}", "rm_hit_max": f"{max(r['hit10'] for r in free):.2f}",
            "rm_acc_min": f"{min(ka):.2f}", "rm_acc_max": f"{max(ka):.2f}",
            "rm_d1": f"{d['d1']['accuracy']['mean']:.2f}",
            "rm_bm25_acc": f"{K['bm25']['acc']:.3f}", "rm_live_acc": f"{K['rrf_dense_bm25']['acc']:.3f}",
            "rm_hnsw40_ms": f"{K['hnsw_ef40']['ms']:.2f}", "rm_dense_ms": f"{K['dense']['ms']:.2f}",
            "rm_hnsw40_recall": f"{K['hnsw_ef40']['recall']:.3f}",
            "rm_deck_empty": f"{100 * K['deck_dense']['empty']:.0f}%",
            "rm_kg_min": f"{min(ga):.2f}", "rm_kg_max": f"{max(ga):.2f}",
            "rm_kg_mrr_min": f"{min(gm):.3f}", "rm_kg_mrr_max": f"{max(gm):.3f}", "rm_kg_mrr_rand": f"{rnd['mrr']:.3f}",
            "rm_abl_recall40": f"{hn[40]['recall@10']:.3f}" if 40 in hn else "?",
            "rm_abl_ms40": f"{hn[40]['ms_p50']:.2f}" if 40 in hn else "?",
            "rm_abl_exact_ms": f"{hn['exact']['ms_p50']:.2f}" if "exact" in hn else "?",
        })
    except (KeyError, TypeError, IndexError, ValueError, StopIteration) as e:
        for k in ("rm_kb", "rm_kg", "rm_meta"):
            DATA_ROWS.pop(k, None)
        warn(f"retrieval_methods_v1.json schema not recognised ({e!r}); retrieval table stays PENDING")


def load_value_overrides(stats_path: Path | None):
    """Fill VALUES / DATA_ROWS from result files that are still pending. Strict: a missing key -> stays red."""
    load_vs_pro()
    load_answer_rules()
    load_context_conflict()
    load_tap_landing()
    load_decision_models()
    load_retrieval_methods()


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
        pr = {(r["a"], r["b"]): r for r in d["paired"]["test"] if r.get("why") == "spec"}
        r1 = pr[("d1_tac", "lgbm")]
        VALUES["ci_vs_lgbm"] = ("not significant" if r1["mcnemar_p"] >= 0.05 else "significant") + f" ({fmt_p(r1['mcnemar_p'])})"
        r2 = pr[("d1_tac", "placement_prior")]
        VALUES["ci_vs_prior"] = f"{100 * r2['est']:+.1f} pts, {fmt_p(r2['mcnemar_p'])}"
        return d
    except Exception as e:
        warn(f"{p.name} found but schema not recognised ({e!r}); using v1 numbers without CIs")
        return None


N_SLIDES = 16   # <= 16 allowed: 14 presented (13 + the decision-model slide) + 2 backups
LAT_STAGES = ("capture", "perception", "live_state", "trigger", "candidates", "rank", "execute", "total")
LIVE_RUN = "20261107-111554"


def latency_stages() -> tuple[list[tuple], str, bool, dict]:
    """Rows (stage, p50, p95, p50_sd, p95_sd) in ms. Prefers docs/data/latency_v2.json (h6 = 3 repeated dry runs,
    mean and sd of each run's p50 / p95), else a recorded run's summary.json (sd None).
    Last element: extras (live-run total p50/p95, runs, target_met)."""
    extras: dict = {"live_p50": 338.9, "live_p95": 571.8, "n_runs": None, "target_met": None, "decisions": None}
    d = jload("latency_v2.json")
    if d:
        try:
            st = d["h6"]["stages"]
            rows = [(k, float(st[k]["p50"]["mean"]), float(st[k]["p95"]["mean"]),
                     float(st[k]["p50"]["sd"]), float(st[k]["p95"]["sd"])) for k in LAT_STAGES]
            live = next(r for r in d["runs"] if r["run"] == LIVE_RUN)
            extras.update(live_p50=float(live["stages"]["total"]["p50"]), live_p95=float(live["stages"]["total"]["p95"]),
                          live_exec=float(live["stages"]["execute"]["p50"]),
                          n_runs=len(d["h6"]["runs"]), target_met=int(d["h6"]["target_met"]),
                          decisions=d["h6"]["decisions"])
            return rows, "docs/data/latency_v2.json", True, extras
        except (KeyError, TypeError, StopIteration, ValueError) as e:
            warn(f"latency_v2.json schema not recognised ({e!r}); using run {LIVE_RUN}")
    run = DOCS.parent / "runs" / "poc" / LIVE_RUN / "summary.json"
    if run.exists():
        sm = json.loads(run.read_text())["decision_ms"]
        rows = [(k, sm[k]["p50"], sm[k]["p95"], None, None) for k in LAT_STAGES]
        return rows, f"run {LIVE_RUN} summary.json", False, extras
    rows = [("capture", 21.2, 32.1, None, None), ("perception", 102.4, 230.0, None, None), ("live_state", 0.13, 0.33, None, None),
            ("trigger", 1.5, 8.7, None, None), ("candidates", 26.1, 56.9, None, None), ("rank", 0.0, 136.7, None, None),
            ("execute", 183.3, 188.1, None, None), ("total", 338.9, 571.8, None, None)]
    return rows, f"course_mapping_audit.md C7 (run {LIVE_RUN})", False, extras


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


def _pc1(x: float) -> str:
    return f"{100 * x:.1f}".rstrip("0").rstrip(".")


def trials_ci_text() -> str:
    lo, hi = clopper_pearson(int(TRIALS["wins"]), int(TRIALS["n"]))
    return f"95% CI {_pc1(lo)}-{_pc1(hi)}%"


def trials_text() -> tuple[str, bool]:
    """('k/n (95% CI a-b%)', pending?)"""
    k, n = TRIALS["wins"], TRIALS["n"]
    if k is None or n is None:
        return PENDING["trials_score"], True
    return f"{k}/{n} ({trials_ci_text()})", False


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on [[a, b], [c, d]] (a, c = wins; b, d = losses)."""
    n, r1, c1 = a + b + c + d, a + b, a + c

    def pr(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)

    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9)))


def resolve(key: str) -> tuple[str, bool]:
    """-> (text, is_pending). Order: data file VALUES, FINAL/TRIALS, PENDING (red)."""
    if key == "trials_score":
        txt, pend = trials_text()
    elif key == "trials_n":
        txt, pend = (str(TRIALS["n"]), False) if TRIALS["n"] is not None else (PENDING["trials_n"], True)
    elif key == "trials_ci":
        if TRIALS["wins"] is not None and TRIALS["n"] is not None:
            txt, pend = trials_ci_text(), False
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
        return PENDING[key] if TRIALS["wins"] is None else trials_ci_text()
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


def chart_decision(rows, w, h) -> Path:
    """Decision models on the 766 held-out lines: left, 4-way accuracy with 95% CI whiskers (rows sorted by accuracy);
    right, latency p50 per decision on a log axis (same rows)."""
    colors = {"base": C_LGRAY, "lgbm": "#505050", "zero": C_PGOLD, "ft": C_GOLD, "live": C_GOLD}
    n = len(rows)
    fig, (ax, bx) = plt.subplots(1, 2, sharey=True, figsize=(w, h), gridspec_kw=dict(width_ratios=[1.05, 1.0], wspace=0.05))
    ys = list(range(n))[::-1]
    for y, r in zip(ys, rows):
        ax.barh(y, r["est"], color=colors[r["kind"]], edgecolor=C_BLACK, linewidth=2.6 if r["kind"] == "live" else 1.2, height=0.68)
        ax.errorbar(r["est"], y, xerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], color=C_BLACK, capsize=4, lw=1.7, capthick=1.7)
        ax.text(r["hi"] + 0.012, y, pct1(r["est"]), va="center", ha="left", fontsize=13, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1.0), zorder=5)
        bx.plot([r["p50"]], [y], marker="o", ms=11 if r["kind"] == "live" else 9, color=colors[r["kind"]], mec=C_BLACK,
                mew=2.0 if r["kind"] == "live" else 1.2, ls="none", zorder=4)
        bx.annotate(ms_text(r["p50"]), (r["p50"], y), xytext=(11 if r["kind"] == "live" else 9, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=13, fontweight="bold" if r["kind"] == "live" else "normal")
    ax.axvline(0.25, color=C_GRAY, ls="--", lw=1.5, zorder=1)
    ax.text(0.253, n - 0.52, "chance 0.25", fontsize=12, color=C_GRAY, va="bottom", ha="left")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=14)
    ax.set_ylim(-0.6, n + 0.1)
    ax.set_xlim(0, 0.66)
    ax.set_xticks([0, 0.25, 0.5])
    ax.set_xticklabels(["0", ".25", ".50"])
    ax.set_xlabel("4-way accuracy vs the pro (95% CI)", fontsize=13)
    ax.tick_params(axis="x", labelsize=13)
    bx.set_xscale("log")
    bx.set_xlim(0.0013, 150000)
    bx.minorticks_off()
    bx.set_xticks([0.001, 0.1, 10, 1000])
    bx.set_xticklabels(["0.001", "0.1", "10", "1,000"])
    bx.grid(axis="x", color="#E3E3E3", lw=1)
    bx.set_axisbelow(True)
    bx.tick_params(axis="x", labelsize=13)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("ms per decision (p50, log scale)", fontsize=13)
    fig.subplots_adjust(left=0.215, right=0.99, top=0.985, bottom=0.17)
    return _save(fig, "decision_models.png")


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


def chart_latency(rows, w, h, final: bool, live_click: float) -> Path:
    """Per-stage p50 (gold) and p95 (black) bars; whiskers = sd over the 3 repeated runs (latency_v2 h6)."""
    names = {"capture": "Capture", "perception": "Perception", "live_state": "Live state", "trigger": "When to act",
             "candidates": "Candidates", "rank": "d1 ranker", "execute": "Taps (2 clicks)", "total": "Total"}
    fig, ax = plt.subplots(figsize=(w, h))
    n = len(rows)
    bh = 0.36

    def lab(v, sd):
        base = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
        if sd is None:
            return base
        return base + (f" \u00b1 {sd:.0f}" if v >= 10 else f" \u00b1 {sd:.1f}")

    ys = []
    for i, (k, p50, p95, s50, s95) in enumerate(rows):
        y = n - 1 - i - (0.35 if k == "total" else 0)
        ys.append(y)
        ax.barh(y + bh / 2, p50, height=bh, color=C_GOLD, edgecolor=C_BLACK, lw=1.0)
        ax.barh(y - bh / 2, p95, height=bh, color=C_BLACK, edgecolor=C_BLACK, lw=1.0)
        if k == "execute" and final:
            ax.text(8, y, f"0: dry runs do not click (live: {live_click:.0f})", va="center", fontsize=12, color=C_GRAY, style="italic")
            continue
        if p95 < 40 and s50 is not None:  # tiny stage: one label line instead of two overlapping ones
            ax.text(max(p50, p95) + 10, y, f"p50 {lab(p50, s50)},  p95 {lab(p95, s95)}", va="center", fontsize=12)
            continue
        e50 = e95 = 0
        if s50 is not None:
            ax.errorbar(p50, y + bh / 2, xerr=s50, color=C_BLACK, capsize=3, lw=1.4, capthick=1.4)
            ax.errorbar(p95, y - bh / 2, xerr=s95, color="#9A9A9A", capsize=3, lw=1.4, capthick=1.4)
            e50, e95 = s50, s95
        ax.text(p50 + e50 + 8, y + bh / 2, lab(p50, s50), va="center", fontsize=12)
        ax.text(p95 + e95 + 8, y - bh / 2, lab(p95, s95), va="center", fontsize=12, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([names[r[0]] for r in rows], fontsize=14)
    ax.axvline(300, color=C_GRAY, ls="--", lw=1.5)
    ax.text(305, n - 0.25, "p95 target 300 ms", fontsize=12, color=C_GRAY, va="top")
    ax.set_xlim(0, 690)
    ax.set_xlabel("ms per decision" + (" (mean \u00b1 sd over 3 runs)" if final else ""), fontsize=14)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C_GOLD, ec=C_BLACK, label="p50"), Patch(fc=C_BLACK, ec=C_BLACK, label="p95")],
              frameon=False, fontsize=13, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.tick_params(axis="x", labelsize=13)
    fig.tight_layout(pad=0.4)
    return _save(fig, "latency_by_stage.png")


CC_H = 1.75  # context-conflict chart height on the slide (in)


def chart_cc(rows, w, h) -> Path:
    """Context-conflict test: test-split 4-way accuracy of d1-ft-joint-tac under C0..C4 with 95% CI whiskers."""
    labels = {"C0": "C0\nas is", "C1": "C1\nremoved", "C2": "C2\nshuffled", "C3": "C3\npoisoned", "C4": "C4\nswapped"}
    fig, ax = plt.subplots(figsize=(w, h))
    for i, (c, est, lo, hi) in enumerate(rows):
        ax.bar(i, est, width=0.62, color=C_GOLD if c == "C0" else C_PGOLD, edgecolor=C_BLACK, lw=1.2)
        ax.errorbar(i, est, yerr=[[est - lo], [hi - est]], color=C_BLACK, capsize=4, lw=1.6, capthick=1.6)
        ax.text(i, 0.02, f"{est:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.axhline(0.25, color=C_GRAY, ls="--", lw=1.3)
    ax.text(5.28, 0.265, "chance", fontsize=12, color=C_GRAY, ha="right", va="bottom")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([labels[r[0]] for r in rows], fontsize=13, linespacing=0.95)
    ax.set_ylim(0, 0.7)
    ax.set_yticks([0, 0.25, 0.5])
    ax.set_yticklabels(["0", ".25", ".50"], fontsize=13)
    ax.set_xlim(-0.55, 5.3)
    fig.tight_layout(pad=0.3)
    return _save(fig, "context_conflict.png")


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
            "irreversible, so decisions must come in well under a second; we are at 0.32 seconds median from frame to click. "
            "Perception is noisy: greyed cards read as unknown, and our live state agrees with the dataset builder at a feature "
            "cosine of only 0.67. The opponent's deck and hand are hidden, so we need prior knowledge, and that is where 197 "
            "pro games come in. A general LLM does not have that knowledge: the d1 model scores <<d1_zs>> on a four-way pick, below "
            "chance. Scope: Training Camp against the built-in trainer only, never ranked, in line with the game's terms of "
            "service; the executor needs two explicit switches and checks that the banner reads Trainer."),
    3: (80, "Here is the whole system. The top row is the fast loop, running on every frame at about ten frames per second. "
            "Capture grabs the mirrored window; perception is two YOLOv8 detectors from KataCR plus OCR and a Kalman tracker; "
            "that becomes a live state, the same representation we built the dataset with. A small gate decides whether to act "
            "at all: reactive rules plus a LightGBM model, because pros wait about 80 percent of the seconds. If we act, a "
            "tactical engine with a lane simulation, and the knowledge graph, propose and justify moves. The candidate "
            "generator merges those with retrieval from the pro database, kNN plus BM25 fused by reciprocal rank fusion, to "
            "at most four options. The fine-tuned d1 language model picks one, and the executor taps behind a double guard. "
            "The tap changes the game and the loop closes. The dashed box is an optional slow loop, a Qwen coach, off in the "
            "showcase. Gold boxes are where the course material lives."),
    4: (50, "The knowledge comes from 197 pro games in 33 videos by one creator, about twelve match-hours, 10,923 sampled "
            "states. We split by video, not by game, so no match leaks across train and test; that is the contamination "
            "lesson from the RAG lectures. The knowledge graph has two versions. Version 1 came from the wiki and knowledge "
            "base: 590 nodes. Version 2, which the live bot uses, has 840 nodes and 27,112 edges in 19 relations; the gold "
            "bars are what v2 added: lane-simulation edges such as STOPS and KILLS, deck co-occurrence, role transfer, and "
            "mined answer events, 3,620 cases where the opponent played X and a pro answered with Y."),
    5: (70, "Retrieval, lectures 8 to 10. I benchmarked every method on the 766 held-out decision lines, searching only "
            "train states. The live path is hybrid: exact kNN over a 644-dimensional state vector plus BM25 over the state "
            "text, fused by reciprocal rank fusion. The methods are within noise: hit at ten between <<rm_hit_min>> and "
            "<<rm_hit_max>>. Used as a decision maker, by summing the scores of hits that match an option, every one lands at "
            "<<rm_acc_min>> to <<rm_acc_max>> four-way, against 0.25 for chance and <<rm_d1>> for d1. BM25 alone is best at "
            "<<rm_bm25_acc>>, the live fusion gets <<rm_live_acc>>, and the bge text embedding adds nothing. Right: HNSW at "
            "ef 40 reaches recall <<rm_abl_recall40>> in <<rm_abl_ms40>> milliseconds against <<rm_abl_exact_ms>> for exact, but end to end on the "
            "train pool it is <<rm_hnsw40_ms>> against <<rm_dense_ms>>, so we keep exact search. A deck pre-filter leaves "
            "<<rm_deck_empty>> of lines with no hits, so the live path filters by legality only. The knowledge-graph "
            "methods are all at chance as decision makers, <<rm_kg_min>> to <<rm_kg_max>>; their value is as candidates and "
            "context for d1."),
    6: (55, "Knowledge graph, lectures 5 to 7. For link prediction I held out COUNTERS triples and compared four embedding "
            "models. RotatE is the only one clearly above the popularity baseline: MRR 0.159 against 0.110. Neuro-symbolic "
            "part: the lane simulation, top right, gives the engine a number to reason with: a lone level-15 balloon deals "
            "3,162 tower hit points unanswered, 351 with a musketeer behind the tower, zero with a Tesla pull. Hard "
            "constraints then filter the options, though perceived legality is noisy: the pros' own move passes our check "
            "only 54 to 60 percent of the time. Bottom left, the mined answer rules, held out by game: pooled, "
            "<<rules_p_n>> events, hit at one is <<rules_p_full_pt>> against <<rules_p_rand_pt>> for random, significant, but "
            "not better than plain train popularity. A popularity prior is as good."),
    7: (85, "RAG plus recommendation, lectures 11 to 14, is the heart of the decision: a two-stage recommender. Stage one "
            "retrieves at most four candidate plays; stage two is the language model picking one. Each example is the pro's "
            "real play plus three hard negatives in a two-by-two grid: same card other placement, other card same placement, "
            "other card other placement. An earlier version of the negatives leaked the answer, a rule that looked only at "
            "the option keys scored 0.85; we rebuilt the grid and that rule is now 0.25 by construction. The chart is "
            "four-way accuracy on the test split, chance 0.25, with 95 percent whiskers from a bootstrap over whole games. "
            "Zero-shot d1 is at chance, <<d1_zs>>. Fine-tuned it reaches <<d1_ft>>, with the tactics notes <<d1_tac>>. A tuned "
            "LightGBM reaches <<lgbm>>: tuned simple baselines are competitive. The paired test of d1 with tactics against "
            "LightGBM is <<ci_vs_lgbm>>; against the placement prior d1 is ahead by <<ci_vs_prior>>, which is real. The "
            "variant with KG lines trained in was no better; it overfit."),
    8: (60, "Which model should make the decision? Same 766 held-out decisions, 27 games, for every model. Left, accuracy "
            "against the pro with 95 percent intervals; right, latency on a log scale. The fine-tuned 600-million-parameter "
            "d1 gets <<dm_d1_pct>> in <<dm_d1_p50>> milliseconds, locally, at no cost. Zero-shot, nothing does well: d1 itself "
            "is at chance, the Qwen models from 4 to 14 billion parameters reach <<dm_qwen_min>> to <<dm_qwen_max>>, and Haiku "
            "<<dm_haiku_acc>>. Retrieved pro states in the prompt do not help Qwen. Claude Sonnet 5.5 is the best zero-shot at "
            "<<dm_sonnet_pct>>, but <<dm_gap_pts>> points behind d1, at <<dm_lat_ratio>> times the latency and about a third of "
            "a cent a decision. The 2-billion d1 is worse and four times slower. LightGBM is within noise of d1. The "
            "short-output Sonnet run had <<dm_sonnet16_parse>> cut-off replies, so I use the long-output run. Fine-tuning on pro "
            "games beats much larger zero-shot models."),
    9: (60, "Agents, lectures 15 to 18. The language model is the decision maker; the things around it act like tools already "
            "called: tactics lines, a few KG facts, notes from retrieved pro states, all rendered into d1's context. The Qwen "
            "planner I built made zero tool calls, so the results are precomputed. Execution is separate and guarded: two "
            "switches, the Trainer banner, a read hand, and a cooldown after failed taps; taps landed <<tap_runs>> in the two "
            "newest live runs. Top right, knowledge editing: a synthetic plus-one-elixir change on five cards; the fact store "
            "updates answers and leaves unrelated ones alone, which shows the mechanism, not a real patch. Bottom right, the "
            "context-conflict test: with the notes in d1's context corrupted, accuracy stays between <<cc_min>> and <<cc_max>> "
            "in every condition, and d1 follows a poisoned note only <<cc_c3_follow>> of the time. It barely reads its context."),
    10: (50, "I run the full live stack on <<vs_pro_games>> test-split pro games, open loop: <<vs_pro_plays>> pro plays in all. "
             "Bot taps are matched to pro plays within two seconds. Timing: <<vs_pro_recall>> of pro plays have a bot tap "
             "nearby, and <<vs_pro_precision>> of bot taps have a pro play nearby. Read that against the null, the same taps "
             "shifted twenty to sixty seconds: <<vs_pro_recall_null>> and <<vs_pro_precision_null>>. The bot simply taps more, "
             "<<vs_pro_taps_bot>> against <<vs_pro_taps_pro>> a minute. Choice: same card as the pro <<vs_pro_card_bot>>, "
             "against <<vs_pro_card_random>> for a random card from its perceived hand; the pro's card is in that hand only "
             "<<vs_pro_ceiling>> of the time. This is agreement, not quality."),
    11: (60, "Live results, Training Camp, 7 November, demo launcher, code frozen. Zero wins in ten battles, recorded by "
             "hand. Scores, us against them: 0-1, 0-1 in overtime, 0-1, 1-2, 0-2, 1-3, 1-3 in overtime, 0-3, 0-1, 0-1. "
             "Crowns: three for, eighteen against; a tower taken in three battles, two overtimes. The 95 percent interval "
             "for zero of ten is zero to 30.8 percent: the bot is not good yet, and ten battles cannot say how bad. One "
             "earlier win was recorded, in overtime. Per battle, 40 decisions and 27 taps. The loop itself holds up: 323 "
             "milliseconds median and 602 at the 95th percentile from capture to click. The chart is three dry runs; "
             "perception is the biggest stage and the 300 millisecond p95 target is missed in all three. Taps landed "
             "<<tap_runs>> in the two newest runs."),
    12: (40, "Two prior systems. KataCR, Wu et al., is an offline-RL transformer imitating one human; its headline model "
             "won 1 of 20 against the built-in AI. We reuse its detector weights. Claude Royale is a Claude agent reading "
             "screenshots on the ladder, about seven seconds per decision, with no built-in-AI win rate. Ours adds a "
             "pro-play database, a knowledge graph and a simulation engine, and takes 0.32 seconds. Our 0 of 10 against "
             "1 of 20 gives a Fisher p of 1.0: no detectable difference, and with samples this small none could show. "
             "Decks and opponents differ, so this is context, not a benchmark."),
    13: (90, "[Play the 60-second demo clip or switch to the live screen share.] While it plays, point at the dashboard "
             "numbers. One: the ten pro states most like this situation. Two: up to four candidate plays with d1's "
             "probability as the bar, here 88 against 12 percent. Three: the reason from the tactics engine, here that a "
             "Tesla pull saves about 500 tower hit points. Four: the knowledge graph, the counters and facts given to d1. "
             "If anything breaks live, the recorded clip is the fallback."),
    14: (50, "Why does it lose? The trial and its logs give six reasons. Tower hit points flicker, so there is no committed "
             "target tower: damage is split between lanes, and ram riders followed a fallen tower's lane on to the king. "
             "The bigger deploy area after a tower falls is never used, because a destroyed tower never reads as zero. "
             "There is no support behind counter-pushes. Fireball is valued by elixir destroyed, not damage prevented, "
             "and zap counts tanks as a group. And the bot over-plays: 27 taps a battle, 40 emergency defends at two to "
             "five elixir. Next: fixes W1 to W6 from the backlog, then re-run the ten-battle trial with automatic result "
             "logging. The system and the measurement are the contribution; the trial shows where it fails. Thank you; "
             "questions."),
    15: (0, "Backup: the knowledge-graph methods as threat-to-answer rankers and as decision makers, with the chance "
            "lines. Not presented; for questions."),
    16: (0, "Backup: concept map from the course to the system, with the numbers. Not presented."),
}
assert sum(v[0] for v in SCRIPT.values()) == 840


def script_text(n: int) -> str:
    """Talk text with <<key>> replaced by the current headline numbers (so the notes follow the data)."""
    def sub(m):
        k = m.group(1)
        if k in NUM:
            return NUM[k]
        if k in VALUES:
            return VALUES[k]
        if k not in PENDING:
            warn(f"script token <<{k}>> is not defined anywhere (typo?)")
        return "[" + PENDING.get(k, k) + " - pending]"  # still red on the slide; fill from the slide

    return re.sub(r"<<(\w+)>>", sub, SCRIPT[n][1])


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
        {"t": "**Real time.** 3-minute match, elixir regenerates, every play is irreversible: frame to click in 0.32 s (p50).", "bullet": True, "after": 9},
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
    text(s, 6.3, 1.26, 6.5, 0.34, [{"t": f"capture to click: {TRIAL['lat_p50']} ms p50, {TRIAL['lat_p95']} ms p95 (live trial)", "size": 14, "color": GRAY, "after": 0}],
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
    s = new_slide(prs, n, "Retrieval (L8-10): nine methods on 766 decisions", SCRIPT[n][1])
    kb = DATA_ROWS.get("rm_kb")
    caption(s, 0.5, 1.28, 6.6, "KB methods, held-out decision lines (train pool)", bold=True, color=INK)
    if kb:
        rows = [["Method", "hit@10", "4-way acc", "ms p50"]]
        for r in kb:
            lab = r["label"] + (f" (recall@20 {r['recall']:.3f})" if r["recall"] is not None else "")
            rows.append([lab, f"{r['hit10']:.3f}", f"{r['acc']:.3f}", f"{r['ms']:.2f}"])
        live_i = 1 + [r["key"] for r in kb].index("rrf_dense_bm25")
        _, bot = table(s, 0.5, 1.62, [3.5, 0.95, 1.2, 0.95], rows, size=14, row_min=0.31, pad=0.03, name="KB methods table",
                       aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT], hl_rows=(live_i,))
    else:
        text(s, 0.5, 1.7, 6.6, 0.5, [{"t": "{p:rm_note}", "size": 14, "after": 0}], name="KB methods pending")
        bot = 4.75
    caption(s, 7.35, 1.28, 5.5, "HNSW vs exact, search only (10.9 K vectors)", bold=True, color=INK)
    picture(s, charts["hnsw"], 7.35, 1.62, w=5.5, name="Chart HNSW")
    if kb:
        text(s, 0.5, bot + 0.12, 12.35, 6.64 - bot - 0.12, [
            {"t": "**As decision makers all nine are weak:** {p:rm_acc_min}-{p:rm_acc_max} four-way (chance 0.25, d1 {p:rm_d1}). BM25 alone is best ({p:rm_bm25_acc}), the live RRF gets {p:rm_live_acc}, "
                  "bge and a third RRF channel add nothing; hit@10 is {p:rm_hit_min}-{p:rm_hit_max} with overlapping CIs.", "bullet": True, "after": 5},
            {"t": "**Scale:** HNSW ef 40 reaches recall@10 {p:rm_abl_recall40} in {p:rm_abl_ms40} ms vs {p:rm_abl_exact_ms} ms exact (search only); end to end on the train pool it is "
                  "{p:rm_hnsw40_ms} vs {p:rm_dense_ms} ms, so exact stays live. A deck pre-filter finds no hits for {p:rm_deck_empty} of lines.", "bullet": True, "after": 5},
            {"t": "**KG methods** (v1 counters and wiki, v2 per source, RotatE, TransE, popularity): all at chance as decision makers ({p:rm_kg_min}-{p:rm_kg_max} four-way); "
                  "their value is as candidates and context for d1.", "bullet": True, "after": 0},
        ], size=15, name="Retrieval bullets")
    return s


def slide_kg(prs, n, kge_rows, charts):
    s = new_slide(prs, n, "Knowledge graph and rules (L5-7)", SCRIPT[n][1])
    caption(s, 0.5, 1.28, 6.4, "Held-out COUNTERS link prediction (test, filtered)", bold=True, color=INK)
    rows = [["Model", "MRR", "Hits@10"]] + kge_rows
    table(s, 0.5, 1.66, [3.2, 1.5, 1.7], rows, size=15, row_min=0.36, pad=0.03, name="KGE table",
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT], hl_rows=(1,))
    caption(s, 7.2, 1.28, 5.65, "Lane simulation: what an answer saves", bold=True, color=INK)
    picture(s, charts["lane"], 7.2, 1.66, w=5.65, name="Chart lane simulation")
    caption(s, 0.5, 4.3, 6.2, "Mined answer rules, held-out hit@1 [95% CI]", bold=True, color=INK)
    rrows = [["Method", "Test ({p:rules_t_n})", "Pooled ({p:rules_p_n})"],
             ["Mined rules (KG v2)", "{p:rules_t_full}", "{p:rules_p_full}"],
             ["Train popularity", "{p:rules_t_pop}", "{p:rules_p_pop}"],
             ["Random in deck", "{p:rules_t_rand}", "{p:rules_p_rand}"]]
    table(s, 0.5, 4.66, [2.1, 2.0, 2.05], rrows, size=14, row_min=0.36, pad=0.03, name="Rules table",
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER], hl_rows=(1,))
    text(s, 6.95, 4.3, 5.9, 2.35, [
        {"t": "**Mined rules beat random** when pooled ({p:rules_p_vs_rand}) **but not train popularity** ({p:rules_p_vs_pop}); on test alone the gap to random is {p:rules_t_vs_rand}.", "bullet": True, "after": 4},
        {"t": "**Hard constraints** filter the LLM's options (card in hand, elixir, legal tile, no blind spells); the pros' own move passes only 54-60%.", "bullet": True, "after": 4},
        {"t": "RotatE is the only embedding clearly above popularity; DistMult is symmetric by construction.", "bullet": True, "after": 0},
    ], size=14, name="KG bullets")
    return s


def slide_ranker(prs, n, ranker_chart, stats):
    s = new_slide(prs, n, "The ranker: RAG + recommendation (L11-14)", SCRIPT[n][1])
    picture(s, ranker_chart, 0.5, 1.3, w=7.55, name="Chart ranker accuracy")
    if stats:
        note = f"Whiskers: 95% CI, bootstrap over the {stats['n_games']['test']} test games (stats_audit_v1)."
    else:
        note = "{p:ci_note}"
    text(s, 0.5, 5.3, 7.55, 1.3, [
        {"t": f"d1 + tactics ({NUM['d1_tac']}) vs LightGBM ({NUM['lgbm']}): {{p:ci_vs_lgbm}}; vs placement prior {{p:ci_vs_prior}}.", "size": 15, "after": 4},
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


# parameter counts (billions) behind the "7-23x larger" claim: d1 600M vs Qwen3-4B / Qwen3-14B
PARAMS_B = {"d1": 0.6, "qwen3-4b": 4.0, "qwen3-14b": 14.0}


def slide_decision(prs, n, charts):
    V = VALUES
    s = new_slide(prs, n, f"Which decision model? ({V.get('dm_n', 'N')} held-out decisions)", SCRIPT[n][1])
    if "dm" not in DATA_ROWS:
        text(s, 0.5, 1.5, 12.35, 0.6, [{"t": "{p:dm_note}", "size": 18, "after": 0}], name="Decision pending")
        return s
    picture(s, charts["decision"], 0.5, 1.28, w=8.4, name="Chart decision models")
    x, w = 9.15, 3.7
    tile(s, x, 1.3, w, 1.3, f"{V['dm_d1_pct']}: d1, live",
         f"95% CI {V['dm_d1_ci']}; {V['dm_d1_p50']} / {V['dm_d1_p95']} ms (p50 / p95), runs locally, no cost",
         big_size=24, name="Tile d1")
    tile(s, x, 2.7, w, 1.6, f"{V['dm_sonnet_pct']}: Sonnet 5.5",
         f"best zero-shot; {V['dm_sonnet_p50']} ms p50, {V['dm_sonnet_cost']} a decision; {V['dm_gap_pts']} points behind d1 (paired CI {V['dm_gap_ci']}, {V['dm_gap_p']})",
         big_size=24, name="Tile Sonnet")
    text(s, x, 4.38, w, 1.24, [
        {"t": f"Sonnet 5.5 = the long-output run; the 16-token run had {V['dm_sonnet16_parse']} cut-off replies. d1 zero-shot is {V['dm_zs_acc']} here, 0.22 on slide 7's plain lines.",
         "size": 14, "color": GRAY, "after": 0}], name="Decision footnote")
    lo, hi = round(PARAMS_B["qwen3-4b"] / PARAMS_B["d1"]), round(PARAMS_B["qwen3-14b"] / PARAMS_B["d1"])
    bn = box(s, 0.5, 5.78, 12.35, 0.84, fill=BLACK, name="Takeaway banner")
    text(s, 0, 0, 0, 0, [{"t": f"Fine-tuning on pro games beats {lo}\u2013{hi}\u00d7 larger zero-shot models; Sonnet 5.5 is the best zero-shot at "
                              f"{V['dm_sonnet_pct']} but {V['dm_gap_pts']} points behind d1 at ~{V['dm_lat_ratio']}\u00d7 the latency",
                          "size": 18, "bold": True, "color": GOLD, "after": 0}], anchor=MSO_ANCHOR.MIDDLE, shape=bn, inset=(0.2, 0.05))
    return s


def slide_agents(prs, n, edit, charts):
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
    ex = box(s, 0.5, 4.32, 6.4, 1.85, fill=BLACK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, name="Executor box")
    text(s, 0, 0, 0, 0, [
        {"t": "Executor: separate, guarded", "size": 16, "bold": True, "color": GOLD, "after": 3},
        {"t": "--live + CLASHBOT_ALLOW_LIVE=1, checked before each tap", "size": 14, "color": WHITE, "bullet": True, "after": 2},
        {"t": "taps only when the banner reads \"Trainer\"", "size": 14, "color": WHITE, "bullet": True, "after": 2},
        {"t": "no tap from an unread hand; failed taps get a 2 s cooldown", "size": 14, "color": WHITE, "bullet": True, "after": 2},
        {"t": "taps landed: {p:tap_runs} (two newest runs)", "size": 14, "color": WHITE, "bullet": True, "after": 0},
    ], anchor=MSO_ANCHOR.MIDDLE, shape=ex, inset=(0.2, 0.05))
    text(s, 0.5, 6.22, 6.4, 0.4, [{"t": "The Qwen planner made 0 tool calls: context is precomputed.", "size": 14, "color": GRAY, "after": 0}],
         name="Agents note")
    # right: knowledge editing table, then the context-conflict chart
    x = 7.2
    caption(s, x, 1.28, 5.65, "Knowledge editing (synthetic +1 elixir, 5 cards)", bold=True, color=INK)
    rows = [["Criterion", "Edited", "No edit"]] + edit
    _, bot = table(s, x, 1.64, [3.05, 1.3, 1.3], rows, size=14, row_min=0.33, pad=0.025, name="Edit table",
                   aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])
    caption(s, x, bot + 0.06, 5.65, "Context conflict: d1 accuracy under corrupted notes (test)", bold=True, color=INK)
    if "cc" in charts:
        picture(s, charts["cc"], x, bot + 0.4, w=5.65, name="Chart context conflict")
        ytxt = bot + 0.4 + 5.65 * CC_H / 5.65 + 0.04
    else:
        ytxt = bot + 0.44
    text(s, x, ytxt, 5.65, 6.64 - ytxt, [
        {"t": "d1 barely reads its context: poisoned notes (C3) leave accuracy at {p:cc_c3_acc} vs {p:cc_c0_acc}; it follows them only {p:cc_c3_follow} of the time.", "size": 14, "after": 0},
    ], name="Context conflict result")
    return s


def slide_vs_pro(prs, n):
    s = new_slide(prs, n, "Bot vs pros on their own videos", SCRIPT[n][1])
    games = "{p:vs_pro_games} test games"
    steps = [("Pro video", games), ("Full live stack", "open loop, dry-run executor"),
             ("Match by time", "bot tap vs pro play, +/- 2 s"), ("Score", "timing + choice, vs baselines")]
    bw, gap = 2.85, 0.32
    for i, (t, d) in enumerate(steps):
        x = 0.5 + i * (bw + gap)
        b = box(s, x, 1.3, bw, 0.95, fill=(GOLD if i in (1,) else LIGHT), line=BLACK, lw=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1, name=f"Step {t}")
        text(s, 0, 0, 0, 0, [{"t": t, "size": 16, "bold": True, "after": 1}, {"t": d, "size": 14, "after": 0}], align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, shape=b, inset=(0.05, 0.03))
        if i < 3:
            arrow(s, x + bw + 0.02, 1.775, x + bw + gap - 0.02, 1.775)
    rows = [["Metric", "Clashbot (95% CI)", "Baseline", "Baseline value (95% CI)"],
            ["Pro-play recall, +/- 2 s", "{p:vs_pro_recall}", "same taps shifted 20-60 s", "{p:vs_pro_recall_null}"],
            ["Bot-tap precision, +/- 2 s", "{p:vs_pro_precision}", "same taps shifted 20-60 s", "{p:vs_pro_precision_null}"],
            ["Same card (n = {p:vs_pro_card_n} pairs)", "{p:vs_pro_card_bot}", "random card in perceived hand", "{p:vs_pro_card_random}"],
            ["Taps per minute", "{p:vs_pro_taps_bot}", "the pros' plays per minute", "{p:vs_pro_taps_pro}"]]
    _, bot = table(s, 0.5, 2.42, [3.3, 2.5, 3.65, 2.9], rows, size=14, row_min=0.38, pad=0.03, name="Vs pro table",
                   aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.CENTER], hl_rows=())
    text(s, 0.5, bot + 0.1, 12.35, 0.55, [
        {"t": "**Perception ceiling:** the pro's card is in the hand the bot perceived only {p:vs_pro_ceiling} of the time ({p:vs_pro_ceiling_ci}), so same-card agreement cannot be high.", "size": 15, "after": 0},
    ], name="Ceiling sentence")
    ey = bot + 0.74
    eh = 6.62 - ey
    for k, (verdict, hfill, hcol) in enumerate((("agree", GOLD, BLACK), ("disagree", BLACK, GOLD))):
        ex_ = 0.5 + k * 6.25
        box(s, ex_, ey, 6.1, eh, fill=WHITE, line=BLACK, lw=1.5, name=f"Example {verdict} box")
        box(s, ex_, ey, 6.1, 0.34, fill=hfill, name=f"Example {verdict} head")
        text(s, ex_, ey, 6.1, 0.34, [{"t": f"{{p:vs_pro_ex_{verdict}_head}}", "size": 14, "bold": True, "color": hcol, "after": 0}],
             anchor=MSO_ANCHOR.MIDDLE, inset=(0.12, 0.02), name=f"Example {verdict} title")
        text(s, ex_ + 0.05, ey + 0.38, 6.0, eh - 0.42, [
            {"t": f"{{p:vs_pro_ex_{verdict}_pro}}", "size": 14, "after": 2},
            {"t": f"{{p:vs_pro_ex_{verdict}_bot}}", "size": 14, "after": 2},
            {"t": f"{{p:vs_pro_ex_{verdict}_why}}", "size": 14, "color": GRAY, "after": 0},
        ], inset=(0.1, 0.03), name=f"Example {verdict} text")
    return s


def slide_live(prs, n, lat_rows, lat_note, lat_final, lat_x, charts):
    s = new_slide(prs, n, "Live results: Training Camp", SCRIPT[n][1])
    picture(s, charts["latency"], 0.5, 1.62, w=7.2, name="Chart latency")
    caption(s, 0.5, 1.28, 7.2, "Decision latency by stage (" + ("3 dry runs, mean \u00b1 sd" if lat_final else f"run {LIVE_RUN}") + ")", bold=True, color=INK)
    tot = [r for r in lat_rows if r[0] == "total"][0]
    x, w, th, gap = 8.0, 4.85, 1.05, 0.1
    y = 1.3
    tile(s, x, y, w, th, "{p:trials_score}", "battles won, demo launcher, frozen code; one earlier win (overtime)", big_size=24, name="Tile trials")
    y += th + gap
    # per-battle scores (us-them), gold = we took a tower
    caption(s, x, y - 0.02, w, "Scores us-them; OT = overtime; gold = took a tower", size=14, color=GRAY)
    cw_, cg_ = (w - 4 * 0.07) / 5, 0.07
    for i, (a, b, ot) in enumerate(TRIAL["scores"]):
        cx = x + (i % 5) * (cw_ + cg_)
        cy = y + 0.34 + (i // 5) * 0.52
        cell = box(s, cx, cy, cw_, 0.46, fill=(GOLD if a > 0 else LIGHT), line=BLACK, lw=1.0, name=f"Score {i + 1}")
        text(s, 0, 0, 0, 0, [{"t": f"{a}-{b}" + (" OT" if ot else ""), "size": 16, "bold": True, "after": 0}], align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, shape=cell, inset=(0.03, 0.02))
    y += 0.34 + 2 * 0.52 + 0.02
    tile(s, x, y, w, th, f"{TRIAL['crowns_for']} for / {TRIAL['crowns_against']} against",
         f"crowns; a tower taken in {TRIAL['tower_battles']} of {TRIALS['n']} battles, {TRIAL['overtime']} went to overtime", big_size=24, name="Tile crowns")
    y += th + gap
    tile(s, x, y, w, th, f"{TRIAL['lat_p50']} / {TRIAL['lat_p95']} ms",
         f"capture to click, p50 / p95, whole session; per battle {TRIAL['decisions_per_battle']} decisions, {TRIAL['taps_per_battle']} taps",
         big_size=24, name="Tile latency")
    if not lat_final:
        text(s, 0.5, 6.1, 7.2, 0.5, [{"t": "{p:latency_note}", "size": 14, "after": 0}], name="Latency note")
    else:
        text(s, 0.5, 6.08, 7.2, 0.54, [{"t": f"Dry runs: total {tot[1]:.0f} \u00b1 {tot[3]:.0f} / {tot[2]:.0f} \u00b1 {tot[4]:.0f} ms (p50 / p95, no clicks); p95 target 300 ms met in {lat_x['target_met']} of {lat_x['n_runs']} runs. Taps landed {{p:tap_rates}} in the two newest runs.",
                                      "size": 14, "color": GRAY, "after": 0}], name="Latency note")
    return s


def slide_compare(prs, n):
    s = new_slide(prs, n, "Comparison with prior bots", SCRIPT[n][1])
    rows = [
        ["", "KataCR\n(Wu et al. 2025)", "Claude Royale\n(Houseworth 2025)", "Clashbot (ours)"],
        ["Setting", "phone video + ADB taps; Training Camp built-in AI", "BlueStacks on a Mac; ladder vs humans", "iPhone Mirroring on a Mac; Training Camp only"],
        ["Perception", "YOLOv8-l x2 (mAP 67.4), OCR, ResNets", "Claude vision reads screenshots", "same KataCR YOLOv8 weights, OCR, Kalman tracker"],
        ["Decision", "offline-RL transformer imitating 1 human", "Claude Haiku + hand-written prompts", "rules + LightGBM gate, engine, KG, retrieval, d1 picks 1 of 4"],
        ["Latency", "120 ms decide + 240 ms fusion", "~7 s per agent", f"{TRIAL['lat_p50']} ms p50 / {TRIAL['lat_p95']} ms p95, capture to click"],
        ["Win rate vs built-in AI", "5% (1/20); 0-10% over 9 models", "not reported (plays humans)", "{p:trials_score}; one earlier win (overtime)"],
    ]
    _, bot = table(s, 0.5, 1.4, [2.15, 3.35, 3.35, 3.5], rows, size=16, row_min=0.6, name="Comparison table", first_col_bold=True, hl_rows=(5,))
    text(s, 0.5, bot + 0.18, 12.35, 6.65 - bot - 0.18, [
        {"t": f"KataCR 1/20 has a 95% CI of 0.1-24.9%. Ours {TRIALS['wins']}/{TRIALS['n']} vs 1/20: Fisher exact p = {fisher_two_sided(TRIALS['wins'], TRIALS['n'] - TRIALS['wins'], *TRIAL['katacr'][:1], TRIAL['katacr'][1] - TRIAL['katacr'][0]):.1f}, no detectable difference. Decks, opponent versions and n differ: read win rates as context, not a benchmark.", "size": 14, "color": GRAY, "after": 0},
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
    s = new_slide(prs, n, "Why the bot loses, and what is next", SCRIPT[n][1])
    rows = [
        ("W1-2", "**No committed target tower.** Tower HP reads flicker (4424 to 424 and back), so damage is split between lanes and ram riders follow a fallen tower's lane on to the king."),
        ("W3", "**The bigger deploy area is never used.** It opens when a princess tower reads exactly 0; perception never reports 0, so the pocket is never legal."),
        ("W4", "**No support behind counter-pushes.** The win condition goes alone and dies to a witch: no musketeer behind it, no knight in front as a shield."),
        ("W5", "**Fireball is valued by elixir, not by damage prevented.** A giant + balloon + bats push counts 3.3 elixir against a 5-elixir spell, so no fireball was proposed."),
        ("W6", "**Zap counts tanks as group members.** It is cast on a giant plus 1-2 bats; it should need 3-4 troops it really hurts."),
        ("W8", f"**Over-playing.** {TRIAL['taps_per_battle']} taps per battle, 40 emergency defends at 2-5 elixir, and {{p:vs_pro_taps_bot}} vs {{p:vs_pro_taps_pro}} taps per minute against the pro."),
    ]
    rh, y0, tw = 0.74, 1.35, 0.95
    for i, (tag, t) in enumerate(rows):
        y = y0 + i * rh
        tb = box(s, 0.5, y, tw, rh - 0.08, fill=GOLD, line=BLACK, lw=1.5, name=f"Reason tag {i + 1}")
        text(s, 0, 0, 0, 0, [{"t": tag, "size": 16, "bold": True, "after": 0}], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
             shape=tb, inset=(0.03, 0.02))
        box(s, 0.5 + tw, y, 12.35 - tw, rh - 0.08, fill=WHITE, line=BLACK, lw=1.5, name=f"Reason card {i + 1}")
        text(s, 0.5 + tw + 0.05, y, 12.35 - tw - 0.1, rh - 0.08, [{"t": t, "size": 16, "after": 0}], anchor=MSO_ANCHOR.MIDDLE,
             inset=(0.1, 0.03), name=f"Reason text {i + 1}")
    by = y0 + len(rows) * rh + 0.08
    bn = box(s, 0.5, by, 12.35, 6.6 - by, fill=BLACK, name="Next banner")
    text(s, 0, 0, 0, 0, [{"t": "next: W1\u2013W6, then re-run the 10-battle trial with automatic result logging",
                          "size": 18, "bold": True, "color": GOLD, "after": 0}], anchor=MSO_ANCHOR.MIDDLE, shape=bn, inset=(0.2, 0.05))
    return s


def slide_backup_kg(prs, n):
    s = new_slide(prs, n, "Backup: KG methods vs chance and d1", SCRIPT[n][1])
    kg, meta = DATA_ROWS.get("rm_kg"), DATA_ROWS.get("rm_meta")
    if not kg:
        text(s, 0.5, 1.5, 12.35, 0.6, [{"t": "{p:rm_note}", "size": 18, "after": 0}], name="KG methods pending")
        return s
    rows = [["KG method", "hit@1", "MRR", "4-way acc", "card acc", "ms p50 / p95"]]
    for r in kg:
        lat = f"{r['p50']:.2f} / {r['p95']:.2f}" if r["p50"] is not None else "-"
        rows.append([r["label"], f"{r['hit1']:.3f}", f"{r['mrr']:.3f}", f"{r['acc']:.3f}", f"{r['card']:.3f}", lat])
    _, bot = table(s, 0.5, 1.35, [4.75, 1.15, 1.15, 1.5, 1.5, 2.3], rows, size=14, row_min=0.33, pad=0.03, name="KG methods table",
                   aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 5, first_col_bold=False)
    rnd = next(r for r in kg if r["key"] == "random")
    text(s, 0.5, bot + 0.15, 12.35, 6.64 - bot - 0.15, [
        {"t": f"hit@1 and MRR rank the pro's deck for the answer to a threat: {meta['events']} events from the {meta['games']} test games "
              f"(random: hit@1 {rnd['hit1']:.3f}, MRR {rnd['mrr']:.3f}). 4-way and card accuracy use all {meta['n']} decision lines "
              f"({meta['with_threat']} have a threat). Chance is 0.25 four-way and 0.50 card-level: the options are a 2x2 grid, "
              f"so a card-level KG score cannot choose a placement. d1 reaches {{p:rm_d1}}.", "size": 14, "color": GRAY, "after": 0},
    ], name="KG methods note")
    return s


def slide_backup(prs, n):
    s = new_slide(prs, n, "Backup: course-concept map", SCRIPT[n][1])
    live = next((r for r in DATA_ROWS.get("rm_kb", []) if r["key"] == "rrf_dense_bm25"), None)
    BK_RRF = (f"hit@10 {live['hit10']:.3f}, 4-way {live['acc']:.3f}" if live else "hit@10 0.16 (test), 26 ms")
    BK_HNSW = (f"recall {{p:rm_abl_recall40}} @ {{p:rm_abl_ms40}} ms\ndeck filter: {{p:rm_deck_empty}} empty" if live
               else "recall 0.978 @ 0.31 ms; 89% empty")
    rows = [
        ["Block", "Concept", "In clashbot", "Evidence"],
        ["L8-10", "BM25 + dense + RRF", "kNN over 644-d vector + FTS5, k = 60", BK_RRF],
        ["L8-10", "HNSW, pre/post filter", "exact live; HNSW + filters offline", BK_HNSW],
        ["L5-7", "KG + KGE link prediction", "KG v2 840 / 27,112; RotatE", "MRR 0.159 vs 0.110 popularity"],
        ["L5-7", "Rule mining, neuro-symbolic", "mined answers; legality filter", "hit@1 {p:rules_p_full_pt} vs {p:rules_p_rand_pt} random"],
        ["L11-14", "Two-stage RAG / recsys", "candidates, then d1 picks 1 of 4", f"{NUM['d1_tac']} vs {NUM['prior']} placement prior"],
        ["L11-14", "Hard negatives, baselines", "2x2 grid; LightGBM", f"shortcut 0.85 to 0.25; LGBM {NUM['lgbm']}"],
        ["L15-18", "Typed tools, executor split", "engine + KG context; guarded taps", "taps {p:tap_rates}; 0 planner calls"],
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
    if len(slides) != N_SLIDES:
        problems.append(f"expected {N_SLIDES} slides, found {len(slides)}")
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
            if sh_.has_text_frame and ("{p:" in sh_.text_frame.text or "<<" in sh_.text_frame.text):
                problems.append(f"slide {i}: '{sh_.name}' has an unresolved {{p:...}} / <<...>> token")
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
                        if "{p:" in c_.text_frame.text or "<<" in c_.text_frame.text:
                            problems.append(f"slide {i}: table '{sh_.name}' has an unresolved token")
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
              6: "Knowledge graph and rules (L5-7)", 7: "The ranker (L11-14)", 8: "Which decision model?", 9: "Agents (L15-18)",
              10: "Bot vs pros on videos", 11: "Live results", 12: "Comparison with prior bots", 13: "Demo (clip or screen share)",
              14: "Why it loses, and what is next", 15: "Backup: KG methods (not presented)",
              16: "Backup: concept map (not presented)"}
    cum = 0
    for n_, (sec, _) in SCRIPT.items():
        cum += sec
        L.append(f"| {n_} | {titles[n_]} | {sec // 60}:{sec % 60:02d} | {cum // 60}:{cum % 60:02d} |")
    L.append(f"\nDemo: slide 13 has a placeholder where the video goes (or switch to a live screen share of the dashboard, "
             f"`python -m clashbot.viz --follow runs/demo --open`). The {SCRIPT[13][0]} s budget includes a ~60 s clip.\n")
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
                                    "`docs/data/context_conflict_v1.json` (MAC-AR, E2)" if k.startswith("cc_") else
                                    "`docs/data/stats_audit_v1.json`" if k.startswith("ci_") else
                                    "`docs/data/tap_landing_*.md`" if k.startswith("tap_") else "?")
            L.append(f"| `{k}` | {PENDING.get(k)} | {', '.join(str(x) for x in sorted(used_slides[k]))} | {origin} |")
        L.append("")
    else:
        L.append("None: every number is final.\n")
    L.append("Data-file loaders in `build_slides.py` (`load_vs_pro`, `load_answer_rules`, `load_context_conflict`, `load_tap_landing`, "
             "`load_stats_audit`, `latency_stages`, `load_decision_models`, `load_retrieval_methods`) read exact JSON paths / markdown lines from `docs/data/`; if a file changes shape the "
             "loader prints a WARNING and the number stays red. Re-run `build_slides.py` after `vs_pro_v1.json` is rewritten: the games "
             "count, every number in the slide-10 table, the two example boxes and the script text follow the file.\n")
    L.append("Where the numbers on slides 5-12 and 15-16 come from:\n\n"
             "| slide | numbers | file |\n|---|---|---|\n"
             "| 5 | KB methods: hit@10, 4-way accuracy as a decision maker, query ms p50, HNSW recall@20 vs exact (table); KG accuracy range; HNSW search-only chart | `retrieval_methods_v1.json` (`kb.*`, `kg.*`, `d1`), `retrieval_ablation_v1.json` (`hnsw`) |\n"
             "| 6 | held-out mined-rule hit@1 (test, val+test) vs random vs popularity | `answer_rules_v1.json` (`headline.splits.*.methods`, `.paired`) |\n"
             "| 7 | ranker accuracy and 95% CI whiskers; d1+tactics vs LightGBM p, vs placement prior | `stats_audit_v1.json` (`accuracy.test`, `paired.test`) |\n"
             "| 8 | accuracy vs the pro with 95% CI whiskers, latency p50 / p95, cost, parse failures, paired diff to d1; Sonnet 5.5 row = `claude-sonnet-5-5-long` (the 16-token run `claude-sonnet-5-5` had 62.8% cut-off replies, footnoted only) | `decision_models_v1.json` (`models.*`) |\n"
             "| 9 | context-conflict accuracy C0-C4 (test) and follow rate | `context_conflict_v1.json` (`splits.test`) |\n"
             "| 10 | recall / precision +-2 s vs shifted null, same card vs random hand, taps per minute, ceiling, examples | `vs_pro_v1.json` (`aggregate.metrics`, `examples`); final on 24 test games |\n"
             "| 11 | per-stage latency, mean +- sd of p50 / p95 over 3 dry runs; tap landing | `latency_v2.json` (`h6.stages`), `tap_landing_*_v2.md` |\n"
             "| 11-12 | Training Camp trial 2026-11-07 (demo launcher, frozen code): wins, scores, crowns, overtime, decisions and taps per battle, session latency p50 / p95; Clopper-Pearson CI; Fisher exact vs KataCR 1/20 | `TRIALS` / `TRIAL` in `build_slides.py` (user-recorded; session log `runs/demo/20261107-213709`) |\n"
             "| 15 | all KG methods: threat-to-answer hit@1 / MRR, 4-way and card accuracy as decision makers, decision latency | `retrieval_methods_v1.json` (`kg.*`) |\n")
    L.append("Reading notes for the two MAC-AU slides: (a) d1 is 0.5065 in `decision_models_v1.json` (shown 50.6%) and 0.50653 in "
             "`retrieval_methods_v1.json` (shown 0.51 on slide 5); same predictions, different rounding. (b) d1 zero-shot is 0.26 on the 766 "
             "HEAD tactics lines of slide 8 (transformers backend) and 0.22 on the plain lines of slide 7 (`d1_zero_shot_v1.json`, mlx); both "
             "are at chance. (c) Slide 5 shows two HNSW views: search only on the full 10.9 K index (`retrieval_ablation_v1.json`, 0.31 vs 1.51 "
             "ms) and end to end on the train-filtered pool (`retrieval_methods_v1.json`, 1.11 vs 1.16 ms, includes featurising). (d) The "
             "\"7-23x\" on slide 8 is 4B/0.6B and 14B/0.6B parameters (Haiku and Sonnet sizes are not public); \"~33x\" is 2,467 / 75 ms.\n")
    L.append("Status at build time: " + ("stats_audit_v1.json FOUND (CIs on the chart)" if stats_found else "stats_audit_v1.json not found (chart without CIs, red note)")
             + "; latency_v2.json " + ("FOUND (3-run mean +- sd chart)" if lat_final else "not found (chart uses run 20261107-111554 summary.json)") + ".\n")
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
    lat_rows, lat_src, lat_final, lat_x = latency_stages()

    # --- charts
    charts = {
        "hnsw": chart_hnsw(ret["hnsw"], 5.5, 3.1),
        "lane": chart_lanesim(5.65, 2.5),
        "kg": chart_kg(6.2, 5.0),
        "latency": chart_latency(lat_rows, 7.2, 4.4, lat_final, lat_x.get("live_exec", 183.0)),
    }
    if "cc" in DATA_ROWS:
        charts["cc"] = chart_cc(DATA_ROWS["cc"], 5.65, CC_H)
    if "dm" in DATA_ROWS:
        charts["decision"] = chart_decision(DATA_ROWS["dm"], 8.4, 4.3)
    rrows = ranker_rows(stats)
    byk = {r["key"]: r["est"] for r in rrows}
    NUM.update({"d1_zs": f"{byk['d1_zero_shot']:.2f}", "d1_ft": f"{byk['d1_joint']:.2f}", "d1_tac": f"{byk['d1_tac']:.2f}",
                "lgbm": f"{byk['lgbm']:.2f}", "prior": f"{byk['placement_prior']:.2f}", "d1_kg": f"{byk['d1_kg']:.2f}"})
    if "cc" in DATA_ROWS:
        accs = [r[1] for r in DATA_ROWS["cc"]]
        NUM.update({"cc_min": f"{min(accs):.2f}", "cc_max": f"{max(accs):.2f}"})
    if "tap_old" in DATA_ROWS:
        NUM["tap_old"] = f"{DATA_ROWS['tap_old'][1]} of {DATA_ROWS['tap_old'][2]}"
    tot = [r for r in lat_rows if r[0] == "total"][0]
    NUM.update({"lat_total_p50": f"{tot[1]:.0f}" + (f" plus or minus {tot[3]:.0f}" if tot[3] is not None else ""),
                "lat_total_p95": f"{tot[2]:.0f}" + (f" plus or minus {tot[4]:.0f}" if tot[4] is not None else "")})
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
    slide_decision(prs, 8, charts)
    slide_agents(prs, 9, edit_rows, charts)
    slide_vs_pro(prs, 10)
    slide_live(prs, 11, lat_rows, lat_src, lat_final, lat_x, charts)
    slide_compare(prs, 12)
    slide_demo(prs, 13, crops)
    slide_lessons(prs, 14)
    slide_backup_kg(prs, 15)
    slide_backup(prs, 16)
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
    print(f"verify: {N_SLIDES} slides, footer + logos on every slide, fonts >= 14 pt (titles >= 28), no overflow detected")


if __name__ == "__main__":
    main()
