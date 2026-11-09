"""MAC-AT: the annotated demo clip: the agent's decisions drawn over a held-out pro video, rendered from a finished MAC-AQ run dir
(no model, no app, no emulator needed).

  python -m tools.eval.render_clip --run runs/eval/vs_pro/<game_id> [--start S --seconds 90 | --auto 90] --out dist/media/<name>.mp4
      [--stills DIR]     also save 3 PNG stills (a same-card moment, another moment, the title card)

Layout, 1920x1080: the game area (the calibration's content rect) scaled to full height on the left, with the bot's tap as a
cyan ring and the pro's play as a magenta ring (each fades over 1.5 s); a black panel on the right with the clock, the bot's mode,
its latest decision, candidates (d1 probability bars, source chips), the chosen candidate's note, a tactics line, a KG line and
the top 3 retrieved pro states; a bottom strip with the latest pro play and a badge (green same card, amber different card,
red the bot did not act) plus the bot's hold reason while it holds. A 4 s title card opens and a 3 s headline card closes the clip.

Time: a run's `t` is VIDEO seconds (see tools/eval/eval_vs_pro.py), so the clip reads frame `idx / fps` and looks up every
row at that `t`. The tile -> pixel mapping is the executor's own (`MirroringExecutor` on a virtual window equal to the frame,
as `clashbot.recorder_shots` does); nothing here sends input. Pairs of (bot tap, pro play) are the MAC-AQ ones: `pair_up`
within `MATCH_S`. A pair is shown when both events have happened (the later one), a pro play nobody answered 2 s after it.
Text is ASCII only (cv2's Hershey fonts have no other glyphs).
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import HOLD_KEY  # noqa: E402
from clashbot.executor import MirroringExecutor, tile_target, tile_to_screen  # noqa: E402
from clashbot.game_rules import parse_action_key  # noqa: E402
from tools.eval.eval_vs_pro import MATCH_S, bot_acts, pair_up, read_jsonl  # noqa: E402

OUT_W, OUT_H = 1920, 1080
TITLE_S, OUTRO_S = 4.0, 3.0
RING_S = 1.5                  # a ring fades out over this long
STRIP_S = 6.0                 # a resolved pro play stays in the bottom strip this long
STATE_AGE_S = 3.0             # a hold / decision row older than this no longer describes the bot
STRIP_H = 150
N_CANDS = 4
SHOWN_RETRIEVED = 3
EXAMPLES_MD = "docs/data/vs_pro_v1.md"
STATS_JSON = "docs/data/vs_pro_v1.json"

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, GREY, BLACK = (255, 255, 255), (160, 160, 160), (0, 0, 0)    # BGR
CYAN, MAGENTA = (255, 255, 0), (255, 0, 255)
GREEN, AMBER, RED = (90, 200, 80), (0, 191, 255), (70, 70, 235)
BADGES = {"same": (GREEN, "SAME CARD"), "diff": (AMBER, "DIFFERENT CARD"), "miss": (RED, "BOT DID NOT ACT")}
MODE_COLORS = {"defend": (230, 150, 60), "punish": (70, 70, 235), "counter": (0, 150, 255), "leak": (200, 90, 170),
               "spell": (200, 100, 255), "learned": (90, 200, 80), "hold": (120, 120, 120)}
CHIP_COLORS = {"tac": (0, 150, 255), "knn": (230, 150, 60), "kg": (200, 90, 170)}
CLOCK_RE = re.compile(r"^\[(\w+)\|(\d+):(\d{2})\]")
EXAMPLE_RE = re.compile(r"\*\*\d+\. (AGREE|DISAGREE)\*\* game (\S+), video t = ([\d.]+) s")


# ------------------------------------------------------------------------------------------------ the run

@dataclass
class Run:
    dir: Path
    meta: dict
    config: dict
    decisions: list[dict]
    holds: list[dict]
    pro: list[dict]
    acts: list[dict]         # the bot's taps (eval_vs_pro.bot_acts)
    events: list[dict]       # one per pro play, sorted by `resolved` (see pair_events)
    anchors: list[tuple[float, int, str]]    # (t, seconds left, phase) read from the decisions' context text

    @property
    def game_id(self) -> str:
        return self.meta.get("game_id") or self.dir.name

    @property
    def span(self) -> tuple[float, float]:
        """First and last video second the run has rows for."""
        ts = [r["t"] for r in self.decisions + self.holds]
        return float(self.config.get("start") or min(ts)), max(ts)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def pair_events(acts: list[dict], pro: list[dict]) -> list[dict]:
    """One event per pro play: its matched bot tap (or None), dt = bot - pro, same card, and `resolved`, the video second the
    pair is known (both events have happened; for an unanswered play, MATCH_S after it)."""
    pairs = dict((j, i) for i, j in pair_up([a["t"] for a in acts], [p["t"] for p in pro], MATCH_S))
    events = []
    for j, p in enumerate(pro):
        act = acts[pairs[j]] if j in pairs else None
        events.append({"t": float(p["t"]), "play": p, "act": act,
                       "dt": act["t"] - p["t"] if act else None,
                       "same_card": act["card"] == p["card"] if act else None,
                       "resolved": max(act["t"], p["t"]) if act else p["t"] + MATCH_S})
    return sorted(events, key=lambda e: e["resolved"])


def clock_anchors(decisions: list[dict]) -> list[tuple[float, int, str]]:
    out = []
    for d in decisions:
        m = CLOCK_RE.match(d.get("context_text") or "")
        if m:
            out.append((float(d["t"]), int(m.group(2)) * 60 + int(m.group(3)), m.group(1)))
    return sorted(out)


def load_run(run_dir: str | Path) -> Run:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "vs_pro_meta.json").read_text()) if (run_dir / "vs_pro_meta.json").exists() else {}
    config = json.loads((run_dir / "config.json").read_text())
    decisions = read_jsonl(run_dir / "decisions.jsonl")
    holds = read_jsonl(run_dir / "holds.jsonl") if (run_dir / "holds.jsonl").exists() else []
    pro = sorted(read_jsonl(run_dir / "pro_plays.jsonl"), key=lambda p: p["t"])
    acts, _ = bot_acts(decisions)
    return Run(run_dir, meta, config, decisions, holds, pro, acts, pair_events(acts, pro), clock_anchors(decisions))


def row_at(rows: list[dict], t: float) -> dict | None:
    """The latest row with row["t"] <= t (rows sorted by t)."""
    i = bisect.bisect_right([r["t"] for r in rows], t)
    return rows[i - 1] if i else None


def clock_at(anchors: list[tuple[float, int, str]], t: float) -> str | None:
    """Time left on the game clock at video second t: the latest decision's reading, counted down."""
    if not anchors:
        return None
    i = max(bisect.bisect_right([a[0] for a in anchors], t) - 1, 0)
    t0, left, phase = anchors[i]
    left = max(int(left - (t - t0)), 0)
    return ("OT " if phase == "overtime" else "") + f"{left // 60}:{left % 60:02d}"


# ------------------------------------------------------------------------------------------------ choosing the window

def parse_examples(text: str) -> list[dict]:
    """The MAC-AQ example moments in docs/data/vs_pro_v1.md: [{kind, game, t}]."""
    return [{"kind": k.lower(), "game": g, "t": float(t)} for k, g, t in EXAMPLE_RE.findall(text)]


def window_score(events: list[dict], s: float, e: float, examples: list[float]) -> tuple:
    """Higher is better: matched pairs inside [s, e], then an example moment inside, then same-card pairs, then earlier."""
    inside = [ev for ev in events if ev["act"] and s <= ev["t"] <= e and s <= ev["act"]["t"] <= e]
    return (len(inside), any(s <= x <= e for x in examples), sum(1 for ev in inside if ev["same_card"]), -s)


def pick_window(events: list[dict], lo: float, hi: float, seconds: float, examples: list[float] = (), step: float = 1.0) -> float:
    """Start of the `seconds`-long window in [lo, hi] with the most bot taps matched to pro plays (ties: an example moment inside)."""
    n = min(seconds, hi - lo)
    best, s = None, lo
    while s + n <= hi + 1e-9:
        score = window_score(events, s, s + n, list(examples))
        if best is None or score > best[0]:
            best = (score, s)
        s += step
    return best[1] if best else lo


# ------------------------------------------------------------------------------------------------ tile -> pixel

class TileMapper:
    """Arena tile -> pixel of the left view (the content rect of the frame, scaled to `view_h`), through the executor's geometry:
    the tap target of the tile (`tile_target`) on a virtual window equal to the frame, like the shot recorder's."""

    def __init__(self, calib: dict | str | Path, frame_w: int, frame_h: int, view_h: int = OUT_H):
        self.ex = MirroringExecutor.from_calib(calib, [0.0, 0.0, float(frame_w), float(frame_h)], live=False)
        cx, cy, cw, ch = self.ex.screen_map.content_rect_frac
        self.x0, self.y0 = int(round(cx * frame_w)), int(round(cy * frame_h))
        self.w, self.h = int(round(cw * frame_w)), int(round(ch * frame_h))
        self.scale = view_h / self.h
        self.view_w, self.view_h = int(round(self.w * self.scale)), view_h

    def to_view(self, tile) -> tuple[float, float]:
        x, y = tile_to_screen(tile_target([float(tile[0]), float(tile[1])]), self.ex.H)
        return (x - self.x0) * self.scale, (y - self.y0) * self.scale

    def view(self, frame: np.ndarray) -> np.ndarray:
        crop = frame[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        return cv2.resize(crop, (self.view_w, self.view_h), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------------------ what the panel says

def card_line(key: str | None) -> str:
    """'card / lane / zone' of an action key; 'HOLD' for a hold; anything unparsable as it is."""
    if not key or key == HOLD_KEY:
        return "HOLD"
    try:
        p = parse_action_key(key)
    except ValueError:
        return key
    return f"{p['card']} / {p['lane']} / {p['zone']}"


def decision_panel(row: dict) -> dict:
    """The text of one decision row: title, tile, reason, up to N_CANDS candidates (by d1 probability, the chosen one always
    shown) with source chips, the chosen candidate's note, one tactics line, the first KG line, the top retrieved pro states."""
    key = row.get("key") or HOLD_KEY
    cands = row.get("candidates") or []
    probs = row.get("probs") or []
    rows = [{"key": c["key"], "prob": probs[i] if i < len(probs) else None, "chosen": c["key"] == key,
             "chips": [s for s, v in (c.get("sources") or {}).items() if v and v > 0], "note": c.get("note") or ""}
            for i, c in enumerate(cands)]
    top = sorted(rows, key=lambda r: -(r["prob"] if r["prob"] is not None else -1))[:N_CANDS]
    chosen = next((r for r in rows if r["chosen"]), None)
    if chosen and chosen not in top:
        top[-1] = chosen
    tile = row.get("tile")
    return {"title": card_line(key), "held": key == HOLD_KEY, "reason": row.get("reason") or "",
            "tile": f"tile ({tile[0]:g}, {tile[1]:g})" if tile else "no tile",
            "cands": [{k: v for k, v in r.items() if k != "note"} for r in top],
            "note": chosen["note"] if chosen else "",
            "tactic": (row.get("tactics_lines") or [""])[0], "kg": (row.get("kg_lines") or [""])[0],
            "retrieved": [{"play": r.get("action_key") or "?", "score": r.get("score"), "verdict": r.get("verdict") or ""}
                          for r in (row.get("retrieval") or [])[:SHOWN_RETRIEVED]]}


def where(p: dict) -> str:
    return f" ({p['lane']}, {p['zone']})" if p.get("lane") and p.get("zone") else ""


def strip_text(ev: dict) -> tuple[str, str, str]:
    """(badge name, line 1, line 2) of a resolved pro play for the bottom strip."""
    p, act = ev["play"], ev["act"]
    line1 = f"pro played {p['card']}{where(p)} at t={ev['t']:.1f} s"
    if act is None:
        return "miss", line1, f"the bot did not act within +-{MATCH_S:g} s"
    side = "bot earlier" if ev["dt"] < 0 else "bot later"
    return ("same" if ev["same_card"] else "diff"), line1, f"bot played {act['card']}{where(act)}   dt = {ev['dt']:+.1f} s ({side})"


def panel_model(run: Run, t: float) -> dict:
    """Everything the right panel and the strip show at video second t (plain data: equal models draw equal panels)."""
    d, h = row_at(run.decisions, t), row_at(run.holds, t)
    latest_is_decision = d is not None and (h is None or d["t"] >= h["t"])
    last = d if latest_is_decision else h
    fresh = last is not None and t - last["t"] <= STATE_AGE_S
    mode, hold_line = None, None
    if fresh and latest_is_decision:
        mode = "hold" if d.get("key") == HOLD_KEY else d.get("mode")
        if mode == "hold":
            hold_line = f"decision held: {(d.get('executor') or {}).get('status', 'hold')}  ({d.get('reason') or ''})"
    elif fresh:
        mode = h["mode"] if h.get("act") else "hold"
        if not h.get("act"):
            hold_line = f"{h['mode']}: {h.get('reason') or ''}"
    resolved = [e["resolved"] for e in run.events]
    i = bisect.bisect_right(resolved, t) - 1
    strip = None
    if i >= 0 and t - resolved[i] <= STRIP_S:
        strip = strip_text(run.events[i])
    return {"clock": clock_at(run.anchors, t), "video_t": round(t, 1), "mode": mode,
            "decision": decision_panel(d) if d else None, "age": int(t - d["t"]) if d else None,
            "strip": strip, "hold_line": hold_line}


# ------------------------------------------------------------------------------------------------ drawing

def ascii_text(s: str) -> str:
    s = str(s)
    return s if s.isascii() else s.encode("ascii", "replace").decode()


def put(img, text, org, scale=0.9, color=WHITE, thick=1, outline=False) -> None:
    text, org = ascii_text(text), (int(org[0]), int(org[1]))
    if outline:
        cv2.putText(img, text, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_w(text: str, scale: float, thick: int = 1) -> int:
    return cv2.getTextSize(ascii_text(text), FONT, scale, thick)[0][0]


def wrap(text: str, width: int, scale: float, max_lines: int, thick: int = 1) -> list[str]:
    """Greedy word wrap to `width` pixels, at most `max_lines` lines (the last one ends in '...' when cut)."""
    lines, cur = [], ""
    for word in ascii_text(text).split():
        trial = f"{cur} {word}".strip()
        if cur and text_w(trial, scale, thick) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and text_w(lines[-1] + "...", scale, thick) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "..."
    return lines


def draw_ring(view: np.ndarray, xy, label: str, color, age: float, below: bool = False) -> None:
    """A ring at xy that grows and fades over RING_S, with its label."""
    a = max(0.0, 1.0 - age / RING_S)
    layer = view.copy()
    x, y, r = int(xy[0]), int(xy[1]), int(24 + 16 * age / RING_S)
    cv2.circle(layer, (x, y), r, BLACK, 8, cv2.LINE_AA)
    cv2.circle(layer, (x, y), r, color, 4, cv2.LINE_AA)
    cv2.circle(layer, (x, y), 5, color, -1, cv2.LINE_AA)
    scale = 0.7
    tx = min(max(x - text_w(label, scale, 2) // 2, 4), view.shape[1] - text_w(label, scale, 2) - 4)
    ty = min(y + r + 28, view.shape[0] - 6) if below else max(y - r - 12, 24)
    put(layer, label, (tx, ty), scale, color, 2, outline=True)
    cv2.addWeighted(layer, a, view, 1 - a, 0, dst=view)


def draw_panel(model: dict, w: int, h: int = OUT_H) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    pad, body_w = 40, w - 80
    # clock and mode
    put(img, "TIME LEFT", (pad, 38), 0.7, GREY)
    put(img, model["clock"] or "--:--", (pad, 112), 2.4, WHITE, 3)
    put(img, f"video {model['video_t']:.1f} s", (330, 112), 0.9, GREY)
    mode = model["mode"]
    cv2.rectangle(img, (w - 440, 24), (w - pad, 108), MODE_COLORS.get(mode, (70, 70, 70)), -1)
    chip = (mode or "-").upper()
    put(img, chip, (w - 240 - text_w(chip, 1.5, 2) // 2, 82), 1.5, BLACK, 2)
    cv2.line(img, (pad, 134), (w - pad, 134), (70, 70, 70), 1)
    d = model["decision"]
    if d:
        # latest decision
        put(img, f"LATEST DECISION  ({model['age']} s ago)", (pad, 172), 0.7, GREY)
        put(img, d["title"], (pad, 228), 1.5, WHITE, 2)
        put(img, d["tile"], (w - pad - text_w(d["tile"], 1.0), 228), 1.0, GREY)
        for i, line in enumerate(wrap("why: " + d["reason"], body_w, 0.85, 2)):
            put(img, line, (pad, 274 + 34 * i), 0.85, WHITE)
        # candidates
        put(img, "CANDIDATES   d1 probability   sources", (pad, 362), 0.7, GREY)
        for i, c in enumerate(d["cands"]):
            y = 412 + 50 * i
            col = WHITE if c["chosen"] else GREY
            put(img, ">" if c["chosen"] else " ", (pad, y), 0.9, CYAN)
            put(img, c["key"], (pad + 34, y), 0.9, col)
            p = c["prob"]
            cv2.rectangle(img, (760, y - 24), (1060, y + 4), (50, 50, 50), -1)
            if p is not None:
                cv2.rectangle(img, (760, y - 24), (760 + int(300 * min(max(p, 0.0), 1.0)), y + 4), CYAN if c["chosen"] else GREY, -1)
            put(img, "-" if p is None else f"{p:.2f}", (1076, y), 0.9, col)
            for k, chip_name in enumerate(c["chips"][:3]):
                x0 = 1170 + 80 * k
                cv2.rectangle(img, (x0, y - 24), (x0 + 70, y + 4), CHIP_COLORS.get(chip_name, (100, 100, 100)), -1)
                put(img, chip_name, (x0 + 35 - text_w(chip_name, 0.7) // 2, y - 3), 0.7, BLACK)
        for i, line in enumerate(wrap("chosen: " + d["note"], body_w, 0.8, 2)):
            put(img, line, (pad, 618 + 34 * i), 0.8, WHITE)
        # tactics / KG on the left, retrieved pro states on the right
        half = body_w // 2 - 20
        put(img, "TACTICS / KNOWLEDGE GRAPH", (pad, 712), 0.7, GREY)
        y = 750
        for text, col in ((d["tactic"], (130, 200, 255)), (d["kg"], (200, 160, 255))):
            for line in wrap(text, half, 0.7, 3):
                put(img, line, (pad, y), 0.7, col)
                y += 30
            y += 12
        x1 = pad + body_w // 2 + 20
        put(img, "RETRIEVED PRO STATES   pro's play, fused score", (x1, 712), 0.7, GREY)
        for i, r in enumerate(d["retrieved"]):
            score = "" if r["score"] is None else f"  {r['score']:.4f}"
            put(img, f"{i + 1}. {r['play']}{score}", (x1, 752 + 40 * i), 0.8, WHITE)
    else:
        put(img, "waiting for the bot's first decision", (pad, 228), 1.0, GREY)
    # bottom strip
    y0 = h - STRIP_H
    cv2.rectangle(img, (0, y0), (w, h), (28, 28, 28), -1)
    badge, line1, line2 = model["strip"] or (None, "waiting for a pro play", "")
    color, name = BADGES.get(badge, ((70, 70, 70), "-"))
    cv2.rectangle(img, (0, y0), (400, h), color, -1)
    put(img, name, (200 - text_w(name, 1.0, 2) // 2, y0 + 88), 1.0, BLACK, 2)
    put(img, line1, (424, y0 + 42), 0.85, WHITE)
    put(img, line2, (424, y0 + 84), 0.85, WHITE)
    if model["hold_line"]:
        put(img, "bot holds - " + model["hold_line"], (424, y0 + 126), 0.8, GREY)
    return img


def screen(items: list[tuple[str, float, tuple, int]], y: int = 120, x: int = 110, gap: int = 0) -> np.ndarray:
    """A black full-size card: lines of (text, scale, color, thickness), wrapped to the width, blank text = a gap."""
    img = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    for text, scale, color, thick in items:
        for line in wrap(text, OUT_W - 2 * x, scale, 3, thick) or [""]:
            y += int(40 * scale) + 14
            put(img, line, (x, y), scale, color, thick)
        y += gap
    return img


def title_card(run: Run, start: float, end: float) -> np.ndarray:
    cfg = run.config
    items = [("Clashbot on a held-out pro game (open loop):", 1.9, WHITE, 3),
             ("what the bot would do vs what the pro did", 1.6, WHITE, 2), ("", 1.0, WHITE, 1),
             (f"game {run.game_id}   pro result: {run.meta.get('result', '?')}   clip: {start:.0f} to {end:.0f} s of the video", 1.0, GREY, 1),
             ("deck: " + ", ".join(str(c) for c in (run.meta.get("own_deck") or str(cfg.get("deck", "")).split(","))), 0.9, WHITE, 1),
             (f"model: {Path(str(cfg.get('d1_model', '?'))).name} (d1 ranker) + {Path(str(cfg.get('act_model', '?'))).name} "
              f"(act policy) + tactics + knowledge graph; decision code frozen at {run.meta.get('frozen_commit', '?')}", 0.9, WHITE, 1)]
    img = screen(items)
    y = 600
    for color, text, ring in ((CYAN, "the bot's tap: what Clashbot would play, drawn at its target tile", True),
                              (MAGENTA, "the pro's play: what the pro actually played", True)):
        cv2.circle(img, (150, y - 10), 26, color, 4, cv2.LINE_AA)
        put(img, text, (210, y), 1.0, WHITE)
        y += 80
    for k, (color, name) in enumerate(BADGES.values()):
        cv2.rectangle(img, (110 + 360 * k, y - 34), (110 + 340 + 360 * k, y + 16), color, -1)
        put(img, name, (110 + 170 + 360 * k - text_w(name, 0.9, 2) // 2, y), 0.9, BLACK, 2)
    put(img, "badge: how the pro's play compares with the bot's tap within 2 s", (110, y + 70), 1.0, WHITE)
    put(img, "Open loop: the bot's taps are not sent anywhere, the video is the pro's. Source: YouTube " + str(run.meta.get("video_id", "")),
        (110, OUT_H - 70), 0.8, GREY)
    return img


def outro_card(run: Run, stats: dict | None, clip: dict) -> np.ndarray:
    items = [("Headline", 1.9, WHITE, 3), ("", 1.0, WHITE, 1)]
    if stats:
        agg = stats["aggregate"]
        g = next((x for x in agg["per_game"] if x["game_id"] == run.game_id), None)
        if g:
            items.append((f"This game ({run.game_id}): a bot tap within 2 s of {g['recall_2s']:.0%} of the pro's plays "
                          f"({g['pro_plays']} plays), the same card on {g['same_card']:.0%} of the matched pairs.", 1.3, WHITE, 1))
        m = agg["metrics"]
        items.append((f"All {agg['n_games']} test games: recall within 2 s {m['recall_2s']['value']:.1%}, "
                      f"same card {m['same_card']['value']:.1%} (a random card of its hand: "
                      f"{agg['baselines']['pair']['rand_hand']['value']:.1%}).", 1.3, WHITE, 1))
    items.append((f"This clip ({clip['seconds']:.0f} s): {clip['matched']} of {clip['pro_plays']} pro plays answered by a bot tap "
                  f"within 2 s, {clip['same_card']} with the same card.", 1.3, CYAN, 1))
    items.append(("Open loop on perceived frames: perception noise and the pro's hidden information limit agreement.", 1.0, GREY, 1))
    return screen(items, gap=40)


# ------------------------------------------------------------------------------------------------ rendering

def open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """H.264 ('avc1') so it plays in QuickTime and browsers; 'mp4v' when this OpenCV build cannot write it."""
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            return w, codec
        w.release()
    raise RuntimeError(f"cannot open a video writer for {path}")


def pick_still_times(events: list[dict], start: float, end: float) -> dict[str, float]:
    """Video seconds of two stills: a same-card moment and a different-card (else unanswered) one, just after they resolve
    (events that resolve alone, so the strip shows them)."""
    inside = [e for e in events if start + 5 <= e["resolved"] <= end - 5
              and not any(o is not e and 0 <= o["resolved"] - e["resolved"] < 1.5 for o in events)]
    same = next((e for e in inside if e["same_card"]), None)
    other = next((e for e in inside if e["act"] and not e["same_card"]), None) or next((e for e in inside if not e["act"]), None)
    return {name: e["resolved"] + 0.4 for name, e in (("same", same), ("other", other)) if e}


def clip_stats(run: Run, start: float, end: float) -> dict:
    inside = [e for e in run.events if start <= e["t"] <= end]
    return {"seconds": end - start, "pro_plays": len(inside), "matched": sum(1 for e in inside if e["act"]),
            "same_card": sum(1 for e in inside if e["same_card"])}


def render_clip(run: Run, start: float, seconds: float, out: Path, *, title_s: float = TITLE_S, outro_s: float = OUTRO_S,
                stats: dict | None = None, stills: Path | None = None) -> dict:
    """Write the clip; returns {frames, fps, codec, start, end, ...}."""
    cap = cv2.VideoCapture(str(resolve(run.config["video"])))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {run.config['video']}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mapper = TileMapper(resolve(run.config["calib"]), fw, fh)
    panel_w = OUT_W - mapper.view_w
    end = start + seconds
    still_t = pick_still_times(run.events, start, end) if stills else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, codec = open_writer(out, fps, (OUT_W, OUT_H))
    title = title_card(run, start, end)
    if stills:
        stills.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(stills / f"{out.stem}_still_title.png"), title)
    n_title = int(round(title_s * fps))
    for _ in range(n_title):
        writer.write(title)

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    idx, n_clip, last_model, panel = int(cap.get(cv2.CAP_PROP_POS_FRAMES)), 0, None, None
    canvas = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    while True:
        t = idx / fps
        ok, frame = cap.read()
        if not ok or t >= end:
            break
        view = mapper.view(frame)
        for a in run.acts:
            if 0 <= t - a["t"] < RING_S and a.get("tile"):
                draw_ring(view, mapper.to_view(a["tile"]), a["card"], CYAN, t - a["t"])
        for e in run.events:
            if 0 <= t - e["t"] < RING_S and e["play"].get("tile"):
                draw_ring(view, mapper.to_view(e["play"]["tile"]), "pro: " + str(e["play"]["card"]), MAGENTA, t - e["t"], below=True)
        model = panel_model(run, t)
        if model != last_model:
            panel, last_model = draw_panel(model, panel_w), model
        canvas[:, :mapper.view_w] = view
        canvas[:, mapper.view_w:] = panel
        writer.write(canvas)
        for name, t_still in list(still_t.items()):
            if t >= t_still:
                cv2.imwrite(str(stills / f"{out.stem}_still_{name}.png"), canvas)
                del still_t[name]
        n_clip += 1
        idx += 1
    cap.release()

    clip = clip_stats(run, start, start + n_clip / fps)
    outro = outro_card(run, stats, clip)
    n_outro = int(round(outro_s * fps))
    for _ in range(n_outro):
        writer.write(outro)
    writer.release()
    return {"out": str(out), "codec": codec, "fps": fps, "frames": n_title + n_clip + n_outro, "clip_frames": n_clip,
            "start": start, "end": start + n_clip / fps, "seconds": (n_title + n_clip + n_outro) / fps, **clip}


def verify(path: Path) -> dict:
    """Reopen the file: frame count, fps, size in MB."""
    cap = cv2.VideoCapture(str(path))
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS),
            "size": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "mb": round(path.stat().st_size / 1e6, 1)}
    cap.release()
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="a MAC-AQ run dir, runs/eval/vs_pro/<game_id>")
    ap.add_argument("--start", type=float, help="first video second of the clip (with --seconds)")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--auto", type=float, metavar="N", help="use the N-second window with the most bot taps matched to pro plays")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stills", help="directory for 3 PNG stills")
    ap.add_argument("--examples", default=EXAMPLES_MD, help="vs_pro markdown with the example moments (tie-break of --auto)")
    ap.add_argument("--stats", default=STATS_JSON, help="vs_pro json for the headline card")
    args = ap.parse_args(argv)

    run = load_run(resolve(args.run))
    lo, hi = run.span
    seconds = args.auto or args.seconds
    if args.start is None:
        ex_path = resolve(args.examples)
        examples = [e["t"] for e in parse_examples(ex_path.read_text()) if e["game"] == run.game_id] if ex_path.exists() else []
        start = pick_window(run.events, lo, hi, seconds, examples)
    else:
        start = args.start
    stats = json.loads(resolve(args.stats).read_text()) if resolve(args.stats).exists() else None
    info = render_clip(run, start, min(seconds, hi - start), Path(args.out), stats=stats,
                       stills=Path(args.stills) if args.stills else None)
    print(json.dumps({**info, "verify": verify(Path(args.out))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
