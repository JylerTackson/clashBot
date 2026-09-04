"""Build Phase 2 game-state samples from the perception runs.

Reads, per game, `runs/videos/<vid>/match_<m>/[game_<k>/]context.json` (plus the
match-level `states.jsonl` when the video was processed locally) and the Phase 1
match file `knowledge_base/matches/<key>.md`, and writes one JSON object per
line to `knowledge_base/states/<key>.jsonl` following
`knowledge_base/meta/game_state_schema.json`.

  python3 tools/extract_states.py [--only <key> ...] [--periodic 10] [--horizon 15]

Sample kinds:
  key       one per `## Key moments` bullet, state at t_start - 1.0 s
  play      one per own play event that is not inside a key moment, state at t - 1.0 s
  periodic  every --periodic seconds, skipping instants near a key/play sample

Nothing here runs perception: the pipeline output is only re-read. The tool is
deterministic and idempotent; re-running keeps any `enrichment` already written
into the jsonl (merged by sample id).
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base"
RUNS = ROOT / "runs" / "videos"
STATES_DIR = KB / "states"
MATCHES = KB / "matches"
CARDS_DIR = KB / "cards"
DECKS_DIR = KB / "decks"
CARD_INDEX = KB / "meta" / "card_index.json"

sys.path.insert(0, str(ROOT / "tools"))
from dispatch_matches import ready  # noqa: E402

try:  # the only perception import: pure label -> slug lookup + class vocabulary
    from cr_perception.detect import to_card_slug, TOWER_CLASSES
except Exception:  # pragma: no cover - fallback when cv2/torch are absent
    TOWER_CLASSES = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
                     "royal-chef-tower"}

    def to_card_slug(label: str) -> str:
        return label.replace("-evolution", "")

SCHEMA_VERSION = 1
CREATOR = "ryleycr1"
NEAR_S = 4.0          # a periodic instant this close to another sample is dropped
PLAY_NEAR_KEY_S = 2.0  # a play instant this close to a key sample is dropped
PLAY_LEAD_S = 1.0     # sample the state this far before the play / key moment
ACTION_WINDOW_S = 6.0  # "what he did next" window for key/periodic samples
RECENT_S = 12.0       # recent_plays window
COMMENTARY_S = 6.0    # transcript cues within +-6 s
STATE_TOL_S = 0.6     # max distance of a states.jsonl row from the wanted instant

PHASES = {"single_elixir", "double_elixir", "triple_elixir", "overtime"}
VALID_ZONES = ("back", "mid", "bridge", "enemy_half")

# --------------------------------------------------------------------------- parsing helpers

_T_PAT = re.compile(r"\bt\s*(?:=|≈|~)\s*[~≈]?\s*(\d+(?:\.\d+)?)"
                    r"(?:\s*(?:-|–|—|->|→|to)\s*(\d+(?:\.\d+)?))?")
_CLOCK_PAT = re.compile(r"\b(\d{1,2}):(\d{2})\b")
# "314-321s", "~1498s": some bullets give video seconds without a t= prefix
_SEC_PAT = re.compile(r"~?(\d{1,5}(?:\.\d+)?)\s*(?:-|–|—|to)?\s*(\d{1,5}(?:\.\d+)?)?\s*s\b")
_HERO_PAT = re.compile(r"[Hh]ero(?:ic)?[ -]([A-Z][A-Za-z.'’]*(?:\s+[A-Z][A-Za-z.'’]*){0,2})")


def load_card_index() -> tuple[dict[str, float], dict[str, str]]:
    """{slug: elixir cost}, {lowercase card name: slug}."""
    cards = json.loads(CARD_INDEX.read_text())["cards"]
    costs: dict[str, float] = {}
    names: dict[str, str] = {}
    for c in cards:
        slug = c["slug"]
        try:
            costs[slug] = float(c.get("elixir_cost"))
        except (TypeError, ValueError):
            pass
        for n in (c.get("name"), c.get("title")):
            if n:
                names[n.lower()] = slug
        names[slug.replace("-", " ")] = slug
    return costs, names


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    head, body = text[3:end], text[end + 4:]
    fm: dict[str, str] = {}
    for line in head.splitlines():
        if line[:1] in (" ", "\t", "") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        fm[k.strip()] = v.strip()
    return fm, body


def flow_list(value: str | None) -> list[str]:
    if not value:
        return []
    v = value.strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    return [x.strip().strip("'\"") for x in v.split(",") if x.strip()]


def key_moment_bullets(body: str) -> list[str]:
    """The `## Key moments` bullets, continuation lines folded into one string."""
    m = re.search(r"^##+\s*Key moments\s*$(.*?)(^##+\s|\Z)", body, re.S | re.M)
    if not m:
        return []
    out: list[str] = []
    cur: list[str] | None = None
    for line in m.group(1).splitlines():
        if line.startswith("- "):
            if cur:
                out.append(" ".join(cur))
            cur = [line[2:].strip()]
        elif cur is not None and line.strip() and line.startswith((" ", "\t")):
            cur.append(line.strip())
        elif cur is not None and line.strip():
            cur.append(line.strip())
        elif cur is not None:
            out.append(" ".join(cur))
            cur = None
    if cur:
        out.append(" ".join(cur))
    return [b.strip() for b in out if b.strip()]


def parse_bullet(bullet: str) -> dict:
    """{t_start, t_end, clock, text} - t_start/t_end None when the bullet has no t=."""
    head = bullet[:140]
    clock = None
    mc = _CLOCK_PAT.search(head)
    if mc:
        clock = f"{int(mc.group(1))}:{mc.group(2)}"
    mt = _T_PAT.search(head)
    t0 = float(mt.group(1)) if mt else None
    t1 = float(mt.group(2)) if (mt and mt.group(2)) else None
    alt = []
    if t0 is None:
        for m in _SEC_PAT.finditer(head):
            alt.append((float(m.group(1)), float(m.group(2)) if m.group(2) else None))
    return {"t_start": t0, "t_end": t1, "clock": clock, "text": bullet, "seconds_alt": alt}


def hero_cards_for(body: str, deck: list[str], hero_note: str | None,
                   names: dict[str, str]) -> list[str]:
    deck_set = set(deck)
    found: list[str] = []
    for m in _HERO_PAT.finditer(body):
        words = m.group(1).split()
        for n in (3, 2, 1):
            if len(words) < n:
                continue
            slug = names.get(" ".join(words[:n]).lower())
            if slug and slug in deck_set and slug not in found:
                found.append(slug)
                break
    if hero_note:
        mh = re.match(r"\s*(.+?)\s+as a Hero", hero_note)
        if mh:
            slug = names.get(mh.group(1).strip().lower())
            name = mh.group(1).strip()
            if slug and slug in deck_set and slug not in found:
                pat = re.compile(rf"[Hh]ero(?:ic)?[^.\n]{{0,60}}{re.escape(name)}"
                                 rf"|{re.escape(name)}[^.\n]{{0,60}}[Hh]ero", re.I)
                if pat.search(body):
                    found.append(slug)
    return found


def read_match_file(key: str, hero_note: str | None, names: dict[str, str]) -> dict:
    path = MATCHES / f"{key}.md"
    if not path.exists():
        return {"path": path, "exists": False, "deck": [], "deck_key": None,
                "result": None, "bullets": [], "hero_cards": []}
    text = path.read_text()
    fm, body = split_frontmatter(text)
    deck = flow_list(fm.get("own_deck"))
    result = (fm.get("result") or "").strip().strip("'\"") or None
    if result not in ("win", "loss", "draw", "unknown", None):
        result = "unknown"
    return {"path": path, "exists": True, "deck": deck,
            "deck_key": (fm.get("own_deck_key") or "").strip() or None,
            "result": result,
            "bullets": [parse_bullet(b) for b in key_moment_bullets(body)],
            "hero_cards": hero_cards_for(body, deck, hero_note, names)}


# --------------------------------------------------------------------------- small utilities

def fmt_num(x) -> str:
    if x is None:
        return "?"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def pad4(hand) -> list:
    hand = list(hand or [])[:4]
    return hand + [None] * (4 - len(hand))


def towers_of(d) -> dict:
    d = d or {}
    return {"king": d.get("king"), "left": d.get("left"), "right": d.get("right")}


def norm_phase(phase) -> tuple[str | None, str | None]:
    """Schema phase + a quality note when the pipeline value had to be folded."""
    if not phase:
        return None, None
    if phase in PHASES:
        return phase, None
    if phase.endswith("_overtime") or phase == "overtime":
        return "overtime", f"phase '{phase}' mapped to overtime"
    return None, f"phase '{phase}' not in schema enum"


def lane_of(tile) -> str | None:
    if not tile:
        return None
    col = tile[0]
    if col <= 5:
        return "left"
    if col >= 12:
        return "right"
    return "middle"


def zone_of(tile) -> str | None:
    if not tile:
        return None
    row = tile[1]
    if row <= 4:
        return "back"
    if row <= 11:
        return "mid"
    if row <= 16:
        return "bridge"
    return "enemy_half"


def play_record(ev: dict, costs: dict[str, float]) -> dict:
    card = ev.get("card")
    return {"t": round(float(ev["timestamp"]), 3), "player": ev.get("player"),
            "card": card, "tile": ev.get("tile"),
            "elixir_cost": costs.get(card) if card else None,
            "detect_source": ev.get("detect_source"), "confidence": ev.get("confidence")}


def unit_from_timeline(u: dict) -> dict:
    return {"unit": u.get("class") or "unknown_unit", "side": u.get("side") or "enemy",
            "tile": list(u.get("tile") or [0, 0])[:2],
            "heading": u.get("heading"), "speed_tiles_s": u.get("speed"),
            "predicted_tile_2s": u.get("pred_2s"), "eta_tower": u.get("eta_tower")}


def unit_from_state(u: dict) -> dict:
    m = u.get("motion") or {}
    return {"unit": u.get("class") or "unknown_unit", "side": u.get("side") or "enemy",
            "tile": list(u.get("tile") or m.get("tile") or [0, 0])[:2],
            "heading": m.get("heading"), "speed_tiles_s": m.get("speed"),
            "predicted_tile_2s": m.get("pred_2s"), "eta_tower": m.get("eta_tower")}


def render_threats(units: list[dict]) -> list[str]:
    """Same shape the pipeline renders in the timeline, for states.jsonl rows."""
    out = []
    for u in units:
        eta = u.get("eta_tower")
        if u.get("side") != "enemy" or not isinstance(eta, dict):
            continue
        secs = eta.get("s")
        if not str(eta.get("tower", "")).startswith("own") or secs is None or secs > 8.0:
            continue
        head = u.get("heading") or "approaching"
        out.append(f"{u['unit']}(e) {head}, tower in {secs}s")
    return out


# --------------------------------------------------------------------------- states.jsonl access

_STATE_T = re.compile(rb'"type":\s*"state".{0,40}?"t":\s*(-?\d+(?:\.\d+)?)')


def states_path_for(context_json: Path) -> Path | None:
    for p in (context_json.parent / "states.jsonl", context_json.parent.parent / "states.jsonl"):
        if p.exists():
            return p
    return None


def fetch_state_rows(path: Path, targets: list[float]) -> dict[float, dict]:
    """Nearest `state` row (readiness=match) for each target instant, streaming once."""
    if not targets:
        return {}
    ts = sorted(set(targets))
    best: dict[float, tuple[float, dict]] = {}
    with path.open("rb") as fh:
        for raw in fh:
            m = _STATE_T.search(raw[:120])
            if not m:
                continue
            t = float(m.group(1))
            i = bisect.bisect_left(ts, t)
            for j in (i - 1, i):
                if not (0 <= j < len(ts)):
                    continue
                dt = abs(ts[j] - t)
                if dt > STATE_TOL_S:
                    continue
                cur = best.get(ts[j])
                if cur is None or dt < cur[0]:
                    try:
                        rec = json.loads(raw)
                    except ValueError:
                        continue
                    if rec.get("readiness") != "match" or not rec.get("own"):
                        continue
                    best[ts[j]] = (dt, rec)
    return {k: v[1] for k, v in best.items()}


# --------------------------------------------------------------------------- per-game extraction

class Game:
    def __init__(self, key: str, context_json: Path, ctx: dict, match: dict,
                 costs: dict[str, float]):
        self.key, self.context_json, self.ctx, self.match, self.costs = key, context_json, ctx, match, costs
        period = ctx.get("period") or [ctx.get("start_t"), ctx.get("end_t")]
        self.t0 = float(ctx.get("start_t") if ctx.get("start_t") is not None else period[0])
        self.t1 = float(ctx.get("end_t") if ctx.get("end_t") is not None else period[1])
        self.timeline = sorted(ctx.get("timeline") or [], key=lambda r: r["t"])
        self.tl_ts = [r["t"] for r in self.timeline]
        self.events = sorted((e for e in ctx.get("events") or [] if e.get("timestamp") is not None),
                             key=lambda e: e["timestamp"])
        self.own_events = [e for e in self.events if e.get("player") == "own"]
        self.transcript = sorted(ctx.get("transcript") or [], key=lambda c: c.get("t", 0))
        self.deck = match["deck"] or (ctx.get("own_deck_video") or {}).get("deck") or ctx.get("own_deck_observed") or []
        self.deck_key = match["deck_key"] or ctx.get("own_deck_key")
        self.clock_read = any(r.get("clock") for r in self.timeline)
        self.states_path = states_path_for(context_json)
        self.state_rows: dict[float, dict] = {}
        self.unmatched: list[str] = []
        # HUD hand reads are noisy: when the 8-card deck is known, anything outside
        # it is a misread. Cards read from an in-game deploy label are trusted.
        self.deck_set = set(self.deck) if len(set(self.deck)) >= 8 else None
        self.stats = {"hand_slots_nulled": 0, "next_card_nulled": 0,
                      "hud_plays_dropped": 0, "hud_play_entries_dropped": 0,
                      "actions_unknown": 0}
        self.stats["hud_plays_dropped"] = sum(1 for e in self.own_events if self.foreign_hud(e))

    def foreign(self, card) -> bool:
        """The card is not one of the 8 he is playing (only judged when the deck is known)."""
        return self.deck_set is not None and card is not None and card not in self.deck_set

    def foreign_hud(self, ev: dict) -> bool:
        return ev.get("detect_source") == "hud" and self.foreign(ev.get("card"))

    def own_plays_in(self, lo: float, hi: float) -> list[dict]:
        """Own plays in (lo, hi], minus HUD reads of cards he does not play."""
        out = []
        for e in self.own_events:
            if not (lo < e["timestamp"] <= hi):
                continue
            if self.foreign_hud(e):
                self.stats["hud_play_entries_dropped"] += 1
                continue
            out.append(play_record(e, self.costs))
        return out

    def sanitize_hand(self, own: dict, notes: list[str]) -> None:
        if self.deck_set is None:
            return
        for i, card in enumerate(own["hand"]):
            if self.foreign(card):
                notes.append(f"hand slot {i} read as {card} (not in deck)")
                own["hand"][i] = None
                self.stats["hand_slots_nulled"] += 1
        if self.foreign(own.get("next_card")):
            notes.append(f"next card read as {own['next_card']} (not in deck)")
            own["next_card"] = None
            self.stats["next_card_nulled"] += 1

    # -- sample instants ---------------------------------------------------
    def clamp(self, t: float) -> float:
        return round(min(max(t, self.t0), self.t1), 1)

    def timeline_row(self, t: float) -> dict | None:
        if not self.timeline:
            return None
        i = bisect.bisect_right(self.tl_ts, t) - 1
        if i < 0:
            i = 0
        return self.timeline[i]

    def t_for_clock(self, clock: str | None) -> float | None:
        if not clock:
            return None
        for r in self.timeline:
            if r.get("clock") == clock:
                return float(r["t"])
        return None

    def plan(self):
        """Sample specs: {t, kind, key_moment?, event?}. Later kinds never collide
        with earlier ones (key > play > periodic)."""
        specs: list[dict] = []
        key_ts: list[float] = []
        taken: list[float] = []

        for b in self.match["bullets"]:
            t_start, t_end = b["t_start"], b["t_end"]
            if t_start is None:
                t_start = self.t_for_clock(b["clock"])
            if t_start is None:  # bullets that give plain video seconds ("314-321s")
                for cand, cand_end in b.get("seconds_alt") or []:
                    if self.t0 <= cand <= self.t1:
                        t_start, t_end = cand, cand_end
                        break
            if t_start is None:
                self.unmatched.append(b["text"][:120])
                continue
            t = self.clamp(t_start - PLAY_LEAD_S)
            if any(abs(t - x) < 0.05 for x in taken):
                continue
            taken.append(t)
            key_ts.append(t)
            specs.append({"t": t, "kind": "key",
                          "key_moment": {"t_start": round(float(t_start), 3),
                                         "t_end": round(float(t_end), 3) if t_end is not None else None,
                                         "text": b["text"], "clock": b["clock"]}})

        for ev in self.own_events:
            t = self.clamp(float(ev["timestamp"]) - PLAY_LEAD_S)
            if any(abs(t - x) <= PLAY_NEAR_KEY_S for x in key_ts):
                continue
            if any(abs(t - x) < 0.05 for x in taken):
                continue
            taken.append(t)
            specs.append({"t": t, "kind": "play", "event": ev})

        return specs, taken, key_ts

    def periodic_specs(self, period: float, taken: list[float]) -> list[dict]:
        specs = []
        t = self.t0
        n = 0
        while t <= self.t1 + 1e-9:
            tt = self.clamp(t)
            if not any(abs(tt - x) <= NEAR_S for x in taken):
                taken.append(tt)
                specs.append({"t": tt, "kind": "periodic"})
            n += 1
            t = self.t0 + n * period
        return specs

    # -- state -------------------------------------------------------------
    def opponent_seen(self, t: float) -> list[str]:
        seen: list[str] = []
        for e in self.events:
            if e["timestamp"] > t:
                break
            if e.get("player") == "opponent" and e.get("card") and e["card"] not in seen:
                seen.append(e["card"])
        return seen

    def build_state(self, t: float) -> tuple[dict, str, list[str]]:
        notes: list[str] = []
        rec = self.state_rows.get(round(t, 1))
        row = self.timeline_row(t)
        if rec is not None:
            source = "states.jsonl"
            own, opp = rec.get("own") or {}, rec.get("opponent") or {}
            phase, note = norm_phase(rec.get("phase"))
            confs = [c for c in (own.get("hand_conf") or []) if isinstance(c, (int, float))]
            units = [unit_from_state(u) for u in (rec.get("units") or [])
                     if u.get("category") != "tower" and u.get("class") not in TOWER_CLASSES]
            threats = render_threats(units)
            state = {
                "clock": rec.get("match_clock"), "phase": phase,
                "match_seconds": rec.get("match_seconds"),
                "own": {"elixir": own.get("elixir"), "hand": pad4(own.get("hand")),
                        "hand_confidence": round(sum(confs) / len(confs), 3) if confs else None,
                        "next_card": own.get("next_card"), "deck": list(self.deck),
                        "deck_key": self.deck_key,
                        "hero_cards": list(self.match["hero_cards"]),
                        "towers": towers_of(own.get("towers")), "recent_plays": []},
                "opponent": {"elixir_estimate": opp.get("elixir_est"),
                             "towers": towers_of(opp.get("towers")),
                             "deck_known": list(opp.get("deck_known") or []),
                             "deck_complete": bool(opp.get("deck_complete")),
                             "recent_plays": []},
                "units": units, "threats": threats}
        else:
            source = "context.timeline"
            row = row or {}
            phase, note = norm_phase(row.get("phase"))
            units = [unit_from_timeline(u) for u in (row.get("units") or [])]
            seen = self.opponent_seen(t)
            clock = row.get("clock")
            state = {
                "clock": clock, "phase": phase,
                "match_seconds": clock_to_seconds(clock, phase),
                "own": {"elixir": row.get("own_elixir"), "hand": pad4(row.get("hand")),
                        "hand_confidence": None, "next_card": row.get("next"),
                        "deck": list(self.deck), "deck_key": self.deck_key,
                        "hero_cards": list(self.match["hero_cards"]),
                        "towers": towers_of(row.get("towers_own")), "recent_plays": []},
                "opponent": {"elixir_estimate": row.get("opp_elixir_est"),
                             "towers": towers_of(row.get("towers_enemy")),
                             "deck_known": seen, "deck_complete": len(seen) >= 8,
                             "recent_plays": []},
                "units": units, "threats": list(row.get("threats") or [])}
            if not self.timeline:
                notes.append("no timeline rows for this game")
        if note:
            notes.append(note)
        if state["own"]["elixir"] is not None:
            state["own"]["elixir"] = max(0.0, min(10.0, float(state["own"]["elixir"])))
        self.sanitize_hand(state["own"], notes)
        state["own"]["recent_plays"] = self.own_plays_in(t - RECENT_S - 1e-9, t)
        state["opponent"]["recent_plays"] = [play_record(e, self.costs) for e in self.events
                                             if t - RECENT_S <= e["timestamp"] <= t and e.get("player") == "opponent"]
        return state, source, notes

    # -- action / outcome --------------------------------------------------
    def next_own_play(self, t: float, window: float):
        for e in self.own_events:
            if t < e["timestamp"] <= t + window:
                return e
        return None

    def build_action(self, t: float, event: dict | None, horizon: float,
                     notes: list[str]) -> dict:
        ev = event if event is not None else self.next_own_play(t, ACTION_WINDOW_S)
        if ev is None:
            return {"type": "hold", "card": None, "tile": None, "lane": None, "zone": None,
                    "delay_s": None, "elixir_before": None, "elixir_after": None,
                    "detect_source": None, "confidence": None,
                    "window_s": ACTION_WINDOW_S,
                    "following_plays": self.own_plays_in(t, t + horizon)}
        te = float(ev["timestamp"])
        tile = ev.get("tile")
        following = self.own_plays_in(te, t + horizon)
        if self.foreign_hud(ev):
            notes.append(f"action read as {ev.get('card')} by the HUD (not in deck): type unknown")
            self.stats["actions_unknown"] += 1
            return {"type": "unknown", "card": None, "tile": tile,
                    "lane": lane_of(tile), "zone": zone_of(tile),
                    "delay_s": round(te - t, 2), "elixir_before": ev.get("elixir_before"),
                    "elixir_after": ev.get("elixir_after"), "detect_source": ev.get("detect_source"),
                    "confidence": ev.get("confidence"), "window_s": ACTION_WINDOW_S,
                    "following_plays": following}
        return {"type": "play", "card": ev.get("card"), "tile": tile,
                "lane": lane_of(tile), "zone": zone_of(tile),
                "delay_s": round(te - t, 2), "elixir_before": ev.get("elixir_before"),
                "elixir_after": ev.get("elixir_after"), "detect_source": ev.get("detect_source"),
                "confidence": ev.get("confidence"), "window_s": ACTION_WINDOW_S,
                "following_plays": following}

    def tower_gone(self, side_key: str, tower: str, t: float) -> bool:
        """True when the tower reads a HP value at/before t and never again after t."""
        before = [r for r in self.timeline if r["t"] <= t and (r.get(side_key) or {}).get(tower) is not None]
        after = [r for r in self.timeline if r["t"] > t]
        if not before or len(after) < 3:
            return False
        return all((r.get(side_key) or {}).get(tower) is None for r in after)

    def build_outcome(self, t: float, horizon: float) -> dict:
        r0 = self.timeline_row(t)
        t_end = min(t + horizon, self.t1)
        r1 = self.timeline_row(t_end)

        def delta(side_key: str):
            if not r0 or not r1:
                return None
            out = {}
            for k in ("king", "left", "right"):
                a = (r0.get(side_key) or {}).get(k)
                b = (r1.get(side_key) or {}).get(k)
                out[k] = round(float(b) - float(a), 1) if (a is not None and b is not None) else None
            return out

        own_d, enemy_d = delta("towers_own"), delta("towers_enemy")
        lost = [k for k in ("left", "right", "king") if self.tower_gone("towers_own", k, t)
                and not self.tower_gone("towers_own", k, t - horizon)]
        taken = [k for k in ("left", "right", "king") if self.tower_gone("towers_enemy", k, t)
                 and not self.tower_gone("towers_enemy", k, t - horizon)]

        own_spend = sum(self.costs.get(e.get("card"), 0.0) for e in self.events
                        if t < e["timestamp"] <= t + horizon and e.get("player") == "own"
                        and not self.foreign_hud(e))  # misread HUD cards are not priced
        opp_spend = sum(self.costs.get(e.get("card"), 0.0) for e in self.events
                        if t < e["timestamp"] <= t + horizon and e.get("player") == "opponent")
        own_dmg = -sum(v for v in (own_d or {}).values() if v is not None and v < 0)
        enemy_dmg = -sum(v for v in (enemy_d or {}).values() if v is not None and v < 0)
        swing = enemy_dmg - own_dmg
        if taken or swing > 300:
            verdict = "positive"
        elif lost or swing < -300:
            verdict = "negative"
        else:
            verdict = "neutral"
        if taken and lost:
            verdict = "neutral"
        return {"horizon_s": horizon, "own_tower_hp_delta": own_d, "enemy_tower_hp_delta": enemy_d,
                "towers_lost": lost, "towers_taken": taken,
                "own_elixir_end": (r1 or {}).get("own_elixir"),
                "opponent_plays": [play_record(e, self.costs) for e in self.events
                                   if t < e["timestamp"] <= t + horizon and e.get("player") == "opponent"],
                "threats_after": list((r1 or {}).get("threats") or []),
                "elixir_trade": round(opp_spend - own_spend, 1),
                "verdict": verdict, "game_result": self.match["result"] or "unknown"}

    def commentary(self, t: float) -> list[dict]:
        out = []
        for c in self.transcript:
            ct = float(c.get("t", 0))
            if abs(ct - t) <= COMMENTARY_S:
                out.append({"t": round(ct, 2), "text": (c.get("text") or "").strip()})
        return out

    def context_refs(self, state: dict, action: dict) -> dict:
        slugs: list[str] = []
        for s in list(state["own"]["hand"]) + [state["own"].get("next_card"), action.get("card")]:
            if s and s not in slugs:
                slugs.append(s)
        for u in state["units"]:
            s = to_card_slug(u["unit"])
            if s and s not in slugs:
                slugs.append(s)
        cards = [f"knowledge_base/cards/{s}.md" for s in slugs if (CARDS_DIR / f"{s}.md").exists()]
        deck_file = None
        if self.deck_key and (DECKS_DIR / f"{self.deck_key}.md").exists():
            deck_file = f"knowledge_base/decks/{self.deck_key}.md"
        opp_file = None
        best = None
        for m in (self.ctx.get("opponent") or {}).get("kb_matches") or []:
            if m.get("matched", 0) >= 3 and m.get("score", -99) > 0 and (best is None or m["score"] > best["score"]):
                best = m
        if best and (DECKS_DIR / f"{best['deck_key']}.md").exists():
            opp_file = f"knowledge_base/decks/{best['deck_key']}.md"
        return {"match_file": f"knowledge_base/matches/{self.key}.md", "deck_file": deck_file,
                "cards": cards, "opponent_deck_file": opp_file}

    def build_sample(self, spec: dict, horizon: float) -> dict:
        t = spec["t"]
        state, source, notes = self.build_state(t)
        action = self.build_action(t, spec.get("event"), horizon, notes)
        outcome = self.build_outcome(t, horizon)
        sample = {
            "id": f"{self.key}#{t:.1f}", "schema_version": SCHEMA_VERSION,
            "source": {"video_id": self.ctx["video_id"], "match_index": str(self.ctx["match_index"]),
                       "match_file": f"knowledge_base/matches/{self.key}.md",
                       "video_title": self.ctx.get("title") or "", "creator": CREATOR,
                       "video_url": self.ctx.get("url") or ""},
            "kind": spec["kind"], "t": t, "state": state, "action": action, "outcome": outcome,
            "commentary": self.commentary(t),
            "state_text": state_text(state, action, t),
            "context_refs": self.context_refs(state, action),
            "quality": {"clock_read": bool(self.clock_read),
                        "hand_confidence": (self.ctx.get("quality") or {}).get("hand_conf_mean"),
                        "calibration": self.ctx.get("calibration_method") or "unknown",
                        "state_source": source, "notes": notes},
        }
        if spec["kind"] == "key":
            sample["key_moment"] = spec["key_moment"]
        return sample

    def samples(self, period: float, horizon: float) -> list[dict]:
        specs, taken, _ = self.plan()
        specs = specs + self.periodic_specs(period, taken)
        if self.states_path is not None:
            self.state_rows = fetch_state_rows(self.states_path, [round(s["t"], 1) for s in specs])
        out, seen = [], set()
        for spec in sorted(specs, key=lambda s: (s["t"], {"key": 0, "play": 1, "periodic": 2}[s["kind"]])):
            sample = self.build_sample(spec, horizon)
            if sample["id"] in seen:
                continue
            seen.add(sample["id"])
            out.append(sample)
        return out


def clock_to_seconds(clock: str | None, phase: str | None):
    """m:ss remaining -> seconds elapsed (regulation 3:00, overtime counts on)."""
    if not clock:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", clock)
    if not m:
        return None
    remaining = int(m.group(1)) * 60 + int(m.group(2))
    if phase == "overtime":
        return 180 + max(0, 120 - remaining)
    return max(0, 180 - remaining)


# --------------------------------------------------------------------------- state_text

def state_text(state: dict, action: dict, t: float | None = None) -> str:
    own, opp = state["own"], state["opponent"]
    hand = ", ".join(x or "?" for x in own["hand"])
    line1 = (f"[{state['phase'] or 'unknown'}|{state['clock'] or '?'}] "
             f"elixir {fmt_num(own['elixir'])} (opp ~{fmt_num(opp['elixir_estimate'])}). "
             f"hand: {hand} (next {own.get('next_card') or '?'}). "
             f"deck: {', '.join(own['deck']) if own['deck'] else 'unknown'}")
    field = []
    for u in state["units"]:
        s = f"{u['unit']}({'a' if u['side'] == 'ally' else 'e'})@[{fmt_num(u['tile'][0])},{fmt_num(u['tile'][1])}]"
        if u.get("heading"):
            s += f" {u['heading']}"
        eta = u.get("eta_tower")
        if isinstance(eta, dict) and eta.get("tower") is not None:
            s += f" eta {eta['tower']} {fmt_num(eta.get('s'))}s"
        field.append(s)
    line2 = "field: " + ("; ".join(field) if field else "none")
    line3 = "threats: " + ("; ".join(state["threats"]) if state["threats"] else "none")
    line4_items = []
    for tag, plays in (("own", own["recent_plays"]), ("opp", opp["recent_plays"])):
        for p in plays:
            tile = f"@[{fmt_num(p['tile'][0])},{fmt_num(p['tile'][1])}]" if p.get("tile") else ""
            ago = f" {round(t - p['t'], 1)}s ago" if t is not None else ""
            line4_items.append(f"{tag} {p['card'] or '?'}{tile}{ago}")
    line4 = "recent: " + ("; ".join(line4_items) if line4_items else "none")
    ot, et = own["towers"], opp["towers"]
    line5 = (f"towers own {fmt_num(ot['left'])}/{fmt_num(ot['right'])}/{fmt_num(ot['king'])}, "
             f"enemy {fmt_num(et['left'])}/{fmt_num(et['right'])}/{fmt_num(et['king'])}")
    if action["type"] == "play":
        tile = action.get("tile")
        where = f" at [{fmt_num(tile[0])},{fmt_num(tile[1])}]" if tile else ""
        placement = f" ({action.get('zone') or '?'} {action.get('lane') or '?'})" if tile else ""
        delay = f" after {fmt_num(action.get('delay_s'))}s" if action.get("delay_s") is not None else ""
        line6 = f"action: play {action.get('card') or '?'}{where}{placement}{delay}"
    else:
        line6 = f"action: {action['type']}"
    return "\n".join([line1, line2, line3, line4, line5, line6])


# --------------------------------------------------------------------------- io

def existing_enrichment(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("enrichment"):
            out[rec["id"]] = rec["enrichment"]
    return out


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = existing_enrichment(path)
    lines = []
    for s in samples:
        if s["id"] in keep:
            s["enrichment"] = keep[s["id"]]
        lines.append(json.dumps(s, sort_keys=False, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


FLAGS = ("--only", "--periodic", "--horizon", "--out-dir")


def pull_only(argv: list[str]) -> tuple[list[str], list[str]]:
    """argparse cannot take `--only -V4H_YeMGGk-m0.0` (keys may start with '-'),
    so the --only values are pulled out of argv by hand."""
    if "--only" not in argv:
        return argv, []
    i = argv.index("--only")
    j = i + 1
    keys: list[str] = []
    while j < len(argv) and argv[j] not in FLAGS:
        keys.append(argv[j])
        j += 1
    return argv[:i] + argv[j:], keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=[], help="only these <video>-m<index> keys")
    ap.add_argument("--periodic", type=float, default=10.0, help="periodic sample interval (s)")
    ap.add_argument("--horizon", type=float, default=15.0, help="outcome horizon (s)")
    ap.add_argument("--out-dir", default=str(STATES_DIR))
    argv, only = pull_only(sys.argv[1:])
    a = ap.parse_args(argv)
    a.only = only or a.only

    costs, names = load_card_index()
    out_dir = Path(a.out_dir)
    totals = {"key": 0, "play": 0, "periodic": 0}
    sanitised: dict[str, int] = {}
    unmatched_total = 0
    games = [g for g in ready() if not a.only or g["key"] in a.only]
    if a.only:
        missing = sorted(set(a.only) - {g["key"] for g in games})
        for k in missing:
            print(f"! no context for {k}", file=sys.stderr)
    for g in games:
        ctx_path = Path(g["context_json"])
        ctx = json.loads(ctx_path.read_text())
        key = g["key"]
        match = read_match_file(key, ctx.get("hero_note"), names)
        game = Game(key, ctx_path, ctx, match, costs)
        samples = game.samples(a.periodic, a.horizon)
        write_jsonl(out_dir / f"{key}.jsonl", samples)
        counts = {k: sum(1 for s in samples if s["kind"] == k) for k in totals}
        for k, v in counts.items():
            totals[k] += v
        unmatched_total += len(game.unmatched)
        for k, v in game.stats.items():
            sanitised[k] = sanitised.get(k, 0) + v
        print(json.dumps({"key": key, "counts": counts, "total": len(samples),
                          "key_bullets_unmatched": len(game.unmatched),
                          "state_source": "states.jsonl" if game.states_path else "context.timeline",
                          "match_file": match["exists"], "sanitised": game.stats}))
        for u in game.unmatched:
            print(f"  ! unmatched key bullet: {u}", file=sys.stderr)
    print(json.dumps({"games": len(games), **totals, "total": sum(totals.values()),
                      "key_bullets_unmatched": unmatched_total, "sanitised": sanitised}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
