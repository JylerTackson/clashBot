"""Mine the pro's answers to opponent plays (MAC-AK K2; supersedes build_kg.py::answered_by_edges).

  python3 tools/data/mine_pro_answers.py [--dataset DIR] [--out DIR] [--no-unit-threats]

Reads <dataset>/{states.jsonl,games.jsonl,act_dense/states_<split>.jsonl}; writes into <out> (default <dataset>/kg2):
  answer_events.jsonl  one line per (opponent play -> first defensive own play), all splits
  answers.jsonl        per (threat, answer) aggregate, train split only
  situations.jsonl     per situation key (`situation_key`), train split only
  sequences.jsonl      play -> next play within 5 s, own and opponent, train split only
and prints the sanity report.

Plays come from states.jsonl only (`recent_plays`, the play `action`, `following_plays`); the 1 Hz dense sidecar
states only supply tracked units and tower HP between plays. Tiles are [col, row], own side = low rows.
`situation_key` is a pure top-level function (K1 imports it).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.card_roles import load_roles  # noqa: E402
from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.game_rules import lane_of, load_cards  # noqa: E402
from clashbot.units import resolve  # noqa: E402

# --- answer events
WINDOW_S = 8.0                  # the answer is the first defensive own play in [P.t, P.t + WINDOW_S]
DEFENSIVE_MAX_ROW = 16          # a non-spell answer lands on our half or the river
SPELL_NEAR_TILES = 4.0          # ... or a spell landing this close to the threat's tracked position
TRACK_TOL_S = 1.0               # a snapshot this close to the answer time gives the threat's tracked tile
DEDUP_S = 1.5                   # same card within this many seconds and DEDUP_TILES = the same play
DEDUP_TILES = 3.0
PERSIST_S = 3.5                 # deploy labels linger and are re-read every ~3 s: a repeat inside this is the same play
CENTRE_X = 9.0                  # threat x >= this is the right half: its offsets are mirrored
OWN_PRINCESS = ((3.5, 4.3), (14.5, 4.3))    # own princess towers (cr_perception/tracking.py::TOWER_TILES)
OWN_HALF_MAX_ROW = 16           # an enemy unit at or below this row is on our side of the arena
BOARD_ETA_S = 10.0              # ... or arrives within this many seconds
ELIXIR_MAX_AGE_S = 2.5          # the elixir / board read for an answer is the last snapshot this recent
CROSS_QUIET_S = 6.0             # unit-threat: the card was not on our half in this many seconds before
CROSS_EXPLAINED_S = 20.0        # ... and no opponent play of it in this many seconds before
CROSS_RADIUS = 4.0              # units of one card within this of each other are one group
# --- held
HELD_WINDOW_S = 12.0            # the princess tower in the threat's lane loses < HELD_MAX_LOSS HP in this window
HELD_MAX_LOSS = 300.0
HELD_START_AGE_S = 5.0          # the starting HP reading is at most this old
HELD_MIN_COVER_S = 6.0          # and some reading is at least this far after the answer
# --- aggregates
MIN_SUPPORT = 3                 # answers.jsonl keeps pairs with at least this many events
MIN_SITUATION_N = 2             # situations.jsonl keeps keys seen this often
SEQ_WINDOW_S = 5.0
SEQ_MIN_SUPPORT = 3
# --- situation key
ETA_BANDS = ("0-3", "3-6", "6-10", ">10")
ELIXIR_BANDS = ("0-3", "4-6", "7-10")
LANE_RELS = ("same", "split", "centre")
BUILDING_TARGETERS = ("giant", "hog-rider", "balloon", "royal-giant", "ram-rider", "battle-ram")

PRIO = {"action": 0, "deploy_label": 1, "hud": 2}      # which detection's tile a merged play keeps


# --------------------------------------------------------------------------- cards
_CARDS: dict | None = None
_ROLES: dict | None = None


def _cards() -> dict:
    global _CARDS
    if _CARDS is None:
        _CARDS = load_cards()
    return _CARDS


def _roles() -> dict:
    global _ROLES
    if _ROLES is None:
        _ROLES = load_roles()
    return _ROLES


def card_cost(card: str | None) -> float | None:
    c = _cards().get(card) if card else None
    return None if c is None else c.get("elixir_cost")


def card_kind(card: str | None) -> str | None:
    """troop | building | spell (card_roles type, else the card table's type)."""
    if not card:
        return None
    r = _roles().get(card)
    if r:
        return r.get("type")
    c = _cards().get(card)
    return c["card_type"].lower() if c and c.get("card_type") else None


def is_threat_card(card: str | None) -> bool:
    """Every card is a threat except spells that are not win conditions (zap, fireball, tornado, ...): no card is
    played to answer those. Graveyard and goblin-barrel are spells *and* win conditions, so they stay."""
    if card_kind(card) != "spell":
        return True
    return "win_condition" in (_roles().get(card) or {}).get("roles", [])


# --------------------------------------------------------------------------- situation key (K1 imports this)

def eta_band(eta: float | None) -> str:
    if eta is None or eta > 10:
        return ">10"
    if eta < 3:
        return "0-3"
    if eta < 6:
        return "3-6"
    return "6-10"


def elixir_band(elixir: float | None) -> str:
    if elixir is None:
        return "?"
    if elixir < 4:
        return "0-3"
    if elixir < 7:
        return "4-6"
    return "7-10"


def top_threat_cards(threat_cards, k: int = 2) -> list[str]:
    """The k distinct cards with the highest elixir cost (ties: name), sorted by name."""
    uniq = {c for c in threat_cards if c}
    top = sorted(uniq, key=lambda c: (-(card_cost(c) or 0.0), c))[:k]
    return sorted(top)


def lane_rel_of(tiles) -> str:
    """centre = every threat is in the middle band, split = threats in both side lanes, else same (one lane)."""
    lanes = [lane_of(t) for t in tiles if t]
    if not lanes or all(l == "middle" for l in lanes):
        return "centre" if lanes else "same"
    return "split" if ("left" in lanes and "right" in lanes) else "same"


def situation_key(threat_cards, lane_rel: str, eta: float | None, elixir: float | None) -> str:
    """'<card>+<card>|<lane_rel>|<eta band>|<elixir band>', e.g. 'giant+witch|same|3-6|4-6'. The cards are the top 2
    distinct threats by elixir cost (sorted); eta = seconds until the soonest of them hits a tower (None = > 10)."""
    return "|".join(["+".join(top_threat_cards(threat_cards)), lane_rel, eta_band(eta), elixir_band(elixir)])


def parse_situation_key(key: str) -> dict:
    cards, lane_rel, eta, elixir = key.split("|")
    return {"key": key, "threat_cards": cards.split("+"), "lane_rel": lane_rel, "eta_band": eta,
            "elixir_band": elixir}


# --------------------------------------------------------------------------- geometry
def _xy(tile) -> tuple[float, float] | None:
    if tile is None or len(tile) < 2 or tile[0] is None or tile[1] is None:
        return None
    return (float(tile[0]), float(tile[1]))


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mirror_offset(answer_tile, threat_tile) -> tuple[float, float, bool]:
    """(dx, dy, mirrored): answer - threat, with dx negated when the threat is in the right half (x >= 9) so that
    an offset toward the centre of the arena is positive. dy is not touched (negative = in front of the threat)."""
    dx, dy = answer_tile[0] - threat_tile[0], answer_tile[1] - threat_tile[1]
    mirrored = threat_tile[0] >= CENTRE_X
    return (-dx if mirrored else dx), dy, mirrored


def dist_to_own_princess(tile) -> float:
    return min(_dist(tile, p) for p in OWN_PRINCESS)


def lane_name(tile) -> str:
    """Which own princess tower defends a threat at `tile`."""
    return "right" if tile[0] >= CENTRE_X else "left"


# --------------------------------------------------------------------------- snapshots
@dataclass(frozen=True)
class EnemyUnit:
    cands: frozenset             # cards this unit class may come from
    card: str | None             # states.jsonl's resolved card
    tile: tuple[float, float]
    eta: float | None            # seconds to the tower it is heading for


@dataclass(frozen=True)
class Snap:
    t: float
    own_elixir: float | None
    opp_elixir: float | None
    tower_left: float | None     # own princess tower HP
    tower_right: float | None
    units: tuple


_CAND_CACHE: dict = {}


def unit_cands(unit_class: str | None, card: str | None) -> frozenset:
    key = (unit_class, card)
    if key not in _CAND_CACHE:
        c = set(resolve(unit_class)["candidates"]) if unit_class else set()
        if card:
            c.add(card)
        _CAND_CACHE[key] = frozenset(c)
    return _CAND_CACHE[key]


def snap_from_state(t: float, state: dict) -> Snap:
    own, opp = state.get("own") or {}, state.get("opponent") or {}
    towers = own.get("towers") or {}
    units = []
    for u in state.get("units") or []:
        xy = _xy(u.get("tile"))
        if u.get("side") != "enemy" or xy is None:
            continue
        eta = u.get("eta_tower")
        units.append(EnemyUnit(unit_cands(u.get("unit_class"), u.get("card")), u.get("card"), xy,
                               eta.get("s") if isinstance(eta, dict) else None))
    return Snap(float(t), own.get("elixir"), opp.get("elixir_estimate"), towers.get("left"), towers.get("right"),
                tuple(units))


def _median3(values: list[float]) -> list[float]:
    """3-point median: kills a one-sample OCR glitch in a tower HP series (4858, 658, 4858)."""
    if len(values) < 3:
        return list(values)
    out = [values[0]]
    out += [sorted(values[i - 1:i + 2])[1] for i in range(1, len(values) - 1)]
    out.append(values[-1])
    return out


class Timeline:
    """All snapshots of one game (dense + sparse) in time order, with the tower HP series per lane."""

    def __init__(self, snaps: list[Snap]):
        self.snaps = sorted(snaps, key=lambda s: s.t)
        self.times = [s.t for s in self.snaps]
        self.tower: dict[str, tuple[list[float], list[float]]] = {}
        for lane, attr in (("left", "tower_left"), ("right", "tower_right")):
            pts = [(s.t, getattr(s, attr)) for s in self.snaps
                   if isinstance(getattr(s, attr), (int, float)) and not isinstance(getattr(s, attr), bool)]
            self.tower[lane] = ([p[0] for p in pts], _median3([float(p[1]) for p in pts]))

    def near(self, t: float, tol: float) -> list[Snap]:
        lo, hi = bisect_left(self.times, t - tol), bisect_right(self.times, t + tol)
        return sorted(self.snaps[lo:hi], key=lambda s: abs(s.t - t))

    def before(self, t: float, max_age: float, need=None) -> Snap | None:
        """The latest snapshot at or before t (at most max_age old) for which need(snap) is true."""
        i = bisect_right(self.times, t) - 1
        while i >= 0 and t - self.times[i] <= max_age:
            if need is None or need(self.snaps[i]):
                return self.snaps[i]
            i -= 1
        return None

    def track(self, card: str, t: float, ref_tile=None, tol: float = TRACK_TOL_S):
        """(tile, eta, snapshot t) of the enemy units of `card` in the snapshot nearest t (within tol): the group of
        matching units within CROSS_RADIUS of the one closest to ref_tile; tile = the group's mean. None if absent."""
        for s in self.near(t, tol):
            match = [u for u in s.units if u.card == card] or [u for u in s.units if card in u.cands]
            if not match:
                continue
            pivot = min(match, key=lambda u: _dist(u.tile, ref_tile)) if ref_tile else match[0]
            group = [u for u in match if _dist(u.tile, pivot.tile) <= CROSS_RADIUS]
            tile = (sum(u.tile[0] for u in group) / len(group), sum(u.tile[1] for u in group) / len(group))
            etas = [u.eta for u in group if u.eta is not None]
            return tile, (min(etas) if etas else None), s.t
        return None

    def held(self, lane: str, t0: float) -> bool | None:
        """The own princess tower of `lane` loses < HELD_MAX_LOSS HP within HELD_WINDOW_S after t0. None = unknown."""
        ts, hp = self.tower[lane]
        i = bisect_right(ts, t0) - 1
        if i < 0 or t0 - ts[i] > HELD_START_AGE_S:
            return None
        j = bisect_right(ts, t0 + HELD_WINDOW_S)
        if j <= i + 1 or ts[j - 1] < t0 + HELD_MIN_COVER_S:
            return None
        return (hp[i] - min(hp[i + 1:j])) < HELD_MAX_LOSS


# --------------------------------------------------------------------------- plays
def dedupe_plays(raw: list[dict]) -> tuple[list[dict], Counter]:
    """Collapse detections of the same play. raw: {t, card, tile (xy|None), prio?}. Detections of one card join a play
    when the gap to its previous detection is <= PERSIST_S and the tiles are within DEDUP_TILES (an unknown tile
    matches). Gaps <= DEDUP_S count as `dup` (the same play read in several records), longer ones as `persist`
    (a deploy label that lingers and is read again every ~3 s). The play keeps the earliest t and the tile of the
    best detection (action > deploy_label > hud > arena)."""
    stats: Counter = Counter()
    plays: list[dict] = []
    open_by_card: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(raw, key=lambda d: (d["t"], d.get("prio", 9))):
        tile = _xy(d.get("tile"))
        hit = None
        for p in open_by_card[d["card"]]:      # oldest first: the newest matching play wins
            if d["t"] - p["last_t"] <= PERSIST_S and (tile is None or p["anchor"] is None
                                                      or _dist(tile, p["anchor"]) <= DEDUP_TILES):
                hit = p
        if hit is None:
            p = {"t": d["t"], "card": d["card"], "tile": tile, "anchor": tile, "last_t": d["t"],
                 "prio": d.get("prio", 9) if tile else 99, "n": 1}
            plays.append(p)
            open_by_card[d["card"]].append(p)
            continue
        stats["dup" if d["t"] - hit["last_t"] <= DEDUP_S else "persist"] += 1
        hit["last_t"] = max(hit["last_t"], d["t"])
        hit["n"] += 1
        if tile is not None and d.get("prio", 9) < hit["prio"]:
            hit["tile"], hit["prio"] = tile, d.get("prio", 9)
        if hit["anchor"] is None and tile is not None:
            hit["anchor"] = tile
    return [{"t": p["t"], "card": p["card"], "tile": p["tile"], "n": p["n"]} for p in plays], stats


def _prio(src: str | None) -> int:
    return PRIO.get(src or "", 3)


def raw_plays(records: list[dict], who: str) -> list[dict]:
    """Every detection of `who` ('own' | 'opponent') plays in the sparse records: recent_plays, plus for the pro the
    play action (at t + delay_s) and following_plays."""
    out = []
    for r in records:
        for p in (r["state"].get(who) or {}).get("recent_plays") or []:
            out.append({"t": p["t"], "card": p.get("card"), "tile": p.get("tile"), "prio": _prio(p.get("detect_source"))})
        a = r.get("action") or {}
        if who == "own" and a.get("type") == "play" and a.get("card"):
            out.append({"t": r["t"] + (a.get("delay_s") or 0.0), "card": a["card"], "tile": a.get("tile"), "prio": 0})
        for f in a.get("following_plays") or []:
            if f.get("player") == who:
                out.append({"t": f["t"], "card": f.get("card"), "tile": f.get("tile"),
                            "prio": _prio(f.get("detect_source"))})
    return out


def clean_plays(raw: list[dict], stats: Counter, prefix: str, deck: list[str] | None = None) -> list[dict]:
    """Drop detections with no card (inferred plays) or a card outside the table (or outside `deck`, if given),
    then dedupe. Counts under `<prefix>_...` in stats."""
    stats[f"{prefix}_detections"] += len(raw)
    kept = []
    for d in raw:
        if not d["card"]:
            stats[f"{prefix}_unknown_card"] += 1
        elif d["card"] not in _cards():
            stats[f"{prefix}_card_not_in_table"] += 1
        elif deck is not None and d["card"] not in deck:
            stats[f"{prefix}_off_deck"] += 1
        else:
            kept.append(d)
    plays, dd = dedupe_plays(kept)
    stats[f"{prefix}_dup_collapsed"] += dd["dup"]
    stats[f"{prefix}_persist_collapsed"] += dd["persist"]
    stats[f"{prefix}_plays"] += len(plays)
    stats[f"{prefix}_plays_no_tile"] += sum(1 for p in plays if p["tile"] is None)
    return sorted(plays, key=lambda p: p["t"])


# --------------------------------------------------------------------------- threats
def unit_card(u: EnemyUnit, deck: set, strict: bool = True) -> str | None:
    """The card an enemy unit stands for: the single candidate in the opponent's deck. Otherwise (several candidates
    in the deck, e.g. skeletons from a witch or a graveyard) the unit's resolved card, unless strict."""
    both = u.cands & deck
    if len(both) == 1:
        return next(iter(both))
    return None if strict else u.card


def unit_threats(tl: Timeline, opp_plays: list[dict], opp_deck: list[str]) -> list[dict]:
    """Opponent cards that show up on our half without any recorded play of them: a card is a threat the first time
    a unit of it is on rows <= OWN_HALF_MAX_ROW after CROSS_QUIET_S without one, and no play of it was seen in the
    CROSS_EXPLAINED_S before. A unit class is read as a card only when exactly one of its candidates (or its resolved
    card) is in the opponent's deck, so spawns (skeletons, goblins, ...) are credited to the card that made them."""
    deck = set(opp_deck)
    if len(deck) < 7:
        return []
    last_on_our_half: dict[str, float] = {}
    out = []
    for s in tl.snaps:
        here: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for u in s.units:
            if u.tile[1] > OWN_HALF_MAX_ROW:
                continue
            card = unit_card(u, deck)
            if card:
                here[card].append(u.tile)
        for card, tiles in here.items():
            quiet = s.t - last_on_our_half.get(card, -1e9) > CROSS_QUIET_S
            explained = any(p["card"] == card and s.t - CROSS_EXPLAINED_S <= p["t"] <= s.t + 1.0 for p in opp_plays)
            if quiet and not explained and _cards().get(card):
                tile = (sum(x for x, _ in tiles) / len(tiles), sum(y for _, y in tiles) / len(tiles))
                out.append({"t": s.t, "card": card, "tile": tile, "n": 1})
            last_on_our_half[card] = s.t
    return out


# --------------------------------------------------------------------------- events of one game
def _round(x, n=3):
    return None if x is None else round(float(x), n)


def mine_game(game: dict, records: list[dict], dense: list[Snap], stats: Counter,
              use_unit_threats: bool = True) -> tuple[list[dict], dict]:
    """Answer events of one game and its cleaned plays {'own': [...], 'opp': [...]} (for the sequences).
    game: a games.jsonl row; records: its states.jsonl rows (sorted by t); dense: Snaps of its 1 Hz sidecar."""
    gid, split = game["game_id"], game["split"]
    tl = Timeline(dense + [snap_from_state(r["t"], r["state"]) for r in records])
    own_deck = game.get("own_deck") or []
    own_valid = game.get("deck_status") == "valid"
    opp = clean_plays(raw_plays(records, "opponent"), stats, "opp")
    if game.get("deck_status") == "invalid":       # the pro's cards cannot be told from misreads
        stats["games_own_deck_invalid"] += 1
        own = []
    else:
        own = clean_plays(raw_plays(records, "own"), stats, "own", own_deck if own_valid else None)
    plays = {"own": own, "opp": opp}

    threats = []
    for p in opp:
        if is_threat_card(p["card"]):
            threats.append(dict(p, source="play"))
        else:
            stats["threat_dropped_nonthreat_spell"] += 1
    if use_unit_threats:
        ut = unit_threats(tl, opp, game.get("opponent_deck") or [])
        stats["threats_from_units"] += len(ut)
        threats += [dict(p, source="unit") for p in ut]
    stats["threats"] += len(threats)

    events = []
    for P in sorted(threats, key=lambda p: p["t"]):
        cand = [a for a in own if P["t"] <= a["t"] <= P["t"] + WINDOW_S]
        found = None
        for A in cand:
            a_tile = _xy(A["tile"])
            if a_tile is None:
                stats["answer_candidates_no_tile"] += 1
                continue
            tr = tl.track(P["card"], A["t"], _xy(P["tile"]))
            pos = tr[0] if tr else _xy(P["tile"])
            if a_tile[1] <= DEFENSIVE_MAX_ROW:
                defensive = True
            else:
                defensive = card_kind(A["card"]) == "spell" and pos is not None and _dist(a_tile, pos) <= SPELL_NEAR_TILES
            if defensive:
                found = (A, a_tile, tr, pos)
                break
        if found is None:
            stats["threats_unanswered"] += 1
            continue
        A, a_tile, tr, pos = found
        if pos is None:
            stats["events_dropped_threat_tile_missing"] += 1
            continue
        dx, dy, mirrored = mirror_offset(a_tile, pos)
        dt = A["t"] - P["t"]
        tracked = tr is not None
        lane = lane_name(pos)
        eta = tr[1] if tracked else None
        el = tl.before(A["t"], ELIXIR_MAX_AGE_S, lambda s: s.own_elixir is not None)
        elo = tl.before(A["t"], ELIXIR_MAX_AGE_S, lambda s: s.opp_elixir is not None)
        tc, ac = card_cost(P["card"]), card_cost(A["card"])
        sit = _situation(tl, game, P, pos, eta, el.own_elixir if el else None, A["t"])
        events.append({
            "game_id": gid, "split": split, "threat": P["card"], "threat_source": P["source"],
            "threat_t": _round(P["t"]), "threat_tile": P["tile"] and list(P["tile"]),
            "answer": A["card"], "answer_t": _round(A["t"]), "answer_tile": list(a_tile),
            "answer_kind": card_kind(A["card"]), "dt_s": _round(dt),
            "threat_tile_at_answer": [_round(pos[0]), _round(pos[1])],
            "threat_tile_src": "tracked" if tracked else "play",
            "offset": [_round(dx), _round(dy)], "mirrored": mirrored, "lane": lane,
            # the offset is relative to where the threat stood at the answer time: only trusted when we saw it there,
            # or when it had hardly had time to move
            "offset_ok": tracked or dt <= 2.0,
            "eta_at_answer": _round(eta), "dist_at_answer": _round(dist_to_own_princess(pos)),
            "own_elixir": _round(el.own_elixir if el else None, 2),
            "opp_elixir_est": _round(elo.opp_elixir if elo else None, 2),
            "held": tl.held(lane, A["t"]),
            "threat_cost": tc, "answer_cost": ac, "trade": None if tc is None or ac is None else tc - ac,
            "situation": sit,
        })
    stats["events"] += len(events)
    return events, plays


def _situation(tl: Timeline, game: dict, P: dict, pos, eta, elixir, a_t: float) -> dict:
    """Situation just before the answer: the enemy threats on the board (rows <= 16 or ETA <= 10 s) plus the threat
    itself; the top 2 by elixir cost give the key."""
    deck = set(game.get("opponent_deck") or [])
    board = [(P["card"], pos, eta)]
    s = tl.before(a_t, ELIXIR_MAX_AGE_S)
    for u in (s.units if s else ()):
        if u.tile[1] > OWN_HALF_MAX_ROW and not (u.eta is not None and u.eta <= BOARD_ETA_S):
            continue
        card = unit_card(u, deck, strict=False)
        if card in _cards():
            board.append((card, u.tile, u.eta))
    top = top_threat_cards([b[0] for b in board])
    tiles: dict[str, tuple[float, float]] = {}
    etas = []
    for card, tile, e in board:
        if card in top:
            tiles.setdefault(card, tile)
            if e is not None:
                etas.append(e)
    return parse_situation_key(situation_key(top, lane_rel_of(list(tiles.values())), min(etas) if etas else None,
                                             elixir))


# --------------------------------------------------------------------------- aggregates
def _med(xs):
    xs = [x for x in xs if x is not None]
    return _round(float(np.median(xs))) if xs else None


def _iqr(xs):
    xs = [x for x in xs if x is not None]
    return _round(float(np.percentile(xs, 75) - np.percentile(xs, 25))) if xs else None


def _held_rate(evs) -> tuple[int, int, float | None]:
    known = [e["held"] for e in evs if e["held"] is not None]
    h = sum(1 for x in known if x)
    return h, len(known), (_round(h / len(known)) if known else None)


def aggregate_answers(events: list[dict], min_support: int = MIN_SUPPORT) -> list[dict]:
    """Per (threat, answer), train events only."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in events:
        if e["split"] == "train":
            groups[(e["threat"], e["answer"])].append(e)
    out = []
    for (threat, answer), evs in groups.items():
        if len(evs) < min_support:
            continue
        held, held_n, rate = _held_rate(evs)
        off = [e["offset"] for e in evs if e["offset_ok"] and e["offset"]]
        trades = [e["trade"] for e in evs if e["trade"] is not None]
        out.append({
            "threat": threat, "answer": answer, "support": len(evs), "held": held, "held_n": held_n,
            "held_rate": rate, "mean_trade": _round(sum(trades) / len(trades)) if trades else None,
            "mean_dt_s": _round(sum(e["dt_s"] for e in evs) / len(evs)),
            "offset": {"dx_med": _med([o[0] for o in off]), "dy_med": _med([o[1] for o in off]),
                       "dx_iqr": _iqr([o[0] for o in off]), "dy_iqr": _iqr([o[1] for o in off]), "n": len(off)},
            "eta_at_answer_med": _med([e["eta_at_answer"] for e in evs]),
            "dist_at_answer_med": _med([e["dist_at_answer"] for e in evs]),
        })
    return sorted(out, key=lambda a: (-a["support"], a["threat"], a["answer"]))


def aggregate_situations(events: list[dict], min_n: int = MIN_SITUATION_N) -> list[dict]:
    """Per situation key, train events only. One answer play counts once, even if it answers several threats: the
    event of its most expensive threat represents it."""
    plays: dict[tuple, dict] = {}
    for e in events:
        if e["split"] != "train":
            continue
        k = (e["game_id"], e["answer_t"], e["answer"])
        if k not in plays or (e["threat_cost"] or 0) > (plays[k]["threat_cost"] or 0):
            plays[k] = e
    by_key: dict[str, list[dict]] = defaultdict(list)
    for e in plays.values():
        by_key[e["situation"]["key"]].append(e)
    out = []
    for key, evs in by_key.items():
        if len(evs) < min_n:
            continue
        by_ans: dict[str, list[dict]] = defaultdict(list)
        for e in evs:
            by_ans[e["answer"]].append(e)
        s = evs[0]["situation"]
        out.append({"key": key, "threat_cards": s["threat_cards"], "lane_rel": s["lane_rel"],
                    "eta_band": s["eta_band"], "elixir_band": s["elixir_band"], "n": len(evs),
                    "answers": {c: {"n": len(v), "held_rate": _held_rate(v)[2]}
                                for c, v in sorted(by_ans.items(), key=lambda kv: (-len(kv[1]), kv[0]))}})
    return sorted(out, key=lambda s: (-s["n"], s["key"]))


def sequence_pairs(plays: list[dict], window: float = SEQ_WINDOW_S) -> list[tuple[str, str, float]]:
    """(first, then, lag) for each play and the next one within `window` seconds (a card does not follow itself)."""
    ps = sorted(plays, key=lambda p: p["t"])
    return [(a["card"], b["card"], b["t"] - a["t"]) for a, b in zip(ps, ps[1:])
            if b["t"] - a["t"] <= window and a["card"] != b["card"]]


def aggregate_sequences(pairs: list[tuple[str, str, str, float]], min_support: int = SEQ_MIN_SUPPORT) -> list[dict]:
    """pairs: (who, first, then, lag) from train games only."""
    g: dict[tuple, list[float]] = defaultdict(list)
    for who, a, b, lag in pairs:
        g[(who, a, b)].append(lag)
    out = [{"who": w, "first": a, "then": b, "support": len(v), "lag_med_s": _round(float(np.median(v)))}
           for (w, a, b), v in g.items() if len(v) >= min_support]
    return sorted(out, key=lambda r: (-r["support"], r["who"], r["first"], r["then"]))


# --------------------------------------------------------------------------- io
def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_dense(ds: Path) -> dict[str, list[Snap]]:
    """game_id -> Snaps of the 1 Hz sidecar states (act_dense/states_<split>.jsonl; id = game_id#t)."""
    out: dict[str, list[Snap]] = defaultdict(list)
    for split in ("train", "val", "test"):
        path = ds / "act_dense" / f"states_{split}.jsonl"
        if not path.exists():
            continue
        for d in read_jsonl(path):
            out[d["id"].rsplit("#", 1)[0]].append(snap_from_state(d["t"], d["state"]))
    return out


def mine_dataset(ds: Path, use_unit_threats: bool = True):
    """(events, sequence pairs (train only), stats) over every game of games.jsonl."""
    games = {g["game_id"]: g for g in read_jsonl(ds / "games.jsonl")}
    dense = load_dense(ds)
    recs: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(ds / "states.jsonl"):
        recs[r["game_id"]].append({"t": r["t"], "state": r["state"], "action": r.get("action")})
    stats: Counter = Counter()
    events, pairs = [], []
    for gid, game in games.items():
        rs = sorted(recs.get(gid, []), key=lambda r: r["t"])
        stats["games"] += 1
        stats[f"games_{game['split']}"] += 1
        if not rs:
            stats["games_without_records"] += 1
            continue
        if gid not in dense:
            stats["games_without_dense"] += 1
        ev, plays = mine_game(game, rs, dense.get(gid, []), stats, use_unit_threats)
        events += ev
        if game["split"] == "train":
            for who, ps in plays.items():
                pairs += [(who, a, b, lag) for a, b, lag in sequence_pairs(ps)]
    return events, pairs, stats


# --------------------------------------------------------------------------- sanity report
STAT_LABELS = (
    ("games", "games in games.jsonl"),
    ("games_own_deck_invalid", "games whose own deck is invalid: no pro plays, no events (opponent plays still count)"),
    ("games_without_dense", "games without 1 Hz sidecar states (tracking and tower HP come from the sparse records)"),
    ("opp_detections", "opponent play detections read"),
    ("opp_unknown_card", "  dropped: card unknown (inferred / arena detections without a card)"),
    ("opp_card_not_in_table", "  dropped: card not in the card table"),
    ("opp_dup_collapsed", "  collapsed: same card within 1.5 s and 3 tiles (the same play read in several records)"),
    ("opp_persist_collapsed", "  collapsed: re-read of a lingering deploy label (1.5-3.5 s, same card and place)"),
    ("opp_plays", "opponent plays after dedupe"),
    ("opp_plays_no_tile", "  of which without a tile"),
    ("own_detections", "pro play detections read"),
    ("own_unknown_card", "  dropped: card unknown (inferred HUD reads)"),
    ("own_card_not_in_table", "  dropped: card not in the card table"),
    ("own_off_deck", "  dropped: card not in the pro's (valid) deck, a misread"),
    ("own_dup_collapsed", "  collapsed: same card within 1.5 s and 3 tiles (same play: action + HUD + label + records)"),
    ("own_persist_collapsed", "  collapsed: re-read of a lingering deploy label (1.5-3.5 s)"),
    ("own_plays", "pro plays after dedupe"),
    ("own_plays_no_tile", "  of which without a tile (cannot be judged defensive; skipped as answer candidates)"),
    ("threat_dropped_nonthreat_spell", "opponent plays dropped as threats: damage / utility spells (zap, fireball, tornado, ...)"),
    ("threats", "threats considered (opponent plays + unit crossings)"),
    ("threats_from_units", "  of which an enemy unit first seen on our half with no play of it (--no-unit-threats to drop)"),
    ("threats_unanswered", "threats with no defensive pro play in the 8 s window"),
    ("answer_candidates_no_tile", "pro plays inside a window skipped for having no tile"),
    ("events_dropped_threat_tile_missing", "events dropped: the threat has no tile and was not tracked"),
    ("events", "answer events written"),
)


def _dist_line(xs, label: str) -> str:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return f"  {label}: n=0"
    q = [float(np.percentile(xs, p)) for p in (10, 25, 50, 75, 90)]
    return f"  {label}: n={len(xs)} p10/25/50/75/90 = " + " / ".join(f"{v:+.1f}" for v in q)


def _hist(xs, edges, fmt="{:+.0f}") -> str:
    counts = [0] * (len(edges) + 1)
    for x in xs:
        counts[bisect_right(edges, x)] += 1
    labels = [f"<{fmt.format(edges[0])}"] + [f"[{fmt.format(a)},{fmt.format(b)})" for a, b in zip(edges, edges[1:])] \
        + [f">={fmt.format(edges[-1])}"]
    return "  " + "  ".join(f"{l}:{c}" for l, c in zip(labels, counts))


def sanity_report(events: list[dict], answers: list[dict], situations: list[dict], sequences: list[dict],
                  stats: Counter) -> str:
    L: list[str] = []
    train = [e for e in events if e["split"] == "train"]
    L.append("== answer events per split")
    by_split = Counter(e["split"] for e in events)
    for s in ("train", "val", "test"):
        L.append(f"  {s}: {by_split.get(s, 0)} events, {stats.get('games_' + s, 0)} games")
    L.append(f"  threat source: {dict(Counter(e['threat_source'] for e in events))}; threat tile at answer: "
             f"{dict(Counter(e['threat_tile_src'] for e in events))}; offset_ok: "
             f"{sum(1 for e in events if e['offset_ok'])}/{len(events)}")

    L.append("== data checks (all splits; raw detections are the entries of recent_plays / actions / following_plays)")
    for k, label in STAT_LABELS:
        L.append(f"  {stats.get(k, 0):>7}  {k}: {label}")

    L.append("== top 3 answers for the 10 most common threats (train)")
    tc = Counter(e["threat"] for e in train)
    by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for e in train:
        by_pair[(e["threat"], e["answer"])].append(e)
    for threat, n in tc.most_common(10):
        ans = sorted(((a, v) for (t, a), v in by_pair.items() if t == threat), key=lambda kv: -len(kv[1]))[:3]
        parts = []
        for a, v in ans:
            h, hn, r = _held_rate(v)
            parts.append(f"{a} n={len(v)} held={'-' if r is None else f'{r:.2f}'}({hn})")
        L.append(f"  {threat} (n={n}): " + "; ".join(parts))

    L.append("== building answers vs building-targeter threats (train, offset_ok): offset in the canonical lane")
    L.append("   dx > 0 = toward the centre, dy < 0 = in front of the threat")
    bld = [e for e in train if e["answer_kind"] == "building" and e["threat"] in BUILDING_TARGETERS and e["offset_ok"]]
    L.append(_dist_line([e["offset"][0] for e in bld], "dx"))
    L.append(_dist_line([e["offset"][1] for e in bld], "dy"))
    if bld:
        L.append(f"  dx>0 (toward centre): {sum(1 for e in bld if e['offset'][0] > 0) / len(bld):.0%}   "
                 f"dx==0: {sum(1 for e in bld if e['offset'][0] == 0) / len(bld):.0%}   "
                 f"dy<0 (in front): {sum(1 for e in bld if e['offset'][1] < 0) / len(bld):.0%}")
        L.append("  dx hist" + _hist([e["offset"][0] for e in bld], [-6, -3, -1, 1, 3, 6]))
        L.append("  dy hist" + _hist([e["offset"][1] for e in bld], [-12, -8, -4, -1, 1, 4]))
        for t in BUILDING_TARGETERS:
            sub = [e for e in bld if e["threat"] == t]
            if sub:
                L.append(f"  {t}: n={len(sub)} dx_med={float(np.median([e['offset'][0] for e in sub])):+.1f} "
                         f"dy_med={float(np.median([e['offset'][1] for e in sub])):+.1f}")
    allk = [e for e in train if e["answer_kind"] == "building" and e["offset_ok"]]
    L.append(f"  (all building answers, any threat: n={len(allk)})")

    L.append("== held")
    held = Counter("unknown" if e["held"] is None else ("held" if e["held"] else "tower hit") for e in train)
    L.append(f"  train events: {dict(held)}")
    rates = [a["held_rate"] for a in answers if a["held_rate"] is not None]
    L.append(f"  held_rate of {len(rates)} aggregated (threat, answer) pairs, support >= {MIN_SUPPORT}: "
             + _hist(rates, [0.2, 0.4, 0.6, 0.8, 1.0], "{:.1f}").strip())
    if rates:
        L.append(f"  mean {float(np.mean(rates)):.2f}  median {float(np.median(rates)):.2f}")

    L.append(f"== aggregates: {len(answers)} answer pairs (support>={MIN_SUPPORT}), {len(situations)} situations "
             f"(n>={MIN_SITUATION_N}), {len(sequences)} sequences (support>={SEQ_MIN_SUPPORT})")
    L.append("== example situations")
    for s in situations[:6]:
        top = list(s["answers"].items())[:3]
        L.append(f"  {s['key']}  n={s['n']}: " + "; ".join(
            f"{c} n={v['n']} held={'-' if v['held_rate'] is None else v['held_rate']}" for c, v in top))
    L.append("== example sequences")
    for who in ("own", "opp"):
        for q in [q for q in sequences if q["who"] == who][:4]:
            L.append(f"  {who}: {q['first']} -> {q['then']}  support={q['support']} lag_med={q['lag_med_s']}s")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--out", type=Path, default=None, help="default: <dataset>/kg2")
    ap.add_argument("--no-unit-threats", action="store_true", help="only opponent plays are threats")
    args = ap.parse_args(argv)
    out = args.out or args.dataset / "kg2"
    out.mkdir(parents=True, exist_ok=True)

    events, pairs, stats = mine_dataset(args.dataset, use_unit_threats=not args.no_unit_threats)
    events.sort(key=lambda e: (e["split"], e["game_id"], e["threat_t"], e["threat"]))
    answers = aggregate_answers(events)
    situations = aggregate_situations(events)
    sequences = aggregate_sequences(pairs)
    write_jsonl(out / "answer_events.jsonl", events)
    write_jsonl(out / "answers.jsonl", answers)
    write_jsonl(out / "situations.jsonl", situations)
    write_jsonl(out / "sequences.jsonl", sequences)
    print(sanity_report(events, answers, situations, sequences, stats))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
