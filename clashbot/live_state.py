"""Live state builder (D-015): perception GameState + PlayEvents -> the clean `state` dict of dataset/v1/states.jsonl.

Re-implements, for one live instant, the dataset path `tools/extract_states.py::Game.build_state`
(states.jsonl branch) followed by `tools/data/build_clean_dataset.py::clean_sample`, so retrieval,
features, candidates and d1 requests see live states the way they saw training states. The small
helpers are private copies (the runtime package must not import build scripts);
tests/test_live_state.py proves they match the originals.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from clashbot.game_rules import is_playable, load_cards
from clashbot.state_text import render_context, render_threats
from clashbot.units import load_unit_map, resolve

try:  # pure class vocabulary; fall back to the literal set when cv2 is absent
    from cr_perception.detect import TOWER_CLASSES
except Exception:  # pragma: no cover
    TOWER_CLASSES = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
                     "royal-chef-tower"}

RECENT_S = 12.0                     # same as tools/extract_states.py
SCHEMA_PHASES = ("single_elixir", "double_elixir", "triple_elixir", "overtime")
MAX_OPP_DECK = 8
_UNSET = object()                   # "no match seen yet" (distinct from a match_id of None)


@dataclass
class LiveConfig:
    own_deck: list[str]                       # the 8 card slugs the user plays
    hero_cards: list[str] = field(default_factory=list)
    recent_s: float = RECENT_S
    hand_hold_s: float = 2.0                  # reuse the last confident hand this long when all 4 slots are unread; 0 = off


# --------------------------------------------------------------------------- private helpers
# Copies of tools/extract_states.py and tools/data/build_clean_dataset.py (same output, same key order).

def _pad4(hand) -> list:
    hand = list(hand or [])[:4]
    return hand + [None] * (4 - len(hand))


def _towers_of(d) -> dict:
    d = d or {}
    return {"king": d.get("king"), "left": d.get("left"), "right": d.get("right")}


def _norm_phase(phase) -> str | None:
    if not phase:
        return None
    if phase in SCHEMA_PHASES:
        return phase
    if phase.endswith("_overtime") or phase == "overtime":
        return "overtime"
    return None


def _play_record(ev: dict, cards: dict) -> dict:
    card = ev.get("card")
    return {"t": round(float(ev["timestamp"]), 3), "player": ev.get("player"),
            "card": card, "tile": ev.get("tile"),
            "elixir_cost": cards[card]["elixir_cost"] if card in cards else None,
            "detect_source": ev.get("detect_source"), "confidence": ev.get("confidence")}


def _clean_plays(plays, cards: dict) -> list[dict]:
    out = []
    for p in plays or []:
        q = dict(p)
        card = q.get("card")
        if not (card and is_playable(card, cards)):
            q["card"] = None
        out.append(q)
    return out


def _unit_from_state(u: dict) -> dict:
    m = u.get("motion") or {}
    return {"unit": u.get("class") or "unknown_unit", "side": u.get("side") or "enemy",
            "tile": list(u.get("tile") or m.get("tile") or [0, 0])[:2],
            "heading": m.get("heading"), "speed_tiles_s": m.get("speed"),
            "predicted_tile_2s": m.get("pred_2s"), "eta_tower": m.get("eta_tower")}


def _clean_units(raw_units: list[dict], deck_status: str, own_deck: list[str], opp_deck: list[str],
                 cards: dict, unit_map: dict) -> list[dict]:
    units = []
    own_set, opp_set = set(own_deck), set(opp_deck)
    for u in raw_units or []:
        r = resolve(u["unit"], cards, unit_map)
        side, repaired, eta = u["side"], False, u.get("eta_tower")
        if deck_status != "invalid":
            cand = set(r["candidates"])
            own_ok, opp_ok = bool(cand & own_set), bool(cand & opp_set)
            if side == "ally" and not own_ok and opp_ok:
                side, repaired = "enemy", True
            elif side == "enemy" and not opp_ok and own_ok:
                side, repaired = "ally", True
            if repaired:
                eta = None
        units.append({"unit_class": u["unit"], "card": r["card"], "is_evo": r["is_evo"],
                      "side": side, "side_repaired": repaired, "tile": u.get("tile"),
                      "heading": u.get("heading"), "speed_tiles_s": u.get("speed_tiles_s"),
                      "predicted_tile_2s": u.get("predicted_tile_2s"), "eta_tower": eta})
    return units


def _as_json(obj) -> dict:
    return obj.to_json() if hasattr(obj, "to_json") else obj


# --------------------------------------------------------------------------- builder

class LiveStateBuilder:
    """Stateful: remembers the play events and when each opponent card was first seen, per match."""

    def __init__(self, config: LiveConfig, cards: dict | None = None, unit_map: dict | None = None):
        self.config = config
        self.cards = load_cards() if cards is None else cards
        self.unit_map = load_unit_map() if unit_map is None else unit_map
        self.deck = sorted({c for c in config.own_deck if is_playable(c, self.cards)})
        self.deck_status = ("valid" if len(self.deck) == 8 else
                            "incomplete" if len(self.deck) == 7 else "invalid")
        # HUD hand reads are noisy: with the 8-card deck known, anything outside it is a misread.
        self.deck_set = set(config.own_deck) if len(set(config.own_deck)) >= 8 else None
        self.reset()

    def reset(self) -> None:
        """Forget the current match: events, opponent first-seen times, last match_id."""
        self._events: list[dict] = []          # normalised play-event dicts, sorted by timestamp
        self._event_keys: set[tuple] = set()
        self._opp_first: dict[str, float] = {}  # playable opponent card -> first-seen time
        self._match_id = _UNSET
        self._last_hand: tuple[float, list] | None = None  # (t, hand) of the last read with >= 2 known slots
        self.last_hand_reused = False          # the hand of the latest build() came from _last_hand

    # -- inputs ------------------------------------------------------------
    def observe_events(self, events: list) -> None:
        for e in events or []:
            ev = dict(_as_json(e))
            ts = ev.get("timestamp")
            if ts is None:
                continue
            ev["timestamp"] = ts = float(ts)
            # the tile keeps two unidentified plays (card None) at the same instant apart
            tile = ev.get("tile")
            key = (round(ts, 1), ev.get("player"), ev.get("card"), tuple(tile) if tile else None)
            if key in self._event_keys:
                continue
            self._event_keys.add(key)
            if ev.get("tile") is not None:
                ev["tile"] = list(ev["tile"])
            bisect.insort(self._events, ev, key=lambda x: x["timestamp"])
            card = ev.get("card")
            if ev.get("player") == "opponent" and card and is_playable(card, self.cards):
                self._opp_first[card] = min(self._opp_first.get(card, ts), ts)

    def _foreign(self, card) -> bool:
        """The card is not one of the 8 he is playing (only judged when the deck is known)."""
        return self.deck_set is not None and card is not None and card not in self.deck_set

    def _plays(self, player: str, t: float, drop_foreign_hud: bool) -> list[dict]:
        lo = t - self.config.recent_s - 1e-9
        out = []
        for e in self._events:
            if e.get("player") != player or not (lo <= e["timestamp"] <= t):
                continue
            if drop_foreign_hud and e.get("detect_source") == "hud" and self._foreign(e.get("card")):
                continue
            out.append(_play_record(e, self.cards))
        return _clean_plays(out, self.cards)

    def _hold_hand(self, hand: list, t: float) -> list:
        """Greyed (unaffordable) cards are missed by the matcher: when all four slots read None, reuse the
        last confident hand for at most `hand_hold_s`, minus cards played since (they left the hand)."""
        known = sum(c is not None for c in hand)
        if known >= 2:
            self._last_hand = (t, list(hand))
            return hand
        hold = self.config.hand_hold_s
        if known or hold <= 0 or self._last_hand is None:
            return hand
        t_last, last = self._last_hand
        if not 0.0 <= t - t_last <= hold:
            return hand
        played = {e.get("card") for e in self._events
                  if e.get("player") == "own" and e["timestamp"] > t_last}
        reused = [None if c in played else c for c in last]
        self.last_hand_reused = any(c is not None for c in reused)
        return reused if self.last_hand_reused else hand

    # -- output ------------------------------------------------------------
    def build(self, gs) -> tuple[dict, str] | None:
        g = _as_json(gs)
        self.last_hand_reused = False
        if g.get("readiness") != "match" or not g.get("own"):
            return None
        mid = g.get("match_id")
        if self._match_id is not _UNSET and mid != self._match_id:
            self.reset()                       # new match: forget the old one's events
        self._match_id = mid
        cards = self.cards
        t = float(g["t"])
        own, opp = g.get("own") or {}, g.get("opponent") or {}

        confs = [c for c in (own.get("hand_conf") or []) if isinstance(c, (int, float))]
        elixir = own.get("elixir")
        if elixir is not None:
            elixir = max(0.0, min(10.0, float(elixir)))
        hand = [None if self._foreign(c) else c for c in _pad4(own.get("hand"))]
        hand = [c if (c and is_playable(c, cards)) else None for c in hand]
        hand = self._hold_hand(hand, t)
        confidence = None if self.last_hand_reused else (round(sum(confs) / len(confs), 3) if confs else None)
        nxt = own.get("next_card")
        nxt = None if self._foreign(nxt) else nxt
        nxt = nxt if (nxt and is_playable(nxt, cards)) else None

        for c in opp.get("deck_known") or []:  # the tracker's own list counts from the first t it shows
            if is_playable(c, cards):
                self._opp_first[c] = min(self._opp_first.get(c, t), t)
        seen = sorted((c for c, ts in self._opp_first.items() if ts <= t),
                      key=lambda c: (self._opp_first[c], c))[:MAX_OPP_DECK]

        raw_units = [_unit_from_state(u) for u in (g.get("units") or [])
                     if u.get("category") != "tower" and u.get("class") not in TOWER_CLASSES]
        units = _clean_units(raw_units, self.deck_status, self.deck, seen, cards, self.unit_map)

        state = {
            "clock": g.get("match_clock"), "phase": _norm_phase(g.get("phase")),
            "match_seconds": g.get("match_seconds"),
            "own": {
                "elixir": elixir,
                "hand_confidence": confidence,
                "hero_cards": list(self.config.hero_cards), "towers": _towers_of(own.get("towers")),
                "hand": hand, "next_card": nxt, "deck": list(self.deck),
                "deck_status": self.deck_status,
                "recent_plays": self._plays("own", t, drop_foreign_hud=True),
            },
            "opponent": {
                "elixir_estimate": opp.get("elixir_est"), "towers": _towers_of(opp.get("towers")),
                "deck_known": seen, "deck_complete": len(seen) == MAX_OPP_DECK,
                "recent_plays": self._plays("opponent", t, drop_foreign_hud=False),
            },
            "units": units,
            "threats": render_threats(units),
        }
        return state, render_context(state, t)
