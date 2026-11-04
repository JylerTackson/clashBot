"""Reactive act/hold policy (D-041): save elixir, answer threats in time, leak at a full bar, punish an overspent opponent.

`ReactivePolicy.decide` says whether to play now (and in which lane); `filter_candidates` narrows the candidate
generator's output to moves that fit that decision (no blind spells). `clashbot.loop.should_decide` stays as the
replay trigger and the `--policy trigger` fallback of the live loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from clashbot.game_rules import lane_of, legal_cards, load_cards, zone_of

DEFEND_ETA_S = 4.0          # an enemy unit this close (seconds) to one of our towers must be answered now
LEAK_ELIXIR = 9.5           # at (near) full elixir, waiting wastes elixir
PUNISH_OPP_ELIXIR = 3.0     # opponent estimated this low after spending -> push the other lane
PUNISH_OWN_ELIXIR = 6.0
OWN_HALF_MAX_ROW = 15       # rows 0..15 are our side of the river
NEAR_BACK_ROW = 8           # an enemy unit without ETA this deep in our half is as urgent as one at the tower
DEFEND_ZONES = ("back", "mid", "bridge")
LEAK_ZONES = ("back", "bridge")
PUNISH_ZONES = ("bridge", "enemy_half")
OPPOSITE_LANE = {"left": "right", "right": "left"}       # middle has no opposite


@dataclass
class ActDecision:
    act: bool
    mode: str                 # "defend" | "leak" | "punish" | "hold" | "none"
    lane: str | None          # lane to play in (None = any)
    reason: str               # short human-readable reason, logged
    threat: dict | None = None   # the enemy unit being answered (defend)


def _tile(unit: dict) -> list | None:
    tile = unit.get("tile")
    return tile if isinstance(tile, (list, tuple)) and len(tile) >= 2 else None


def _enemy_units(state: dict) -> list[dict]:
    return [u for u in state.get("units", []) if u.get("side") == "enemy"]


def _own_eta(unit: dict) -> tuple[float | None, str]:
    """(seconds, tower name) to one of our towers, or (None, "")."""
    eta = unit.get("eta_tower")
    if isinstance(eta, dict) and str(eta.get("tower", "")).startswith("own") and eta.get("s") is not None:
        return float(eta["s"]), str(eta["tower"])
    return None, ""


def _tower_lane(tower: str) -> str | None:
    return {"own_left": "left", "own_right": "right", "own_king": "middle"}.get(tower)


class _Cand(NamedTuple):
    obj: Any                  # the Candidate (or dict) as given
    card: str | None          # card / lane / zone are None for keys that are not `card|lane|zone`
    lane: str | None
    zone: str | None
    sources: dict


def _parse(cand) -> _Cand:
    key = cand["key"] if isinstance(cand, dict) else cand.key
    sources = (cand.get("sources") if isinstance(cand, dict) else cand.sources) or {}
    parts = str(key).split("|")
    return _Cand(cand, *(parts if len(parts) == 3 else (None, None, None)), sources)


class ReactivePolicy:
    def __init__(self, cards: dict | None = None, defend_eta_s: float = DEFEND_ETA_S,
                 leak_elixir: float = LEAK_ELIXIR, punish_opp_elixir: float = PUNISH_OPP_ELIXIR,
                 punish_own_elixir: float = PUNISH_OWN_ELIXIR):
        self.cards = cards if cards is not None else load_cards()
        self.defend_eta_s = defend_eta_s
        self.leak_elixir = leak_elixir
        self.punish_opp_elixir = punish_opp_elixir
        self.punish_own_elixir = punish_own_elixir

    # -- decide --------------------------------------------------------------------------------------------------------
    def _threats(self, state: dict) -> list[tuple[float, dict, str | None]]:
        """(eta, unit, lane) of every enemy unit that must be answered: close to our towers or already on our half."""
        out: list[tuple[float, dict, str | None]] = []
        for u in _enemy_units(state):
            tile = _tile(u)
            eta, tower = _own_eta(u)
            on_our_half = tile is not None and tile[1] <= OWN_HALF_MAX_ROW
            if eta is not None and eta <= self.defend_eta_s:
                pass
            elif on_our_half:
                if eta is None:
                    eta = 0.0 if tile[1] <= NEAR_BACK_ROW else 2.0
            else:
                continue
            out.append((eta, u, lane_of(tile) or _tower_lane(tower)))
        return out

    def decide(self, state: dict, t: float) -> ActDecision:
        if not legal_cards(state, self.cards):
            return ActDecision(False, "none", None, "no affordable card")

        threats = self._threats(state)
        if threats:
            eta, unit, lane = min(threats, key=lambda x: x[0])      # stable: first of the most urgent
            name = unit.get("card") or unit.get("unit_class") or "unit"
            return ActDecision(True, "defend", lane, f"defend {name} {lane or 'any'} eta {eta:.1f}s", threat=unit)

        elixir = state["own"].get("elixir") or 0.0
        if elixir >= self.leak_elixir:
            lanes = {lane_of(_tile(u)) for u in _enemy_units(state) if _tile(u)}
            lane = OPPOSITE_LANE.get(next(iter(lanes))) if len(lanes) == 1 else None
            return ActDecision(True, "leak", lane, f"elixir {elixir:.1f} near full, lane {lane or 'any'}")

        opp = state.get("opponent") or {}
        opp_elixir = opp.get("elixir_estimate")
        if (opp_elixir is not None and opp_elixir <= self.punish_opp_elixir
                and elixir >= self.punish_own_elixir):
            plays = [p for p in opp.get("recent_plays", []) if _tile(p)]
            last = max(reversed(plays), key=lambda p: p["t"] if p.get("t") is not None else float("-inf")) \
                if plays else None                      # newest play with a tile (later in the list wins a tie)
            lane = OPPOSITE_LANE.get(lane_of(_tile(last))) if last else None
            return ActDecision(True, "punish", lane,
                               f"opponent ~{opp_elixir:.1f} elixir after spending, own {elixir:.1f}, lane {lane or 'any'}")

        return ActDecision(False, "hold", None, "save elixir: no threat, not full, opponent not overspent")

    # -- candidates ----------------------------------------------------------------------------------------------------
    def _is_spell(self, card: str | None) -> bool:
        return card is not None and (self.cards.get(card) or {}).get("card_type") == "Spell"

    def filter_candidates(self, candidates: list, decision: ActDecision, state: dict) -> list:
        """Keep the candidates that fit `decision`, in the generator's order. [] means "hold" (punish with no fit,
        or every candidate was a blind spell)."""
        enemy_cells = {(lane_of(_tile(u)), zone_of(_tile(u))) for u in _enemy_units(state) if _tile(u)}
        parsed = [_parse(c) for c in candidates]
        # no blind spells: a spell needs an enemy unit in its own lane and zone
        kept = [p for p in parsed if not self._is_spell(p.card) or (p.lane, p.zone) in enemy_cells]
        spell_filtered = [p.obj for p in kept]
        lane = decision.lane

        if decision.mode == "defend":
            fits = [p.obj for p in kept
                    if (lane is None or p.lane == lane) and (self._is_spell(p.card) or p.zone in DEFEND_ZONES)]
            if fits:
                return fits
            kg = [p.obj for p in kept if p.sources.get("kg", 0) > 0]
            return kg or spell_filtered
        if decision.mode == "leak":
            fits = [p.obj for p in kept if p.zone in LEAK_ZONES and (lane is None or p.lane == lane)]
            return fits or spell_filtered
        if decision.mode == "punish":
            return [p.obj for p in kept
                    if not self._is_spell(p.card) and (lane is None or p.lane == lane) and p.zone in PUNISH_ZONES]
        return spell_filtered
