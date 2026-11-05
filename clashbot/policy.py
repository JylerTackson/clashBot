"""Reactive act/hold policy (D-041): save elixir, answer threats in time, leak at a full bar, punish an overspent opponent.

`ReactivePolicy.decide` says whether to play now (and in which lane); `filter_candidates` narrows the candidate
generator's output to moves that fit that decision (no blind spells). A damage spell must also be worth its elixir
(MAC-Y2): `spell_target` finds the densest cluster of non-tank enemy units in the spell's lane and zone, and
`target_tile` hands that aim to the live loop instead of the pro median tile. `LearnedActPolicy` (MAC-X) adds a
learned play/wait head on top: the rules keep the safety modes (defend / leak), a d1 act-head decides the rest.
`clashbot.loop.should_decide` stays as the replay trigger and the `--policy trigger` fallback of the live loop.
Candidates from the tactics engine (`"tac"` in their sources, MAC-Z3) were placed on purpose: they skip the lane and
zone constraints of defend / leak / punish, but damage spells among them still need a worthwhile target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

from clashbot import d1_format
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

# spells (MAC-Y2): radius in tiles, travel time before the spell lands, and what is not worth a damage spell
SPELL_RADIUS = {"zap": 2.5, "fireball": 2.5, "arrows": 4.0, "poison": 3.5, "lightning": 3.5, "rocket": 2.0,
                "the-log": 2.0, "barbarian-barrel": 2.0, "giant-snowball": 2.5, "earthquake": 3.5, "tornado": 5.5,
                "freeze": 3.0, "void": 2.5, "goblin-curse": 3.0, "royal-delivery": 3.0, "vines": 2.5}
DEFAULT_SPELL_RADIUS = 2.5
SPELL_LEAD_S = {"fireball": 1.0, "rocket": 1.5, "arrows": 0.8, "poison": 0.5, "lightning": 0.6}   # travel time
DEFAULT_SPELL_LEAD_S = 0.3
TANK_CARDS = {"giant", "royal-giant", "golem", "p-e-k-k-a", "mega-knight", "lava-hound", "electro-giant",
              "goblin-giant", "giant-skeleton", "elixir-golem", "goblinstein", "rune-giant"}
NON_DAMAGE_SPELLS = {"rage", "clone", "mirror", "goblin-barrel", "graveyard"}   # keep the old lane+zone rule
SWARM_MIN_UNITS = 3
ARENA_COLS, ARENA_ROWS = 18, 32


@dataclass
class ActDecision:
    act: bool
    mode: str                 # "defend" | "leak" | "punish" | "learned" | "hold" | "none"
    lane: str | None          # lane to play in (None = any)
    reason: str               # short human-readable reason, logged
    threat: dict | None = None   # the enemy unit being answered (defend)


def _tile(unit: dict) -> list | None:
    tile = unit.get("tile")
    return tile if isinstance(tile, (list, tuple)) and len(tile) >= 2 else None


def _lead_tile(unit: dict, lead_s: float) -> list[float] | None:
    """The unit's tile moved toward its predicted position (`predicted_tile_2s`) by `lead_s` of the 2 s, or None."""
    tile = _tile(unit)
    if tile is None:
        return None
    pred = unit.get("predicted_tile_2s")
    if (isinstance(pred, (list, tuple)) and len(pred) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in pred)):
        k = lead_s / 2.0
        return [tile[0] + (pred[0] - tile[0]) * k, tile[1] + (pred[1] - tile[1]) * k]
    return [float(tile[0]), float(tile[1])]


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

    def decide(self, state: dict, t: float, context_text: str | None = None) -> ActDecision:
        """`context_text` is accepted for the policy interface of the live loop and ignored here."""
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

    def _counted(self, unit: dict) -> bool:
        """A unit a damage spell can be aimed at: it has a tile and is neither a tank nor a spell."""
        if _tile(unit) is None:
            return False
        card = unit.get("card")
        return card not in TANK_CARDS and not self._is_spell(card)

    def spell_target(self, card: str, lane: str | None, zone: str | None,
                     state: dict) -> tuple[list[float], float, int] | None:
        """(tile, elixir value, unit count) of the best place for the damage spell `card` in `lane`/`zone`, or None
        when no counted enemy unit stands there or the cluster is not worth the spell (value below its cost and
        fewer than SWARM_MIN_UNITS units). Centres are the lead tiles of the counted units in the lane and zone; the
        members of a centre are all counted units, in any lane, whose lead tile is within the spell radius."""
        r = SPELL_RADIUS.get(card, DEFAULT_SPELL_RADIUS)
        lead = SPELL_LEAD_S.get(card, DEFAULT_SPELL_LEAD_S)
        counted = [(u, _lead_tile(u, lead)) for u in _enemy_units(state) if self._counted(u)]
        centres = [lt for u, lt in counted if (lane_of(_tile(u)), zone_of(_tile(u))) == (lane, zone)]
        best: tuple[float, int, float] | None = None
        best_leads: list[list[float]] = []
        for c in centres:
            members = [(u, lt) for u, lt in counted if math.dist(c, lt) <= r]
            known = {u.get("card") for u, _ in members if (self.cards.get(u.get("card")) or {}).get("elixir_cost")}
            value = sum(float(self.cards[k]["elixir_cost"]) for k in known)
            rank = (value, len(members), -c[1])             # ties: the smaller row (our side), then the first centre
            if best is None or rank > best:
                best, best_leads = rank, [lt for _, lt in members]
        if best is None:
            return None
        value, count = best[0], best[1]
        cost = (self.cards.get(card) or {}).get("elixir_cost")
        if not (cost is not None and value >= float(cost)) and count < SWARM_MIN_UNITS:
            return None
        x = min(max(sum(m[0] for m in best_leads) / count, 0.0), ARENA_COLS - 1.0)
        y = min(max(sum(m[1] for m in best_leads) / count, 0.0), ARENA_ROWS - 1.0)
        return [round(x, 1), round(y, 1)], value, count

    def target_tile(self, key: str, state: dict) -> list[float] | None:
        """Where the live loop should drop `key` when it is a damage spell with something worth hitting; None for
        every other key (the loop then uses the pro placement prior)."""
        parts = str(key).split("|")
        if len(parts) != 3 or not self._is_spell(parts[0]) or parts[0] in NON_DAMAGE_SPELLS:
            return None
        target = self.spell_target(parts[0], parts[1], parts[2], state)
        return target[0] if target else None

    def _aims_at_threat(self, card: str, target: list[float], threat: dict | None) -> bool:
        """defend: the threat is a counted unit and, where the spell lands, within the spell radius of the aim."""
        if threat is None or not self._counted(threat):
            return False
        hit = _lead_tile(threat, SPELL_LEAD_S.get(card, DEFAULT_SPELL_LEAD_S))
        return hit is not None and math.dist(hit, target) <= SPELL_RADIUS.get(card, DEFAULT_SPELL_RADIUS)

    def _unblind(self, parsed: list[_Cand], state: dict, decision: ActDecision | None = None) -> list[_Cand]:
        """No blind or wasted spells. A damage spell needs a worthwhile target (`spell_target`), and in defend mode that
        target must be the threat; a non-damage spell (rage, ...) needs an enemy unit in its own lane and zone."""
        enemy_cells = {(lane_of(_tile(u)), zone_of(_tile(u))) for u in _enemy_units(state) if _tile(u)}
        out: list[_Cand] = []
        for p in parsed:
            if not self._is_spell(p.card):
                out.append(p)
            elif p.card in NON_DAMAGE_SPELLS:
                # a tactical win-condition spell (goblin barrel, graveyard) is aimed at a tower, not at a unit
                if (p.lane, p.zone) in enemy_cells or "tac" in p.sources:
                    out.append(p)
            else:
                target = self.spell_target(p.card, p.lane, p.zone, state)
                if target is None:
                    continue
                if (decision is not None and decision.mode == "defend"
                        and not self._aims_at_threat(p.card, target[0], decision.threat)):
                    continue
                out.append(p)
        return out

    def no_blind_spells(self, candidates: list, state: dict) -> list:
        """`candidates` without the spells that have nothing worth hitting in their lane and zone, order kept."""
        return [p.obj for p in self._unblind([_parse(c) for c in candidates], state)]

    def filter_candidates(self, candidates: list, decision: ActDecision, state: dict) -> list:
        """Keep the candidates that fit `decision`, in the generator's order. [] means "hold" (punish with no fit,
        or every candidate was a blind spell)."""
        kept = self._unblind([_parse(c) for c in candidates], state, decision)
        spell_filtered = [p.obj for p in kept]
        lane = decision.lane
        tac = lambda p: "tac" in p.sources          # noqa: E731  tactical candidates: placed on purpose, no lane / zone rule

        if decision.mode == "defend":
            fits = [p.obj for p in kept
                    if tac(p) or ((lane is None or p.lane == lane)
                                  and (self._is_spell(p.card) or p.zone in DEFEND_ZONES))]
            if fits:
                return fits
            kg = [p.obj for p in kept if p.sources.get("kg", 0) > 0]
            return kg or spell_filtered
        if decision.mode == "leak":
            fits = [p.obj for p in kept if tac(p) or (p.zone in LEAK_ZONES and (lane is None or p.lane == lane))]
            return fits or spell_filtered
        if decision.mode == "punish":
            return [p.obj for p in kept
                    if tac(p) or (not self._is_spell(p.card) and (lane is None or p.lane == lane)
                                  and p.zone in PUNISH_ZONES)]
        return spell_filtered


class LearnedActPolicy:
    """Learned play/wait decision (MAC-X, D-041): a d1 act-head scores "deploy now rather than wait" from the context.

    The rules stay in charge of safety: no affordable card -> none, a threat to answer (defend) or a full bar (leak)
    -> act, whatever the head says. In every other moment the head decides: p(act) >= threshold plays (keeping the
    rule's mode and lane when the rules also wanted a punish), else hold. `scorer.probabilities(state, questions)`
    is a `D1Ranker` on the act checkpoint ({"act": [p_false, p_true]})."""

    def __init__(self, scorer, cards: dict | None = None, threshold: float = 0.5,
                 rules: ReactivePolicy | None = None, min_check_gap_s: float = 0.5):
        self.scorer = scorer
        self.threshold = threshold
        self.rules = rules if rules is not None else ReactivePolicy(cards)
        self.cards = self.rules.cards
        self.min_check_gap_s = min_check_gap_s
        self._last_t: float | None = None         # time of the last real scorer call
        self._last_p = 0.0

    def p_act(self, context_text: str) -> float:
        state, questions = d1_format.act_request(context_text)
        return float(self.scorer.probabilities(state, questions)[d1_format.ACT_NAME][1])

    def _p(self, context_text: str, t: float) -> float:
        """p(act), reused when the previous scorer call was less than `min_check_gap_s` ago."""
        if self._last_t is not None and 0.0 <= t - self._last_t < self.min_check_gap_s:
            return self._last_p
        self._last_p = self.p_act(context_text)
        self._last_t = t
        return self._last_p

    def decide(self, state: dict, t: float, context_text: str | None = None) -> ActDecision:
        r = self.rules.decide(state, t)
        if r.mode == "none":
            return r
        if r.act and r.mode in ("defend", "leak"):
            return r
        if context_text is None:
            return r
        p = self._p(context_text, t)
        if p >= self.threshold:
            return ActDecision(True, r.mode if r.act else "learned", r.lane if r.act else None,
                               f"learned act p={p:.2f}")
        return ActDecision(False, "hold", None, f"learned wait p={p:.2f}")

    def target_tile(self, key: str, state: dict) -> list[float] | None:
        return self.rules.target_tile(key, state)

    def filter_candidates(self, candidates: list, decision: ActDecision, state: dict) -> list:
        if decision.mode == "learned":
            return self.rules.no_blind_spells(candidates, state)
        return self.rules.filter_candidates(candidates, decision, state)
