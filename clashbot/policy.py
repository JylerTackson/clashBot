"""Reactive act/hold policy (D-041): save elixir, answer threats in time, leak at a full bar, punish an overspent opponent.

`ReactivePolicy.decide` says whether to play now (and in which lane); `filter_candidates` narrows the candidate
generator's output to moves that fit that decision (no blind spells). A damage spell must also be worth its elixir
(MAC-Y2): `spell_target` finds the densest cluster of non-tank enemy units in the spell's lane and zone, and
`target_tile` hands that aim to the live loop instead of the pro median tile. `LearnedActPolicy` (MAC-X) adds a
learned play/wait head on top: the rules keep the safety modes (defend / leak), a d1 act-head decides the rest.
`clashbot.loop.should_decide` stays as the replay trigger and the `--policy trigger` fallback of the live loop.
Defend discipline (MAC-AC C1, FIX_LIST #32-#34): only a unit that is really coming counts as a threat (not a retreating
one, not a far unknown_unit, not a spell). The tactical engine then says whether the hand already holds an adequate
answer; if not, and a better card is affordable before the push lands, the policy holds in mode "save" instead of
spending the elixir on the cheapest card. A unit that is only coming soon (watch list) keeps the elixir of its answer
out of a punish. When the engine fails, the old "any threat -> defend" rule applies.
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

# defend discipline (MAC-AC C1)
UNKNOWN_ETA_S = 4.0         # an unknown_unit (no card) only counts as a threat this close
WATCH_ETA_S = 10.0          # an enemy unit this close that is not a threat yet is watched (its answer keeps its elixir)
SAVE_MARGIN_S = 1.0         # spare seconds between having the elixir for the best answer and the push landing
EMERGENCY_ETA_S = 1.5       # a push this close is answered with the best card affordable now
ADEQUATE_MARGIN = 0.15      # a move this close to the best score (engine scale) is as good as the best
SECONDS_PER_ELIXIR = {"single_elixir": 2.8, "double_elixir": 1.4, "triple_elixir": 0.93, "overtime": 0.93}
FULL_ELIXIR = 10.0

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
    mode: str                 # "defend" | "leak" | "punish" | "learned" | "save" | "hold" | "none"
    lane: str | None          # lane to play in (None = any)
    reason: str               # short human-readable reason, logged
    threat: dict | None = None   # the enemy unit being answered (defend) or waited for (save)
    allowed_cards: frozenset[str] | None = None   # defend: only these cards answer the threat adequately (None = any)


def seconds_per_elixir(phase) -> float:
    """Seconds the bar needs for one elixir in `phase` (an unknown phase counts as single elixir)."""
    return SECONDS_PER_ELIXIR.get(phase, SECONDS_PER_ELIXIR["single_elixir"])


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


def _side_lane(unit: dict) -> str | None:
    """left / right of the unit's tile (a middle tile goes to the nearer side, col < 9 left), else its tower's lane."""
    tile = _tile(unit)
    lane = lane_of(tile)
    if lane == "middle":
        return "left" if tile[0] < 9 else "right"
    lane = lane or _tower_lane(_own_eta(unit)[1])
    return lane if lane in ("left", "right") else None


_Seen = tuple[float, dict, str | None]          # (eta, enemy unit, lane) of a threat or a watched unit


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
                 punish_own_elixir: float = PUNISH_OWN_ELIXIR, engine=None, save_margin_s: float = SAVE_MARGIN_S,
                 emergency_eta_s: float = EMERGENCY_ETA_S, adequate_margin: float = ADEQUATE_MARGIN):
        self.cards = cards if cards is not None else load_cards()
        self.defend_eta_s = defend_eta_s
        self.leak_elixir = leak_elixir
        self.punish_opp_elixir = punish_opp_elixir
        self.punish_own_elixir = punish_own_elixir
        self.engine = engine                    # a TacticalEngine (anything with `analyse(state)`); built lazily if None
        self.save_margin_s = save_margin_s
        self.emergency_eta_s = emergency_eta_s
        self.adequate_margin = adequate_margin
        self._engine_failed = False             # building the default engine raised: stay on the old defend rule

    # -- decide --------------------------------------------------------------------------------------------------------
    def _ignored(self, unit: dict) -> bool:
        """A unit that neither threatens nor needs watching: it retreats, or it is a spell object (not graveyard)."""
        if "retreat" in str(unit.get("heading") or "").lower():
            return True
        card = unit.get("card")
        return self._is_spell(card) and card != "graveyard"

    def _classify(self, state: dict) -> tuple[list[_Seen], list[_Seen]]:
        """(threats, watch) as (eta, unit, lane) lists. A threat is an enemy unit that must be answered now: its ETA to
        an own tower is within `defend_eta_s` (an unknown_unit, i.e. no card: within UNKNOWN_ETA_S), or it has no ETA
        and stands at row NEAR_BACK_ROW or deeper on our half (an unknown_unit never counts without an ETA).
        Retreating units and spells are neither. The watch list holds the other units whose ETA is within
        WATCH_ETA_S, soonest first."""
        threats: list[_Seen] = []
        watch: list[_Seen] = []
        for u in _enemy_units(state):
            if self._ignored(u):
                continue
            tile = _tile(u)
            eta, tower = _own_eta(u)
            lane = lane_of(tile) or _tower_lane(tower)
            known = bool(u.get("card"))
            if eta is not None:
                if eta <= (self.defend_eta_s if known else min(self.defend_eta_s, UNKNOWN_ETA_S)):
                    threats.append((eta, u, lane))
                elif eta <= WATCH_ETA_S:
                    watch.append((eta, u, lane))
            elif known and tile is not None and tile[1] <= NEAR_BACK_ROW:
                threats.append((0.0, u, lane))
        watch.sort(key=lambda x: x[0])
        return threats, watch

    def _threats(self, state: dict) -> list[_Seen]:
        """(eta, unit, lane) of every enemy unit that must be answered now."""
        return self._classify(state)[0]

    def _watch(self, state: dict) -> list[_Seen]:
        """(eta, unit, lane) of the enemy units that are not threats yet but land within WATCH_ETA_S, soonest first."""
        return self._classify(state)[1]

    # -- the tactical engine -------------------------------------------------------------------------------------------
    def _get_engine(self):
        """The engine, built on first use; None when it cannot be built (the old rules then apply)."""
        if self.engine is None and not self._engine_failed:
            try:
                from clashbot.tactics import TacticalEngine        # tactics imports this module: no top-level import
                self.engine = TacticalEngine(self.cards)
            except Exception:                                      # noqa: BLE001  any failure -> old behaviour
                self._engine_failed = True
        return self.engine

    @staticmethod
    def _find_push(pushes, unit: dict):
        """The push whose units include `unit` (the same object), else the push of the unit's lane, else None."""
        for push in pushes:
            if any(x is unit for x in push.units):
                return push
        lane = _side_lane(unit)
        return next((p for p in pushes if p.lane == lane), None) if lane else None

    def _answers(self, state: dict, unit: dict):
        """(push, defend moves best first) of the engine for the push that holds `unit`; (None, []) when the engine
        has no such push. With several pushes the analysis is repeated on that push alone, so the moves of another
        lane can neither crowd out nor stand in for its answers."""
        engine = self._get_engine()
        if engine is None:
            return None, []
        report = engine.analyse(state)
        push = self._find_push(report.pushes, unit)
        if push is None:
            return None, []
        if len(report.pushes) > 1:
            alone = {**state, "units": [u for u in state.get("units", [])
                                        if u.get("side") != "enemy" or any(u is x for x in push.units)]}
            report = engine.analyse(alone)
            push = self._find_push(report.pushes, unit) or push
        return push, [m for m in report.moves if m.intent == "defend"]

    def _need_s(self, card: str, state: dict) -> float:
        """Seconds until the bar holds the cost of `card`."""
        cost = float((self.cards.get(card) or {}).get("elixir_cost") or 0.0)
        elixir = state["own"].get("elixir") or 0.0
        return max(0.0, cost - elixir) * seconds_per_elixir(state.get("phase"))

    @staticmethod
    def _name(unit: dict) -> str:
        return unit.get("card") or unit.get("unit_class") or "unit"

    def _defend(self, state: dict, threats: list[_Seen]) -> ActDecision:
        """The decision for the most urgent threat: answer it now when the hand holds an adequate card, hold and save
        the elixir when a clearly better card is affordable before the push lands, else play the best card at hand."""
        eta, unit, lane = min(threats, key=lambda x: x[0])      # stable: first of the most urgent
        name = self._name(unit)
        what = f"{name} {lane or 'any'}"                        # the threat as the reasons name it
        old = ActDecision(True, "defend", lane, f"defend {what} eta {eta:.1f}s", threat=unit)
        try:
            push, now = self._answers(state, unit)
            if push is None:
                return old
            full = self._answers({**state, "own": {**state["own"], "elixir": FULL_ELIXIR}}, unit)[1]
        except Exception:                                       # noqa: BLE001  never worse than the old rule
            return old
        best_now = max(now, key=lambda m: m.score, default=None)
        best_any = max(full, key=lambda m: m.score, default=None)
        margin = self.adequate_margin
        if best_now is not None and (best_any is None or best_now.score >= best_any.score - margin):
            allowed = frozenset(m.card for m in now if m.score >= best_now.score - margin)
            return ActDecision(True, "defend", lane, f"defend {what} with an adequate answer", threat=unit,
                               allowed_cards=allowed)
        if best_any is not None:
            need = self._need_s(best_any.card, state)
            if need + self.save_margin_s <= push.eta_s:
                return ActDecision(False, "save", None,
                                   f"save for {best_any.card} vs {what}: {need:.1f}s, eta {push.eta_s:.1f}s",
                                   threat=unit)
        if best_now is not None:
            how = (f"emergency, eta {push.eta_s:.1f}s" if push.eta_s <= self.emergency_eta_s
                   else f"best card {best_now.card}")
            return ActDecision(True, "defend", lane, f"defend {what} ({how})", threat=unit,
                               allowed_cards=frozenset({best_now.card}))
        return ActDecision(False, "save", None, f"no answer in hand for {what}", threat=unit)

    def _punish_reserve(self, state: dict, watch: list[_Seen]) -> tuple[float, str] | None:
        """(elixir cost of the cheapest adequate answer to the soonest watched push, the unit's name), or None when
        the engine knows no answer (then nothing is reserved)."""
        unit = watch[0][1]
        try:
            moves = self._answers({**state, "own": {**state["own"], "elixir": FULL_ELIXIR}}, unit)[1]
        except Exception:                                       # noqa: BLE001
            return None
        if not moves:
            return None
        top = max(m.score for m in moves)
        costs = [float((self.cards.get(m.card) or {}).get("elixir_cost") or 0.0)
                 for m in moves if m.score >= top - self.adequate_margin]
        return min(costs), self._name(unit)

    def decide(self, state: dict, t: float, context_text: str | None = None) -> ActDecision:
        """`context_text` is accepted for the policy interface of the live loop and ignored here."""
        if not legal_cards(state, self.cards):
            return ActDecision(False, "none", None, "no affordable card")

        threats, watch = self._classify(state)
        if threats:
            return self._defend(state, threats)

        elixir = state["own"].get("elixir") or 0.0
        if elixir >= self.leak_elixir:
            lanes = {lane_of(_tile(u)) for u in _enemy_units(state) if _tile(u)}
            lane = OPPOSITE_LANE.get(next(iter(lanes))) if len(lanes) == 1 else None
            return ActDecision(True, "leak", lane, f"elixir {elixir:.1f} near full, lane {lane or 'any'}")

        opp = state.get("opponent") or {}
        opp_elixir = opp.get("elixir_estimate")
        if (opp_elixir is not None and opp_elixir <= self.punish_opp_elixir
                and elixir >= self.punish_own_elixir):
            reserve = self._punish_reserve(state, watch) if watch else None
            if reserve is not None and elixir - reserve[0] < 0:
                return ActDecision(False, "save", None,
                                   f"keep {reserve[0]:g} elixir for the answer to {reserve[1]}, no punish")
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
            if decision.allowed_cards is not None:         # only the adequate answers; tactical candidates too
                kept = [p for p in kept if p.card in decision.allowed_cards]
                spell_filtered = [p.obj for p in kept]
                if not kept:
                    return []
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
    -> act, whatever the head says; a save (MAC-AC: wait for a better answer) -> hold. In every other moment the head
    decides: p(act) >= threshold plays (keeping the rule's mode and lane when the rules also wanted a punish), else
    hold. `scorer.probabilities(state, questions)` is a `D1Ranker` on the act checkpoint ({"act": [p_false, p_true]})."""

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
        if r.mode in ("none", "save"):         # no card, or the rules wait for a better answer: the head cannot override
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
