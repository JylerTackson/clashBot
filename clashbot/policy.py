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
Real defence (MAC-AD): the pushes come from the engine (every unit within ~10 s of a tower), not only the units 4 s
out. Each push has a deploy window per card role (a building vs a building targeter from the river until the targeter
locks on its tower, a ranged troop 6 s, a melee card 4.5 s, a kamikaze card or spell 2.5 s) and an answer is adequate
when it stops the push (stopping class high / medium against a push with medium or tank units; a spell only when it
hits its cost back). The policy acts when an adequate card is affordable and its window is open, saves the elixir when
one is affordable soon, waits for the window otherwise, and holds when our own plays (known from the tap) already
answer the push. A low-stopping card ("slows ...") goes only in an emergency or when the hand holds nothing adequate.
Candidates from the tactics engine (`"tac"` in their sources, MAC-Z3) were placed on purpose: they skip the lane and
zone constraints of defend / leak / punish, but damage spells among them still need a worthwhile target.
Offence windows and placement (MAC-AG): besides the defend rules the policy has a `counter` mode, an act when every
push is answered or still more than 5 s out and the engine's best offence move (counter_push, split, punish, leak) is
worth it and leaves the elixir of the cheapest adequate answer; a full-bar leak plays the win condition when that is the
engine's best move. Every act carries `engine_tiles` (card -> the engine's tile against this push), which the live loop
uses instead of the pro-median tile of a lookup candidate. A flyer is never answered with a ground-only card, a building
is a pull only while the threat can still be pulled (it has crossed the river and not locked on its tower), and a second
card of the same role class is not played against a push within the commitment window.
Threat model (MAC-AH): the engine's pushes come ranked by what they would do to the tower. A push the tower handles is a
hold (mode `save`, "tower handles ..."); a move is adequate when the engine's simulation says so (the role rule is the
fallback for cards without stats); a smaller push may not spend the elixir of the best adequate answer to a bigger one
that reaches us within `RESERVE_WINDOW_S`; a damage spell is worth casting when what it kills and damages pays its cost
(`price_spells=False` brings back the elixir-sum rule).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from clashbot import d1_format
from clashbot.card_roles import hits_air, load_roles
from clashbot.game_rules import action_key, lane_of, legal_cards, load_cards, zone_of
from clashbot.unit_stats import spell_stat, spell_value

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

# real defence (MAC-AD): deploy windows (seconds of the push's ETA), answered pushes, low-stop cards
BUILDING_WINDOW_S = 9.0     # kept as a constructor argument only: a pull window is geometric since MAC-AG (see below)
RANGED_WINDOW_S = 6.0       # a ranged troop, a building vs a push without a targeter, any card vs a flying push
MELEE_WINDOW_S = 4.5        # melee / mini tank / tank killer
BURST_WINDOW_S = 2.5        # kamikaze card or spell
ANSWERED_MIN_ETA_S = 1.0    # an answered push this close to the tower is acted on all the same
NO_ADEQUATE_ETA_S = 2.5     # the hand holds nothing adequate: a low-stopping card may go this close
COMMIT_S = 4.0              # a card the policy chose to answer a push counts as answering it this long
DEFAULT_ANSWERED_FRAC = 0.6
RESERVE_WINDOW_S = 10.0     # a bigger push this close (tracker or simulated ETA) keeps the elixir of its answer (MAC-AH)

# offence windows (MAC-AG)
COUNTER_MIN_SCORE = 0.5     # the engine's best offence move needs this score for a `counter` act
COUNTER_CALM_ETA_S = 5.0    # a push that is not answered and this close blocks a counter (it is a defend matter)
COUNTER_INTENTS = ("counter_push", "split", "punish", "leak")

# placement geometry (MAC-AG): our tower tiles (the perception's TOWER_TILES), pulls, flyers
OWN_TOWER_TILES = {"own_left": (3.5, 4.3), "own_right": (14.5, 4.3), "own_king": (9.0, 0.5)}
PULL_LEAD_S = 1.0           # a threat is judged where it will be this many seconds ahead
RIVER_CROSSED_ROW = 16      # a threat at or beyond this row (towards our side) has crossed the river
LOCK_ON_MARGIN = 1.0        # a building targeter locks on its tower once it is within its attack range + this many tiles
ATTACK_RANGE_TILES = {"melee": 1.0, "short": 3.0, "long": 5.0}
AIR_GROUND_NEAR_TILES = 2.0     # a ground unit this close to a flying lead is part of the same group (a ground card helps)

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
    mode: str                 # "defend" | "counter" | "leak" | "punish" | "learned" | "save" | "hold" | "none"
    lane: str | None          # lane to play in (None = any)
    reason: str               # short human-readable reason, logged
    threat: dict | None = None   # the enemy unit being answered (defend) or waited for (save)
    allowed_cards: frozenset[str] | None = None   # defend / counter: only these cards fit the decision (None = any)
    engine_tiles: dict[str, list[float]] = field(default_factory=dict)   # card -> the engine's tile for it (MAC-AG)


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


def nearest_own_tower_dist(tile) -> float:
    return min(math.dist(tile, xy) for xy in OWN_TOWER_TILES.values())


def locked_on(tile, role: dict) -> bool:
    """A building targeter at `tile` has locked on its tower (it is within its attack range + LOCK_ON_MARGIN of one of
    ours): a building placed now no longer pulls it."""
    reach = ATTACK_RANGE_TILES.get(role.get("range") or "melee", ATTACK_RANGE_TILES["melee"]) + LOCK_ON_MARGIN
    return nearest_own_tower_dist(tile) <= reach


def targeter_of(units: list[dict], role_of) -> dict | None:
    """The most urgent unit of the push that goes for buildings (giant, balloon, hog, ram ...), or None."""
    return next((u for u in units if role_of(u.get("card")).get("targets") == "buildings" and _tile(u)), None)


def air_lead_blocks_ground(units: list[dict], role_of) -> bool:
    """The push's lead (most urgent) unit flies, and no ground unit is close behind it: a card that cannot hit air
    cannot answer it. With ground units around the lead (the most urgent ground unit within AIR_GROUND_NEAR_TILES of
    it) a ground card still helps against those."""
    if not units or not role_of(units[0].get("card")).get("flying"):
        return False
    lead = _tile(units[0])
    ground = next((u for u in units[1:] if not role_of(u.get("card")).get("flying")), None)
    if ground is None or lead is None or _tile(ground) is None:
        return True
    return math.dist(_tile(ground), lead) > AIR_GROUND_NEAR_TILES


_Seen = tuple[float, dict, str | None]          # (eta, enemy unit, lane) of a threat or a watched unit


@dataclass
class _Plan:
    """What the engine's pushes ask of us (MAC-AD), most urgent push first."""
    act: ActDecision | None = None                   # the first push that must be answered now
    commit: tuple | None = None                      # ((push lane, lead card), card chosen) of that act
    hold: ActDecision | None = None                  # else the first push's hold (answered, saving, waiting, nothing)
    reserve: tuple[float, str] | None = None         # elixir the answer to the first unanswered push still needs, its name
    reserve_push: Any = None                         # ... and that push (a smaller push may not spend the reserve, MAC-AH)
    calm: bool = True                                # every push is answered or farther out than `counter_eta_s`


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
                 emergency_eta_s: float = EMERGENCY_ETA_S, adequate_margin: float = ADEQUATE_MARGIN,
                 building_window_s: float = BUILDING_WINDOW_S, commit_s: float = COMMIT_S,
                 counter_min_score: float = COUNTER_MIN_SCORE, counter_eta_s: float = COUNTER_CALM_ETA_S,
                 price_spells: bool = True):
        self.cards = cards if cards is not None else load_cards()
        self.price_spells = price_spells        # MAC-AH: a damage spell is valued by what it kills (False: elixir sum)
        self.defend_eta_s = defend_eta_s
        self.leak_elixir = leak_elixir
        self.punish_opp_elixir = punish_opp_elixir
        self.punish_own_elixir = punish_own_elixir
        self.engine = engine                    # a TacticalEngine (anything with `analyse(state)`); built lazily if None
        self.save_margin_s = save_margin_s
        self.emergency_eta_s = emergency_eta_s
        self.adequate_margin = adequate_margin
        self.building_window_s = building_window_s    # kept for compatibility; the building window is geometric (MAC-AG)
        self.counter_min_score = counter_min_score    # the engine score its best offence move needs for a counter
        self.counter_eta_s = counter_eta_s            # a push not answered and this close (s) rules a counter out
        self.commit_s = commit_s                # the card chosen against a push counts as answering it this long (0 = off)
        self._engine_failed = False             # building the default engine raised: stay on the old defend rule
        self._role_table: dict | None = None    # card_roles, when the engine has none of its own
        self._acted: dict[tuple, tuple[float, str]] = {}      # (push lane, lead card) -> (t, card) of our last answer
        self._commits: dict[tuple, list[tuple[float, str]]] = {}   # (push lane, lead card) -> every (t, card) we chose

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

    def _need_s(self, card: str, state: dict) -> float:
        """Seconds until the bar holds the cost of `card`."""
        elixir = state["own"].get("elixir") or 0.0
        return max(0.0, self._cost(card) - elixir) * seconds_per_elixir(state.get("phase"))

    def _cost(self, card: str) -> float:
        return float((self.cards.get(card) or {}).get("elixir_cost") or 0.0)

    @staticmethod
    def _name(unit: dict) -> str:
        return unit.get("card") or unit.get("unit_class") or "unit"

    # -- real defence (MAC-AD) -----------------------------------------------------------------------------------------
    def _roles(self) -> dict:
        """The card role table: the engine's own when it has one, else the committed Z1 table."""
        roles = getattr(self.engine, "roles", None)
        if isinstance(roles, dict) and roles:
            return roles
        if self._role_table is None:
            try:
                self._role_table = load_roles()
            except Exception:                                   # noqa: BLE001  no table: windows fall back to the card type
                self._role_table = {}
        return self._role_table

    def _traits(self, push) -> tuple[bool, bool, bool]:
        """(heavy, building targeter, flying) of the push: the engine's flags (`heavy`, `needs`), and for a push built
        without them (a stand-in engine) what the card roles of its units say. An unknown card counts as a medium
        ground unit, as in the engine."""
        roles = self._roles()
        units = [roles.get(u.get("card")) or {"hp_class": "medium"} for u in push.units]
        heavy = push.heavy or "tank" in push.needs or any(r.get("hp_class") in ("medium", "tank") for r in units)
        targeter = "building_targeter" in push.needs or any(r.get("targets") == "buildings" for r in units)
        air = "air" in push.needs or any(r.get("flying") for r in units)
        return heavy, targeter, air

    def _role_of(self, card):
        """The role entry of `card` ({} when the table has none)."""
        return self._roles().get(card) or {}

    def _ctype(self, card: str) -> str:
        role = self._role_of(card)
        return role.get("type") or ("spell" if self._is_spell(card) else
                                    "building" if (self.cards.get(card) or {}).get("card_type") == "Building" else "troop")

    def _pull_lost(self, card: str, push) -> bool:
        """`card` is a building and the push has a building targeter that has locked on its tower: placing it now
        pulls nothing (MAC-AG). False for every other card, and when no unit of the push is a known targeter."""
        if self._ctype(card) != "building" or not self._traits(push)[1]:
            return False
        unit = targeter_of(push.units, self._role_of)
        tile = _lead_tile(unit, PULL_LEAD_S) if unit is not None else None
        return tile is not None and locked_on(tile, self._role_of(unit.get("card")))

    def _air_blocked(self, card: str, push) -> bool:
        """A flyer leads the push and `card` cannot hit air (a card without a role entry is not judged)."""
        role = self._roles().get(card)
        return bool(role) and not hits_air(role) and air_lead_blocks_ground(push.units, self._role_of)

    def _window_s(self, card: str, push) -> float:
        """Seconds of the push's ETA within which `card` may be deployed against it (inf = always, -inf = never): a
        building vs a building targeter (giant, balloon, hog ...) from the moment the targeter has crossed the river
        (row <= RIVER_CROSSED_ROW) until it locks on its tower (MAC-AG: pulling is geometric, not a clock); a ranged
        troop, a building vs another push and any card vs a flying push 6 s; a melee / mini tank / tank killer card
        4.5 s; a kamikaze card or spell 2.5 s."""
        role = self._role_of(card)
        ctype = self._ctype(card)
        tags = role.get("roles") or []
        _, targeter, air = self._traits(push)
        if ctype == "building":
            if not targeter:
                return RANGED_WINDOW_S
            unit = targeter_of(push.units, self._role_of) or (push.units[0] if push.units else None)
            tile = _lead_tile(unit, PULL_LEAD_S) if unit is not None else None
            if tile is None:
                return math.inf
            return math.inf if tile[1] <= RIVER_CROSSED_ROW and not self._pull_lost(card, push) else -math.inf
        if air:
            return RANGED_WINDOW_S
        if ctype == "spell" or "kamikaze" in tags:
            return BURST_WINDOW_S
        if "mini_tank" in tags or "tank_killer" in tags or role.get("range", "melee") == "melee":
            return MELEE_WINDOW_S
        return RANGED_WINDOW_S

    def _adequate(self, move, push) -> bool:
        """The move stops the push. Against a push with a medium or tank unit that takes a high / medium stopping card
        (a building, a tank killer, a troop with dps), or a spell that hits its cost back (MAC-Y2); a push of swarm or
        light units is stopped by anything. A move that does not say how well it stops (`n/a`) counts. A card that
        cannot hit the flyer leading the push does not (MAC-AG), and a building does not once the building targeter
        has locked on its tower. MAC-AH: when the engine simulated the move (`adequate` is not None) its verdict stands -
        the tower damage left is small, or nothing reaches the tower - and the role rule above is only the fallback."""
        if self._air_blocked(move.card, push) or self._pull_lost(move.card, push):
            return False
        verdict = getattr(move, "adequate", None)
        if verdict is not None:
            return bool(verdict)
        if not self._traits(push)[0]:
            return True
        if move.stop == "low":
            return False
        if move.stop == "spell":
            return move.value is not None and move.value >= self._cost(move.card)
        return True

    @staticmethod
    def _defence_moves(report) -> list:
        """The defend moves of `report` (best first): the report's full defence list when it has one, else its moves."""
        moves = getattr(report, "defence", None) or report.moves
        return [m for m in moves if m.intent == "defend"]

    def _push_moves(self, engine, state: dict, report, push, t: float) -> tuple[list, list]:
        """(now, full): the defend moves for `push` with the elixir we have, and as if the bar were full. With several
        pushes the analysis is repeated on that push alone, so the moves of another lane can neither crowd out nor
        stand in for its answers."""
        base = state
        if len(report.pushes) > 1:
            base = {**state, "units": [u for u in state.get("units", [])
                                       if u.get("side") != "enemy" or any(u is x for x in push.units)]}
            now_report = engine.analyse(base, t)
        else:
            now_report = report
        full_report = engine.analyse({**base, "own": {**base["own"], "elixir": FULL_ELIXIR}}, t)
        return self._defence_moves(now_report), self._defence_moves(full_report)

    def _answered_names(self, engine, push, t: float) -> list[str]:
        """The cards that answer the push: the engine's (`push.answered_by`), plus the card we chose against it in the
        last `commit_s` seconds when that alone pays for `answered_frac` of the push (our tap may not show yet). Only a
        later call counts it: asking again at the same instant gives the same answer."""
        names = list(push.answered_by) if push.answered else []
        entry = self._acted.get(self._push_key(push))
        if entry is not None and 0.0 < t - entry[0] <= self.commit_s and entry[1] not in names:   # later calls only
            frac = (getattr(engine, "w", None) or {}).get("answered_frac", DEFAULT_ANSWERED_FRAC)
            cost = self._cost(entry[1])
            if cost > 0 and cost >= frac * push.elixir:
                names.append(entry[1])
        return names

    @staticmethod
    def _push_key(push) -> tuple:
        return push.lane, (push.units[0].get("card") or push.units[0].get("unit_class")) if push.units else None

    @staticmethod
    def _engine_tiles(moves: list) -> dict[str, list[float]]:
        """card -> the tile of the engine's best move for it (`moves` best first)."""
        tiles: dict[str, list[float]] = {}
        for m in moves:
            tiles.setdefault(m.card, [float(v) for v in m.tile])
        return tiles

    def _role_class(self, card: str) -> str:
        """building / spell / kamikaze / swarm / melee (incl. mini tank, tank killer) / ranged: what a card does against
        a push, for the rule that a push is not answered twice with the same kind of card."""
        ctype = self._ctype(card)
        tags = self._role_of(card).get("roles") or []
        if ctype in ("building", "spell"):
            return ctype
        for tag in ("kamikaze", "swarm"):
            if tag in tags:
                return tag
        return "ranged" if self._role_of(card).get("range") in ("short", "long") else "melee"

    def _without_repeats(self, moves: list, push, t: float) -> list:
        """`moves` without the cards of a role class we already chose against this push in the last `commit_s`
        seconds (no second knight on a push the first one stands against), unless the push is worth at least twice
        what we spent on it. Only a later call counts a choice (the same instant gives the same answer)."""
        spent, classes = 0.0, set()
        for t0, card in self._commits.get(self._push_key(push), []):
            if 0.0 < t - t0 <= self.commit_s:
                spent += self._cost(card)
                classes.add(self._role_class(card))
        if not classes or push.elixir >= 2 * spent:
            return moves
        return [m for m in moves if self._role_class(m.card) not in classes]

    def _judge(self, state: dict, push, now: list, full: list) -> tuple[ActDecision, str | None]:
        """(decision, committed card) for one push that is not answered. First that applies: an adequate card is
        affordable and its window is open -> defend with those; the best adequate card is not affordable yet but
        will be in time -> save; the push is at the tower -> the best card at hand; the hand holds nothing adequate
        and the push is close -> the best card at hand; else wait."""
        unit = push.units[0]
        lead = _tile(unit)
        lane = lane_of(lead) or push.lane
        what = f"{self._name(unit)} {lane}"
        margin = self.adequate_margin
        tiles = self._engine_tiles(now)
        adequate_now = [m for m in now if self._adequate(m, push)]
        open_now = [m for m in adequate_now if push.eta_s <= self._window_s(m.card, push)]
        if open_now:
            best = max(open_now, key=lambda m: m.score)
            allowed = frozenset(m.card for m in open_now if m.score >= best.score - margin)
            return ActDecision(True, "defend", lane, f"defend {what}: {best.card} ({best.stop})", threat=unit,
                               allowed_cards=allowed, engine_tiles=tiles), best.card
        adequate_full = [m for m in full if self._adequate(m, push)]
        best_full = max(adequate_full, key=lambda m: m.score, default=None)
        if best_full is not None and best_full.card not in {m.card for m in now}:
            need = self._need_s(best_full.card, state)
            if need + self.save_margin_s <= push.eta_s:
                return ActDecision(False, "save", None,
                                   f"save for {best_full.card} vs {what}: {need:.1f}s, eta {push.eta_s:.1f}s",
                                   threat=unit), None
        best_now = max(now, key=lambda m: m.score, default=None)
        if best_now is not None:
            if push.eta_s <= self.emergency_eta_s:
                return ActDecision(True, "defend", lane, f"defend {what}: {best_now.card} (emergency)", threat=unit,
                                   allowed_cards=frozenset({best_now.card}), engine_tiles=tiles), best_now.card
            if not adequate_full and push.eta_s <= NO_ADEQUATE_ETA_S:
                return ActDecision(True, "defend", lane,
                                   f"defend {what}: {best_now.card} ({best_now.stop}, nothing adequate in hand)",
                                   threat=unit, allowed_cards=frozenset({best_now.card}),
                                   engine_tiles=tiles), best_now.card
        waiting = max(adequate_now, key=lambda m: m.score, default=None)
        if waiting is not None:
            return ActDecision(False, "save", None, f"waiting for the {waiting.card} window vs {what}: "
                               f"eta {push.eta_s:.1f}s", threat=unit), None
        return ActDecision(False, "save", None, f"no adequate answer in hand for {what}", threat=unit), None

    def _reserve(self, full: list, push) -> tuple[float, str] | None:
        """(elixir cost of the cheapest adequate answer within `adequate_margin` of the best one, the push's lead
        unit name), or None when the hand holds no adequate answer (nothing is reserved)."""
        adequate = [m for m in full if self._adequate(m, push)]
        if not adequate:
            return None
        top = max(m.score for m in adequate)
        return min(self._cost(m.card) for m in adequate if m.score >= top - self.adequate_margin), \
            self._name(push.units[0])

    def _plan(self, state: dict, t: float, report, threats: list[_Seen], watch: list[_Seen]) -> _Plan | None:
        """What the pushes of the engine's `report` say, most urgent first: the first one that must be answered now
        becomes `act`; the first hold is kept as `hold`; `reserve` is the answer the first unanswered push still needs;
        `calm` says that every push is answered or still farther out than `counter_eta_s`. None when the engine is
        missing, fails, or sees no push among the units that are coming (the old rule applies)."""
        engine = self._get_engine()
        coming = [u for eta, u, _ in threats + watch if u.get("card") or eta <= UNKNOWN_ETA_S]
        if report is not None:       # MAC-AH: the units of a push the engine listed for the river / bridge count too
            coming += [u for p in report.pushes if getattr(p, "early", False) for u in p.units if not self._ignored(u)]
        if engine is None or report is None or not coming:
            return None
        try:
            pushes = [p for p in report.pushes if any(x is c for x in p.units for c in coming)]
            plan = _Plan()
            for push in pushes:                                 # the engine ranks them: the biggest threat first
                if getattr(push, "tower_ok", False):
                    plan.hold = plan.hold or ActDecision(
                        False, "save", None, f"tower handles {self._name(push.units[0])} {push.lane}",
                        threat=push.units[0])
                    continue
                names = self._answered_names(engine, push, t)
                if names and push.eta_s > ANSWERED_MIN_ETA_S:
                    plan.hold = plan.hold or ActDecision(
                        False, "save", None, f"push {push.lane} answered by {', '.join(names)}",
                        threat=push.units[0])
                    continue
                if not names and push.eta_s <= self.counter_eta_s:
                    plan.calm = False
                now, full = self._push_moves(engine, state, report, push, t)
                now, full = self._without_repeats(now, push, t), self._without_repeats(full, push, t)
                decision, card = self._judge(state, push, now, full)
                if decision.act and plan.reserve is not None and not getattr(push, "destroys_tower", False):
                    decision, card = self._keep_reserve(state, plan, push, now, decision, card)
                    if not decision.act:
                        plan.hold = decision                    # the clearest reason why this push waits
                if decision.act:
                    plan.act, plan.commit = decision, (self._push_key(push), card)
                    return plan
                plan.hold = plan.hold or decision
                if plan.reserve is None:
                    plan.reserve = self._reserve(full, push)
                    plan.reserve_push = push if plan.reserve is not None else None
        except Exception:                                       # noqa: BLE001  never worse than the old rule
            return None
        return plan if pushes else None

    def _keep_reserve(self, state: dict, plan: _Plan, push, now: list, decision: ActDecision,
                      card: str | None) -> tuple[ActDecision, str | None]:
        """A bigger push (`plan.reserve_push`) that reaches us within RESERVE_WINDOW_S keeps the elixir of its best
        adequate answer: this smaller push may only use the cards that leave it. If none does, hold (MAC-AH)."""
        big = plan.reserve_push
        keep, name = plan.reserve
        sim_eta = getattr(big, "sim_eta_s", None)
        if big is None or min(big.eta_s, sim_eta if sim_eta is not None else big.eta_s) > RESERVE_WINDOW_S:
            return decision, card
        elixir = state["own"].get("elixir") or 0.0
        allowed = frozenset(c for c in (decision.allowed_cards or ()) if elixir - self._cost(c) >= keep)
        if allowed:
            if card in allowed:
                return decision, card
            best = max((m for m in now if m.card in allowed), key=lambda m: m.score)
            what = f"{self._name(push.units[0])} {decision.lane}"
            return ActDecision(True, "defend", decision.lane, f"defend {what}: {best.card} ({best.stop})",
                               threat=decision.threat, allowed_cards=allowed, engine_tiles=decision.engine_tiles), \
                best.card
        return ActDecision(False, "save", None, f"keep {keep:g} elixir for the {name}, not for {push.lane}",
                           threat=decision.threat), None

    def _defend_old(self, threats: list[_Seen]) -> ActDecision:
        """The rule before the engine: the most urgent threat is answered now with whatever fits (any card)."""
        eta, unit, lane = min(threats, key=lambda x: x[0])      # stable: first of the most urgent
        return ActDecision(True, "defend", lane, f"defend {self._name(unit)} {lane or 'any'} eta {eta:.1f}s",
                           threat=unit)

    def _analyse(self, state: dict, t: float):
        """The engine's report of the state, or None when there is no engine or it fails."""
        engine = self._get_engine()
        if engine is None:
            return None
        try:
            return engine.analyse(state, t)
        except Exception:                                       # noqa: BLE001  any failure -> the old rules
            return None

    @staticmethod
    def _best_offence(report) -> Any:
        """The engine's best move among the offence intents of a counter (counter_push, split, punish, leak), or None.
        The report's own `offence` list when it has one (the 6 best moves may all be defend moves), else its moves."""
        moves = getattr(report, "offence", None) or getattr(report, "moves", None) or []
        return next((m for m in moves if m.intent in COUNTER_INTENTS), None)

    def _is_win_condition(self, card: str) -> bool:
        return "win_condition" in (self._role_of(card).get("roles") or [])

    def decide(self, state: dict, t: float, context_text: str | None = None) -> ActDecision:
        """`context_text` is accepted for the policy interface of the live loop and ignored here. Order: the pushes
        of the engine (act when one must be answered now), the counter (an offence window while the pushes are answered
        or far), the full-bar leak, the punish; else the first push's hold."""
        if not legal_cards(state, self.cards):
            return ActDecision(False, "none", None, "no affordable card")

        threats, watch = self._classify(state)
        report = self._analyse(state, t)
        early = report is not None and any(getattr(p, "early", False) for p in report.pushes)
        plan = self._plan(state, t, report, threats, watch) if (threats or watch or early) else None
        if plan is not None and plan.act is not None:
            if self.commit_s > 0:
                self._acted[plan.commit[0]] = (t, plan.commit[1])
                commits = [c for c in self._commits.get(plan.commit[0], []) if 0.0 <= t - c[0] <= 4 * self.commit_s]
                self._commits[plan.commit[0]] = commits + [(t, plan.commit[1])]
            return plan.act
        if plan is None and threats:                    # no engine, or it knows no push for them: the old rule
            return self._defend_old(threats)

        elixir = state["own"].get("elixir") or 0.0
        opp = state.get("opponent") or {}
        opp_elixir = opp.get("elixir_estimate")
        leak_due = elixir >= self.leak_elixir
        punish_due = (opp_elixir is not None and opp_elixir <= self.punish_opp_elixir
                      and elixir >= self.punish_own_elixir)

        # offence window (MAC-AG): the pushes are answered or far, the engine has a worthwhile move for a win condition
        offence = self._best_offence(report)
        if (offence is not None and (plan is None or plan.calm) and offence.score >= self.counter_min_score
                and not (offence.intent == "leak" and leak_due) and not (offence.intent == "punish" and punish_due)):
            reserve = plan.reserve[0] if plan is not None and plan.reserve is not None else 0.0
            if elixir >= self._cost(offence.card) + reserve:
                lane = lane_of(offence.tile)
                return ActDecision(True, "counter", lane, f"counter: {offence.intent} {offence.card} {lane}",
                                   allowed_cards=frozenset({offence.card}),
                                   engine_tiles={offence.card: [float(v) for v in offence.tile]})

        if leak_due:
            lanes = {lane_of(_tile(u)) for u in _enemy_units(state) if _tile(u)}
            lane = OPPOSITE_LANE.get(next(iter(lanes))) if len(lanes) == 1 else None
            allowed, tiles = None, {}
            if offence is not None and offence.intent == "leak":
                tiles = {offence.card: [float(v) for v in offence.tile]}
                if self._is_win_condition(offence.card):            # the full bar goes into the win condition
                    allowed = frozenset({offence.card})
            return ActDecision(True, "leak", lane, f"elixir {elixir:.1f} near full, lane {lane or 'any'}",
                               allowed_cards=allowed, engine_tiles=tiles)

        if punish_due:
            reserve = plan.reserve if plan is not None else None
            if reserve is not None and elixir - reserve[0] < 0:
                return ActDecision(False, "save", None,
                                   f"keep {reserve[0]:g} elixir for the answer to {reserve[1]}, no punish")
            plays = [p for p in opp.get("recent_plays", []) if _tile(p)]
            last = max(reversed(plays), key=lambda p: p["t"] if p.get("t") is not None else float("-inf")) \
                if plays else None                      # newest play with a tile (later in the list wins a tie)
            lane = OPPOSITE_LANE.get(lane_of(_tile(last))) if last else None
            return ActDecision(True, "punish", lane,
                               f"opponent ~{opp_elixir:.1f} elixir after spending, own {elixir:.1f}, lane {lane or 'any'}")

        if plan is not None and plan.hold is not None:
            return plan.hold
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
        priced = self.price_spells and spell_stat(card) is not None   # MAC-AH: valued by what it kills, if it has numbers
        counted = [(u, _lead_tile(u, lead)) for u in _enemy_units(state) if self._counted(u)]
        centres = [lt for u, lt in counted if (lane_of(_tile(u)), zone_of(_tile(u))) == (lane, zone)]
        best: tuple[float, int, float] | None = None
        best_leads: list[list[float]] = []
        for c in centres:
            members = [(u, lt) for u, lt in counted if math.dist(c, lt) <= r]
            if priced:
                value = spell_value(card, [u for u, _ in members], self.cards)[0]
            else:
                known = {u.get("card") for u, _ in members
                         if (self.cards.get(u.get("card")) or {}).get("elixir_cost")}
                value = sum(float(self.cards[k]["elixir_cost"]) for k in known)
            rank = (value, len(members), -c[1])             # ties: the smaller row (our side), then the first centre
            if best is None or rank > best:
                best, best_leads = rank, [lt for _, lt in members]
        if best is None:
            return None
        value, count = best[0], best[1]
        cost = (self.cards.get(card) or {}).get("elixir_cost")
        if priced:
            if cost is not None and value < float(cost):    # kills and damage must pay for the spell (no swarm escape)
                return None
        elif not (cost is not None and value >= float(cost)) and count < SWARM_MIN_UNITS:
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

    @staticmethod
    def _engine_candidates(decision: ActDecision) -> list:
        """The engine's own moves for the allowed cards as tactical candidates: the generator offers only the first
        report moves, which may all be defend moves while the decision is an offence one."""
        from clashbot.candidates import Candidate            # lazily: the candidate module pulls in the retrieval stack

        out = []
        for card in sorted(decision.allowed_cards or ()):
            tile = decision.engine_tiles.get(card)
            if tile is not None:
                key = action_key(card, lane_of(tile), zone_of(tile))
                out.append(Candidate(key, 0.0, {"tac": 0.0}, list(tile), f"tactics: {decision.mode}, {decision.reason}"))
        return out

    def filter_candidates(self, candidates: list, decision: ActDecision, state: dict) -> list:
        """Keep the candidates that fit `decision`, in the generator's order. [] means "hold" (punish with no fit,
        or every candidate was a blind spell). A counter keeps the candidates of its card (the engine's own move when
        there is none), a leak that plays a win condition likewise."""
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
        if decision.mode in ("counter", "leak") and decision.allowed_cards is not None:
            # a counter or a win-condition leak plays that card; the engine places it (`engine_tiles`)
            kept = [p for p in kept if p.card in decision.allowed_cards]
            return [p.obj for p in kept] or self._engine_candidates(decision)
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
        if r.act and r.mode in ("defend", "counter", "leak"):
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
