"""Tactical engine (D-042, MAC-Z2, MAC-A2, MAC-AD): which enemy push threatens which tower, which hand card answers it
and where to put it, and when there is an offence window (counter_push, punish, leak, and for a bait deck bait,
pressure, support and cycle).

`TacticalEngine.analyse(state)` returns a `TacticalReport`: the enemy pushes (most urgent first), up to 6 concrete
moves (card + deploy tile + intent + reason, best first) and 1..4 text lines for the d1 context. The card roles come
from the Z1 table (`clashbot.card_roles`, loaded lazily); spell aim and value come from `ReactivePolicy.spell_target`
(MAC-Y2). Every hand-set number of the scoring lives in `DEFAULT_WEIGHTS`; `tactics_weights.json` (tuned on the pro
data by `tools/train/tune_tactics.py`) and an explicit `weights=` dict override it. Deterministic, no I/O after
`__init__`, no models.
Real defence (MAC-AD): a defend move carries its stopping class against the push (`stop`: high / medium / low / spell),
and a push counts as answered (`Push.answered`, `answered_by`) when the allies near its lead unit, our buildings on our
half (against a building targeter) and the plays we made in the last few seconds (known at once, before perception
sees them) together pay for most of its elixir.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from clashbot.card_roles import GROUND_ONLY_SPELLS, hits_air, load_roles
from clashbot.config import ANYWHERE_CARDS
from clashbot.game_rules import action_key, lane_of, legal_cards, load_cards, zone_of
from clashbot.policy import NON_DAMAGE_SPELLS, OPPOSITE_LANE, SPELL_RADIUS, TANK_CARDS, ReactivePolicy, _lead_tile
from clashbot.state_text import fmt_num

URGENT_ETA_S = 8.0           # a push this close (seconds) to an own tower is acted on; later ones leave room for offence
PUSH_ETA_S = 10.0            # an enemy unit this close to an own tower is listed in a push (a building is placed early)
OWN_HALF_MAX_ROW = 14        # an enemy unit at or behind this row is on our half (rows 15-16 are the river)
NEAR_BACK_ROW = 8            # no ETA and this deep in our half = as urgent as at the tower (ETA 0), else ETA 2 s
MAX_PUSH_LANES = 2
MAX_MOVES = 6
MAX_REASON = 60
TRADE_SUFFIX_ROOM = 12          # characters kept free for ", trade +N" in a reason
LEAD_S = 1.0                 # the threat's position is led by this many seconds along `predicted_tile_2s`
PUNISH_OPP_ELIXIR = 3.0
LEAK_ELIXIR = 9.5
OWN_TOWER_X = {"left": 3.5, "right": 14.5}
ENEMY_TOWER_TILE = {"left": (3.5, 23.3), "right": (14.5, 23.3)}
BRIDGE_ROW = 14.0
MIN_X, MAX_X, MIN_Y, MAX_Y = 0.5, 17.5, 1.0, 14.0
BAIT_RECENT_S = 8.0          # a known small spell played this recently is still cycling (not in their hand)
BAIT_ELIXIR_MARGIN = 1.0     # bait / support / counter_push keep this much elixir after the card
PRESSURE_RESERVE = 3.0       # pressure keeps this much elixir for defence after the win condition
PRESSURE_OPP_MAX = 5.0       # pressure only while the opponent is at or below this estimate (or unknown)
SUPPORT_MIN_ROW = 12         # an ally win condition / tank at or beyond this row (bridge zone, enemy half) is supported
SUPPORT_BEHIND = 2.0         # a supporting ranged troop stands this many tiles behind the ally
SUPPORT_DEFENDER_RANGE = 6.0     # enemy units this close to the ally count as the ones defending against it
CYCLE_MIN_ELIXIR = 7.0
CYCLE_MAX_COST = 2.0
CYCLE_BACK_ROW = 1.5
ANSWERED_RANGE = 3.0         # ally units this close to the push's lead unit are fighting it
ANSWERED_BUILDING_RANGE = 7.0    # our buildings on our half this close to the lead unit answer a building targeter
ANSWERED_PLAY_S = 4.0        # our plays this recent (known from the tap, before perception sees them) count as answers
ANSWERED_PLAY_RANGE = 7.0    # ... when they landed this close to the lead unit (or in the push's lane); spells: ANSWERED_RANGE
ANSWERED_EMERGENCY_ETA = 1.5     # a push this close to the tower with weaker allies is still acted on in full
KAMIKAZE_REPEAT_S = 3.0      # a kamikaze card we played this recently is not proposed again against a push
NEED_ORDER = ("air", "tank", "swarm", "building_targeter")

# Every hand-set number of the move scoring. `TacticalEngine(weights=None)` starts from these, applies
# `resources/tactics_weights.json` if it exists, then an explicit dict; `tools/train/tune_tactics.py` tunes them.
DEFAULT_WEIGHTS: dict[str, float] = {
    # defence score = base + need bonuses / penalties + kg + trade terms - win-condition penalty
    "base": 0.5,
    "air_cannot_hit_penalty": 0.5,          # the card cannot hit air but the push has a flyer (and ground units too)
    "tank_killer_bonus": 0.6,               # tank push: tank_killer role or high dps
    "tank_building_bonus": 0.5,             # tank push with a building targeter: a building pulls it
    "tank_swarm_bonus": 0.3,                # tank push: a swarm card distracts it
    "tank_spell_penalty": 0.6,              # tank push: a spell that is not worth its cost
    "swarm_splash_bonus": 0.6,              # swarm push: splash troop or a spell that kills swarm
    "swarm_melee_penalty": 0.4,             # swarm push: single-target melee troop
    "targeter_building_bonus": 0.5,         # building-targeter push: a building
    "targeter_swarm_bonus": 0.3,            # ... a swarm or kamikaze card
    "targeter_mini_tank_bonus": 0.3,        # ... a mini tank
    "kg_factor": 0.2,                       # x the normalised KG counter score of the most urgent unit
    "trade_overpay_per_elixir": 0.15,       # per elixir of over-payment beyond the slack
    "trade_overpay_slack": 1.0,
    "trade_good_min": 1.0,                  # a trade of at least this many elixir ...
    "trade_good_bonus": 0.1,                # ... earns this
    "win_condition_penalty": 0.4,           # a win condition is kept for offence
    # stopping power against a push with medium / tank units (buildings and tank killers count as high, kamikaze as low)
    "stop_high": 0.3,
    "stop_medium": 0.15,
    "stop_low": -0.35,
    # a push our units already fight (ally units within ANSWERED_RANGE of its lead unit, enough elixir)
    "answered_frac": 0.6,                   # ally elixir / push elixir needed to call the push answered
    "answered_factor": 0.3,                 # the push's defence scores are multiplied by this
    # offence intent scores
    "counter_push": 0.6,
    "punish": 0.55,
    "leak_wc": 0.5,
    "leak": 0.4,
    "bait_counter_used": 0.6,               # their known small spell was played in the last BAIT_RECENT_S
    "bait_unknown": 0.45,                   # no small spell of theirs is known
    "bait_counter_ready": 0.25,             # a known small spell may be in their hand
    "pressure": 0.45,
    "support": 0.5,
    "cycle": 0.35,
    "cycle_block_score": 0.4,               # cycle only when no other move scores at least this
}
WEIGHTS_PATH = Path(__file__).resolve().parent / "resources" / "tactics_weights.json"


def _read_weights_file(path: Path) -> dict[str, float]:
    """The tuned weights of the committed file (`{"weights": {...}}` or a plain mapping); {} when it is missing.
    Keys the engine does not know are ignored, so an older or newer file never breaks the engine."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict) and isinstance(data.get("weights"), dict):
        data = data["weights"]
    if not isinstance(data, dict):
        return {}
    return {k: float(v) for k, v in data.items()
            if k in DEFAULT_WEIGHTS and isinstance(v, (int, float)) and not isinstance(v, bool)}


def resolve_weights(weights: dict | None = None, path: Path | str | None = None) -> dict[str, float]:
    """DEFAULT_WEIGHTS, updated from the weights file (`path`, default `WEIGHTS_PATH`), then from `weights` (an
    explicit dict wins over both; a partial dict only changes the keys it names)."""
    out = dict(DEFAULT_WEIGHTS)
    out.update(_read_weights_file(Path(path) if path is not None else WEIGHTS_PATH))
    if weights:
        unknown = sorted(set(weights) - set(DEFAULT_WEIGHTS))
        if unknown:
            raise ValueError(f"unknown tactics weights: {', '.join(unknown)}")
        out.update({k: float(v) for k, v in weights.items()})
    return out
SWARM_SPELL_UNITS = frozenset({"graveyard"})     # spell objects on the field that stand for a swarm (skeleton tokens)

_DEFAULT_TROOP = {"type": "troop", "targets": "ground", "flying": False, "hp_class": "medium",
                  "dps_class": "medium", "splash": False, "range": "melee", "speed": "medium", "count": 1,
                  "roles": []}


@dataclass
class Push:
    lane: str                    # "left" | "right" (middle units go to the lane of the nearer x: col < 9 left)
    units: list[dict]            # the enemy unit dicts of the state, most urgent first
    elixir: float                # sum of elixir_cost over distinct known cards
    eta_s: float                 # most urgent: min eta to an own tower; units on our half without eta -> 0.0 if row <= 8 else 2.0
    needs: list[str]             # subset of ["air", "tank", "swarm", "building_targeter"], in that order
    summary: str                 # "giant, musketeer (9 elixir), tower in 6.2s"
    answered: bool = False       # our units, buildings and recent plays already pay for most of it (see _answered_by)
    answered_by: list[str] = field(default_factory=list)      # the distinct cards that answer it, when answered
    heavy: bool = False          # the push has a medium or tank unit (by role hp_class): a light card does not stop it


@dataclass
class TacticalMove:
    card: str
    key: str                     # card|lane|zone of `tile` (lane_of / zone_of)
    tile: list[float]            # concrete deploy tile, rounded to 0.5
    intent: str                  # "defend" | "counter_push" | "punish" | "leak" | "bait" | "pressure" | "support" | "cycle"
    reason: str                  # <= 60 chars, e.g. "pulls giant to the centre, trade +5"
    score: float                 # engine preference, higher is better
    trade: float | None          # push elixir - card cost (defend), else None
    stop: str = "n/a"            # defend: the card's stopping class against that push: high | medium | low | spell; else n/a
    value: float | None = None   # defend with a spell: the elixir value of the units it hits (worth it when >= its cost)


@dataclass
class TacticalReport:
    pushes: list[Push]           # most urgent first
    moves: list[TacticalMove]    # best first, distinct keys, at most 6
    best: TacticalMove | None
    hold_reason: str | None      # set when best is None
    lines: list[str]             # 1..4 lines, each starting with "tactics: "
    defence: list[TacticalMove] = field(default_factory=list)   # every defend move, best first (`moves` keeps 6 of all)


@dataclass
class _PushCtx:
    """What the defence rules need about one push: its most urgent unit and where that unit is going."""
    push: Push
    unit: dict
    pos: list[float]             # the most urgent unit's tile, led by LEAD_S
    tower_x: float               # x of the own princess tower of the push lane
    toward_centre: int           # +1 lane left, -1 lane right
    threat: str                  # name of the most urgent unit
    all_air: bool
    kg_norm: dict[str, float]    # KG counter scores of the threat, scaled to max 1
    heavy: bool = False          # the push has a medium or tank unit (by role hp_class)
    swarm_only: bool = False     # every unit of the push is a swarm unit (or a graveyard token)
    answered_by: tuple = ()      # distinct ally cards fighting the push when it counts as being answered, else ()
    blocked: frozenset = frozenset()     # kamikaze cards we played in the last KAMIKAZE_REPEAT_S (not proposed again)


def _tile(unit: dict) -> list | None:
    tile = unit.get("tile")
    ok = (isinstance(tile, (list, tuple)) and len(tile) >= 2
          and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in tile[:2]))
    return list(tile[:2]) if ok else None


def _side_lane(tile) -> str:
    """left / right; middle tiles go to the nearer x (col < 9 left)."""
    lane = lane_of(tile)
    if lane == "middle":
        return "left" if tile[0] < 9 else "right"
    return lane


def _own_eta(unit: dict) -> float | None:
    eta = unit.get("eta_tower")
    if isinstance(eta, dict) and str(eta.get("tower", "")).startswith("own") and eta.get("s") is not None:
        return float(eta["s"])
    return None


def _name(unit: dict) -> str:
    return unit.get("card") or unit.get("unit_class") or "unit"


def _r05(v: float) -> float:
    return math.floor(v * 2 + 0.5) / 2


def _clamp_tile(x: float, y: float) -> list[float]:
    """Clamp a troop / building tile to the deployable part of our half, then round to 0.5."""
    return [_r05(min(max(x, MIN_X), MAX_X)), _r05(min(max(y, MIN_Y), MAX_Y))]


def _with_trade(base: str, trade: float) -> str:
    suffix = f", trade {'+' if trade >= 0 else '-'}{fmt_num(abs(trade))}"
    return base[:max(0, MAX_REASON - len(suffix))] + suffix


def _clip(text: str) -> str:
    return text[:MAX_REASON]


class TacticalEngine:
    def __init__(self, cards: dict | None = None, roles: dict | None = None, kg=None, weights: dict | None = None):
        self.cards = cards if cards is not None else load_cards()
        # the scoring numbers: defaults < tactics_weights.json < the explicit dict (read as `self.w[key]` at use time)
        self.w = resolve_weights(weights)
        if roles is None:
            roles = load_roles()
        # accept both the file layout ({"version", "cards": {...}}) and the plain card -> role mapping
        self.roles = roles["cards"] if isinstance(roles.get("cards"), dict) else roles
        self.kg = kg
        self.policy = ReactivePolicy(self.cards)
        self._fallback: dict[str, dict] = {}

    # -- cards and roles -----------------------------------------------------------------------------------------------
    def _cost(self, card: str) -> float:
        return float((self.cards.get(card) or {}).get("elixir_cost") or 0.0)

    def _role(self, card: str | None) -> dict:
        """The role entry of `card`. A card (or unit) without one counts as a medium ground unit of its card type."""
        role = self.roles.get(card) if card else None
        if role:
            return role
        key = card or ""
        if key not in self._fallback:
            ctype = (self.cards.get(card) or {}).get("card_type") if card else None
            if ctype == "Spell":
                fb = {"type": "spell", "spell_class": "medium", "radius": 2.5, "kills": [], "roles": []}
            else:
                fb = dict(_DEFAULT_TROOP, roles=[])
                if ctype == "Building":
                    fb["type"] = "building"
                    fb["range"] = "long"
                    fb["speed"] = None
            self._fallback[key] = fb
        return self._fallback[key]

    def _hits_air(self, card: str, role: dict) -> bool:
        """`card_roles.hits_air` for a card with a role entry; for one without (fallback role) the same rule keyed on
        the slug: spells hit air except the ground-only ones, everything else by `targets == "air_ground"`."""
        if card in self.roles:
            return hits_air(role)
        if role.get("type") == "spell":
            return card not in GROUND_ONLY_SPELLS
        return role.get("targets") == "air_ground"

    @staticmethod
    def _has(role: dict, name: str) -> bool:
        return name in (role.get("roles") or [])

    # -- pushes --------------------------------------------------------------------------------------------------------
    def _pushes(self, state: dict) -> list[Push]:
        """Enemy units close to an own tower (ETA <= PUSH_ETA_S) or on our half, grouped by lane, most urgent push
        first."""
        per_lane: dict[str, list[tuple[float, float, int, dict]]] = {"left": [], "right": []}
        for i, u in enumerate(state.get("units", [])):
            tile = _tile(u)
            if u.get("side") != "enemy" or tile is None:
                continue
            card = u.get("card")
            if card not in SWARM_SPELL_UNITS and self._role(card).get("type") == "spell":
                continue                                    # a spell object is an effect, not a troop to answer
            eta = _own_eta(u)
            on_our_half = tile[1] <= OWN_HALF_MAX_ROW
            if not ((eta is not None and eta <= PUSH_ETA_S) or on_our_half):
                continue
            if eta is None:
                eta = 0.0 if tile[1] <= NEAR_BACK_ROW else 2.0
            per_lane[_side_lane(tile)].append((eta, tile[1], i, u))
        pushes = []
        for lane, items in per_lane.items():
            if not items:
                continue
            items.sort(key=lambda x: x[:3])                           # most urgent first: ETA, nearest our side, order
            pushes.append(self._make_push(lane, [x[3] for x in items], items[0][0]))
        pushes.sort(key=lambda p: (p.eta_s, -p.elixir, p.lane))
        return pushes[:MAX_PUSH_LANES]

    def _make_push(self, lane: str, units: list[dict], eta_s: float) -> Push:
        roles = [self._role(u.get("card")) for u in units]
        known = {u.get("card") for u in units if (self.cards.get(u.get("card")) or {}).get("elixir_cost")}
        elixir = sum(self._cost(c) for c in known)
        swarm_units = sum(1 for r in roles if r.get("hp_class") == "swarm")
        needs = {
            "air": any(r.get("flying") for r in roles),
            "tank": any(r.get("hp_class") == "tank" for r in roles),
            "swarm": (swarm_units >= 3 or any(r.get("hp_class") == "swarm" and (r.get("count") or 1) >= 3
                                              for r in roles)
                      or any(u.get("card") in SWARM_SPELL_UNITS for u in units)),
            "building_targeter": any(r.get("targets") == "buildings" for r in roles),
        }
        counts: dict[str, int] = {}
        for u in units:
            counts[_name(u)] = counts.get(_name(u), 0) + 1
        parts = [n if k == 1 else f"{n} x{k}" for n, k in counts.items()]
        names = ", ".join(parts[:4]) + (f" +{len(parts) - 4}" if len(parts) > 4 else "")
        summary = f"{names} ({fmt_num(elixir)} elixir), tower in {eta_s:.1f}s"
        return Push(lane, units, elixir, eta_s, [n for n in NEED_ORDER if needs[n]], summary,
                    heavy=any(r.get("hp_class") in ("medium", "tank") for r in roles))

    def _is_building(self, card: str) -> bool:
        return (self.cards.get(card) or {}).get("card_type") == "Building" or self._has(self._role(card), "building")

    def _is_spell(self, card: str) -> bool:
        return (self.cards.get(card) or {}).get("card_type") == "Spell" or self._role(card).get("type") == "spell"

    @staticmethod
    def _clock(state: dict, t: float | None) -> float | None:
        """`t`, else the newest play time of either side (the engine given no clock reads that as "now")."""
        if t is not None:
            return t
        stamps = [p["t"] for side in ("own", "opponent") for p in (state.get(side) or {}).get("recent_plays") or []
                  if p.get("t") is not None]
        return max(stamps, default=None)

    def _answered_by(self, push: Push, pos: list[float], state: dict, t: float | None = None) -> tuple:
        """The distinct cards that answer the push when their elixir is at least `answered_frac` x the push's (and,
        for a push at the tower within ANSWERED_EMERGENCY_ETA, at least the push's); () when it is not answered.
        They are (a) ally units within ANSWERED_RANGE of the push's lead tile, (b) against a building targeter, our
        buildings on our half within ANSWERED_BUILDING_RANGE of it, and (c) our own plays of the last ANSWERED_PLAY_S
        seconds that landed on our half, in the push's lane or within ANSWERED_PLAY_RANGE of the lead tile (a spell
        only within ANSWERED_RANGE). The plays are known from our tap, so a new building counts at once."""
        names: list[str] = []

        def add(card) -> None:
            if card and card not in names:
                names.append(card)

        targeter = "building_targeter" in push.needs
        for u in state.get("units", []):
            tile, card = _tile(u), u.get("card")
            if u.get("side") != "ally" or tile is None or not card or self._is_spell(card):
                continue
            d = math.dist(tile, pos)
            if d <= ANSWERED_RANGE or (targeter and d <= ANSWERED_BUILDING_RANGE and tile[1] <= OWN_HALF_MAX_ROW
                                       and self._is_building(card)):
                add(card)
        now = self._clock(state, t)
        for p in (state.get("own") or {}).get("recent_plays") or []:
            tile, card = _tile(p), p.get("card")
            if (now is None or p.get("t") is None or p["t"] < now - ANSWERED_PLAY_S or tile is None or not card
                    or tile[1] > OWN_HALF_MAX_ROW):
                continue
            d = math.dist(tile, pos)
            if self._is_spell(card):
                if d <= ANSWERED_RANGE:
                    add(card)
            elif _side_lane(tile) == push.lane or d <= ANSWERED_PLAY_RANGE:
                add(card)
        ally_elixir = sum(self._cost(c) for c in names)
        if ally_elixir <= 0 or ally_elixir < self.w["answered_frac"] * push.elixir:
            return ()
        if push.eta_s <= ANSWERED_EMERGENCY_ETA and ally_elixir < push.elixir:
            return ()
        return tuple(names)

    def _recent_kamikazes(self, state: dict, t: float | None) -> frozenset:
        """Kamikaze cards we played in the last KAMIKAZE_REPEAT_S seconds (needs the clock `t`)."""
        if t is None:
            return frozenset()
        return frozenset(p["card"] for p in (state.get("own") or {}).get("recent_plays") or []
                         if p.get("card") and p.get("t") is not None and 0 <= t - p["t"] <= KAMIKAZE_REPEAT_S
                         and self._has(self._role(p["card"]), "kamikaze"))

    def _ctx(self, push: Push, state: dict, t: float | None = None) -> _PushCtx:
        """The defence context of the push; `t` is the clock of the play times (our recent plays that answer the push,
        the kamikaze cards we played a moment ago)."""
        unit = push.units[0]
        pos = _lead_tile(unit, LEAD_S) or list(_tile(unit))
        card = unit.get("card")
        norm: dict[str, float] = {}
        if self.kg is not None and card:
            scores = self.kg.counter_scores(card) or {}
            top = max(scores.values(), default=0.0)
            if top > 0:
                norm = {k: v / top for k, v in scores.items()}
        roles = [self._role(u.get("card")) for u in push.units]
        return _PushCtx(push, unit, pos, OWN_TOWER_X[push.lane], 1 if push.lane == "left" else -1, _name(unit),
                        all(r.get("flying") for r in roles), norm,
                        heavy=any(r.get("hp_class") in ("medium", "tank") for r in roles),
                        swarm_only=all(r.get("hp_class") == "swarm" or u.get("card") in SWARM_SPELL_UNITS
                                       for r, u in zip(roles, push.units)),
                        answered_by=self._answered_by(push, pos, state, t),
                        blocked=self._recent_kamikazes(state, t))

    # -- defence -------------------------------------------------------------------------------------------------------
    def _defends(self, card: str, role: dict) -> bool:
        """Spells that do damage can defend; rage / clone / freeze ... and the win-condition spells cannot. Troops that
        only attack buildings (hog rider, giant, wall breakers ...) are kept for offence."""
        if role.get("type") != "spell":
            return not (role.get("type") == "troop" and role.get("targets") == "buildings")
        return (card not in NON_DAMAGE_SPELLS and role.get("spell_class") != "utility"
                and not self._has(role, "win_condition"))

    def _spell_aim(self, card: str, ctx: _PushCtx, state: dict):
        """`spell_target` picks its centres by the units' current tiles, so ask for the cell the threat stands in now;
        if nothing worthwhile is there, for the cell of its lead tile."""
        now = _tile(ctx.unit)
        for tile in (now, ctx.pos):
            aim = self.policy.spell_target(card, lane_of(tile), zone_of(tile), state)
            if aim is not None:
                return aim
        return None

    def _stop_class(self, role: dict, ctx: _PushCtx) -> str | None:
        """How well the card stops a push: high / medium / low, None for spells (judged by their aim). Buildings and
        tank killers count as high; a kamikaze card as low against a push with medium / tank units. A low-dps card
        with the air_defense role counts as medium against a push with flyers (that is what the role is for)."""
        if role.get("type") == "spell":
            return None
        if role.get("type") == "building" or self._has(role, "tank_killer"):
            return "high"
        if ctx.heavy and self._has(role, "kamikaze"):
            return "low"
        dps = role.get("dps_class")
        dps = dps if dps in ("high", "medium", "low") else "medium"
        if dps == "low" and "air" in ctx.push.needs and self._has(role, "air_defense"):
            return "medium"
        return dps

    def _claims_trade(self, role: dict, ctx: _PushCtx) -> bool:
        """The elixir trade counts (bonus, reason text) when the card stops the push at least at medium, or the push is
        only swarm units; a low-stopping card only slows it."""
        return self._stop_class(role, ctx) in (None, "high", "medium") or ctx.swarm_only

    def _defence_score(self, card: str, role: dict, ctx: _PushCtx, aim) -> tuple[float, float] | None:
        """(score, trade) of `card` against the push, or None when it cannot answer an air-only push."""
        push, needs = ctx.push, ctx.push.needs
        w = self.w
        cost = self._cost(card)
        is_spell, is_building = role.get("type") == "spell", role.get("type") == "building"
        score = w["base"]
        if "air" in needs and not self._hits_air(card, role):
            if ctx.all_air:
                return None
            score -= w["air_cannot_hit_penalty"]
        if "tank" in needs:
            if self._has(role, "tank_killer") or role.get("dps_class") == "high":
                score += w["tank_killer_bonus"]
            if is_building and "building_targeter" in needs:
                score += w["tank_building_bonus"]
            if self._has(role, "swarm"):
                score += w["tank_swarm_bonus"]
            if is_spell and not (aim is not None and aim[1] >= cost):
                score -= w["tank_spell_penalty"]
        if "swarm" in needs:
            if role.get("splash") or (is_spell and "swarm" in (role.get("kills") or [])):
                score += w["swarm_splash_bonus"]
            elif role.get("type") == "troop" and role.get("range") == "melee":
                score -= w["swarm_melee_penalty"]
        if "building_targeter" in needs:
            if is_building:
                score += w["targeter_building_bonus"]
            if self._has(role, "swarm") or self._has(role, "kamikaze"):
                score += w["targeter_swarm_bonus"]
            if self._has(role, "mini_tank"):
                score += w["targeter_mini_tank_bonus"]
        if ctx.heavy:
            stop = self._stop_class(role, ctx)
            if stop is not None:
                score += w["stop_" + stop]
        score += w["kg_factor"] * ctx.kg_norm.get(card, 0.0)
        trade = push.elixir - cost
        score -= w["trade_overpay_per_elixir"] * max(0.0, -trade - w["trade_overpay_slack"])
        if trade >= w["trade_good_min"] and self._claims_trade(role, ctx):
            score += w["trade_good_bonus"]
        if self._has(role, "win_condition"):
            score -= w["win_condition_penalty"]                              # kept for offence
        if ctx.answered_by:
            score *= w["answered_factor"]                                    # our units are already fighting it
        return round(score, 3) + 0.0, trade                                # + 0.0: no -0.0

    def _defence_place(self, card: str, role: dict, ctx: _PushCtx, aim, state: dict) -> tuple[list[float], str]:
        """(tile, reason without the trade suffix) for a non-spell card; spells use their aim."""
        pos, push, threat = ctx.pos, ctx.push, ctx.threat
        if role.get("type") == "spell":
            tile = [_r05(aim[0][0]), _r05(aim[0][1])]
            near = [_name(u) for u in push.units
                    if math.dist(tile, _tile(u)) <= SPELL_RADIUS.get(card, 2.5) + 0.5
                    and _name(u) not in TANK_CARDS and self._role(u.get("card")).get("hp_class") != "tank"]
            names = list(dict.fromkeys(near)) or [threat]
            while True:                                                 # leave room for ", trade +N"
                why = f"{card} {' + '.join(names)} ({fmt_num(aim[1])} elixir)"
                if len(why) <= MAX_REASON - TRADE_SUFFIX_ROOM or len(names) == 1:
                    return tile, why
                names = names[:-1]
        if role.get("type") == "building":
            if "building_targeter" in push.needs:
                x = 7.5 if push.lane == "left" else 10.5
                return _clamp_tile(x, min(max(pos[1] - 4, 6), 10)), f"pulls {threat} to the centre"
            return _clamp_tile(ctx.tower_x + 2 * ctx.toward_centre, 6.5), f"blocks {threat} at the tower"
        # a specific role (kamikaze, swarm) beats the generic range rules
        if self._has(role, "kamikaze"):
            return _clamp_tile(pos[0], pos[1] - 2.5), f"blows up on {threat} up front"
        if self._has(role, "swarm"):
            return _clamp_tile(pos[0], pos[1]), f"swarm on {threat}"
        front = any(self._has(role, r) for r in ("tank", "mini_tank", "tank_killer"))
        if front or role.get("range", "melee") == "melee":
            y = max(pos[1] - 2.5, 1.0) if pos[1] > 8 else pos[1] - 1
            if self._has(role, "tank_killer"):
                why = f"stops {threat} in front of the tower"
            elif self._has(role, "mini_tank") or self._has(role, "tank"):
                why = f"tanks for the tower vs {threat}"
            else:
                why = f"meets {threat} in front of the tower"
            return _clamp_tile(pos[0], y), why
        return (_clamp_tile(ctx.tower_x + 1.5 * ctx.toward_centre, 2.5),
                f"kills {threat} from behind the tower")

    def _defence_moves(self, ctx: _PushCtx, hand: list[str], state: dict) -> list[TacticalMove]:
        moves = []
        for card in hand:
            role = self._role(card)
            if not self._defends(card, role) or card in ctx.blocked:
                continue
            aim = None
            if role.get("type") == "spell":
                aim = self._spell_aim(card, ctx, state)
                if aim is None:
                    continue
            scored = self._defence_score(card, role, ctx, aim)
            if scored is None:
                continue
            score, trade = scored
            tile, why = self._defence_place(card, role, ctx, aim, state)
            key = action_key(card, lane_of(tile), zone_of(tile))
            reason = _with_trade(why, trade) if self._claims_trade(role, ctx) else _clip(f"slows {ctx.threat}")
            moves.append(TacticalMove(card, key, tile, "defend", reason, score, trade,
                                      stop=self._stop_class(role, ctx) or "spell",
                                      value=None if aim is None else float(aim[1])))
        return moves

    # -- offence -------------------------------------------------------------------------------------------------------
    def _is_aimed_wc(self, card: str, role: dict) -> bool:
        """Win conditions that go straight on the enemy princess tower: spells (barrel, graveyard) and miner-likes."""
        return role.get("type") == "spell" or card in ANYWHERE_CARDS

    def _wc_tile(self, card: str, role: dict, lane: str) -> list[float]:
        if self._is_aimed_wc(card, role):
            x, y = ENEMY_TOWER_TILE[lane]
            return [_r05(x), _r05(y)]
        return [OWN_TOWER_X[lane], BRIDGE_ROW]                  # the bridge of the lane (x of the princess towers)

    def _weak_lane(self, state: dict) -> str | None:
        """The lane of the enemy princess tower with the lower HP; None when unknown or equal."""
        towers = (state.get("opponent") or {}).get("towers") or {}
        left, right = towers.get("left"), towers.get("right")
        nums = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (left, right))
        if nums and left != right:
            return "left" if left < right else "right"
        return None

    def _opp_last_lane(self, state: dict) -> str | None:
        plays = [p for p in (state.get("opponent") or {}).get("recent_plays", []) if _tile(p)]
        if not plays:
            return None
        last = max(reversed(plays), key=lambda p: p["t"] if p.get("t") is not None else float("-inf"))
        return OPPOSITE_LANE.get(lane_of(_tile(last)))

    def _bait_lane(self, state: dict) -> str:
        """Lane L of the bait / pressure moves: the weaker enemy princess tower, else opposite their newest play."""
        return self._weak_lane(state) or self._opp_last_lane(state) or "left"

    def _quiet_lane(self, state: dict) -> str:
        """The lane with fewer enemy units on the field (ties: left)."""
        counts = {"left": 0, "right": 0}
        for u in state.get("units", []):
            tile = _tile(u)
            if u.get("side") == "enemy" and tile is not None and self._role(u.get("card")).get("type") != "spell":
                counts[_side_lane(tile)] += 1
        return "left" if counts["left"] <= counts["right"] else "right"

    def _small_spell(self, card: str) -> bool:
        """A damage spell that kills a goblin-barrel swarm: role spell_small, or a spell whose kills include swarm."""
        role = self._role(card)
        return (role.get("type") == "spell" and card not in NON_DAMAGE_SPELLS and role.get("spell_class") != "utility"
                and (self._has(role, "spell_small") or "swarm" in (role.get("kills") or [])))

    def _bait_read(self, state: dict, t: float | None) -> tuple[float, str]:
        """(score, reason) of a bait now: their known small spells that answer it, and whether they are cycling."""
        w = self.w
        opp = state.get("opponent") or {}
        small = [c for c in opp.get("deck_known") or [] if self._small_spell(c)]
        if not small:
            return w["bait_unknown"], "no small spell of theirs seen"
        plays = list(opp.get("recent_plays") or [])
        if t is None:                                   # no clock given: the newest play of either side stands for now
            stamps = [p["t"] for p in plays + list((state.get("own") or {}).get("recent_plays") or [])
                      if p.get("t") is not None]
            t = max(stamps, default=None)
        last: dict[str, float] = {}
        for p in plays:
            if p.get("card") in small and p.get("t") is not None:
                last[p["card"]] = max(last.get(p["card"], float("-inf")), p["t"])
        ready = [c for c in small if not (c in last and t is not None and 0 <= t - last[c] <= BAIT_RECENT_S)]
        if not ready:
            newest = max(small, key=lambda c: (last[c], c))
            return w["bait_counter_used"], f"their {newest} is cycling"
        return w["bait_counter_ready"], f"their {ready[0]} may be ready"

    def _bait_moves(self, state: dict, hand: list[str], elixir: float, t: float | None) -> list[TacticalMove]:
        """A spell-like win condition (goblin barrel, graveyard, miner, goblin drill) on the enemy princess tower."""
        cards = [c for c in hand if self._has(self._role(c), "win_condition")
                 and self._is_aimed_wc(c, self._role(c)) and elixir >= self._cost(c) + BAIT_ELIXIR_MARGIN]
        if not cards:
            return []
        lane = self._bait_lane(state)
        score, why = self._bait_read(state, t)
        moves = []
        for c in cards:
            tile = self._wc_tile(c, self._role(c), lane)
            moves.append(TacticalMove(c, action_key(c, lane_of(tile), zone_of(tile)), tile, "bait",
                                      _clip(why), score, None))
        return moves

    def _pressure_moves(self, state: dict, hand: list[str], elixir: float) -> list[TacticalMove]:
        """A troop win condition at the bridge while the opponent is low and we keep an elixir reserve."""
        opp_elixir = (state.get("opponent") or {}).get("elixir_estimate")
        if opp_elixir is not None and opp_elixir > PRESSURE_OPP_MAX:
            return []
        lane = self._bait_lane(state)
        moves = []
        for c in hand:
            role = self._role(c)
            if (self._has(role, "win_condition") and not self._is_aimed_wc(c, role)
                    and elixir >= self._cost(c) + PRESSURE_RESERVE):
                tile = self._wc_tile(c, role, lane)
                why = f"{c} at the {lane} bridge, {fmt_num(elixir - self._cost(c))} elixir kept back"
                moves.append(TacticalMove(c, action_key(c, lane_of(tile), zone_of(tile)), tile, "pressure",
                                          _clip(why), self.w["pressure"], None))
        return moves

    def _support_spell_aim(self, card: str, ally_tile: list[float], state: dict):
        """(tile, value, count) of the best small-spell aim on the enemy units near the ally (the ones defending)."""
        cells: list[tuple[str, str]] = []
        for u in state.get("units", []):
            tile = _tile(u)
            if (u.get("side") == "enemy" and tile is not None and math.dist(tile, ally_tile) <= SUPPORT_DEFENDER_RANGE
                    and (lane_of(tile), zone_of(tile)) not in cells):
                cells.append((lane_of(tile), zone_of(tile)))
        reach = SUPPORT_DEFENDER_RANGE + SPELL_RADIUS.get(card, 2.5)
        best = None
        for lane, zone in cells:
            aim = self.policy.spell_target(card, lane, zone, state)
            if aim is not None and math.dist(aim[0], ally_tile) <= reach and (best is None or aim[1:] > best[1:]):
                best = aim
        return best

    def _support_moves(self, state: dict, hand: list[str], elixir: float) -> list[TacticalMove]:
        """Back an ally win condition / tank that has crossed to the bridge or the enemy half: a ranged troop two
        tiles behind it, or a small damage spell on the units defending against it."""
        lead: dict[str, tuple[list[float], dict]] = {}              # per lane, the most advanced such ally
        for u in state.get("units", []):
            tile = _tile(u)
            if u.get("side") != "ally" or tile is None or tile[1] < SUPPORT_MIN_ROW:
                continue
            role = self._role(u.get("card"))
            if not (self._has(role, "win_condition") or self._has(role, "tank")):
                continue
            lane = _side_lane(tile)
            if lane not in lead or tile[1] > lead[lane][0][1]:
                lead[lane] = (tile, u)
        moves: list[TacticalMove] = []
        for lane in ("left", "right"):
            if lane not in lead:
                continue
            tile, ally = lead[lane]
            name = _name(ally)
            for c in hand:
                role = self._role(c)
                if elixir < self._cost(c) + BAIT_ELIXIR_MARGIN:
                    continue
                if (role.get("type") == "troop" and role.get("range") in ("short", "long")
                        and role.get("targets") != "buildings" and not self._has(role, "win_condition")):
                    spot = _clamp_tile(tile[0], tile[1] - SUPPORT_BEHIND)
                    why = f"{c} behind {name}"
                elif (role.get("type") == "spell" and role.get("spell_class") == "small"
                      and c not in NON_DAMAGE_SPELLS and not self._has(role, "win_condition")):
                    aim = self._support_spell_aim(c, tile, state)
                    if aim is None:
                        continue
                    spot = [_r05(aim[0][0]), _r05(aim[0][1])]
                    why = f"{c} on the defenders of {name} ({fmt_num(aim[1])} elixir)"
                else:
                    continue
                moves.append(TacticalMove(c, action_key(c, lane_of(spot), zone_of(spot)), spot, "support",
                                          _clip(why), self.w["support"], None))
        return moves

    def _cycle_moves(self, state: dict, hand: list[str], elixir: float,
                     moves: list[TacticalMove]) -> list[TacticalMove]:
        """Rotate the hand with a cheap card when elixir is high and nothing else is worth playing."""
        if elixir < CYCLE_MIN_ELIXIR or any(m.score >= self.w["cycle_block_score"] for m in moves):
            return []
        plain = sorted((c for c in hand if self._role(c).get("type") != "spell"
                        and not self._has(self._role(c), "win_condition")), key=lambda c: (self._cost(c), c))
        if not plain or self._cost(plain[0]) > CYCLE_MAX_COST:
            return []
        c = plain[0]
        if self._has(self._role(c), "kamikaze"):                   # chip damage on the weaker tower
            lane = self._weak_lane(state) or "left"
            tile = [OWN_TOWER_X[lane], BRIDGE_ROW]
            why = f"{c} chips the {lane} tower, cycles"
        else:
            lane = self._quiet_lane(state)
            tile = _clamp_tile(OWN_TOWER_X[lane] + 1.5 * (1 if lane == "left" else -1), CYCLE_BACK_ROW)
            why = f"{c} cycles at the back, {lane} lane is quiet"
        return [TacticalMove(c, action_key(c, lane_of(tile), zone_of(tile)), tile, "cycle", _clip(why),
                             self.w["cycle"], None)]

    def _offence_moves(self, state: dict, hand: list[str], t: float | None = None) -> list[TacticalMove]:
        own, opp = state["own"], state.get("opponent") or {}
        elixir = own.get("elixir") or 0.0
        w = self.w
        wcs = [c for c in hand if self._has(self._role(c), "win_condition")]
        moves: list[TacticalMove] = []

        def add(card: str, lane: str, intent: str, score: float, reason: str) -> None:
            tile = self._wc_tile(card, self._role(card), lane)
            moves.append(TacticalMove(card, action_key(card, lane_of(tile), zone_of(tile)), tile, intent,
                                      _clip(reason), score, None))

        # counter-push: back an ally unit that is already across the middle of our half
        allies: dict[str, dict] = {}
        for u in state.get("units", []):
            tile = _tile(u)
            if u.get("side") == "ally" and tile is not None and tile[1] >= 9:
                lane = _side_lane(tile)
                if lane not in allies or tile[1] > _tile(allies[lane])[1]:
                    allies[lane] = u
        for lane in ("left", "right"):
            if lane in allies:
                for c in wcs:
                    if elixir >= self._cost(c) + 1:
                        add(c, lane, "counter_push", w["counter_push"],
                            f"backs {_name(allies[lane])} in the {lane} lane")

        # punish: the opponent has spent down, hit the lane away from its last play
        opp_elixir = opp.get("elixir_estimate")
        if opp_elixir is not None and opp_elixir <= PUNISH_OPP_ELIXIR:
            lane = self._opp_last_lane(state) or self._weak_lane(state) or "left"
            for c in wcs:
                if elixir >= self._cost(c) + 1:
                    add(c, lane, "punish", w["punish"],
                        f"opponent ~{fmt_num(opp_elixir)} elixir, hits the {lane} lane")

        # leak: a full bar is wasted elixir
        if elixir >= LEAK_ELIXIR:
            lane = self._weak_lane(state) or "left"
            if wcs:
                for c in wcs:
                    add(c, lane, "leak", w["leak_wc"], f"elixir {fmt_num(elixir)} near full, pushes {lane}")
            else:
                plain = sorted((c for c in hand if self._role(c).get("type") != "spell"),
                               key=lambda c: (self._cost(c), c))
                if plain:
                    c = plain[0]
                    if self._role(c).get("hp_class") == "tank":
                        tile = _clamp_tile(9.0, 1.0)                               # behind the king
                    else:
                        tile = _clamp_tile(OWN_TOWER_X[lane] + 1.5 * (1 if lane == "left" else -1), 2.5)
                    moves.append(TacticalMove(c, action_key(c, lane_of(tile), zone_of(tile)), tile, "leak",
                                              _clip(f"elixir {fmt_num(elixir)} near full, cycles {c}"),
                                              w["leak"], None))

        # the bait-deck intents (no push in sight): bait, pressure, support, then cycle if nothing else is worth it
        moves += self._bait_moves(state, hand, elixir, t)
        moves += self._pressure_moves(state, hand, elixir)
        moves += self._support_moves(state, hand, elixir)
        moves += self._cycle_moves(state, hand, elixir, moves)
        return moves

    # -- report --------------------------------------------------------------------------------------------------------
    def _rank(self, moves: list[TacticalMove], limit: int | None = MAX_MOVES) -> list[TacticalMove]:
        """Best first (ties: lower cost, then card name), the first move per key, at most `limit`."""
        out, seen = [], set()
        for m in sorted(moves, key=lambda m: (-m.score, self._cost(m.card), m.card)):
            if m.key not in seen:
                seen.add(m.key)
                out.append(m)
        return out[:limit]

    def _hold_reason(self, state: dict, pushes: list[Push], hand: list[str]) -> str:
        if pushes:
            return "no card in hand answers the push" if hand else "not enough elixir"
        own, opp = state["own"], state.get("opponent") or {}
        return (f"no threat, saving elixir (own {fmt_num(own.get('elixir'))}, "
                f"opp ~{fmt_num(opp.get('elixir_estimate'))})")

    def _lines(self, pushes: list[Push], moves: list[TacticalMove], hold_reason: str | None) -> list[str]:
        lines = [f"tactics: push {p.lane}: {p.summary}; needs {', '.join(p.needs) or 'nothing special'}"
                 + (f" (being answered by {', '.join(p.answered_by[:3])})" if p.answered else "")
                 for p in pushes[:MAX_PUSH_LANES]]

        def at(m: TacticalMove) -> str:
            return f"{m.card} @[{fmt_num(m.tile[0])},{fmt_num(m.tile[1])}]"

        if moves:
            lines.append(f"tactics: best {at(moves[0])} {moves[0].intent}: {moves[0].reason}")
            if len(moves) > 1:
                lines.append("tactics: also " + "; ".join(f"{at(m)} ({m.reason})" for m in moves[1:3]))
        else:
            lines.append(f"tactics: hold: {hold_reason}")
        return lines

    def analyse(self, state: dict, t: float | None = None) -> TacticalReport:
        """`t` is the clock of the state's play times (the live loop and the dataset pass it). `bait` uses it to tell
        whether their small spell was played recently (without it the newest play of either side stands for now), the
        defence to skip a kamikaze card we played in the last KAMIKAZE_REPEAT_S (without it the rule is off), and the
        answered check to find our plays of the last ANSWERED_PLAY_S seconds (without it: the newest play stands for
        now)."""
        hand = legal_cards(state, self.cards)
        pushes = self._pushes(state)
        moves: list[TacticalMove] = []
        for push in pushes:
            ctx = self._ctx(push, state, t)
            push.answered, push.answered_by = bool(ctx.answered_by), list(ctx.answered_by)
            moves += self._defence_moves(ctx, hand, state)
        defence = self._rank(moves, None)
        if not any(p.eta_s <= URGENT_ETA_S for p in pushes):
            moves += self._offence_moves(state, hand, t)
        moves = self._rank(moves)
        best = moves[0] if moves else None
        hold = None if best else self._hold_reason(state, pushes, hand)
        return TacticalReport(pushes, moves, best, hold, self._lines(pushes, moves, hold), defence)

    def score_key(self, key: str, state: dict, report: TacticalReport | None = None) -> float | None:
        """The engine's preference for the action key: the score of the first proposed move with that key, else the
        defence score of the key's card against the most urgent push (hand cards only), else None. None also when
        there is no push, or the card cannot answer it (an air-only push and a card that cannot hit air)."""
        report = report if report is not None else self.analyse(state)
        for m in report.moves:
            if m.key == key:
                return m.score
        card = str(key).split("|")[0]
        if card not in legal_cards(state, self.cards) or not report.pushes:
            return None
        role = self._role(card)
        if not self._defends(card, role):
            return None
        ctx = self._ctx(report.pushes[0], state)
        aim = self._spell_aim(card, ctx, state) if role.get("type") == "spell" else None
        scored = self._defence_score(card, role, ctx, aim)
        return None if scored is None else scored[0]
