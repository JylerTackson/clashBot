"""Tactical engine (D-042, MAC-Z2): which enemy push threatens which tower, which hand card answers it and where to
put it, and when there is an offence window.

`TacticalEngine.analyse(state)` returns a `TacticalReport`: the enemy pushes (most urgent first), up to 6 concrete
moves (card + deploy tile + intent + reason, best first) and 1..4 text lines for the d1 context. The card roles come
from the Z1 table (`clashbot.card_roles`, loaded lazily); spell aim and value come from `ReactivePolicy.spell_target`
(MAC-Y2). Deterministic, no I/O after `__init__`, no models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from clashbot.config import ANYWHERE_CARDS
from clashbot.game_rules import action_key, lane_of, legal_cards, load_cards, zone_of
from clashbot.policy import NON_DAMAGE_SPELLS, OPPOSITE_LANE, SPELL_RADIUS, TANK_CARDS, ReactivePolicy, _lead_tile
from clashbot.state_text import fmt_num

URGENT_ETA_S = 8.0           # a push this close (seconds) to an own tower is acted on; later ones leave room for offence
OWN_HALF_MAX_ROW = 14        # an enemy unit at or behind this row is on our half (rows 15-16 are the river)
NEAR_BACK_ROW = 8            # no ETA and this deep in our half = as urgent as at the tower (ETA 0), else ETA 2 s
MAX_PUSH_LANES = 2
MAX_MOVES = 6
MAX_REASON = 60
TRADE_SUFFIX_ROOM = 12          # characters kept free for ", trade +N" in a reason
LEAD_S = 1.0                 # the threat's position is led by this many seconds along `predicted_tile_2s`
BASE_SCORE = 0.5
PUNISH_OPP_ELIXIR = 3.0
LEAK_ELIXIR = 9.5
OWN_TOWER_X = {"left": 3.5, "right": 14.5}
ENEMY_TOWER_TILE = {"left": (3.5, 23.3), "right": (14.5, 23.3)}
BRIDGE_ROW = 14.0
MIN_X, MAX_X, MIN_Y, MAX_Y = 0.5, 17.5, 1.0, 14.0
OFFENCE_SCORES = {"counter_push": 0.6, "punish": 0.55, "leak_wc": 0.5, "leak": 0.4}
NEED_ORDER = ("air", "tank", "swarm", "building_targeter")
GROUND_ONLY_SPELLS = frozenset({"the-log", "barbarian-barrel", "earthquake", "giant-snowball"})

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


@dataclass
class TacticalMove:
    card: str
    key: str                     # card|lane|zone of `tile` (lane_of / zone_of)
    tile: list[float]            # concrete deploy tile, rounded to 0.5
    intent: str                  # "defend" | "counter_push" | "punish" | "leak"
    reason: str                  # <= 60 chars, e.g. "pulls giant to the centre, trade +5"
    score: float                 # engine preference, higher is better
    trade: float | None          # push elixir - card cost (defend), else None


@dataclass
class TacticalReport:
    pushes: list[Push]           # most urgent first
    moves: list[TacticalMove]    # best first, distinct keys, at most 6
    best: TacticalMove | None
    hold_reason: str | None      # set when best is None
    lines: list[str]             # 1..4 lines, each starting with "tactics: "


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
    def __init__(self, cards: dict | None = None, roles: dict | None = None, kg=None):
        self.cards = cards if cards is not None else load_cards()
        if roles is None:
            from clashbot.card_roles import load_roles        # lazy: the module works with `roles=` without Z1
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

    @staticmethod
    def _hits_air(card: str, role: dict) -> bool:
        """Same semantics as `card_roles.hits_air`, with the card slug at hand: spells hit air except the ground-only
        ones, everything else by `targets == "air_ground"`."""
        if role.get("type") == "spell":
            return card not in GROUND_ONLY_SPELLS
        return role.get("targets") == "air_ground"

    @staticmethod
    def _has(role: dict, name: str) -> bool:
        return name in (role.get("roles") or [])

    # -- pushes --------------------------------------------------------------------------------------------------------
    def _pushes(self, state: dict) -> list[Push]:
        """Enemy units close to an own tower (ETA <= 8 s) or on our half, grouped by lane, most urgent push first."""
        per_lane: dict[str, list[tuple[float, float, int, dict]]] = {"left": [], "right": []}
        for i, u in enumerate(state.get("units", [])):
            tile = _tile(u)
            if u.get("side") != "enemy" or tile is None:
                continue
            eta = _own_eta(u)
            on_our_half = tile[1] <= OWN_HALF_MAX_ROW
            if not ((eta is not None and eta <= URGENT_ETA_S) or on_our_half):
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
            "swarm": swarm_units >= 3 or any(r.get("hp_class") == "swarm" and (r.get("count") or 1) >= 3
                                             for r in roles),
            "building_targeter": any(r.get("targets") == "buildings" for r in roles),
        }
        counts: dict[str, int] = {}
        for u in units:
            counts[_name(u)] = counts.get(_name(u), 0) + 1
        parts = [n if k == 1 else f"{n} x{k}" for n, k in counts.items()]
        names = ", ".join(parts[:4]) + (f" +{len(parts) - 4}" if len(parts) > 4 else "")
        summary = f"{names} ({fmt_num(elixir)} elixir), tower in {eta_s:.1f}s"
        return Push(lane, units, elixir, eta_s, [n for n in NEED_ORDER if needs[n]], summary)

    def _ctx(self, push: Push) -> _PushCtx:
        unit = push.units[0]
        pos = _lead_tile(unit, LEAD_S) or list(_tile(unit))
        card = unit.get("card")
        norm: dict[str, float] = {}
        if self.kg is not None and card:
            scores = self.kg.counter_scores(card) or {}
            top = max(scores.values(), default=0.0)
            if top > 0:
                norm = {k: v / top for k, v in scores.items()}
        return _PushCtx(push, unit, pos, OWN_TOWER_X[push.lane], 1 if push.lane == "left" else -1, _name(unit),
                        all(self._role(u.get("card")).get("flying") for u in push.units), norm)

    # -- defence -------------------------------------------------------------------------------------------------------
    def _defends(self, card: str, role: dict) -> bool:
        """Spells that do damage can defend; rage / clone / freeze ... and the win-condition spells cannot."""
        if role.get("type") != "spell":
            return True
        return (card not in NON_DAMAGE_SPELLS and role.get("spell_class") != "utility"
                and not self._has(role, "win_condition"))

    def _spell_aim(self, card: str, ctx: _PushCtx, state: dict):
        return self.policy.spell_target(card, lane_of(ctx.pos), zone_of(ctx.pos), state)

    def _defence_score(self, card: str, role: dict, ctx: _PushCtx, aim) -> tuple[float, float] | None:
        """(score, trade) of `card` against the push, or None when it cannot answer an air-only push."""
        push, needs = ctx.push, ctx.push.needs
        cost = self._cost(card)
        is_spell, is_building = role.get("type") == "spell", role.get("type") == "building"
        score = BASE_SCORE
        if "air" in needs and not self._hits_air(card, role):
            if ctx.all_air:
                return None
            score -= 0.5
        if "tank" in needs:
            if self._has(role, "tank_killer") or role.get("dps_class") == "high":
                score += 0.6
            if is_building and "building_targeter" in needs:
                score += 0.5
            if self._has(role, "swarm"):
                score += 0.3
            if is_spell and not (aim is not None and aim[1] >= cost):
                score -= 0.6
        if "swarm" in needs:
            if role.get("splash") or (is_spell and "swarm" in (role.get("kills") or [])):
                score += 0.6
            elif role.get("type") == "troop" and role.get("range") == "melee":
                score -= 0.4
        if "building_targeter" in needs:
            if is_building:
                score += 0.5
            if self._has(role, "swarm") or self._has(role, "kamikaze"):
                score += 0.3
            if self._has(role, "mini_tank"):
                score += 0.3
        score += 0.2 * ctx.kg_norm.get(card, 0.0)
        trade = push.elixir - cost
        score -= 0.15 * max(0.0, -trade - 1.0)
        if trade >= 1.0:
            score += 0.1
        if self._has(role, "win_condition"):
            score -= 0.4                                                # kept for offence
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
        if role.get("range", "melee") == "melee":
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
            if not self._defends(card, role):
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
            moves.append(TacticalMove(card, key, tile, "defend", _with_trade(why, trade), score, trade))
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

    def _offence_moves(self, state: dict, hand: list[str]) -> list[TacticalMove]:
        own, opp = state["own"], state.get("opponent") or {}
        elixir = own.get("elixir") or 0.0
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
                        add(c, lane, "counter_push", OFFENCE_SCORES["counter_push"],
                            f"backs {_name(allies[lane])} in the {lane} lane")

        # punish: the opponent has spent down, hit the lane away from its last play
        opp_elixir = opp.get("elixir_estimate")
        if opp_elixir is not None and opp_elixir <= PUNISH_OPP_ELIXIR:
            lane = self._opp_last_lane(state) or self._weak_lane(state) or "left"
            for c in wcs:
                if elixir >= self._cost(c) + 1:
                    add(c, lane, "punish", OFFENCE_SCORES["punish"],
                        f"opponent ~{fmt_num(opp_elixir)} elixir, hits the {lane} lane")

        # leak: a full bar is wasted elixir
        if elixir >= LEAK_ELIXIR:
            lane = self._weak_lane(state) or "left"
            if wcs:
                for c in wcs:
                    add(c, lane, "leak", OFFENCE_SCORES["leak_wc"], f"elixir {fmt_num(elixir)} near full, pushes {lane}")
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
                                              OFFENCE_SCORES["leak"], None))
        return moves

    # -- report --------------------------------------------------------------------------------------------------------
    def _rank(self, moves: list[TacticalMove]) -> list[TacticalMove]:
        """Best first (ties: lower cost, then card name), the first move per key, at most MAX_MOVES."""
        out, seen = [], set()
        for m in sorted(moves, key=lambda m: (-m.score, self._cost(m.card), m.card)):
            if m.key not in seen:
                seen.add(m.key)
                out.append(m)
        return out[:MAX_MOVES]

    def _hold_reason(self, state: dict, pushes: list[Push], hand: list[str]) -> str:
        if pushes:
            return "no card in hand answers the push" if hand else "not enough elixir"
        own, opp = state["own"], state.get("opponent") or {}
        return (f"no threat, saving elixir (own {fmt_num(own.get('elixir'))}, "
                f"opp ~{fmt_num(opp.get('elixir_estimate'))})")

    def _lines(self, pushes: list[Push], moves: list[TacticalMove], hold_reason: str | None) -> list[str]:
        lines = [f"tactics: push {p.lane}: {p.summary}; needs {', '.join(p.needs) or 'nothing special'}"
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
        """`t` is accepted for the live-loop interface; the analysis depends on the state only."""
        hand = legal_cards(state, self.cards)
        pushes = self._pushes(state)
        moves: list[TacticalMove] = []
        for push in pushes:
            moves += self._defence_moves(self._ctx(push), hand, state)
        if not any(p.eta_s <= URGENT_ETA_S for p in pushes):
            moves += self._offence_moves(state, hand)
        moves = self._rank(moves)
        best = moves[0] if moves else None
        hold = None if best else self._hold_reason(state, pushes, hand)
        return TacticalReport(pushes, moves, best, hold, self._lines(pushes, moves, hold))

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
        ctx = self._ctx(report.pushes[0])
        aim = self._spell_aim(card, ctx, state) if role.get("type") == "spell" else None
        scored = self._defence_score(card, role, ctx, aim)
        return None if scored is None else scored[0]
