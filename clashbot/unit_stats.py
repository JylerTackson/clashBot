"""Unit stats and the threat model (MAC-AH): what a unit really is (hitpoints, damage, speed ...) at a card level, what a
spell kills, and what a push does to a tower.

`resources/unit_stats.json` (built by `tools/data/build_unit_stats.py` from the knowledge base's exact per-level tables,
cr-api-data as fallback) holds every card as a list of units with per-level series; `stat_at(card, level)` reads the
table row of the level (beyond the table: x1.1 per level). `resources/levels.json` says at which level we assume the
opponent's cards (15), our own and the towers (15).

`simulate_lane(enemy_units, own_defenders, tower)` is a small deterministic 1-D-ish simulation of one lane, < 2 ms: the
enemy units walk to our princess tower (or to what pulls them), our tower and defenders shoot, and the result says how
much damage reaches the tower. `spell_value(card, units)` prices a damage spell by what it kills.

Approximations of the simulation (it is a magnitude estimate, not the game):
  * time steps of 0.25 s; hits land at once (no projectile flight; a fireball lands LEAD_S after the cast);
  * no collisions or pushback; units walk in straight lines (ground units from the enemy half first through the bridge
    of their lane); all units of a card start at the card's tile with a small spread;
  * a building-targeter (giant, balloon, hog ...) goes for the nearest building, ours or the tower; other units for the
    nearest defender they can hit inside SIGHT_TILES, else the tower; no other retargeting (a building pull is the
    nearest-building rule; the geometry of the pull tile is the engine's, MAC-AG);
  * the tower keeps one target (the nearest in range) until it dies or leaves the range; the king tower is not in it;
  * a unit's attack starts with its first-hit delay when it comes in reach and repeats at its hit speed; stuns (zap)
    delay it; shields, rage, healing, charge bonuses, evolutions and hero abilities are ignored;
  * death damage (balloon bomb after its fuse, golem blast) and death spawns (golemites, lava pups) and periodic spawns
    (night witch bats, witch skeletons; at most SPAWN_CAP per spawner) are simulated; chain and pierce hit one target.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / "resources"
STATS_PATH = RESOURCES / "unit_stats.json"
LEVELS_PATH = RESOURCES / "levels.json"
LEVEL_GROWTH = 1.1                     # per level beyond a table
DEFAULT_OPPONENT_LEVEL = 15
DEFAULT_OWN_LEVEL = 14
DEFAULT_TOWER_LEVEL = 15

# tower tiles of the perception (cr_perception/geometry.py): our princess towers
TOWER_TILES = {"left": (3.5, 4.3), "right": (14.5, 4.3)}
BRIDGE_X = {"left": 3.5, "right": 14.5}
RIVER_ROW = 16.0                       # a ground unit on the enemy half first walks to the bridge of its lane here

# simulation constants
DT = 0.25
SIGHT_TILES = 5.5                      # a unit notices a target this close (plus its range)
TOWER_RADIUS = 1.5                     # a tower is hit when the unit is within its range + this of the tower centre
TROOP_RADIUS = 0.5
BUILDING_RADIUS = 1.0
DEPLOY_LATENCY_S = 0.5                 # a proposed card lands this long after the decision (tap + travel)
SPAWN_CAP = 6                          # spawned units per spawner
MAX_SIM_UNITS = 36
SPELL_LEAD_S = {"fireball": 1.0, "rocket": 1.5, "arrows": 0.8, "lightning": 0.6, "poison": 0.5}
DEFAULT_SPELL_LEAD_S = 0.3
_UNKNOWN_UNIT = {"name": "unit", "count": 1, "hp": 1000, "damage": 150, "dps": 125.0, "hit_speed_s": 1.2,
                 "first_hit_s": 0.5, "range_tiles": 1.0, "speed_tiles_s": 1.0, "targets": "ground", "flying": False,
                 "splash_radius_tiles": 0.0, "death_damage": 0, "death_radius_tiles": 0.0, "death_delay_s": 0.0,
                 "death_spawn": None, "spawn": None, "deploy_s": 1.0, "lifetime_s": None}


# ------------------------------------------------------------------------------------------------------ tables
@lru_cache(maxsize=1)
def load_unit_stats(path: str | None = None) -> dict:
    """The parsed `unit_stats.json` ({} when the file is missing, so the engine falls back to its role rules)."""
    try:
        return json.loads(Path(path or STATS_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def load_levels(path: str | None = None) -> dict:
    try:
        return json.loads(Path(path or LEVELS_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def opponent_level(card: str | None = None) -> int:
    lv = load_levels()
    return int((lv.get("opponent") or {}).get(card) or lv.get("opponent_default") or DEFAULT_OPPONENT_LEVEL)


def own_level(card: str | None = None) -> int:
    lv = load_levels()
    return int((lv.get("own") or {}).get(card) or lv.get("own_default") or DEFAULT_OWN_LEVEL)


def tower_level() -> int:
    return int(load_levels().get("tower_level") or DEFAULT_TOWER_LEVEL)


def series_at(series: dict | None, level: int) -> int | None:
    """The value of a {"l0": first level, "v": [...]} series at `level`: the table row, or beyond the table x1.1 per
    level (below it /1.1), rounded to an integer."""
    if not series:
        return None
    v = series["v"]
    i = int(level) - int(series["l0"])
    if 0 <= i < len(v):
        return int(round(v[i]))
    if i >= len(v):
        return int(round(v[-1] * LEVEL_GROWTH ** (i - len(v) + 1)))
    return int(round(v[0] / LEVEL_GROWTH ** (-i)))


def has_stats(card: str | None) -> bool:
    return bool(card) and card in (load_unit_stats().get("cards") or {})


def _unit_at(u: dict, level: int, hp_override: dict | None = None, dmg_override: dict | None = None) -> dict:
    hp = series_at(hp_override or u.get("hp"), level)
    damage = series_at(dmg_override or u.get("damage"), level)
    hit = u.get("hit_speed_s") or 0.0
    return {
        "name": u["name"], "count": u.get("count", 1), "hp": hp or 0, "damage": damage or 0,
        "dps": round(damage / hit, 1) if damage and hit else 0.0, "hit_speed_s": hit,
        "first_hit_s": u.get("first_hit_s") or 0.0, "range_tiles": u.get("range_tiles") or 0.0,
        "speed_tiles_s": u.get("speed_tiles_s") or 0.0, "targets": u.get("targets") or "ground",
        "flying": bool(u.get("flying")), "kamikaze": bool(u.get("kamikaze")),
        "splash_radius_tiles": u.get("splash_radius_tiles") or 0.0,
        "death_damage": series_at(u.get("death_damage"), level) or 0,
        "death_radius_tiles": u.get("death_radius_tiles") or 0.0, "death_delay_s": u.get("death_delay_s") or 0.0,
        "death_spawn": u.get("death_spawn"), "spawn": u.get("spawn"), "deploy_s": u.get("deploy_s") or 1.0,
        "lifetime_s": u.get("lifetime_s"),
    }


@lru_cache(maxsize=4096)
def _stat_at(card: str, level: int, variant: str | None) -> dict | None:
    rec = (load_unit_stats().get("cards") or {}).get(card)
    if rec is None:
        return None
    var = (rec.get("variants") or {}).get(variant) if variant else None
    overrides = (var or {}).get("series") or {}
    units = []
    for i, u in enumerate(rec.get("units") or []):
        units.append(_unit_at(u, level, overrides.get("hp") if i == 0 else None,
                              overrides.get("damage") if i == 0 else None))
    out: dict = {"card": card, "type": rec["type"], "level": level, "rarity": rec.get("rarity"),
                 "elixir": rec.get("elixir"), "units": units, "lifetime_s": rec.get("lifetime_s"),
                 "source": rec.get("source") or {}}
    sp = rec.get("spell")
    if sp:
        out["spell"] = {"damage": series_at(sp.get("damage"), level) or 0,
                        "crown_tower_damage": series_at(sp.get("crown_tower_damage"), level) or 0,
                        "radius_tiles": sp.get("radius_tiles") or 0.0, "duration_s": sp.get("duration_s"),
                        "stun_s": sp.get("stun_s") or 0.0}
    if var:
        out["abilities"] = _resolve_abilities(var.get("abilities") or {}, level)
    return out


def _resolve_abilities(node, level: int):
    """The ability dict of a variant with every series replaced by its value at `level`."""
    if isinstance(node, dict):
        if "l0" in node and "v" in node:
            return series_at(node, level)
        return {k: _resolve_abilities(v, level) for k, v in node.items()}
    return node


def stat_at(card: str, level: int | None = None, variant: str | None = None) -> dict | None:
    """The stats of `card` at `level` (default: the opponent's level): {"type", "elixir", "rarity", "units": [{name,
    count, hp, damage, dps, hit_speed_s, range_tiles, speed_tiles_s, targets, flying, splash_radius_tiles,
    death_damage, death_spawn ...}], "spell": {damage, crown_tower_damage, radius_tiles}, "lifetime_s"}; None for an
    unknown card. `variant` "hero" / "evo" (musketeer hero, tesla / electro-dragon evolution) swaps the main unit's
    series and adds the resolved `abilities`. The dicts are shared (cached): do not modify them."""
    return _stat_at(card, int(level if level is not None else opponent_level(card)), variant)


def tower_stat(kind: str = "princess", level: int | None = None) -> dict:
    """{"hp", "damage", "hit_speed_s", "first_hit_s", "range_tiles"} of our tower at `level` (default levels.json)."""
    level = int(level if level is not None else tower_level())
    t = (load_unit_stats().get("tower") or {}).get(kind) or {}
    if not t:
        hp = int(4424 * LEVEL_GROWTH ** (level - 15))
        return {"hp": hp, "damage": hp // 28, "hit_speed_s": 0.8, "first_hit_s": 0.8, "range_tiles": 7.5}
    return {"hp": series_at(t["hp"], level), "damage": series_at(t["damage"], level), "hit_speed_s": t["hit_speed_s"],
            "first_hit_s": t.get("first_hit_s") or t["hit_speed_s"], "range_tiles": t["range_tiles"]}


def unit_stat(unit: dict, level: int | None = None) -> dict:
    """The stats of ONE detected unit (a state unit dict with `card` and maybe `unit_class`): the card's unit whose
    name matches the unit class ('bat' of the night witch), else its main unit; a card without stats gives a medium
    ground troop (so an unknown unit still counts). `count` is 1."""
    card = unit.get("card")
    st = stat_at(card, level) if card else None
    if st is None or not st["units"]:
        return dict(_UNKNOWN_UNIT, count=1)
    units = st["units"]
    cls = _norm(unit.get("unit_class"))
    pick = units[0]
    if cls and cls != _norm(card):
        for u in units:
            if _norm(u["name"]) == cls:
                pick = u
                break
    return dict(pick, count=1)


def _norm(text) -> str:
    s = "".join(ch for ch in str(text or "").lower() if ch.isalnum())
    return s[:-1] if s.endswith("s") and len(s) > 3 else s


# ------------------------------------------------------------------------------------------------------ spells
def spell_stat(card: str, level: int | None = None) -> dict | None:
    """{"damage", "crown_tower_damage", "radius_tiles", "stun_s"} of a damage spell at our level; None when the card
    has no damage numbers (rage, clone, freeze, ... or no stats)."""
    st = stat_at(card, level if level is not None else own_level(card))
    sp = (st or {}).get("spell")
    return sp if sp and sp.get("damage") else None


def spell_value(card: str, members: list[dict], cards: dict | None = None, level: int | None = None,
                opp_level: int | None = None) -> tuple[float, int] | None:
    """(elixir value, units killed) of casting the damage spell `card` on `members`, the enemy unit dicts inside its
    radius, by what it does. Per distinct card among them: the units it kills count the card's elixir times their share
    of that card's units seen in the radius (perception may miss some of a swarm, so the units seen stand for the card);
    a survivor counts 0.5 x elixir x damage / its hitpoints (at most 0.5 x elixir). The spell's damage is at our level,
    the units' hitpoints at the opponent's. None when the spell has no damage stats (the caller keeps its old rule)."""
    sp = spell_stat(card, level)
    if sp is None:
        return None
    by_card: dict[str, list[dict]] = {}
    for u in members:
        c = u.get("card")
        if c:
            by_card.setdefault(c, []).append(u)
    value, killed = 0.0, 0
    for c, units in by_card.items():
        elixir = float(((cards or {}).get(c) or {}).get("elixir_cost") or (stat_at(c) or {}).get("elixir") or 0.0)
        if elixir <= 0:
            continue
        n = len(units)
        share = 0.0
        for u in units:
            hp = unit_stat(u, opp_level)["hp"]
            if hp <= sp["damage"]:
                killed += 1
                share += 1.0
            else:
                share += min(0.5, 0.5 * sp["damage"] / hp)
        value += elixir * share / n
    return round(value, 3), killed



# ------------------------------------------------------------------------------------------------------ simulation
@dataclass
class SimResult:
    tower_damage: int            # damage dealt to the princess tower (capped at its hitpoints), death damage included
    tower_destroyed: bool
    destroyed_at_s: float | None
    units_alive_at_end: int      # enemy units still alive when the simulation ends
    time_to_clear: float | None  # when the last enemy unit died; None when some survive the horizon
    eta_s: float | None          # when the first enemy unit came in reach of the tower; None if none did
    tower_hp_left: int
    enemy_killed: int = 0
    defenders_lost: int = 0


class _U:
    """One simulated unit."""
    __slots__ = ("side", "x", "y", "hp", "speed", "rng", "dmg", "hit", "first", "can_air", "bonly", "fly", "bld",
                 "splash", "ddmg", "drad", "ddelay", "dspawn", "spawn", "next_hit", "engaged", "stun", "ready", "expire",
                 "alive", "radius", "tgt", "spawned", "spawn_t", "eta_done", "name", "card", "kam")


def _mk(stat: dict, side: int, x: float, y: float, ready: float, bld: bool, card: str = "") -> _U:
    u = _U()
    targets = stat["targets"]
    u.side, u.x, u.y, u.hp = side, x, y, float(max(stat["hp"], 1))
    u.speed = 0.0 if bld else stat["speed_tiles_s"]
    u.rng, u.dmg = stat["range_tiles"], float(stat["damage"])
    u.hit = stat["hit_speed_s"] or 1.0
    u.first = stat["first_hit_s"] or 0.0
    u.can_air = targets in ("air_ground", "air")
    u.bonly = targets == "buildings"
    u.fly, u.bld = stat["flying"], bld
    u.splash = stat["splash_radius_tiles"]
    u.ddmg, u.drad, u.ddelay = stat["death_damage"], stat["death_radius_tiles"], stat["death_delay_s"] or 0.0
    u.dspawn, u.spawn = stat["death_spawn"], stat["spawn"]
    u.next_hit, u.engaged, u.stun, u.ready = 0.0, False, 0.0, ready
    lt = stat.get("lifetime_s")
    u.expire = ready + lt if lt and bld else math.inf
    u.alive, u.radius, u.tgt, u.spawned = True, BUILDING_RADIUS if bld else TROOP_RADIUS, None, 0
    u.spawn_t = ready + (u.spawn.get("start_s") or 1.0) if u.spawn else math.inf
    u.eta_done = False
    u.name, u.card = stat["name"], card
    u.kam = bool(stat.get("kamikaze"))
    return u


def _spread(i: int, n: int) -> tuple[float, float]:
    """A small offset for the i-th of n units of one card (they do not stand on one point)."""
    if n <= 1:
        return 0.0, 0.0
    ang = 2.0 * math.pi * i / n
    return 0.5 * math.cos(ang), 0.5 * math.sin(ang)


def simulate_lane(enemy_units: list[dict], own_defenders: list[dict], tower: str | dict = "left",
                  horizon_s: float = 20.0, *, opp_level: int | None = None, tower_lv: int | None = None) -> SimResult:
    """One lane for `horizon_s` seconds. `enemy_units`: state unit dicts (`card`, `tile`, maybe `unit_class`), one
    simulated unit per dict, at the opponent's level. `own_defenders`: dicts with `card` and `tile` for our units and
    buildings (a proposed card: every unit of it is simulated, at our level or `level`, acting after `ready_s` seconds -
    default the card's deploy time + the tap latency, 0 when `placed` is true; `single` true = one detected unit of the
    card, as an ally seen on the field), or a damage spell (a spell card: cast at the tile, lands after its travel
    time). `tower` is "left" / "right" or {"lane", "hp"} (the hitpoints now; default full at the tower level)."""
    if isinstance(tower, dict):
        lane, hp_now = tower.get("lane", "left"), tower.get("hp")
    else:
        lane, hp_now = tower, None
    tx, ty = TOWER_TILES[lane]
    bridge_x = BRIDGE_X[lane]
    tstat = tower_stat("princess", tower_lv)
    tower_hp = float(hp_now if isinstance(hp_now, (int, float)) and not isinstance(hp_now, bool) and hp_now > 0
                     else tstat["hp"])
    tower_hp0 = tower_hp
    tower_dmg, tower_hit, tower_rng = float(tstat["damage"]), float(tstat["hit_speed_s"]), float(tstat["range_tiles"])

    E: list[_U] = []                           # enemy units
    O: list[_U] = []                           # our units and buildings
    for e in enemy_units:
        tile = e.get("tile")
        if isinstance(tile, (list, tuple)) and len(tile) >= 2:
            E.append(_mk(unit_stat(e, opp_level), 1, float(tile[0]), float(tile[1]), 0.0, False, e.get("card") or ""))
    n_enemy0 = len(E)
    spells: list[tuple[float, dict]] = []      # (cast time, spell stat + tile)
    for d in own_defenders:
        card, tile = d.get("card"), d.get("tile")
        if not card or not (isinstance(tile, (list, tuple)) and len(tile) >= 2):
            continue
        st = stat_at(card, d.get("level") if d.get("level") is not None else own_level(card))
        if st is None:
            continue
        if st["type"] == "spell":
            sp = st.get("spell")
            if sp and sp.get("damage"):
                ready = d.get("ready_s")
                spells.append((float(SPELL_LEAD_S.get(card, DEFAULT_SPELL_LEAD_S) if ready is None else ready),
                               dict(sp, tile=tile)))
            continue
        bld = st["type"] == "building"
        ready = d.get("ready_s")
        if ready is None:
            ready = 0.0 if d.get("placed") else DEPLOY_LATENCY_S + (st["units"][0]["deploy_s"] if st["units"] else 1.0)
        if d.get("single"):                       # one detected unit of the card (an ally seen on the field)
            O.append(_mk(unit_stat(d, d.get("level") if d.get("level") is not None else own_level(card)), 0,
                         float(tile[0]), float(tile[1]), float(ready), bld, card))
            continue
        for ui, su in enumerate(st["units"]):
            n = su["count"]
            for k in range(n):
                dx, dy = _spread(k, n)
                O.append(_mk(su, 0, float(tile[0]) + dx, float(tile[1]) + dy, float(ready), bld and ui == 0, card))
    spells.sort(key=lambda s: s[0])
    pending: list[tuple[float, float, float, float, float, int]] = []       # (time, x, y, radius, damage, victim side)

    tower_total = 0.0
    eta = last_death = destroyed_at = None
    killed = lost = 0
    alive_enemy = n_enemy0
    tower_tgt: _U | None = None
    tower_next = float(tstat["first_hit_s"])

    def spawn_unit(owner: _U, name: str, count: int, now: float) -> None:
        nonlocal alive_enemy
        st = stat_at(owner.card, opp_level if owner.side == 1 else own_level(owner.card))
        key = _norm(name)
        su = next((u for u in (st or {}).get("units", []) if _norm(u["name"]) == key), None)
        if su is None:
            return
        pool = E if owner.side == 1 else O
        for k in range(min(count, 6)):
            if len(E) + len(O) >= MAX_SIM_UNITS:
                break
            dx, dy = _spread(k, count)
            pool.append(_mk(su, owner.side, owner.x + dx, owner.y + dy, now + 0.5, False, owner.card))
            if owner.side == 1:
                alive_enemy += 1

    def kill(v: _U, now: float) -> None:
        nonlocal alive_enemy, killed, lost, last_death
        v.alive = False
        if v.side == 1:
            alive_enemy -= 1
            killed += 1
            last_death = now
        else:
            lost += 1
        if v.ddmg and v.drad > 0:
            pending.append((now + v.ddelay, v.x, v.y, v.drad, float(v.ddmg), 1 - v.side))
        if v.dspawn:
            spawn_unit(v, v.dspawn["unit"], int(v.dspawn["count"]), now)

    def splash(x: float, y: float, r: float, dmg: float, victims: list[_U], now: float, skip: _U | None) -> None:
        for v in victims:
            if v.alive and v is not skip and v.ready <= now and (v.x - x) ** 2 + (v.y - y) ** 2 <= (r + v.radius) ** 2:
                v.hp -= dmg
                if v.hp <= 0:
                    kill(v, now)

    def act(u: _U, foes: list[_U], now: float) -> None:
        """One step of unit `u` against the living `foes` (our units for an enemy, enemies for ours): pick the target,
        attack it when in reach, else walk toward it."""
        nonlocal tower_hp, tower_total, eta
        ux, uy = u.x, u.y
        best = None
        if u.side == 1:
            bd = 1e9
            for v in foes:
                if v.alive and v.ready <= now and (v.bld or not u.bonly) and (u.can_air or not v.fly):
                    d2 = (v.x - ux) ** 2 + (v.y - uy) ** 2
                    if d2 < bd:
                        best, bd = v, d2
            if best is not None and bd > ((tx - ux) ** 2 + (ty - uy) ** 2 if u.bonly else (SIGHT_TILES + u.rng) ** 2):
                best = None                                 # a building-targeter takes the tower when it is nearer
            if best is None:
                gx, gy, gr = tx, ty, TOWER_RADIUS
            else:
                gx, gy, gr = best.x, best.y, best.radius
        else:
            if u.bonly or u.dmg <= 0:
                return
            bd = (SIGHT_TILES + u.rng) ** 2
            for v in foes:
                if v.alive and (u.can_air or not v.fly):
                    d2 = (v.x - ux) ** 2 + (v.y - uy) ** 2
                    if d2 < bd:
                        best, bd = v, d2
            if best is None:
                return
            gx, gy, gr = best.x, best.y, best.radius
        d = math.hypot(gx - ux, gy - uy)
        reach = u.rng + gr
        if d <= reach + 1e-6:                                   # (the walk stops exactly at reach: no float stall)
            if best is None and not u.eta_done:
                u.eta_done = True
                if eta is None:
                    eta = now
            if u.dmg <= 0:
                return
            if not u.engaged or u.tgt is not best:
                u.engaged, u.tgt = True, best
                u.next_hit = max(u.next_hit, now + u.first)
            hits = 0
            while now >= u.next_hit and hits < 4:
                hits += 1
                u.next_hit += u.hit
                if best is None:
                    applied = min(u.dmg, tower_hp)
                    tower_hp -= applied
                    tower_total += applied
                    if u.splash > 0:
                        splash(gx, gy, u.splash, u.dmg, O, now, None)
                    if u.kam:
                        kill(u, now)                            # a spirit explodes once and is gone
                        return
                    if tower_hp <= 0:
                        return
                else:
                    best.hp -= u.dmg
                    if u.splash > 0:
                        splash(best.x, best.y, u.splash, u.dmg, E if u.side == 0 else O, now, best)
                    if best.hp <= 0 and best.alive:
                        kill(best, now)
                    if u.kam:
                        kill(u, now)
                        return
                    if not best.alive:
                        u.engaged = False
                        return
            return
        u.engaged = False
        if u.speed <= 0:
            return
        wx, wy, stop = gx, gy, max(0.0, d - reach)
        if u.side == 1 and not u.fly and uy > RIVER_ROW and gy < RIVER_ROW and abs(ux - bridge_x) > 0.6:
            wx, wy = bridge_x, RIVER_ROW                      # a ground unit on the enemy half crosses at the bridge
            stop = math.hypot(wx - ux, wy - uy)
        dd = math.hypot(wx - ux, wy - uy)
        step = min(u.speed * DT, stop)
        if dd > 1e-9 and step > 0:
            u.x = ux + (wx - ux) / dd * step
            u.y = uy + (wy - uy) / dd * step

    spawners = [u for u in E + O if u.spawn]
    t = 0.0
    horizon = float(horizon_s)
    while t < horizon and tower_hp > 0:
        while spells and spells[0][0] <= t:
            _, sp = spells.pop(0)
            sx, sy = sp["tile"][0], sp["tile"][1]
            r = sp["radius_tiles"]
            for v in E:
                if v.alive and (v.x - sx) ** 2 + (v.y - sy) ** 2 <= (r + v.radius) ** 2:
                    v.hp -= sp["damage"]
                    if sp.get("stun_s"):
                        v.stun, v.engaged = t + sp["stun_s"], False
                    if v.hp <= 0:
                        kill(v, t)
        if pending:
            due = [p for p in pending if p[0] <= t]
            if due:
                pending[:] = [p for p in pending if p[0] > t]
                for _, x, y, r, dmg, victim in due:
                    if victim == 0:
                        if (x - tx) ** 2 + (y - ty) ** 2 <= (r + TOWER_RADIUS) ** 2:        # lands on the tower
                            applied = min(dmg, tower_hp)
                            tower_hp -= applied
                            tower_total += applied
                        splash(x, y, r, dmg, O, t, None)
                    else:
                        splash(x, y, r, dmg, E, t, None)
        shots = 0
        while tower_next <= t and shots < 3 and alive_enemy > 0:           # the tower: one target at a time
            shots += 1
            if tower_tgt is not None and not (tower_tgt.alive and (tower_tgt.x - tx) ** 2 + (tower_tgt.y - ty) ** 2
                                              <= (tower_rng + tower_tgt.radius) ** 2):
                tower_tgt = None
            if tower_tgt is None:
                bd = 1e9
                for v in E:
                    if v.alive:
                        d2 = (v.x - tx) ** 2 + (v.y - ty) ** 2
                        if d2 <= (tower_rng + v.radius) ** 2 and d2 < bd:
                            tower_tgt, bd = v, d2
            if tower_tgt is None:
                tower_next = t                                             # idle: fires as soon as something is in range
                break
            tower_tgt.hp -= tower_dmg
            tower_next += tower_hit
            if tower_tgt.hp <= 0:
                kill(tower_tgt, t)
                tower_tgt = None
        for u in E:
            if u.alive and t >= u.stun and (u.dmg > 0 or u.speed > 0):
                act(u, O, t)
        for u in O:
            if not u.alive or u.ready > t:
                continue
            if t >= u.expire:
                u.alive = False
                continue
            if t >= u.stun:
                act(u, E, t)
        for u in spawners:
            if u.alive and u.ready <= t and t >= u.spawn_t and u.spawned < SPAWN_CAP:
                spawn_unit(u, u.spawn["unit"], int(u.spawn["count"]), t)
                u.spawned += int(u.spawn["count"])
                u.spawn_t += u.spawn["every_s"]
        if tower_hp <= 0:
            destroyed_at = t
            break
        if alive_enemy <= 0 and not pending:
            break
        t += DT

    alive = max(alive_enemy, 0)
    clear = (last_death if last_death is not None else 0.0) if alive == 0 else None
    return SimResult(int(round(min(tower_total, tower_hp0))), tower_hp <= 0, destroyed_at, alive, clear, eta,
                     int(max(tower_hp, 0)), killed, lost)
