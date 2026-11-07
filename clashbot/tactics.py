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
Placement and windows (MAC-AG): a flyer is never answered with a card that cannot hit air (unless ground units stand
at its side); a building is a pull only at a geometric pull tile (closer to the threat than any of our towers, in range
of both princess towers) and only before the threat locks on its tower, else it is a low-stopping block; a ground
troop-targeter is kited to the centre by the first melee defender. Offence is proposed during pressure too: a
counter-push when the lane's push is answered or gone, a `split` into the lane away from an answered push worth 5+
elixir, and a win-condition leak at a full bar while no push is within 3 s. `TacticalReport.offence` lists them.
Threat model (MAC-AH, `clashbot.unit_stats`): every push is simulated against our princess tower (`damage_if_unanswered`,
`destroys_tower`, a simulated ETA when the tracker gave none) and the pushes are ranked by what they would do, not by
ETA; river / bridge threats (a unit at rows 15-18 walking toward us, a building-targeter or flyer at rows <= 20) are
pushes already; a push worth under `tower_ok_damage` gets no defend moves ("tower handles it"); a defend move is simulated
too (`prevented`, `efficiency`, `adequate`: the damage left is <= max(tower_ok_damage, adequate_frac x the push's)) and the
score gets `prevent` x the share it prevents, the role terms staying as a tie-break at `role_scale`. Our units already in
the lane are simulated as well (a push they hold counts as answered). Weight `threat_model` = 0 switches all of it off.
Knowledge graph (MAC-AK K4b, `clashbot.kg2.KG2` as `kg=` / `kg2=`): for the costliest card of each push the graph says
how our hand cards answered it in the pro games (`answers_for`: pro held-rate, role transfer, wiki, sim); the defend
score gets `kg_prior` x that answer's score (the simulation's adequacy stays the gate), the reason quotes the strongest
evidence ("pros 70% held n=23", "role of cannon"), and a card whose pro answer has an offset from >= 5 events is placed
at the threat's lead tile + that offset (mirrored back to the lane, clamped; a building only where it still pulls;
`tile_source` "pro_offset", else "geometric" = the MAC-AG rule) unless the simulation likes the geometric tile better.
Hero ability (MAC-AI part 2): with the button ready and her Musketeer on the field, the Trusty Turret (KB stats at the
musketeer's level) is simulated at the spawn tile (her tile + 3 rows toward the enemy) against each push; when it prevents
`ability_min_prevent` (1,000, D-049) tower HP the engine proposes an `ability` move (`TacticalReport.abilities`, key
`ability:musketeer|lane|zone`). Without a graph, nothing of this changes.
Timing and spells (MAC-AL P): a push's ETA is the simulation's whenever there is one (`min(sim, tracker)`, `eta_source`:
the tracker's is late); a building targeter or flyer anywhere on the field that walks toward us, or that was played in the
last 3 s, is an early push. With the threat model on, every damage spell in hand is aimed at the best cluster anywhere
(`ReactivePolicy.spell_best`: any half and lane, tanks are damaged survivors, +1 elixir for their princess tower): a
cluster with a unit of a push is that push's defend move (simulated on the push minus the units it kills), any other
worthwhile cluster is a `spell_value` move (`TacticalReport.spells`).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from clashbot import unit_stats as ustats
from clashbot.actions import ABILITY_COST, ABILITY_KEY, ABILITY_TURRET_TILES, ability_key
from clashbot.card_roles import GROUND_ONLY_SPELLS, hits_air, load_roles
from clashbot.config import ANYWHERE_CARDS
from clashbot.game_rules import action_key, lane_of, legal_cards, load_cards, zone_of
from clashbot.policy import (NON_DAMAGE_SPELLS, OPPOSITE_LANE, OWN_TOWER_TILES, PULL_LEAD_S, SPELL_RADIUS, TANK_CARDS,
                             ReactivePolicy, _lead_tile, air_lead_blocks_ground, locked_on, nearest_own_tower_dist,
                             targeter_of)
from clashbot.state_text import fmt_num

URGENT_ETA_S = 8.0           # a push this close (seconds) to an own tower is acted on; later ones leave room for offence
PUSH_ETA_S = 10.0            # an enemy unit this close to an own tower is listed in a push (a building is placed early)
OWN_HALF_MAX_ROW = 14        # an enemy unit at or behind this row is on our half (rows 15-16 are the river)
NEAR_BACK_ROW = 8            # no ETA and this deep in our half = as urgent as at the tower (ETA 0), else ETA 2 s
MAX_PUSH_LANES = 2
MAX_MOVES = 6
MAX_REASON = 60
MAX_REASON_SAVES = 72           # ... a reason that quotes what the move saves (MAC-AH) may be this long
TRADE_SUFFIX_ROOM = 12          # characters kept free for ", trade +N" in a reason
SAVES_SUFFIX_ROOM = 22          # ... and for ", saves ~3,100 tower HP" (MAC-AH)
MAX_REASON_KG = 96              # ... a reason that also quotes the graph's evidence ("pros 70% held n=23") may run to this
PRO_OFFSET_MIN_N = 5            # a pro offset is used for placement when it comes from this many events (K4b)
PRO_OFFSET_MAX_IQR = 8.0        # ... and when its spread (IQR of dx / dy, tiles) is no wider than this
KG_EVIDENCE_MIN_SCORE = 0.05    # an answer scoring less is not quoted
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
RIVER_MIN_ROW, RIVER_MAX_ROW = 15, 18    # enemy units here and walking toward our side are a push already (MAC-AH)
EARLY_FLYER_ROW = 20         # a building-targeter or flyer this close to the river (rows <= 20) is a push, whatever its ETA
RECENT_PLAY_S = 3.0          # ... and one anywhere on the field is a push when it walks toward us or they played it this recently
SIM_HORIZON_S = 20.0         # the lane simulation's default horizon (clashbot.unit_stats.simulate_lane)
FAR_ATTACK_S = 10.0          # a push with a unit beyond our half is simulated until that unit has attacked this long
SIM_MAX_HORIZON_S = 45.0     # ... but never longer than this
SPELL_REASON_NAMES = {"electro-": "e-"}     # shortened in a spell move's reason ("e-wizard")
PREVENT_REF = 1500.0         # the damage a move prevents is measured against at least this much (small pushes stay small)
ALLY_DEPLOY_S = 1.0          # a play of ours this recent is still deploying: its units act after what is left of this
SPLIT_MIN_ELIXIR = 5.0       # an answered push worth this much elixir opens the other lane (split)
LEAK_PUSH_MIN_ETA_S = 3.0    # a win-condition leak is proposed during pressure while no push is closer than this
PULL_CLOSER_TILES = 1.5      # a pull tile is this much closer to the threat than the nearest tower of ours
PULL_MIN_DIST = 3.0          # ... and at least this far from it (the building is not hit at once)
PULL_X = (6.0, 12.0)         # the grid of pull tiles (0.5 steps); the centre band is x 7..11
PULL_Y = (4.0, 11.0)
PRINCESS_RANGE = 7.5         # tiles a princess tower reaches: the threat is hit on the part of its walk inside it
KING_RANGE = 7.0             # ... and the king tower (pulling to the middle brings the king into the fight)
PULL_BAND = (7.0, 11.0)      # the centre band a pull tile prefers
PULL_STEP = 0.25             # sampling step (tiles) of the threat's walk
KITE_X = 9.0                 # a kiting defender stands this far (KITE_DX) toward its lane from the middle of the arena
KITE_DX = 1.5
KITE_BEHIND = 3.0            # ... this many rows closer to our side than the lead unit
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
    "kg_factor": 0.2,                       # x the normalised v1 KG counter score of the most urgent unit (no KG2 attached)
    "kg_prior": 0.25,                       # x the KG2 answer score of the card against the push's costliest card (K4b)
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
    # threat model (MAC-AH): what a push does and what a move prevents is judged by the lane simulation
    # (clashbot.unit_stats.simulate_lane)
    "threat_model": 1.0,                    # 0 switches the whole threat model off: the engine is then exactly MAC-AG
                                            # (no simulation, no river pushes, ETA / order / reasons / answered as before)
    "prevent": 0.8,                         # x (tower damage a move prevents / max(the push's damage, PREVENT_REF))
    "role_scale": 0.5,                      # the role terms above (tank / swarm / targeter / stop / kg) count this much
                                            # once the simulation priced the move (they stay as a tie-break)
    "tower_ok_damage": 300.0,               # a push that would do less than this to the tower: the tower handles it
    "adequate_frac": 0.35,                  # a move is adequate when the tower damage left is <= max(tower_ok_damage,
                                            # this x the push's damage) or no unit reaches the tower
    # hero ability (MAC-AI part 2): the Trusty Turret, simulated at the Musketeer's spawn tile
    "ability": 0.5,                         # base score of the `ability` move (+ `prevent` x the share it prevents)
    "ability_min_prevent": 1000.0,          # proposed when it prevents at least this much tower HP (D-049: used sparingly)
    # a damage spell with a worthwhile cluster and no push to answer (MAC-AL): base score of the `spell_value` move, plus
    # `spell_value_ratio` x (value / cost - 1), at most `spell_value_cap` above the base
    "spell_value": 0.55,
    "spell_value_ratio": 0.1,
    "spell_value_cap": 0.3,
    # offence intent scores
    "counter_push": 0.6,
    "split": 0.7,                           # an answered push worth 5+ elixir: the win condition goes to the other lane
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
    # threat model (MAC-AH): the lane simulation of the push against our tower alone. None = no unit has stats
    damage_if_unanswered: float | None = None    # tower damage if nothing of ours fights it (tower included)
    destroys_tower: bool = False                 # ... and the tower would fall
    sim_eta_s: float | None = None               # when its first unit would reach the tower (simulated)
    tower_ok: bool = False       # damage_if_unanswered < tower_ok_damage: the tower handles it, no defend moves
    damage_with_allies: float | None = None      # tower damage with our units on the field now (None: no allies)
    destroys_with_allies: bool | None = None     # ... and whether the tower would fall then
    allies_hold: bool = False    # our units on the field already keep the damage small (see `adequate_frac`)
    early: bool = False          # listed for a river / bridge unit (MAC-AH), not for ETA <= PUSH_ETA_S or our half
    eta_source: str = "tracker"  # "sim" when `eta_s` is the simulation's (MAC-AL: the tracker's is late), else "tracker"


@dataclass
class TacticalMove:
    card: str
    key: str                     # card|lane|zone of `tile` (lane_of / zone_of)
    tile: list[float]            # concrete deploy tile, rounded to 0.5
    intent: str                  # "defend" | "counter_push" | "split" | "punish" | "leak" | "bait" | "pressure" | "support" | "cycle"
    reason: str                  # <= 60 chars, e.g. "pulls giant to the centre, trade +5"
    score: float                 # engine preference, higher is better
    trade: float | None          # push elixir - card cost (defend), else None
    stop: str = "n/a"            # defend: the card's stopping class against that push: high | medium | low | spell; else n/a
    value: float | None = None   # defend with a spell: the elixir value of the units it hits (worth it when >= its cost)
    # simulated against the push with this move added (None = no stats for the card or the push)
    adequate: bool | None = None         # the tower damage left is small (see `adequate_frac`) or nothing reaches the tower
    prevented: float | None = None       # tower damage the move removes (damage_if_unanswered - damage with it)
    efficiency: float | None = None      # prevented / elixir cost
    damage_with: float | None = None     # tower damage with the move (and our units already there)
    # K4b: where the tile came from - "pro_offset" (the threat's lead tile + a pro offset of the knowledge graph, from
    # >= PRO_OFFSET_MIN_N events), "geometric" (the MAC-AG rule) or "ability" (the turret's spawn tile)
    tile_source: str = "geometric"
    push_lane: str | None = None         # ability: the lane of the push the turret is judged against
    # damage spell aimed at a cluster (MAC-AL): the enemy unit dicts inside its circle and how many of them it kills
    members: tuple = ()
    kills: int = 0


class _SpellAim(NamedTuple):
    """A damage spell's best cluster: (tile, elixir value, counted units) like `ReactivePolicy.spell_target`, plus every unit
    in the circle, those it kills, the kill count and whether the circle hits an enemy princess tower."""
    tile: list
    value: float
    count: int
    members: tuple = ()
    killed: tuple = ()
    kills: int = 0
    tower: bool = False


@dataclass
class TacticalReport:
    pushes: list[Push]           # most urgent first
    moves: list[TacticalMove]    # best first, distinct keys, at most 6
    best: TacticalMove | None
    hold_reason: str | None      # set when best is None
    lines: list[str]             # 1..4 lines, each starting with "tactics: "
    defence: list[TacticalMove] = field(default_factory=list)   # every defend move, best first (`moves` keeps 6 of all)
    offence: list[TacticalMove] = field(default_factory=list)   # every offence move, best first (`moves` keeps 6 of all)
    abilities: list[TacticalMove] = field(default_factory=list)   # the hero ability, when it is worth it (MAC-AI part 2)
    spells: list[TacticalMove] = field(default_factory=list)      # `spell_value` moves: a worthwhile cluster, no push (MAC-AL)


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
    air_blocked: bool = False    # a flyer leads the push: a card that cannot hit air cannot answer it
    pull_unit: dict | None = None        # the push's building targeter (giant, balloon ...), if any
    pull_tile: list | None = None        # where a building would pull it to; None = no pull (locked on, or no such tile)
    kite: bool = False           # a ground troop-targeter and nobody fights it yet: the first melee defender kites it
    tower: dict | None = None    # the tower the lane simulation attacks: {"lane", "hp"}
    allies: list = field(default_factory=list)   # our units in the lane (simulation defenders, already on the field)
    kg_threat: str | None = None         # K4b: the costliest card of the push, the one the graph is asked about
    kg_pos: list | None = None           # ... and where its unit will be (tile led by LEAD_S): pro offsets start there
    kg_answers: dict = field(default_factory=dict)   # card in hand -> kg2.Answer against `kg_threat`


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


def _with_trade(base: str, trade: float, saves: float | None = None) -> str:
    """`base` + ", saves ~N tower HP" (when the simulation says a move prevents N >= 100) + ", trade +N". A reason is
    cut to MAX_REASON; with the saves clause it may run to MAX_REASON_SAVES, and the clause gives way first (shortened
    to ", saves ~N HP", then dropped) before the base is cut."""
    suffix = f", trade {'+' if trade >= 0 else '-'}{fmt_num(abs(trade))}"
    limit = MAX_REASON
    if saves is not None and saves >= 100:
        n = f"{int(round(saves, -2)):,}"
        for clause in (f", saves ~{n} tower HP", f", saves ~{n} HP"):
            if len(base) + len(clause) + len(suffix) <= MAX_REASON_SAVES:
                suffix, limit = clause + suffix, MAX_REASON_SAVES
                break
    return base[:max(0, limit - len(suffix))] + suffix


def _with_evidence(base: str, trade: float, saves: float | None, evidence: tuple[str, ...]) -> str:
    """`_with_trade` that also quotes the graph's evidence ("pros 70% held n=23"): `evidence` lists the clause variants,
    richest first; the first variant that fits MAX_REASON_KG together with ", saves ~N tower HP" (or its short form)
    and ", trade +N" is used, else the reason is the plain `_with_trade` one."""
    suffix = f", trade {'+' if trade >= 0 else '-'}{fmt_num(abs(trade))}"
    saves_clauses = [""]
    if saves is not None and saves >= 100:
        n = f"{int(round(saves, -2)):,}"
        saves_clauses = [f", saves ~{n} tower HP", f", saves ~{n} HP"]
    for ev in evidence:
        for sv in saves_clauses:
            text = f"{base}{sv}, {ev}{suffix}"
            if len(text) <= MAX_REASON_KG:
                return text
    return _with_trade(base, trade, saves)


def _clip(text: str) -> str:
    return text[:MAX_REASON]


class TacticalEngine:
    def __init__(self, cards: dict | None = None, roles: dict | None = None, kg=None, weights: dict | None = None,
                 kg2=None):
        self.cards = cards if cards is not None else load_cards()
        # the scoring numbers: defaults < tactics_weights.json < the explicit dict (read as `self.w[key]` at use time)
        self.w = resolve_weights(weights)
        if roles is None:
            roles = load_roles()
        # accept both the file layout ({"version", "cards": {...}}) and the plain card -> role mapping
        self.roles = roles["cards"] if isinstance(roles.get("cards"), dict) else roles
        # `kg` is the v1 graph (`counter_scores`) or a KG2 (`answers_for`, MAC-AK K4b); `kg2` names the KG2 explicitly
        # when both are given. With a KG2 its answers replace the v1 counter term (they include the wiki counters).
        if kg2 is None and callable(getattr(kg, "answers_for", None)):
            kg2 = kg
            kg = kg if callable(getattr(kg, "counter_scores", None)) else None
        self.kg = kg
        self.kg2 = kg2
        self.policy = ReactivePolicy(self.cards)
        self._fallback: dict[str, dict] = {}
        self._kg_cache: dict[tuple, dict] = {}

    # -- cards and roles -----------------------------------------------------------------------------------------------
    def _cost(self, card: str) -> float:
        if card == ABILITY_KEY:
            return float(ABILITY_COST)
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
    @staticmethod
    def _toward_us(unit: dict, tile) -> bool:
        """An enemy unit at the river is walking toward our side: its heading says advancing, else its velocity / the
        tile it will be in 2 s says its row falls; a retreating or stationary one is not; with no information it is."""
        heading = str(unit.get("heading") or "").lower()
        if "retreat" in heading or "stationary" in heading:
            return False
        if "advanc" in heading:
            return True
        vel = unit.get("vel") or unit.get("velocity")
        if isinstance(vel, (list, tuple)) and len(vel) >= 2 and isinstance(vel[1], (int, float)):
            return vel[1] < 0
        pred = unit.get("predicted_tile_2s")
        if isinstance(pred, (list, tuple)) and len(pred) >= 2 and isinstance(pred[1], (int, float)):
            return pred[1] < tile[1]
        return True

    def _recent_opp_cards(self, state: dict, t: float | None) -> frozenset:
        """The cards the opponent played in the last RECENT_PLAY_S seconds (the clock `t`, else the newest play stands for
        now)."""
        now = self._clock(state, t)
        if now is None:
            return frozenset()
        return frozenset(p["card"] for p in (state.get("opponent") or {}).get("recent_plays") or []
                         if p.get("card") and p.get("t") is not None and 0 <= now - p["t"] <= RECENT_PLAY_S)

    def _is_early(self, unit: dict, tile, card, recent: frozenset = frozenset()) -> bool:
        """A river / bridge threat (MAC-AH): an enemy unit at rows RIVER_MIN_ROW..RIVER_MAX_ROW walking toward our
        side, or a building-targeter / flyer at rows <= EARLY_FLYER_ROW, whatever its ETA - and (MAC-AL) such a unit
        anywhere on the field that walks toward us or whose card they played in the last RECENT_PLAY_S seconds (`recent`),
        so a balloon is budgeted from the moment it is played."""
        row = tile[1]
        played = card in recent
        if "retreat" in str(unit.get("heading") or "").lower() and not played:
            return False
        if RIVER_MIN_ROW <= row <= RIVER_MAX_ROW and self._toward_us(unit, tile):
            return True
        role = self._role(card)
        if role.get("targets") == "buildings" or bool(role.get("flying")):
            return row <= EARLY_FLYER_ROW or played or self._toward_us(unit, tile)
        return False

    def _pushes(self, state: dict, t: float | None = None) -> list[Push]:
        """Enemy units close to an own tower (ETA <= PUSH_ETA_S), on our half, or at the river walking toward us
        (MAC-AH), grouped by lane. Each push is simulated against our tower alone (`damage_if_unanswered`) and, when
        our units already stand in its lane, against them too (`damage_with_allies`). The pushes are ranked by the
        tower destroyed, then the damage left (with our units there), then ETA: the biggest open threat is first."""
        per_lane: dict[str, list[tuple[float, float, int, dict, float | None, bool]]] = {"left": [], "right": []}
        recent = self._recent_opp_cards(state, t) if self._sim_on() else frozenset()
        for i, u in enumerate(state.get("units", [])):
            tile = _tile(u)
            if u.get("side") != "enemy" or tile is None:
                continue
            card = u.get("card")
            if card not in SWARM_SPELL_UNITS and self._role(card).get("type") == "spell":
                continue                                    # a spell object is an effect, not a troop to answer
            eta = _own_eta(u)
            on_our_half = tile[1] <= OWN_HALF_MAX_ROW
            early = False
            if not ((eta is not None and eta <= PUSH_ETA_S) or on_our_half):
                early = self._sim_on() and self._is_early(u, tile, card, recent)
                if not early:
                    continue
            order_eta = eta if eta is not None else (0.0 if tile[1] <= NEAR_BACK_ROW else 2.0)
            per_lane[_side_lane(tile)].append((order_eta, tile[1], i, u, eta, early))
        pushes = []
        for lane, items in per_lane.items():
            if not items:
                continue
            items.sort(key=lambda x: x[:3])                           # most urgent first: ETA, nearest our side, order
            push = self._make_push(lane, [x[3] for x in items], items[0][0])
            self._assess(push, state)
            allies = self._lane_allies(lane, state, t) if push.damage_if_unanswered is not None else []
            if allies and not push.tower_ok:
                res = ustats.simulate_lane(push.units, allies, self._tower_of(lane, state))
                push.damage_with_allies, push.destroys_with_allies = float(res.tower_damage), res.tower_destroyed
                push.allies_hold = res.tower_damage <= self._ok_limit(push) or res.eta_s is None
            # the ETA: the simulation's whenever there is one - real speeds and ranges; the tracker's is late (MAC-AL) -
            # but never later than the tracker's; a unit without a tracker ETA is at the tower (back of our half) or
            # reaches it when the simulation says (else 2 s)
            etas = [x[4] for x in items if x[4] is not None]
            etas += [0.0 if x[1] <= NEAR_BACK_ROW else (push.sim_eta_s if push.sim_eta_s is not None else 2.0)
                     for x in items if x[4] is None]
            push.eta_s = min(etas)
            if push.sim_eta_s is not None and push.sim_eta_s <= push.eta_s:
                push.eta_s, push.eta_source = push.sim_eta_s, "sim"
            push.early = all(x[5] for x in items)
            push.summary = self._summary(push)
            pushes.append(push)
        def threat(p: Push) -> tuple[bool, float]:                  # (the tower falls, damage) left with our units there
            if p.damage_with_allies is not None:
                return bool(p.destroys_with_allies), p.damage_with_allies
            return p.destroys_tower, p.damage_if_unanswered or 0.0

        pushes.sort(key=lambda p: (not threat(p)[0], -threat(p)[1], p.eta_s, -p.elixir, p.lane))
        return pushes[:MAX_PUSH_LANES]

    def _sim_on(self) -> bool:
        """The threat model is on (weight `threat_model` > 0; read at use time so a swapped weights dict applies). The
        policy that aims our spells follows it: it prices them by what they kill only while the model is on."""
        on = self.w["threat_model"] > 0
        self.policy.price_spells = on
        return on

    def _tower_of(self, lane: str, state: dict) -> dict:
        """The tower of the lane for the simulation: its lane and hitpoints now (full when unknown)."""
        hp = ((state.get("own") or {}).get("towers") or {}).get(lane)
        return {"lane": lane, "hp": hp if isinstance(hp, (int, float)) and not isinstance(hp, bool) and hp > 0 else None}

    def _assess(self, push: Push, state: dict) -> None:
        """Simulate the push against our tower alone and set its damage fields; nothing when the threat model is off or
        none of its units has stats."""
        if not self._sim_on() or not any(ustats.has_stats(u.get("card")) for u in push.units):
            return
        tower = self._tower_of(push.lane, state)
        res = ustats.simulate_lane(push.units, [], tower)
        if res.eta_s is None or res.eta_s + FAR_ATTACK_S > SIM_HORIZON_S:
            # a unit far from the tower (a balloon just played) would be cut off by the horizon before it has attacked
            # for long: run until it has attacked FAR_ATTACK_S, so its damage and ETA are the real ones (MAC-AL)
            if any((_tile(u) or (0, 0))[1] > OWN_HALF_MAX_ROW for u in push.units):
                arrive = res.eta_s
                if arrive is None:
                    arrive = ustats.simulate_lane(push.units, [], tower, horizon_s=SIM_MAX_HORIZON_S).eta_s
                if arrive is not None:
                    res = ustats.simulate_lane(push.units, [], tower,
                                               horizon_s=min(SIM_MAX_HORIZON_S, arrive + FAR_ATTACK_S))
        push.damage_if_unanswered = float(res.tower_damage)
        push.destroys_tower = res.tower_destroyed
        push.sim_eta_s = res.eta_s
        push.tower_ok = res.tower_damage < self.w["tower_ok_damage"] and not res.tower_destroyed

    def _ok_limit(self, push: Push) -> float:
        """The tower damage a defence may leave and still be adequate (see the `adequate_frac` weight)."""
        return max(self.w["tower_ok_damage"], self.w["adequate_frac"] * (push.damage_if_unanswered or 0.0))

    @staticmethod
    def _names(units: list[dict]) -> str:
        counts: dict[str, int] = {}
        for u in units:
            counts[_name(u)] = counts.get(_name(u), 0) + 1
        parts = [n if k == 1 else f"{n} x{k}" for n, k in counts.items()]
        return ", ".join(parts[:4]) + (f" +{len(parts) - 4}" if len(parts) > 4 else "")

    def _summary(self, push: Push) -> str:
        """'balloon (5 elixir), ~3,200 tower damage if unanswered, tower in 9.3s' (the damage when it is simulated)."""
        text = f"{self._names(push.units)} ({fmt_num(push.elixir)} elixir)"
        if push.damage_if_unanswered is not None:
            text += f", ~{int(round(push.damage_if_unanswered, -2)):,} tower damage if unanswered"
            if push.destroys_tower:
                text += " (destroys the tower)"
        return text + f", tower in {push.eta_s:.1f}s"

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
        summary = f"{self._names(units)} ({fmt_num(elixir)} elixir), tower in {eta_s:.1f}s"
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

    def _lane_allies(self, lane: str, state: dict, t: float | None = None) -> list[dict]:
        """Our units and buildings in the lane (on our half) as simulation defenders: the ones on the field now (one
        per detection) and the cards we played in the last ANSWERED_PLAY_S seconds that perception does not show yet
        (still deploying). Spells and cards without stats are left out."""
        out: list[dict] = []
        seen: set[str] = set()
        for u in state.get("units", []):
            tile, card = _tile(u), u.get("card")
            if (u.get("side") != "ally" or tile is None or not card or tile[1] > OWN_HALF_MAX_ROW
                    or _side_lane(tile) != lane or self._is_spell(card) or not ustats.has_stats(card)):
                continue
            out.append({"card": card, "unit_class": u.get("unit_class"), "tile": list(tile), "placed": True,
                        "single": True})
            seen.add(card)
        now = self._clock(state, t)
        for p in (state.get("own") or {}).get("recent_plays") or []:
            tile, card = _tile(p), p.get("card")
            if (now is None or p.get("t") is None or tile is None or not card or card in seen
                    or not 0 <= now - p["t"] <= ANSWERED_PLAY_S or tile[1] > OWN_HALF_MAX_ROW
                    or _side_lane(tile) != lane or self._is_spell(card) or not ustats.has_stats(card)):
                continue
            out.append({"card": card, "tile": list(tile), "ready_s": max(0.0, ALLY_DEPLOY_S - (now - p["t"]))})
            seen.add(card)
        return out

    def _recent_kamikazes(self, state: dict, t: float | None) -> frozenset:
        """Kamikaze cards we played in the last KAMIKAZE_REPEAT_S seconds (needs the clock `t`)."""
        if t is None:
            return frozenset()
        return frozenset(p["card"] for p in (state.get("own") or {}).get("recent_plays") or []
                         if p.get("card") and p.get("t") is not None and 0 <= t - p["t"] <= KAMIKAZE_REPEAT_S
                         and self._has(self._role(p["card"]), "kamikaze"))

    def _kg_threat(self, push: Push) -> tuple[str | None, dict | None]:
        """(card, unit) the graph is asked about: the costliest known card of the push (the first of them in the push's
        order of urgency), which is its lead in almost every real push; (None, None) for a push of unknown units."""
        best: tuple[float, str, dict] | None = None
        for u in push.units:
            card = u.get("card")
            if not card or self._role(card).get("type") == "spell":
                continue
            cost = self._cost(card)
            if best is None or cost > best[0]:
                best = (cost, card, u)
        return (best[1], best[2]) if best else (None, None)

    def _kg_answers(self, threat: str, hand: list[str]) -> dict:
        """card -> `kg2.Answer` for the cards of `hand` against `threat` (cached: the graph is static). {} when the
        graph fails."""
        key = (threat, tuple(sorted(hand)))
        got = self._kg_cache.get(key)
        if got is None:
            try:
                got = {a.card: a for a in self.kg2.answers_for(threat, list(hand), k=max(len(hand), 1))}
            except Exception:                                        # noqa: BLE001  the graph never breaks the engine
                got = {}
            if len(self._kg_cache) > 512:
                self._kg_cache.clear()
            self._kg_cache[key] = got
        return got

    def _ctx(self, push: Push, state: dict, t: float | None = None, hand: list[str] | None = None) -> _PushCtx:
        """The defence context of the push; `t` is the clock of the play times (our recent plays that answer the push,
        the kamikaze cards we played a moment ago), `hand` the cards the graph is asked about (default: the legal ones)."""
        unit = push.units[0]
        pos = _lead_tile(unit, LEAD_S) or list(_tile(unit))
        card = unit.get("card")
        norm: dict[str, float] = {}
        if self.kg is not None and self.kg2 is None and card:
            scores = self.kg.counter_scores(card) or {}
            top = max(scores.values(), default=0.0)
            if top > 0:
                norm = {k: v / top for k, v in scores.items()}
        roles = [self._role(u.get("card")) for u in push.units]
        pull_unit, pull_tile = self._pull(push)
        lead_role = self._role(card)
        answered_by = self._answered_by(push, pos, state, t)
        tower = self._tower_of(push.lane, state)
        allies = self._lane_allies(push.lane, state, t)
        if allies and not answered_by and push.allies_hold:
            # our units on the field now, simulated against the push, already stop it (even when the plain "elixir
            # near the lead unit" check of `_answered_by` does not see it)
            answered_by = tuple(dict.fromkeys(a["card"] for a in allies))
        ctx = _PushCtx(push, unit, pos, OWN_TOWER_X[push.lane], 1 if push.lane == "left" else -1, _name(unit),
                        all(r.get("flying") for r in roles), norm,
                        heavy=any(r.get("hp_class") in ("medium", "tank") for r in roles),
                        swarm_only=all(r.get("hp_class") == "swarm" or u.get("card") in SWARM_SPELL_UNITS
                                       for r, u in zip(roles, push.units)),
                        answered_by=answered_by,
                        blocked=self._recent_kamikazes(state, t),
                        air_blocked=air_lead_blocks_ground(push.units, self._role),
                        pull_unit=pull_unit, pull_tile=pull_tile,
                        kite=(not lead_role.get("flying") and lead_role.get("targets") != "buildings"
                              and not answered_by),
                        tower=tower, allies=allies)
        if self.kg2 is not None:
            threat, tunit = self._kg_threat(push)
            if threat is not None:
                ctx.kg_threat = threat
                ctx.kg_pos = _lead_tile(tunit, LEAD_S) or list(_tile(tunit))
                ctx.kg_answers = self._kg_answers(threat, hand if hand is not None else legal_cards(state, self.cards))
        return ctx

    # -- pulling a building targeter -----------------------------------------------------------------------------------
    @staticmethod
    def _both_range_walk(start, end) -> float:
        """Tiles of the straight walk start -> end that lie inside the range of both towers that fight it: the
        princess tower of its side and the king tower."""
        princess = min((OWN_TOWER_TILES[k] for k in ("own_left", "own_right")), key=lambda xy: math.dist(start, xy))
        length = math.dist(start, end)
        n = max(1, int(length / PULL_STEP))
        inside = 0
        for i in range(n):
            f = (i + 0.5) / n
            p = (start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f)
            if math.dist(p, princess) <= PRINCESS_RANGE and math.dist(p, OWN_TOWER_TILES["own_king"]) <= KING_RANGE:
                inside += 1
        return length * inside / n

    def _pull_tile(self, pos) -> list[float] | None:
        """Where a building pulls the threat at `pos`: a tile of a 0.5 grid (x 6..12, y 4..11) at least PULL_MIN_DIST
        from it and at least PULL_CLOSER_TILES closer to it than the nearest tower of ours (so it walks to the building
        first); of those, inside the centre band PULL_BAND, the one where the threat walks the longest inside the range
        of the princess tower of its side and the king tower (ties: nearer to the threat, then to the middle of the
        arena). None when no tile qualifies."""
        reach = nearest_own_tower_dist(pos) - PULL_CLOSER_TILES
        best, best_rank = None, None
        for ix in range(int((PULL_X[1] - PULL_X[0]) * 2) + 1):
            for iy in range(int((PULL_Y[1] - PULL_Y[0]) * 2) + 1):
                tile = (PULL_X[0] + ix / 2, PULL_Y[0] + iy / 2)
                d = math.dist(tile, pos)
                if d < PULL_MIN_DIST or d > reach:
                    continue
                rank = (PULL_BAND[0] <= tile[0] <= PULL_BAND[1], round(self._both_range_walk(pos, tile), 3), -d,
                        -abs(tile[0] - 9.0), tile[1])
                if best_rank is None or rank > best_rank:
                    best, best_rank = tile, rank
        return None if best is None else [best[0], best[1]]

    def _pull(self, push: Push) -> tuple[dict | None, list[float] | None]:
        """(the push's building targeter, the tile that pulls it); the tile is None once it has locked on its tower
        (its distance to the nearest tower of ours is within its attack range + 1 tile) or when no tile qualifies."""
        unit = targeter_of(push.units, self._role)
        if unit is None:
            return None, None
        pos = _lead_tile(unit, PULL_LEAD_S) or list(_tile(unit))
        if locked_on(pos, self._role(unit.get("card"))):
            return unit, None
        return unit, self._pull_tile(pos)

    # -- defence -------------------------------------------------------------------------------------------------------
    def _defends(self, card: str, role: dict) -> bool:
        """Spells that do damage can defend; rage / clone / freeze ... and the win-condition spells cannot. Troops that
        only attack buildings (hog rider, giant, wall breakers ...) are kept for offence."""
        if role.get("type") != "spell":
            return not (role.get("type") == "troop" and role.get("targets") == "buildings")
        return (card not in NON_DAMAGE_SPELLS and role.get("spell_class") != "utility"
                and not self._has(role, "win_condition"))

    def _spell_aim(self, card: str, ctx: _PushCtx, state: dict):
        """The role-model engine's aim (threat model off): `spell_target` picks its centres by the units' current tiles,
        so ask for the cell the threat stands in now; if nothing worthwhile is there, for the cell of its lead tile."""
        self._sim_on()                                  # (syncs how the policy prices spells with the threat model)
        now = _tile(ctx.unit)
        for tile in (now, ctx.pos):
            aim = self.policy.spell_target(card, lane_of(tile), zone_of(tile), state)
            if aim is not None:
                return aim
        return None

    def _spell_clusters(self, state: dict, hand: list[str]) -> dict[str, _SpellAim] | None:
        """card -> the best cluster anywhere (`ReactivePolicy.spell_best`: any half and lane, tanks are damaged survivors,
        +1 elixir for an enemy princess tower) for every damage spell in `hand`; None when the threat model is off (the
        old lead-cell aim of `_spell_aim` then applies). `killed` are the members the spell's damage kills."""
        if not self._sim_on():
            return None
        out: dict[str, _SpellAim] = {}
        for card in hand:
            role = self._role(card)
            if role.get("type") != "spell" or not self._defends(card, role):
                continue
            best = self.policy.spell_best(card, state)
            if best is None:
                continue
            tile, value, members, kills = best
            damage = (ustats.spell_stat(card) or {}).get("damage") or 0
            killed = tuple(u for u in members if u.get("card") and ustats.unit_stat(u)["hp"] <= damage)
            out[card] = _SpellAim(list(tile), float(value), sum(1 for u in members if self.policy._counted(u)),
                                  tuple(members), killed, int(kills), self.policy._hits_tower(card, tile))
        return out

    @staticmethod
    def _touches(aim: _SpellAim, push: Push) -> bool:
        """A unit of the push stands in the spell's circle."""
        return any(u is m for u in push.units for m in aim.members)

    def _spell_why(self, card: str, aim: _SpellAim, room: int = MAX_REASON) -> str:
        """'fireball night-witch + e-wizard + bats x3: kills 4, ~9 elixir [+ tower]' within `room` characters: the
        non-tank units of the circle, the costliest first, dropped from the end until it fits."""
        counts: dict[str, int] = {}
        for u in aim.members:
            name = _name(u)
            if name in TANK_CARDS or self._role(u.get("card")).get("type") == "spell":
                continue
            counts[name] = counts.get(name, 0) + 1
        order = sorted(counts, key=lambda n: (-self._cost(n), n))
        names = [f"{self._short(n)} x{counts[n]}" if counts[n] > 1 else self._short(n) for n in order]
        tail = f": kills {aim.kills}, ~{round(aim.value, 1):g} elixir" + (" + tower" if aim.tower else "")
        while True:
            why = f"{card} {' + '.join(names)}{tail}" if names else f"{card}{tail}"
            if len(why) <= room or len(names) <= 1:
                return why
            names = names[:-1]

    @staticmethod
    def _short(name: str) -> str:
        for long, short in SPELL_REASON_NAMES.items():
            if name.startswith(long):
                return short + name[len(long):]
        return name

    def _stop_class(self, role: dict, ctx: _PushCtx) -> str | None:
        """How well the card stops a push: high / medium / low, None for spells (judged by their aim). Buildings and
        tank killers count as high; a kamikaze card as low against a push with medium / tank units. A low-dps card
        with the air_defense role counts as medium against a push with flyers (that is what the role is for)."""
        if role.get("type") == "spell":
            return None
        if role.get("type") == "building":
            # a building stops a targeter only by pulling it: with no pull (locked on its tower) it only chips
            return "low" if "building_targeter" in ctx.push.needs and ctx.pull_tile is None else "high"
        if self._has(role, "tank_killer"):
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

    def _defence_score(self, card: str, role: dict, ctx: _PushCtx, aim,
                       prevent: float | None = None) -> tuple[float, float] | None:
        """(score, trade) of `card` against the push, or None when it cannot answer an air-only push. `prevent` is the
        share of the push's tower damage the move removes by the lane simulation (0..1; None = not simulated): it adds
        `w.prevent` x that, and the role terms (tank / swarm / targeter / stop / kg) then count `w.role_scale` as a
        tie-break."""
        push, needs = ctx.push, ctx.push.needs
        w = self.w
        cost = self._cost(card)
        is_spell, is_building = role.get("type") == "spell", role.get("type") == "building"
        score = w["base"]
        if "air" in needs and not self._hits_air(card, role):
            if ctx.air_blocked:
                return None                                  # a flyer leads the push: a ground-only card is wasted
            score -= w["air_cannot_hit_penalty"]
        pulls = is_building and ctx.pull_tile is not None            # a building only counts as a pull where one works
        rs = 0.0                                                     # the role terms
        if "tank" in needs:
            if self._has(role, "tank_killer") or role.get("dps_class") == "high":
                rs += w["tank_killer_bonus"]
            if pulls and "building_targeter" in needs:
                rs += w["tank_building_bonus"]
            if self._has(role, "swarm"):
                rs += w["tank_swarm_bonus"]
            if is_spell and not (aim is not None and aim[1] >= cost):
                rs -= w["tank_spell_penalty"]
        if "swarm" in needs:
            if role.get("splash") or (is_spell and "swarm" in (role.get("kills") or [])):
                rs += w["swarm_splash_bonus"]
            elif role.get("type") == "troop" and role.get("range") == "melee":
                rs -= w["swarm_melee_penalty"]
        if "building_targeter" in needs:
            if pulls:
                rs += w["targeter_building_bonus"]
            if self._has(role, "swarm") or self._has(role, "kamikaze"):
                rs += w["targeter_swarm_bonus"]
            if self._has(role, "mini_tank"):
                rs += w["targeter_mini_tank_bonus"]
        if ctx.heavy:
            stop = self._stop_class(role, ctx)
            if stop is not None:
                rs += w["stop_" + stop]
        rs += w["kg_factor"] * ctx.kg_norm.get(card, 0.0)
        if prevent is None:
            score += rs
        else:
            score += w["role_scale"] * rs + w["prevent"] * prevent
        ans = ctx.kg_answers.get(card)
        if ans is not None:
            score += w["kg_prior"] * ans.score                               # how the pros (and the wiki) answered it
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
            if isinstance(aim, _SpellAim) and aim.members:
                return tile, self._spell_why(card, aim)
            near = [_name(u) for u in push.units
                    if math.dist(tile, _tile(u)) <= SPELL_RADIUS.get(card, 2.5) + 0.5
                    and _name(u) not in TANK_CARDS and self._role(u.get("card")).get("hp_class") != "tank"]
            names = list(dict.fromkeys(near)) or [threat]
            room = (MAX_REASON_SAVES - TRADE_SUFFIX_ROOM - SAVES_SUFFIX_ROOM if self.w["threat_model"] > 0
                    else MAX_REASON - TRADE_SUFFIX_ROOM)
            while True:                                                 # leave room for ", saves ~N HP, trade +N"
                why = f"{card} {' + '.join(names)} ({fmt_num(round(aim[1], 1))} elixir)"
                if len(why) <= room or len(names) == 1:
                    return tile, why
                names = names[:-1]
        if role.get("type") == "building":
            if "building_targeter" in push.needs and ctx.pull_tile is not None:
                return list(ctx.pull_tile), f"pulls {_name(ctx.pull_unit)} to the centre"
            return _clamp_tile(ctx.tower_x + 2 * ctx.toward_centre, 6.5), f"blocks {threat} at the tower"
        # a specific role (kamikaze, swarm) beats the generic range rules
        if self._has(role, "kamikaze"):
            return _clamp_tile(pos[0], pos[1] - 2.5), f"blows up on {threat} up front"
        if self._has(role, "swarm"):
            return _clamp_tile(pos[0], pos[1]), f"swarm on {threat}"
        front = any(self._has(role, r) for r in ("tank", "mini_tank", "tank_killer"))
        if ctx.kite and (self._has(role, "mini_tank") or role.get("range", "melee") == "melee"):
            # a troop-targeter walks to the nearest troop: stand towards the middle so both towers (and the king) hit it
            return _clamp_tile(KITE_X - KITE_DX * ctx.toward_centre, max(pos[1] - KITE_BEHIND, 1.0)), \
                f"kites {threat} to the centre"
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

    # -- pro offsets (K4b) ---------------------------------------------------------------------------------------------
    def _pro_tile(self, card: str, role: dict, ans, ctx: _PushCtx) -> tuple[list[float], str] | None:
        """(tile, reason without the suffixes) from the pro offset of `ans` (the card's, or its `via` card's, offset
        against the push's costliest card), or None. The offset is stored in the canonical lane (positive dx = toward the
        centre): it is mirrored back for a threat at x >= 9, added to the threat's lead tile and clamped to our half. It
        needs >= PRO_OFFSET_MIN_N events and a spread of at most PRO_OFFSET_MAX_IQR; a spell keeps its aim, and a
        building against a building targeter is placed there only where it still pulls (a pull tile, threat not yet
        locked on its tower)."""
        off = getattr(ans, "offset", None)
        if role.get("type") == "spell" or not isinstance(off, dict) or ctx.kg_pos is None:
            return None
        n = off.get("n")
        n = getattr(ans, "support", 0) if n is None else n
        nums = [off.get(k) for k in ("dx_med", "dy_med")]
        if not (isinstance(n, (int, float)) and n >= PRO_OFFSET_MIN_N
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nums)):
            return None
        if any(isinstance(off.get(k), (int, float)) and off[k] > PRO_OFFSET_MAX_IQR for k in ("dx_iqr", "dy_iqr")):
            return None
        pos = ctx.kg_pos
        dx = -nums[0] if pos[0] >= 9.0 else nums[0]
        tile = _clamp_tile(pos[0] + dx, pos[1] + nums[1])
        if role.get("type") == "building" and "building_targeter" in ctx.push.needs:
            if ctx.pull_tile is None or ctx.pull_unit is None:
                return None
            at = _lead_tile(ctx.pull_unit, PULL_LEAD_S) or list(_tile(ctx.pull_unit))
            if not PULL_MIN_DIST <= math.dist(tile, at) <= nearest_own_tower_dist(at) - PULL_CLOSER_TILES:
                return None
            return tile, f"pulls {_name(ctx.pull_unit)} to the centre"
        return tile, f"{card} at the pro spot vs {ctx.kg_threat}"

    def _sim_adequate(self, sim, push: Push) -> bool:
        """The simulated tower damage left is small (see `adequate_frac`) or nothing reaches the tower."""
        return bool(sim.tower_damage <= self._ok_limit(push) or sim.eta_s is None)

    def _choose_tile(self, card: str, role: dict, ctx: _PushCtx, aim, state: dict):
        """(tile, reason, simulation, tile_source) of a defend move: the geometric rule of MAC-AG, or the pro offset when
        the graph has one - unless the simulation finds the pro tile inadequate and the geometric one adequate."""
        tile, why = self._defence_place(card, role, ctx, aim, state)
        removed = aim.killed if isinstance(aim, _SpellAim) else ()
        sim = self._simulate_move(card, tile, ctx, removed) if removed else self._simulate_move(card, tile, ctx)
        ans = ctx.kg_answers.get(card)
        pro = None if ans is None else self._pro_tile(card, role, ans, ctx)
        if pro is None:
            return tile, why, sim, "geometric"
        psim = self._simulate_move(card, pro[0], ctx)
        if sim is None or psim is None or self._sim_adequate(psim, ctx.push) or not self._sim_adequate(sim, ctx.push):
            return pro[0], pro[1], psim, "pro_offset"
        return tile, why, sim, "geometric"

    @staticmethod
    def _kg_evidence(ans, pro_spot: bool = False) -> tuple[str, ...]:
        """The strongest evidence of the graph for `ans` as reason clauses, richest first: "pros 70% held n=23",
        "role of cannon", "wiki counter" (+ ", pro spot" when the tile is the pros'); () when it says nothing worth
        quoting (a weak score, or only the simulation's STOPS edge, which the reason's "saves ~N" already quotes)."""
        if ans is None or ans.score < KG_EVIDENCE_MIN_SCORE:
            return ()
        from clashbot.kg2 import SOURCE_WEIGHTS                  # the graph's own weights: what really carried the score
        contrib = {k: SOURCE_WEIGHTS.get(k, 0.0) * float(ans.sources.get(k) or 0.0)
                   for k in ("pro", "role_transfer", "wiki", "sim")}
        top = max(contrib, key=lambda k: contrib[k])
        if contrib[top] <= 0 or top == "sim":
            return ()
        if top == "pro":
            held = "" if ans.held_rate is None else f" {ans.held_rate * 100:.0f}% held"
            full, short = f"pros{held} n={ans.support}", f"pros n={ans.support}"
        elif top == "role_transfer":
            full = short = f"role of {ans.via}"
        else:
            full = short = "wiki counter"
        spot = f"pro spot via {ans.via}" if ans.via and top != "role_transfer" else "pro spot"   # (a transferred offset is the via card's)
        out = [f"{full}, {spot}", full] if pro_spot else [full]
        if short != full:
            out.append(short)
        return tuple(out)

    def _simulate_move(self, card: str, tile, ctx: _PushCtx, removed: tuple = ()):
        """The lane simulation of the push with `card` played at `tile` on top of our units already there (`removed`: units
        of the push a damage spell kills, left out); None when the push or the card has no stats (the move is then judged
        by the roles alone)."""
        if ctx.push.damage_if_unanswered is None or not ustats.has_stats(card):
            return None
        units = [u for u in ctx.push.units if not any(u is r for r in removed)] if removed else ctx.push.units
        return ustats.simulate_lane(units, ctx.allies + [{"card": card, "tile": list(tile)}], ctx.tower)

    def _defence_moves(self, ctx: _PushCtx, hand: list[str], state: dict,
                       clusters: dict[str, _SpellAim] | None = None) -> list[TacticalMove]:
        """The defend moves of one push. `clusters` (threat model on): a damage spell is a move for the push when its best
        cluster holds a unit of it; None: the old lead-cell aim."""
        moves = []
        if ctx.push.tower_ok:
            return moves                                  # the tower handles it: no card is spent on it (MAC-AH)
        unanswered = ctx.push.damage_if_unanswered
        # what a move prevents is measured from what our units already there leave (the whole push when none are)
        base = unanswered if ctx.push.damage_with_allies is None else ctx.push.damage_with_allies
        for card in hand:
            role = self._role(card)
            if not self._defends(card, role) or card in ctx.blocked:
                continue
            aim = None
            if role.get("type") == "spell":
                if clusters is None:
                    aim = self._spell_aim(card, ctx, state)
                else:
                    aim = clusters.get(card)
                    if aim is not None and not self._touches(aim, ctx.push):
                        aim = None
                if aim is None:
                    continue
            tile, why, sim, source = self._choose_tile(card, role, ctx, aim, state)
            prevented = None if sim is None else max(0.0, base - sim.tower_damage)
            scored = self._defence_score(card, role, ctx, aim,
                                         None if prevented is None else prevented / max(unanswered, PREVENT_REF))
            if scored is None:
                continue
            score, trade = scored
            key = action_key(card, lane_of(tile), zone_of(tile))
            evidence = self._kg_evidence(ctx.kg_answers.get(card), source == "pro_offset")
            cluster = isinstance(aim, _SpellAim) and bool(aim.members)
            if cluster:
                reason = _clip(why)                      # a spell move: what it hits and kills, at most MAX_REASON
                for ev in evidence:                      # ... plus the graph's evidence when it fits
                    if len(f"{reason}, {ev}") <= MAX_REASON_KG:
                        reason = f"{reason}, {ev}"
                        break
            elif not self._claims_trade(role, ctx):
                reason = _clip(f"slows {ctx.threat}")
            elif evidence:
                reason = _with_evidence(why, trade, prevented, evidence)
            else:
                reason = _with_trade(why, trade, prevented)
            cost = self._cost(card)
            moves.append(TacticalMove(card, key, tile, "defend", reason, score, trade,
                                      stop=self._stop_class(role, ctx) or "spell",
                                      value=None if aim is None else float(aim[1]),
                                      adequate=None if sim is None else self._sim_adequate(sim, ctx.push),
                                      prevented=None if sim is None else round(prevented, 1),
                                      efficiency=None if sim is None or cost <= 0 else round(prevented / cost, 1),
                                      damage_with=None if sim is None else float(sim.tower_damage),
                                      tile_source=source, members=tuple(aim.members) if cluster else (),
                                      kills=aim.kills if cluster else 0))
        return moves

    def _spell_value_moves(self, clusters: dict[str, _SpellAim] | None, taken: set[str]) -> list[TacticalMove]:
        """A `spell_value` move for every damage spell whose best cluster belongs to no push that wants an answer (not
        in `taken`): cast it for what it kills (and the tower). Score: `spell_value` + `spell_value_ratio` x (value /
        cost - 1), at most `spell_value_cap` above the base."""
        w = self.w
        out: list[TacticalMove] = []
        for card, aim in (clusters or {}).items():
            if card in taken:
                continue
            tile = [_r05(aim.tile[0]), _r05(aim.tile[1])]
            cost = self._cost(card)
            bonus = min(w["spell_value_cap"], max(0.0, w["spell_value_ratio"] * (aim.value / cost - 1.0))) if cost > 0 else 0.0
            out.append(TacticalMove(card, action_key(card, lane_of(tile), zone_of(tile)), tile, "spell_value",
                                    _clip(self._spell_why(card, aim)), round(w["spell_value"] + bonus, 3), None,
                                    stop="spell", value=float(aim.value), tile_source="spell",
                                    members=tuple(aim.members), kills=aim.kills))
        return out

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

    def _offence_moves(self, state: dict, hand: list[str], t: float | None = None,
                       pushes: list[Push] | None = None) -> list[TacticalMove]:
        """The offence moves. With no push within URGENT_ETA_S: every intent. During pressure (such a push exists)
        only the windows of MAC-AG: a counter-push in a lane whose push is answered or gone, a split into the lane
        away from an answered push, and a win-condition leak at a full bar while no push is within LEAK_PUSH_MIN_ETA_S."""
        own, opp = state["own"], state.get("opponent") or {}
        elixir = own.get("elixir") or 0.0
        w = self.w
        pushes = pushes or []
        pressure = any(p.eta_s <= URGENT_ETA_S for p in pushes)
        unanswered = {p.lane for p in pushes if not p.answered}         # lanes with a push that is still on
        wcs = [c for c in hand if self._has(self._role(c), "win_condition")]
        moves: list[TacticalMove] = []

        def add(card: str, lane: str, intent: str, score: float, reason: str) -> None:
            tile = self._wc_tile(card, self._role(card), lane)
            moves.append(TacticalMove(card, action_key(card, lane_of(tile), zone_of(tile)), tile, intent,
                                      _clip(reason), score, None))

        # counter-push: back an ally unit (not a building) that is already across the middle of our half; during
        # pressure only in a lane whose push is answered or gone
        allies: dict[str, dict] = {}
        for u in state.get("units", []):
            tile = _tile(u)
            if (u.get("side") == "ally" and tile is not None and tile[1] >= 9
                    and not self._is_building(u.get("card") or "")):
                lane = _side_lane(tile)
                if lane not in allies or tile[1] > _tile(allies[lane])[1]:
                    allies[lane] = u
        for lane in ("left", "right"):
            if lane in allies and not (pressure and lane in unanswered):
                for c in wcs:
                    if elixir >= self._cost(c) + 1:
                        add(c, lane, "counter_push", w["counter_push"],
                            f"backs {_name(allies[lane])} in the {lane} lane")

        # split: they committed a push worth 5+ elixir that our side already answers; hit the other lane
        for push in pushes:
            other = OPPOSITE_LANE[push.lane]
            if push.answered and push.elixir >= SPLIT_MIN_ELIXIR and other not in unanswered:
                for c in wcs:
                    if elixir >= self._cost(c) + 1:
                        add(c, other, "split", w["split"], f"they committed {fmt_num(push.elixir)} elixir {push.lane}")

        if pressure:
            # a full bar is wasted even now: the win condition at the bridge, away from the push, unless it is at the tower
            if elixir >= LEAK_ELIXIR and wcs and min(p.eta_s for p in pushes) > LEAK_PUSH_MIN_ETA_S:
                lanes = {p.lane for p in pushes}
                lane = OPPOSITE_LANE[next(iter(lanes))] if len(lanes) == 1 else (self._weak_lane(state) or "left")
                for c in wcs:
                    add(c, lane, "leak", w["leak_wc"], f"elixir {fmt_num(elixir)} near full, pushes {lane}")
            return moves

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

    # -- hero ability (MAC-AI part 2) ----------------------------------------------------------------------------------
    @staticmethod
    def _turret_spot(state: dict) -> list[float] | None:
        """Where Trusty Turret lands, or None when the ability is not usable: the button is visible and affordable, the
        own elixir (when known) covers its cost and an ally Musketeer is on the field. The spot is her tile
        ABILITY_TURRET_TILES rows toward the enemy, clamped to the arena - the same tile (so the same key) as
        `candidates.hero_ability_move`."""
        own = state.get("own") or {}
        ab = own.get("hero_ability")
        if not isinstance(ab, dict) or not ab.get("visible") or not ab.get("affordable"):
            return None
        elixir = own.get("elixir")
        if isinstance(elixir, (int, float)) and not isinstance(elixir, bool) and elixir < ABILITY_COST:
            return None
        for u in state.get("units") or []:
            tile = _tile(u)
            if u.get("side") == "ally" and u.get("card") == "musketeer" and tile is not None:
                return [min(max(float(tile[0]), 0.0), 17.0), min(max(float(tile[1]) + ABILITY_TURRET_TILES, 0.0), 31.0)]
        return None

    @staticmethod
    def _turret_defender(spot: list[float]) -> dict | None:
        """The simulation defender of the turret: the KB stats of the hero Musketeer's Trusty Turret at her level (a
        building that hits air and ground, acts one deploy time + the tap latency after the decision, lives 10 s); None
        when the table has no hero ability."""
        st = ustats.stat_at("musketeer", ustats.own_level("musketeer"), "hero")
        tur = (((st or {}).get("abilities") or {}).get("trusty-turret") or {}).get("turret")
        if not tur or not tur.get("hitpoints") or not tur.get("damage"):
            return None
        dmg = float(tur["damage"])
        hit = dmg / float(tur["damage_per_second"]) if tur.get("damage_per_second") else 1.0
        deploy = float(tur.get("deploy_s") or 1.0)
        stat = {"name": "Trusty Turret", "count": 1, "hp": int(tur["hitpoints"]), "damage": int(dmg),
                "dps": float(tur.get("damage_per_second") or dmg / hit), "hit_speed_s": hit, "first_hit_s": hit,
                "range_tiles": float(tur.get("range_tiles") or 4.0), "speed_tiles_s": 0.0,
                "targets": tur.get("targets") or "air_ground", "flying": False, "kamikaze": False,
                "splash_radius_tiles": 0.0, "death_damage": 0, "death_radius_tiles": 0.0, "death_delay_s": 0.0,
                "death_spawn": None, "spawn": None, "deploy_s": deploy, "lifetime_s": float(tur.get("lifetime_s") or 10.0)}
        return {"card": "musketeer-turret", "stat": stat, "tile": list(spot), "ready_s": ustats.DEPLOY_LATENCY_S + deploy}

    def _ability_moves(self, state: dict, ctxs: list[_PushCtx]) -> list[TacticalMove]:
        """The `ability` move (at most one): the turret simulated at its spawn tile against each push on top of our units
        already in the lane. It is proposed for the push where it prevents the most, and only when that is at least
        `ability_min_prevent` (1,000, D-049) tower HP; a push the tower handles never asks for it. Needs the threat
        model."""
        if not self._sim_on():
            return []
        spot = self._turret_spot(state)
        turret = self._turret_defender(spot) if spot is not None else None
        if turret is None:
            return []
        w = self.w
        best: tuple[float, _PushCtx, ustats.SimResult] | None = None
        for ctx in ctxs:
            push = ctx.push
            if push.tower_ok or push.damage_if_unanswered is None:
                continue
            base = push.damage_if_unanswered if push.damage_with_allies is None else push.damage_with_allies
            res = ustats.simulate_lane(push.units, ctx.allies + [turret], ctx.tower)
            prevented = max(0.0, base - res.tower_damage)
            if prevented >= w["ability_min_prevent"] and (best is None or prevented > best[0]):
                best = (prevented, ctx, res)
        if best is None:
            return []
        prevented, ctx, res = best
        push = ctx.push
        pulls = "building_targeter" in push.needs and ctx.pull_unit is not None
        n = f"{int(round(prevented, -2)):,}"
        why = f"turret {'pulls' if pulls else 'holds'} {_name(ctx.pull_unit) if pulls else ctx.threat} {push.lane}"
        for clause in (f", saves ~{n} tower HP, 3 elixir", f", saves ~{n} HP, 3 elixir", ", 3 elixir"):
            if len(why) + len(clause) <= MAX_REASON_SAVES:
                why += clause
                break
        score = round(w["ability"] + w["prevent"] * prevented / max(push.damage_if_unanswered, PREVENT_REF), 3)
        return [TacticalMove(ABILITY_KEY, ability_key(lane_of(spot), zone_of(spot)), list(spot), "ability", why, score,
                             None, stop="turret", adequate=self._sim_adequate(res, push), prevented=round(prevented, 1),
                             efficiency=round(prevented / ABILITY_COST, 1), damage_with=float(res.tower_damage),
                             tile_source="ability", push_lane=push.lane)]

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
        if pushes and all(p.tower_ok for p in pushes):
            return "tower handles it"
        if pushes:
            return "no card in hand answers the push" if hand else "not enough elixir"
        own, opp = state["own"], state.get("opponent") or {}
        return (f"no threat, saving elixir (own {fmt_num(own.get('elixir'))}, "
                f"opp ~{fmt_num(opp.get('elixir_estimate'))})")

    def _lines(self, pushes: list[Push], moves: list[TacticalMove], hold_reason: str | None) -> list[str]:
        lines = [f"tactics: push {p.lane}: {p.summary}; needs {', '.join(p.needs) or 'nothing special'}"
                 + ("; tower handles it" if p.tower_ok else "")
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
        pushes = self._pushes(state, t)
        clusters = self._spell_clusters(state, hand)
        moves: list[TacticalMove] = []
        ctxs: list[_PushCtx] = []
        # spells that get no `spell_value` move: their cluster is the defend move of a push that is still on, or holds
        # only units of pushes the tower handles
        taken: set[str] = set()
        for push in pushes:
            ctx = self._ctx(push, state, t, hand)
            ctxs.append(ctx)
            push.answered, push.answered_by = bool(ctx.answered_by), list(ctx.answered_by)
            new = self._defence_moves(ctx, hand, state, clusters)
            moves += new
            if not push.answered:
                taken |= {m.card for m in new if m.members}
        for card, aim in (clusters or {}).items():
            ok_units = [u for p in pushes if p.tower_ok for u in p.units]
            if all(any(u is o for o in ok_units) for u in aim.members):
                taken.add(card)                                  # only units the tower handles: no card is spent on it (MAC-AH)
        defence = self._rank(moves, None)
        offence = self._offence_moves(state, hand, t, pushes)
        abilities = self._ability_moves(state, ctxs)
        spells = self._spell_value_moves(clusters, taken)
        moves = self._rank(moves + offence + abilities + spells)
        best = moves[0] if moves else None
        hold = None if best else self._hold_reason(state, pushes, hand)
        return TacticalReport(pushes, moves, best, hold, self._lines(pushes, moves, hold), defence,
                              self._rank(offence, None), abilities, self._rank(spells, None))

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
        if report.pushes[0].tower_ok:
            return None
        ctx = self._ctx(report.pushes[0], state)
        aim = None
        if role.get("type") == "spell":
            clusters = self._spell_clusters(state, [card])
            if clusters is None:
                aim = self._spell_aim(card, ctx, state)
            else:
                aim = clusters.get(card)
                if aim is not None and not self._touches(aim, report.pushes[0]):
                    aim = None
        sim, push = None, report.pushes[0]
        unanswered = push.damage_if_unanswered
        base = unanswered if push.damage_with_allies is None else push.damage_with_allies
        if aim is not None or role.get("type") != "spell":
            sim = self._choose_tile(card, role, ctx, aim, state)[2]
        prevent = None if sim is None else max(0.0, base - sim.tower_damage) / max(unanswered, PREVENT_REF)
        scored = self._defence_score(card, role, ctx, aim, prevent)
        return None if scored is None else scored[0]


# -- one rendering and one factory for the live loop and the d1 dataset builder (MAC-AK K4b) -------------------------------
def build_engine(cards: dict | None = None, kg_v1=None, kg2=None, weights: dict | None = None) -> TacticalEngine:
    """The engine as the live loop runs it (`clashbot.live.main`) and `tools/data/build_d1_tac.py` builds the training
    lines with it: the committed weights file (plus `weights`), the threat model, the v1 graph's counter scores (only
    while there is no `kg2`) and the KG2 (the pro-answer prior, the pro offsets). Same arguments, same lines."""
    return TacticalEngine(cards, kg=kg_v1, kg2=kg2, weights=weights)


def kg_fact_lines(kg2, state: dict, max_lines: int = 3) -> list[str]:
    """The `KG:` fact lines of the state for the LLM context (`KG2.facts_lines` with the state's own deck); [] without a
    graph or when it fails - the graph never stops a decision."""
    if kg2 is None:
        return []
    try:
        deck = [c for c in ((state.get("own") or {}).get("deck") or []) if c]
        return [str(x) for x in kg2.facts_lines(state, deck, max_lines=max_lines)]
    except Exception:                                        # noqa: BLE001
        return []


def rank_context(ctx: str, report, facts_lines=()) -> str:
    """The text the ranker reads: `ctx` (the state's context), then the engine's `tactics:` lines (when there is a
    report), then the graph's `KG:` fact lines, newline-joined. The live loop and `build_d1_tac` both render it here, so
    the training lines and the live ranking context are the same string."""
    text = ctx
    if report is not None:
        text += "\n" + "\n".join(report.lines)
    if facts_lines:
        text += "\n" + "\n".join(facts_lines)
    return text
