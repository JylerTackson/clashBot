"""TacticalEngine (MAC-Z2): pushes, defence scores and placement, offence, hold reasons, lines, score_key - hand-made
states, inline roles in the Z1 schema, no models."""
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.game_rules import lane_of, load_cards, zone_of  # noqa: E402
from clashbot.tactics import DEFAULT_WEIGHTS, TacticalEngine, TacticalMove, TacticalReport  # noqa: E402

CARDS = load_cards()


# -- inline roles (Z1 schema) ------------------------------------------------------------------------------------------
def _troop(targets="ground", hp="medium", dps="medium", rng="melee", roles=(), count=1, flying=False, splash=False,
           speed="medium"):
    return {"type": "troop", "targets": targets, "flying": flying, "hp_class": hp, "dps_class": dps,
            "splash": splash, "range": rng, "speed": speed, "count": count, "roles": list(roles), "source": "manual"}


def _building(targets="ground", dps="medium", roles=("building",)):
    return {"type": "building", "targets": targets, "flying": False, "hp_class": "medium", "dps_class": dps,
            "splash": False, "range": "long", "speed": None, "count": 1, "roles": list(roles), "source": "manual"}


def _spell(cls, radius, kills=(), roles=(), hits_air=True):
    role = {"type": "spell", "spell_class": cls, "radius": radius, "kills": list(kills), "roles": list(roles),
            "source": "manual"}
    if not hits_air:
        role["hits_air"] = False                                  # as card_roles.json does for the-log etc.
    return role


ROLES = {
    "knight": _troop(roles=["mini_tank"]),
    "valkyrie": _troop(roles=["splash_defense", "mini_tank"], splash=True),
    "mini-p-e-k-k-a": _troop(dps="high", roles=["mini_tank", "tank_killer"]),
    "mega-knight": _troop(hp="tank", dps="high", splash=True, roles=["tank"]),
    "giant": _troop(targets="buildings", hp="tank", roles=["win_condition", "tank"]),
    "hog-rider": _troop(targets="buildings", speed="very_fast", roles=["win_condition"]),
    "ram-rider": _troop(targets="buildings", roles=["win_condition"]),
    "battle-ram": _troop(targets="buildings", dps="high", roles=["win_condition", "kamikaze"]),
    "wall-breakers": _troop(targets="buildings", hp="light", dps="low", count=2, splash=True, speed="very_fast",
                            roles=["win_condition", "kamikaze"]),
    "miner": _troop(roles=["win_condition"], speed="fast"),
    "berserker": _troop(roles=["mini_tank"]),
    "goblinstein": _troop(targets="air_ground", hp="tank", rng="short", roles=["tank"], speed="slow"),
    "musketeer": _troop(targets="air_ground", rng="long", roles=["support"]),
    "archers": _troop(targets="air_ground", hp="light", rng="long", count=2, roles=["support"]),
    "bats": _troop(targets="air_ground", hp="swarm", dps="low", count=3, flying=True, speed="very_fast",
                   roles=["swarm", "cycle"]),
    "minions": _troop(targets="air_ground", hp="light", rng="short", count=3, flying=True, roles=["swarm"]),
    "skeletons": _troop(hp="swarm", dps="low", count=3, roles=["swarm", "cycle"]),
    "ice-spirit": _troop(targets="air_ground", hp="swarm", dps="low", rng="short", splash=True,
                         roles=["kamikaze", "cycle"]),
    "baby-dragon": _troop(targets="air_ground", rng="short", flying=True, splash=True, roles=["splash_defense"]),
    "tesla": _building(targets="air_ground", roles=["building", "air_defense"]),
    "cannon": _building(roles=["building"]),
    "zap": _spell("small", 2.5, ["swarm"], ["spell_small"]),
    "the-log": _spell("small", 2.0, ["swarm"], ["spell_small"], hits_air=False),
    "fireball": _spell("medium", 2.5, ["swarm", "light"], ["spell_medium"]),
    "rage": _spell("utility", 5.0, [], ["spell_small"]),
    "tornado": _spell("utility", 5.5),
    "royal-delivery": _spell("small", 3.0, ["swarm"], ["spell_small"]),
    "graveyard": _spell("utility", 4.0, [], ["win_condition"]),
    "goblin-barrel": _spell("small", 1.5, [], ["win_condition"]),
    "magic-archer": _troop(targets="air_ground", rng="long", splash=True, roles=["support", "air_defense"]),
    "balloon": _troop(targets="buildings", hp="medium", dps="high", flying=True, roles=["win_condition"]),
    "bomb-tower": _building(roles=["building", "splash_defense"]),
    "fire-spirit": _troop(targets="air_ground", hp="swarm", dps="low", rng="short", splash=True, speed="very_fast",
                          roles=["kamikaze", "cycle"]),
}

# the numeric assertions below are about the default weights, whatever clashbot/resources/tactics_weights.json holds
# MAC-AH adds the threat model (the lane simulation: damage per push, `prevent`, a push the tower handles gets no moves,
# river pushes, simulated ETA). The MAC-Z2 / A2 / AD / AG tests below check the role arithmetic and weak pushes (3 bats,
# a lone skeleton), so `ENGINE` runs the engine with the threat model off (weight `threat_model` 0 = exactly the MAC-AG
# engine); `SIM_ENGINE` is the full engine, and the MAC-AH tests at the end of the file use it.
ROLE_WEIGHTS = {**DEFAULT_WEIGHTS, "threat_model": 0.0}
ENGINE = TacticalEngine(CARDS, ROLES, weights=ROLE_WEIGHTS)
SIM_ENGINE = TacticalEngine(CARDS, ROLES, weights=DEFAULT_WEIGHTS)


# -- state fixtures (as tests/test_policy.py) --------------------------------------------------------------------------
def _state(elixir=7.0, units=None, hand=None, deck=None, opp_elixir=4.0, opp_plays=None, opp_towers=None,
           next_card=None, deck_known=None):
    hand = ["tesla", "knight", "zap", "musketeer"] if hand is None else hand
    return {
        "phase": "double_elixir", "match_seconds": 150,
        "own": {"elixir": elixir, "hand": hand, "next_card": next_card,
                "deck": deck or ["tesla", "knight", "zap", "musketeer", "giant", "fireball", "bats", "skeletons"],
                "towers": {"king": None, "left": 4000, "right": 4000}, "recent_plays": []},
        "opponent": {"elixir_estimate": opp_elixir,
                     "towers": opp_towers or {"king": None, "left": 4000, "right": 4000},
                     "deck_known": list(deck_known or []), "recent_plays": opp_plays or []},
        "units": units or [], "threats": [],
    }


def _enemy(tile, card="knight", eta=None, tower="own_left", pred=None, unit_class=None):
    unit = {"unit_class": unit_class or card, "card": card, "side": "enemy", "tile": list(tile),
            "eta_tower": None if eta is None else {"tower": tower, "s": eta}}
    if pred is not None:
        unit["predicted_tile_2s"] = list(pred)
    return unit


def _ally(tile, card="knight"):
    return {"unit_class": card, "card": card, "side": "ally", "tile": list(tile), "eta_tower": None}


def _giant_right(**kw):
    return _state(units=[_enemy((15, 13), "giant", eta=8.0, tower="own_right")], **kw)


def _bats_left(**kw):
    units = [_enemy(t, "bats", eta=6.0) for t in ((3, 14), (4, 14), (3.5, 15))]
    return _state(units=units, hand=["zap", "knight", "musketeer", "fireball"], **kw)


# 1 -----------------------------------------------------------------------------------------------------------------
def test_giant_right_lane_tesla_pulls_it_to_the_centre():
    rep = ENGINE.analyse(_giant_right())
    assert isinstance(rep, TacticalReport) and isinstance(rep.best, TacticalMove)
    best = rep.best
    assert best.card == "tesla" and best.intent == "defend" and best.trade == 1.0
    assert best.tile == [11.0, 7.0] and best.key == "tesla|middle|mid"           # MAC-AG: the geometric pull tile
    assert best.reason == "pulls giant to the centre, trade +1"
    assert rep.hold_reason is None
    push = rep.pushes[0]
    assert push.lane == "right" and push.needs == ["tank", "building_targeter"] and push.elixir == 5.0
    assert push.summary == "giant (5 elixir), tower in 8.0s"
    assert any("push right" in ln for ln in rep.lines) and any("best tesla" in ln for ln in rep.lines)
    # the knight meets the giant in front, the zap has nothing to hit, the musketeer is a weak answer
    by_card = {m.card: m for m in rep.moves}
    assert by_card["knight"].tile == [15.0, 10.5] and "zap" not in by_card
    assert rep.moves[0].score > by_card["knight"].score > by_card["musketeer"].score


# 2 -----------------------------------------------------------------------------------------------------------------
def test_bats_zap_is_the_best_answer_and_the_knight_is_not_proposed():
    rep = ENGINE.analyse(_bats_left())
    push = rep.pushes[0]
    assert push.lane == "left" and push.needs == ["air", "swarm"] and push.elixir == 2.0
    assert rep.best.card == "zap" and rep.best.trade == 0.0 and rep.best.intent == "defend"
    assert rep.best.tile == [3.5, 14.5] and rep.best.key == "zap|left|bridge"
    assert rep.best.reason == "zap bats (2 elixir), trade +0"
    cards = [m.card for m in rep.moves]
    assert "knight" not in cards and {"zap", "fireball", "musketeer"} <= set(cards)
    assert rep.lines[0] == "tactics: push left: bats x3 (2 elixir), tower in 6.0s; needs air, swarm"
    assert rep.lines[1].startswith("tactics: best zap @[3.5,14.5] defend: ")
    assert rep.lines[2].startswith("tactics: also ")


# 3 -----------------------------------------------------------------------------------------------------------------
def test_air_only_push_proposes_nothing_that_cannot_hit_air():
    state = _bats_left()
    state["own"]["hand"] = ["the-log", "knight", "zap", "musketeer"]
    rep = ENGINE.analyse(state)
    assert {m.card for m in rep.moves} == {"zap", "musketeer"}
    state["own"]["hand"] = ["the-log", "knight", "cannon", "giant"]
    rep = ENGINE.analyse(state)
    assert rep.best is None and rep.hold_reason == "no card in hand answers the push"
    assert rep.lines[-1] == "tactics: hold: no card in hand answers the push"


def test_mixed_push_penalises_but_keeps_a_card_that_cannot_hit_air():
    units = [_enemy((3, 12), "bats", eta=5.0), _enemy((4, 13), "knight", eta=5.5)]
    rep = ENGINE.analyse(_state(units=units, hand=["knight", "musketeer", "valkyrie", "zap"]))
    by_card = {m.card: m for m in rep.moves}
    assert "knight" in by_card and "valkyrie" in by_card
    assert by_card["musketeer"].score > by_card["knight"].score


# 4 -----------------------------------------------------------------------------------------------------------------
def test_overpay_lone_skeleton_fireball_is_not_best_and_a_cheap_troop_wins():
    state = _state(units=[_enemy((3, 10), "skeletons")], hand=["fireball", "knight", "musketeer", "ice-spirit"])
    rep = ENGINE.analyse(state)
    assert rep.pushes[0].eta_s == 2.0 and rep.pushes[0].needs == ["swarm"]      # one unit of a count-3 swarm card
    assert rep.best.card == "ice-spirit"
    assert "fireball" not in [m.card for m in rep.moves]                       # 1 elixir of target for 4: no aim


def test_trade_bonus_and_overpay_penalty():
    state = _state(units=[_enemy((3, 10), "knight", eta=4.0)], hand=["skeletons", "mega-knight", "musketeer"],
                   elixir=9.0)
    by_card = {m.card: m for m in ENGINE.analyse(state).moves}
    # a knight is a medium unit, so the stopping term applies: medium dps +0.15, high +0.3 (MAC-A2 addendum)
    assert by_card["musketeer"].trade == -1.0 and by_card["musketeer"].score == pytest.approx(0.5 + 0.15)
    assert by_card["mega-knight"].trade == -4.0 and by_card["mega-knight"].score == pytest.approx(0.05 + 0.3)
    # skeletons have low dps: -0.35, no trade bonus, and the reason does not claim the +2 trade (the field still holds it)
    assert by_card["skeletons"].trade == 2.0 and by_card["skeletons"].reason == "slows knight"
    assert by_card["skeletons"].score == pytest.approx(0.5 - 0.35)
    assert by_card["mega-knight"].reason.endswith("trade -4")


def test_win_condition_is_kept_for_offence_on_defence():
    state = _state(units=[_enemy((3, 10), "knight", eta=4.0)], hand=["miner", "musketeer"])
    by_card = {m.card: m for m in ENGINE.analyse(state).moves}
    assert by_card["musketeer"].score == pytest.approx(0.5 + 0.15)               # + stop_medium vs the knight
    assert by_card["miner"].score == pytest.approx(0.1 + 0.15)


# 5 -----------------------------------------------------------------------------------------------------------------
def test_punish_the_lane_opposite_the_last_play():
    plays = [{"t": 5.0, "card": "giant", "tile": [3, 25]}, {"t": 8.0, "card": "knight", "tile": [14, 24]}]
    state = _state(elixir=7.0, opp_elixir=2.0, opp_plays=plays, hand=["ram-rider", "zap", "knight", "musketeer"])
    rep = ENGINE.analyse(state)
    assert rep.pushes == []
    assert rep.best.card == "ram-rider" and rep.best.intent == "punish" and rep.best.score == 0.55
    assert rep.best.tile == [3.5, 14.0] and rep.best.key == "ram-rider|left|bridge" and rep.best.trade is None
    assert rep.lines[0].startswith("tactics: best ram-rider @[3.5,14] punish: ")


def test_goblin_barrel_punish_is_aimed_at_the_enemy_tower():
    plays = [{"t": 8.0, "card": "knight", "tile": [3, 24]}]
    state = _state(elixir=7.0, opp_elixir=2.0, opp_plays=plays, hand=["goblin-barrel", "knight", "zap", "musketeer"])
    rep = ENGINE.analyse(state)
    assert rep.best.card == "goblin-barrel" and rep.best.intent == "punish"
    assert rep.best.tile == [14.5, 23.5] and rep.best.key == "goblin-barrel|right|enemy_half"
    # the play list order does not matter, the newest play does
    plays = [{"t": 8.0, "card": "knight", "tile": [3, 24]}, {"t": 5.0, "card": "giant", "tile": [14, 25]}]
    assert ENGINE.analyse(_state(elixir=7.0, opp_elixir=2.0, opp_plays=plays,
                                 hand=["goblin-barrel", "knight"])).best.tile[0] == 14.5


def test_punish_needs_elixir_and_falls_back_to_the_weaker_tower():
    base = dict(opp_elixir=2.0, hand=["ram-rider", "knight", "zap", "musketeer"])
    assert ENGINE.analyse(_state(elixir=5.5, **base)).best is None                    # ram-rider 5 + 1 not met
    weak = {"king": None, "left": 4000, "right": 1500}
    assert ENGINE.analyse(_state(elixir=7.0, opp_towers=weak, **base)).best.key == "ram-rider|right|bridge"
    assert ENGINE.analyse(_state(elixir=7.0, **base)).best.key == "ram-rider|left|bridge"        # equal HP: left
    middle = [{"t": 3.0, "card": "knight", "tile": [9, 22]}]                          # no opposite of the middle
    assert ENGINE.analyse(_state(elixir=7.0, opp_plays=middle, opp_towers=weak, **base)).best.key == \
        "ram-rider|right|bridge"


def test_counter_push_backs_an_ally_across_the_middle():
    state = _state(elixir=6.0, units=[_ally((14.5, 10), "knight"), _ally((3.5, 5), "knight")],
                   hand=["hog-rider", "knight", "zap", "musketeer"])
    rep = ENGINE.analyse(state)
    assert rep.best.card == "hog-rider" and rep.best.intent == "counter_push" and rep.best.score == 0.6
    assert rep.best.tile == [14.5, 14.0] and rep.best.key == "hog-rider|right|bridge"
    assert len(rep.moves) == 1                                                    # the left ally is still at row 5
    assert ENGINE.analyse(_state(elixir=4.5, units=state["units"], hand=state["own"]["hand"])).best is None


def test_leak_at_a_full_bar():
    state = _state(elixir=10.0, hand=["hog-rider", "knight", "zap", "musketeer"],
                   opp_towers={"king": None, "left": 3000, "right": 4000})
    rep = ENGINE.analyse(state)
    assert rep.best.card == "hog-rider" and rep.best.intent == "leak" and rep.best.score == 0.5
    assert rep.best.tile == [3.5, 14.0]
    no_wc = ENGINE.analyse(_state(elixir=10.0, hand=["zap", "musketeer", "knight", "tesla"]))
    assert no_wc.best.card == "knight" and no_wc.best.intent == "leak" and no_wc.best.score == 0.4
    assert no_wc.best.tile == [5.0, 2.5] and no_wc.best.key == "knight|left|back"      # left: equal tower HP
    tank = ENGINE.analyse(_state(elixir=10.0, hand=["mega-knight", "zap"], opp_towers={"king": None, "left": 5, "right": 2}))
    assert tank.best.card == "mega-knight" and tank.best.tile == [9.0, 1.0]
    assert ENGINE.analyse(_state(elixir=10.0, hand=["zap", "fireball"])).best is None     # only spells


# 6 -----------------------------------------------------------------------------------------------------------------
def test_no_threat_saves_elixir():
    rep = ENGINE.analyse(_state(elixir=6.0, opp_elixir=6.0))
    assert rep.best is None and rep.moves == [] and rep.pushes == []
    assert rep.hold_reason == "no threat, saving elixir (own 6, opp ~6)"
    assert rep.lines == ["tactics: hold: no threat, saving elixir (own 6, opp ~6)"]


def test_not_enough_elixir_when_the_hand_is_unaffordable():
    rep = ENGINE.analyse(_giant_right(elixir=0.4))
    assert rep.best is None and rep.hold_reason == "not enough elixir"
    assert rep.pushes and rep.lines[-1] == "tactics: hold: not enough elixir"


# 7 -----------------------------------------------------------------------------------------------------------------
def test_unknown_hand_uses_the_deck_fallback():
    state = _giant_right(hand=[None, None, None, None], elixir=10.0)
    deck = state["own"]["deck"]
    rep = ENGINE.analyse(state)
    assert rep.best is not None and {m.card for m in rep.moves} <= set(deck)
    state["own"]["next_card"] = "tesla"                                        # the next card cannot be played
    assert "tesla" not in [m.card for m in ENGINE.analyse(state).moves]


# 8 -----------------------------------------------------------------------------------------------------------------
def test_score_key():
    state = _giant_right()
    rep = ENGINE.analyse(state)
    assert ENGINE.score_key(rep.best.key, state, rep) == rep.best.score
    assert ENGINE.score_key(rep.best.key, state) == rep.best.score                      # report recomputed
    # a hand card that is not proposed (zap has no aim) gets its defence score: 0.5 - 0.6 (spell vs tank) + 0.1 (trade)
    assert "zap" not in [m.card for m in rep.moves]
    assert ENGINE.score_key("zap|left|back", state, rep) == pytest.approx(0.0)
    # a card outside the hand, or no push to defend against
    assert ENGINE.score_key("mega-knight|right|mid", state, rep) is None
    assert ENGINE.score_key("tesla|right|mid", _state()) is None
    # a card that cannot answer an air-only push
    bats = _bats_left()
    assert ENGINE.score_key("knight|left|bridge", bats) is None
    assert ENGINE.score_key("musketeer|right|back", bats) is not None


# 9 -----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("units,hand", [
    ([_enemy((0.0, 2), "giant", eta=2.0), _enemy((0.2, 3), "musketeer", eta=2.5)], ["tesla", "knight", "musketeer", "valkyrie"]),
    ([_enemy((17.9, 14), "giant", eta=7.0, tower="own_right", pred=(17.9, 18))], ["tesla", "knight", "ice-spirit", "minions"]),
    ([_enemy((9, 0.2), "knight", eta=0.5, tower="own_king")], ["cannon", "mini-p-e-k-k-a", "skeletons", "archers"]),
    ([_enemy((14, 12), "skeletons", pred=(14, 4)), _enemy((15, 12), "skeletons"), _enemy((13, 11), "skeletons")],
     ["fireball", "zap", "ice-spirit", "musketeer"]),
    ([_enemy((6, 20), "giant", eta=7.9)], ["tesla", "knight", "musketeer", "baby-dragon"]),
])
def test_tiles_are_clamped_on_our_half_and_keys_match(units, hand):
    rep = ENGINE.analyse(_state(units=units, hand=hand, elixir=9.0))
    assert rep.moves
    for m in rep.moves:
        x, y = m.tile
        assert m.intent == "defend" and 0.5 <= x <= 17.5 and y <= 14.0
        assert x * 2 == int(x * 2) and y * 2 == int(y * 2)                       # multiples of 0.5
        assert m.key == f"{m.card}|{lane_of(m.tile)}|{zone_of(m.tile)}"
        assert len(m.reason) <= 60
        if ROLES[m.card]["type"] != "spell":
            assert y >= 1.0


# 10 ----------------------------------------------------------------------------------------------------------------
def test_analyse_takes_under_5_ms_on_15_units():
    units = [_enemy((3 + i, 8 + (i % 6)), c, eta=3.0 + i % 4, tower="own_left" if i < 6 else "own_right",
                    pred=(3 + i, 5 + (i % 6)))
             for i, c in enumerate(["giant", "musketeer", "bats", "skeletons", "knight", "archers", "minions",
                                    "valkyrie", "hog-rider", "bats"])]
    units += [_ally((5 + i, 9 + i), "knight") for i in range(5)]
    assert len(units) == 15
    state = _state(units=units, hand=["zap", "fireball", "tesla", "mini-p-e-k-k-a"], elixir=8.0)
    ENGINE.analyse(state)
    runs = []
    for _ in range(20):
        t0 = time.perf_counter()
        ENGINE.analyse(state)
        runs.append(time.perf_counter() - t0)
    assert min(runs) < 0.005, f"{min(runs) * 1000:.2f} ms"


# pushes ------------------------------------------------------------------------------------------------------------
def test_push_selection_grouping_and_eta():
    units = [
        _enemy((14, 25), "giant", eta=6.0, tower="own_right"),           # far away but within 8 s: a push
        _enemy((3, 25), "knight", eta=12.0),                             # far and slow: not a push
        _enemy((8.9, 10), "knight"),                                     # middle, col < 9: left, row <= 8? no -> 2.0
        _enemy((9.0, 6), "musketeer"),                                   # middle, col 9: right, row <= 8 -> 0.0
        {**_enemy((4, 12), "knight"), "side": "ally"},                   # not an enemy
        {**_enemy((5, 12), "knight"), "tile": None},                     # no tile: ignored
        _enemy((4, 16), "knight", eta=3.0, tower="own_king"),            # eta to the king tower counts as an own tower
    ]
    pushes = ENGINE.analyse(_state(units=units)).pushes
    by_lane = {p.lane: p for p in pushes}
    assert [p.lane for p in pushes] == ["right", "left"]                  # right has the 0.0 s unit
    assert by_lane["right"].eta_s == 0.0 and [u["card"] for u in by_lane["right"].units] == ["musketeer", "giant"]
    assert by_lane["left"].eta_s == 2.0 and by_lane["left"].units[0]["tile"] == [8.9, 10]
    assert by_lane["right"].elixir == 9.0                                 # giant 5 + musketeer 4
    assert by_lane["right"].summary == "musketeer, giant (9 elixir), tower in 0.0s"


def test_unit_on_our_half_with_a_long_eta_is_a_push_but_leaves_room_for_offence():
    state = _state(units=[_enemy((3, 12), "knight", eta=9.0)], elixir=10.0, hand=["hog-rider", "musketeer", "knight"])
    rep = ENGINE.analyse(state)
    assert rep.pushes[0].eta_s == 9.0
    assert {m.intent for m in rep.moves} == {"defend", "leak"}


def test_needs_in_order():
    units = [_enemy((3, 12), "bats", eta=4.0), _enemy((4, 12), "giant", eta=5.0), _enemy((4, 13), "minions", eta=5.0)]
    push = ENGINE.analyse(_state(units=units)).pushes[0]
    assert push.needs == ["air", "tank", "swarm", "building_targeter"]
    three = [_enemy((3, 12), "skeletons", eta=4.0)] * 3                    # three units of a swarm-class card
    assert ENGINE.analyse(_state(units=three)).pushes[0].needs == ["swarm"]
    two = [_enemy((3, 12), "archers", eta=4.0)] * 2                        # two light units: no swarm
    assert ENGINE.analyse(_state(units=two)).pushes[0].needs == []
    assert "nothing special" in ENGINE.analyse(_state(units=two)).lines[0]


def test_two_pushes_two_lanes_and_at_most_four_lines():
    units = [_enemy((3, 12), "knight", eta=4.0), _enemy((15, 12), "giant", eta=5.0, tower="own_right")]
    rep = ENGINE.analyse(_state(units=units, elixir=9.0))
    assert [p.lane for p in rep.pushes] == ["left", "right"]
    assert 1 <= len(rep.lines) <= 4 and all(ln.startswith("tactics: ") for ln in rep.lines)
    assert [ln.split(":")[1].split()[0] for ln in rep.lines[:2]] == ["push", "push"]
    assert len(rep.moves) <= 6 and len({m.key for m in rep.moves}) == len(rep.moves)
    assert [m.score for m in rep.moves] == sorted((m.score for m in rep.moves), reverse=True)


# defence rules -----------------------------------------------------------------------------------------------------
def test_placement_rules():
    # building vs a non-targeter goes beside the tower; vs a building-targeter it pulls to the centre
    knight_left = _state(units=[_enemy((3, 12), "knight", eta=4.0)], hand=["cannon", "tesla"])
    by_card = {m.card: m for m in ENGINE.analyse(knight_left).moves}
    assert by_card["cannon"].tile == [5.5, 6.5]
    giant_left = _state(units=[_enemy((3, 12), "giant", eta=4.0)], hand=["cannon"])
    assert ENGINE.analyse(giant_left).best.tile == [7.0, 11.0]
    # melee on our side in front; deep in our half just in front of the unit
    deep = _state(units=[_enemy((3, 7), "giant", eta=2.0)], hand=["mini-p-e-k-k-a"])
    assert ENGINE.analyse(deep).best.tile == [3.0, 6.0]
    # ranged behind the tower, swarm on the threat, kamikaze in front, led by 1 s along the prediction
    right = _state(units=[_enemy((15, 12), "knight", eta=4.0, tower="own_right", pred=(15, 8))],
                   hand=["musketeer", "skeletons", "ice-spirit"], elixir=9.0)
    by_card = {m.card: m for m in ENGINE.analyse(right).moves}
    assert by_card["musketeer"].tile == [13.0, 2.5]
    assert by_card["skeletons"].tile == [15.0, 10.0]
    assert by_card["ice-spirit"].tile == [15.0, 7.5]


def test_fireball_on_support_and_tank_penalty_for_spells():
    # a giant with a musketeer and bats behind it: the fireball is worth its cost
    units = [_enemy((3, 12), "giant", eta=4.0), _enemy((3.5, 13), "musketeer", eta=5.0), _enemy((4, 13.5), "bats", eta=5)]
    state = _state(units=units, hand=["fireball", "knight", "zap", "tesla"], elixir=9.0)
    rep = ENGINE.analyse(state)
    fireball = next(m for m in rep.moves if m.card == "fireball")
    assert fireball.reason.startswith("fireball ") and "elixir)" in fireball.reason
    assert fireball.key == f"fireball|{lane_of(fireball.tile)}|{zone_of(fireball.tile)}"
    # a lone giant with a cheap supporter next to it: the spell value (2) < cost 4 and 1 unit: no spell move
    lone = _state(units=[_enemy((3, 12), "giant", eta=4.0), _enemy((3.5, 12.5), "skeletons", eta=5.0)],
                  hand=["fireball", "tesla"])
    assert "fireball" not in [m.card for m in ENGINE.analyse(lone).moves]
    # support worth less than the spell (3 skeletons: swarm rule) vs a tank: the -0.6 tank penalty applies
    swarmy = _state(units=[_enemy((3, 12), "giant", eta=4.0)] + [_enemy((3 + i * 0.3, 12.5), "skeletons", eta=5.0)
                                                                 for i in range(3)], hand=["fireball", "tesla"])
    spell = next(m for m in ENGINE.analyse(swarmy).moves if m.card == "fireball")
    assert spell.score < 0.5 + 0.6                                            # swarm +0.6, tank -0.6, overpay


def test_non_damage_spells_are_never_defence():
    state = _state(units=[_enemy((3, 12), "musketeer", eta=4.0), _enemy((3.5, 12), "archers", eta=4.0)],
                   hand=["rage", "goblin-barrel", "fireball", "knight"], elixir=9.0)
    assert {m.card for m in ENGINE.analyse(state).moves} <= {"fireball", "knight"}


class _FakeKG:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def counter_scores(self, card):
        self.calls.append(card)
        return self.scores


def test_kg_counter_scores_add_up_to_0_2():
    state = _giant_right(hand=["knight", "musketeer"])
    plain = {m.card: m.score for m in ENGINE.analyse(state).moves}
    kg = _FakeKG({"musketeer": 4.0, "knight": 1.0, "zap": 2.0})
    engine = TacticalEngine(CARDS, ROLES, kg=kg, weights=ROLE_WEIGHTS)
    boosted = {m.card: m.score for m in engine.analyse(state).moves}
    assert kg.calls == ["giant"]
    assert boosted["musketeer"] == pytest.approx(plain["musketeer"] + 0.2)
    assert boosted["knight"] == pytest.approx(plain["knight"] + 0.05)
    empty = TacticalEngine(CARDS, ROLES, kg=_FakeKG({}), weights=ROLE_WEIGHTS)
    assert empty.analyse(state).best.card == ENGINE.analyse(state).best.card


# misc --------------------------------------------------------------------------------------------------------------
def test_deterministic_and_does_not_mutate_the_state():
    state = _bats_left()
    before = repr(state)
    a, b = ENGINE.analyse(state), ENGINE.analyse(state, t=12.0)
    assert a == b and repr(state) == before


def test_accepts_the_role_file_layout_and_unknown_cards_count_as_medium_ground():
    wrapped = TacticalEngine(CARDS, {"version": 1, "cards": ROLES}, weights=ROLE_WEIGHTS)
    assert wrapped.analyse(_giant_right()).best.card == "tesla"
    sparse = TacticalEngine(CARDS, {}, weights=ROLE_WEIGHTS)      # no roles at all: everything is a medium unit
    rep = sparse.analyse(_state(units=[_enemy((3, 12), "giant", eta=4.0)], hand=["knight", "zap"]))
    assert rep.pushes[0].needs == [] and rep.best.card == "knight"
    unnamed = _state(units=[_enemy((3, 12), None, eta=4.0, unit_class="skeleton")], hand=["knight"])
    assert ENGINE.analyse(unnamed).pushes[0].summary == "skeleton (0 elixir), tower in 4.0s"


def test_real_role_table():
    engine = TacticalEngine(CARDS, weights=ROLE_WEIGHTS)                     # roles=None: load_roles()
    rep = engine.analyse(_giant_right(hand=["tesla", "knight", "zap", "musketeer"]))
    assert rep.best is not None and rep.pushes[0].lane == "right" and "tank" in rep.pushes[0].needs
    # card_roles.hits_air: the-log carries "hits_air": false
    bats = engine.analyse(_state(units=[_enemy(t, "bats", eta=6.0) for t in ((3, 14), (4, 14), (3.5, 15))],
                                 hand=["the-log", "zap", "knight", "musketeer"]))
    assert {m.card for m in bats.moves} == {"zap", "musketeer"}


# follow-up fixes (real pro states) ----------------------------------------------------------------------------------
def _tokens(*tiles):
    return [{"unit_class": "skeleton", "card": None, "side": "enemy", "tile": list(t), "eta_tower": None}
            for t in tiles]


# fix 1: spell objects as push units
def test_graveyard_is_a_swarm_source_and_other_spell_units_are_ignored():
    grave = _enemy((4, 6), "graveyard")
    state = _state(units=[grave] + _tokens((4, 6), (4.5, 6.5), (3.5, 6.5)),
                   hand=["berserker", "knight", "zap", "musketeer"])
    rep = ENGINE.analyse(state)
    push = rep.pushes[0]
    assert push.needs == ["swarm"] and push.elixir == 5.0 and grave in push.units      # its value is its cost
    assert rep.best.card == "zap" and rep.best.intent == "defend"
    scores = {m.card: m.score for m in rep.moves}
    assert scores["musketeer"] > scores["berserker"] and scores["musketeer"] > scores["knight"]   # melee: -0.4
    # a fireball or rage object on the field is an effect, not a push
    effects = _state(units=[_enemy((4, 6), "fireball"), _enemy((9, 5), "rage", eta=2.0)])
    rep = ENGINE.analyse(effects)
    assert rep.pushes == [] and rep.hold_reason.startswith("no threat")
    mixed = ENGINE.analyse(_state(units=[_enemy((4, 6), "fireball"), _enemy((3, 12), "knight", eta=4.0)]))
    assert [u["card"] for u in mixed.pushes[0].units] == ["knight"] and mixed.pushes[0].needs == []


# fix 2: building-only troops never defend
def test_building_only_troops_never_defend_but_stay_available_for_offence():
    hand = ["wall-breakers", "hog-rider", "giant", "knight"]
    state = _state(units=[_enemy((3, 10), "knight", eta=4.0)], hand=hand, elixir=10.0)
    rep = ENGINE.analyse(state)
    assert {m.card for m in rep.moves if m.intent == "defend"} == {"knight"}
    assert ENGINE.score_key("hog-rider|left|mid", state, rep) is None
    assert ENGINE.score_key("wall-breakers|left|mid", state) is None
    # no push at a full bar: they are the leak / counter-push cards
    leak = ENGINE.analyse(_state(elixir=10.0, hand=hand))
    assert leak.best.intent == "leak" and leak.best.card in {"wall-breakers", "hog-rider", "giant"}
    ally = ENGINE.analyse(_state(elixir=6.0, units=[_ally((14.5, 10))], hand=["wall-breakers", "knight"]))
    assert ally.best.card == "wall-breakers" and ally.best.intent == "counter_push"


# fix 3: tanks are not ranged support
def test_tanks_use_the_front_rule_whatever_their_range():
    state = _state(units=[_enemy((3, 10), "musketeer", eta=4.0)], hand=["goblinstein", "musketeer"], elixir=9.0)
    by_card = {m.card: m for m in ENGINE.analyse(state).moves}
    assert by_card["goblinstein"].tile == [3.0, 7.5]                           # meets it in front, not behind the tower
    assert by_card["goblinstein"].reason.startswith("tanks for the tower vs musketeer")
    assert by_card["musketeer"].tile == [5.0, 2.5]                             # a plain ranged troop stays behind
    deep = _state(units=[_enemy((3, 7), "knight", eta=2.0)], hand=["goblinstein"], elixir=9.0)
    assert ENGINE.analyse(deep).best.tile == [3.0, 6.0]
    # kamikaze and swarm roles still come first
    spirit = _state(units=[_enemy((3, 10), "musketeer", eta=4.0)], hand=["ice-spirit"])
    assert ENGINE.analyse(spirit).best.tile == [3.0, 7.5]


# fix 4: the spell aim is asked for the cell the threat stands in now
def test_spell_aim_uses_the_current_cell_of_the_threat():
    ram = _enemy((5, 14), "battle-ram", eta=2.0, pred=(5, 8))                  # bridge now, mid after the 1 s lead
    state = _state(units=[ram], hand=["tornado", "royal-delivery"], elixir=7.0)
    rep = ENGINE.analyse(state)
    assert rep.pushes[0].needs == ["building_targeter"]
    assert rep.best is not None and rep.best.card == "royal-delivery", rep.lines          # value 4 >= cost 3
    assert rep.best.key == "royal-delivery|left|bridge" and rep.best.tile == [5.0, 13.0]
    assert "tornado" not in [m.card for m in rep.moves]                         # utility spell
    assert ENGINE.score_key("royal-delivery|left|bridge", state, rep) == rep.best.score


def test_spell_aim_falls_back_to_the_lead_cell():
    # the giant is not a spell target; its own cell (left, mid) holds nothing, the musketeer ahead is in (left, bridge)
    units = [_enemy((3, 10), "giant", eta=3.0, pred=(3, 14)), _enemy((3, 12.5), "musketeer", eta=4.0)]
    rep = ENGINE.analyse(_state(units=units, hand=["fireball", "tesla"], elixir=9.0))
    fireball = next(m for m in rep.moves if m.card == "fireball")
    assert fireball.tile == [3.0, 12.5] and fireball.key == "fireball|left|bridge"
    # the current cell wins when it has a target; here nothing is worth a spell anywhere
    lone = _state(units=[_enemy((3, 10), "giant", eta=3.0, pred=(3, 14))], hand=["fireball"], elixir=9.0)
    assert ENGINE.analyse(lone).best is None


def test_four_follow_up_situations_on_the_real_role_table():
    engine = TacticalEngine(CARDS, weights=ROLE_WEIGHTS)
    grave = engine.analyse(_state(units=[_enemy((4, 6), "graveyard")], hand=["berserker", "knight", "zap", "musketeer"]))
    assert grave.pushes[0].needs == ["swarm"]
    wall = engine.analyse(_state(units=[_enemy((3, 10), "knight", eta=4.0)],
                                 hand=["wall-breakers", "hog-rider", "giant", "musketeer"]))
    assert {m.card for m in wall.moves} == {"musketeer"}
    stein = engine.analyse(_state(units=[_enemy((3, 10), "musketeer", eta=4.0)], hand=["goblinstein", "knight"],
                                  elixir=9.0))
    assert next(m for m in stein.moves if m.card == "goblinstein").tile == [3.0, 7.5]
    ram = engine.analyse(_state(units=[_enemy((5, 14), "battle-ram", eta=2.0, pred=(5, 8))],
                                hand=["tornado", "royal-delivery"], elixir=7.0))
    assert ram.best.card == "royal-delivery", ram.lines


# =====================================================================================================================
# MAC-A2: offence intents for the bait deck (bait, pressure, support, cycle), the weights, and the tuner
# =====================================================================================================================
def _intents(rep):
    return {m.card: m for m in rep.moves}


# bait --------------------------------------------------------------------------------------------------------------
def test_bait_after_their_log_was_played_3s_ago():
    plays = [{"t": 7.0, "card": "the-log", "tile": [14, 20]}]
    state = _state(elixir=6.0, hand=["goblin-barrel", "knight", "musketeer", "zap"], opp_plays=plays,
                   deck_known=["the-log", "knight"])
    rep = ENGINE.analyse(state, 10.0)
    best = rep.best
    assert best.card == "goblin-barrel" and best.intent == "bait" and best.score == 0.6
    assert best.reason == "their the-log is cycling"
    # the lane opposite their newest play (the log landed in the right lane), aimed at the enemy princess tower
    assert best.tile == [3.5, 23.5] and best.key == "goblin-barrel|left|enemy_half" and best.trade is None
    assert rep.lines[0] == "tactics: best goblin-barrel @[3.5,23.5] bait: their the-log is cycling"
    assert ENGINE.analyse(state).best.score == 0.6                        # no clock: the newest play stands for now
    # the lower enemy princess tower wins over the opposite lane
    weak = {"king": None, "left": 4000, "right": 1200}
    assert ENGINE.analyse(dict(state, opponent=dict(state["opponent"], towers=weak)), 10.0).best.key == \
        "goblin-barrel|right|enemy_half"


def test_bait_is_not_preferred_when_their_zap_is_ready():
    plays = [{"t": 18.0, "card": "zap", "tile": [14, 20]}]            # 12 s ago: back in their hand
    state = _state(elixir=8.0, hand=["goblin-barrel", "ice-spirit", "musketeer", "knight"], opp_plays=plays,
                   deck_known=["zap", "knight"])
    rep = ENGINE.analyse(state, 30.0)
    bait = _intents(rep)["goblin-barrel"]
    assert bait.intent == "bait" and bait.score == 0.25 and bait.reason == "their zap may be ready"
    assert rep.best.card == "ice-spirit" and rep.best.intent == "cycle"             # the cycle (0.35) beats the bait
    # played 3 s ago instead: the bait is on, and it blocks the cycle
    recent = _state(elixir=8.0, hand=state["own"]["hand"], opp_plays=[{"t": 27.0, "card": "zap", "tile": [14, 20]}],
                    deck_known=["zap"])
    rep = ENGINE.analyse(recent, 30.0)
    assert rep.best.card == "goblin-barrel" and rep.best.score == 0.6 and "cycle" not in {m.intent for m in rep.moves}
    # two small spells known, one of them still cycling: the other may be in hand
    both = _state(elixir=6.0, hand=["goblin-barrel", "knight"], deck_known=["zap", "the-log"],
                  opp_plays=[{"t": 27.0, "card": "zap", "tile": [14, 20]}])
    assert ENGINE.analyse(both, 30.0).best.score == 0.25


def test_bait_with_no_known_small_spell_and_its_elixir_condition():
    state = _state(elixir=6.0, hand=["goblin-barrel", "knight"], deck_known=["knight", "giant"])
    bait = ENGINE.analyse(state, 10.0).best
    assert bait.card == "goblin-barrel" and bait.score == 0.45 and bait.reason == "no small spell of theirs seen"
    assert ENGINE.analyse(_state(elixir=3.5, hand=["goblin-barrel", "knight"]), 10.0).best is None   # needs cost + 1
    # graveyard is a bait too (5 elixir + 1); the miner-likes go on the tower as well
    assert ENGINE.analyse(_state(elixir=6.0, hand=["graveyard", "knight"])).best.intent == "bait"
    assert ENGINE.analyse(_state(elixir=5.5, hand=["graveyard", "knight"])).best is None
    miner = ENGINE.analyse(_state(elixir=6.0, hand=["miner", "knight"])).best
    assert miner.card == "miner" and miner.intent == "bait" and miner.tile[1] == 23.5
    assert all(len(m.reason) <= 60 for m in ENGINE.analyse(state, 10.0).moves)


def test_bait_needs_no_push_in_sight():
    state = _giant_right(hand=["goblin-barrel", "tesla", "knight", "musketeer"], elixir=7.0)
    assert "bait" not in {m.intent for m in ENGINE.analyse(state, 10.0).moves}


# pressure ----------------------------------------------------------------------------------------------------------
def test_pressure_needs_the_reserve_and_a_low_opponent():
    weak = {"king": None, "left": 3000, "right": 4000}
    hand = ["hog-rider", "knight", "zap", "musketeer"]
    rep = ENGINE.analyse(_state(elixir=7.0, hand=hand, opp_towers=weak))
    best = rep.best
    assert best.card == "hog-rider" and best.intent == "pressure" and best.score == 0.45
    assert best.tile == [3.5, 14.0] and best.key == "hog-rider|left|bridge" and len(best.reason) <= 60
    assert ENGINE.analyse(_state(elixir=6.5, hand=hand, opp_towers=weak)).best is None        # cost 4 + reserve 3
    assert ENGINE.analyse(_state(elixir=9.0, hand=hand, opp_elixir=6.5)).best is None          # they are not low
    assert ENGINE.analyse(_state(elixir=7.0, hand=hand, opp_elixir=None)).best.intent == "pressure"
    # a goblin barrel is a bait, never a pressure move
    barrel = ENGINE.analyse(_state(elixir=9.0, hand=["goblin-barrel", "knight"]))
    assert {m.intent for m in barrel.moves} == {"bait"}


# support -----------------------------------------------------------------------------------------------------------
def test_support_behind_an_ally_hog_on_the_enemy_half():
    ally = _ally((3.5, 20), "hog-rider")
    state = _state(elixir=6.0, hand=["musketeer", "knight", "tesla", "zap"], units=[ally])
    rep = ENGINE.analyse(state)
    support = _intents(rep)["musketeer"]
    assert support.intent == "support" and support.score == 0.5 and support.trade is None
    assert support.tile == [3.5, 14.0] and support.key == "musketeer|left|bridge"        # 2 behind (y 18), clamped to our half
    assert "hog-rider" in support.reason and len(support.reason) <= 60
    assert "knight" not in _intents(rep) and "tesla" not in _intents(rep)                  # melee troops and buildings do not
    assert "zap" not in _intents(rep)                                                      # nothing to zap
    # an ally at the bridge: 2 tiles behind it, not clamped
    bridge = ENGINE.analyse(_state(elixir=6.0, hand=["musketeer", "knight"], units=[_ally((14.5, 13), "giant")]))
    assert bridge.best.tile == [14.5, 11.0] and bridge.best.key == "musketeer|right|mid"
    # a small damage spell on the defenders of the ally
    defenders = [_enemy(t, "skeletons") for t in ((3, 21), (4, 21), (3.5, 22))]
    rep = ENGINE.analyse(_state(elixir=6.0, hand=["zap", "knight"], units=[ally] + defenders))
    zap = _intents(rep)["zap"]
    assert zap.intent == "support" and zap.key.startswith("zap|left|enemy_half") and 21 <= zap.tile[1] <= 22
    # no support without elixir, for an ally still on its way, or for a cheap ally that is no win condition / tank
    assert ENGINE.analyse(_state(elixir=3.5, hand=["musketeer"], units=[ally])).best is None
    assert ENGINE.analyse(_state(elixir=6.0, hand=["musketeer"], units=[_ally((3.5, 10), "hog-rider")])).best is None
    assert ENGINE.analyse(_state(elixir=6.0, hand=["musketeer"], units=[_ally((3.5, 20), "knight")])).best is None


# cycle -------------------------------------------------------------------------------------------------------------
def test_cycle_with_spirits_chips_the_weaker_tower():
    weak = {"king": None, "left": 4000, "right": 2500}
    rep = ENGINE.analyse(_state(elixir=8.0, hand=["ice-spirit", "knight", "zap", "musketeer"], opp_towers=weak))
    best = rep.best
    assert best.card == "ice-spirit" and best.intent == "cycle" and best.score == 0.35
    assert best.tile == [14.5, 14.0] and best.key == "ice-spirit|right|bridge" and len(best.reason) <= 60
    # the other conditions: elixir >= 7, a card of cost <= 2, no other move scoring >= 0.4
    assert ENGINE.analyse(_state(elixir=6.5, hand=["ice-spirit", "knight"], opp_towers=weak)).best is None
    assert ENGINE.analyse(_state(elixir=8.0, hand=["knight", "musketeer", "zap", "tesla"])).best is None
    blocked = ENGINE.analyse(_state(elixir=8.0, hand=["ice-spirit", "goblin-barrel"]))      # bait (unknown) = 0.45
    assert blocked.best.card == "goblin-barrel" and "cycle" not in {m.intent for m in blocked.moves}
    # the cheapest non-spell, non-win-condition card is cycled: fire-spirit (1) before ice-spirit (1) by name
    assert ENGINE.analyse(_state(elixir=8.0, hand=["ice-spirit", "fire-spirit", "zap", "hog-rider"],
                                 opp_elixir=6.0)).best.card == "fire-spirit"


def test_cycle_a_non_kamikaze_card_goes_back_in_the_quiet_lane():
    units = [_enemy((3, 20), "knight"), _enemy((4, 21), "knight")]                        # enemy units on the left
    rep = ENGINE.analyse(_state(elixir=8.0, hand=["skeletons", "knight", "zap", "musketeer"], units=units))
    best = rep.best
    assert best.card == "skeletons" and best.intent == "cycle"
    assert best.tile == [13.0, 1.5] and best.key == "skeletons|right|back"
    quiet_left = ENGINE.analyse(_state(elixir=8.0, hand=["skeletons", "knight"], units=[_enemy((15, 20), "knight")]))
    assert quiet_left.best.tile == [5.0, 1.5] and quiet_left.best.key == "skeletons|left|back"


def test_new_intents_are_listed_in_the_report_lines_and_scores_come_from_the_weights():
    state = _state(elixir=7.0, hand=["hog-rider", "knight", "zap", "musketeer"])
    boosted = TacticalEngine(CARDS, ROLES, weights={**ROLE_WEIGHTS, "pressure": 0.9})
    assert boosted.analyse(state).best.score == 0.9 and ENGINE.analyse(state).best.score == 0.45
    assert ENGINE.analyse(state).lines[0].startswith("tactics: best hog-rider @[3.5,14] pressure: ")


# defence fixes from live match B (balloon vs ice-spirit): stopping power, already-answered pushes, kamikaze repeats -----
def _balloon(**kw):
    kw.setdefault("hand", ["ice-spirit", "magic-archer", "bomb-tower", "berserker"])
    return _state(units=[_enemy((14.5, 12), "balloon", eta=5.0, tower="own_right")], **kw)


def test_a_balloon_is_not_answered_by_an_ice_spirit():
    for engine in (ENGINE, TacticalEngine(CARDS, weights=ROLE_WEIGHTS)):          # inline roles and the real table
        rep = engine.analyse(_balloon(elixir=7.0), 10.0)
        assert rep.pushes[0].needs == ["air", "building_targeter"] and rep.pushes[0].elixir == 5.0
        assert rep.best.card in ("magic-archer", "bomb-tower") and rep.best.intent == "defend"
        assert rep.best.reason.endswith("trade +1")                              # a medium-stopping card keeps its trade
        spirit = next((m for m in rep.moves if m.card == "ice-spirit"), None)
        assert spirit is not None and spirit.score < rep.best.score
        assert spirit.reason == "slows balloon" and spirit.trade == 4.0           # no trade claimed for a low-stopping card
        assert "ice-spirit" not in rep.lines[1]                                   # the best line names the real answer
    # the stopping terms: building / tank_killer high, medium dps, kamikaze low (vs a push with a medium unit)
    knight_push = _state(units=[_enemy((3, 10), "knight", eta=4.0)], elixir=9.0,
                         hand=["tesla", "mini-p-e-k-k-a", "musketeer", "ice-spirit"])
    scores = {m.card: m.score for m in ENGINE.analyse(knight_push).moves}
    assert scores["tesla"] == pytest.approx(0.5 + 0.3) and scores["mini-p-e-k-k-a"] == pytest.approx(0.5 + 0.3)
    assert scores["musketeer"] == pytest.approx(0.5 + 0.15) and scores["ice-spirit"] == pytest.approx(0.5 - 0.35)
    # and they are weights
    flat = TacticalEngine(CARDS, ROLES, weights={**ROLE_WEIGHTS, "stop_low": 0.0, "stop_medium": 0.0})
    flat_scores = {m.card: m.score for m in flat.analyse(knight_push).moves}
    assert flat_scores["ice-spirit"] == pytest.approx(0.5) and flat_scores["musketeer"] == pytest.approx(0.5)


def test_ice_spirit_against_bats_is_still_a_fine_answer():
    state = _state(units=[_enemy(t, "bats", eta=6.0) for t in ((3, 14), (4, 14), (3.5, 15))],
                   hand=["ice-spirit", "knight", "musketeer", "tesla"], elixir=7.0)
    rep = ENGINE.analyse(state)
    assert rep.pushes[0].needs == ["air", "swarm"]
    assert rep.best.card == "ice-spirit" and rep.best.reason.endswith("trade +1")   # swarm-only push: the trade counts
    assert rep.best.score == pytest.approx(0.5 + 0.6 + 0.1)                         # no stopping term: nothing medium in it


def test_a_push_our_units_are_already_fighting_is_scaled_down_and_marked():
    giant = _enemy((15, 13), "giant", eta=8.0, tower="own_right")
    base = {m.card: m.score for m in ENGINE.analyse(_state(units=[giant], elixir=9.0)).moves}
    allies = [_ally((15, 11), "bomb-tower"), _ally((14, 12), "berserker")]           # 6 elixir vs the giant's 5
    rep = ENGINE.analyse(_state(units=[giant] + allies, elixir=9.0))
    assert {m.card: m.score for m in rep.moves} == {c: pytest.approx(round(v * 0.3, 3)) for c, v in base.items()}
    assert rep.lines[0] == ("tactics: push right: giant (5 elixir), tower in 8.0s; needs tank, building_targeter"
                            " (being answered by bomb-tower, berserker)")
    assert rep.best.card == "tesla"                                                 # the order of the answers is kept
    # an ally 4 tiles away, or allies worth less than 0.6 x the push, do not count (a building does: see MAC-AD below)
    far = ENGINE.analyse(_state(units=[giant, _ally((15, 9), "knight")], elixir=9.0))
    assert "being answered" not in " ".join(far.lines) and far.best.score == pytest.approx(max(base.values()))
    weak = ENGINE.analyse(_state(units=[giant, _ally((15, 12), "skeletons")], elixir=9.0))     # 1 < 0.6 x 5
    assert "being answered" not in " ".join(weak.lines) and weak.best.score == pytest.approx(max(base.values()))
    # the factor and the fraction are weights
    mild = TacticalEngine(CARDS, ROLES, weights={**ROLE_WEIGHTS, "answered_factor": 0.5, "answered_frac": 1.5})
    assert "being answered" not in " ".join(mild.analyse(_state(units=[giant] + allies, elixir=9.0)).lines)   # 6 < 1.5 x 5
    half = TacticalEngine(CARDS, ROLES, weights={**ROLE_WEIGHTS, "answered_factor": 0.5})
    assert half.analyse(_state(units=[giant] + allies, elixir=9.0)).best.score == pytest.approx(round(base["tesla"] * 0.5, 3))


def test_an_emergency_at_the_tower_with_weak_allies_is_not_scaled_down():
    giant = _enemy((15, 13), "giant", eta=1.0, tower="own_right")
    base = ENGINE.analyse(_state(units=[giant], elixir=9.0))
    some = [_ally((15, 12), "knight")]                                              # 3 elixir: >= 0.6 x 5 but < 5
    rep = ENGINE.analyse(_state(units=[giant] + some, elixir=9.0))
    assert [m.score for m in rep.moves] == [m.score for m in base.moves] and "being answered" not in " ".join(rep.lines)
    strong = [_ally((15, 12), "knight"), _ally((14, 12), "bomb-tower")]            # 7 elixir >= 5: it is answered
    rep = ENGINE.analyse(_state(units=[giant] + strong, elixir=9.0))
    assert rep.best.score == pytest.approx(round(base.best.score * 0.3, 3)) and "answered by knight, bomb-tower" in rep.lines[0]
    late = _enemy((15, 13), "giant", eta=2.0, tower="own_right")                    # beyond 1.5 s: the weak allies count
    assert "answered by knight" in ENGINE.analyse(_state(units=[late] + some, elixir=9.0)).lines[0]


def test_a_kamikaze_card_we_just_played_is_not_proposed_again():
    bats = [_enemy(t, "bats", eta=6.0) for t in ((3, 14), (4, 14), (3.5, 15))]
    spirit = {"t": 8.0, "player": "own", "card": "ice-spirit", "tile": [3, 12]}
    state = _state(units=bats, hand=["ice-spirit", "zap", "knight", "musketeer"], elixir=8.0)
    state["own"]["recent_plays"] = [spirit]
    assert ENGINE.analyse(state, 10.0).best.card != "ice-spirit"                    # played 2 s ago
    assert "ice-spirit" not in [m.card for m in ENGINE.analyse(state, 10.9).moves]
    assert "ice-spirit" in [m.card for m in ENGINE.analyse(state, 11.5).moves]      # 3.5 s ago: fine again
    assert "ice-spirit" in [m.card for m in ENGINE.analyse(state).moves]            # no clock: the rule is off
    state["own"]["recent_plays"] = [dict(spirit, card="knight")]                     # a non-kamikaze play does not block
    assert "ice-spirit" in [m.card for m in ENGINE.analyse(state, 10.0).moves]
    state["own"]["recent_plays"] = [dict(spirit, t=None)]
    assert "ice-spirit" in [m.card for m in ENGINE.analyse(state, 10.0).moves]


# weights -----------------------------------------------------------------------------------------------------------
def test_default_weights_cover_every_scoring_number_and_the_committed_file_is_valid():
    import json

    from clashbot.tactics import WEIGHTS_PATH
    for key in ("base", "kg_factor", "win_condition_penalty", "counter_push", "punish", "leak", "leak_wc",
                "bait_counter_used", "bait_unknown", "bait_counter_ready", "pressure", "support", "cycle",
                "stop_high", "stop_medium", "stop_low", "answered_frac", "answered_factor"):
        assert key in DEFAULT_WEIGHTS
    assert (DEFAULT_WEIGHTS["bait_counter_used"], DEFAULT_WEIGHTS["bait_unknown"],
            DEFAULT_WEIGHTS["bait_counter_ready"]) == (0.6, 0.45, 0.25)
    assert (DEFAULT_WEIGHTS["pressure"], DEFAULT_WEIGHTS["support"], DEFAULT_WEIGHTS["cycle"]) == (0.45, 0.5, 0.35)
    assert (DEFAULT_WEIGHTS["stop_high"], DEFAULT_WEIGHTS["stop_medium"], DEFAULT_WEIGHTS["stop_low"]) == (0.3, 0.15, -0.35)
    assert (DEFAULT_WEIGHTS["answered_frac"], DEFAULT_WEIGHTS["answered_factor"]) == (0.6, 0.3)
    assert all(isinstance(v, float) for v in DEFAULT_WEIGHTS.values())
    data = json.loads(WEIGHTS_PATH.read_text())                               # committed by tools/train/tune_tactics.py
    # the file may lack a weight added after the last tuning run (it falls back to the default): MAC-AG `split`, and the
    # MAC-AH threat-model weights (the tuner does not know them yet, see the report)
    added_after_tuning = {"split", "threat_model", "prevent", "role_scale", "tower_ok_damage", "adequate_frac"}
    assert set(data["weights"]) <= set(DEFAULT_WEIGHTS) and set(DEFAULT_WEIGHTS) - set(data["weights"]) <= added_after_tuning
    assert all(isinstance(v, float) for v in data["weights"].values())


def test_the_weights_file_is_loaded_and_an_explicit_dict_overrides_it(tmp_path, monkeypatch):
    import json

    from clashbot import tactics
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"version": 1, "weights": {"pressure": 0.9, "support": 0.1, "no_such_weight": 5}}))
    monkeypatch.setattr(tactics, "WEIGHTS_PATH", path)
    state = _state(elixir=7.0, hand=["hog-rider", "knight", "zap", "musketeer"])
    from_file = TacticalEngine(CARDS, ROLES)
    assert from_file.w["pressure"] == 0.9 and from_file.w["support"] == 0.1 and "no_such_weight" not in from_file.w
    assert from_file.w["cycle"] == DEFAULT_WEIGHTS["cycle"]                      # untouched keys stay at the default
    assert from_file.analyse(state).best.score == 0.9
    explicit = TacticalEngine(CARDS, ROLES, weights={"pressure": 0.2})           # explicit beats the file, file beats defaults
    assert explicit.analyse(state).best.score == 0.2 and explicit.w["support"] == 0.1
    assert TacticalEngine(CARDS, ROLES, weights=DEFAULT_WEIGHTS).w == DEFAULT_WEIGHTS
    with pytest.raises(ValueError, match="no_such_weight"):
        TacticalEngine(CARDS, ROLES, weights={"no_such_weight": 1.0})
    path.write_text(json.dumps({"pressure": 0.8}))                               # a plain mapping is accepted too
    assert TacticalEngine(CARDS, ROLES).w["pressure"] == 0.8
    path.write_text("not json")                                                  # a broken or missing file = defaults
    assert TacticalEngine(CARDS, ROLES).w == DEFAULT_WEIGHTS
    path.unlink()
    assert TacticalEngine(CARDS, ROLES).w == DEFAULT_WEIGHTS


def test_the_defaults_score_as_written():
    """The Z2 numbers plus the stopping term: the weights refactor must not move any other defence or offence score."""
    rep = ENGINE.analyse(_giant_right())
    # Z2 had tesla 1.6, knight 0.9, musketeer 0.6; the giant is a tank, so stop_high / stop_medium / stop_medium now apply
    assert {m.card: m.score for m in rep.moves} == {"tesla": 1.9, "knight": 1.05, "musketeer": 0.75}
    bats = ENGINE.analyse(_bats_left())                                    # a swarm-only push: unchanged since Z2
    assert {m.card: m.score for m in bats.moves} == {"zap": 1.1, "fireball": 0.95, "musketeer": 0.35}
    assert DEFAULT_WEIGHTS["counter_push"] == 0.6 and DEFAULT_WEIGHTS["punish"] == 0.55
    assert DEFAULT_WEIGHTS["leak_wc"] == 0.5 and DEFAULT_WEIGHTS["leak"] == 0.4


# the tuner ---------------------------------------------------------------------------------------------------------
def _tuning_fixtures():
    """A knight push where the KG favours the knight (0.85) over the tesla (0.8): a pro tesla needs the KG factor down."""
    from tools.train import tune_tactics as tt
    kg = _FakeKG({"knight": 1.0})
    engine = TacticalEngine(CARDS, ROLES, kg=kg, weights=DEFAULT_WEIGHTS)

    def state():
        return _state(units=[_enemy((3, 10), "knight", eta=4.0)], hand=["knight", "zap", "musketeer", "tesla"])

    moves = {m.card: m for m in engine.analyse(state()).moves}
    assert engine.analyse(state()).best.card == "knight" and "tesla" in moves
    lane = moves["tesla"].key.split("|")[1]

    def recs(split, card, n, lane=lane):
        return [tt.Rec(f"{split}{i}", split, state(), 10.0, card, lane, 4, True) for i in range(n)]

    return tt, engine, recs


def test_the_tuner_improves_its_objective_on_a_tiny_record_set():
    tt, engine, recs = _tuning_fixtures()
    train, val = recs("train", "tesla", 6), recs("val", "tesla", 3)
    base = tt.mean_objective(engine, train, DEFAULT_WEIGHTS)
    assert base == pytest.approx(0.5)                                                           # top-3 only
    out = tt.tune(engine, train, val, passes=2, log=lambda *a: None)
    assert out["tuned"] and out["picked_pass"] >= 1 and out["weights"] != DEFAULT_WEIGHTS
    assert out["history"][0]["train"] == pytest.approx(base) and out["history"][1]["train"] > base
    assert tt.mean_objective(engine, train, out["weights"]) > base                              # +1 top-1 (+0.25 lane)
    assert tt.mean_objective(engine, val, out["weights"]) > tt.mean_objective(engine, val, DEFAULT_WEIGHTS)
    assert engine.w == DEFAULT_WEIGHTS                                                          # the engine is left as it was


def test_the_tuner_keeps_the_defaults_when_val_does_not_improve():
    tt, engine, recs = _tuning_fixtures()
    train = recs("train", "tesla", 6)
    best = engine.analyse(_state(units=[_enemy((3, 10), "knight", eta=4.0)],
                                 hand=["knight", "zap", "musketeer", "tesla"])).best
    worse = tt.tune(engine, train, recs("val", best.card, 3, lane=best.key.split("|")[1]), passes=2,
                    log=lambda *a: None)                                                        # val wants the default best
    assert not worse["tuned"] and worse["weights"] == DEFAULT_WEIGHTS
    assert "did not improve" in worse["reason"] and worse["picked_pass"] == 0
    equal = tt.tune(engine, train, recs("val", "musketeer", 3), passes=1, log=lambda *a: None)   # val unaffected = no gain
    assert not equal["tuned"] and equal["weights"] == DEFAULT_WEIGHTS


def test_the_tuner_vetoes_changes_that_break_a_design_guard():
    tt, engine, recs = _tuning_fixtures()
    train, val = recs("train", "tesla", 6), recs("val", "tesla", 3)
    free = tt.tune(engine, train, val, passes=2, log=lambda *a: None)
    assert free["weights"]["kg_factor"] < 0.2 and free["vetoed"] == {}
    kept = tt.tune(engine, train, val, passes=2, log=lambda *a: None, guards=[lambda e: e.w["kg_factor"] >= 0.2])
    assert kept["weights"]["kg_factor"] == 0.2 and kept["vetoed"]["kg_factor"] >= 1
    with pytest.raises(RuntimeError, match="default weights break"):
        tt.tune(engine, train, val, passes=1, log=lambda *a: None, guards=[lambda e: False])


def test_the_default_design_guards_hold_for_the_defaults_and_catch_regressions():
    from tools.train import tune_tactics as tt
    # MAC-AH: the guards (MAC-AD design statements about the role terms, e.g. "ice-spirit still answers bats") were
    # written before the threat model; with it, 3 bats are a push the tower handles. They are checked on the role-only
    # configuration until tools/train/tune_tactics.py gets simulation-aware guards (Opus).
    engine = TacticalEngine(CARDS, weights=ROLE_WEIGHTS)                          # the real role table
    guards = tt.default_guards()
    assert tt.guards_hold(engine, ROLE_WEIGHTS, guards) and engine.w == ROLE_WEIGHTS
    broken = {
        "a balloon answered by an ice-spirit": {"targeter_swarm_bonus": 0.9},
        "an answered push not scaled down": {"answered_factor": 1.0},
        "bait unknown above bait cycling": {"bait_unknown": 0.9},
        "a positive penalty for low stopping power": {"stop_low": 0.2},
        "cycle above the other intents": {"cycle": 0.7, "cycle_block_score": 0.7},
    }
    for what, change in broken.items():
        assert not tt.guards_hold(engine, {**ROLE_WEIGHTS, **change}, guards), what
    assert tt.guards_hold(engine, {**ROLE_WEIGHTS, "support": 0.8, "kg_factor": 0.1}, guards)


def test_the_incremental_objective_matches_a_full_recompute_and_candidates():
    tt, engine, recs = _tuning_fixtures()
    rows = recs("train", "tesla", 4)
    obj = tt.Objective(engine, rows, dict(DEFAULT_WEIGHTS))
    assert obj.touch["kg_factor"] == {0, 1, 2, 3} and obj.touch["bait_unknown"] == set()   # only read weights are touched
    total, updates = obj.propose("kg_factor", 0.05)
    assert total / 4 == pytest.approx(tt.mean_objective(engine, rows, {**DEFAULT_WEIGHTS, "kg_factor": 0.05}))
    assert dict.__getitem__(obj.tracer, "kg_factor") == 0.2                                      # propose restores the value
    obj.accept("kg_factor", 0.05, total, updates)
    assert obj.total == total and obj.weights["kg_factor"] == 0.05
    assert tt.candidates(0.4) == [0.3, 0.5, 0.2, 0.6, 0.8]                                       # smallest step first
    assert tt.candidates(0.1) == [0.075, 0.125, 0.05, 0.15, 0.0, 0.2]                            # near 0: +-0.1, sign kept
    assert 0.0 in tt.candidates(0.1) and min(tt.candidates(0.1)) >= 0
    assert 0.1 in tt.candidates(0.0) and -0.1 in tt.candidates(0.0)
    assert min(tt.candidates(-0.35)) >= -0.7 and max(tt.candidates(-0.35)) <= -0.175          # negative weights stay negative


def test_the_tuner_main_writes_the_weights_and_the_report(tmp_path, monkeypatch):
    import json

    from tools.train import tune_tactics as tt
    monkeypatch.setattr(tt, "DEFAULT_WEIGHTS", ROLE_WEIGHTS)       # the guards predate the threat model (see above)
    tt_state = _state(units=[_enemy((3, 10), "knight", eta=4.0)], hand=["knight", "zap", "musketeer", "tesla"])
    rows = [{"id": f"{split}#{i}", "game_id": split, "split": split, "t": 10.0, "state": tt_state,
             "action": {"type": "play", "card": "zap", "lane": "left", "zone": "mid", "tile": [3, 10]}}
            for split in ("train", "val", "test") for i in range(3)]
    rows.append({"id": "hold", "split": "train", "t": 1.0, "state": tt_state, "action": {"type": "hold"}})
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(tt, "KnowledgeGraph", lambda: _FakeKG({"knight": 1.0}))
    out, report = tmp_path / "w.json", tmp_path / "rep" / "tuning.json"
    assert tt.main(["--dataset", str(ds), "--passes", "2", "--out", str(out), "--report", str(report)]) == 0
    written = json.loads(out.read_text())
    assert set(written["weights"]) == set(ROLE_WEIGHTS) and written["version"] == 1
    result = json.loads(report.read_text())
    assert report.with_suffix(".md").exists() and result["kg"] is True
    for split in ("train", "val", "test"):
        for side in ("before", "after"):
            row = result[side][split]["situations"]["all"]
            assert row["n"] == 3 and row["n_eligible"] == 3 and row["objective"] is not None
    assert result["tuned"] == (written["weights"] != ROLE_WEIGHTS) and set(result["changed"]) <= set(ROLE_WEIGHTS)
    assert result["before"]["val"]["by_intent"]["defend"]["n"] == 3


# MAC-AD: stopping class on defend moves, pushes from 10 s out, answered by buildings and our recent plays -------------
def test_defend_moves_carry_their_stopping_class_and_offence_moves_do_not():
    stops = {m.card: m.stop for m in ENGINE.analyse(_giant_right(elixir=9.0)).defence}
    assert stops == {"tesla": "high", "knight": "medium", "musketeer": "medium"}
    knight_push = _state(units=[_enemy((3, 10), "knight", eta=4.0)], elixir=9.0,
                         hand=["tesla", "mini-p-e-k-k-a", "musketeer", "ice-spirit"])
    stops = {m.card: m.stop for m in ENGINE.analyse(knight_push).defence}
    assert stops == {"tesla": "high", "mini-p-e-k-k-a": "high", "musketeer": "medium", "ice-spirit": "low"}
    zap = next(m for m in ENGINE.analyse(_bats_left(elixir=9.0)).defence if m.card == "zap")
    assert zap.stop == "spell" and zap.value == 2.0                       # the elixir value of what it hits
    # the air-defense rule: a low-dps card that hits air stops a balloon (medium); a kamikaze card does not
    balloon = _state(units=[_enemy((3, 12), "balloon", eta=6.0)], elixir=9.0,
                     hand=["magic-archer", "ice-spirit", "tesla", "knight"])
    stops = {m.card: m.stop for m in ENGINE.analyse(balloon).defence}
    assert stops == {"magic-archer": "medium", "ice-spirit": "low", "tesla": "high"}      # the knight cannot hit air
    off = ENGINE.analyse(_state(elixir=10.0, hand=["hog-rider", "musketeer", "knight", "zap"]))
    assert off.moves and {m.stop for m in off.moves} == {"n/a"} and off.defence == []
    assert TacticalMove("zap", "zap|left|mid", [3.0, 8.0], "defend", "x", 0.5, 1.0).stop == "n/a"     # the default


def test_the_defence_list_keeps_every_defend_move_while_moves_keeps_six():
    units = [_enemy((3, 12), "knight", eta=4.0), _enemy((15, 12), "knight", eta=4.5, tower="own_right")]
    rep = ENGINE.analyse(_state(units=units, elixir=10.0, hand=["tesla", "knight", "musketeer", "ice-spirit"]))
    # eight defend moves, but the two kiting knights stand at the same key (both lanes, x 7.5 / 10.5 = "middle")
    assert len(rep.moves) <= 6 and len(rep.defence) == 7 and {m.intent for m in rep.defence} == {"defend"}
    assert [m.score for m in rep.defence] == sorted((m.score for m in rep.defence), reverse=True)
    assert [m for m in rep.moves if m.intent == "defend"] == rep.defence[:len(rep.moves)] or len(rep.moves) == 6


def test_a_unit_ten_seconds_out_is_listed_but_a_later_one_is_not():
    near = _state(units=[_enemy((15, 15), "giant", eta=9.5, tower="own_right")], elixir=6.0)
    rep = ENGINE.analyse(near)
    assert rep.pushes[0].eta_s == 9.5 and rep.best.card == "tesla" and rep.best.tile == [11.0, 7.0]
    assert ENGINE.analyse(_state(units=[_enemy((15, 15), "giant", eta=10.5, tower="own_right")])).pushes == []
    assert rep.pushes[0].heavy is True
    assert ENGINE.analyse(_bats_left()).pushes[0].heavy is False          # a swarm push is not heavy


def test_a_building_of_ours_near_a_building_targeter_answers_it():
    giant = _enemy((15, 13), "giant", eta=8.0, tower="own_right")
    base = {m.card: m.score for m in ENGINE.analyse(_state(units=[giant], elixir=9.0)).moves}
    rep = ENGINE.analyse(_state(units=[giant, _ally((15, 8), "bomb-tower")], elixir=9.0))      # 5 tiles from the giant
    push = rep.pushes[0]
    assert push.answered is True and push.answered_by == ["bomb-tower"]
    assert rep.lines[0].endswith("(being answered by bomb-tower)")
    assert {m.card: m.score for m in rep.moves} == {c: pytest.approx(round(v * 0.3, 3)) for c, v in base.items()}
    # 7+ tiles away, on their half, or not a building: not answered
    for ally in (_ally((15, 5), "bomb-tower"), _ally((15, 20), "bomb-tower"), _ally((15, 8), "knight")):
        out = ENGINE.analyse(_state(units=[giant, ally], elixir=9.0)).pushes[0]
        assert out.answered is False and out.answered_by == []
    # a push without a building targeter is not answered by a building 5 tiles away (only within 3 tiles)
    knight = _enemy((15, 13), "knight", eta=4.0, tower="own_right")
    assert ENGINE.analyse(_state(units=[knight, _ally((15, 8), "tesla")], elixir=9.0)).pushes[0].answered is False
    assert ENGINE.analyse(_state(units=[knight, _ally((15, 11), "tesla")], elixir=9.0)).pushes[0].answered is True


def test_our_recent_plays_answer_a_push_at_once():
    giant = _enemy((15, 13), "giant", eta=8.0, tower="own_right")

    def with_plays(*plays, t=100.0):
        state = _state(units=[giant], elixir=9.0)
        state["own"]["recent_plays"] = [{"player": "own", **p} for p in plays]
        return ENGINE.analyse(state, t).pushes[0]

    tower = {"t": 99.0, "card": "bomb-tower", "tile": [10.5, 9.0]}
    push = with_plays(tower)                                                 # 5 tiles from the giant, in its lane
    assert push.answered and push.answered_by == ["bomb-tower"]
    assert not with_plays(dict(tower, t=95.9)).answered                      # older than 4 s
    assert with_plays(dict(tower, t=96.0)).answered                          # exactly 4 s: still counts
    assert with_plays(dict(tower, tile=[14.5, 2.0])).answered                # in the push's lane, far from the lead unit
    assert not with_plays(dict(tower, tile=[3.5, 9.0])).answered             # the other lane, 11 tiles away
    assert with_plays(dict(tower, tile=[8.5, 12.0])).answered                # left of the middle, 6.6 tiles away
    assert not with_plays(dict(tower, tile=[15.0, 20.0])).answered           # on their half
    assert not with_plays(dict(tower, tile=None)).answered                   # unknown place
    assert not with_plays(dict(tower, t=None)).answered
    assert not with_plays(dict(tower, card=None)).answered
    # a spell counts only on the lead unit (within 3 tiles): a fireball on the giant (4 elixir >= 0.6 x 5)
    assert with_plays({"t": 99.0, "card": "fireball", "tile": [15.0, 12.0]}).answered_by == ["fireball"]
    assert not with_plays({"t": 99.0, "card": "fireball", "tile": [15.0, 8.0]}).answered
    # elixir adds up over distinct cards: a skeleton army (1) and a repeated card (once) are not enough alone
    assert not with_plays({"t": 99.0, "card": "skeletons", "tile": [15.0, 9.0]},
                          {"t": 99.5, "card": "skeletons", "tile": [14.0, 9.0]}).answered
    both = with_plays({"t": 98.0, "card": "skeletons", "tile": [15.0, 9.0]}, {"t": 99.0, "card": "knight", "tile": [15.0, 9.0]})
    assert both.answered_by == ["skeletons", "knight"]
    # no clock: the newest play stands for "now"
    state = _state(units=[giant], elixir=9.0)
    state["own"]["recent_plays"] = [{"player": "own", **dict(tower, t=40.0)}]
    assert ENGINE.analyse(state).pushes[0].answered
    # the defence scores shrink as for answered allies
    plain = {m.card: m.score for m in ENGINE.analyse(_state(units=[giant], elixir=9.0)).moves}
    state["own"]["recent_plays"] = [{"player": "own", **tower}]
    assert {m.card: m.score for m in ENGINE.analyse(state, 100.0).moves} \
        == {c: pytest.approx(round(v * 0.3, 3)) for c, v in plain.items()}


# MAC-AG: flyers, pulls, kites, offence during pressure ------------------------------------------------------------------
# The real role table (the Z1 file), the weights pinned to the defaults, and the confirmed deck (D-047).
REAL = TacticalEngine(CARDS, weights=DEFAULT_WEIGHTS)
DECK47 = ["electro-dragon", "musketeer", "tesla", "fireball", "knight", "zap", "ram-rider", "mini-p-e-k-k-a"]
OWN_TOWERS = {"left": (3.5, 4.3), "right": (14.5, 4.3)}


def _defence_cards(report):
    return [m.card for m in report.defence]


def test_ag_a_flyer_is_never_answered_with_a_ground_only_card():
    bats = [_enemy(t, "bats", eta=4.0) for t in ((4, 14), (3, 14), (5, 14))]
    hand = ["electro-dragon", "knight", "musketeer", "zap"]
    witch = _enemy((4, 20), "night-witch", eta=8.0)                                # 6 tiles behind the bats
    rep = REAL.analyse(_state(units=bats + [witch], hand=hand, elixir=7.0))
    assert rep.pushes[0].units[0]["card"] == "bats"
    cards = _defence_cards(rep)
    assert "knight" not in cards and {"zap", "electro-dragon", "musketeer"} <= set(cards)
    # the same hand against bats alone: no knight either
    assert "knight" not in _defence_cards(REAL.analyse(_state(units=bats, hand=hand, elixir=7.0)))
    # a ground unit within 2 tiles of the lead is part of the same group: the knight still helps against it
    near = _enemy((4, 15.5), "night-witch", eta=4.5)
    assert "knight" in _defence_cards(REAL.analyse(_state(units=bats + [near], hand=hand, elixir=7.0)))
    # a ground lead with a flyer behind it is the old rule: a penalty, not a ban
    knight_bats = [_enemy((3, 12), "knight", eta=4.0), _enemy((3, 14), "bats", eta=5.0)]
    assert "knight" in _defence_cards(REAL.analyse(_state(units=knight_bats, hand=hand, elixir=7.0)))
    # a lone balloon: no ground-only card or building that cannot hit air
    balloon = _enemy((3, 13), "balloon", eta=6.0)
    assert not {"knight", "bomb-tower", "cannon"} & set(
        _defence_cards(REAL.analyse(_state(units=[balloon], hand=["knight", "bomb-tower", "musketeer", "zap"]))))


def test_ag_a_building_pulls_a_giant_only_at_a_pull_tile_and_only_before_it_locks_on():
    giant = _enemy((14, 15), "giant", eta=9.0, tower="own_right")
    rep = REAL.analyse(_state(units=[giant], hand=["musketeer", "knight", "tesla", "mini-p-e-k-k-a"], elixir=8.0))
    tesla = next(m for m in rep.defence if m.card == "tesla")
    d_giant = ((tesla.tile[0] - 14) ** 2 + (tesla.tile[1] - 15) ** 2) ** 0.5
    d_tower = ((OWN_TOWERS["right"][0] - 14) ** 2 + (OWN_TOWERS["right"][1] - 15) ** 2) ** 0.5
    assert tesla.stop == "high" and tesla.reason.startswith("pulls giant to the centre")
    assert 7.0 <= tesla.tile[0] <= 11.0 and 4.0 <= tesla.tile[1] <= 11.0         # the centre band of our half
    assert d_giant <= d_tower - 1.5 and d_giant >= 3.0                           # nearer to it than the tower, not on it
    assert rep.best.card == "tesla"
    # the giant has reached its tower: nothing pulls it, the building is a block and the tank killer is preferred
    attacking = _enemy((14, 5.5), "giant", eta=0.5, tower="own_right")
    rep = REAL.analyse(_state(units=[attacking], hand=["tesla", "mini-p-e-k-k-a", "musketeer", "knight"], elixir=8.0))
    by_card = {m.card: m for m in rep.defence}
    assert rep.best.card == "mini-p-e-k-k-a" and by_card["tesla"].stop == "low" and by_card["tesla"].reason == "slows giant"
    assert REAL._pull(rep.pushes[0])[1] is None
    # between the lock-on distance and the nearest pull tile there is no pull either (the tower is only 3.7 tiles away)
    close = _enemy((14, 8), "giant", eta=3.0, tower="own_right")
    push = REAL.analyse(_state(units=[close])).pushes[0]
    assert REAL._pull(push)[0] is close and REAL._pull(push)[1] is None
    # a flying targeter is pulled in the same way, in the left lane
    pull = REAL._pull(REAL.analyse(_state(units=[_enemy((3, 15), "balloon", eta=9.0)])).pushes[0])[1]
    assert pull is not None and 7.0 <= pull[0] <= 11.0
    assert REAL._pull(REAL.analyse(_state(units=[_enemy((3, 12), "knight", eta=4.0)])).pushes[0]) == (None, None)


def test_ag_a_melee_defender_kites_a_ground_troop_targeter_to_the_centre():
    hand = ["mini-p-e-k-k-a", "musketeer", "tesla", "fireball"]
    rep = REAL.analyse(_state(units=[_enemy((3, 12), "knight", eta=4.0)], hand=hand, elixir=7.0))
    by_card = {m.card: m for m in rep.defence}
    assert by_card["mini-p-e-k-k-a"].tile == [7.5, 9.0] and by_card["mini-p-e-k-k-a"].reason.startswith(
        "kites knight to the centre")
    assert by_card["musketeer"].tile == [5.0, 2.5]                                 # ranged support stays behind the tower
    right = REAL.analyse(_state(units=[_enemy((15, 12), "knight", eta=4.0, tower="own_right")], hand=hand, elixir=7.0))
    assert next(m for m in right.defence if m.card == "mini-p-e-k-k-a").tile == [10.5, 9.0]
    deep = REAL.analyse(_state(units=[_enemy((3, 3), "knight", eta=0.5)], hand=hand, elixir=7.0))
    assert next(m for m in deep.defence if m.card == "mini-p-e-k-k-a").tile[1] == 1.0       # never behind the back line
    # a building targeter is met in front of the tower, a push already fought by a troop of ours is not kited
    giant = REAL.analyse(_state(units=[_enemy((15, 12), "giant", eta=4.0, tower="own_right")], hand=hand, elixir=7.0))
    assert next(m for m in giant.defence if m.card == "mini-p-e-k-k-a").reason.startswith("stops giant in front")
    fought = REAL.analyse(_state(units=[_enemy((3, 12), "knight", eta=4.0), _ally((3.5, 11), "knight")], hand=hand,
                                 elixir=7.0))
    assert fought.pushes[0].answered
    assert not next(m for m in fought.defence if m.card == "mini-p-e-k-k-a").reason.startswith("kites")


# offence during pressure ------------------------------------------------------------------------------------------------
OFFENCE_HAND = ["ram-rider", "musketeer", "electro-dragon", "zap"]


def _answered_giant_right(elixir=7.0, extra=(), hand=None, eta=5.0):
    giant = _enemy((14, 12), "giant", eta=eta, tower="own_right")
    return _state(units=[giant, _ally((11, 8), "tesla"), *extra], elixir=elixir, hand=hand or OFFENCE_HAND)


def test_ag_split_into_the_other_lane_when_they_commit_5_elixir_into_an_answered_push():
    rep = REAL.analyse(_answered_giant_right())
    assert rep.pushes[0].answered and rep.pushes[0].eta_s == 5.0                    # pressure: a push within 8 s
    assert [(m.intent, m.card, m.tile, m.score) for m in rep.offence] == [("split", "ram-rider", [3.5, 14.0], 0.7)]
    assert rep.best.card == "ram-rider" and rep.best.key == "ram-rider|left|bridge"
    assert rep.best.reason == "they committed 5 elixir right"
    assert rep.lines[-2] == "tactics: best ram-rider @[3.5,14] split: they committed 5 elixir right"
    # a counter-push in the left lane takes the same key; split scores higher
    rep = REAL.analyse(_answered_giant_right(extra=[_ally((4, 10), "knight")]))
    assert [m.intent for m in rep.offence] == ["split"]
    # not enough elixir (cost 5 + 1), a push worth less than 5, or a push nobody answers: no split
    assert REAL.analyse(_answered_giant_right(elixir=5.5)).offence == []
    knight = _enemy((14, 11), "knight", eta=5.0, tower="own_right")
    small = _state(units=[knight, _ally((14, 8.5), "knight")], elixir=8.0, hand=OFFENCE_HAND)
    assert REAL.analyse(small).pushes[0].answered and REAL.analyse(small).offence == []
    assert REAL.analyse(_state(units=[_enemy((14, 12), "giant", eta=5.0, tower="own_right")], elixir=8.0,
                               hand=OFFENCE_HAND)).offence == []
    # a second push in the target lane that is still on: no split into it
    balloon = _enemy((3, 13), "balloon", eta=4.0)
    assert [m for m in REAL.analyse(_answered_giant_right(extra=[balloon], elixir=8.0)).offence
            if m.intent == "split"] == []


def test_ag_counter_push_during_pressure_needs_the_lane_push_answered_or_gone():
    ally = _ally((4, 10), "knight")
    giant_left = _enemy((3.5, 14), "giant", eta=5.0)
    # their giant is in the left lane, where our knight is: not answered (too far from the lead), so no counter-push there
    state = _state(units=[giant_left, ally], elixir=8.0, hand=OFFENCE_HAND)
    assert REAL.analyse(state).pushes[0].answered is False
    assert [m for m in REAL.analyse(state).offence if m.intent == "counter_push"] == []
    # answered by our tesla at the tower: the knight's lane may counter-push
    state = _state(units=[giant_left, _ally((6, 8), "tesla"), ally], elixir=8.0, hand=OFFENCE_HAND)
    rep = REAL.analyse(state)
    assert rep.pushes[0].answered and ("counter_push", [3.5, 14.0]) in [(m.intent, m.tile) for m in rep.offence]
    # the push is in the other lane: the knight's lane has none, so it counter-pushes while the right lane is held
    state = _state(units=[_enemy((14, 12), "giant", eta=5.0, tower="own_right"), ally], elixir=8.0, hand=OFFENCE_HAND)
    assert [(m.intent, m.tile) for m in REAL.analyse(state).offence] == [("counter_push", [3.5, 14.0])]
    # a building is not a unit that wins a lane
    building = _ally((4, 10), "bomb-tower")
    state = _state(units=[_enemy((14, 12), "giant", eta=5.0, tower="own_right"), building], elixir=8.0,
                   hand=OFFENCE_HAND)
    assert REAL.analyse(state).offence == []


def test_ag_a_full_bar_leaks_the_win_condition_during_pressure_unless_a_push_is_at_the_tower():
    hand = ["ram-rider", "knight", "mini-p-e-k-k-a", "fireball"]
    far = _state(units=[_enemy((3, 14), "knight", eta=6.0)], elixir=10.0, hand=hand)
    leak = REAL.analyse(far).offence
    assert [(m.intent, m.card, m.tile, m.score) for m in leak] == [("leak", "ram-rider", [14.5, 14.0], 0.5)]   # away from the push
    assert leak[0].reason == "elixir 10 near full, pushes right"
    # a push within 3 s: no leak; below 9.5 elixir: none either
    assert REAL.analyse(_state(units=[_enemy((3, 10), "knight", eta=2.5)], elixir=10.0, hand=hand)).offence == []
    assert REAL.analyse(_state(units=[_enemy((3, 14), "knight", eta=6.0)], elixir=9.0, hand=hand)).offence == []
    # pressure leaks only win conditions (the plain-card leak of a quiet board stays quiet-board only)
    plain = _state(units=[_enemy((3, 14), "knight", eta=6.0)], elixir=10.0, hand=["knight", "musketeer", "zap", "tesla"])
    assert REAL.analyse(plain).offence == []
    # both lanes pushed: the weaker enemy tower
    both = _state(units=[_enemy((3, 14), "knight", eta=6.0), _enemy((15, 14), "knight", eta=6.5, tower="own_right")],
                  elixir=10.0, hand=hand, opp_towers={"king": None, "left": 4000, "right": 1500})
    assert [m.tile for m in REAL.analyse(both).offence] == [[14.5, 14.0]]


def test_ag_quiet_board_offence_is_unchanged_and_the_report_lists_every_offence_move():
    rep = REAL.analyse(_state(units=[_ally((4, 10), "knight")], elixir=8.0, hand=OFFENCE_HAND))
    assert [(m.intent, m.card) for m in rep.offence] == [("counter_push", "ram-rider")]
    assert [m.key for m in rep.moves if m.intent == "counter_push"] == [m.key for m in rep.offence]
    # the offence list holds every offence move even when six defend moves fill `moves`
    units = [_enemy((3, 12), "knight", eta=4.0), _enemy((15, 12), "knight", eta=4.5, tower="own_right"),
             _ally((8, 10), "knight")]
    big = REAL.analyse(_state(units=units, elixir=10.0, hand=["ram-rider", "mini-p-e-k-k-a", "tesla", "musketeer"]))
    assert all(m.intent != "defend" for m in big.offence)


# MAC-AH: the threat model ---------------------------------------------------------------------------------------------
# REAL is the full engine (real role table, default weights, threat model on). The deck of D-047, the user's scenario of
# runs/poc/20261106-170722 (t = 240.7: a night witch, an electro wizard and two bats on the left, a balloon at the river on
# the right) and the numbers of the lane simulation (level 15 opponent, princess tower 4,424 HP).
def _heading(unit, text):
    return {**unit, "heading": text}


def _scenario(left=True, balloon=True, allies=True, elixir=7.0, hand=("musketeer", "tesla", "knight", "zap"),
              balloon_tile=(16, 15), balloon_eta=10.7, towers=None):
    units = []
    if left:
        units += [_heading(_enemy((4, 10), "night-witch", eta=5.1), "advancing left lane"),
                  _heading(_enemy((3, 10), "electro-wizard", eta=34.9), "stationary"),
                  _heading(_enemy((4, 10), "bats", eta=2.1, unit_class="bat"), "advancing left lane"),
                  _heading(_enemy((2, 10), "bats", eta=2.0, unit_class="bat"), "advancing left lane")]
    if balloon:
        units.append(_heading(_enemy(balloon_tile, "balloon", eta=balloon_eta, tower="own_right"), "advancing right lane"))
    if allies:
        units += [_ally((4, 7), "electro-dragon"), _ally((5, 5), "musketeer")]
    state = _state(elixir=elixir, units=units, hand=list(hand), deck=DECK47)
    state["own"]["towers"] = towers or {"king": None, "left": 4082, "right": 4424}
    return state


def test_ah_a_balloon_at_the_river_is_a_push_with_its_magnitude_and_a_simulated_eta():
    balloon = _heading(_enemy((16, 15), "balloon"), "advancing right lane")          # no tracker ETA at all
    rep = REAL.analyse(_state(units=[balloon], hand=["tesla", "knight", "musketeer", "zap"]))
    push = rep.pushes[0]
    assert push.lane == "right" and push.early is True and push.needs == ["air", "building_targeter"]
    assert push.damage_if_unanswered == 3162.0 and push.destroys_tower is False and not push.tower_ok
    assert push.sim_eta_s == 9.25 and push.eta_s == 9.25                              # the simulation's, there is no tracker's
    assert rep.lines[0] == ("tactics: push right: balloon (5 elixir), ~3,200 tower damage if unanswered, tower in 9.2s;"
                            " needs air, building_targeter")
    assert rep.best.card == "tesla" and rep.best.reason == "pulls balloon to the centre, saves ~3,200 tower HP, trade +1"
    assert rep.best.adequate is True and rep.best.prevented == 3162.0 and rep.best.damage_with == 0.0
    assert rep.best.efficiency == pytest.approx(3162.0 / 4, abs=0.1)
    # not heading our way / on their half: no push
    assert REAL.analyse(_state(units=[_heading(_enemy((16, 15), "balloon"), "retreating")])).pushes == []
    assert REAL.analyse(_state(units=[_enemy((16, 24), "balloon")])).pushes == []        # row > 20
    assert [p.early for p in REAL.analyse(_state(units=[_enemy((16, 20), "balloon")])).pushes] == [True]   # a flyer at row 20
    assert REAL.analyse(_state(units=[_enemy((16, 21), "bats", unit_class="bat")])).pushes == []


def test_ah_river_units_walking_toward_us_count_and_a_unit_without_information_does_too():
    advancing = _heading(_enemy((15, 17), "knight"), "advancing right lane")
    assert [p.early for p in REAL.analyse(_state(units=[advancing])).pushes] == [True]
    vel = {**_enemy((15, 17), "knight"), "vel": [0.0, -1.0]}
    assert len(REAL.analyse(_state(units=[vel])).pushes) == 1
    away = {**_enemy((15, 17), "knight"), "vel": [0.0, 1.0]}
    assert REAL.analyse(_state(units=[away])).pushes == []
    pred = _enemy((15, 17), "knight", pred=(15, 19))                                  # its row grows: away from us
    assert REAL.analyse(_state(units=[pred])).pushes == []
    assert len(REAL.analyse(_state(units=[_enemy((15, 17), "knight", pred=(15, 15))])).pushes) == 1
    assert len(REAL.analyse(_state(units=[_enemy((15, 17), "knight")])).pushes) == 1  # nothing says otherwise
    assert REAL.analyse(_state(units=[_enemy((15, 19), "knight")])).pushes == []     # row 19: not at the river yet


def test_ah_the_user_scenario_the_balloon_comes_first_and_the_left_lane_gets_nothing_that_spends_its_answer():
    rep = REAL.analyse(_scenario())
    assert [p.lane for p in rep.pushes] == ["right", "left"]
    balloon, left = rep.pushes
    assert balloon.damage_if_unanswered == 3162.0 and not balloon.answered
    assert left.damage_if_unanswered > 4000 and left.destroys_tower          # unanswered the left would fall ...
    assert left.damage_with_allies < 300 and left.answered and left.answered_by == ["electro-dragon", "musketeer"]   # ... ours hold it
    assert rep.best.card == "tesla" and rep.best.tile == [11.0, 11.0]
    right_moves = [m for m in rep.defence if "balloon" in m.reason]
    assert {m.card for m in right_moves} == {"tesla", "musketeer"} and all(m.adequate for m in right_moves)
    assert "knight" not in {m.card for m in right_moves} and "zap" not in {m.card for m in right_moves}
    left_moves = [m for m in rep.defence if lane_of(m.tile) == "left"]
    assert left_moves and max(m.score for m in left_moves) < rep.best.score / 4          # nothing on the left competes
    assert [ln.split(":")[1].split()[0] for ln in rep.lines[:2]] == ["push", "push"]
    assert "(being answered by electro-dragon, musketeer)" in rep.lines[1]
    assert rep.lines[0].startswith("tactics: push right: balloon (5 elixir), ~3,200 tower damage if unanswered, tower in 10.7s")


def test_ah_without_allies_the_left_push_that_destroys_the_tower_is_first():
    rep = REAL.analyse(_scenario(allies=False))
    assert [p.lane for p in rep.pushes] == ["left", "right"]
    left = rep.pushes[0]
    assert left.destroys_tower and left.damage_if_unanswered == 4082.0 and left.damage_with_allies is None
    assert "destroys the tower" in rep.lines[0]
    # the balloon does less (3,162) but the left would take the tower: the left answers are better by the simulation
    best_left = max((m for m in rep.defence if lane_of(m.tile) == "left"), key=lambda m: m.score)
    assert best_left.card in {"musketeer", "knight"} and best_left.prevented > 3000


def test_ah_pushes_are_ranked_by_magnitude_not_by_eta():
    units = [_enemy((3, 12), "knight", eta=4.0), _enemy((15, 12), "giant", eta=5.0, tower="own_right")]
    rep = REAL.analyse(_state(units=units, elixir=9.0))
    assert [p.lane for p in rep.pushes] == ["right", "left"]                  # the giant (~3,300) before the knight (~2,100)
    assert rep.pushes[0].damage_if_unanswered > rep.pushes[1].damage_if_unanswered
    # a tower the push would destroy ranks first whatever the damage: 600 HP left on the left
    low = _state(units=units, elixir=9.0)
    low["own"]["towers"] = {"king": None, "left": 600, "right": 4424}
    assert [p.lane for p in REAL.analyse(low).pushes] == ["left", "right"]
    assert REAL.analyse(low).pushes[0].destroys_tower


def test_ah_two_bats_are_a_push_the_tower_handles():
    bats = [_enemy((3, 14), "bats", eta=2.0, unit_class="bat"), _enemy((4, 14), "bats", eta=2.0, unit_class="bat")]
    rep = REAL.analyse(_state(units=bats, hand=["zap", "knight", "musketeer", "fireball"]))
    push = rep.pushes[0]
    assert push.tower_ok and push.damage_if_unanswered < 300 and not push.destroys_tower
    assert rep.defence == [] and rep.best is None and rep.hold_reason == "tower handles it"
    assert rep.lines[-1] == "tactics: hold: tower handles it" and rep.lines[0].endswith("needs air, swarm; tower handles it")
    assert REAL.score_key("zap|left|bridge", _state(units=bats, hand=["zap", "knight"])) is None
    # five bats and a witch are not
    pack = bats + [_enemy((4, 15), "night-witch", eta=3.0)]
    assert not REAL.analyse(_state(units=pack)).pushes[0].tower_ok


def test_ah_a_move_is_adequate_when_the_tower_damage_left_is_small():
    state = _state(units=[_heading(_enemy((16, 15), "balloon", eta=9.0, tower="own_right"), "advancing right lane")],
                   hand=["tesla", "knight", "musketeer", "zap"], elixir=9.0)
    rep = REAL.analyse(state)
    by_card = {m.card: m for m in rep.defence}
    assert set(by_card) == {"tesla", "musketeer"}                              # no zap (nothing to gain), no knight (no air)
    assert by_card["tesla"].adequate and by_card["tesla"].prevented == 3162.0 and by_card["tesla"].damage_with == 0.0
    assert by_card["musketeer"].adequate and by_card["musketeer"].damage_with == 351.0     # only the death bomb lands
    assert by_card["tesla"].score > by_card["musketeer"].score
    assert by_card["musketeer"].reason.endswith("trade +1") and "saves ~2,800 tower HP" in by_card["musketeer"].reason
    # a magic archer behind the tower leaves too much of a level-15 balloon's damage: not adequate
    weak = REAL.analyse(_state(units=state["units"], hand=["magic-archer", "ice-spirit", "bomb-tower", "knight"], elixir=9.0))
    magic = next(m for m in weak.defence if m.card == "magic-archer")
    assert magic.adequate is False and magic.prevented > 1000 and magic.damage_with > 0.35 * 3162
    assert "bomb-tower" not in [m.card for m in weak.defence]


def test_ah_the_simulated_prevention_decides_over_the_role_tie_breakers():
    # a giant: the tesla pull and the mini pekka (a tank killer) both stop it; the knight does not stop it alone
    giant = _enemy((15, 13), "giant", eta=8.0, tower="own_right")
    rep = REAL.analyse(_state(units=[giant], hand=["tesla", "knight", "musketeer", "mini-p-e-k-k-a"], elixir=9.0))
    by_card = {m.card: m for m in rep.defence}
    assert rep.best.card == "tesla" and by_card["mini-p-e-k-k-a"].adequate
    assert by_card["knight"].damage_with > by_card["mini-p-e-k-k-a"].damage_with


def test_ah_zap_on_a_lone_balloon_is_not_proposed_and_zap_on_bats_with_support_is():
    balloon = _heading(_enemy((3, 14), "balloon", eta=6.0, tower="own_left"), "advancing left lane")
    rep = REAL.analyse(_state(units=[balloon], hand=["zap", "musketeer", "tesla", "knight"], elixir=9.0))
    assert "zap" not in [m.card for m in rep.moves]
    witch = _scenario(allies=False, balloon=False, hand=("zap", "knight", "fireball", "musketeer"), elixir=7.0)
    zap = next(m for m in REAL.analyse(witch).defence if m.card == "zap")
    assert zap.value == pytest.approx(2.962, abs=0.001) and zap.value >= 2.0           # bats dead + a bit of the witch pair
    assert zap.tile == [3.0, 10.0]


def test_ah_the_threat_model_switch_restores_the_mac_ag_engine():
    off = TacticalEngine(CARDS, weights={**DEFAULT_WEIGHTS, "threat_model": 0.0})
    state = _scenario()
    rep = off.analyse(state)
    assert all(p.damage_if_unanswered is None and not p.early and not p.tower_ok and p.damage_with_allies is None
               for p in rep.pushes)
    assert [p.lane for p in rep.pushes] == ["left"]                              # the balloon at the river is not a push
    assert all(m.adequate is None and m.prevented is None and m.efficiency is None for m in rep.defence)
    assert all("saves" not in m.reason for m in rep.moves) and "tower damage" not in " ".join(rep.lines)
    on = REAL.analyse(state)
    assert on.pushes[0].damage_if_unanswered is not None


def test_ah_a_push_without_stats_falls_back_to_the_role_rules():
    unknown = [{"unit_class": "mystery", "card": None, "side": "enemy", "tile": [4, 8],
                "eta_tower": {"tower": "own_left", "s": 3.0}}]
    state = _state(units=unknown, hand=["knight", "musketeer", "zap", "tesla"])
    rep = REAL.analyse(state)
    assert rep.pushes[0].damage_if_unanswered is None and not rep.pushes[0].tower_ok
    assert rep.defence and all(m.adequate is None for m in rep.defence)
    assert "tower damage" not in rep.lines[0]
    # a card of ours without stats: judged by its role alone
    ghost = TacticalEngine({**CARDS, "ghost-card": {**CARDS["knight"], "slug": "ghost-card"}}, weights=DEFAULT_WEIGHTS)
    st = _state(units=[_enemy((3, 10), "knight", eta=4.0)], hand=["ghost-card", "musketeer"])
    moves = {m.card: m for m in ghost.analyse(st).defence}
    assert moves["ghost-card"].adequate is None and moves["musketeer"].adequate is not None


def test_ah_the_engine_with_the_threat_model_stays_fast():
    state = _scenario()
    REAL.analyse(state)
    runs = []
    for _ in range(20):
        t0 = time.perf_counter()
        REAL.analyse(state)
        runs.append(time.perf_counter() - t0)
    assert min(runs) < 0.015, f"{min(runs) * 1000:.2f} ms"
