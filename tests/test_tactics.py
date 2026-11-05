"""TacticalEngine (MAC-Z2): pushes, defence scores and placement, offence, hold reasons, lines, score_key - hand-made
states, inline roles in the Z1 schema, no models."""
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.game_rules import lane_of, load_cards, zone_of  # noqa: E402
from clashbot.tactics import TacticalEngine, TacticalMove, TacticalReport  # noqa: E402

CARDS = load_cards()


# -- inline roles (Z1 schema) ------------------------------------------------------------------------------------------
def _troop(targets="ground", hp="medium", dps="medium", rng="melee", roles=(), count=1, flying=False, splash=False,
           speed="medium"):
    return {"type": "troop", "targets": targets, "flying": flying, "hp_class": hp, "dps_class": dps,
            "splash": splash, "range": rng, "speed": speed, "count": count, "roles": list(roles), "source": "manual"}


def _building(targets="ground", dps="medium", roles=("building",)):
    return {"type": "building", "targets": targets, "flying": False, "hp_class": "medium", "dps_class": dps,
            "splash": False, "range": "long", "speed": None, "count": 1, "roles": list(roles), "source": "manual"}


def _spell(cls, radius, kills=(), roles=()):
    return {"type": "spell", "spell_class": cls, "radius": radius, "kills": list(kills), "roles": list(roles),
            "source": "manual"}


ROLES = {
    "knight": _troop(roles=["mini_tank"]),
    "valkyrie": _troop(roles=["splash_defense", "mini_tank"], splash=True),
    "mini-p-e-k-k-a": _troop(dps="high", roles=["mini_tank", "tank_killer"]),
    "mega-knight": _troop(hp="tank", dps="high", splash=True, roles=["tank"]),
    "giant": _troop(targets="buildings", hp="tank", roles=["win_condition", "tank"]),
    "hog-rider": _troop(targets="buildings", speed="very_fast", roles=["win_condition"]),
    "ram-rider": _troop(targets="buildings", roles=["win_condition"]),
    "musketeer": _troop(targets="air_ground", rng="long", roles=["support"]),
    "archers": _troop(targets="air_ground", hp="light", rng="long", count=2, roles=["support"]),
    "bats": _troop(targets="air_ground", hp="swarm", dps="low", count=3, flying=True, speed="very_fast",
                   roles=["swarm", "cycle"]),
    "minions": _troop(targets="air_ground", hp="light", rng="short", count=3, flying=True, roles=["swarm"]),
    "skeletons": _troop(hp="swarm", dps="low", count=3, roles=["swarm", "cycle"]),
    "ice-spirit": _troop(hp="swarm", dps="low", rng="short", splash=True, roles=["kamikaze", "cycle"]),
    "baby-dragon": _troop(targets="air_ground", rng="short", flying=True, splash=True, roles=["splash_defense"]),
    "tesla": _building(targets="air_ground", roles=["building", "air_defense"]),
    "cannon": _building(roles=["building"]),
    "zap": _spell("small", 2.5, ["swarm"], ["spell_small"]),
    "the-log": _spell("small", 2.0, ["swarm"], ["spell_small"]),
    "fireball": _spell("medium", 2.5, ["swarm", "light"], ["spell_medium"]),
    "rage": _spell("utility", 5.0, [], ["spell_small"]),
    "goblin-barrel": _spell("small", 1.5, [], ["win_condition"]),
}

ENGINE = TacticalEngine(CARDS, ROLES)


# -- state fixtures (as tests/test_policy.py) --------------------------------------------------------------------------
def _state(elixir=7.0, units=None, hand=None, deck=None, opp_elixir=4.0, opp_plays=None, opp_towers=None,
           next_card=None):
    hand = ["tesla", "knight", "zap", "musketeer"] if hand is None else hand
    return {
        "phase": "double_elixir", "match_seconds": 150,
        "own": {"elixir": elixir, "hand": hand, "next_card": next_card,
                "deck": deck or ["tesla", "knight", "zap", "musketeer", "giant", "fireball", "bats", "skeletons"],
                "towers": {"king": None, "left": 4000, "right": 4000}, "recent_plays": []},
        "opponent": {"elixir_estimate": opp_elixir,
                     "towers": opp_towers or {"king": None, "left": 4000, "right": 4000},
                     "deck_known": [], "recent_plays": opp_plays or []},
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
    assert best.tile == [10.5, 9.0] and best.key == "tesla|middle|mid"
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
    assert by_card["musketeer"].trade == -1.0 and by_card["musketeer"].score == pytest.approx(0.5)
    assert by_card["mega-knight"].trade == -4.0 and by_card["mega-knight"].score == pytest.approx(0.05)
    assert by_card["skeletons"].trade == 2.0 and by_card["skeletons"].reason.endswith("trade +2")
    assert by_card["mega-knight"].reason.endswith("trade -4")


def test_win_condition_is_kept_for_offence_on_defence():
    state = _state(units=[_enemy((3, 10), "knight", eta=4.0)], hand=["hog-rider", "musketeer"])
    by_card = {m.card: m for m in ENGINE.analyse(state).moves}
    assert by_card["musketeer"].score == pytest.approx(0.5)
    assert by_card["hog-rider"].score == pytest.approx(0.1)


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
    assert ENGINE.analyse(giant_left).best.tile == [7.5, 8.0]
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
    boosted = {m.card: m.score for m in TacticalEngine(CARDS, ROLES, kg=kg).analyse(state).moves}
    assert kg.calls == ["giant"]
    assert boosted["musketeer"] == pytest.approx(plain["musketeer"] + 0.2)
    assert boosted["knight"] == pytest.approx(plain["knight"] + 0.05)
    assert TacticalEngine(CARDS, ROLES, kg=_FakeKG({})).analyse(state).best.card == ENGINE.analyse(state).best.card


# misc --------------------------------------------------------------------------------------------------------------
def test_deterministic_and_does_not_mutate_the_state():
    state = _bats_left()
    before = repr(state)
    a, b = ENGINE.analyse(state), ENGINE.analyse(state, t=12.0)
    assert a == b and repr(state) == before


def test_accepts_the_role_file_layout_and_unknown_cards_count_as_medium_ground():
    wrapped = TacticalEngine(CARDS, {"version": 1, "cards": ROLES})
    assert wrapped.analyse(_giant_right()).best.card == "tesla"
    sparse = TacticalEngine(CARDS, {})                                       # no roles at all: everything is a medium unit
    rep = sparse.analyse(_state(units=[_enemy((3, 12), "giant", eta=4.0)], hand=["knight", "zap"]))
    assert rep.pushes[0].needs == [] and rep.best.card == "knight"
    unnamed = _state(units=[_enemy((3, 12), None, eta=4.0, unit_class="skeleton")], hand=["knight"])
    assert ENGINE.analyse(unnamed).pushes[0].summary == "skeleton (0 elixir), tower in 4.0s"


def test_real_role_table_when_z1_is_merged():
    pytest.importorskip("clashbot.card_roles")
    engine = TacticalEngine(CARDS)                                           # roles=None: load_roles()
    rep = engine.analyse(_giant_right(hand=["tesla", "knight", "zap", "musketeer"]))
    assert rep.best is not None and rep.pushes[0].lane == "right" and "tank" in rep.pushes[0].needs
