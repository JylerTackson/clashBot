"""ReactivePolicy (MAC-W): decide() modes, candidate filtering, no blind spells - hand-made states, no models."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.candidates import Candidate  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.policy import ActDecision, LearnedActPolicy, ReactivePolicy  # noqa: E402

CARDS = load_cards()
POLICY = ReactivePolicy(CARDS)


def _state(elixir=5.0, units=None, opp_plays=None, opp_elixir=4.0, hand=None):
    return {
        "phase": "double_elixir", "match_seconds": 150,
        "own": {"elixir": elixir, "hand": hand or ["zap", "archers", "knight", "giant"],
                "deck": ["zap", "archers", "knight", "giant"],
                "towers": {"king": None, "left": 4000, "right": 5000}, "recent_plays": []},
        "opponent": {"elixir_estimate": opp_elixir, "towers": {"king": None, "left": 3000, "right": 4858},
                     "deck_known": [], "recent_plays": opp_plays or []},
        "units": units or [], "threats": [],
    }


def _enemy(tile, eta=None, tower="own_left", card="knight", pred=None):
    unit = {"unit_class": card, "card": card, "side": "enemy", "tile": list(tile),
            "eta_tower": None if eta is None else {"tower": tower, "s": eta}}
    if pred is not None:
        unit["predicted_tile_2s"] = list(pred)
    return unit


def _cands(*keys, kg=()):
    return [Candidate(k, 1.0 - 0.1 * i, {"kg": 1.0} if k in kg else {"knn": 1.0}) for i, k in enumerate(keys)]


def _keys(cands):
    return [c.key for c in cands]


# 1 -----------------------------------------------------------------------------------------------------------------
def test_no_legal_cards_is_none():
    d = POLICY.decide(_state(elixir=0.0, units=[_enemy((5, 9), eta=1.0)]), 10.0)
    assert d == ActDecision(False, "none", None, "no affordable card")


# 2 -----------------------------------------------------------------------------------------------------------------
def test_close_eta_defends_the_threat_lane():
    unit = _enemy((5, 20), eta=3.0)                  # still on their half but 3 s from our left tower
    d = POLICY.decide(_state(units=[unit]), 10.0)
    assert d.act and d.mode == "defend" and d.lane == "left" and d.threat is unit
    assert "knight" in d.reason and "left" in d.reason and "3.0" in d.reason


def test_far_eta_on_the_enemy_half_holds():
    d = POLICY.decide(_state(units=[_enemy((5, 25), eta=6.0)]), 10.0)
    assert not d.act and d.mode == "hold" and d.lane is None and d.reason.startswith("save elixir")


def test_most_urgent_threat_wins():
    slow, fast = _enemy((14, 12), eta=3.5, tower="own_right"), _enemy((4, 10), eta=1.5)
    d = POLICY.decide(_state(units=[slow, fast]), 10.0)
    assert d.mode == "defend" and d.lane == "left" and d.threat is fast


def test_own_units_and_other_towers_are_not_threats():
    own = {**_enemy((5, 9), eta=1.0), "side": "own"}
    toward_enemy = _enemy((5, 25), eta=1.0, tower="enemy_left")
    d = POLICY.decide(_state(units=[own, toward_enemy]), 10.0)
    assert d.mode == "hold"


# 3 -----------------------------------------------------------------------------------------------------------------
def test_enemy_unit_on_our_half_without_eta_defends():
    d = POLICY.decide(_state(units=[_enemy((10, 10))]), 10.0)
    assert d.act and d.mode == "defend" and d.lane == "middle"


def test_no_eta_urgency_depends_on_depth():
    deep, shallow = _enemy((3, 6)), _enemy((15, 14))                # ETA 0 vs 2.0
    d = POLICY.decide(_state(units=[shallow, deep]), 10.0)
    assert d.threat is deep and d.lane == "left" and "0.0" in d.reason
    d = POLICY.decide(_state(units=[shallow]), 10.0)
    assert d.threat is shallow and d.lane == "right" and "2.0" in d.reason


def test_unit_on_our_half_with_long_eta_still_defends():
    d = POLICY.decide(_state(units=[_enemy((5, 14), eta=9.0)]), 10.0)
    assert d.mode == "defend" and d.lane == "left"


def test_threat_without_tile_uses_the_tower_lane():
    d = POLICY.decide(_state(units=[{**_enemy((0, 0), eta=2.0, tower="own_right"), "tile": None}]), 10.0)
    assert d.mode == "defend" and d.lane == "right"


# 4 -----------------------------------------------------------------------------------------------------------------
def test_full_elixir_leaks_with_no_lane():
    d = POLICY.decide(_state(elixir=10.0), 10.0)
    assert d.act and d.mode == "leak" and d.lane is None
    assert not POLICY.decide(_state(elixir=9.4), 10.0).act
    assert POLICY.decide(_state(elixir=9.5), 10.0).mode == "leak"


def test_leak_plays_away_from_the_occupied_lane():
    right = _state(elixir=10.0, units=[_enemy((14, 25), eta=9.0, tower="own_right"), _enemy((13, 28))])
    d = POLICY.decide(right, 10.0)
    assert d.mode == "leak" and d.lane == "left"
    left = _state(elixir=10.0, units=[_enemy((3, 25))])
    assert POLICY.decide(left, 10.0).lane == "right"
    both = _state(elixir=10.0, units=[_enemy((3, 25)), _enemy((14, 25))])
    assert POLICY.decide(both, 10.0).lane is None
    middle = _state(elixir=10.0, units=[_enemy((9, 25))])
    assert POLICY.decide(middle, 10.0).lane is None


def test_defend_beats_leak():
    d = POLICY.decide(_state(elixir=10.0, units=[_enemy((5, 9), eta=2.0)]), 10.0)
    assert d.mode == "defend"


# 5 -----------------------------------------------------------------------------------------------------------------
def test_punish_the_opposite_lane_of_the_last_opponent_play():
    plays = [{"t": 5.0, "card": "giant", "tile": [3, 25]}, {"t": 8.0, "card": "knight", "tile": [14, 24]}]
    d = POLICY.decide(_state(elixir=7.0, opp_elixir=2.5, opp_plays=plays), 10.0)
    assert d.act and d.mode == "punish" and d.lane == "left"
    # the order of the list does not matter, the newest play with a tile does
    d = POLICY.decide(_state(elixir=7.0, opp_elixir=2.5, opp_plays=plays[::-1]), 10.0)
    assert d.mode == "punish" and d.lane == "left"


def test_punish_needs_own_elixir_and_a_low_opponent_estimate():
    plays = [{"t": 8.0, "card": "knight", "tile": [14, 24]}]
    assert POLICY.decide(_state(elixir=4.0, opp_elixir=2.5, opp_plays=plays), 10.0).mode == "hold"
    assert POLICY.decide(_state(elixir=7.0, opp_elixir=3.5, opp_plays=plays), 10.0).mode == "hold"
    assert POLICY.decide(_state(elixir=7.0, opp_elixir=None, opp_plays=plays), 10.0).mode == "hold"
    assert POLICY.decide(_state(elixir=6.0, opp_elixir=3.0, opp_plays=plays), 10.0).mode == "punish"


def test_punish_without_a_play_tile_or_in_the_middle_has_no_lane():
    assert POLICY.decide(_state(elixir=7.0, opp_elixir=2.0), 10.0).lane is None
    mid = [{"t": 8.0, "card": "knight", "tile": [9, 24]}]
    assert POLICY.decide(_state(elixir=7.0, opp_elixir=2.0, opp_plays=mid), 10.0).lane is None
    no_tile = [{"t": 8.0, "card": "knight", "tile": None}]
    d = POLICY.decide(_state(elixir=7.0, opp_elixir=2.0, opp_plays=no_tile), 10.0)
    assert d.mode == "punish" and d.lane is None


# 6 -----------------------------------------------------------------------------------------------------------------
def test_fresh_opponent_play_still_on_their_side_waits():
    state = _state(elixir=5.0, units=[_enemy((5, 25))], opp_plays=[{"t": 9.5, "card": "knight", "tile": [5, 25]}])
    d = POLICY.decide(state, 10.0)
    assert not d.act and d.mode == "hold"
    # the same unit becomes a threat once it nears the tower
    state["units"] = [_enemy((5, 14), eta=3.0)]
    assert POLICY.decide(state, 12.0).mode == "defend"


# 7 -----------------------------------------------------------------------------------------------------------------
def test_spells_without_an_enemy_in_their_lane_and_zone_are_dropped():
    state = _state(units=[_enemy((5, 9), eta=2.0, card="musketeer")])   # left lane, mid zone, worth a fireball (4 >= 4)
    d = POLICY.decide(state, 10.0)
    cands = _cands("fireball|left|mid", "fireball|right|mid", "zap|left|back", "knight|left|mid", "giant|left|back")
    out = POLICY.filter_candidates(cands, d, state)
    assert _keys(out) == ["fireball|left|mid", "knight|left|mid", "giant|left|back"]


def test_no_blind_spells_in_any_mode():
    cands = _cands("fireball|left|bridge", "zap|right|mid")
    empty = _state(elixir=10.0)
    for mode, lane in (("leak", None), ("hold", None), ("none", None), ("punish", None), ("defend", "left")):
        out = POLICY.filter_candidates(cands, ActDecision(True, mode, lane, "x"), empty)
        assert out == [], mode


def test_defend_keeps_the_threat_lane_and_defensive_zones():
    state = _state(units=[_enemy((5, 9), eta=2.0)])
    d = POLICY.decide(state, 10.0)
    cands = _cands("knight|right|mid", "knight|left|enemy_half", "archers|left|bridge", "giant|left|back",
                   "musketeer|middle|mid", "knight|left|mid")
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["archers|left|bridge", "giant|left|back",
                                                                "knight|left|mid"]      # generator order kept


def test_defend_spell_may_use_any_zone_in_the_threat_lane():
    state = _state(units=[_enemy((5, 20), eta=3.0, card="musketeer")])   # left lane, enemy_half zone
    d = POLICY.decide(state, 10.0)
    cands = _cands("fireball|left|enemy_half", "knight|left|enemy_half", "knight|left|mid")
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["fireball|left|enemy_half", "knight|left|mid"]


def test_defend_falls_back_to_kg_counters_then_to_the_spell_filtered_list():
    state = _state(units=[_enemy((5, 9), eta=2.0)])
    d = POLICY.decide(state, 10.0)
    cands = _cands("knight|right|mid", "giant|middle|back", "zap|right|mid", kg=("giant|middle|back",))
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["giant|middle|back"]
    no_kg = _cands("knight|right|mid", "giant|middle|back", "zap|right|mid")
    assert _keys(POLICY.filter_candidates(no_kg, d, state)) == ["knight|right|mid", "giant|middle|back"]


def test_leak_prefers_back_and_bridge_and_the_free_lane():
    state = _state(elixir=10.0, units=[_enemy((14, 25), eta=9.0, tower="own_right")])
    d = POLICY.decide(state, 10.0)
    assert d.mode == "leak" and d.lane == "left"
    cands = _cands("knight|left|mid", "knight|right|back", "giant|left|back", "archers|left|bridge",
                   "musketeer|left|enemy_half")
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["giant|left|back", "archers|left|bridge"]
    none_fit = _cands("knight|left|mid", "knight|right|back")
    assert _keys(POLICY.filter_candidates(none_fit, d, state)) == ["knight|left|mid", "knight|right|back"]


def test_punish_keeps_only_bridge_and_enemy_half_troops_in_the_lane():
    plays = [{"t": 8.0, "card": "knight", "tile": [14, 24]}]
    state = _state(elixir=7.0, opp_elixir=2.5, opp_plays=plays, units=[_enemy((3, 20))])
    d = POLICY.decide(state, 10.0)
    assert d.mode == "punish" and d.lane == "left"
    cands = _cands("knight|left|mid", "giant|left|bridge", "fireball|left|enemy_half", "knight|right|bridge",
                   "archers|left|enemy_half")
    out = POLICY.filter_candidates(cands, d, state)
    assert _keys(out) == ["giant|left|bridge", "archers|left|enemy_half"]                # spells never punish
    assert POLICY.filter_candidates(_cands("knight|left|mid", "knight|right|bridge"), d, state) == []


def test_punish_without_a_lane_accepts_any_lane():
    state = _state(elixir=7.0, opp_elixir=2.0)
    d = POLICY.decide(state, 10.0)
    cands = _cands("knight|right|bridge", "knight|left|mid", "giant|middle|enemy_half")
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["knight|right|bridge", "giant|middle|enemy_half"]


def test_filter_accepts_candidate_dicts_and_empty_input():
    state = _state(units=[_enemy((5, 9), eta=2.0)])
    d = POLICY.decide(state, 10.0)
    assert POLICY.filter_candidates([], d, state) == []
    as_dicts = [{"key": "knight|left|mid", "score": 1.0, "sources": {}}, {"key": "knight|right|mid", "score": 0.5}]
    assert POLICY.filter_candidates(as_dicts, d, state) == [as_dicts[0]]


# 8 (MAC-Y2: aimed and worthwhile spells) -----------------------------------------------------------------------------
def _hold(mode="hold", lane=None, threat=None):
    return ActDecision(mode != "hold", mode, lane, "x", threat=threat)


def test_fireball_needs_a_target_worth_its_elixir_and_ignores_tanks():
    lone_giant = _state(units=[_enemy((5, 9), card="giant")])
    cands = _cands("fireball|left|mid")
    assert POLICY.filter_candidates(cands, _hold(), lone_giant) == []
    assert POLICY.target_tile("fireball|left|mid", lone_giant) is None
    # a giant with a musketeer 1 tile away: the fireball goes to the musketeer, the giant is not a member
    pair = _state(units=[_enemy((4, 9), card="giant"), _enemy((5, 9), card="musketeer")])
    assert _keys(POLICY.filter_candidates(cands, _hold(), pair)) == ["fireball|left|mid"]
    assert POLICY.target_tile("fireball|left|mid", pair) == [5.0, 9.0]
    assert POLICY.spell_target("fireball", "left", "mid", pair) == ([5.0, 9.0], 4.0, 1)


def test_zap_value_count_and_swarm_rule():
    bats = _state(units=[_enemy((3, 8), card="bats"), _enemy((4, 10), card="bats")])       # distinct value 2 >= 2
    assert POLICY.target_tile("zap|left|mid", bats) == [3.5, 9.0]
    assert POLICY.spell_target("zap", "left", "mid", bats)[1:] == (2.0, 2)
    skeleton = _state(units=[_enemy((3, 8), card="skeletons")])                              # value 1 < 2, count 1
    assert POLICY.filter_candidates(_cands("zap|left|mid"), _hold(), skeleton) == []
    archers = _state(units=[_enemy((3, 8), card="archers"), _enemy((4, 9), card="archers")])
    assert POLICY.target_tile("fireball|left|mid", archers) is None                          # distinct: 3 < 4, count 2
    unknown = _state(units=[_enemy((3, 8), card=None), _enemy((4, 9), card=None), _enemy((4, 10), card=None)])
    assert _keys(POLICY.filter_candidates(_cands("zap|left|mid"), _hold(), unknown)) == ["zap|left|mid"]
    assert POLICY.spell_target("zap", "left", "mid", unknown)[1:] == (0.0, 3)                 # swarm at value 0
    two_unknown = _state(units=[_enemy((3, 8), card=None), _enemy((4, 9), card=None)])
    assert POLICY.target_tile("zap|left|mid", two_unknown) is None


def test_spell_cluster_counts_units_of_other_lanes_and_spells_and_picks_the_best_centre():
    # musketeer in the left lane, bats just across the lane border: both are hit, the aim is their mean
    state = _state(units=[_enemy((5, 9), card="musketeer"), _enemy((7, 9), card="bats")])
    assert POLICY.spell_target("zap", "left", "mid", state) == ([6.0, 9.0], 6.0, 2)
    # a unit of ours and an enemy spell object are never targets
    own = {**_enemy((5, 9), card="musketeer"), "side": "own"}
    assert POLICY.target_tile("zap|left|mid", _state(units=[own, _enemy((5, 9), card="fireball")])) is None
    # two equal clusters in the lane and zone: the one closer to our side (smaller row) wins, whatever the order
    two = _state(units=[_enemy((3, 11), card="musketeer"), _enemy((5, 6), card="musketeer")])
    assert POLICY.target_tile("fireball|left|mid", two) == [5.0, 6.0]
    # a denser cluster beats a closer one
    dense = _state(units=[_enemy((5, 6), card="musketeer"), _enemy((3, 11), card="musketeer"),
                          _enemy((4, 10), card="bats")])
    assert POLICY.target_tile("zap|left|mid", dense) == [3.5, 10.5]


def test_spell_target_leads_the_moving_unit_and_stays_in_the_arena():
    ahead = _state(units=[_enemy((5, 9), card="musketeer", pred=(5, 11))])               # 2 tiles ahead in 2 s
    assert POLICY.target_tile("fireball|left|mid", ahead) == [5.0, 10.0]                  # lead 1.0 s -> 1.0 tile
    assert POLICY.target_tile("zap|left|mid", ahead) == [5.0, 9.3]                        # lead 0.3 s -> 0.3 tile
    assert POLICY.target_tile("arrows|left|mid", ahead) == [5.0, 9.8]                     # lead 0.8 s -> 0.8 tile
    edge = _state(units=[_enemy((17, 31), card="musketeer", pred=(17, 40))])
    assert POLICY.target_tile("fireball|right|enemy_half", edge) == [17.0, 31.0]
    # the key still names the lane and zone of the unit's current tile
    assert POLICY.target_tile("fireball|left|bridge", ahead) is None


def test_target_tile_is_none_for_everything_but_damage_spells():
    state = _state(units=[_enemy((5, 9), card="musketeer")])
    for key in ("knight|left|mid", "rage|left|mid", "zap|right|mid", "zap|left", "zap", "not-a-card|left|mid"):
        assert POLICY.target_tile(key, state) is None, key
    assert POLICY.target_tile("zap|left|mid", state) == [5.0, 9.0]


def test_defend_drops_a_spell_aimed_away_from_the_threat():
    knight = _enemy((5, 6), eta=2.0)                                                       # left lane, mid zone
    far = [_enemy((9, 24), card=None), _enemy((10, 24), card=None), _enemy((9, 25), card=None)]   # their half
    state = _state(units=[knight, *far])
    d = POLICY.decide(state, 10.0)
    assert d.mode == "defend" and d.threat is knight
    cands = _cands("zap|middle|enemy_half", "knight|left|mid")
    assert _keys(POLICY.no_blind_spells(cands, state)) == ["zap|middle|enemy_half", "knight|left|mid"]   # a fine swarm
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["knight|left|mid"]
    # the fallbacks use the same rule: an off-threat spell does not come back through the spell-filtered list
    assert _keys(POLICY.filter_candidates(_cands("zap|middle|enemy_half", "giant|right|mid"), d, state)) \
        == ["giant|right|mid"]
    # a cluster in the threat's own lane and zone but far from it is dropped too
    apart = _state(units=[_enemy((5, 3), eta=1.0, card="musketeer"), *[_enemy((5, 11), card=None)
                                                                         for _ in range(3)]])
    d = POLICY.decide(apart, 10.0)
    assert d.mode == "defend" and d.threat["tile"] == [5, 3]
    assert POLICY.filter_candidates(_cands("zap|left|mid"), d, apart) == []


def test_defend_keeps_a_spell_on_the_threat_cluster_and_aims_at_it():
    knight, archers = _enemy((5, 6), eta=1.5), _enemy((4, 6), eta=3.0, card="archers")
    state = _state(units=[knight, archers])
    d = POLICY.decide(state, 10.0)
    assert d.threat is knight
    assert _keys(POLICY.filter_candidates(_cands("zap|left|mid", "zap|right|mid"), d, state)) == ["zap|left|mid"]
    assert POLICY.target_tile("zap|left|mid", state) == [4.5, 6.0]


def test_defend_drops_a_spell_when_the_threat_is_a_tank():
    giant = _enemy((5, 6), eta=1.0, card="giant")
    state = _state(units=[giant, _enemy((4, 6), eta=2.0, card="musketeer")])
    d = POLICY.decide(state, 10.0)
    assert d.mode == "defend" and d.threat is giant
    cands = _cands("fireball|left|mid", "knight|left|mid")
    assert _keys(POLICY.no_blind_spells(cands, state)) == ["fireball|left|mid", "knight|left|mid"]   # musketeer is a target
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["knight|left|mid"]
    assert POLICY.filter_candidates(_cands("fireball|left|mid"), ActDecision(True, "defend", "left", "x"), state) == []


def test_non_damage_spells_keep_the_lane_and_zone_rule():
    giant = _state(units=[_enemy((5, 9), card="giant")])
    cands = _cands("rage|left|mid", "rage|right|mid", "clone|left|back", "fireball|left|mid")
    assert _keys(POLICY.no_blind_spells(cands, giant)) == ["rage|left|mid"]
    assert POLICY.target_tile("rage|left|mid", giant) is None
    d = POLICY.decide(giant, 10.0)
    assert d.mode == "defend" and _keys(POLICY.filter_candidates(_cands("rage|left|mid"), d, giant)) == ["rage|left|mid"]


def test_learned_policy_delegates_target_tile_and_filters_with_the_new_rule():
    policy = LearnedActPolicy(None, CARDS)
    state = _state(units=[_enemy((5, 25), eta=6.0, card="musketeer"), _enemy((14, 25), card="skeletons")])
    assert policy.target_tile("fireball|left|enemy_half", state) == POLICY.target_tile("fireball|left|enemy_half", state)
    assert policy.target_tile("fireball|left|enemy_half", state) == [5.0, 25.0]
    assert policy.target_tile("zap|right|enemy_half", state) is None            # one skeleton is not worth a zap
    assert policy.target_tile("knight|left|mid", state) is None
    cands = _cands("fireball|left|enemy_half", "zap|right|enemy_half", "knight|right|back")
    d = ActDecision(True, "learned", None, "learned act p=0.90")
    assert _keys(policy.filter_candidates(cands, d, state)) == ["fireball|left|enemy_half", "knight|right|back"]


def test_act_policy_eval_scores_and_report(tmp_path):
    from tools.eval import eval_act_policy as ev

    s = ev.score([(1, 1), (1, 0), (0, 0), (0, 1), (0, 0)])
    assert s["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
    assert s["accuracy"] == pytest.approx(0.6) and s["act_rate"] == pytest.approx(0.4)
    assert s["balanced_accuracy"] == pytest.approx((0.5 + 2 / 3) / 2) and s["pro_play_rate"] == pytest.approx(0.4)
    assert ev.score([])["accuracy"] is None

    def rec(kind, typ, state):
        return {"split": "val", "kind": kind, "t": 10.0, "state": state, "type": typ}

    records = [rec("play", "play", _state(units=[_enemy((5, 9), eta=2.0)])),       # defend
               rec("periodic", "hold", _state(elixir=5.0)),                        # hold
               rec("periodic", "unknown", _state(elixir=10.0))]                    # skipped
    res = ev.evaluate(records, CARDS)["val"]
    assert res["all"]["n"] == 2 and res["all"]["pro_play_rate"] == 0.5
    assert res["all"]["policies"]["reactive"]["accuracy"] == 1.0
    assert res["all"]["policies"]["always"]["accuracy"] == 0.5
    assert res["all"]["reactive_modes"] == {"defend": 1, "hold": 1}
    assert res["periodic"]["n"] == 1 and res["periodic"]["policies"]["reactive"]["confusion"]["tn"] == 1
    assert "reactive" in ev.format_markdown({"val": res})


def test_thresholds_are_configurable():
    strict = ReactivePolicy(CARDS, defend_eta_s=1.0, leak_elixir=8.0)
    state = _state(elixir=8.0, units=[_enemy((5, 25), eta=3.0)])
    assert strict.decide(state, 10.0).mode == "leak"
    assert POLICY.decide(state, 10.0).mode == "defend"
