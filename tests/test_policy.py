"""ReactivePolicy (MAC-W, MAC-AC C1, MAC-AD): decide() modes, defend discipline, deploy windows and adequate answers,
answered pushes, candidate filtering, no blind spells - hand-made states, fake engines and the real (model-free)
tactical engine, no models."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.candidates import Candidate  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live_state import LiveConfig, LiveStateBuilder  # noqa: E402
from clashbot.policy import ActDecision, LearnedActPolicy, ReactivePolicy, seconds_per_elixir  # noqa: E402
from clashbot.tactics import (DEFAULT_WEIGHTS, WEIGHTS_PATH, Push, TacticalEngine, TacticalMove,  # noqa: E402
                              TacticalReport)

CARDS = load_cards()
# The real tactical engine, pinned to its hand-set default weights: the tests below read its scores (which card is
# adequate, when to save), so a retuned clashbot/resources/tactics_weights.json must not move them.
ENGINE = TacticalEngine(CARDS, weights=DEFAULT_WEIGHTS)
POLICY = ReactivePolicy(CARDS, engine=ENGINE)


@pytest.fixture(autouse=True)
def _forget_commitments():
    """The shared policy remembers the card it chose against a push for 4 s (MAC-AD); every test starts clean."""
    POLICY._acted.clear()
    LEGACY._acted.clear()


class _BrokenEngine:
    def analyse(self, state, t=None):
        raise RuntimeError("engine down")


LEGACY = ReactivePolicy(CARDS, engine=_BrokenEngine())     # engine failure -> today's defend rule (lane / zone tests)


STOP = {"bomb-tower": "high", "p-e-k-k-a": "high", "magic-archer": "medium", "knight": "medium", "zap": "spell",
        "ice-spirit": "low", "fire-spirit": "low"}          # the stopping class the real engine gives these cards


def _move(card, score, intent="defend", stop=None, value=None):
    return TacticalMove(card, f"{card}|left|mid", [4.5, 8.5], intent, "x", score, 1.0,
                        stop=stop or STOP.get(card, "medium"), value=value)


class FakeEngine:
    """One push holding the enemy units of the state (eta_s as given; a heavy one needs a high / medium stopping card),
    the defend moves in `moves` the own elixir pays (cost <= elixir + 0.5, like legal_cards); every call is recorded
    as (own elixir, enemy unit count)."""

    def __init__(self, moves, eta_s, lane="left", heavy=True, needs=("tank", "building_targeter")):
        self.moves, self.eta_s, self.lane, self.heavy, self.needs = moves, eta_s, lane, heavy, list(needs)
        self.calls = []

    def analyse(self, state, t=None):
        enemies = [u for u in state["units"] if u["side"] == "enemy"]
        elixir = state["own"]["elixir"]
        self.calls.append((elixir, len(enemies)))
        moves = [m for m in self.moves if CARDS[m.card]["elixir_cost"] <= elixir + 0.5]
        pushes = [Push(self.lane, enemies, 5.0, self.eta_s, list(self.needs), "fake", heavy=self.heavy)] \
            if enemies else []
        return TacticalReport(pushes, moves, moves[0] if moves else None, None, [])


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
    d = LEGACY.decide(_state(units=[unit]), 10.0)
    assert d.act and d.mode == "defend" and d.lane == "left" and d.threat is unit and d.allowed_cards is None
    assert "knight" in d.reason and "left" in d.reason and "3.0" in d.reason
    d = POLICY.decide(_state(units=[unit]), 10.0)                      # the real engine: an adequate answer is at hand
    assert d.act and d.mode == "defend" and d.lane == "left" and d.threat is unit
    assert "knight" in d.reason and "left" in d.reason and "knight" in d.allowed_cards


def test_far_eta_on_the_enemy_half_holds():
    d = POLICY.decide(_state(units=[_enemy((5, 25), eta=12.0)]), 10.0)         # beyond the 10 s watch horizon
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
def test_enemy_unit_deep_on_our_half_without_eta_defends():
    d = POLICY.decide(_state(units=[_enemy((10, 8))]), 10.0)
    assert d.act and d.mode == "defend" and d.lane == "middle"


def test_no_eta_unit_counts_only_in_the_back_of_our_half():
    deep, shallow = _enemy((3, 6)), _enemy((15, 14))                # row <= 8: ETA 0; further forward: no threat
    d = LEGACY.decide(_state(units=[shallow, deep]), 10.0)
    assert d.threat is deep and d.lane == "left" and "0.0" in d.reason
    assert POLICY.decide(_state(units=[shallow]), 10.0).mode == "hold"
    assert POLICY.decide(_state(units=[_enemy((10, 9))]), 10.0).mode == "hold"          # row 9: no longer a threat
    assert [u for _, u, _ in POLICY._threats(_state(units=[shallow, deep]))] == [deep]


def test_unit_on_our_half_with_long_eta_is_watched_not_answered():
    state = _state(units=[_enemy((5, 14), eta=9.0)])
    assert POLICY._threats(state) == []
    assert [(eta, lane) for eta, _, lane in POLICY._watch(state)] == [(9.0, "left")]
    d = POLICY.decide(state, 10.0)               # a push 9 s out: nothing is played yet, no card's window is open
    assert not d.act and d.mode == "save" and d.reason.startswith("waiting for the") and "knight left" in d.reason


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
    d = LEGACY.decide(state, 10.0)
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
    d = LEGACY.decide(state, 10.0)
    cands = _cands("knight|right|mid", "knight|left|enemy_half", "archers|left|bridge", "giant|left|back",
                   "musketeer|middle|mid", "knight|left|mid")
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["archers|left|bridge", "giant|left|back",
                                                                "knight|left|mid"]      # generator order kept


def test_defend_spell_may_use_any_zone_in_the_threat_lane():
    state = _state(units=[_enemy((5, 20), eta=3.0, card="musketeer")])   # left lane, enemy_half zone
    d = LEGACY.decide(state, 10.0)
    cands = _cands("fireball|left|enemy_half", "knight|left|enemy_half", "knight|left|mid")
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["fireball|left|enemy_half", "knight|left|mid"]


def test_defend_falls_back_to_kg_counters_then_to_the_spell_filtered_list():
    state = _state(units=[_enemy((5, 9), eta=2.0)])
    d = LEGACY.decide(state, 10.0)
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
    d = LEGACY.decide(state, 10.0)
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
    d = LEGACY.decide(apart, 10.0)
    assert d.mode == "defend" and d.threat["tile"] == [5, 3]
    assert POLICY.filter_candidates(_cands("zap|left|mid"), d, apart) == []


def test_defend_keeps_a_spell_on_the_threat_cluster_and_aims_at_it():
    knight, archers = _enemy((5, 6), eta=1.5), _enemy((4, 6), eta=3.0, card="archers")
    state = _state(units=[knight, archers])
    d = LEGACY.decide(state, 10.0)
    assert d.threat is knight
    assert _keys(POLICY.filter_candidates(_cands("zap|left|mid", "zap|right|mid"), d, state)) == ["zap|left|mid"]
    assert POLICY.target_tile("zap|left|mid", state) == [4.5, 6.0]


def test_defend_drops_a_spell_when_the_threat_is_a_tank():
    giant = _enemy((5, 6), eta=1.0, card="giant")
    state = _state(units=[giant, _enemy((4, 6), eta=2.0, card="musketeer")])
    d = LEGACY.decide(state, 10.0)
    assert d.mode == "defend" and d.threat is giant
    cands = _cands("fireball|left|mid", "knight|left|mid")
    assert _keys(POLICY.no_blind_spells(cands, state)) == ["fireball|left|mid", "knight|left|mid"]   # musketeer is a target
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["knight|left|mid"]
    assert POLICY.filter_candidates(_cands("fireball|left|mid"), ActDecision(True, "defend", "left", "x"), state) == []


def test_non_damage_spells_keep_the_lane_and_zone_rule():
    giant = _state(units=[_enemy((5, 8), card="giant")])
    cands = _cands("rage|left|mid", "rage|right|mid", "clone|left|back", "fireball|left|mid")
    assert _keys(POLICY.no_blind_spells(cands, giant)) == ["rage|left|mid"]
    assert POLICY.target_tile("rage|left|mid", giant) is None
    d = LEGACY.decide(giant, 10.0)
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
    strict = ReactivePolicy(CARDS, engine=_BrokenEngine(), defend_eta_s=1.0, leak_elixir=8.0)
    state = _state(elixir=8.0, units=[_enemy((5, 25), eta=3.0)])
    assert strict.decide(state, 10.0).mode == "leak"        # no engine: the unit 3 s out is only watched
    assert LEGACY.decide(state, 10.0).mode == "defend"
    assert POLICY.decide(state, 10.0).mode == "defend"      # the engine's push is 3 s out: a card's window is open


# 9 (MAC-AC C1: defend discipline) ---------------------------------------------------------------------------------
HAND = ["ice-spirit", "bomb-tower", "fire-spirit", "magic-archer"]


def _giant(eta=None, tile=(14, 7), **kw):
    return {**_enemy(tile, eta=eta, tower="own_right", card="giant"), **kw}


class _EmptyEngine:
    def analyse(self, state, t=None):
        return TacticalReport([], [], None, "nothing", [])


class _ElsewhereEngine:
    """One push, in the right lane, that does not hold any unit of the state."""

    def analyse(self, state, t=None):
        other = _enemy((14, 9), eta=2.0)
        return TacticalReport([Push("right", [other], 3.0, 2.0, [], "x")], [_move("zap", 0.9)], None, None, [])


def test_threats_drop_retreating_far_unknown_and_spell_units():
    retreating = _giant(eta=2.0, heading="retreating")
    unknown_far = _enemy((5, 9), eta=8.0, card=None)
    healer = _enemy((9, 10), eta=20.6, card="battle-healer")
    fireball = _enemy((5, 9), eta=1.0, card="fireball")
    state = _state(units=[retreating, unknown_far, healer, fireball], hand=HAND)
    assert POLICY._threats(state) == []
    d = POLICY.decide(state, 10.0)
    assert not d.act and d.mode == "hold"
    # the watch list: the unknown unit lands within 10 s; a retreating unit, a spell and a 20 s unit are not watched
    assert [u for _, u, _ in POLICY._watch(state)] == [unknown_far]


def test_unknown_unit_is_a_threat_only_within_four_seconds_and_with_an_eta():
    near = _enemy((5, 9), eta=3.5, card=None)
    assert [u for _, u, _ in POLICY._threats(_state(units=[near]))] == [near]
    assert POLICY._threats(_state(units=[_enemy((5, 9), eta=4.5, card=None)])) == []
    assert POLICY._threats(_state(units=[_enemy((5, 5), card=None)])) == []        # deep on our half, but no ETA


def test_graveyard_is_a_threat_but_other_spells_are_not():
    assert len(POLICY._threats(_state(units=[_enemy((5, 9), eta=2.0, card="graveyard")]))) == 1
    assert POLICY._threats(_state(units=[_enemy((5, 9), eta=2.0, card="poison")])) == []


def test_watch_list_is_the_non_threats_within_ten_seconds_soonest_first():
    soon, late, far = _giant(eta=6.0), _enemy((5, 12), eta=9.5, card="musketeer"), _enemy((5, 20), eta=10.5)
    state = _state(units=[late, far, soon, _enemy((5, 9), eta=2.0)])
    assert [(eta, u["card"]) for eta, u, _ in POLICY._watch(state)] == [(6.0, "giant"), (9.5, "musketeer")]
    assert [eta for eta, _, _ in POLICY._threats(state)] == [2.0]


def test_seconds_per_elixir_by_phase():
    assert [seconds_per_elixir(p) for p in ("single_elixir", "double_elixir", "triple_elixir", "overtime")] \
        == [2.8, 1.4, 0.93, 0.93]
    assert seconds_per_elixir(None) == seconds_per_elixir("what") == 2.8


def _save_policy(eta_s, extra=(), **kw):
    engine = FakeEngine([_move("bomb-tower", 0.9), *extra, _move("ice-spirit", 0.5)], eta_s)
    return engine, ReactivePolicy(CARDS, engine=engine, **kw)


def test_far_push_with_a_better_card_affordable_in_time_saves():
    engine, policy = _save_policy(9.5)
    state = _state(elixir=2.0, units=[_giant()], hand=HAND)          # double elixir: bomb-tower needs 2 x 1.4 = 2.8 s
    d = policy.decide(state, 10.0)
    assert not d.act and d.mode == "save" and d.lane is None and d.allowed_cards is None
    assert d.reason.startswith("save for bomb-tower") and "2.8s" in d.reason and "eta 9.5s" in d.reason
    assert d.threat is state["units"][0]
    assert sorted(engine.calls) == [(2.0, 1), (10.0, 1)]             # now and "as if affordable" (own elixir 10)
    # single elixir: 2 x 2.8 = 5.6 s, still in time at eta 9.5, but no longer at eta 6.5 (5.6 + 1.0 > 6.5): the spirit
    # only slows the giant, so the policy still does not tap it (MAC-AD) and waits
    single = {**state, "phase": "single_elixir"}
    d = policy.decide(single, 10.0)
    assert d.mode == "save" and "5.6s" in d.reason
    d = _save_policy(6.5)[1].decide(single, 10.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("no adequate answer in hand for giant")


def test_emergency_plays_the_best_affordable_card_in_any_case():
    state = _state(elixir=2.0, units=[_giant()], hand=HAND)
    d = _save_policy(1.2)[1].decide(state, 10.0)
    assert d.act and d.mode == "defend" and d.lane == "right" and d.allowed_cards == frozenset({"ice-spirit"})
    assert "emergency" in d.reason
    # not an emergency, and too late to wait for the bomb tower (2.8 + 1.0 > 3.0): a card that only slows the giant
    # is not played (MAC-AD); it goes when the push is within the emergency horizon
    d = _save_policy(3.0)[1].decide(state, 10.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("no adequate answer in hand")
    d = _save_policy(3.0, emergency_eta_s=3.0)[1].decide(state, 10.0)
    assert d.act and d.allowed_cards == frozenset({"ice-spirit"}) and "emergency" in d.reason


def test_adequate_answer_in_hand_defends_and_narrows_the_cards():
    _, policy = _save_policy(5.5, extra=[_move("musketeer", 0.8), _move("fire-spirit", 0.85)])
    state = _state(elixir=5.0, units=[_giant()], hand=HAND)
    d = policy.decide(state, 10.0)
    assert d.act and d.mode == "defend" and d.lane == "right"
    # within 0.15 of the best (0.9 vs 0.8) and with its window open (ranged: 6 s); the fire spirit only slows the giant
    assert d.allowed_cards == frozenset({"bomb-tower", "musketeer"})
    assert d.reason == "defend giant right: bomb-tower (high)"
    cands = _cands("ice-spirit|left|mid", "bomb-tower|right|mid", "musketeer|right|mid", "fire-spirit|right|mid")
    assert _keys(policy.filter_candidates(cands, d, state)) == ["bomb-tower|right|mid", "musketeer|right|mid"]


def test_filter_applies_allowed_cards_before_lane_zone_and_to_tactical_candidates():
    state = _state(units=[_enemy((5, 9), eta=2.0)])
    d = ActDecision(True, "defend", "left", "x", allowed_cards=frozenset({"knight"}))
    tac = Candidate("giant|right|enemy_half", 1.0, {"tac": 1.0})
    tac_ok = Candidate("knight|right|enemy_half", 0.9, {"tac": 1.0})
    plain = _cands("knight|left|mid", "archers|left|mid")
    out = POLICY.filter_candidates([tac, tac_ok, *plain], d, state)
    assert _keys(out) == ["knight|right|enemy_half", "knight|left|mid"]     # tactical: exempt from lane / zone only
    # nothing of the allowed cards: [] (the loop records hold_policy), no kg / spell fallback
    assert POLICY.filter_candidates(_cands("archers|left|mid", "giant|left|back", kg=("giant|left|back",)), d, state) == []
    # an allowed card outside the lane / zone rules falls back to its own candidates, never to other cards
    assert _keys(POLICY.filter_candidates(_cands("knight|right|mid", "archers|left|mid"), d, state)) == ["knight|right|mid"]
    # no allowed_cards: unchanged
    assert _keys(POLICY.filter_candidates(plain, ActDecision(True, "defend", "left", "x"), state)) \
        == ["knight|left|mid", "archers|left|mid"]


def test_no_answer_in_hand_holds():
    d = ReactivePolicy(CARDS, engine=FakeEngine([], 2.0)).decide(_state(elixir=5.0, units=[_giant()], hand=HAND), 10.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("no adequate answer in hand for giant")
    # only an unaffordable answer, and no time to wait for it
    late = FakeEngine([_move("bomb-tower", 0.9)], 2.0)
    d = ReactivePolicy(CARDS, engine=late).decide(_state(elixir=2.0, units=[_giant()], hand=HAND), 10.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("no adequate answer in hand")


def test_bats_with_an_affordable_zap_defend():
    bats = [_enemy((5, 10), eta=2.0, card="bats"), _enemy((6, 10), card="bats"), _enemy((4, 10), card="bats")]
    d = POLICY.decide(_state(elixir=5.0, units=bats), 10.0)             # the real engine: zap is the adequate answer
    assert d.act and d.mode == "defend" and d.lane == "left" and "zap" in d.allowed_cards
    swarm = FakeEngine([_move("zap", 1.1), _move("archers", 0.5)], 2.0, heavy=False, needs=("air", "swarm"))
    d = ReactivePolicy(CARDS, engine=swarm).decide(_state(elixir=5.0, units=bats), 10.0)
    assert d.act and d.mode == "defend" and d.allowed_cards == frozenset({"zap"})


def test_engine_failure_keeps_todays_behaviour():
    state = _state(units=[_enemy((5, 20), eta=3.0)])
    d = LEGACY.decide(state, 10.0)
    assert d == ActDecision(True, "defend", "left", "defend knight left eta 3.0s", threat=state["units"][0])
    # no pushes, or a push that does not hold the threat
    for engine in (_EmptyEngine(), _ElsewhereEngine()):
        d = ReactivePolicy(CARDS, engine=engine).decide(state, 10.0)
        assert d.act and d.mode == "defend" and d.allowed_cards is None and d.reason.endswith("eta 3.0s")

    class Flaky(FakeEngine):                 # the "as if affordable" call fails
        def analyse(self, state, t=None):
            if state["own"]["elixir"] >= 10:
                raise RuntimeError("boom")
            return super().analyse(state, t)

    d = ReactivePolicy(CARDS, engine=Flaky([_move("zap", 0.9)], 3.0)).decide(state, 10.0)
    assert d.allowed_cards is None and d.reason.endswith("eta 3.0s")


def test_default_engine_is_built_lazily_and_a_failing_build_falls_back(monkeypatch):
    import clashbot.tactics as tactics

    policy = ReactivePolicy(CARDS)
    assert policy.engine is None
    policy.decide(_state(), 10.0)                                          # no threat: no engine needed
    assert policy.engine is None
    policy.decide(_state(units=[_enemy((5, 9), eta=2.0)]), 10.0)
    assert isinstance(policy.engine, tactics.TacticalEngine) and policy.engine.cards is CARDS

    def boom(*a, **k):
        raise OSError("roles file missing")

    monkeypatch.setattr(tactics, "TacticalEngine", boom)
    broken = ReactivePolicy(CARDS)
    d = broken.decide(_state(units=[_enemy((5, 9), eta=2.0)]), 10.0)
    assert d.act and d.mode == "defend" and d.allowed_cards is None and broken._engine_failed


def test_defend_answers_only_the_push_of_the_threat():
    # knight in the left lane (the threat), a giant in the right lane: the giant's answer is not the knight's
    knight, giant = _enemy((4, 9), eta=2.0), _giant(eta=3.0, tile=(14, 9))
    state = _state(elixir=5.0, units=[knight, giant], hand=HAND)
    d = POLICY.decide(state, 10.0)
    assert d.threat is knight and d.act and d.mode == "defend"        # not "save for bomb-tower" (the giant's answer)

    class TwoLanes(FakeEngine):
        """Left push: the knight (magic-archer); right push: the giant (bomb-tower). Alone, only that lane's moves."""

        def analyse(self, state, t=None):
            enemies = [u for u in state["units"] if u["side"] == "enemy"]
            self.calls.append((state["own"]["elixir"], len(enemies)))
            pushes = [p for p in (Push("left", [knight], 3.0, 2.0, [], "k"),
                                  Push("right", [giant], 5.0, 3.0, ["tank"], "g", heavy=True))
                      if p.units[0] in enemies]
            moves = ([_move("magic-archer", 0.6)] if knight in enemies else []) \
                + ([_move("bomb-tower", 1.6)] if giant in enemies else [])
            moves = [m for m in moves if CARDS[m.card]["elixir_cost"] <= state["own"]["elixir"] + 0.5]
            return TacticalReport(pushes, moves, moves[0] if moves else None, None, [])

    engine = TwoLanes([], 0.0)
    d = ReactivePolicy(CARDS, engine=engine).decide(state, 10.0)
    assert d.act and d.mode == "defend" and d.allowed_cards == frozenset({"magic-archer"})
    assert engine.calls[0] == (5.0, 2) and (5.0, 1) in engine.calls and (10.0, 1) in engine.calls


def _punish_state(elixir, watch_unit, opp_elixir=2.5):
    plays = [{"t": 8.0, "card": "knight", "tile": [14, 24]}]
    return _state(elixir=elixir, opp_elixir=opp_elixir, opp_plays=plays, units=[watch_unit], hand=HAND)


def test_punish_never_spends_the_elixir_of_the_next_defence():
    watched = _giant(eta=8.0, tile=(14, 13))                          # lands in 8 s: not a threat, but watched
    state = _punish_state(8.0, watched)
    assert POLICY._threats(state) == [] and len(POLICY._watch(state)) == 1
    assert CARDS["p-e-k-k-a"]["elixir_cost"] == 7
    dear = ReactivePolicy(CARDS, engine=FakeEngine([_move("p-e-k-k-a", 0.9), _move("ice-spirit", 0.3)], 8.0))
    d = dear.decide(_punish_state(6.0, watched), 10.0)             # only the 7-elixir pekka answers it adequately
    assert not d.act and d.mode == "save" and "giant" in d.reason and "7 elixir" in d.reason
    d = dear.decide(_punish_state(7.0, watched), 10.0)
    assert d.act and d.mode == "punish" and d.lane == "left"          # enough for the answer: punish as before
    # a cheap adequate answer: the reserve is small, the punish goes ahead
    cheap = ReactivePolicy(CARDS, engine=FakeEngine([_move("magic-archer", 0.9), _move("p-e-k-k-a", 0.5)], 8.0))
    assert cheap.decide(_punish_state(6.0, watched), 10.0).mode == "punish"
    # no watched unit: the punish rule is untouched and the engine is never asked
    engine = FakeEngine([_move("p-e-k-k-a", 0.9)], 8.0)
    d = ReactivePolicy(CARDS, engine=engine).decide(_state(elixir=6.0, opp_elixir=2.5, hand=HAND), 10.0)
    assert d.mode == "punish" and engine.calls == []
    # an engine failure reserves nothing
    assert ReactivePolicy(CARDS, engine=_BrokenEngine()).decide(_punish_state(6.0, watched), 10.0).mode == "punish"


def test_real_engine_giant_far_off_with_two_elixir_is_not_answered_with_the_cheapest_card():
    # MAC-AC test 7. The giant (row 13, heading right, 9 s from our right tower) is only watched, not a threat yet
    giant = _giant(eta=9.0, tile=(14, 13), heading="toward own_right")
    state = _state(elixir=2.0, units=[giant], hand=HAND)
    d = POLICY.decide(state, 10.0)          # MAC-AD: the push is listed from 10 s out, and the policy saves for the bomb tower
    assert not d.act and d.mode == "save" and d.reason.startswith("save for bomb-tower vs giant")
    d = ReactivePolicy(CARDS, engine=ENGINE, defend_eta_s=10.0).decide(state, 10.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("save for bomb-tower vs giant")
    # 3.5 s out with 3 elixir: the bomb tower is 1 elixir (1.4 s) away and still in time
    near = _state(elixir=3.0, units=[_giant(eta=3.5, tile=(14, 10))], hand=HAND)
    d = POLICY.decide(near, 10.0)
    assert not d.act and d.mode == "save" and "bomb-tower" in d.reason
    # with the 4 elixir for it, it is the adequate answer and a spirit is no longer one
    near4 = {**near, "own": {**near["own"], "elixir": 4.0}}
    d = POLICY.decide(near4, 10.0)
    assert d.act and d.mode == "defend" and d.allowed_cards == frozenset({"bomb-tower"})
    cands = _cands("ice-spirit|right|mid", "bomb-tower|middle|mid", "magic-archer|right|back")
    assert _keys(POLICY.filter_candidates(cands, d, near4)) == ["bomb-tower|middle|mid"]
    # the push is on the tower: whatever is affordable goes
    d = POLICY.decide(_state(elixir=2.0, units=[_giant(eta=1.2, tile=(14, 10))], hand=HAND), 20.0)
    assert d.act and d.mode == "defend" and "emergency" in d.reason and len(d.allowed_cards) == 1


def test_learned_policy_does_not_override_a_save():
    class Scorer:
        calls = 0

        def probabilities(self, state, questions):
            Scorer.calls += 1
            return {"act": [0.0, 1.0]}

    _, rules = _save_policy(9.5)
    policy = LearnedActPolicy(Scorer(), CARDS, rules=rules)
    d = policy.decide(_state(elixir=2.0, units=[_giant()], hand=HAND), 10.0, "CTX")
    assert not d.act and d.mode == "save" and Scorer.calls == 0


def test_constructor_knobs_move_the_save_boundary():
    engine, _ = _save_policy(4.0)
    state = _state(elixir=2.0, units=[_giant()], hand=HAND)
    assert ReactivePolicy(CARDS, engine=engine, save_margin_s=1.0).decide(state, 10.0).mode == "save"      # 2.8+1.0 <= 4
    d = ReactivePolicy(CARDS, engine=engine, save_margin_s=1.5).decide(state, 10.0)     # 2.8+1.5 > 4: no time to wait
    assert not d.act and d.reason.startswith("no adequate answer in hand")
    # the adequate margin sets how many cards are allowed: the musketeer is 0.3 behind the bomb tower
    both = FakeEngine([_move("bomb-tower", 0.9), _move("musketeer", 0.6)], 5.5)
    five = _state(elixir=5.0, units=[_giant()], hand=HAND)
    assert ReactivePolicy(CARDS, engine=both).decide(five, 10.0).allowed_cards == frozenset({"bomb-tower"})
    wide = ReactivePolicy(CARDS, engine=both, adequate_margin=0.5)
    assert wide.decide(five, 10.0).allowed_cards == frozenset({"bomb-tower", "musketeer"})


# 10 (MAC-AD: real defence - deploy windows, adequate = stops the push, our own answers) ----------------------------------
def _tuned_weights():
    try:
        return json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))["weights"]
    except (OSError, ValueError, KeyError):
        return None


TUNED = _tuned_weights()
# the scenarios hold with the hand-set defaults and with the committed tuned file, loaded explicitly
WEIGHT_SETS = [pytest.param(DEFAULT_WEIGHTS, id="default"),
               pytest.param(TUNED, id="tuned", marks=pytest.mark.skipif(TUNED is None, reason="no tuned weights file"))]
AD_HAND = ["bomb-tower", "ice-spirit", "magic-archer", "berserker"]


def _ad(weights, **kw):
    """A fresh policy (it remembers what it chose against a push) on the real engine with these weights."""
    return ReactivePolicy(CARDS, engine=TacticalEngine(CARDS, weights=weights), **kw)


def _bridge_giant(eta=9.0, tile=(14, 15)):
    return _giant(eta=eta, tile=tile, heading="toward own_right")


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ad_giant_on_the_bridge_is_answered_by_the_bomb_tower_not_a_spirit(weights):
    policy = _ad(weights)
    state = _state(elixir=5.0, units=[_bridge_giant()], hand=AD_HAND)           # row 15, 9 s from the right tower
    d = policy.decide(state, 100.0)
    assert d.act and d.mode == "defend" and d.lane == "right" and d.threat is state["units"][0]
    assert "bomb-tower" in d.allowed_cards and "ice-spirit" not in d.allowed_cards
    assert d.reason == "defend giant right: bomb-tower (high)"
    best = next(m for m in policy.engine.analyse(state, 100.0).defence if m.card == "bomb-tower")
    assert best.tile == [10.5, 10.0] and best.stop == "high" and best.reason.startswith("pulls giant to the centre")
    cands = [Candidate("bomb-tower|middle|mid", 1.0, {"tac": 1.0}),
             Candidate("ice-spirit|right|bridge", 0.9, {"tac": 1.0})]
    assert _keys(policy.filter_candidates(cands, d, state)) == ["bomb-tower|middle|mid"]


DECK8 = ["bomb-tower", "ice-spirit", "magic-archer", "berserker", "fire-spirit", "goblin-barrel", "royal-delivery",
         "wall-breakers"]


def _gs(t, elixir, units):
    """A perception GameState as a plain dict (what LiveStateBuilder.build takes)."""
    return {"t": t, "frame_index": int(t * 10), "readiness": "match", "match_id": 1, "match_clock": "1:20",
            "match_seconds": 100.0, "phase": "double_elixir",
            "own": {"elixir": elixir, "hand": list(AD_HAND), "hand_conf": [0.9] * 4, "next_card": "fire-spirit",
                    "towers": {"king": 4000, "left": 3000, "right": 2500}},
            "opponent": {"elixir_est": 4.0, "towers": {"king": 4800, "left": 4000, "right": 3500}, "deck_known": []},
            "units": units}


def _giant_obs(row, eta):
    motion = {"heading": "toward own_right", "speed": 1.0, "pred_2s": [14, row - 2],
              "eta_tower": {"tower": "own_right", "s": eta}, "tile": [14, row]}
    return {"class": "giant", "side": "enemy", "tile": [14, row], "conf": 0.9, "bbox": [0, 0, 1, 1], "track_id": 7,
            "category": "unit", "motion": motion}


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ad_a_second_after_our_bomb_tower_tap_the_push_is_answered(weights):
    policy = _ad(weights)
    builder = LiveStateBuilder(LiveConfig(own_deck=DECK8, hand_tracker=True))
    state, _ = builder.build(_gs(100.0, 6.0, [_giant_obs(15, 9.0)]))
    d = policy.decide(state, 100.0)
    assert d.act and "bomb-tower" in d.allowed_cards
    pull = next(m for m in policy.engine.analyse(state, 100.0).defence if m.card == "bomb-tower").tile
    # the executor taps the bomb tower (slot 0) on the pull tile; the HUD and the perception have not seen it yet
    builder.note_own_play("bomb-tower", 0, 100.2, tile=pull)
    state, _ = builder.build(_gs(101.2, 6.0, [_giant_obs(14, 8.0)]))
    assert [(p["card"], p["tile"], p["detect_source"]) for p in state["own"]["recent_plays"]] \
        == [("bomb-tower", pull, "executor")]
    assert state["own"]["elixir"] < 3.0                                        # min(read, expected): the HUD still lags
    d = _ad(weights).decide(state, 101.2)                                      # a policy that remembers nothing
    assert not d.act and d.mode == "save" and d.reason == "push right answered by bomb-tower"
    # without the tap the push is not answered
    other = LiveStateBuilder(LiveConfig(own_deck=DECK8, hand_tracker=True))
    state2, _ = other.build(_gs(101.2, 6.0, [_giant_obs(14, 8.0)]))
    assert _ad(weights).decide(state2, 101.2).act

    class Scorer:
        calls = 0

        def probabilities(self, state, questions):
            Scorer.calls += 1
            return {"act": [0.0, 1.0]}

    # and the learned head cannot override the hold
    learned = LearnedActPolicy(Scorer(), CARDS, rules=_ad(weights))
    assert learned.decide(state, 101.2, "CTX").mode == "save" and Scorer.calls == 0


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ad_giant_far_off_with_two_elixir_saves_for_the_bomb_tower_and_taps_no_spirit(weights):
    policy = _ad(weights)
    state = _state(elixir=2.0, units=[_bridge_giant()], hand=AD_HAND)
    d = policy.decide(state, 100.0)
    assert not d.act and d.mode == "save" and d.allowed_cards is None
    assert d.reason == "save for bomb-tower vs giant right: 2.8s, eta 9.0s"


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ad_balloon_in_the_left_lane_is_answered_by_the_magic_archer(weights):
    policy = _ad(weights)
    balloon = _enemy((3, 13), eta=6.0, tower="own_left", card="balloon")
    state = _state(elixir=5.0, units=[balloon], hand=AD_HAND)
    d = policy.decide(state, 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left" and d.reason == "defend balloon left: magic-archer (medium)"
    assert d.allowed_cards == frozenset({"magic-archer"})                       # not the fire / ice spirit
    cards = [m.card for m in policy.engine.analyse(state, 100.0).defence]
    assert "bomb-tower" not in cards and {"magic-archer", "ice-spirit"} <= set(cards)   # a tower cannot hit air
    # 6.5 s out the ranged window (6 s) is not open yet
    late = _state(elixir=5.0, units=[_enemy((3, 14), eta=6.5, tower="own_left", card="balloon")], hand=AD_HAND)
    d = _ad(weights).decide(late, 100.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("waiting for the magic-archer window")


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ad_balloon_at_the_tower_with_only_a_spirit_affordable_is_an_emergency(weights):
    balloon = _enemy((3, 10), eta=1.2, tower="own_left", card="balloon")
    d = _ad(weights).decide(_state(elixir=1.5, units=[balloon], hand=AD_HAND), 100.0)
    assert d.act and d.mode == "defend" and d.reason == "defend balloon left: ice-spirit (emergency)"
    assert d.allowed_cards == frozenset({"ice-spirit"})
    # 2.4 s out the spirit is only a slow-down and the magic archer is not affordable in time: no spirit
    far = _enemy((3, 10), eta=2.4, tower="own_left", card="balloon")
    d = _ad(weights).decide(_state(elixir=1.5, units=[far], hand=AD_HAND), 100.0)
    assert not d.act and d.mode == "save"


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ad_swarm_may_be_answered_with_a_low_stopping_card(weights):
    bats = [_enemy((5, 10), eta=2.0, card="bats"), _enemy((6, 10), card="bats"), _enemy((4, 10), card="bats")]
    hand = ["zap", "fire-spirit", "knight", "giant"]
    d = _ad(weights).decide(_state(elixir=5.0, units=bats, hand=hand), 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left"
    assert d.allowed_cards & {"zap", "fire-spirit"}
    d = _ad(weights).decide(_state(elixir=5.0, units=bats, hand=["fire-spirit", "knight", "giant", "archers"]), 100.0)
    assert d.act and "fire-spirit" in d.allowed_cards                           # a low stop is fine against a swarm


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ad_a_push_our_bomb_tower_stands_against_is_answered(weights):
    ally = {"unit_class": "bomb-tower", "card": "bomb-tower", "side": "ally", "tile": [10.5, 10], "eta_tower": None}
    giant = _giant(eta=8.0, tile=(14, 13), heading="toward own_right")              # the tower is 4.6 tiles from the giant
    policy = _ad(weights)
    state = _state(elixir=5.0, units=[giant, ally], hand=AD_HAND)
    push = policy.engine.analyse(state, 100.0).pushes[0]
    assert push.answered and push.answered_by == ["bomb-tower"]
    d = policy.decide(state, 100.0)
    assert not d.act and d.mode == "save" and d.reason == "push right answered by bomb-tower"
    # at the tower itself (<= 1 s) an answered push is acted on all the same
    close = _state(elixir=5.0, units=[_giant(eta=0.8, tile=(14, 10)), ally], hand=AD_HAND)
    assert _ad(weights).decide(close, 100.0).act


def test_ad_a_push_without_engine_flags_is_read_from_the_roles_of_its_units():
    class Bare(FakeEngine):
        def analyse(self, state, t=None):
            report = super().analyse(state, t)
            for p in report.pushes:                                  # a stand-in engine that sets no flags
                p.heavy, p.needs = False, []
            return report

    moves = [_move("bomb-tower", 0.9), _move("ice-spirit", 0.8)]
    d = ReactivePolicy(CARDS, engine=Bare(moves, 5.0)).decide(_state(elixir=5.0, units=[_giant()], hand=HAND), 10.0)
    assert d.act and d.allowed_cards == frozenset({"bomb-tower"})      # the giant is heavy, a targeter: the spirit is low
    swarm = [_enemy((5, 10), eta=2.0, card="bats"), _enemy((6, 10), card="bats"), _enemy((4, 10), card="bats")]
    d = ReactivePolicy(CARDS, engine=Bare([_move("zap", 1.1), _move("ice-spirit", 1.0)], 2.0)).decide(
        _state(elixir=5.0, units=swarm, hand=HAND), 10.0)
    assert d.act and d.allowed_cards == frozenset({"zap", "ice-spirit"})     # bats are not heavy: any card stops them


def test_ad_melee_card_waits_for_its_window():
    hand = ["berserker", "ice-spirit", "fire-spirit", "magic-archer"]
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[_bridge_giant(eta=5.0)], hand=hand), 100.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("waiting for the berserker window vs giant right")
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[_bridge_giant(eta=4.0)], hand=hand), 100.0)
    assert d.act and d.allowed_cards == frozenset({"berserker"}) and d.reason == "defend giant right: berserker (medium)"


def test_ad_building_window_is_nine_seconds_or_a_lead_unit_on_our_half():
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[_bridge_giant(eta=9.5)], hand=AD_HAND), 100.0)
    assert not d.act and d.reason.startswith("waiting for the bomb-tower window")       # row 15, 9.5 s out
    on_half = _giant(eta=9.5, tile=(14, 13), heading="toward own_right")                 # across the river: pull it now
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[on_half], hand=AD_HAND), 100.0)
    assert d.act and "bomb-tower" in d.allowed_cards
    wide = _ad(DEFAULT_WEIGHTS, building_window_s=10.0)
    d = wide.decide(_state(elixir=5.0, units=[_bridge_giant(eta=9.5)], hand=AD_HAND), 100.0)
    assert d.act and "bomb-tower" in d.allowed_cards


def test_ad_low_stopping_cards_only_when_the_hand_holds_nothing_adequate_and_the_push_is_close():
    hand = ["ice-spirit", "fire-spirit", "magic-archer", "wall-breakers"]            # nothing here stops a giant
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=6.0, units=[_bridge_giant(eta=4.0)], hand=hand), 100.0)
    assert not d.act and d.reason == "no adequate answer in hand for giant right"
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=6.0, units=[_bridge_giant(eta=2.0)], hand=hand), 100.0)
    assert d.act and d.mode == "defend" and "nothing adequate in hand" in d.reason and len(d.allowed_cards) == 1
    assert d.allowed_cards <= {"ice-spirit", "fire-spirit"}


def test_ad_the_card_chosen_against_a_push_counts_as_answering_it_for_four_seconds():
    state = _state(elixir=5.0, units=[_bridge_giant()], hand=AD_HAND)
    policy = _ad(DEFAULT_WEIGHTS)
    d = policy.decide(state, 100.0)
    assert d.act and "bomb-tower" in d.allowed_cards
    d = policy.decide(state, 101.5)                          # no tap is visible yet, the choice stands in for it
    assert not d.act and d.mode == "save" and d.reason == "push right answered by bomb-tower"
    assert policy.decide(state, 104.5).act                   # 4.5 s on: the choice has expired
    off = _ad(DEFAULT_WEIGHTS, commit_s=0.0)
    off.decide(state, 100.0)
    assert off.decide(state, 101.5).act
    # a cheap card does not answer a push worth much more (a spirit is 1 of the giant's 5 elixir)
    spirit = _ad(DEFAULT_WEIGHTS)
    spirit._acted[("right", "giant")] = (100.0, "ice-spirit")
    assert spirit.decide(state, 101.0).act


def test_ad_pushes_beyond_ten_seconds_and_retreating_ones_are_not_answered():
    far = _state(elixir=5.0, units=[_bridge_giant(eta=10.5)], hand=AD_HAND)
    d = _ad(DEFAULT_WEIGHTS).decide(far, 100.0)
    assert not d.act and d.mode == "hold"
    back = _state(elixir=5.0, units=[_giant(eta=3.0, tile=(14, 10), heading="retreating")], hand=AD_HAND)
    assert _ad(DEFAULT_WEIGHTS).decide(back, 100.0).mode == "hold"


def test_ad_leak_follows_the_pushes():
    # a push in its window takes precedence over a full bar
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=10.0, units=[_bridge_giant()], hand=AD_HAND), 100.0)
    assert d.act and d.mode == "defend"
    # an answered push does not: the full bar leaks as before
    ally = {"unit_class": "bomb-tower", "card": "bomb-tower", "side": "ally", "tile": [10.5, 10], "eta_tower": None}
    state = _state(elixir=10.0, units=[_giant(eta=8.0, tile=(14, 13), heading="toward own_right"), ally], hand=AD_HAND)
    assert _ad(DEFAULT_WEIGHTS).decide(state, 100.0).mode == "leak"
