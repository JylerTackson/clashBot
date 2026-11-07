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
# MAC-AH: the MAC-W ... AG tests check the policy rules on the role-based engine, so ENGINE / POLICY / `_ad` run with the
# threat model off (weight `threat_model` 0 = the MAC-AG engine) and spells priced by elixir (`price_spells=False`);
# SIM_ENGINE / SIM_POLICY at the end of the file are the full MAC-AH stack (simulation, spells priced by what they kill).
ROLE_MODEL = {"threat_model": 0.0}
ENGINE = TacticalEngine(CARDS, weights={**DEFAULT_WEIGHTS, **ROLE_MODEL})
POLICY = ReactivePolicy(CARDS, engine=ENGINE, price_spells=False)


@pytest.fixture(autouse=True)
def _forget_commitments():
    """The shared policy remembers the card it chose against a push for 4 s (MAC-AD); every test starts clean."""
    POLICY._acted.clear()
    LEGACY._acted.clear()


class _BrokenEngine:
    def analyse(self, state, t=None):
        raise RuntimeError("engine down")


LEGACY = ReactivePolicy(CARDS, engine=_BrokenEngine(), price_spells=False)     # engine failure -> today's defend rule


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
    # (their units are 12 s out: a push 9 s out is pre-placed against at a full bar, MAC-AL P1.4)
    right = _state(elixir=10.0, units=[_enemy((14, 25), eta=12.0, tower="own_right"), _enemy((13, 28))])
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
    # a hand without a win condition: the leak keeps the zone and lane rule over any card
    state = _state(elixir=10.0, units=[_enemy((14, 25), eta=12.0, tower="own_right")],     # (12 s out: no push yet)
                   hand=["knight", "archers", "musketeer", "zap"])
    d = POLICY.decide(state, 10.0)
    assert d.mode == "leak" and d.lane == "left" and d.allowed_cards is None
    cands = _cands("knight|left|mid", "knight|right|back", "giant|left|back", "archers|left|bridge",
                   "musketeer|left|enemy_half")
    assert _keys(POLICY.filter_candidates(cands, d, state)) == ["giant|left|back", "archers|left|bridge"]
    none_fit = _cands("knight|left|mid", "knight|right|back")
    assert _keys(POLICY.filter_candidates(none_fit, d, state)) == ["knight|left|mid", "knight|right|back"]
    # with a win condition in hand the full bar goes into it (MAC-AG), and nothing else is offered
    wc = POLICY.decide(_state(elixir=10.0, units=[_enemy((14, 25), eta=12.0, tower="own_right")]), 10.0)
    assert wc.mode == "leak" and wc.lane == "left" and wc.allowed_cards == frozenset({"giant"})
    assert _keys(POLICY.filter_candidates(cands, wc, state)) == ["giant|left|back"]


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
    # no candidate at all: the allowed cards still reach the LLM as the engine's own candidates (MAC-AL P3.7)
    injected = POLICY.filter_candidates([], d, state)
    assert injected and {c.key.split("|")[0] for c in injected} == set(d.allowed_cards)
    assert all(c.sources == {"tac": 0.0} and c.tile == d.engine_tiles[c.key.split("|")[0]] for c in injected)
    as_dicts = [{"key": "knight|left|mid", "score": 1.0, "sources": {}}, {"key": "knight|right|mid", "score": 0.5}]
    kept = POLICY.filter_candidates(as_dicts, d, state)
    assert kept[0] == as_dicts[0] and as_dicts[1] not in kept                  # (the other allowed cards are the engine's)
    assert {c.key.split("|")[0] for c in kept[1:]} == set(d.allowed_cards) - {"knight"}


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
    # (MAC-AM M1: the lead is the spell's real delay - 0.5 s latency + the flight from our king tower at the KB's projectile
    # speed - at the unit's speed; the unit is out of reach of our towers, else it would stand and fight)
    ahead = _state(units=[_enemy((5, 13), card="musketeer", pred=(5, 15))])              # 1 tile/s toward their side
    assert POLICY.target_tile("fireball|left|bridge", ahead) == [5.0, 15.0]               # 0.5 + 15.1 / 10 = 2.0 s -> 2 tiles
    assert POLICY.target_tile("zap|left|bridge", ahead) == [5.0, 13.5]                    # an instant spell: 0.5 s -> 0.5 tile
    assert POLICY.target_tile("arrows|left|bridge", ahead) == [5.0, 14.3]                 # 0.5 + 14.9 / 18.3 = 1.3 s
    edge = _state(units=[_enemy((17, 30), card="musketeer", pred=(17, 40))])
    assert POLICY.target_tile("fireball|right|enemy_half", edge) == [17.0, 31.0]
    # the key still names the lane and zone of the unit's current tile
    assert POLICY.target_tile("fireball|left|mid", ahead) is None


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
    policy = LearnedActPolicy(None, CARDS, rules=ReactivePolicy(CARDS, price_spells=False))   # MAC-AH: elixir-priced spells
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
    assert policy.engine is None                                           # nothing is built before the first decision
    policy.decide(_state(), 10.0)                                          # the engine also reads the offence windows
    assert isinstance(policy.engine, tactics.TacticalEngine) and policy.engine.cards is CARDS

    def boom(*a, **k):
        raise OSError("roles file missing")

    monkeypatch.setattr(tactics, "TacticalEngine", boom)
    broken = ReactivePolicy(CARDS)
    assert broken.decide(_state(), 10.0).mode == "hold" and broken._engine_failed      # no engine: no offence window
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
    # no watched unit: the punish rule is untouched and the engine sees no enemy unit
    engine = FakeEngine([_move("p-e-k-k-a", 0.9)], 8.0)
    d = ReactivePolicy(CARDS, engine=engine).decide(_state(elixir=6.0, opp_elixir=2.5, hand=HAND), 10.0)
    assert d.mode == "punish" and engine.calls == [(6.0, 0)]
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
    return ReactivePolicy(CARDS, engine=TacticalEngine(CARDS, weights={**weights, **ROLE_MODEL}),
                          **{"price_spells": False, **kw})


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
    assert best.tile == [10.0, 10.0] and best.stop == "high" and best.reason.startswith("pulls giant to the centre")
    assert d.engine_tiles["bomb-tower"] == best.tile                              # the live loop places it there
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
    # 7 s out the ranged window (6 s, judged against the ETA minus the 0.5 s decision latency, MAC-AL P1.2) is not open yet
    late = _state(elixir=5.0, units=[_enemy((3, 14), eta=7.0, tower="own_left", card="balloon")], hand=AD_HAND)
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
    # (the 4.5 s window is judged against the ETA minus the 0.5 s decision latency: 5.5 s out is not open, 5 s is, MAC-AL P1.2)
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[_bridge_giant(eta=5.5)], hand=hand), 100.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("waiting for the berserker window vs giant right")
    assert _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[_bridge_giant(eta=5.0)], hand=hand), 100.0).act
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[_bridge_giant(eta=4.0)], hand=hand), 100.0)
    assert d.act and d.allowed_cards == frozenset({"berserker"}) and d.reason == "defend giant right: berserker (medium)"


def test_ad_building_window_opens_at_the_river_and_closes_when_the_threat_locks_on():
    # MAC-AG: the window is geometric. MAC-AM M3.3: it opens at the bridge approach - the targeter at row 20 or nearer,
    # heading to our side - not only after the river. Row 21 is not there yet, whatever the ETA says
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[_bridge_giant(eta=9.5, tile=(14, 21))],
                                           hand=AD_HAND), 100.0)
    assert not d.act and d.reason.startswith("waiting for the bomb-tower window")
    # a giant that walks away (retreating) is not pulled
    back = _giant(eta=9.5, tile=(14, 18), heading="retreating")
    assert not _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[back], hand=AD_HAND), 100.0).act
    for row in (20, 17, 16, 15, 13):                                                 # at the approach / across the river: pull it
        d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=5.0, units=[_bridge_giant(eta=9.5, tile=(14, row))],
                                               hand=AD_HAND), 100.0)
        assert d.act and "bomb-tower" in d.allowed_cards, row
    # at the tower (within 1 + 1 tiles of it) the giant has locked on: no pull, and the tower is not an answer
    locked = _state(elixir=5.0, units=[_bridge_giant(eta=1.0, tile=(14, 5.5))], hand=AD_HAND)
    d = _ad(DEFAULT_WEIGHTS).decide(locked, 100.0)
    assert d.act and d.allowed_cards and "bomb-tower" not in d.allowed_cards


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


# 11 (MAC-AG: offence windows, engine placement, flyers, pulls, no second card of a kind) ---------------------------------
OFFENCE_HAND = ["ram-rider", "musketeer", "electro-dragon", "zap"]
DECK47 = ["electro-dragon", "musketeer", "tesla", "fireball", "knight", "zap", "ram-rider", "mini-p-e-k-k-a"]


def _ally(tile, card="knight"):
    return {"unit_class": card, "card": card, "side": "ally", "tile": list(tile), "eta_tower": None}


def _counter_scene(elixir=7.0, extra=()):
    """Their giant (5 elixir) in the right lane 5 s out, answered by our tesla; our knight across the bridge, left."""
    giant = _enemy((14, 12), eta=5.0, tower="own_right", card="giant")
    return _state(elixir=elixir, units=[giant, _ally((11, 8), "tesla"), _ally((4, 10), "knight"), *extra],
                  hand=OFFENCE_HAND)


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ag_counter_plays_the_win_condition_in_the_open_lane_while_the_giant_is_answered(weights):
    policy = _ad(weights)
    d = policy.decide(_counter_scene(), 100.0)
    # the deck's win condition at the left bridge; split with the default weights, counter_push when the tuned file
    # scores that higher (both ask for the same key)
    intent = "split" if weights is DEFAULT_WEIGHTS else "counter_push"
    assert d.act and d.mode == "counter" and d.lane == "left" and d.reason == f"counter: {intent} ram-rider left"
    assert d.allowed_cards == frozenset({"ram-rider"}) and d.engine_tiles == {"ram-rider": [3.5, 14.0]}
    # the filter keeps the card's candidates; with none of them the engine's own move is offered
    cands = _cands("ram-rider|left|mid", "musketeer|left|back", "ram-rider|right|bridge")
    assert _keys(policy.filter_candidates(cands, d, _counter_scene())) == ["ram-rider|left|mid", "ram-rider|right|bridge"]
    only_others = policy.filter_candidates(_cands("musketeer|left|back", "zap|left|mid"), d, _counter_scene())
    assert [(c.key, c.tile, c.sources) for c in only_others] == [("ram-rider|left|bridge", [3.5, 14.0], {"tac": 0.0})]


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ag_an_unanswered_balloon_comes_first_and_the_reserve_keeps_the_answer_affordable(weights):
    balloon = _enemy((3, 13), eta=4.0, tower="own_left", card="balloon")
    scene = _counter_scene(elixir=5.0, extra=[balloon])
    scene["units"] = [u for u in scene["units"] if u.get("card") != "knight"] + [_ally((5, 10), "knight")]
    d = _ad(weights).decide(scene, 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left" and d.reason == "defend balloon left: musketeer (medium)"
    # a balloon 7 s out (no window open yet - 6 s + the 0.5 s latency - and not within 5 s): the counter is allowed only
    # above the answer's price
    far = _enemy((3, 14), eta=7.0, tower="own_left", card="balloon")
    hand = ["ram-rider", "musketeer", "electro-dragon", "knight"]            # the cheapest adequate answer costs 4

    def scene_at(elixir):
        giant = _enemy((14, 12), eta=7.0, tower="own_right", card="giant")
        units = [giant, _ally((11, 8), "tesla"), _ally((13, 10), "knight"), far]
        return _state(elixir=elixir, units=units, hand=hand)

    d = _ad(weights).decide(scene_at(8.0), 100.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("waiting for the musketeer window")
    d = _ad(weights).decide(scene_at(9.0), 100.0)
    assert d.act and d.mode == "counter" and d.reason == "counter: counter_push ram-rider right"
    # a balloon 4.5 s out is a defend matter, whatever the bar holds
    near = _enemy((3, 12), eta=4.5, tower="own_left", card="balloon")
    d = _ad(weights).decide(_state(elixir=9.0, units=[near, _ally((13, 10), "knight")], hand=hand), 100.0)
    assert d.act and d.mode == "defend"


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ag_a_full_bar_leaks_the_win_condition_but_a_push_at_the_tower_is_defended_first(weights):
    hand = ["ram-rider", "knight", "mini-p-e-k-k-a", "fireball"]
    # a push 6 s out whose answer (knight / mini pekka, window 4.5 s) cannot go yet: the full bar pre-places it instead of
    # leaking (MAC-AL P1.4) ...
    d = _ad(weights).decide(_state(elixir=10.0, units=[_enemy((3, 14), eta=6.0)], hand=hand), 100.0)
    assert d.act and d.mode == "defend" and d.reason.startswith("full bar: pre-place ") and d.lane == "left"
    assert d.allowed_cards and d.allowed_cards <= {"knight", "mini-p-e-k-k-a"}
    # ... while a hand with no card that answers it (no ranged card, no melee card) leaks the win condition
    leak_hand = ["ram-rider", "fireball", "zap", "ice-spirit"]
    d = _ad(weights).decide(_state(elixir=10.0, units=[_enemy((3, 14), eta=6.0)], hand=leak_hand), 100.0)
    assert d.act and d.mode == "leak" and d.allowed_cards == frozenset({"ram-rider"}) and d.lane == "right"
    assert d.reason == "elixir 10.0 near full, lane right" and d.engine_tiles == {"ram-rider": [14.5, 14.0]}
    cands = _cands("knight|right|back", "ram-rider|right|bridge", "mini-p-e-k-k-a|right|bridge")
    assert _keys(_ad(weights).filter_candidates(cands, d, _state(elixir=10.0, hand=leak_hand))) == ["ram-rider|right|bridge"]
    d = _ad(weights).decide(_state(elixir=10.0, units=[_enemy((3, 10), eta=2.0)], hand=hand), 100.0)
    assert d.act and d.mode == "defend" and d.allowed_cards & {"knight", "mini-p-e-k-k-a"}


def test_ag_decisions_carry_the_engines_tile_for_every_card_they_may_pick():
    policy = _ad(DEFAULT_WEIGHTS)
    giant = _enemy((14, 15), eta=9.0, tower="own_right", card="giant", pred=(14, 13))
    d = policy.decide(_state(elixir=5.0, units=[giant], hand=["tesla", "knight", "musketeer", "zap"]), 100.0)
    assert d.act and d.allowed_cards == frozenset({"tesla"}) and d.reason == "defend giant right: tesla (high)"
    assert set(d.engine_tiles) == {"tesla", "knight", "musketeer"}              # the zap has nothing to hit
    assert d.engine_tiles["tesla"] == [10.0, 10.0] and d.engine_tiles["musketeer"] == [13.0, 2.5]     # (the centre box, MAC-AM M3)
    assert _ad(DEFAULT_WEIGHTS).decide(_state(), 100.0).engine_tiles == {}          # holds carry no tiles


def test_ag_a_ground_only_card_does_not_answer_a_flyer_in_the_policy_either():
    bats = [_enemy((4, 14), eta=4.0, card="bats"), _enemy((3, 14), card="bats"), _enemy((5, 14), card="bats")]
    witch = _enemy((4, 20), eta=8.0, card="night-witch")
    hand = ["electro-dragon", "knight", "musketeer", "zap"]
    policy = _ad(DEFAULT_WEIGHTS)
    state = _state(elixir=7.0, units=bats + [witch], hand=hand)
    d = policy.decide(state, 100.0)
    assert d.act and d.mode == "defend" and "knight" not in d.allowed_cards
    push = policy.engine.analyse(state, 100.0).pushes[0]
    assert push.units[0]["card"] == "bats"
    for card, ok in (("zap", True), ("electro-dragon", True), ("musketeer", True), ("knight", False)):
        move = TacticalMove(card, f"{card}|left|mid", [4.5, 8.5], "defend", "x", 1.0, 1.0,
                            stop="spell" if card == "zap" else "medium", value=2.0 if card == "zap" else None)
        assert policy._adequate(move, push) is ok, card
    # a stand-in engine that proposes the knight all the same: the policy does not accept it
    fake = ReactivePolicy(CARDS, engine=FakeEngine([_move("knight", 1.2), _move("musketeer", 0.6)], 4.0,
                                                   heavy=False, needs=("air", "swarm")))
    d = fake.decide(_state(elixir=7.0, units=bats, hand=hand), 100.0)
    assert d.act and d.allowed_cards == frozenset({"musketeer"})


@pytest.mark.parametrize("weights", WEIGHT_SETS)
def test_ag_a_giant_at_the_tower_is_met_by_the_tank_killer_not_the_building(weights):
    hand = ["tesla", "mini-p-e-k-k-a", "musketeer", "knight"]
    attacking = _enemy((14, 5.5), eta=0.5, tower="own_right", card="giant")
    d = _ad(weights).decide(_state(elixir=8.0, units=[attacking], hand=hand), 100.0)
    assert d.act and d.mode == "defend" and d.reason == "defend giant right: mini-p-e-k-k-a (high)"
    assert "tesla" not in d.allowed_cards and d.engine_tiles["mini-p-e-k-k-a"] == [14.0, 4.5]
    # the pull is open from the river on: a giant across it at row 16 is answered by the tesla at its pull tile
    crossing = _enemy((14, 16), eta=10.0, tower="own_right", card="giant")
    d = _ad(weights).decide(_state(elixir=8.0, units=[crossing], hand=hand), 100.0)
    assert d.act and "tesla" in d.allowed_cards and d.engine_tiles["tesla"][0] <= 11.0


def test_ag_no_second_card_of_the_same_kind_against_a_push_we_just_answered():
    prince = _enemy((3, 12), eta=4.0, card="prince")                           # 5 elixir
    state = _state(elixir=8.0, units=[prince], hand=["knight", "mini-p-e-k-k-a", "musketeer", "fireball"])
    key = ("left", "prince")
    policy = _ad(DEFAULT_WEIGHTS)
    first = policy.decide(state, 100.0)
    assert first.act and first.mode == "defend"
    assert [t for t, _ in policy._commits[key]] == [100.0] and policy._commits[key][0][1] in first.allowed_cards

    def later(commits, t):
        policy._acted.clear()                                                  # only the repeat rule is under test
        policy._commits[key] = commits
        return policy.decide(state, t)

    # 1.2 s on the tap is not on the board yet and the knight alone does not answer the prince: a second melee card
    # (knight, mini-pekka) is out, a card of another kind is not
    d = later([(100.0, "knight")], 101.2)
    assert d.act and "musketeer" in d.allowed_cards and not {"knight", "mini-p-e-k-k-a"} & d.allowed_cards
    # the choice expires after 4 s; a push worth twice what we spent is answered again; the same instant counts nothing
    assert {"knight", "mini-p-e-k-k-a"} & later([(100.0, "knight")], 104.5).allowed_cards
    assert {"knight", "mini-p-e-k-k-a"} & later([(100.0, "ice-spirit")], 101.2).allowed_cards       # 5 >= 2 x 1
    assert {"knight", "mini-p-e-k-k-a"} & later([(100.0, "knight")], 100.0).allowed_cards
    # two kinds already spent: only the fireball is left, and its window (2.5 s) is not open yet
    d = later([(99.0, "knight"), (100.0, "musketeer")], 101.2)
    assert not d.act and d.mode == "save" and d.reason.startswith("waiting for the fireball window")
    off = _ad(DEFAULT_WEIGHTS, commit_s=0.0)
    off.decide(state, 100.0)
    assert off._commits == {}


def test_ag_learned_policy_passes_a_counter_through_without_asking_the_head():
    class Scorer:
        calls = 0

        def probabilities(self, state, questions):
            Scorer.calls += 1
            return {"act": [1.0, 0.0]}

    learned = LearnedActPolicy(Scorer(), CARDS, rules=_ad(DEFAULT_WEIGHTS))
    d = learned.decide(_counter_scene(), 100.0, "CTX")
    assert d.act and d.mode == "counter" and d.allowed_cards == frozenset({"ram-rider"}) and Scorer.calls == 0
    assert d.engine_tiles == {"ram-rider": [3.5, 14.0]}


# MAC-AH: the full stack - the threat model in the engine, adequacy by simulation, the reserve, spells priced by kills -------
from tests.test_tactics import _heading, _scenario  # noqa: E402


def _sim(**kw):
    """A fresh policy on the real engine with the threat model on and the default weights pinned."""
    return ReactivePolicy(CARDS, engine=TacticalEngine(CARDS, weights=DEFAULT_WEIGHTS), **kw)


def test_ah_the_balloon_is_answered_with_the_tesla_pull_and_the_covered_left_lane_waits():
    d = _sim().decide(_scenario(), 100.0)
    assert d.act and d.mode == "defend" and d.lane == "right" and d.reason == "defend balloon right: tesla (high)"
    assert d.allowed_cards == frozenset({"tesla"}) and d.engine_tiles["tesla"] == [10.0, 10.0]
    assert d.threat["card"] == "balloon"
    # without the balloon the left push (a night witch pack) is held by our electro dragon and musketeer: no card is spent
    d = _sim().decide(_scenario(balloon=False), 100.0)
    assert not d.act and d.mode == "save" and d.reason == "push left answered by electro-dragon, musketeer"


def test_ah_with_three_elixir_the_balloon_answer_is_saved_and_nothing_goes_on_the_left():
    state = _scenario(elixir=3.0, hand=("knight", "zap", "fireball", "musketeer"))
    d = _sim().decide(state, 100.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("save for musketeer vs balloon right")
    # no knight, no zap on the left lane, whatever the head says: the decision is a hold
    assert d.allowed_cards is None


def test_ah_a_push_the_tower_handles_is_a_hold_with_that_reason():
    bats = [_heading(_enemy((3, 14), eta=2.0, card="bats"), "advancing left lane") for _ in range(2)]
    d = _sim().decide(_state(elixir=7.0, units=bats, hand=["zap", "knight", "musketeer", "fireball"]), 10.0)
    assert not d.act and d.mode == "save" and d.reason == "tower handles bats left"
    # ... and it does not block a counter or a leak: a full bar still leaks
    full = _sim().decide(_state(elixir=10.0, units=bats, hand=["zap", "knight", "musketeer", "fireball"]), 10.0)
    assert full.act and full.mode == "leak"


def test_ah_a_smaller_push_may_not_spend_the_elixir_of_the_bigger_pushs_answer():
    hand = ("tesla", "knight", "musketeer", "zap")
    # the balloon is still far on their side (row 23, simulated ETA 17.25 s: the musketeer's 9 s air window and the tesla's
    # pull are not open yet, the 20 s reserve window is); a knight pushes the left. The answer kept for it is the tesla
    # (4 elixir, the better of the two 4-elixir answers): its pull opens when the balloon reaches row 20 (MAC-AM M3.3), 3 s
    # from now, less the 0.5 s latency; the double-elixir bar regenerates 2.5 / 1.4 = 1.79 by then: keep 4 - 1.79 = 2.21
    balloon = _heading(_enemy((16, 23), eta=15.0, tower="own_right", card="balloon"), "advancing right lane")
    knight = _heading(_enemy((4, 11), eta=4.0, card="knight"), "advancing left lane")

    def state(elixir):
        st = _scenario(left=False, balloon=False, allies=False, elixir=elixir, hand=hand)
        st["units"] = [balloon, knight]
        return st

    d = _sim().decide(state(5.0), 100.0)                      # 5 elixir: a knight (3) would leave 2 < 2.21
    assert not d.act and d.mode == "save" and d.reason.startswith("keep 2.2 elixir for the balloon")
    d = _sim().decide(state(8.0), 100.0)                      # 8 elixir: the knight leaves 5 >= 2.21
    assert d.act and d.mode == "defend" and d.lane == "left"
    assert all(c in ("knight", "musketeer", "tesla") for c in d.allowed_cards)
    cost = {c: CARDS[c]["elixir_cost"] for c in d.allowed_cards}
    assert all(8.0 - cost[c] >= 4 - 2.5 / 1.4 for c in cost)   # every allowed card leaves the reserve
    # a push that is more than RESERVE_WINDOW_S away (row 28: 22.25 s simulated) keeps no reserve
    far = _heading(_enemy((16, 28), eta=17.0, tower="own_right", card="balloon"), "advancing right lane")
    st = state(5.0)
    st["units"] = [far, knight]
    d = _sim().decide(st, 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left"


def test_ah_zap_on_a_lone_balloon_is_not_worth_it_and_the_filter_drops_it():
    policy = _sim()
    balloon = _heading(_enemy((3, 12), eta=6.0, tower="own_left", card="balloon"), "advancing left lane")
    state = _state(elixir=9.0, units=[balloon], hand=["zap", "musketeer", "tesla", "knight"])
    assert policy.spell_target("zap", "left", "bridge", state) is None
    assert policy.spell_target("zap", "left", "mid", state) is None
    assert policy.target_tile("zap|left|bridge", state) is None
    decision = policy.decide(state, 10.0)
    assert decision.act and "zap" not in (decision.allowed_cards or ())
    assert policy.filter_candidates(_cands("zap|left|bridge", "zap|left|mid"), ActDecision(True, "learned", None, "x"), state) == []
    assert policy.no_blind_spells(_cands("zap|left|bridge"), state) == []


def test_ah_spells_are_valued_by_what_they_kill():
    policy = _sim()
    # (MAC-AM M2: a 2-elixir zap needs 2 kills and 2.5 of value; a bat is 1/5 of its card's 2 elixir)
    bats = [_enemy((3, 8), card="bats"), _enemy((4, 10), card="bats")]
    assert policy.spell_target("zap", "left", "mid", _state(units=bats)) is None             # 2 kills, but worth 0.8
    witch = _state(units=[*bats, _enemy((4, 9), card="night-witch"), _enemy((3, 9), card="electro-wizard")])
    assert policy.spell_target("zap", "left", "mid", witch) is None                           # + a bit of the pair: 1.76
    five = [_enemy((3 + 0.3 * i, 8 + 0.4 * i), card="bats") for i in range(5)]
    assert policy.spell_target("zap", "left", "mid", _state(units=five)) is None             # a whole play of bats: 2.0 < 2.5
    goblins = [_enemy((3 + 0.3 * i, 9), card="spear-goblins") for i in range(3)]
    target = policy.spell_target("zap", "left", "mid", _state(units=five + goblins))        # + a whole play of goblins: 4.0
    assert target is not None and target[1] == pytest.approx(4.0) and target[2] == 8
    # a fireball on a pair of archers is not worth its 4 elixir (they are worth 3); with a musketeer next to them (4.9) it
    # is still short of the 5 a fireball must clear; with a night witch (a bit of it) as well it clears it
    archers = [_enemy((3, 8), card="archers"), _enemy((4, 9), card="archers")]
    assert policy.spell_target("fireball", "left", "mid", _state(units=archers)) is None
    mixed = _state(units=[*archers, _enemy((3, 9), card="musketeer")])
    assert policy.spell_target("fireball", "left", "mid", mixed) is None
    mixed = _state(units=[*archers, _enemy((3, 9), card="musketeer"), _enemy((4, 8), card="night-witch")])
    assert policy.spell_target("fireball", "left", "mid", mixed)[1] == pytest.approx(6.43, abs=0.01)   # 3 + 1.91 + 1.52
    # three skeletons (1 elixir) are not worth a zap (2), a lone giant never is
    assert policy.spell_target("zap", "left", "mid", _state(units=[_enemy((3, 8 + i * 0.3), card="skeletons")
                                                                   for i in range(3)])) is None
    # the old elixir-sum rule is one flag away
    old = _sim(price_spells=False)
    assert old.spell_target("zap", "left", "mid", _state(units=[_enemy((3, 8), card="knight")])) is not None
    assert policy.spell_target("zap", "left", "mid", _state(units=[_enemy((3, 8), card="knight")])) is None


def test_ah_a_move_without_a_simulation_verdict_keeps_the_role_rule():
    moves = [_move("ice-spirit", 0.5, stop="low"), _move("magic-archer", 0.9, stop="medium")]
    assert all(m.adequate is None for m in moves)
    fake = FakeEngine(moves, 3.0, lane="left", heavy=True)
    d = ReactivePolicy(CARDS, engine=fake).decide(_state(elixir=5.0, units=[_enemy((4, 10), eta=3.0)]), 10.0)
    assert d.act and d.allowed_cards == frozenset({"magic-archer"})
    # ... and a verdict from the simulation wins over the role class
    moves = [TacticalMove("ice-spirit", "ice-spirit|left|mid", [4.5, 8.5], "defend", "x", 0.5, 1.0, stop="low", adequate=True),
             TacticalMove("magic-archer", "magic-archer|left|mid", [4.5, 8.5], "defend", "x", 0.9, 1.0, stop="medium",
                          adequate=False)]
    d = ReactivePolicy(CARDS, engine=FakeEngine(moves, 2.0, lane="left", heavy=True)).decide(
        _state(elixir=5.0, units=[_enemy((4, 10), eta=2.0)]), 10.0)
    assert d.act and d.allowed_cards == frozenset({"ice-spirit"})


# MAC-AK K4b: the graph's placement reaches the live loop, the hero ability is an answer, the reserve for predicted cards ----
from clashbot.actions import ABILITY_KEY  # noqa: E402
from tests.test_tactics import GIANT_BRIDGE, FakeKG2, _ability_state, _ally, _ans, _kg_engine, _off  # noqa: E402
from tests.test_tactics import _enemy as _t_enemy  # noqa: E402


class FakeDeckKG:
    """The opponent-deck side of a KG2: `predict_opp_deck(seen, top)` from a fixed list (seen cards left out)."""

    def __init__(self, preds=(), boom=False):
        self.preds, self.boom, self.calls = list(preds), boom, []

    def predict_opp_deck(self, seen, top=8):
        self.calls.append(list(seen))
        if self.boom:
            raise RuntimeError("model down")
        return [(c, p) for c, p in self.preds if c not in seen][:top]


def _air_state(elixir=6.5, hand=("musketeer", "knight", "ice-spirit", "giant"), seen=("miner", "valkyrie"),
                  opp_elixir=2.0):
    st = _state(elixir=elixir, opp_elixir=opp_elixir, hand=list(hand))
    st["opponent"]["deck_known"] = list(seen)
    return st


def _reserve_policy(preds=(("balloon", 0.62), ("miner", 0.4)), **kw):
    kg = FakeDeckKG(preds, **kw)
    return ReactivePolicy(CARDS, engine=ENGINE, price_spells=False, kg=kg), kg


def test_k4b_a_predicted_balloon_keeps_the_air_answer_out_of_a_punish():
    policy, kg = _reserve_policy()
    state = _air_state()
    assert policy.air_reserve(state) == (4.0, "musketeer", "balloon", 0.62)         # the ground cards in hand are no answer
    d = policy.decide(state, 100.0)
    assert d.act and d.mode == "punish"                                            # a cheap card still fits the punish
    cands = _cands("knight|left|bridge", "ice-spirit|left|bridge", "giant|left|bridge", "musketeer|left|bridge")
    assert _keys(policy.filter_candidates(cands, d, state)) == ["ice-spirit|left|bridge"]    # 6.5 - 1 >= 4; knight leaves 3.5
    plain = ReactivePolicy(CARDS, engine=ENGINE, price_spells=False)                # the same without a graph
    assert plain.air_reserve(state) is None and _keys(plain.filter_candidates(cands, d, state)) == _keys(cands)
    # nothing leaves the elixir of the answer: the punish waits
    tight = _air_state(elixir=6.0, hand=("musketeer", "knight", "giant", "mini-p-e-k-k-a"))
    d = policy.decide(tight, 100.0)
    assert not d.act and d.mode == "save"
    assert d.reason == "keep 4 elixir for musketeer: opponent likely has balloon 0.62, no punish"
    assert plain.decide(tight, 100.0).mode == "punish"


def test_k4b_the_reserve_follows_the_prediction_and_the_hand():
    state = _air_state()
    low, _ = _reserve_policy(preds=(("balloon", 0.45),))
    assert low.air_reserve(state) is None                                          # under 0.5
    edge, _ = _reserve_policy(preds=(("bats", 0.5), ("balloon", 0.3)))
    assert edge.air_reserve(state)[2:] == ("bats", 0.5)                            # the likeliest air card counts
    for card in ("minions", "minion-horde", "lava-hound", "mega-minion"):
        assert _reserve_policy(preds=((card, 0.7),))[0].air_reserve(state)[2] == card
    ground, _ = _reserve_policy(preds=(("miner", 0.9), ("skeletons", 0.8)))
    assert ground.air_reserve(state) is None                                       # a likely miner is no air threat
    policy, kg = _reserve_policy()
    assert policy.air_reserve(_air_state(hand=("knight", "giant", "zap", "mini-p-e-k-k-a"))) is None   # no air answer
    cheap = policy.air_reserve(_air_state(hand=("electro-dragon", "tesla", "knight", "zap")))
    assert cheap[:2] == (4.0, "tesla")                                             # the cheapest answer sets the reserve
    assert policy.air_reserve(_air_state(hand=("electro-dragon", "knight", "zap", "giant")))[:2] == (5.0, "electro-dragon")
    assert policy.air_reserve(_air_state(hand=("ram-rider", "knight", "zap", "giant"))) is None   # a win condition is no answer
    # the balloon was already seen: the model does not predict a card it has seen
    assert _reserve_policy()[0].air_reserve(_air_state(seen=("miner", "balloon"))) is None
    # nothing seen yet: the model is not asked
    quiet, kg2 = _reserve_policy()
    assert quiet.air_reserve(_air_state(seen=())) is None and kg2.calls == []
    # the prediction is cached per set of seen cards
    policy.air_reserve(state)
    policy.air_reserve(state)
    assert kg.calls == [["miner", "valkyrie"]]
    # a failing model means no reserve, not a crash
    broken, _ = _reserve_policy(boom=True)
    assert broken.air_reserve(state) is None and broken.decide(state, 100.0).mode == "punish"


def test_k4b_the_reserve_is_for_non_urgent_plays_only():
    policy, _ = _reserve_policy()
    # a full bar leaks whatever it likes (counter and leak are excepted)
    full = _air_state(elixir=10.0, opp_elixir=6.0, hand=("musketeer", "knight", "mini-p-e-k-k-a", "zap"))
    leak = policy.decide(full, 100.0)
    assert leak.act and leak.mode == "leak" and policy.air_reserve(full) is not None
    cands = _cands("knight|left|back", "mini-p-e-k-k-a|left|bridge", "musketeer|left|back")
    assert _keys(policy.filter_candidates(cands, leak, full)) == _keys(cands)
    # an answer to a push is never held back by it: the musketeer itself may defend
    giant = _state(elixir=6.0, units=[_enemy((14, 15), eta=3.0, tower="own_right", card="giant")],
                   hand=["musketeer", "knight", "giant", "mini-p-e-k-k-a"])
    giant["opponent"]["deck_known"] = ["miner", "valkyrie"]
    d = policy.decide(giant, 100.0)
    assert d.act and d.mode == "defend"
    # a counter is not held either
    counter = policy.decide(_counter_scene(), 100.0)
    assert counter.mode == "counter" and policy.filter_candidates(
        _cands("ram-rider|left|bridge"), counter, _counter_scene()) != []


def test_k4b_a_learned_play_keeps_the_reserve_too():
    class Scorer:
        def probabilities(self, state, questions):
            return {"act": [0.0, 1.0]}

    policy, _ = _reserve_policy()
    learned = LearnedActPolicy(Scorer(), CARDS, rules=policy)
    state = _air_state(elixir=5.0, opp_elixir=6.0, hand=("musketeer", "knight", "ice-spirit", "giant"))
    d = learned.decide(state, 100.0, "CTX")
    assert d.act and d.mode == "learned"
    cands = _cands("knight|left|mid", "ice-spirit|left|mid", "musketeer|left|back")
    assert _keys(learned.filter_candidates(cands, d, state)) == ["ice-spirit|left|mid"]      # 5 - 1 >= 4 only
    assert _keys(policy.no_blind_spells(cands, state)) == ["ice-spirit|left|mid"]
    assert _keys(ReactivePolicy(CARDS, engine=ENGINE, price_spells=False).no_blind_spells(cands, state)) == _keys(cands)


def test_k4b_the_pro_offset_source_reaches_the_decision():
    table = {"giant": [_ans("knight", 0.5, pro=0.6, support=9, held=0.8, offset=_off(2.0, -3.0))]}
    engine = _kg_engine(table)
    policy = ReactivePolicy(CARDS, engine=engine)
    state = _state(elixir=9.0, units=[_enemy((15, 13), eta=3.0, tower="own_right", card="giant")],
                   hand=["knight", "tesla", "musketeer", "zap"])
    d = policy.decide(state, 100.0)
    assert d.act and d.mode == "defend"
    assert d.engine_tiles["knight"] == [13.0, 10.0] and d.tile_sources["knight"] == "pro_offset"
    assert d.tile_sources["tesla"] == "geometric"
    # without the graph every source is geometric; a counter carries its move's source
    plain = ReactivePolicy(CARDS, engine=ENGINE).decide(state, 100.0)
    assert set(plain.tile_sources.values()) == {"geometric"} and set(plain.tile_sources) == set(plain.engine_tiles)
    c = ReactivePolicy(CARDS, engine=ENGINE).decide(_counter_scene(), 100.0)
    assert c.tile_sources == {"ram-rider": "geometric"}


def test_k4b_a_policy_without_an_engine_gives_the_one_it_builds_the_graph():
    kg = FakeKG2({})
    policy = ReactivePolicy(CARDS, kg=kg)
    assert policy._get_engine().kg2 is kg
    late = ReactivePolicy(CARDS)
    late.attach_kg(kg)
    assert late.kg is kg and late._get_engine().kg2 is kg
    built = ReactivePolicy(CARDS, engine=TacticalEngine(CARDS))
    built.attach_kg(kg)
    assert built.engine.kg2 is kg


# the hero ability as an answer -----------------------------------------------------------------------------------------
def test_k4b_the_ability_is_offered_to_the_llm_as_an_allowed_card_when_it_is_the_answer():
    state = _ability_state([GIANT_BRIDGE, _ally((14, 5), "musketeer")])
    d = _sim().decide(state, 100.0)
    assert d.act and d.mode == "defend" and d.lane == "right"
    assert ABILITY_KEY in d.allowed_cards
    assert d.engine_tiles[ABILITY_KEY] == [14.0, 8.0] and d.tile_sources[ABILITY_KEY] == "ability"
    # no button, no ability: the decision is the card one
    for kw in ({"visible": None}, {"visible": False}, {"affordable": False}, {"elixir": 2.5}):
        d = _sim().decide(_ability_state([GIANT_BRIDGE, _ally((14, 5), "musketeer")], **kw), 100.0)
        assert ABILITY_KEY not in (d.allowed_cards or ()), kw
    assert ABILITY_KEY not in (_sim().decide(_ability_state([GIANT_BRIDGE]), 100.0).allowed_cards or ())   # not her
    # the ability counts for what it costs (3), and the candidate with its key passes the defend filter
    assert _sim()._cost(ABILITY_KEY) == 3.0
    d = _sim().decide(state, 100.0)
    cand = Candidate("ability:musketeer|right|mid", 0.0, {"ability": 1.0}, [14.0, 8.0], "hero ability")
    assert _sim().filter_candidates([cand], d, state) == [cand]
    assert _sim().filter_candidates([cand], ActDecision(True, "defend", "right", "x", allowed_cards=frozenset({"tesla"})),
                                    state) == []


def test_k4b_the_ability_can_be_the_only_adequate_answer_and_joins_the_cards_only_when_it_saves_much_more():
    # the hand holds nothing that stops a giant: the turret is the answer
    state = _ability_state([GIANT_BRIDGE, _ally((14, 5), "musketeer")], hand=("zap", "fireball", "ram-rider", "electro-dragon"),
                           elixir=5.0)
    d = _sim().decide(state, 100.0)
    assert d.act and ABILITY_KEY in d.allowed_cards
    # the hand holds the tesla, which stops the giant as well as the turret does: the card alone (D-049: the ability joins
    # an open card answer only when it prevents 1.5x as much)
    rich = _ability_state([GIANT_BRIDGE, _ally((14, 5), "musketeer")], hand=("tesla", "knight", "zap", "fireball"), elixir=8.0)
    d = _sim().decide(rich, 100.0)
    assert d.act and "tesla" in d.allowed_cards and ABILITY_KEY not in d.allowed_cards
    # a giant already on its tower cannot be pulled: the turret is no adequate answer there (like any building), it is
    # only the best card at hand in the last second
    late = _ability_state([_heading(_t_enemy((14, 5.5), "giant", eta=0.5, tower="own_right"), "advancing right lane"),
                           _ally((14, 4), "musketeer")], hand=("zap", "fireball", "ram-rider", "electro-dragon"), elixir=8.0)
    policy = _sim()
    report = policy._analyse(late, 100.0)
    assert report.abilities and not policy._adequate(report.abilities[0], report.pushes[0])
    d = policy.decide(late, 100.0)
    assert not d.act or "(emergency)" in d.reason or ABILITY_KEY not in d.allowed_cards


# MAC-AL P: timing, big threats, the full bar, air answers, blind buildings, spell clusters, the ability ---------------------
import dataclasses  # noqa: E402
import math  # noqa: E402

from clashbot.policy import (AIR_WINDOW_S, BIG_PUSH_DAMAGE, DECISION_LATENCY_S, GREAT_SPELL_RATIO,  # noqa: E402
                             RESERVE_WINDOW_S, _Plan)
from tests.test_tactics import DECK47, _support_pack  # noqa: E402


def _al(units, hand, elixir, plays=None, towers=None):
    """A state with the D-047 deck, our towers at the scenario HP and `plays` as their recent plays."""
    st = _state(elixir=elixir, units=units, hand=list(hand))
    st["own"]["deck"] = DECK47
    st["own"]["towers"] = towers or {"king": None, "left": 4082, "right": 4424}
    if plays:
        st["opponent"]["recent_plays"] = plays
    return st


def test_al_constants_are_the_specs():
    assert (DECISION_LATENCY_S, BIG_PUSH_DAMAGE, RESERVE_WINDOW_S, AIR_WINDOW_S, GREAT_SPELL_RATIO) == (
        0.5, 1500.0, 20.0, 9.0, 2.0)


# 1 -----------------------------------------------------------------------------------------------------------------
def test_al_a_window_opens_half_a_second_early_and_the_push_eta_is_the_simulations():
    musketeer = _move("musketeer", 0.9, stop="medium")
    hand = ["musketeer", "zap", "archers", "knight"]
    for eta, acts in ((6.0, True), (6.5, True), (6.6, False), (7.0, False)):       # the ranged window is 6 s
        fake = FakeEngine([musketeer], eta, lane="left", heavy=True)
        d = ReactivePolicy(CARDS, engine=fake).decide(_state(elixir=5.0, units=[_enemy((4, 10), eta=3.0)], hand=hand), 10.0)
        assert d.act is acts, eta
    # the real engine: the tracker says 6 s, the simulation 4.25 s, a melee window (4.5 s) is open at 3.75 s
    knight = _heading(_t_enemy((3, 11), "knight", eta=6.0), "advancing left lane")
    state = _al([knight], ["knight", "zap", "musketeer", "fireball"], 6.0)
    report = _sim()._analyse(state, 100.0)
    assert report.pushes[0].eta_s == 4.25 and report.pushes[0].eta_source == "sim"
    d = _sim().decide(state, 100.0)
    assert d.act and d.mode == "defend" and d.reason == "defend knight left: knight (medium)"
    # with the tracker's ETA (the role-model engine has no simulation) the knight's window is still closed
    d = _ad(DEFAULT_WEIGHTS).decide(_state(elixir=6.0, units=[_enemy((3, 11), eta=6.0)],
                                           hand=["knight", "zap", "musketeer", "fireball"]), 100.0)
    assert d.act and "knight" not in d.allowed_cards                                   # (only the musketeer's 6 s window)


# 2 -----------------------------------------------------------------------------------------------------------------
def _balloon_and_knight(towers=None, elixir=5.0, phase="single_elixir", balloon_tile=(16, 19), balloon_eta=None):
    balloon = _heading(_t_enemy(balloon_tile, "balloon", eta=balloon_eta, tower="own_right"), "advancing right lane")
    balloon["vel"] = [0.0, -1.0]                                                       # walking toward us, just played
    knight = _heading(_t_enemy((3, 12), "knight", eta=4.0), "advancing left lane")
    st = _al([balloon, knight], ["musketeer", "zap", "knight", "fireball"], elixir,
             plays=[{"t": 99.0, "card": "balloon", "tile": list(balloon_tile)}], towers=towers)
    st["phase"] = phase
    return st


def test_al_a_balloon_just_played_keeps_the_elixir_of_its_answer_less_what_the_bar_regenerates():
    # The balloon's answer is the musketeer (4 elixir, air window 9 s). The balloon is 13.25 s out (simulated), so the
    # window opens in 13.25 - 0.5 - 9 = 3.75 s; the bar regenerates 3.75 / 2.8 = 1.34 elixir by then (single elixir),
    # 3.75 / 1.4 = 2.68 (double): the other lane must leave 4 - 1.34 = 2.66 (single) or 4 - 2.68 = 1.32 (double).
    policy = _sim()
    report = policy._analyse(_balloon_and_knight(), 100.0)
    balloon = next(p for p in report.pushes if p.units[0]["card"] == "balloon")
    assert balloon.early and balloon.eta_source == "sim" and balloon.eta_s == 13.25 and balloon.damage_if_unanswered >= BIG_PUSH_DAMAGE
    d = policy.decide(_balloon_and_knight(), 100.0)
    # single elixir, 5 elixir: the knight (3) would leave 2 < 2.66 - and so would every other card of the hand
    assert not d.act and d.mode == "save"
    assert d.reason == "keep 2.7 elixir for the balloon (~3,200 if unanswered), not for knight left"
    # double elixir: the knight leaves 2 >= 1.32 and may go (the old full reserve of 4 held it)
    d = _sim().decide(_balloon_and_knight(phase="double_elixir"), 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left" and "knight" in d.allowed_cards
    # 9 elixir, single: what the left lane may play leaves the 2.66
    d = _sim().decide(_balloon_and_knight(elixir=9.0), 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left"
    assert d.allowed_cards and all(9.0 - CARDS[c]["elixir_cost"] >= 4 - 3.75 / 2.8 for c in d.allowed_cards)
    # the left tower at 300 HP: the knight alone takes it down, so it is answered whatever the balloon needs
    d = _sim().decide(_balloon_and_knight(towers={"king": None, "left": 300, "right": 4424}), 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left" and "knight" in d.allowed_cards


def test_al_a_balloon_played_at_their_back_row_binds_from_the_play_and_a_far_one_keeps_nothing():
    # row 22: simulated ETA 16.25 s, inside RESERVE_WINDOW_S = 20; the musketeer's window opens in 16.25 - 0.5 - 9 = 6.75 s,
    # 6.75 / 2.8 = 2.41 elixir regenerate (single): keep 4 - 2.41 = 1.59
    state = _balloon_and_knight(elixir=4.5, balloon_tile=(16, 22))
    d = _sim().decide(state, 100.0)                                # the knight (3) would leave 1.5 < 1.59
    assert not d.act and d.mode == "save"
    assert d.reason == "keep 1.6 elixir for the balloon (~3,200 if unanswered), not for knight left"
    d = _sim().decide(_balloon_and_knight(elixir=5.0, balloon_tile=(16, 22)), 100.0)    # 5 - 3 = 2 >= 1.59
    assert d.act and d.mode == "defend" and d.lane == "left" and "knight" in d.allowed_cards
    # row 28: 22.25 s, beyond the window: nothing is kept
    far = _balloon_and_knight(elixir=4.5, balloon_tile=(16, 28))
    d = _sim().decide(far, 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left"


def test_al_the_whole_price_is_kept_once_the_answers_window_is_open():
    # the balloon at the river on our side (row 14): 8.25 s out, the musketeer's window (9 s, less 0.5 s) is open - no
    # regeneration to wait for. With 3 elixir the musketeer is out of reach (save for it) and the knight (3) would leave 0
    state = _balloon_and_knight(elixir=3.0, balloon_tile=(16, 14), balloon_eta=9.0)    # (on our half: the tracker gives an ETA)
    d = _sim().decide(state, 100.0)
    assert not d.act and d.reason == "keep 4 elixir for the balloon (~3,200 if unanswered), not for knight left"
    policy = _sim()
    report = policy._analyse(state, 100.0)
    plan = _Plan()
    big = policy._big_push(policy._get_engine(), state, report, report.pushes, 100.0, plan, {})
    knight = next(p for p in report.pushes if p.units[0]["card"] == "knight")
    assert big.eta_s == 8.25 and plan.reserve_card == "musketeer" and plan.reserve[0] == 4.0
    assert policy._answer_lead_s("musketeer", big) == 0.0
    assert policy._bound_reserve(plan, knight, state) == 4.0 and policy._bound_reserve(plan, big, state) == 0.0
    # the same push 3.75 s earlier in its walk (13.25 s out): 4 - 3.75 / 2.8 single, 4 - 3.75 / 1.4 double
    later = _balloon_and_knight(elixir=3.0, balloon_tile=(16, 19))
    report = policy._analyse(later, 100.0)
    plan = _Plan()
    big = policy._big_push(policy._get_engine(), later, report, report.pushes, 100.0, plan, {})
    knight = next(p for p in report.pushes if p.units[0]["card"] == "knight")
    assert policy._answer_lead_s("musketeer", big) == pytest.approx(3.75)
    assert policy._bound_reserve(plan, knight, later) == pytest.approx(4 - 3.75 / 2.8)
    later["phase"] = "double_elixir"
    assert policy._bound_reserve(plan, knight, later) == pytest.approx(4 - 3.75 / 1.4)
    later["phase"] = "overtime"                                  # triple speed: 0.93 s per elixir, 4.03 regenerate: nothing kept
    assert policy._bound_reserve(plan, knight, later) == 0.0


def test_al_a_buildings_window_counts_from_the_bridge_approach_and_a_lost_pull_keeps_the_whole_price():
    # the tesla (4) is the answer: a balloon at row 23 reaches the bridge approach (row 20, MAC-AM M3.3) in 3 s, the window
    # opens 0.5 s sooner
    balloon = _heading(_t_enemy((16, 23), "balloon", tower="own_right"), "advancing right lane")
    state = _al([balloon], ["tesla", "knight", "zap", "fireball"], 3.0)
    state["phase"] = "single_elixir"
    policy = _sim()
    report = policy._analyse(state, 100.0)
    plan = _Plan()
    big = policy._big_push(policy._get_engine(), state, report, report.pushes, 100.0, plan, {})
    assert plan.reserve_card == "tesla" and policy._window_s("tesla", big) == float("-inf")
    assert policy._answer_lead_s("tesla", big) == pytest.approx(3.0 - 0.5)
    other = Push("left", [_t_enemy((3, 12), "knight", eta=4.0)], 3.0, 4.0, [], "x")
    assert policy._bound_reserve(plan, other, state) == pytest.approx(4 - 2.5 / 2.8)
    # locked on its tower: no pull left, no window, no regeneration to count on
    locked = _al([_heading(_t_enemy((15, 6), "balloon", eta=1.0, tower="own_right"), "advancing right lane")],
                 ["tesla", "knight", "zap", "fireball"], 3.0)
    report = policy._analyse(locked, 100.0)
    big = report.pushes[0]
    assert policy._pull_lost("tesla", big) and policy._answer_lead_s("tesla", big) == 0.0


def test_al_two_bats_the_tower_handles_spend_nothing_while_a_balloon_is_coming():
    balloon = _heading(_t_enemy((16, 19), "balloon", tower="own_right"), "advancing right lane")
    bats = [_heading(_t_enemy((3, 12), "bats", eta=2.0, unit_class="bat"), "advancing left lane") for _ in range(2)]
    d = _sim().decide(_al([balloon, *bats], ["musketeer", "zap", "knight", "fireball"], 5.0,
                          plays=[{"t": 99.0, "card": "balloon", "tile": [16, 19]}]), 100.0)
    assert not d.act and d.mode == "save" and d.reason.startswith("waiting for the musketeer window vs balloon right")


# 3 -----------------------------------------------------------------------------------------------------------------
def test_al_full_bar_balloon_at_the_left_bridge_is_defended_with_the_musketeer_not_the_knight():       # t = 367.5
    balloon = _heading(_t_enemy((1.5, 16), "balloon", eta=7.0, tower="own_left"), "advancing left lane")
    firecracker = _heading(_t_enemy((2.5, 16.5), "firecracker", eta=7.5, tower="own_left"), "advancing left lane")
    state = _al([balloon, firecracker], ["zap", "fireball", "musketeer", "knight"], 10.0)
    d = _sim().decide(state, 367.5)
    assert d.act and d.mode == "defend" and d.lane == "left" and d.reason == "defend balloon left: musketeer (medium)"
    assert "musketeer" in d.allowed_cards and "knight" not in d.allowed_cards
    assert d.engine_tiles["musketeer"] == [5.0, 2.5]                                   # behind the left tower
    assert d.threat["card"] == "balloon" and len(d.push_units) == 2


def test_al_a_full_bar_places_the_answer_before_its_window_opens_instead_of_leaking():
    far = _heading(_t_enemy((1.5, 19), "balloon", eta=14.0, tower="own_left"), "advancing left lane")
    hand = ["zap", "fireball", "musketeer", "knight"]
    d = _sim().decide(_al([far], hand, 10.0), 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left" and d.reason == "full bar: pre-place musketeer vs balloon left"
    assert d.allowed_cards == frozenset({"musketeer"}) and d.engine_tiles["musketeer"] == [5.0, 2.5]
    assert d.tile_sources["musketeer"] in ("geometric", "pro_offset")
    # not a full bar: it waits for its window as before
    d = _sim().decide(_al([far], hand, 8.0), 100.0)
    assert not d.act and d.reason.startswith("waiting for the musketeer window vs balloon left")
    # nothing to pre-place (only a win condition and spells in hand): the leak stays
    d = _sim().decide(_al([far], ["ram-rider", "fireball", "zap", "knight"], 10.0), 100.0)
    assert d.mode == "leak" and d.allowed_cards == frozenset({"ram-rider"})


# 4 -----------------------------------------------------------------------------------------------------------------
def test_al_a_ranged_air_answer_has_nine_seconds_against_a_big_flying_push():
    hand = ["zap", "fireball", "musketeer", "knight"]
    river = _heading(_t_enemy((3, 15.5), "balloon", eta=8.5, tower="own_left"), "advancing left lane")
    d = _sim().decide(_al([river], hand, 6.0), 100.0)
    assert d.act and d.mode == "defend" and d.allowed_cards == frozenset({"musketeer"})
    late = _heading(_t_enemy((3, 18), "balloon", eta=11.5, tower="own_left"), "advancing left lane")
    d = _sim().decide(_al([late], hand, 6.0), 100.0)
    assert not d.act and d.reason.startswith("waiting for the musketeer window")
    # a small flying push keeps the 6 s window (the role rule: 3 bats are not worth 9 s)
    policy = _sim()
    push = policy._analyse(_al([river], hand, 6.0), 100.0).pushes[0]
    assert policy._window_s("musketeer", push) == AIR_WINDOW_S
    small = dataclasses.replace(push, damage_if_unanswered=900.0)
    assert policy._window_s("musketeer", small) == 6.0
    assert policy._window_s("tesla", push) == float("inf")                             # buildings keep the geometric rule


# 5 -----------------------------------------------------------------------------------------------------------------
def test_al_a_ground_card_is_never_the_emergency_answer_to_a_balloon():                  # t = 492.0
    giant = _heading(_t_enemy((4, 7), "giant", eta=0.0, tower="own_left"), "advancing left lane")
    balloon = _heading(_t_enemy((3, 8), "balloon", eta=0.0, tower="own_left"), "advancing left lane")
    state = _al([balloon, giant], ["fireball", "ram-rider", "knight", "zap"], 3.0)
    policy = _sim()
    assert [m.card for m in policy._analyse(state, 492.0).defence] == ["knight"]       # the engine does list the knight ...
    d = policy.decide(state, 492.0)
    assert not d.act and d.mode == "save" and d.reason == "no air answer at hand vs balloon left"   # ... the policy does not play it
    # the same push 2 s out (the nothing-adequate rule): no knight either
    for u in state["units"]:
        u["eta_tower"] = {"tower": "own_left", "s": 2.0}
    assert _sim().decide(state, 492.0).reason == "no air answer at hand vs balloon left"
    # without the balloon the knight is the emergency card against the giant
    d = _sim().decide(_al([giant], ["fireball", "ram-rider", "knight", "zap"], 3.0), 492.0)
    assert d.act and d.allowed_cards == frozenset({"knight"}) and "(emergency)" in d.reason
    # and the hero ability is never an emergency card
    assert ABILITY_KEY not in (d.allowed_cards or ())


# 6 -----------------------------------------------------------------------------------------------------------------
def test_al_a_defensive_building_is_not_placed_with_no_push_on_the_field():
    policy = _sim()
    hand = ["tesla", "knight", "zap", "musketeer"]
    state = _al([], hand, 6.0)
    cands = _cands("tesla|middle|mid", "knight|left|mid", "tesla|left|back", "musketeer|right|back")
    kept = policy.no_blind_spells(cands, state)                                         # a `learned` play
    assert _keys(kept) == ["knight|left|mid", "musketeer|right|back"]
    # the leak (a non-win-condition one) and the punish drop it as well
    leak = ActDecision(True, "leak", None, "x")
    assert "tesla" not in {c.key.split("|")[0] for c in policy.filter_candidates(_cands("tesla|left|back", "knight|left|back"), leak, state)}
    punish = ActDecision(True, "punish", "left", "x")
    assert _keys(policy.filter_candidates(_cands("tesla|left|bridge", "knight|left|bridge"), punish, state)) == ["knight|left|bridge"]
    learned = LearnedActPolicy(None, CARDS, rules=policy)
    assert _keys(learned.filter_candidates(cands, ActDecision(True, "learned", None, "x"), state)) == _keys(kept)
    # with an enemy push on the field it is a pull and stays - when its tile is in the centre box (MAC-AM M3.5: a
    # candidate without a tile, or with one next to a tower, is dropped)
    giant = _heading(_t_enemy((14, 15), "giant", eta=9.0, tower="own_right"), "advancing right lane")
    assert _keys(policy.no_blind_spells(cands, _al([giant], hand, 6.0))) == ["knight|left|mid", "musketeer|right|back"]
    centred = [Candidate("tesla|middle|mid", 0.9, {"tac": 0.9}, [10.0, 10.0]), *cands[1:]]
    assert _keys(policy.no_blind_spells(centred, _al([giant], hand, 6.0))) == [
        "tesla|middle|mid", "knight|left|mid", "musketeer|right|back"]
    # a unit on their half that is no flyer and no targeter, and a retreating one, are no push
    for far in (_t_enemy((9, 26), "knight"), _heading(_t_enemy((9, 12), "knight"), "retreating")):
        assert "tesla|middle|mid" not in _keys(policy.no_blind_spells(cands, _al([far], hand, 6.0)))
    # a flyer or building targeter anywhere is one
    assert "tesla|middle|mid" in _keys(policy.no_blind_spells(centred, _al([_t_enemy((16, 26), "balloon")], hand, 6.0)))


# 7 -----------------------------------------------------------------------------------------------------------------
def _cluster_behind_a_giant(elixir=9.0):
    giant = _heading(_t_enemy((4, 12), "giant", eta=5.0), "advancing left lane")
    return _al([giant, *_support_pack()], ["fireball", "electro-dragon", "knight", "musketeer"], elixir)


def test_al_a_great_spell_on_the_support_behind_a_giant_is_the_only_allowed_card_and_reaches_the_llm():
    policy = _sim()
    state = _cluster_behind_a_giant()
    d = policy.decide(state, 100.0)
    assert d.act and d.mode == "defend" and d.lane == "left" and d.allowed_cards == frozenset({"fireball"})
    assert d.reason.startswith("defend giant left: fireball hits ") and " units (~" in d.reason and d.reason.endswith(" elixir)")
    assert len(d.push_units) == 10 and d.push_units[0]["card"] == "giant" or any(u["card"] == "giant" for u in d.push_units)
    aim = d.engine_tiles["fireball"]
    # the generator offered a knight and an electro dragon only: the fireball is the engine's own candidate
    out = policy.filter_candidates(_cands("knight|left|mid", "electro-dragon|left|mid"), d, state)
    assert [c.key.split("|")[0] for c in out] == ["fireball"] and out[0].sources == {"tac": 0.0}
    assert math.dist(out[0].tile, [4.3, 15.5]) <= 1.0 and out[0].tile == aim
    # a lookup fireball candidate in the cluster's cell passes the defend filter (a giant-led push: any unit of the push)
    assert _keys(policy.filter_candidates(_cands("fireball|left|bridge", "knight|left|mid"), d, state)) == ["fireball|left|bridge"]
    # ... today's rule (no push units) only looks at the lead, a giant, which no spell is aimed at
    assert not policy._aims_at_threat("fireball", aim, state["units"][0])
    assert policy._aims_at_threat("fireball", aim, state["units"][0], tuple(state["units"]))
    assert policy.target_tile("fireball|left|bridge", state) is not None


def test_al_the_same_cluster_on_their_side_with_no_push_is_a_spell_act():
    pack = [dict(u, heading="stationary") for u in _support_pack(y=21.0, eta=False)]
    state = _al(pack, ["fireball", "electro-dragon", "knight", "musketeer"], 9.0)
    policy = _sim()
    d = policy.decide(state, 100.0)
    assert d.act and d.mode == "spell" and d.allowed_cards == frozenset({"fireball"})
    assert d.reason.startswith("spell: fireball ") and "kills" in d.reason and d.reason.endswith("+ tower")
    assert d.tile_sources == {"fireball": "spell"} and d.engine_tiles["fireball"][1] >= 17
    assert d.lane == "left" or d.lane == "middle"
    # the filter keeps the spell's candidates, and the engine's own when there are none
    out = policy.filter_candidates(_cands("knight|left|mid", "electro-dragon|left|back"), d, state)
    assert [c.key.split("|")[0] for c in out] == ["fireball"] and out[0].tile == d.engine_tiles["fireball"]
    # not enough elixir for the spell plus the balloon's answer (a big push is coming): no spell
    balloon = _heading(_t_enemy((16, 17), "balloon", eta=11.0, tower="own_right"), "advancing right lane")
    busy = _al(pack + [balloon], ["fireball", "electro-dragon", "knight", "musketeer"], 6.0)
    assert policy.decide(busy, 100.0).mode != "spell"
    # an air answer the opponent is likely to need keeps its elixir, too
    kg_policy = ReactivePolicy(CARDS, engine=TacticalEngine(CARDS, weights=DEFAULT_WEIGHTS),
                               kg=FakeDeckKG([("balloon", 0.7)]))
    st = _al(pack, ["fireball", "electro-dragon", "knight", "musketeer"], 6.0)
    st["opponent"]["deck_known"] = ["miner", "valkyrie"]
    assert kg_policy.air_reserve(st)[0] == 4.0 and kg_policy.decide(st, 100.0).mode != "spell"     # 6 - 4 < the reserve of 4
    st["own"]["elixir"] = 9.0
    assert kg_policy.decide(st, 100.0).mode == "spell"


def test_al_a_cheap_cluster_is_no_spell_act_and_the_old_engine_has_none():
    pack = [dict(u, heading="stationary") for u in _support_pack(y=21.0, eta=False)]
    state = _al(pack[:3], ["fireball", "electro-dragon", "knight", "musketeer"], 9.0)      # a witch, a wizard, one bat
    assert _sim().decide(state, 100.0).mode != "spell"
    assert _ad(DEFAULT_WEIGHTS).decide(_al(pack, ["fireball", "electro-dragon", "knight", "musketeer"], 9.0), 100.0).mode != "spell"


# 8 -----------------------------------------------------------------------------------------------------------------
def test_al_every_allowed_card_reaches_the_llm_even_when_the_generator_left_it_out():          # t = 261.5
    policy = _sim()
    state = _al([_enemy((3.5, 14.5), eta=3.0, card="bats"), _enemy((4, 14.5), card="bats")],
                ["fireball", "electro-dragon", "knight", "zap"], 7.0)
    tiles = {"fireball": [3.5, 15.0], "electro-dragon": [5.0, 2.5]}
    d = ActDecision(True, "defend", "left", "defend bats left: fireball (spell)", allowed_cards=frozenset({"fireball", "electro-dragon"}),
                    engine_tiles=tiles)
    out = policy.filter_candidates(_cands("electro-dragon|left|back", "knight|left|mid"), d, state)
    assert _keys(out) == ["electro-dragon|left|back", "fireball|left|bridge"]           # kept first, the missing one appended
    assert out[1].tile == [3.5, 15.0] and out[1].sources == {"tac": 0.0} and "tactics: defend" in out[1].note
    # nothing left at all: both are the engine's
    out = policy.filter_candidates(_cands("knight|left|mid"), d, state)
    assert [c.key.split("|")[0] for c in out] == ["electro-dragon", "fireball"]
    # a card without an engine tile (and the hero ability) is not invented
    d2 = dataclasses.replace(d, allowed_cards=frozenset({"fireball", "zap", ABILITY_KEY}))
    assert [c.key.split("|")[0] for c in policy.filter_candidates([], d2, state)] == ["fireball"]
    # the counter, spell and win-condition leak modes inject too
    for mode in ("counter", "spell", "leak"):
        dm = ActDecision(True, mode, "left", "x", allowed_cards=frozenset({"fireball"}), engine_tiles=tiles)
        assert _keys(policy.filter_candidates(_cands("knight|left|mid"), dm, state)) == ["fireball|left|bridge"]


# 9 -----------------------------------------------------------------------------------------------------------------
def _ability_report(card_prevented, ability_prevented, card_adequate=True, ability_adequate=True, eta=5.0):
    """A stand-in engine: one giant push, a tesla move and the turret, with the damage each prevents."""
    giant = _heading(_t_enemy((14, 13), "giant", eta=eta, tower="own_right"), "advancing right lane")
    push = Push("right", [giant], 5.0, eta, ["tank", "building_targeter"], "fake", heavy=True)
    tesla = TacticalMove("tesla", "tesla|right|mid", [11.0, 7.0], "defend", "x", 0.9, 1.0, stop="high",
                         adequate=card_adequate, prevented=card_prevented)
    ability = TacticalMove(ABILITY_KEY, "ability:musketeer|right|mid", [14.0, 8.0], "ability", "x", 0.8, None,
                           stop="turret", adequate=ability_adequate, prevented=ability_prevented,
                           tile_source="ability", push_lane="right")

    class Engine:
        def analyse(self, state, t=None):
            return TacticalReport([push], [tesla, ability], tesla, None, [], defence=[tesla], abilities=[ability])

    return Engine(), _state(elixir=8.0, units=[giant], hand=["tesla", "knight", "zap", "fireball"])


def test_al_the_ability_joins_an_open_card_answer_only_when_it_saves_one_and_a_half_times_as_much():
    for card, ability, listed in ((1000.0, 1600.0, True), (1000.0, 1500.0, True), (1000.0, 1400.0, False),
                                  (1110.0, 1110.0, False)):
        engine, state = _ability_report(card, ability)
        d = ReactivePolicy(CARDS, engine=engine).decide(state, 100.0)
        assert d.act and d.mode == "defend" and "tesla" in d.allowed_cards, (card, ability)
        assert (ABILITY_KEY in d.allowed_cards) is listed, (card, ability)
    # no adequate card has its window open (the tesla is not adequate here): the turret alone
    engine, state = _ability_report(0.0, 1110.0, card_adequate=False)
    d = ReactivePolicy(CARDS, engine=engine).decide(state, 100.0)
    assert d.act and d.allowed_cards == frozenset({ABILITY_KEY}) and d.reason.endswith("(turret)")
    # the turret is no adequate answer (the damage it leaves is too big): never added, not even in an emergency
    engine, state = _ability_report(0.0, 1110.0, card_adequate=False, ability_adequate=False, eta=1.0)
    d = ReactivePolicy(CARDS, engine=engine).decide(state, 100.0)
    assert d.act and "(emergency)" in d.reason and ABILITY_KEY not in d.allowed_cards
    # and the ability is no answer the biggest push keeps elixir for
    engine, state = _ability_report(0.0, 1110.0, card_adequate=False)
    policy = ReactivePolicy(CARDS, engine=engine)
    report = engine.analyse(state)
    assert policy._reserve(policy._push_moves(engine, state, report, report.pushes[0], 100.0)[1], report.pushes[0]) is None


def test_al_the_ability_is_proposed_for_a_balloon_at_the_tower_when_no_air_card_is_affordable_and_not_for_bats():
    balloon = _heading(_t_enemy((14, 7), "balloon", eta=3.0, tower="own_right"), "advancing right lane")
    state = _ability_state([balloon, _ally((14, 4), "musketeer")], hand=("knight", "zap", "fireball", "ram-rider"), elixir=5.0)
    d = _sim().decide(state, 100.0)
    assert d.act and d.mode == "defend" and d.allowed_cards == frozenset({ABILITY_KEY})
    assert d.reason == "defend balloon right: ability:musketeer (turret)" and d.tile_sources[ABILITY_KEY] == "ability"
    # a bats push with the Musketeer on the field and the button ready: the engine proposes nothing, the policy allows nothing
    bats = [_heading(_t_enemy((3, 12), "bats", eta=2.0, unit_class="bat"), "advancing left lane") for _ in range(3)]
    state = _ability_state(bats + [_ally((4, 5), "musketeer")], hand=("knight", "zap", "fireball", "ram-rider"), elixir=5.0)
    d = _sim().decide(state, 100.0)
    assert not d.act and ABILITY_KEY not in (d.allowed_cards or ())


# the air reserve also covers a counter and the generic leak (P2.3) ----------------------------------------------------------
def test_al_the_air_reserve_covers_a_counter_and_a_leak_too():
    policy, _ = _reserve_policy()
    scene = _counter_scene(elixir=7.0)
    scene["opponent"]["deck_known"] = ["miner", "valkyrie"]                              # a balloon is likely: keep the musketeer's 4
    d = policy.decide(scene, 100.0)
    assert d.mode != "counter" and not d.act                                              # ram-rider (5) would leave 2 < 4
    full = _counter_scene(elixir=9.0)
    full["opponent"]["deck_known"] = ["miner", "valkyrie"]
    d = policy.decide(full, 100.0)
    assert d.act and d.mode == "counter" and d.allowed_cards == frozenset({"ram-rider"})   # 9 - 5 = 4 covers it
    # the generic leak drops the plays that spend the reserve; the win-condition leak is allowed when 4 stay
    hand = ("musketeer", "knight", "p-e-k-k-a", "zap")
    st = _air_state(elixir=10.0, opp_elixir=6.0, hand=hand)
    leak = policy.decide(st, 100.0)
    assert leak.act and leak.mode == "leak" and leak.allowed_cards is None
    out = policy.filter_candidates(_cands("p-e-k-k-a|left|back", "knight|left|back"), leak, st)
    assert _keys(out) == ["knight|left|back"]                                             # p.e.k.k.a (7) would leave 3 < 4
    wc = _air_state(elixir=10.0, opp_elixir=6.0, hand=("musketeer", "ram-rider", "knight", "zap"))
    d = policy.decide(wc, 100.0)
    assert d.mode == "leak" and d.allowed_cards == frozenset({"ram-rider"})              # 10 - 5 = 5 >= 4
    policy.air_reserve = lambda state: (6.0, "musketeer", "balloon", 0.6)                 # a dearer reserve: 5 would leave < 6
    assert policy.decide(wc, 100.0).allowed_cards is None


# =================================================================================================================
# MAC-AM: spells lead their targets by the real delay, are cast only for value, and a tesla stands in the centre
# =================================================================================================================
from clashbot.policy import (CENTER_BOX, GROUP_TILE_SOURCE, LANE_ANCHOR, PULL_LEAD_S, PULL_OPEN_ROW,  # noqa: E402
                             ZAP_GROUP_MIN_PREVENT, _lead_tile, in_center_box, lead_tile_s, projectile_speeds,
                             spell_delay_s, spell_flight_s, spell_worth)
from clashbot import unit_stats as _us  # noqa: E402

KING = (9.0, 0.5)


def _mover(tile, pred, card="knight", kalman=0.7, heading="advancing left lane", **kw):
    """An enemy walking toward our side: the tracker's velocity (`kalman` tiles/s) and the tile it will be in 2 s."""
    return {**_enemy(tile, card=card, pred=pred), "speed_tiles_s": kalman, "heading": heading, **kw}


def test_am_constants_and_the_kb_projectile_speeds():
    speeds = projectile_speeds()
    assert speeds["fireball"] == 10.0 and speeds["rocket"] == pytest.approx(350 / 60) and speeds["arrows"] == pytest.approx(1100 / 60)
    assert "zap" in speeds and speeds["zap"] is None                      # a card with no projectile column: instant
    assert PULL_LEAD_S == 1.5 and PULL_OPEN_ROW == 20 and ZAP_GROUP_MIN_PREVENT == 300.0
    assert CENTER_BOX == ((7.0, 11.0), (8.5, 11.5))
    assert LANE_ANCHOR == {"left": (8.0, 10.0), "right": (10.0, 10.0), "middle": (9.0, 10.0)}
    assert in_center_box((8.0, 10.0)) and in_center_box((7.0, 8.5)) and in_center_box((11.0, 11.5))
    assert not in_center_box((5.5, 6.5)) and not in_center_box((9.0, 12.0)) and not in_center_box(None)
    # the coordinate frame: our princess towers at row 4.3, the river at rows 15-16: the box is the middle of our half
    from clashbot.policy import OWN_TOWER_TILES
    assert OWN_TOWER_TILES["own_left"][1] < CENTER_BOX[1][0] < CENTER_BOX[1][1] < 15


# 1 (lead) --------------------------------------------------------------------------------------------------------
def test_am_a_fireball_is_led_by_the_latency_plus_the_flight_from_the_king_tower():
    unit = _mover((4, 16), (4, 14.6))                 # walks down at the knight's 1.0 tiles/s; the filter says 0.7
    # the unit's speed is max(0.7, 0.9 x 1.0) = 0.9: after the delay it has walked delay x 0.9 tiles
    aim = lead_tile_s(unit, 0.0)
    assert aim == [4.0, 16.0]
    delay = spell_delay_s("fireball", aim)
    for _ in range(2):                                # the policy's two rounds
        aim = lead_tile_s(unit, delay)
        delay = spell_delay_s("fireball", aim)
    assert delay == pytest.approx(0.5 + math.dist(KING, aim) / 10.0, abs=0.02) and delay == pytest.approx(1.95, abs=0.03)
    assert aim == pytest.approx([4.0, 14.25], abs=0.05)                  # (the old lead: [4, 15.3])
    assert _lead_tile(unit, 1.0) == pytest.approx([4.0, 15.3])
    # a zap lands 0.5 s after the decision, whatever the distance
    assert spell_delay_s("zap", (4, 14.25)) == 0.5 and spell_delay_s("zap", (17, 31)) == 0.5
    assert lead_tile_s(unit, 0.5) == pytest.approx([4.0, 15.55])
    # rocket and arrows use their own speeds
    assert spell_flight_s("rocket", (4, 14)) == pytest.approx(math.dist(KING, (4, 14)) / (350 / 60))
    assert spell_flight_s("arrows", (4, 14)) == pytest.approx(math.dist(KING, (4, 14)) / (1100 / 60))
    # a card the KB does not know keeps the old flat lead
    assert spell_flight_s("not-a-card", (4, 14)) == 0.3 and spell_flight_s("lightning", (4, 14)) == 0.0   # (KB: no projectile)


def test_am_a_unit_that_stands_or_fights_is_not_led():
    still = _mover((3.5, 22.0), (3.5, 20.6), heading="stationary")        # at their tower
    assert lead_tile_s(still, 2.0) == [3.5, 22.0]
    at_tower = _mover((3.5, 6.0), (3.5, 4.6))                              # within a knight's reach of our left tower
    assert lead_tile_s(at_tower, 2.0) == [3.5, 6.0]
    free = _mover((3.5, 9.0), (3.5, 7.6))
    assert lead_tile_s(free, 2.0)[1] == pytest.approx(9.0 - 1.8)
    # ... or of one of our units: a knight of ours next to it stops it, one 4 tiles away does not
    state = {"units": [_ally((3.5, 8.0))]}
    assert lead_tile_s(free, 2.0, state) == [3.5, 9.0]
    state = {"units": [_ally((3.5, 5.0))]}
    assert lead_tile_s(free, 2.0, state)[1] == pytest.approx(7.2)
    # a unit with no velocity information, a unit moving away at the filter's speed, a lead past the arena edge
    assert lead_tile_s(_enemy((4, 12), card="knight"), 2.0) == [4.0, 12.0]
    away = _mover((4, 20), (4, 21.4), heading="retreating")
    assert lead_tile_s(away, 2.0) == pytest.approx([4.0, 21.4])
    edge = _mover((17, 30), (17, 40), card="musketeer", kalman=5.0)
    assert lead_tile_s(edge, 2.0) == [17.0, 31.0]


def test_am_the_spell_aim_uses_the_lead_and_the_defend_check_uses_it_too():
    policy = _sim()
    # a worthwhile cluster walking down toward the bridge: the aim is where the units will be when the fireball lands
    pack = [_mover((4 + 0.6 * i, 16 + 0.3 * i), (4 + 0.6 * i, 14.6 + 0.3 * i), card=c)
            for i, c in enumerate(("archers", "archers", "firecracker"))]
    state = _al(pack, ["fireball", "knight", "zap", "musketeer"], 9.0)
    aim, value, members, kills = policy.spell_best("fireball", state)
    assert kills == 3 and len(members) == 3 and value >= 6.0
    still = [dict(u, heading="stationary") for u in pack]
    aim0 = policy.spell_best("fireball", _al(still, ["fireball", "knight", "zap", "musketeer"], 9.0))[0]
    assert aim0[1] - aim[1] > 1.2                                          # ~1.8 tiles further down than where they stand
    assert math.dist(aim, aim0) < 3.0
    assert policy.target_tile("fireball|left|bridge", state) == aim
    # the defend check: the push units at their lead tiles must be in the circle of the aim
    assert policy._aims_at_threat("fireball", aim, pack[0], tuple(pack), state)
    assert not policy._aims_at_threat("fireball", [aim[0], aim[1] + 6], pack[0], tuple(pack), state)


# 2 (worth) -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("card, value, kills, expected", [
    ("zap", 2.4, 2, False), ("zap", 2.5, 2, True), ("zap", 5.0, 1, False), ("zap", 6.0, 0, False), ("zap", 4.0, 8, True),
    ("fireball", 4.0, 1, False), ("fireball", 4.99, 1, False), ("fireball", 5.0, 1, True), ("fireball", 7.0, 2, True),
    ("fireball", 5.9, 0, False), ("fireball", 6.0, 0, True), ("fireball", 3.9, None, False), ("fireball", 4.0, None, True),
    ("arrows", 3.9, 1, False), ("arrows", 4.0, 1, True), ("rocket", 6.5, 2, False), ("rocket", 7.0, 2, True),
])
def test_am_spell_worth_is_one_bar_for_every_spell(card, value, kills, expected):
    assert spell_worth(card, value, kills) is expected


def test_am_a_lone_firecracker_with_the_tower_bonus_is_not_worth_a_fireball_in_any_mode():
    policy = _sim()
    lone = _al([dict(_enemy((4.5, 22.0), card="firecracker"), heading="stationary")], ["fireball", "knight", "zap", "musketeer"], 9.0)
    assert policy.spell_best("fireball", lone) is None                    # 3 + the +1 tower bonus = 4 < 5 (t = 325.5)
    assert policy.target_tile("fireball|left|enemy_half", lone) is None
    assert policy._analyse(lone, 100.0).spells == []
    cand = Candidate("fireball|left|enemy_half", 0.9, {"tac": 0.9}, [4.5, 22.0])
    for mode in ("learned", "leak", "punish", "counter", "spell"):
        assert policy.filter_candidates([cand], ActDecision(True, mode, "left", "x"), lone) == [], mode
    # the same bar in an emergency: the engine's move would be a fireball worth 4 with one kill, a push at the tower
    firecracker = _enemy((4, 6), eta=0.5, card="firecracker")
    weak = TacticalMove("fireball", "fireball|left|mid", [4.0, 6.0], "defend", "x", 0.9, 1.0, stop="spell", value=4.0,
                        members=(firecracker,), kills=1)
    d = ReactivePolicy(CARDS, engine=FakeEngine([weak], 0.5, lane="left", heavy=False, needs=())).decide(
        _state(elixir=5.0, units=[firecracker], hand=["fireball", "knight", "zap", "giant"]), 100.0)
    assert not d.act and "fireball" not in (d.allowed_cards or ()) and d.reason.startswith("no worthwhile card at hand")
    # ... and a fireball worth 5 with a kill is taken in the same situation
    good = TacticalMove("fireball", "fireball|left|mid", [4.0, 6.0], "defend", "x", 0.9, 1.0, stop="spell", value=5.0,
                        members=(firecracker,), kills=1)
    d = ReactivePolicy(CARDS, engine=FakeEngine([good], 0.5, lane="left", heavy=False, needs=())).decide(
        _state(elixir=5.0, units=[firecracker], hand=["fireball", "knight", "zap", "giant"]), 100.0)
    assert d.act and d.allowed_cards == frozenset({"fireball"})


def test_am_archers_and_a_firecracker_next_to_their_tower_are_worth_a_fireball():
    policy = _sim()
    pack = [dict(_enemy(t, card=c), heading="stationary") for t, c in
            (((4, 20), "archers"), ((4.5, 20.5), "archers"), ((5, 21), "firecracker"))]
    state = _al(pack, ["fireball", "knight", "zap", "musketeer"], 9.0)
    aim, value, members, kills = policy.spell_best("fireball", state)
    assert kills == 3 and value == pytest.approx(7.0) and len(members) == 3        # archers 3 + firecracker 3 + the tower 1
    report = policy._analyse(state, 100.0)
    assert [m.card for m in report.spells] == ["fireball"] and report.spells[0].value == pytest.approx(7.0)


def test_am_zaps_are_cast_for_two_kills_and_value_never_for_nothing():
    policy = _sim()
    zap_hand = ["zap", "knight", "fireball", "musketeer"]
    def at(units): return _al([dict(u, heading="stationary") for u in units], zap_hand, 9.0)
    bats = lambda n, x=4.0, y=28.0: [{**_enemy((x + 0.3 * i, y), card="bats"), "unit_class": "bat"} for i in range(n)]    # noqa: E731
    # one bat of five (0.4), three bats (1.2), a whole play of bats (2.0): none clears the 2.5 a zap needs on its own
    for n in (1, 3, 5):
        assert policy.spell_best("zap", at(bats(n))) is None, n
    # kills 0 never: a lone balloon, a musketeer, a night witch and an e-wizard (they only get scratched)
    for card in ("balloon", "musketeer", "night-witch", "electro-wizard", "giant"):
        assert policy.spell_best("zap", at([_enemy((4, 28), card=card)])) is None, card
    # a whole play of bats + a spear goblin (2.67, six kills) is worth it; with a whole play of goblins, 4.0 (a great spell)
    best = policy.spell_best("zap", at(bats(5) + [_enemy((4.5, 28.5), card="spear-goblins")]))
    assert best is not None and best[1] == pytest.approx(2.0 + 2 / 3, abs=0.01) and best[3] == 6
    goblins = [_enemy((4 + 0.4 * i, 28.5), card="spear-goblins") for i in range(3)]
    best = policy.spell_best("zap", at(bats(5) + goblins))
    assert best is not None and best[1] == pytest.approx(4.0, abs=0.01) and best[3] == 8
    # next to their tower the +1 of the crown-tower damage counts: a whole play of bats is then worth 3.0
    assert policy.spell_best("zap", at(bats(5, y=24.0)))[1] == pytest.approx(3.0)
    assert policy.spell_best("zap", at(bats(3, y=24.0))) is None                                      # 1.2 + 1 = 2.2 < 2.5


# 2.1b (the group zap, the user's amendment) --------------------------------------------------------------------------
def _group(cards, ys=9.0, eta=2.0, lane_x=3.5, heading="advancing left lane"):
    offsets = ((0, 0), (0.5, 0.5), (-0.5, 0.5), (0, 1.0), (1.0, 0.0), (-1.0, 0.0))
    return [_heading(_enemy((lane_x + dx, ys + dy), card=c, eta=eta), heading) for (dx, dy), c in zip(offsets, cards)]


def test_am_a_zap_on_a_group_the_tower_finishes_is_worth_it_when_it_saves_300_tower_hp():
    engine = TacticalEngine(CARDS, weights=DEFAULT_WEIGHTS)
    state = _al(_group(["archers", "archers", "firecracker"]), ["zap", "knight", "fireball", "musketeer"], 9.0)
    rep = engine.analyse(state, 100.0)
    zap = next(m for m in rep.defence if m.card == "zap")
    # no kills (the archers and the firecracker have 490 hp, a zap does 281): rule (a) fails, rule (b) holds
    assert zap.kills == 0 and not spell_worth("zap", zap.value, zap.kills)
    assert zap.group and zap.tile_source == GROUP_TILE_SOURCE and len(zap.members) == 3
    assert zap.prevented >= ZAP_GROUP_MIN_PREVENT and zap.prevented == pytest.approx(1126.0, abs=1.0)
    assert zap.reason == "zap archers x2 + firecracker: tower finishes them, saves ~1,100 HP"
    # a group of two is no group; three units far on their side that are not coming are no push
    assert not [m for m in engine.analyse(_al(_group(["archers", "firecracker"]), ["zap", "knight"], 9.0), 100.0).defence
                if m.card == "zap"]
    away = _group(["archers", "archers", "firecracker"], ys=26.0, eta=None, heading="stationary")
    rep = engine.analyse(_al(away, ["zap", "knight", "fireball", "musketeer"], 9.0), 100.0)
    assert rep.pushes == [] and not [m for m in rep.defence if m.card == "zap"]
    back = _group(["archers", "archers", "firecracker"], ys=26.0, eta=None, heading="retreating")
    assert not [m for m in engine.analyse(_al(back, ["zap", "knight"], 9.0), 100.0).defence if m.card == "zap"]


def test_am_the_group_zap_passes_the_policys_filter_and_a_kill_zap_is_still_a_kill_zap():
    policy = _sim()
    # archers x2 + minions: the sim says the zap alone stops this push (prevented 1,633, adequate): the policy allows it
    state = _al(_group(["archers", "archers", "minions"]), ["zap", "knight", "tesla", "musketeer"], 9.0)
    report = policy._analyse(state, 100.0)
    zap = next(m for m in report.defence if m.card == "zap")
    assert zap.group and zap.adequate
    d = policy.decide(state, 100.0)
    assert d.act and d.mode == "defend" and d.allowed_cards == frozenset({"zap"})
    assert d.tile_sources["zap"] == GROUP_TILE_SOURCE and d.engine_tiles["zap"] == [3.5, 9.5]
    cand = Candidate("zap|left|mid", 0.9, {"knn": 1.0})
    assert _keys(policy.filter_candidates([cand], d, state)) == ["zap|left|mid"]       # (spell_target knows no such cluster)
    assert policy.spell_target("zap", "left", "mid", state) is None
    # with spear goblins in the group the zap kills three of them: rule (a) takes it (not flagged as a group zap) ...
    kills = _group(["archers", "archers", "firecracker", "spear-goblins", "spear-goblins", "spear-goblins"])
    mixed = _al(kills, ["zap", "knight", "fireball", "musketeer"], 9.0)
    rep = policy._analyse(mixed, 100.0)
    zap = next(m for m in rep.defence if m.card == "zap")
    assert not zap.group and zap.kills == 3 and spell_worth("zap", zap.value, zap.kills)
    assert zap.prevented >= 2000
    # ... and rule (b) holds there too (the amendment's example): 6 units of the push in the circle, 2,568 tower HP saved
    engine = policy.engine
    push = engine._pushes(mixed, 100.0)[0]
    ctx = engine._ctx(push, mixed, 100.0, ["zap", "knight", "fireball", "musketeer"])
    aim = engine._group_aim("zap", ctx, mixed)
    assert aim is not None and aim.group and len(aim.members) == 6 and aim.kills == 3
    # three units of a push that retreat on their side of the river (eta given) are not a group the tower finishes
    gone = _al(_group(["archers", "archers", "firecracker"], ys=18.0, eta=5.0, heading="retreating"), ["zap", "knight"], 9.0)
    push = engine._pushes(gone, 100.0)[0]
    assert engine._group_aim("zap", engine._ctx(push, gone, 100.0, ["zap", "knight"]), gone) is None


# 3 (tesla) -------------------------------------------------------------------------------------------------------------
def _tesla_scene(units, hand=("tesla", "knight", "musketeer", "zap"), elixir=9.0):
    return _al(units, hand, elixir)


def _tesla_move(units, **kw):
    rep = TacticalEngine(CARDS, weights=DEFAULT_WEIGHTS).analyse(_tesla_scene(units, **kw), 100.0)
    return next((m for m in rep.defence if m.card == "tesla"), None), rep


def test_am_a_giant_at_the_left_bridge_is_pulled_to_the_centre_box_anchor():
    giant = _mover((3.5, 15.0), (3.5, 13.5), card="giant", kalman=0.75)
    giant["eta_tower"] = {"tower": "own_left", "s": 9.0}
    tesla, rep = _tesla_move([giant])
    assert tesla is not None and tesla.tile == [8.0, 10.0] and in_center_box(tesla.tile)
    assert tesla.reason.startswith("pulls giant to the centre") and tesla.adequate
    # the right bridge: the right anchor
    giant_r = _mover((14.5, 15.0), (14.5, 13.5), card="giant", kalman=0.75, heading="advancing right lane")
    giant_r["eta_tower"] = {"tower": "own_right", "s": 9.0}
    assert _tesla_move([giant_r])[0].tile == [10.0, 10.0]
    # every box tile that qualifies is in the box; the anchor itself is the pick whenever it qualifies
    engine = TacticalEngine(CARDS, weights=DEFAULT_WEIGHTS)
    for y in (13.0, 15.0, 18.0, 20.0):
        tile = engine._pull_tile([3.5, y], anchor=LANE_ANCHOR["left"])
        assert tile is not None and in_center_box(tile)
        assert tile == [8.0, 10.0]
    # near the tower there is no pull tile at all (a tile must be 3 tiles from the giant and 1.5 closer than the tower)
    assert engine._pull_tile([3.5, 6.5]) is None and engine._pull_tile([4, 8]) is None


def test_am_a_giant_that_is_already_deep_gets_no_tesla():
    deep = _mover((4, 8), (4, 6.5), card="giant", kalman=0.75)
    deep["eta_tower"] = {"tower": "own_left", "s": 3.0}
    tesla, rep = _tesla_move([deep])
    assert tesla is None and [p.lane for p in rep.pushes] == ["left"]
    assert "tesla" not in {m.card for m in rep.moves}
    # the policy does not ask for one either
    d = _sim().decide(_tesla_scene([deep]), 100.0)
    assert "tesla" not in (d.allowed_cards or ()) and "tesla" not in d.engine_tiles


def test_am_a_troop_push_is_kited_to_the_anchor_while_the_lead_is_above_it():
    cracker = _mover((14.5, 15.0), (14.5, 12.0), card="firecracker", kalman=1.5, heading="advancing right lane")
    cracker["eta_tower"] = {"tower": "own_right", "s": 7.0}
    tesla, _ = _tesla_move([cracker])
    assert tesla is not None and tesla.tile == [10.0, 10.0] and tesla.reason.startswith("kites firecracker to the centre")
    # the lead unit is already below the anchor's row + 1: no tesla (it would stand behind the unit)
    low = _mover((14.5, 10.0), (14.5, 7.0), card="firecracker", kalman=1.5, heading="advancing right lane")
    low["eta_tower"] = {"tower": "own_right", "s": 4.0}
    assert _tesla_move([low])[0] is None
    # ... and a lead within 3 tiles of the anchor: no tesla either
    near = _mover((9.0, 12.5), (9.0, 10.0), card="firecracker", kalman=1.5, heading="advancing left lane")
    near["eta_tower"] = {"tower": "own_left", "s": 4.0}
    assert _tesla_move([near])[0] is None                                    # 3 tiles from the middle lane's anchor (9, 10)? no: 2.5


def test_am_a_balloon_at_the_tower_may_get_the_old_tower_tile_when_nothing_else_hits_air():
    balloon = _mover((3.5, 6.0), (3.5, 5.0), card="balloon", kalman=1.0)
    balloon["eta_tower"] = {"tower": "own_left", "s": 1.0}
    tesla, _ = _tesla_move([balloon], hand=("tesla", "knight", "zap", "fireball"))
    assert tesla is not None and tesla.tile == [5.5, 6.5] and tesla.reason.startswith("air dps at the tower")
    assert not in_center_box(tesla.tile)
    # ... not when another card in the hand hits air (a musketeer)
    assert _tesla_move([balloon], hand=("tesla", "knight", "zap", "musketeer"))[0] is None
    # a giant (no air) locked on the tower: no exception
    giant = _mover((3.5, 6.0), (3.5, 5.0), card="giant", kalman=0.75)
    giant["eta_tower"] = {"tower": "own_left", "s": 1.0}
    assert _tesla_move([giant], hand=("tesla", "knight", "zap", "fireball"))[0] is None


def test_am_a_tesla_candidate_outside_the_centre_box_is_dropped_unless_it_is_the_engines_tile():
    policy = _sim()
    giant = _heading(_t_enemy((3.5, 15.0), "giant", eta=9.0), "advancing left lane")
    state = _al([giant], ["tesla", "knight", "zap", "musketeer"], 9.0)
    leak = Candidate("tesla|left|back", 0.8, {"knn": 1.0}, [5.0, 2.5])        # the full-bar leak tesla of the run
    centre = Candidate("tesla|middle|mid", 0.7, {"tac": 0.7}, [8.0, 10.0])
    knight = Candidate("knight|left|mid", 0.6, {"knn": 1.0})
    for mode in ("leak", "punish", "learned", "counter"):
        decision = ActDecision(True, mode, "left", "x")
        out = policy.filter_candidates([leak, centre, knight], decision, state)
        assert leak not in out and centre in out, mode
    assert _keys(policy.no_blind_spells([leak, centre, knight], state)) == ["tesla|middle|mid", "knight|left|mid"]
    # a defend decision whose engine places the tesla keeps it (the live loop puts it at the engine's tile)
    d = policy.decide(state, 100.0)
    assert d.act and "tesla" in d.allowed_cards and d.engine_tiles["tesla"] == [8.0, 10.0]
    assert _keys(policy.filter_candidates([leak, knight], d, state)) == ["tesla|left|back"]
    # the air exception tile is the engine's too
    balloon = _mover((3.5, 6.0), (3.5, 5.0), card="balloon", kalman=1.0, eta_tower={"tower": "own_left", "s": 0.25})
    st = _al([balloon], ["tesla", "knight", "zap", "fireball"], 9.0)
    d = policy.decide(st, 100.0)
    if d.act and "tesla" in (d.allowed_cards or ()):
        assert d.engine_tiles["tesla"] == [5.5, 6.5]
        assert _keys(policy.filter_candidates([Candidate("tesla|left|back", 0.8, {"tac": 0.8}, [5.5, 6.5])], d, st))


def test_am_a_building_window_opens_at_the_bridge_approach_and_not_for_a_giant_that_walks_away():
    policy = _sim()
    hand = ("tesla", "knight", "zap", "fireball")
    for row, acts in ((22, False), (20, True), (18, True), (16, True)):
        giant = _heading(_enemy((3.5, row), card="giant", eta=9.0, pred=(3.5, row - 1.5)), "advancing left lane")
        d = _sim().decide(_al([giant], hand, 6.0), 100.0)
        assert d.act is acts and (not acts or "tesla" in d.allowed_cards), row
    giant = _heading(_enemy((3.5, 18), card="giant", eta=9.0), "retreating")
    assert not _sim().decide(_al([giant], hand, 6.0), 100.0).act
