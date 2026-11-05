"""MAC-C: live state builder (perception GameState/PlayEvents -> clean state dict, D-015).

Small fakes built from cr_perception.state dataclasses; the real-data round trip is skipped when
dataset/v1/states.jsonl is missing. No model is loaded.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.features import featurize  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live_state import HandTracker, LiveConfig, LiveStateBuilder  # noqa: E402
from clashbot.state_text import render_context, render_threats, unit_token  # noqa: E402
from clashbot.units import load_unit_map  # noqa: E402
from cr_perception.state import GameState, PlayEvent, UnitObs  # noqa: E402
from tools.data import build_clean_dataset as bcd  # noqa: E402
from tools.eval.live_state_parity import main as parity_main  # noqa: E402


def _load_extract_states():
    """tools/extract_states.py by path, without replacing the 'extract_states' entry other tests use."""
    spec = importlib.util.spec_from_file_location("_live_state_test_extract_states",
                                                  ROOT / "tools" / "extract_states.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load_extract_states()
CARDS = load_cards()
UNIT_MAP = load_unit_map()
COSTS = {s: c["elixir_cost"] for s, c in CARDS.items() if c["elixir_cost"] is not None}
DECK = ["archers", "berserker", "bomb-tower", "goblinstein", "magic-archer", "royal-delivery",
        "tornado", "wall-breakers"]
OPP_CARDS = ["electro-wizard", "graveyard", "mighty-miner", "miner", "p-e-k-k-a", "ronin",
             "royal-ghost", "bats", "battle-healer", "fire-spirit"]

STATE_KEYS = ["clock", "phase", "match_seconds", "own", "opponent", "units", "threats"]
OWN_KEYS = ["elixir", "hand_confidence", "hero_cards", "towers", "hand", "next_card", "deck",
            "deck_status", "recent_plays"]
OPP_KEYS = ["elixir_estimate", "towers", "deck_known", "deck_complete", "recent_plays"]
TOWER_KEYS = ["king", "left", "right"]
PLAY_KEYS = ["t", "player", "card", "tile", "elixir_cost", "detect_source", "confidence"]
UNIT_KEYS = ["unit_class", "card", "is_evo", "side", "side_repaired", "tile", "heading",
             "speed_tiles_s", "predicted_tile_2s", "eta_tower"]


def builder(deck=DECK, **kw) -> LiveStateBuilder:
    return LiveStateBuilder(LiveConfig(own_deck=list(deck), **kw))


def make_gs(t=100.0, *, match_id=1, readiness="match", phase="double_elixir", elixir=7.5,
            hand=("tornado", "berserker", "archers", "bomb-tower"), next_card="wall-breakers",
            hand_conf=(0.9, 0.8, 0.7, 0.6), units=(), opp_known=(), own=None) -> GameState:
    if own is None:
        own = {"elixir": elixir, "hand": list(hand), "hand_conf": list(hand_conf),
               "next_card": next_card, "towers": {"king": 4000, "left": 3000, "right": 2500}}
    return GameState(t=t, frame_index=int(t * 10), readiness=readiness, match_id=match_id,
                     match_clock="1:20", match_seconds=100.0, phase=phase, own=own,
                     opponent={"elixir_est": 5.5, "towers": {"king": 4800, "left": 4000, "right": None},
                               "deck_known": list(opp_known), "deck_complete": False},
                     units=list(units))


def make_ev(ts, player="own", card="tornado", tile=(8, 10), source="arena") -> PlayEvent:
    return PlayEvent(timestamp=ts, match_clock="1:20", player=player, card=card, tile=tile,
                     elixir_before=None, elixir_after=None, detect_source=source, confidence="high")


def unit_obs(cls="knight", side="enemy", tile=(3, 12), category="unit", motion=None) -> UnitObs:
    return UnitObs(cls=cls, side=side, tile=tile, conf=0.9, bbox=(0, 0, 10, 10), track_id=5,
                   category=category, motion=motion)


# --------------------------------------------------------------------------- 1-2: gating and shape

def test_menu_and_empty_own_give_none():
    b = builder()
    assert b.build(make_gs(readiness="menu")) is None
    assert b.build(make_gs(own={})) is None
    assert b.build(make_gs(readiness="unreadable").to_json()) is None


def test_shape_and_key_order_and_context():
    b = builder(hero_cards=["wall-breakers"])
    b.observe_events([make_ev(95.0), make_ev(96.0, "opponent", "miner", (3, 20))])
    out = b.build(make_gs(100.0, opp_known=["bats"]))
    assert out is not None
    state, text = out
    assert list(state) == STATE_KEYS
    assert list(state["own"]) == OWN_KEYS
    assert list(state["opponent"]) == OPP_KEYS
    assert list(state["own"]["towers"]) == TOWER_KEYS == list(state["opponent"]["towers"])
    for p in state["own"]["recent_plays"] + state["opponent"]["recent_plays"]:
        assert list(p) == PLAY_KEYS
    assert state["own"]["deck"] == DECK and state["own"]["deck_status"] == "valid"
    assert state["own"]["hero_cards"] == ["wall-breakers"]
    assert state["own"]["hand_confidence"] == 0.75
    assert state["opponent"]["elixir_estimate"] == 5.5
    assert state["opponent"]["towers"] == {"king": 4800, "left": 4000, "right": None}
    assert text == render_context(state, 100.0)
    json.dumps(state)                                  # plain JSON types only


def test_dict_inputs_equal_object_inputs():
    gs, ev = make_gs(100.0), make_ev(95.0)
    a, b = builder(), builder()
    a.observe_events([ev])
    b.observe_events([ev.to_json()])
    assert a.build(gs) == b.build(gs.to_json())


# --------------------------------------------------------------------------- 3: sanitising

def test_hand_phase_elixir_sanitising():
    b = builder()
    gs = make_gs(hand=["tornado", "mega-minion", None, "berserker"], next_card="mega-minion",
                 elixir=12, phase="double_elixir_overtime")
    state, _ = b.build(gs)
    assert state["own"]["hand"] == ["tornado", None, None, "berserker"]
    assert state["own"]["next_card"] is None
    assert state["own"]["elixir"] == 10.0 and isinstance(state["own"]["elixir"], float)
    assert state["phase"] == "overtime"
    assert b.build(make_gs(phase="overtime"))[0]["phase"] == "overtime"
    assert b.build(make_gs(phase="weird"))[0]["phase"] is None
    assert b.build(make_gs(phase=None))[0]["phase"] is None
    assert b.build(make_gs(elixir=-1))[0]["own"]["elixir"] == 0.0
    assert b.build(make_gs(elixir=None))[0]["own"]["elixir"] is None
    short = b.build(make_gs(hand=["tornado"], hand_conf=[]))[0]
    assert short["own"]["hand"] == ["tornado", None, None, None]
    assert short["own"]["hand_confidence"] is None
    assert b.build(make_gs(next_card="tornado"))[0]["own"]["next_card"] == "tornado"


def test_unplayable_cards_and_incomplete_deck():
    # 7 cards: no deck_set (fewer than 8 distinct), so only the playable check applies
    b = builder(DECK[:7])
    state, _ = b.build(make_gs(hand=["tornado", "mega-minion", "tower-princess", "not-a-card"],
                               next_card="tower-princess"))
    assert state["own"]["deck_status"] == "incomplete"
    assert state["own"]["deck"] == sorted(DECK[:7])
    assert state["own"]["hand"] == ["tornado", "mega-minion", None, None]
    assert state["own"]["next_card"] is None
    assert "[deck incomplete]" in render_context(state, 100.0)
    state, _ = builder(["knight"]).build(make_gs())
    assert state["own"]["deck_status"] == "invalid"


# --------------------------------------------------------------------------- 4: recent plays

def test_recent_plays_windows_and_shape():
    t = 100.0
    own_events = [make_ev(t - 13, "own", "tornado"), make_ev(t - 5, "own", "berserker"),
                  make_ev(t - 2, "own", "mega-minion", source="hud"),   # HUD misread of a foreign card
                  make_ev(t - 3, "own", "not-a-card", source="arena"),  # unknown slug -> card None
                  make_ev(t - 1, "own", "mega-minion", source="arena")]  # trusted deploy label, kept
    opp_events = [make_ev(t - 12, "opponent", "miner", (3, 20), "arena"),
                  make_ev(t + 1, "opponent", "bats", (5, 20), "arena"),
                  make_ev(t - 4, "opponent", "mega-minion", (5, 20), "hud")]  # opponent HUD reads stay
    b = builder()
    b.observe_events(own_events + opp_events)
    state, _ = b.build(make_gs(t))

    def expected(evs):
        return bcd.clean_plays([ex.play_record(e.to_json(), COSTS) for e in evs], CARDS)

    assert state["own"]["recent_plays"] == expected([own_events[1], own_events[3], own_events[4]])
    assert state["opponent"]["recent_plays"] == expected([opp_events[0], opp_events[2]])
    assert [p["card"] for p in state["own"]["recent_plays"]] == ["berserker", None, "mega-minion"]
    assert [p["t"] for p in state["own"]["recent_plays"]] == [95.0, 97.0, 99.0]
    assert state["own"]["recent_plays"][0]["elixir_cost"] == 2.0
    assert state["opponent"]["recent_plays"][0]["tile"] == [3, 20]


def test_recent_s_is_configurable():
    b = builder(recent_s=3.0)
    b.observe_events([make_ev(96.0), make_ev(98.5, card="archers")])
    state, _ = b.build(make_gs(100.0))
    assert [p["card"] for p in state["own"]["recent_plays"]] == ["archers"]


# --------------------------------------------------------------------------- 5: opponent deck

def test_deck_known_caps_at_eight_in_first_seen_order():
    b = builder()
    # played out of alphabetical order: first-seen order is OPP_CARDS order
    b.observe_events([make_ev(10.0 * (i + 1), "opponent", c, (3, 20)) for i, c in enumerate(OPP_CARDS)])
    state, _ = b.build(make_gs(45.0))
    assert state["opponent"]["deck_known"] == OPP_CARDS[:4]       # later plays are in the future
    assert state["opponent"]["deck_complete"] is False
    state, _ = b.build(make_gs(105.0))
    assert state["opponent"]["deck_known"] == OPP_CARDS[:8]       # 10 seen, the 8 earliest kept
    assert state["opponent"]["deck_complete"] is True
    assert OPP_CARDS[:8] != sorted(OPP_CARDS[:8])


def test_deck_known_from_tracker_counts_from_first_appearance():
    b = builder()
    b.observe_events([make_ev(10.0, "opponent", "miner"), make_ev(30.0, "opponent", "ronin"),
                      make_ev(35.0, "opponent", "not-a-card")])
    assert b.build(make_gs(20.0))[0]["opponent"]["deck_known"] == ["miner"]
    s40 = b.build(make_gs(40.0, opp_known=["bats", "tower-princess"]))[0]["opponent"]
    assert s40["deck_known"] == ["miner", "ronin", "bats"]        # bats first shows at t=40
    b.observe_events([make_ev(45.0, "opponent", "fire-spirit")])
    s50 = b.build(make_gs(50.0, opp_known=["bats"]))[0]["opponent"]
    assert s50["deck_known"] == ["miner", "ronin", "bats", "fire-spirit"]
    s60 = b.build(make_gs(60.0, opp_known=[]))[0]["opponent"]     # the tracker forgetting changes nothing
    assert s60["deck_known"] == ["miner", "ronin", "bats", "fire-spirit"]
    assert s60["deck_complete"] is False


# --------------------------------------------------------------------------- 6: units

def test_units_tower_dropped_and_motion_matches_clean_units():
    motion = {"heading": "advancing left lane", "speed": 1.2, "pred_2s": [3, 10],
              "eta_tower": {"tower": "own_left", "s": 3.0}, "tile": [3, 12]}
    knight = unit_obs("knight", "enemy", (3, 12), motion=motion)
    archers = unit_obs("archers", "enemy", (9, 8), motion={"heading": "holding", "speed": 0.0,
                                                           "pred_2s": [9, 8], "eta_tower": None})
    towers = [unit_obs("queen-tower", "enemy", (3, 25), category="tower"),
              unit_obs("king-tower", "enemy", (8, 28))]           # dropped by class, not category
    gs = make_gs(units=[towers[0], knight, towers[1], archers])
    b = builder()
    state, text = b.build(gs)
    raw = [ex.unit_from_state(u.to_json()) for u in (knight, archers)]
    deck_known = state["opponent"]["deck_known"]
    expected, flipped, _ = bcd.clean_units(raw, "valid", DECK, deck_known, CARDS, UNIT_MAP)
    assert state["units"] == expected
    assert flipped == 1                                            # my own archers read as enemy -> ally
    assert [list(u) for u in state["units"]] == [UNIT_KEYS, UNIT_KEYS]
    assert state["units"][0]["unit_class"] == "knight" and state["units"][0]["eta_tower"] == motion["eta_tower"]
    assert state["units"][1]["side"] == "ally" and state["units"][1]["side_repaired"] is True
    assert state["threats"] == render_threats(state["units"])
    assert state["threats"] == [f"{unit_token(state['units'][0])}(e) advancing left lane, tower in 3.0s"]
    assert "knight(e)@[3,12] advancing left lane eta own_left 3s" in text
    assert text == render_context(state, 100.0)


def test_unit_without_motion_defaults():
    state, _ = builder().build(make_gs(units=[unit_obs("mystery-unit", "unknown", None)]))
    assert state["units"] == bcd.clean_units([ex.unit_from_state({"class": "mystery-unit", "side": "unknown"})],
                                             "valid", DECK, [], CARDS, UNIT_MAP)[0]
    assert state["units"][0]["tile"] == [0, 0] and state["threats"] == []


# --------------------------------------------------------------------------- 7-8: reset and dedupe

def test_new_match_forgets_events_and_first_seen():
    b = builder()
    b.observe_events([make_ev(95.0, "own", "tornado"), make_ev(96.0, "opponent", "miner", (3, 20))])
    s1, _ = b.build(make_gs(100.0, match_id=1))
    assert len(s1["own"]["recent_plays"]) == 1 and s1["opponent"]["deck_known"] == ["miner"]
    s2, _ = b.build(make_gs(101.0, match_id=2))
    assert s2["own"]["recent_plays"] == [] and s2["opponent"]["recent_plays"] == []
    assert s2["opponent"]["deck_known"] == []
    b.observe_events([make_ev(105.0, "own", "archers")])           # events of the new match accumulate
    s3, _ = b.build(make_gs(106.0, match_id=2))
    assert [p["card"] for p in s3["own"]["recent_plays"]] == ["archers"]


def test_events_observed_before_the_first_build_survive():
    b = builder()
    b.observe_events([make_ev(95.0)])
    assert len(b.build(make_gs(100.0, match_id=7))[0]["own"]["recent_plays"]) == 1


def test_reset_clears_everything():
    b = builder()
    b.observe_events([make_ev(95.0), make_ev(96.0, "opponent", "miner")])
    b.build(make_gs(100.0))
    b.reset()
    state, _ = b.build(make_gs(100.0))
    assert state["own"]["recent_plays"] == [] and state["opponent"]["deck_known"] == []


def test_duplicate_events_counted_once():
    b = builder()
    e1, e2 = make_ev(95.02), make_ev(95.04)                        # same rounded timestamp
    b.observe_events([e1, e2, e1.to_json()])
    b.observe_events([e1])
    b.observe_events([make_ev(95.02, "opponent", "tornado"), make_ev(95.02, card="archers")])
    b.observe_events([{"timestamp": None, "player": "own", "card": "tornado"}])  # no timestamp: ignored
    state, _ = b.build(make_gs(100.0))
    assert [(p["player"], p["card"]) for p in state["own"]["recent_plays"]] == [("own", "tornado"),
                                                                              ("own", "archers")]
    assert [p["card"] for p in state["opponent"]["recent_plays"]] == ["tornado"]


def test_unidentified_plays_at_same_time_dedupe_on_tile():
    b = builder()
    left = make_ev(95.02, "opponent", None, tile=[3, 20])
    right = make_ev(95.03, "opponent", None, tile=[14, 20])
    b.observe_events([left, right, left.to_json()])
    state, _ = b.build(make_gs(100.0))
    assert [p["tile"] for p in state["opponent"]["recent_plays"]] == [[3, 20], [14, 20]]
    assert all(p["card"] is None for p in state["opponent"]["recent_plays"])
    b2 = builder()                                                 # identical events (same tile): one
    b2.observe_events([make_ev(95.02, "opponent", None, tile=[3, 20]),
                       make_ev(95.04, "opponent", None, tile=(3, 20))])
    state2, _ = b2.build(make_gs(100.0))
    assert [p["tile"] for p in state2["opponent"]["recent_plays"]] == [[3, 20]]


def test_events_stay_sorted_when_observed_late():
    b = builder()
    b.observe_events([make_ev(98.0, card="archers")])
    b.observe_events([make_ev(92.0, card="tornado")])
    state, _ = b.build(make_gs(100.0))
    assert [p["t"] for p in state["own"]["recent_plays"]] == [92.0, 98.0]


# --------------------------------------------------------------------------- hand hold (FIX_LIST #21)

FULL_HAND = ("tornado", "berserker", "archers", "bomb-tower")
GREY = (None, None, None, None)


def test_greyed_hand_reuses_last_confident_hand():
    b = builder()
    assert b.build(make_gs(100.0, hand=FULL_HAND))[0]["own"]["hand"] == list(FULL_HAND)
    assert b.last_hand_reused is False
    state, text = b.build(make_gs(101.0, hand=GREY, hand_conf=()))
    assert state["own"]["hand"] == list(FULL_HAND)
    assert state["own"]["hand_confidence"] is None
    assert b.last_hand_reused is True
    assert text == render_context(state, 101.0)
    assert list(state["own"]) == OWN_KEYS                           # (state, context_text) contract unchanged
    b.build(make_gs(102.0, hand=FULL_HAND))                         # a fresh read clears the flag
    assert b.last_hand_reused is False


def test_greyed_hand_stays_empty_after_hold_window():
    b = builder()
    b.build(make_gs(100.0, hand=FULL_HAND))
    state, _ = b.build(make_gs(102.5, hand=GREY, hand_conf=()))
    assert state["own"]["hand"] == [None] * 4 and b.last_hand_reused is False
    state, _ = b.build(make_gs(102.0, hand=GREY, hand_conf=()).to_json())   # exactly hand_hold_s: still held
    assert state["own"]["hand"] == list(FULL_HAND) and b.last_hand_reused is True


def test_reused_hand_drops_cards_played_since_the_read():
    b = builder()
    b.build(make_gs(100.0, hand=FULL_HAND))
    b.observe_events([make_ev(99.5, card="archers"),                # before the read: already reflected in it
                      make_ev(100.5, card="tornado"),
                      make_ev(100.6, "opponent", "berserker", (3, 20))])  # opponent plays do not matter
    state, _ = b.build(make_gs(101.0, hand=GREY, hand_conf=()))
    assert state["own"]["hand"] == [None, "berserker", "archers", "bomb-tower"]
    assert b.last_hand_reused is True
    b.observe_events([make_ev(100.8, card="berserker"), make_ev(100.9, card="archers"),
                      make_ev(100.95, card="bomb-tower")])
    state, _ = b.build(make_gs(101.0, hand=GREY, hand_conf=()))     # nothing left to reuse
    assert state["own"]["hand"] == [None] * 4 and b.last_hand_reused is False


def test_hand_hold_disabled_and_partial_reads_not_reused():
    b = builder(hand_hold_s=0)
    b.build(make_gs(100.0, hand=FULL_HAND))
    state, _ = b.build(make_gs(100.5, hand=GREY, hand_conf=()))
    assert state["own"]["hand"] == [None] * 4 and b.last_hand_reused is False
    b2 = builder()
    assert b2.build(make_gs(100.0, hand=GREY, hand_conf=()))[0]["own"]["hand"] == [None] * 4   # nothing to reuse yet
    assert b2.last_hand_reused is False
    b2.build(make_gs(101.0, hand=("tornado", None, None, None), hand_conf=(0.9,)))  # 1 known slot: not remembered
    assert b2.build(make_gs(101.5, hand=GREY, hand_conf=()))[0]["own"]["hand"] == [None] * 4
    state, _ = b2.build(make_gs(102.0, hand=("tornado", None, "archers", None), hand_conf=(0.9, 0.9)))
    assert state["own"]["hand_confidence"] == 0.9
    state, _ = b2.build(make_gs(102.5, hand=("berserker", None, None, None), hand_conf=(0.9,)))
    assert state["own"]["hand"] == ["berserker", None, None, None] and b2.last_hand_reused is False


def test_reused_hand_does_not_extend_the_hold_and_reset_clears_it():
    b = builder()
    b.build(make_gs(100.0, hand=FULL_HAND))
    assert b.build(make_gs(101.5, hand=GREY, hand_conf=()))[0]["own"]["hand"] == list(FULL_HAND)
    assert b.build(make_gs(103.0, hand=GREY, hand_conf=()))[0]["own"]["hand"] == [None] * 4   # 3.0 s after the read
    b.build(make_gs(110.0, hand=FULL_HAND))
    b.reset()
    assert b.build(make_gs(110.5, hand=GREY, hand_conf=()))[0]["own"]["hand"] == [None] * 4
    assert b.last_hand_reused is False
    b.build(make_gs(120.0, match_id=1, hand=FULL_HAND))
    state, _ = b.build(make_gs(120.5, match_id=2, hand=GREY, hand_conf=()))   # a new match forgets the old hand
    assert state["own"]["hand"] == [None] * 4 and b.last_hand_reused is False


def test_last_hand_reused_resets_when_build_returns_none():
    b = builder()
    b.build(make_gs(100.0, hand=FULL_HAND))
    b.build(make_gs(101.0, hand=GREY, hand_conf=()))
    assert b.last_hand_reused is True
    assert b.build(make_gs(101.5, readiness="menu")) is None
    assert b.last_hand_reused is False


# --------------------------------------------------------------------------- positional hand memory (FIX_LIST #26)

TDECK = list("abcdefgh")                      # 8 distinct slugs: the tracker never looks them up in the card table
NONE4 = [None] * 4


def test_tracker_remembers_a_greyed_hand_for_ever():
    tr = HandTracker(TDECK)
    assert tr.update(list("abcd"), "e", 100.0) == list("abcd") and tr.filled == [False] * 4
    for k in range(1, 101):                                          # 10 s of all-None reads
        assert tr.update(NONE4, None, 100.0 + k * 0.1) == list("abcd")
    assert tr.filled == [True] * 4 and tr.next == "e"
    assert tr.update(["a", None, None, None], None, 111.0) == list("abcd")
    assert tr.filled == [False, True, True, True]


def test_tracker_play_moves_the_next_card_into_the_slot():
    tr = HandTracker(TDECK)
    tr.update(list("abcd"), "e", 100.0)
    tr.note_play("b", 1, 100.5)
    assert tr.slots == ["a", "e", "c", "d"] and tr.log == ["b"] and tr.pending == {1: ("b", 100.5)}
    assert tr.next is None                                           # < 4 plays logged: the queue is unknown
    assert tr.update(NONE4, "f", 100.6) == ["a", "e", "c", "d"] and tr.next == "f"
    assert tr.update(["a", "e", "c", "d"], "f", 102.5) == ["a", "e", "c", "d"] and tr.pending == {}   # it landed


def test_tracker_ignores_a_lagging_read_and_restores_a_failed_play():
    tr = HandTracker(TDECK)
    tr.update(list("abcd"), "e", 100.0)
    tr.note_play("b", 1, 100.5)
    assert tr.update(list("abcd"), "e", 101.0)[1] == "e"             # 0.5 s after the tap: the screen still shows b
    assert tr.log == ["b"] and 1 in tr.pending
    assert tr.update(list("abcd"), "e", 102.0) == list("abcd")       # 1.5 s on: the tap did not land
    assert tr.log == [] and tr.pending == {} and tr.filled == [False] * 4
    tr.note_play("b", 1, 103.0)
    assert tr.update(["a", "x", "c", "d"], None, 103.2)[1] == "x" and tr.pending == {}   # another card: a normal read


def test_tracker_failed_play_recomputes_a_next_derived_from_the_log():
    tr = HandTracker(TDECK)
    for k, card in enumerate("abcd"):
        tr.note_play(card, k, 100.0 + k)
    assert tr.log == list("abcd") and tr.next == "a"
    tr.update(["a", None, None, None], None, 105.0)                  # slot 0 shows a again: that tap never landed
    assert tr.log == list("bcd") and tr.next is None and tr.slots[0] == "a"


def test_tracker_one_card_sits_in_one_slot_only():
    tr = HandTracker(TDECK)
    tr.update(list("abcd"), None, 100.0)
    assert tr.update([None, None, "a", None], None, 100.1) == [None, "b", "a", "d"]
    tr2 = HandTracker(TDECK)
    tr2.update(list("abcd"), None, 100.0)
    tr2.note_play("e", 0, 100.1)
    tr2.update(NONE4, None, 102.0)                                   # the play landed (slot unread for 1.5 s+)
    assert tr2.log == ["e"] and tr2.pending == {}
    tr2.update(["e", None, None, None], None, 105.0)                 # e is in the hand: that play did not happen
    assert tr2.log == [] and tr2.slots[0] == "e"


def test_tracker_cycle_fills_the_single_unknown_slot_and_clears_stale_ones():
    tr = HandTracker(TDECK)
    tr.update(list("abcd"), "e", 100.0)
    tr.note_play("a", 0, 101.0)                                      # a -> e
    tr.update(NONE4, "f", 101.1)
    tr.note_play("b", 1, 102.0)                                      # b -> f
    tr.update(NONE4, "g", 102.1)
    tr.note_play("c", 2, 103.0)                                      # c -> g
    assert tr.next is None and tr.slots == ["e", "f", "g", "d"]
    tr.note_play("d", 3, 104.0)                                      # d -> unknown: the next card was never read
    assert tr.slots == ["e", "f", "g", None] and tr.log == list("abcd")
    assert tr.update(NONE4, None, 104.1) == ["e", "f", "g", "h"]     # the one card left in the deck
    assert tr.next == "a" == tr.log[-4]
    assert tr.filled == [True] * 4
    tr.slots[2] = "b"                                                # a memory slot holding a queued card is stale
    assert tr.update(NONE4, None, 104.2) == ["e", "f", "g", "h"]     # cleared, then refilled by the cycle
    tr.slots[0], tr.slots[3] = "c", "d"
    assert tr.update(NONE4, None, 104.3) == [None, "f", "g", None]   # two unknown slots: no guess


def test_tracker_cycle_needs_a_known_deck_and_four_plays():
    tr = HandTracker(TDECK[:7])                                      # not 8 cards: no cycle inference
    assert tr.deck is None
    tr.update(["e", "f", "g", None], None, 100.0)
    for k, card in enumerate("abcd"):
        tr.note_play(card, 3, 101.0 + k)
    assert tr.update(NONE4, None, 106.0)[3] is None
    tr2 = HandTracker(TDECK)
    tr2.update(["e", "f", "g", None], None, 100.0)
    tr2.note_play("a", 3, 101.0)
    assert tr2.update(NONE4, None, 106.0)[3] is None                 # only one play logged


def test_tracker_reset_forgets_everything():
    tr = HandTracker(TDECK)
    tr.update(list("abcd"), "e", 100.0)
    tr.note_play("a", 0, 101.0)
    tr.reset()
    assert tr.slots == NONE4 and tr.next is None and tr.log == [] and tr.pending == {} and tr.filled == [False] * 4


def test_builder_tracker_is_opt_in_and_sized_by_the_deck():
    assert LiveConfig(own_deck=DECK).hand_tracker is False
    assert builder().tracker is None
    b = builder(hand_tracker=True)
    assert b.tracker is not None and b.tracker.deck == sorted(DECK)
    assert builder(deck=DECK[:7], hand_tracker=True).tracker.deck is None
    builder().note_own_play("tornado", 0, 100.0)                     # no tracker: a no-op


def test_builder_without_tracker_matches_the_default_output():
    plain, off = builder(), builder(hand_tracker=False)
    for t, hand in ((100.0, FULL_HAND), (101.0, GREY), (103.5, GREY), (104.0, FULL_HAND)):
        gs = make_gs(t, hand=hand, hand_conf=() if hand == GREY else (0.9, 0.8, 0.7, 0.6))
        assert json.dumps(off.build(gs)) == json.dumps(plain.build(gs))
        assert off.last_hand_reused == plain.last_hand_reused


def test_builder_tracker_keeps_greyed_cards_and_flags_reuse():
    b = builder(hand_tracker=True)
    state, _ = b.build(make_gs(100.0, hand=FULL_HAND))
    assert state["own"]["hand"] == list(FULL_HAND) and b.last_hand_reused is False
    assert state["own"]["hand_confidence"] == 0.75 and state["own"]["next_card"] == "wall-breakers"
    for t in (101.0, 110.0, 160.0):                                  # no time limit: the memory lasts
        state, text = b.build(make_gs(t, hand=GREY, hand_conf=(), next_card=None))
        assert state["own"]["hand"] == list(FULL_HAND) and b.last_hand_reused is True
        assert state["own"]["hand_confidence"] is None and state["own"]["next_card"] == "wall-breakers"
        assert list(state["own"]) == OWN_KEYS and text == render_context(state, t)
    state, _ = b.build(make_gs(161.0, hand=("tornado", None, "archers", "bomb-tower"), hand_conf=(0.9, 0.9, 0.9)))
    assert state["own"]["hand"] == list(FULL_HAND) and b.last_hand_reused is True      # slot 1 still remembered
    b.build(make_gs(162.0, hand=FULL_HAND))
    assert b.last_hand_reused is False


def test_builder_note_own_play_moves_the_next_card_into_the_slot():
    b = builder(hand_tracker=True)
    b.build(make_gs(100.0, hand=FULL_HAND))                          # next card: wall-breakers
    b.note_own_play("tornado", 0, 100.5)
    state, _ = b.build(make_gs(101.0, hand=GREY, hand_conf=(), next_card=None))
    assert state["own"]["hand"] == ["wall-breakers", "berserker", "archers", "bomb-tower"]
    assert b.last_hand_reused is True


def test_builder_new_match_resets_the_tracker():
    b = builder(hand_tracker=True)
    b.build(make_gs(100.0, match_id=1, hand=FULL_HAND))
    b.note_own_play("tornado", 0, 100.5)
    state, _ = b.build(make_gs(101.0, match_id=2, hand=GREY, hand_conf=()))
    assert state["own"]["hand"] == [None] * 4 and b.last_hand_reused is False
    assert b.tracker.log == [] and b.tracker.pending == {}
    b.build(make_gs(110.0, match_id=2, hand=FULL_HAND))
    b.reset()
    assert b.tracker.slots == [None] * 4


# --------------------------------------------------------------------------- 9: real-data round trip

DATASET_STATES = DATASET_DIR / "states.jsonl"


def _val_records(n: int) -> list[dict]:
    out = []
    with DATASET_STATES.open(encoding="utf-8") as fh:
        for line in fh:
            if '"split": "val"' not in line:
                continue
            rec = json.loads(line)
            if rec["split"] == "val":
                out.append(rec)
                if len(out) == n:
                    break
    return out


def _perception_like(rec: dict) -> tuple[dict, list[dict]]:
    st = rec["state"]
    own, opp = st["own"], st["opponent"]
    units = [{"class": u["unit_class"], "side": u["side"], "tile": u["tile"], "category": "unit",
              "motion": {"heading": u["heading"], "speed": u["speed_tiles_s"],
                         "pred_2s": u["predicted_tile_2s"], "eta_tower": u["eta_tower"]}}
             for u in st["units"]]
    gs = {"t": rec["t"], "readiness": "match", "match_id": 1, "match_clock": st["clock"],
          "match_seconds": st["match_seconds"], "phase": st["phase"],
          "own": {"elixir": own["elixir"], "hand": own["hand"], "hand_conf": None,
                  "next_card": own["next_card"], "towers": own["towers"]},
          "opponent": {"elixir_est": opp["elixir_estimate"], "towers": opp["towers"],
                       "deck_known": opp["deck_known"]},
          "units": units}
    events = [{"timestamp": p["t"], "player": p["player"], "card": p["card"], "tile": p["tile"],
               "detect_source": p["detect_source"], "confidence": p["confidence"]}
              for p in own["recent_plays"] + opp["recent_plays"]]
    return gs, events


@pytest.mark.skipif(not DATASET_STATES.exists(), reason="dataset/v1/states.jsonl not built")
def test_round_trip_on_real_records():
    recs = _val_records(50)
    assert len(recs) == 50
    close = 0
    for rec in recs:
        st, t = rec["state"], rec["t"]
        gs, events = _perception_like(rec)
        b = builder(st["own"]["deck"], hero_cards=st["own"]["hero_cards"] or [])
        b.observe_events(events)
        live, text = b.build(gs)
        assert live["own"]["hand"] == st["own"]["hand"], rec["id"]
        assert live["own"]["elixir"] == st["own"]["elixir"], rec["id"]
        assert live["phase"] == st["phase"], rec["id"]
        assert live["own"]["deck"] == st["own"]["deck"] and live["own"]["deck_status"] == st["own"]["deck_status"]
        assert text == render_context(live, t)
        close += float(featurize(live, t) @ featurize(st, t)) >= 0.99
    assert close >= 45, f"only {close}/50 records reach cosine 0.99"


# --------------------------------------------------------------------------- 10: parity tool

def test_parity_tool_on_tiny_game(tmp_path):
    states = [make_gs(20.0, match_id=3), make_gs(30.0, match_id=3, hand=("archers", "tornado", None, None))]
    events = [make_ev(25.0, "own", "tornado"), make_ev(26.0, "opponent", "miner", (3, 20))]
    perception = tmp_path / "states.jsonl"
    perception.write_text("\n".join(json.dumps(r.to_json()) for r in [*states[:1], *events, *states[1:]]) + "\n")
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "games.jsonl").write_text(json.dumps({"game_id": "g1", "own_deck": DECK}) + "\n")
    recs = []
    for gs in states:
        b = builder()
        b.observe_events([e for e in events if e.timestamp <= gs.t])
        state, text = b.build(gs)
        recs.append({"id": f"g1#{gs.t:.1f}", "game_id": "g1", "t": gs.t, "state": state, "context_text": text})
    recs.append({"id": "other#1.0", "game_id": "other", "t": 1.0, "state": recs[0]["state"], "context_text": ""})
    (dataset / "states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    out = tmp_path / "parity.json"
    assert parity_main(["--perception", str(perception), "--game", "g1", "--dataset", str(dataset),
                        "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["game_id"] == "g1" and res["n_records"] == 2 and res["n_compared"] == 2
    assert res["no_state"] == 0
    assert res["mean"]["phase"] == 1.0 and res["mean"]["context_exact"] == 1.0
    assert res["mean"]["units_count_diff"] == 0.0
    assert res["median_feature_cosine"] == pytest.approx(1.0, abs=1e-4)
    assert len(res["examples"]) == 2 and res["examples"][0]["id"] == "g1#20.0"

    # default output path (<dataset>/eval/...) and an exact-match tolerance
    assert parity_main(["--perception", str(perception), "--game", "g1", "--dataset", str(dataset),
                        "--tol", "0.0"]) == 0
    default = json.loads((dataset / "eval" / "live_state_parity_g1.json").read_text())
    assert default["n_compared"] == 2

    far = tmp_path / "far.jsonl"
    far.write_text(json.dumps(states[0].to_json()) + "\n")
    assert parity_main(["--perception", str(far), "--game", "g1", "--dataset", str(dataset),
                        "--tol", "5.0", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["n_compared"] == 1 and res["no_state"] == 1
