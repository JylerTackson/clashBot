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
from clashbot.live_state import LiveConfig, LiveStateBuilder  # noqa: E402
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
