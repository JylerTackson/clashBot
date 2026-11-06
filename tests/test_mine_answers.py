"""MAC-AK K2: pro-answer mining (tools/data/mine_pro_answers.py) on synthetic games; a real-data smoke test is skipped
when dataset/v1 is missing."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402
from tools.data import mine_pro_answers as m  # noqa: E402

OWN_DECK = ["cannon", "knight", "zap", "musketeer", "ice-spirit", "skeletons", "fireball", "the-log"]
OPP_DECK = ["giant", "zap", "witch", "miner", "musketeer", "bats", "fireball", "hog-rider"]


# --------------------------------------------------------------------------- builders
def play(t, card, tile, src="deploy_label", player="opponent"):
    return {"t": t, "player": player, "card": card, "tile": tile, "elixir_cost": None, "detect_source": src,
            "confidence": "high"}


def rec(t, opp=(), own=(), action=None, units=(), left=4858, right=4858, elixir=8.0):
    return {"t": t, "action": action or {"type": "unknown"},
            "state": {"own": {"elixir": elixir, "towers": {"king": None, "left": left, "right": right},
                              "recent_plays": list(own)},
                      "opponent": {"elixir_estimate": 5.0, "recent_plays": list(opp)}, "units": list(units)}}


def act(card, tile, delay=1.0):
    return {"type": "play", "card": card, "tile": tile, "delay_s": delay}


def unit(card, tile, eta=None, side="enemy"):
    return {"unit_class": card, "card": card, "side": side, "tile": tile,
            "eta_tower": None if eta is None else {"tower": "own_right", "s": eta}}


def snap(t, units=(), left=4858, right=4858, elixir=8.0):
    """A dense Snap whose enemy units are (card, tile, eta) triples."""
    return m.Snap(float(t), elixir, 5.0, left, right,
                  tuple(m.EnemyUnit(frozenset({c}), c, (float(x), float(y)), eta) for c, (x, y), eta in units))


def game(**kw):
    g = {"game_id": "g1", "split": "train", "own_deck": OWN_DECK, "deck_status": "valid", "opponent_deck": OPP_DECK}
    g.update(kw)
    return g


def event(threat="giant", answer="cannon", split="train", offset=(5.0, -3.0), held=True, trade=2.0, dt=2.0,
          kind="building", offset_ok=True, eta=4.0, dist=9.0, game_id="g1", answer_t=12.0, key="giant|same|3-6|7-10"):
    return {"game_id": game_id, "split": split, "threat": threat, "answer": answer, "answer_kind": kind,
            "answer_t": answer_t, "offset": list(offset), "offset_ok": offset_ok, "held": held, "trade": trade,
            "dt_s": dt, "eta_at_answer": eta, "dist_at_answer": dist, "threat_cost": m.card_cost(threat),
            "situation": m.parse_situation_key(key)}


# --------------------------------------------------------------------------- dedupe
def test_dedupe_same_play_in_many_records():
    raw = [{"t": 10.0, "card": "giant", "tile": [14, 20], "prio": 1},
           {"t": 10.2, "card": "giant", "tile": [14, 21], "prio": 1},
           {"t": 10.9, "card": "giant", "tile": None, "prio": 2},          # unknown tile joins
           {"t": 30.0, "card": "giant", "tile": [14, 20], "prio": 1},      # a later play of its own
           {"t": 10.1, "card": "giant", "tile": [3, 20], "prio": 1}]       # same time, 11 tiles away: another play
    plays, st = m.dedupe_plays(raw)
    assert sorted((p["t"], p["tile"]) for p in plays) == [(10.0, (14.0, 20.0)), (10.1, (3.0, 20.0)), (30.0, (14.0, 20.0))]
    assert st["dup"] == 2 and st["persist"] == 0


def test_dedupe_lingering_label_chain_and_distinct_cards():
    raw = [{"t": t, "card": "void", "tile": [7, 8], "prio": 1} for t in (582.0, 585.0, 588.0, 591.0)]
    raw += [{"t": 582.0, "card": "miner", "tile": [7, 8], "prio": 1}]      # another card: never merged
    plays, st = m.dedupe_plays(raw)
    assert Counter(p["card"] for p in plays) == {"void": 1, "miner": 1}
    assert st["persist"] == 3 and st["dup"] == 0
    # 6 s apart is beyond any label and any real card cycle: two plays
    plays, _ = m.dedupe_plays([{"t": 0.0, "card": "miner", "tile": [2, 28], "prio": 1},
                               {"t": 6.0, "card": "miner", "tile": [2, 28], "prio": 1}])
    assert len(plays) == 2


def test_dedupe_keeps_the_best_tile_and_first_time():
    plays, _ = m.dedupe_plays([{"t": 5.0, "card": "cannon", "tile": None, "prio": 2},
                               {"t": 5.2, "card": "cannon", "tile": [8, 11], "prio": 2},
                               {"t": 5.3, "card": "cannon", "tile": [9, 10], "prio": 0}])
    assert len(plays) == 1 and plays[0]["t"] == 5.0 and plays[0]["tile"] == (9.0, 10.0) and plays[0]["n"] == 3


def test_raw_and_clean_plays_drop_unknown_cards_and_off_deck():
    records = [rec(10.5, opp=[play(10.0, "giant", [14, 20]), play(10.0, None, [9, 20], "inferred")],
                   own=[play(9.0, "mega-knight", [8, 5], player="own"), play(9.5, "cannon", [8, 9], player="own")]),
               rec(11.0, action=act("knight", [6, 7]), opp=[play(10.0, "giant", [14, 20])])]
    st = Counter()
    opp = m.clean_plays(m.raw_plays(records, "opponent"), st, "opp")
    own = m.clean_plays(m.raw_plays(records, "own"), st, "own", OWN_DECK)
    assert [(p["card"], p["t"]) for p in opp] == [("giant", 10.0)]
    assert st["opp_unknown_card"] == 1 and st["opp_dup_collapsed"] == 1
    assert [(p["card"], p["t"]) for p in own] == [("cannon", 9.5), ("knight", 12.0)]     # action at t + delay_s
    assert st["own_off_deck"] == 1


# --------------------------------------------------------------------------- lane mirroring
def test_mirror_offset_left_and_right_lane():
    # threat in the left lane, building toward the centre (larger x) and in front (smaller y)
    assert m.mirror_offset((8, 9), (3, 12)) == (5, -3, False)
    # the same play in the right lane: dx is negated so the centre is still positive
    assert m.mirror_offset((9, 9), (14, 12)) == (5, -3, True)
    # x = 9 is the right half
    dx, dy, mirrored = m.mirror_offset((12, 9), (9, 12))
    assert (dx, dy, mirrored) == (-3, -3, True)
    assert m.lane_name((8.9, 10)) == "left" and m.lane_name((9, 10)) == "right"


def test_events_of_mirrored_lanes_agree():
    def run(threat_x, answer_x):
        recs = [rec(10.5, opp=[play(10.0, "giant", [threat_x, 17])]), rec(11.0, action=act("cannon", [answer_x, 9]))]
        ev, _ = m.mine_game(game(), recs, [], Counter(), use_unit_threats=False)
        assert len(ev) == 1
        return ev[0]
    left, right = run(3, 8), run(14, 9)
    assert left["offset"] == right["offset"] == [5.0, -8.0]
    assert (left["mirrored"], right["mirrored"], left["lane"], right["lane"]) == (False, True, "left", "right")


# --------------------------------------------------------------------------- defensive filter
def _answer_of(plays_own, threat=("giant", [9, 22]), dense=()):
    recs = [rec(10.5, opp=[play(10.0, threat[0], threat[1])])]
    for t, card, tile in plays_own:
        recs.append(rec(t - 1.0, action=act(card, tile)))
    ev, _ = m.mine_game(game(), recs, list(dense), Counter(), use_unit_threats=False)
    return ev


def test_defensive_filter_rows_and_window():
    # t=11.5: a troop on the enemy half is not defensive; t=12: row 16 (the river) is; the first defensive wins
    ev = _answer_of([(11.5, "knight", [8, 20]), (12.0, "musketeer", [8, 16]), (13.0, "cannon", [8, 9])])
    assert [(e["answer"], e["dt_s"]) for e in ev] == [("musketeer", 2.0)]
    # row 17 is the enemy half
    assert _answer_of([(12.0, "knight", [8, 17])]) == []
    # outside [P.t, P.t + 8]
    assert _answer_of([(18.5, "cannon", [8, 9])]) == []
    assert [e["answer"] for e in _answer_of([(18.0, "cannon", [8, 9])])] == ["cannon"]
    # a play with no tile cannot be judged: skipped, the next one answers
    recs = [rec(10.5, opp=[play(10.0, "giant", [9, 22])]),
            rec(11.0, action=act("knight", None)), rec(12.0, action=act("cannon", [8, 9]))]
    st = Counter()
    ev, _ = m.mine_game(game(), recs, [], st, use_unit_threats=False)
    assert [e["answer"] for e in ev] == ["cannon"] and st["answer_candidates_no_tile"] == 1


def test_defensive_spell_within_four_tiles_of_the_threat():
    near = _answer_of([(12.0, "zap", [9, 19])])              # 3 tiles from the giant at (9, 22)
    assert [e["answer"] for e in near] == ["zap"] and near[0]["answer_kind"] == "spell"
    assert _answer_of([(12.0, "zap", [9, 17.5])]) == []      # 4.5 tiles away, on the enemy half
    # the spell is measured against the tracked position of the threat when we have one
    dense = [snap(12.0, [("giant", (9, 17.5), 9.0)])]
    assert _answer_of([(12.0, "zap", [9, 22.5])]) != []                                     # 0.5 tiles from where it was played
    assert _answer_of([(12.0, "zap", [9, 22.5])], dense=dense) == []                        # ... but 5 from where it is
    assert [e["answer"] for e in _answer_of([(12.0, "zap", [9, 20])], dense=dense)] == ["zap"]
    # a troop on the enemy half is not rescued by being close
    assert _answer_of([(12.0, "knight", [9, 20])]) == []


def test_non_threat_spells_are_not_threats():
    assert not m.is_threat_card("zap") and not m.is_threat_card("fireball")
    assert m.is_threat_card("graveyard") and m.is_threat_card("goblin-barrel")        # spell cards that are win conditions
    assert m.is_threat_card("giant") and m.is_threat_card("cannon")
    st = Counter()
    recs = [rec(10.5, opp=[play(10.0, "zap", [9, 22])]), rec(11.0, action=act("cannon", [8, 9]))]
    assert m.mine_game(game(), recs, [], st, use_unit_threats=False)[0] == [] and st["threat_dropped_nonthreat_spell"] == 1


# --------------------------------------------------------------------------- held
def test_held_from_tower_hp():
    def tl(series, lane="left"):
        return m.Timeline([snap(t, **{lane: hp}) for t, hp in series])
    flat = tl([(t, 4858) for t in range(0, 30)])
    assert flat.held("left", 10.0) is True
    hit = tl([(t, 4858 if t < 15 else 4400) for t in range(0, 30)])
    assert hit.held("left", 10.0) is False                       # -458 inside the 12 s window
    assert hit.held("left", 3.0) is False                        # the hit at 15 is the last second of the window (3, 15]
    assert hit.held("left", 2.0) is True                         # ... and one second beyond this one
    late = tl([(t, 4858 if t < 25 else 4000) for t in range(0, 40)])
    assert late.held("left", 10.0) is True                       # the hit at 25 is beyond 10 + 12
    small = tl([(t, 4858 if t < 15 else 4600) for t in range(0, 30)])
    assert small.held("left", 10.0) is True                      # -258 < 300
    glitch = tl([(t, 658 if t == 14 else 4858) for t in range(0, 30)])
    assert glitch.held("left", 10.0) is True                     # a one-sample OCR glitch is filtered
    assert tl([(t, 4858) for t in range(0, 8)]).held("left", 5.0) is None            # no reading 6 s later
    assert tl([(t, None) for t in range(0, 30)]).held("left", 10.0) is None          # no HP at all
    assert tl([(t, 4858) for t in range(20, 40)]).held("left", 10.0) is None         # no starting reading
    # the right lane reads the right tower
    both = m.Timeline([snap(t, left=4858, right=4858 if t < 15 else 3000) for t in range(0, 30)])
    assert both.held("left", 10.0) is True and both.held("right", 10.0) is False


# --------------------------------------------------------------------------- one game end to end
def test_mine_game_event_fields():
    recs = [rec(10.5, opp=[play(10.0, "giant", [14, 20]), play(10.0, "zap", [9, 20])], elixir=8.0),
            rec(11.0, action=act("cannon", [9, 9], delay=1.0), opp=[play(10.0, "giant", [14, 20])], elixir=6.0)]
    dense = [snap(t, [("giant", (14, 20 - 2 * (t - 10)), 12.0 - (t - 10))], elixir=6.0 + (t - 11) * 0.5,
                  right=4858 if t < 40 else 4000) for t in range(9, 26)]
    st = Counter()
    ev, plays = m.mine_game(game(), recs, dense, st, use_unit_threats=False)
    assert len(ev) == 1
    e = ev[0]
    assert (e["threat"], e["answer"], e["split"], e["game_id"]) == ("giant", "cannon", "train", "g1")
    assert (e["threat_t"], e["answer_t"], e["dt_s"]) == (10.0, 12.0, 2.0)
    assert e["threat_tile_src"] == "tracked" and e["threat_tile_at_answer"] == [14.0, 16.0]
    assert e["answer_tile"] == [9.0, 9.0] and e["offset"] == [5.0, -7.0] and e["mirrored"] is True
    assert e["eta_at_answer"] == 10.0 and e["offset_ok"] is True
    assert e["dist_at_answer"] == pytest.approx(((14 - 14.5) ** 2 + (16 - 4.3) ** 2) ** 0.5, abs=1e-2)
    assert e["own_elixir"] == 6.5 and e["opp_elixir_est"] == 5.0
    assert e["held"] is True and e["threat_cost"] == 5.0 and e["answer_cost"] == 3.0 and e["trade"] == 2.0
    assert e["situation"]["key"] == "giant|same|6-10|4-6"
    assert st["threat_dropped_nonthreat_spell"] == 1 and st["threats_unanswered"] == 0
    assert [p["card"] for p in plays["own"]] == ["cannon"] and [p["card"] for p in plays["opp"]] == ["giant", "zap"]
    # the tower in the threat's lane falls 12 s later, inside the window of an answer 30 s later: held = False
    late = [rec(10.5, opp=[play(30.0, "giant", [14, 20])]), rec(31.0, action=act("cannon", [9, 9]))]
    dense2 = [snap(t, [("giant", (14, 16), 5.0)], right=4858 if t < 36 else 3000) for t in range(25, 60)]
    assert m.mine_game(game(), late, dense2, Counter(), use_unit_threats=False)[0][0]["held"] is False
    # without the dense states the threat stays where it was played
    ev, _ = m.mine_game(game(), recs, [], Counter(), use_unit_threats=False)
    assert ev[0]["threat_tile_src"] == "play" and ev[0]["threat_tile_at_answer"] == [14.0, 20.0]
    assert ev[0]["held"] is None or ev[0]["held"] is True


def test_event_dropped_without_any_threat_tile():
    recs = [rec(10.5, opp=[play(10.0, "giant", None)]), rec(11.0, action=act("cannon", [9, 9]))]
    st = Counter()
    assert m.mine_game(game(), recs, [], st, use_unit_threats=False)[0] == []
    assert st["events_dropped_threat_tile_missing"] == 1
    # ... but a tracked unit gives the tile
    ev, _ = m.mine_game(game(), recs, [snap(12.0, [("giant", (3, 14), None)])], Counter(), use_unit_threats=False)
    assert len(ev) == 1 and ev[0]["threat_tile_at_answer"] == [3.0, 14.0]


def test_invalid_own_deck_has_no_answers():
    recs = [rec(10.5, opp=[play(10.0, "giant", [14, 20])]), rec(11.0, action=act("cannon", [9, 9]))]
    ev, plays = m.mine_game(game(deck_status="invalid"), recs, [], Counter(), use_unit_threats=False)
    assert ev == [] and plays["own"] == []


def test_unit_threats_need_an_unexplained_crossing():
    recs = [rec(20.0, action=act("cannon", [9, 9], delay=0.0))]
    dense = [snap(t, [("hog-rider", (3, 15), 4.0)] if 17 <= t <= 22 else []) for t in range(10, 30)]
    st = Counter()
    ev, _ = m.mine_game(game(), recs, dense, st)
    assert [(e["threat"], e["threat_source"], e["threat_t"], e["dt_s"]) for e in ev] == [("hog-rider", "unit", 17.0, 3.0)]
    assert st["threats_from_units"] == 1
    # the hog was played 8 s before: the play explains the unit
    recs2 = recs + [rec(12.0, opp=[play(11.0, "hog-rider", [3, 24])])]
    st2 = Counter()
    m.mine_game(game(), recs2, dense, st2)
    assert st2["threats_from_units"] == 0
    # a deck of fewer than 7 known cards cannot tell what a unit is
    st3 = Counter()
    m.mine_game(game(opponent_deck=["giant"]), recs, dense, st3)
    assert st3["threats_from_units"] == 0
    # a unit on the opponent's half is no crossing
    st4 = Counter()
    m.mine_game(game(), recs, [snap(t, [("hog-rider", (3, 24), 20.0)]) for t in range(10, 30)], st4)
    assert st4["threats_from_units"] == 0


# --------------------------------------------------------------------------- aggregates
def test_aggregate_answers_train_only():
    evs = [event(offset=(5.0, -3.0), held=True, trade=2.0, dt=1.0, eta=4.0, dist=8.0),
           event(offset=(3.0, -5.0), held=False, trade=0.0, dt=2.0, eta=6.0, dist=10.0),
           event(offset=(7.0, -1.0), held=True, trade=1.0, dt=3.0, eta=None, dist=12.0),
           event(offset=(9.0, 9.0), offset_ok=False, held=None, trade=None, dt=4.0),     # counts, but no offset
           event(split="val"), event(split="test"),                                        # never in the aggregate
           event(answer="tesla", kind="building"), event(answer="tesla"),                  # support 2 < 3: dropped
           ]
    agg = m.aggregate_answers(evs)
    assert [(a["threat"], a["answer"]) for a in agg] == [("giant", "cannon")]
    a = agg[0]
    assert a["support"] == 4 and a["held"] == 2 and a["held_n"] == 3 and a["held_rate"] == pytest.approx(0.667, abs=1e-3)
    assert a["mean_trade"] == pytest.approx(1.0) and a["mean_dt_s"] == pytest.approx(2.5)
    assert a["offset"] == {"dx_med": 5.0, "dy_med": -3.0, "dx_iqr": 2.0, "dy_iqr": 2.0, "n": 3}
    assert a["eta_at_answer_med"] == 4.0 and a["dist_at_answer_med"] == pytest.approx(9.5)       # nulls are skipped
    assert len(m.aggregate_answers(evs, min_support=2)) == 2
    # unknown held everywhere -> null, not 0
    nulls = m.aggregate_answers([event(held=None, trade=None) for _ in range(3)])
    assert nulls[0]["held_rate"] is None and nulls[0]["mean_trade"] is None and nulls[0]["held"] == 0


def test_aggregate_situations_counts_each_answer_once():
    k = "giant+witch|same|3-6|7-10"
    evs = [event(answer="cannon", key=k, held=True, answer_t=12.0),
           event(threat="witch", answer="cannon", key=k, held=True, answer_t=12.0),     # the same cannon answers both
           event(answer="cannon", key=k, held=False, answer_t=40.0),
           event(answer="tesla", key=k, held=True, answer_t=60.0),
           event(answer="cannon", key="giant|same|0-3|0-3", answer_t=70.0),               # n=1: dropped
           event(answer="cannon", key=k, split="val", answer_t=80.0)]
    sit = m.aggregate_situations(evs)
    assert [s["key"] for s in sit] == [k]
    s = sit[0]
    assert s["n"] == 3 and s["threat_cards"] == ["giant", "witch"]
    assert (s["lane_rel"], s["eta_band"], s["elixir_band"]) == ("same", "3-6", "7-10")
    assert s["answers"]["cannon"] == {"n": 2, "held_rate": 0.5}
    assert s["answers"]["tesla"] == {"n": 1, "held_rate": 1.0}
    assert list(s["answers"]) == ["cannon", "tesla"]


# --------------------------------------------------------------------------- situation keys
def test_situation_key_top_two_by_cost_sorted():
    # giant 5, witch 5, musketeer 4, zap 2: the two most expensive, alphabetically
    assert m.situation_key(["zap", "musketeer", "witch", "giant"], "same", 4.0, 5.0) == "giant+witch|same|3-6|4-6"
    assert m.situation_key(["witch", "giant", "witch"], "split", 8.0, 9.0) == "giant+witch|split|6-10|7-10"
    assert m.situation_key(["knight", "giant"], "same", 4.0, 5.0) == "giant+knight|same|3-6|4-6"      # giant 5, knight 3
    assert m.situation_key(["giant"], "centre", None, None) == "giant|centre|>10|?"
    assert m.situation_key(("hog-rider",), "same", 1.0, 0.0) == "hog-rider|same|0-3|0-3"
    assert m.top_threat_cards(["zap", "knight", "giant", "witch"], k=3) == ["giant", "knight", "witch"]


def test_bands_and_lane_rel():
    assert [m.eta_band(x) for x in (0, 2.99, 3, 5.99, 6, 10, 10.1, None)] == ["0-3", "0-3", "3-6", "3-6", "6-10", "6-10", ">10", ">10"]
    assert [m.elixir_band(x) for x in (0, 3.9, 4, 6.9, 7, 10, None)] == ["0-3", "0-3", "4-6", "4-6", "7-10", "7-10", "?"]
    assert m.lane_rel_of([(3, 10)]) == "same"
    assert m.lane_rel_of([(3, 10), (4, 12)]) == "same"
    assert m.lane_rel_of([(3, 10), (14, 12)]) == "split"
    assert m.lane_rel_of([(8, 10), (9, 12)]) == "centre"
    assert m.lane_rel_of([(8, 10), (3, 12)]) == "same"
    assert m.lane_rel_of([]) == "same"
    assert set(m.LANE_RELS) == {"same", "split", "centre"} and len(m.ETA_BANDS) == 4 and len(m.ELIXIR_BANDS) == 3


def test_situation_key_roundtrip_and_event_situation():
    key = m.situation_key(["giant", "witch"], "split", 2.0, 8.0)
    assert m.parse_situation_key(key) == {"key": key, "threat_cards": ["giant", "witch"], "lane_rel": "split",
                                          "eta_band": "0-3", "elixir_band": "7-10"}
    # the board before the answer contributes its threats: a witch on our half joins the giant
    recs = [rec(10.5, opp=[play(10.0, "giant", [14, 20])]), rec(11.0, action=act("cannon", [9, 9]))]
    dense = [snap(12.0, [("giant", (14, 14), 4.0), ("witch", (3, 12), 7.0), ("miner", (9, 22), None)], elixir=9.0)]
    ev, _ = m.mine_game(game(), recs, dense, Counter(), use_unit_threats=False)
    assert ev[0]["situation"]["key"] == "giant+witch|split|3-6|7-10"      # miner is on their half and 'no eta'


# --------------------------------------------------------------------------- sequences
def test_sequence_pairs_and_aggregate():
    plays = [{"t": 0.0, "card": "giant"}, {"t": 2.0, "card": "witch"}, {"t": 9.0, "card": "zap"},
             {"t": 10.0, "card": "zap"}, {"t": 12.0, "card": "giant"}]
    assert m.sequence_pairs(plays) == [("giant", "witch", 2.0), ("zap", "giant", 2.0)]     # >5 s and same-card pairs skipped
    assert m.sequence_pairs(plays, window=10.0)[1] == ("witch", "zap", 7.0)
    pairs = [("own", "giant", "witch", lag) for lag in (1.0, 2.0, 4.0)] + [("own", "zap", "giant", 1.0)] * 2 \
        + [("opp", "giant", "witch", 3.0)] * 3
    agg = m.aggregate_sequences(pairs)
    assert agg == [{"who": "opp", "first": "giant", "then": "witch", "support": 3, "lag_med_s": 3.0},
                   {"who": "own", "first": "giant", "then": "witch", "support": 3, "lag_med_s": 2.0}]


# --------------------------------------------------------------------------- the dataset on disk
def _write_dataset(root: Path) -> Path:
    ds = root / "v1"
    (ds / "act_dense").mkdir(parents=True)
    games, states, dense = [], [], []
    for i, split in enumerate(["train", "train", "train", "val"]):
        g = game(game_id=f"G{i}", split=split)
        games.append(g)
        for k in range(3):                                   # three giant -> cannon exchanges per game
            t0 = 100.0 * k
            for r in (rec(t0 + 10.5, opp=[play(t0 + 10.0, "giant", [14, 20])]),
                      rec(t0 + 11.0, action=act("cannon", [9, 9]), opp=[play(t0 + 10.0, "giant", [14, 20])])):
                states.append(dict(r, game_id=g["game_id"], split=split))
        for t in range(0, 300):
            dense.append((split, {"id": f"G{i}#{t}.0", "t": float(t), "state": {
                "own": {"elixir": 6.0, "towers": {"king": None, "left": 4858, "right": 4858}},
                "opponent": {"elixir_estimate": 5.0}, "units": []}}))
    (ds / "games.jsonl").write_text("\n".join(json.dumps(g) for g in games) + "\n")
    (ds / "states.jsonl").write_text("\n".join(json.dumps(s) for s in states) + "\n")
    for split in ("train", "val", "test"):
        (ds / "act_dense" / f"states_{split}.jsonl").write_text(
            "".join(json.dumps(d) + "\n" for sp, d in dense if sp == split))
    return ds


def test_main_writes_the_four_files(tmp_path, capsys):
    ds = _write_dataset(tmp_path)
    assert m.main(["--dataset", str(ds), "--no-unit-threats"]) == 0
    out = ds / "kg2"
    rows = {n: [json.loads(l) for l in (out / f"{n}.jsonl").read_text().splitlines()]
            for n in ("answer_events", "answers", "situations", "sequences")}
    assert Counter(e["split"] for e in rows["answer_events"]) == {"train": 9, "val": 3}      # val events are written too
    assert [(a["threat"], a["answer"], a["support"]) for a in rows["answers"]] == [("giant", "cannon", 9)]   # train only
    assert rows["answers"][0]["offset"]["dx_med"] == 5.0
    assert rows["situations"] and rows["situations"][0]["n"] == 9 and rows["situations"][0]["answers"]["cannon"]["n"] == 9
    assert rows["sequences"] == []
    text = capsys.readouterr().out
    assert "answer events per split" in text and "giant (n=9)" in text and "building answers" in text


@pytest.mark.skipif(not (DATASET_DIR / "states.jsonl").exists() or not (DATASET_DIR / "act_dense").exists(),
                    reason="dataset/v1 not built")
def test_real_dataset_smoke():
    events, pairs, stats = m.mine_dataset(DATASET_DIR)
    assert Counter(e["split"] for e in events).keys() == {"train", "val", "test"}
    assert stats["games"] > 0 and stats["events"] == len(events)
    for e in events[:200]:
        assert e["threat"] and e["answer"] and 0.0 <= e["dt_s"] <= m.WINDOW_S
        assert m.parse_situation_key(e["situation"]["key"])["key"] == e["situation"]["key"]
    assert pairs and all(0.0 <= lag <= m.SEQ_WINDOW_S for *_, lag in pairs)
    bld = [e for e in events if e["split"] == "train" and e["answer_kind"] == "building" and e["offset_ok"]
           and e["threat"] in m.BUILDING_TARGETERS]
    if len(bld) >= 10:                                           # buildings go toward the centre, in front of the threat
        assert sum(1 for e in bld if e["offset"][0] > 0) / len(bld) > 0.5
        assert sum(1 for e in bld if e["offset"][1] < 0) / len(bld) > 0.5
