"""MAC-Z3 wiring: candidates with a tactics report, d1 option notes, the tactics ranker, the policy exemption, the live
loop and the two offline tools (eval_tactics, build_d1_tac) - fakes only, no models, no screen."""
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot import d1_format  # noqa: E402
from clashbot.actions import placement_tile  # noqa: E402
from clashbot.candidates import FILLER_NOTE, PROS_ALSO, Candidate, CandidateGenerator  # noqa: E402
from clashbot.config import HOLD_KEY  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live import _build_ranker, build_parser, main, rank_keys, top_source  # noqa: E402
from clashbot.policy import ActDecision, ReactivePolicy  # noqa: E402
from clashbot.ranker import D1Ranker, RankResult, TacticsRanker  # noqa: E402
from clashbot.state_text import render_context  # noqa: E402
from clashbot.tactics import TacticalEngine, TacticalMove, TacticalReport  # noqa: E402
from tests.test_live_loop import (PRIOR, FakeBuilder, FakeExecutor, FakeGenerator, FakeKeyRanker,  # noqa: E402
                                  FakePerception, FakeSource, _argv, _loop, _state)
from tools.data import build_d1_tac  # noqa: E402
from tools.eval import eval_tactics  # noqa: E402

CARDS = load_cards()


# -- fakes --------------------------------------------------------------------------------------------------------------
class FakeIndex:
    """StateIndex stand-in: search() returns fixed (action_key, score) hits and records the filters it got."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, state, t, context_text, k=10, depth=50, **filters):
        self.calls.append(filters)
        return [types.SimpleNamespace(action_key=a, score=s) for a, s in self.hits]


class FakeKG:
    def __init__(self, table=None):
        self.table = table or {}

    def counter_scores(self, threat):
        return dict(self.table.get(threat, {}))


def _move(card, lane, zone, tile, score, intent="defend", reason="meets knight in front of the tower, trade +1"):
    return TacticalMove(card, f"{card}|{lane}|{zone}", list(tile), intent, reason, score, 1.0)


def _report(moves, lines=("tactics: best move",)):
    return TacticalReport([], list(moves), moves[0] if moves else None, None if moves else "hold", list(lines))


def _cand_state():
    """Hand zap / archers / knight / giant at 5 elixir; an enemy knight 3 s from the left tower (a KG threat)."""
    st = _state(elixir=5.0)
    st["units"] = [{"unit_class": "knight", "card": "knight", "side": "enemy", "tile": [4, 12],
                    "eta_tower": {"tower": "own_left", "s": 3.0}}]
    return st


def _generator():
    hits = [("knight|left|mid", 0.03), ("zap|left|mid", 0.02), ("archers|left|back", 0.01), (HOLD_KEY, 0.5),
            ("fireball|left|mid", 0.5)]                       # hold and an illegal key never become candidates
    return CandidateGenerator(FakeIndex(hits), FakeKG({"knight": {"archers": 1.0, "giant": 0.5}}), CARDS)


def _moves():
    return [_move("knight", "left", "mid", [4.5, 9.5], 0.9),
            _move("fireball", "left", "mid", [4.0, 9.0], 0.85),               # not in the hand: skipped
            _move("giant", "right", "mid", [14.5, 9.0], 0.8, reason="tanks for the tower vs giant, trade +0"),
            _move("archers", "left", "back", [4.0, 2.5], 0.7)]


# 1 -----------------------------------------------------------------------------------------------------------------
def test_generate_with_a_report_puts_legal_tactical_moves_first_with_tile_and_note():
    gen = _generator()
    cands = gen.generate(_cand_state(), 10.0, "ctx", n=4, report=_report(_moves()))
    assert [c.key for c in cands] == ["knight|left|mid", "giant|right|mid", "zap|left|mid", "archers|left|mid"]
    knight, giant, zap, archers = cands
    assert knight.tile == [4.5, 9.5] and knight.score == 0.9
    assert knight.note == "tactics: defend, meets knight in front of the tower, trade +1; pros played this too"
    assert knight.sources == {"tac": 0.9, "knn": 1.0, "kg": 0.0}          # the lookup sources are merged in
    assert giant.tile == [14.5, 9.0] and giant.sources == {"tac": 0.8}
    assert giant.note == "tactics: defend, tanks for the tower vs giant, trade +0"
    assert zap.tile is None and zap.sources["knn"] == pytest.approx(0.6667)
    assert zap.note == "pros played this in similar states (support 0.67)"
    assert archers.tile is None and archers.note == "known counter of knight"
    assert not any(c.key.startswith("fireball") for c in cands)


def test_tac_n_limits_the_tactical_candidates_and_never_exceeds_n():
    gen = _generator()
    one = gen.generate(_cand_state(), 10.0, "ctx", n=4, report=_report(_moves()), tac_n=1)
    assert [c.key for c in one][:2] == ["knight|left|mid", "zap|left|mid"] and sum("tac" in c.sources for c in one) == 1
    none = gen.generate(_cand_state(), 10.0, "ctx", n=4, report=_report(_moves()), tac_n=0)
    assert not any("tac" in c.sources for c in none) and all(c.note for c in none)
    small = gen.generate(_cand_state(), 10.0, "ctx", n=1, report=_report(_moves()))
    assert [c.key for c in small] == ["knight|left|mid"]


def test_report_without_moves_gives_lookup_candidates_with_notes():
    cands = _generator().generate(_cand_state(), 10.0, "ctx", n=3, report=_report([]))
    assert [c.key for c in cands] == ["knight|left|mid", "zap|left|mid", "archers|left|mid"]
    assert all(c.tile is None and c.note for c in cands)


def test_filters_reach_the_index_and_report_does_not():
    gen = _generator()
    gen.generate(_cand_state(), 10.0, "ctx", report=_report(_moves()), tac_n=2, splits=("train",), exclude_game="g1")
    assert gen.index.calls == [{"splits": ("train",), "exclude_game": "g1"}]


# 2 -----------------------------------------------------------------------------------------------------------------
def test_without_a_report_nothing_changes():
    gen = _generator()
    base = gen.generate(_cand_state(), 10.0, "ctx", n=4)
    assert base == gen.generate(_cand_state(), 10.0, "ctx", n=4, report=None)
    assert [c.key for c in base] == ["knight|left|mid", "zap|left|mid", "archers|left|mid", "archers|left|back"]
    assert [c.sources for c in base][0] == {"knn": 1.0, "kg": 0.0}
    assert all(c.tile is None and c.note is None for c in base)
    padded = gen.generate(_cand_state(), 10.0, "ctx", n=7)           # lookup keys run out: legal keys fill in
    assert padded[5].sources == {"fallback": 1} and all(c.tile is None and c.note is None for c in padded)
    assert Candidate("a|left|mid", 1.0, {"knn": 1.0}) == Candidate("a|left|mid", 1.0, {"knn": 1.0}, None, None)


# 3 -----------------------------------------------------------------------------------------------------------------
def test_option_notes_match_the_notes_generate_attaches():
    gen = _generator()
    report = _report(_moves())
    cands = gen.generate(_cand_state(), 10.0, "ctx", n=8, report=report)
    assert any(c.note == FILLER_NOTE for c in cands) and any(c.note.startswith("tactics:") for c in cands)
    keys = [c.key for c in cands]
    live = {c.key: c.note for c in cands}
    assert gen.option_notes(_cand_state(), 10.0, "ctx", keys, report, tac_n=2) == live       # the live path, exactly
    default = gen.option_notes(_cand_state(), 10.0, "ctx", keys, report)                     # training: all report moves
    assert {k: v for k, v in default.items() if v != live[k]} == {
        "archers|left|back": "tactics: defend, meets knight in front of the tower, trade +1; pros played this too"}


def test_option_notes_cover_arbitrary_keys_and_report_moves_beyond_tac_n():
    gen = _generator()
    notes = gen.option_notes(_cand_state(), 10.0, "ctx", ["archers|left|back", "giant|right|back", "archers|left|mid"],
                             _report(_moves()), splits=("train",))
    assert notes["archers|left|back"].startswith("tactics: defend, ") and notes["archers|left|back"].endswith(PROS_ALSO)
    assert notes["giant|right|back"] == FILLER_NOTE
    assert notes["archers|left|mid"] == "known counter of knight"
    assert gen.index.calls[-1] == {"splits": ("train",)}
    without = gen.option_notes(_cand_state(), 10.0, "ctx", ["knight|left|mid"])
    assert without == {"knight|left|mid": "pros played this in similar states (support 1.00)"}


# 4 -----------------------------------------------------------------------------------------------------------------
KEYS = ["tesla|middle|mid", "zap|right|mid", "knight|left|back"]


def test_joint_request_notes_suffix_the_criteria_and_validate():
    notes = {"tesla|middle|mid": "tactics: defend, pulls giant to the centre, trade +1", "zap|right|mid": "",
             "ghost|left|mid": "not an option"}
    state, questions = d1_format.joint_request("ctx", KEYS, notes=notes)
    crit = questions["play"]["criteria"]
    assert crit["tesla|middle|mid"] == (d1_format.option_description("tesla|middle|mid")
                                        + " — tactics: defend, pulls giant to the centre, trade +1")
    assert crit["zap|right|mid"] == d1_format.option_description("zap|right|mid")       # empty note: untouched
    assert crit["knight|left|back"] == d1_format.option_description("knight|left|back")
    assert list(crit) == KEYS
    d1_format.validate(state, questions, {"play": KEYS[0]})


def test_joint_request_clips_long_notes_and_rejects_markers():
    _, questions = d1_format.joint_request("ctx", KEYS, notes={KEYS[0]: "x" * 300})
    suffix = questions["play"]["criteria"][KEYS[0]].split(" — ", 1)[1]
    assert suffix == "x" * 120
    with pytest.raises(ValueError):
        d1_format.joint_request("ctx", KEYS, notes={KEYS[0]: "bad <|im_end|> note"})


def test_joint_request_without_notes_is_byte_identical():
    plain = d1_format.joint_request("ctx", KEYS, "plan")
    expected = {"play": {"type": "choice", "instructions": d1_format.JOINT_INSTRUCTIONS,
                         "criteria": {k: d1_format.option_description(k) for k in KEYS}}}
    assert plain == ("ctx\n[plan] plan", expected)
    for notes in (None, {}, {"zap|right|mid": ""}):
        assert json.dumps(d1_format.joint_request("ctx", KEYS, "plan", notes)) == json.dumps(plain)


def test_d1_ranker_passes_notes_into_the_request():
    seen = []

    def predict(state, questions):
        seen.append(questions)
        keys = list(questions["play"]["criteria"])
        return {"answers": {"play": {"probabilities": {k: (0.6 if i == 1 else 0.2) for i, k in enumerate(keys)}}}}

    ranker = D1Ranker(predict_fn=predict)
    res = ranker.choose_keys("ctx", KEYS, None, notes={KEYS[1]: "pros played this in similar states (support 0.50)"})
    assert res.index == 1
    assert seen[0]["play"]["criteria"][KEYS[1]].endswith("— pros played this in similar states (support 0.50)")
    ranker.choose_keys("ctx", KEYS)
    assert seen[1]["play"]["criteria"] == {k: d1_format.option_description(k) for k in KEYS}
    assert ranker.build_request("ctx", KEYS, None, {KEYS[0]: "n"})[1]["play"]["criteria"][KEYS[0]].endswith("— n")


def test_rank_keys_forwards_notes_only_to_rankers_that_accept_them():
    class WithNotes:
        def __init__(self):
            self.calls = []

        def choose_keys(self, context, keys, plan=None, notes=None):
            self.calls.append(notes)
            return RankResult(0, [1.0] + [0.0] * (len(keys) - 1), "n")

    class WithKwargs(WithNotes):
        def choose_keys(self, context, keys, plan=None, **kw):
            self.calls.append(kw.get("notes"))
            return RankResult(0, [1.0] + [0.0] * (len(keys) - 1), "k")

    notes = {KEYS[0]: "n"}
    for cls in (WithNotes, WithKwargs):
        r = cls()
        rank_keys(r, "ctx", KEYS, None, notes=notes)
        rank_keys(r, "ctx", KEYS, None)
        assert r.calls == [notes, None]
    plain = FakeKeyRanker()                                          # no notes parameter: called as before
    assert rank_keys(plain, "ctx", KEYS, "p", notes=notes) == (1, [0.2, 0.8, 0.0])
    assert plain.calls == [("ctx", KEYS, "p")]
    assert rank_keys(plain, "ctx", KEYS, "p")[0] == 1


def test_tactics_ranker_picks_the_highest_tac_source():
    r = TacticsRanker()
    assert r.name == "tactics"
    cands = [Candidate("a|left|mid", 0.9, {"knn": 1.0}), Candidate("b|left|mid", 0.4, {"tac": 0.4}),
             Candidate("c|left|mid", 0.7, {"tac": 0.7}), Candidate("d|left|mid", 0.7, {"tac": 0.7})]
    res = r.choose_candidates(cands)
    assert (res.index, res.probs, res.model) == (2, [0.0, 0.0, 1.0, 0.0], "tactics")           # ties -> first
    no_tac = r.choose_candidates([Candidate("a|left|mid", 0.9, {"knn": 1.0}), Candidate("b|left|mid", 0.4, {"kg": 1.0})])
    assert (no_tac.index, no_tac.probs) == (0, [1.0, 0.0])
    assert r.choose_candidates([Candidate("a|left|mid", 0.0, {"tac": 0.0}), Candidate("b|left|mid", 0.0, {})]).index == 0
    assert r.choose_candidates([{"key": "a", "sources": {}}, {"key": "b", "sources": {"tac": 0.1}}]).index == 1


# 5 -----------------------------------------------------------------------------------------------------------------
def _threat_state(units=None, elixir=5.0):
    st = _state(elixir=elixir)
    st["units"] = units if units is not None else [{"unit_class": "giant", "card": "giant", "side": "enemy",
                                                    "tile": [15, 9], "eta_tower": {"tower": "own_right", "s": 2.0}}]
    return st


def test_policy_keeps_a_tactical_candidate_outside_the_defended_lane_and_zone():
    pol = ReactivePolicy(CARDS)
    state = _threat_state()
    decision = pol.decide(state, 10.0)
    assert decision.mode == "defend" and decision.lane == "right"
    tesla = Candidate("tesla|middle|mid", 0.9, {"tac": 0.9}, [10.5, 9.0])
    knight = Candidate("knight|right|mid", 0.5, {"knn": 1.0})
    assert pol.filter_candidates([tesla, knight], decision, state) == [tesla, knight]       # order unchanged
    plain_tesla = replace(tesla, sources={"knn": 1.0})
    assert pol.filter_candidates([plain_tesla, knight], decision, state) == [knight]         # no "tac": old rule


def test_policy_exemption_covers_leak_and_punish_but_not_the_spell_rule():
    pol = ReactivePolicy(CARDS)
    state = _threat_state(units=[])
    mid = Candidate("knight|middle|mid", 0.6, {"tac": 0.6}, [9.0, 8.0])
    back = Candidate("giant|left|back", 0.5, {"knn": 1.0})
    leak = ActDecision(True, "leak", "left", "full bar")
    assert pol.filter_candidates([mid, back], leak, state) == [mid, back]                    # mid is no leak zone
    assert pol.filter_candidates([replace(mid, sources={"knn": 1.0}), back], leak, state) == [back]
    punish = ActDecision(True, "punish", "left", "overspent")
    assert pol.filter_candidates([mid, back], punish, state) == [mid]                        # no lane / zone rule
    barrel = Candidate("goblin-barrel|left|enemy_half", 0.55, {"tac": 0.55}, [3.5, 23.5])
    assert pol.filter_candidates([barrel], punish, state) == [barrel]                        # aimed at a tower
    assert pol.filter_candidates([replace(barrel, sources={"knn": 1.0})], punish, state) == []


def test_policy_still_drops_a_tactical_spell_without_a_worthwhile_target():
    pol = ReactivePolicy(CARDS)
    state = _threat_state()                                  # only a giant: a tank is not a spell target
    decision = pol.decide(state, 10.0)
    fireball = Candidate("fireball|right|mid", 0.9, {"tac": 0.9}, [14.5, 9.0])
    tesla = Candidate("tesla|middle|mid", 0.8, {"tac": 0.8}, [10.5, 9.0])
    assert pol.filter_candidates([fireball, tesla], decision, state) == [tesla]
    assert pol.no_blind_spells([fireball, tesla], state) == [tesla]
    # ... and a spell with a target in the threat's reach is kept
    swarm = [{"unit_class": "skeleton", "card": "skeletons", "side": "enemy", "tile": [15, 9 + i * 0.3],
              "eta_tower": {"tower": "own_right", "s": 2.0}} for i in range(3)]
    state2 = _threat_state(units=swarm, elixir=5.0)
    zap = Candidate("zap|right|mid", 0.9, {"tac": 0.9}, [15.0, 9.3])
    assert pol.filter_candidates([zap], pol.decide(state2, 10.0), state2) == [zap]


# 6 -----------------------------------------------------------------------------------------------------------------
LINES = ["tactics: push right: giant (5 elixir), tower in 6.0s; needs tank",
         "tactics: best knight @[9.5,8] defend: tanks for the tower vs giant, trade +2"]
TAC_NOTE = "tactics: defend, tanks for the tower vs giant, trade +2"


class FakeEngine:
    def __init__(self, lines=LINES):
        self.lines = list(lines)
        self.calls = []

    def analyse(self, state, t=None):
        self.calls.append(t)
        return TacticalReport([], [], None, "fake", list(self.lines))


class TacGenerator:
    """Hands out fixed candidates and records (ctx, report, filters)."""

    def __init__(self, cands=None):
        self.cands = cands or [Candidate("knight|middle|mid", 0.9, {"tac": 0.9}, [9.5, 8.0], TAC_NOTE),
                               Candidate("zap|left|mid", 0.5, {"knn": 0.5}, None, "pros played this in similar states"),
                               Candidate("giant|left|back", 0.0, {"fallback": 1}, None, None)]
        self.calls = []

    def generate(self, state, t, ctx, n=4, report=None, tac_n=2, **filters):
        self.calls.append((ctx, report, filters))
        return [replace(c) for c in self.cands]


class NotesRanker:
    name = "notes"

    def __init__(self, index=0):
        self.index = index
        self.calls = []

    def choose_keys(self, context, keys, plan=None, notes=None):
        self.calls.append((context, list(keys), plan, notes))
        return RankResult(self.index, [1.0 if i == self.index else 0.0 for i in range(len(keys))], "fake")


class RecPolicy:
    """Always defends; records the context it was given; target_tile is configurable."""

    def __init__(self, tile=None):
        self.tile = tile
        self.ctxs = []

    def decide(self, state, t, context_text=None):
        self.ctxs.append(context_text)
        return ActDecision(True, "defend", None, "fake")

    def filter_candidates(self, candidates, decision, state):
        return candidates

    def target_tile(self, key, state):
        return self.tile


def _tac_loop(tmp_path=None, ranker=None, policy=None, generator=None, engine="default", **kw):
    return _loop(tmp_path=tmp_path, builder=FakeBuilder(warmup=0), generator=generator or TacGenerator(),
                 ranker=ranker or NotesRanker(), policy=policy or RecPolicy(), executor=FakeExecutor(),
                 tactics=FakeEngine() if engine == "default" else engine, **kw)


def test_ranker_context_has_the_lines_and_the_policy_context_does_not():
    pol, rk, gen, eng = RecPolicy(), NotesRanker(), TacGenerator(), FakeEngine()
    lp = _tac_loop(ranker=rk, policy=pol, generator=gen, engine=eng)
    d = lp.step("f", 1.0, 0, 2.0)
    assert d is not None and eng.calls == [1.0]
    assert pol.ctxs == ["CTX"]                                       # the act policy keeps the plain context
    assert gen.calls[0][0] == "CTX" and gen.calls[0][1].lines == LINES and gen.calls[0][2] == {}
    assert rk.calls[0][0] == "CTX\n" + "\n".join(LINES) and rk.calls[0][2] == "PLAN"
    assert d.tactics_lines == LINES


def test_notes_reach_the_ranker_without_the_empty_ones():
    rk = NotesRanker()
    _tac_loop(ranker=rk).step("f", 1.0, 0, 2.0)
    assert rk.calls[0][3] == {"knight|middle|mid": TAC_NOTE, "zap|left|mid": "pros played this in similar states"}


@pytest.mark.parametrize("aim,pick,expected", [
    ([1.0, 2.0], 0, [1.0, 2.0]),                                     # 1. the policy's spell aim beats everything
    (None, 0, [9.5, 8.0]),                                           # 2. else the chosen candidate's own tile
    (None, 2, [3.0, 2.0]),                                           # 3. else the placement prior
])
def test_tile_precedence(aim, pick, expected):
    ex = FakeExecutor()
    cands = [Candidate("knight|middle|mid", 0.9, {"tac": 0.9}, [9.5, 8.0], TAC_NOTE),
             Candidate("zap|left|mid", 0.5, {"knn": 0.5}),
             Candidate("giant|left|back", 0.3, {"kg": 0.3})]          # supported, so no fallback hold at 5 elixir
    lp = _loop(builder=FakeBuilder(warmup=0), generator=TacGenerator(cands), ranker=NotesRanker(pick),
               policy=RecPolicy(aim), executor=ex, tactics=FakeEngine())
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.tile == expected and ex.calls[0][1] == expected
    if pick == 2:
        assert expected == placement_tile("giant|left|back", PRIOR)


def test_chosen_candidate_tile_is_copied_not_shared():
    gen = TacGenerator()
    lp = _tac_loop(generator=gen, policy=RecPolicy(None))
    d = lp.step("f", 1.0, 0, 2.0)
    d.tile.append(99.0)
    assert gen.cands[0].tile == [9.5, 8.0]


def test_tactics_ranker_picks_the_highest_tac_candidate_in_the_loop(tmp_path):
    cands = [Candidate("zap|left|mid", 0.5, {"knn": 1.0}), Candidate("archers|left|mid", 0.3, {"tac": 0.3}, [4.0, 8.0]),
             Candidate("knight|middle|mid", 0.9, {"tac": 0.9}, [9.5, 8.0])]
    lp = _tac_loop(tmp_path, ranker=TacticsRanker(), generator=TacGenerator(cands))
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.key == "knight|middle|mid" and d.probs == [0.0, 0.0, 1.0] and d.tile == [9.5, 8.0]
    assert lp.summary()["decisions_by_source"] == {"tac": 1}


def test_no_engine_is_identical_to_today(tmp_path):
    gen, rk = FakeGenerator(), FakeKeyRanker()
    lp = _loop(tmp_path, builder=FakeBuilder(warmup=0), generator=gen, ranker=rk)
    assert lp.tactics is None
    d = lp.step("f", 1.0, 7, 2.0)
    assert gen.calls == [(1.0, 4, {})]                               # generate() got no report
    assert rk.calls == [("CTX", ["zap|left|mid", "giant|left|back"], "PLAN")]       # plain context, no notes
    assert d.tile == [3.0, 2.0] and d.tactics_lines == []
    assert d.candidates[0] == {"key": "zap|left|mid", "score": 1.0, "sources": {"knn": 1.0}, "tile": None,
                               "note": None}


def test_log_fields_and_decisions_by_source(tmp_path):
    lp = _tac_loop(tmp_path, ranker=NotesRanker(1))                  # picks the knn candidate
    lp.step("f", 1.0, 0, 2.0)
    line = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert line["tactics_lines"] == LINES
    assert line["candidates"][0] == {"key": "knight|middle|mid", "score": 0.9, "sources": {"tac": 0.9},
                                     "tile": [9.5, 8.0], "note": TAC_NOTE}
    assert line["candidates"][1]["tile"] is None and line["candidates"][2]["note"] is None
    assert lp.summary()["decisions_by_source"] == {"knn": 1}


def test_decisions_by_source_counts_every_kind(tmp_path):
    lp = _tac_loop(tmp_path, ranker=NotesRanker(2), policy=RecPolicy())          # a fallback candidate ...
    lp.step("f", 1.0, 0, 2.0)                                                    # ... at 5 elixir: held
    assert lp.decisions[0].key == HOLD_KEY and lp.decisions[0].executor["status"] == "hold_fallback"
    lp2 = _tac_loop(tmp_path, ranker=NotesRanker(0))
    lp2.step("f", 1.0, 0, 2.0)
    assert lp.summary()["decisions_by_source"] == {"hold": 1} and lp2.summary()["decisions_by_source"] == {"tac": 1}
    assert top_source({"knn": 0.2, "kg": 0.5}) == "kg" and top_source({"knn": 0.0, "kg": 0.0}) == "fallback"
    assert top_source({"tac": 0.1, "knn": 1.0}) == "tac" and top_source({"knn": 0.3, "kg": 0.3}) == "knn"


def test_cli_ranker_and_tactics_flags(monkeypatch, tmp_path, capsys):
    import clashbot.live as live_mod

    ap = build_parser()
    assert ap.parse_args(_argv()).ranker == "score" and ap.parse_args(_argv()).no_tactics is False
    assert ap.parse_args(_argv("--ranker", "tactics", "--no-tactics")).no_tactics is True
    with pytest.raises(SystemExit):
        ap.parse_args(_argv("--ranker", "bogus"))
    assert isinstance(_build_ranker(ap.parse_args(_argv("--ranker", "tactics"))), TacticsRanker)
    assert main(_argv("--ranker", "tactics", "--no-tactics")) == 2
    assert "--no-tactics" in capsys.readouterr().err

    seen = {}

    class StubLoop:
        def __init__(self, *a, **k):
            seen.clear()
            seen.update(k)
            self.summary = _loop().summary()

        def run(self, **k):
            return self.summary

    monkeypatch.setattr(live_mod, "_build_source", lambda args: FakeSource())
    monkeypatch.setattr(live_mod, "_build_perception", lambda args, source: FakePerception())
    monkeypatch.setattr(live_mod, "_build_plan_text", lambda args, deck: None)
    monkeypatch.setattr(live_mod, "load_cards", lambda: CARDS)
    monkeypatch.setattr(live_mod, "load_placement_prior", lambda: PRIOR)
    monkeypatch.setattr(live_mod, "StateIndex", lambda: None)
    kgs = []
    monkeypatch.setattr(live_mod, "KnowledgeGraph", lambda: kgs.append(FakeKG()) or kgs[-1])
    monkeypatch.setattr(live_mod, "CandidateGenerator", lambda index, kg, cards: FakeGenerator())
    monkeypatch.setattr(live_mod, "LiveLoop", StubLoop)
    base = ("--source", "video", "--video", "x.mp4", "--device", "cpu")
    assert main(_argv(*base, "--out", str(tmp_path / "a"))) == 0
    assert isinstance(seen["tactics"], TacticalEngine) and seen["tactics"].kg is kgs[-1]    # the generator's KG
    assert json.loads((tmp_path / "a" / "config.json").read_text())["no_tactics"] is False
    assert main(_argv(*base, "--no-tactics", "--out", str(tmp_path / "b"))) == 0
    assert seen["tactics"] is None
    assert json.loads((tmp_path / "b" / "config.json").read_text())["no_tactics"] is True
    assert main(_argv(*base, "--ranker", "tactics", "--out", str(tmp_path / "c"))) == 0
    assert isinstance(seen["tactics"], TacticalEngine)


# 7 -----------------------------------------------------------------------------------------------------------------
DECK = ["tesla", "knight", "zap", "musketeer", "giant", "fireball", "bats", "skeletons"]
OTHER_KEYS = ["knight|left|back", "musketeer|right|back", "tesla|left|mid"]


def _full_state():
    giant = {"unit_class": "giant", "card": "giant", "side": "enemy", "tile": [15, 13],
             "eta_tower": {"tower": "own_right", "s": 8.0}, "heading": "advancing"}
    return {"clock": "1:30", "phase": "double_elixir", "match_seconds": 90,
            "own": {"elixir": 7.0, "hand": ["tesla", "knight", "zap", "musketeer"], "next_card": "giant",
                    "deck": DECK, "deck_status": "valid", "hand_confidence": None, "hero_cards": [],
                    "towers": {"king": None, "left": 4000, "right": 4000}, "recent_plays": []},
            "opponent": {"elixir_estimate": 5.0, "towers": {"king": None, "left": 4000, "right": 4000},
                         "deck_known": [], "deck_complete": False, "recent_plays": []},
            "units": [giant], "threats": ["giant(e) advancing, tower in 8.0s"]}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def synthetic(tmp_path):
    """A tiny dataset dir (one record per split) whose pro play is whatever the real engine ranks first."""
    ds = tmp_path / "ds"
    cards = load_cards()
    engine = TacticalEngine(cards)
    state = _full_state()
    best = engine.analyse(state, 10.0).best
    assert best is not None, "the synthetic giant push must produce a tactical move"
    keys = [best.key, *[k for k in OTHER_KEYS if k != best.key][:3]]
    context = render_context(state, 10.0)
    games = {"train": "g1", "val": "g2", "test": "g3"}
    states, joint = [], {}
    for split, game in games.items():
        rid = f"{game}#10.0"
        action = {"type": "play", "card": best.card, "tile": best.tile, "lane": best.key.split("|")[1],
                  "zone": best.key.split("|")[2], "key": best.key}
        states.append({"id": rid, "game_id": game, "split": split, "kind": "play", "t": 10.0, "state": state,
                       "action": action})
        _write(ds / "ranker" / f"{split}.jsonl", [{
            "id": rid, "game_id": game, "split": split, "kind": "play", "context": context,
            "options": [{"key": k, "text": k, "source": "pos" if k == best.key else "G_card"} for k in keys],
            "label": 0, "positive_key": best.key, "pos_in_legal": True}])
        req_state, questions = d1_format.joint_request(context, keys)
        joint[split] = [{"id": rid, "game_id": game, "split": split, "kind": "play", "variant": "joint",
                         "state": req_state, "questions": questions, "answers": {"play": best.key}}]
        _write(ds / "d1" / "joint" / f"{split}.jsonl", joint[split])
    _write(ds / "states.jsonl", states)
    return ds, best, joint


def test_eval_tactics_main_writes_json_and_markdown(synthetic, monkeypatch, capsys):
    ds, best, _ = synthetic
    monkeypatch.setattr(eval_tactics, "KnowledgeGraph", lambda: FakeKG())
    out = ds / "eval" / "t.json"
    assert eval_tactics.main(["--dataset", str(ds), "--splits", "val,test", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert out.with_suffix(".md").exists() and result["kg"] is True and result["splits"] == ["val", "test"]
    for split in ("val", "test"):
        allrow = result["play"][split]["all"]
        assert allrow["n"] == 1 and allrow["coverage"] == 1.0 and allrow["top1"] == 1.0 and allrow["top3"] == 1.0
        assert result["play"][split]["push"]["n"] == 1 and result["play"][split]["no_push"]["n"] == 0
        assert allrow["top1_matched"] == {"n": 1, "lane": 1.0, "zone": 1.0, "tile_dist": 0.0}
        grid = result["grid"][split]
        assert grid["n"] == 1 and grid["scored_share"] == 1.0 and grid["acc"] in (0.0, 1.0)
        assert grid["by_situation"]["push"]["n"] == 1
    assert result["reference"]["d1_finetuned"] == {"val": 0.524, "test": 0.483}
    text = capsys.readouterr().out
    assert "| val | all | 1 |" in text and "tactics ranker" in text and "placement prior" in text


def test_eval_tactics_reports_intents_and_takes_a_weights_option(synthetic, monkeypatch, tmp_path, capsys):
    from clashbot.tactics import DEFAULT_WEIGHTS
    ds, best, _ = synthetic
    monkeypatch.setattr(eval_tactics, "KnowledgeGraph", lambda: FakeKG())
    out = ds / "eval" / "w.json"
    assert eval_tactics.main(["--dataset", str(ds), "--splits", "val", "--out", str(out), "--weights", "default"]) == 0
    result = json.loads(out.read_text())
    assert result["weights"] == "default"
    row = result["intents"]["val"][best.intent]                                  # the giant push: a defend move is best
    assert row["n"] == 1 and row["share"] == 1.0 and row["precision"] == 1.0 and row["lane"] == 1.0
    assert row["top3_n"] >= 1 and row["top3_precision"] is not None
    assert "## By intent of the best move" in capsys.readouterr().out and "| val | defend | 1 |" in out.with_suffix(".md").read_text()
    # a weights file: its values reach the engine (a huge base score moves every defence score)
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"weights": {"base": DEFAULT_WEIGHTS["base"] + 5.0}}))
    seen = {}
    real = eval_tactics.TacticalEngine

    def spy(*a, **kw):
        seen["w"] = kw.get("weights")
        return real(*a, **kw)

    monkeypatch.setattr(eval_tactics, "TacticalEngine", spy)
    assert eval_tactics.main(["--dataset", str(ds), "--splits", "val", "--out", str(ds / "p.json"), "--weights", str(path)]) == 0
    assert seen["w"]["base"] == DEFAULT_WEIGHTS["base"] + 5.0 and seen["w"]["kg_factor"] == DEFAULT_WEIGHTS["kg_factor"]
    assert json.loads((ds / "p.json").read_text())["weights"] == str(path)
    assert eval_tactics.main(["--dataset", str(ds), "--splits", "val", "--out", str(ds / "q.json")]) == 0
    assert seen["w"] is None and json.loads((ds / "q.json").read_text())["weights"] == "committed"


def test_eval_tactics_without_a_knowledge_graph_still_runs(synthetic, monkeypatch):
    ds, _, _ = synthetic

    def boom():
        raise OSError("no db")

    monkeypatch.setattr(eval_tactics, "KnowledgeGraph", boom)
    assert eval_tactics.main(["--dataset", str(ds), "--splits", "val", "--out", str(ds / "o.json")]) == 0
    assert json.loads((ds / "o.json").read_text())["kg"] is False


def test_build_d1_tac_main_writes_validated_lines_with_notes_and_stats(synthetic, monkeypatch):
    ds, best, joint = synthetic
    index = FakeIndex([(best.key, 0.02)])
    monkeypatch.setattr(build_d1_tac, "StateIndex", lambda: index)
    monkeypatch.setattr(build_d1_tac, "KnowledgeGraph", lambda: FakeKG())
    out = ds / "d1" / "joint_tac"
    assert build_d1_tac.main(["--dataset", str(ds), "--out", str(out)]) == 0
    stats = json.loads((out / "stats.json").read_text())
    assert set(stats) == {"train", "val", "test"}
    assert {c["exclude_game"] for c in index.calls} == {"g1", "g2", "g3"}
    assert all(c["splits"] == ("train",) for c in index.calls)
    for split in ("train", "val", "test"):
        [new] = [json.loads(x) for x in (out / f"{split}.jsonl").read_text().splitlines()]
        [old] = joint[split]
        assert {k: new[k] for k in ("id", "game_id", "split", "kind", "variant", "answers")} == \
            {k: old[k] for k in ("id", "game_id", "split", "kind", "variant", "answers")}
        d1_format.validate(new["state"], new["questions"], new["answers"])
        assert new["state"].startswith(old["state"] + "\ntactics: push right")
        assert "\ntactics: best " in new["state"]
        crit = new["questions"]["play"]["criteria"]
        assert list(crit) == list(old["questions"]["play"]["criteria"])
        assert crit[best.key].startswith(d1_format.option_description(best.key) + " — tactics: ")
        assert crit[best.key].endswith("; pros played this too")                 # the fake index backs this key
        assert all(" — " in v for v in crit.values())                       # every option got a note
        s = stats[split]
        assert s["lines"] == 1 and s["answer_has_tactics_note"] == 1.0 and s["tactics_top_is_answer"] == 1.0
        assert s["context_mismatches"] == 0 and s["skipped_no_state"] == 0 and s["lines_with_push"] == 1.0
        assert s["any_lookup_note"] == 0.0 and s["mean_state_chars"] == len(new["state"])


def test_build_line_uses_the_lines_own_text_when_the_context_differs(synthetic):
    ds, best, joint = synthetic
    rec = json.loads((ds / "states.jsonl").read_text().splitlines()[0])
    line = json.loads(json.dumps(joint["train"][0]))
    line["state"] = "OTHER CONTEXT"
    gen = CandidateGenerator(FakeIndex([]), FakeKG(), load_cards())
    new, facts = build_d1_tac.build_line(line, rec, TacticalEngine(load_cards()), gen)
    assert facts["mismatch"] is True and new["state"].startswith("OTHER CONTEXT\ntactics: ")
    assert build_d1_tac.split_stats([facts], 2)["context_mismatches"] == 1
