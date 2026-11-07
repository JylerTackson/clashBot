"""MAC-X: learned play/wait decision - LearnedActPolicy, the act-data sampler, the offline eval and the live CLI.

Fakes only: a scorer that returns a fixed p per context string, a monkeypatched D1Ranker; no model is ever loaded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clashbot.live as live_mod  # noqa: E402
import clashbot.ranker as ranker_mod  # noqa: E402
from clashbot import d1_format  # noqa: E402
from clashbot.candidates import Candidate  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live import build_parser, main  # noqa: E402
from clashbot.policy import ActDecision, LearnedActPolicy, ReactivePolicy  # noqa: E402
from tests.test_live_loop import (PRIOR, FakeBuilder, FakeExecutor, FakeGenerator, FakePerception,  # noqa: E402
                                  FakeSource, _argv, _loop, _state)
from tools.data import sample_act_train as sat  # noqa: E402
from tools.eval import eval_act_head as eah  # noqa: E402

CARDS = load_cards()
RULES = ReactivePolicy(CARDS)


def _enemy(tile, eta=None, tower="own_left", card="knight"):
    return {"unit_class": card, "card": card, "side": "enemy", "tile": list(tile),
            "eta_tower": None if eta is None else {"tower": tower, "s": eta}}


def _with(units=None, **kw):
    return {**_state(**kw), "units": units or []}


class FakeScorer:
    """probabilities() -> {"act": [1 - p, p]} with p looked up by the state string (default `p`)."""

    def __init__(self, p: float = 0.5, by_state: dict | None = None):
        self.p = p
        self.by_state = by_state or {}
        self.calls: list[str] = []

    def probabilities(self, state, questions):
        d1_format.validate(state, questions)
        self.calls.append(state)
        p = self.by_state.get(state, self.p)
        return {"act": [1.0 - p, p]}


# 1 -----------------------------------------------------------------------------------------------------------------
def test_p_act_builds_a_valid_act_request_and_returns_p_true():
    scorer = FakeScorer(by_state={"CTX one": 0.8})
    policy = LearnedActPolicy(scorer, CARDS)
    assert policy.p_act("CTX one") == pytest.approx(0.8)
    assert scorer.calls == ["CTX one"]
    state, questions = d1_format.act_request("CTX one")
    d1_format.validate(state, questions)
    assert questions["act"]["type"] == "noul"


# 2 -----------------------------------------------------------------------------------------------------------------
def test_no_affordable_card_is_none_and_never_asks_the_head():
    scorer = FakeScorer(1.0)
    d = LearnedActPolicy(scorer, CARDS).decide(_with(elixir=0.0), 10.0, "CTX")
    assert d == ActDecision(False, "none", None, "no affordable card") and scorer.calls == []


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_defend_and_leak_pass_through_whatever_p_is(p):
    scorer = FakeScorer(p)
    policy = LearnedActPolicy(scorer, CARDS, threshold=0.5)
    threat = _with(units=[_enemy((5, 9), eta=2.0)])
    d = policy.decide(threat, 10.0, "CTX")
    assert d.act and d.mode == "defend" and d.lane == "left"
    d = policy.decide(_with(elixir=10.0), 20.0, "CTX")
    assert d.act and d.mode == "leak"
    assert scorer.calls == []              # the safety modes do not need the head


def test_p_above_threshold_acts_in_mode_learned_without_a_lane():
    scorer = FakeScorer(0.7)
    d = LearnedActPolicy(scorer, CARDS, threshold=0.5).decide(_with(), 10.0, "CTX")
    assert d.act and d.mode == "learned" and d.lane is None and d.reason == "learned act p=0.70"
    assert d.threat is None


def test_p_at_the_threshold_acts_and_below_holds():
    assert LearnedActPolicy(FakeScorer(0.5), CARDS, threshold=0.5).decide(_with(), 10.0, "CTX").act
    d = LearnedActPolicy(FakeScorer(0.49), CARDS, threshold=0.5).decide(_with(), 10.0, "CTX")
    assert d == ActDecision(False, "hold", None, "learned wait p=0.49")


def test_threshold_is_the_policy_threshold():
    assert LearnedActPolicy(FakeScorer(0.3), CARDS, threshold=0.25).decide(_with(), 10.0, "CTX").act
    assert not LearnedActPolicy(FakeScorer(0.3), CARDS, threshold=0.35).decide(_with(), 10.0, "CTX").act


def test_punish_keeps_the_rules_mode_and_lane_when_the_head_says_act():
    plays = [{"t": 8.0, "card": "knight", "tile": [14, 24]}]
    state = {**_with(elixir=7.0), "opponent": {**_with()["opponent"], "elixir_estimate": 2.5, "recent_plays": plays}}
    assert RULES.decide(state, 10.0).mode == "punish"
    d = LearnedActPolicy(FakeScorer(0.9), CARDS).decide(state, 10.0, "CTX")
    assert d.act and d.mode == "punish" and d.lane == "left" and d.reason == "learned act p=0.90"
    d = LearnedActPolicy(FakeScorer(0.1), CARDS).decide(state, 10.0, "CTX")        # the head can veto a punish
    assert not d.act and d.mode == "hold" and d.lane is None


def test_without_context_text_the_rules_decide():
    scorer = FakeScorer(1.0)
    policy = LearnedActPolicy(scorer, CARDS)
    assert policy.decide(_with(), 10.0) == RULES.decide(_with(), 10.0)
    assert scorer.calls == []


def test_reactive_policy_ignores_context_text():
    state = _with(units=[_enemy((5, 9), eta=2.0)])
    assert RULES.decide(state, 10.0, context_text="anything") == RULES.decide(state, 10.0)
    assert RULES.decide(state, 10.0, "anything").mode == "defend"


def test_cache_reuses_p_within_min_check_gap_s():
    scorer = FakeScorer(0.9)
    policy = LearnedActPolicy(scorer, CARDS, min_check_gap_s=0.5)
    assert policy.decide(_with(), 10.0, "CTX a").act
    scorer.p = 0.0                                              # a new query would now say wait
    assert policy.decide(_with(), 10.3, "CTX b").act            # 0.3 s later: cached p = 0.9
    assert policy.decide(_with(), 10.49, "CTX c").act
    assert scorer.calls == ["CTX a"]
    assert not policy.decide(_with(), 10.5, "CTX d").act        # the gap since the real query (10.0) is over
    assert scorer.calls == ["CTX a", "CTX d"]
    assert not policy.decide(_with(), 10.7, "CTX e").act        # gap counted from 10.5, not from 10.3
    assert scorer.calls == ["CTX a", "CTX d"]


def test_cache_is_not_used_when_time_goes_backwards():
    scorer = FakeScorer(0.9)
    policy = LearnedActPolicy(scorer, CARDS, min_check_gap_s=0.5)
    policy.decide(_with(), 10.0, "CTX a")
    policy.decide(_with(), 5.0, "CTX b")
    assert scorer.calls == ["CTX a", "CTX b"]


def test_rules_default_to_a_reactive_policy_on_the_given_cards():
    policy = LearnedActPolicy(FakeScorer(), CARDS)
    assert isinstance(policy.rules, ReactivePolicy) and policy.rules.cards is CARDS
    custom = ReactivePolicy(CARDS, leak_elixir=7.0)
    assert LearnedActPolicy(FakeScorer(0.0), rules=custom).decide(_with(elixir=7.0), 10.0, "CTX").mode == "leak"


# 3 -----------------------------------------------------------------------------------------------------------------
def _cands(*keys):
    return [Candidate(k, 1.0 - 0.1 * i, {"knn": 1.0}) for i, k in enumerate(keys)]


def test_learned_mode_drops_blind_spells_only():
    # one enemy far away on their half (beyond the 10 s a push is answered from, MAC-AD): not a threat, but a spell can
    # still target it (left lane, enemy_half zone)
    # a whole play of bats and a spear goblin: a zap kills six, 2.7 elixir (MAC-AM M2: a zap needs 2 kills and 2.5 of value;
    # not 2x its cost, so no `spell` act)
    swarm = ([_enemy((5 + 0.3 * i, 25), eta=12.0, card="bats") for i in range(5)]
             + [_enemy((4.5, 24.5), eta=12.0, card="spear-goblins")])
    state = _with(units=swarm)
    policy = LearnedActPolicy(FakeScorer(0.9), CARDS)
    d = policy.decide(state, 10.0, "CTX")
    assert d.mode == "learned"
    cands = _cands("zap|right|mid", "zap|left|enemy_half", "knight|right|back", "giant|left|bridge", "fireball|left|mid")
    out = policy.filter_candidates(cands, d, state)
    assert [c.key for c in out] == ["zap|left|enemy_half", "knight|right|back", "giant|left|bridge"]   # order kept
    assert policy.filter_candidates(_cands("zap|left|mid"), d, _with()) == []       # no enemy at all: every spell blind


def test_other_modes_delegate_to_the_rules():
    policy = LearnedActPolicy(FakeScorer(0.9), CARDS)
    state = _with(units=[_enemy((5, 9), eta=2.0)])
    d = policy.decide(state, 10.0, "CTX")
    assert d.mode == "defend"
    cands = _cands("knight|right|mid", "knight|left|mid", "giant|left|back", "zap|left|back")
    assert policy.filter_candidates(cands, d, state) == RULES.filter_candidates(cands, d, state)
    # MAC-AC: defend narrows to the cards the tactics engine calls an adequate answer. The giant only attacks
    # buildings, so it is not a defender (the zap is a blind spell here): only the knight survives of the generator's
    # candidates; the other cards the policy allowed come as the engine's own (MAC-AL P3.7: the LLM always sees them).
    out = policy.filter_candidates(cands, d, state)
    assert out[0].key == "knight|left|mid" and all(c.sources == {"tac": 0.0} for c in out[1:])
    assert {c.key.split("|")[0] for c in out} == set(d.allowed_cards)


def test_live_loop_runs_the_learned_policy_with_the_context_text():
    scorer = FakeScorer(0.9)
    ex = FakeExecutor()
    lp = _loop(builder=FakeBuilder(_with(), warmup=0), executor=ex,
               policy=LearnedActPolicy(scorer, CARDS, threshold=0.5))
    d = lp.step("f", 10.0, 0, 2.0)
    assert scorer.calls == ["CTX"]                              # LiveLoop.step passes its context text
    assert d is not None and d.mode == "learned" and d.reason == "learned act p=0.90"
    assert d.key == "giant|left|back"                           # the blind zap candidate was filtered out
    assert [c["key"] for c in d.candidates] == ["giant|left|back"] and len(ex.calls) == 1


def test_live_loop_holds_when_the_head_says_wait():
    lp = _loop(builder=FakeBuilder(_with(), warmup=0), policy=LearnedActPolicy(FakeScorer(0.1), CARDS))
    assert lp.step("f", 10.0, 0, 2.0) is None
    assert lp.policy_holds == {"hold": 1}


# 4 -----------------------------------------------------------------------------------------------------------------
def _line(game, split, t, act, since=None) -> dict:
    s, q = d1_format.act_request(f"ctx {game} {t:.1f}")
    return {"id": f"{game}#{t:.1f}", "game_id": game, "split": split, "kind": "dense", "variant": "act",
            "state": s, "questions": q, "answers": {"act": "true" if act else "false"},
            "meta": {"since_opp_play": since, "threat": False, "own_elixir": 5.0, "opp_elixir": 4.0,
                     "next_deploy_in": None}}


def _write_split(src: Path, split: str, lines: list[dict], states: dict | None = None) -> None:
    src.mkdir(parents=True, exist_ok=True)
    (src / f"{split}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    side = [{"id": l["id"], "t": float(l["id"].rsplit("#", 1)[1]),
             "state": (states or {}).get(l["id"], _state())} for l in lines]
    (src / f"states_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in side))


def _game(game, split, n, n_act):
    """n lines at t = 0..n-1 s; the first n_act are acts spread by index (every n // n_act-th line)."""
    act_at = {int(i * n / n_act) for i in range(n_act)} if n_act else set()
    return [_line(game, split, float(t), t in act_at) for t in range(n)]


def _make_src(tmp_path: Path) -> Path:
    src = tmp_path / "act_dense"
    _write_split(src, "train", _game("gA", "train", 100, 30) + _game("gB", "train", 10, 3))
    _write_split(src, "val", _game("gV", "val", 50, 10) + _game("gW", "val", 7, 2))
    _write_split(src, "test", _game("gT", "test", 5, 1))
    return src


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _sample(tmp_path: Path, *extra: str) -> tuple[Path, Path]:
    src, out = _make_src(tmp_path), tmp_path / "bal"
    assert sat.main(["--src", str(src), "--out", str(out), "--pos-per-game", "5", "--neg-per-game", "10",
                     "--eval-per-game", "20", *extra]) == 0
    return src, out


def test_sampler_caps_train_per_game_and_shuffles(tmp_path):
    src, out = _sample(tmp_path)
    train = _read(out / "train.jsonl")
    per_game: dict = {}
    for l in train:
        per_game.setdefault(l["game_id"], []).append(l)
    pos = {g: sum(l["answers"]["act"] == "true" for l in ls) for g, ls in per_game.items()}
    neg = {g: sum(l["answers"]["act"] == "false" for l in ls) for g, ls in per_game.items()}
    assert pos == {"gA": 5, "gB": 3} and neg == {"gA": 10, "gB": 7}      # caps, or all there is (gB has 3 acts)
    assert len({l["id"] for l in train}) == len(train) == 25            # without replacement
    assert {l["id"] for l in train} <= {l["id"] for l in _read(src / "train.jsonl")}
    games = [l["game_id"] for l in train]
    assert games != sorted(games)                                       # shuffled across games
    assert [l["id"] for l in train] != sorted(l["id"] for l in train)


def test_sampler_is_deterministic_per_seed(tmp_path):
    _, out = _sample(tmp_path)
    again = tmp_path / "again"
    assert sat.main(["--src", str(tmp_path / "act_dense"), "--out", str(again), "--pos-per-game", "5",
                     "--neg-per-game", "10", "--eval-per-game", "20"]) == 0
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "states_train.jsonl"):
        assert (out / name).read_text() == (again / name).read_text()
    other = tmp_path / "other"
    assert sat.main(["--src", str(tmp_path / "act_dense"), "--out", str(other), "--pos-per-game", "5",
                     "--neg-per-game", "10", "--eval-per-game", "20", "--seed", "1"]) == 0
    assert (out / "train.jsonl").read_text() != (other / "train.jsonl").read_text()


def test_sampler_eval_files_are_uniform_per_game_and_keep_file_order(tmp_path):
    src, out = _sample(tmp_path)
    val, src_val = _read(out / "val.jsonl"), _read(src / "val.jsonl")
    assert [l["game_id"] for l in val].count("gV") == 20 and [l["game_id"] for l in val].count("gW") == 7
    order = {l["id"]: i for i, l in enumerate(src_val)}
    idx = [order[l["id"]] for l in val]
    assert idx == sorted(idx)                                           # a subsequence of the source file
    assert _read(out / "test.jsonl") == _read(src / "test.jsonl")       # smaller than the cap: the game as it is


def test_sampler_keeps_the_natural_act_rate_when_nothing_is_cut(tmp_path):
    src = _make_src(tmp_path)
    out = tmp_path / "all"
    assert sat.main(["--src", str(src), "--out", str(out), "--eval-per-game", "1000"]) == 0
    assert _read(out / "val.jsonl") == _read(src / "val.jsonl")
    stats = json.loads((out / "stats.json").read_text())
    assert stats["splits"]["val"] == {"lines": 57, "act_rate": round(12 / 57, 4), "games": 2, "states_missing": 0}
    assert stats["splits"]["test"]["lines"] == 5 and stats["splits"]["test"]["act_rate"] == pytest.approx(0.2)
    assert stats["splits"]["train"]["games"] == 2


def test_sampler_sidecars_are_aligned_with_the_act_files(tmp_path):
    marked = {"gA#0.0": {**_state(), "clock": "marker"}}          # t = 0 is an act line: always sampled
    src = tmp_path / "act_dense"
    _write_split(src, "train", _game("gA", "train", 100, 30), marked)
    _write_split(src, "val", _game("gV", "val", 50, 10))
    _write_split(src, "test", _game("gT", "test", 5, 1))
    out = tmp_path / "bal"
    assert sat.main(["--src", str(src), "--out", str(out), "--pos-per-game", "50", "--neg-per-game", "50"]) == 0
    for split in ("train", "val", "test"):
        side = _read(out / f"states_{split}.jsonl")
        assert [r["id"] for r in side] == [l["id"] for l in _read(out / f"{split}.jsonl")]
    train_side = {r["id"]: r for r in _read(out / "states_train.jsonl")}
    assert train_side["gA#0.0"]["state"]["clock"] == "marker"        # the sidecar row of that very line was copied
    assert all(r["state"].get("clock") is None for i, r in train_side.items() if i != "gA#0.0")
    stats = json.loads((out / "stats.json").read_text())
    assert stats["splits"]["train"]["lines"] == 80 and stats["args"]["seed"] == 0


def test_sampler_without_source_sidecars_writes_none(tmp_path):
    src = tmp_path / "act_dense"
    _write_split(src, "train", _game("gA", "train", 10, 3))
    for f in src.glob("states_*.jsonl"):
        f.unlink()
    out = tmp_path / "bal"
    assert sat.main(["--src", str(src), "--out", str(out)]) == 0
    assert not list(out.glob("states_*.jsonl")) and _read(out / "val.jsonl") == []


# 5 -----------------------------------------------------------------------------------------------------------------
class FakeD1:
    """Stand-in for D1Ranker in the eval script: p per state string, constructor kwargs recorded."""
    inits: list = []
    p_by_state: dict = {}

    def __init__(self, **kwargs):
        FakeD1.inits.append(kwargs)

    def probabilities(self, state, questions):
        d1_format.validate(state, questions)
        p = FakeD1.p_by_state[state]
        return {"act": [1.0 - p, p]}


# (id, pro acts, p, state kind, since_opp_play)
VAL_ROWS = [("v1", True, 0.90, "hold", 1.0), ("v2", True, 0.70, "hold", 2.0), ("v3", True, 0.35, "hold", None),
            ("v4", False, 0.30, "threat", 5.0), ("v5", False, 0.20, "hold", None), ("v6", False, 0.10, "hold", 0.5)]
TEST_ROWS = [("t1", True, 0.80, "hold", 3.0), ("t2", False, 0.40, "hold", 10.0),
             ("t3", False, 0.60, "threat", 2.0), ("t4", True, 0.20, "hold", None)]


def _eval_data(tmp_path: Path) -> Path:
    data = tmp_path / "ds" / "act_dense_bal"
    FakeD1.p_by_state = {}
    for split, rows in (("val", VAL_ROWS), ("test", TEST_ROWS)):
        lines, states = [], {}
        for i, (ident, act, p, kind, since) in enumerate(rows):
            line = _line(f"g_{split}", split, float(i), act, since)       # id "g_<split>#<i>.0"; ident = row label
            lines.append(line)
            FakeD1.p_by_state[line["state"]] = p
            states[line["id"]] = _with(units=[_enemy((5, 9), eta=2.0)]) if kind == "threat" else _with()
        _write_split(data, split, lines, states)
    return data


@pytest.fixture
def fake_d1(monkeypatch):
    FakeD1.inits = []
    monkeypatch.setattr(eah, "D1Ranker", FakeD1)
    return FakeD1


def test_eval_picks_the_val_threshold_closest_to_half_among_ties(tmp_path, fake_d1):
    data = _eval_data(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    result = eah.main(["--model", str(model), "--data", str(data), "--backend", "mlx"])
    # val separates perfectly for any threshold in (0.30, 0.35]; the one closest to 0.5 is 0.35
    assert result["threshold"] == 0.35 and result["val_balanced_acc"] == 1.0
    assert json.loads((model / "act_threshold.json").read_text()) == {"threshold": 0.35, "val_balanced_acc": 1.0}
    assert fake_d1.inits == [{"model_path": str(model), "backend": "mlx"}]


def test_pick_threshold_sweeps_005_to_095():
    assert eah.THRESHOLDS[0] == 0.05 and eah.THRESHOLDS[-1] == 0.95 and len(eah.THRESHOLDS) == 91
    # perfect separation for any threshold in (0.1, 0.8]: the one closest to 0.5 is 0.5 itself
    assert eah.pick_threshold([True, True, False], [0.9, 0.8, 0.1]) == (0.5, 1.0)
    # bacc 0.75 is reached for thresholds in (0.1, 0.2] and in (0.9, 0.95]: 0.20 is closer to 0.5 than 0.91
    assert eah.pick_threshold([True, False, True, False], [0.95, 0.9, 0.2, 0.1]) == (0.2, 0.75)
    assert eah.pick_threshold([True, True], [0.4, 0.6]) == (0.4, 1.0)          # one class only: its recall counts
    assert eah.pick_threshold([], []) == (0.5, None)


def test_eval_reports_all_policies_per_split(tmp_path, fake_d1):
    data = _eval_data(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    out = tmp_path / "out" / "act_head.json"
    result = eah.main(["--model", str(model), "--data", str(data), "--out", str(out)])
    assert json.loads(out.read_text()) == result and out.with_suffix(".md").exists()
    assert set(result["splits"]) == {"val", "test"}
    val = result["splits"]["val"]
    assert set(val["policies"]) == {"learned", "rules", "combined", "always", "never"}
    p = val["policies"]
    assert p["learned"]["accuracy"] == 1.0 and p["learned"]["act_rate"] == 0.5 == val["pro_act_rate"]
    assert p["rules"]["confusion"] == {"tp": 0, "fp": 1, "fn": 3, "tn": 2}      # the rules only see the threat (v4)
    assert p["rules"]["act_rate"] == pytest.approx(1 / 6)
    assert p["combined"]["confusion"] == {"tp": 3, "fp": 1, "fn": 0, "tn": 2}   # learned + defend on v4
    assert p["combined"]["act_precision"] == 0.75 and p["combined"]["act_recall"] == 1.0
    assert p["combined"]["wait_precision"] == 1.0 and p["combined"]["wait_recall"] == pytest.approx(2 / 3)
    assert p["always"]["act_rate"] == 1.0 and p["always"]["balanced_accuracy"] == 0.5
    assert p["never"]["act_rate"] == 0.0 and p["never"]["accuracy"] == 0.5 and p["never"]["act_precision"] is None
    test = result["splits"]["test"]["policies"]
    assert test["learned"]["confusion"] == {"tp": 1, "fp": 2, "fn": 1, "tn": 0}    # thr 0.35: t1 T, t2 T, t3 T, t4 F
    assert test["rules"]["confusion"] == {"tp": 0, "fp": 1, "fn": 2, "tn": 1}      # t3 is the threat
    shown = out.with_suffix(".md").read_text()
    for name in ("learned", "rules", "combined", "always", "never", "reaction slice"):
        assert name in shown


def test_eval_reaction_slice_covers_recent_opponent_plays(tmp_path, fake_d1):
    data = _eval_data(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    result = eah.main(["--model", str(model), "--data", str(data), "--out", str(tmp_path / "o.json")])
    r = result["splits"]["val"]["reaction"]
    assert r["n"] == 3 and r["pro_act_rate"] == pytest.approx(2 / 3)            # v1, v2, v6 (since <= 3.0)
    pol = r["policies"]
    assert pol["learned"]["act_rate"] == pytest.approx(2 / 3) and pol["learned"]["act_recall"] == 1.0
    assert pol["rules"]["act_rate"] == 0.0 and pol["rules"]["act_recall"] == 0.0
    assert pol["combined"]["act_rate"] == pytest.approx(2 / 3) and pol["combined"]["act_precision"] == 1.0
    assert pol["always"]["act_recall"] == 1.0 and pol["never"]["act_rate"] == 0.0
    assert result["splits"]["test"]["reaction"]["n"] == 2                       # t1 (3.0, inclusive) and t3 (2.0)


def test_eval_limit_and_split_options(tmp_path, fake_d1):
    data = _eval_data(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    result = eah.main(["--model", str(model), "--data", str(data), "--out", str(tmp_path / "o.json"),
                       "--splits", "val", "--limit", "3"])
    assert set(result["splits"]) == {"val"} and result["splits"]["val"]["n"] == 3 and result["limit"] == 3
    with pytest.raises(SystemExit):
        eah.main(["--model", str(model), "--data", str(data), "--splits", "test"])        # val is needed
    with pytest.raises(SystemExit):
        eah.main(["--model", str(model), "--data", str(data), "--splits", "val,dev"])


def test_eval_default_out_is_next_to_the_data_dir(tmp_path, fake_d1):
    data = _eval_data(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    eah.main(["--model", str(model), "--data", str(data), "--splits", "val"])
    assert (tmp_path / "ds" / "eval" / "act_head.json").exists()


def test_eval_without_the_sidecar_fails_clearly(tmp_path, fake_d1):
    data = _eval_data(tmp_path)
    (data / "states_val.jsonl").unlink()
    with pytest.raises(SystemExit, match="states_val.jsonl"):
        eah.main(["--model", str(tmp_path), "--data", str(data), "--splits", "val"])


# 6 -----------------------------------------------------------------------------------------------------------------
class RecordingD1:
    inits: list = []

    def __init__(self, **kwargs):
        RecordingD1.inits.append(kwargs)


def _patch_main(monkeypatch, seen: dict) -> None:
    class StubLoop:
        def __init__(self, *a, **k):
            seen.update(k)
            self.summary = _loop().summary()

        def run(self, **k):
            return self.summary

    RecordingD1.inits = []
    monkeypatch.setattr(ranker_mod, "D1Ranker", RecordingD1)
    monkeypatch.setattr(live_mod, "_build_source", lambda args: FakeSource())
    monkeypatch.setattr(live_mod, "_build_perception", lambda args, source: FakePerception())
    monkeypatch.setattr(live_mod, "_build_ranker", lambda args: None)
    monkeypatch.setattr(live_mod, "_build_plan_text", lambda args, deck: None)
    monkeypatch.setattr(live_mod, "load_cards", lambda: {})
    monkeypatch.setattr(live_mod, "load_placement_prior", lambda: PRIOR)
    monkeypatch.setattr(live_mod, "StateIndex", lambda: None)
    monkeypatch.setattr(live_mod, "KnowledgeGraph", lambda: None)
    monkeypatch.setattr(live_mod, "CandidateGenerator", lambda *a: FakeGenerator())
    monkeypatch.setattr(live_mod, "LiveLoop", StubLoop)


def _run_main(tmp_path: Path, *extra: str) -> int:
    return main(_argv("--source", "video", "--video", "x.mp4", "--device", "cpu", "--out", str(tmp_path / "run"),
                      *extra))


def test_cli_act_flags_and_policy_choices():
    args = build_parser().parse_args(_argv())
    assert args.policy == "reactive" and args.act_model is None and args.act_threshold is None
    assert args.act_backend == "transformers"
    args = build_parser().parse_args(_argv("--policy", "learned", "--act-model", "/m/act", "--act-threshold", "0.4",
                                           "--act-backend", "mlx"))
    assert args.policy == "learned" and args.act_model == "/m/act" and args.act_threshold == 0.4
    assert args.act_backend == "mlx"
    with pytest.raises(SystemExit):
        build_parser().parse_args(_argv("--act-backend", "onnx"))


def test_cli_learned_without_act_model_exits_2(capsys):
    assert main(_argv("--policy", "learned")) == 2
    assert "--act-model" in capsys.readouterr().err


def test_cli_learned_reads_the_threshold_file_of_the_model(monkeypatch, tmp_path):
    seen: dict = {}
    _patch_main(monkeypatch, seen)
    model = tmp_path / "act_model"
    model.mkdir()
    (model / "act_threshold.json").write_text(json.dumps({"threshold": 0.62, "val_balanced_acc": 0.7}))
    assert _run_main(tmp_path, "--policy", "learned", "--act-model", str(model)) == 0
    policy = seen["policy"]
    assert isinstance(policy, LearnedActPolicy) and policy.threshold == 0.62
    assert isinstance(policy.scorer, RecordingD1)
    assert RecordingD1.inits == [{"model_path": str(model), "backend": "transformers"}]
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["policy"] == "learned" and config["act_threshold"] == 0.62 and config["act_model"] == str(model)


def test_cli_act_threshold_flag_beats_the_file_and_defaults_to_half(monkeypatch, tmp_path):
    seen: dict = {}
    _patch_main(monkeypatch, seen)
    model = tmp_path / "act_model"
    model.mkdir()
    (model / "act_threshold.json").write_text(json.dumps({"threshold": 0.62}))
    assert _run_main(tmp_path, "--policy", "learned", "--act-model", str(model), "--act-threshold", "0.3",
                     "--act-backend", "mlx") == 0
    assert seen["policy"].threshold == 0.3 and RecordingD1.inits[-1]["backend"] == "mlx"
    bare = tmp_path / "bare"
    bare.mkdir()
    assert _run_main(tmp_path, "--policy", "learned", "--act-model", str(bare)) == 0
    assert seen["policy"].threshold == 0.5


def test_cli_learned_with_a_broken_threshold_file_exits_2(monkeypatch, tmp_path, capsys):
    seen: dict = {}
    _patch_main(monkeypatch, seen)
    model = tmp_path / "act_model"
    model.mkdir()
    (model / "act_threshold.json").write_text("{not json")
    assert _run_main(tmp_path, "--policy", "learned", "--act-model", str(model)) == 2
    assert "act_threshold.json" in capsys.readouterr().err and "policy" not in seen


def test_cli_reactive_and_trigger_still_pass_their_names(monkeypatch, tmp_path):
    seen: dict = {}
    _patch_main(monkeypatch, seen)
    assert _run_main(tmp_path) == 0 and seen["policy"] == "reactive"
    assert _run_main(tmp_path, "--policy", "trigger") == 0 and seen["policy"] == "trigger"
    assert RecordingD1.inits == []             # no act model is built unless --policy learned
