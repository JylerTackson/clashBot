"""MAC-AB A4: the lookup-vs-tactics arbiter (d1 always decides, the arbiter only narrows its options), the offline
eval script on a tiny synthetic dataset, and the live-loop wiring - fakes only, no models, no screen."""
import json
import random
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.arbiter import (ArbiterConfig, RuleArbiter, argmax, pick_within, urgent_from_report)  # noqa: E402
from clashbot.candidates import Candidate  # noqa: E402
from clashbot.live import build_parser, main  # noqa: E402
from clashbot.ranker import RankResult, TacticsRanker  # noqa: E402
from clashbot.tactics import Push, TacticalReport  # noqa: E402
from tests.test_live_loop import (PRIOR, FakeBuilder, FakeExecutor, FakeGenerator, FakeKeyRanker,  # noqa: E402
                                  FakePerception, FakeSource, _argv, _loop)
from tools.eval import eval_arbiter  # noqa: E402


def C(key, **sources):
    return types.SimpleNamespace(key=key, sources=sources)


def within(probs, subset):
    """Reference d1 pick within a subset: the highest probability, the first index on ties."""
    best = subset[0]
    for i in subset:
        if probs[i] > probs[best]:
            best = i
    return best


# 1. the four rules, in order --------------------------------------------------------------------------------------
def test_rule1_confident_d1_chooses_among_everything():
    cands = [C("a", knn=1.0), C("b", tac=0.9), C("c", knn=0.9)]
    arb = RuleArbiter()
    d = arb.decide(cands, [0.1, 0.8, 0.1], urgent=True)        # urgent push and a tac candidate, but d1 is sure
    assert (d.index, d.reason, d.subset) == (1, "d1 confident", [0, 1, 2])
    assert d.probs == pytest.approx([0.1, 0.8, 0.1])
    d = arb.decide(cands, [0.9, 0.05, 0.05], urgent=True)      # d1 sure of a non-tactical, non-pro candidate
    assert (d.index, d.reason) == (0, "d1 confident")
    assert arb.choose(cands, [0.55, 0.3, 0.15], urgent=True) == (0, "d1 confident")        # conf is inclusive


def test_rule2_urgent_d1_chooses_among_tactics_and_the_best_lookup():
    cands = [C("plain", kg=0.4), C("tac", tac=0.9), C("pro-lo", knn=0.3), C("pro-hi", knn=0.8), C("pro-mid", knn=0.5)]
    d = RuleArbiter().decide(cands, [0.40, 0.15, 0.10, 0.25, 0.10], urgent=True)
    assert d.reason == "urgent push: d1 among tactics+lookup"
    assert d.subset == [1, 3]                                  # the tac candidate + the single best-kNN lookup
    assert d.index == 3                                        # d1 prefers pro-hi (0.25) to tac (0.15); plain (0.40) is out
    assert d.probs == pytest.approx([0.15 / 0.40, 0.25 / 0.40])
    d = RuleArbiter().decide(cands, [0.40, 0.30, 0.10, 0.10, 0.10], urgent=True)
    assert (d.index, d.subset) == (1, [1, 3])                  # here d1 prefers the tactical candidate


def test_rule2_best_lookup_is_not_a_tactical_candidate_and_needs_knn():
    cands = [C("tac", tac=0.9, knn=0.9), C("lo", knn=0.4), C("none", kg=0.2)]
    d = RuleArbiter().decide(cands, [0.3, 0.3, 0.4], urgent=True)
    assert d.subset == [0, 1] and d.index == 0                 # the lookup is the best *non-tactical* knn candidate
    d = RuleArbiter().decide([C("tac", tac=0.9), C("x", kg=1), C("y", fallback=1)], [0.3, 0.4, 0.3], urgent=True)
    assert d.subset == [0] and d.index == 0                    # no lookup candidate: tactics alone


def test_rule2_needs_urgency_and_a_tactical_candidate():
    cands = [C("a", knn=0.9), C("b", knn=0.1)]
    d = RuleArbiter().decide(cands, [0.3, 0.5], urgent=True)       # urgent but no tac candidate -> rule 3
    assert d.reason == "d1 among pro-supported + tactics" and d.subset == [0]
    cands = [C("a", kg=0.9), C("t", tac=0.9)]
    d = RuleArbiter().decide(cands, [0.5, 0.5], urgent=False)            # tac candidate but no push, no knn -> rule 4
    assert (d.reason, d.subset, d.index) == ("d1", [0, 1], 0)


def test_rule3_pro_supported_plus_tactics():
    cands = [C("plain", kg=1.0), C("pro", knn=0.6), C("weak", knn=0.4), C("tac", tac=0.5)]
    d = RuleArbiter().decide(cands, [0.40, 0.15, 0.25, 0.20], urgent=False)
    assert d.reason == "d1 among pro-supported + tactics" and d.subset == [1, 3]
    assert d.index == 3                                        # tac (0.20) beats pro (0.15) in d1's eyes
    d = RuleArbiter(ArbiterConfig(knn_min=0.4)).decide(cands, [0.40, 0.15, 0.25, 0.20], urgent=False)
    assert d.subset == [1, 2, 3] and d.index == 2              # knn_min is inclusive: weak (0.4) joins, d1 likes it
    d = RuleArbiter(ArbiterConfig(knn_min=0.7)).decide(cands, [0.40, 0.15, 0.25, 0.20], urgent=False)
    assert (d.reason, d.index) == ("d1", 0)                    # nothing reaches 0.7 -> plain d1


def test_rule4_plain_d1():
    cands = [C("a", kg=0.5), C("b", fallback=1), C("c", knn=0.1)]
    d = RuleArbiter().decide(cands, [0.2, 0.45, 0.35], urgent=False)
    assert (d.index, d.reason, d.subset) == (1, "d1", [0, 1, 2])
    assert d.probs == pytest.approx([0.2, 0.45, 0.35])


def test_order_of_the_rules():
    cands = [C("plain"), C("tac", tac=0.9), C("pro", knn=0.9)]
    arb = RuleArbiter()
    assert arb.decide(cands, [0.6, 0.2, 0.2], urgent=True).reason == "d1 confident"             # 1 beats 2, 3
    assert arb.decide(cands, [0.4, 0.3, 0.3], urgent=True).reason == "urgent push: d1 among tactics+lookup"   # 2 beats 3
    assert arb.decide(cands, [0.4, 0.3, 0.3], urgent=False).reason == "d1 among pro-supported + tactics"
    assert arb.decide([C("a"), C("b")], [0.4, 0.3], urgent=True).reason == "d1"           # 4 is the fall-through


def test_config_is_respected():
    cands = [C("a"), C("b", knn=0.6)]
    assert RuleArbiter(ArbiterConfig(conf=0.9)).decide(cands, [0.7, 0.3], urgent=False).index == 1
    assert RuleArbiter(ArbiterConfig(conf=0.6)).decide(cands, [0.7, 0.3], urgent=False).index == 0
    assert ArbiterConfig() == ArbiterConfig(conf=0.55, urgent_eta_s=4.0, knn_min=0.5)


# 2. ties, empty probs ---------------------------------------------------------------------------------------------
def test_ties_go_to_the_first():
    cands = [C("a"), C("b", tac=0.5), C("c", tac=0.5), C("d", knn=0.9)]
    assert RuleArbiter().choose(cands, [0.25, 0.25, 0.25, 0.25], urgent=False)[0] == 1     # first of subset {1, 2, 3}
    assert RuleArbiter().choose([C("a"), C("b")], [0.5, 0.5], urgent=False)[0] == 0
    assert argmax([0.1, 0.4, 0.4]) == 1 and pick_within([0.3, 0.3, 0.3], [1, 2])[0] == 1


def test_empty_probs_pick_index_zero():
    cands = [C("a", knn=0.9), C("b", tac=0.9)]
    d = RuleArbiter().decide(cands, [], urgent=True)
    assert d.index == 0 and d.probs == [1.0]
    assert RuleArbiter().choose([C("only")], [], urgent=False)[0] == 0
    with pytest.raises(ValueError):
        RuleArbiter().decide([], [], urgent=False)


def test_short_probs_count_missing_as_zero():
    d = RuleArbiter().decide([C("a"), C("b", knn=0.9), C("c", knn=0.9)], [0.3, 0.2], urgent=False)
    assert d.subset == [1, 2] and d.index == 1                 # c has no d1 probability
    assert pick_within([0.0, 0.0], [0, 1])[1] == [0.5, 0.5]    # all-zero subset renormalises to uniform


# 3. d1 always decides: the pick is the argmax of d1's probabilities within the subset ------------------------------
def test_pick_is_always_d1_argmax_within_the_subset():
    rng = random.Random(7)
    arb_rules = set()
    for _ in range(2000):
        n = rng.randint(1, 6)
        cands = []
        for i in range(n):
            src = {}
            if rng.random() < 0.3:
                src["tac"] = round(rng.random(), 2)
            if rng.random() < 0.5:
                src["knn"] = round(rng.choice([0.0, 0.1, 0.3, 0.5, 0.7, 1.0]), 2)
            if rng.random() < 0.2:
                src["kg"] = 0.5
            cands.append(C(f"c{i}", **src))
        raw = [rng.choice([0.0, 0.1, 0.25, 0.25, 0.4, 0.6, rng.random()]) for _ in range(n)]
        total = sum(raw)
        probs = [r / total for r in raw] if total > 0 else [1.0 / n] * n          # ties on purpose
        cfg = ArbiterConfig(conf=rng.choice([0.3, 0.55, 0.8]), urgent_eta_s=4.0, knn_min=rng.choice([0.2, 0.5, 1.0]))
        urgent = rng.random() < 0.5
        d = RuleArbiter(cfg).decide(cands, probs, urgent)
        arb_rules.add(d.reason)
        assert d.index in d.subset and d.subset == sorted(set(d.subset))
        assert d.index == within(probs, d.subset)                                # d1's argmax inside the subset
        assert sum(d.probs) == pytest.approx(1.0) and len(d.probs) == len(d.subset)
        assert d.index == d.subset[argmax(d.probs)]                              # renormalising keeps the argmax
        if max(probs) >= cfg.conf:                                               # confident: d1's global argmax
            assert d.index == within(probs, list(range(n))) and d.subset == list(range(n))
        assert RuleArbiter(cfg).choose(cands, probs, urgent) == (d.index, d.reason)
    assert arb_rules == {"d1 confident", "urgent push: d1 among tactics+lookup",
                         "d1 among pro-supported + tactics", "d1"}


def test_urgent_from_report():
    push = lambda eta: Push("left", [], 4.0, eta, [], "x")      # noqa: E731
    rep = TacticalReport([push(6.0), push(3.5)], [], None, "hold", ["tactics: hold"])
    assert urgent_from_report(rep, 4.0) is True and urgent_from_report(rep, 3.5) is True
    assert urgent_from_report(rep, 3.0) is False
    assert urgent_from_report(TacticalReport([], [], None, "hold", ["tactics: hold"]), 4.0) is False
    assert urgent_from_report(None, 4.0) is False


# 4. note parsing and the eval script --------------------------------------------------------------------------------
def test_parse_sources_from_notes():
    desc = "giant left lane, back"
    assert eval_arbiter.parse_sources(desc) == {}
    assert eval_arbiter.parse_sources(desc + " — filler: no pro or counter support") == {}
    assert eval_arbiter.parse_sources(desc + " — tactics: defend, tanks for the tower vs giant, trade +2") == {"tac": 1.0}
    assert eval_arbiter.parse_sources(desc + " — pros played this in similar states (support 0.42)") == {"knn": 0.42}
    assert eval_arbiter.parse_sources(desc + " — pros played this in similar states (support 1.00)") == {"knn": 1.0}
    assert eval_arbiter.parse_sources(desc + " — known counter of battle-ram") == {"kg": 1}
    assert eval_arbiter.parse_sources(desc + " — tactics: defend, blocks bandit; pros played this too") == {
        "tac": 1.0, "knn": 0.5}
    given = desc + " — pros played this in similar states (support 0.30); pros played this too"
    assert eval_arbiter.parse_sources(given) == {"knn": 0.3}                       # a given value beats the 0.5 default


def test_push_etas_from_the_state_lines():
    state = ("[single_elixir|2:48] elixir 10\nfield: none\n"
             "tactics: push left: battle-ram (4 elixir), tower in 2.0s; needs building_targeter\n"
             "tactics: push right: p-e-k-k-a (7 elixir), tower in 23.5s; needs tank\n"
             "tactics: best bomb-tower @[10.5,6] defend: pulls battle-ram to the centre, trade +0\n"
             "tactics: hold: no threat, saving elixir (own 6, opp ~7.72)")
    assert eval_arbiter.push_etas(state) == [2.0, 23.5]
    assert eval_arbiter.push_etas("tactics: hold: no threat") == []


def _line(i, split, keys, notes, gold, push_eta=None):
    state = "[single_elixir|2:48] elixir 10\nfield: none"
    if push_eta is not None:
        state += f"\ntactics: push left: giant (5 elixir), tower in {push_eta}s; needs tank"
    criteria = {k: f"{k.split('|')[0]} desc" + (f" — {notes[j]}" if notes[j] else "") for j, k in enumerate(keys)}
    return {"id": f"g{i}", "split": split, "variant": "joint", "state": state,
            "questions": {"play": {"type": "choice", "instructions": "x", "criteria": criteria}},
            "answers": {"play": keys[gold]}}


KEYS4 = ["a|left|mid", "b|left|mid", "c|right|mid", "d|right|mid"]
TAC = "tactics: defend, tanks for the tower vs giant, trade +2"
PRO = "pros played this in similar states (support 0.90)"


def _synthetic(tmp_path, scramble=False):
    """Split val: d1 is unsure and wrong on lines where the tactical option (with an urgent push) is the answer, and
    right elsewhere; test is the same shape. The d1 eval JSON carries the probs in the criteria order."""
    lines = {"val": [], "test": []}
    per_example = []
    for split in ("val", "test"):
        for i in range(8):
            if i < 4:       # urgent push, tactical option 'c' is the answer, d1 unsure and prefers 'a'
                notes, gold, eta, probs = [None, None, TAC, None], 2, 3.0, [0.34, 0.2, 0.3, 0.16]
            else:           # no push, d1 confident and right
                notes, gold, eta, probs = [None, PRO, None, None], 1, None, [0.1, 0.7, 0.1, 0.1]
            line = _line(i, split, KEYS4, notes, gold, eta)
            lines[split].append(line)
            keys = list(line["questions"]["play"]["criteria"])
            shown = list(reversed(probs)) if scramble else probs
            per_example.append({"id": line["id"], "variant": "joint", "split": split, "pred": keys[argmax(probs)],
                                "gold": keys[gold], "probs": shown})
    for split, rows in lines.items():
        (tmp_path / "joint_tac").mkdir(exist_ok=True)
        (tmp_path / "joint_tac" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "d1_eval.json").write_text(json.dumps({"per_example": per_example}))
    return tmp_path


def test_eval_main_chooses_on_val_and_reports_test(tmp_path, capsys):
    ds = _synthetic(tmp_path)
    out = tmp_path / "out" / "arbiter.json"
    assert eval_arbiter.main(["--dataset", str(ds), "--d1-eval", str(ds / "d1_eval.json"),
                              "--lines", str(ds / "joint_tac"), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    st = res["strategies"]
    assert st["d1"]["val"]["acc"] == 0.5 and st["d1"]["test"]["acc"] == 0.5          # right only on the confident half
    arb = st["arbiter"]
    assert arb["val"]["acc"] == 1.0 and arb["test"]["acc"] == 1.0                    # tactics subset rescues the urgent half
    assert arb["params"]["conf"] >= 0.35 and arb["params"]["urgent_eta_s"] in (4.0, 6.0)    # eta 2 would not see 3.0 s
    assert arb["val"]["fixed"] == 4 and arb["val"]["broken"] == 0
    rules = arb["val"]["rules"]
    assert rules["urgent push: d1 among tactics+lookup"]["fired"] == 4 and rules["d1 confident"]["fired"] == 4
    assert rules["urgent push: d1 among tactics+lookup"]["acc"] == 1.0
    assert rules["urgent push: d1 among tactics+lookup"]["d1_acc"] == 0.0
    # the pure baselines are labelled as non-LLM diagnostics
    assert st["tactics_first"]["llm_decides"] is False and st["knn_first"]["llm_decides"] is False
    assert st["tactics_first"]["val"]["acc"] == 1.0 and st["knn_first"]["val"]["acc"] == 0.5
    assert len(res["arbiter_grid"]) == 11 * 5 * 3
    assert res["checks"]["val"] == {"lines": 8, "joined": 8, "missing_d1": 0, "size_mismatch": 0, "pred_mismatch": 0,
                                    "gold_mismatch": 0}
    assert res["coverage"]["val"]["with_tac"] == 0.5 and res["coverage"]["val"]["urgent_4s"] == 0.5
    md = out.with_suffix(".md").read_text()
    assert "arbiter, grid chosen on val" in md and "Rule firing" in md and "non-LLM baseline" in md
    assert "d1 alone" in capsys.readouterr().out


def test_eval_chosen_point_is_picked_on_val_not_test(tmp_path):
    ds = _synthetic(tmp_path)
    # make the test half different: there d1 is right on the urgent lines, so the val-chosen restriction hurts
    rows = [json.loads(line) for line in (ds / "joint_tac" / "test.jsonl").read_text().splitlines()]
    d1 = json.loads((ds / "d1_eval.json").read_text())
    for row in rows[:4]:
        row["answers"]["play"] = KEYS4[0]
    for e in d1["per_example"]:
        if e["split"] == "test" and int(e["id"][1:]) < 4:
            e["gold"] = KEYS4[0]
    (ds / "joint_tac" / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ds / "d1_eval.json").write_text(json.dumps(d1))
    out = ds / "arbiter.json"
    eval_arbiter.main(["--dataset", str(ds), "--d1-eval", str(ds / "d1_eval.json"), "--lines", str(ds / "joint_tac"),
                       "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["strategies"]["arbiter"]["val"]["acc"] == 1.0           # chosen for its val accuracy ...
    assert res["strategies"]["arbiter"]["test"]["acc"] == 0.5          # ... and reported on test as it comes out


def test_eval_detects_probs_in_the_wrong_order(tmp_path):
    ds = _synthetic(tmp_path, scramble=True)
    entries = eval_arbiter.load_d1_entries(ds / "d1_eval.json")
    lines = eval_arbiter.read_jsonl(ds / "joint_tac" / "val.jsonl")
    examples, check = eval_arbiter.load_examples(lines, entries, "val")
    assert check["pred_mismatch"] == 8 and len(examples) == 8
    assert eval_arbiter.main(["--dataset", str(ds), "--d1-eval", str(ds / "d1_eval.json"),
                              "--lines", str(ds / "joint_tac"), "--out", str(ds / "o.json")]) == 0
    assert json.loads((ds / "o.json").read_text())["checks"]["test"]["pred_mismatch"] == 8


def test_eval_missing_d1_probs_are_skipped_and_counted(tmp_path):
    ds = _synthetic(tmp_path)
    d1 = json.loads((ds / "d1_eval.json").read_text())
    d1["per_example"] = [e for e in d1["per_example"] if not (e["split"] == "val" and e["id"] == "g0")]
    (ds / "d1_eval.json").write_text(json.dumps(d1))
    entries = eval_arbiter.load_d1_entries(ds / "d1_eval.json")
    examples, check = eval_arbiter.load_examples(eval_arbiter.read_jsonl(ds / "joint_tac" / "val.jsonl"), entries, "val")
    assert (check["lines"], check["joined"], check["missing_d1"]) == (8, 7, 1) and len(examples) == 7


# 5. live loop -------------------------------------------------------------------------------------------------------
class ProbsRanker:
    """A d1 stand-in with fixed probabilities; counts its calls (the arbiter must not need a second one)."""
    name = "probs"

    def __init__(self, probs):
        self.probs = probs
        self.calls = 0

    def choose_keys(self, context, keys, plan=None, notes=None):
        self.calls += 1
        i = argmax(self.probs[:len(keys)])
        return RankResult(i, list(self.probs[:len(keys)]), "fake")


class FakeEngine:
    def __init__(self, eta):
        self.report = TacticalReport([Push("left", [], 5.0, eta, [], "giant (5 elixir)")], [], None, "fake",
                                     ["tactics: push left: giant (5 elixir), tower in 3.0s; needs tank"])

    def analyse(self, state, t=None):
        return self.report


class SetGenerator:
    def __init__(self):
        self.cands = [Candidate("knight|middle|mid", 0.1, {"kg": 0.3}), Candidate("zap|left|mid", 0.5, {"knn": 1.0}),
                      Candidate("giant|left|back", 0.9, {"tac": 0.9}, [4.0, 6.0], "tactics: defend, x"),
                      Candidate("archers|right|mid", 0.2, {"knn": 0.1})]

    def generate(self, state, t, ctx, n=4, report=None, tac_n=2, **filters):
        return list(self.cands)


PROBS = [0.40, 0.20, 0.30, 0.10]         # d1 alone picks knight (index 0); unsure (< 0.55)


def _arb_loop(tmp_path=None, arbiter=None, ranker=None, eta=3.0, **kw):
    return _loop(tmp_path, builder=FakeBuilder(warmup=0), generator=SetGenerator(),
                 ranker=ranker or ProbsRanker(PROBS), executor=FakeExecutor(), tactics=FakeEngine(eta),
                 arbiter=arbiter, **kw)


def test_live_arbiter_rules_uses_the_arbiters_pick_and_logs_the_reason(tmp_path):
    rk = ProbsRanker(PROBS)
    lp = _arb_loop(tmp_path, arbiter=RuleArbiter(), ranker=rk)
    d = lp.step("f", 1.0, 0, 2.0)
    # urgent (3.0 s <= 4.0), unsure: d1 chooses between the tactical giant (0.30) and the best lookup zap (0.20)
    assert d.key == "giant|left|back" and d.tile == [4.0, 6.0]
    assert d.arbiter_reason == "urgent push: d1 among tactics+lookup"
    assert d.arbiter_subset == [{"key": "zap|left|mid", "p": 0.4}, {"key": "giant|left|back", "p": 0.6}]
    assert d.probs == PROBS and rk.calls == 1                       # d1's own probabilities, one d1 call
    line = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert line["arbiter_reason"] == d.arbiter_reason and line["arbiter_subset"] == d.arbiter_subset


def test_live_arbiter_follows_the_configuration(tmp_path):
    arb = RuleArbiter(ArbiterConfig(conf=0.35))                     # d1's 0.40 is now confident
    d = _arb_loop(arbiter=arb).step("f", 1.0, 0, 2.0)
    assert d.key == "knight|middle|mid" and d.arbiter_reason == "d1 confident" and len(d.arbiter_subset) == 4
    arb = RuleArbiter(ArbiterConfig(urgent_eta_s=2.0))              # the push (3.0 s) is no longer urgent
    d = _arb_loop(arbiter=arb).step("f", 1.0, 0, 2.0)
    assert d.arbiter_reason == "d1 among pro-supported + tactics"
    assert [s["key"] for s in d.arbiter_subset] == ["zap|left|mid", "giant|left|back"]
    assert d.key == "giant|left|back"


def test_live_without_a_tactics_engine_the_report_is_none_and_not_urgent():
    lp = _loop(builder=FakeBuilder(warmup=0), generator=SetGenerator(), ranker=ProbsRanker(PROBS),
               executor=FakeExecutor(), arbiter=RuleArbiter())
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.arbiter_reason == "d1 among pro-supported + tactics" and d.key == "giant|left|back"


def test_live_arbiter_none_changes_nothing(tmp_path):
    rk = ProbsRanker(PROBS)
    lp = _arb_loop(tmp_path, arbiter=None, ranker=rk)
    assert lp.arbiter is None
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.key == "knight|middle|mid" and d.arbiter_reason is None and d.arbiter_subset is None
    plain = _loop(builder=FakeBuilder(warmup=0), generator=FakeGenerator(), ranker=FakeKeyRanker(),
                  executor=FakeExecutor())
    assert plain.step("f", 1.0, 0, 2.0).key == "giant|left|back"    # the pre-arbiter behaviour (ranker's pick)


def test_live_arbiter_is_not_used_with_the_tactics_ranker_or_a_single_candidate(tmp_path):
    lp = _arb_loop(tmp_path, arbiter=RuleArbiter(), ranker=TacticsRanker())
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.key == "giant|left|back" and d.arbiter_reason is None      # the engine's own pick, no arbiter
    one = _loop(builder=FakeBuilder(warmup=0), generator=FakeGenerator(keys=["zap|left|mid"]),
                ranker=ProbsRanker([1.0]), executor=FakeExecutor(), arbiter=RuleArbiter())
    d = one.step("f", 1.0, 0, 2.0)
    assert d.key == "zap|left|mid" and d.arbiter_reason is None


def test_cli_arbiter_flags(monkeypatch, tmp_path, capsys):
    import clashbot.live as live_mod

    args = build_parser().parse_args(_argv())
    assert (args.arbiter, args.arbiter_conf, args.arbiter_knn, args.arbiter_eta) == ("none", 0.55, 0.5, 4.0)
    args = build_parser().parse_args(_argv("--arbiter", "rules", "--arbiter-conf", "0.4", "--arbiter-knn", "0.7",
                                           "--arbiter-eta", "6"))
    assert (args.arbiter, args.arbiter_conf, args.arbiter_knn, args.arbiter_eta) == ("rules", 0.4, 0.7, 6.0)
    with pytest.raises(SystemExit):
        build_parser().parse_args(_argv("--arbiter", "bogus"))

    seen = {}

    class StubLoop:
        def __init__(self, *a, **k):
            seen.clear()
            seen.update(k)
            self.summary = _loop().summary()

        def run(self, **k):
            return self.summary

    kg = types.SimpleNamespace(counter_scores=lambda threat: {})
    monkeypatch.setattr(live_mod, "_build_source", lambda a: FakeSource())
    monkeypatch.setattr(live_mod, "_build_perception", lambda a, source: FakePerception())
    monkeypatch.setattr(live_mod, "_build_plan_text", lambda a, deck: None)
    monkeypatch.setattr(live_mod, "load_placement_prior", lambda: PRIOR)
    monkeypatch.setattr(live_mod, "StateIndex", lambda: None)
    monkeypatch.setattr(live_mod, "KnowledgeGraph", lambda: kg)
    monkeypatch.setattr(live_mod, "CandidateGenerator", lambda index, k, cards: FakeGenerator())
    monkeypatch.setattr(live_mod, "TacticalEngine", lambda cards, kg=None: None)
    monkeypatch.setattr(live_mod, "LiveLoop", StubLoop)
    base = ("--source", "video", "--video", "x.mp4", "--device", "cpu")
    assert main(_argv(*base, "--out", str(tmp_path / "a"))) == 0
    assert seen["arbiter"] is None                                                   # default: none
    assert main(_argv(*base, "--arbiter", "rules", "--arbiter-conf", "0.4", "--arbiter-knn", "0.7",
                      "--arbiter-eta", "6", "--out", str(tmp_path / "b"))) == 0
    assert isinstance(seen["arbiter"], RuleArbiter)
    assert seen["arbiter"].cfg == ArbiterConfig(conf=0.4, urgent_eta_s=6.0, knn_min=0.7)
    cfg = json.loads((tmp_path / "b" / "config.json").read_text())
    assert cfg["arbiter"] == "rules" and cfg["arbiter_conf"] == 0.4
    assert main(_argv(*base, "--ranker", "tactics", "--arbiter", "rules", "--out", str(tmp_path / "c"))) == 0
    assert seen["arbiter"] is None                                                   # the tactics ranker: no arbiter
    capsys.readouterr()
