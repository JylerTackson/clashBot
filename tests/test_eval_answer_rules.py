"""MAC-AR R1: held-out evaluation of the mined answer rules (toy fixtures with known answers; a tiny real KG2 store)."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.kg2 import KG2, SCHEMA  # noqa: E402
from tools.eval import eval_answer_rules as E  # noqa: E402

DECK = ["a", "b", "c", "d", "e", "f", "g", "h"]
REAL = ROOT / "dataset" / "v1" / "kg2" / "answer_events.jsonl"


def _edge(conn, src, rel, dst, weight=1.0, support=0, source="x", props=None):
    conn.execute("INSERT INTO edges VALUES (?,?,?,?,?,?,?)",
                 (f"card:{src}", rel, f"card:{dst}", weight, support, source, json.dumps(props or {})))


@pytest.fixture()
def toy_kg(tmp_path):
    """Threat T. a: pro rule (support 10, held 0.8). y (not in deck) is a pro answer with support 8, SIMILAR_ROLE y~c:
    c gets role transfer. b: wiki counter (2 bullets). d: simulation STOPS edge. e..h: no evidence."""
    path = tmp_path / "kg2.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    _edge(conn, "T", "ANSWERED_BY", "a", support=10, props={"support": 10, "held_rate": 0.8})
    _edge(conn, "T", "ANSWERED_BY", "y", support=8, props={"support": 8, "held_rate": 0.5})
    _edge(conn, "y", "SIMILAR_ROLE", "c", weight=0.9)
    _edge(conn, "b", "COUNTERS", "T", props={"count": 2})
    _edge(conn, "d", "STOPS", "T", weight=0.6)
    conn.commit()
    conn.close()
    kg = KG2(path, deck_cooc_path=tmp_path / "none.json")
    yield kg
    kg.close()


# ---------------------------------------------------------------------------------------------- ranking metrics
def test_tie_metrics_strict_ranking():
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    top = E.tie_metrics(scores, DECK[:4], "a")
    assert (top["hit1"], top["hit3"], top["rr"], top["rank_best"]) == (1.0, 1.0, 1.0, 1)
    second = E.tie_metrics(scores, DECK[:4], "b")
    assert (second["hit1"], second["hit3"], second["rr"]) == (0.0, 1.0, 0.5)
    last = E.tie_metrics(scores, DECK[:4], "d")           # d has no score: 4th of 4
    assert (last["hit1"], last["hit3"], last["rr"], last["answer_scored"]) == (0.0, 0.0, 0.25, 0.0)
    assert last["covered"] == 1.0


def test_tie_metrics_ties_are_expectations_and_no_evidence_is_random():
    nothing = E.tie_metrics({}, DECK[:4], "c")
    assert nothing["hit1"] == pytest.approx(0.25)
    assert nothing["hit3"] == pytest.approx(0.75)
    assert nothing["rr"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)
    assert nothing["covered"] == 0.0
    tied = E.tie_metrics({"a": 1.0, "b": 1.0}, DECK[:4], "a")   # a and b share positions 1-2
    assert tied["hit1"] == pytest.approx(0.5) and tied["hit3"] == 1.0 and tied["rr"] == pytest.approx(0.75)
    behind = E.tie_metrics({"a": 2.0, "b": 1.0, "c": 1.0}, DECK[:4], "c")    # b, c share positions 2-3
    assert behind["hit1"] == 0.0 and behind["hit3"] == 1.0 and behind["rr"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    with pytest.raises(ValueError):
        E.tie_metrics({}, DECK[:4], "zzz")


def test_random_expectation_over_decks_of_8():
    m = E.tie_metrics({}, DECK, "h")
    assert m["hit1"] == pytest.approx(1 / 8) and m["hit3"] == pytest.approx(3 / 8)
    assert m["rr"] == pytest.approx(sum(1 / i for i in range(1, 9)) / 8)


def test_game_bootstrap_clusters_and_pairs():
    games = ["g1", "g1", "g2", "g2", "g3"]
    boot = E.GameBootstrap(games, boot=2000, seed=1)
    same = boot.ci([1, 1, 1, 1, 1])
    assert (same["mean"], same["lo"], same["hi"]) == (1.0, 1.0, 1.0)
    ci = boot.ci([1, 1, 0, 0, 0])
    assert ci["mean"] == pytest.approx(0.4) and ci["lo"] < 0.4 < ci["hi"] and ci["n"] == 5
    assert boot.ci([1, 1, 0, 0, 0]) == ci                              # deterministic
    masked = boot.ci([1, 1, 0, 0, 0], mask=[1, 1, 0, 0, 0])
    assert masked["mean"] == 1.0 and masked["n"] == 2
    d = boot.diff([1, 1, 0, 0, 1], [1, 1, 0, 0, 1])
    assert d["mean"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0 and d["p"] == 1.0
    up = boot.diff([1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
    assert up["mean"] == 1.0 and up["p"] == 0.0
    assert np.isnan(boot.ci([1, 1, 1, 1, 1], mask=[0, 0, 0, 0, 0])["mean"])


# ---------------------------------------------------------------------------------------------- source ablation wiring
def test_each_source_alone_scores_only_its_own_card(toy_kg):
    s = E.source_scores(toy_kg, "T", DECK)
    nz = lambda name: {c for c, v in s[name].items() if v > 0}       # noqa: E731
    assert nz("only:pro") == {"a"}
    assert nz("only:role_transfer") == {"c"}
    assert nz("only:wiki") == {"b"}
    assert nz("only:sim") == {"d"}
    assert nz("full") == {"a", "b", "c", "d"}
    assert max(s["full"], key=s["full"].get) == "a"
    # one source alone is renormalised over itself: its single card scores that source's own value
    assert s["only:pro"]["a"] == pytest.approx(0.8 * 10 / 15, abs=1e-3)
    assert s["only:wiki"]["b"] == pytest.approx(0.75)
    assert s["only:sim"]["d"] == pytest.approx(0.6)
    assert s["only:role_transfer"]["c"] == pytest.approx(0.9 * (0.5 * 8 / 13) * 0.7, abs=1e-3)


def test_leave_one_out_removes_exactly_that_source(toy_kg):
    s = E.source_scores(toy_kg, "T", DECK)
    assert s["without:pro"].get("a", 0.0) == 0.0 and {"b", "c", "d"} <= set(s["without:pro"])
    assert s["without:wiki"].get("b", 0.0) == 0.0 and s["without:wiki"]["a"] > 0
    assert s["without:sim"].get("d", 0.0) == 0.0 and s["without:sim"]["a"] > 0
    assert s["without:role_transfer"].get("c", 0.0) == 0.0 and s["without:role_transfer"]["d"] > 0
    # dropping the pro source renormalises over wiki / role transfer / sim: sim (weight .15, value .6) now leads
    assert max(s["without:pro"], key=s["without:pro"].get) == "d"


def test_full_equals_answers_for_and_check_is_tight(toy_kg):
    answers = toy_kg.answers_for("T", DECK, k=8)
    assert E.score_gap(toy_kg, answers) < 5e-4
    s = E.source_scores(toy_kg, "T", DECK, answers)
    for a in answers:
        assert s["full"][a.card] == pytest.approx(a.score, abs=5e-4)
    assert E.combine_sources([], {"pro": 0.45}) == {}


def test_set_rules_swaps_only_the_pro_relation(toy_kg):
    saved = toy_kg._answered
    E.set_rules(toy_kg, [{"threat": "T", "answer": "e", "support": 4, "held_rate": 1.0}])
    s = E.source_scores(toy_kg, "T", DECK)
    assert {c for c, v in s["only:pro"].items() if v > 0} == {"e"}
    assert {c for c, v in s["only:wiki"].items() if v > 0} == {"b"}      # the other relations are untouched
    assert s["only:role_transfer"] == {}                                   # y is no longer a pro answer
    toy_kg._answered = saved
    assert {c for c, v in E.source_scores(toy_kg, "T", DECK)["only:pro"].items() if v > 0} == {"a"}


def test_eval_rows_toy_ranks_and_baselines(toy_kg):
    decks = {"g1": DECK, "g2": DECK, "g3": DECK[:4]}
    events = [{"game_id": "g1", "split": "val", "threat": "T", "answer": "a", "held": True, "threat_source": "play"},
              {"game_id": "g2", "split": "val", "threat": "T", "answer": "d", "held": False, "threat_source": "unit"},
              {"game_id": "g3", "split": "test", "threat": "T", "answer": "zz", "held": None},        # not in the deck
              {"game_id": "g2", "split": "val", "threat": "Q", "answer": "e", "held": True, "threat_source": "play"}]

    class Wiki:                                   # KnowledgeGraph stand-in
        inc = {("card:T", "COUNTERS"): [("card:b", {}), ("card:b", {}), ("card:c", {})]}

        def counter_scores(self, threat):
            return {"b": 2.0, "c": 1.0, "a": 0.25}

    pop = E.Counter({"d": 5, "a": 1})
    res = E.eval_rows(toy_kg, events, decks, pop, Wiki())
    assert res["skipped"] == {"answer_not_in_deck": 1} and len(res["rows"]) == 3
    r_a, r_d, r_q = res["rows"]
    assert r_a["m"]["full"]["hit1"] == 1.0 and r_d["m"]["full"]["rr"] == pytest.approx(1 / 2)   # a, then d, c, b
    assert r_a["pro_support_max"] == 10 and r_q["pro_support_max"] == 0
    assert r_a["m"]["wiki_counters"]["rank_best"] == 3                     # b 2, c 1, then a, d, ... tied at 0
    assert r_d["m"]["popularity"]["hit1"] == 1.0 and r_a["m"]["popularity"]["rank_best"] == 2
    assert r_a["m"]["random"]["hit1"] == pytest.approx(1 / 8)
    assert r_q["m"]["full"]["covered"] == 0.0 and r_q["m"]["full"]["hit1"] == pytest.approx(1 / 8)  # no evidence = random
    assert res["max_score_gap"] < 5e-4


def test_held_by_rank_and_calibration_toy():
    def row(game, held, score, rank):
        return {"game_id": game, "held": held, "m": {"full": {"answer_scored": float(score > 0), "rank_best": rank,
                                                              "score": score}}}
    rows = [row("g1", True, 0.5, 1), row("g1", False, 0.5, 1), row("g2", True, 0.3, 2), row("g2", None, 0.3, 2),
            row("g3", True, 0.0, 5), row("g3", False, 0.0, 5), row("g3", True, 0.1, 6)]
    boot = E.GameBootstrap([r["game_id"] for r in rows], boot=500, seed=0)
    h = E.held_by_rank(rows, boot)
    assert h["rank 1"]["events"] == 2 and h["rank 1"]["held"]["mean"] == 0.5
    assert h["rank 2"]["events"] == 2 and h["rank 2"]["held"]["n"] == 1 and h["rank 2"]["held"]["mean"] == 1.0
    assert h["rank 4+"]["events"] == 1 and h["no evidence"]["events"] == 2 and h["no evidence"]["held"]["mean"] == 0.5
    cal = E.score_calibration(rows, boot)
    assert cal["bins"]["0 (none)"]["events"] == 2 and cal["n_pos"] == 4 and cal["n_neg"] == 2
    # held events scored .5 .3 0 .1 (the fourth event has no flag), not held .5 0: AUROC by hand
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in (0.5, 0.3, 0.0, 0.1) for n in (0.5, 0.0))
    assert cal["auroc_held"] == pytest.approx(wins / 8)


def test_counters_only_ignores_answered_by():
    class Wiki:
        inc = {("card:T", "COUNTERS"): [("card:b", {}), ("card:b", {}), ("card:c", {})],
               ("card:T", "ANSWERED_BY"): [("card:z", {"support": 9})]}
    assert E.counters_only(Wiki(), "T") == {"b": 2.0, "c": 1.0}
    assert E.counters_only(Wiki(), "nothing") == {}


# ---------------------------------------------------------------------------------------------- rules from train only
def _ev(split, threat="X", answer="Y", game="g", held=True):
    return {"game_id": f"{split}-{game}", "split": split, "threat": threat, "answer": answer, "held": held,
            "offset": [0.0, 0.0], "offset_ok": True, "trade": 1.0, "dt_s": 1.0, "eta_at_answer": 2.0,
            "dist_at_answer": 5.0}


def test_rules_are_built_from_train_events_only():
    events = [_ev("train")] * 3 + [_ev("val")] * 5 + [_ev("test")] * 4
    rules = E.aggregate_rules(events, 1)
    assert [(r["threat"], r["answer"], r["support"]) for r in rules] == [("X", "Y", 3)]
    shipped = E.aggregate_rules([e for e in events if e["split"] == "train"], 3)
    ok = E.leak_check(events, shipped)
    assert ok["rules_all_events_equal_train_only"] and ok["rules_equal_shipped_answers_jsonl"]
    assert ok["support_within_train_events"] and ok["events"] == {"train": 3, "val": 5, "test": 4}
    assert E.leak_check(events, [])["rules_equal_shipped_answers_jsonl"] is False
    assert E.popularity_counts(events) == {"Y": 3}                          # popularity ignores val / test too


def test_remine_refuses_to_write_over_dataset_or_db(tmp_path):
    from clashbot.config import DATASET_DIR, DB_PATH
    with pytest.raises(ValueError):
        E.remine(4.0, tmp_path, DATASET_DIR / "kg2")
    with pytest.raises(ValueError):
        E.remine(4.0, tmp_path, DB_PATH.parent / "ar_tmp")


def test_remine_cache_is_read_back(tmp_path):
    out = tmp_path / "ar_w4"
    out.mkdir()
    (out / "answer_events.jsonl").write_text(json.dumps(_ev("train")) + "\n")
    assert E.remine(4.0, tmp_path, out) == [_ev("train")]


@pytest.mark.skipif(not REAL, reason="dataset/v1/kg2 not built")
def test_real_shipped_rules_come_from_train_events_only():
    ds = ROOT / "dataset" / "v1"
    events = E.read_jsonl(ds / "kg2" / "answer_events.jsonl")
    shipped = E.read_jsonl(ds / "kg2" / "answers.jsonl")
    lc = E.leak_check(events, shipped)
    assert lc["rules_all_events_equal_train_only"] and lc["rules_equal_shipped_answers_jsonl"]
    assert lc["support_within_train_events"]
    games = {g["game_id"]: g["split"] for g in E.read_jsonl(ds / "games.jsonl")}
    assert all(games[e["game_id"]] == e["split"] for e in events)           # the split field is the game's split
