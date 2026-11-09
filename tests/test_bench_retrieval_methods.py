"""MAC-AU part C: retrieval-method benchmark (toy fixtures; parsing, metrics, tie logic, leakage filter, KB / KG wiring)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.features import dim, featurize, vocab  # noqa: E402
from clashbot.retrieval import StateIndex  # noqa: E402
from tools.eval import bench_retrieval_methods as B  # noqa: E402

DATASET = ROOT / "dataset" / "v1"
KEYS = ["a|left|mid", "b|left|mid", "a|right|back", "b|right|back"]       # 2x2 grid: gold a|left|mid


# ---------------------------------------------------------------------------------------------- lines
def _row(i: int, game: str, gold: str) -> dict:
    return {"id": f"{game}#{i}.0", "game_id": game, "kind": "play",
            "questions": {"play": {"criteria": {k: "note" for k in KEYS}}}, "answers": {"play": gold}}


def test_build_and_verify_lines(tmp_path):
    ds = tmp_path / "v1"
    (ds / "d1" / "joint_tac_v2").mkdir(parents=True)
    rows = [_row(1, "g1", "b|left|mid"), _row(2, "g2", "a|right|back")]
    (ds / "d1" / "joint_tac_v2" / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lines = B.build_lines(ds)
    assert [l["line_id"] for l in lines] == ["g1#1.0", "g2#2.0"]
    assert [l["gold"] for l in lines] == [1, 2] and lines[0]["keys"] == KEYS
    assert B.verify_lines(lines, lines) == []
    other = [dict(lines[0], gold=0), lines[1]]
    assert B.verify_lines(other, lines) and B.verify_lines(lines[:1], lines)

    path = tmp_path / "au" / "lines.jsonl"
    got, info = B.load_lines(path, ds)                       # absent: built and written
    assert got == lines and path.exists() and info["identical_to_dataset_rows"]
    got2, _ = B.load_lines(path, ds)                         # present: read back and verified
    assert got2 == lines
    path.write_text(json.dumps(dict(lines[0], gold=3)) + "\n" + json.dumps(lines[1]) + "\n")
    with pytest.raises(ValueError):
        B.load_lines(path, ds)


@pytest.mark.skipif(not (DATASET / "d1" / "joint_tac_v2" / "test.jsonl").exists(), reason="dataset missing")
def test_real_lines_are_all_test_rows():
    lines = B.build_lines(DATASET)
    assert len(lines) == 766 and len({l["game_id"] for l in lines}) == 27
    assert len({l["line_id"] for l in lines}) == 766
    assert all(len(l["keys"]) == 4 and 0 <= l["gold"] < 4 for l in lines)


# ---------------------------------------------------------------------------------------------- metrics
def test_mrr_exact():
    keys = ["x|l|m", None, "a|left|mid", "a|left|mid"]
    assert B.mrr_exact(keys, "a|left|mid") == pytest.approx(1 / 3)
    assert B.mrr_exact(keys, "x|l|m") == 1.0
    assert B.mrr_exact(keys, "zzz") == 0.0
    assert B.mrr_exact(keys, "hold") == 0.0 and B.mrr_exact(keys, None) == 0.0


def test_option_scores_sums_hits_by_exact_key():
    got = B.option_scores(KEYS, ["a|left|mid", "b|right|back", "a|left|mid", None, "other"], [0.5, 0.2, 0.25, 9.0, 3.0])
    assert got == [0.75, 0.0, 0.0, 0.2]
    assert B.option_scores(KEYS, [], []) == [0.0] * 4


def test_decide_ties_are_expectations():
    nothing = B.decide([0, 0, 0, 0], KEYS, 0, "s")
    assert nothing["credit"] == 0.25 and nothing["supported"] is False and nothing["n_tied"] == 4
    assert nothing["card_credit"] == 0.5                       # two of the four options share the gold card
    unique = B.decide([0.9, 0.1, 0.0, 0.0], KEYS, 0, "s")
    assert (unique["credit"], unique["card_credit"], unique["supported"], unique["pred"]) == (1.0, 1.0, True, 0)
    wrong = B.decide([0.0, 0.9, 0.1, 0.0], KEYS, 0, "s")
    assert (wrong["credit"], wrong["card_credit"], wrong["pred"]) == (0.0, 0.0, 1)
    pair = B.decide([0.4, 0.0, 0.4, 0.0], KEYS, 0, "s")        # same card, two placements: tie
    assert pair["credit"] == 0.5 and pair["card_credit"] == 1.0 and pair["pred"] in (0, 2)
    near = B.decide([0.5, 0.5 + 1e-12, 0, 0], KEYS, 0, "s")    # equal to 9 decimals
    assert near["credit"] == 0.5
    assert B.decide([0, 0, 0, 0], KEYS, 0, "s")["pred"] == B.decide([0, 0, 0, 0], KEYS, 0, "s")["pred"]    # seeded
    draws = {B.decide([0, 0, 0, 0], KEYS, 0, f"s{i}")["pred"] for i in range(40)}
    assert draws == {0, 1, 2, 3}


def test_mcnemar_exact():
    assert B.mcnemar_exact([True] * 5, [False] * 5) == {"a_only": 5, "b_only": 0, "p": pytest.approx(2 / 32)}
    assert B.mcnemar_exact([True, False], [False, True])["p"] == 1.0
    assert B.mcnemar_exact([True, False], [True, False])["p"] == 1.0
    out = B.mcnemar_exact([True] * 9 + [False] * 3, [False] * 9 + [True] * 3)
    assert (out["a_only"], out["b_only"]) == (9, 3) and out["p"] == pytest.approx(598 / 4096)
    from scipy.stats import binomtest
    assert out["p"] == pytest.approx(binomtest(3, 12, 0.5).pvalue)


def test_latency_row():
    row = B.latency_row([1.0, 2.0, 3.0, 4.0, 100.0])
    assert row["ms_p50"] == 3.0 and row["ms_mean"] == pytest.approx(22.0) and row["n"] == 5 and row["ms_p95"] > 4
    assert B.latency_row([])["ms_p50"] is None


# ---------------------------------------------------------------------------------------------- leakage
class FakeIndex:
    split = np.array(["train", "train", "val", "test"], dtype=object)
    game_id = np.array(["a", "b", "c", "t1"], dtype=object)
    id = np.array(["a#1", "b#1", "c#1", "t1#1"], dtype=object)


def test_check_leakage():
    B.check_leakage(FakeIndex, [0, 1], ["train"], {"t1"})
    with pytest.raises(B.LeakageError):
        B.check_leakage(FakeIndex, [0, 2], ["train"], {"t1"})          # a val row
    with pytest.raises(B.LeakageError):
        B.check_leakage(FakeIndex, [3], ["train", "test"], {"t1"})     # an evaluated game, even if its split were allowed
    B.check_leakage(FakeIndex, [0, 2], ["train", "val"], {"t1"})       # train+val allowed when stated


def test_drop_nontrain_decks():
    games = [{"split": "train", "own_deck_key": "d-train"}, {"split": "test", "own_deck_key": "d-test"},
             {"split": "val", "own_deck_key": "d-train"}, {"split": "test", "own_deck_key": "d-both"},
             {"split": "train", "own_deck_key": "d-both"}]
    triples = [("card:x", "IN_DECK", "deck:d-test"), ("card:x", "IN_DECK", "deck:d-train"),
               ("card:x", "IN_DECK", "deck:d-both"), ("deck:d-test", "HAS_ARCHETYPE", "archetype:cycle"),
               ("deck:d-other", "HAS_ARCHETYPE", "archetype:cycle"), ("card:x", "COUNTERS", "card:y")]
    kept = B.drop_nontrain_decks(triples, games)
    assert kept == [("card:x", "IN_DECK", "deck:d-train"), ("card:x", "IN_DECK", "deck:d-both"),
                    ("deck:d-other", "HAS_ARCHETYPE", "archetype:cycle"), ("card:x", "COUNTERS", "card:y")]


# ---------------------------------------------------------------------------------------------- KB methods on a toy index
def _toy_state(rng: random.Random) -> dict:
    cards = vocab()
    hand = rng.sample(cards, 4)
    return {"phase": "double_elixir", "match_seconds": rng.uniform(0, 280),
            "own": {"elixir": rng.uniform(0, 10), "hand": hand, "deck": rng.sample(cards, 8),
                    "towers": {"king": None, "left": 3000, "right": 3000}, "recent_plays": []},
            "opponent": {"elixir_estimate": 5.0, "towers": {"king": None, "left": 3000, "right": 3000},
                         "deck_known": [], "recent_plays": []},
            "units": [], "threats": []}


@pytest.fixture(scope="module")
def toy():
    rng = random.Random(0)
    n = 40
    splits = ["train"] * 16 + ["val"] * 8 + ["test"] * 16
    states = [_toy_state(rng) for _ in range(n)]
    idx = StateIndex.__new__(StateIndex)
    idx.n = n
    idx.id = np.array([f"g{i % 8}#{i}" for i in range(n)], dtype=object)
    idx.game_id = np.array([f"g{i % 8}" if splits[i] == "train" else f"x{i % 4}" for i in range(n)], dtype=object)
    idx.split = np.array(splits, dtype=object)
    idx.own_deck_key = np.array([f"deck{i % 3}" for i in range(n)], dtype=object)
    idx.phase = np.array(["double_elixir"] * n, dtype=object)
    idx.action_key = np.array([f"c{i % 5}|left|mid" for i in range(n)], dtype=object)
    idx._has_action = np.ones(n, dtype=bool)
    idx.M = np.stack([featurize(s, 10.0) for s in states]).astype(np.float32)
    idx.use_hnsw, idx._hnsw, idx.db_path = False, None, None
    import hnswlib
    h = hnswlib.Index(space="ip", dim=dim())
    h.init_index(max_elements=n, ef_construction=100, M=16, random_seed=0)
    h.add_items(idx.M, np.arange(n))
    E = np.random.default_rng(1).normal(size=(n, 16)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return idx, h, E, states


def test_hnsw_filtered_respects_the_pool(toy):
    idx, h, _E, states = toy
    allowed = idx.mask(splits=["train"])
    got = B.hnsw_filtered(h, featurize(states[20], 10.0), allowed, 10, ef=40)
    assert len(got) == 10 and all(allowed[r] for r, _ in got)
    exact = [r for r, _ in idx.knn(featurize(states[20], 10.0), 10, splits=["train"])]
    assert set(r for r, _ in got) == set(exact)                 # a 40-point graph at ef 40 is exact
    assert got[0][1] == pytest.approx(float(idx.M[exact[0]] @ featurize(states[20], 10.0)), abs=1e-5)


def test_kb_methods_stay_in_the_train_pool(toy):
    idx, h, E, states = toy
    fns = B.kb_methods(idx, h, E)
    assert list(fns) == ["dense", "bm25", "rrf_dense_bm25", "bge", "rrf_all3", "hnsw_ef10", "hnsw_ef40", "hnsw_ef160",
                         "deck_dense"]
    q = {"state": states[30], "t": 10.0, "context_text": "x", "emb": E[30], "deck_key": "deck1"}
    forbidden = {str(g) for g in idx.game_id[idx.split != "train"]}
    for m in ("dense", "bge", "hnsw_ef10", "hnsw_ef40", "hnsw_ef160", "deck_dense"):
        hits = fns[m](q)
        assert 0 < len(hits) <= B.TOP, m
        B.check_leakage(idx, [r for r, _ in hits], ["train"], forbidden)
    assert all(idx.own_deck_key[r] == "deck1" for r, _ in fns["deck_dense"](q))
    assert fns["deck_dense"](dict(q, deck_key="unseen-deck")) == [] and fns["deck_dense"](dict(q, deck_key=None)) == []
    dense = [r for r, _ in fns["dense"](q)]
    assert dense == [r for r, _ in fns["hnsw_ef160"](q)]           # tiny index: HNSW at high ef is exact
    assert B.recall_overlap([r for r, _ in fns["hnsw_ef10"](q)], dense, B.TOP) <= 1.0
    # bge: precomputed embedding vs a per-query encode function give the same list
    enc = {"x": E[30]}
    fns2 = B.kb_methods(idx, h, E, embed=lambda text: enc[text])
    assert fns2["bge"](q) == fns["bge"](q)


def test_kb_line_record_metrics_and_leak_guard(toy):
    idx, _h, _E, _states = toy
    gold_row = int(np.flatnonzero(idx.split == "train")[0])
    gold_key = str(idx.action_key[gold_row])
    q = {"line_id": "L", "game_id": "t", "gold": 0, "keys": [gold_key, "zz|left|mid", "zz|right|back", "yy|left|mid"]}
    hits = [(gold_row, 0.8), (int(np.flatnonzero(idx.split == "train")[5]), 0.5)]
    rec = B.kb_line_record("dense", q, hits, idx, {"t"}, None)
    assert rec["hit@10"] == 1.0 and rec["hit@1"] == 1.0 and rec["mrr"] == 1.0 and rec["card_hit@10"] == 1.0
    assert rec["credit"] == 1.0 and rec["supported"] and rec["n_hits"] == 2 and "recall@20" not in rec
    rec2 = B.kb_line_record("hnsw_ef10", q, hits, idx, {"t"}, [hits[0][0]])
    assert rec2["recall@20"] == 1.0
    empty = B.kb_line_record("deck_dense", q, [], idx, {"t"}, None)
    assert empty["credit"] == 0.25 and not empty["supported"] and empty["n_hits"] == 0 and empty["hit@10"] == 0.0
    val_row = int(np.flatnonzero(idx.split == "val")[0])
    with pytest.raises(B.LeakageError):
        B.kb_line_record("dense", q, [(val_row, 1.0)], idx, {"t"}, None)


# ---------------------------------------------------------------------------------------------- KG
def test_line_threats_reads_recent_plays_and_units_heading_to_our_towers():
    state = {"opponent": {"recent_plays": [{"card": "hog-rider", "t": 98.0, "tile": [3, 10]},
                                           {"card": "old", "t": 50.0, "tile": [1, 1]},
                                           {"card": "hog-rider", "t": 99.0, "tile": [3, 9]}]},
             "units": [{"side": "enemy", "card": "balloon", "tile": [4, 12], "eta_tower": {"tower": "own_left", "s": 4.0}},
                       {"side": "enemy", "card": "golem", "tile": [4, 12], "eta_tower": {"tower": "own_left", "s": 30.0}},
                       {"side": "ally", "card": "knight", "tile": [4, 12], "eta_tower": {"tower": "enemy_left", "s": 2.0}},
                       {"side": "enemy", "card": "giant", "tile": [4, 12], "eta_tower": {"tower": "own_right", "s": 8.0}}]}
    assert B.line_threats(state, 100.0) == ["hog-rider", "balloon", "giant"]
    assert B.line_threats({"opponent": {"recent_plays": []}, "units": []}, 5.0) == []


def test_shift_scores_keeps_order_and_is_positive():
    raw = {"a": -3.0, "b": -1.0, "c": -2.0}
    out = B.shift_scores(raw)
    assert out == {"a": 1.0, "b": 3.0, "c": 2.0}
    assert sorted(raw, key=raw.get) == sorted(out, key=out.get)
    assert B.shift_scores({}) == {}


def test_kge_scorer_maps_deck_cards_and_unknown_threats():
    class FakeScorer:
        ent = {"card:t": 0}

        def score_heads(self, tails):
            return np.array([[-2.0, -1.0, -5.0]])

    k = B.KGEScorer(FakeScorer(), ["card:a", "card:b", "card:c"])
    got = k("t", ["a", "c", "zz", "a"])
    assert got == {"a": 4.0, "c": 1.0}                  # b not in the deck, zz unknown, duplicates collapse
    assert k("unknown-threat", ["a"]) == {}


def test_kg_line_record_sums_over_threats_and_ties_without_a_threat():
    q = {"line_id": "L", "game_id": "g", "gold": 0, "keys": KEYS, "own_deck": ["a", "b", "c"]}
    scores = {"t1": {"a": 2.0, "b": 1.0}, "t2": {"a": 0.5, "c": 9.0}}
    fn = lambda threat, deck: scores[threat]                          # noqa: E731
    rec = B.kg_line_record("m", fn, q, ["t1", "t2"])
    assert rec["n_threats"] == 2 and rec["supported"]
    assert rec["credit"] == 0.5 and rec["card_credit"] == 1.0         # card a wins, both a-options tie
    none = B.kg_line_record("m", fn, q, [])
    assert none["credit"] == 0.25 and none["supported"] is False and none["n_tied"] == 4
    rand = B.kg_line_record("random", lambda threat, deck: {}, q, ["t1"])
    assert rand["credit"] == 0.25 and not rand["supported"]
    seen = {}
    B.kg_line_record("m", lambda threat, deck: seen.setdefault("deck", list(deck)) and {}, q, ["t1"])
    assert seen["deck"] == ["a", "b", "c"]                            # own deck plus option cards, no duplicates


# ---------------------------------------------------------------------------------------------- aggregation
def test_decision_summary_cluster_bootstrap_and_mcnemar_vs_d1():
    from tools.eval.eval_answer_rules import GameBootstrap
    games = ["g1", "g1", "g2", "g2", "g3", "g3"]
    rows = [{"line_id": f"L{i}", "game_id": g, "gold": 0, "pred": 0 if i % 2 == 0 else 1,
             "credit": 1.0 if i % 2 == 0 else 0.25, "card_credit": 1.0, "supported": i < 4}
            for i, g in enumerate(games)]
    d1 = {f"L{i}": {"pred": 0, "latency_ms": 1.0} for i in range(6)}
    boot = GameBootstrap(games, 500, 0)
    out = B.decision_summary(rows, boot, d1, {})
    assert out["accuracy"]["mean"] == pytest.approx((3 * 1.0 + 3 * 0.25) / 6)
    assert out["accuracy"]["lo"] <= out["accuracy"]["mean"] <= out["accuracy"]["hi"]
    assert out["coverage"]["mean"] == pytest.approx(4 / 6)
    assert out["accuracy_supported"]["n"] == 4 and out["accuracy_unsupported"]["n"] == 2
    assert out["mcnemar_vs_d1"] == {"a_only": 0, "b_only": 3, "p": pytest.approx(0.25)}
    assert out["delta_vs_d1"]["mean"] == pytest.approx(out["accuracy"]["mean"] - 1.0)       # d1 is right on all six
    assert "mcnemar_vs_d1" not in B.decision_summary(rows, boot, None, {})


def test_d1_correct_prefers_part_b_preds(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "PREDS_DIR", tmp_path)
    monkeypatch.setattr(B, "D1_EVAL", tmp_path / "missing.json")
    assert B.d1_correct(["L0"], {"L0": 0}, {"L0": KEYS}) == (None, "missing")
    (tmp_path / f"{B.D1_ID}.jsonl").write_text(json.dumps({"line_id": "L0", "pred": 2, "latency_ms": 12.5}) + "\n")
    got, src = B.d1_correct(["L0"], {"L0": 0}, {"L0": KEYS})
    assert got == {"L0": {"pred": 2, "latency_ms": 12.5}} and "Part B" in src
    (tmp_path / "ev.json").write_text(json.dumps({"per_example": [{"id": "L1", "split": "test", "pred": KEYS[3]}]}))
    monkeypatch.setattr(B, "D1_EVAL", tmp_path / "ev.json")
    got, src = B.d1_correct(["L1"], {"L1": 0}, {"L1": KEYS})
    assert got == {"L1": {"pred": 3, "latency_ms": None}} and "provisional" in src
