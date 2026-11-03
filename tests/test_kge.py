"""MAC-R2 tests: realistic rank, COUNTERS split, popularity baseline, tiny PyKEEN runs, real-data split counts."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval.kge_link_prediction import (HITS_KS, MODELS, PopularityScorer, PykeenScorer, RandomScorer,  # noqa: E402
                                            asymmetry, build_splits, evaluate, load_triples, make_factory,
                                            metrics_from_ranks, novel_predictions, rank_and_hits, rank_triples,
                                            realistic_hits, realistic_rank, split_counters, train_model)

DATASET = ROOT / "dataset" / "v1"
needs_kg = pytest.mark.skipif(not (DATASET / "kg" / "edges.jsonl").exists(), reason="dataset/v1/kg/edges.jsonl missing")


def synthetic_triples(n_cards: int = 12, n_counters: int = 40, n_synergy: int = 20, seed: int = 0):
    """Random card-card COUNTERS / SYNERGY triples (plus duplicates and ANSWERED_BY edges to exercise the filters)."""
    rng = random.Random(seed)
    cards = [f"card:c{i:02d}" for i in range(n_cards)]
    pairs = [(a, b) for a in cards for b in cards if a != b]
    rng.shuffle(pairs)
    counters = [(a, "COUNTERS", b) for a, b in pairs[:n_counters]]
    synergy = [(a, "SYNERGY", b) for a, b in pairs[n_counters:n_counters + n_synergy]]
    answered = [(a, "ANSWERED_BY", b) for a, b in pairs[-6:]]
    return counters + synergy + answered + counters[:5]       # last 5 are duplicates


def test_realistic_rank():
    s = np.array([0.9, 0.5, 0.5, 0.1])
    assert realistic_rank(s, 1, []) == 2.5
    assert realistic_rank(s, 1, [0]) == 1.5
    assert realistic_rank(np.zeros(4), 2, []) == 2.5
    assert realistic_rank(s, 0, []) == 1.0
    assert realistic_rank(s, 3, [1, 2]) == 2.0
    assert realistic_rank(s, 1, [1]) == 2.5        # the true index is never filtered


def test_realistic_hits():
    tied = np.zeros(4)                                       # 4 all-tied candidates: the true one is in the top k w.p. k/4
    assert realistic_hits(tied, 2, [], 1) == 0.25
    assert realistic_hits(tied, 2, [], 3) == 0.75
    assert realistic_hits(tied, 2, [], 4) == 1.0
    s = np.array([0.9, 0.5, 0.5, 0.1])
    assert realistic_hits(s, 1, [], 1) == 0.0                # one better item, tied with one other
    assert realistic_hits(s, 1, [], 2) == 0.5
    assert realistic_hits(s, 1, [0], 1) == 0.5               # filtering the better item makes it a 2-way tie for rank 1
    untied = np.array([0.9, 0.5, 0.3, 0.1])                  # no ties: plain 0/1 hits (item i has rank i + 1)
    for true in range(4):
        for k in (1, 2, 3, 10):
            assert realistic_hits(untied, true, [], k) == float(true + 1 <= k)
    assert realistic_hits(s, 3, [1, 2], 1) == 0.0 and realistic_hits(s, 3, [1, 2], 2) == 1.0
    big = np.zeros(126)                                      # a full tie over 126 candidates: Hits@10 = 10/126, not 0
    assert realistic_hits(big, 5, [], 10) == pytest.approx(10 / 126)
    assert realistic_rank(big, 5, []) > 10                   # ... although the expected rank (63.5) is beyond 10
    assert realistic_hits(s, 1, [1], 2) == 0.5               # the true index is never filtered


def test_metrics_from_ranks():
    ranks = np.array([[1.0, 2.5], [4.0, 11.0]])
    hits = np.array([[[1.0, 1.0, 1.0], [0.5, 1.0, 1.0]], [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]])   # (n, 2, len(HITS_KS))
    assert HITS_KS == (1, 3, 10)
    m = metrics_from_ranks(ranks, hits)
    assert m["n"] == 4
    assert m["hits@1"] == pytest.approx(1.5 / 4) and m["hits@3"] == pytest.approx(2.5 / 4) and m["hits@10"] == 0.75
    assert m["mrr"] == pytest.approx((1 + 0.4 + 0.25 + 1 / 11) / 4)
    assert math.isnan(metrics_from_ranks(np.zeros((0, 2)), np.zeros((0, 2, 3)))["hits@1"])


def test_split_deterministic_and_disjoint():
    triples = synthetic_triples()
    a, b = split_counters([t for t in triples if t[1] == "COUNTERS"]), split_counters(
        list(reversed([t for t in triples if t[1] == "COUNTERS"])))
    assert a == b
    train, val, test = a
    assert len(train) + len(val) + len(test) == 40            # 5 duplicates removed
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    sp1, sp2 = build_splits(triples, "full"), build_splits(list(reversed(triples)), "full")
    assert (sp1.train, sp1.val_c, sp1.test_c, sp1.candidates) == (sp2.train, sp2.val_c, sp2.test_c, sp2.candidates)
    assert len(sp1.train) == len(set(sp1.train))
    assert not (set(sp1.val_c) | set(sp1.test_c)) & set(sp1.train)     # held-out COUNTERS never leak into training
    assert len(sp1.train_c) + len(sp1.val_c) + len(sp1.test_c) + sp1.dropped_unseen == 40


def test_no_inverse_has_no_answered_by():
    triples = synthetic_triples()
    full, noinv = build_splits(triples, "full"), build_splits(triples, "no_inverse")
    assert any(t[1] == "ANSWERED_BY" for t in full.train)
    assert not any(t[1] == "ANSWERED_BY" for t in noinv.train)
    assert full.train_c == noinv.train_c                    # the COUNTERS split is the same in both settings
    with pytest.raises(ValueError):
        build_splits(triples, "bogus")


def test_unseen_entities_dropped():
    triples = [("card:a", "COUNTERS", "card:b"), ("card:b", "COUNTERS", "card:c"), ("card:c", "COUNTERS", "card:d"),
               ("card:a", "SYNERGY", "card:b"), ("card:x", "COUNTERS", "card:y")]
    sp = build_splits(triples, "full")
    kept = set(sp.train) | set(sp.val_c) | set(sp.test_c)
    assert len(sp.train_c) + len(sp.val_c) + len(sp.test_c) + sp.dropped_unseen == 4
    seen = {x for h, _, t in sp.train for x in (h, t)}
    assert all(h in seen and t in seen for h, _, t in sp.val_c + sp.test_c)
    assert kept and all(c.startswith("card:") for c in sp.candidates)


def test_popularity_star_graph():
    hub = "card:hub"
    leaves = [f"card:x{i:02d}" for i in range(15)]
    triples = [(x, "COUNTERS", hub) for x in leaves] + [(x, "SYNERGY", hub) for x in leaves]   # keeps leaves seen
    sp = build_splits(triples, "full")
    assert sp.test_c, "star graph should leave at least one held-out triple"
    pop = PopularityScorer(sp.train_c, sp.candidates)
    cidx = {c: i for i, c in enumerate(sp.candidates)}
    assert int(np.argmax(pop.score_tails([leaves[0]])[0])) == cidx[hub]
    ranks, hits = rank_and_hits(pop, sp.test_c, sp.candidates, sp.all_counters)
    assert np.all(ranks[:, 0] == 1.0) and np.all(hits[:, 0, :] == 1.0)    # tail prediction: the hub is ranked first
    assert np.array_equal(ranks, rank_triples(pop, sp.test_c, sp.candidates, sp.all_counters))
    rnd = rank_triples(RandomScorer(len(sp.candidates)), sp.test_c, sp.candidates, sp.all_counters)
    assert np.all(rnd[:, 0] > 1.0)


def test_random_baseline_hits_are_expectations():
    """A full tie gives Hits@k = k / #filtered candidates (> 0 for k = 10), not Hits@k of the expected rank."""
    cands = [f"card:c{i:02d}" for i in range(30)]
    known = [(cands[0], "COUNTERS", cands[1]), (cands[0], "COUNTERS", cands[2])]
    ranks, hits = rank_and_hits(RandomScorer(len(cands)), [known[0]], cands, known)
    assert ranks[0, 0] == pytest.approx((29 + 1) / 2)       # tail: 30 candidates minus the other known answer c02
    assert hits[0, 0].tolist() == pytest.approx([1 / 29, 3 / 29, 10 / 29])
    assert hits[0, 1].tolist() == pytest.approx([1 / 30, 3 / 30, 10 / 30])
    m = metrics_from_ranks(ranks, hits)
    assert m["hits@10"] == pytest.approx((10 / 29 + 10 / 30) / 2) and m["hits@1"] < m["hits@3"] < m["hits@10"]


def test_filtered_rank_removes_other_known_answers():
    cands = ["card:a", "card:b", "card:c", "card:d"]
    known = [("card:a", "COUNTERS", t) for t in ("card:b", "card:c", "card:d")]

    class Fixed:
        def score_tails(self, heads):
            return np.tile(np.array([0.0, 0.2, 0.9, 0.8]), (len(heads), 1))     # c and d outrank the true tail b

        def score_heads(self, tails):
            return np.tile(np.array([0.5, 0.0, 0.0, 0.0]), (len(tails), 1))

    ranks, hits = rank_and_hits(Fixed(), [("card:a", "COUNTERS", "card:b")], cands, known)
    assert ranks[0, 0] == 1.0                               # c, d are other true answers -> filtered; a (0.0) is lower
    assert ranks[0, 1] == 1.0                               # head a scores highest
    assert np.all(hits[0] == 1.0)


def test_tiny_pykeen_runs():
    sp = build_splits(synthetic_triples(), "full")
    tf = make_factory(sp)
    for name in MODELS:
        model, secs = train_model(name, tf, dim=8, lr=1e-2, epochs=2, seed=0)
        scorer = PykeenScorer(model, tf, sp.candidates)
        m = evaluate(scorer, sp.test_c, sp)
        assert math.isfinite(m["mrr"]) and 0 < m["mrr"] <= 1, (name, m)
        assert m["n"] == 2 * len(sp.test_c)
        asym = asymmetry(scorer, sp.test_c)
        assert 0.0 <= asym <= 1.0
        if name == "DistMult":
            assert asym == 0.0                              # symmetric scoring function
    preds = novel_predictions(scorer, sp, {}, top=5)
    known = sp.all_counters
    assert len(preds) == 5 and all((p["h"], "COUNTERS", p["t"]) not in known and p["h"] != p["t"] for p in preds)
    assert [p["score"] for p in preds] == sorted((p["score"] for p in preds), reverse=True)


@needs_kg
def test_real_data_split_counts():
    triples = load_triples(DATASET)
    raw = set()
    with (DATASET / "kg" / "edges.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            if e["rel"] == "COUNTERS":
                raw.add((e["src"], e["rel"], e["dst"]))
    assert not any(t[1].startswith("GAME_") for t in triples)
    for setting in ("full", "no_inverse"):
        sp = build_splits(triples, setting)
        assert len(sp.all_counters) == len(raw)
        assert len(sp.train_c) + len(sp.val_c) + len(sp.test_c) == len(raw) - sp.dropped_unseen
        assert sp.candidates and all(c.startswith("card:") for c in sp.candidates)
        assert not (set(sp.val_c) | set(sp.test_c)) & set(sp.train)
    assert not any(t[1] == "ANSWERED_BY" for t in sp.train)
