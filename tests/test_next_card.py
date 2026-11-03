"""MAC-R4 tests: opponent play sequences, copy / cycle baselines, SASRec, metrics and the next_card CLI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.features import vocab  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from tools.eval.next_card import (SASRec, Target, aggregate_seeds, copy_frequency, cycle, main,  # noqa: E402
                                  make_targets, opponent_sequences, popularity_order, rank_metrics, rank_of,
                                  sasrec_scores, summarize, train_sasrec, training_windows)

DATASET = ROOT / "dataset" / "v1"
needs_dataset = pytest.mark.skipif(not (DATASET / "states.jsonl").exists(), reason="dataset/v1/states.jsonl missing")
CARDS = load_cards()


def play(t, card):
    return {"t": t, "player": "opponent", "card": card, "tile": None, "elixir_cost": None,
            "detect_source": "arena", "confidence": "medium"}


def rec(game, recent, outcome_plays=None, split="train"):
    return {"game_id": game, "split": split, "state": {"opponent": {"recent_plays": recent}},
            "outcome": None if outcome_plays is None else {"opponent_plays": outcome_plays}}


def test_opponent_sequences_dedupe_collapse_filter_order():
    recs = [
        rec("g", [play(10.0, "knight"), play(14.0, "archers")], []),
        rec("g", [play(10.0, "knight"), play(20.0, "giant"), play(14.0, "archers"), play(10.4, "knight")],
            [play(14.04, "archers")]),
        rec("g", [play(20.0, "giant"), play(30.0, "miner"), play(31.0, "tower-princess"), play(32.0, None),
                  play(33.0, "not-a-card"), play(None, "zap")], None),
    ]
    seqs = opponent_sequences(recs, CARDS)
    assert seqs["g"]["split"] == "train"
    assert seqs["g"]["plays"] == [(10.0, "knight"), (14.0, "archers"), (20.0, "giant"), (30.0, "miner")]


def test_opponent_sequences_collapse_threshold_and_outcome():
    recs = [rec("g", [play(10.0, "knight")], [play(11.0, "knight"), play(11.5, "zap")], split="val")]
    seqs = opponent_sequences(recs, CARDS)
    # a re-read exactly 1.0 s later is a new play; both lists feed the sequence
    assert seqs["g"]["plays"] == [(10.0, "knight"), (11.0, "knight"), (11.5, "zap")]
    assert seqs["g"]["split"] == "val"


def test_make_targets_positions_and_buckets():
    seqs = {"g": {"split": "test", "plays": [(float(i), c) for i, c in enumerate("abcdefghij")]},
            "h": {"split": "val", "plays": [(0.0, "a")]}}
    tg = make_targets(seqs)
    assert [t.i for t in tg] == list(range(1, 10))
    assert tg[0].history == ("a",) and tg[0].truth == "b" and tg[0].bucket == "early"
    assert tg[7].history == tuple("abcdefgh") and tg[7].bucket == "late"      # 8 distinct cards seen


def test_copy_frequency_order():
    v = ["a", "b", "c", "d", "e", "f"]
    out = copy_frequency(["a", "b", "a", "c"], v)
    assert out[:3] == ["a", "c", "b"]
    assert out[3:] == ["d", "e", "f"] and sorted(out) == v


def test_popularity_order_ties_by_slug():
    assert popularity_order([["b", "b", "c"], ["c", "a"]], ["a", "b", "c", "d"]) == ["b", "c", "a", "d"]


def test_cycle_rule():
    v = list("abcdefgh")
    hist = list("abcde")
    assert cycle(hist, v, 4)[0] == "a"                       # only seen card outside the last 4 plays
    assert cycle(hist, v, 4) == ["a", "e", "d", "c", "b", "f", "g", "h"]
    assert cycle(hist, v, 0) == copy_frequency(hist, v)
    assert cycle(hist, v, 1)[:5] == ["d", "c", "b", "a", "e"]


def test_metrics_and_ranks():
    assert rank_of(["x", "y", "z"], "z") == 3
    m = rank_metrics([3])
    assert m["MRR"] == pytest.approx(1 / 3) and m["acc@1"] == 0.0 and m["acc@3"] == 1.0 and m["n"] == 1
    m = rank_metrics([1, 3, 5, 6])
    assert (m["acc@1"], m["acc@3"], m["acc@5"]) == (0.25, 0.5, 0.75)
    assert m["MRR"] == pytest.approx((1 + 1 / 3 + 1 / 5 + 1 / 6) / 4)
    assert rank_metrics([])["n"] == 0
    tg = [Target("g", "val", 1, ("a",), "b"), Target("g", "val", 9, tuple("abcdefgh"), "i")]
    s = summarize(tg, [1, 4])
    assert s["early"]["acc@1"] == 1.0 and s["late"]["acc@1"] == 0.0 and s["late"]["n"] == 1


def test_aggregate_seeds_mean_std():
    a = {"acc@1": 0.2, "n": 5, "early": {"acc@1": 0.1, "n": 2}}
    b = {"acc@1": 0.4, "n": 5, "early": {"acc@1": 0.3, "n": 2}}
    agg = aggregate_seeds([a, b])
    assert agg["acc@1"] == pytest.approx(0.3) and agg["std"]["acc@1"] == pytest.approx(0.1)
    assert agg["n"] == 5 and agg["early"]["acc@1"] == pytest.approx(0.2)


def test_training_windows():
    assert training_windows([1], 5) == []
    assert training_windows([1, 2, 3], 5) == [[1, 2, 3]]
    w = training_windows(list(range(1, 21)), 5)
    assert all(len(x) == 6 for x in w) and w[0][0] == 1 and w[-1][-1] == 20


def test_sasrec_causal_mask():
    torch.manual_seed(0)
    model = SASRec(8, d=16, n_blocks=2, n_heads=2, d_ff=32, dropout=0.0, max_len=12).eval()
    ids = torch.randint(1, 9, (2, 10))
    ids2 = ids.clone()
    ids2[:, 6:] = (ids2[:, 6:] % 8) + 1                      # change the future tokens
    with torch.no_grad():
        a, b = model(ids), model(ids2)
    assert a.shape == (2, 10, 8)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5)
    assert not torch.allclose(a[:, 6:], b[:, 6:], atol=1e-5)
    # right padding does not change the logits of earlier positions
    padded = torch.cat([ids, torch.zeros(2, 2, dtype=torch.long)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(padded)[:, :10], a, atol=1e-5)


def test_sasrec_learns_periodic_sequences():
    rng = np.random.default_rng(0)

    def seq(n=24):
        off = int(rng.integers(8))
        return [(off + j) % 8 + 1 for j in range(n)]

    train = [seq() for _ in range(200)]
    val = [seq() for _ in range(30)]
    ctx = [(s[max(0, i - 50):i], s[i]) for s in val for i in range(1, len(s))]
    model, info = train_sasrec(train, ctx, 8, d=16, n_blocks=1, n_heads=2, d_ff=32, epochs=30, patience=30,
                               seed=0)
    scores = sasrec_scores(model, [c for c, _ in ctx])
    acc = float((scores.argmax(1) + 1 == np.array([t for _, t in ctx])).mean())
    assert acc >= 0.9, (acc, info)
    assert info["val_acc@1"] >= 0.9


def _write_states(path: Path) -> None:
    cards = ["knight", "archers", "giant", "miner", "zap", "p-e-k-k-a", "knight", "giant", "archers", "zap"]
    rows = []
    for gid, split in (("g1", "train"), ("g2", "train"), ("g3", "val"), ("g4", "test")):
        plays = [play(10.0 + 5.0 * j, c) for j, c in enumerate(cards)]
        for k in range(3):
            rows.append(rec(gid, plays[3 * k:3 * k + 4], plays[3 * k + 2:3 * k + 6], split=split))
    with open(path / "states.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_quick_end_to_end(tmp_path):
    _write_states(tmp_path)
    out = tmp_path / "nc.json"
    main(["--dataset", str(tmp_path), "--quick", "--out", str(out)])
    res = json.loads(out.read_text())
    assert set(res["methods"]) == {"popularity", "copy_frequency", "cycle", "sasrec", "sasrec+cycle"}
    assert res["n_games"] == {"train": 2, "val": 1, "test": 1}
    assert res["n_targets"]["val"] == 9 and res["n_targets"]["test"] == 9
    assert set(res["cycle_w_val"]) == {"0", "1", "2", "3", "4"} and res["chosen_w"] in range(5)
    for m, by_split in res["methods"].items():
        for split in ("val", "test"):
            met = by_split[split]
            assert {"acc@1", "acc@3", "acc@5", "MRR", "n", "early", "late"} <= set(met), m
            assert met["n"] == 9 and 0.0 <= met["acc@1"] <= met["acc@3"] <= met["acc@5"] <= 1.0
    assert res["methods"]["sasrec"]["val"]["std"]["acc@1"] == 0.0           # one seed in --quick
    md = out.with_name("nc.json.md").read_text()
    assert "| method | val acc@1 | val acc@3 | val MRR | test acc@1 | test acc@3 | test MRR |" in md
    assert "sasrec+cycle" in md


@needs_dataset
def test_real_data_train_game_count(tmp_path):
    games = [json.loads(line) for line in open(DATASET / "games.jsonl") if line.strip()]
    recs = [json.loads(line) for line in open(DATASET / "states.jsonl") if line.strip()]
    seqs = opponent_sequences(recs, CARDS)
    expected = sum(1 for g in games if g["split"] == "train" and len(seqs.get(g["game_id"], {"plays": []})["plays"]) >= 2)
    assert expected > 0
    out = tmp_path / "real.json"
    main(["--dataset", str(DATASET), "--methods", "popularity,copy_frequency,cycle", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["n_games"]["train"] == expected
    assert res["n_targets"]["val"] > 0 and 0.0 < res["oracle_seen"]["val"] <= 1.0
    assert set(vocab()) >= {c for s in seqs.values() for _, c in s["plays"]}
