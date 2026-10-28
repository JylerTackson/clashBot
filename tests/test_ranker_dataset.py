"""WP4 tests: option text, placement prior lookup, negative selection, dataset, act files and baselines integration."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.actions import option_text, placement_tile  # noqa: E402
from clashbot.game_rules import load_cards, possible_action_keys  # noqa: E402
from tools.data.build_ranker_dataset import select_negatives  # noqa: E402

DATASET = ROOT / "dataset" / "v1"
needs_dataset = pytest.mark.skipif(not (DATASET / "states.jsonl").exists(), reason="dataset/v1/states.jsonl missing")


def test_option_text():
    assert option_text("hold") == "hold elixir and play nothing yet"
    assert option_text("knight|left|bridge") == "play knight in the left lane, at the bridge"
    assert option_text("giant|right|back") == "play giant in the right lane, back, behind the king tower"


def test_placement_tile_fallback():
    prior = {"by_key": {"knight|left|bridge": {"count": 3, "median_tile": [4.0, 14.0]}},
             "by_lane_zone": {"left|bridge": {"count": 9, "median_tile": [3.5, 13.0]}}}
    assert placement_tile("knight|left|bridge", prior) == [4.0, 14.0]
    assert placement_tile("giant|left|bridge", prior) == [3.5, 13.0]
    assert placement_tile("giant|right|back", prior) is None
    assert placement_tile("hold", prior) is None


def test_select_negatives_priority_order():
    pools = {"N1": ["a", "b", "c"], "N2": ["x"], "N3": ["y"], "N5": ["z"]}
    assert select_negatives(pools, "p") == [("N1", "a"), ("N2", "x"), ("N3", "y")]


def test_select_negatives_n3_used_when_n2_empty():
    pools = {"N1": ["a", "b"], "N2": [], "N3": ["y"], "N5": ["z"]}
    assert select_negatives(pools, "p") == [("N1", "a"), ("N3", "y"), ("N1", "b")]


def test_select_negatives_duplicates_across_pools():
    pools = {"N1": ["a", "b", "c"], "N2": ["a"], "N3": ["b"], "N5": ["d"]}
    # N1 first = a; N2 a is a duplicate; N3 b; then remaining N1: a dup, b dup, c
    assert select_negatives(pools, "p") == [("N1", "a"), ("N3", "b"), ("N1", "c")]


def test_select_negatives_fewer_than_three():
    assert select_negatives({"N1": ["a"], "N2": [], "N3": [], "N5": ["p"]}, "p") == [("N1", "a")]
    assert select_negatives({"N1": [], "N2": [], "N3": [], "N5": []}, "p") == []


def test_select_negatives_falls_back_to_n5():
    pools = {"N1": [], "N2": ["x"], "N3": [], "N5": ["p", "z", "w", "v"]}
    assert select_negatives(pools, "p") == [("N2", "x"), ("N5", "z"), ("N5", "w")]


def _read(split: str, sub: str = "ranker") -> list[dict]:
    return [json.loads(line) for line in open(DATASET / sub / f"{split}.jsonl")]


@pytest.fixture(scope="module")
def built():
    subprocess.run([sys.executable, str(ROOT / "tools/data/build_ranker_dataset.py")], check=True,
                   capture_output=True, text=True)
    return {s: _read(s) for s in ("train", "val", "test")}


@needs_dataset
def test_built_examples(built):
    cards = load_cards()
    states = {}
    with open(DATASET / "states.jsonl") as f:
        for line in f:
            r = json.loads(line)
            states[r["id"]] = r["state"]
    for split, examples in built.items():
        assert examples, split
        for ex in examples:
            assert 2 <= len(ex["options"]) <= 4
            assert sum(o["source"] == "pos" for o in ex["options"]) == 1
            assert ex["options"][ex["label"]]["key"] == ex["positive_key"]
            assert "action:" not in ex["context"]
            keys = [o["key"] for o in ex["options"]]
            assert len(set(keys)) == len(keys)
            assert "hold" not in keys
            possible = set(possible_action_keys(states[ex["id"]], cards))
            assert all(k in possible for k in keys)
        assert len({ex["label"] for ex in examples}) > 1
    train_games = {ex["game_id"] for ex in built["train"]}
    for split in ("val", "test"):
        assert not train_games & {ex["game_id"] for ex in built[split]}
    assert (DATASET / "placement_prior.json").exists()
    stats = json.loads((DATASET / "ranker" / "stats.json").read_text())
    assert all("skipped_pos_impossible" in stats[s] for s in stats)


@needs_dataset
def test_act_files(built):
    for split in ("train", "val", "test"):
        labels = {ex["label"] for ex in _read(split, "act")}
        assert labels <= {0, 1}
        if split == "train":
            assert labels == {0, 1}


@needs_dataset
def test_baselines(built):
    subprocess.run([sys.executable, str(ROOT / "tools/eval/eval_ranker_baselines.py")], check=True,
                   capture_output=True, text=True)
    res = json.loads((DATASET / "ranker" / "baselines.json").read_text())
    assert "knn_vote" in res["note"]
    for split in ("val", "test"):
        assert set(res["ranker"][split]) == {"random_expected", "popularity", "popularity_by_phase",
                                             "card_popularity_in_deck", "placement_prior"}
        assert set(res["act"][split]) == {"always_play", "phase_majority"}
        for b in list(res["ranker"][split].values()) + list(res["act"][split].values()):
            assert 0.0 <= b["acc"] <= 1.0
            assert b["n"] > 0
        assert set(res["candidate_recall"][split]) == {"knn_vote", "popularity"}
        for m in res["candidate_recall"][split].values():
            for prefix in ("recall", "card_recall"):
                vals = [m[f"{prefix}@{k}"] for k in (1, 3, 5, 10)]
                assert all(0.0 <= v <= 1.0 for v in vals)
                assert vals == sorted(vals)
