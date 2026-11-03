"""WP4/MAC-H tests: option text, placement prior lookup, 2x2 grid negatives (D-037), dataset, act files and baselines integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.actions import option_text, placement_tile  # noqa: E402
from clashbot.game_rules import legal_cards, load_cards, next_card_blocked, possible_action_keys, possible_cards  # noqa: E402
from tools.data.build_ranker_dataset import (GRID_SOURCES, grid_options, placement_block,  # noqa: E402
                                             placement_candidates, shortcut_value)


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


# --- 2x2 grid negatives (D-037) on hand-made states

_GRID_DECK = ["knight", "archers", "giant", "fireball", "musketeer", "valkyrie", "minions", "zap"]


def _grid_rec(hand, next_card="musketeer", elixir=10.0):
    return {"state": {"own": {"hand": hand, "next_card": next_card, "deck": list(_GRID_DECK), "elixir": elixir},
                      "opponent": {"towers": {"left": 1, "right": 1}}}}


def _possible(rec):
    return possible_action_keys(rec["state"], load_cards())


def _parts(keys):
    return [k.split("|") for k in keys]


def test_placement_candidates_variant_0():
    out = placement_candidates("knight", "left", "mid", {"knight": {"bridge": 5, "back": 1}}, 0)
    assert out[:3] == [("left", "bridge"), ("left", "back"), ("left", "enemy_half")]
    assert out[3:5] == [("middle", "mid"), ("right", "mid")]
    assert len(out) == 3 + 2 + 6
    assert ("left", "mid") not in out
    assert len(set(out)) == len(out)


def test_placement_candidates_variants_rotate_blocks():
    freq = {"knight": {"bridge": 5, "back": 1}}
    v0, v1, v2 = (placement_candidates("knight", "left", "mid", freq, v) for v in range(3))
    assert v1[0] == ("middle", "mid")
    assert v2[0][0] != "left" and v2[0][1] != "mid"
    for out in (v0, v1, v2):
        assert ("left", "mid") not in out
        assert len(set(out)) == len(out)
    assert sorted(v0) == sorted(v1) == sorted(v2)
    # unknown card: zones fall back to alphabetical order
    assert placement_candidates("zap", "right", "back", {}, 0)[0] == ("right", "bridge")


def test_grid_options_forms_a_grid():
    cards = load_cards()
    rec = _grid_rec(["knight", "archers", "giant", "fireball"])
    possible = _possible(rec)
    pos = "knight|left|mid"
    negs = grid_options(rec, pos, possible, [], cards, {"knight": {"bridge": 5, "back": 1}}, 0)
    assert negs is not None and len(negs) == 3
    assert [src for src, _ in negs] == list(GRID_SOURCES)
    keys = [pos] + [k for _, k in negs]
    assert len(set(keys)) == 4 and all(k in set(possible) for k in keys)
    assert len({p[0] for p in _parts(keys)}) == 2
    assert len({(p[1], p[2]) for p in _parts(keys)}) == 2
    # c2 = first possible card, placement = first Z candidate (knight's most frequent other zone)
    assert dict(negs) == {"G_card": "knight|left|bridge", "G_place": "archers|left|mid", "G_diag": "archers|left|bridge"}


def test_grid_options_prefers_neighbour_card():
    cards = load_cards()
    rec = _grid_rec(["knight", "archers", "giant", "fireball"])
    negs = grid_options(rec, "knight|left|mid", _possible(rec), ["hold", "giant|right|back", "archers|left|mid"],
                        cards, {}, 0)
    assert dict(negs)["G_place"] == "giant|left|mid"
    assert dict(negs)["G_diag"].startswith("giant|")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_grid_options_all_variants_valid(seed):
    cards = load_cards()
    rec = _grid_rec(["knight", "archers", "giant", "fireball"])
    possible = _possible(rec)
    pos = "knight|left|mid"
    negs = grid_options(rec, pos, possible, [], cards, {}, seed)
    keys = [pos] + [k for _, k in negs]
    assert all(k in set(possible) for k in keys)
    assert len({p[0] for p in _parts(keys)}) == 2 and len({(p[1], p[2]) for p in _parts(keys)}) == 2
    assert placement_block(pos, dict(negs)["G_card"]) == "ZLB"[seed % 3]


def test_grid_options_none_when_one_card_possible():
    cards = load_cards()
    rec = _grid_rec(["knight", "knight", "knight", "knight"])
    assert possible_cards(rec["state"], cards) == ["knight"]
    assert grid_options(rec, "knight|left|mid", _possible(rec), [], cards, {}, 0) is None


def test_grid_options_never_offers_next_card():
    cards = load_cards()
    rec = _grid_rec(["knight", None, None, None], next_card="archers")
    negs = grid_options(rec, "knight|left|mid", _possible(rec), ["archers|left|mid"], cards, {}, 0)
    assert negs is not None
    assert all(k.split("|")[0] != "archers" for _, k in negs)


def test_option_set_shortcut_neutral_on_grid():
    cards = load_cards()
    rec = _grid_rec(["knight", "archers", "giant", "fireball"])
    pos = "knight|left|mid"
    keys = [pos] + [k for _, k in grid_options(rec, pos, _possible(rec), [], cards, {}, 0)]
    cs = [p[0] for p in _parts(keys)]
    ps = [(p[1], p[2]) for p in _parts(keys)]
    assert [cs.count(c) + ps.count(p) for c, p in zip(cs, ps)] == [4, 4, 4, 4]
    for label in range(4):
        assert shortcut_value(keys, label) == pytest.approx(0.25)


def test_option_set_shortcut_catches_intersection_option():
    # old-style negatives: one shares the card, one shares lane+zone with the positive, one is unrelated
    keys = ["knight|left|mid", "knight|left|bridge", "giant|left|mid", "archers|right|back"]
    assert shortcut_value(keys, 0) == 1.0
    assert shortcut_value(keys, 1) == 0.0


# --- built data (temp build from the session fixture in conftest.py)

def _read(root: Path, split: str, sub: str = "ranker") -> list[dict]:
    with open(root / sub / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def _states(root: Path) -> dict:
    out = {}
    with open(root / "states.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["state"]
    return out


@pytest.fixture(scope="module")
def built(built_dataset):
    return {s: _read(built_dataset, s) for s in ("train", "val", "test")}


def test_built_examples(built, built_dataset):
    cards = load_cards()
    states = _states(built_dataset)
    for split, examples in built.items():
        assert examples, split
        for ex in examples:
            assert len(ex["options"]) == 4
            assert sum(o["source"] == "pos" for o in ex["options"]) == 1
            assert sorted(o["source"] for o in ex["options"] if o["source"] != "pos") == sorted(GRID_SOURCES)
            assert ex["options"][ex["label"]]["key"] == ex["positive_key"]
            assert "action:" not in ex["context"]
            keys = [o["key"] for o in ex["options"]]
            assert len(set(keys)) == len(keys)
            assert "hold" not in keys
            possible = set(possible_action_keys(states[ex["id"]], cards))
            assert all(k in possible for k in keys)
            parts = _parts(keys)
            assert len({p[0] for p in parts}) == 2
            assert len({(p[1], p[2]) for p in parts}) == 2
            assert len({tuple(p) for p in parts}) == 4
        assert len({ex["label"] for ex in examples}) > 1
    train_games = {ex["game_id"] for ex in built["train"]}
    for split in ("val", "test"):
        assert not train_games & {ex["game_id"] for ex in built[split]}
    assert (built_dataset / "placement_prior.json").exists()
    stats = json.loads((built_dataset / "ranker" / "stats.json").read_text())
    for split, examples in built.items():
        st = stats[split]
        assert {"skipped_no_grid", "skipped_pos_impossible", "placement_variant_counts"} <= set(st)
        assert "skipped_no_negatives" not in st
        assert st["examples"] == len(examples)
        assert st["mean_options"] == 4.0
        assert st["option_set_shortcut"] == pytest.approx(0.25, abs=1e-9)
        assert set(st["negative_source_counts"]) == set(GRID_SOURCES)
        assert all(v == st["examples"] for v in st["negative_source_counts"].values())
        assert set(st["placement_variant_counts"]) == {"Z", "L", "B"}
        assert sum(st["placement_variant_counts"].values()) == st["examples"]


def test_act_files(built, built_dataset):
    for split in ("train", "val", "test"):
        labels = {ex["label"] for ex in _read(built_dataset, split, "act")}
        assert labels <= {0, 1}
        if split == "train":
            assert labels == {0, 1}


def test_baselines(built, built_dataset):
    res = json.loads((built_dataset / "ranker" / "baselines.json").read_text())
    assert "knn_vote" in res["note"]
    for split in ("val", "test"):
        assert set(res["ranker"][split]) == {"random_expected", "popularity", "popularity_by_phase",
                                             "card_popularity_in_deck", "placement_prior", "option_set_shortcut"}
        assert res["ranker"][split]["option_set_shortcut"]["acc"] == pytest.approx(0.25, abs=1e-9)
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


_NEXT_DECK = ["knight", "archers", "giant", "fireball", "musketeer", "valkyrie", "minions", "zap"]


def _next_state(hand, next_card, elixir=10.0):
    return {"own": {"hand": hand, "next_card": next_card, "deck": list(_NEXT_DECK), "elixir": elixir}}


def test_possible_cards_excludes_next_card():
    cards = load_cards()
    out = possible_cards(_next_state(["knight", None, None, None], "archers"), cards)
    assert "archers" not in out
    assert "knight" in out


def test_possible_cards_next_card_in_hand_not_blocked():
    cards = load_cards()
    state = _next_state(["archers", None, None, None], "archers")
    assert next_card_blocked(state) is None
    assert "archers" in possible_cards(state, cards)


def test_legal_cards_fallback_excludes_next_card():
    cards = load_cards()
    out = legal_cards(_next_state([None, None, None, None], "archers", elixir=10.0), cards)
    assert "archers" not in out
    assert "knight" in out


def test_next_card_never_offered_in_built_data(built, built_dataset):
    states = _states(built_dataset)
    fact_dir = built_dataset / "d1" / "factorized"
    for path in sorted(fact_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                blocked = next_card_blocked(states[ex["id"]])
                if blocked:
                    assert blocked not in ex["questions"]["card"]["criteria"], (path.name, ex["id"])
    for split, examples in built.items():
        for ex in examples:
            blocked = next_card_blocked(states[ex["id"]])
            if blocked:
                assert all(o["key"].split("|")[0] != blocked for o in ex["options"]), (split, ex["id"])
