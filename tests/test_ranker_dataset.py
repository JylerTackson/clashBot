"""WP4/MAC-H tests: option text, placement prior lookup, 2x2 grid negatives (D-037), dataset, act files and baselines integration."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.actions import option_text, placement_tile  # noqa: E402
from clashbot.config import LANES  # noqa: E402
from clashbot.game_rules import legal_cards, load_cards, next_card_blocked, possible_action_keys, possible_cards  # noqa: E402
from tools.data.build_ranker_dataset import (GRID_SOURCES, grid_options, placement_block,  # noqa: E402
                                             placement_candidates, positive_rates, rate_rule_accuracy,
                                             shortcut_value)


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


_FREQ = {"knight": {"bridge": 5, "back": 1}}
_Z_BLOCK = [("left", "bridge"), ("left", "back"), ("left", "enemy_half")]


def _cands(variant, rng_seed=0):
    return placement_candidates("knight", "left", "mid", _FREQ, variant, random.Random(rng_seed))


def test_placement_candidates_variant_0():
    out = _cands(0)
    assert out[:3] == _Z_BLOCK                                                  # Z keeps zone-frequency order
    assert sorted(out[3:5]) == [("middle", "mid"), ("right", "mid")]            # L (shuffled)
    assert all(lane != "left" and zone != "mid" for lane, zone in out[5:])      # B (shuffled)
    assert len(out) == 3 + 2 + 6
    assert ("left", "mid") not in out
    assert len(set(out)) == len(out)


def test_placement_candidates_variants_rotate_blocks():
    v0, v1, v2 = (_cands(v) for v in range(3))
    assert v1[0] in {("middle", "mid"), ("right", "mid")}
    assert v1[2:5] == _Z_BLOCK
    assert v2[0][0] != "left" and v2[0][1] != "mid"
    assert v2[6:9] == _Z_BLOCK
    for out in (v0, v1, v2):
        assert ("left", "mid") not in out
        assert len(set(out)) == len(out)
    assert sorted(v0) == sorted(v1) == sorted(v2)
    # unknown card: zones fall back to alphabetical order
    assert placement_candidates("zap", "right", "back", {}, 0, random.Random(0))[0] == ("right", "bridge")


def test_placement_candidates_shuffle_is_deterministic_and_varies():
    assert _cands(1, 7) == _cands(1, 7)
    assert _cands(2, 7) == _cands(2, 7)
    # across rngs the candidate set is the same, but the first L entry (variant 1) and first B entry (variant 2) vary
    assert {_cands(1, s)[0] for s in range(20)} == {("middle", "mid"), ("right", "mid")}
    assert len({_cands(2, s)[0] for s in range(20)}) > 1
    assert all(sorted(_cands(1, s)) == sorted(_cands(1, 0)) for s in range(20))
    # Z is never shuffled
    assert all(_cands(0, s)[:3] == _Z_BLOCK for s in range(20))


def test_grid_options_negative_lane_is_not_fixed_by_lanes_order():
    cards = load_cards()
    rec = _grid_rec(["knight", "archers", "giant", "fireball"])
    possible = _possible(rec)
    # seed % 3 == 1 takes the first L placement: both other lanes must show up across seeds (FIX_LIST #13)
    lanes = {dict(grid_options(rec, "knight|left|mid", possible, [], cards, {}, seed))["G_card"].split("|")[1]
             for seed in range(1, 60, 3)}
    assert lanes == {"middle", "right"}
    assert grid_options(rec, "knight|left|mid", possible, [], cards, {}, 4) == \
        grid_options(rec, "knight|left|mid", possible, [], cards, {}, 4)


def test_rate_rules_on_hand_made_examples():
    # per example: (lane, zone) of each option and the label
    ex = [([("right", "mid"), ("left", "mid"), ("right", "back"), ("left", "back")], 0),
          ([("left", "mid"), ("right", "mid"), ("left", "back"), ("right", "back")], 1),
          ([("left", "mid"), ("left", "back"), ("left", "bridge"), ("left", "mid")], 0)]
    lane = lambda p: p[0]  # noqa: E731
    rates = positive_rates(ex, lane)
    assert rates == {"right": 2 / 4, "left": 1 / 8}
    # ex 1 and 2: the two 'right' options tie at the top (label among them); ex 3: all 'left', 4-way tie
    assert rate_rule_accuracy(ex, rates, lane) == pytest.approx((0.5 + 0.5 + 0.25) / 3)
    lz = positive_rates(ex, lambda p: p)
    assert lz[("right", "mid")] == 1.0 and lz[("left", "mid")] == 0.25 and lz[("left", "back")] == 0.0
    assert rate_rule_accuracy(ex, lz, lambda p: p) == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert rate_rule_accuracy([], rates, lane) == 0.0


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
        assert {"lane_only_rule", "lane_zone_rule"} <= set(st)
        assert set(st["negative_source_counts"]) == set(GRID_SOURCES)
        assert all(v == st["examples"] for v in st["negative_source_counts"].values())
        assert set(st["placement_variant_counts"]) == {"Z", "L", "B"}
        assert sum(st["placement_variant_counts"].values()) == st["examples"]


def test_lane_only_rule_is_not_a_shortcut(built, built_dataset):
    stats = json.loads((built_dataset / "ranker" / "stats.json").read_text())
    for split in ("val", "test"):
        assert stats[split]["lane_only_rule"] <= 0.28, split
    assert set(stats["train"]["p_pos_given_lane"]) <= set(LANES)
    assert "p_pos_given_lane" not in stats["val"]
    # recompute both rules from the written files (train rates -> each split) and compare with the stats
    per_split = {s: [([tuple(o["key"].split("|")[1:]) for o in ex["options"]], ex["label"]) for ex in exs]
                 for s, exs in built.items()}
    lane, lane_zone = (lambda p: p[0]), (lambda p: p)
    lane_rates = positive_rates(per_split["train"], lane)
    lane_zone_rates = positive_rates(per_split["train"], lane_zone)
    assert stats["train"]["p_pos_given_lane"] == {k: pytest.approx(v, abs=1e-4) for k, v in lane_rates.items()}
    for split, examples in per_split.items():
        assert stats[split]["lane_only_rule"] == pytest.approx(rate_rule_accuracy(examples, lane_rates, lane), abs=1e-4)
        assert stats[split]["lane_zone_rule"] == pytest.approx(
            rate_rule_accuracy(examples, lane_zone_rates, lane_zone), abs=1e-4)


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
