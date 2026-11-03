"""MAC-R3 tests: pointwise features, leave-one-game-out statistics, leakage guards, option-set shortcut, listwise loss, CLI."""
from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval import feature_ranker as fr  # noqa: E402

REAL = ROOT / "dataset" / "v1"
needs_dataset = pytest.mark.skipif(not (REAL / "ranker" / "val.jsonl").exists(), reason="dataset/v1/ranker/val.jsonl missing")
CARDS = ("knight", "zap", "cannon")           # Troop, Spell, Building
POS = "knight|left|bridge"


def _state(elixir=7.0, units=None, opp_plays=None, hand=None, phase="double_elixir"):
    return {
        "phase": phase, "match_seconds": 150,
        "own": {"elixir": elixir, "hand": hand or ["knight", "zap", "cannon", "giant"],
                "deck": ["knight", "zap", "cannon", "giant"],
                "towers": {"king": None, "left": 4000, "right": 5000}, "recent_plays": []},
        "opponent": {"elixir_estimate": 4.0, "towers": {"king": None, "left": 3000, "right": 4858},
                     "deck_known": [], "recent_plays": opp_plays or []},
        "units": units or [], "threats": [],
    }


def _record(rid, game, split, key, t=60.0, state=None):
    card, lane, zone = key.split("|")
    return {"id": rid, "game_id": game, "split": split, "kind": "play", "t": t,
            "state": state or _state(),
            "action": {"type": "play", "card": card, "lane": lane, "zone": zone, "key": key},
            "train_eligible": True}


def _example(rid, game, split, keys, label=0, kind="play"):
    return {"id": rid, "game_id": game, "split": split, "kind": kind,
            "options": [{"key": k, "text": k, "source": "pos" if i == label else "N1"} for i, k in enumerate(keys)],
            "label": label, "positive_key": keys[label], "pos_in_legal": True}


INTERSECTION = [POS, "knight|left|back", "zap|left|bridge", "cannon|right|mid"]    # pos = pos / N2 / N3 / N1
DISTINCT = ["knight|left|back", "zap|middle|mid", "cannon|right|bridge", "giant|left|enemy_half"]


def _stats_records():
    return [_record("a1", "g1", "train", POS), _record("a2", "g1", "train", POS),
            _record("a3", "g1", "train", "zap|left|bridge"),
            _record("b1", "g2", "train", POS), _record("b2", "g2", "train", "knight|left|back"),
            _record("c1", "g3", "train", "cannon|right|mid"),
            _record("v1", "gv", "val", POS)]


def test_build_features_shape_and_pointwise():
    records = _stats_records()
    stats = fr.train_stats(records)
    rec = records[0]
    ex = _example("a1", "g1", "train", INTERSECTION)
    F = fr.n_features()
    rows = fr.build_features(ex, rec, stats, None)
    assert rows.shape == (4, F)
    assert len(fr.all_feature_names()) == F and np.isfinite(rows).all()
    # changing / reordering the other options does not change an option's row
    other = _example("a1", "g1", "train", [POS, "giant|right|enemy_half", "zap|middle|mid", "cannon|left|back"])
    assert np.array_equal(fr.build_features(other, rec, stats, None)[0], rows[0])
    swapped = _example("a1", "g1", "train", [INTERSECTION[2], INTERSECTION[3], INTERSECTION[0], INTERSECTION[1]])
    assert np.array_equal(fr.build_features(swapped, rec, stats, None)[2], rows[0])
    # option block sanity: card / lane / zone one-hots, in_hand, cost
    names = fr.all_feature_names()
    assert rows[0, names.index("opt:card:knight")] == 1 and rows[0, names.index("opt:lane:left")] == 1
    assert rows[0, names.index("opt:zone:bridge")] == 1 and rows[0, names.index("opt:type:Troop")] == 1
    assert rows[2, names.index("opt:type:Spell")] == 1 and rows[3, names.index("opt:type:Building")] == 1
    assert rows[0, names.index("opt:in_hand")] == 1
    assert rows[0, names.index("opt:kg_counter")] == 0.0


def test_leave_one_game_out_statistics():
    records = _stats_records()
    full = fr.train_stats(records)
    no_g1 = fr.train_stats(records, exclude_game="g1")
    assert full.key_count(POS) == 3 and no_g1.key_count(POS) == 1          # val record never counts
    assert full.card_count("knight") == 4 and no_g1.card_count("knight") == 2
    assert full.zone_count("zap", "bridge") == 1 and no_g1.zone_count("zap", "bridge") == 0
    assert no_g1.key_phase_count("double_elixir", POS) == 1
    assert "g1" not in no_g1.per_game
    # exclusion by view equals building the statistics without the game
    assert full.key_count(POS, game="g1") == no_g1.key_count(POS)
    assert full.p_zone_given_card("knight", "bridge", game="g1") == pytest.approx((1 + 1) / (2 + 4))
    assert full.p_zone_given_card("knight", "bridge") == pytest.approx((3 + 1) / (4 + 4))

    # a train example of g1 sees the g1-free counts, identically for all four options
    names = fr.all_feature_names()
    cols = [names.index(n) for n in ("opt:log_pop_key", "opt:log_pop_card", "opt:log_pop_key_phase", "opt:p_zone_given_card")]
    ex = _example("a1", "g1", "train", INTERSECTION)
    rows = fr.build_features(ex, records[0], full, None)
    assert np.array_equal(rows, fr.build_features(ex, records[0], no_g1, None))
    assert rows[0, cols[0]] == pytest.approx(math.log1p(1))               # pop_key(POS) without g1
    assert rows[0, cols[1]] == pytest.approx(math.log1p(2))               # pop_card(knight) without g1
    assert rows[0, cols[2]] == pytest.approx(math.log1p(1))
    assert rows[0, cols[3]] == pytest.approx(2 / 6)
    assert rows[2, cols[0]] == 0.0                                        # zap|left|bridge only occurs in g1
    # a val example (game not in train) sees the full counts
    exv = _example("v1", "gv", "val", INTERSECTION)
    assert fr.build_features(exv, records[-1], full, None)[0, cols[0]] == pytest.approx(math.log1p(3))


def test_forbidden_fields_do_not_change_features():
    records = _stats_records()
    stats = fr.train_stats(records)
    ex = _example("a1", "g1", "train", INTERSECTION)
    base = fr.build_features(ex, records[0], stats, None)
    changed = copy.deepcopy(ex)
    changed["positive_key"], changed["pos_in_legal"], changed["label"] = "zap|left|back", False, 3
    for o in changed["options"]:
        o["source"] = "N5"
        o["text"] = "something else"
    assert np.array_equal(fr.build_features(changed, records[0], stats, None), base)
    deleted = copy.deepcopy(ex)
    for k in ("positive_key", "pos_in_legal", "label", "kind"):
        del deleted[k]
    for o in deleted["options"]:
        del o["source"]
    assert np.array_equal(fr.build_features(deleted, records[0], stats, None), base)
    # the record's own action (the label) is never read
    rec2 = copy.deepcopy(records[0])
    rec2["action"] = {"type": "play", "card": "zap", "key": "zap|right|back"}
    assert np.array_equal(fr.build_features(ex, rec2, stats, None), base)


def test_option_set_shortcut():
    assert fr.option_set_accuracy(_example("x", "g", "val", INTERSECTION, label=0)) == 1.0
    assert fr.option_set_accuracy(_example("x", "g", "val", INTERSECTION, label=1)) == 0.0
    assert fr.option_set_accuracy(_example("x", "g", "val", DISTINCT, label=2)) == 0.25
    assert fr.option_set_scores(_example("x", "g", "val", INTERSECTION)) == [4.0, 3.0, 3.0, 2.0]


def test_kg_counter_uses_threats():
    class FakeKG:
        def counter_scores(self, threat):
            return {"knight": 0.4, "zap": 1.5} if threat == "giant" else {"knight": 9.0}

    state = _state(units=[{"side": "enemy", "card": "giant", "tile": [8, 20], "eta_tower": {"tower": "own_left", "s": 6.0}},
                          {"side": "enemy", "card": "far", "tile": [8, 30], "eta_tower": {"tower": "own_left", "s": 12.0}}],
                   opp_plays=[{"card": "old", "t": 10.0}])
    assert fr.threat_cards(state, 60.0) == ["giant"]
    rec = _record("a1", "g1", "train", POS, state=state)
    rows = fr.build_features(_example("a1", "g1", "train", INTERSECTION), rec, fr.Stats(), FakeKG())
    col = fr.all_feature_names().index("opt:kg_counter")
    assert rows[:, col].tolist() == pytest.approx([0.4, 0.4, 1.5, 0.0])


def test_listwise_loss_and_scorers():
    labels = torch.tensor([0, 2, 3])
    sharp = torch.full((3, 4), 0.0)
    sharp[torch.arange(3), labels] = 20.0
    assert fr.listwise_loss(sharp, labels).item() < 0.01
    assert fr.listwise_loss(torch.zeros(3, 4), labels).item() == pytest.approx(math.log(4), abs=1e-6)
    x = torch.randn(5, 4, 12)
    assert fr.LinearScorer(12)(x).shape == (5, 4)
    assert fr.DCNv2Scorer(12)(x).shape == (5, 4)
    assert fr.accuracy(np.array([[0.1, 0.9, 0, 0], [1, 0, 0, 0]]), np.array([1, 2])) == 0.5


# ---------------------------------------------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------------------------------------------

def _write_synthetic_dataset(root: Path) -> None:
    rng = random.Random(0)
    sizes = {"train": (40, 8), "val": (10, 2), "test": (10, 2)}         # (examples, games)
    records, ranker = [], {}
    lanes, zones = fr.LANES, fr.ZONES
    for split, (n, n_games) in sizes.items():
        ranker[split] = []
        for i in range(n):
            game = f"{split}{i % n_games}"
            card, lane, zone = rng.choice(CARDS), rng.choice(lanes), rng.choice(zones)
            pos = f"{card}|{lane}|{zone}"
            n2 = f"{card}|{lane}|{rng.choice([z for z in zones if z != zone])}"
            n3 = f"{rng.choice([c for c in CARDS if c != card])}|{lane}|{zone}"
            n1 = f"{rng.choice(CARDS)}|{rng.choice(lanes)}|{rng.choice(zones)}"
            keys = list(dict.fromkeys([pos, n2, n3, n1]))
            while len(keys) < 4:
                keys.append(f"giant|{rng.choice(lanes)}|{rng.choice(zones)}")
                keys = list(dict.fromkeys(keys))
            rng.shuffle(keys)
            rid = f"{game}#{i}.0"
            rec = _record(rid, game, split, pos, t=float(i), state=_state(elixir=float(rng.randint(2, 10))))
            records.append(rec)
            ranker[split].append(_example(rid, game, split, keys, label=keys.index(pos), kind="play" if i % 3 else "key"))
    (root / "ranker").mkdir(parents=True)
    (root / "states.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    for split, exs in ranker.items():
        (root / "ranker" / f"{split}.jsonl").write_text("\n".join(json.dumps(e) for e in exs) + "\n")
    base = {name: {"acc": acc, "n": 10, "by_kind": {"play": {"acc": acc, "n": 7}, "key": {"acc": acc, "n": 3}}}
            for name, acc in (("random_expected", 0.25), ("popularity", 0.3), ("placement_prior", 0.4))}
    (root / "ranker" / "baselines.json").write_text(json.dumps({"ranker": {"val": base, "test": base}}))


def test_quick_end_to_end_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "DB_PATH", tmp_path / "missing.sqlite")        # kg_counter falls back to 0
    _write_synthetic_dataset(tmp_path / "ds")
    out = tmp_path / "out" / "f.json"
    fr.main(["--dataset", str(tmp_path / "ds"), "--quick", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["n"] == {"train": 40, "val": 10, "test": 10}
    assert res["features"] == fr.n_features() and res["kg_counter_available"] is False
    assert set(res["models"]) == {"logreg", "lgbm_lambdarank", "dcn_v2"}
    for m in res["models"].values():
        assert len(m["grid"]) == 1 and len(m["best"]["seeds"]) == 1
        assert 0.0 <= m["val_acc_mean"] <= 1.0 and 0.0 <= m["test_acc_mean"] <= 1.0
        assert set(m["test_by_kind"]) == {"play", "key"}
    assert set(res["ablation"]) == {"full", "no_kg_counter", "no_popularity", "no_state_block", "no_train_stats"}
    assert len(res["importance"]) == 20 and res["importance"][0]["feature"].split(":")[0] in {"state", "opt"}
    assert res["reference"]["popularity"]["test_acc"] == 0.3
    assert res["reference"]["option_set_shortcut"]["test_acc"] > 0.5      # the leak is visible even on synthetic data
    md = out.with_suffix(".md").read_text()
    assert "| model | val acc | test acc |" in md and "NOT A VALID MODEL" in md and "lgbm_lambdarank" in md


@needs_dataset
def test_quick_real_data(tmp_path):
    res = fr.main(["--quick", "--limit", "200", "--out", str(tmp_path / "f.json")])
    assert res["n"] == {"train": 200, "val": 200, "test": 200}
    for m in res["models"].values():
        assert 0.0 <= m["val_acc_mean"] <= 1.0 and 0.0 <= m["test_acc_mean"] <= 1.0
    assert (tmp_path / "f.json").exists()
