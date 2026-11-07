"""MAC-AB A1: fast play/wait model - features, labels, the trainer, the tree walker, FastActPolicy and the live CLI.

Synthetic data: a toy dense set where the pro acts when own elixir is high, tiny LightGBM models trained in a second,
a fake model with a fixed p for the policy tests. One test compares the runtime tree walker with LightGBM on real val
rows (skipped without the data / models). The runtime never imports lightgbm (it crashes next to torch): LightGBM's own
predictions are always computed in a subprocess here, and the torch tests run in subprocesses too. No GPU.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot import act_model  # noqa: E402
from clashbot.act_model import FEATURES, ActGBDT, FastActPolicy, TreeEnsemble, act_features  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live import build_parser, main  # noqa: E402
from clashbot.policy import ActDecision, ReactivePolicy  # noqa: E402
from tests.test_act_head import _cands, _enemy, _patch_main, _run_main, _with  # noqa: E402
from clashbot.tactics import DEFAULT_WEIGHTS  # noqa: E402
from tests.test_policy import (ENGINE, HAND as POLICY_HAND, _ad, _counter_scene, _giant,  # noqa: E402
                               _state as _policy_state)
from tests.test_live_loop import FakeBuilder, FakeExecutor, _argv, _loop  # noqa: E402
from tools.eval.eval_act_head import pick_threshold  # noqa: E402
from tools.train import train_act_gbdt as tag  # noqa: E402

CARDS = load_cards()
RULES = ReactivePolicy(CARDS)
DECK = ["zap", "archers", "knight", "giant", "miner", "tornado", "bomb-tower", "berserker"]


def F(state: dict, t: float) -> dict[str, float]:
    return dict(zip(FEATURES, act_features(state, t)))


# 1 features --------------------------------------------------------------------------------------------------------
def test_features_order_and_length_match_the_names():
    assert len(FEATURES) == len(set(FEATURES)) == 29
    row = act_features(_with(), 10.0)
    assert len(row) == len(FEATURES) and all(isinstance(v, float) for v in row)
    assert F(_with(elixir=7.0), 10.0)["own_elixir"] == 7.0 and row[FEATURES.index("own_elixir")] == 5.0


def _hand_made() -> dict:
    def unit(card, side, tile, eta=None):
        return {"unit_class": card or "unknown_unit", "card": card, "side": side, "tile": tile,
                "eta_tower": None if eta is None else {"tower": "own_left", "s": eta}}

    return {
        "phase": "double_elixir", "match_seconds": 100, "clock": "1:20",
        "own": {"elixir": 6.0, "hand": ["tornado", "bomb-tower", "berserker", None], "next_card": "royal-delivery",
                "deck": DECK, "towers": {"king": None, "left": 4000, "right": 2500},
                "recent_plays": [{"t": 60.0, "card": "archers", "tile": [8, 5], "elixir_cost": 3.0}]},
        "opponent": {"elixir_estimate": 4.0, "towers": {"king": None, "left": 5000, "right": None},
                     "deck_known": [],
                     "recent_plays": [{"t": 90.0, "card": "knight", "tile": [3, 20], "elixir_cost": 3.0},
                                      {"t": 97.0, "card": "miner", "tile": [5, 14], "elixir_cost": 3.0}]},
        "units": [unit("knight", "enemy", [3, 12], eta=3.0), unit("knight", "enemy", [5, 13], eta=6.0),
                  unit("giant", "enemy", [9, 20]), unit("hog-rider", "ally", [8, 18]),
                  unit("archers", "ally", [8, 5]), unit(None, "ally", [2, 3])],
        "threats": ["knight(e) advancing left lane, tower in 3.0s", "x"],
    }


def test_features_of_a_hand_made_state():
    f = F(_hand_made(), 100.0)
    assert f == {
        # elixir
        "own_elixir": 6.0, "opp_elixir_est": 4.0, "elixir_diff": 2.0, "own_full": 0.0,
        # time
        "phase_idx": 1.0, "match_seconds": 100.0,
        # enemy: 3 units (two knights on rows 12 / 13, a giant on row 20), nearest eta 3 s,
        # distinct cards knight 3 + giant 5, only the knight on our half
        "enemy_count": 3.0, "enemy_count_own_half": 2.0, "enemy_min_eta_own": 3.0,
        "enemy_elixir_field": 8.0, "enemy_elixir_own_half": 3.0,
        # ally: 3 units (one unread card), the hog rider on row 18 is past the river, hog 4 + archers 3
        "ally_count": 3.0, "ally_count_enemy_half": 1.0, "ally_elixir_field": 7.0,
        # plays: their newest is the miner at t=97 (3 s ago, cost 3), the knight at t=90 is outside the 5 s window;
        # our only play was 40 s ago (capped to 20) and cost nothing in the last 5 s
        "since_opp_play": 3.0, "opp_last_cost": 3.0, "opp_spent_5s": 3.0, "since_own_play": 20.0, "own_spent_5s": 0.0,
        # hand: tornado 3, bomb-tower 4, berserker 2 (one slot unread); all affordable at 6; next card costs 3
        "hand_known": 3.0, "hand_cost_min": 2.0, "hand_cost_mean": 3.0, "hand_affordable": 3.0, "next_card_cost": 3.0,
        # towers: min princess HP / 5000; their right tower is gone from a live HUD
        "own_princess_hp_min": 0.5, "opp_princess_hp_min": 1.0, "own_towers_down": 0.0, "opp_towers_down": 1.0,
        "n_threats": 2.0}


def test_spend_window_is_inclusive_and_counts_every_play_in_it():
    state = _hand_made()
    state["opponent"]["recent_plays"] = [{"t": 95.0, "card": "knight"}, {"t": 98.0, "card": "giant"},
                                         {"t": 99.9, "card": None}]
    f = F(state, 100.0)
    assert f["opp_spent_5s"] == 3.0 + 5.0                    # t=95 is exactly 5 s ago: inside; the unknown card is free
    assert f["since_opp_play"] == pytest.approx(0.1) and f["opp_last_cost"] == 0.0     # newest play: unknown card


def test_eta_is_capped_and_defaults_to_the_cap():
    far = {**_with(), "units": [_enemy((5, 25), eta=35.0)]}
    assert F(far, 10.0)["enemy_min_eta_own"] == 20.0
    assert F(_with(), 10.0)["enemy_min_eta_own"] == 20.0 and F(_with(), 10.0)["since_opp_play"] == 20.0
    mixed = {**_with(), "units": [_enemy((5, 25), eta=35.0), _enemy((5, 9), eta=2.5), _enemy((5, 12), eta=None)]}
    assert F(mixed, 10.0)["enemy_min_eta_own"] == 2.5
    ally_eta = {"unit_class": "hog-rider", "card": "hog-rider", "side": "enemy", "tile": [5, 9],
                "eta_tower": {"tower": "enemy_left", "s": 1.0}}      # heading for THEIR tower: not our danger
    assert F({**_with(), "units": [ally_eta]}, 10.0)["enemy_min_eta_own"] == 20.0


def test_since_play_is_capped_and_never_negative():
    state = _hand_made()
    state["own"]["recent_plays"] = [{"t": 5.0, "card": "knight"}]
    assert F(state, 100.0)["since_own_play"] == 20.0
    state["own"]["recent_plays"] = [{"t": 101.0, "card": "knight"}]     # a play stamped after `t`
    assert F(state, 100.0)["since_own_play"] == 0.0


def test_unknown_values_map_to_minus_one():
    towers = {"king": None, "left": None, "right": None}
    state = {"phase": None, "match_seconds": None,
             "own": {"elixir": None, "hand": [None] * 4, "next_card": None, "towers": dict(towers),
                     "recent_plays": []},
             "opponent": {"elixir_estimate": None, "towers": dict(towers), "recent_plays": []},
             "units": [], "threats": []}
    assert F(state, 3.0) == {
        "own_elixir": -1.0, "opp_elixir_est": -1.0, "elixir_diff": 0.0, "own_full": 0.0,
        "phase_idx": -1.0, "match_seconds": -1.0,
        "enemy_count": 0.0, "enemy_count_own_half": 0.0, "enemy_min_eta_own": 20.0, "enemy_elixir_field": 0.0,
        "enemy_elixir_own_half": 0.0, "ally_count": 0.0, "ally_count_enemy_half": 0.0, "ally_elixir_field": 0.0,
        "since_opp_play": 20.0, "opp_last_cost": 0.0, "opp_spent_5s": 0.0, "since_own_play": 20.0,
        "own_spent_5s": 0.0, "hand_known": 0.0, "hand_cost_min": -1.0, "hand_cost_mean": -1.0,
        "hand_affordable": 0.0, "next_card_cost": -1.0,
        "own_princess_hp_min": -1.0, "opp_princess_hp_min": -1.0, "own_towers_down": 0.0, "opp_towers_down": 0.0,
        "n_threats": 0.0}


def test_phase_index_and_full_flag():
    for phase, idx in (("single_elixir", 0), ("double_elixir", 1), ("triple_elixir", 2), ("overtime", 3)):
        assert F({**_with(), "phase": phase}, 1.0)["phase_idx"] == idx
    assert F(_with(elixir=9.5), 1.0)["own_full"] == 1.0 and F(_with(elixir=9.4), 1.0)["own_full"] == 0.0


# 2 labels ----------------------------------------------------------------------------------------------------------
def _line(i: int, split: str, nd: float | None, act: bool | None = None) -> dict:
    act = (nd is not None and nd <= 1.0) if act is None else act
    return {"id": f"g{split}#{i:.1f}", "game_id": f"g{split}", "split": split, "kind": "dense", "variant": "act",
            "state": "ctx", "questions": {}, "answers": {"act": "true" if act else "false"},
            "meta": {"since_opp_play": 2.0, "threat": False, "own_elixir": 5.0, "opp_elixir": 4.0,
                     "next_deploy_in": nd}}


def test_act_1_is_answers_act_and_horizons_are_nested():
    nds = [None, 0.4, 1.0, 1.01, 1.5, 2.0, 2.5, 3.0, 5.0]
    lines = [_line(i, "val", nd) for i, nd in enumerate(nds)]
    lines.append(_line(99, "val", 1.0, act=False))        # build_dense_act rounds next_in to 0.01: 1.004 -> 1.0, act False
    for line in lines:
        a1, a2, a3 = (tag.act_label(line, h) for h in (1.0, 2.0, 3.0))
        assert a1 == (line["answers"]["act"] == "true")
        assert (not a1 or a2) and (not a2 or a3)             # act_1 => act_2 => act_3
    assert [tag.act_label(l, 2.0) for l in lines[:9]] == [False, True, True, True, True, True, False, False, False]
    assert [tag.act_label(l, 3.0) for l in lines[:9]] == [False, True, True, True, True, True, True, True, False]
    assert tag.act_label(lines[-1], 1.0) is False and tag.act_label(lines[-1], 2.0) is True     # edge line
    assert tag.label_check(lines) == {"n": 10, "meta_mismatch": 1}
    assert tag.act_label(lines[4], 0.5) is False and tag.act_label(lines[1], 0.5) is True       # shorter than 1 s


# 3 trainer ---------------------------------------------------------------------------------------------------------
def _syn_state(elixir: float, t: float) -> dict:
    return {"phase": "double_elixir", "match_seconds": int(t), "clock": "1:00",
            "own": {"elixir": elixir, "hand": ["zap", "archers", "knight", "giant"], "next_card": None, "deck": DECK,
                    "towers": {"king": None, "left": 4000, "right": 5000}, "recent_plays": []},
            "opponent": {"elixir_estimate": 4.0, "towers": {"king": None, "left": 3000, "right": 4858},
                         "deck_known": [], "recent_plays": []},
            "units": [], "threats": []}


def _write_dense(root: Path, n: dict[str, int]) -> Path:
    """Toy dense set: the pro deploys within 1 s when own elixir >= 6, within 2 s from 4, within 3 s from 2."""
    root.mkdir(parents=True, exist_ok=True)
    for k, (split, count) in enumerate(n.items()):
        rng = random.Random(k)
        lines, side = [], []
        for i in range(count):
            e = round(rng.uniform(0.0, 10.0), 2)
            nd = 0.5 if e >= 6 else 1.8 if e >= 4 else 2.8 if e >= 2 else None
            line = _line(i, split, nd)
            lines.append(line)
            side.append({"id": line["id"], "t": float(i), "state": _syn_state(e, float(i))})
        (root / f"{split}.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        (root / f"states_{split}.jsonl").write_text("\n".join(json.dumps(s) for s in side) + "\n")
    return root


def _fake_d1_json(path: Path) -> Path:
    pol = {"n": 1, "accuracy": 0.5, "balanced_accuracy": 0.55, "act_rate": 0.5, "pro_act_rate": 0.3,
           "act_precision": 0.3, "act_recall": 0.6, "wait_precision": 0.8, "wait_recall": 0.5}
    react = {"n": 1, "act_rate": 0.5, "pro_act_rate": 0.3, "act_recall": 0.6, "act_precision": 0.3}
    block = {"policies": {"learned": pol, "rules": pol, "combined": pol},
             "reaction": {"policies": {"learned": react, "rules": react, "combined": react}}}
    path.write_text(json.dumps({"threshold": 0.5, "splits": {"val": block, "test": block}}))
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("act_gbdt")
    data = _write_dense(root / "ds" / "act_dense", {"train": 600, "val": 300, "test": 300})
    bal = root / "ds" / "act_dense_bal"
    bal.mkdir()
    for split in ("val", "test"):                              # every other line, as a stand-in for the balanced subset
        lines = (data / f"{split}.jsonl").read_text().splitlines()[::2]
        (bal / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    out = root / "model"
    res = tag.run(data, out, [1.0, 2.0, 3.0], 0, bal, _fake_d1_json(root / "d1.json"), root / "ds" / "eval" / "act_gbdt.json",
                  quiet=True)
    return {"root": root, "data": data, "out": out, "res": res, "bal": bal}


def test_trainer_writes_models_meta_and_reports(trained):
    out, res = trained["out"], trained["res"]
    for h in ("1", "2", "3"):
        assert (out / f"model_h{h}.txt").is_file()
        dump = json.loads((out / f"model_h{h}.json").read_text())          # the runtime reads these, not the .txt
        assert dump["objective"].startswith("binary") and dump["tree_info"] and dump["feature_names"] == FEATURES
    meta = json.loads((out / "meta.json").read_text())
    assert meta["json_files"] == {"1": "model_h1.json", "2": "model_h2.json", "3": "model_h3.json"}
    assert meta["features"] == FEATURES and meta["horizons"] == [1.0, 2.0, 3.0] and meta["default_horizon"] == 2.0
    assert set(meta["thresholds"]) == {"1", "2", "3"}
    assert all(0.05 <= t <= 0.95 for t in meta["thresholds"].values())
    assert meta["params"]["n_jobs"] == 1 and meta["params"]["is_unbalance"] and meta["params"]["num_leaves"] == 31
    assert meta["params"]["learning_rate"] == 0.05 and meta["params"]["min_data_in_leaf"] == 50
    assert meta["params"]["max_rounds"] == 2000 and meta["params"]["early_stopping_rounds"] == 50
    assert set(meta["metrics"]["2"]) >= {"threshold", "val", "test", "best_iteration", "pro_act_rate"}
    report = trained["root"] / "ds" / "eval"
    assert json.loads((report / "act_gbdt.json").read_text())["features"] == FEATURES
    md = (report / "act_gbdt.md").read_text()
    assert "Horizon 2 s" in md and "Feature importance" in md and "Predict latency" in md
    assert res["label_check"]["train"]["meta_mismatch"] == 0


def test_separable_toy_feature_is_learned_and_the_threshold_is_picked_on_val(trained):
    res = trained["res"]
    for key in ("1", "2", "3"):
        h = res["horizons"][key]
        assert h["splits"]["test"]["auc"] > 0.95
        assert h["splits"]["test"]["policies"]["gbdt"]["balanced_accuracy"] > 0.9
        assert h["importance"][0]["feature"] in ("own_elixir", "elixir_diff", "hand_affordable")
    # the stored threshold is the best-balanced-accuracy one of the saved model's own val predictions
    model = ActGBDT.load(trained["out"], 1.0)
    side = [json.loads(l) for l in (trained["data"] / "states_val.jsonl").read_text().splitlines()]
    lines = [json.loads(l) for l in (trained["data"] / "val.jsonl").read_text().splitlines()]
    probs = [model.p_act(s["state"], s["t"]) for s in side]
    thr, _ = pick_threshold([l["answers"]["act"] == "true" for l in lines], probs)
    assert thr == pytest.approx(model.threshold) == pytest.approx(res["horizons"]["1"]["threshold"])


def test_trainer_bal_comparison_lists_the_d1_rows_and_the_subset_size(trained):
    bal = trained["res"]["bal"]
    assert bal["splits"]["val"]["n"] == 150 and bal["splits"]["test"]["n"] == 150
    rows = bal["splits"]["val"]["rows"]
    assert {"d1 learned", "d1 rules", "d1 combined", "rules (recomputed)", "gbdt", "fast (rules + gbdt)",
            "gbdt@bal", "always", "never"} <= set(rows)
    assert bal["d1_threshold"] == 0.5 and bal["splits"]["test"]["reaction"]["rows"]["d1 learned"]["act_rate"] == 0.5
    assert "Apples to apples" in (trained["root"] / "ds" / "eval" / "act_gbdt.md").read_text()


def test_trainer_main_defaults_and_cli(tmp_path):
    data = _write_dense(tmp_path / "ds" / "act_dense", {"train": 300, "val": 150, "test": 150})
    out = tmp_path / "m"
    res = tag.main(["--data", str(data), "--out", str(out), "--horizons", "2,1", "--seed", "3"])
    assert (out / "model_h1.txt").is_file() and (out / "model_h2.txt").is_file() and not (out / "model_h3.txt").exists()
    assert (out / "model_h1.json").is_file() and (out / "model_h2.json").is_file() and not (out / "model_h3.json").exists()
    assert (tmp_path / "ds" / "eval" / "act_gbdt.json").is_file()            # default report next to the data dir
    assert list(res["horizons"]) == ["1", "2"] and res["bal"] is None and res["seed"] == 3
    with pytest.raises(SystemExit):
        tag.main(["--data", str(data), "--out", str(out), "--horizons", "0"])


def test_trainer_stops_when_the_label_semantics_changed(tmp_path):
    data = _write_dense(tmp_path / "ds" / "act_dense", {"train": 100, "val": 50, "test": 50})
    path = data / "train.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    for l in lines:
        l["answers"]["act"] = "false" if l["answers"]["act"] == "true" else "true"
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    with pytest.raises(SystemExit, match="label semantics"):
        tag.run(data, tmp_path / "m", [1.0], 0, None, None, tmp_path / "r.json", quiet=True)


def test_trainer_missing_sidecar_fails_clearly(tmp_path):
    data = _write_dense(tmp_path / "ds" / "act_dense", {"train": 60, "val": 30, "test": 30})
    (data / "states_val.jsonl").write_text("")
    with pytest.raises(SystemExit, match="states_val.jsonl"):
        tag.run(data, tmp_path / "m", [1.0], 0, None, None, tmp_path / "r.json", quiet=True)


def test_auc_ranks_with_ties():
    import numpy as np
    y = np.array([0, 0, 1, 1])
    assert tag.auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert tag.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert tag.auc(np.array([1, 1]), np.array([0.1, 0.2])) is None


# 4 model ----------------------------------------------------------------------------------------------------------
def test_load_default_and_explicit_horizon(trained):
    default = ActGBDT.load(trained["out"])
    meta = json.loads((trained["out"] / "meta.json").read_text())
    assert default.horizon == 2.0 and default.threshold == meta["thresholds"]["2"]
    h3 = ActGBDT.load(trained["out"], horizon=3.0)
    assert h3.horizon == 3.0 and h3.threshold == meta["thresholds"]["3"]
    state = _syn_state(8.0, 10.0)
    p_hi, p_lo = default.p_act(state, 10.0), default.p_act(_syn_state(0.5, 10.0), 10.0)
    assert 0.0 <= p_lo < p_hi <= 1.0
    assert default.p_act(state, 10.0) == pytest.approx(
        float(default.trees.predict(np.array([act_features(state, 10.0)]))[0]), abs=1e-12)


def test_load_rejects_stale_features_and_unknown_horizons(trained, tmp_path):
    import shutil
    stale = tmp_path / "stale"
    shutil.copytree(trained["out"], stale)
    meta = json.loads((stale / "meta.json").read_text())
    with pytest.raises(ValueError, match="horizon 5"):
        ActGBDT.load(stale, horizon=5.0)
    meta["features"] = meta["features"][:-1]
    (stale / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="FEATURES"):
        ActGBDT.load(stale)
    with pytest.raises(OSError):
        ActGBDT.load(tmp_path / "missing")


def test_p_act_on_one_row_is_fast(trained):
    model = ActGBDT.load(trained["out"])
    state = _syn_state(5.0, 10.0)
    model.p_act(state, 10.0)
    t0 = time.perf_counter()
    for _ in range(200):
        model.p_act(state, 10.0)
    assert (time.perf_counter() - t0) / 200 < 0.005            # the trainer reports < 1 ms; keep headroom for a busy CI box


# 4b tree walker (no lightgbm in the runtime) ------------------------------------------------------------------------
def _py(code: str, *args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter from the repo root (so `clashbot` and `tests` import)."""
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, cwd=ROOT,
                          timeout=timeout)


_LGB_PREDICT = """
import sys
import lightgbm as lgb
import numpy as np
X = np.load(sys.argv[1])
np.savez(sys.argv[2], *[lgb.Booster(model_file=p).predict(X, num_threads=1) for p in sys.argv[3:]])
"""


def lightgbm_predictions(models: list[Path], X, tmp: Path) -> list:
    """LightGBM's own `Booster.predict` of each text model on X, computed in a subprocess (lightgbm and torch must not
    meet in one process)."""
    np.save(tmp / "X.npy", np.asarray(X, dtype=np.float64))
    proc = _py(_LGB_PREDICT, str(tmp / "X.npy"), str(tmp / "ref.npz"), *map(str, models))
    assert proc.returncode == 0, proc.stderr[-800:]
    ref = np.load(tmp / "ref.npz")
    return [ref[f"arr_{i}"] for i in range(len(models))]


def _leaf(i: int, v: float) -> dict:
    return {"leaf_index": i, "leaf_value": v}


def _split(i: int, feat: int, thr: float, left: dict, right: dict, default_left: bool = True,
           missing: str = "None") -> dict:
    return {"split_index": i, "split_feature": feat, "threshold": thr, "decision_type": "<=",
            "default_left": default_left, "missing_type": missing, "left_child": left, "right_child": right}


def _dump(*roots: dict, objective: str = "binary sigmoid:1", **extra) -> dict:
    return {"objective": objective, "num_class": 1, "num_tree_per_iteration": 1, "max_feature_idx": 1,
            "tree_info": [{"tree_index": i, "tree_structure": r} for i, r in enumerate(roots)], **extra}


NAN = float("nan")


def _raw(root: dict, *row: float) -> float:
    return TreeEnsemble(_dump(root)).raw(list(row))


def test_walker_goes_left_iff_value_le_threshold():
    tree = _split(0, 0, 1.0, _leaf(0, -1.0), _leaf(1, 2.0))
    assert _raw(tree, 0.5, 0.0) == -1.0 and _raw(tree, 1.0, 0.0) == -1.0           # equality goes left
    assert _raw(tree, 1.0000001, 0.0) == 2.0 and _raw(tree, -5.0, 0.0) == -1.0


def test_walker_missing_type_none_reads_nan_as_zero_and_ignores_default_left():
    for default_left in (True, False):
        low = _split(0, 0, 0.5, _leaf(0, -1.0), _leaf(1, 2.0), default_left, "None")        # 0.0 <= 0.5: left
        high = _split(0, 0, -0.5, _leaf(0, -1.0), _leaf(1, 2.0), default_left, "None")      # 0.0 > -0.5: right
        assert _raw(low, NAN, 0.0) == -1.0 and _raw(high, NAN, 0.0) == 2.0
        assert _raw(low, 0.0, 0.0) == -1.0 and _raw(high, 0.0, 0.0) == 2.0                  # zero is an ordinary value


def test_walker_missing_type_nan_sends_nan_the_default_way_and_only_nan():
    for default_left, expected in ((True, -1.0), (False, 2.0)):
        tree = _split(0, 0, -0.5, _leaf(0, -1.0), _leaf(1, 2.0), default_left, "NaN")       # threshold would say: right
        assert _raw(tree, NAN, 0.0) == expected
        assert _raw(tree, 0.0, 0.0) == 2.0 and _raw(tree, -1.0, 0.0) == -1.0               # zero is not missing here


def test_walker_missing_type_zero_sends_zero_and_nan_the_default_way():
    for default_left, expected in ((True, -1.0), (False, 2.0)):
        tree = _split(0, 0, 0.5, _leaf(0, -1.0), _leaf(1, 2.0), default_left, "Zero")       # threshold would say: left
        for v in (0.0, NAN, 1e-36, -1e-36, 1e-35):                                          # |v| <= 1e-35: missing
            assert _raw(tree, v, 0.0) == expected, v
        assert _raw(tree, 2e-35, 0.0) == -1.0 and _raw(tree, 0.7, 0.0) == 2.0               # above: an ordinary value


def test_walker_sums_trees_and_applies_the_sigmoid_to_the_raw_score():
    a = _split(0, 0, 1.0, _leaf(0, -1.0), _leaf(1, 2.0))
    b = _split(0, 1, 0.0, _leaf(0, 0.25), _split(1, 0, 3.0, _leaf(1, 1.0), _leaf(2, 4.0)))
    const = {"leaf_value": 0.5}                                      # a tree that never split
    model = TreeEnsemble(_dump(a, b, const))
    rows = [[0.0, -1.0], [2.0, 1.0], [5.0, 1.0], [NAN, NAN]]
    raws = [-1.0 + 0.25 + 0.5, 2.0 + 1.0 + 0.5, 2.0 + 4.0 + 0.5, -1.0 + 0.25 + 0.5]
    for row, raw in zip(rows, raws):
        assert model.raw(row) == pytest.approx(raw, abs=1e-15)
        assert model.predict_row(row) == pytest.approx(1.0 / (1.0 + np.exp(-raw)), abs=1e-15)
    assert model.predict(np.array(rows)) == pytest.approx([model.predict_row(r) for r in rows], abs=1e-15)
    scaled = TreeEnsemble(_dump(a, objective="binary sigmoid:2.5"))
    assert scaled.predict_row([5.0, 0.0]) == pytest.approx(1.0 / (1.0 + np.exp(-2.5 * 2.0)), abs=1e-15)


def test_walker_handles_init_score_and_average_output():
    a = _split(0, 0, 1.0, _leaf(0, -1.0), _leaf(1, 2.0))
    b = {"leaf_value": 4.0}
    assert TreeEnsemble(_dump(a, b, init_score=0.5)).raw([0.0, 0.0]) == 0.5 - 1.0 + 4.0
    assert TreeEnsemble(_dump(a, b, average_output=True)).raw([0.0, 0.0]) == (-1.0 + 4.0) / 2
    assert TreeEnsemble(_dump(a, b, average_output=True)).predict(np.array([[5.0, 0.0]]))[0] == pytest.approx(
        1.0 / (1.0 + np.exp(-(2.0 + 4.0) / 2)))
    assert TreeEnsemble(_dump(a)).raw([0.0, 0.0]) == -1.0           # neither key: plain sum


def test_walker_rejects_what_it_cannot_evaluate():
    cat = _split(0, 0, 1.0, _leaf(0, -1.0), _leaf(1, 2.0))
    cat["decision_type"] = "=="
    with pytest.raises(ValueError, match="numerical"):
        TreeEnsemble(_dump(cat))
    for objective, extra in (("regression", {}), ("multiclass num_class:3", {"num_class": 3}),
                             ("binary sigmoid:1", {"num_tree_per_iteration": 2})):
        with pytest.raises(ValueError, match="objective"):
            TreeEnsemble(_dump({"leaf_value": 0.0}, objective=objective, **extra))


_LGB_TRAIN_MISSING = """
import json, sys
import lightgbm as lgb
import numpy as np
seed, zero_as_missing, dump_path, data_path = int(sys.argv[1]), bool(int(sys.argv[2])), sys.argv[3], sys.argv[4]
rng = np.random.default_rng(seed)
def draw(n):
    X = rng.normal(size=(n, 6))
    X[:, 4] = np.round(X[:, 4])                        # many exact zeros
    X[rng.random(X.shape) < 0.15] = np.nan
    X[rng.random(X.shape) < 0.15] = 0.0
    return X
X = draw(4000)
f = lambda c, nan: np.nan_to_num(X[:, c], nan=nan)
y = ((f(0, 1.5) + 0.5 * f(1, -1.0) + 0.8 * (X[:, 2] == 0) + 1.0 * np.isnan(X[:, 3]) - 0.5 * f(4, 0.0)
      + 0.3 * f(5, 0.0)) > 0.4).astype(int)
params = dict(objective="binary", num_leaves=15, min_data_in_leaf=20, verbose=-1, n_jobs=1, use_missing=True,
              zero_as_missing=zero_as_missing, seed=seed)
b = lgb.train(params, lgb.Dataset(X, y), 40)
Xt = draw(1500)
Xt[:8, 0] = [1e-36, -1e-36, 1e-35, -1e-35, 2e-35, -2e-35, 0.0, np.nan]      # around kZeroThreshold
json.dump(b.dump_model(), open(dump_path, "w"))
np.savez(data_path, X=Xt, p=b.predict(Xt, num_threads=1))
"""


@pytest.mark.parametrize("zero_as_missing, kind", [(0, "NaN"), (1, "Zero")])
def test_walker_equals_lightgbm_with_nans_and_zeros(tmp_path, zero_as_missing, kind):
    proc = _py(_LGB_TRAIN_MISSING, "7", str(zero_as_missing), str(tmp_path / "dump.json"), str(tmp_path / "d.npz"))
    assert proc.returncode == 0, proc.stderr[-800:]
    dump = json.loads((tmp_path / "dump.json").read_text())
    kinds: set = set()
    stack = [t["tree_structure"] for t in dump["tree_info"]]
    while stack:
        node = stack.pop()
        if "split_index" in node:
            kinds.add((node["missing_type"], node["default_left"]))
            stack += [node["left_child"], node["right_child"]]
    assert {(kind, True), (kind, False)} <= kinds, kinds            # both default directions of this missing type occur
    data = np.load(tmp_path / "d.npz")
    model = TreeEnsemble(dump)
    assert np.isnan(data["X"]).any() and (data["X"] == 0.0).any() and 0.05 < data["p"].mean() < 0.95
    assert np.abs(model.predict(data["X"]) - data["p"]).max() < 1e-9
    assert np.abs(np.array([model.predict_row(r) for r in data["X"].tolist()]) - data["p"]).max() < 1e-9


def test_walker_equals_lightgbm_on_the_trained_toy_models(trained, tmp_path):
    rng = random.Random(1)
    rows = [act_features(_syn_state(round(rng.uniform(0.0, 10.0), 2), float(i)), float(i)) for i in range(400)]
    hs = (1.0, 2.0, 3.0)
    refs = lightgbm_predictions([trained["out"] / f"model_h{h:g}.txt" for h in hs], rows, tmp_path)
    for h, ref in zip(hs, refs):
        model = ActGBDT.load(trained["out"], h)
        assert ref.std() > 0.05
        assert np.abs(model.trees.predict(np.array(rows)) - ref).max() < 1e-9
        assert np.abs(np.array([model.trees.predict_row(r) for r in rows]) - ref).max() < 1e-9


REAL_DENSE = ROOT / "dataset" / "v1" / "act_dense"
REAL_MODELS = ROOT / "db" / "models" / "act-gbdt"
HORIZONS = (1.0, 2.0, 3.0)


@pytest.mark.skipif(not ((REAL_DENSE / "states_val.jsonl").exists()
                         and all((REAL_MODELS / f"model_h{h:g}.txt").exists() for h in HORIZONS)
                         and (REAL_MODELS / "meta.json").exists()),
                    reason="dataset/v1/act_dense or db/models/act-gbdt not built")
def test_walker_equals_lightgbm_on_2000_real_val_rows_for_every_horizon(tmp_path):
    side = [json.loads(l) for l in (REAL_DENSE / "states_val.jsonl").read_text().splitlines() if l.strip()]
    sample = random.Random(0).sample(side, 2000)
    cards = load_cards()
    X = np.array([act_features(s["state"], s["t"], cards) for s in sample]).reshape(-1, len(FEATURES))
    model_dir = tmp_path / "model"                    # .txt + meta only: ActGBDT.load exports the .json in a subprocess
    model_dir.mkdir()
    for name in ["meta.json"] + [f"model_h{h:g}.txt" for h in HORIZONS]:
        shutil.copy(REAL_MODELS / name, model_dir / name)
    refs = lightgbm_predictions([model_dir / f"model_h{h:g}.txt" for h in HORIZONS], X, tmp_path)
    for h, ref in zip(HORIZONS, refs):
        model = ActGBDT.load(model_dir, h)
        assert (model_dir / f"model_h{h:g}.json").is_file()
        assert len(np.unique(ref)) > 3 and ref.std() > 0.01   # a real spread of probabilities, not a constant
        assert np.abs(model.trees.predict(X) - ref).max() < 1e-9
        assert np.abs(np.array([model.trees.predict_row(r) for r in X.tolist()]) - ref).max() < 1e-9
        assert all(model.p_act(s["state"], s["t"]) == pytest.approx(r, abs=1e-9)
                   for s, r in zip(sample[:200], ref[:200]))
        shipped = REAL_MODELS / f"model_h{h:g}.json"
        if shipped.exists():                              # the exported file in use is the one the exporter writes now
            assert json.loads(shipped.read_text()) == json.loads((model_dir / shipped.name).read_text())


_LOAD_AFTER_TORCH = """
import json, sys
import torch
from clashbot.act_model import ActGBDT
args = json.loads(sys.argv[2])
model = ActGBDT.load(sys.argv[1])
p = model.p_act(args["state"], args["t"])
assert 'lightgbm' not in sys.modules, 'lightgbm was imported in the live process'
print(repr(p))
"""


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_torch_first_then_load_and_p_act_does_not_crash_and_gives_the_same_value(trained):
    arg = json.dumps({"state": _syn_state(8.0, 10.0), "t": 10.0})
    proc = _py(_LOAD_AFTER_TORCH, str(trained["out"]), arg)
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-800:])         # 139 = the OpenMP segfault
    expected = ActGBDT.load(trained["out"]).p_act(_syn_state(8.0, 10.0), 10.0)
    assert float(proc.stdout.strip().splitlines()[-1]) == expected


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_a_missing_json_is_exported_in_a_subprocess_even_after_torch(trained, tmp_path):
    model_dir = tmp_path / "old"                      # a model directory from before the .json export existed
    shutil.copytree(trained["out"], model_dir, ignore=shutil.ignore_patterns("model_h*.json"))
    assert [p.name for p in model_dir.glob("*.json")] == ["meta.json"]
    arg = json.dumps({"state": _syn_state(8.0, 10.0), "t": 10.0})
    proc = _py(_LOAD_AFTER_TORCH, str(model_dir), arg)
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-800:])
    assert sorted(p.name for p in model_dir.glob("model_h*.json")) == ["model_h1.json", "model_h2.json", "model_h3.json"]
    for name in ("model_h1.json", "model_h2.json", "model_h3.json"):
        assert json.loads((model_dir / name).read_text()) == json.loads((trained["out"] / name).read_text())
    expected = ActGBDT.load(trained["out"]).p_act(_syn_state(8.0, 10.0), 10.0)
    assert float(proc.stdout.strip().splitlines()[-1]) == expected


def test_a_failing_export_or_no_model_files_raise_oserror(trained, tmp_path, monkeypatch):
    old = tmp_path / "old"
    shutil.copytree(trained["out"], old, ignore=shutil.ignore_patterns("model_h*.json"))
    monkeypatch.setattr(act_model, "TRAINER", tmp_path / "no_such_trainer.py")
    with pytest.raises(OSError, match="exporting"):
        ActGBDT.load(old)
    empty = tmp_path / "empty"
    shutil.copytree(trained["out"], empty, ignore=shutil.ignore_patterns("model_h*"))
    with pytest.raises(OSError):                       # neither .json nor .txt: nothing to export, nothing to read
        ActGBDT.load(empty)


def test_export_json_cli_rewrites_the_files_of_existing_models(trained, tmp_path):
    model_dir = tmp_path / "m"
    shutil.copytree(trained["out"], model_dir, ignore=shutil.ignore_patterns("model_h*.json"))
    assert tag.main(["--export-json", str(model_dir)]) == {}
    for name in ("model_h1.json", "model_h2.json", "model_h3.json"):
        assert json.loads((model_dir / name).read_text()) == json.loads((trained["out"] / name).read_text())
    with pytest.raises(SystemExit, match="model_h"):
        tag.main(["--export-json", str(tmp_path)])


def test_the_runtime_module_never_imports_lightgbm(trained):
    src = (ROOT / "clashbot" / "act_model.py").read_text()
    assert not re.search(r"^\s*(import|from)\s+lightgbm", src, re.M)
    code = ("import sys\n"
            "from clashbot.act_model import ActGBDT, FastActPolicy\n"
            "import clashbot.live\n"
            "import json\n"
            "state = json.loads(sys.argv[2])\n"
            "m = ActGBDT.load(sys.argv[1])\n"
            "d = FastActPolicy(m).decide(state, 10.0)\n"
            "assert d.reason.startswith('fast '), d\n"
            "assert 0.0 <= m.p_act(state, 10.0) <= 1.0\n"
            "assert 'lightgbm' not in sys.modules, 'lightgbm imported'\n"
            "print('ok')\n")
    proc = _py(code, str(trained["out"]), json.dumps(_syn_state(5.0, 10.0)))
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr[-800:]


# 5 policy ---------------------------------------------------------------------------------------------------------
class FakeModel:
    """p_act() -> a fixed p; counts the calls."""

    def __init__(self, p: float = 0.5, threshold: float = 0.5, horizon: float = 2.0):
        self.p, self.threshold, self.horizon = p, threshold, horizon
        self.calls: list[float] = []

    def p_act(self, state, t):
        self.calls.append(t)
        return self.p


def test_no_affordable_card_is_none_and_never_asks_the_model():
    model = FakeModel(1.0)
    d = FastActPolicy(model, CARDS).decide(_with(elixir=0.0), 10.0)
    assert d == ActDecision(False, "none", None, "no affordable card") and model.calls == []


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_defend_and_leak_pass_through_whatever_p_is(p):
    model = FakeModel(p)
    policy = FastActPolicy(model, CARDS, threshold=0.5)
    d = policy.decide(_with(units=[_enemy((5, 9), eta=2.0)]), 10.0)
    assert d.act and d.mode == "defend" and d.lane == "left"
    d = policy.decide(_with(elixir=10.0), 20.0, "ignored context")
    assert d.act and d.mode == "leak"
    assert model.calls == []                                   # the safety modes do not need the model


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_a_counter_passes_through_unchanged_and_the_model_is_not_called(p):
    # MAC-AG: their giant is answered by our tesla, our knight is across the bridge: the rules counter with the ram rider
    state = _counter_scene()
    rules = _ad(DEFAULT_WEIGHTS)
    rule = rules.decide(state, 100.0)
    assert rule.act and rule.mode == "counter" and rule.allowed_cards == frozenset({"ram-rider"})
    assert rule.engine_tiles == {"ram-rider": [3.5, 14.0]}
    model = FakeModel(p)
    d = FastActPolicy(model, CARDS, threshold=0.5, rules=_ad(DEFAULT_WEIGHTS)).decide(state, 100.0, "ignored context")
    assert d == rule and model.calls == []                     # allowed cards and engine tiles survive, no model call


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_a_save_is_a_rule_level_hold_the_model_never_overrides(p):
    # MAC-AC: the rules wait for a better answer (giant 3.5 s out, 3 elixir, bomb tower one elixir away)
    # (the engine is the default-weights one of test_policy: the test reads its scores)
    state = _policy_state(elixir=3.0, units=[_giant(eta=3.5, tile=(14, 10))], hand=POLICY_HAND)
    rules = ReactivePolicy(CARDS, engine=ENGINE)
    rule = rules.decide(state, 10.0)
    assert not rule.act and rule.mode == "save" and rule.reason.startswith("save for bomb-tower")
    model = FakeModel(p)
    d = FastActPolicy(model, CARDS, threshold=0.5, rules=rules).decide(state, 10.0)
    assert d == rule and model.calls == []                     # passed through unchanged, the model is never asked


def test_p_at_or_above_the_threshold_acts_below_holds():
    d = FastActPolicy(FakeModel(0.7), CARDS, threshold=0.5).decide(_with(), 10.0)
    assert d.act and d.mode == "learned" and d.lane is None and d.reason == "fast act p=0.70 h=2s"
    assert FastActPolicy(FakeModel(0.5), CARDS, threshold=0.5).decide(_with(), 10.0).act
    d = FastActPolicy(FakeModel(0.49), CARDS, threshold=0.5).decide(_with(), 10.0)
    assert d == ActDecision(False, "hold", None, "fast wait p=0.49 h=2s")


def test_threshold_and_horizon_come_from_the_model_unless_given():
    policy = FastActPolicy(FakeModel(0.3, threshold=0.25, horizon=3.0), CARDS)
    assert policy.threshold == 0.25 and policy.decide(_with(), 10.0).reason == "fast act p=0.30 h=3s"
    assert FastActPolicy(FakeModel(0.3, threshold=0.25), CARDS, threshold=0.35).decide(_with(), 10.0).act is False


def test_punish_keeps_the_rules_mode_and_lane_when_the_model_says_act():
    plays = [{"t": 8.0, "card": "knight", "tile": [14, 24]}]
    state = {**_with(elixir=7.0), "opponent": {**_with()["opponent"], "elixir_estimate": 2.5, "recent_plays": plays}}
    assert RULES.decide(state, 10.0).mode == "punish"
    d = FastActPolicy(FakeModel(0.9), CARDS).decide(state, 10.0)
    assert d.act and d.mode == "punish" and d.lane == "left" and d.reason == "fast act p=0.90 h=2s"
    d = FastActPolicy(FakeModel(0.1), CARDS).decide(state, 10.0)           # the model can veto a punish
    assert not d.act and d.mode == "hold" and d.lane is None


def test_learned_mode_drops_blind_spells_and_other_modes_use_the_rules():
    # one enemy far away on their half: ETA <= 10 s is a push now (MAC-AD PUSH_ETA_S), so the knight is 12 s out
    # a whole play of bats and a spear goblin: a zap kills six, 2.7 elixir (MAC-AM M2: a zap needs 2 kills and 2.5 of value;
    # not 2x its cost, so no `spell` act)
    swarm = ([_enemy((5 + 0.3 * i, 25), eta=12.0, card="bats") for i in range(5)]
             + [_enemy((4.5, 24.5), eta=12.0, card="spear-goblins")])
    state = _with(units=swarm)
    policy = FastActPolicy(FakeModel(0.9), CARDS)
    d = policy.decide(state, 10.0)
    assert d.mode == "learned"
    cands = _cands("zap|right|mid", "zap|left|enemy_half", "knight|right|back", "giant|left|bridge")
    assert [c.key for c in policy.filter_candidates(cands, d, state)] == [
        "zap|left|enemy_half", "knight|right|back", "giant|left|bridge"]
    threat = _with(units=[_enemy((5, 9), eta=2.0)])
    d = policy.decide(threat, 10.0)
    cands = _cands("knight|right|mid", "knight|left|mid", "giant|left|back", "zap|left|back")
    assert policy.filter_candidates(cands, d, threat) == RULES.filter_candidates(cands, d, threat)
    assert policy.target_tile("zap|left|enemy_half", state) == RULES.target_tile("zap|left|enemy_half", state)


def test_offline_fast_predictions_equal_the_policy_decisions():
    plays = [{"t": 8.0, "card": "knight", "tile": [14, 24]}]
    punish = {**_with(elixir=7.0), "opponent": {**_with()["opponent"], "elixir_estimate": 2.5, "recent_plays": plays}}
    states = [{"state": s, "t": 10.0} for s in (_with(), _with(elixir=0.0), _with(elixir=10.0), punish,
                                                _with(units=[_enemy((5, 9), eta=2.0)]))]
    decisions = tag.rule_decisions(states, RULES)
    for p in (0.2, 0.8):
        model = FakeModel(p)
        policy = FastActPolicy(model, CARDS, threshold=0.5)
        expected = [policy.decide(s["state"], s["t"]).act for s in states]
        assert tag.fast_predictions(decisions, [p] * len(states), 0.5) == expected


def test_live_loop_runs_the_fast_policy():
    model = FakeModel(0.9)
    ex = FakeExecutor()
    lp = _loop(builder=FakeBuilder(_with(), warmup=0), executor=ex, policy=FastActPolicy(model, CARDS, threshold=0.5))
    d = lp.step("f", 10.0, 0, 2.0)
    assert model.calls == [10.0]
    assert d is not None and d.mode == "learned" and d.reason == "fast act p=0.90 h=2s"
    assert d.key == "giant|left|back" and len(ex.calls) == 1             # the blind zap candidate was filtered out
    lp = _loop(builder=FakeBuilder(_with(), warmup=0), policy=FastActPolicy(FakeModel(0.1), CARDS))
    assert lp.step("f", 10.0, 0, 2.0) is None and lp.policy_holds == {"hold": 1}


# 6 live CLI -------------------------------------------------------------------------------------------------------
def test_cli_policy_choices_include_fast():
    assert build_parser().parse_args(_argv("--policy", "fast", "--act-model", "/m/gbdt")).policy == "fast"
    for name in ("reactive", "learned", "trigger"):
        assert build_parser().parse_args(_argv("--policy", name)).policy == name
    with pytest.raises(SystemExit):
        build_parser().parse_args(_argv("--policy", "bogus"))


def test_cli_fast_without_act_model_exits_2(capsys):
    assert main(_argv("--policy", "fast")) == 2
    assert "--act-model" in capsys.readouterr().err


def test_cli_fast_reads_the_threshold_from_meta_json(monkeypatch, tmp_path):
    seen: dict = {}
    _patch_main(monkeypatch, seen)
    loads: list = []

    def fake_load(cls, directory, horizon=None):
        loads.append((str(directory), horizon))
        return FakeModel(0.5, threshold=0.37)

    monkeypatch.setattr(ActGBDT, "load", classmethod(fake_load))
    assert _run_main(tmp_path, "--policy", "fast", "--act-model", "/m/gbdt") == 0
    policy = seen["policy"]
    assert isinstance(policy, FastActPolicy) and policy.threshold == 0.37 and loads == [("/m/gbdt", None)]
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["policy"] == "fast" and config["act_threshold"] == 0.37 and config["act_model"] == "/m/gbdt"
    assert _run_main(tmp_path, "--policy", "fast", "--act-model", "/m/gbdt", "--act-threshold", "0.2") == 0
    assert seen["policy"].threshold == 0.2                                  # the flag beats meta.json


def test_cli_fast_with_a_real_model_dir(monkeypatch, tmp_path, trained):
    seen: dict = {}
    _patch_main(monkeypatch, seen)
    assert _run_main(tmp_path, "--policy", "fast", "--act-model", str(trained["out"])) == 0
    meta = json.loads((trained["out"] / "meta.json").read_text())
    policy = seen["policy"]
    assert isinstance(policy.model, ActGBDT) and policy.horizon == 2.0
    assert policy.threshold == meta["thresholds"]["2"]
    assert policy.decide(_syn_state(0.4, 5.0) | {"threats": []}, 5.0).mode in ("none", "hold")


def test_cli_fast_with_a_bad_model_dir_exits_3(monkeypatch, tmp_path, capsys):
    seen: dict = {}
    _patch_main(monkeypatch, seen)
    assert _run_main(tmp_path, "--policy", "fast", "--act-model", str(tmp_path / "nope")) == 3
    assert "error" in capsys.readouterr().err and "policy" not in seen


# MAC-AL P3.6: a `spell` act is passed through like a counter ------------------------------------------------------------
def test_al_the_fast_policy_passes_a_spell_decision_through_without_asking_the_model():
    spell = ActDecision(True, "spell", "left", "spell: fireball night-witch + e-wizard: kills 4, ~9 elixir",
                        allowed_cards=frozenset({"fireball"}), engine_tiles={"fireball": [4.0, 20.5]},
                        tile_sources={"fireball": "spell"})

    class Rules:
        cards = CARDS

        def decide(self, state, t, context_text=None):
            return spell

    model = FakeModel(0.0)                                  # the model would wait
    policy = FastActPolicy(model, CARDS, rules=Rules())
    d = policy.decide(_with(), 10.0)
    assert d is spell and d.mode == "spell" and model.calls == []
    # the learned (d1 act-head) policy as well
    from clashbot.policy import LearnedActPolicy

    class Scorer:
        calls = 0

        def probabilities(self, state, questions):
            Scorer.calls += 1
            return {"act": [1.0, 0.0]}

    learned = LearnedActPolicy(Scorer(), CARDS, rules=Rules())
    assert learned.decide(_with(), 10.0, "CTX") is spell and Scorer.calls == 0


def test_al_a_spell_act_is_a_rule_act_for_the_trainer_and_is_counted_by_the_act_policy_eval():
    spell = ActDecision(True, "spell", "left", "spell: x", allowed_cards=frozenset({"fireball"}))
    counter = ActDecision(True, "counter", "left", "counter: x", allowed_cards=frozenset({"ram-rider"}))
    wait = ActDecision(False, "hold", None, "x")
    # whatever the model says (p = 0), a spell act is an act - like a counter; a hold follows the model
    assert tag.fast_predictions([spell, counter, wait], np.array([0.0, 0.0, 0.0]), 0.5) == [True, True, False]
    assert tag.fast_predictions([wait], np.array([0.9]), 0.5) == [True]
    from tools.eval import eval_act_policy
    assert "spell" in eval_act_policy.MODES and "counter" in eval_act_policy.MODES
