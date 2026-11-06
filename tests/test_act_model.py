"""MAC-AB A1: fast play/wait model - features, labels, the trainer, FastActPolicy and the live CLI.

Synthetic data only: a toy dense set where the pro acts when own elixir is high, tiny LightGBM models trained in a
second, a fake model with a fixed p for the policy tests. No real data, no torch, no GPU.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.act_model import FEATURES, ActGBDT, FastActPolicy, act_features  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live import build_parser, main  # noqa: E402
from clashbot.policy import ActDecision, ReactivePolicy  # noqa: E402
from tests.test_act_head import _cands, _enemy, _patch_main, _run_main, _with  # noqa: E402
from tests.test_policy import HAND as POLICY_HAND, _giant, _state as _policy_state  # noqa: E402
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
    meta = json.loads((out / "meta.json").read_text())
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
        float(default.booster.predict([act_features(state, 10.0)], num_threads=1)[0]))


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
def test_a_save_is_a_rule_level_hold_the_model_never_overrides(p):
    # MAC-AC: the rules wait for a better answer (giant 3.5 s out, 3 elixir, bomb tower one elixir away)
    state = _policy_state(elixir=3.0, units=[_giant(eta=3.5, tile=(14, 10))], hand=POLICY_HAND)
    rule = RULES.decide(state, 10.0)
    assert not rule.act and rule.mode == "save" and rule.reason.startswith("save for bomb-tower")
    model = FakeModel(p)
    d = FastActPolicy(model, CARDS, threshold=0.5).decide(state, 10.0)
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
    state = _with(units=[_enemy((5, 25), eta=6.0)])
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
