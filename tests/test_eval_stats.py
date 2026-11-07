"""Tests for tools/eval/eval_stats.py (MAC-AP): toy sets with known answers, then a tiny end-to-end run on fake files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval import eval_stats as es  # noqa: E402


# -- bootstrap ---------------------------------------------------------------------------------------------------------
def test_cluster_bootstrap_collapses_when_every_game_has_the_same_accuracy():
    games = np.repeat(np.arange(20), 5)
    values = np.tile([1.0, 1.0, 1.0, 0.0, 0.0], 20)            # 0.6 in every game: resampling games cannot move it
    ci = es.cluster_bootstrap_ci(values, games, n_boot=500)
    assert ci["est"] == pytest.approx(0.6)
    assert ci["lo"] == pytest.approx(0.6) and ci["hi"] == pytest.approx(0.6)


def test_cluster_bootstrap_interval_brackets_the_estimate_and_has_the_right_width():
    games = np.repeat(np.arange(20), 5)
    values = np.repeat(np.array([1.0, 0.0] * 10), 5)           # 10 games all right, 10 all wrong: p = 0.5, sd of a game = 0.5
    ci = es.cluster_bootstrap_ci(values, games, n_boot=4000)
    assert ci["est"] == pytest.approx(0.5)
    assert ci["lo"] < 0.5 < ci["hi"]
    half = 1.96 * np.sqrt(0.25 / 20)                            # 0.219: the normal-theory half width over 20 clusters
    assert ci["hi"] - ci["lo"] == pytest.approx(2 * half, abs=0.06)
    assert es.cluster_bootstrap_ci(np.ones(100), np.repeat(np.arange(10), 10), n_boot=200)["lo"] == 1.0


def test_bootstrap_is_reproducible_with_a_fixed_seed():
    games = np.repeat(np.arange(12), 4)
    v = np.random.default_rng(0).integers(0, 2, 48).astype(float)
    assert es.cluster_bootstrap_ci(v, games, 300, seed=7) == es.cluster_bootstrap_ci(v, games, 300, seed=7)


def test_paired_diff_uses_the_same_games_and_matches_the_hand_count():
    games = np.repeat(np.arange(10), 4)
    a = np.ones(40)
    b = np.tile([1.0, 1.0, 0.0, 0.0], 10)
    ctx = es.make_ctx(games, 500, 1)
    d = es.paired_diff(ctx, a, b)
    assert d["est"] == pytest.approx(0.5)
    assert d["lo"] == pytest.approx(0.5) and d["hi"] == pytest.approx(0.5)
    assert (d["mcnemar_a_only"], d["mcnemar_b_only"]) == (20, 0)
    assert d["mcnemar_p"] < 1e-5


# -- McNemar -----------------------------------------------------------------------------------------------------------
def test_mcnemar_exact_on_a_2x2():
    a = np.array([1] * 7 + [0] * 1 + [1] * 10 + [0] * 5)        # 7 a-only, 1 b-only, 10 both right, 5 both wrong
    b = np.array([0] * 7 + [1] * 1 + [1] * 10 + [0] * 5)
    r = es.mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (7, 1)
    assert r["p"] == pytest.approx(2 * (1 + 8) / 256)           # 2 * P(X <= 1), X ~ Binomial(8, 1/2) = 0.0703125


def test_mcnemar_exact_edge_cases():
    assert es.mcnemar_exact([1, 0, 1], [1, 0, 1])["p"] == 1.0                   # no discordant pair
    assert es.mcnemar_exact([1] * 5 + [0] * 5, [0] * 5 + [1] * 5)["p"] == 1.0   # 5 vs 5
    assert es.mcnemar_exact([1] * 10, [0] * 10)["p"] == pytest.approx(2 / 1024)


# -- calibration -------------------------------------------------------------------------------------------------------
def test_ece_of_a_perfectly_calibrated_toy_is_zero():
    conf = np.array([0.8] * 10 + [0.3] * 10 + [0.55] * 20)
    hit = np.array([1] * 8 + [0] * 2 + [1] * 3 + [0] * 7 + [1] * 11 + [0] * 9, dtype=float)
    ece, rows = es.expected_calibration_error(conf, hit)
    assert ece == pytest.approx(0.0, abs=1e-12)
    assert sum(r["n"] for r in rows) == 40 and len(rows) == es.ECE_BINS


def test_ece_of_an_overconfident_toy():
    ece, _ = es.expected_calibration_error(np.ones(10), np.array([1, 0] * 5, dtype=float))   # conf 1.0 -> last bin
    assert ece == pytest.approx(0.5)


def test_ece_ci_point_matches_and_brier_of_a_perfect_forecast_is_zero():
    games = np.repeat(np.arange(8), 5)
    rng = np.random.default_rng(3)
    conf, hit = rng.uniform(0.25, 1, 40), rng.integers(0, 2, 40).astype(float)
    ctx = es.make_ctx(games, 200, 2)
    assert es.ece_ci(ctx, conf, hit)["est"] == pytest.approx(es.expected_calibration_error(conf, hit)[0])
    probs = np.eye(4)[[0, 1, 2, 3]]
    assert es.brier_per_line(probs, np.array([0, 1, 2, 3])).sum() == 0.0
    assert es.brier_per_line(np.full((1, 4), 0.25), np.array([2]))[0] == pytest.approx(0.75)


# -- selective prediction ----------------------------------------------------------------------------------------------
def test_risk_coverage_and_the_val_chosen_threshold():
    conf, hit = np.array([0.9, 0.8, 0.7, 0.6]), np.array([1.0, 1.0, 0.0, 0.0])
    rc = es.risk_coverage(conf, hit)
    np.testing.assert_allclose(rc["coverage"], [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(rc["utility"], [0.25, 0.5, 0.25, 0.0])
    np.testing.assert_allclose(rc["risk"], [0.0, 0.0, 1 / 3, 0.5])
    assert es.choose_threshold(conf, hit) == 0.8
    assert es.choose_threshold(conf, np.zeros(4)) == float("inf")               # always wrong: abstain


def test_risk_coverage_answers_ties_together():
    rc = es.risk_coverage(np.array([0.5, 0.5, 0.5, 0.9]), np.array([1.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(rc["threshold"], [0.9, 0.5])
    np.testing.assert_allclose(rc["n_answered"], [1, 4])


# -- ensemble and terciles ---------------------------------------------------------------------------------------------
def test_zscore_rows():
    z = es.zscore_rows(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]]))
    assert z[0].mean() == pytest.approx(0.0) and z[0].std() == pytest.approx(1.0)
    assert (z[1] == 0).all()


def test_pick_ensemble_weight_sees_only_the_lines_it_is_given():
    labels = np.array([0, 1, 2, 3, 0, 1])
    good = np.eye(4)[labels] * 2.0                                                # d1: always right
    bad = np.eye(4)[(labels + 1) % 4] * 2.0                                       # b: always picks another option
    w, accs = es.pick_ensemble_weight(es.zscore_rows(good), es.zscore_rows(bad), labels)
    assert w > 0.5 and max(accs) == 1.0 and accs[0] == 0.0
    w2, _ = es.pick_ensemble_weight(es.zscore_rows(bad), es.zscore_rows(good), labels)
    assert w2 < 0.5
    w3, _ = es.pick_ensemble_weight(es.zscore_rows(good), es.zscore_rows(good), labels)
    assert w3 == 0.5                                                              # every weight ties: the even blend


def test_terciles():
    counts = np.array([0, 0, 0, 1, 2, 3, 10, 20, 30])
    edges = es.tercile_edges(counts)
    assert edges == (0.0, 3.0)                                                    # 'lower' quantiles of the 9 counts
    assert list(es.tercile_of(counts, edges)) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


# -- loaders and an end-to-end run on fake files -----------------------------------------------------------------------
def _fake_world(tmp_path: Path, n_games: int = 6, per_game: int = 6, seed: int = 0):
    """ranker/{val,test}.jsonl, placement_prior.json, states.jsonl and d1 eval JSONs for a tiny made-up task."""
    rng = np.random.default_rng(seed)
    cards = ["archers", "knight", "tesla", "zap", "musketeer", "giant"]
    pool = [f"{c}|{lane}|{zone}" for c in cards for lane in ("left", "right") for zone in ("back", "mid")]
    ds, docs = tmp_path / "ds", tmp_path / "docs"
    (ds / "ranker").mkdir(parents=True)
    docs.mkdir()
    with open(ds / "states.jsonl", "w") as f:
        for i in range(120):
            f.write(json.dumps({"split": "train", "train_eligible": True, "game_id": f"tr{i % 9}",
                                "action": {"key": pool[int(rng.integers(0, 6))] if i % 3 else pool[int(rng.integers(0, len(pool)))]}}) + "\n")
    (ds / "placement_prior.json").write_text(json.dumps({"zone_freq": {c: {"back": 3, "mid": 5} for c in cards}}))
    per_example, truth = [], {}
    for sp in ("val", "test"):
        rows = []
        for g in range(n_games):
            for j in range(per_game):
                keys = list(rng.choice(pool, 4, replace=False))
                label = int(rng.integers(0, 4))
                rows.append({"id": f"{sp}-g{g}#{j}", "game_id": f"{sp}-g{g}", "split": sp, "kind": "play",
                             "options": [{"key": k} for k in keys], "label": label, "positive_key": keys[label]})
                p = rng.dirichlet(np.ones(4))
                p[label] += 0.6
                p /= p.sum()
                per_example.append({"id": rows[-1]["id"], "variant": "joint", "split": sp, "pred": keys[int(np.argmax(p))],
                                    "gold": keys[label], "probs": p.tolist()})
        (ds / "ranker" / f"{sp}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        truth[sp] = rows
    def eval_json(name):
        (docs / name).write_text(json.dumps({"backend": "x", "variants": {"joint": {sp: {"acc": None} for sp in ("val", "test")}},
                                             "per_example": per_example}))
    eval_json("d1_finetune_joint_tac_eval_v2.json")
    eval_json("d1_finetune_joint_eval_v1.json")
    return ds, docs, truth


def test_load_d1_probs_aligns_by_id_and_rejects_a_mismatch(tmp_path):
    ds, docs, _ = _fake_world(tmp_path)
    lines = es.load_lines(ds)
    probs, _rec = es.load_d1_probs(docs / "d1_finetune_joint_tac_eval_v2.json", lines)
    assert probs["val"].shape == (36, 4) and np.allclose(probs["test"].sum(axis=1), 1.0)
    bad = json.loads((docs / "d1_finetune_joint_tac_eval_v2.json").read_text())
    bad["per_example"][0]["gold"] = "nope|left|mid"
    (docs / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        es.load_d1_probs(docs / "bad.json", lines)


def test_dump_preds_round_trips_into_the_audit_loader(tmp_path):
    from tools.eval import feature_ranker as fr
    ds, docs, truth = _fake_world(tmp_path)
    collected = {"lgbm_lambdarank": {s: {sp: np.random.default_rng(s).normal(size=(len(truth[sp]), 4)) for sp in ("val", "test")}
                                     for s in (0, 1)}}
    fr.dump_preds(docs / "p.json", truth, collected, [0, 1])
    lines = es.load_lines(ds)
    got = es.load_ranker_preds(docs / "p.json", lines)
    assert sorted(got["lgbm_lambdarank"]) == [0, 1]
    np.testing.assert_allclose(got["lgbm_lambdarank"][1]["test"], collected["lgbm_lambdarank"][1]["test"], atol=1e-6)


def test_main_end_to_end_writes_json_md_and_csv(tmp_path):
    from tools.eval import feature_ranker as fr
    ds, docs, truth = _fake_world(tmp_path)
    rng = np.random.default_rng(9)
    collected = {name: {s: {sp: rng.normal(size=(len(truth[sp]), 4)) + np.eye(4)[[r["label"] for r in truth[sp]]]
                            for sp in ("val", "test")} for s in (0, 1, 2)}
                 for name in ("logreg", "lgbm_lambdarank", "dcn_v2")}
    fr.dump_preds(docs / "feature_ranker_preds_v2.json", truth, collected, [0, 1, 2])
    out = docs / "stats_audit_v1.json"
    res = es.main(["--dataset", str(ds), "--docs-data", str(docs), "--out", str(out), "--boot", "300"])
    assert out.exists() and out.with_suffix(".md").exists() and (docs / "stats_audit_v1_risk_coverage.csv").exists()
    data = json.loads(out.read_text())
    assert data["n_lines"] == {"val": 36, "test": 36} and data["n_boot"] == 300
    for m in ("random", "popularity", "placement_prior", "logreg", "lgbm", "dcn_v2", "d1_joint", "d1_tac"):
        c = data["accuracy"]["test"][m]
        assert c["lo"] <= c["est"] <= c["hi"]
    assert {(r["a"], r["b"]) for r in data["paired"]["test"]} >= {("d1_tac", "lgbm"), ("d1_tac", "d1_joint"), ("d1_tac", "placement_prior")}
    assert 0 <= data["calibration"]["splits"]["test"]["ece"] <= 1 and "curves" not in data["selective"]
    assert data["ensemble"]["primary"]["weight_d1"] in es.WEIGHT_GRID
    assert "d1_kg" not in data["accuracy"]["test"] and any("d1_kg" in n for n in data["meta"]["notes"])
    assert res["accuracy"]["test"]["d1_tac"]["est"] == pytest.approx(data["accuracy"]["test"]["d1_tac"]["est"], abs=1e-6)
    assert (docs / "stats_audit_v1_risk_coverage.csv").read_text().startswith("model,split,threshold")


def test_ensemble_weight_does_not_depend_on_the_test_labels(tmp_path):
    ds, docs, truth = _fake_world(tmp_path)
    lines = es.load_lines(ds)
    probs, _ = es.load_d1_probs(docs / "d1_finetune_joint_tac_eval_v2.json", lines)
    rng = np.random.default_rng(4)
    gb = {sp: rng.normal(size=(len(truth[sp]), 4)) for sp in ("val", "test")}
    models = {"d1_tac": {"label": "d1", "kind": "prob", "scores": probs},
              "lgbm": {"label": "gb", "kind": "score", "scores": gb}}
    gold = {sp: np.zeros(len(truth[sp])) for sp in ("val", "test")}
    base = es.audit({sp: {"game_ids": lines[sp]["game_ids"], "labels": lines[sp]["labels"]} for sp in ("val", "test")},
                     models, gold, n_boot=100)
    flipped = {sp: {"game_ids": lines[sp]["game_ids"],
                    "labels": lines[sp]["labels"] if sp == "val" else (lines[sp]["labels"] + 1) % 4} for sp in ("val", "test")}
    other = es.audit(flipped, models, gold, n_boot=100)
    assert base["ensemble"]["primary"]["weight_d1"] == other["ensemble"]["primary"]["weight_d1"]
    assert base["selective"]["threshold"] == other["selective"]["threshold"]
    assert base["ensemble"]["primary"]["test"]["acc"]["est"] != other["ensemble"]["primary"]["test"]["acc"]["est"]
