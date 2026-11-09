"""Tests for tools/eval/bench_d1_models.py and tools/eval/score_decision_models.py (MAC-AU Part B): toy lines, fake rankers,
a toy checkpoint dir for the d1-2B loader fix, and a tiny end-to-end scorer run. No model is loaded."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval import bench_d1_models as bench  # noqa: E402
from tools.eval import score_decision_models as sc  # noqa: E402


# -- toy fixtures ------------------------------------------------------------------------------------------------------
def toy_lines() -> list[dict]:
    """6 lines in 3 games; options are card|lane|zone keys, the gold card is repeated in some wrong options."""
    out = []
    for i in range(6):
        game = f"g{i // 2}"
        keys = [f"knight|left|mid", f"archers|left|mid", f"knight|right|mid", f"giant|left|back"]
        out.append({"line_id": f"{game}#{i}", "game_id": game, "state_id": f"{game}#{i}", "kind": "play",
                    "gold": i % 4, "keys": keys, "notes": ["n"] * 4, "state": "s"})
    return out


def rows_for(lines: list[dict], preds: list[int | None], latency: float = 10.0, cost: float | None = 0.0) -> dict:
    return {ln["line_id"]: bench.pred_row(ln, p, None, latency, cost_usd=cost) for ln, p in zip(lines, preds)}


# -- scorer: hits ------------------------------------------------------------------------------------------------------
def test_pick_hits_counts_exact_card_and_null():
    lines = toy_lines()                                   # gold = 0,1,2,3,0,1 -> knight, archers, knight, giant, knight, archers
    preds = [0, 2, 0, 3, None, 9]                         # exact, wrong card, card only (knight), exact, null, out of range
    covered, hit, card = sc.pick_hits(rows_for(lines, preds), lines)
    assert len(covered) == 6
    assert hit.tolist() == [1, 0, 0, 1, 0, 0]
    assert card.tolist() == [1, 0, 1, 1, 0, 0]


def test_pick_hits_skips_lines_without_a_row():
    lines = toy_lines()
    rows = rows_for(lines[:4], [0, 1, 2, 3])
    covered, hit, _ = sc.pick_hits(rows, lines)
    assert [ln["line_id"] for ln in covered] == [ln["line_id"] for ln in lines[:4]]
    assert hit.tolist() == [1, 1, 1, 1]


def test_score_model_reports_accuracy_latency_cost_and_parse_failures():
    lines = toy_lines()
    rows = rows_for(lines, [0, 1, 2, None, 0, 1], latency=0.0, cost=0.0)
    for i, r in enumerate(rows.values()):
        r["latency_ms"] = 10.0 * (i + 1)
        r["cost_usd"] = 0.001
        r["in_tokens"], r["out_tokens"] = 100, 4
    rows[lines[3]["line_id"]]["error"] = "refusal"
    m = sc.score_model(rows, lines, sc.Boot(200, 0))
    assert m["n"] == 6 and m["n_games"] == 3
    assert m["acc"]["est"] == pytest.approx(5 / 6, abs=1e-4)           # pred 0 vs gold 0 ok; 1,2 ok; None wrong; 0 ok; 1 ok
    assert m["acc"]["lo"] <= m["acc"]["est"] <= m["acc"]["hi"]
    assert m["parse_failures"] == {"n": 1, "rate": pytest.approx(1 / 6, abs=1e-4)}
    assert m["errors"] == 1
    assert m["latency_ms"]["p50"] == pytest.approx(35.0) and m["latency_ms"]["mean"] == pytest.approx(35.0)
    assert m["latency_ms"]["p95"] == pytest.approx(57.5)
    assert m["cost_usd_per_decision"] == pytest.approx(0.001)
    assert m["in_tokens_mean"] == 100.0 and m["out_tokens_mean"] == 4.0
    assert set(m["per_game"]) == {"g0", "g1", "g2"}


def test_bootstrap_clusters_over_games():
    # every game has accuracy 0.5, so resampling whole games cannot move the estimate: the CI collapses to a point
    lines = [{"line_id": f"g{g}#{i}", "game_id": f"g{g}", "gold": 0, "keys": ["a|l|m", "b|l|m", "c|l|m", "d|l|m"]}
             for g in range(8) for i in range(4)]
    rows = {ln["line_id"]: {"pred": 0 if int(ln["line_id"].split("#")[1]) < 2 else 1} for ln in lines}
    m = sc.score_model(rows, lines, sc.Boot(500, 0))
    assert m["acc"]["est"] == pytest.approx(0.5)
    assert m["acc"]["lo"] == pytest.approx(0.5) and m["acc"]["hi"] == pytest.approx(0.5)


def test_bootstrap_is_seeded():
    lines = toy_lines()
    rows = rows_for(lines, [0, 0, 2, 3, 1, 1])
    a = sc.score_model(rows, lines, sc.Boot(300, 0))["acc"]
    b = sc.score_model(rows, lines, sc.Boot(300, 0))["acc"]
    assert a == b


def test_mcnemar_against_the_reference_uses_common_lines():
    lines = toy_lines()
    ref = rows_for(lines, [0, 1, 2, 3, 0, 1])                           # all right
    other = rows_for(lines[:5], [0, 0, 0, 0, 0])                        # partial: 5 lines, right on 0 and 4 only... and 2? gold 2 -> pred 0 wrong
    m = sc.score_model(other, lines, sc.Boot(200, 0), ref)
    v = m["vs_reference"]
    assert v["n_common"] == 5
    assert v["reference_only"] == 3 and v["model_only"] == 0           # model right on lines 0 and 4 (both also right in ref)
    assert v["mcnemar_p"] == pytest.approx(0.25)                       # exact two-sided: 2 * P(X <= 0 | n=3) = 0.25
    assert v["diff"]["est"] == pytest.approx(-3 / 5)


def test_reference_has_no_comparison_with_itself():
    lines = toy_lines()
    ref = rows_for(lines, [0, 1, 2, 3, 0, 1])
    assert "vs_reference" not in sc.score_model(ref, lines, sc.Boot(100, 0), ref)


def test_random_expected_is_a_quarter_and_card_share_follows_the_options():
    lines = toy_lines()
    r = sc.random_expected(lines)
    assert r["acc"]["est"] == pytest.approx(0.25)
    # knight gold: 2 of 4 options share the card; archers: 1 of 4; giant: 1 of 4
    gold_cards = [ln["keys"][ln["gold"]].split("|")[0] for ln in lines]
    want = np.mean([{"knight": 0.5, "archers": 0.25, "giant": 0.25}[c] for c in gold_cards])
    assert r["card_acc"]["est"] == pytest.approx(round(float(want), 4))


# -- scorer: loading and the end-to-end report -------------------------------------------------------------------------
def write_bench(tmp_path: Path, lines: list[dict], preds: dict[str, dict]) -> Path:
    (tmp_path / "preds").mkdir()
    (tmp_path / "lines.jsonl").write_text("".join(json.dumps(ln) + "\n" for ln in lines))
    for mid, rows in preds.items():
        (tmp_path / "preds" / f"{mid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    return tmp_path


def test_load_preds_checks_gold_drops_unknown_lines_and_keeps_the_last_duplicate(tmp_path):
    lines = toy_lines()
    rows = rows_for(lines, [0, 1, 2, 3, 0, 1])
    extra = bench.pred_row({"line_id": "other#1", "game_id": "x", "gold": 0}, 0, None, 1.0)
    dup = dict(rows[lines[0]["line_id"]], pred=3)
    bd = write_bench(tmp_path, lines, {"m": rows})
    with open(bd / "preds" / "m.jsonl", "a") as f:
        f.write(json.dumps(extra) + "\n" + json.dumps(dup) + "\n")
    preds, warnings = sc.load_preds(bd / "preds", lines)
    assert len(preds["m"]) == 6 and preds["m"][lines[0]["line_id"]]["pred"] == 3
    assert any("not benchmark lines" in w for w in warnings) and any("repeated" in w for w in warnings)

    bad = dict(rows[lines[1]["line_id"]], gold=3)
    (bd / "preds" / "bad.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="lines.jsonl says"):
        sc.load_preds(bd / "preds", lines)


def test_main_writes_json_and_markdown_on_partial_inputs(tmp_path):
    lines = toy_lines()
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    write_bench(bench_dir, lines, {
        "d1-ft-joint-tac": rows_for(lines, [0, 1, 2, 3, 0, 0], latency=60.0),
        "claude-haiku-5-5": rows_for(lines[:3], [0, 1, 3], latency=900.0, cost=0.0003),   # partial
        "my-new-model": rows_for(lines, [1, 1, 1, 1, 1, 1], latency=5.0),                 # not in the registry
    })
    (bench_dir / "preds" / "d1-ft-joint-tac.meta.json").write_text(json.dumps({"dtype": "auto", "secret": "x"}))
    out = tmp_path / "out" / "decision_models_v1"
    rep = sc.main(["--bench-dir", str(bench_dir), "--out", str(out), "--boot", "200"])
    assert out.with_suffix(".json").exists() and out.with_suffix(".md").exists()
    saved = json.loads(out.with_suffix(".json").read_text())
    assert saved["reference"] == "d1-ft-joint-tac"
    assert saved["models"]["d1-ft-joint-tac"]["acc"]["est"] == pytest.approx(5 / 6, abs=1e-4)
    assert saved["models"]["claude-haiku-5-5"]["n"] == 3
    assert saved["models"]["my-new-model"]["label"] == "my-new-model"
    assert "qwen3-4b" in saved["pending"] and "claude-haiku-5-5" not in saved["pending"]
    assert "random-expected" in saved["models"]
    md = out.with_suffix(".md").read_text()
    assert saved["runs"] == {"d1-ft-joint-tac": {"dtype": "auto"}} and "`d1-ft-joint-tac`: dtype auto" in md
    assert "3 (partial)" in md and "(reference)" in md and "0.0003" in md and "`qwen3-4b`" in md
    assert rep["lines"]["n"] == 6 and rep["lines"]["n_games"] == 3


def test_report_without_the_reference_model_still_scores(tmp_path):
    lines = toy_lines()
    preds = {"popularity": rows_for(lines, [0, 0, 0, 0, 0, 0])}
    rep = sc.build_report(lines, preds, n_boot=100)
    assert rep["reference"] is None
    assert "vs_reference" not in rep["models"]["popularity"]
    assert "(reference)" not in sc.render_markdown({**rep, "warnings": []})


# -- bench_d1_models: the line file ------------------------------------------------------------------------------------
def write_dataset(root: Path, rows: list[dict]) -> Path:
    (root / "d1" / "joint_tac_v2").mkdir(parents=True)
    (root / "d1" / "joint_tac_v2" / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "states.jsonl").write_text("".join(json.dumps({"id": r["id"], "game_id": r["game_id"]}) + "\n"
                                               for r in rows if r.get("has_state", True)))
    return root


def source_row(i: int, game: str, n_options: int = 4, has_state: bool = True) -> dict:
    keys = ["knight|left|mid", "archers|left|mid", "giant|right|back", "bats|right|mid"][:n_options]
    return {"id": f"{game}#{i}", "game_id": game, "kind": "play", "state": f"state {i}", "has_state": has_state,
            "questions": {"play": {"type": "choice", "instructions": "x", "criteria": {k: f"note {k}" for k in keys}}},
            "answers": {"play": keys[1]}}


def test_build_lines_keeps_four_way_lines_in_source_order(tmp_path):
    rows = [source_row(0, "a"), source_row(1, "b"), source_row(2, "a", n_options=3), source_row(3, "a")]
    ds = write_dataset(tmp_path, rows)
    lines = bench.build_lines(ds)
    assert [ln["line_id"] for ln in lines] == ["a#0", "b#1", "a#3"]
    ln = lines[0]
    assert ln["state_id"] == "a#0" and ln["gold"] == 1 and len(ln["keys"]) == 4 and len(ln["notes"]) == 4
    assert ln["keys"][ln["gold"]] == "archers|left|mid"
    assert [x["line_id"] for x in bench.build_lines(ds, games=("b",))] == ["b#1"]


def test_build_lines_requires_a_dataset_state(tmp_path):
    ds = write_dataset(tmp_path, [source_row(0, "a", has_state=False)])
    with pytest.raises(ValueError, match="no dataset state"):
        bench.build_lines(ds)


def test_write_lines_round_trips(tmp_path):
    ds = write_dataset(tmp_path / "ds", [source_row(0, "a"), source_row(1, "a")])
    path = tmp_path / "out" / "lines.jsonl"
    lines = bench.write_lines(path, dataset=ds)
    assert bench.load_lines(path) == lines


# -- bench_d1_models: predictions and baselines ------------------------------------------------------------------------
def test_pred_row_follows_the_contract():
    ln = toy_lines()[0]
    row = bench.pred_row(ln, 2, [0.1, 0.2, 0.3333333333, 0.4], 12.34567, raw="x" * 500)
    assert list(row) == list(bench.PRED_FIELDS)
    assert row["probs"][2] == 0.333333 and row["latency_ms"] == 12.3457 and len(row["raw"]) == 200
    assert bench.pred_row(ln, None, None, None, error="boom")["pred"] is None


def test_argmax_first_ties_go_to_the_earliest_option():
    assert bench.argmax_first([0.2, 0.4, 0.4, 0.1]) == 1
    assert bench.argmax_first([0, 0, 0, 0]) == 0


def test_baselines_are_deterministic_and_use_the_train_counts():
    lines = toy_lines()
    pop = {"archers|left|mid": 7, "knight|left|mid": 3}
    zone = {"knight": {"mid": 3, "back": 1}, "archers": {"mid": 1, "back": 1}, "giant": {"back": 4}}
    a = bench.baseline_rows(lines, pop, zone, seed=0)
    b = bench.baseline_rows(lines, pop, zone, seed=0)
    assert [r["pred"] for r in a["random"]] == [r["pred"] for r in b["random"]]
    assert all(r["pred"] == 1 for r in a["popularity"])                  # archers|left|mid has the most train plays
    assert all(r["pred"] == 3 for r in a["placement-prior"])             # giant always goes back: share 1.0
    assert all(r["latency_ms"] >= 0 for rows in a.values() for r in rows)
    assert bench.placement_score("unknown|left|mid", zone) == 0.0


def test_lightgbm_rows_filter_to_the_lines_and_check_the_options():
    lines = toy_lines()[:2]
    dump_lines = [{"id": ln["line_id"], "label": ln["gold"], "keys": ln["keys"]} for ln in toy_lines()]
    dump = {"lines": {"test": dump_lines},
            "models": {"lgbm_lambdarank": {"0": {"test": [[0.0, 3.0, 0.0, 0.0]] * 6}}}}
    rows = bench.lightgbm_rows(lines, dump)
    assert [r["pred"] for r in rows] == [1, 1] and sum(rows[0]["probs"]) == pytest.approx(1.0, abs=1e-4)
    assert [r["latency_ms"] for r in rows] == [None, None]                 # the dump holds scores only: not measured
    timed = bench.lightgbm_rows(lines, dump, latency_ms={lines[0]["line_id"]: 0.5, lines[1]["line_id"]: 0.7})
    assert [r["latency_ms"] for r in timed] == [0.5, 0.7]
    dump["lines"]["test"][0]["label"] = 3
    with pytest.raises(ValueError, match="disagrees"):
        bench.lightgbm_rows(lines, dump)


# -- bench_d1_models: the d1 runner (fake ranker) ----------------------------------------------------------------------
class FakeRanker:
    def __init__(self, fail_on: str | None = None):
        self.calls = 0
        self.fail_on = fail_on

    def probabilities(self, state: str, questions: dict) -> dict:
        self.calls += 1
        if self.fail_on and state == self.fail_on:
            raise RuntimeError("boom")
        keys = list(questions["play"]["criteria"])
        return {"play": [0.1, 0.6, 0.2, 0.1][:len(keys)]}


def test_run_d1_times_each_decision_after_the_warmup_and_records_failures(tmp_path):
    ds = write_dataset(tmp_path / "ds", [source_row(i, "a") for i in range(5)])
    lines = bench.build_lines(ds)
    ranker = FakeRanker(fail_on="state 3")
    rows = bench.run_d1("fake-d1", None, lines=lines, ranker=ranker, preds_dir=tmp_path / "preds", dataset=ds, warmup=2)
    assert ranker.calls == 2 + 5                                         # warm-up calls are not scored
    assert [r["pred"] for r in rows] == [1, 1, 1, None, 1]
    assert rows[3]["error"].startswith("RuntimeError") and rows[3]["latency_ms"] >= 0
    assert all(r["cost_usd"] == 0.0 for r in rows)
    saved = [json.loads(x) for x in (tmp_path / "preds" / "fake-d1.jsonl").read_text().splitlines()]
    assert saved == rows
    meta = json.loads((tmp_path / "preds" / "fake-d1.meta.json").read_text())
    assert meta["n"] == 5 and meta["warmup"] == 2
    # every gold is option 1, so the fake ranker is right except where it failed
    assert sum(r["pred"] == r["gold"] for r in rows) == 4


def test_run_d1_limit(tmp_path):
    ds = write_dataset(tmp_path / "ds", [source_row(i, "a") for i in range(5)])
    rows = bench.run_d1("fake-d1", None, lines=bench.build_lines(ds), ranker=FakeRanker(), limit=3,
                        preds_dir=tmp_path / "p", dataset=ds, warmup=1)
    assert len(rows) == 3


# -- the d1-2B loader fix ----------------------------------------------------------------------------------------------
def make_checkpoint(path: Path, auto_config: str) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "lfm3_vl", "auto_map": {
        "AutoConfig": auto_config, "AutoModel": "modeling_d1_lfm3_vl.D1Model"}}))
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "native_lfm3_vl.py").write_text("# native")
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer.json").chmod(0o444)                               # the real one is read-only
    return path


def test_needs_loader_fix_only_for_the_vendored_config_module():
    assert bench.needs_loader_fix({"auto_map": {"AutoConfig": "configuration_lfm3_vl.Lfm3VlConfig"}})
    assert not bench.needs_loader_fix({"auto_map": {"AutoConfig": "native_lfm3_vl.Lfm3VlConfig"}})
    assert not bench.needs_loader_fix({"architectures": ["D1Model"]})       # the 600M checkpoints
    assert not bench.needs_loader_fix(None)


def test_make_loadable_mirrors_without_touching_the_source(tmp_path):
    src = make_checkpoint(tmp_path / "ft", "configuration_lfm3_vl.Lfm3VlConfig")
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    dst = bench.make_loadable(src, tmp_path / "loadable")
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before      # source untouched
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["auto_map"]["AutoConfig"] == bench.GOOD_AUTO_CONFIG
    assert cfg["auto_map"]["AutoModel"] == "modeling_d1_lfm3_vl.D1Model" and cfg["model_type"] == "lfm3_vl"
    assert (dst / "model.safetensors").is_symlink() and (dst / "model.safetensors").read_bytes() == b"weights"
    assert not (dst / "native_lfm3_vl.py").is_symlink() and (dst / "tokenizer.json").read_text() == "{}"
    assert not bench.needs_loader_fix(cfg)
    bench.make_loadable(src, dst)                                         # again: no error, same result
    assert json.loads((dst / "config.json").read_text()) == cfg


def test_resolve_model_path_mirrors_only_when_needed(tmp_path):
    mirror = tmp_path / "mirror"
    bad = make_checkpoint(tmp_path / "bad", "configuration_lfm3_vl.Lfm3VlConfig")
    good = make_checkpoint(tmp_path / "good", "native_lfm3_vl.Lfm3VlConfig")
    path, note = bench.resolve_model_path(str(bad), mirror)
    assert path == str(mirror) and "loader fix" in note
    assert bench.resolve_model_path(str(good), mirror) == (str(good), None)
    assert bench.resolve_model_path(None, mirror) == (None, None)                  # zero-shot: the hub id is used as is
