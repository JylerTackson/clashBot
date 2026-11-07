"""Tests for tools/eval/latency_table.py (MAC-AP): aggregation of fake summary.json files; no model is loaded."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval import latency_table as lt  # noqa: E402


def fake_summary(scale: float, decisions: int = 10, target_met: bool = False) -> dict:
    """decision_ms stages whose p50 is 10 * k * scale and p95 20 * k * scale (k = stage index + 1)."""
    block = {s: {"p50": 10.0 * (i + 1) * scale, "p95": 20.0 * (i + 1) * scale} for i, s in enumerate(lt.STAGES)}
    return {"decisions": decisions, "frames": 100, "target_met": target_met, "decision_ms": block,
            "frame_ms": {"capture": {"p50": 5.0 * scale, "p95": 9.0 * scale}}}


def write_run(root: Path, name: str, summary: dict | None, live: bool = False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary))
    (d / "config.json").write_text(json.dumps({"live": live, "source": "video", "ranker": "d1", "seconds": 120}))
    return d


def test_aggregate_two_fake_summaries_gives_mean_and_sample_sd():
    agg = lt.aggregate_runs([fake_summary(1.0), fake_summary(3.0)])
    capture = agg["capture"]["p50"]                            # 10 and 30
    assert capture["mean"] == pytest.approx(20.0)
    assert capture["sd"] == pytest.approx(14.142135, abs=1e-5)  # sample sd (n - 1) of {10, 30}
    assert capture["values"] == [10.0, 30.0] and capture["n"] == 2
    total = agg["total"]["p95"]                                 # stage 8: 160 and 480
    assert total["mean"] == pytest.approx(320.0) and total["sd"] == pytest.approx(226.274, abs=1e-3)
    assert set(agg) == set(lt.STAGES)


def test_single_run_has_no_sd_and_missing_stages_are_skipped():
    s = fake_summary(1.0)
    del s["decision_ms"]["trigger"]
    agg = lt.aggregate_runs([s])
    assert agg["capture"]["p50"]["sd"] is None
    assert "trigger" not in agg
    assert lt.aggregate_runs([fake_summary(2.0)], "frame_ms") == {"capture": agg_of(10.0, 18.0)}


def agg_of(p50: float, p95: float) -> dict:
    return {"p50": {"n": 1, "mean": p50, "sd": None, "values": [p50]}, "p95": {"n": 1, "mean": p95, "sd": None, "values": [p95]}}


def test_live_rows_one_row_per_run_with_a_summary(tmp_path):
    write_run(tmp_path, "20261106-100000", fake_summary(1.0), live=True)
    write_run(tmp_path, "20261106-110000", fake_summary(2.0), live=False)
    write_run(tmp_path, "20261106-120000", None)                    # crashed run: no summary.json
    write_run(tmp_path, "20261001-000000", fake_summary(9.0))        # outside the glob
    rows = lt.live_rows(tmp_path, "2026110*")
    assert [r["run"] for r in rows] == ["20261106-100000", "20261106-110000"]
    assert [r["mode"] for r in rows] == ["live", "dry"]
    assert rows[1]["stages"]["total"] == {"p50": 160.0, "p95": 320.0}
    assert rows[0]["decisions"] == 10


def test_percentile_stats_and_time_calls():
    st = lt.percentile_stats(list(range(1, 101)))
    assert st["n"] == 100 and st["p50_ms"] == pytest.approx(50.5) and st["p95_ms"] == pytest.approx(95.05)
    calls = []
    ms = lt.time_calls(lambda: calls.append(1), n=7, warmup=3)
    assert len(ms) == 7 and len(calls) == 10 and all(m >= 0 for m in ms)
    seen = []
    lt.time_calls(seen.append, n=2, warmup=1, args_for=lambda i: i)
    assert seen == [-1, 0, 1]


def test_main_without_micro_writes_json_and_markdown(tmp_path):
    h6 = [write_run(tmp_path / "rerun", f"lat_{i}", fake_summary(float(i), target_met=(i == 3))) for i in (1, 2, 3)]
    poc = tmp_path / "poc"
    write_run(poc, "20261107-111554", fake_summary(2.0), live=True)
    out = tmp_path / "docs" / "latency_v2.json"
    res = lt.main(["--h6", *map(str, h6), "--poc-root", str(poc), "--out", str(out), "--skip-micro"])
    data = json.loads(out.read_text())
    assert data["h6"]["runs"] == ["lat_1", "lat_2", "lat_3"] and data["h6"]["target_met"] == 1
    assert data["h6"]["stages"]["capture"]["p50"]["mean"] == pytest.approx(20.0)          # {10, 20, 30}
    assert data["h6"]["stages"]["capture"]["p50"]["sd"] == pytest.approx(10.0)
    assert [r["run"] for r in data["runs"]] == ["20261107-111554"] and "micro" not in data
    md = out.with_suffix(".md").read_text()
    assert "20.0 ± 10.0" in md and "20261107-111554" in md and "median of live runs" in md
    assert res["h6"]["decisions"] == [10, 10, 10]


def test_markdown_renders_micro_benchmarks_and_errors():
    micro = {"katacr": {"mps": lt.percentile_stats([40.0, 41.0, 60.0]), "cpu": {"error": "RuntimeError: boom"},
                        "frame": "f.mp4 t=90"},
             "d1": {**lt.percentile_stats([60.0, 70.0]), "cold_first_call_ms": 1234.5}}
    md = lt.render_markdown({"meta": {"git_head": "abc"}, "runs": [], "micro": micro})
    assert "| katacr / mps | 3 |" in md and "ERROR: RuntimeError: boom" in md and "| d1 | 2 |" in md
    assert "cold_first_call_ms = 1234.5" in md and "mean_ms" not in md.split("Isolated")[1].split("- d1")[1]
