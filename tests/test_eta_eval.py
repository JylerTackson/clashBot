"""tools/eval/eval_eta.py (MAC-AL T2): ETA errors against a known arrival, on synthetic tracks (no model, no video)."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_perception.tracking import UnitTracker, eta_to_tower, load_range_priors, load_speed_priors  # noqa: E402
from tools.eval import eval_eta as ee  # noqa: E402

KB = ROOT / "knowledge_base"


@dataclass
class Det:
    cls: str
    side: str
    category: str
    conf: float
    bbox: tuple
    pos_f: tuple


def _track(samples, kind="melee", cls="knight", rng=0.0):
    return {"key": "m1#1", "cls": cls, "kind": kind, "speed_prior": 1.0, "range": rng, "samples": samples}


def _s(t, old, new, at=False):
    return [t, 3.5, 12.0, old, new, at]


def test_errors_are_prediction_minus_time_left_before_the_arrival():
    # arrives at t = 10; at t = 4 the new rule says 7.0 (late by 1.0), the old says 9.0 (late by 3.0)
    tr = _track([_s(4.0, 9.0, 7.0), _s(6.0, 6.0, 4.0), _s(8.0, None, 2.5), _s(10.0, 0.0, 0.0, at=True),
                 _s(11.0, 1.0, 0.0, at=True)])
    assert ee.arrival_t(tr) == 10.0
    errs = ee.track_errors(tr)
    assert errs["new"] == pytest.approx([1.0, 0.0, 0.5])          # 7-6, 4-4, 2.5-2
    assert errs["old"] == pytest.approx([3.0, 2.0])               # the None ETA at t = 8 is skipped
    # only the frames within a horizon of the arrival
    assert ee.track_errors(tr, max_horizon_s=4.0)["new"] == pytest.approx([0.0, 0.5])      # t = 6 and 8, not t = 4


def test_track_without_an_arrival_has_no_truth():
    tr = _track([_s(1.0, 5.0, 4.0), _s(2.0, 4.0, 3.0)])
    assert ee.arrival_t(tr) is None and ee.track_errors(tr) == {}
    assert ee.evaluate([tr])["all"]["new"]["n"] == 0


def test_summary_statistics():
    s = ee.summarize([-1.0, 0.0, 0.5, 1.0, 3.0])
    assert s["n"] == 5 and s["median"] == 0.5 and s["p25"] == 0.0 and s["p75"] == 1.0
    assert s["within_1s"] == pytest.approx(0.8) and s["median_abs"] == 1.0          # |3.0| is the only miss
    assert ee.summarize([])["n"] == 0 and ee.summarize([])["median"] is None


def test_evaluate_splits_by_kind_and_rule_and_acceptance():
    melee = _track([_s(0.0, 12.0, 10.5), _s(5.0, 7.0, 5.5), _s(10.0, 0.0, 0.0, at=True)])
    ranged = _track([_s(0.0, 16.0, 10.2), _s(5.0, 11.0, 5.1), _s(10.0, 0.0, 0.0, at=True)], kind="ranged", cls="archer")
    flyer = _track([_s(0.0, 11.0, 9.5), _s(5.0, 6.0, 4.5), _s(10.0, 0.0, 0.0, at=True)], kind="flyer", cls="bat")
    table = ee.evaluate([melee, ranged, flyer])
    assert table["melee"]["old"]["median"] == pytest.approx(2.0) and table["melee"]["new"]["median"] == pytest.approx(0.5)
    assert table["ranged"]["old"]["median"] == pytest.approx(6.0) and table["ranged"]["new"]["median"] == pytest.approx(0.15)
    assert table["flyer"]["new"]["median"] == pytest.approx(-0.5) and table["flyer"]["new"]["n"] == 2
    assert table["all"]["new"]["n"] == 6 and table["unknown"]["old"]["n"] == 0
    acc = ee.acceptance(table)
    assert acc["melee"]["pass"] and acc["ranged"]["pass"] and acc["flyer"]["pass"]
    # a kind with no samples fails the acceptance instead of passing silently
    assert not ee.acceptance(ee.evaluate([melee]))["ranged"]["pass"]
    # a new rule that is no better than the old one fails even when it is within the limit
    same = _track([_s(0.0, 10.5, 10.5), _s(10.0, 0.0, 0.0, at=True)])
    assert not ee.acceptance(ee.evaluate([same]))["melee"]["pass"]
    assert "| melee | new |" in "\n".join(ee.format_table(table, "t"))
    assert ee.tracks_by_kind([melee, ranged, flyer, _track([_s(1.0, 3.0, 2.0)])])["melee"] == {"tracks": 2, "arrived": 1}


def test_unit_kind_from_unit_stats():
    cards = ee.load_cards_stats()
    assert ee.unit_kind("knight", cards) == "melee" and ee.unit_kind("giant", cards) == "melee"
    assert ee.unit_kind("archer", cards) == "ranged"             # the detector label 'archer' is the card 'archers'
    assert ee.unit_kind("musketeer", cards) == "ranged"
    assert ee.unit_kind("balloon", cards) == "flyer" and ee.unit_kind("bat", cards) == "flyer"
    assert ee.unit_kind("unknown_unit", cards) == "unknown" and ee.unit_kind("whatever", {}) == "unknown"


def test_in_reach_and_samples_flag_the_arrival_by_attack_range():
    # our left tower is at (3.5, 4.3): an archer (range 5) is in reach up to 6.5 tiles from the centre
    assert ee.in_reach((3.5, 4.3 + 6.4), 5.0) and not ee.in_reach((3.5, 4.3 + 6.6), 5.0)
    assert ee.in_reach((3.5, 4.3 + 1.4), 0.0) and not ee.in_reach((3.5, 4.3 + 1.6), 0.0)
    s = ee.make_sample(2.0, (3.5, 4.3 + 8.0), (0.0, -1.0), "enemy", 5.0, ("own_left", 1.5))
    assert s == [2.0, 3.5, 12.3, ee.legacy_eta((3.5, 12.3), (0.0, -1.0), "enemy")[1], 1.5, False]
    assert ee.make_sample(2.0, (3.5, 10.0), (0.0, -1.0), "enemy", 5.0, ("own_left", 0.0))[5] is True


def test_legacy_eta_is_the_tracker_rule_without_priors():
    rng = np.random.default_rng(3)
    for _ in range(200):
        pos, vel = (rng.uniform(0, 18), rng.uniform(5, 31)), tuple(rng.uniform(-2, 2, size=2))
        for side in ("enemy", "ally"):
            assert eta_to_tower(pos, vel, side) == ee.legacy_eta(pos, vel, side)


def test_recorder_on_a_walking_archer_new_rule_is_right_and_old_rule_is_late():
    """An enemy archer (range 5, 1 tile/s) walks down the left lane; the tracker sees it every 0.3 s with 0.3 tiles of
    noise. It is in reach of our left tower at row 10.8: the new ETA tracks that, the old one is ~5 s late."""
    rng = np.random.default_rng(11)
    tracker = UnitTracker(speed_priors=load_speed_priors(KB), range_priors=load_range_priors(),
                          categories={"archers": "unit"})
    rec = ee.EtaRecorder(ee.load_cards_stats())
    t = 0.0
    while t < 16.0:
        y = 22.0 - 1.0 * t
        tracker.update(t, [Det("archer", "enemy", "unit", 0.9, (0, 0, 0, 0), (3.5 + rng.normal(0, 0.3), y + rng.normal(0, 0.3)))])
        rec.observe(t, 1, tracker.confirmed())
        t = round(t + 0.3, 3)
    tracks = rec.result()
    assert len(tracks) == 1 and tracks[0]["kind"] == "ranged" and tracks[0]["range"] == 5.0
    t_arr = ee.arrival_t(tracks[0])
    assert 10.0 <= t_arr <= 12.5                                          # row 10.8 at t = 11.2, give or take the noise
    table = ee.evaluate(tracks)["ranged"]
    assert abs(table["new"]["median"]) < 1.0 and table["new"]["within_1s"] > 0.5
    assert table["old"]["median"] > 4.0 and table["old"]["within_1s"] < 0.2     # the old rule: to the tower edge
    report = ee.build_report(tracks, {"video": "synthetic"})
    json.dumps(report)                                                    # the report is JSON
    assert report["arrived"] == 1 and report["coverage"]["archer"]["kind"] == "ranged"
    assert report["by_kind"]["ranged"]["new"]["median_abs"] < report["by_kind"]["ranged"]["old"]["median_abs"]


def test_recorder_skips_allies_and_buildings():
    tracker = UnitTracker(categories={"cannon": "building"})
    for k in range(3):
        tracker.update(float(k), [Det("cannon", "enemy", "unit", 0.9, (0, 0, 0, 0), (3.5, 12.0)),
                                  Det("knight", "ally", "unit", 0.9, (0, 0, 0, 0), (3.5, 8.0 + k))])
    rec = ee.EtaRecorder({})
    rec.observe(2.0, 1, tracker.confirmed())
    assert rec.result() == []
