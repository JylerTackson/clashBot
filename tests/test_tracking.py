"""Kalman unit tracking on synthetic motion."""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cr_perception.tracking import UnitTracker, load_speed_priors, KalmanTrack

KB = Path(__file__).resolve().parents[1] / "knowledge_base"


@dataclass
class Det:
    cls: str
    side: str
    category: str
    conf: float
    bbox: tuple
    pos_f: tuple


def test_speed_priors_from_phase1():
    sp = load_speed_priors(KB)
    assert sp.get("knight") == 1.0          # Medium (60)
    assert sp.get("hog-rider") == 2.0       # Very Fast (120)
    assert sp.get("golem") == 0.75          # Slow (45)


def test_straight_line_track_velocity_and_prediction():
    rng = np.random.default_rng(1)
    tr = UnitTracker(speed_priors={"knight": 1.0}, categories={"knight": "unit"})
    # an ally knight walking up the left lane at 1 tile/s, detections at 1 Hz with 0.3-tile noise
    truth = lambda t: (3.5, 2.0 + 1.0 * t)
    last = None
    for k in range(9):
        t = float(k)
        c, r = truth(t)
        tr.update(t, [Det("knight", "ally", "unit", 0.9, (0, 0, 0, 0), (c + rng.normal(0, 0.3), r + rng.normal(0, 0.3)))])
        last = tr.confirmed()
    assert len(last) == 1
    k = last[0]
    vc, vr = k.velocity()
    assert abs(vr - 1.0) < 0.25 and abs(vc) < 0.25, (vc, vr)
    assert k.heading() == "advancing left lane"
    # predict 2 s ahead: within one tile of the truth
    pc, pr = k.predict_position(2.0)
    tc, tr_ = truth(8.0 + 2.0)
    assert math.hypot(pc - tc, pr - tr_) < 1.0
    assert k.eta_to_row(15.5) is not None and 3 < k.eta_to_row(15.5) < 7


def test_gap_handling_and_reassociation():
    tr = UnitTracker(speed_priors={"knight": 1.0}, categories={"knight": "unit"})
    for k in range(4):
        tr.update(float(k), [Det("knight", "enemy", "unit", 0.9, (0, 0, 0, 0), (14.0, 25.0 - k))])
    tid = tr.confirmed()[0].id
    # 2 s without detections: the track survives on prediction and keeps its id
    tr.predict(5.0)
    assert tid in tr.tracks
    p = tr.tracks[tid].position()
    assert abs(p[1] - 20.0) < 1.0             # predicted onward at ~1 tile/s downward
    assert tr.tracks[tid].pos_std() > 0.3     # uncertainty grew during the gap
    tr.update(6.0, [Det("knight", "enemy", "unit", 0.9, (0, 0, 0, 0), (14.2, 19.1))])
    assert tr.confirmed()[0].id == tid and len(tr.tracks) == 1
    assert tr.tracks[tid].pos_std() < 0.4     # shrank again after the measurement


def test_static_building_and_class_gating():
    tr = UnitTracker(categories={"cannon": "building", "knight": "unit"})
    for k in range(3):
        tr.update(float(k), [Det("cannon", "ally", "unit", 0.9, (0, 0, 0, 0), (9.0 + 0.1 * (k % 2), 10.0)),
                             Det("knight", "enemy", "unit", 0.9, (0, 0, 0, 0), (9.0, 10.5 - 0.3 * k))])
    tracks = {t.cls: t for t in tr.confirmed()}
    assert set(tracks) == {"cannon", "knight"}
    assert tracks["cannon"].speed() == 0.0 and tracks["cannon"].heading() == "stationary"
    assert tracks["cannon"].category == "building"
    s = tracks["knight"].summary()
    assert s["pred_1s"] and s["eta_tower"] is not None and s["heading"].startswith("advancing")


def test_unknown_unit_upgrades_to_class():
    tr = UnitTracker()
    tr.update(0.0, [Det("unknown_unit", "enemy", "unit", 0.4, (0, 0, 0, 0), (5.0, 20.0))])
    tr.update(1.0, [Det("mega-knight", "enemy", "unit", 0.8, (0, 0, 0, 0), (5.0, 19.2))])
    assert [t.cls for t in tr.confirmed()] == ["mega-knight"]
