"""Kalman unit tracking on synthetic motion."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from cr_perception.tracking import (TOWER_TILES, KalmanTrack, UnitTracker, eta_to_tower, load_range_priors,
                                    load_speed_priors)

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


# --- MAC-AL T1: ETA to attack range at the card's real speed --------------------------------------------------------
def _old_eta(x, y, vc, vr, side):
    """The pre-MAC-AL rule: straight line to the tower edge at the filter's speed."""
    sp = math.hypot(vc, vr)
    if sp < 0.1:
        return None
    names = ("enemy_left", "enemy_right", "enemy_king") if side == "ally" else ("own_left", "own_right", "own_king")
    best = None
    for nm in names:
        tc, tr = TOWER_TILES[nm]
        d = math.hypot(tc - x, tr - y) - 1.5
        if best is None or d < best[1]:
            best = (nm, max(0.0, d))
    return best[0], round(best[1] / sp, 1)


def _track(cls="knight", side="enemy", pos=(3.5, 14.3), vel=(0.0, -1.0), speed_prior=None, range_prior=0.0,
           category="unit"):
    x = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=float)
    return KalmanTrack(1, cls, side, category, x, np.eye(4) * 0.1, 0.0, speed_prior=speed_prior, range_prior=range_prior)


def test_range_priors_are_the_first_units_range_keyed_like_the_speed_priors(tmp_path):
    rp = load_range_priors()
    assert rp["archers"] == 5.0 and rp["musketeer"] == 6.0 and rp["balloon"] == 0.1 and rp["bats"] == 1.2
    assert set(load_speed_priors(KB)) <= set(rp)             # a track's slug finds both priors
    assert isinstance(rp["giant"], float) and rp["zap"] >= 0.0
    # a card with no unit row gets 0.0; a missing or broken file gives {} (the old tower-edge ETA)
    p = tmp_path / "stats.json"
    p.write_text(json.dumps({"cards": {"a": {"units": [{"range_tiles": 4.5}, {"range_tiles": 9}]}, "b": {"units": []},
                             "c": {"units": [{"name": "x"}]}, "d": None}}))
    assert load_range_priors(p) == {"a": 4.5, "b": 0.0, "c": 0.0, "d": 0.0}
    assert load_range_priors(tmp_path / "missing.json") == {}
    p.write_text("not json")
    assert load_range_priors(p) == {}


def test_ranged_track_counts_to_attack_range_not_to_the_tower_edge():
    # archers (range 5) 7 tiles beyond the tower edge: they stop 2 tiles from walking further
    x, y = 3.5, 4.3 + 1.5 + 7.0
    assert eta_to_tower((x, y), (0.0, -1.0), "enemy", 1.0, 5.0) == ("own_left", 2.0)
    assert eta_to_tower((x, y), (0.0, -0.8), "enemy", 0.8, 5.0) == ("own_left", 2.5)        # 2 / 0.8
    assert eta_to_tower((x, y), (0.0, -1.0), "enemy", 1.0, 0.0) == ("own_left", 7.0)        # without range: the edge
    tr = _track("archer", pos=(x, y), vel=(0.0, -1.0), speed_prior=1.0, range_prior=5.0)
    assert tr.eta_to_nearest_enemy_tower() == ("own_left", 2.0)
    assert tr.summary()["eta_tower"] == {"tower": "own_left", "s": 2.0, "range": 5.0}


def test_slow_filter_on_a_unit_walking_at_its_card_speed_uses_90_percent_of_the_prior():
    pos = (3.5, 4.3 + 1.5 + 9.0)                              # 9 tiles of walking left
    tr = _track("knight", pos=pos, vel=(0.0, -0.6), speed_prior=1.0)          # filter reads 0.6x the card's speed
    assert tr.eta_to_nearest_enemy_tower() == ("own_left", round(9.0 / 0.9, 1))
    # a filter already faster than 0.9 x prior keeps its own speed
    assert _track("knight", pos=pos, vel=(0.0, -1.2), speed_prior=1.0).eta_to_nearest_enemy_tower() == ("own_left", 7.5)
    # a sideways drift (cosine to the tower < 0.5) is not "toward the tower": the filter speed stays
    side = eta_to_tower((9.0, 4.3 + 1.5 + 9.0), (-0.6, 0.0), "enemy", 1.0, 0.0)
    assert side == _old_eta(9.0, 4.3 + 1.5 + 9.0, -0.6, 0.0, "enemy")


def test_unit_in_reach_of_a_tower_has_eta_zero_even_when_it_reads_stationary():
    assert _track("knight", pos=(3.5, 5.5), vel=(0.0, 0.0), speed_prior=1.0).eta_to_nearest_enemy_tower() == ("own_left", 0.0)
    assert eta_to_tower((3.5, 5.5), (0.0, 0.05), "enemy") == ("own_left", 0.0)               # no priors needed
    assert eta_to_tower((3.5, 5.5), (0.0, 0.0), "enemy", 1.0, 0.1) == ("own_left", 0.0)      # balloon on the tower
    # in reach by range: archers 6 tiles from the tower centre (range 5 + edge 1.5 = 6.5)
    assert eta_to_tower((3.5, 4.3 + 6.0), (0.0, 0.0), "enemy", 1.0, 5.0) == ("own_left", 0.0)
    # out of reach and not moving: still no ETA
    assert eta_to_tower((3.5, 4.3 + 8.0), (0.0, 0.0), "enemy", 1.0, 5.0) is None
    assert eta_to_tower((3.5, 4.3 + 8.0), (0.0, -0.05), "enemy", 1.0, 5.0) is None


def test_track_moving_away_from_the_tower_keeps_the_old_speed_rule():
    pos = (3.5, 4.3 + 1.5 + 9.0)
    away = eta_to_tower(pos, (0.0, 0.7), "enemy", 2.0, 0.0)          # backing off at 0.7: the prior does not speed it up
    assert away == _old_eta(pos[0], pos[1], 0.0, 0.7, "enemy") == ("own_left", round(9.0 / 0.7, 1))
    # with a range the distance still shrinks by it, the speed is the filter's
    assert eta_to_tower(pos, (0.0, 0.7), "enemy", 2.0, 5.0) == ("own_left", round(4.0 / 0.7, 1))


def test_no_priors_gives_the_old_eta():
    rng = np.random.default_rng(7)
    n_checked = 0
    for _ in range(300):
        x, y = rng.uniform(0, 18), rng.uniform(1, 31)
        vc, vr = rng.uniform(-2, 2, size=2)
        for side in ("enemy", "ally"):
            prefix = "own" if side == "enemy" else "enemy"
            if min(math.hypot(a - x, b - y) for n, (a, b) in TOWER_TILES.items() if n.startswith(prefix)) <= 1.5:
                continue                                         # at a tower: ETA 0.0 even when stationary (new by design)
            assert eta_to_tower((x, y), (vc, vr), side) == _old_eta(x, y, vc, vr, side)
            n_checked += 1
    assert n_checked > 300
    # through a tracker without priors: the same numbers as the old rule
    tr = UnitTracker(categories={"knight": "unit"})
    for k in range(4):
        tr.update(float(k), [Det("knight", "enemy", "unit", 0.9, (0, 0, 0, 0), (3.5, 20.0 - k))])
    t = tr.confirmed()[0]
    assert t.speed_prior is None and t.range_prior == 0.0
    assert t.eta_to_nearest_enemy_tower() == _old_eta(t.x[0], t.x[1], t.x[2], t.x[3], "enemy")


def test_buildings_and_spells_have_no_tower_eta():
    assert _track("cannon", pos=(3.5, 5.5), vel=(0.0, 0.0), category="building", range_prior=5.5).eta_to_nearest_enemy_tower() is None
    assert _track("fireball", pos=(3.5, 5.5), vel=(0.0, 0.0), category="spell").summary()["eta_tower"] is None


def test_tracker_gives_its_tracks_the_priors_of_their_card_slug():
    tr = UnitTracker(speed_priors={"archers": 1.0, "knight": 1.0}, range_priors={"archers": 5.0, "knight": 1.2},
                     categories={"archers": "unit", "knight": "unit"})
    # the detector says 'archer', the card slug is 'archers'
    tr.update(0.0, [Det("archer", "enemy", "unit", 0.9, (0, 0, 0, 0), (3.5, 20.0)),
                    Det("unknown_unit", "enemy", "unit", 0.4, (0, 0, 0, 0), (14.5, 20.0))])
    tr.update(1.0, [Det("archer", "enemy", "unit", 0.9, (0, 0, 0, 0), (3.5, 19.0)),
                    Det("knight", "enemy", "unit", 0.8, (0, 0, 0, 0), (14.5, 19.0))])   # the unknown unit becomes a knight
    by_cls = {t.cls: t for t in tr.confirmed()}
    assert (by_cls["archer"].speed_prior, by_cls["archer"].range_prior) == (1.0, 5.0)
    assert (by_cls["knight"].speed_prior, by_cls["knight"].range_prior) == (1.0, 1.2)
    eta = by_cls["archer"].summary()["eta_tower"]
    assert eta["tower"] == "own_left" and eta["range"] == 5.0
    assert eta["s"] == pytest.approx(math.hypot(0.0, 19.0 - 4.3) - 1.5 - 5.0, abs=1.0)   # ~8.2 s at ~1 tile/s


def test_perception_gives_the_tracker_range_priors(tmp_path, monkeypatch):
    from cr_perception import Perception
    from cr_perception.config import Calibration
    from cr_perception.hud import DEFAULT_ROIS
    from cr_perception.sources import FrameSource
    calib = Calibration(source={"type": "synthetic", "frame_size": [576, 1024]}, rois=dict(DEFAULT_ROIS))
    cfg = tmp_path / "calib.json"
    calib.save(cfg)
    monkeypatch.setattr("cr_perception.perception.REPO", tmp_path)       # learned templates: neither read nor written

    class NoFrames(FrameSource):
        def frames(self):
            return iter(())

    p = Perception(cfg, NoFrames(), detector=None, use_ocr=False)
    try:
        assert p.tracker.range_priors["musketeer"] == 6.0 and p.tracker.speed_priors["knight"] == 1.0
    finally:
        p.close()
