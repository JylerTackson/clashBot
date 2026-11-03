"""Perception's slow-reader rates count processed frames (`n_processed`), not
the source's frame index: a stride-5 source that starts off-grid must trigger
clock OCR / deploy labels / detection exactly as often as consecutive frames.
The slow readers are replaced by call counters; no model is loaded."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_perception import Perception  # noqa: E402
from cr_perception.config import Calibration  # noqa: E402
from cr_perception.events import PlayDetector  # noqa: E402
from cr_perception.hud import DEFAULT_ROIS  # noqa: E402
from cr_perception.screen import Readiness  # noqa: E402
from cr_perception.sources import FrameSource  # noqa: E402
from tests.test_hud_synthetic import synthetic_hud  # noqa: E402
from tests.test_pipeline_synthetic import arena_frame  # noqa: E402

N_FRAMES = 60
DT = 0.1                                   # keeps the 60 frames inside the 9 s slot-crop window


class NullSource(FrameSource):
    fps = 10.0

    def frames(self):
        return iter(())


class FakeOcr:
    """Stands in for OcrReader: counts clock reads, always returns a valid clock."""

    def __init__(self):
        self.reads = 0

    def read(self, img, allow="0123456789:"):
        self.reads += 1
        return "2:30", 0.9


class FakeLabels:
    def __init__(self):
        self.reads = 0

    def read(self, content, arena_crop):
        self.reads += 1
        return [], []


class FakeTowerReader:
    def __init__(self):
        self.reads = 0

    def read(self, content):
        self.reads += 1
        return {f"{side}_{k}": (None, 0.0) for side in ("own", "enemy") for k in ("king", "left", "right")}


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, content, arena_crop):
        self.calls += 1
        return []


@pytest.fixture
def make_perception(tmp_path, monkeypatch):
    """Perception whose slow readers are counters and whose gate is open from
    the first frame (ENTER_FRAMES=1, readiness forced to "match")."""
    calib = Calibration(source={"type": "synthetic", "frame_size": [576, 1024]}, rois=dict(DEFAULT_ROIS))
    calib.arena_corners_frac = [[0.072, 0.769], [0.922, 0.769], [0.905, 0.079], [0.089, 0.079]]
    cfg = tmp_path / "calib.json"
    calib.save(cfg)
    monkeypatch.setattr("cr_perception.perception.assess", lambda *a, **k: Readiness("match", 1.0, {}))

    unit_passes = []
    orig_update_units = PlayDetector.update_units

    def counting_update_units(self, *a, **k):
        unit_passes.append(1)
        return orig_update_units(self, *a, **k)

    monkeypatch.setattr(PlayDetector, "update_units", counting_update_units)

    def make() -> Perception:
        p = Perception(cfg, NullSource(), detector=FakeDetector(), use_ocr=False, tower_ocr=True, ENTER_FRAMES=1)
        p.ocr, p.labels, p.tower_reader = FakeOcr(), FakeLabels(), FakeTowerReader()
        p.unit_passes = unit_passes
        return p

    return make


FRAME = arena_frame(synthetic_hud(["knight", "fireball", "musketeer", "ice-spirit"], "hog-rider", 7))


def run(p: Perception, indices) -> list:
    return [p.process(FRAME, k * DT, idx) for k, idx in enumerate(indices)]


@pytest.mark.parametrize("indices", [
    pytest.param([23676 + 5 * k for k in range(N_FRAMES)], id="stride5_offgrid"),
    pytest.param(list(range(N_FRAMES)), id="consecutive"),
])
def test_rates_independent_of_source_index(make_perception, indices):
    p = make_perception()
    states = run(p, indices)
    assert all(s.readiness == "match" for s in states)
    assert p.n_processed == N_FRAMES
    assert p.ocr.reads == 6                    # clock OCR: n % ocr_every(10) == 0
    assert p.tower_reader.reads == 6           # tower OCR shares ocr_every
    assert p.labels.reads == 4                 # deploy labels: n % label_every(15) == 0
    assert p.detector.calls == 20              # detection: n % detect_every(3) == 0
    assert len(p.unit_passes) == 20            # unit-event pass shares detect_every
    assert len(p._slot_crops) == 20            # hand-crop ring buffer: n % 3 == 0
    # the source index stays the frame identity in the output
    assert [s.frame_index for s in states] == indices
    assert p.frame_i == indices[-1]


def test_n_processed_survives_reset_match(make_perception):
    p = make_perception()
    run(p, range(10))
    assert p.n_processed == 10
    p.reset_match(1.0)
    assert p.n_processed == 10
    p.process(FRAME, 1.0, 10)
    p.process(FRAME, 1.1, 11)
    assert p.n_processed == 12
    # the gate keeps counting from 12: next clock-OCR frame is n == 20, not a fresh 0
    reads = p.ocr.reads                        # passes so far: n = 0 and n = 10
    assert reads == 2
    for k in range(12, 20):
        p.process(FRAME, k * DT, k)
    assert p.ocr.reads == reads
    p.process(FRAME, 2.0, 20)
    assert p.ocr.reads == reads + 1
