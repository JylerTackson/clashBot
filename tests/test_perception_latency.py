"""Perception latency for the live loop (MAC-Q): the hand-read cache is
parity-safe, and `async_ocr=True` keeps slow OCR off the frame path while
`async_ocr=False` stays the synchronous path. Slow readers are fakes; no model
is loaded."""
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_perception import Perception  # noqa: E402
from cr_perception.config import Calibration  # noqa: E402
from cr_perception.events import PlayDetector  # noqa: E402
from cr_perception.hud import DEFAULT_ROIS  # noqa: E402
from cr_perception.labels import DeployLabel  # noqa: E402
from cr_perception.screen import Readiness  # noqa: E402
from tests.test_hud_synthetic import synthetic_hud  # noqa: E402
from tests.test_perception_rates import FRAME, NullSource  # noqa: E402
from tests.test_pipeline_synthetic import arena_frame  # noqa: E402

CHANGED = arena_frame(synthetic_hud(["knight", "fireball", "golem", "ice-spirit"], "hog-rider", 7))


class SlowOcr:
    """Stands in for OcrReader: sleeps, records the calling thread, returns a
    scripted clock (the i-th read returns values[min(i, len-1)])."""

    def __init__(self, values=("1:59",), delay=0.2, conf=0.9):
        self.values, self.delay, self.conf = list(values), delay, conf
        self.threads: list[str] = []

    def read(self, img, allow="0123456789:"):
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        return self.values[min(len(self.threads) - 1, len(self.values) - 1)], self.conf


class OneLabel:
    """Stands in for DeployLabelReader: one deploy label per pass."""

    def __init__(self, delay=0.1):
        self.delay, self.threads = delay, []

    def read(self, content, arena_crop):
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        return [DeployLabel("knight", "Knight", 1.0, 0.9, (250, 400, 330, 430), 11)], []


@pytest.fixture
def make_perception(tmp_path, monkeypatch):
    """Perception with the gate open from the first frame (ENTER_FRAMES=1,
    readiness forced to "match"), no detector, no real OCR."""
    calib = Calibration(source={"type": "synthetic", "frame_size": [576, 1024]}, rois=dict(DEFAULT_ROIS))
    calib.arena_corners_frac = [[0.072, 0.769], [0.922, 0.769], [0.905, 0.079], [0.089, 0.079]]
    cfg = tmp_path / "calib.json"
    calib.save(cfg)
    monkeypatch.setattr("cr_perception.perception.assess", lambda *a, **k: Readiness("match", 1.0, {}))
    made: list[Perception] = []

    def make(ocr=None, labels=None, **kw) -> Perception:
        p = Perception(cfg, NullSource(), detector=None, use_ocr=False, ENTER_FRAMES=1, **kw)
        p.ocr, p.labels = ocr, labels
        made.append(p)
        return p

    yield make
    for p in made:
        p.close()


def count_hand_reads(p: Perception) -> list[float]:
    """Wrap hand_reader.read; the returned list collects the frame time of each real read."""
    reads: list[float] = []
    orig = p.hand_reader.read

    def counting(content):
        reads.append(p._now)
        return orig(content)

    p.hand_reader.read = counting
    return reads


def feed(p: Perception, frames, dt=0.1):
    out = []
    for k, frame in enumerate(frames):
        p._now = k * dt
        out.append(p.process(frame, p._now, k))
    return out


def hand_view(s) -> tuple:
    return (tuple(s.own["hand"]), tuple(s.own["hand_conf"]), s.own["next_card"], s.own["next_conf"])


# ----------------------------------------------------------------------------
# 1. hand-read cache
# ----------------------------------------------------------------------------

def test_hand_cache_is_parity_safe(make_perception, monkeypatch):
    # reference: the cache disabled (an entry is never younger than 0 s)
    with monkeypatch.context() as m:
        m.setattr("cr_perception.perception.HAND_CACHE_MAX_AGE_S", 0.0)
        ref = make_perception()
        ref_reads = count_hand_reads(ref)
        ref_states = feed(ref, [FRAME] * 30, dt=0.1)
    assert len(ref_reads) == 30 and ref.hand_cache_hits == 0

    p = make_perception()
    reads = count_hand_reads(p)
    states = feed(p, [FRAME] * 30, dt=0.1)
    assert len([t for t in reads if t <= 1.0]) <= 2        # one real read per second of an unchanged hand
    assert len(reads) <= 4 and p.hand_cache_hits + p.hand_cache_misses == 30
    assert p.hand_cache_hits >= 26 and p.hand_cache_misses == len(reads)
    assert [hand_view(s) for s in states] == [hand_view(s) for s in ref_states]
    assert states[-1].own["hand"] == ["knight", "fireball", "musketeer", "ice-spirit"]
    assert p.timing_report()["hand_cache"] == {"hits": p.hand_cache_hits, "misses": p.hand_cache_misses}

    # one slot's pixels change -> the very next frame reads again, and the new card comes through
    n_reads = len(reads)
    p._now = 3.0
    p.process(CHANGED, 3.0, 30)
    assert len(reads) == n_reads + 1
    last = None
    for k in range(31, 37):
        p._now = 3.0 + (k - 30) * 0.1
        last = p.process(CHANGED, p._now, k)
    assert last.own["hand"][2] == "golem"

    # reset_match drops the cache
    p.reset_match(4.0)
    assert p._last_hand_raw is None
    n_reads = len(reads)
    p._now = 4.0
    p.process(CHANGED, 4.0, 37)
    assert len(reads) == n_reads + 1


def test_hand_cache_ignores_sensor_noise_but_not_age(make_perception):
    p = make_perception()
    reads = count_hand_reads(p)
    noisy = FRAME.copy()
    noisy[::7] = (noisy[::7].astype("int16") + 2).clip(0, 255).astype("uint8")   # mean change ~0.3 grey levels
    feed(p, [FRAME, noisy, FRAME], dt=0.2)
    assert len(reads) == 1
    p._now = 1.5                                           # older than HAND_CACHE_MAX_AGE_S: forced re-read
    p.process(FRAME, 1.5, 3)
    assert len(reads) == 2


# ----------------------------------------------------------------------------
# 2. async_ocr=False keeps today's synchronous path
# ----------------------------------------------------------------------------

def test_sync_ocr_runs_on_the_main_thread_with_todays_clock_sequence(make_perception):
    ocr = SlowOcr(values=("2:30", "2:20", "2:10"), delay=0.2)
    p = make_perception(ocr=ocr)
    assert p.async_ocr is False
    states = feed(p, [FRAME] * 21, dt=0.1)
    # clock OCR on n % ocr_every == 0 (n = 0, 10, 20); every frame carries the last good read
    assert len(ocr.threads) == 3
    assert set(ocr.threads) == {threading.current_thread().name}
    assert [s.match_clock for s in states] == ["2:30"] * 10 + ["2:20"] * 10 + ["2:10"]
    assert p._ocr_pool is None and "ocr_async" not in p.timing_report()
    p.close()
    p.close()


# ----------------------------------------------------------------------------
# 3. async_ocr=True: OCR on the cr-ocr worker only, frames never wait for it
# ----------------------------------------------------------------------------

def test_async_ocr_never_blocks_a_frame(make_perception):
    ocr = SlowOcr(values=("1:59",), delay=0.2)
    p = make_perception(ocr=ocr, async_ocr=True)
    p.hand_reader.read(FRAME)                              # warm-up so frame 0 does not pay first-use costs
    states, worst = [], 0.0
    deadline = time.perf_counter() + 5.0
    k = 0
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        states.append(p.process(FRAME, k * 0.1, k))
        worst = max(worst, time.perf_counter() - t0)
        k += 1
        if states[-1].match_clock == "1:59":
            break
        time.sleep(0.01)
    assert states[-1].match_clock == "1:59"
    assert states[0].match_clock != "1:59"                 # frame 0 did not wait for the 0.2 s OCR
    assert worst < 0.15, worst
    assert ocr.threads and all(name.startswith("cr-ocr") for name in ocr.threads)
    rep = p.timing_report()["ocr_async"]
    assert rep["submitted"] >= 1 and rep["applied"] >= 1
    p.close()
    p.close()                                              # idempotent
    assert p._ocr_pool is None


def test_async_clock_job_is_not_resubmitted_while_pending(make_perception):
    ocr = SlowOcr(values=("1:59",), delay=0.5)
    p = make_perception(ocr=ocr, async_ocr=True)
    feed(p, [FRAME] * 30, dt=0.1)                          # gate fires at n = 0, 10, 20 within ~50 ms
    assert p.ocr_async_submitted == 1 and p._clock_job is not None
    p.close()
    assert p._clock_job is None


def test_async_worker_error_does_not_kill_the_loop(make_perception):
    class Boom:
        def read(self, img, allow="0123456789:"):
            raise RuntimeError("ocr died")

    p = make_perception(ocr=Boom(), async_ocr=True)
    p.process(FRAME, 0.0, 0)
    time.sleep(0.1)
    s = p.process(FRAME, 0.1, 1)
    assert s.readiness == "match" and p.ocr_async_errors == 1 and p.ocr_async_applied == 0


# ----------------------------------------------------------------------------
# 4. async labels carry the submit time
# ----------------------------------------------------------------------------

def test_async_labels_use_the_submit_time(make_perception, monkeypatch):
    calls = []
    orig = PlayDetector.update_labels

    def wrapped(self, t, labels, tiles, clock):
        calls.append((t, len(labels), p.n_processed))
        return orig(self, t, labels, tiles, clock)

    monkeypatch.setattr(PlayDetector, "update_labels", wrapped)
    labels = OneLabel(delay=0.1)
    p = make_perception(labels=labels, async_ocr=True)
    t_first = 5.0
    deadline = time.perf_counter() + 5.0
    k = 0
    while not calls and time.perf_counter() < deadline:
        p.process(FRAME, t_first + k * 0.1, k)
        k += 1
        time.sleep(0.02)
    assert calls, "no label result was applied"
    t_applied, n_labels, n_at_apply = calls[0]
    assert n_labels == 1
    assert n_at_apply > 1                                  # applied on a later frame ...
    assert t_applied == t_first                            # ... but stamped with the submit time (n = 0)
    assert all(name.startswith("cr-ocr") for name in labels.threads)


def test_sync_labels_use_the_frame_time(make_perception, monkeypatch):
    calls = []
    orig = PlayDetector.update_labels

    def wrapped(self, t, labels, tiles, clock):
        calls.append(t)
        return orig(self, t, labels, tiles, clock)

    monkeypatch.setattr(PlayDetector, "update_labels", wrapped)
    labels = OneLabel(delay=0.0)
    p = make_perception(labels=labels)
    feed(p, [FRAME] * 16, dt=0.1)                          # label gate: n = 0 and n = 15
    assert calls == [0.0, 1.5]
    assert set(labels.threads) == {threading.current_thread().name}
