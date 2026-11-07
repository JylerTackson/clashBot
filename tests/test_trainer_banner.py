"""Training Camp banner (MAC-AN, D-050): the box, the OCR wrapper with a fake engine, `is_trainer`, the verdict, and the real
shots of runs/poc (skipped when runs/ is absent; the real-OCR check is slow: CLASHBOT_SLOW_TESTS=1)."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_perception.config import Calibration  # noqa: E402
from cr_perception.trainer_banner import (BANNER_FRAC, MIN_TRAINER_FRAMES, banner_box, edit_distance, is_trainer,  # noqa: E402
                                          read_banner, verdict)

CALIB = ROOT / "runs" / "poc" / "mirror_calib.json"
SHOTS = sorted((ROOT / "runs" / "poc" / "20261107-111554" / "shots").glob("*_a.jpg"))[:3]
FULL = ROOT / "runs" / "poc" / "mirror_frame_quartz.png"
needs_runs = pytest.mark.skipif(not CALIB.exists() or len(SHOTS) < 3, reason="runs/poc shots missing")
slow = pytest.mark.skipif(os.environ.get("CLASHBOT_SLOW_TESTS") != "1", reason="real OCR: set CLASHBOT_SLOW_TESTS=1")


@pytest.mark.parametrize("text", ["Trainer Cheddar II", "TRAINER", "Traner Cheddar", "TraiNeR CHeddaR I", "Trainer:", "trainer",
                                  "Tralner Cheddar", "TrainerCheddar", "Trainar Bob", "  Trainer  "])
def test_is_trainer_true(text):
    assert is_trainer(text)


@pytest.mark.parametrize("text", ["", None, " ", "36ono scope", "Tyler", "Training", "Tra", "Ryley", "Mia Khalifa",
                                  "Brainless", "Training Camp", "12345"])
def test_is_trainer_false(text):
    # "Training" is 3 edits away (the Training Camp screens say it, the banner does not): not a trainer. Empty text never is.
    assert not is_trainer(text)


def test_is_trainer_documented_tolerance():
    # within 2 edits is the spec's tolerance: a real player called "Trainee" or "Rainer" would also read as a trainer
    # (they would need 2 of the checked frames of one battle to flip the verdict)
    assert is_trainer("Trainee") and is_trainer("Rainer")


def test_edit_distance():
    assert edit_distance("trainer", "trainer") == 0 and edit_distance("traner", "trainer") == 1
    assert edit_distance("", "abc") == 3 and edit_distance("training", "trainer") == 3


def test_verdict_needs_two_trainer_frames():
    assert MIN_TRAINER_FRAMES == 2
    assert not verdict([]) and not verdict([True]) and not verdict([False] * 10)
    assert verdict([True, True]) and verdict([False, True, False, True])


def test_banner_box_scales_with_the_frame_and_stays_inside_it():
    calib = Calibration(content_rect_frac=[0.0253, 0.0546, 0.9494, 0.9339])
    full = banner_box((1392, 632, 3), calib)
    half = banner_box((696, 316, 3), calib)
    assert all(abs(a - 2 * b) <= 2 for a, b in zip(full, half))
    x0, y0, x1, y1 = full
    assert 0 <= x0 < x1 <= 632 and 0 <= y0 < y1 <= 1392
    assert y1 - y0 == pytest.approx(BANNER_FRAC[3] * 0.9339 * 1392, abs=1.5)         # one text line, not the clan line too
    assert (x1 - x0) / 632 == pytest.approx(BANNER_FRAC[2] * 0.9494, abs=0.01)
    assert banner_box((100, 100), Calibration())[2] <= 100                       # no content rect: the whole frame


def test_banner_box_accepts_a_dict_and_a_path(tmp_path):
    calib = Calibration(content_rect_frac=[0.1, 0.1, 0.8, 0.8])
    path = tmp_path / "c.json"
    calib.save(path)
    assert banner_box((200, 100), calib) == banner_box((200, 100), calib.__dict__) == banner_box((200, 100), path)


class FakeOcr:
    """A RapidOCR stand-in: records the crop it was given and the stage switches; answers rows in `rows`."""

    def __init__(self, rows, boom=False):
        self.rows, self.boom, self.calls = rows, boom, []

    def __call__(self, img, **kw):
        self.calls.append((img.shape, kw))
        if self.boom:
            raise RuntimeError("ocr died")
        return self.rows, None


def test_read_banner_crops_upscales_and_recognises_only():
    frame = np.zeros((1392, 632, 3), np.uint8)
    ocr = FakeOcr([["Trainer Cheddar II", 0.9]])
    assert read_banner(frame, Calibration(content_rect_frac=[0.0253, 0.0546, 0.9494, 0.9339]), ocr) == "Trainer Cheddar II"
    (h, w, _), kw = ocr.calls[0]
    x0, y0, x1, y1 = banner_box(frame.shape, Calibration(content_rect_frac=[0.0253, 0.0546, 0.9494, 0.9339]))
    assert w >= 2 * (x1 - x0) - 2 and h >= 2 * (y1 - y0) - 2                    # at least x2
    assert kw == {"use_det": False, "use_cls": False, "use_rec": True}


def test_read_banner_takes_detection_rows_left_to_right_and_an_ocr_reader_object():
    rows = [[[[80, 0], [120, 0], [120, 9], [80, 9]], "Cheddar", 0.9], [[[0, 0], [60, 0], [60, 9], [0, 9]], "Trainer", 0.9]]
    reader = type("Reader", (), {"ocr": FakeOcr(rows)})()          # an OcrReader keeps its RapidOCR in `.ocr`
    assert read_banner(np.zeros((696, 316, 3), np.uint8), Calibration(), reader) == "Trainer Cheddar"


def test_read_banner_is_empty_on_no_text_an_error_or_an_empty_crop():
    frame = np.zeros((696, 316, 3), np.uint8)
    assert read_banner(frame, Calibration(), FakeOcr(None)) == ""
    assert read_banner(frame, Calibration(), FakeOcr([], boom=True)) == ""
    assert read_banner(np.zeros((0, 0, 3), np.uint8), Calibration(), FakeOcr([["Trainer", 1.0]])) == ""
    assert read_banner(frame, Calibration(), FakeOcr([[None, 1.0], ["", 1.0]])) == ""


def test_read_banner_falls_back_to_a_plain_call_for_an_engine_without_stage_switches():
    seen = []

    def plain(img):
        seen.append(img.shape)
        return [[[[0, 0], [5, 0], [5, 5], [0, 5]], "Trainer", 0.9]], None

    assert read_banner(np.zeros((696, 316, 3), np.uint8), Calibration(), plain) == "Trainer" and len(seen) == 1


@needs_runs
def test_the_crop_of_real_shots_holds_the_pink_banner_text():
    for path in SHOTS:
        im = cv2.imread(str(path))
        x0, y0, x1, y1 = banner_box(im.shape, CALIB)
        crop = im[y0:y1, x0:x1]
        assert crop.shape[0] >= 12 and crop.shape[1] >= 80
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pink = ((hsv[..., 0] >= 140) | (hsv[..., 0] <= 5)) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 140)
        assert pink.mean() > 0.03, path.name                                      # the name line is in the box
        clan_row = im[y1 + 2:y1 + 10, x0:x0 + 60]                                  # the white clan line starts just below it
        assert clan_row.size and clan_row.mean() > 100


@needs_runs
@slow
def test_real_ocr_reads_trainer_on_real_shots_and_the_full_size_frame():
    from cr_perception.hud import OcrReader
    ocr = OcrReader()
    frames = [cv2.imread(str(p)) for p in SHOTS] + ([cv2.imread(str(FULL))] if FULL.exists() else [])
    for im in frames:
        text = read_banner(im, CALIB, ocr)
        assert is_trainer(text), text
    # a frame without the banner (the arena only) never reads as one
    plain = np.full((1392, 632, 3), (60, 140, 60), np.uint8)
    assert not is_trainer(read_banner(plain, CALIB, ocr))
