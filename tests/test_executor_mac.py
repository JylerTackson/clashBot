"""MAC-F: screen mapping from the perception calibration, MirroringExecutor with the double live guard,
the Quartz poster (never called here: tests use a recording poster) and the overlay tool."""
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot import executor as ex_mod  # noqa: E402
from clashbot.executor import (LIVE_ENV, ARENA_TILE_CORNERS, Calibration, DryRunExecutor,  # noqa: E402
                               MirroringExecutor, calibration_from_screen_map, card_slot_points,
                               check_live_allowed, content_to_screen, homography, quartz_click,
                               screen_map_from_calib, screen_to_image, tile_target, tile_to_screen)
from cr_perception.geometry import Homography  # noqa: E402
from cr_perception.hud import DEFAULT_ROIS  # noqa: E402

RECT = [100, 50, 300, 600]
CORNERS = [[0.1, 0.75], [0.9, 0.75], [0.88, 0.1], [0.12, 0.1]]
CALIB = {
    "content_rect_frac": [0.0, 0.1, 1.0, 0.8],
    "arena_corners_frac": CORNERS,
    "rois": {f"hand_{i}": DEFAULT_ROIS[f"hand_{i}"] for i in range(4)},
}


def _smap():
    return screen_map_from_calib(CALIB, RECT)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, x, y):
        self.calls.append((x, y))


def _live_executor(monkeypatch, poster, **kw):
    monkeypatch.setenv(LIVE_ENV, "1")
    monkeypatch.setattr(sys, "platform", "darwin")
    return MirroringExecutor.from_calib(CALIB, RECT, live=True, poster=poster, sleep=lambda s: None, **kw)


def test_content_to_screen_and_back():
    smap = _smap()
    assert content_to_screen(0.5, 0.5, smap) == (250.0, 350.0)
    # image of a 600x1200 frame (Retina: 2x the 300x600 points)
    x, y = screen_to_image(250.0, 350.0, smap, 600, 1200)
    assert (x, y) == pytest.approx((300.0, 600.0))
    assert screen_to_image(100.0, 50.0, smap, 600, 1200) == pytest.approx((0.0, 0.0))
    assert screen_to_image(400.0, 650.0, smap, 600, 1200) == pytest.approx((600.0, 1200.0))


def test_screen_map_inputs(tmp_path):
    class Rect:
        left, top, width, height = 100, 50, 300, 600

    assert screen_map_from_calib(CALIB, Rect()).rect == [100.0, 50.0, 300.0, 600.0]
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(CALIB))
    assert screen_map_from_calib(p, RECT).arena_corners_frac == CORNERS
    assert screen_map_from_calib(str(p), RECT).content_rect_frac == [0.0, 0.1, 1.0, 0.8]
    # missing content rect and hand ROIs fall back to the defaults
    smap = screen_map_from_calib({"arena_corners_frac": CORNERS}, RECT)
    assert smap.content_rect_frac == [0.0, 0.0, 1.0, 1.0]
    assert smap.hand_rois == [list(DEFAULT_ROIS[f"hand_{i}"]) for i in range(4)]
    with pytest.raises(ValueError, match="arena_corners_frac"):
        screen_map_from_calib({"content_rect_frac": [0, 0, 1, 1]}, RECT)


def test_executor_homography_maps_tile_corners():
    smap = _smap()
    cal = calibration_from_screen_map(smap)
    assert cal.src_tiles == ARENA_TILE_CORNERS
    H = homography(cal)
    bl = content_to_screen(*CORNERS[0], smap)
    tr = content_to_screen(*CORNERS[2], smap)
    assert bl == pytest.approx((130.0, 470.0))
    assert tile_to_screen([0, 0], H) == pytest.approx(bl, abs=1e-6)
    assert tile_to_screen([18, 32], H) == pytest.approx(tr, abs=1e-6)


def test_perception_parity():
    smap = _smap()
    H_exec = homography(calibration_from_screen_map(smap))
    cw, ch = 300, 480
    percep = Homography.from_corners([(x * cw, y * ch) for x, y in CORNERS])
    for c, r in [(0, 0), (8, 14), (17, 31), (3, 4)]:
        px, py = percep.tile_to_pixel(c, r, centre=True)
        want = content_to_screen(px / cw, py / ch, smap)
        got = tile_to_screen(tile_target([c, r]), H_exec)
        assert got == pytest.approx(want, abs=1e-6)


def test_card_slot_points_and_tile_target():
    smap = _smap()
    pts = card_slot_points(smap)
    assert len(pts) == 4
    for i, pt in enumerate(pts):
        x, y, w, h = DEFAULT_ROIS[f"hand_{i}"]
        assert pt == pytest.approx(content_to_screen(x + w / 2, y + h / 2, smap))
    assert tile_target([3, 14]) == [3.5, 14.5]
    assert tile_target([-2, 40]) == [0.5, 31.5]
    assert tile_target([17, 31]) == [17.5, 31.5]
    assert tile_target([99, -1]) == [17.5, 0.5]


def test_plan_clicks():
    ex = MirroringExecutor.from_calib(CALIB, RECT)
    assert ex.plan_clicks("hold", None, ["a", "b", "knight", "c"]) == {"status": "hold", "slot": None, "clicks": []}
    assert ex.plan_clicks("knight|left|bridge", None, ["knight"])["status"] == "hold"
    p = ex.plan_clicks("knight|left|bridge", [3, 14], ["a", "b", "c", "d"])
    assert p == {"status": "skipped_card_not_in_hand", "slot": None, "clicks": []}
    assert ex.plan_clicks("knight|left|bridge", [3, 14], None)["status"] == "skipped_card_not_in_hand"

    p = ex.plan_clicks("knight|left|bridge", [3, 14], ["a", "b", "knight", "c"])
    assert p["status"] == "planned" and p["slot"] == 2 and len(p["clicks"]) == 2
    assert p["clicks"][0] == card_slot_points(_smap())[2]
    assert p["clicks"][1] == pytest.approx(tile_to_screen([3.5, 14.5], ex.H))

    # a target outside the window bounds is not planned
    small = MirroringExecutor(ex.calibration, ex.card_slot_pixels, bounds=[100, 50, 300, 100])
    p = small.plan_clicks("knight|left|bridge", [3, 2], ["a", "b", "knight", "c"])
    assert p["status"] == "skipped_out_of_window" and p["slot"] == 2 and len(p["clicks"]) == 2


def test_update_rect_shifts_every_click():
    hand = ["a", "b", "knight", "c"]
    ex = MirroringExecutor.from_calib(CALIB, RECT)
    keys = [("knight|left|bridge", [3, 14]), ("knight|right|mid", [14, 8]), ("knight|middle|back", [9, 2])]
    before = [ex.plan_clicks(k, t, hand) for k, t in keys]
    assert all(p["status"] == "planned" for p in before)

    class Rect:   # WindowRect-like object, moved by (+40, +25)
        left, top, width, height = RECT[0] + 40, RECT[1] + 25, RECT[2], RECT[3]

    ex.update_rect(Rect)
    after = [ex.plan_clicks(k, t, hand) for k, t in keys]
    for b, a in zip(before, after):
        assert a["status"] == "planned" and a["slot"] == b["slot"]
        for (bx, by), (ax, ay) in zip(b["clicks"], a["clicks"]):
            assert ax - bx == pytest.approx(40.0, abs=1e-6)
            assert ay - by == pytest.approx(25.0, abs=1e-6)
    want = card_slot_points(screen_map_from_calib(CALIB, [140, 75, 300, 600]))
    assert np.allclose(ex.card_slot_pixels, want, atol=1e-6)
    assert ex.bounds == [140.0, 75.0, 300.0, 600.0] and ex.screen_map.rect == ex.bounds
    assert ex.log[-1] == {"event": "rect_update", "rect": [140.0, 75.0, 300.0, 600.0]}
    # bounds moved too: a click that was inside the old window's right edge is now checked against the new one
    ex.update_rect([100, 50, 300, 600])
    assert np.allclose(ex.plan_clicks(*keys[0], hand)["clicks"], before[0]["clicks"], atol=1e-6)
    assert [e["event"] for e in ex.log] == ["rect_update", "rect_update"]


def test_update_rect_needs_from_calib():
    ex = MirroringExecutor.from_calib(CALIB, RECT)
    direct = MirroringExecutor(ex.calibration, ex.card_slot_pixels, bounds=list(RECT))
    with pytest.raises(ValueError, match="from_calib"):
        direct.update_rect([140, 75, 300, 600])
    assert direct.log == [] and direct.bounds == list(RECT)


def test_update_rect_keeps_live_and_poster(monkeypatch):
    rec = Recorder()
    ex = _live_executor(monkeypatch, rec)
    hand = ["a", "b", "knight", "c"]
    ex.update_rect([140, 75, 300, 600])
    assert ex.live is True and ex.poster is rec
    e = ex.execute("knight|left|bridge", [3, 14], 1.0, hand=hand)
    assert e["sent"] is True and rec.calls == [tuple(c) for c in e["clicks"]]
    # clicks went to the moved window, inside its bounds
    assert all(140 <= x <= 440 and 75 <= y <= 675 for x, y in rec.calls)
    # the live guard is still re-checked after a move
    monkeypatch.delenv(LIVE_ENV)
    with pytest.raises(PermissionError):
        ex.execute("knight|left|bridge", [3, 14], 2.0, hand=hand)
    assert len(rec.calls) == 2
    # a dry executor stays dry
    dry = MirroringExecutor.from_calib(CALIB, RECT, poster=rec)
    dry.update_rect([140, 75, 300, 600])
    assert dry.live is False and dry.execute("knight|left|bridge", [3, 14], 3.0, hand=hand)["sent"] is False
    assert len(rec.calls) == 2


def test_not_live_never_posts():
    rec = Recorder()
    ex = MirroringExecutor.from_calib(CALIB, RECT, live=False, poster=rec, sleep=lambda s: None)
    e = ex.execute("knight|left|bridge", [3, 14], 1.0, hand=["a", "b", "knight", "c"])
    assert e["status"] == "planned" and e["sent"] is False
    assert rec.calls == []
    assert ex.log == [e] and ex.screen_map is not None
    assert MirroringExecutor(ex.calibration, ex.card_slot_pixels).screen_map is None


def test_live_requires_env_and_darwin(monkeypatch):
    rec = Recorder()
    monkeypatch.delenv(LIVE_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(PermissionError, match=LIVE_ENV):
        MirroringExecutor.from_calib(CALIB, RECT, live=True, poster=rec)
    monkeypatch.setenv(LIVE_ENV, "1")
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(PermissionError, match="macOS"):
        MirroringExecutor.from_calib(CALIB, RECT, live=True, poster=rec)
    monkeypatch.setenv(LIVE_ENV, "true")      # only "1" counts
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(PermissionError):
        check_live_allowed()
    assert rec.calls == []


def test_live_sends_two_clicks_then_guard_recheck(monkeypatch):
    rec = Recorder()
    sleeps = []
    ex = _live_executor(monkeypatch, rec)
    ex.sleep = sleeps.append
    hand = ["a", "b", "knight", "c"]
    plan = ex.plan_clicks("knight|left|bridge", [3, 14], hand)
    e = ex.execute("knight|left|bridge", [3, 14], 2.0, hand=hand)
    assert e["sent"] is True and e["status"] == "planned"
    assert rec.calls == [tuple(c) for c in plan["clicks"]] and len(rec.calls) == 2
    assert sleeps == [ex.click_gap_s, ex.click_gap_s] == [0.05, 0.05]

    # holds and skipped plays are never sent even when live
    assert ex.execute("hold", None, 3.0, hand=hand)["sent"] is False
    assert ex.execute("zap|left|bridge", [3, 14], 3.5, hand=hand)["sent"] is False
    assert len(rec.calls) == 2

    # the env switch disappearing after construction stops the next tap
    monkeypatch.delenv(LIVE_ENV)
    with pytest.raises(PermissionError):
        ex.execute("knight|left|bridge", [3, 14], 4.0, hand=hand)
    assert len(rec.calls) == 2
    assert len(ex.log) == 3


def test_dry_run_executor_entries():
    dry = DryRunExecutor()
    a = dry.execute("hold", None, 1.0)
    b = dry.execute("knight|left|bridge", [3, 14], 1.0, hand=["knight", "a", "b", "c"])
    assert a == {"t": 1.0, "key": "hold", "tile": None}
    assert b == {"t": 1.0, "key": "knight|left|bridge", "tile": [3, 14]}
    assert dry.log == [a, b]


def test_old_api_unchanged():
    src = [[0, 0], [10, 0], [10, 10], [0, 10]]
    dst = [[100 + 10 * x, 100 + 10 * y] for x, y in src]
    H = homography(Calibration(src, dst))
    x, y = tile_to_screen([1, 2], H)
    assert abs(x - 110) < 1e-6 and abs(y - 120) < 1e-6
    with pytest.raises(ValueError):
        homography(Calibration(src[:3], dst[:3]))


def _load_overlay():
    spec = importlib.util.spec_from_file_location("executor_overlay", ROOT / "tools" / "executor_overlay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_overlay_tool(tmp_path, capsys):
    overlay = _load_overlay()
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps(CALIB))
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), np.full((1200, 600, 3), 40, np.uint8))
    out = tmp_path / "overlay.png"
    entries = overlay.main(["--calib", str(calib), "--image", str(img), "--out", str(out),
                            "--prior", str(tmp_path / "no_prior.json")])
    assert out.exists()
    drawn = cv2.imread(str(out))
    assert drawn.shape == (1200, 600, 3) and not np.array_equal(drawn, cv2.imread(str(img)))
    assert len(entries) == 4
    assert all(e["status"] == "planned" and len(e["clicks_screen"]) == 2 and len(e["clicks_image"]) == 2 for e in entries)
    assert [e["key"] for e in entries][0] == "knight|left|bridge"
    assert entries[0]["tile"] == [3.0, 14.0]
    printed = json.loads(capsys.readouterr().out)
    assert [e["key"] for e in printed] == [e["key"] for e in entries]

    # explicit rect (points) on a 2x capture: image coordinates are 2x the screen points minus the origin
    entries = overlay.main(["--calib", str(calib), "--image", str(img), "--out", str(out), "--rect", "100,50,300,600",
                            "--keys", "knight|left|bridge", "--hand", "knight,a,b,c",
                            "--prior", str(tmp_path / "no_prior.json")])
    (sx, sy), (tx, ty) = entries[0]["clicks_screen"]
    (ix, iy), (jx, jy) = entries[0]["clicks_image"]
    assert (ix, iy) == pytest.approx(((sx - 100) * 2, (sy - 50) * 2))
    assert (jx, jy) == pytest.approx(((tx - 100) * 2, (ty - 50) * 2))


def test_overlay_never_live():
    src = (ROOT / "tools" / "executor_overlay.py").read_text()
    assert "live=True" not in src and "quartz_click" not in src


def test_cgeventpost_only_in_quartz_click():
    src = Path(ex_mod.__file__).read_text()
    assert "CGEventPost" in inspect.getsource(quartz_click)
    assert src.count("CGEventPost") == inspect.getsource(quartz_click).count("CGEventPost")
    offenders = []
    for sub in ("clashbot", "cr_perception", "tools"):
        for path in (ROOT / sub).rglob("*.py"):
            if path.resolve() != Path(ex_mod.__file__).resolve() and "CGEventPost" in path.read_text():
                offenders.append(str(path))
    assert offenders == []


def test_click_gap_default_and_custom(monkeypatch):
    """FIX_LIST #22: one planned play sleeps click_gap_s after each of its two clicks (default 0.05 s)."""
    hand = ["a", "b", "knight", "c"]
    monkeypatch.setenv(LIVE_ENV, "1")
    monkeypatch.setattr(sys, "platform", "darwin")

    rec, sleeps = Recorder(), []
    ex = MirroringExecutor.from_calib(CALIB, RECT, live=True, poster=rec, sleep=sleeps.append)
    assert ex.click_gap_s == 0.05
    plan = ex.plan_clicks("knight|left|bridge", [3, 14], hand)
    assert ex.execute("knight|left|bridge", [3, 14], 1.0, hand=hand)["sent"] is True
    assert sleeps == [0.05, 0.05]
    assert rec.calls == [tuple(c) for c in plan["clicks"]]

    rec2, sleeps2 = Recorder(), []
    ex2 = MirroringExecutor.from_calib(CALIB, RECT, live=True, poster=rec2, sleep=sleeps2.append, click_gap_s=0.2)
    assert ex2.execute("knight|left|bridge", [3, 14], 1.0, hand=hand)["sent"] is True
    assert sleeps2 == [0.2, 0.2]
    assert rec2.calls == rec.calls                       # same points regardless of timing


def test_quartz_click_event_gap_default_and_custom(monkeypatch):
    """quartz_click paces its three events with event_gap_s (default 0.01 s). Tested with a fake Quartz
    module and a fake sleep, so no input is ever posted."""
    posted, sleeps = [], []

    class FakeQuartz:
        kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp = "move", "down", "up"
        kCGMouseButtonLeft, kCGHIDEventTap = "left", "hid"

        @staticmethod
        def CGEventCreateMouseEvent(source, kind, pos, button):
            return (kind, pos, button)

        @staticmethod
        def CGEventPost(tap, ev):
            posted.append(ev)

    monkeypatch.setitem(sys.modules, "Quartz", FakeQuartz)
    monkeypatch.setattr(ex_mod.time, "sleep", sleeps.append)
    quartz_click(10, 20)
    assert [e[0] for e in posted] == ["move", "down", "up"] and all(e[1] == (10.0, 20.0) for e in posted)
    assert sleeps == [0.01, 0.01, 0.01]
    sleeps.clear()
    quartz_click(10, 20, event_gap_s=0.03)
    assert sleeps == [0.03, 0.03, 0.03]
