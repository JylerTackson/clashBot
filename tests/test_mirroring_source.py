"""MirroringSource: window lookup, permission check, steady-rate scheduler (fake clock, no real capture)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_perception import sources  # noqa: E402
from cr_perception.sources import MirroringSource, WindowRect, find_mirroring_window  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.slept = 0.0

    def clock(self):
        return self.now

    def sleep(self, dt):
        self.slept += max(0.0, dt)
        self.now += max(0.0, dt)


def make_grab(clock, cost_s, shape=(40, 30, 3)):
    rng = np.random.default_rng(0)

    def grab(mon):
        clock.now += cost_s
        return rng.integers(0, 255, size=shape, dtype=np.uint8)
    return grab


def WIN(owner, w, h, layer=0):
    return {"owner": owner, "title": "", "window_id": 1, "layer": layer,
            "rect": WindowRect(0, 0, w, h, owner, "")}


def make_source(fc, fps, cost_s, **kw):
    kw.setdefault("check_permission", False)
    return MirroringSource(rect=WindowRect(0, 0, 300, 650), target_fps=fps, grab_fn=make_grab(fc, cost_s),
                           clock=fc.clock, sleep=fc.sleep, **kw)


def test_find_mirroring_window():
    wins = [WIN("Finder", 800, 600), WIN("iPhone Mirroring", 300, 650), WIN("iPhone Mirroring", 50, 50)]
    r = find_mirroring_window(windows=wins)
    assert (r.width, r.height) == (300, 650)
    # layer-1 window is ignored even if bigger
    r = find_mirroring_window(windows=wins + [WIN("iPhone Mirroring", 900, 900, layer=1)])
    assert (r.width, r.height) == (300, 650)
    assert find_mirroring_window(windows=[WIN("Finder", 800, 600)]) is None
    assert find_mirroring_window(windows=[]) is None
    r = find_mirroring_window(owner_hint="finder", windows=wins)
    assert r is not None and r.owner == "Finder"


def test_permission_check_blank_and_empty():
    rect = WindowRect(0, 0, 300, 650)
    with pytest.raises(PermissionError, match="Screen Recording"):
        MirroringSource(rect=rect, grab_fn=lambda mon: np.zeros((40, 30, 3), np.uint8))
    with pytest.raises(RuntimeError, match="empty capture"):
        MirroringSource(rect=rect, grab_fn=lambda mon: np.zeros((0, 0, 3), np.uint8))


def test_permission_check_passes_and_looks_like_game_is_informational():
    fc = FakeClock()
    src = MirroringSource(rect=WindowRect(0, 0, 300, 650), grab_fn=make_grab(fc, 0.0), clock=fc.clock,
                          sleep=fc.sleep)   # random noise is not a match screen: must not raise
    assert src.looks_like_game in (True, False)


def test_window_not_found_raises(monkeypatch):
    monkeypatch.setattr(sources, "find_mirroring_window", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="iPhone Mirroring window not found"):
        MirroringSource(grab_fn=lambda mon: np.zeros((1, 1, 3), np.uint8))


def test_steady_rate():
    fc = FakeClock()
    src = make_source(fc, 10, 0.01)
    ts = []
    for frame, t, i in src.frames():
        assert frame.shape == (40, 30, 3) and i == len(ts)
        ts.append(t)
        if t >= 1.0 - 1e-6:   # fake-clock float drift puts frame 10 at 0.99999999999
            break
    assert 10 <= len(ts) <= 11
    deltas = np.diff(ts)
    assert np.allclose(deltas, 0.1, atol=0.02)
    st = src.stats()
    assert st["dropped"] == 0
    assert 9.5 <= st["fps"] <= 11
    assert 9.0 <= st["grab_ms_p50"] <= 11.0 and src.last_grab_ms == pytest.approx(10.0, abs=0.5)


def test_too_slow_drops_frames():
    fc = FakeClock()
    src = make_source(fc, 10, 0.25)
    ts = []
    for frame, t, i in src.frames():
        ts.append(t)
        if len(ts) == 8:
            break
    assert src.dropped >= 10
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_unthrottled_never_sleeps():
    fc = FakeClock()
    src = make_source(fc, 0, 0.01)
    ts = []
    for frame, t, i in src.frames():
        ts.append(t)
        if len(ts) == 5:
            break
    assert fc.slept == 0.0
    assert fc.now == pytest.approx(100.0 + 5 * 0.01)
    assert src.dropped == 0


def test_refind_follows_moved_window(monkeypatch):
    first = WindowRect(10, 20, 300, 650, "iPhone Mirroring", "")
    moved = WindowRect(50, 60, 300, 650, "iPhone Mirroring", "")
    calls = {"n": 0}

    def fake_find(owner_hint=None, windows=None):
        calls["n"] += 1
        return first if calls["n"] == 1 else moved
    monkeypatch.setattr(sources, "find_mirroring_window", fake_find)
    fc = FakeClock()
    src = MirroringSource(grab_fn=make_grab(fc, 0.01), clock=fc.clock, sleep=fc.sleep, target_fps=10,
                          check_permission=False, refind_every_s=0.5)
    assert src._auto and src.mon["left"] == 10 and src.rect_changes == 0
    for frame, t, i in src.frames():
        if t >= 1.0:
            break
    assert src.rect_changes == 1
    assert src.mon["left"] == 50 and src.mon["top"] == 60 and src.rect.left == 50


def test_manual_rect_is_not_refound(monkeypatch):
    monkeypatch.setattr(sources, "find_mirroring_window", lambda *a, **k: pytest.fail("must not look up"))
    fc = FakeClock()
    src = make_source(fc, 10, 0.01, refind_every_s=0.1)
    for frame, t, i in src.frames():
        if t >= 0.5:
            break
    assert src.rect_changes == 0


def test_stats_keys_and_scale():
    fc = FakeClock()
    src = make_source(fc, 10, 0.01)
    st0 = src.stats()
    assert st0["elapsed_s"] == 0.0 and st0["fps"] == 0.0 and st0["grab_ms_p50"] is None
    assert st0["frame_shape"] is None and st0["scale"] is None and src.last_grab_ms == 0.0
    for frame, t, i in src.frames():
        if i == 3:
            break
    st = src.stats()
    assert set(st) == {"frames", "dropped", "elapsed_s", "fps", "target_fps", "grab_ms_p50", "grab_ms_p95",
                       "frame_shape", "scale", "rect", "rect_changes", "looks_like_game"}
    assert st["frame_shape"] == [40, 30]
    assert st["scale"] == 30 / src.rect.width
    assert st["rect"]["width"] == 300 and st["target_fps"] == 10


def test_sources_module_is_read_only():
    text = (ROOT / "cr_perception" / "sources.py").read_text()
    assert "CGEventCreate" not in text and "CGEventPost" not in text


def test_benchmark_cli_help():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_capture.py"), "--help"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "--sweep" in r.stdout and "--list-windows" in r.stdout
