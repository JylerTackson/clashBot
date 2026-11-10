"""MAC-AW: VideoFrameSource(realtime=True) paces a recording on the wall clock (fake clock, a tiny generated mp4), the
`--realtime` flag, and live_meta.json (the tile -> frame mapping is the executor's own)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from clashbot import live  # noqa: E402
from clashbot.executor import GRID_COLS, GRID_ROWS, MirroringExecutor, tile_target, tile_to_screen  # noqa: E402
from clashbot.viz import livemeta  # noqa: E402
from cr_perception.sources import VideoFrameSource  # noqa: E402

FPS = 10.0
N_FRAMES = 60                                      # a 6 s clip at 10 fps: frame k is at k / 10 s, its pixels are all ~4 * k


class Clock:
    """A clock that only moves when somebody sleeps or the 'consumer' works."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        assert s >= 0
        self.now += s


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("rt") / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    if not w.isOpened():
        pytest.skip("no mp4 writer")
    for i in range(N_FRAMES):
        w.write(np.full((48, 64, 3), 4 * i, np.uint8))
    w.release()
    return path


def source(clip, clock=None, **kw):
    clock = clock or Clock()
    return VideoFrameSource(clip, realtime=True, clock=clock, sleep=clock.sleep, wall=lambda: 1_700_000_000.0, **kw), clock


def drain(src, clock, work=0.0, limit=None):
    """Consume the source; `work` seconds of fake time pass between two frames (the loop's processing time)."""
    out = []
    for frame, t, idx in src.frames():
        out.append((clock.now, t, idx, int(frame[0, 0, 0])))
        if limit and len(out) >= limit:
            break
        clock.now += work
    return out


def test_a_fast_consumer_gets_every_frame_exactly_on_time(clip):
    src, clock = source(clip)
    got = drain(src, clock)
    assert [g[2] for g in got] == list(range(N_FRAMES)) and src.dropped == 0
    wall0, t0 = got[0][0], got[0][1]
    for wall, t, _idx, _px in got:
        assert wall == pytest.approx(wall0 + (t - t0), abs=1e-9)          # not a moment early, not late
    assert src.wall_start == 1_700_000_000.0 and src.video_t0 == 0.0
    st = src.stats()
    assert st["frames"] == N_FRAMES and st["dropped"] == 0 and st["realtime"] is True and st["late_ms_max"] == pytest.approx(0.0, abs=1e-6)


def test_no_frame_is_ever_delivered_ahead_of_its_time_whatever_the_consumer_does(clip):
    for work in (0.0, 0.03, 0.1, 0.25, 0.7):
        src, clock = source(clip)
        got = drain(src, clock, work=work)
        wall0, t0 = got[0][0], got[0][1]
        assert all(wall >= wall0 + (t - t0) - 1e-9 for wall, t, *_ in got), work
        assert [g[1] for g in got] == sorted(g[1] for g in got) and len({g[2] for g in got}) == len(got), work


def test_a_slow_loop_drops_frames_and_never_queues_them(clip):
    src, clock = source(clip)
    got = drain(src, clock, work=0.45)                                    # 4.5 frame periods per frame
    idxs = [g[2] for g in got]
    assert idxs[0] == 0 and len(idxs) < N_FRAMES // 3
    assert src.dropped >= (idxs[-1] + 1) - len(idxs)                       # every slot up to the last delivered one: delivered or dropped
    # each delivered frame is the newest one whose time had passed when the loop asked: it is at most one frame period old
    wall0 = got[0][0]
    for wall, t, idx, px in got[1:]:
        assert abs(px - 4 * idx) <= 8                                     # the pixels belong to the index (lossy mp4v, no stale buffer)
        assert wall - (wall0 + t) < 1.0 / FPS + 1e-9, (wall, t)           # never behind by more than a frame period
    gaps = [b - a for a, b in zip(idxs, idxs[1:])]
    assert min(gaps) >= 4 and max(gaps) <= 6                              # ~4.5 periods of work -> every 4th-6th frame
    assert src.stats()["dropped"] == src.dropped > 0


def test_every_frame_of_the_clip_is_either_delivered_or_counted_as_dropped(clip):
    for stride in (1, 2, 3):
        for work in (0.0, 0.25, 0.55):
            src, clock = source(clip, stride=stride)
            drain(src, clock, work=work)
            assert src.frames_out + src.dropped == -(-N_FRAMES // stride), (stride, work)       # the slots of the stride grid, none twice


def test_stride_still_caps_the_rate(clip):
    src, clock = source(clip, stride=3)
    got = drain(src, clock)
    assert [g[2] for g in got] == list(range(0, N_FRAMES, 3)) and src.dropped == 0
    assert [round(b[1] - a[1], 6) for a, b in zip(got, got[1:])] == [pytest.approx(0.3)] * (len(got) - 1)
    assert all(wall == pytest.approx(got[0][0] + t, abs=1e-9) for wall, t, *_ in got)
    # a slow loop with a stride only ever sees frames of the stride grid
    src, clock = source(clip, stride=3)
    slow = drain(src, clock, work=0.5)
    assert all(g[2] % 3 == 0 for g in slow) and src.dropped > 0


def test_start_and_end_are_honoured(clip):
    src, clock = source(clip, start=2.0, end=3.5)
    got = drain(src, clock)
    assert got[0][1] == pytest.approx(2.0, abs=0.11) and got[-1][1] <= 3.5
    assert src.video_t0 == pytest.approx(got[0][1])
    assert got[-1][0] - got[0][0] == pytest.approx(got[-1][1] - got[0][1], abs=1e-9)


def test_the_decode_time_excludes_the_pacing_wait(clip):
    src, clock = source(clip)
    drain(src, clock, limit=5)
    assert src.last_grab_ms == pytest.approx(0.0, abs=1e-6)             # the fake clock does not move while decoding


def test_a_plain_source_is_unchanged(clip):
    plain = VideoFrameSource(clip, start=1.0, end=2.5, stride=3)
    got = [(round(t, 3), idx) for _f, t, idx in plain.frames()]
    plain.close()
    assert got[0] == (1.0, 10) and [i for _t, i in got] == [10, 13, 16, 19, 22, 25]
    assert not hasattr(plain, "stats") and not hasattr(plain, "wall_start") and not hasattr(plain, "last_grab_ms")
    assert plain.realtime is False


def test_the_realtime_flag_is_off_by_default_and_needs_a_video():
    args = live.build_parser().parse_args(["--calib", "c.json", "--deck", "a,b,c,d,e,f,g,h"])
    assert args.realtime is False
    args = live.build_parser().parse_args(["--calib", "c.json", "--deck", "a,b,c,d,e,f,g,h", "--source", "video", "--video", "v.mp4", "--realtime"])
    assert args.realtime is True
    assert live.main(["--calib", "c.json", "--deck", "a,b,c,d,e,f,g,h", "--realtime"]) == 2        # mirroring source: refused before anything loads


def test_the_loop_runs_the_first_frame_hook_once_with_the_first_frames_time():
    calls = []

    class Src:
        last_grab_ms = 1.0

        def frames(self):
            for i in range(4):
                yield f"f{i}", 5.0 + 0.1 * i, 50 + i

        def close(self):
            pass

    from tests.test_live_loop import FakeBuilder, FakePerception, _loop
    lp = _loop(source=Src(), builder=FakeBuilder(warmup=9), perception=FakePerception())
    lp.on_first_frame = lambda t, idx: calls.append((t, idx))
    lp.run()
    assert calls == [(5.0, 50)]


# -- live_meta.json ---------------------------------------------------------------------------------------------------
CALIB = ROOT / "configs" / "mirror_calib.example.json"
needs_calib = pytest.mark.skipif(not CALIB.exists(), reason="configs/mirror_calib.example.json missing")


@needs_calib
def test_the_arena_mapping_is_the_executors(tmp_path):
    geo = livemeta.arena_geometry(CALIB)
    assert geo["grid"] == {"cols": GRID_COLS, "rows": GRID_ROWS} and len(geo["corners"]) == 4 and len(geo["slots"]) == 4
    x0, y0, x1, y1 = geo["rect"]
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    import numpy as _np
    h = _np.array(geo["homography"])
    ex = MirroringExecutor.from_calib(CALIB, [0.0, 0.0, 596.0, 1280.0], live=False)       # what the shot recorder draws on a 596x1280 frame
    for tile in ([3.0, 9.0], [0.0, 0.0], [17.0, 31.0], [8.5, 15.2]):
        want = tile_to_screen(tile_target(tile), ex.H)                                     # frame pixels, from the executor itself
        got = tile_to_screen(tile_target(tile), h)                                         # frame fractions, from live_meta
        assert got[0] * 596 == pytest.approx(want[0], abs=0.05) and got[1] * 1280 == pytest.approx(want[1], abs=0.05)
    # the 4 arena corners are the images of the tile corners
    assert tile_to_screen([0.0, 0.0], h) == pytest.approx(tuple(geo["corners"][0]), abs=1e-4)
    assert tile_to_screen([18.0, 32.0], h) == pytest.approx(tuple(geo["corners"][2]), abs=1e-4)


@needs_calib
def test_live_meta_file_has_what_the_browser_needs(tmp_path, clip):
    src, clock = source(clip, stride=3)
    args = live.build_parser().parse_args(["--calib", str(CALIB), "--deck", "a,b,c,d,e,f,g,h", "--source", "video", "--video", str(clip),
                                           "--stride", "3", "--realtime"])
    next(iter(src.frames()))
    path = live.write_live_meta(tmp_path, args, src, 0.0, 0)
    meta = json.loads(path.read_text())
    assert path.name == "live_meta.json" and not list(tmp_path.glob("*.tmp"))
    assert meta["wall_start"] == 1_700_000_000.0 and meta["video"] == str(clip) and meta["video_t0"] == 0.0
    assert meta["fps"] == pytest.approx(FPS) and meta["stride"] == 3 and meta["realtime"] is True and meta["rate_hz"] == pytest.approx(FPS / 3, abs=1e-3)
    assert meta["frame_size"] == [64, 48] and meta["arena"]["grid"] == {"cols": 18, "rows": 32}
    assert livemeta.read(tmp_path) == meta


def test_derive_from_a_config_has_no_wall_start(tmp_path, clip):
    cfg = {"video": str(clip), "start": 12.5, "stride": 3, "fps": 10.0, "calib": str(CALIB)}
    meta = livemeta.derive(cfg)
    assert meta["wall_start"] is None and meta["video_t0"] == 12.5 and meta["stride"] == 3 and meta["realtime"] is False
    assert meta["fps"] is None and (meta["arena"] is None or meta["arena"]["grid"]["rows"] == 32)
    assert livemeta.derive({"video": None}) is None and livemeta.derive({}) is None


def test_video_path_serves_only_an_existing_mp4(tmp_path, clip):
    assert livemeta.video_path({"video": str(clip)}) == clip
    other = tmp_path / "x.avi"
    other.write_bytes(b"x")
    assert livemeta.video_path({"video": str(other)}) is None                      # not an mp4
    assert livemeta.video_path({"video": str(tmp_path / "missing.mp4")}) is None   # not there
    assert livemeta.video_path({"video": None}) is None and livemeta.video_path({"video": 5}) is None and livemeta.video_path({}) is None
    rel = livemeta.video_path({"video": "clip.mp4"}, root=clip.parent)               # relative paths are taken from the repository root
    assert rel == clip.parent / "clip.mp4"


def test_last_row_reads_the_tail_of_a_growing_file(tmp_path):
    p = tmp_path / "frames.jsonl"
    assert livemeta.last_row(p) is None
    p.write_text("".join(json.dumps({"t": k * 0.1, "pad": "x" * 200}) + "\n" for k in range(200)) + '{"t": 99.9, "pa')
    assert livemeta.last_row(p)["t"] == pytest.approx(19.9)                        # the half-written last line is skipped
    p.write_text(json.dumps({"t": 1.0}) + "\n")
    assert livemeta.last_row(p) == {"t": 1.0}


def test_the_docker_entrypoint_plays_the_clip_in_real_time_and_still_cannot_tap():
    text = (ROOT / "docker" / "entrypoint.sh").read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "--realtime" in code and "--stride 3" in code and "DEMO_SECONDS:-60" in code and "--source video" in code
    assert "--live" not in code and "CLASHBOT_ALLOW_LIVE" not in code
