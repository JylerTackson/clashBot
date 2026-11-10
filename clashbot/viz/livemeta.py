"""What the browser needs to play a run's recording and draw the agent's taps on it (MAC-AW).

`live_meta.json` is written into the run dir by `python -m clashbot.live --source video --realtime` when the first frame is
delivered: the wall-clock start (epoch seconds), the video path and the video time of that first frame, the frame rate, the
stride, and the arena geometry of the calibration, so the page can map a tile to a pixel of the video. The mapping is the
executor's own (`clashbot.executor`: the tile -> screen homography the dry run and the shot recorder use, with the
tile-centre rule of `tile_target`), evaluated on a unit window so that its result is a fraction of the frame; the page
multiplies by the size it draws the video at.

A run without the file (an older run, or one made without `--realtime`) gets the same fields derived from its `config.json`
(`derive`), minus the wall-clock start: such a run can be replayed but was not paced on the wall clock.

Standard library plus `clashbot.executor` (numpy); nothing here writes outside the run dir or sends input.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from clashbot.config import ROOT

LIVE_META = "live_meta.json"
VERSION = 1


def arena_geometry(calib) -> dict | None:
    """The arena of a calibration in fractions of the video frame, from the executor's tile -> screen mapping evaluated on a
    unit window ([0, 0, 1, 1]): `grid` {cols, rows}; `corners` bl, br, tr, tl; `rect` [x0, y0, x1, y1] (their bounding box);
    `homography` 3x3, tile-target (col, row) -> frame fraction (apply it to `tile_target(tile)`, i.e. tile + 0.5 clamped
    into the arena); `slots` the 4 hand-slot centres. None when the calibration cannot be read."""
    try:
        from clashbot.executor import (GRID_COLS, GRID_ROWS, calibration_from_screen_map, card_slot_points, homography,
                                       screen_map_from_calib)
        smap = screen_map_from_calib(calib, [0.0, 0.0, 1.0, 1.0])
        cal = calibration_from_screen_map(smap)
        h = homography(cal)
        slots = card_slot_points(smap)
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return None
    corners = [[round(float(x), 5), round(float(y), 5)] for x, y in cal.dst_pixels]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return {"grid": {"cols": GRID_COLS, "rows": GRID_ROWS}, "corners": corners,
            "rect": [min(xs), min(ys), max(xs), max(ys)],
            "homography": [[float(v) for v in row] for row in h.tolist()],
            "slots": [[round(float(x), 5), round(float(y), 5)] for x, y in slots]}


def build(video, video_t0: float, wall_start: float | None, fps: float, stride: int, realtime: bool, calib,
          frame_size=None, first_idx: int | None = None) -> dict:
    """The live_meta.json document."""
    return {"version": VERSION, "wall_start": wall_start, "video": str(video), "video_t0": float(video_t0),
            "first_idx": first_idx, "fps": float(fps), "stride": int(stride), "rate_hz": round(float(fps) / max(int(stride), 1), 3),
            "realtime": bool(realtime), "frame_size": list(frame_size) if frame_size else None,
            "calib": str(calib) if calib is not None and not isinstance(calib, dict) else None,
            "arena": arena_geometry(calib) if calib is not None else None}


def write(out_dir: Path | str, meta: dict) -> Path:
    """Write live_meta.json atomically (a reader never sees half a file)."""
    path = Path(out_dir) / LIVE_META
    tmp = path.with_name(LIVE_META + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)
    return path


def read(run_dir: Path | str) -> dict | None:
    try:
        data = json.loads((Path(run_dir) / LIVE_META).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def resolve(path: str | os.PathLike | None, root: Path = ROOT) -> Path:
    """A path out of a config: relative ones are taken from the repository root (where the runs were started)."""
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else root / p


def video_path(config: dict, root: Path = ROOT) -> Path | None:
    """The source video named in a run's config: an existing `.mp4` file, else None. This is the only file the video
    endpoint ever serves for the run."""
    name = config.get("video") if isinstance(config, dict) else None
    if not name or not isinstance(name, str):
        return None
    path = resolve(name, root)
    if path.suffix.lower() != ".mp4" or not path.is_file():
        return None
    return path


def derive(config: dict, root: Path = ROOT) -> dict | None:
    """The live_meta of a run that has none, from its config.json: the video, its start, stride and the calibration it was
    run with. `wall_start` stays None (such a run was not paced on the wall clock). None without a video in the config."""
    if not isinstance(config, dict) or not config.get("video"):
        return None
    calib = config.get("calib")
    calib_path = resolve(calib, root) if calib else None
    meta = build(config["video"], float(config.get("start") or 0.0), None, 0.0, int(config.get("stride") or 1),
                 bool(config.get("realtime")), calib_path if calib_path is not None and calib_path.is_file() else None)
    meta["fps"] = meta["rate_hz"] = None                 # the config's --fps is the live capture rate, not the video's
    return meta


def last_row(path: Path, tail_bytes: int = 8192) -> dict | None:
    """The last complete JSON object line of a (possibly growing) JSON-lines file, read from its tail."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            start = max(size - tail_bytes, 0)
            fh.seek(start)
            chunk = fh.read()
    except OSError:
        return None
    lines = chunk.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()                                      # the file ends with a newline: every line is complete
    elif lines:
        lines.pop()                                      # the last line is still being written
    if start > 0 and lines:
        lines.pop(0)                                     # the first line of the window may be cut
    for raw in reversed(lines):
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict):
            return row
    return None
