#!/usr/bin/env python3
"""Artefact builders of the Docker submission (MAC-AV): called by scripts/make_docker_submission.sh.

    python scripts/docker_artifacts.py clip   SRC.mp4 DST.mp4 [--start 60] [--seconds 90]
    python scripts/docker_artifacts.py slim-db SRC.sqlite DST.sqlite

Each one only reads its source (the sqlite file is opened read-only) and writes a new file. No network, no model load, no
input sent anywhere. (The d1 weights are not packaged: the image downloads them from Hugging Face at build time.)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

def cut_clip(src: Path, dst: Path, start: float = 60.0, seconds: float = 90.0) -> dict:
    """Re-encode `seconds` of `src` from `start` with OpenCV (H.264 'avc1' where the platform has it, else 'mp4v').
    The frame size is kept: the calibration is a set of fractions of the frame and the readers were tuned on it."""
    import cv2
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    codec = "avc1"
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        codec = "mp4v"
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write mp4 here (neither avc1 nor mp4v)")
    want, n = int(round(seconds * fps)), 0
    while n < want:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
    writer.release()
    cap.release()
    if n == 0:
        raise RuntimeError(f"no frames read from {src} at {start}s")
    return {"frames": n, "fps": round(fps, 3), "size": [width, height], "codec": codec, "bytes": dst.stat().st_size}


def slim_db(src: Path, dst: Path) -> dict:
    """A copy of db/clashbot.sqlite holding what the video path and the dashboard read: states (metadata, texts, FTS5 index),
    games, provenance, kg_edges / kg_nodes, cards, meta. Dropped: the `documents` tables (planner tools only) and the `record`
    column (the full state JSON, 60+ MB, read by nothing at run time; the column stays, set to NULL). The source is opened read-only."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    ro = sqlite3.connect(f"file:{Path(src).resolve()}?mode=ro", uri=True)
    out = sqlite3.connect(str(dst))
    try:
        ro.backup(out)
    finally:
        ro.close()
    try:
        names = {r[0] for r in out.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for table in ("documents_fts", "documents"):
            if table in names:
                out.execute(f"DROP TABLE {table}")
        cols = {r[1] for r in out.execute("PRAGMA table_info(states)")}
        if "record" in cols:
            out.execute("UPDATE states SET record = NULL")
        out.commit()
        out.execute("VACUUM")
        counts = {t: out.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("states", "games", "provenance", "kg_edges") if t in names}
    finally:
        out.close()
    return {"bytes": dst.stat().st_size, "rows": counts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clip")
    c.add_argument("src", type=Path)
    c.add_argument("dst", type=Path)
    c.add_argument("--start", type=float, default=60.0)
    c.add_argument("--seconds", type=float, default=90.0)
    s = sub.add_parser("slim-db")
    s.add_argument("src", type=Path)
    s.add_argument("dst", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "clip":
        info = cut_clip(args.src, args.dst, args.start, args.seconds)
    else:
        info = slim_db(args.src, args.dst)
    print(args.cmd, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
