#!/usr/bin/env python3
"""Artefact builders of the Docker submission (MAC-AV): called by scripts/make_docker_submission.sh.

    python scripts/docker_artifacts.py clip   SRC.mp4 DST.mp4 [--start 60] [--seconds 90]
    python scripts/docker_artifacts.py slim-db SRC.sqlite DST.sqlite
    python scripts/docker_artifacts.py slim-pool SRC_DB_DIR DST_DIR [--exclude-video Z8-4VhLjrGU]

Each one only reads its source (the sqlite file is opened read-only) and writes new files. No network, no model load, no
input sent anywhere. (The d1 weights are not packaged: the image downloads them from Hugging Face at build time.)
"""
from __future__ import annotations

import argparse
import json
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
        out.execute("PRAGMA journal_mode = DELETE")        # no -wal / -shm files next to the shipped file
        out.execute("VACUUM")
        counts = {t: out.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("states", "games", "provenance", "kg_edges") if t in names}
    finally:
        out.close()
    return {"bytes": dst.stat().st_size, "rows": counts}


def slim_pool(src_dir: Path, dst_dir: Path, exclude_video: str | None = None) -> dict:
    """The retrieval pool of the image: `clashbot.sqlite` (slimmed as `slim_db`), `state_features.npy` and
    `state_features.meta.json` in `dst_dir`, from the `db/` folder `src_dir`. With `exclude_video`, every state of that source
    video (all its games) leaves the pool, so the demo clip never retrieves the pro's own play from the same match. The three
    files stay consistent the way `StateIndex` expects: the surviving states are renumbered 0..n-1 in their old order
    (`states.vec_row`), the FTS5 index is rebuilt with rowid = vec_row + 1, and the feature matrix keeps exactly those rows.
    The games and provenance rows of the excluded video go too. Nothing under `src_dir` is modified."""
    import numpy as np
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    db = dst_dir / "clashbot.sqlite"
    info = slim_db(src_dir / "clashbot.sqlite", db)
    features = np.load(src_dir / "state_features.npy", mmap_mode="r")
    meta = json.loads((src_dir / "state_features.meta.json").read_text())
    conn = sqlite3.connect(str(db))
    try:
        old_rows = [r[0] for r in conn.execute("SELECT vec_row FROM states ORDER BY vec_row")]
        if old_rows != list(range(len(old_rows))) or features.shape[0] != len(old_rows):
            raise ValueError(f"the source pool is not aligned: {len(old_rows)} states, {features.shape[0]} feature rows")
        before = len(old_rows)
        if exclude_video:
            games = {r[0] for r in conn.execute("SELECT game_id FROM provenance WHERE video_id = ?", (exclude_video,))}
            games |= {r[0] for r in conn.execute("SELECT game_id FROM games WHERE game_id LIKE ?", (exclude_video + "-%",))}
            games |= {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM states WHERE game_id LIKE ?", (exclude_video + "-%",))}
            marks = ",".join("?" * len(games))
            if games:
                conn.execute(f"DELETE FROM states WHERE game_id IN ({marks})", sorted(games))
                conn.execute(f"DELETE FROM games WHERE game_id IN ({marks})", sorted(games))
                conn.execute(f"DELETE FROM provenance WHERE game_id IN ({marks})", sorted(games))
            conn.execute("DELETE FROM provenance WHERE video_id = ?", (exclude_video,))
        kept = [r[0] for r in conn.execute("SELECT vec_row FROM states ORDER BY vec_row")]
        # renumber in ascending order: every new number is <= the old one and already free, so the primary key never collides
        conn.executemany("UPDATE states SET vec_row = ? WHERE vec_row = ?", [(new, old) for new, old in enumerate(kept) if new != old])
        conn.execute("DELETE FROM states_fts")
        conn.execute("INSERT INTO states_fts(rowid, id, context_text) SELECT vec_row + 1, id, context_text FROM states ORDER BY vec_row")
        conn.execute("INSERT INTO states_fts(states_fts) VALUES ('optimize')")
        conn.execute("UPDATE meta SET value = ? WHERE key = 'n_states'", (str(len(kept)),))
        conn.commit()
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("VACUUM")
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("states", "states_fts", "games", "provenance")}
    finally:
        conn.close()
    np.save(dst_dir / "state_features.npy", np.ascontiguousarray(features[kept], dtype=np.float32))
    meta["n"] = len(kept)
    if exclude_video:
        meta["excluded_video"] = exclude_video
    (dst_dir / "state_features.meta.json").write_text(json.dumps(meta, indent=2))
    return {"before": before, "after": len(kept), "removed": before - len(kept), "rows": counts,
            "db_bytes": db.stat().st_size, "features_bytes": (dst_dir / "state_features.npy").stat().st_size, "slim": info}


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
    q = sub.add_parser("slim-pool")
    q.add_argument("src", type=Path, help="the db/ folder: clashbot.sqlite, state_features.npy, state_features.meta.json")
    q.add_argument("dst", type=Path)
    q.add_argument("--exclude-video", default=None, help="drop every state of this source video id from the pool")
    args = ap.parse_args(argv)
    if args.cmd == "clip":
        info = cut_clip(args.src, args.dst, args.start, args.seconds)
    elif args.cmd == "slim-pool":
        info = slim_pool(args.src, args.dst, args.exclude_video)
    else:
        info = slim_db(args.src, args.dst)
    print(args.cmd, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
