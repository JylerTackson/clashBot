"""Download the N most recent videos of a YouTube channel with yt-dlp, in
parallel, into data/videos/<video_id>/ with auto-generated English subtitles
and the info JSON. Skips ids that already have a finished .mp4. Writes
data/videos/manifest.json.

  python3 scripts/download_videos.py --channel https://www.youtube.com/@ryleycr1/videos --n 50 --workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "videos"


def list_channel(url: str, n: int) -> list[dict]:
    cmd = ["yt-dlp", "--flat-playlist", "--playlist-end", str(n),
           "--print", "%(id)s\t%(duration)s\t%(title)s", url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    vids = []
    for line in out.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            vids.append({"id": parts[0], "duration": parts[1], "title": parts[2]})
    return vids


def download(v: dict, height: int) -> dict:
    d = OUT / v["id"]
    d.mkdir(parents=True, exist_ok=True)
    done = list(d.glob("*.mp4"))
    if done and not list(d.glob("*.part")):
        return {**v, "status": "cached", "file": done[0].name}
    cmd = ["yt-dlp", "--no-progress", "-q",
           "-f", f"bv*[height<={height}][ext=mp4]/bv*[height<={height}]/b[height<={height}]",
           "--write-auto-subs", "--write-subs", "--sub-langs", "en.*,en", "--sub-format", "vtt/srt",
           "--write-info-json", "--no-playlist", "-N", "4",
           "-o", str(d / "%(id)s.%(ext)s"),
           f"https://www.youtube.com/watch?v={v['id']}"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    files = sorted(x.name for x in d.iterdir())
    mp4 = [f for f in files if f.endswith(".mp4") or f.endswith(".webm")]
    subs = [f for f in files if f.endswith(".vtt") or f.endswith(".srt")]
    ok = p.returncode == 0 and bool(mp4)
    return {**v, "status": "ok" if ok else "failed", "file": mp4[0] if mp4 else None,
            "subs": subs, "seconds": round(time.time() - t0, 1),
            "error": (p.stderr[-400:] if not ok else None),
            "bytes": sum(x.stat().st_size for x in d.iterdir())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    vids = list_channel(a.channel, a.n)
    print(f"{len(vids)} videos listed", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(download, v, a.height): v for v in vids}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"[{len(results)}/{len(vids)}] {r['status']:6s} {r['id']} {r.get('bytes', 0)//1_000_000}MB subs={len(r.get('subs', []))} {r['title'][:60]}", flush=True)
            if r["status"] == "failed":
                print("   ", (r.get("error") or "")[-300:], flush=True)
    order = {v["id"]: i for i, v in enumerate(vids)}
    results.sort(key=lambda r: order[r["id"]])
    (OUT / "manifest.json").write_text(json.dumps({"channel": a.channel, "requested": a.n,
                                                    "videos": results}, indent=1))
    n_ok = sum(r["status"] in ("ok", "cached") for r in results)
    print(f"done: {n_ok}/{len(vids)} downloaded, total {sum(r.get('bytes', 0) for r in results)//1_000_000} MB")
    return 0 if n_ok == len(vids) else 1


if __name__ == "__main__":
    sys.exit(main())
