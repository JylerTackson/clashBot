"""Download the N most recent videos of a YouTube channel with yt-dlp, in
parallel, into data/videos/<video_id>/ with auto-generated English subtitles
and the info JSON. Skips ids that already have a finished .mp4. Writes
data/videos/manifest.json.

  python3 scripts/download_videos.py --channel https://www.youtube.com/@ryleycr1/videos --n 50 --workers 4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRA: list[str] = []
OUT = ROOT / "data" / "videos"


def list_channel(url: str, n: int) -> list[dict]:
    cmd = ["yt-dlp", *EXTRA, "--flat-playlist", "--playlist-end", str(n),
           "--print", "%(id)s\t%(duration)s\t%(title)s", url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    vids = []
    for line in out.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            vids.append({"id": parts[0], "duration": parts[1], "title": parts[2]})
    return vids


NON_CR = re.compile(r"marvel snap|plants on fire|brawl stars|clash of clans|squad busters|clash mini|fortnite|minecraft|"
                    r"valorant|roblox|among us|hay day|boom beach|mario|pokemon|chess|overwatch|apex", re.I)
CR = re.compile(r"clash royale|\bcr\b|crl|log bait|hog rider|royal hogs|x-?bow|mortar|golem|graveyard|miner|mega knight|"
                r"pekka|evolution|\bevo\b|elixir|arena|ultimate champion|trophies|\bdeck\b|\bcard\b|hero (knight|valkyrie|"
                r"ice wizard|berserker|giant|musketeer|wizard|goblins|mega minion|balloon|bowler|dark prince|tombstone|barbarian)", re.I)


def is_clash_royale(v: dict, info: dict | None = None) -> tuple[bool, str]:
    """Title says another game -> no. Otherwise title/tags/description must
    mention Clash Royale (or its vocabulary). The perception pipeline's
    readiness scan is the final check: no match periods -> not usable."""
    title = v.get("title", "")
    if NON_CR.search(title):
        return False, f"title names another game: {title}"
    if CR.search(title):
        return True, "title"
    if info:
        blob = " ".join([info.get("description", ""), " ".join(info.get("tags", []) or []), " ".join(info.get("categories", []) or [])])
        if re.search(r"clash royale", blob, re.I) and not NON_CR.search(blob[:400]):
            return True, "tags/description"
    return False, "no Clash Royale reference in title/tags/description"


def fetch_info(v: dict) -> dict | None:
    d = OUT / v["id"]
    d.mkdir(parents=True, exist_ok=True)
    existing = list(d.glob("*.info.json"))
    if existing:
        return json.loads(existing[0].read_text())
    cmd = ["yt-dlp", *EXTRA, "-q", "--skip-download", "--write-info-json", "--no-playlist",
           "-o", str(d / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={v['id']}"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    existing = list(d.glob("*.info.json"))
    return json.loads(existing[0].read_text()) if existing else None


def download(v: dict, height: int) -> dict:
    d = OUT / v["id"]
    d.mkdir(parents=True, exist_ok=True)
    done = [f for f in d.iterdir() if f.suffix in (".mp4", ".webm", ".mkv")]
    if done and not list(d.glob("*.part")):
        return {**v, "status": "cached", "file": done[0].name}
    cmd = ["yt-dlp", *EXTRA, "--no-progress", "-q",
           "-f", "bv*", "-S", f"res:{height},vcodec:h264,ext:mp4",   # short side = {height}; H.264 so OpenCV can decode (no AV1)
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
    ap.add_argument("--cookies", help="Netscape cookies.txt exported from a browser logged into YouTube "
                                     "(needed when YouTube answers 'Sign in to confirm you are not a bot')")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between requests (be gentle: 429s otherwise)")
    a = ap.parse_args()
    global EXTRA
    EXTRA = ["--js-runtimes", "node", "--sleep-requests", str(a.sleep)] + (["--cookies", a.cookies] if a.cookies else [])
    vids = list_channel(a.channel, a.n)
    print(f"{len(vids)} videos listed", flush=True)
    # metadata-first pass: keep only Clash Royale videos
    keep, skipped = [], []
    for v in vids:
        ok, why = is_clash_royale(v)
        if not ok and "another game" not in why:
            ok, why = is_clash_royale(v, fetch_info(v))
        (keep if ok else skipped).append({**v, "reason": why})
    for v in skipped:
        print(f"skip   {v['id']} {v['title'][:60]}  ({v['reason']})", flush=True)
    print(f"{len(keep)} Clash Royale videos to download, {len(skipped)} skipped", flush=True)
    vids = keep
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
                                                    "skipped_non_clash_royale": skipped,
                                                    "videos": results}, indent=1))
    n_ok = sum(r["status"] in ("ok", "cached") for r in results)
    print(f"done: {n_ok}/{len(vids)} downloaded, total {sum(r.get('bytes', 0) for r in results)//1_000_000} MB")
    return 0 if n_ok == len(vids) else 1


if __name__ == "__main__":
    sys.exit(main())
