"""List match contexts that are ready for an agent and track which ones have
been handed off / completed (runs/videos/agent_manifest.json).

  python3 tools/dispatch_matches.py list            # ready, not yet assigned
  python3 tools/dispatch_matches.py assign <video_id>-m<n> ... --batch <name>
  python3 tools/dispatch_matches.py done <video_id>-m<n> ...
  python3 tools/dispatch_matches.py status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "videos"
AM = RUNS / "agent_manifest.json"


def load():
    return json.loads(AM.read_text()) if AM.exists() else {"matches": {}}


def save(m):
    AM.write_text(json.dumps(m, indent=1))


def ready() -> list[dict]:
    out = []
    for ctx in sorted(list(RUNS.glob("*/match_*/context.json")) + list(RUNS.glob("*/match_*/game_*/context.json"))):
        c = json.loads(ctx.read_text())
        if c.get("empty") or c.get("split_into"):
            continue
        key = f"{c['video_id']}-m{c['match_index']}"
        q = c.get("quality", {})
        out.append({"key": key, "context_md": str(ctx.with_suffix(".md")), "context_json": str(ctx),
                    "title": c.get("title"), "seconds": q.get("readable_seconds"), "events": q.get("events_total"),
                    "transcript_cues": len(c.get("transcript", [])), "own_deck": c.get("own_deck_observed")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "assign", "done", "status"])
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--batch", default="")
    a = ap.parse_args()
    m = load()
    if a.cmd == "list":
        for r in ready():
            st = m["matches"].get(r["key"], {}).get("status", "ready")
            if st == "ready":
                print(json.dumps(r))
    elif a.cmd == "assign":
        for k in a.keys:
            m["matches"][k] = {"status": "assigned", "batch": a.batch}
        save(m)
    elif a.cmd == "done":
        for k in a.keys:
            m["matches"].setdefault(k, {})["status"] = "done"
        save(m)
    else:
        from collections import Counter
        allr = ready()
        c = Counter(m["matches"].get(r["key"], {}).get("status", "ready") for r in allr)
        print(json.dumps({"ready_total": len(allr), **c}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
