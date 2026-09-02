"""Regenerate context packs for existing runs from their states.jsonl (after
context-builder changes). Idempotent; does not touch states or calibration.

  python3 tools/rebuild_contexts.py [--runs runs/videos] [--only <video_id> ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cr_perception.context import build_context, render_context_md, split_matches  # noqa: E402
from cr_perception.decktracker import load_kb_decks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs" / "videos"))
    ap.add_argument("--videos", default=str(ROOT / "data" / "videos"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    idx = json.loads((ROOT / "knowledge_base" / "meta" / "card_index.json").read_text())
    names = {c["slug"]: c["name"] for c in idx["cards"]}
    kb_decks = load_kb_decks(ROOT / "knowledge_base")
    n = 0
    for states in sorted(Path(a.runs).glob("*/match_*/states.jsonl")):
        mdir = states.parent
        vid = mdir.parent.name
        if a.only and vid not in a.only:
            continue
        i = int(mdir.name.split("_")[1])
        vdir = Path(a.videos) / vid
        info = json.loads(next(vdir.glob("*.info.json")).read_text()) if list(vdir.glob("*.info.json")) else {}
        vtt = next(iter(sorted(vdir.glob("*.en.vtt"))), None)
        method = json.loads((mdir / "calib.json").read_text()).get("notes", {}).get("arena_method", "?") if (mdir / "calib.json").exists() else "?"
        for old in mdir.glob("game_*"):
            shutil.rmtree(old)
        segs = split_matches(states)
        written = []
        for k, (gi, s0, s1) in enumerate(segs):
            sub = f"{i}" if len(segs) == 1 else f"{i}.{k}"
            ctx = build_context(states, vtt, {"video_id": vid, "title": info.get("title", vid), "match_index": sub,
                                              "period": [s0, s1], "calibration_method": method,
                                              "url": f"https://www.youtube.com/watch?v={vid}"}, names, window=(s0, s1), kb_decks=kb_decks)
            cdir = mdir if len(segs) == 1 else mdir / f"game_{k}"
            cdir.mkdir(exist_ok=True)
            (cdir / "context.json").write_text(json.dumps(ctx, indent=1))
            (cdir / "context.md").write_text(render_context_md(ctx, names))
            written.append({"match": sub, "seconds": round(s1 - s0, 1), "events": len(ctx.get("events", [])),
                            "own_deck": ctx.get("own_deck_observed"), "opp_deck": ctx.get("opponent", {}).get("deck_known")})
            n += 1
        if len(segs) > 1:
            (mdir / "context.json").write_text(json.dumps({"video_id": vid, "match_index": i, "split_into": written}, indent=1))
        print(vid, mdir.name, "->", len(segs), "game(s)", [(w["match"], w["seconds"], w["events"]) for w in written])
    print(f"{n} contexts written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
