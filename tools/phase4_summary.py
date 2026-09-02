"""Print a Phase 4 coverage summary (markdown) from the knowledge base and the
download manifest: videos covered, matches, decks/cards with creator insights,
and downloaded videos with no match file."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base"


def main() -> None:
    dl = json.loads((ROOT / "data" / "videos" / "manifest.json").read_text()) if (ROOT / "data" / "videos" / "manifest.json").exists() else {"videos": []}
    titles = {v["id"]: v.get("title", "") for v in dl["videos"]}
    matches = sorted((KB / "matches").glob("*.md"))
    per_video: Counter = Counter()
    results: Counter = Counter()
    for p in matches:
        vid = re.sub(r"-m[\d.]+$", "", p.stem)
        per_video[vid] += 1
        m = re.search(r"^result:\s*(\w+)", p.read_text(), re.M)
        results[m.group(1) if m else "unknown"] += 1
    decks = [p for p in (KB / "decks").glob("*.md")]
    decks_creator = [p for p in decks if "Creator matches (ryleycr1)" in p.read_text()]
    decks_agent = [p for p in decks if "classification_source: agent" in p.read_text()]
    cards = [p for p in (KB / "cards").glob("*.md") if "Creator insights (ryleycr1)" in p.read_text()]
    covered = sorted(per_video)
    missing = [v for v in titles if v not in per_video]
    print(f"- Videos with match files: {len(covered)} of {len(titles)} downloaded; match files: {len(matches)}")
    print(f"- Results recorded: " + ", ".join(f"{k} {n}" for k, n in results.most_common()))
    print(f"- Deck files: {len(decks)} total, {len(decks_agent)} created from creator matches, {len(decks_creator)} carrying creator match blocks")
    print(f"- Card files with creator insights: {len(cards)} of {len(list((KB / 'cards').glob('*.md')))}")
    print("- Per video (id, title, games):")
    for v in covered:
        print(f"  - `{v}` {titles.get(v, '')[:70]}: {per_video[v]}")
    if missing:
        print("- Downloaded but no match file:")
        for v in missing:
            print(f"  - `{v}` {titles.get(v, '')[:70]}")


if __name__ == "__main__":
    main()
