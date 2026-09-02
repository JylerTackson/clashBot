"""Plan extraction-agent batches from the manifest.

Groups cards (with their evolution / hero sub-pages) into batches of
`--size` cards, skipping anything already `done`. Prints JSON:
  [{"batch": 1, "items": [{"slug", "kind", "target", "source"}...]}, ...]
"""
from __future__ import annotations

import argparse
import json
import sys

from common import CACHE_DIR, HERO_INDEX_JSON, INDEX_JSON, KB, load_json, load_manifest

SRC = CACHE_DIR / "src"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=9, help="cards per batch")
    args = ap.parse_args()
    index = load_json(INDEX_JSON, None)
    hindex = load_json(HERO_INDEX_JSON, None)
    items = load_manifest()["items"]
    heroes_by_base = {h["base_card"]: h for h in hindex["heroes"]}

    def entry(slug, kind):
        e = items.get(slug, {})
        if e.get("status") == "done" or e.get("stage") not in ("fetched",):
            return None
        sub = {"card": "cards", "evolution": "evolutions", "hero": "heroes"}[kind]
        return {"slug": slug, "kind": kind,
                "target": str(KB / sub / f"{slug}.md"),
                "source": str(SRC / f"{slug}.md")}

    groups = []
    for c in index["cards"]:
        g = [entry(c["slug"], "card")]
        if c["evolution"]:
            g.append(entry(c["evolution"]["slug"], "evolution"))
        h = heroes_by_base.get(c["name"])
        if h:
            g.append(entry(h["slug"], "hero"))
        g = [x for x in g if x]
        if g:
            groups.append(g)
    # heroes without a base card in the list (should not happen)
    for h in hindex["heroes"]:
        if not h.get("has_base_card"):
            e = entry(h["slug"], "hero")
            if e:
                groups.append([e])

    batches = []
    cur = []
    for g in groups:
        cur.append(g)
        if len(cur) >= args.size:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    out = [{"batch": i + 1, "items": [x for g in b for x in g]} for i, b in enumerate(batches)]
    json.dump(out, sys.stdout, indent=1)
    print()
    print(f"# {len(out)} batches, {sum(len(b['items']) for b in out)} items", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
