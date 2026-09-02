"""Apply agent-produced creator insights to the knowledge base.

Agents never edit card/deck files directly (several run in parallel on the
same cards); they write `knowledge_base/matches/<key>.md` plus an
`insights.json` sidecar next to the match context. This tool merges the
sidecars into `knowledge_base/cards/<slug>.md` and `knowledge_base/decks/`
as marked blocks, replacing a block with the same marker on re-run.

  python3 tools/merge_insights.py [--only <key> ...] [--dry-run]

insights.json:
{
  "key": "<video_id>-m<n>",
  "match_file": "matches/<key>.md",
  "video_title": "...",
  "cards": {"<slug>": ["bullet text", ...]},                      # per card Ryley played/discussed
  "deck": {"deck_key": "<sorted-8>|null", "bullets": ["..."],     # block for the deck file
           "new_deck": {"display_name": "...", "archetype_primary": "...", "archetype_secondary": "...",
                        "rationale": "...", "why_it_works_md": "..."}   # only if no deck file exists
  }
}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base"
RUNS = ROOT / "runs" / "videos"
CREATOR = "ryleycr1"
CARD_HEADING = f"## Creator insights ({CREATOR})"
DECK_HEADING = f"## Creator matches ({CREATOR})"


def block(key: str, match_rel: str, title: str, bullets: list[str]) -> str:
    lines = [f"<!-- creator-insights:{CREATOR}:{key}:start -->"]
    lines.append(f"Match [{title}]({match_rel}):")
    for b in bullets:
        b = b.strip()
        lines.append(b if b.startswith("- ") else f"- {b}")
    lines.append(f"<!-- creator-insights:{CREATOR}:{key}:end -->")
    return "\n".join(lines)


def upsert(text: str, heading: str, key: str, blk: str) -> str:
    marker = re.compile(rf"<!-- creator-insights:{CREATOR}:{re.escape(key)}:start -->.*?<!-- creator-insights:{CREATOR}:{re.escape(key)}:end -->\n?", re.S)
    if marker.search(text):
        return marker.sub(blk + "\n", text, count=1)
    if heading in text:
        # append to the end of that section (before the next '## ' heading)
        i = text.index(heading) + len(heading)
        m = re.compile(r"^## ", re.M).search(text, i)
        j = m.start() if m else len(text)
        body = text[i:j].rstrip("\n")
        return text[:i] + body + "\n\n" + blk + "\n\n" + text[j:]
    section = f"{heading}\n\n{blk}\n\n"
    if "\n## Source" in text:
        k = text.index("\n## Source") + 1
        return text[:k] + section + text[k:]
    return text.rstrip("\n") + "\n\n" + section


def card_link(slug: str, names: dict) -> str:
    return f"[{names.get(slug, slug)}](../cards/{slug}.md)"


def new_deck_file(deck_key: str, ins: dict, names: dict, cards_idx: dict) -> str:
    nd = ins["deck"].get("new_deck") or {}
    slugs = ins["deck"].get("cards") or []
    if len(slugs) != 8:
        raise ValueError(f"{ins['key']}: new_deck needs deck.cards (8 slugs)")
    costs = [cards_idx.get(s, {}).get("elixir_cost") for s in slugs]
    known = [c for c in costs if isinstance(c, (int, float))]
    avg = round(sum(known) / len(known), 2) if len(known) == 8 else "n/a"
    fm = {
        "deck_key": deck_key, "display_name": nd.get("display_name", "Ryley deck"),
        "archetype_primary": nd.get("archetype_primary", "null"), "archetype_secondary": nd.get("archetype_secondary", '"none"'),
        "classification_source": "agent", "classification_rationale": nd.get("rationale", ""),
        "avg_elixir": avg, "rating": "n/a", "usage": "n/a", "wins": "n/a", "draws": "n/a", "losses": "n/a",
        "stat_unit": "n/a (deck observed in a creator video, no site statistics)",
        "source_url": f'"https://www.youtube.com/watch?v={ins["key"].split("-m")[0]}"', "creator": CREATOR,
    }
    out = ["---"] + [f"{k}: {v}" for k, v in fm.items()] + ["---", "", f"# {fm['display_name']}", "", "## Cards", ""]
    for s in slugs:
        c = cards_idx.get(s, {})
        out.append(f"- {card_link(s, names)} — {c.get('elixir_cost', '?')} elixir, {c.get('rarity', '?')} {c.get('card_type', '?')}, targets {c.get('targets', '?')}")
    out += ["", "## Classification", "", f"Primary archetype: [{fm['archetype_primary']}](../archetypes/{fm['archetype_primary']}.md). {fm['classification_rationale']}", "",
            "## Why this deck works", "", nd.get("why_it_works_md", "(see creator matches below)"), "",
            "## Source", "", f"Observed in ryleycr1's video {fm['source_url']}; classification by agent.", ""]
    return "\n".join(out)


def apply(ins: dict, names: dict, cards_idx: dict, dry: bool) -> list[str]:
    key, title = ins["key"], ins.get("video_title", ins["key"])
    match_rel = "../" + ins.get("match_file", f"matches/{key}.md")
    changed = []
    for slug, bullets in (ins.get("cards") or {}).items():
        p = KB / "cards" / f"{slug}.md"
        if not p.exists() or not bullets:
            changed.append(f"SKIP card {slug} (missing file or no bullets)")
            continue
        new = upsert(p.read_text(), CARD_HEADING, key, block(key, match_rel, title, bullets))
        if not dry:
            p.write_text(new)
        changed.append(f"card {slug}")
    d = ins.get("deck") or {}
    dk = d.get("deck_key")
    if dk and d.get("bullets"):
        p = KB / "decks" / f"{dk}.md"
        if not p.exists():
            if not dry:
                p.write_text(new_deck_file(dk, ins, names, cards_idx))
            changed.append(f"deck {dk} (created)")
        new = upsert(p.read_text() if p.exists() else new_deck_file(dk, ins, names, cards_idx), DECK_HEADING, key,
                     block(key, match_rel, title, d["bullets"]))
        if not dry:
            p.write_text(new)
        changed.append(f"deck {dk}")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    idx = json.loads((KB / "meta" / "card_index.json").read_text())
    names = {c["slug"]: c["name"] for c in idx["cards"]}
    cards_idx = {c["slug"]: c for c in idx["cards"]}
    n = 0
    for p in sorted(RUNS.glob("*/match_*/**/insights.json")):
        ins = json.loads(p.read_text())
        if a.only and ins["key"] not in a.only:
            continue
        for line in apply(ins, names, cards_idx, a.dry_run):
            print(ins["key"], line)
        n += 1
    print(f"{n} insight file(s) merged{' (dry run)' if a.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
