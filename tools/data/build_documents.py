"""Build retrieval documents (pro advice, wiki and match sections) into <dataset>/documents.jsonl (WP2).

  python3 tools/data/build_documents.py [--dataset DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR, KB  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from tools.data.build_clean_dataset import scrub  # noqa: E402
from tools.data.kb_markdown import (  # noqa: E402
    build_aliases, creator_blocks, mentions, parse_frontmatter, section,
)

TYPE_ORDER = ["card_insight", "deck_insight", "deck_why", "archetype_guide",
              "match_lessons", "match_card_usage", "match_opponent"]


def md_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.md"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    args = ap.parse_args(argv)
    ds: Path = args.dataset

    cards = load_cards()
    aliases = build_aliases(cards)
    deck_index = {d["deck_key"]: d for d in json.loads(
        (KB / "meta" / "deck_index.json").read_text(encoding="utf-8"))["decks"]}
    games = {}
    with (ds / "games.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                g = json.loads(line)
                games[g["game_id"]] = g

    docs: list[dict] = []

    def add(doc_id, dtype, title, raw, *, cards_override=None, deck_key=None, archetype=None,
            game_id=None, cards_exclude=None, cards_prefix=None):
        text = scrub(raw)
        if not text:
            return
        if cards_override is not None:
            card_list = list(cards_override)
        else:
            card_list = mentions(text, aliases, exclude=cards_exclude)
            if cards_prefix:
                card_list = list(cards_prefix) + card_list
        split = games[game_id]["split"] if game_id and game_id in games else None
        docs.append({"doc_id": doc_id, "type": dtype, "title": title, "text": text,
                     "cards": card_list, "deck_key": deck_key, "archetype": archetype,
                     "game_id": game_id, "split": split})

    for p in md_files(KB / "cards"):
        slug = p.stem
        if slug not in cards:
            continue
        raw = p.read_text(encoding="utf-8")
        for game_id, body in creator_blocks(raw):
            add(f"card_insight:{slug}:{game_id}", "card_insight", f"{cards[slug]['name']} — pro notes",
                body, cards_exclude=slug, cards_prefix=[slug], game_id=game_id)

    skipped_decks = 0
    for p in md_files(KB / "decks"):
        key = p.stem
        d = deck_index.get(key)
        if d is None:
            skipped_decks += 1
            continue
        raw = p.read_text(encoding="utf-8")
        _, body = parse_frontmatter(raw)
        name, arch = d["display_name"], d["archetype_primary"]
        for game_id, blk in creator_blocks(raw):
            add(f"deck_insight:{key}:{game_id}", "deck_insight", f"{name} — pro notes", blk,
                deck_key=key, archetype=arch, game_id=game_id)
        why = section(body, "Why this deck works")
        if why is not None:
            add(f"deck_why:{key}", "deck_why", f"Why {name} works", why,
                cards_override=d["cards"], deck_key=key, archetype=arch)

    for p in md_files(KB / "archetypes"):
        fm, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        gameplan = section(body, "Gameplan") or ""
        why = section(body, "Why it works") or ""
        add(f"archetype_guide:{fm.get('archetype')}", "archetype_guide", f"{fm.get('archetype_name')} gameplan",
            gameplan + "\n\n" + why, archetype=fm.get("archetype"))

    for p in md_files(KB / "matches"):
        gid = p.stem
        _, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        g = games.get(gid, {})
        for dtype, heading, title in (
            ("match_lessons", "Lessons stated by Ryley", f"Lessons — {gid}"),
            ("match_card_usage", "How Ryley uses his cards", f"Card usage — {gid}"),
            ("match_opponent", "Opponent", f"Opponent — {gid}"),
        ):
            sec = section(body, heading)
            if sec is None:
                continue
            add(f"{dtype}:{gid}", dtype, title, sec, deck_key=g.get("own_deck_key"),
                archetype=g.get("own_archetype"), game_id=gid)

    docs.sort(key=lambda d: (TYPE_ORDER.index(d["type"]), d["doc_id"]))
    with (ds / "documents.jsonl").open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("documents:", len(docs), dict(sorted(Counter(d["type"] for d in docs).items(),
                                               key=lambda kv: TYPE_ORDER.index(kv[0]))))
    print(f"deck files not in deck_index.json (skipped): {skipped_decks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
