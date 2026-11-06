"""Build dataset/v1/kg2/deck_cooc.json (MAC-AK K3): the known decks and their card co-occurrence counts.

  python tools/data/build_deck_cooc.py [--dataset DIR] [--kb DIR] [--out PATH] [--temperature T] [--count-power P]

Deck sources (identical decks are merged and counted):
  kb   - knowledge_base/decks/*.md card lists (the 68 pro decks);
  own  - games.jsonl `own_deck` where deck_status == "valid" (exactly 8 cards);
  opp  - games.jsonl `opponent_deck` when 7 or 8 cards are known.
Archetype of a merged deck: the KB deck file, else the game's own_archetype, else the opponent_archetype_guess.
The JSON is `OppDeckModel.to_dict()`: cards, pair_counts, card_counts, decks, deck_counts, deck_archetypes, params, meta.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import CARD_INDEX, DATASET_DIR, KB  # noqa: E402
from clashbot.opp_model import DECK_SIZE, DEFAULT_COUNT_POWER, DEFAULT_TEMPERATURE, NO_ARCHETYPE, OppDeckModel  # noqa: E402
from tools.data.kb_markdown import _CARD_LINK, bullets, parse_frontmatter, section  # noqa: E402

SOURCE_RANK = {"kb": 0, "own": 1, "opp": 2}        # archetype label priority
MIN_OPP_CARDS = 7


def parse_deck_file(path: Path) -> dict | None:
    """{deck_key, cards, archetype} of one knowledge_base/decks/*.md file (cards from the `## Cards` links)."""
    fm, body = parse_frontmatter(Path(path).read_text(encoding="utf-8"))
    cards = []
    for b in bullets(section(body, "Cards") or ""):
        m = _CARD_LINK.search(b)
        if m and m.group(1) not in cards:
            cards.append(m.group(1))
    if not cards:
        return None
    return {"deck_key": fm.get("deck_key") or Path(path).stem, "cards": cards, "archetype": fm.get("archetype_primary")}


def load_kb_decks(kb_dir: Path = KB) -> list[dict]:
    out = []
    for p in sorted(Path(kb_dir, "decks").glob("*.md")):
        d = parse_deck_file(p)
        if d:
            out.append(d)
    return out


def known_card_slugs(path: Path = CARD_INDEX) -> set[str] | None:
    try:
        return {c["slug"] for c in json.loads(Path(path).read_text(encoding="utf-8"))["cards"]}
    except (OSError, KeyError, ValueError):
        return None


def _label(value) -> str | None:
    return None if value in NO_ARCHETYPE else value


def collect_entries(dataset_dir: Path = DATASET_DIR, kb_dir: Path = KB, card_slugs: set[str] | None = None,
                    stats: Counter | None = None) -> list[dict]:
    """One entry per deck occurrence: {cards: sorted tuple, source: kb|own|opp, game_id, archetype}.

    `stats` (optional) counts what was dropped: own_not_valid, opp_too_small, unknown_card."""
    stats = stats if stats is not None else Counter()
    entries: list[dict] = []

    def add(cards, source, game_id, archetype):
        uniq = tuple(sorted(set(cards)))
        if card_slugs is not None and any(c not in card_slugs for c in uniq):
            stats["unknown_card"] += 1
            return
        entries.append({"cards": uniq, "source": source, "game_id": game_id, "archetype": _label(archetype)})

    for d in load_kb_decks(kb_dir):
        add(d["cards"], "kb", d["deck_key"], d["archetype"])
    with open(Path(dataset_dir) / "games.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            own = set(g.get("own_deck") or [])
            if g.get("deck_status") == "valid" and len(own) == DECK_SIZE:
                add(own, "own", g["game_id"], g.get("own_archetype"))
            else:
                stats["own_not_valid"] += 1
            opp = set(g.get("opponent_deck") or [])
            if MIN_OPP_CARDS <= len(opp) <= DECK_SIZE:
                add(opp, "opp", g["game_id"], g.get("opponent_archetype_guess"))
            else:
                stats["opp_too_small"] += 1
    return entries


def aggregate(entries: list[dict]) -> tuple[list[tuple[str, ...]], list[int], list[str | None]]:
    """Unique decks (sorted), how many times each occurs, and the best archetype label of each."""
    count: Counter = Counter()
    label: dict[tuple, tuple[int, str]] = {}
    for e in entries:
        count[e["cards"]] += 1
        a = e["archetype"]
        if a is not None:
            rank = SOURCE_RANK[e["source"]]
            if e["cards"] not in label or rank < label[e["cards"]][0]:
                label[e["cards"]] = (rank, a)
    decks = sorted(count)
    return decks, [count[d] for d in decks], [label[d][1] if d in label else None for d in decks]


def build_model(entries: list[dict], temperature: float = DEFAULT_TEMPERATURE,
                count_power: float = DEFAULT_COUNT_POWER) -> OppDeckModel:
    decks, counts, archetypes = aggregate(entries)
    return OppDeckModel(decks, counts, archetypes, temperature=temperature, count_power=count_power)


def summary(entries: list[dict], model: OppDeckModel, stats: Counter) -> dict:
    by_source = Counter(e["source"] for e in entries)
    unique_by_source = {s: len({e["cards"] for e in entries if e["source"] == s}) for s in SOURCE_RANK}
    sizes = Counter(len(d) for d in model.decks)
    return {"total_decks": len(entries), "unique_decks": len(model.decks), "cards": len(model.cards),
            "total_by_source": dict(by_source), "unique_by_source": unique_by_source,
            "unique_deck_sizes": dict(sorted(sizes.items())), "dropped": dict(stats),
            "decks_with_archetype": sum(a is not None for a in model.archetypes)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--kb", type=Path, default=KB)
    ap.add_argument("--out", type=Path, default=None, help="default <dataset>/kg2/deck_cooc.json")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--count-power", type=float, default=DEFAULT_COUNT_POWER)
    args = ap.parse_args(argv)
    out = args.out or args.dataset / "kg2" / "deck_cooc.json"

    stats: Counter = Counter()
    entries = collect_entries(args.dataset, args.kb, known_card_slugs(), stats)
    model = build_model(entries, args.temperature, args.count_power)
    info = summary(entries, model, stats)
    model.save(out, meta=info)
    print(json.dumps(info, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return info


if __name__ == "__main__":
    main()
