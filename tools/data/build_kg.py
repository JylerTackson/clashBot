"""Build the knowledge graph (nodes, edges) and card_stats from knowledge_base + dataset/v1 (WP2).

  python3 tools/data/build_kg.py [--dataset DIR]

Writes <dataset>/kg/nodes.jsonl, <dataset>/kg/edges.jsonl, <dataset>/card_stats.jsonl.
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
from clashbot.units import load_unit_map, resolve  # noqa: E402
from tools.data.build_clean_dataset import scrub  # noqa: E402
from tools.data.kb_markdown import (  # noqa: E402
    bullets, build_aliases, first_table, load_deck_registry, mentions, parse_frontmatter, section,
)

NOT_SPECIFIED = "Not specified on source page"
WINDOW_S = 4.0


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def md_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.md"))


def build_nodes(cards, deck_index, games, unit_classes) -> list[dict]:
    groups: list[list[dict]] = []

    g = []
    for slug, c in cards.items():
        g.append({"id": f"card:{slug}", "type": "card", "name": c["name"], "props": {
            "rarity": c["rarity"], "card_type": c["card_type"], "elixir_cost": c["elixir_cost"],
            "targets": c["targets"], "has_evolution": c["has_evolution"],
            "has_hero_variant": c["has_hero_variant"]}})
    groups.append(g)

    g = []
    for p in md_files(KB / "evolutions"):
        fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
        g.append({"id": f"evo:{fm.get('slug')}", "type": "evolution", "name": fm.get("name"), "props": {
            k: fm.get(k) for k in ("base_card_slug", "elixir_cost", "cycles", "card_type")}})
    groups.append(g)

    g = []
    for p in md_files(KB / "heroes"):
        fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
        g.append({"id": f"hero:{fm.get('slug')}", "type": "hero", "name": fm.get("name"), "props": {
            k: fm.get(k) for k in ("base_card_slug", "elixir_cost", "ability_name", "ability_cost",
                                   "total_elixir_cost", "card_type", "targets")}})
    groups.append(g)

    g = []
    for cls in sorted(set(load_unit_map()) | set(unit_classes)):
        r = resolve(cls)
        g.append({"id": f"unit:{cls}", "type": "unit", "name": cls, "props": {
            "kind": r["kind"], "primary": r["card"], "is_evo": r["is_evo"]}})
    groups.append(g)

    g = []
    for d in deck_index:
        g.append({"id": f"deck:{d['deck_key']}", "type": "deck", "name": d.get("display_name"), "props": {
            "archetype_primary": d.get("archetype_primary"),
            "archetype_secondary": d.get("archetype_secondary"),
            "avg_elixir": d.get("avg_elixir"), "cards": d.get("cards"),
            "evolutions": d.get("evolutions"), "heroes": d.get("heroes"),
            "in_index": d.get("in_index")}})
    groups.append(g)

    g = []
    for p in md_files(KB / "archetypes"):
        fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
        g.append({"id": f"archetype:{fm.get('archetype')}", "type": "archetype",
                  "name": fm.get("archetype_name"), "props": {
                      "deck_count": fm.get("deck_count"),
                      "secondary_deck_count": fm.get("secondary_deck_count")}})
    groups.append(g)

    g = []
    for gm in games:
        g.append({"id": f"game:{gm['game_id']}", "type": "game", "name": gm["game_id"], "props": {
            k: gm.get(k) for k in ("split", "deck_status", "result", "own_archetype",
                                   "opponent_archetype_guess")}})
    groups.append(g)

    out: list[dict] = []
    for g in groups:
        out.extend(sorted(g, key=lambda n: n["id"]))
    return out


def edge(src, rel, dst, props=None) -> dict:
    return {"src": src, "rel": rel, "dst": dst, "props": props or {}}


def sort_group(edges: list[dict]) -> list[dict]:
    return sorted(edges, key=lambda e: (e["src"], e["dst"], e["props"].get("source", "")))


def wiki_edges(cards, aliases) -> list[dict]:
    merged: dict[tuple, dict] = {}

    def add(src, rel, dst, source, bullet):
        key = (src, rel, dst, source)
        e = merged.get(key)
        if e is None:
            e = merged[key] = edge(src, rel, dst, {"source": source, "count": 0, "evidence": []})
        e["props"]["count"] += 1
        if len(e["props"]["evidence"]) < 3:
            e["props"]["evidence"].append(scrub(bullet)[:300])

    for p in md_files(KB / "cards"):
        slug = p.stem
        if slug not in cards:
            continue
        _, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        for heading in ("Strong against", "Weak against", "Notes / synergies"):
            sec = section(body, heading)
            if sec is None or sec.strip() == NOT_SPECIFIED:
                continue
            for b in bullets(sec):
                if b == NOT_SPECIFIED:
                    continue
                for x in mentions(b, aliases, exclude=slug):
                    if heading == "Strong against":
                        add(f"card:{slug}", "COUNTERS", f"card:{x}", "wiki_strong_against", b)
                    elif heading == "Weak against":
                        add(f"card:{x}", "COUNTERS", f"card:{slug}", "wiki_weak_against", b)
                    elif b.lower().startswith("substitutes"):
                        add(f"card:{slug}", "SUBSTITUTE", f"card:{x}", "wiki_notes", b)
                    else:
                        add(f"card:{slug}", "SYNERGY", f"card:{x}", "wiki_notes", b)
    return list(merged.values())


def answered_by_edges(states_path: Path) -> list[dict]:
    seen: set[tuple] = set()
    agg: dict[tuple[str, str], dict] = {}
    for s in read_jsonl(states_path):
        if s.get("split") != "train":
            continue
        action = s.get("action") or {}
        if action.get("type") != "play" or not action.get("card"):
            continue
        if "deck_invalid" in (s.get("flags") or []):
            continue
        delay = action.get("delay_s")
        play_time = s["t"] + (delay if delay is not None else 0.0)
        outcome = s.get("outcome") or {}
        for p in s["state"]["opponent"]["recent_plays"]:
            if not p.get("card"):
                continue
            if not (0.0 <= play_time - p["t"] <= WINDOW_S):
                continue
            key = (s["game_id"], round(play_time, 1), action["card"], round(p["t"], 1), p["card"])
            if key in seen:
                continue
            seen.add(key)
            a = agg.setdefault((p["card"], action["card"]), {
                "support": 0, "positive": 0, "neutral": 0, "negative": 0, "trades": []})
            a["support"] += 1
            v = outcome.get("verdict")
            if v in ("positive", "neutral", "negative"):
                a[v] += 1
            if outcome.get("elixir_trade") is not None:
                a["trades"].append(outcome["elixir_trade"])
    out = []
    for (opp, own), a in agg.items():
        if a["support"] < 3:
            continue
        trades = a["trades"]
        out.append(edge(f"card:{opp}", "ANSWERED_BY", f"card:{own}", {
            "support": a["support"], "positive": a["positive"], "neutral": a["neutral"],
            "negative": a["negative"],
            "mean_elixir_trade": round(sum(trades) / len(trades), 3) if trades else None,
            "confidence": round(a["positive"] / a["support"], 3),
            "source": "mined_train_states", "window_s": WINDOW_S}))
    return out


def build_edges(cards, deck_index, games, nodes, aliases, states_path) -> list[dict]:
    ids = {n["id"] for n in nodes}
    groups: list[list[dict]] = []

    g = []
    for n in nodes:
        if n["type"] == "evolution" and f"card:{n['props']['base_card_slug']}" in ids:
            g.append(edge(n["id"], "EVOLUTION_OF", f"card:{n['props']['base_card_slug']}"))
    groups.append(sort_group(g))

    g = []
    for n in nodes:
        if n["type"] == "hero" and f"card:{n['props']['base_card_slug']}" in ids:
            g.append(edge(n["id"], "HERO_OF", f"card:{n['props']['base_card_slug']}"))
    groups.append(sort_group(g))

    g = []
    for n in nodes:
        if n["type"] != "unit":
            continue
        r = resolve(n["name"])
        for slug in r["candidates"]:
            g.append(edge(n["id"], "SPAWNED_BY", f"card:{slug}", {"primary": slug == r["card"]}))
    groups.append(sort_group(g))

    g = []
    for d in deck_index:
        for slug in d["cards"]:
            if slug in cards:
                g.append(edge(f"card:{slug}", "IN_DECK", f"deck:{d['deck_key']}", {
                    "evolved": slug in d["evolutions"], "hero": slug in d["heroes"]}))
    groups.append(sort_group(g))

    g = []
    for d in deck_index:
        prim, sec = d.get("archetype_primary"), d.get("archetype_secondary")
        if f"archetype:{prim}" in ids:
            g.append(edge(f"deck:{d['deck_key']}", "HAS_ARCHETYPE", f"archetype:{prim}", {"role": "primary"}))
        if isinstance(sec, str) and sec and sec != "none" and sec != prim and f"archetype:{sec}" in ids:
            g.append(edge(f"deck:{d['deck_key']}", "HAS_ARCHETYPE", f"archetype:{sec}", {"role": "secondary"}))
    groups.append(sort_group(g))

    groups.append(sort_group(wiki_edges(cards, aliases)))

    g = []
    for gm in games:
        gid = f"game:{gm['game_id']}"
        for c in gm.get("own_deck") or []:
            g.append(edge(gid, "GAME_OWN_CARD", f"card:{c}"))
        for c in gm.get("opponent_deck") or []:
            g.append(edge(gid, "GAME_OPP_CARD", f"card:{c}"))
        k = gm.get("own_deck_key")
        if k and f"deck:{k}" in ids:
            g.append(edge(gid, "GAME_DECK", f"deck:{k}"))
    groups.append(sort_group(g))

    groups.append(sort_group(answered_by_edges(states_path)))

    out: list[dict] = []
    for g in groups:
        out.extend(g)
    return out


def build_card_stats(cards) -> list[dict]:
    out = []
    for p in md_files(KB / "cards"):
        slug = p.stem
        if slug not in cards:
            continue
        _, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        c = cards[slug]
        attrs = first_table(section(body, "Attributes") or "")
        out.append({
            "slug": slug, "name": c["name"], "card_type": c["card_type"], "rarity": c["rarity"],
            "elixir_cost": c["elixir_cost"], "targets": c["targets"],
            "attributes": attrs[0] if attrs else {},
            "stats_by_level": first_table(section(body, "Stats by level") or "") or [],
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    args = ap.parse_args(argv)
    ds: Path = args.dataset
    kg = ds / "kg"
    kg.mkdir(parents=True, exist_ok=True)

    cards = load_cards()
    deck_index = list(load_deck_registry().values())
    games = list(read_jsonl(ds / "games.jsonl"))
    states_path = ds / "states.jsonl"
    unit_classes = set()
    for s in read_jsonl(states_path):
        for u in s["state"].get("units") or []:
            if u.get("unit_class"):
                unit_classes.add(u["unit_class"])

    aliases = build_aliases(cards)
    nodes = build_nodes(cards, deck_index, games, unit_classes)
    edges = build_edges(cards, deck_index, games, nodes, aliases, states_path)
    write_jsonl(kg / "nodes.jsonl", nodes)
    write_jsonl(kg / "edges.jsonl", edges)
    stats = build_card_stats(cards)
    write_jsonl(ds / "card_stats.jsonl", stats)

    print("nodes:", dict(sorted(Counter(n["type"] for n in nodes).items())))
    print("edges:", dict(sorted(Counter(e["rel"] for e in edges).items())))
    print("COUNTERS by source:", dict(sorted(Counter(
        e["props"]["source"] for e in edges if e["rel"] == "COUNTERS").items())))
    print("ANSWERED_BY edges:", sum(1 for e in edges if e["rel"] == "ANSWERED_BY"))
    print("card_stats rows:", len(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
