"""Step 1: enumerate every card, evolution and hero from the wiki index pages.

Ground truth for the card list is the wiki's Card Overviews page (the page the
`Cards` article links to for "the individual cards"), which lists every
playable card grouped by type and rarity. Elixir cost is taken from the
comparison tables on the `Cards` page where available. Evolutions come from
the `Card Evolution` list page plus the evolution table on `Cards`; heroes from
the `Heroes` page. Every sub-page's existence is verified against the API.

Outputs:
  knowledge_base/meta/card_index.json
  knowledge_base/meta/hero_index.json
  knowledge_base/meta/card_index.md
  knowledge_base/meta/scrape_manifest.json  (pending entries added; existing
                                             statuses preserved)
"""
from __future__ import annotations

import html
import re
import sys

from common import (
    HERO_INDEX_JSON,
    INDEX_JSON,
    INDEX_MD,
    api,
    load_json,
    load_manifest,
    now_iso,
    page_exists,
    page_url,
    save_json,
    save_manifest,
    slugify,
    wikitext_to_md,
)


def get_wikitext(title: str) -> str:
    return api(action="parse", page=title, prop="wikitext")["parse"]["wikitext"]["*"]


def _iter_templates(text: str, name: str):
    """Yield (start, end, inner) for every top-level {{name|...}} in text."""
    i = 0
    tag = "{{" + name
    while True:
        i = text.find(tag, i)
        if i < 0:
            return
        depth = 0
        j = i
        while j < len(text):
            if text.startswith("{{", j):
                depth += 1
                j += 2
            elif text.startswith("}}", j):
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        yield i, j, text[i + 2 : j - 2]
        i = j


def parse_card_overviews(text: str) -> list[dict]:
    """Walk ==Type== / ===Rarity=== headings and collect {{CardOverview|Card=..}}."""
    cards = []
    # heading lines may contain html (<span id="..">) so match on the '=' fence only
    headings = [(m.start(), len(m.group(1)),
                 html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).replace("‌", "").strip())
                for m in re.finditer(r"^(==+)(.+?)\1\s*$", text, flags=re.M)]
    for start, _end, inner in _iter_templates(text, "CardOverview"):
        card_type = rarity = None
        for hs, lvl, title in headings:
            if hs > start:
                break
            if lvl == 2:
                card_type, rarity = title, None
            elif lvl == 3:
                rarity = title
        fields = {}
        for part in re.split(r"\|(?=\s*\w+\s*=)", inner[len("CardOverview"):]):
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.strip()] = v.strip()
        name = fields.get("Card", "").strip()
        if not name:
            continue
        singular = {"Troops": "Troop", "Spells": "Spell", "Buildings": "Building",
                    "Tower Troops": "Tower Troop"}[card_type]
        cards.append({
            "name": name,
            "slug": slugify(name),
            "title": name,
            "url": page_url(name),
            "card_type": singular,
            "rarity": rarity,
            "overview_description": wikitext_to_md(fields.get("Description", "")),
        })
    return cards


def parse_cards_page(text: str) -> tuple[dict[str, str], dict[str, dict]]:
    """Return (cost_by_title, evolution_rows_by_title) from the `Cards` page tables."""
    costs: dict[str, str] = {}
    evos: dict[str, dict] = {}
    # comparison tables: rows like |[[Archers]]||3||... ; column 2 is Cost for
    # troops/buildings/spells tables; spawner table has Type in col 2, cost col 3.
    section = None
    for line in text.split("\n"):
        m = re.search(r"\{\{StatisticsSubheader\|(.*?)\}\}", line)
        if m:
            section = m.group(1).strip()
        if line.startswith("|[[") and section:
            cells = [c.strip() for c in line[1:].split("||")]
            lm = re.match(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", cells[0])
            if not lm:
                continue
            target, label = lm.group(1), lm.group(2)
            if section == "Evolutions":
                base = target.split("/")[0]
                evos[base] = {"cost": cells[1], "cycles": cells[2],
                              "overall_cost": cells[3],
                              "stat_boosts": wikitext_to_md(cells[4]) if len(cells) > 4 else ""}
                continue
            if label and label != target:
                continue  # spawned sub-unit (e.g. Cursed Hog), not a card
            if section == "Spawners":
                cost = cells[2] if len(cells) > 2 else ""
            elif section == "Tower Troops":
                cost = ""
            else:
                cost = cells[1] if len(cells) > 1 else ""
            cost = re.sub(r"\s*\(.*?\)", "", cost).strip()
            if cost and cost.upper() != "N/A" and target not in costs:
                costs[target] = cost
    return costs, evos


def parse_card_evolution_page(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[\[([^\]|]+)/Evolution\|", text)))


def parse_heroes_page(text: str) -> list[dict]:
    i = text.find("==List of Heroes==")
    j = text.find("\n==", i + 5)
    table = text[i:j]
    heroes = []
    for row in table.split("|-")[1:]:
        cells = [c.strip() for c in re.split(r"\n\|", "\n" + row.strip()) if c.strip()]
        m = re.search(r"\[\[([^\]|]+)/Hero\|([^\]]+)\]\]", cells[0])
        if not m:
            continue
        base = m.group(1)
        ability = wikitext_to_md(cells[4]) if len(cells) > 4 else ""
        am = re.match(r"\*\*(.+?)\*\*\s*-?\s*(.*)", ability, flags=re.S)
        heroes.append({
            "base_card": base,
            "base_slug": slugify(base),
            "title": f"{base}/Hero",
            "slug": slugify(f"{base}/Hero"),
            "url": page_url(f"{base}/Hero"),
            "elixir_cost": cells[1] if len(cells) > 1 else "",
            "ability_cost": cells[2] if len(cells) > 2 else "",
            "total_elixir_cost": cells[3] if len(cells) > 3 else "",
            "ability_name": am.group(1).strip() if am else "",
            "ability_summary": " ".join(am.group(2).split()) if am else ability,
        })
    return heroes


def main() -> int:
    print("fetching index pages ...")
    overviews = get_wikitext("Card Overviews")
    cards_page = get_wikitext("Cards")
    evo_page = get_wikitext("Card Evolution")
    heroes_page = get_wikitext("Heroes")

    cards = parse_card_overviews(overviews)
    names = [c["name"] for c in cards]
    assert len(names) == len(set(names)), "duplicate card names in Card Overviews"
    costs, evo_rows = parse_cards_page(cards_page)
    evo_names = set(parse_card_evolution_page(evo_page)) | set(evo_rows)
    heroes = parse_heroes_page(heroes_page)
    hero_bases = {h["base_card"] for h in heroes}

    # verify sub-pages exist
    evo_titles = [f"{n}/Evolution" for n in sorted(evo_names)]
    hero_titles = [h["title"] for h in heroes]
    exists = page_exists(evo_titles + hero_titles + names)
    missing_cards = [n for n in names if not exists.get(n)]
    if missing_cards:
        print("WARNING: card pages missing:", missing_cards)
    evo_names = {n for n in evo_names if exists.get(f"{n}/Evolution")}
    unknown_evo = sorted(evo_names - set(names))
    if unknown_evo:
        print("WARNING: evolutions whose base card is not in the card list:", unknown_evo)

    for c in cards:
        c["elixir_cost"] = costs.get(c["name"], "")
        c["targets"] = "pending"  # filled from the card page attributes table
        c["has_evolution"] = c["name"] in evo_names
        c["has_hero_variant"] = c["name"] in hero_bases
        c["evolution"] = None
        if c["has_evolution"]:
            row = evo_rows.get(c["name"], {})
            c["evolution"] = {
                "title": f"{c['name']}/Evolution",
                "slug": f"{c['slug']}-evolution",
                "url": page_url(f"{c['name']}/Evolution"),
                "cycles": row.get("cycles", ""),
                "stat_boosts": row.get("stat_boosts", ""),
            }
    for h in heroes:
        h["has_base_card"] = h["base_card"] in names

    # preserve targets already resolved by a previous run
    prev = {c["slug"]: c for c in load_json(INDEX_JSON, {}).get("cards", [])}
    for c in cards:
        p = prev.get(c["slug"])
        if p and p.get("targets") not in (None, "", "pending"):
            c["targets"] = p["targets"]
            if not c["elixir_cost"]:
                c["elixir_cost"] = p.get("elixir_cost", "")

    save_json(INDEX_JSON, {"generated_at": now_iso(),
                           "source_pages": [page_url("Cards"), page_url("Card Overviews"),
                                            page_url("Card Evolution")],
                           "card_count": len(cards),
                           "evolution_count": sum(c["has_evolution"] for c in cards),
                           "cards": cards})
    save_json(HERO_INDEX_JSON, {"generated_at": now_iso(), "source_page": page_url("Heroes"),
                                "hero_count": len(heroes), "heroes": heroes})
    write_index_md(cards, heroes)

    # manifest: add pending entries, never downgrade existing ones
    m = load_manifest()
    items = m["items"]
    def ensure(slug, kind, title, url):
        if slug not in items:
            items[slug] = {"kind": kind, "title": title, "url": url,
                           "status": "pending", "stage": "enumerated",
                           "reason": None, "updated_at": now_iso()}
    for c in cards:
        ensure(c["slug"], "card", c["name"], c["url"])
        if c["evolution"]:
            ensure(c["evolution"]["slug"], "evolution", c["evolution"]["title"], c["evolution"]["url"])
    for h in heroes:
        ensure(h["slug"], "hero", h["title"], h["url"])
    m["counts"] = {"cards": len(cards), "evolutions": sum(c["has_evolution"] for c in cards),
                   "heroes": len(heroes)}
    save_manifest(m)
    print(f"cards={len(cards)} evolutions={m['counts']['evolutions']} heroes={len(heroes)}")
    return 0


def write_index_md(cards: list[dict], heroes: list[dict]) -> None:
    from common import MANIFEST
    manifest = load_json(MANIFEST, {"items": {}})["items"]
    lines = []
    lines.append("# Card index\n")
    lines.append(f"Generated {now_iso()} from "
                 f"[Cards]({page_url('Cards')}), [Card Overviews]({page_url('Card Overviews')}), "
                 f"[Card Evolution]({page_url('Card Evolution')}) and [Heroes]({page_url('Heroes')}).\n")
    n_evo = sum(c["has_evolution"] for c in cards)
    lines.append(f"**Cards: {len(cards)}** · **Evolutions: {n_evo}** · **Heroes: {len(heroes)}**\n")
    lines.append("`targets` values: `ground`, `air`, `ground_and_air`, `buildings`, `n/a` "
                 "(`pending` until the card page has been scraped). `status` mirrors "
                 "`scrape_manifest.json`.\n")
    lines.append("| # | Card | Slug | Rarity | Elixir | Type | Targets | Has evolution | Has hero | Status | Wiki |")
    lines.append("|---|------|------|--------|--------|------|---------|---------------|----------|--------|------|")
    for i, c in enumerate(cards, 1):
        st = manifest.get(c["slug"], {}).get("status", "pending")
        lines.append(
            f"| {i} | {c['name']} | `{c['slug']}` | {c['rarity']} | {c['elixir_cost'] or 'n/a'} | "
            f"{c['card_type']} | {c['targets']} | {'yes' if c['has_evolution'] else 'no'} | "
            f"{'yes' if c['has_hero_variant'] else 'no'} | {st} | [link]({c['url']}) |")
    lines.append("\n## Heroes\n")
    lines.append("| # | Hero | Slug | Base card | Elixir | Ability cost | Ability | Status | Wiki |")
    lines.append("|---|------|------|-----------|--------|--------------|---------|--------|------|")
    for i, h in enumerate(heroes, 1):
        st = manifest.get(h["slug"], {}).get("status", "pending")
        lines.append(
            f"| {i} | {h['title']} | `{h['slug']}` | `{h['base_slug']}` | {h['elixir_cost']} | "
            f"{h['ability_cost']} | {h['ability_name']} | {st} | [link]({h['url']}) |")
    INDEX_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
