"""Step 2-4 (deterministic half): fetch every card / evolution / hero page,
cache it, download the header image and write a skeleton markdown file.

The skeleton contains everything that can be extracted mechanically and
reliably: frontmatter, the in-game description, the lead paragraph, every
statistics table (rendered by the wiki, so the per-level formulas are already
evaluated), ability / modifier sections and cross-links. Sections that need
judgement (Strong against / Weak against / Notes & synergies) are left as
`<!-- AGENT:FILL -->` placeholders for the extraction agents (see
`scripts/kb/agent_batches.py`).

Re-runnable: items whose manifest stage is already `fetched` or `done` are
skipped unless --force is passed. A per-item cache of the API responses is
kept in $KB_CACHE_DIR (default: <repo>/.kb_cache, git-ignored).
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup

from common import (
    CACHE_DIR,
    CARD_IMG_DIR,
    CARDS_DIR,
    EVO_DIR,
    HERO_DIR,
    HERO_IMG_DIR,
    HERO_INDEX_JSON,
    INDEX_JSON,
    KB,
    SESSION,
    api,
    load_json,
    load_manifest,
    now_iso,
    save_json,
    save_manifest,
    split_sections,
    wikitext_to_md,
)

EVO_IMG_DIR = EVO_DIR / "images"
SRC_DIR = CACHE_DIR / "src"

PLACEHOLDER = "<!-- AGENT:FILL -->"

SKIP_SECTIONS = {"history", "gallery", "trivia", "in other languages", "sound effects",
                 "see also", "references", "navigation", "card mastery", "strategy",
                 "statistics", "mastery"}

IMAGE_BLACKLIST = re.compile(r"^(Elixir|Cycles|Hero_Shard|Wild_Shard|MergeTacticStar|.*stoplight|.*Modifier|.*_Preview\.gif|Cards|Boost|Damage.*|Hitpoint.*)\b", re.I)


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def fetch_page(title: str) -> dict:
    cache = CACHE_DIR / "pages" / (title.replace("/", "__") + ".json")
    if cache.exists():
        return load_json(cache, None)
    r = api(action="parse", page=title, prop="wikitext|text|images|displaytitle")
    if "error" in r:
        raise RuntimeError(r["error"].get("info", "api error"))
    data = {"title": r["parse"]["title"], "wikitext": r["parse"]["wikitext"]["*"],
            "html": r["parse"]["text"]["*"], "images": r["parse"]["images"],
            "displaytitle": re.sub(r"<[^>]+>", "", r["parse"].get("displaytitle", "")).strip(),
            "fetched_at": now_iso()}
    save_json(cache, data)
    return data


def pick_header_image(page: dict, kind: str) -> str | None:
    """The infobox image is the first image on the page and follows the wiki's
    naming convention (<Name>Card.png / <Name>CardEvolution.png /
    Hero<Name>Card.png). Fall back to the first non-icon image."""
    imgs = page["images"]
    # infobox image as it appears in the rendered html (most reliable)
    soup = BeautifulSoup(page["html"], "lxml")
    box = soup.select_one(".portable-infobox img, .infobox img, aside img")
    if box is not None:
        name = box.get("data-image-name") or box.get("data-image-key")
        if name:
            return name.replace(" ", "_")
    pats = {"card": r"Card\.png$", "evolution": r"CardEvolution\.png$|Evolution.*\.png$",
            "hero": r"^Hero.*Card\.png$|Hero.*\.png$"}
    for im in imgs:
        if re.search(pats[kind], im, re.I) and not IMAGE_BLACKLIST.search(im):
            return im
    for im in imgs:
        if im.lower().endswith(".png") and not IMAGE_BLACKLIST.search(im):
            return im
    return None


def download_image(file_name: str, dest: Path) -> dict:
    r = api(action="query", titles=f"File:{file_name}", prop="imageinfo",
            iiprop="url|size|mime")
    pages = r["query"]["pages"]
    info = next(iter(pages.values()))
    if "imageinfo" not in info:
        raise RuntimeError(f"no imageinfo for {file_name}")
    ii = info["imageinfo"][0]
    url = ii["url"]
    url += ("&" if "?" in url else "?") + "format=png"
    resp = SESSION.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.content
    if not data.startswith(b"\x89PNG"):
        raise RuntimeError(f"downloaded image for {file_name} is not PNG ({resp.headers.get('content-type')})")
    if len(data) < 1024:
        raise RuntimeError(f"downloaded image for {file_name} is suspiciously small ({len(data)} bytes)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return {"file": file_name, "source_url": ii["url"], "bytes": len(data),
            "width": ii.get("width"), "height": ii.get("height")}


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------

def infobox_params(wikitext: str) -> dict:
    m = re.search(r"\{\{(?:Card|Evolved Card|Heroic Card) Infobox(.*?)\}\}", wikitext, re.S)
    if not m:
        return {}
    out = {}
    for part in m.group(1).split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def quote_text(wikitext: str) -> str:
    m = re.search(r"\{\{Quote\|(.*?)\}\}", wikitext, re.S)
    return " ".join(m.group(1).split()) if m else ""


def html_tables(html: str) -> list[tuple[str, str]]:
    """Return [(caption, markdown_table)] for the statistics tables."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for t in soup.select("table.wikitable"):
        hdr = t.find_previous(["h2", "h3", "h4"])
        caption = hdr.get_text(" ", strip=True) if hdr else "Table"
        caption = re.sub(r"\[.*?\]", "", caption).strip()
        if re.search(r"mastery|gallery|trivia|history|other languages", caption, re.I):
            continue
        # expand rowspan/colspan into a rectangular grid so columns line up
        grid: list[list[str]] = []
        pending: dict[tuple[int, int], str] = {}  # (row, col) -> text carried by rowspan
        r_i = 0
        for tr in t.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            row: list[str] = []
            c_i = 0
            for c in cells:
                while (r_i, c_i) in pending:
                    row.append(pending.pop((r_i, c_i)))
                    c_i += 1
                for br in c.find_all("br"):
                    br.replace_with(" ")
                txt = c.get_text(" ", strip=True)
                txt = re.sub(r"\s+", " ", txt).replace("|", "/")
                if not txt:
                    alts = [i.get("alt") or i.get("data-image-name") for i in c.find_all("img")]
                    txt = " ".join(a for a in alts if a) or ""
                try:
                    rs = int(c.get("rowspan", 1) or 1)
                    cs = int(c.get("colspan", 1) or 1)
                except ValueError:
                    rs = cs = 1
                for k in range(cs):
                    row.append(txt if k == 0 else "")
                    for dr in range(1, rs):
                        pending[(r_i + dr, c_i)] = txt if k == 0 else ""
                    c_i += 1
            while (r_i, c_i) in pending:
                row.append(pending.pop((r_i, c_i)))
                c_i += 1
            grid.append(row)
            r_i += 1
        rows = grid
        if not rows:
            continue
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
        for r in rows[1:]:
            md.append("| " + " | ".join(r) + " |")
        out.append((caption, "\n".join(md)))
    return out


def normalize_targets(raw: str) -> str:
    s = raw.lower().replace("&", "and")
    s = re.sub(r"\s+", " ", s).strip()
    if not s or s in ("n/a", "none", "-"):
        return "n/a"
    has_air = "air" in s
    has_ground = "ground" in s
    if "building" in s and not has_air and not has_ground:
        return "buildings"
    if has_air and has_ground:
        return "ground_and_air"
    if has_air:
        return "air"
    if has_ground:
        return "ground"
    return s.replace(" ", "_")


def attribute_value(tables: list[tuple[str, str]], column_regex: str) -> str:
    """Look up a column in the first (primary) attributes table."""
    for caption, md in tables:
        lines = md.split("\n")
        if len(lines) < 3:
            continue
        head = [h.strip() for h in lines[0].strip("|").split("|")]
        idx = next((i for i, h in enumerate(head) if re.search(column_regex, h, re.I)), None)
        if idx is None:
            continue
        cells = [c.strip() for c in lines[2].strip("|").split("|")]
        if idx < len(cells) and cells[idx]:
            return cells[idx]
    return ""


def section_md(wikitext: str, heading_regex: str) -> list[tuple[str, str]]:
    out = []
    for h, body in split_sections(wikitext):
        if h and re.search(heading_regex, h, re.I):
            out.append((wikitext_to_md(h), wikitext_to_md(body)))
    return out


def lead_paragraph(wikitext: str) -> str:
    lead = strip_stats_block(split_sections(wikitext)[0][1])
    lead = re.sub(r"\{\{Quote\|.*?\}\}", "", lead, flags=re.S)
    md = wikitext_to_md(lead)
    paras = [p.strip() for p in md.split("\n\n") if p.strip() and not p.strip().startswith(">")]
    return "\n\n".join(paras)


def strategy_md(wikitext: str) -> str:
    parts = section_md(wikitext, r"^strategy")
    return "\n\n".join(body for _h, body in parts)


def other_sections_md(wikitext: str) -> list[tuple[str, str]]:
    """Ability:, Modifiers, and any other substantive top-level section."""
    out = []
    for h, body in split_sections(wikitext):
        if not h:
            continue
        key = wikitext_to_md(h).strip().lower()
        if any(key.startswith(s) for s in SKIP_SECTIONS) or key == "":
            continue
        # skip subsections of skipped sections handled by wikitext order: we only
        # take h2-level content by re-splitting on == only
        out.append((wikitext_to_md(h), wikitext_to_md(body)))
    return out


def norm_heading(h: str) -> str:
    """'\'\'\'Strategy\'\'\'' -> 'strategy'; 'Strategies' -> 'strategies'."""
    return re.sub(r"[*'\s]+", " ", wikitext_to_md(h)).strip().lower()


def strip_stats_block(body: str) -> str:
    """The statistics div is appended to whatever section precedes it (usually
    Strategy) without its own heading; cut it off so it never leaks as raw
    wikitext. The tables themselves are taken from the rendered html."""
    return re.split(r'<center>\s*<div[^>]*unit-statistics|<div[^>]*id="unit-statistics"|\{\{Statistics\b',
                    body, maxsplit=1)[0]


def top_level_sections(wikitext: str) -> list[tuple[str, str]]:
    """Split on level-2 headings only (keeps ===sub=== inside their parent)."""
    parts = re.split(r"^==([^=].*?)==\s*$", wikitext, flags=re.M)
    secs = [("", parts[0])]
    for i in range(1, len(parts), 2):
        secs.append((parts[i].strip(), parts[i + 1]))
    return secs


def substantive_sections(wikitext: str) -> list[tuple[str, str]]:
    out = []
    for h, body in top_level_sections(wikitext):
        if not h:
            continue
        key = norm_heading(h)
        if any(key.startswith(s) for s in SKIP_SECTIONS) or key.startswith("strateg"):
            continue
        body_md = wikitext_to_md(strip_stats_block(body))
        if body_md.strip():
            out.append((wikitext_to_md(h).strip("*' "), body_md))
    return out


def strategy_block(wikitext: str) -> str:
    for h, body in top_level_sections(wikitext):
        if norm_heading(h).startswith("strateg"):
            return wikitext_to_md(strip_stats_block(body))
    return ""


# --------------------------------------------------------------------------
# skeleton writers
# --------------------------------------------------------------------------

PROSE_HEADINGS = ("## Strong against", "## Weak against", "## Notes / synergies",
                  "## What changes mechanically", "## Notes")


def existing_prose(path: Path) -> dict[str, str]:
    """Agent-written section bodies from a previously generated file, so a
    mechanical regeneration does not throw away the extraction work."""
    if not path.exists():
        return {}
    text = path.read_text()
    out = {}
    for m in re.finditer(r"^(## [^\n]+)\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S):
        h, body = m.group(1).strip(), m.group(2).strip()
        if h in PROSE_HEADINGS and body and PLACEHOLDER not in body:
            out[h] = body
    return out


def prose(kept: dict[str, str], heading: str) -> str:
    return kept.get(heading, PLACEHOLDER)


def fm(fields: list[tuple[str, object]]) -> str:
    lines = ["---"]
    for k, v in fields:
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
            continue
        v = "" if v is None else str(v)
        if v == "" :
            v = '""'
        elif re.search(r"[:#\[\]{}&*!|>'\"%@`]|^\s|\s$", v) or v.lower() in ("yes", "no", "null", "true", "false", "n/a", "on", "off"):
            v = '"' + v.replace('"', '\\"') + '"'
        lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def split_tables(tables):
    """Per-level tables (first column 'Level') vs. everything else (attributes).
    Captions are relabelled: the wiki renders a bare {{Statistics}} header as
    'Statistics' for both kinds."""
    level, attr = [], []
    for cap, md in tables:
        is_level = md.startswith("| Level")
        if cap.strip().lower() == "statistics":
            cap = "Card Statistics" if is_level else "Attributes"
        elif is_level and "statistic" not in cap.lower():
            cap = "Card Statistics (per level)"
        (level if is_level else attr).append((cap, md))
    return level, attr


def tables_md(tables: list[tuple[str, str]]) -> str:
    if not tables:
        return "Not specified on source page"
    chunks = []
    for cap, md in tables:
        chunks.append(f"**{cap}**\n\n{md}")
    return "\n\n".join(chunks)


def write_card_skeleton(card: dict, page: dict, image_rel: str, tables, heroes_by_base: dict) -> dict:
    w = page["wikitext"]
    ib = infobox_params(w)
    cost = card.get("elixir_cost") or ib.get("cost", "") or attribute_value(tables, r"^Cost")
    cost = re.sub(r"\s*\(.*?\)", "", cost).strip() or "n/a"
    targets_raw = attribute_value(tables, r"^Targets?$|^Target\b")
    targets = normalize_targets(targets_raw) if targets_raw else ("n/a" if card["card_type"] in ("Spell", "Building") else "")
    if not targets:
        targets = "n/a"
    ability_cost = ib.get("abilitycost", "")
    level_tables, attr_tables = split_tables(tables)

    links = []
    if card["has_evolution"]:
        links.append(f"[Evolution: {card['evolution']['slug']}](../evolutions/{card['evolution']['slug']}.md)")
    hero = heroes_by_base.get(card["name"])
    if hero:
        links.append(f"[Hero variant: {hero['slug']}](../heroes/{hero['slug']}.md)")

    fields = [
        ("name", card["name"]), ("slug", card["slug"]), ("rarity", card["rarity"]),
        ("elixir_cost", cost), ("card_type", card["card_type"]), ("targets", targets),
        ("has_evolution", card["has_evolution"]), ("has_hero_variant", bool(hero)),
        ("source_url", card["url"]), ("image_path", image_rel), ("scraped_at", now_iso()),
    ]
    if ability_cost:
        fields.insert(4, ("ability_cost", ability_cost))
    if card["has_evolution"]:
        fields.append(("evolution_file", f"evolutions/{card['evolution']['slug']}.md"))
    if hero:
        fields.append(("hero_file", f"heroes/{hero['slug']}.md"))
    fields.append(("arena", ib.get("arena", "")))
    fields.append(("release_date", ib.get("releasedate", "")))

    body = [fm(fields), "", f"# {card['name']}", ""]
    q = quote_text(w)
    if q:
        body += [f"> *In-game description:* \"{q}\"", ""]
    if links:
        body += ["**Related:** " + " · ".join(links), ""]
    body += ["## Overview", "", lead_paragraph(w) or "Not specified on source page", ""]
    body += ["<!-- Card Overviews summary (wiki) -->", card.get("overview_description") or "", ""]
    body += ["## Attributes", "", tables_md(attr_tables), ""]
    body += ["## Stats by level", "", tables_md(level_tables), ""]
    extra = substantive_sections(w)
    if extra:
        body += ["## Abilities and special mechanics", ""]
        for h, md in extra:
            body += [f"### {h}", "", md, ""]
    else:
        body += ["## Abilities and special mechanics", "", "Not specified on source page", ""]
    kept = existing_prose(CARDS_DIR / f"{card['slug']}.md")
    body += ["## Strong against", "", prose(kept, "## Strong against"), ""]
    body += ["## Weak against", "", prose(kept, "## Weak against"), ""]
    body += ["## Notes / synergies", "", prose(kept, "## Notes / synergies"), ""]
    if card["has_evolution"]:
        ev = card["evolution"]
        body += ["## Evolution", "",
                 f"This card has an evolution: see [{ev['slug']}.md](../evolutions/{ev['slug']}.md). "
                 f"Cycles to evolve: {ev.get('cycles') or 'see evolution file'}. "
                 f"Stat boosts: {ev.get('stat_boosts') or 'see evolution file'}.", ""]
    if hero:
        body += ["## Hero variant", "",
                 f"This card has a Hero form ({hero['ability_name']}, ability cost {hero['ability_cost']} Elixir): "
                 f"see [{hero['slug']}.md](../heroes/{hero['slug']}.md).", ""]
    body += ["## Source", "", f"- {card['url']} (scraped {now_iso()})", ""]
    (CARDS_DIR / f"{card['slug']}.md").write_text("\n".join(body))
    # source pack for the agent
    src = [f"# SOURCE PACK: {card['name']} ({card['url']})", "",
           "## Lead", lead_paragraph(w), "", "## Card Overviews summary", card.get("overview_description") or "", "",
           "## Strategy (wiki)", strategy_block(w) or "(none)", ""]
    for h, md in extra:
        src += [f"## {h}", md, ""]
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    (SRC_DIR / f"{card['slug']}.md").write_text("\n".join(src))
    return {"elixir_cost": cost, "targets": targets, "targets_raw": targets_raw}


def write_evo_skeleton(card: dict, page: dict, image_rel: str, tables) -> None:
    ev = card["evolution"]
    w = page["wikitext"]
    ib = infobox_params(w)
    level_tables, attr_tables = split_tables(tables)
    fields = [
        ("base_card", card["name"]), ("base_card_slug", card["slug"]),
        ("base_card_file", f"cards/{card['slug']}.md"),
        ("name", page.get("displaytitle") or f"Evolved {card['name']}"), ("slug", ev["slug"]),
        ("rarity", ib.get("rarity", card["rarity"])), ("elixir_cost", ib.get("cost", card.get("elixir_cost", ""))),
        ("cycles", ib.get("cyclecost", ev.get("cycles", ""))),
        ("card_type", ib.get("type", card["card_type"])),
        ("source_url", ev["url"]), ("image_path", image_rel), ("scraped_at", now_iso()),
        ("release_date", ib.get("releasedate", "")),
    ]
    body = [fm(fields), "", f"# {page.get('displaytitle') or 'Evolved ' + card['name']}", ""]
    q = quote_text(w)
    if q:
        body += [f"> *In-game description:* \"{q}\"", ""]
    body += [f"**Base card:** [{card['slug']}.md](../cards/{card['slug']}.md)", ""]
    body += ["## Overview", "", lead_paragraph(w) or "Not specified on source page", ""]
    kept = existing_prose(EVO_DIR / f"{ev['slug']}.md")
    body += ["## What changes mechanically", "", prose(kept, "## What changes mechanically"), ""]
    body += ["## Evolution-specific stats/behavior", "",
             f"Cycles to evolve: {ib.get('cyclecost') or ev.get('cycles') or 'Not specified on source page'}. "
             f"Stat boosts vs. base card (from the Cards page evolution table): {ev.get('stat_boosts') or 'Not specified on source page'}.", "",
             "**Attributes**" if attr_tables else "", tables_md(attr_tables) if attr_tables else "", "",
             "**Stats by level**", "", tables_md(level_tables), ""]
    extra = substantive_sections(w)
    if extra:
        body += ["## Abilities (wiki sections)", ""]
        for h, md in extra:
            body += [f"### {h}", "", md, ""]
    body += ["## Notes", "", prose(kept, "## Notes"), ""]
    body += ["## Source", "", f"- {ev['url']} (scraped {now_iso()})", ""]
    (EVO_DIR / f"{ev['slug']}.md").write_text("\n".join(body))
    src = [f"# SOURCE PACK: {ev['title']} ({ev['url']})", "", "## Lead", lead_paragraph(w), "",
           "## Strategy (wiki)", strategy_block(w) or "(none)", ""]
    for h, md in extra:
        src += [f"## {h}", md, ""]
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    (SRC_DIR / f"{ev['slug']}.md").write_text("\n".join(src))


def write_hero_skeleton(hero: dict, card: dict | None, page: dict, image_rel: str, tables) -> None:
    w = page["wikitext"]
    ib = infobox_params(w)
    level_tables, attr_tables = split_tables(tables)
    targets_raw = attribute_value(tables, r"^Targets?$|^Target\b")
    dt = page.get("displaytitle") or ""
    hero_name = dt if dt and "/" not in dt else f"Heroic {hero['base_card']}"
    page = dict(page, displaytitle=hero_name)
    fields = [
        ("name", hero_name), ("slug", hero["slug"]),
        ("base_card", hero["base_card"]), ("base_card_slug", hero["base_slug"]),
        ("base_card_file", f"cards/{hero['base_slug']}.md" if card else ""),
        ("rarity", ib.get("rarity", card["rarity"] if card else "")),
        ("elixir_cost", ib.get("cost", hero["elixir_cost"])),
        ("ability_name", hero["ability_name"]), ("ability_cost", ib.get("abilitycost", hero["ability_cost"])),
        ("total_elixir_cost", hero["total_elixir_cost"]),
        ("card_type", ib.get("type", card["card_type"] if card else "")),
        ("targets", normalize_targets(targets_raw) if targets_raw else "n/a"),
        ("source_url", hero["url"]), ("image_path", image_rel), ("scraped_at", now_iso()),
        ("release_date", ib.get("releasedate", "")),
    ]
    title = page.get("displaytitle") or f"Heroic {hero['base_card']}"
    body = [fm(fields), "", f"# {title}", ""]
    q = quote_text(w)
    if q:
        body += [f"> *In-game description:* \"{q}\"", ""]
    if card:
        body += [f"**Base card:** [{hero['base_slug']}.md](../cards/{hero['base_slug']}.md)", ""]
    body += ["## Overview", "", lead_paragraph(w) or "Not specified on source page", ""]
    body += [f"## Ability: {hero['ability_name']}", "", f"*{hero['ability_summary']}* (costs {hero['ability_cost']} Elixir; "
             f"{hero['elixir_cost']} + {hero['ability_cost']} = {hero['total_elixir_cost']} total)", ""]
    extra = substantive_sections(w)
    for h, md in extra:
        body += [f"### {h}", "", md, ""]
    body += ["## Attributes", "", tables_md(attr_tables), ""]
    body += ["## Stats by level", "", tables_md(level_tables), ""]
    kept = existing_prose(HERO_DIR / f"{hero['slug']}.md")
    body += ["## Strong against", "", prose(kept, "## Strong against"), ""]
    body += ["## Weak against", "", prose(kept, "## Weak against"), ""]
    body += ["## Notes / synergies", "", prose(kept, "## Notes / synergies"), ""]
    body += ["## Source", "", f"- {hero['url']} (scraped {now_iso()})", ""]
    (HERO_DIR / f"{hero['slug']}.md").write_text("\n".join(body))
    src = [f"# SOURCE PACK: {hero['title']} ({hero['url']})", "", "## Lead", lead_paragraph(w), "",
           "## Strategy (wiki)", strategy_block(w) or "(none)", ""]
    for h, md in extra:
        src += [f"## {h}", md, ""]
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    (SRC_DIR / f"{hero['slug']}.md").write_text("\n".join(src))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def process(item: dict, heroes_by_base: dict, cards_by_name: dict) -> dict:
    kind, slug = item["kind"], item["slug"]
    page = fetch_page(item["title"])
    tables = html_tables(page["html"])
    img_name = pick_header_image(page, kind)
    if kind == "card":
        dest = CARD_IMG_DIR / f"{slug}.png"
    elif kind == "evolution":
        dest = EVO_IMG_DIR / f"{slug}.png"
    else:
        dest = HERO_IMG_DIR / f"{slug}.png"
    image_rel = str(dest.relative_to(KB))
    img_info = None
    img_err = None
    if img_name:
        try:
            if not (dest.exists() and dest.stat().st_size > 1024):
                img_info = download_image(img_name, dest)
            else:
                img_info = {"file": img_name, "bytes": dest.stat().st_size, "cached": True}
        except Exception as e:  # noqa: BLE001
            img_err = f"{img_name}: {e}"
    else:
        img_err = "no header image found on page"
    extra = {}
    if kind == "card":
        extra = write_card_skeleton(item["card"], page, image_rel, tables, heroes_by_base)
    elif kind == "evolution":
        write_evo_skeleton(item["card"], page, image_rel, tables)
    else:
        write_hero_skeleton(item["hero"], cards_by_name.get(item["hero"]["base_card"]), page, image_rel, tables)
    return {"slug": slug, "kind": kind, "image": img_info, "image_error": img_err,
            "n_tables": len(tables), **extra}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-fetch/re-write even if already fetched")
    ap.add_argument("--only", nargs="*", help="limit to these slugs")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    index = load_json(INDEX_JSON, None)
    hindex = load_json(HERO_INDEX_JSON, None)
    if not index or not hindex:
        print("run enumerate_cards.py first", file=sys.stderr)
        return 2
    cards = index["cards"]
    heroes = hindex["heroes"]
    cards_by_name = {c["name"]: c for c in cards}
    heroes_by_base = {h["base_card"]: h for h in heroes}
    manifest = load_manifest()
    items = manifest["items"]

    work = []
    for c in cards:
        work.append({"kind": "card", "slug": c["slug"], "title": c["title"], "card": c})
        if c["evolution"]:
            work.append({"kind": "evolution", "slug": c["evolution"]["slug"], "title": c["evolution"]["title"], "card": c})
    for h in heroes:
        work.append({"kind": "hero", "slug": h["slug"], "title": h["title"], "hero": h})
    if args.only:
        work = [w for w in work if w["slug"] in set(args.only)]
    if not args.force:
        work = [w for w in work if items.get(w["slug"], {}).get("stage") not in ("fetched", "done")]
    print(f"{len(work)} items to fetch")

    for d in (CARDS_DIR, CARD_IMG_DIR, EVO_DIR, EVO_IMG_DIR, HERO_DIR, HERO_IMG_DIR, SRC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, w, heroes_by_base, cards_by_name): w for w in work}
        for fut in as_completed(futs):
            w = futs[fut]
            entry = items.setdefault(w["slug"], {"kind": w["kind"], "title": w["title"], "status": "pending"})
            try:
                res = fut.result()
                entry.update({"stage": "fetched", "status": "pending", "reason": None,
                              "image": res["image"], "image_error": res["image_error"],
                              "n_tables": res["n_tables"], "updated_at": now_iso()})
                if res["image_error"]:
                    entry["status"] = "failed"
                    entry["reason"] = f"image: {res['image_error']}"
                if w["kind"] == "card":
                    c = w["card"]
                    c["elixir_cost"] = res["elixir_cost"]
                    c["targets"] = res["targets"]
                done += 1
                print(f"  ok  {w['kind']:9s} {w['slug']}  tables={res['n_tables']}"
                      + (f"  IMAGE-ERR {res['image_error']}" if res["image_error"] else ""))
            except Exception as e:  # noqa: BLE001
                entry.update({"stage": "enumerated", "status": "failed",
                              "reason": f"fetch: {e}", "updated_at": now_iso()})
                print(f"  FAIL {w['kind']:9s} {w['slug']}: {e}")
                traceback.print_exc()
            if done % 10 == 0:
                save_manifest(manifest)
                save_json(INDEX_JSON, index)
    save_manifest(manifest)
    save_json(INDEX_JSON, index)
    # refresh card_index.md with resolved costs/targets/status
    from enumerate_cards import write_index_md
    write_index_md(cards, heroes)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
