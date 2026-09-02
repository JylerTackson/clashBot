"""Step 6: verify the knowledge base and write meta/qa_report.md.

Also promotes manifest entries to `done` once their file is complete (no
agent placeholders left, frontmatter complete, image present), so it doubles
as the "finalize" step after each agent batch.

Checks:
  1. cards/*.md count == card_index card_count; missing slugs listed by name
  2. every card md has all required frontmatter fields, non-empty
  3. every image_path exists, is > 1KB and is a real PNG
  4. every has_evolution:true card has evolutions/<slug>-evolution.md;
     every hero has heroes/<slug>.md
  5. no `<!-- AGENT:FILL -->` placeholders remain; cross-links resolve
"""
from __future__ import annotations

import re
import sys

from common import (
    CARDS_DIR,
    EVO_DIR,
    HERO_DIR,
    HERO_INDEX_JSON,
    INDEX_JSON,
    KB,
    META_DIR,
    load_json,
    load_manifest,
    now_iso,
    save_manifest,
)

PLACEHOLDER = "<!-- AGENT:FILL -->"
CARD_REQUIRED = ["name", "slug", "rarity", "elixir_cost", "card_type", "targets",
                 "has_evolution", "has_hero_variant", "source_url", "image_path", "scraped_at"]
EVO_REQUIRED = ["base_card", "source_url", "image_path", "scraped_at"]
HERO_REQUIRED = ["name", "slug", "base_card", "source_url", "image_path", "scraped_at",
                 "ability_name", "ability_cost"]


def parse_frontmatter(text: str) -> dict | None:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    out = {}
    for line in m.group(1).split("\n"):
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] == '"':
            v = v[1:-1].replace('\\"', '"')
        out[k.strip()] = v
    return out


def check_image(rel: str) -> str | None:
    p = KB / rel
    if not p.exists():
        return "missing"
    if p.stat().st_size < 1024:
        return f"too small ({p.stat().st_size} bytes)"
    with open(p, "rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            return "not a PNG"
    return None


def check_file(path, required: list[str]) -> tuple[dict | None, list[str]]:
    problems = []
    if not path.exists():
        return None, ["file missing"]
    text = path.read_text()
    fm = parse_frontmatter(text)
    if fm is None:
        return None, ["no frontmatter"]
    for k in required:
        if fm.get(k, "") == "":
            problems.append(f"frontmatter `{k}` empty")
    if PLACEHOLDER in text:
        n = text.count(PLACEHOLDER)
        problems.append(f"{n} agent placeholder(s) unfilled")
    if "image_path" in fm and fm["image_path"]:
        err = check_image(fm["image_path"])
        if err:
            problems.append(f"image {fm['image_path']}: {err}")
    # relative links must resolve
    for link in re.findall(r"\]\((\.\./[^)]+\.md)\)", text):
        if not (path.parent / link).resolve().exists():
            problems.append(f"broken link {link}")
    return fm, problems


def main() -> int:
    index = load_json(INDEX_JSON, None)
    hindex = load_json(HERO_INDEX_JSON, None)
    if not index or not hindex:
        print("run enumerate_cards.py first", file=sys.stderr)
        return 2
    cards = index["cards"]
    heroes = hindex["heroes"]
    manifest = load_manifest()
    items = manifest["items"]

    report = ["# QA report", "", f"Generated {now_iso()}", ""]
    passed, failed = [], []

    def record(ok: bool, msg: str):
        (passed if ok else failed).append(msg)

    # 1. counts
    expected = index["card_count"]
    md_files = sorted(p.stem for p in CARDS_DIR.glob("*.md"))
    expected_slugs = [c["slug"] for c in cards]
    missing = [c["name"] for c in cards if c["slug"] not in md_files]
    extra = [s for s in md_files if s not in expected_slugs]
    record(not missing and len(md_files) == expected,
           f"Card file count: {len(md_files)} files in cards/ vs {expected} cards in card_index.md"
           + (f"; MISSING: {', '.join(missing)}" if missing else "")
           + (f"; UNEXPECTED: {', '.join(extra)}" if extra else ""))

    # 2/3/5 per card
    per_item: dict[str, list[str]] = {}
    fm_by_slug: dict[str, dict] = {}
    for c in cards:
        fm, probs = check_file(CARDS_DIR / f"{c['slug']}.md", CARD_REQUIRED)
        if fm:
            fm_by_slug[c["slug"]] = fm
            if fm.get("has_evolution") != ("true" if c["has_evolution"] else "false"):
                probs.append(f"has_evolution={fm.get('has_evolution')} disagrees with index ({c['has_evolution']})")
            if fm.get("targets") in ("", "pending"):
                probs.append("targets unresolved")
        per_item[c["slug"]] = probs
    n_bad_fm = sum(1 for s, p in per_item.items() if any("frontmatter" in x for x in p))
    record(n_bad_fm == 0, f"Card frontmatter: {len(cards) - n_bad_fm}/{len(cards)} cards have all required fields non-empty")
    n_bad_img = sum(1 for s, p in per_item.items() if any(x.startswith("image") for x in p))
    record(n_bad_img == 0, f"Card images: {len(cards) - n_bad_img}/{len(cards)} image_path files exist, are PNG and > 1KB")
    n_ph = sum(1 for s, p in per_item.items() if any("placeholder" in x for x in p))
    record(n_ph == 0, f"Card prose: {len(cards) - n_ph}/{len(cards)} cards have Strong/Weak/Notes sections filled")

    # 4. evolutions
    evo_cards = [c for c in cards if c["has_evolution"]]
    evo_missing = []
    for c in evo_cards:
        slug = c["evolution"]["slug"]
        fm, probs = check_file(EVO_DIR / f"{slug}.md", EVO_REQUIRED)
        if fm is None:
            evo_missing.append(c["name"])
        per_item[slug] = probs
    record(not evo_missing, f"Evolutions: {len(evo_cards) - len(evo_missing)}/{len(evo_cards)} cards with has_evolution:true have an evolutions/ file"
           + (f"; MISSING: {', '.join(evo_missing)}" if evo_missing else ""))
    n_evo_bad = sum(1 for c in evo_cards if per_item[c["evolution"]["slug"]])
    record(n_evo_bad == 0, f"Evolution files: {len(evo_cards) - n_evo_bad}/{len(evo_cards)} pass frontmatter/image/placeholder/link checks")

    hero_missing = []
    for h in heroes:
        fm, probs = check_file(HERO_DIR / f"{h['slug']}.md", HERO_REQUIRED)
        if fm is None:
            hero_missing.append(h["title"])
        per_item[h["slug"]] = probs
    record(not hero_missing, f"Heroes: {len(heroes) - len(hero_missing)}/{len(heroes)} heroes in the Heroes index have a heroes/ file"
           + (f"; MISSING: {', '.join(hero_missing)}" if hero_missing else ""))
    n_hero_bad = sum(1 for h in heroes if per_item[h["slug"]])
    record(n_hero_bad == 0, f"Hero files: {len(heroes) - n_hero_bad}/{len(heroes)} pass frontmatter/image/placeholder/link checks")

    # cross-link sanity: card -> evolution and evolution -> card, card -> hero
    xl_bad = []
    for c in evo_cards:
        ctext = (CARDS_DIR / f"{c['slug']}.md").read_text() if (CARDS_DIR / f"{c['slug']}.md").exists() else ""
        etext = (EVO_DIR / f"{c['evolution']['slug']}.md").read_text() if (EVO_DIR / f"{c['evolution']['slug']}.md").exists() else ""
        if f"../evolutions/{c['evolution']['slug']}.md" not in ctext or f"../cards/{c['slug']}.md" not in etext:
            xl_bad.append(c["name"])
    for h in heroes:
        if not h.get("has_base_card"):
            continue
        ctext = (CARDS_DIR / f"{h['base_slug']}.md").read_text() if (CARDS_DIR / f"{h['base_slug']}.md").exists() else ""
        htext = (HERO_DIR / f"{h['slug']}.md").read_text() if (HERO_DIR / f"{h['slug']}.md").exists() else ""
        if f"../heroes/{h['slug']}.md" not in ctext or f"../cards/{h['base_slug']}.md" not in htext:
            xl_bad.append(h["title"])
    record(not xl_bad, "Cross-links: every card<->evolution and card<->hero pair links both ways"
           + (f"; BROKEN: {', '.join(xl_bad)}" if xl_bad else ""))

    # update manifest statuses
    n_done = 0
    for slug, probs in per_item.items():
        e = items.get(slug)
        if e is None:
            continue
        if not probs:
            e["status"], e["stage"], e["reason"] = "done", "done", None
            n_done += 1
        else:
            if e.get("status") == "done":
                e["status"], e["stage"] = "pending", "fetched"
            if e.get("stage") in ("fetched", "done"):
                e["reason"] = "; ".join(probs)
            elif not e.get("reason"):
                e["reason"] = "; ".join(probs)
        e["updated_at"] = now_iso()
    save_manifest(manifest)

    # counts by kind
    from collections import Counter
    by_kind = Counter((e["kind"], e["status"]) for e in items.values())
    report.append("## Summary")
    report.append("")
    for kind in ("card", "evolution", "hero"):
        d = by_kind[(kind, "done")]; p = by_kind[(kind, "pending")]; f = by_kind[(kind, "failed")]
        report.append(f"- {'heroes' if kind == 'hero' else kind + 's'}: {d} done, {p} pending, {f} failed")
    report.append("")
    report.append("## Checks")
    report.append("")
    for m in passed:
        report.append(f"- PASS: {m}")
    for m in failed:
        report.append(f"- FAIL: {m}")
    report.append("")
    problems = {s: p for s, p in per_item.items() if p}
    report.append("## Per-item problems")
    report.append("")
    if not problems:
        report.append("None.")
    else:
        for s, p in sorted(problems.items()):
            report.append(f"- `{s}`: " + "; ".join(p))
    # sections the wiki did not cover (explicit sentinel, not an error)
    sentinel = "Not specified on source page"
    gaps = []
    for d in (CARDS_DIR, EVO_DIR, HERO_DIR):
        for path in sorted(d.glob("*.md")):
            text = path.read_text()
            secs = [m.group(1) for m in re.finditer(r"^## ([^\n]+)\n+" + re.escape(sentinel), text, flags=re.M)]
            if secs:
                gaps.append(f"- `{d.name}/{path.name}`: " + ", ".join(secs))
    report.append("")
    report.append(f"## Sections marked \"{sentinel}\" ({len(gaps)} files)")
    report.append("")
    report.append("These are expected outcomes (the wiki page has no such content), listed so they are visible.")
    report.append("")
    report.extend(gaps or ["None."])
    (META_DIR / "qa_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))

    # refresh index md statuses
    from enumerate_cards import write_index_md
    write_index_md(cards, heroes)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
