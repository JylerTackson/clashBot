# clashBot

A Clash Royale playing agent. This repository currently contains **Phase 1**:
a static, file-based knowledge base of every card, card evolution and hero,
scraped from the [Clash Royale Fandom wiki](https://clashroyale.fandom.com).

## Layout

```
knowledge_base/
  cards/<card_slug>.md              one file per card (frontmatter + stats + strategy)
  cards/images/<card_slug>.png      the card's header image from the wiki infobox
  evolutions/<card_slug>-evolution.md
  evolutions/images/<card_slug>-evolution.png
  heroes/<card_slug>-hero.md
  heroes/images/<card_slug>-hero.png
  meta/card_index.md                master table: every card, rarity, cost, type, targets,
                                    has_evolution, has_hero, scrape status
  meta/card_index.json              same data, machine readable (+ Card Overviews blurbs)
  meta/hero_index.json              hero list with ability names/costs
  meta/scrape_manifest.json         per-item progress log (pending | done | failed)
  meta/qa_report.md                 verification results
scripts/kb/                         the scraper (see below)
```

Slugs are the wiki page title lower-cased with non-alphanumerics collapsed to
`-` (`Mega Knight` -> `mega-knight`, `P.E.K.K.A.` -> `p-e-k-k-a`,
`Knight/Hero` -> `knight-hero`).

### Card file schema

```yaml
---
name: Mega Knight
slug: mega-knight
rarity: Legendary
elixir_cost: 7            # "n/a" for Tower Troops
card_type: Troop          # Troop | Spell | Building | Tower Troop
targets: ground           # ground | air | ground_and_air | buildings | n/a
has_evolution: true
has_hero_variant: false
source_url: https://clashroyale.fandom.com/wiki/Mega_Knight
image_path: cards/images/mega-knight.png   # relative to knowledge_base/
scraped_at: 2026-09-02T02:31:39+00:00
evolution_file: evolutions/mega-knight-evolution.md   # only when has_evolution
hero_file: heroes/knight-hero.md                      # only when has_hero_variant
ability_cost: 1           # Champions only
arena: Electro Valley
release_date: 8 September 2017
---
```

Sections: Overview, Attributes, Stats by level, Abilities and special
mechanics, Strong against, Weak against, Notes / synergies, Evolution, Hero
variant, Source. A section the wiki does not cover reads
`Not specified on source page` so "unknown" is distinguishable from "missed".

## Re-running the scraper

```
pip install -r scripts/kb/requirements.txt
cd scripts/kb
python3 enumerate_cards.py   # step 1: index pages -> card_index / hero_index / manifest
python3 fetch_cache.py       # steps 2-4 (mechanical): pages, images, skeleton md files
python3 agent_batches.py     # plan LLM extraction batches for the Strong/Weak/Notes sections
python3 verify.py            # step 6: QA report + promote manifest entries to done
```

All fetches go through the wiki's MediaWiki API (`api.php`) because the
rendered pages sit behind a Cloudflare challenge; images come from the static
CDN with `format=png`. Page responses are cached in `.kb_cache/` (git-ignored)
and items already marked `fetched`/`done` in the manifest are skipped, so the
run is resumable. `fetch_cache.py --force --only <slug ...>` re-does specific
items.

The extraction agents follow `scripts/kb/AGENT_INSTRUCTIONS.md`: they only
replace `<!-- AGENT:FILL -->` placeholders using the per-page source pack in
`.kb_cache/src/`, never the mechanically generated parts.
