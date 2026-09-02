# Extraction agent instructions (Clash Royale knowledge base, phase 1)

You are filling the judgement-based sections of pre-generated markdown files.
Everything mechanical (frontmatter, stats tables, images, cross-links) is
already written by `scripts/kb/fetch_cache.py` and must NOT be changed.

For every slug in your batch you get two files:

- **Target** (edit this): `knowledge_base/cards/<slug>.md`,
  `knowledge_base/evolutions/<slug>.md` or `knowledge_base/heroes/<slug>.md`
- **Source pack** (read only): `<cache>/src/<slug>.md` — the wiki page's lead
  paragraph, the Card Overviews summary, the full Strategy section and any
  Ability / Modifier sections, converted from wikitext. Use ONLY this source
  (plus the tables already in the target file). Do not invent facts from
  memory; the wiki is the ground truth for this dataset.

## What to write

Replace each `<!-- AGENT:FILL -->` placeholder with content. Nothing else in
the file may change (do not touch frontmatter, tables, headings, links).

### Cards and heroes

- `## Strong against` — bullet list. Each bullet names the enemy card(s) or
  card class (e.g. "ground swarms", "single-target tanks") this card handles
  well, and *why* in a few words, taken from the Strategy text. Prefer
  concrete card names when the source gives them. Aim for 4-10 bullets.
- `## Weak against` — same format, for what counters this card. Aim for 4-10
  bullets.
- `## Notes / synergies` — bullets with everything else substantive and
  decision-relevant from the Strategy text: placement tips, timing, elixir
  trades, combos/synergy partners, interactions (e.g. "resets Inferno Tower",
  "activates King Tower"), typical role (win condition / defensive /
  support / cycle / spell bait). Skip pure trivia and flavor text. Aim for
  5-15 bullets; more is fine when the source is rich.

If the source pack genuinely has nothing for a section, write exactly:
`Not specified on source page` (this is a valid, expected outcome).

### Evolutions

- `## What changes mechanically` — bullets: the new ability, how it differs
  from the base card, stat changes, cycle count (from the source lead /
  Ability sections and the "Evolution Attributes" table already in the file).
- `## Notes` — bullets with strategy-relevant points from the Strategy text:
  what it now beats, what still counters it, best uses. Use
  `Not specified on source page` when the page has no Strategy section.

## Style

- Plain markdown bullets (`- `), one idea per bullet, keep card names exactly
  as the wiki spells them (e.g. `P.E.K.K.A.`, `Mini P.E.K.K.A.`).
- Be concrete: numbers, tile distances and elixir values from the source are
  more useful than adjectives.
- Do not add headings, do not reorder sections, do not leave any
  `<!-- AGENT:FILL -->` behind.
