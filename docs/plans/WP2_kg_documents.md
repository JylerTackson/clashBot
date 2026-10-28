# WP2 — Knowledge graph, card stats and pro-advice documents (spec for a Sonnet implementer)

Author: Opus · 2026-10-28 · Implements D-008, D-009, D-013, D-014 and the KG rows of `docs/research/course_mapping.md` §3.
Depends on WP1 being present: `clashbot/config.py`, `clashbot/game_rules.py` (`load_cards`, `is_playable`), `clashbot/units.py` (`load_unit_map`), `tools/data/build_clean_dataset.py` (function `scrub`), and the built `dataset/v1/{states.jsonl,games.jsonl,splits.json}`.

**You are implementing, not designing.** If anything is unclear or contradicts the data, stop and report the question; do not choose.

## 0. Ground rules
- Python 3.11, stdlib only (pytest for tests). Do not modify `knowledge_base/`, WP1 files, `tools/extract_states.py`, `cr_perception/`.
- Do not git commit/push. Do not create files other than those listed.
- At the top of each script: `ROOT = Path(__file__).resolve().parents[2]` and `sys.path.insert(0, str(ROOT))` before importing `clashbot` or `tools.data.build_clean_dataset` (`tools/` is a namespace package; that import works).

## 1. Files to create
```
tools/data/kb_markdown.py        shared markdown helpers (frontmatter, sections, bullets, tables, alias matcher)
tools/data/build_kg.py           writes dataset/v1/kg/nodes.jsonl, dataset/v1/kg/edges.jsonl, dataset/v1/card_stats.jsonl
tools/data/build_documents.py    writes dataset/v1/documents.jsonl
tests/test_kg_documents.py
```

## 2. `tools/data/kb_markdown.py`

```python
def parse_frontmatter(text: str) -> tuple[dict, str]
```
If `text` starts with `---\n`: the frontmatter is the lines up to the next line that equals `---`; the body is everything after that line. For each frontmatter line matching `^([A-Za-z0-9_]+):\s*(.*)$`: value = group 2 stripped; if it starts with `[` and ends with `]` → list = `[x.strip().strip('"') for x in value[1:-1].split(",") if x.strip()]`; else strip one pair of surrounding double quotes if present. Lines not matching (indented continuation lines, nested keys) are ignored. No frontmatter → `({}, text)`.

```python
def section(body: str, heading: str) -> str | None
```
Returns the text between the line that equals `"## " + heading` and the next line starting with `"## "` (or end of text), without the heading line; None if the heading is absent.

```python
def bullets(section_text: str) -> list[str]
```
A bullet starts at a line beginning with `"- "`; the bullet text is the rest of that line. Following lines that begin with two spaces and are not blank are appended to the current bullet with a single space (after stripping). Any other line ends the current bullet. Returns the bullets, stripped.

```python
def first_table(section_text: str) -> list[dict]
```
Finds the first run of consecutive lines starting with `|`. Row 1 = header, row 2 = separator (skip), remaining rows = data. Split each row on `|`, drop the first and last empty pieces, strip cells. Returns a list of `{header_cell: cell}` dicts (rows with a different cell count than the header are skipped). Empty list if no table.

```python
def creator_blocks(text: str) -> list[tuple[str, str]]
```
`re.finditer(r"<!-- creator-insights:ryleycr1:(?P<game>[^:]+):start -->\n?(?P<body>.*?)<!-- creator-insights:ryleycr1:(?P=game):end -->", text, re.S)` → list of `(game_id, body.strip())` in file order.

```python
def build_aliases(cards: dict) -> list[tuple[str, str]]
```
For every slug in `cards` (all cards, including tower troops), aliases: `name.lower()`, `slug.replace("-", " ")`, and if `"." in name`: `name.lower().replace(".", "").strip()`. Plus these fixed extras (alias → slug), only added when the slug exists in `cards`:
`{"pekka": "p-e-k-k-a", "mini pekka": "mini-p-e-k-k-a", "e-spirit": "electro-spirit", "e spirit": "electro-spirit", "ewiz": "electro-wizard", "e-wiz": "electro-wizard", "x bow": "x-bow"}`.
Deduplicate (alias, slug) pairs; drop aliases shorter than 3 characters; return sorted by `(-len(alias), alias)`.

```python
def mentions(text: str, aliases: list[tuple[str, str]], exclude: str | None = None) -> list[str]
```
`low = text.lower()`; `consumed = [False] * len(low)`; `found = []` of `(start, slug)`. For each `(alias, slug)` in the given order: for each `m` in `re.finditer(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", low)`: if none of `consumed[m.start():m.end()]` is True → mark them True and append `(m.start(), slug)`. Return slugs ordered by start, duplicates removed (first occurrence), with `exclude` removed.

## 3. `tools/data/build_kg.py`

CLI: `python3 tools/data/build_kg.py [--dataset DIR]` (default `clashbot.config.DATASET_DIR`). Writes into `<dataset>/kg/` (create) and `<dataset>/card_stats.jsonl`.

### 3.1 Nodes — `nodes.jsonl`, one JSON object per line: `{"id", "type", "name", "props"}`
Written in this order, each group sorted by id:

| type | id | name | props |
|---|---|---|---|
| card | `card:<slug>` for every slug in `load_cards()` | card name | `rarity, card_type, elixir_cost, targets, has_evolution, has_hero_variant` |
| evolution | `evo:<slug>` for each `knowledge_base/evolutions/*.md` (slug from frontmatter `slug`) | frontmatter `name` | `base_card_slug, elixir_cost, cycles, card_type` (strings as in frontmatter; missing → None) |
| hero | `hero:<slug>` for each `knowledge_base/heroes/*.md` | frontmatter `name` | `base_card_slug, elixir_cost, ability_name, ability_cost, total_elixir_cost, card_type, targets` |
| unit | `unit:<class>` for every key of `load_unit_map()` **plus** every `unit_class` appearing in `dataset/v1/states.jsonl` | class | `kind, primary, is_evo` from `clashbot.units.resolve(class)` (`primary` = resolve()["card"]) |
| deck | `deck:<deck_key>` for every deck in `knowledge_base/meta/deck_index.json` | `display_name` | `archetype_primary, archetype_secondary, avg_elixir, cards, evolutions, heroes` (from deck_index) |
| archetype | `archetype:<archetype>` for each `knowledge_base/archetypes/*.md` (frontmatter `archetype`) | frontmatter `archetype_name` | `deck_count, secondary_deck_count` |
| game | `game:<game_id>` for every line of `dataset/v1/games.jsonl` | game_id | `split, deck_status, result, own_archetype, opponent_archetype_guess` |

### 3.2 Edges — `edges.jsonl`, one object per line: `{"src", "rel", "dst", "props"}`

1. `EVOLUTION_OF`: `evo:<slug>` → `card:<base_card_slug>` (skip if the card node does not exist). props `{}`.
2. `HERO_OF`: `hero:<slug>` → `card:<base_card_slug>` (same skip rule). props `{}`.
3. `SPAWNED_BY`: for every unit node, for each slug in `resolve(class)["candidates"]`: `unit:<class>` → `card:<slug>`, props `{"primary": slug == resolve(class)["card"]}`.
4. `IN_DECK`: for each deck in deck_index, for each slug in `cards`: `card:<slug>` → `deck:<deck_key>`, props `{"evolved": slug in deck["evolutions"], "hero": slug in deck["heroes"]}` (skip unknown cards).
5. `HAS_ARCHETYPE`: `deck:<key>` → `archetype:<archetype_primary>` props `{"role": "primary"}`; and → `archetype:<archetype_secondary>` props `{"role": "secondary"}` when secondary is a non-empty string that is not `"none"` and differs from primary. Skip if the archetype node does not exist.
6. Wiki relations, for each `knowledge_base/cards/<slug>.md` (page card = `slug`), using `aliases = build_aliases(load_cards())`:
   - `## Strong against` bullets: for each `x in mentions(bullet, aliases, exclude=slug)` → edge `card:<slug>` -COUNTERS-> `card:<x>` with source `"wiki_strong_against"`.
   - `## Weak against` bullets: each mention `x` → `card:<x>` -COUNTERS-> `card:<slug>` with source `"wiki_weak_against"`.
   - `## Notes / synergies` bullets: if `bullet.lower().startswith("substitutes")` → each mention `x` → `card:<slug>` -SUBSTITUTE-> `card:<x>` source `"wiki_notes"`; else each mention → `card:<slug>` -SYNERGY-> `card:<x>` source `"wiki_notes"`.
   - Skip bullets equal to `"Not specified on source page"` and sections whose stripped text equals that string.
   - Merge: one edge per `(src, rel, dst, source)`; props `{"source": source, "count": <number of bullets producing it>, "evidence": [first 3 bullets, each scrub()-ed and cut to 300 chars]}`.
7. Game edges, from `games.jsonl`: `game:<id>` -GAME_OWN_CARD-> `card:<c>` for c in `own_deck`; -GAME_OPP_CARD-> `card:<c>` for c in `opponent_deck`; -GAME_DECK-> `deck:<own_deck_key>` when not null and the deck node exists. props `{}`.
8. `ANSWERED_BY` (mined, **train split only**), from `dataset/v1/states.jsonl`:
   - Use samples with `split == "train"`, `action.type == "play"`, `action.card` not null, and `"deck_invalid"` not in `flags`.
   - `play_time = t + (action.delay_s if action.delay_s is not None else 0.0)`.
   - For each `p` in `state.opponent.recent_plays` with `p.card` not null and `0.0 <= play_time - p.t <= 4.0`: candidate pair `(p.card, action.card)`; dedupe key `(game_id, round(play_time, 1), action.card, round(p.t, 1), p.card)`.
   - Aggregate per pair: `support` (deduped count), `positive`, `neutral`, `negative` (from `outcome.verdict`; other values count only in support), `mean_elixir_trade` (mean of non-null `outcome.elixir_trade`, rounded to 3, null if none), `confidence = round(positive / support, 3)`.
   - Emit `card:<opp_card>` -ANSWERED_BY-> `card:<own_card>` for pairs with `support >= 3`, props = the aggregates plus `{"source": "mined_train_states", "window_s": 4.0}`.

Edge file order: groups in the numbered order above; within a group sorted by `(src, dst, props.get("source", ""))`.

### 3.3 `card_stats.jsonl`
One line per `knowledge_base/cards/<slug>.md` that exists for a slug in `load_cards()` (sorted by slug):
`{"slug", "name", "card_type", "rarity", "elixir_cost", "targets", "attributes": <first_table(section(body,"Attributes"))[0] or {}>, "stats_by_level": first_table(section(body, "Stats by level")) or []}` (`elixir_cost` etc. from `load_cards()`).

### 3.4 Output summary
Print node counts by type, edge counts by rel (and COUNTERS split by source), and the number of ANSWERED_BY edges.

## 4. `tools/data/build_documents.py`

CLI: `python3 tools/data/build_documents.py [--dataset DIR]`. Writes `<dataset>/documents.jsonl`. Each line:
`{"doc_id", "type", "title", "text", "cards", "deck_key", "archetype", "game_id", "split"}`
`text` is always `scrub(...)`-ed; skip a document whose scrubbed text is empty. `cards` = `mentions(text, aliases)` unless stated otherwise. `split` = the game's split from games.jsonl when `game_id` is set and known, else null. Unset fields are null.

| type | source | doc_id | title | extra fields |
|---|---|---|---|---|
| `card_insight` | each `creator_blocks()` entry in `knowledge_base/cards/<slug>.md` | `card_insight:<slug>:<game_id>` | `"<card name> — pro notes"` | cards = `[slug]` + mentions (excluding slug), game_id |
| `deck_insight` | each `creator_blocks()` entry in `knowledge_base/decks/<key>.md` | `deck_insight:<key>:<game_id>` | `"<display_name> — pro notes"` | deck_key = key, game_id, archetype = deck's archetype_primary |
| `deck_why` | `section(body, "Why this deck works")` of each deck file | `deck_why:<key>` | `"Why <display_name> works"` | deck_key, archetype = archetype_primary, cards = the deck's card list from deck_index |
| `archetype_guide` | `section(body,"Gameplan")` + `"\n\n"` + `section(body,"Why it works")` of each archetype file (missing parts → "") | `archetype_guide:<archetype>` | `"<archetype_name> gameplan"` | archetype |
| `match_lessons` | `section(body, "Lessons stated by Ryley")` of each `knowledge_base/matches/<game_id>.md` | `match_lessons:<game_id>` | `"Lessons — <game_id>"` | game_id, deck_key = games.jsonl own_deck_key, archetype = games.jsonl own_archetype |
| `match_card_usage` | `section(body, "How Ryley uses his cards")` | `match_card_usage:<game_id>` | `"Card usage — <game_id>"` | as above |
| `match_opponent` | `section(body, "Opponent")` | `match_opponent:<game_id>` | `"Opponent — <game_id>"` | as above |

`display_name` and `archetype_primary` come from `deck_index.json`. Match files whose game_id is not in games.jsonl are still processed (split null). Order: by type in the table order, then by doc_id. Print counts by type.

## 5. Tests — `tests/test_kg_documents.py`
1. `parse_frontmatter` on a synthetic string with a list value and a quoted value.
2. `section` / `bullets` on a synthetic section with a two-space continuation line.
3. `first_table` on a synthetic 3-column table with 2 rows.
4. `mentions`: `"Skeleton Army and Skeletons beat Mini P.E.K.K.A., not Zappies or Zap"` with real aliases → `["skeleton-army", "skeletons", "mini-p-e-k-k-a", "zappies", "zap"]`; `"Spear Goblins"` → `["spear-goblins"]` only.
5. `creator_blocks` on a synthetic two-block string.
6. Integration: run both scripts via subprocess on the real `dataset/v1` (skip with `pytest.skip` if `dataset/v1/games.jsonl` is missing). Assert: every edge's src and dst ids exist in nodes.jsonl; there is at least one COUNTERS edge from each source (`wiki_strong_against`, `wiki_weak_against`); `card:archers` -COUNTERS-> `card:minions` exists (from the archers page); no ANSWERED_BY edge has support < 3; every documents.jsonl text is non-empty and contains no `"http"` and no `"t="`; doc_ids are unique.

## 6. Acceptance (run, paste outputs)
```
python3 -m pytest -q tests/test_kg_documents.py
python3 tools/data/build_kg.py
python3 tools/data/build_documents.py
python3 -c "import json,collections;e=[json.loads(l) for l in open('dataset/v1/kg/edges.jsonl')];print(collections.Counter(x['rel'] for x in e))"
grep -m3 '"rel": "ANSWERED_BY"' dataset/v1/kg/edges.jsonl
git status --short
```

## 7. Report back (≤ 40 lines)
Files created; test results; node/edge/document counts; three example ANSWERED_BY edges; any spec item you could not follow exactly and why.

---

## Amendment B (Opus review, 2026-10-28)

**B1. Deck registry from deck files (D-017).** `deck_index.json` lists only 19 of the 68 deck files; every one of the 55 own-deck keys in `games.jsonl` has a deck file. Replace every use of `deck_index.json` (deck nodes, `IN_DECK`, `HAS_ARCHETYPE`, `GAME_DECK`, `deck_insight`, `deck_why`, the `display_name`/`archetype_primary`/card list lookups) with a registry built from `knowledge_base/decks/*.md` (skip non-`.md` files), implemented as `load_deck_registry() -> dict[deck_key, dict]` in `tools/data/kb_markdown.py`:
- From frontmatter: `deck_key`, `display_name`, `archetype_primary`, `archetype_secondary`, `avg_elixir`, `evolutions`, `heroes`. For `evolutions` and `heroes`: a list value stays a list; a string equal to `"none"` or empty → `[]`; any other string → `[x.strip() for x in s.split(",") if x.strip()]`.
- `cards`: from `section(body, "Cards")`, for each bullet, the first match of `r"\(\.\./cards/([a-z0-9-]+)\.md\)"`, in bullet order.
- `in_index`: True if the deck_key is in `deck_index.json`, else False (add to deck node props).
Deck node props become `archetype_primary, archetype_secondary, avg_elixir, cards, evolutions, heroes, in_index`.

**B2. scrub() fix (in WP1's `tools/data/build_clean_dataset.py`, allowed for this amendment only).** Replace regex 2 with exactly:
`r"\bt\s*[=≈~]\s*[~≈]?\s*\d+(?:\.\d+)?(?:\s*[-–]\s*[~≈]?\d+(?:\.\d+)?)?\s*(?:—|–|:)?\s*"`
Add to `tests/test_clean_dataset.py` a test: `"t=" not in scrub('(called at t=~316 (0:37)): "I have to go"')` and `"t=" not in scrub("t≈12-~15 — he waits")`.

**B3. Rebuild order.** Another agent is reading `dataset/v1/states.jsonl` right now: do **not** run `build_clean_dataset.py`. Run `build_kg.py`, then `build_documents.py`, then `python3 tools/db/seed.py --hnsw`.

**B4. Tests.** Integration: deck nodes = 68; every `games.jsonl` row with non-null `own_deck_key` has a `GAME_DECK` edge.

Re-run the §6 acceptance commands plus `python3 -m pytest -q tests/test_clean_dataset.py` and the seed command, and report counts (nodes by type, edges by rel, documents by type, seed table counts).

**B5. Head-first mention rule for wiki relations (precision fix).** Bullets are written as `<subject> — <explanation>`, and explanations often name other cards only for comparison (e.g. "Goblin Barrel — a great counter; slightly harder to time than Arrows" produced a false `the-log COUNTERS arrows`). In §3.2 item 6 (COUNTERS, SYNERGY, SUBSTITUTE), replace `mentions(bullet, aliases, exclude=slug)` with `relation_mentions(bullet, aliases, exclude=slug)`, a new function in `kb_markdown.py`:
- If `" — "` (space, em dash U+2014, space) occurs in the bullet: `head = bullet.split(" — ", 1)[0]`; `h = mentions(head, aliases, exclude)`; if `h` is non-empty return `h`.
- Otherwise (no separator, or the head has no card) return `mentions(bullet, aliases, exclude)`.
Add tests: `relation_mentions("Goblin Barrel — a great counter; harder to time than Arrows", aliases)` == `["goblin-barrel"]`; `relation_mentions("Low-hitpoint swarms — Minions, Goblins", aliases)` == `["minions", "goblins"]`. Report the new COUNTERS/SYNERGY/SUBSTITUTE counts next to the old ones.
