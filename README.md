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

## Phase 2 — deck archetypes and meta decks

```
knowledge_base/
  decks/<deck_key>.md            one file per scraped deck (deck_key = 8 sorted card slugs)
  archetypes/<archetype>.md      beatdown, control, cycle, bait, bridge-spam, siege
  meta/deck_index.md|json        every deck: archetype, classification source, site stats, avg elixir
  cards/<slug>.md                gains an idempotent "## Deck archetypes" section
```

```
cd scripts/kb
python3 decks_fetch.py                    # step 0-1: policy check, fetch/render, enumerate decks
python3 decks_fetch.py --html page.html   # ...or parse a page saved from a normal browser
python3 decks_build.py build              # step 2-3: classify (heuristic) + write deck files
python3 agent_batches.py                  # (phase 1 planner; phase 2 agents read AGENT_INSTRUCTIONS_DECKS.md)
python3 decks_build.py finalize           # step 4-6: archetypes, index, card cross-links, verify, QA
python3 tests/test_phase2.py              # end-to-end smoke test on a synthetic fixture
```

Status: the live site cannot be fetched by automation — royaleapi.com serves
a Cloudflare managed challenge (interactive Turnstile) to both plain HTTP and
headless Chromium, and its robots.txt disallows AI agents (details in
`meta/qa_report.md`). The current data therefore comes from a copy of the
popular-decks page saved from a normal browser session and passed in with
`--html`: 20 entries, 19 unique decks (Cycle 7, Bridge Spam 3, Siege 3,
Beatdown 2, Control 2, Bait 2), with hero variants and evolutions recorded per
deck. Classifications are heuristic unless a reviewing agent changed them
(`classification_source: agent`); the page shows no archetype labels of its
own. Re-running with a newer saved page updates decks, archetype files and
card cross-links in place.

## Phase 4 — perception pipeline (`cr_perception/`)

Read-only state extraction from Clash Royale footage (recorded videos now;
a macOS emulator window via `mss` later). Nothing in this package can send
input.

```
cr_perception/
  sources.py     VideoFrameSource / ImageDirSource / ScreenSource (mss + Quartz window lookup,
                 Screen-Recording-permission check that fails loudly)
  screen.py      black-bar stripping, game-panel detection from the elixir bar (streaming layouts),
                 per-frame readiness (match | match_weak | menu | unreadable) + MatchGate hysteresis
  geometry.py    18x32 tile grid, perspective homography, bottom-centre rule, legal placement mask
  config.py      calib.json: every ROI/corner as a FRACTION of the content rect
  hud.py         elixir (bar fill + digit templates), hand/next card (template matching vs Phase 1 art),
                 clock (digit templates, RapidOCR fallback), tower HP; every reader returns (value, conf)
  detect.py      KataCR YOLOv8 x2 (direct inference, ally/enemy channel) and BuildABot ONNX backends
  events.py      own plays from hand change + elixir drop; opponent plays from new arena tracks;
                 unidentified spells from tower-HP drops -> PlayEvent(card=None, confidence=low)
  elixir_sim.py  regen simulator; own-side drift vs HUD is the error bar on the opponent estimate
  decktracker.py opponent deck / cycle tracking + Phase 2 KB inference (labelled as inference)
  state.py       GameState / PlayEvent contract;  recorder.py  JSONL;  overlay.py  debug rendering
tools/calibrate.py        frame -> calib.json (+ verification grid image); auto arena from towers
tools/run_video.py        video -> states.jsonl + overlay.mp4 + summary.json (fps, drift, deck)
tools/segment_transcript.py  keep only subtitle cues spoken during readable match periods
tools/benchmark_capture.py   macOS capture fps + permission check
scripts/download_videos.py   yt-dlp channel downloader (parallel, subtitles, --cookies)
tests/                     15 tests: grid ground truth, homography, bottom-centre rule, legal mask,
                           elixir sim + drift, deck cycle + KB inference, HUD readers on synthetic
                           HUDs built from the Phase 1 art, end-to-end synthetic match
```

Status: see the Phase 4 report in the session log. Real-video validation (ROI
tuning, detector accuracy, fps, deck-tracker replay, elixir drift) requires a
downloaded match; YouTube currently blocks this session's IP with a bot check,
so `scripts/download_videos.py --cookies cookies.txt` is the way in.
