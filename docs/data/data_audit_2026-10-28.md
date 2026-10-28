# Data audit — 2026-10-28 (Opus)

Scope: everything tracked under `knowledge_base/` after Phase 2 data closed (commit `abfce9e`).
`runs/` (perception outputs) and `data/` (videos, transcripts, cookies) are gitignored and are
**not present** in this container; nothing below depends on them.

## Inventory

| path | files | size | what it is | role going forward |
|---|---|---|---|---|
| `knowledge_base/states/*.jsonl` | 197 (one per game) | 67 MB | 10,923 GameStateSample v1 records, all enriched | **core training + retrieval corpus** |
| `knowledge_base/cards/*.md` + png | 126 + 126 | 28 MB (png ≈ 24 MB) | wiki card pages; frontmatter, stats tables, strong/weak/synergy lists, creator-insight blocks (≈ 2.0 MB of the 3.4 MB of markdown) | **KG source** (nodes, counter/synergy edges, stats) |
| `knowledge_base/evolutions`, `heroes` | 42 + 16 md (+png) | 17 MB | wiki pages for evolutions / heroes | KG source (EVOLVES / HERO_OF) |
| `knowledge_base/decks/*.md` | 68 | 0.8 MB | RoyaleAPI decks + archetype classification + creator insights | KG source (IN_DECK, archetype) |
| `knowledge_base/archetypes/*.md` | 6 | 28 KB | archetype pages | KG source |
| `knowledge_base/matches/*.md` | 197 | 4.8 MB | per-game narrative written by agents from video + transcript | pre-game "professional context" docs; game-level metadata (own deck, opp deck, result) |
| `knowledge_base/meta/*` | 9 | 0.4 MB | indexes, schema, QA report | reference |

## YouTube-specific content (candidates for removal from the clean layer)

- In every state sample: `source.video_title`, `source.video_url`, `source.creator`, `commentary[]` (raw ±6 s transcript lines), `context_refs` (file paths), `quality.state_source`.
- In match files: video title/url, `video_time`, verbatim transcript quotes, "readable window" notes.
- In card/deck files: `creator-insights:ryleycr1:*` blocks (distilled pro advice with quotes).

The distilled advice (enrichment fields, creator insights, match "lessons") is the professional context the agent is supposed to load pre-game, so it is *game knowledge*, not YouTube noise. Raw transcript lines and video metadata are not needed at runtime.

## Consistency findings (state samples, n = 10,923)

| check | result | notes |
|---|---|---|
| own deck size ≠ 8 | 12 games / 669 samples | `vp7jP_vjxcU-m0.0/m0.1` carry 29- and 35-card "decks" (multi-deck video union); 8 games have 7 cards (`CInrqMTlVkg-*` = Minion Giant, which is missing from the KB); 2 games have 9 |
| opponent `deck_known` > 8 | 44 games | misattributed own plays and detector noise inflate the opponent deck |
| ally units whose card is not in own deck | 3,076 / 16,466 (18.7%) | side attribution or class noise; many are opponent cards |
| unit class names that are not card slugs | 48 names, 135 distinct total | YOLO class vocabulary (`skeleton`, `zappy`, `archer-evolution`, `elixir-golem-small`, `unknown_unit` = 5,097 occurrences) needs a unit → card mapping |
| hand slots unknown | 0 null: 2,664; 1: 3,625; 2: 2,725; 3: 1,286; 4: 623 | median `hand_confidence` = 0 |
| action type | play 8,555 · hold 700 · unknown 1,668 | unknown = no usable label for a ranker |
| `state.phase` null | 2,017 | clock unread; `match_seconds` still present |
| elixir out of [0,10] | 0 own, 0 opponent | fine |
| game result | win 8,684 · loss 1,120 · unknown 1,119 | strong win bias (one creator's highlight videos): behavioural cloning learns a winner's policy but outcome labels are imbalanced |

## Implications

1. A deterministic repair pass is needed before anything is indexed or trained on (deck validity, unit → card mapping, side re-attribution, opponent deck cap).
2. Labels for a ranker exist for ≈ 9,255 samples (play + hold); `unknown` actions are usable for retrieval context only.
3. Splits must be by video, not by sample, or near-duplicate states from the same game leak across splits (the RETRO contamination lesson from the RAG lecture).
