# Match-context agent instructions (Ryley videos -> knowledge base)

You receive, for one match from a ryleycr1 YouTube video:
- `context.md` / `context.json`: what the perception pipeline saw, second by
  second (clock, Ryley's elixir, his hand, opponent elixir estimate, units on
  the field with tiles), every play event (who, which card, where, elixir
  before/after, how it was detected and how confident), and the auto-
  transcript lines placed at the time they were spoken.
- The knowledge base: `knowledge_base/cards/<slug>.md` (Phase 1 card facts +
  strategy), `knowledge_base/decks/*.md` (Phase 2 popular decks),
  `knowledge_base/archetypes/*.md`.

Reliability notes you must respect:
- Ryley's own plays come from his HUD (hand slot emptied + elixir drop) and
  are exact for card and time; the tile comes from the in-game deploy label
  when it was read (otherwise `tile: None`).
- Opponent plays come from deploy labels (reliable for any card) or from a
  2024-era unit detector (may miss or mislabel new cards; `unknown_unit`
  means something was there). `UNIDENTIFIED` events mean elixir moved but
  no card could be read: treat as a hole, do not guess the card.
- The transcript is auto-generated: card names are often mangled ("log
  bait", "e-barbs", "hog"). Map them to the knowledge base's card names.
- Everything in "Timeline" is observation; your job is interpretation, and
  you must keep the two apart in what you write.

## Deliverable 1: the match context file

Write `knowledge_base/matches/<video_id>-m<match_index>.md`:

```
---
video_id: <id>
video_title: <title>
video_url: https://www.youtube.com/watch?v=<id>
match_index: <n>
video_time: [<start>, <end>]
creator: ryleycr1
own_deck: [<8 slugs, or the observed subset>]
own_deck_key: <sorted-8-slugs or null>
own_archetype: <one of the six, or null>          # your judgement, from cards + how he plays
opponent_deck_seen: [<slugs>]
opponent_archetype_guess: <archetype or null>
result: <win | loss | draw | unknown>            # only if the video/transcript shows it
quality: {...copy the quality block from context.json...}
links:
  cards: [<slugs mentioned in this file>]
  decks: [<deck_keys from knowledge_base/decks that share >= 6 cards with own_deck, if any>]
  archetypes: [<archetype slugs>]
---
```
Sections:
- `## Summary` (3-6 sentences): the deck, the opponent's deck, how the match went.
- `## Key moments`: bullets `t=<video s> (clock m:ss)`: what happened (from
  the plays/timeline) and what Ryley said about it (quote or paraphrase the
  transcript). Every bullet must cite observation first, commentary second.
- `## How Ryley uses his cards`: one bullet per card he played, with the
  situations (defence/offence, placement tiles, elixir state, what it
  answered) drawn from the events, plus what he said about the card.
- `## Opponent`: their cards, how he read/countered them.
- `## Lessons stated by Ryley`: explicit advice he gives, each with the
  timestamp. Only things he actually said.
- `## Data gaps`: unidentified events, low-confidence stretches, transcript
  segments with no visual context.

## Deliverable 2: card and deck updates (graph links)

For each card Ryley played or discussed, append or update ONE block in
`knowledge_base/cards/<slug>.md`, placed before `## Source`, with these
markers so re-runs replace rather than duplicate:

```
## Creator insights (ryleycr1)

<!-- creator-insights:ryleycr1:<video_id>-m<match_index>:start -->
- Match [<video_title>](../matches/<video_id>-m<n>.md), clock <m:ss>: <how he used it / what he said>
<!-- creator-insights:ryleycr1:<video_id>-m<match_index>:end -->
```
If the `## Creator insights (ryleycr1)` heading already exists, add your
block inside that section (after existing blocks); if a block with your
exact marker exists, replace it. Never edit anything else in the card file.

For the deck: if `own_deck_key` matches a file in `knowledge_base/decks/`,
add the same kind of marked block under a `## Creator matches (ryleycr1)`
section in that deck file. If it does not match and the 8 cards are known,
create `knowledge_base/decks/<own_deck_key>.md` with the Phase 2 frontmatter
fields (display_name = Ryley's name for it if he gives one, otherwise the
win condition + "Ryley"), `classification_source: agent`,
`source_url` = the video URL, stats `n/a`, and the standard sections
(Cards, Classification, Why this deck works) written from the match.

Style: concrete, timestamps everywhere, card names as in the knowledge
base, no speculation presented as observation.
