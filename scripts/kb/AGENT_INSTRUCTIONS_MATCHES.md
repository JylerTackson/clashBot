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

## Deliverable 2: card and deck insights (graph links)

Do NOT edit card or deck files yourself: several agents work on the same
cards at once. Write `insights.json` next to the match `context.json`
(same directory) and `tools/merge_insights.py` merges it into
`knowledge_base/cards/<slug>.md` (section `## Creator insights (ryleycr1)`)
and `knowledge_base/decks/<deck_key>.md` (section
`## Creator matches (ryleycr1)`) as marked blocks; re-runs replace blocks.

```
{
  "key": "<video_id>-m<match_index>",
  "match_file": "matches/<video_id>-m<match_index>.md",
  "video_title": "<title>",
  "cards": {
    "<slug>": ["clock 2:34: <how he used it, from the events> — <what he said>", "..."]
  },
  "deck": {
    "deck_key": "<sorted-8-slugs or null>",
    "cards": ["<8 slugs>"],
    "bullets": ["<how the deck was piloted in this match, with timestamps>"],
    "new_deck": {                       # only when knowledge_base/decks/<deck_key>.md does not exist
      "display_name": "<Ryley's name for it if given, else '<win condition> Ryley'>",
      "archetype_primary": "<one of: bait beatdown bridge-spam control cycle siege>",
      "archetype_secondary": "<archetype or \"none\">",
      "rationale": "<one sentence: why that archetype>",
      "why_it_works_md": "<2-5 sentences, markdown, from the match and commentary>"
    }
  }
}
```
Rules for the bullets: one to four per card, each grounded in an event or
timeline row (give the clock) and, where he talked about it, his words
(quote or close paraphrase). A card he only *mentioned* (not played) gets a
bullet that says so. Cards not in the knowledge base (unknown slug) go in
the match file's Data gaps, not in `cards`.

Own deck: `context.md` gives a per-game read and a video-level consensus
("Own deck, video-level consensus"). Use the consensus unless the
commentary says he changed decks; if fewer than 8 cards are known, set
`deck_key` to null and list the known cards.

Style: concrete, timestamps everywhere, card names as in the knowledge
base, no speculation presented as observation.

Cost discipline: read `context.md` once, in large chunks; for card files read
only the `## Overview`, `## Notes / synergies` and `## Deck archetypes`
sections of the cards you will actually cite (`grep -n "^## "` first), and
skip archetype files you already know. Do not re-read files. Reply briefly.
