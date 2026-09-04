# State-enrichment agent instructions (one game, Phase 2 data)

You enrich the extracted game-state samples of exactly one game from a
ryleycr1 video. `<key>` = `<video_id>-m<match_index>` (e.g. `6qYQNJ1Uaeg-m0.4`).
You write one file, the sidecar `knowledge_base/states/<key>.enrich.json`, then
merge and validate it. You never edit the jsonl, the match file, card files or
the schema.

## Inputs

- `python3 tools/states_view.py <key>` — **your primary input**: a compact
  digest of every sample in `knowledge_base/states/<key>.jsonl`, one block per
  sample in time order. Run it once and work from that output; it is a fifth of
  the raw file. `--kind key` (or `play,periodic`) narrows it when you want to do
  the key moments first.
- `knowledge_base/states/<key>.jsonl` — the raw samples, the source of truth.
  Only open it (or grep one id out of it) when you need a field the digest
  leaves out: full unit lists, `recent_plays` detail, `context_refs`,
  `quality.notes`. Never read it end to end.
- `knowledge_base/matches/<key>.md` — the Phase 1 match file: summary, the
  `## Key moments` bullets (each `key` sample carries its bullet verbatim in
  `key_moment.text`), how Ryley uses his cards, his stated lessons, data gaps.
- `knowledge_base/meta/game_state_schema.md` — what every field means, the
  coordinate convention (`[col 0-17, row 0-31]`, rows 0-14 his half, 15-16 the
  river) and the reliability notes.
- `knowledge_base/cards/<slug>.md`, `decks/<deck_key>.md`, `heroes/` — only for
  cards you actually need to reason about, and only once each.

## What a digest block looks like

```
-- <video_id>-m<match_index>#<t> | key | t=798.8 | 1:50
[double_elixir|1:50] elixir 9 (opp ~10). hand: ... (next ...). deck: ...
field: ...            threats: ...          recent: ...
towers own L/R/K, enemy L/R/K
action: play berserker at [8,0] (back middle) after 1s
key: <the Phase 1 key-moment bullet, trimmed>
say: <what Ryley said within ±6 s>
out: own k/l/r 0/0/0 | enemy 0/-216/0 | took right | positive | trade +2 | game win
```

The id on the `--` line is the key you write in the sidecar. The block header
of the digest also carries the deck, hero cards, result and read quality.

## What each sample already contains

`state` (clock, phase, elixir both sides, hand + next card, his 8-card deck and
hero cards, tower HP, units on the field with tiles/headings/ETAs, threats,
both sides' plays in the last 12 s), `action` (what he did next, or `hold`),
`outcome` (tower HP deltas, towers taken/lost, opponent plays and elixir trade
over the next 15 s, `verdict`, `game_result`), `commentary` (transcript lines
within ±6 s) and `quality`. `state_text` (the six lines after the `--` header)
is the embedding rendering — never change it.

## What to write, per kind

Sidecar shape (ids exactly as in the jsonl, nothing else in the file):

```json
{
  "<video_id>-m<match_index>#<t>": {
    "situation_read": "...", "reaction": "...",
    "pro_action_rationale": "...", "principle": "...",
    "alternatives": ["..."], "outcome_note": "...",
    "confidence": "high|medium|low", "tags": ["defend-bridge-push", "elixir-lead"]
  }
}
```

- `kind: key` — all of it: `situation_read`, `reaction`, `pro_action_rationale`
  (why *he* did it, quoting his words from `commentary` / `key_moment.text` when
  he gave them), `principle` (one transferable sentence), `alternatives` (1-3
  other reasonable responses, each with why it is better or worse),
  `outcome_note`, `confidence`, `tags`.
- `kind: play` and `kind: periodic` — `situation_read`, `reaction`, `tags`,
  `confidence`; add `outcome_note` only when the outcome fields actually say
  something (a tower taken or lost, a swing above ~300 HP, a clear elixir
  trade). Nothing else: no `principle`, no `alternatives` on these.
- Every sample in the file gets an entry. Enrich the `key` samples first, then
  the rest — the key moments give you the game's story and make the short ones
  quick.

## Grounding rules

- Cite the state: threat ETAs, tile coordinates, elixir counts, tower HP, cycle
  position ("Miner is next"). If a claim is not visible in the sample or the
  match file, do not make it.
- Ryley's words are the only source for intent. Quote briefly (a clause, not a
  paragraph) and only from this sample's `commentary` or its `key_moment.text`.
  When he says nothing, say what the play does — never invent a motive.
- Only cards in `state.own.deck`, `state.own.hand`, `state.opponent.deck_known`
  or on the field may be named. `unknown_unit` stays unknown (the detector is
  2024-era); `hand_confidence` below ~0.55 means the hand slots are noisy — lean
  on the deck list, deploy labels and elixir instead, and drop to
  `confidence: medium|low`.
- A `null` hand slot, a `null` `next_card` or `action.type: "unknown"` means the
  extractor discarded a HUD read of a card outside his 8-card deck;
  `quality.notes` says which. Treat those as unknown — never as the card named
  in the note — and reason from the elixir drop and the rest of the hand.
- `reaction` is advice to the playing agent in the knowledge base's card
  vocabulary ("Bomb Tower at [9,10] and hold The Log for the Bats"), not a
  narration of what happened.
- `outcome.verdict` is a heuristic. If the outcome fields disagree with it, say
  so in `outcome_note` rather than repeating the label.
- 1-2 sentences per field. No markdown, no bullet lists, no links inside the
  strings. `tags` are 2-5 short retrieval slugs
  (`defend-bridge-push`, `counter-push`, `spell-cycle`, `punish`, `elixir-lead`,
  `overtime`, `tower-race`, `hero-ability`, ...).

## Finish

```
python3 tools/merge_state_enrichment.py <key>          # folds the sidecar in, deletes it
python3 tools/validate_states.py knowledge_base/states/<key>.jsonl
```

Both must exit 0. If the merge reports unknown ids or bad fields, fix the
sidecar (it is kept on failure) and re-run — do not hand-edit the jsonl.

## Cost discipline

Run `tools/states_view.py <key>` once and read the match file once; keep both
in context instead of re-reading, and do not read the raw jsonl in full. Do not open a card file you already know, and never open all card
files "to be safe". Do not run the extractor, perception, torch or OCR. Write
the sidecar in one pass (one Write call), not sample by sample.

## Report (5 lines)

1. key + samples enriched by kind.
2. merge + validate result.
3. what the game turned on, in one sentence.
4. samples you were unsure about (ids) and why.
5. anything wrong in the data (bad hand reads, missing tiles, misattributed plays).
