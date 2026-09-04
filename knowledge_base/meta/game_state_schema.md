# Game-state samples (Phase 2 data: in-game retrieval)

Machine schema: `game_state_schema.json` (JSON Schema draft 2020-12). Records live
in `knowledge_base/states/<video_id>-m<match_index>.jsonl`, one sample per line,
built by `tools/extract_states.py` from the perception run of each game and
enriched by Opus agents (`scripts/kb/AGENT_INSTRUCTIONS_STATES.md`).

Purpose: while the agent plays, it renders its current state the same way
(`state_text`), embeds it, retrieves the nearest samples and reads what the
professional (Ryley) did next, how it turned out, and, for key moments, why.

## Sample kinds

| kind | when it is taken | what it carries |
|---|---|---|
| `key` | the state just before a Phase 1 key moment (`## Key moments` in the match file, matched by `t=`) | full `enrichment` (rationale, principle, alternatives) plus the key-moment text and commentary |
| `play` | the state ~1 s before each of Ryley's plays that is not inside a key moment | `action` = the play; short `situation_read` + `reaction` |
| `periodic` | every 10 s of match time when nothing above applies | `action` = his next play within 6 s or `hold`; short `situation_read` + `reaction` |

## Coordinates and vocabulary

Tiles are `[col, row]`, col 0-17 left to right, row 0-31 from Ryley's king
tower to the opponent's; rows 0-14 are his half, 15-16 the river, 17-31 the
opponent's. Card names are Phase 1 slugs (`knowledge_base/cards/<slug>.md`).
Units come from the 2024-era detector: newer cards can appear as
`unknown_unit`; deploy-label plays are exact for any card.

## Reliability

`quality.hand_confidence` below ~0.55 means the hand slots are noisy; deploy
labels, elixir deltas and the deck list are the reliable signals. `clock_read`
false means the clock was not readable for that game (times are video seconds
only). `outcome.verdict` is a heuristic (tower HP swing and elixir trade) and
should be read together with `enrichment.outcome_note`.

## `state_text` template (deterministic, produced by the extractor)

```
[phase|clock] elixir E (opp ~O). hand: A, B, C, D (next N). deck: ...
field: <unit>(side)@[c,r] <heading> eta <tower> <s>s; ...
threats: ...
recent: own X@[c,r] 3s ago; opp Y@[c,r] 5s ago
towers own L/R/K, enemy L/R/K
action: play <card> at [c,r] (<zone> <lane>) after <d>s | hold
```
The playing agent must render its own state with the same template so the
embeddings are comparable.
