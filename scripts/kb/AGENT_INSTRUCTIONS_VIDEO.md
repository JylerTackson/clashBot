# Video coordinator agent (one per ryleycr1 video)

You own one video: every game the perception pipeline split it into. Your job
is to get one match file and one `insights.json` written per game, in
parallel, then verify them and report. You do not write match files yourself
unless a game sub-agent fails.

Inputs: a video id `<vid>`; the game contexts live at
`runs/videos/<vid>/match_<m>/game_<k>/context.md` (+ `context.json`), or at
`runs/videos/<vid>/match_<m>/context.md` when a period has a single game.
`runs/videos/<vid>/video_deck.json` holds the video-level deck consensus
(`mixed: true` means the games use different decks). The video title and
duration are in `data/videos/<vid>/*.info.json`.

## Steps

1. `python3 tools/dispatch_matches.py list --video <vid>` prints one JSON line
   per ready game (key, context paths, seconds, events, own_deck). Read the
   first 12 lines of each game's `context.md` (deck lines, hero note, cards
   named) and `video_deck.json` so you can brief the sub-agents.
2. Decide the video format from the title and the consensus: single deck
   across games (consensus deck present) or deck showcase (`mixed`).
3. For EVERY game, spawn one sub-agent (model opus, general-purpose, in
   parallel, in one message) with this brief, filling the placeholders:

   ---
   You are a Clash Royale analyst turning one match from a creator video
   (ryleycr1) into knowledge-base content. Repo: /home/user/clashBot. Match
   key: <key>
   1. Read /home/user/clashBot/scripts/kb/AGENT_INSTRUCTIONS_MATCHES.md in full
      and follow it exactly.
   2. Read <context_md> in full (in chunks); <context_json> has the same data
      (use it for the `quality` block).
   3. Read the KB card files for every card Ryley played or discussed
      (/home/user/clashBot/knowledge_base/cards/<slug>.md), the archetype files
      (/home/user/clashBot/knowledge_base/archetypes/), the hero files in
      /home/user/clashBot/knowledge_base/heroes/ when a hero variant is in play,
      and check /home/user/clashBot/knowledge_base/decks/ for a file matching
      the deck key or sharing >= 6 cards.
   4. Write /home/user/clashBot/knowledge_base/matches/<key>.md and
      <dir>/insights.json (schema in the instructions). Do NOT edit
      card/deck/hero/archetype files.
   5. Validate with `python3 -c "import json;json.load(open('<dir>/insights.json'))"`
      and `python3 /home/user/clashBot/tools/merge_insights.py --dry-run --only <key>`.
   Video notes: <title>; <single-deck or deck-showcase sentence>; <this game's
   index, whether it is the first/last, whether it looks short/partial>; <any
   hero note>; cards missing from the KB (wiki stubs such as Minion Giant) go
   in prose and Data gaps only, never as an insights slug. HUD hand reads are
   the least reliable signal (confidence often ~0.5): prefer deploy labels,
   elixir deltas and commentary, and record every override in Data gaps. Keep
   observation and interpretation separate. Do not run video processing,
   torch, or OCR. Reply with a 5-line report: files written, own deck used,
   result if known, number of card insight entries, anything unsure about.
   ---

4. When all sub-agents report: check every game has both files, run
   `python3 tools/merge_insights.py --dry-run --only <all keys>` and make sure
   it succeeds. Cross-check the decks: in a single-deck video all games should
   carry the same `own_deck_key`; if one differs, ask that sub-agent (send it
   a message) to reconcile or justify in Data gaps. Re-run a failed sub-agent
   once. Do not merge, do not commit.
5. Reply with: video id/title, number of games, the deck(s) used, results per
   game, total card insight entries, and the open doubts the sub-agents
   flagged (one line each).
