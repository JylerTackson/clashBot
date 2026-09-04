# State-enrichment worker session (Phase 2 data)

You are one of five worker sessions. Your slice is a group in
`scripts/kb/state_worker_assignments.json` (the orchestrator names it). No
videos, cookies, perception or torch are needed: every input is in git.

## Inputs (all committed)

- `knowledge_base/states/<key>.jsonl`: extracted game-state samples for each
  game (`<key>` = `<video_id>-m<match_index>`), schema in
  `knowledge_base/meta/game_state_schema.json` and `.md`.
- `knowledge_base/matches/<key>.md`: the Phase 1 match file (key moments,
  card usage, lessons, data gaps).
- `knowledge_base/cards/<slug>.md`, `knowledge_base/decks/`, `heroes/`.
- `scripts/kb/AGENT_INSTRUCTIONS_STATES.md`: the per-game enrichment brief.

## Steps

1. `git checkout -b phase5/<your-worker-name>` from the checked-out branch.
2. For every game key in your slice, spawn one sub-agent (model **opus**,
   general-purpose), at most 6 in parallel, with this brief:

   ---
   You are enriching game-state samples for one Clash Royale match from a
   ryleycr1 video. Repo: /home/user/clashBot. Match key: <key>.
   Read /home/user/clashBot/scripts/kb/AGENT_INSTRUCTIONS_STATES.md in full
   and follow it exactly. Inputs: /home/user/clashBot/knowledge_base/states/<key>.jsonl,
   /home/user/clashBot/knowledge_base/matches/<key>.md, and card files only
   as needed. Write /home/user/clashBot/knowledge_base/states/<key>.enrich.json,
   then run `python3 tools/merge_state_enrichment.py <key>` and
   `python3 tools/validate_states.py knowledge_base/states/<key>.jsonl`
   (both must succeed). Do not edit any other file. Reply with a 4-line
   report: samples enriched by kind, validation result, anything unsure.
   ---

3. After each batch of sub-agents: run
   `python3 tools/validate_states.py --summary knowledge_base/states/<key>.jsonl ...`
   for their keys; re-run a failed sub-agent once; then
   `git add knowledge_base/states/<keys>.jsonl && git commit -m "states: <video_id> enriched (<n> games)" && git push -u origin phase5/<your-worker-name>`.
   Commit per video, do not wait for the whole slice.
4. Never edit `knowledge_base/cards`, `decks`, `heroes`, `matches` or the
   schema; never touch `runs/` or `data/`; never push to the main feature
   branch. When the slice is pushed, reply with one line per video (id,
   games enriched, key samples, any failures).
