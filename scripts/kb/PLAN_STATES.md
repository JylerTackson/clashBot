# Plan: Phase 2 data (match-state samples) — implementation brief for the builder agent

Owner: orchestrator (Fable). Builder: one Opus agent. Reviewer: orchestrator.
Schema is fixed: `knowledge_base/meta/game_state_schema.json` (+ `.md`). Do not
change the schema; if a field is impossible to fill, leave it null and say so
in the report.

## Inputs that already exist (do not recompute perception)

- `runs/videos/<vid>/match_<m>/[game_<k>/]context.json`: header, `events`
  (play_event records with `timestamp`, `player`, `card`, `tile`,
  `elixir_before/after`, `detect_source`, `confidence`, `detail`), `timeline`
  (one row per 2 s: `t, clock, phase, own_elixir, opp_elixir_est, hand, next,
  towers_own, towers_enemy, units[{class, side, tile, heading, speed, pred_2s,
  pos_std, eta_tower}], threats[]`), `transcript` cues, `own_deck_observed`,
  `own_deck_key`, `own_deck_video` (consensus; `mixed` flag), `hero_note`,
  `opponent.deck_known`, `quality`.
- `runs/videos/<vid>/match_<m>/states.jsonl` for 15 locally processed videos:
  per-frame `state` records (~6 Hz) with `own.hand`, `own.hand_conf`,
  `own.elixir`, `own.towers`, `opponent.*`, `units`, `match_clock`,
  `match_seconds`, `phase`, `readiness`. Use it, when present, to take the
  pre-play state at exactly t_play - 1.0 s; otherwise use the last timeline
  row at or before t_play.
- `knowledge_base/matches/<key>.md`: Phase 1 match file. Frontmatter has
  `own_deck`, `own_deck_key`, `result`; `## Key moments` bullets start with
  `- t=<a>` or `- t=<a>-<b>` (video seconds; sometimes `(clock m:ss)`). The
  deck in the match file frontmatter overrides `own_deck_observed` (agents
  corrected many pipeline reads).
- `knowledge_base/meta/card_index.json` (slugs, names, elixir_cost),
  `knowledge_base/decks/<deck_key>.md`.
- `tools/dispatch_matches.py` (`ready()` lists games and context paths).

## Deliverables

1. `tools/extract_states.py [--only <key>...] [--periodic 10] [--horizon 15]`
   - For every game with a context (skip `empty`/`split_into`), write
     `knowledge_base/states/<key>.jsonl` with samples per the schema:
     * key samples: one per key-moment bullet, state at `t_start - 1.0`
       (clamped into the game window); `kind: key`, `key_moment` filled.
       If a bullet has no `t=`, try to map its `clock m:ss` via the timeline;
       else skip and count it in the report.
     * play samples: for every own play event not within ±4 s of a key
       moment: state at `t_play - 1.0`, `action` = that play.
     * periodic samples: every `--periodic` seconds of the game window,
       skipping instants within ±4 s of an existing key/play sample;
       `action` = the next own play within 6 s, else `hold`.
   - `state.own.deck` = match-file frontmatter deck (fallback context);
     `hero_cards` from the match file text ("Hero <Card>" / "Heroic <Card>")
     or the context `hero_note` when the match file confirms it.
   - `recent_plays` (both sides) from events in [t-12, t]; `outcome` from
     timeline rows t..t+horizon and events in that window; `elixir_trade`
     from card costs (card_index) of plays in the window; `verdict`:
     positive if (enemy tower damage - own tower damage) > 300 or a tower
     taken, negative if < -300 or a tower lost, else neutral; `game_result`
     from the match-file frontmatter.
   - `commentary`: transcript cues within ±8 s.
   - `state_text` exactly per the template in `game_state_schema.md`.
   - `context_refs`: match file, deck file if it exists, card files for
     hand + action card + on-field unit classes that map to a card slug
     (`cr_perception.detect.to_card_slug`), opponent deck file if the
     `opponent.kb_matches` has a strong match.
   - `quality`: `clock_read` (any clock in the game), `hand_confidence`
     (context quality.hand_conf_mean), `calibration`, `state_source`, notes.
   - Deterministic and idempotent; never overwrites an existing sample's
     `enrichment` (re-running merges: keep enrichment by `id`).
   - Print per game: samples by kind, key bullets unmatched.
2. `tools/validate_states.py [paths]`: validates every line against the JSON
   schema WITHOUT third-party packages (write a small validator covering the
   subset of JSON Schema used: type/enum/const/required/properties/
   additionalProperties/items/min-max/pattern/$ref/$defs), checks id
   uniqueness and that `state_text` matches the template's first line
   pattern; exit non-zero on any error; `--summary` prints counts by kind and
   enrichment coverage.
3. `tools/merge_state_enrichment.py <key>...`: folds
   `knowledge_base/states/<key>.enrich.json` (`{"<id>": {enrichment...}}`)
   into the jsonl (only the `enrichment` field), validates, deletes the
   sidecar on success.
4. `scripts/kb/AGENT_INSTRUCTIONS_STATES.md`: brief for the per-game Opus
   enrichment agent: inputs (the game's jsonl, the match file, card files
   on demand, `game_state_schema.md`), what to write for each kind
   (key: all enrichment fields; play/periodic: `situation_read`,
   `reaction`, `tags`, `confidence`; `outcome_note` when the outcome fields
   say something), grounding rules (cite state fields and Ryley's words;
   never invent cards not in the state/deck; keep each field to 1-2
   sentences), output = the `.enrich.json` sidecar keyed by sample id, then
   run `tools/merge_state_enrichment.py <key>` and
   `tools/validate_states.py knowledge_base/states/<key>.jsonl`, and a
   5-line report. Include cost discipline (read the jsonl once; do not
   re-read card files you already know).
5. Tests in `tests/test_states.py`: schema validator on a hand-written valid
   and invalid sample; extractor on a synthetic context (key/play/periodic
   counts, no overlap, outcome deltas, state_text template); merge keeps
   enrichment on re-extract.

## Constraints

- Python 3.11 stdlib + the repo's existing modules only.
- Do not run perception, torch or OCR; do not modify `cr_perception/`.
- Run the extractor on ALL games at the end and report totals by kind, the
  number of unmatched key bullets, and the 3 largest jsonl files with sizes.
- Report back with: files written, test results, totals, open questions.
