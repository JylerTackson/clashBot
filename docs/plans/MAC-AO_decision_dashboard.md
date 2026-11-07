# MAC-AO — decision dashboard: show the retrievals, the knowledge graph, the candidates and why the bot acted, live and on replay (spec for one Sonnet implementer)

Author: Opus · 2026-11-07 · For the class showcase. Read `docs/plans/MAC-00_common.md` first. No decision logic changes.

User: *"We also need a way to visualize the retrievals, KG, and the actions being taken and why so I can demonstrate that to my teacher in the presentation."*

Default taken (user did not pick): a local browser dashboard beside the iPhone Mirroring window. It follows the run being written (live) and replays any past run (for slides and screenshots).

## Owns
```
clashbot/viz/ (new package: __init__.py, __main__.py, server.py, data.py, static/index.html, static/app.js, static/style.css)
clashbot/candidates.py (only: remember the last retrieval hits and KG contributions; no change to candidates or scores)
clashbot/live.py (only: extra fields in the decisions.jsonl row written at the end of `LiveLoop.step`)
tests/test_viz.py (new)   tests/test_candidates*.py / tests/test_live_loop.py (only for the new fields)
```
Another implementer (MAC-AN, demo launcher) edits `live.py` (`main` split into `build_stack`, a tap gate, a battle
field on rows) and adds `runs/demo/<session>/battles.jsonl`. Stay inside the decision-row lines of `live.py`. The
launcher starts you as a subprocess: `python -m clashbot.viz --follow runs/demo --open`.

## Facts you need
- A run dir (`runs/poc/<ts>/`, `runs/smoke/<name>/`, later `runs/demo/<ts>/`) has `config.json`, `decisions.jsonl` (one
  row per decision: t, key, tile, candidates [key, score, sources {knn, kg, tac, ability}, tile, note], probs (the d1
  ranker's distribution over the candidates), executor {status, sent, clicks}, timings_ms, mode, reason, tactics_lines,
  kg_lines (KG v2 fact lines given to d1), tile_source, elixir, hand, context_text), `holds.jsonl` (MAC-AM M4: t, mode,
  act, reason, elixir, hand, pushes), `frames.jsonl`, `summary.json` (at the end), `shots/` (`--record-shots`: images
  `d<idx>_t<t>_{a,b,c}.jpg` — a = decision frame with the tap drawn, b/c = +0.5 s / +1.5 s — and the recorder's index;
  see `clashbot/recorder_shots.py`).
- Retrieval: `clashbot/retrieval.StateIndex.search` fuses dense kNN (`featurize`) and BM25 (`context_text`) with RRF and
  returns `Hit(row, id, game_id, kind, action_key, verdict, elixir_trade, score, ranks)`.
  `CandidateGenerator._channels` (candidates.py) calls it with k=20 and sums hit scores per action key (the "knn" source);
  the KG channel uses `KnowledgeGraph.counter_scores(threat)` for each recent threat (the "kg" source). The state rows
  live in `db/clashbot.sqlite` (find the table holding the state id, game / video, time, context text and the pro's
  action; see `tools/db/seed.py` and `clashbot/db.py`).
- KG: `clashbot/kg.py` (`KnowledgeGraph`: counter_scores, neighbors(node_id, rel, direction)) over `db/clashbot.sqlite`;
  KG v2 `clashbot/kg2.py` (`KG2`: answers_for, facts_lines, roles, stats) over `db/kg2.sqlite`.

## O1. Log what the dashboard needs (`candidates.py`, `live.py`)
- In `_channels`, keep `self.last_retrieval = [{"id", "game_id", "kind", "action_key", "verdict", "elixir_trade", "score",
  "ranks"}]` (the k=20 hits, in order) and `self.last_kg = [{"threat", "card", "key", "score"}]` (each KG contribution
  that reached a legal key). Reset both at the start of each call. Nothing else in the generator changes (test: the
  candidates of a fixed state are identical before and after).
- In `LiveLoop.step`, add to the decisions.jsonl row: `"retrieval": generator.last_retrieval[:10]` and
  `"kg_used": generator.last_kg` (empty lists when absent, e.g. hold paths that skip the generator). Cost < 0.5 ms;
  measure it.

## O2. Server (`clashbot/viz/server.py`, `data.py`, `__main__.py`)
`python -m clashbot.viz [--run DIR | --follow PARENT] [--port 8765] [--open]`
- Python stdlib only (`http.server.ThreadingHTTPServer`, json, sqlite3); bind 127.0.0.1. `--open` opens the browser.
- `--follow PARENT` (default `runs/demo`): always show the newest run dir under PARENT and switch when a newer one
  appears. `--run DIR`: a fixed run (replay).
- JSON API:
  - `GET /api/runs` — run dirs under runs/poc, runs/demo, runs/smoke (name, start time, decisions, has retrieval logs).
  - `GET /api/run/<name>/meta` — config (deck, heroes, evos, models, flags), summary if present, battles.jsonl if present.
  - `GET /api/run/<name>/decisions?since=N` — rows from index N; each enriched with the shot file names for that decision.
    Read files incrementally (remember byte offsets); tolerate a partial last line.
  - `GET /api/run/<name>/holds?since=N` — likewise; also bucketed reasons (digits → "#").
  - `GET /api/shot/<name>/<file>` — a jpg from that run's shots/ only (reject `..`, absolute paths, non-jpg).
  - `GET /api/state/<id>` — one retrieved pro state: game/video id, time in the video, context text, the pro's action,
    verdict, elixir trade.
  - `GET /api/kg?cards=a,b,c&focus=x` — the subgraph among those cards plus each card's top-3 counters from
    `KnowledgeGraph` (and KG v2 answers when available): nodes {id, label, side: own|enemy|other}, edges {src, dst, rel,
    weight}. Use the KG's own direction convention for COUNTERS (check it and document it in the UI legend).
- Load the KG / state DB lazily once; read-only sqlite connections (`mode=ro` URI).

## O3. Page (`static/`)
No external JS/CSS/fonts (offline presentation room). Plain JS, SVG for the graph, CSS grid. Poll every 1 s. Panels:
1. **Header**: run name, live/replay, deck with evo/hero marks, ranker / policy / models, decisions, taps, latency
   p50/p95 so far, current battle (battles.jsonl).
2. **Screen**: shot a of the selected decision (tap marked by the recorder), b/c thumbnails; "no screenshot" placeholder.
3. **Decision**: t, mode, the policy's reason, the chosen key, tile and tile source; every candidate with its d1
   probability as a bar, its sources (knn / kg / tac / ability as chips with values) and its note; the chosen one
   highlighted; executor status (sent / dry_run / gated / hold).
4. **Retrieval**: the top-10 pro states: rank, fused score, kNN rank and BM25 rank (from `ranks`), game, the pro's play,
   verdict / elixir trade; click a row → its context text (`/api/state`). Old runs without `retrieval`: show
   "not logged in this run (logged since MAC-AO)".
5. **Knowledge graph**: SVG of the selected decision's enemy cards (parse the `field:` / `threats:` lines of
   `context_text`) and our hand, edges by relation (legend), the `kg_used` edges highlighted with their scores; below it
   the `kg_lines` (KG v2 facts given to d1) and the `tactics_lines`.
6. **Timeline**: decisions over game time as marks coloured by mode, holds as a band coloured by reason bucket; click a
   mark to select that decision; live mode auto-selects the newest decision unless the user clicked (a "follow" toggle).
Keep it legible on a projector: large fonts, high contrast, no animation beyond the 1 s refresh.

## Tests
- `test_viz.py`: API on a fixture run dir built in tmp (old-format rows without retrieval + new-format rows); incremental
  `since` reads with a partial last line; shot path traversal rejected (404/400); `/api/kg` returns nodes/edges for real
  cards when `db/` exists (skip otherwise); `/api/state/<id>` on a real id when `db/` exists.
- Generator: `last_retrieval` / `last_kg` filled, candidates unchanged; live loop row has `retrieval` and `kg_used`.
- Full suite green: `../.venv/bin/python -m pytest -q --deselect tests/test_perception_latency.py`.

## Verification
1. Video dry run with the new logging (heavy jobs one at a time; no `--live`):
   `HF_HUB_OFFLINE=1 ../.venv/bin/python -m clashbot.live --source video --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60 --calib runs/poc/mirror_calib.json --deck electro-dragon,musketeer,tesla,fireball,knight,zap,ram-rider,mini-p-e-k-k-a --hero-cards musketeer --ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac --policy fast --act-model db/models/act-gbdt --record-shots --seconds 180 --out runs/smoke/AO_demo`
   Compare latency p50/p95 with `runs/smoke/AM_merged` (same video, 120 s) — the logging must not move them.
2. Start `python -m clashbot.viz --run runs/smoke/AO_demo` and `curl` every endpoint; save the JSON of 2 decisions to the
   report. Also replay `runs/poc/20261107-111554` (old format) to show the fallbacks.
3. Do NOT run anything with `--live` or `CLASHBOT_ALLOW_LIVE=1`.

## Report
Branch, commit, files; logging cost; endpoint samples; how the KG direction convention is shown; latency comparison;
deviations and why; anything for Opus (e.g. data the dashboard wanted but the row does not have).
