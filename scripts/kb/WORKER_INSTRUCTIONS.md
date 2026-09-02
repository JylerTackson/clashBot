# Perception worker session (one machine, a slice of ryleycr1 videos)

You are one of several worker sessions. Your slice is a group in
`scripts/kb/worker_assignments.json` (the orchestrator tells you which). The
orchestrator merges your output; you never edit card/deck/hero/archetype files
and never push to the main feature branch.

## 1. Setup (once)

```
cd /home/user/clashBot            # the repo clone; branch claude/clash-royale-kb-phase-1-4rtfm6
export CR_SCRATCH=$HOME/cr_scratch
bash tools/worker_setup.sh        # deps, KataCR source + weights, tests
```
Write the YouTube cookies you were given to `data/cookies.txt` (Netscape
format, exactly as given) and `chmod 600` it. `data/` is gitignored; never
print, commit or paste the cookies anywhere.

## 2. Download your videos (sequential, gentle)

```
python3 scripts/download_videos.py --ids <id1> <id2> ... --cookies data/cookies.txt --workers 1 --sleep 3
```
Expected: `data/videos/<id>/*.mp4` (H.264, 720 short side), `*.en.vtt`,
`*.info.json`, and `data/videos/manifest.json`. If YouTube answers 403/429 or
"Sign in to confirm", wait 5 minutes and retry once; report persistent
failures instead of hammering.

## 3. Perception (serial, CPU-bound: never run two at once)

```
export CR_SCRATCH=$HOME/cr_scratch
nohup python3 tools/batch_process.py --detect-every 12 --label-every 6 > $CR_SCRATCH/batch.log 2>&1 &
```
It processes every downloaded video (resumable via `runs/videos/manifest.json`),
~20-40 min per video, and writes `runs/videos/<id>/match_<m>/[game_<k>/]context.{md,json}`
plus `runs/videos/<id>/video_deck.json`. Do not run torch/OCR/tests while it
runs. Watch `$CR_SCRATCH/batch.log` for `period N:` lines (one per finished
video) and `FAILED`/`Traceback`.

## 4. Match agents (as each video finishes)

For each finished video: `python3 tools/dispatch_matches.py list --video <id>`
lists its games. Read the first 12 lines of each game's `context.md` and
`runs/videos/<id>/video_deck.json` (`mixed: true` = deck-showcase video, each
game a different deck; otherwise the consensus deck applies to all games).
Spawn ONE sub-agent per game, all in parallel, model **opus**, general-purpose,
with this brief (fill the placeholders):

---
You are a Clash Royale analyst turning one match from a creator video
(ryleycr1) into knowledge-base content. Repo: /home/user/clashBot. Match key: <key>
1. Read /home/user/clashBot/scripts/kb/AGENT_INSTRUCTIONS_MATCHES.md in full
   and follow it exactly (including its cost-discipline section).
2. Read <context_md> in full (large chunks); <context_json> beside it has the
   same data (use it for the `quality` block).
3. Read only the sections you need of the KB card files for cards you will
   cite (/home/user/clashBot/knowledge_base/cards/<slug>.md), the hero files
   (/home/user/clashBot/knowledge_base/heroes/) when a hero variant is in
   play, the archetype files, and check /home/user/clashBot/knowledge_base/decks/
   for a file matching the deck key or sharing >= 6 cards.
4. Write /home/user/clashBot/knowledge_base/matches/<key>.md and
   <game_dir>/insights.json. Do NOT edit card/deck/hero/archetype files.
5. Validate with `python3 -c "import json;json.load(open('<game_dir>/insights.json'))"`
   and `python3 /home/user/clashBot/tools/merge_insights.py --dry-run --only <key>`.
Video notes: "<title>"; <single-deck session: the consensus deck is X, use it
unless this game's commentary contradicts it | deck-showcase video: each game
uses a different deck, use the per-game read plus commentary>; game index <k>
of <n> (<first/last/short-partial note>); <hero note if any>. Cards missing
from the KB (wiki stubs such as Minion Giant) go in prose and Data gaps only,
never as an insights slug. HUD hand reads are the least reliable signal
(confidence often ~0.5): prefer deploy labels, elixir deltas and commentary,
and record every override in Data gaps. Keep observation and interpretation
separate. Do not run video processing, torch, or OCR (a batch job owns the
CPU). Reply with a 5-line report: files written, own deck used, result if
known, number of card insight entries, anything unsure about.
---

When the sub-agents report: check both files exist per game, run
`python3 tools/merge_insights.py --dry-run --only <keys>` (must succeed), and
in a single-deck video make sure all games carry the same `own_deck_key`
(message a sub-agent to reconcile if not). Re-run a failed sub-agent once.
Then `python3 tools/dispatch_matches.py done <keys>`.

## 5. Hand back (per finished video, do not wait for the whole slice)

```
git add knowledge_base/matches/<id>-m*.md
git add -f runs/videos/<id>/video_deck.json runs/videos/<id>/match_*/summary.json \
    runs/videos/<id>/match_*/context.json runs/videos/<id>/match_*/context.md \
    runs/videos/<id>/match_*/game_*/context.json runs/videos/<id>/match_*/game_*/context.md \
    runs/videos/<id>/match_*/game_*/insights.json runs/videos/<id>/match_*/insights.json 2>/dev/null
git add -f runs/videos/manifest.json runs/videos/agent_manifest.json
git commit -m "worker: <id> (<n> games) contexts, match files, insights"
git push -u origin <your-branch>
```
Never `git add` states.jsonl, videos, weights, templates or cookies. Never
run merge_insights without `--dry-run`. Never push to
`claude/clash-royale-kb-phase-1-4rtfm6`.

When the whole slice is pushed, reply with one line per video: id, games,
deck(s), any failures.
