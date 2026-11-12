# clashBot in Docker: one command, the agent end to end

clashBot is a Clash Royale playing agent. This package runs its **whole decision stack on a bundled clip of a pro match** (60 seconds of it by default)
and shows every decision in a live dashboard in your browser. Nothing is tapped or sent anywhere: the executor is the dry-run one.
(Playing for real through macOS iPhone Mirroring cannot run in a container; for a Mac see "Live play on a Mac (optional)" below.)

## Run it

Needs Docker (Docker Desktop, or Docker Engine with the Compose plugin) on **Linux, Windows or macOS, Intel or ARM**, and an internet
connection for the build. The image builds on your machine for your CPU.

```bash
unzip clashbot_docker_submission.zip && cd clashbot_docker
docker compose up --build
```

Then open **http://localhost:8765**: the Recording panel plays the clip in real time and the agent's taps appear as it makes them.
Stop with Ctrl-C.

- **The build downloads** PyTorch (CPU) and the Python packages, and the fine-tuned d1 ranker (1.17 GB) from Hugging Face
  (`Jyler/clashbot-d1-ft-joint-tac`, public, pinned to one revision, no account or token). First build: about 5 to 10 minutes on
  a normal connection (longer on a machine that has to emulate another CPU architecture); later starts reuse the cache.
- **Resources:** give Docker at least **6 GB of RAM** (Docker Desktop: Settings, Resources; the models need about 5 GB while
  the run is in progress) and about 12 GB of disk.
- **Run time:** the demo plays the first **60 seconds** of the clip **in real time** (the agent is shown it like a live stream: a
  frame is never handed over before its time, and the frames the agent is too slow for are skipped, as a live capture drops them),
  at most 10 frames per second (the rate of the live capture; every third video frame). So the run takes **about 60 seconds plus the
  model loading** (longer to load on a slower machine). Measured with the image already built, on a 10-core Apple-silicon Docker VM (CPU only):
  **66 seconds** from `docker compose up` to the end of the run (about 5 seconds of loading), 10 decisions; the agent processed 145 of
  the 607 frames of the 10 fps grid and skipped the rest, and its newest processed frame was a median of 1.1 seconds behind the
  playhead (up to 7.6 seconds while the d1 ranker was thinking). The first decision is on the page about 12 seconds into the clip. On a slower or emulated
  CPU (not measured) the run still ends after about 60 seconds, but the agent processes fewer frames, decides less often and lags
  further behind. `DEMO_SECONDS=90` plays the whole clip, `DEMO_SECONDS=20` is a quick smoke run.
- **When the run ends** the container prints the run summary (frames, decisions, a per-stage latency table) and the dashboard
  keeps serving the finished run until you press Ctrl-C. Runs are kept in the volume `clashbot-runs`
  (`docker compose down -v` deletes them).

## What you will see

The bot "plays" the left side of a recorded pro game (the deck of the creator in the video), exactly as it would play live:

1. **Perception** reads each video frame: KataCR (YOLOv8) finds the units, OCR and template readers read the elixir bar, the
   four-card hand, the clock and the tower health, and a Kalman tracker follows the units.
2. **Live state and policy**: one clean game state per frame, then the act policy (rules plus a LightGBM model) decides *when* to
   act or to save elixir.
3. **Candidates**: a tactics engine (lane simulation, knowledge-graph pro answers) proposes moves; kNN / BM25 retrieval over
   10,650 pro game states (reciprocal rank fusion; the 10,923 of the full database minus the clip's own video) and knowledge-graph counters add more; four options go to the ranker.
4. **Ranker**: the fine-tuned d1 language model (600M parameters) gives a probability per option and picks one.
5. **Executor** in dry run: it logs the two taps it *would* make. Nothing is sent.

On the page, pick a decision on the timeline to see its annotated screenshot, every candidate with its probability and its
sources, the retrieved pro states (click one for its context), the knowledge-graph neighbourhood, the facts given to the model and
the reason the bot waited whenever it held back.

At the top of the page the **Recording** panel plays the clip itself, in step with the agent. The agent watches the recording the way it
would watch a live stream (`--realtime`): it never gets a frame before its time and skips the frames it is too slow for, so the run takes
about as long as the part of the clip it plays. Each decision is drawn on the video the moment the agent makes it: a ring at the tile it
would tap with the card's name, a fading trail of its recent taps, a banner with the mode, the card, the lane and the reason, and a strip
with the reason it is holding back whenever it waits. A badge says "LIVE: the agent is watching this recording now" and shows the agent's
lag (how far behind the playhead its newest processed frame is). When the run is over the same panel replays it with normal video controls,
and clicking a decision in the timeline seeks the video there. For a run without a video (the prior live games below) the panel is a
screenshot player: play, pause and 1x to 4x speed through the decision screenshots in game time.

The retrieved pro states come from **other** games: the clip's own source video (all four of its games, 273 states) is removed
from the retrieval pool in this package, so the bot cannot look up the pro's own play from the same match. (The aggregate
knowledge-graph statistics, such as how often pros answered a card, are still mined from all 197 games.)

## Browse every prior game

The image also carries **every game the agent has played so far**, so you can open them in the same dashboard. The run you just
started is the one shown; the run picker at the top of the page lists the others, grouped by folder (41 runs, about 170 MB):

- `poc/` and `demo/`: the **live games** played through iPhone Mirroring on a Mac against Training Camp: 15 sessions between
  2026-11-03 and 2026-11-09, for example "Live 2026-11-07 01:34 (first win)" and "Live trial 2026-11-07 (10 battles)". Sessions with
  no decision logged (empty or aborted starts) are left out.
- `vs_pro/`: the 24 runs of the bot on recorded pro matches ("Bot vs pro: 6qYQNJ1Uaeg m0.3"), each with the pro's own plays next to
  the bot's decisions.
- `smoke/`: two dry runs on the test clip (the same clip this demo plays).

Each of these keeps what the page shows: every decision (candidates, probabilities, retrieved pro states, reasons), the hold
reasons, the run summary and the annotated screenshots. The screenshots are re-encoded at 60% of their size (JPEG quality 70) to keep the
package small; the decision data is exactly what the bot wrote. The per-frame logs, videos and review sheets are not included.
They are read-only and separate from the demo run: starting or removing the demo does not touch them.

On a Mac, `scripts/setup_mac.sh` (see below) copies them into `runs/`, and `python -m clashbot.viz --open` lists them there too.

## If the d1 download fails

The build does not fail when the Hugging Face download does (no internet access to huggingface.co, a proxy, ...). The image then
has no d1, and the container prints the banner *"d1 weights not present - the tactics engine is choosing moves"*: the tactics
engine picks the moves and the rest of the stack is unchanged. To fetch the weights later, rebuild just that layer:

```bash
D1_RETRY=1 docker compose up --build        # any value that differs from the last build
```

## Other commands and settings

```bash
docker compose run --rm clashbot tests     # a small CPU test subset (about 500 tests, under a minute)
docker compose run --rm clashbot bash      # a shell inside the image
```

Settings, in a `.env` file next to `docker-compose.yml` or in the shell: `DEMO_SECONDS` (video seconds to process, default 60 of the 90 s
clip), `RANKER=tactics` (skip d1), `KEEP_SERVING=0` (exit when the clip ends instead of serving the dashboard),
`D1_RETRY` (above). If port 8765 is taken, change the left number of `"8765:8765"` in `docker-compose.yml`.

## What is in this package

- `clashbot/`, `cr_perception/`, `tools/`, `knowledge_base/` (card tables and art), `configs/`: the code and the reference data
  the stack reads. `tests/` is the subset that `tests` runs.
- `docker/`, `docker-compose.yml`: the image (python:3.12-slim, CPU PyTorch, pinned packages) and the entrypoint.
- `artifacts/`: the KataCR detector weights and the one KataCR source file they need to load (both MIT, from
  github.com/wty-yy/katacr, commit 36ceb9f; licence in `artifacts/katacr/`), the pro-state database and the knowledge graph the
  retrieval reads, the LightGBM act model, the calibration and the clip. The d1 weights are not in the zip (LFM Open License v1.0,
  included in the Hugging Face repo; the image downloads them while it is built).
- `prior_runs/`: the packed prior games (`prior_runs/<group>/<run>/`, see "Browse every prior game"); the image keeps them in
  `/app/prior_runs` and the dashboard lists them.
- `scripts/setup_mac.sh`, `scripts/demo.sh`, `requirements-mac.txt`: the native Mac path below.
- The clip is 90 seconds (1:00 to 2:30) of a public YouTube video by the creator ryleycr1 (video id Z8-4VhLjrGU), re-encoded; it is
  included only so this demo can run, for course grading.

The full repository (all evaluations, the report) is the other submission file.

## Live play on a Mac (optional)

The real agent plays battles on a Mac through iPhone Mirroring (use Training Camp or friendly battles). Docker cannot reach the macOS window server, so
this runs natively, outside the container. Skip this section if you only want to see the stack work.

**You need:** a Mac (macOS 15 or newer; Apple silicon recommended), an iPhone paired with iPhone Mirroring and showing Clash Royale,
Python 3.12 (`brew install python@3.12`), the Xcode command line tools (`xcode-select --install`), about 10 GB of disk and an
internet connection for the one-time setup.

```bash
cd clashbot_docker                      # the unzipped folder
scripts/setup_mac.sh                    # once: a .venv here, the packages, d1 from Hugging Face, the data in place, an environment check
source .venv/bin/activate
export CR_SCRATCH="$PWD/cr_scratch"     # where setup put the KataCR weights and source
scripts/demo.sh                         # asks for your 8 cards, then watches only: the bot decides and logs, nothing is tapped
CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live --dashboard    # taps ON: plays every battle it sees
```

- `setup_mac.sh` is safe to re-run. It leaves an existing `.venv` alone unless you pass `--reinstall`, never overwrites files that
  are already in place, and does not touch anything outside this folder. It ends by printing the live command above.
- **Permissions:** the terminal app you launch it from needs **Screen Recording** (to read the mirrored window) and
  **Accessibility** (to post the taps) in System Settings, Privacy & Security. Restart the terminal after granting them.
- **Any battle (D-055).** The launcher asks for your 8 cards, plays every battle it sees once taps are ON and only records the
  opponent banner it reads in `battles.jsonl`; it does not check the opponent. Start only Training Camp or friendly battles: playing
  ranked matches with a bot violates Supercell's terms of service and risks a ban.
- **Double guard (D-031).** Taps need both `--live` on the command line **and** `CLASHBOT_ALLOW_LIVE=1` in the environment, checked when
  the executor is built and again before every tap. Without both the launcher says so and only watches.
- **The calibration** (`runs/poc/mirror_calib.json`, copied from `configs/mirror_calib.example.json`) holds the geometry of the
  mirrored window. If your window looks different (another iPhone model or scale), make your own from a clean in-match frame:
  `python tools/calibrate.py --help`.
- Everything the session records (decisions, holds, screenshots of every tap, one row per battle) is in `runs/demo/<timestamp>/`;
  `--dashboard` shows it in the browser while it plays.

A dry run on the recorded clip, natively, to check the setup without a phone:

```bash
python -m clashbot.live --dry-run --source video --video artifacts/clip.mp4 --calib runs/poc/mirror_calib.json \
  --deck berserker,dart-goblin,electro-spirit,elite-barbarians,fireball,goblin-gang,suspicious-bush,the-log \
  --ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac --policy fast --act-model db/models/act-gbdt \
  --record-shots --seconds 20 --out runs/smoke/native_check
python -m clashbot.viz --run runs/smoke/native_check --open
```
