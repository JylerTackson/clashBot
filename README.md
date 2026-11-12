# clashBot

A Clash Royale playing agent for Apple-silicon Macs. It reads the game from an iPhone Mirroring window (or a recorded video), decides
what to play with retrieval over 10,923 pro game states, a knowledge graph, a tactics engine and a fine-tuned 600M-parameter LLM, and
taps through iPhone Mirroring. With live taps enabled it plays whatever battle is on screen, so start only Training Camp or friendly
battles (see "Data provenance and terms of service"). Course project; the agent, the evaluations and the demo tooling are all in this
repository.

```
iPhone Mirroring window  (or a recorded video)
  -> cr_perception/        screen gate, HUD readers (elixir, hand, clock, tower HP), KataCR YOLOv8 units, OCR, Kalman tracker
  -> clashbot/live_state   one clean game state, in the format of the training data
  -> clashbot/policy       WHEN to act: rules + a LightGBM "act now?" model (act_model.py), hold and save elixir otherwise
  -> clashbot/tactics      WHAT could work: lane simulation (unit_stats.py), pro answers from the knowledge graph (kg2.py)
  -> clashbot/candidates   4 options: engine moves + kNN/BM25 retrieval over pro states (RRF) + KG counters
  -> clashbot/ranker       d1-omni-600M, fine-tuned (db/models/d1-ft-joint-tac), picks 1 of the 4
  -> clashbot/executor     two taps via iPhone Mirroring, behind a double guard (D-031); or a dry-run log
every decision is logged (decisions.jsonl, screenshots) and shown in the dashboard (clashbot/viz)
```

Fast path for a grader without the iPhone or the model weights: [docs/QUICKSTART.md](docs/QUICKSTART.md) (environment check, tests, one eval,
a 60 s video dry run, the dashboard on it; about 10 minutes on an M-series Mac).

## Run with Docker (any OS)

The Docker submission (`dist/clashbot_docker_submission.zip`, built by `scripts/make_docker_submission.sh`) needs only Docker on Linux,
Windows or macOS, Intel or ARM: `unzip`, `cd clashbot_docker`, `docker compose up --build`, then open http://localhost:8765. It builds the
image for your CPU (PyTorch CPU wheels, the d1 weights downloaded from Hugging Face at build time), runs the whole live stack in dry run on a
bundled clip of a pro match and shows every decision in the dashboard. Nothing is tapped. Details, timings and settings:
[docs/DOCKER.md](docs/DOCKER.md) (it is the README of that zip).

## Live play on a Mac

Playing for real (through iPhone Mirroring; Training Camp or friendly battles) runs natively, because Docker cannot reach the macOS window server. On a Mac
with iPhone Mirroring and Clash Royale, run [scripts/setup_mac.sh](scripts/setup_mac.sh) once from the unzipped Docker package (it makes a
`.venv`, downloads d1, puts the data in place, runs the environment check and prints the live command and the two permissions it needs).
Taps still need `--live` and `CLASHBOT_ALLOW_LIVE=1` (D-031); with both set the launcher plays every battle it sees (D-055); see the
section "Live play on a Mac (optional)" of [docs/DOCKER.md](docs/DOCKER.md) and "Run the demo" below.

## What is in the zip, and what is not

The zip is `git archive` of the repository: code, tests, docs, the card knowledge base (with the card art the perception matches
against) and the 197 extracted pro games. It does **not** contain model weights (d1 fine-tune 2.35 GB, KataCR 2 x 88 MB), built data
(`db/`, `dataset/`), the 33 source videos (4 GB), or run logs; those stay on the author's Mac, where the live trials and the demo are run.
Parts that need them are marked below. `python scripts/check_env.py` prints what is available on a machine and what can run.

## Course map

Built and evaluated items only. Lecture block, component, file, and the file that holds its evaluation (`docs/data/BENCHMARKS.md` indexes
every result file with the command that produced it).

| Lecture block | Component | Code | Evaluation |
|---|---|---|---|
| Retrieval (8-10) | BM25 (SQLite FTS5) + 644-d structured kNN + reciprocal rank fusion; HNSW vs exact; pre/post filtering | `clashbot/retrieval.py`, `features.py`, `db.py` | `docs/data/retrieval_ablation_v1.md` (`tools/eval/eval_retrieval.py`) |
| Retrieval | bge-small text-embedding channel (measured, not in the runtime: no gain) | `clashbot/text_embed.py` | same file |
| RAG / recommendation (11-14) | Two-stage serving: candidate generation, then a setwise LLM ranker | `clashbot/candidates.py`, `ranker.py` | `docs/data/ranker_baselines_v1.json`, `d1_finetune_joint_tac_eval_v1.json` |
| RAG / recommendation | Hard negatives (2x2 card x placement grid, leak checks), listwise fine-tune of d1-omni-600M (zero-shot 0.258 -> 0.540 val, 4-way, random 0.25) | `tools/data/build_ranker_dataset.py`, `build_d1_dataset.py`, `tools/train/train_d1.py` | `docs/data/d1_zero_shot_v1.json`, `d1_finetune_*_v1.json`, `ranker_stats_v1.json` |
| RAG / recommendation | Tuned simple baselines vs the LLM (popularity, placement prior, logreg, LightGBM LambdaRank, DCN-V2) | `tools/eval/eval_ranker_baselines.py`, `feature_ranker.py` | `docs/data/feature_ranker_v1.md`, `ranker_baselines_v1.json` |
| RAG / recommendation | Sequential recommendation: opponent next card (copy frequency vs SASRec) | `tools/eval/next_card.py` | `docs/data/next_card_v1.md` |
| RAG / recommendation | Adaptive "when to act" trigger: LightGBM act model, d1 act head, rule arbiter | `clashbot/act_model.py`, `policy.py`, `arbiter.py`, `tools/train/train_act_gbdt.py` | `docs/data/act_gbdt_v1.md`, `act_head_v1.md`, `arbiter_v1.md` |
| RAG / recommendation | Contamination-safe splits (by video), data cleaning and flags | `tools/data/build_clean_dataset.py` | `docs/data/CLEANING_REPORT_v1.md` |
| Knowledge graphs (5-7) | KG v1 (wiki + mined, 590 nodes / 12,030 edges) and v2 (840 / 27,112, pro answers mined from games) | `clashbot/kg.py`, `kg2.py`, `tools/data/build_kg.py`, `build_kg_v2.py`, `mine_pro_answers.py` | `docs/data/kge_link_prediction_v1.md` |
| Knowledge graphs | Link prediction with TransE / DistMult / ComplEx / RotatE, equal tuning, filtered ranks | `tools/eval/kge_link_prediction.py` | `docs/data/kge_link_prediction_v1.md` |
| Knowledge graphs | Neuro-symbolic layer: legal-move constraints, tactics engine with lane simulation, KG facts in the LLM context | `clashbot/game_rules.py`, `tactics.py`, `unit_stats.py`, `card_roles.py` | `docs/data/tactics_eval_v1.md`, `tactics_tuning_v1.md` |
| Knowledge graphs | Opponent-deck model (leave-one-game-out) | `clashbot/opp_model.py` | `tools/eval/eval_opp_model.py` |
| Agents, memory, editing (15-17) | Guarded executor, separate verification (screenshots of every tap, landing rate) | `clashbot/executor.py`, `recorder_shots.py` | `tools/eval/review_shots.py` |
| Agents, memory, editing | Typed read-only tools, Qwen planner / coach (optional, off in the live default) | `clashbot/tools.py`, `planner.py` | `docs/data/planner_bench_v1.json` |
| Agents, memory, editing | Versioned fact memory and knowledge editing (synthetic patches: reliability, locality, portability) | `clashbot/memory.py` | `docs/data/edit_eval_v1.md` |
| Agents, memory, editing | Trajectory logging, decision dashboard, end-of-battle frames | `clashbot/live.py`, `demo.py`, `viz/` | `runs/<run>/summary.json` |
| Perception | Screen reading and detection from the live window | `cr_perception/` | `docs/data/live_state_parity_v1.json`, `tools/eval/eval_eta.py` |

## Install

macOS on Apple silicon, Python 3.12 (the pins were verified on 3.12.8; PyObjC, MLX and MPS are Mac-only, everything else also runs on CPU).

```bash
xcode-select --install                 # Apple clang: hnswlib builds from source
brew install libomp                    # OpenMP for LightGBM
python3.12 -m venv ../.venv            # scripts/demo.sh looks for ../.venv/bin/python, else python3 (activate your own venv)
source ../.venv/bin/activate
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -r requirements-mac.txt
pip install imageio-ffmpeg             # optional: only tools/demo_clip.py, when ffmpeg is not installed
```

The commands below say `python`; use the activated venv's interpreter. Offline runs: `export HF_HUB_OFFLINE=1`.

**KataCR (unit detector), outside the repo.** The checkpoints unpickle against KataCR's source (MIT, github.com/wty-yy/katacr).
`bash tools/worker_setup.sh` clones it to `~/cr_scratch/katacr`, downloads the two public weight files `katacr_d1.pt`, `katacr_d2.pt` to
`~/cr_scratch/` and finishes with the test suite (it also pip-installs unpinned packages: skip it if the venv above is already set up and
do the clone and the two downloads by hand). Env overrides: `CR_SCRATCH` (default `~/cr_scratch`), `CLASHBOT_KATACR_ROOT`,
`CLASHBOT_KATACR_WEIGHTS` (the two files joined with `:`), `CLASHBOT_TORCH_DEVICE` (`cpu` or `mps`).

**Models and built data under the repo (gitignored, not in the zip).**

| Path | What | Needed for |
|---|---|---|
| `db/models/d1-ft-joint-tac/` | fine-tuned d1-omni-600M (`model.safetensors` + its own `modeling_d1.py`, config, tokenizer; loads offline) | `--ranker d1` (the live ranker) |
| `db/models/act-gbdt/` | LightGBM act model (`meta.json`, `model_h{1,2,3}.json`) | `--policy fast` (the live policy) |
| `db/kg2.sqlite`, `dataset/v1/kg2/` | knowledge graph v2 (prebuilt: mining needs the 33 videos) | engine, policy, `KG:` lines |
| `db/clashbot.sqlite`, `db/state_features.*`, `dataset/v1/` | states, KG v1, documents, FTS5, feature vectors (rebuilt from `knowledge_base/`, below) | retrieval, evals, tests that need data |
| `configs/mirror_calib.example.json` (tracked) | window geometry, no secrets: pass `--calib configs/mirror_calib.example.json` to `clashbot.live`. The demo launcher reads it from `runs/poc/mirror_calib.json`: `mkdir -p runs/poc && cp configs/mirror_calib.example.json runs/poc/mirror_calib.json` (after running the tests, see below) | every `clashbot.live` / demo run |
| `data/videos/<id>/<id>.mp4` | a recorded video for dry runs (the author's are not redistributed; `scripts/download_videos.py` re-downloads the creator's public videos with yt-dlp) | `--source video` |

**Rebuild the data from the tracked `knowledge_base/`** (about 1 minute; writes `dataset/` and `db/` in the checkout):

```bash
python tools/data/build_clean_dataset.py && python tools/data/build_kg.py && python tools/data/build_documents.py
python tools/db/seed.py --hnsw && python tools/data/build_ranker_dataset.py && python tools/eval/eval_ranker_baselines.py
python -m clashbot.loop --replay LGb0mz8Sb8w-m0.0 --limit 15       # fast loop on a recorded game, no model: ~30 ms per decision
```

`db/kg2.sqlite`, `db/models/act-gbdt` and `dataset/v1/act_dense` come from dense perception over the 33 videos (about 17 hours) and are not
rebuilt by this; they are the prebuilt part. Without them the engine runs without pro answers and `--policy reactive` replaces `fast`.

Known limits of a plain unzipped checkout (checked on one, after the rebuild above and without a `.git` folder): the suite gives 1419 passed,
15 skipped, 3 failed. Three tests assume the author's setup: `tests/test_clean_dataset.py::test_data_dir_not_ignored` runs a git command
(needs a git repository), `tests/test_coach.py::test_build_d1_tac_plans_adds_the_latest_plan_at_or_before_each_line` needs the prebuilt
`db/kg2.sqlite`, and `tests/test_opp_model.py::test_real_live_runs_give_a_few_decks` expects recorded runs whenever a `runs/poc` folder exists
(so copy the calibration there only after the tests). None of the three touches the agent's behaviour.

## Run the demo (iPhone Mirroring)

Needs macOS 15+ with iPhone Mirroring and a paired iPhone showing Clash Royale, Screen Recording and Accessibility permission for the
terminal, the calibration JSON above, KataCR, `db/` and the weights. From the repository root:

```bash
CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live --dashboard
```

The launcher asks for the 8 cards (names are matched loosely: "mini pekka", "e dragon"), which are evolutions and which heroes, loads every
model, prints `READY`, then watches the mirrored window until Ctrl-C. It plays every battle it sees and only records the opponent
banner it reads (`battles.jsonl`); it does not check who the opponent is (D-055), so start only Training Camp or friendly battles:
playing ranked matches with a bot violates Supercell's terms of service and risks a ban. Taps need both `--live` and
`CLASHBOT_ALLOW_LIVE=1` (D-031); without them it only watches and says so. `--dashboard` opens the browser dashboard on the session. The session lives in
`runs/demo/<timestamp>/`: `decisions.jsonl`, `holds.jsonl`, `frames.jsonl`, `battles.jsonl` (one row per battle), `summary.json`,
`shots/`, and `battles/<n>_end_{0,1,2}.jpg`, the last three frames of each battle (the result screen), listed in the battle's `end_frames`.

No phone: the same launcher runs on a recorded video, watch only (nothing is tapped):

```bash
scripts/demo.sh --deck "electro dragon, musketeer, tesla, fireball, knight, zap, ram rider, mini pekka" --evo-cards "tesla, electro dragon" \
  --hero-cards musketeer --yes --source video --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60 --seconds 60
```

The plain live loop (no launcher) is `python -m clashbot.live --help`; its dry-run video form is step 4 of the quickstart.

## Replay a run in the dashboard

```bash
python -m clashbot.viz --run runs/smoke/quickstart --open       # any run folder (replay); the page polls every second
python -m clashbot.viz --follow runs/demo --open                # follow the newest run under runs/demo while the bot plays
```

A local page (127.0.0.1:8765, no external assets): the decision with each candidate's d1 probability and sources, the retrieved pro
states, the knowledge-graph neighbourhood, the screenshots, and a timeline of decisions and holds.

## Run the tests

```bash
python -m pytest -q --deselect tests/test_perception_latency.py
```

About 1,430 tests (1429 passed, 8 skipped on the author's Mac), 2.5-3 minutes of test time. Tests that need the built data, `db/`, `runs/` or the KataCR files skip when they are missing, so a
clean checkout runs fewer of them; `test_perception_latency.py` is timing-sensitive and left out. Nothing in the suite loads a real model
unless `CLASHBOT_SLOW_TESTS=1`, and nothing ever sends input.

## Run the evals

All evals read `dataset/v1` (and `db/`), run on CPU, and take `--out PATH` (default `dataset/v1/eval/`). Results, numbers and the commit
they were produced at: [docs/data/BENCHMARKS.md](docs/data/BENCHMARKS.md).

```bash
python tools/eval/eval_tactics.py --out /tmp/tactics.json            # tactics engine vs the pros' plays             (~2 s)
python tools/eval/eval_ranker_baselines.py                           # recall and 4-way baselines, writes dataset/v1/ranker/
python tools/eval/eval_retrieval.py --embedder hash --cache-dir /tmp/emb --out /tmp/retrieval.json   # retrieval ablation (~2 min); the bge channel needs the HF model
python tools/eval/feature_ranker.py --quick --out /tmp/fr.json       # logreg / LightGBM / DCN-V2 rankers          (~20 s)
python tools/eval/next_card.py --quick --out /tmp/next_card.json     # opponent next card: copy frequency vs SASRec (seconds)
python tools/eval/eval_act_policy.py --out /tmp/act_policy.json      # act / hold policies
python tools/eval/eval_opp_model.py --out /tmp/opp_model.json        # opponent-deck model, leave-one-game-out
python tools/eval/edit_eval.py --out /tmp/edit_eval.json             # knowledge editing, synthetic patches        (about 70 s on a quiet machine)
python tools/eval/kge_link_prediction.py --out /tmp/kge.json         # KG embeddings (about 1 hour)
```

With models or video (author's Mac): `tools/eval/eval_d1_zero_shot.py`, `eval_act_head.py` (d1 weights), `eval_arbiter.py` (reads the d1 eval
output in `dataset/v1/eval/`), `eval_eta.py`, `live_state_parity.py` (KataCR + a video), `review_shots.py RUN_DIR` (a recorded run),
`bench_planner.py` (Qwen). Training: `tools/train/train_d1.py`,
`train_act_gbdt.py`.

## Repository map

| Path | What |
|---|---|
| `clashbot/` | the agent: live loop (`live.py`), demo launcher (`demo.py`), policy, tactics, candidates, retrieval, KG, ranker, executor, dashboard (`viz/`) |
| `cr_perception/` | read-only perception: screen gate, HUD readers, KataCR detector, OCR, tracker, trainer-banner reader. Sends no input |
| `tools/` | `data/` builders, `db/seed.py`, `eval/` evaluations, `train/` fine-tuning, perception tools (`calibrate.py`, `run_video.py`, `batch_process.py`) |
| `scripts/` | `demo.sh`, `check_env.py`, `make_submission.sh`, `check_submission.sh`, `download_videos.py`, `kb/` (the wiki / deck scrapers) |
| `knowledge_base/` | cards, evolutions, heroes (wiki), decks and archetypes, and `states/` + `matches/`: 197 pro games extracted from video |
| `tests/` | the pytest suite |
| `configs/` | `mirror_calib.example.json`, the window geometry (no secrets) |
| `docs/` | `decisions/DECISIONS.md` (every design decision), `mac/STATUS.md` (dated log), `data/` (result files), `plans/` (specs), `research/` (audits), `design/`, `report/refs.bib` |

## Data provenance and terms of service

- Card, evolution and hero data come from the Clash Royale Fandom wiki (scripts in `scripts/kb/`, through its MediaWiki API); the card art
  is Supercell's and is included only so the perception can match hand cards. The deck list comes from one page saved from a browser
  session of royaleapi.com (its robots.txt disallows automated agents, so it was not scraped). Clash Royale is a trademark of Supercell;
  this project is not affiliated with or endorsed by Supercell.
- The 197 games (10,923 states) were extracted by this repository's own perception from 33 public YouTube videos by one creator
  (@ryleycr1, downloaded with yt-dlp for research). The videos are not redistributed; the extracted states and the creator's insights are
  in `knowledge_base/states` and `knowledge_base/matches`. Splits are by video, so no game appears in two splits.
- Supercell's terms prohibit bots in normal play. The launcher does not check the opponent (D-055): with `--live` and
  `CLASHBOT_ALLOW_LIVE=1` together (D-031) it plays any battle on screen, so run it only in Training Camp or friendly battles; ranked
  play with a bot violates the terms and risks a ban. Nothing in the test suite or the evals sends input. Everything else (evals, dry
  runs, dashboards) is read-only.
- No credentials are used or stored. `data/cookies.txt` (an optional yt-dlp export) is gitignored; `scripts/check_submission.sh` scans the zip.
