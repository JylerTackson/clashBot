# Quickstart: see it work in about 10 minutes, without the iPhone

For a grader on the author's Mac, where the weights and the built data are in place. Run from the repository root with the environment
from the README ("Install") active. Every step is read-only: nothing here sends input to any app or device.

```bash
cd /Users/tylerjackson/Projects/clashbot/clashBot
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1
```

Times were measured on this Mac (Apple M5, 32 GB) on 2026-11-07, while other jobs were running on it (load average 3 to 6, the wall times below are upper bounds; the suite took
386 s wall in one run and 154 s in a quieter one). About 6 minutes in total on a quiet machine.

| # | Step | Command | Time |
|---|---|---|---|
| 1 | environment check | `python scripts/check_env.py` | 7 s |
| 2 | test suite | `python -m pytest -q --deselect tests/test_perception_latency.py` | 154 s (386 s under load) |
| 3 | one CPU evaluation | `python tools/eval/eval_tactics.py --out /tmp/clashbot_tactics.json` | 2 s |
| 4 | 60 s video dry run, live configuration | the command below | 163 s (60 s of video; model loading included) |
| 5 | open the dashboard on it | `python -m clashbot.viz --run runs/smoke/quickstart --open` | seconds |

## 1. Environment check

`python scripts/check_env.py` prints one line per requirement (packages and versions, MPS, the built data, KataCR, the calibration
JSON, the video, the model folders) and which parts can run on this machine. Expected on this Mac: every line `ok`, and
`result: required packages are installed`.

## 2. Tests

`python -m pytest -q --deselect tests/test_perception_latency.py`. Expected: `1429 passed, 8 skipped, 8 deselected` (tests skip when an opt-in real-model switch, `CLASHBOT_SLOW_TESTS=1`, or an input file is missing).
`test_perception_latency.py` measures wall-clock latency and is flaky next to other jobs, so it is left out.

## 3. One CPU evaluation

`python tools/eval/eval_tactics.py --out /tmp/clashbot_tactics.json` replays the tactics engine (lane simulation, knowledge-graph pro
answers) over the validation and test games and compares its top moves with what the pro played. It prints a table and writes
`/tmp/clashbot_tactics.json` and `.md`. Expected: the 4-way tactics ranker is at chance level against the pro (about 26% vs 25%,
`docs/data/tactics_eval_v1.md`); the engine's value is downstream, as context for the fine-tuned d1.

## 4. A 60 second video dry run with the live configuration

The same stack as the live runs (perception on the Metal GPU, live state, act policy, tactics engine, retrieval + KG candidates,
fine-tuned d1 ranker), fed from a recorded pro game instead of the iPhone window, with the executor in dry-run mode (it logs the taps it
would make and sends nothing):

```bash
python -m clashbot.live --dry-run --source video --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60 --seconds 60 \
  --calib runs/poc/mirror_calib.json --deck electro-dragon,musketeer,tesla,fireball,knight,zap,ram-rider,mini-p-e-k-k-a \
  --hero-cards musketeer --ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac \
  --policy fast --act-model db/models/act-gbdt --device mps --record-shots --out runs/smoke/quickstart
```

Expected: it prints `frames=1799 decisions=21 ... status={'dry_run': 21}`, a per-stage latency table (total p50 about 257 ms and p95 about
397 ms; perception 123 ms and the d1 ranker 96 ms at p50 are the largest stages; the 300 ms p95 target is not met) and a `shots:` line with
`errors: 0`. Files in `runs/smoke/quickstart/`: `decisions.jsonl` (one row per decision: candidates with sources and d1 probabilities, the
retrieved pro states, KG facts, the tactics lines, timings), `holds.jsonl` (why the bot waited), `frames.jsonl`, `summary.json`,
`config.json`, and `shots/` (an annotated screenshot per decision).

## 5. The dashboard on that run

`python -m clashbot.viz --run runs/smoke/quickstart --open` serves a local page (127.0.0.1:8765) and opens it: pick a decision on the
timeline to see the screenshot, every candidate with its d1 probability and its sources (kNN, KG, tactics), the retrieved pro states
(click one for its context), the knowledge-graph neighbourhood and the facts given to the model. Ctrl-C stops the server.

## Without the weights and the built data (a plain unzipped checkout)

What works from the zip alone, measured on an unzipped `git archive` with the README's install done (KataCR files in `~/cr_scratch`, a
recorded video in `data/videos/`, nothing else). Rebuild the data first (README, "Rebuild the data"): about 35 s in total.

```bash
python scripts/check_env.py                     # 9 s: the data / weights lines say what is missing
python -m pytest -q --deselect tests/test_perception_latency.py     # 2.3 min: 1419 passed, 15 skipped, 3 failed (README: known limits)
python tools/eval/eval_tactics.py --out /tmp/clashbot_tactics.json  # 2 s, no kg2 needed
python -m clashbot.loop --replay LGb0mz8Sb8w-m0.0 --limit 15        # 1 s, about 29 ms per decision
python -m clashbot.live --dry-run --source video --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60 --seconds 30 \
  --calib configs/mirror_calib.example.json --deck electro-dragon,musketeer,tesla,fireball,knight,zap,ram-rider,mini-p-e-k-k-a \
  --hero-cards musketeer --ranker tactics --policy reactive --device mps --record-shots --out runs/smoke/quickstart_light
```

The last command is step 4 without the d1 weights, act-gbdt and kg2 (the engine picks the move, the rule policy decides when; it prints
`kg2=off`): 82 s wall for 30 s of video, 4 decisions, total p50 205 ms and p95 261 ms. Without KataCR, a video and the calibration there is
no perception, so the video runs and the demo need them; the tests, the data rebuild and the CPU evals do not.
