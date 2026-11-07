# Packaging and benchmarks audit (read-only)

Audit date 2026-11-07. Repo `/Users/tylerjackson/Projects/clashbot/clashBot`, HEAD `015368f` (docs only; last code commit `e578ef1` / merge `9234be4`, 2026-11-07 02:54-02:56), working tree clean (the only untracked file is another agent's `docs/research/course_mapping_audit.md`). No training, model loading, installs or test runs; only read-only inspection (file listings, `du`, `git log`, small python scripts that parse JSON/AST or read video metadata). Sizes come from `du`/`git ls-files`, numbers from the files named in each row. "est." marks my own estimates (not recorded in any log).

Grading weights used: difficulty 30 / runnability 30 / presentation + demo + benchmarking 20 / course learning 20.

---

## Part 1 - runnability and packaging

### 1.0 Top blockers (ordered)

| # | Blocker | Why it matters | Fix (effort) |
|---|---|---|---|
| 1 | Nothing runnable is in git except tests/evals-after-rebuild: no model weights, no calibration JSON, no video, no `db/`, no `dataset/` | A clean clone cannot run `clashbot.live` at all: `--calib` is required and the only calib files live in gitignored `runs/` (`runs/poc/mirror_calib.json`, 2.8 KB) | Track the calib JSON under `clashbot/resources/calib/` (5 min); provide weights + a short clip (see 1.7) |
| 2 | Weights: fine-tuned d1 (`db/models/d1-ft-joint-tac`, 2.35 GB fp32) and KataCR (`katacr_d1.pt`, `katacr_d2.pt`, 2 x 87.8 MB, Google Drive) plus the KataCR clone (needed to unpickle) | 2.4 GB + 176 MB cannot ride in a course zip; KataCR download relies on two Google Drive ids in `tools/worker_setup.sh:33-34` | Host d1 as fp16 (about 1.2 GB) on HF Hub / Drive; vendor KataCR's 7.5 KB `katacr/yolov8/custom_model.py` (MIT) so no git clone is needed (1-2 h incl. test) |
| 3 | Mac-only, Python 3.12-only environment | `requirements-mac.txt` pins `mlx*`, `pyobjc-framework-Quartz`, torch 2.15, numpy 2.5 and was verified on 3.12.8 only; the shell `python3` on this machine is 3.13.5; `requirements-perception.txt` is unpinned and lacks lightgbm/transformers/hnswlib/sentence-transformers/pykeen, so it cannot run the full test suite | State "macOS + Apple silicon + python3.12" in the README, or split into a portable `requirements.txt` plus a Mac extras file (30 min, untested on Linux) |
| 4 | `hnswlib` has no wheels (needs `CC=/usr/bin/clang CXX=/usr/bin/clang++`) and LightGBM needs `brew install libomp` | A plain `pip install -r requirements-mac.txt` fails on a stock Mac without these two steps; README does not say so (only comments in the requirements file) | Put both steps in the quickstart. hnswlib is optional at runtime (only `seed.py --hnsw`, `eval_retrieval`, `test_retrieval`, `test_eval_retrieval`); the live loop uses exact search (`StateIndex(use_hnsw=False)`) |
| 5 | Two built artefacts cannot be rebuilt from tracked files: `db/kg2.sqlite` / `dataset/v1/kg2/*` (mining reads `dataset/v1/act_dense/states_<split>.jsonl`, which comes from 33 videos x dense perception, about 17 h) and `db/models/act-gbdt` (trained on `act_dense`) | `--policy fast` (the live default in every recent run) and the engine's pro answers need them | Ship them prebuilt: `db/kg2.sqlite` 8.4 MB + `dataset/v1/kg2` 2.8 MB + `db/models/act-gbdt` 1.4 MB, about 2 MB zipped in total |
| 6 | README describes Phase 1 only; `docs/PHASE3_OVERVIEW.md` is dated 2026-10-28 and says "no models trained yet", "next steps proposed, not started" | A grader reads the README first and sees a wiki scraper, not the bot | Rewrite README (1-2 h); see 1.8 |
| 7 | Demo launcher (`clashbot.demo`, MAC-AN) and decision dashboard (MAC-AO) exist only as specs (HEAD `015368f`) | Live play needs iPhone Mirroring (macOS 15+ and a phone); graders cannot reproduce it, and the presentation/demo grade rides on it | Implement MAC-AN/AO or ship a recorded demo video plus the video dry run as the reproducible demo |
| 8 | Project folder contains `.claude/` (10 GB of agent worktrees), `data/` 4.0 GB, `db/` 13 GB, `runs/` 1.5 GB, `.DS_Store`, `.pytest_cache`, `.git` 75 MB | Zipping the folder instead of `git archive` produces a 30 GB file | Build the zip from `git archive` / `git ls-files` only (below) |
| 9 | Several shipped benchmark outputs are stale or live only in gitignored paths (`runs/smoke/AM/*`, `runs/smoke/AL/eta.*`, `dataset/v1/eval/opp_model.*`) | Report numbers would not match the code at HEAD | Re-run plan in Part 2, section 2.3; copy the current ones into `docs/data/` |

### 1.1 What a grader needs

| Item | Detail |
|---|---|
| Python | 3.12.8 (venv at `../.venv`). Pins verified against the installed venv: all 23 pins in `requirements-mac.txt` match, `pip check` clean. |
| OS / hardware | macOS 26.5, Apple M5, 32 GB (STATUS 2026-11-02). MPS used for KataCR and d1; MLX for the Qwen coach/planner and the optional d1 MLX backend. d1 `transformers` backend falls back to CPU (`clashbot/ranker.py:171`), KataCR to CPU (`--device cpu`, about 590 ms per detect vs about 100 ms on MPS, STATUS 2026-11-03). Slow but functional without a GPU. |
| Requirements files | `requirements-mac.txt` (pinned, 23 packages) is the real one. `requirements-perception.txt` (unpinned, 9 packages incl. `yt-dlp-ejs`) is the older cloud/Linux set. `scripts/kb/requirements.txt` (requests, beautifulsoup4, lxml) is for the Phase 1/2 scrapers only. No pyproject/setup. |
| System deps | Homebrew `libomp` (LightGBM); Xcode command-line tools Apple clang (hnswlib source build); `git` + `curl` (only if running `tools/worker_setup.sh`); `ffmpeg` is not installed here and is optional (`tools/demo_clip.py` only: falls back to `imageio_ffmpeg`, which is not in any requirements file). |
| Permissions (live only) | Screen Recording + Accessibility for the terminal, iPhone Mirroring window, `CLASHBOT_ALLOW_LIVE=1` plus `--live` (D-031). Not needed for any item below. |
| Network at run time | None once weights are local (runs use `HF_HUB_OFFLINE=1`). `d1-ft-joint-tac` ships its own `modeling_d1.py`, `configuration`, tokenizer, so no HF access is needed to load it. |
| Env vars | `CR_SCRATCH` (default `~/cr_scratch`), `CLASHBOT_KATACR_ROOT`, `CLASHBOT_KATACR_WEIGHTS`, `CLASHBOT_TORCH_DEVICE` (`cr_perception/detect.py:26-51`), `CLASHBOT_BUILDABOT_ROOT` (unused fallback detector), `CLASHBOT_SLOW_TESTS=1` (opt-in real-model tests), `CLASHBOT_ALLOW_LIVE`. |

#### Requirements vs actual imports (static scan of all tracked `.py`)

| Package | Imported by | In requirements-mac.txt? | Verdict |
|---|---|---|---|
| numpy, scipy, cv2, torch, transformers, mss, onnxruntime, rapidocr_onnxruntime, hnswlib, lightgbm, pykeen, sentence_transformers, tokenizers, mlx, mlx_lm, mlx_vlm, Quartz, pytest, torchvision | clashbot / cr_perception / tools / tests | yes | OK |
| `huggingface_hub` | `clashbot/ranker.py:97`, `tools/train/train_d1.py:739` | no (transitive of transformers) | list it explicitly |
| `imageio_ffmpeg` | `tools/demo_clip.py:144` (optional fallback) | no | optional, list as extra |
| `bs4`, `requests` | `scripts/kb/*` | no, but in `scripts/kb/requirements.txt` | OK (scraper only) |
| `pillow` | no direct import (ultralytics / mlx-vlm use it) | yes | harmless, transitive |
| `ultralytics==8.1.24` | no direct import; KataCR checkpoints unpickle against it (`custom_model.py` imports `ultralytics.nn.tasks`) | yes | required indirectly |
| `soundfile` | d1 remote code | yes | required indirectly |
| `yt-dlp` | `scripts/download_videos.py` | yes | only for re-downloading videos |
| `networkx` | mentioned in `PHASE3_OVERVIEW.md` ("later"), never imported | no | stale doc line |
| `pandas`, `scikit-learn`, `matplotlib` | not imported by repo code | no | pulled by pykeen/ultralytics |

#### External weights per mode

| Mode | Needs | Size |
|---|---|---|
| Tests, dataset/db rebuild, CPU evals (tactics, act policy, retrieval with `--embedder hash`, KGE, next card, edit eval, opp model) | nothing external | - |
| `eval_retrieval.py --embedder bge` (default) | HF `BAAI/bge-small-en-v1.5` | 383 MB (HF cache) |
| Video dry run, perception only (`--detector katacr`, `--ranker score` or `tactics`) | KataCR weights `~/cr_scratch/katacr_d1.pt`, `katacr_d2.pt` + KataCR clone `~/cr_scratch/katacr` (upstream `github.com/wty-yy/katacr`, MIT, commit `36ceb9f`) | 2 x 87.8 MB + 41 MB (only `katacr/yolov8/custom_model.py`, 7.5 KB, is used) |
| Live loop as run in every recent session (`--ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac --policy fast --act-model db/models/act-gbdt`) | the fine-tuned d1 directory + act-gbdt | 2.35 GB (fp32) + 1.4 MB |
| `--ranker d1` without `--d1-model` | HF `LiquidAI/d1-omni-600M-MLX-fp16` (zero-shot = chance level) | 1.1 GB |
| `--planner llm` / `--planner coach` | HF `mlx-community/Qwen3-4B-Instruct-2507-4bit` / `mlx-community/Qwen3-14B-4bit` | 2.1 GB / 7.8 GB |
| Retraining d1 | HF `LiquidAI/d1-omni-600M` (2.2 GB) or `LiquidAI/d1-2B` (3.6 GB), MPS with 20-28 GiB | - |
| Not used by current code but in the HF cache | `convaiinnovations/laya` (1.6 GB), `Qwen/Qwen3-Embedding-0.6B` (1.1 GB) | skip |

### 1.2 Built artefacts: who loads what, and how to rebuild

| Artefact (gitignored) | Size on disk | Loaded by | Rebuilt by (inputs) | Rebuildable from tracked files? |
|---|---|---|---|---|
| `dataset/v1/{states,games,provenance,splits}` | 64 MB states | every eval, `live` (placement prior), tests (skip when absent) | `tools/data/build_clean_dataset.py` (knowledge_base) | yes |
| `dataset/v1/kg`, `documents.jsonl` | 3.2 MB, 4.8 MB | `seed.py`, `kge`, planner | `build_kg.py`, `build_documents.py` | yes |
| `db/clashbot.sqlite`, `state_features.npy`, `.hnsw` | 128 MB, 27 MB, 28 MB | `live` (StateIndex + KnowledgeGraph, exact search), `edit_eval`, `feature_ranker` (kg counter), `eval_tactics` (KG) | `tools/db/seed.py --hnsw` | yes |
| `db/state_text_emb.npy` | 16 MB | `eval_retrieval` text channel only | built by `eval_retrieval.py` (bge) | yes |
| `dataset/v1/ranker`, `act`, `placement_prior.json` | 6.4 MB, 4.7 MB, 68 KB | `live` (`placement_prior.json`), `eval_tactics` (4-way grid), `feature_ranker`, `eval_ranker_baselines` | `build_ranker_dataset.py`, `eval_ranker_baselines.py` | yes |
| `dataset/v1/d1/{joint,factorized,act}` | 45 MB | `eval_d1_zero_shot`, `train_d1` | `build_d1_dataset.py --tokenizer db/hf/d1_tokenizer.json` (tokenizer from HF `LiquidAI/d1-omni-600M`, 4.5 MB) | yes (tokenizer download) |
| `dataset/v1/act_dense` (144 MB), `act_dense_bal` (53 MB) | | `train_act_gbdt`, `eval_act_head`, `mine_pro_answers`, `test_act_model` | `batch_process.py` over 33 videos (4.0 GB, about 17 h, 2026-11-04) -> `runs/videos` (1.2 GB) -> `build_dense_act.py` -> `sample_act_train.py` | **no** (needs the videos; ship prebuilt, about 16 MB zipped) |
| `dataset/v1/kg2/*`, `db/kg2.sqlite` | 2.8 MB, 8.4 MB | `live` (engine pro answers, opponent-deck model, `KG:` lines), `eval_tactics`, tests | `mine_pro_answers.py` -> `build_deck_cooc.py` -> `build_kg_v2.py` | **no** (mining reads `act_dense` sidecars); ship prebuilt |
| `dataset/v1/d1/joint_tac` | in `d1/` | `train_d1`, `eval_arbiter`, d1 evals | `build_d1_tac.py` (needs kg2, engine) | yes once kg2 exists (about 2 min, `next3.log`) |
| `db/models/act-gbdt` | 1.4 MB | `live --policy fast` | `train_act_gbdt.py` (needs `act_dense`) | **no**; ship prebuilt |
| `db/models/d1-ft-joint-tac` (live), `d1-ft-joint`, `d1-ft-joint-kg`, `d1-ft-act` | 2.2 GB each (12 GB total with `d1-2b-ft-joint-tac` 3.6 GB) | `live`, d1 evals | `train_d1.py` (1.2-3.7 h MPS each) | no (hours); ship weights externally |
| `data/templates` (101 learned card crops) | 34 MB | `cr_perception` CardMatcher (optional extra templates; perception also **writes** here, `perception.py:659`) | produced during perception runs | optional; base templates come from tracked `knowledge_base/*/images` |
| `knowledge_base/{cards,evolutions,heroes}/images` | 42 MB (tracked) | **required** by `CardMatcher` (hand reading) | tracked | n/a |
| `data/videos` (33 mp4, 55-245 MB each, 4.0 GB), `runs/*` | | dry-run source, dense perception | `scripts/download_videos.py` (yt-dlp, a creator's public videos) | re-download only |
| Calibration | `runs/poc/mirror_calib.json` (2.8 KB), per-video `runs/videos/<id>/match_0/calib.json` | `live --calib` (required), `eval_eta` | `tools/calibrate.py --video ... --detector katacr` | regenerable only with KataCR weights; ship the JSON |

### 1.3 Entry points that work without an iPhone

| Entry point | Command (from repo root) | Needs | Time |
|---|---|---|---|
| Test suite | `../.venv/bin/python -m pytest -q --deselect tests/test_perception_latency.py` | venv; for no skips also `dataset/v1` + `db/` + kg2 (see skips below) | 1284 passed / 7 skipped / about 2.5 min (reported); `test_perception_latency.py` is the timing-sensitive file that is deselected |
| Fast loop replay (no models) | `python -m clashbot.loop --replay LGb0mz8Sb8w-m0.0 --limit 15` | `db/`, `dataset/v1` | about 30 ms per decision (STATUS 2026-11-02) |
| Video dry run (the reproducible live demo) | `python -m clashbot.live --dry-run --source video --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --start 60 --seconds 120 --calib runs/poc/mirror_calib.json --deck electro-dragon,musketeer,tesla,fireball,knight,zap,ram-rider,mini-p-e-k-k-a --hero-cards musketeer --ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac --policy fast --act-model db/models/act-gbdt --device mps --record-shots` (exactly `runs/smoke/AM_merged/config.json`) | calib, db, dataset, kg2, act-gbdt, d1 dir, KataCR, video | about 4.5 min for 120 s of video on the M5 (log timestamps 03:00:38 -> 03:05); result 45 decisions, 0 shot errors, total p50/p95 248/402 ms |
| Same, without the 2.4 GB d1 | replace the ranker flags with `--ranker tactics` (engine's own pick) or `--ranker score` | KataCR still needed | shorter (no 85 ms rank stage) |
| Perception demo clip with overlay | `python tools/demo_clip.py --video <mp4> --out runs/demo --seconds 30` | KataCR | est. 1-2 min |
| Perception over a video | `python tools/run_video.py --video <mp4> --calib <calib> --out runs/x [--overlay]` | KataCR | 1607 frames in 237 s (`runs/parity/.../perception2/summary.json`) |
| Evals | all of `tools/eval/*` except `eval_eta`, `live_state_parity`, `bench_planner`, `eval_d1_zero_shot`, `review_shots` run on CPU from `dataset/v1` (+ `db/` for retrieval/edit/feature_ranker/tactics) | see 2.1 | seconds to 1 h (KGE) |
| `review_shots.py RUN_DIR` | contact sheets + landing rate for a recorded run | `runs/poc/*/shots` | est. 1 min |
| Rebuild everything but dense/kg2 | `build_clean_dataset.py`, `build_kg.py`, `build_documents.py`, `seed.py --hnsw`, `build_ranker_dataset.py`, `eval_ranker_baselines.py`, `build_d1_dataset.py --tokenizer ...` | tracked files + tokenizer | each seconds to about 1 min (est.; the test fixtures build most of them in temp dirs inside the 2.5 min suite) |

Tests that skip on a fresh clone (from `skipif` / `pytest.skip` in `tests/`): everything using `built_dataset` / `built_kg` fixtures (`conftest.py:22`, needs `dataset/v1/states.jsonl`), `test_runtime`/`test_eval_retrieval`/`test_memory` (need `db/clashbot.sqlite`), `test_tactics:1680`/`test_kg2:676` (need `db/kg2.sqlite`), `test_mine_answers:399`/`test_act_model:530` (need `act_dense`), `test_hero_ability:195,206` (need `runs/` shots), `test_opp_model:306` (needs `runs/poc`), `test_card_roles:221,228` (need `data/external/cr-api-data`), `test_config_paths:95` (needs `~/cr_scratch`), the `CLASHBOT_SLOW_TESTS` ones. The count of skips on a clean clone is much higher than 7; **never verified from a clean clone** (do this once before submitting).

### 1.4 Hard-coded absolute paths in tracked files

No executable code contains an absolute user path. Defaults are `Path.home()/"cr_scratch"` and env-overridable.

| File:line | Content | Severity |
|---|---|---|
| `cr_perception/detect.py:26`, `:31` | `Path.home() / "cr_scratch"` default scratch dir | none (env override) |
| `tools/worker_setup.sh:9` | `${CR_SCRATCH:-$HOME/cr_scratch}` | none |
| `tests/test_config_paths.py:35,82,95` | home-relative default, negative check for `/home/user` | none |
| `docs/data/act_head_v1.json:4` | `/Users/tylerjackson/Projects/clashbot/clashBot/dataset/v1/act_dense_bal` | cosmetic (leaks username) |
| `docs/data/act_policy_v1.json:2`, `arbiter_v1.json:2-3`, `tactics_tuning_v1.json:2`, `tactics_tuning_v1.md:3` | `/Users/tylerjackson/.../.claude/worktrees/agent-*/dataset/v1/...` | cosmetic |
| `docs/data/tactics_eval_v1.json:2` | `/Users/tylerjackson/Projects/clashbot/clashBot/dataset/v1` | cosmetic |
| `docs/plans/MAC-00_common.md:9,11,14,15` | `/Users/tylerjackson/Projects/clashbot/.venv/bin/python`, `ln -s /Users/tylerjackson/...` | doc |
| `docs/plans/MAC-A_config_paths.md:17,83,85,94,117,118`, `docs/plans/WP1_clean_dataset.md:9`, `docs/mac/MAC_DEV_BRIEF.md:28,72`, `docs/mac/FIX_LIST.md:9`, `docs/decisions/DECISIONS.md:105` | `/home/user/...` (old cloud layout) or Mac paths | doc |
| `docs/mac/STATUS.md:9` | `~/cr_scratch` | doc |
| `scripts/kb/WORKER_INSTRUCTIONS.md:11,52,53,58,59,60,62,65`, `scripts/kb/WORKER_INSTRUCTIONS_STATES.md:25-29` | `/home/user/clashBot` (cloud agent prompts) | doc |
| `knowledge_base/meta/scrape_manifest.json:3462` | `/home/user/clashBot/.kb_cache/phase2/...` | data |
| `tools/ingest_states.py:17` | branch name `origin/claude/clash-royale-kb-phase-1-4rtfm6` (cloud-pipeline script, not runnable by a grader) | n/a |
| `runs/overnight/*.sh` (gitignored) | hard-coded `/Users/tylerjackson/...` | not shipped |

Recommend: `sed` the JSON/MD reports to repo-relative paths when they are copied/refreshed; exclude or keep `docs/plans`, `scripts/kb/WORKER_*`, `docs/mac/MAC_DEV_BRIEF.md` by a conscious decision (they document an AI-agent workflow, including git commit trailers naming Claude; 354 commits, 161 of them authored under a `noreply@anthropic.com` identity, and commit messages carry `Co-Authored-By` / `Claude-Session` trailers). A `git archive` zip has no history.

### 1.5 Secrets scan (tracked files, 1120 files)

Pattern scan for HF/OpenAI/GitHub/AWS/Slack/Google keys, private-key blocks, `key|secret|password|token = "..."` assignments, Netscape cookie lines, YouTube cookie names, `.env`/`cookies`/`*.pem` filenames, and git history (added files matching cookie/.env/secret/token/.pt/.onnx/.npy/.sqlite/.mp4 across all branches).

| Result | Detail |
|---|---|
| Credentials / keys | **none found** |
| Cookie / env files | none tracked; `data/cookies.txt` is referenced in docs (README:152, `scripts/download_videos.py:103`) but is not in the repo; `/data/` is gitignored |
| Large-string hits (all false positives) | `clashbot/planner.py:19` HF model id; `tools/ingest_states.py:17` git branch name; `knowledge_base/meta/scrape_manifest.json:3464` sha256 of a saved HTML; `knowledge_base/meta/deck_index.json` deck keys; `knowledge_base/states/*.jsonl` own-deck keys; `tests/test_states.py:95` deck key; `docs/plans/MAC-E_llm_planner.md:15` model id |
| `scripts/kb/render_page.js:4` | reads `process.env.HTTPS_PROXY`; launches Chromium with `--ignore-certificate-errors` and `--disable-blink-features=AutomationControlled` to get past Cloudflare (README admits the site disallows automated agents). Not a secret; consider removing or noting for the ethics section |
| Public Google Drive ids | `tools/worker_setup.sh:33-34` (KataCR upstream weights) - public share ids, not secrets |
| Email-like strings | `tools/ingest_states.py:77` and `docs/plans/MAC-00_common.md:43` contain `noreply@anthropic.com` (commit trailer text); git author is a GitHub noreply address |
| Username leakage | `tylerjackson` appears in the docs/JSON reports listed in 1.4 |

### 1.6 Size of a code-only zip and proposed layout

`git ls-files`: 1120 files, **121.4 MiB raw, about 52 MiB zipped** (estimated with zlib; PNGs are already compressed).

| Top level (tracked) | Raw MiB | Zipped MiB (est.) |
|---|---|---|
| `knowledge_base/cards` + `evolutions` + `heroes` (PNG art, **needed at runtime**) | 44.3 | 41.9 |
| `knowledge_base/states` (197 game JSONL, the training data) | 65.8 | 6.7 |
| `knowledge_base/matches`, `decks`, `meta`, `archetypes` | 5.2 | 1.8 |
| `docs` (88 files) | 3.1 | 0.6 |
| `tests` (48 files) | 1.1 | 0.3 |
| `tools`, `clashbot`, `cr_perception`, `scripts` | 1.9 | 0.7 |

Largest tracked files: `docs/data/d1_zero_shot_v1.json` 1.1 MB; `knowledge_base/states/vp7jP_vjxcU-m0.1.jsonl` 884 KB; `.../5nOaTBlYLlg-m0.1.jsonl` 816 KB; `knowledge_base/evolutions/images/royal-ghost-evolution.png` 764 KB; then 0.5-0.7 MB state files and PNGs. Nothing is oversized.

Prebuilt data worth adding (zipped, est.): `dataset/v1/kg2` 0.3 MB, `db/kg2.sqlite` 1.4 MB, `db/models/act-gbdt` 0.4 MB, `dataset/v1/act_dense` 10.3 MB, `act_dense_bal` 6.1 MB, `dataset/v1/placement_prior.json` ~0, `db/hf/d1_tokenizer.json` 1.0 MB. Full `dataset/v1` + non-model `db/` would be about 40 + 40 MB; unnecessary because the chain rebuilds in minutes.

| Part | Content | Size (zipped) | How |
|---|---|---|---|
| **A. `clashbot-code.zip`** | `git archive HEAD` (code, tests, docs, knowledge_base incl. images and states) + new: `clashbot/resources/calib/mirror_calib.json`, `docs/data/*_v2.*` re-run outputs, a live-runs table (`docs/data/live_runs.md`), the rewritten README, `requirements.txt` | about 52 MB | no `.git`, no `runs/`, no `.claude/` |
| **B. `clashbot-prebuilt.zip`** (optional) | `db/kg2.sqlite`, `dataset/v1/kg2/`, `db/models/act-gbdt/`, `dataset/v1/act_dense*`, `db/hf/d1_tokenizer.json`, a 60 s demo clip (about 8-10 MB, cut from a pro video; check the creator's terms before including video) | about 20-30 MB | unzip over the repo root |
| **C. Weights (download)** | `d1-ft-joint-tac` as fp16 safetensors (about 1.2 GB; live inference already runs fp16, `resolve_dtype` auto) with its code files + tokenizer; KataCR `katacr_d1.pt`, `katacr_d2.pt` (176 MB; or fetch upstream via `tools/worker_setup.sh`) | 1.2 GB + 176 MB | HF Hub repo (public or access-restricted) or Drive link; `scripts/fetch_weights.sh` (to write) that puts them under `db/models/` and `~/cr_scratch/` |
| **D. Never in the zip** | the 33 pro videos (4.0 GB, third-party creator content; keep `scripts/download_videos.py --ids`), `runs/`, `data/templates`, other d1 checkpoints (`d1-ft-joint`, `-kg`, `-act`, `d1-2b-ft-joint-tac`, 10 GB), HF models for coach/planner (Qwen3-14B 7.8 GB, Qwen3-4B 2.1 GB), `.claude/` | - | document as optional |

Grader quickstart (target under 10 minutes after the one-time environment install; commands marked "to add" need the small additions above):

```bash
# 0. environment (once, network; not counted)
python3.12 -m venv .venv && source .venv/bin/activate
brew install libomp
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -r requirements-mac.txt

# 1. rebuild data from the tracked knowledge_base (est. 2-3 min)
python tools/data/build_clean_dataset.py && python tools/data/build_kg.py && python tools/data/build_documents.py
python tools/db/seed.py --hnsw && python tools/data/build_ranker_dataset.py && python tools/eval/eval_ranker_baselines.py
#    unzip clashbot-prebuilt.zip here for kg2 / act-gbdt / act_dense

# 2. tests (about 2.5 min)
python -m pytest -q --deselect tests/test_perception_latency.py

# 3. one offline eval, CPU (about 1-2 min) - the tactics engine vs the pro's plays
python tools/eval/eval_tactics.py --out /tmp/tactics_eval.json

# 4. fast replay (about 10 s)
python -m clashbot.loop --replay LGb0mz8Sb8w-m0.0 --limit 15

# 5. video dry run, 60 s, no big model (about 2.5-3 min on an M-series Mac; needs KataCR weights; clip + calib "to add")
python -m clashbot.live --dry-run --source video --video data/clips/demo_60s.mp4 --seconds 60 \
  --calib clashbot/resources/calib/mirror_calib.json \
  --deck electro-dragon,musketeer,tesla,fireball,knight,zap,ram-rider,mini-p-e-k-k-a --hero-cards musketeer \
  --ranker tactics --policy fast --act-model db/models/act-gbdt --device mps --out runs/quickstart
#    full stack: --ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint-tac   (adds the 1.2 GB weights; +4.5 min per 120 s)
```

Steps 1-4 total about 7-8 min; step 5 pushes the whole thing to about 10-11 min (so use `--seconds 30`, or skip steps 2/3 for the faster path).

### 1.7 README accuracy (README.md vs code)

| README statement | Reality | Action |
|---|---|---|
| "This repository currently contains **Phase 1**" (line 3) | Phases 1-4 plus Phase 3 data/KG/retrieval, the whole `clashbot/` runtime, d1 fine-tuning, tactics engine, live loop, 1284 tests | rewrite the intro |
| Layout block lists only `knowledge_base/` and `scripts/kb/` | missing `clashbot/`, `cr_perception/` (partly covered later), `tools/{data,db,eval,train}`, `tests/`, `docs/`, `requirements-mac.txt`, `knowledge_base/states` + `matches` (the 197-game data) | add a map |
| `cr_perception/` module list (Phase 4) | omits `clock.py`, `context.py`, `labels.py`, `perception.py`, `tracking.py`; the listed ones exist | update |
| "tests/ 15 tests" | 48 test files, 1284 passing | update |
| "Status: see the Phase 4 report in the session log" | dangling reference (no such file) | link `docs/mac/STATUS.md` |
| "YouTube currently blocks this session's IP ... `--cookies` is the way in" | the 33 videos were downloaded on 2026-11-03 (4.2 GB); the cloud block is obsolete | update, keep `--cookies` note |
| "Read-only state extraction ... Nothing in this package can send input" | true for `cr_perception`; the repo as a whole can tap via `clashbot/executor.py` (guarded by `--live` + `CLASHBOT_ALLOW_LIVE=1`, Training Camp only, D-031/D-050) | add safety + ToS paragraph |
| All commands use `python3` | system `python3` here is 3.13.5; pins verified on 3.12.8 | use `python3.12` / venv |
| `pip install -r scripts/kb/requirements.txt` only | the bot needs `requirements-mac.txt` + libomp + Apple clang for hnswlib | add |
| KataCR / BuildABot "backends" | no instruction where weights come from (`tools/worker_setup.sh`, D-030); BuildABot root is never provisioned | add; mark BuildABot unused |
| Tool commands (`scripts/download_videos.py --cookies`, `tools/batch_process.py`, `tools/dispatch_matches.py list\|assign\|done\|status`, `tools/calibrate.py`, `tools/run_video.py --detector`, `tools/benchmark_capture.py`, `tools/segment_transcript.py`) | flags/files exist and match the argparse | OK |
| `docs/PHASE3_OVERVIEW.md` (linked from the docs index, not README) | dated 2026-10-28; rebuild list omits `mine_pro_answers`, `build_deck_cooc`, `build_kg_v2`, `build_d1_tac`, `build_dense_act`, `sample_act_train`, `train_act_gbdt`; "pip install numpy hnswlib pytest"; "no models trained yet"; "Next steps ... not started" | refresh or replace with a single `docs/REPRODUCE.md` |

---

## Part 2 - benchmark inventory

### 2.1 Every eval/benchmark output and whether it reflects the current system

Verdict rule (as requested): compare the output's date (file mtime / `git log -1` of the output) with the last commit that changed the measured code or model (`git log -1 -- <paths>`, model dir mtime in `db/models`). If the measured code/model changed after the output, it is **stale-rerun**; else **current-keep**. Nothing is deleted: re-runs write `_v2` files (see 2.3). Key reference times: `policy.py` and `tactics.py` last changed `e578ef1` 11-07 02:54 (+1406 / +1436 lines since the 11-05 evals); `candidates.py`/`act_model.py`/`actions.py` 11-07 01:13-01:24; `live_state.py`, `state_text.py`, `d1_format.py`, `hud.py` `f04d922` 11-06 17:58; `perception.py`/`tracking.py` `af10598` 11-07 00:26; `opp_model.py`/`kg2.py` `2654dec` 11-06 20:01; `build_d1_tac.py` `4a258d3` 11-06 20:31; `dataset/v1/d1/joint_tac` rebuilt 11-06 20:37; models: `d1-ft-joint` 11-03 23:37, `d1-ft-act` 11-05 19:21, `act-gbdt` trees 11-05 19:40, `d1-ft-joint-tac` 11-05 22:24 (**live model**), `d1-ft-joint-kg` 11-06 22:00, `d1-2b-ft-joint-tac` 11-06 07:11. HEAD after 03:06 on 11-07 is docs only, so the live path did not change after `9234be4`.

Runtimes are from logs/JSON (`total_s`, `loop_seconds`, NEXT*.sh timestamps) where recorded; "est." otherwise.

| # | Script (spec) | Measures | Inputs | Output (mtime; git) | Headline numbers | Runtime | Verdict and reason |
|---|---|---|---|---|---|---|---|
| 1 | `tools/eval/eval_ranker_baselines.py` (WP4, MAC-H) | candidate recall@k of kNN/popularity; 4-way baselines (random, popularity, placement prior, leak checks) | `dataset/v1` ranker files | `docs/data/ranker_baselines_v1.json`, `ranker_stats_v1.json` (11-03 02:15; `cd67e95`) | 4-way val/test: random 0.250, option-set shortcut 0.250, lane-only 0.266/0.272, popularity 0.468/0.325, **placement prior 0.493/0.384**; recall@10 exact key kNN 0.361/0.317, popularity 0.499/0.402; card-level kNN 0.720/0.651 | seconds (est.) | **current-keep**: script (11-02 22:50), `build_ranker_dataset.py` (11-02 23:40) and the ranker files (11-03 00:43) all predate the output |
| 2 | `tools/eval/eval_d1_zero_shot.py` (MAC-D) | d1-omni-600M with no fine-tuning, joint and factorized | `dataset/v1/d1/{joint,factorized}` | `docs/data/d1_zero_shot_v1.json` (backend mlx, run 11-03 19:47; copy 11-04 00:01) | joint **0.258/0.221** (chance 0.25); factorized card 0.134/0.162, lane 0.369/0.355, zone 0.354/0.332; latency 44/38 ms p50 (joint) | about 1-2 min joint (44 ms x 1527), about 5 min factorized (179-214 ms x 1527); est. | **current-keep**: frozen model and frozen data (11-03 00:43) |
| 3 | `tools/train/train_d1.py` + `eval_d1_zero_shot.py --model` (MAC-T) | fine-tune d1 on `joint` (3 ep., 16 layers, 313.6 M trainable) and evaluate | `d1/joint`, HF d1-omni-600M | `d1_finetune_joint_train_v1.json`, `d1_finetune_joint_eval_v1.json` (11-04 00:01) | **0.524/0.483** (val/test) vs placement prior 0.493/0.384; 67-70 ms p50; train 13,242 s (3 h 41 min), peak 17.3 GiB | train 3.7 h MPS; eval about 2 min | **current-keep**: model (11-03 23:37) and data (11-03 00:43) unchanged; the no-tactics reference for ablations |
| 4 | same, `joint_tac` | fine-tune from `d1-ft-joint` on state + `tactics:` lines + option notes | `d1/joint_tac` (as built 11-05) | `d1_finetune_joint_tac_train_v1.json`, `..._eval_v1.json` (11-05 23:51) | **0.540/0.497**; without training on notes 0.480/0.470; 62-64 ms p50; train 4,554 s, peak 21.2 GiB | train 1.3 h; eval about 2 min | **current-keep** as the as-trained record (live model dir unchanged since 22:24); it measures the model on the 11-05 engine's notes, not the engine at HEAD. See row 5 for the parity re-run |
| 5 | same model on the rebuilt `joint_tac` (KG lines, K4b engine, 11-06 20:37) | d1-ft-joint-tac on newer notes | `d1/joint_tac` | `d1_finetune_joint_tac_on_kg_eval_v1.json` (11-06 23:36); also `dataset/v1/eval/d1_eval_..._rebuilt.json` (11-06 07:13, MAC-AD engine: 0.5375/0.4935) | **0.536/0.507**, 66-77 ms | about 2 min | **stale-rerun**: `tactics.py`, `policy.py`, `candidates.py`, `live_state.py` changed 11-07 after the data it scores; FIX_LIST #49/#50 (KG leakage, new ETAs not in the dataset) still open. Needs `joint_tac_v2` built from HEAD |
| 6 | `train_d1.py` joint_tac + KG facts | retrain from `d1-ft-joint-tac` with `KG:` lines | `joint_tac` (11-06 20:37) | `d1_finetune_joint_kg_{train,eval}_v1.json` (11-06 23:36) | **0.531/0.475**, worse than the model it started from (overfit: train loss 0.82 -> 0.43, val 1.19 -> 1.67); train 4,984 s | train 1.4 h | **current-keep** (negative result; model dir 11-06 22:00 unchanged); not the live model |
| 7 | `train_d1.py --variant act` + `eval_act_head.py` (MAC-X) | d1 play/wait head vs reactive rules on dense balanced data | `act_dense_bal`, `d1-ft-act` | `d1_finetune_act_train_v1.json`, `act_head_v1.{json,md}` (11-05 19:26) | test balanced acc: learned **0.558**, rules 0.535, combined 0.545 (n 3,221); train 6,286 s | eval about 7 min MPS (est.) | **stale-rerun, partial**: the learned-head columns are valid (model 11-05 19:21, data 17:31), but the `rules`/`combined` comparator columns call `ReactivePolicy`, which was rewritten (policy.py +1406 lines). Low priority: the head is not in the live path; the GBDT re-run (row 8) refreshes the rules baseline |
| 8 | `tools/train/train_act_gbdt.py` (MAC-AB A1) | LightGBM play/wait model, horizons 1/2/3 s; rules and `fast` policy comparators | `act_dense` (11-04 17:47) | `act_gbdt_v1.{json,md}` (11-05 19:45) | test AUC 0.632/0.659/0.672, balanced acc **0.594/0.607/0.613** (h 1/2/3) vs rules 0.529/0.533/0.530; 0.05 ms per decision | about 1-3 min CPU (est.) | **stale-rerun, partial**: features and trees are unchanged (act_model `FEATURES` identical; `act-gbdt` .txt 11-05 19:40), but the rules/`fast` comparator columns use `policy.py` (11-07) and `act_model.py` (11-07 01:13, counter mode) |
| 9 | `tools/eval/eval_act_policy.py` (MAC-W) | reactive policy vs 3 s trigger vs always, offline on `states.jsonl` | `states.jsonl` | `act_policy_v1.json` (11-03 19:56); newer: `runs/smoke/AM/act_policy.{json,md}` (11-07 02:44) | v1: reactive balanced acc 0.598 (test, all). AM: reactive **0.545**, act rate **0.220** (pro play rate 0.941; periodic: reactive 0.474, trigger 0.554) | under 1 min | **v1 stale-rerun** (policy.py rewritten since 11-03). **`runs/smoke/AM/*` current-keep**: run on the MAC-AM worktree 10 min before the commit; STATUS 11-07 lists the same 0.232 -> 0.220; promote to `docs/data/act_policy_v2` (a 1-min re-run gives strict provenance) |
| 10 | `tools/eval/eval_arbiter.py` (MAC-AB A4, D-045) | does restricting d1's options with kNN/tactics support help | d1 per-example probs, `joint_tac` | `arbiter_v1.{json,md}` (11-05 19:45) | no gain: d1 alone 52.4%/48.3%, best arbiter grid 52.6%/48.3%, default config 50.2%/47.6% | under 1 min CPU | **stale-rerun**: it used probabilities of `d1-ft-joint` (not the live `-tac`) on the 11-05 `joint_tac`; `arbiter` is off by default so priority is low. Re-run with the `-tac` per-example file |
| 11 | `tools/eval/eval_tactics.py` (MAC-Z3) | tactics engine vs pro plays: top-1/top-3 card, lane, zone, tile distance; 4-way grid | `states.jsonl`, `ranker/*`, KG | `tactics_eval_v1.{json,md}` (11-05 18:35); newer: `runs/smoke/AM/tactics.{json,md}` (11-07 02:45), `runs/smoke/AL/tactics.*` (01:03) | v1 (test): top-1 card 9.0%, push 18.0%, 4-way 27.5%/25.5%. AM (current engine): val coverage 52.1%, top-1 13.1%, push 15.9%; test 10.0% (random about 17-20%); 4-way **29.3%/26.5%** vs random 25.0%, placement prior 49.3%/38.4%, d1 52.4%/48.3% | about 1-2 min CPU | **v1 stale-rerun** (tactics.py +1436 lines, weights 11-05 19:56). **AM copy current-keep** (same caveat as row 9) |
| 12 | `tools/train/tune_tactics.py` (MAC-A2) | coordinate search of engine weights on pro plays, before/after | `states.jsonl` | `tactics_tuning_v1.{json,md}` (11-05 19:58; copy `dataset/v1/eval`, 19:53) | kept pass 2: val objective 0.513 -> 0.532, test 0.461 -> 0.476; top-1 test 11.6% -> 12.3% | 11.7 s recorded | **stale-rerun**: tuned against the 11-05 engine; engine rewritten since. Run it with a scratch `--out` so `clashbot/resources/tactics_weights.json` (committed) is not overwritten |
| 13 | `tools/eval/feature_ranker.py` (MAC-R3) | logreg / LightGBM LambdaRank / DCN-V2 on the 4-way task, 3 seeds, ablations | `ranker/*`, KG | `feature_ranker_v1.{json,md}` (11-04 10:00) | test: **LightGBM 0.491 +/- 0.013**, logreg 0.476, DCN-V2 0.412; val 0.548/0.546/0.526; top gain: `log_pop_key` 0.178, `p_zone_given_card` 0.124 | about 50 s (`r3_final.log`) | **current-keep**: script 11-02, ranker data 11-03 00:43; the earlier deterministic-grid numbers (11-02, inflated) in `dataset/v1/eval/feature_ranker.json` are superseded |
| 14 | `tools/eval/eval_retrieval.py` (MAC-R1) | BM25 vs structured vs bge text vs RRF; exact vs HNSW; pre- vs post-filter | `states.jsonl`, `db/`, bge | `retrieval_ablation_v1.{json,md}` (11-02 23:38) | hit@10 exact key 0.14-0.17 (channels within noise), card-level 0.40-0.51; **HNSW ef 40: recall@10 0.978 in 0.31 ms vs exact 1.51 ms**; deck pre-filter pool empty for 88.9% of queries, post@10 returns nothing for 50% (deck) / 87% (phase) | est. 5-10 min (bge embedding of 916+902 queries) | **current-keep**: `retrieval.py` unchanged since 10-28, `text_embed.py` 11-02 22:50, db 11-02 22:10 |
| 15 | `tools/eval/kge_link_prediction.py` (MAC-R2) | PyKEEN TransE/DistMult/ComplEx/RotatE on held-out COUNTERS, 3 seeds | `dataset/v1/kg` | `kge_link_prediction_v1.{json,md}` (11-03) | **RotatE MRR 0.159 / H@10 0.326** vs popularity 0.110 / 0.239, random 0.022 / 0.113; ComplEx 0.116; DistMult 0.091; TransE 0.082 | **3,634 s (about 61 min)** recorded | **current-keep**: script 11-02, KG built 11-02 22:11 |
| 16 | `tools/eval/next_card.py` (MAC-R4) | opponent next-card: popularity / copy-frequency / cycle / SASRec | `states.jsonl` | `next_card_v1.{json,md}` (11-02 23:38) | acc@1 val/test: **copy-frequency 0.207/0.243**, SASRec 0.129/0.222 (+/- 0.01), cycle w=0 (no help), popularity about 0 | 13.6 s recorded | **current-keep** |
| 17 | `tools/eval/edit_eval.py` (MAC-R5, D-034) | non-parametric knowledge edit: reliability, generalization, locality, portability, conflict, versioning, rollback | `states.jsonl`, `db/` | `edit_eval_v1.{json,md}` (11-03 02:15; run 00:42) | reliability 1.00 vs 0.00 (no edit), generalization 1.00/0.00 (n 465), locality 1.00/1.00, portability 1.00/0.07, conflict hits 0.124 -> 0.002, versioning + rollback pass | under 1 min (est.) | **current-keep** (`memory.py` 11-02 22:50). `edit_eval` imports `d1_format.card_description` (touched 11-06 for option notes; probe logic is rule based): a 1-min CPU re-run closes the provenance gap (optional) |
| 18 | `tools/eval/live_state_parity.py` (D-015, D-029) | live state builder vs training dataset on one re-perceived game | perception output of `B7SW-i94v8s-m0.3`, `states.jsonl` | `live_state_parity_v1.json` (11-03 02:15; run 11-02 23:40) | phase 0.62, towers 0.77, elixir 0.55, hand slots 0.57, units Jaccard 0.34, median feature cosine **0.67** | perception about 4-5 min MPS (237 s on the original run) + under 1 min | **stale-rerun**: `live_state.py`/`hud.py`/`state_text.py` (11-06), `perception.py`/`tracking.py` (11-07), `events.py`/`decktracker.py` (11-05) all changed; the perception input (`runs/parity/.../perception2`, 11-02 23:40) is older than all of them. Also the weakest perception number in the repo |
| 19 | `tools/eval/bench_planner.py` (MAC-E, E1) | Qwen3-4B planner timings; `--coach` Qwen3-14B coach | `db/`, MLX | `planner_bench_v1.json` (4B, 11-03 02:15); coach: `runs/smoke/E1/b_final.json` (11-06 01:08) | 4B: pregame 2.6-3.6 s, update 2.4-3.0 s, 0 tool calls. 14B coach: pregame p50 9.7 s (max 11.4), update **p50 4.1 s / p95 4.45 s**, prefill about 1.9 s, peak 8.6 GB | about 2-4 min each | **4B v1 stale-rerun** (`bench_planner.py`/`planner.py` changed 11-06 01:11; 4B no longer used). **Coach `b_final.json` current-keep** (planner unchanged since 3 min after), not in `docs/data`: promote as `coach_bench_v1` |
| 20 | `tools/eval/eval_eta.py` (MAC-AL T2) | tracker ETA error vs observed arrival, old vs new rule | video `Z8-4VhLjrGU` 60-300 s, KataCR | `runs/smoke/AL/eta.{json,md}` (11-07 00:23) | median abs error all frames **1.49 s -> 0.30 s**; melee 1.30 -> 0.30, ranged 2.19 -> 0.40, flyer 2.30 -> 0.20; 80% within 1 s; n 384 frames, 11 arrivals | 227 s + load (log) | **current-keep** (`tracking.py` last changed 3 min later in the same commit `af10598`); not in `docs/data` - promote. Small n |
| 21 | `tools/eval/review_shots.py` (MAC-AF) | did each tap land (elixir drop / slot change) | `runs/poc/<run>/shots` | `runs/smoke/AL/review/review.md` (11-07 00:20); `runs/poc/20261106-170722/shots/sheets` (86%) | run 233907: **95/100 landed** after the ability-verdict fix (91% before; ability 6/6); run 170722: 37/43 = 86% | under 1 min | **current-keep** for 233907. No review exists for the two newest runs (`013441` crashed before finishing, `111554` never reviewed): run it |
| 22 | `tools/eval/eval_opp_model.py` (MAC-AK K3) | opponent-deck model, leave-one-game-out | `states.jsonl`, KB | `dataset/v1/eval/opp_model.{json,md}` (11-06 17:53) - **not in docs/data** | recall@(8-k) k=1..3: **0.234 / 0.291 / 0.327** vs popularity 0.130 / 0.124 / 0.118 vs random 0.059 / 0.051 / 0.043 (155 games) | under 1 min | **stale-rerun**: `opp_model.py` changed `2654dec` 11-06 20:01, after the eval at 17:53; then copy to `docs/data/opp_model_v1` |
| 23 | `train_d1.py` LFM3 adapter + `eval_d1_zero_shot.py --model LiquidAI/d1-2B` (MAC-AE) | d1-2B zero-shot (200 lines) and its fine-tune | `joint`, `joint_tac` | `runs/smoke/E2/d1-2B_joint*.json`, `d1-2B_joint_tac*.json` (11-06 00:41); fine-tune: `db/models/d1-2b-ft-joint-tac/train_report.json` | zero-shot joint 0.250/0.235, joint_tac 0.345/0.285 (the fine-tuned 600M `d1-ft-joint` gets 0.480/0.430 on joint and 0.390/0.430 on joint_tac, same 200 lines); 250-325 ms p50 warm (645 ms cold); fine-tune val play-accuracy 0.419 (best epoch 3; 21,634 s = 6 h, 26 GiB) | train 6 h | zero-shot **current-keep**. **Fine-tune evaluation never produced**: `next3.log` step 3/4 crashed with `KeyError: 'lfm3'` loading the saved checkpoint (config mapping of the remote code); val play 0.419 is below the 600M's 0.54, so it would be a negative result. Fix needed first (see 2.3 H9) |
| 24 | `build_clean_dataset.py` report, KB audit | data cleaning counts | KB | `CLEANING_REPORT_v1.md`, `data_audit_2026-10-28.md` (`4ac5526`) | 10,923 samples, 6,658 train-eligible, 197 games, splits 142/27/28 | seconds | **current-keep** (builder unchanged since 10-28) |
| 25 | `build_d1_dataset.py` stats | token budgets of the d1 exports | `states.jsonl` | `d1_stats_v1.json` (11-03) | joint 3,641/761/766, states 90-733 tokens, 0 options over budget | seconds | **current-keep** |
| 26 | `clashbot.live --dry-run --source video` summaries | per-stage latency and decision mix on a recorded video | video, whole stack | `runs/smoke/AM_merged/summary.json` (11-07 03:05) and AL/AM variants | total **p50 248 / p95 402 ms** (perception 140/196, candidates 45/79, rank 84/174), 45 decisions, 0 shot errors, target p95 < 300 ms not met | about 4.5 min per 120 s | **current-keep** (HEAD code); one sample: needs repetitions for variance |
| 27 | older smoke dirs: `runs/smoke/{A2,A4,AD,AG,AF,Z3,Z310,Z310b,K4b,AI,fastfix,final_check,fullstack,k4b_check}` | intermediate evals of earlier engine/policy versions | | various (11-05 .. 11-06) | | | **superseded** (keep on disk, do not cite): all older than a later version of the same eval |

### 2.2 Live-run history (`runs/poc/*`, `runs/smoke/*` dry runs)

Source: `config.json`, `summary.json` where present, else recomputed from `decisions.jsonl` (marked *). "Taps" = decisions with `executor.sent`. Latency is total decision latency p50/p95 in ms (includes the 183 ms two-click `execute` stage in live runs). **The p95 < 300 ms target was missed in every run** (`target_met: false`). Outcomes (win/loss/crowns) are **not logged anywhere** (`summary.json` has no result field and no result-screen detector exists); the only wins on record are in `docs/mac/STATUS.md`.

| Run (2026-11-..) | Mode | Setup | Seconds (cfg / elapsed) | Decisions | By mode / source | Taps | Latency tot p50/p95 (perception, rank p95) | Shot errors | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 03 `192225` | LIVE | score ranker, rules planner, MPS | 60 / 60.1 | 28 | n/a (pre-policy) | 26 | 527 / 850 (94/545, 0) | - | first live run; macOS blocked clicks for ~30 s, taps worked after |
| 03 `192709` | LIVE | score ranker | 200 / 200.0 | 25 | n/a | 13 | 441 / 624 (93/216, 0) | - | 6 `hold_fallback`, 6 `skipped_card_not_in_hand`; match lost quickly |
| 05 `172506` | LIVE | d1-ft-joint, reactive | 200 / (no summary) | 27 | defend 22, punish 3, leak 2 | 8* | 236 / 710* | - | 14 of 27 skipped (hand `?`); lost in about 1 min |
| 05 `193555` | LIVE | d1-ft-joint, reactive, `--no-tactics` (match A) | 200 / (no summary) | 56 | defend 53, leak 3 | 42* | 422 / 673* (163/366) | - | lookup only; damage dealt one tower -830; lost |
| 05 `194101` | LIVE | d1-ft-joint, reactive + tactics (match B) | 200 / 200.0 | 80 | defend 76, punish 2, leak 2; tac 55, kNN 18, KG 7 | 80 | 393 / 577 (107/223, 146) | - | both towers -830; lost |
| 05 `205720` | dry | d1-ft-joint | 15 / 15.0 | 0 | - | 0 | - | - | aborted start |
| 05 `205934` | LIVE | d1-ft-joint, reactive | 300 / 300.1 | 61 | defend 46, punish 9, leak 6; tac 40, kNN 9, KG 4, hold 8 | 53 | 331 / 492 (96/202, 77) | - | about 2 matches |
| 06 `133529` | LIVE | d1-ft-joint-tac, `--policy fast`, **coach planner** (Qwen3-14B), bait deck | 900 / 900.0 | 181 | defend 157, learned 22, leak 2; tac 110, kNN 32, KG 19, hold 20 | 161 | 371 / 791 (107/312, 315) | - | about 4 matches; coach plans in `plans.jsonl` (4.7-6.2 s per update) |
| 06 `135724` | LIVE | tac + fast, electro-dragon deck | 900 / 900.0 | 61 | defend 46, learned 12, leak 3; hold 20, tac 26 | 41 | 263 / 572 (45/225, 142) | - | about 2 matches |
| 06 `170722` | LIVE | tac + fast, `--record-shots` | 900 / 900.1 | 56 | defend 39, learned 13, counter 3, leak 1 | 43 | 303 / 486 (94/198, 74) | 0 | landing 37/43 = 86% |
| 06 `233907` | LIVE | tac + fast + ability | 900 / 900.0 | 116 | defend 73, learned 25, leak 8, counter 8, punish 2; tac 64, KG 14, kNN 7, ability 6, fallback 9, hold 16 | 100 (6 ability) | 337 / 506 (97/225, 120) | 0 | 4 full games + part of a 5th (STATUS); landing 95/100 after verdict fix; hero ability 6/6 |
| 07 `013441` | LIVE | MAC-AL code, tac + fast | 900 / (no summary: crashed when the window closed, FIX_LIST #56) | 115 | defend 69, learned 34, leak 7, counter 4, spell 1 | 84* (1 ability) | 341 / 552* (110/205, 141) | - | 3 games: game 1 took a tower, game 2 3-crown loss, **game 3 won in overtime: the first win** (STATUS); Ram Rider 7x; fireball and tesla issues -> MAC-AM |
| 07 `111554` | LIVE | MAC-AM code, tac + fast, `holds.jsonl` | 900 / 900.0 | 125 | defend 74, learned 34, counter 9, leak 7, punish 1; tac 72, kNN 10, KG 6, fallback 7, hold 30 | 95 | 339 / 572 (102/230, 137) | 0 | not mentioned in STATUS.md; outcomes unknown; tap landing not reviewed |
| 03 `mirror-dryrun-1` | dry, mirrored screen | score | 120 / 120.6 | 59 | - | 0 | 136 / 672 (90/630, 0) | - | CPU detector (592 ms per detect) |
| 02 `video-dryrun-d1` | dry, video | zero-shot d1 | 250 / - | 692 | trigger policy (no min gap) | 0 | 171 / 471 (87/386, 76) | - | before fixes #14/#15 |
| 07 `smoke/AL_dry`, `AL_merged` | dry, video `Z8-4VhLjrGU` | d1-ft-joint-tac + fast | 120 / 90 | 50 / 34 | defend 34 / 31, learned 9 / 0 | 0 | 242 / 404; 246 / 394 | 0 | MAC-AL checks |
| 07 `smoke/AM_dry`, `AM_merged` | dry, video | same | 120 | 45 / 45 | defend 36, punish 6, learned 2, leak 1 | 0 | 248 / 424; **248 / 402** (140/196, 175) | 0 | MAC-AM merge check; 3 teslas in box, 1 fireball |
| 06 `smoke/fullstack`, `final_check`, `k4b_check`, 05 `fastfix` | dry, video | various | 60 / 40 / 25 / 15 | 34 / 7 / 3 / 5 | | 0 | 211/388, 262/2677 (cold d1), 312/2954 (cold), 556/8245 (cold) | 0 | first-call load spikes dominate p95 on tiny clips |

Aggregate (13 live runs, about 6,500 s of configured live time): since the 50 ms click-gap fix the `execute` stage is 181-184 ms p50 (383-393 ms before); perception 45-110 ms p50 (2x quartz capture, MPS detector); rank p95 74-315 ms; in `233907` capture ran at 7.95 fps against a 10 fps target (1,849 frames dropped), the perception loop at 12.9 fps. Match counts in the Notes column are approximate (readiness gaps); only `233907` and `013441` are described in STATUS.

### 2.3 Re-run plan (stale outputs only; heavy jobs one at a time)

Conventions: run from `/Users/tylerjackson/Projects/clashbot/clashBot`; `PY=../.venv/bin/python`; `export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1`. Every command writes a new `_v2` (or scratch) file; **none overwrites** `docs/data/*_v1.*`, `dataset/v1/eval/*`, the committed `clashbot/resources/tactics_weights.json` or any model dir. Heavy = MPS/GPU or large memory (the Mac's memory guard killed queued jobs on 2026-11-03: close other apps, never overlap two heavy jobs). Order below is the order to run.

| Step | Re-runs (stale output) | Exact command | Expected runtime | Resource | Writes |
|---|---|---|---|---|---|
| C1 | act policy (row 9) | `$PY tools/eval/eval_act_policy.py --out docs/data/act_policy_v2.json` | about 1 min | CPU | `act_policy_v2.{json,md}` |
| C2 | tactics eval (row 11) | `$PY tools/eval/eval_tactics.py --out docs/data/tactics_eval_v2.json` | about 1-2 min | CPU (reads KG) | `tactics_eval_v2.{json,md}` |
| C3 | act GBDT comparator columns (row 8) | `mkdir -p runs/rerun && $PY tools/train/train_act_gbdt.py --out runs/rerun/act-gbdt --report docs/data/act_gbdt_v2.json --bal dataset/v1/act_dense_bal --d1-json docs/data/act_head_v1.json` (do **not** use the default `--out db/models/act-gbdt`; compare `runs/rerun/act-gbdt/meta.json` thresholds with the shipped model as a determinism check) | about 1-3 min | CPU, 1 thread, no torch in process | `act_gbdt_v2.{json,md}` |
| C4 | tactics tuning (row 12) | `$PY tools/train/tune_tactics.py --out runs/rerun/tactics_weights_rerun.json --report docs/data/tactics_tuning_v2.json` (non-default `--out` protects the committed weights) | about 1-5 min | CPU | `tactics_tuning_v2.{json,md}` |
| C5 | opponent-deck model (row 22) | `$PY tools/eval/eval_opp_model.py --out docs/data/opp_model_v1.json` | under 1 min | CPU | `opp_model_v1.{json,md}` |
| C6 | edit eval, provenance only (row 17) | `$PY tools/eval/edit_eval.py --out docs/data/edit_eval_v2.json` | under 1 min | CPU | `edit_eval_v2.{json,md}` |
| C7 | tap-landing review of the two newest runs (row 21, new) | `$PY tools/eval/review_shots.py runs/poc/20261107-111554` (and `...013441` if its shots index is complete) | about 1 min each | CPU | `runs/poc/<run>/shots/sheets/review.md` (copy the table to `docs/data`) |
| H0 | build `joint_tac` with the HEAD engine (needed by rows 5, 10) | `$PY tools/data/build_d1_tac.py --out dataset/v1/d1/joint_tac_v2` (confirm it does not touch `dataset/v1/d1/stats.json`) | about 2-4 min (`next3.log`: 2 min 14 s) | CPU, builds on `db/kg2.sqlite` | `d1/joint_tac_v2/` |
| H1 | d1-ft-joint-tac on the HEAD data (row 5) | `mkdir -p runs/eval_tac_v2/d1 && ln -sfn $PWD/dataset/v1/ranker runs/eval_tac_v2/ranker && ln -sfn $PWD/dataset/v1/d1/joint_tac_v2 runs/eval_tac_v2/d1/joint && $PY tools/eval/eval_d1_zero_shot.py --dataset runs/eval_tac_v2 --variant joint --backend transformers --model db/models/d1-ft-joint-tac --out docs/data/d1_finetune_joint_tac_eval_v2.json` | about 2 min (`next3.log` step 4: 1 min 48 s) | MPS, 2.2 GB model | `d1_finetune_joint_tac_eval_v2.json` |
| H2 | ablation columns on the same data: no-tactics model and KG-trained model | same command with `--model db/models/d1-ft-joint` (`--out docs/data/d1_joint_on_tac_v2.json`) and `--model db/models/d1-ft-joint-kg` (`--out docs/data/d1_joint_kg_on_tac_v2.json`) | about 2 min each | MPS | two JSON files |
| C8 | arbiter (row 10), after H1 | `$PY tools/eval/eval_arbiter.py --d1-eval docs/data/d1_finetune_joint_tac_eval_v2.json --lines dataset/v1/d1/joint_tac_v2 --out docs/data/arbiter_v2.json` | under 1 min | CPU | `arbiter_v2.{json,md}` |
| H3 | act-head comparator columns (row 7), optional | `cp db/models/d1-ft-act/act_threshold.json /tmp/act_threshold.bak && $PY tools/eval/eval_act_head.py --model db/models/d1-ft-act --out docs/data/act_head_v2.json` (it rewrites `act_threshold.json` inside the model dir with the val-best threshold; restore from the backup if it differs) | about 7 min (est.: 6.2 k lines x about 65 ms) | MPS | `act_head_v2.{json,md}` |
| H4 | live-state parity (row 18): re-perceive the game, then compare | `$PY tools/run_video.py --video data/videos/B7SW-i94v8s/B7SW-i94v8s.mp4 --calib runs/parity/B7SW-i94v8s-m0.3/calib.json --out runs/parity/B7SW-i94v8s-m0.3/perception3 --start 750 --end 1018 --stride 5 --device mps` then `$PY tools/eval/live_state_parity.py --perception runs/parity/B7SW-i94v8s-m0.3/perception3/states.jsonl --game B7SW-i94v8s-m0.3 --out docs/data/live_state_parity_v2.json` (window and stride reconstructed from the old run: first state t 750.0, last 1017.9, frame step 5; confirm `--imgsz`/`--detect-every` against the old `summary.json`) | about 4-5 min + under 1 min | MPS (KataCR) | `live_state_parity_v2.json` |
| H5 | coach bench (row 19, promote) | `$PY tools/eval/bench_planner.py --coach --decks 5 --updates 3 --out docs/data/coach_bench_v2.json` | about 3-5 min | MLX Qwen3-14B, 9 GB peak (do not overlap anything) | `coach_bench_v2.json` |
| H6 | whole-stack latency, 3 repetitions for variance (row 26) | the dry-run command of 1.3 with `--out runs/rerun/lat_1`, `lat_2`, `lat_3` | about 4.5 min each | MPS: KataCR + d1 | three `summary.json` |
| H7 | optional: promote current AM/AL artefacts | `cp runs/smoke/AL/eta.json docs/data/eta_eval_v1.json; cp runs/smoke/AL/eta.md docs/data/eta_eval_v1.md; cp runs/smoke/AL/review/review.md docs/data/tap_landing_233907_v1.md` | seconds | - | docs only |
| H8 | optional: d1-2B fine-tune evaluation (row 23) | blocked: fix the checkpoint loader (`KeyError: 'lfm3'` from `CONFIG_MAPPING` when loading `db/models/d1-2b-ft-joint-tac`), then `$PY tools/eval/eval_d1_zero_shot.py --dataset runs/eval_tac_v2 --variant joint --backend transformers --model db/models/d1-2b-ft-joint-tac --out docs/data/d1_2b_ft_eval_v2.json` | fix 30-60 min; eval about 7 min (250-300 ms x 1527) | MPS, 3.6 GB model | one JSON |

Total hands-on compute: CPU group about 10 min, then heavy group about 35-45 min strictly sequential (H1, H2, H3, H4, H5, H6, in that order, one at a time; H6 last because it also validates the whole stack).

### 2.4 Headline numbers for the report (current-keep only)

| Claim | Number | Source (row) |
|---|---|---|
| Fine-tuned d1 (live model) vs baselines, 4-way multiple choice (n 761/766) | **0.540/0.497** (tac) vs placement prior 0.493/0.384, LightGBM 0.548/0.491, popularity 0.468/0.325, zero-shot d1 0.258/0.221, random 0.250. 95% CI about +/- 3.5 points at n = 766: d1-tac and LightGBM are not distinguishable on test | 1, 2, 4, 13 |
| Play/wait timing | LightGBM test balanced acc 0.607 (h = 2 s) vs rules 0.533 vs d1 head 0.558; 0.05 ms per decision | 7, 8 (rules columns pending re-run) |
| Retrieval | HNSW ef 40 recall@10 0.978 at 0.31 ms vs exact 1.51 ms (10.9 k vectors); deck pre-filter pool empty for 89% of queries | 14 |
| KG embeddings | RotatE MRR 0.159 / H@10 0.326 vs popularity 0.110 / 0.239 vs random 0.022 / 0.113 | 15 |
| Opponent next card | copy-frequency 0.243 vs SASRec 0.222 (acc@1, test): a tuned simple baseline wins | 16 |
| Knowledge editing | reliability / generalization / portability 1.00 vs 0.00 / 0.00 / 0.07, locality 1.00 | 17 |
| ETA model | median abs error 1.49 s -> 0.30 s | 20 |
| Tap landing | 95/100 | 21 |
| Live latency | total p50 about 340 ms, p95 about 500-570 ms (target 300 ms not met); execute stage 183 ms | 2.2 |
| Live results | one win on record (013441, game 3, overtime) | STATUS |

### 2.5 Gaps a strong report still needs

| # | Missing benchmark | What exists | What is missing | Effort (dev + compute) |
|---|---|---|---|---|
| 1 | **Win rate / crowns / tower damage from live trials, with variance** (the course asks for trajectory-level evaluation over repeated trials, AGT 7.2) | 13 live runs, `holds.jsonl`, shots, `review_shots` landing rate; tower HP appears in `context_text` | no outcome logging at all (no summary field, no result-screen reader; tower reads are noisy, king HP often `?`); no trial protocol; n = 1 anecdotal win. Needs a result detector (RapidOCR on the Victory/Defeat screen or crown icons; `crown-icon` is a KataCR class) or a manual CSV, a runner that loops N Training Camp battles per config, Wilson CIs + crowns mean/sd | dev 0.5-1 day (result reader + aggregator); play time 3.5-5 min per game x 10 games x 3-4 configs = about 2-4 h of the user's time |
| 2 | **Offline ablation of the live stack on recorded video**: ranker `score` vs `tactics` vs `d1`; kg2 on/off; tactics on/off; policy `reactive` vs `fast` | all switches exist on `clashbot.live` (`--ranker`, `--no-tactics`, `--no-kg2`, `--policy`, `--arbiter`) and write `decisions.jsonl` + `summary.json`; AL/AM smoke runs are single configs | a harness that runs a config matrix over 2-3 fixed clips and tabulates decisions, hold share, source mix, elixir spent per minute, agreement with the pro's own plays in the clip (own-play events from `runs/videos/<id>/match_n/states.jsonl`), reaction time to opponent plays, per-stage latency; a second offline axis: `build_d1_tac --no-kg-facts` + the d1 eval gives a KG-lines on/off number cheaply | dev 0.5-1 day; compute 6-8 configs x 2 clips x about 3.5 min = about 1 h (d1 configs 4.5 min per 120 s) |
| 3 | **Re-run of stale evals on HEAD** (tactics, act policy, d1 on rebuilt `joint_tac`, parity, opp model) so the report is internally consistent | scripts exist; plan in 2.3 | only execution and copying outputs into `docs/data` | about 1 h hands-on, about 45 min compute |
| 4 | **Statistical rigor on the 4-way table**: paired bootstrap / McNemar, CIs | `per_example` probabilities in every d1 eval JSON; LightGBM and the placement prior are deterministic | no script; LightGBM per-example predictions are not persisted by `feature_ranker.py` | 2-3 h, no GPU |
| 5 | **Latency breakdown with variance** | per-stage p50/p95 in every `summary.json` (13 live + 9 dry runs), `benchmark_capture.py` (capture fps), `bench_planner.py` | no aggregator across runs; no isolated micro-benchmarks (KataCR MPS vs CPU, hand reader, d1 fp16 MPS vs MLX vs CPU) in one table; only one run per config | 2-3 h dev; about 15 min compute (H6) |
| 6 | **Retrieval speed: exact vs HNSW** | done at 10.9 k vectors (row 14); candidates use exact search (37-45 ms total candidate stage) | scaling curve (synthetic 10 k -> 1 M vectors, build time, memory) to show where HNSW starts to matter; 2 h | 2 h dev; minutes compute |
| 7 | **Perception accuracy against labelled frames** (hand, elixir, clock, tower HP, units) | `live_state_parity` (stale; weak numbers: units Jaccard 0.34, hand slots 0.57), `eval_eta`, `run_video.py --frames-every` dumps frames for labelling | no ground-truth labels, no per-reader accuracy table; ESCA argument (perception errors dominate agent failures, course KG 20) is only asserted | about 1 day (label about 200 frames) |
| 8 | **Component value of the coach and of memory** (Qwen3-14B plans, episodic memory) | timing only (row 19); `plans.jsonl` in run `133529` | d1 was never trained with `[plan]` text; no A/B | skip or cite as limitation |
| 9 | **Perception/dense-label noise ceiling** quantified | STATUS notes hand fully read in 34-37% of moments, pro card in the visible hand 47% | not tabulated in `docs/data` | 1 h (query `states.jsonl`) |
