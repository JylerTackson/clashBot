# Brief for "MAC - DEV" (Opus on the user's M5 Mac) — 2026-11-03

From: the cloud planning session ("clashbot-ce", Opus) that built Phases 1–3 with Sonnet implementers.
To: you, the Claude Code session running in the clashBot repo on the user's Mac.
The user asked: brief you, then drive the repo forward **from the Mac** — set it up for proof-of-concept (POC) runs, keep developing and pushing from the Mac to GitHub, and build out the course-research enhancements.

Read this whole file, then `docs/PHASE3_OVERVIEW.md`, `docs/decisions/DECISIONS.md`, `docs/design/pipeline_review.md` and `docs/research/course_mapping.md`. Those four documents are the ground truth; this brief tells you what to do next.

---

## 1. Rules you inherit (set by the user — non-negotiable)

1. **Opus plans, audits and re-plans; Sonnet implements.** Every code change and every data edit is done by a Sonnet subagent (`Agent` tool, `model: "sonnet"`) from an Opus-written spec. Specs must be prescriptive enough that Sonnet makes no design decisions (see `docs/plans/WP1…WP7` for the expected level of detail: exact files, signatures, algorithms, field names, tests, acceptance commands, a short report format). You (Opus) may run commands, benchmarks and reviews yourself; you do not write project code or edit data yourself.
2. **Review every Sonnet result** before committing: re-run its tests, spot-check outputs, look for leakage/shortcuts (WP4 Amendment C is the example of why). Send fixes back as numbered amendments appended to the same spec file.
3. **Ask the user before making decisions** that are not covered by the decision log. The user is at the Mac: use `AskUserQuestion` with a recommended option first. Conventional, reversible defaults may be taken and logged as `default (Opus)`.
4. **Document every decision** in `docs/decisions/DECISIONS.md` (next id: **D-025**). Keep `docs/PHASE3_OVERVIEW.md` current.
5. **Security/hygiene:** never print, log or commit cookies, tokens or keys; `data/`, `dataset/`, `db/`, `runs/` are gitignored and must stay so. The perception package is read-only; only `clashbot/executor.py` may ever send input, and only behind an explicit `--live` flag the user turns on.
6. **Terms of service:** Clash Royale prohibits bots in normal play. Live runs only in **Training Camp or friendly battles**, and the report says so.

## 2. Where the project is (summary — details in PHASE3_OVERVIEW.md)

- **Branch:** `claude/clash-royale-kb-phase-1-4rtfm6` on `JylerTackson/clashBot`. All Phase 1–3 work is pushed there (last reviewed commit `b25f272`, plus this brief).
- **Data:** `knowledge_base/` (raw, committed) → `dataset/v1/` (clean states, games, splits, KG, documents, ranker, act, d1 exports — rebuilt by scripts) → `db/clashbot.sqlite` + feature vectors + HNSW (rebuilt by `tools/db/seed.py`).
- **Runtime:** `clashbot/` — game rules, features, hybrid retrieval (BM25 + structured vectors + RRF), KG access, candidate generator (kNN + KG counters + legality; plays only), ranker interface (`ScoreRanker` baseline; `D1Ranker` builds d1 requests but `choose` is a stub), six read-only tools, rule-based planner (+ Qwen stub), executor (`DryRunExecutor`; `MirroringExecutor` stub with tile→screen homography), fast loop with replay (`python3 -m clashbot.loop --replay <game>`): ≈41 ms p50 per decision on a cloud CPU.
- **Perception:** `cr_perception/` (KataCR YOLOv8 + HUD readers + OCR + Kalman tracking). Built for recorded video; `sources.ScreenSource` already captures a macOS window with `mss` + Quartz lookup.
- **Models:** decision model = **LiquidAI d1-omni-600M** (D-022; **do not fine-tune until the user says so**). Training data ready in `dataset/v1/d1/{joint,factorized,act}` (Decision Index format, `clashbot/d1_format.py`; act is provisional, D-020). Slow-loop LLM = Qwen ~3–4B 4-bit on MLX (D-021; exact model to confirm with the user).
- **Results so far (val/test):** candidate recall@10 exact play 0.37/0.32 (kNN) vs 0.50/0.41 (popularity); card-level 0.72/0.65 vs 0.68/0.61; best 4-way multiple-choice baseline (placement prior) 0.37/0.36 vs random 0.25. These are the bars d1 must beat.
- **Known gaps:** perception tests need OpenCV (never ran in the cloud); KataCR paths are hard-coded to the cloud layout (`/home/user/wty-yy/katacr`, see `tools/worker_setup.sh`, `tools/calibrate.py`); the act/hold head needs dense negatives from re-running perception on the videos (approved for **later**, D-020); no live-screen → clean-state builder exists yet.

## 3. Git protocol

- Work on `claude/clash-royale-kb-phase-1-4rtfm6`. Before each push: `git pull --rebase origin claude/clash-royale-kb-phase-1-4rtfm6`. Never force-push. Small, reviewed commits with clear messages.
- From now on the Mac is the primary developer. The cloud session will only push under `docs/` if the user asks it to review something, so conflicts should be rare.
- Commit trailers: follow whatever your own session's attribution rules say.

## 4. How to report (so the user and the cloud session can follow)

Append one dated section per milestone to `docs/mac/STATUS.md` (create it): what was done, test/benchmark numbers, decisions taken (with D-ids), open questions, next step. Commit and push it with the milestone. Keep your end-of-turn summary to the same headline.

## 5. Work plan

Do the milestones in order. Each step = Opus spec in `docs/plans/` → Sonnet implements → Opus review → commit/push → STATUS entry.

### M0 — Mac environment and verification (do first)
1. `git fetch && git checkout claude/clash-royale-kb-phase-1-4rtfm6 && git pull`.
2. Record hardware/software (chip, RAM, macOS, Python) in STATUS.md.
3. Create a project venv (Python 3.11 or 3.12). Install `requirements-perception.txt` (torch with MPS instead of the CPU-wheel note), plus `numpy hnswlib pytest tokenizers`. Ask the user before installing anything large beyond that list.
4. KataCR detector: clone `wty-yy/katacr` and fetch the two checkpoints as `tools/worker_setup.sh` does, to Mac paths you choose; then have Sonnet replace the hard-coded `/home/user/wty-yy/katacr` defaults with a config value / env var (`CLASHBOT_KATACR_ROOT`, `CLASHBOT_KATACR_WEIGHTS`) — one small spec.
5. Rebuild everything with the commands in PHASE3_OVERVIEW.md (`build_clean_dataset` → `build_kg` → `build_documents` → `seed --hnsw` → `build_ranker_dataset` → `eval_ranker_baselines` → pytest → `build_d1_dataset --tokenizer`; the d1 tokenizer comes from `https://huggingface.co/LiquidAI/d1-omni-600M/resolve/main/tokenizer.json` into `db/hf/`). Outputs must match the cloud numbers (state/sample counts, baselines) — report any difference.
6. Run the **full** test suite, including the OpenCV perception tests the cloud could not run. Fix failures via Sonnet.
7. Benchmarks on the M5: `python3 -m clashbot.loop --replay LGb0mz8Sb8w-m0.0 --limit 50` latency; `tools/benchmark_capture.py` capture fps (needs Screen Recording permission for the terminal — ask the user to grant it).

### M1 — Proof-of-concept readiness (no taps until the user enables them)
1. **iPhone Mirroring capture:** extend `ScreenSource` (or a config) to locate the "iPhone Mirroring" window and capture it at a steady rate; measure fps.
2. **Calibration for the mirrored screen:** reuse `tools/calibrate.py` (arena corners, HUD ROIs as fractions of the content rect) on a captured frame; save a Mac calibration file. The same 4 arena points give the executor's `Calibration` (tile → screen pixels).
3. **Live state builder (train/serve parity, D-015):** a module that turns live perception output (`cr_perception` GameState/events + tracks) into the **clean state dict** used in `dataset/v1/states.jsonl` (same fields, same unit→card resolution via `clashbot/units.py`, same `render_context`), so retrieval and d1 see the same inputs as in training. Write a parity test: feed a recorded video segment through the live path and compare against the dataset's states for that game at the same timestamps (the user may need to re-download one video for this; ask).
4. **Dry-run live loop:** `python3 -m clashbot.live --dry-run` = capture → perception → live state → `should_decide` → candidates → ranker → `DryRunExecutor`, printing decisions and per-stage latency (capture, perception, state build, retrieval, ranking). Target: total < 300 ms p95 on the M5; report where time goes.
5. **d1 zero-shot ranker:** implement `D1Ranker.choose` (lazy import; backend = `transformers>=5.15` on MPS per the model card, or the official `LiquidAI/d1-omni-600M-MLX-8bit` build — read its card and ask the user if it is not obvious). Evaluate **zero-shot** d1 on `dataset/v1/d1/joint/val.jsonl` and `test.jsonl` against the baselines (accuracy, latency per request). This is not fine-tuning.
6. **Qwen planner:** implement `LLMPlanner` with `mlx-lm` (model to confirm with the user, ~3–4B 4-bit), using `PREGAME_TEMPLATE` and the six read-only tools in a ReAct loop with a step cap; measure time-to-plan. It must never call the executor.
7. **Executor taps:** implement `MirroringExecutor` with Quartz mouse events (card slot click, then tile click) behind `--live`; test first on a static screenshot overlay (draw the intended click points) before any real click. Ask the user before the first live click.
8. **POC protocol:** Training Camp first, then a friendly battle; log every decision, latency and outcome to `runs/poc/<timestamp>/` (gitignored) and summarise in STATUS.md.

### R — Course-research enhancements (start after M0; interleave with M1 as the user prefers)
Each item is an experiment for the 8-page report. Write the spec with the paper references from `docs/research/course_mapping.md`, keep evaluation protocols honest (video splits, no same-game neighbours, equal tuning budgets), and save result tables to `docs/data/`.

| id | item | papers (course PDF refs) | deliverable |
|---|---|---|---|
| R1 | Retrieval channel ablation: BM25 vs structured vectors vs a small text-embedding channel vs RRF fusion; exact vs HNSW (recall vs latency, `efSearch` sweep); pre- vs post-filtering by deck/archetype (empty-result rate) | RET: BM25 [32], ColBERT/bi-encoder limits [14], RRF [3], HNSW [20], filtering §2.2, BEIR [40] | `tools/eval/eval_retrieval.py` + table |
| R2 | KG embeddings for missing COUNTERS links: TransE, DistMult, ComplEx, RotatE (PyKEEN), same tuning budget, random tie-breaking, held-out wiki edges | KG: [2, 42, 38, 36], audits [34, 37] | link-prediction table; predicted counters as an optional candidate channel |
| R3 | Feature-interaction ranker baseline (GBDT or DCN-V2-style MLP on the structured vector + option features) vs popularity / placement prior | RAG §4.5 FM/DLRM/DCN-V2 [56–58]; "tuned baselines" [5–7] | ranker table |
| R4 | Opponent next-card prediction from the play sequence: copy baseline (CyGNet-style frequency of past cards) vs SASRec-style sequence model | KG temporal [46, 21]; RAG [61] | accuracy table; feeds threat anticipation |
| R5 | Memory + knowledge editing: episodic (own-game states appended to the DB/index), semantic (patch-versioned card facts with provenance/validity), procedural (post-game lessons, Reflexion/ACE playbook); evaluate one balance-change edit for reliability, generalisation, locality, portability; conflict policy for pre-patch pro samples | AGT: memory types §4.1, Mem0 [25], Reflexion [5], ACE [22], editing criteria [46], SERAC/GRACE vs ROME [53, 54, 49], ClashEval [67] | `clashbot/memory.py` + edit-eval table |
| R6 | Planner context budget: plan quality vs number of retrieved documents (inverted-U), position effects | RAG [16, 17]; AGT Lost in the Middle [12], Context Rot [13] | small table |
| R7 | Adaptive arbitration: consult the Qwen planner only when candidate generators disagree and no threat is imminent; measure latency and agreement | RAG FLARE [27], Adaptive-RAG [31], Self-RAG/CRAG [29, 30] | ablation table |

Recommended order: **R1 → R3 → R2 → R5 → R4 → R6/R7**. R1–R3 need no new models and give the report's core tables quickly.

### Not now (explicitly deferred by the user)
- Fine-tuning d1 (data is ready; wait for the user's go-ahead).
- Re-downloading the videos / re-running perception for dense act/hold negatives (approved for later). If a single video is needed for the M1 parity test, ask first.

## 6. First actions for you
1. Pull the branch and read the four documents above.
2. Start M0. When M0 is done, write the STATUS entry, push, and ask the user whether to continue with M1, R1, or both in parallel.
