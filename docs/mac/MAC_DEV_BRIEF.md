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

## 5. Work plan — maximum parallelism, then one fix pass (user instruction, 2026-11-03)

The user wants **as much work as possible in parallel**, followed by a review-and-fix pass for whatever breaks. So do not run milestones one after another. Work in waves:

```
Wave 0  (short, sequential)   setup + one batch of user questions + ALL Wave-1 specs written
Wave 1  (parallel)            every independent work package dispatched at once (Sonnet, isolated worktrees)
Wave 2  (integration)         Opus merges each worktree, runs the full suite, records every failure/finding
Wave 3  (parallel fix pass)   findings grouped by file owner -> parallel Sonnet fix agents -> re-merge -> green
Wave 4  (parallel)            work that needs Wave-1 outputs or the user at the Mac (R6, R7, calibration, live dry-run)
Wave 5  (fix pass)            as Wave 3; then STATUS.md + push; ask the user about POC runs
```

### Parallelism rules
1. **Dispatch in one message.** Launch every Wave-1 agent in a single response (`Agent` calls side by side, `model: "sonnet"`, `isolation: "worktree"`, `run_in_background: true`). Each agent works in its own git worktree, so parallel edits cannot collide.
2. **Disjoint file ownership.** Every spec lists the exact files its agent may create or edit. No two Wave-1 packages own the same file. Shared files (`requirements*.txt`, `docs/decisions/DECISIONS.md`, `docs/PHASE3_OVERVIEW.md`, `docs/mac/STATUS.md`, `README.md`) are edited only by you (Opus) at integration time; agents list what they need added (e.g. a dependency) in their report.
3. **Interfaces fixed up front.** Where one package consumes another (e.g. the live loop uses the capture source and the live-state builder), the spec of each states the exact function signatures and data shapes, so both are written at the same time against the contract; the consumer's tests use small fakes.
4. **Don't block, report.** Agents get a time box: if their tests still fail after a reasonable attempt, they stop and report the failing test output instead of thrashing. Broken work is expected; Wave 3 fixes it.
5. **Heavy jobs are serialised, coding is not.** The M5 has 16 GB shared memory: model downloads, d1/Qwen inference, YOLO runs and the full rebuild must not run concurrently. Coding agents write code and run fast unit tests with fakes; the heavy acceptance runs are queued and executed one at a time by you during integration (or by one designated "runner" agent). Up to ~10 coding agents in parallel is fine.
6. **Integration is sequential and owned by Opus.** Merge worktree branches one at a time (`git merge --no-ff`), run the fast tests after each merge, then the heavy runs once. Collect every failure and every review finding into one numbered list in `docs/mac/FIX_LIST.md` (finding, file owner, evidence, intended fix).
7. **Fix pass in parallel.** Group FIX_LIST items by file owner, write one amendment per group (appended to that package's spec), dispatch all fix agents at once, re-integrate, repeat until the suite is green and the review is clean. Then commit and push.

### Wave 0 — sequential, keep it short
1. `git fetch && git checkout claude/clash-royale-kb-phase-1-4rtfm6 && git pull`; record chip/RAM/macOS/Python in `docs/mac/STATUS.md`.
2. Create the venv (Python 3.11/3.12); install `requirements-perception.txt` (torch with MPS) + `numpy hnswlib pytest tokenizers`. Start the large downloads in the background while you write specs: KataCR repo + two checkpoints (as in `tools/worker_setup.sh`, Mac paths), the d1 tokenizer into `db/hf/`.
3. **One batch of questions to the user** (`AskUserQuestion`, recommended option first), so no agent blocks later: d1 inference backend (transformers on MPS vs the official MLX 8-bit build), the Qwen model for the planner (~3–4B 4-bit on MLX), any extra installs (PyKEEN, LightGBM, sentence-transformers / MLX embedder, mlx-lm), and whether one Ryley video may be re-downloaded for the live-state parity test.
4. Write **all Wave-1 specs** in `docs/plans/` (prefix `MAC-`), each with: owned files, interfaces, algorithm, tests that run with fakes, heavy acceptance commands (for the integration run), report format.

### Wave 1 — dispatch all at once

| pkg | scope | owns (only these files) | heavy run at integration |
|---|---|---|---|
| A | Config: replace hard-coded `/home/user/wty-yy/katacr` with env/config (`CLASHBOT_KATACR_ROOT`, `CLASHBOT_KATACR_WEIGHTS`) | `tools/calibrate.py`, `tools/batch_process.py`, `tools/worker_setup.sh`, `cr_perception/detect.py` (path handling only), `tests/test_config_paths.py` | full rebuild + full test suite incl. OpenCV perception tests |
| B | iPhone Mirroring capture: find the "iPhone Mirroring" window, steady-rate capture, fps benchmark | `cr_perception/sources.py` (new class), `tools/benchmark_capture.py`, `tests/test_mirroring_source.py` | capture fps (needs Screen Recording permission) |
| C | Live state builder: perception GameState/events/tracks → clean state dict identical in shape to `dataset/v1/states.jsonl` (units via `clashbot/units.py`, `render_context`) | `clashbot/live_state.py`, `tests/test_live_state.py` | parity test on one recorded segment (if the user allows the video) |
| D | d1 zero-shot ranker: implement `D1Ranker.choose` (lazy import, chosen backend) + eval on `dataset/v1/d1/joint/{val,test}` vs baselines (accuracy, ms/request) | `clashbot/ranker.py`, `tools/eval/eval_d1_zero_shot.py`, `tests/test_d1_ranker.py` | zero-shot eval |
| E | Qwen planner: `LLMPlanner` with mlx-lm, ReAct loop over the six read-only tools, step cap, never calls the executor | `clashbot/planner.py`, `tests/test_planner_llm.py` | time-to-plan on 5 decks |
| F | Executor taps: `MirroringExecutor` via Quartz events behind `--live`; overlay mode that draws intended click points on a screenshot | `clashbot/executor.py`, `tools/executor_overlay.py`, `tests/test_executor_mac.py` | overlay check only (no live click) |
| G | Live loop CLI `python3 -m clashbot.live --dry-run`: capture → perception → live state → trigger → candidates → ranker → executor, per-stage latency | `clashbot/live.py`, `tests/test_live_loop.py` (fakes for B/C/D) | dry run on the mirrored screen (Wave 4) |
| R1 | Retrieval ablation: BM25 vs structured vectors vs text-embedding channel vs RRF; exact vs HNSW `efSearch` sweep; pre- vs post-filtering | `clashbot/text_embed.py`, `tools/eval/eval_retrieval.py`, `tests/test_eval_retrieval.py` | ablation table → `docs/data/` |
| R2 | KG embeddings (PyKEEN TransE/DistMult/ComplEx/RotatE) for missing COUNTERS, equal tuning budget, random tie-breaking | `tools/eval/kge_link_prediction.py`, `tests/test_kge.py` | link-prediction table |
| R3 | Feature-interaction ranker baseline (GBDT or DCN-V2-style MLP) on the 4-way task | `tools/eval/feature_ranker.py`, `tests/test_feature_ranker.py` | ranker table |
| R4 | Opponent next-card prediction: copy/frequency baseline vs SASRec-style model | `tools/eval/next_card.py`, `tests/test_next_card.py` | accuracy table |
| R5 | Memory + knowledge editing: episodic/semantic/procedural stores, patch-versioned card facts with provenance, one balance-change edit evaluated for reliability/generalisation/locality/portability | `clashbot/memory.py`, `tools/eval/edit_eval.py`, `tests/test_memory.py` | edit-eval table |

Paper references (course PDF numbering, details in `docs/research/course_mapping.md`):
R1 — RET BM25 [32], bi-encoder limits [14], RRF [3], HNSW [20], filtering §2.2, BEIR [40]. R2 — KG TransE [2], DistMult [42], ComplEx [38], RotatE [36]; audits [34, 37]. R3 — RAG FM/DLRM/DCN-V2 [56–58]; tuned baselines [5–7]. R4 — KG CyGNet [46], RE-NET [21]; RAG SASRec [61]. R5 — AGT memory types §4.1, Mem0 [25], Reflexion [5], ACE [22], editing criteria [46], SERAC/GRACE vs ROME [53, 54, 49], ClashEval [67]. R6 — RAG [16, 17]; AGT Lost in the Middle [12], Context Rot [13]. R7 — RAG FLARE [27], Adaptive-RAG [31], Self-RAG/CRAG [29, 30].

### Wave 4 — needs Wave-1 outputs or the user at the Mac (dispatch together)
- **Calibration** of the mirrored screen (`tools/calibrate.py` on a captured frame; saves the Mac calibration + executor `Calibration`). Needs the user to open iPhone Mirroring on a match.
- **Live dry run** with G + B + C + D: latency per stage, target < 300 ms p95; decisions logged to `runs/poc/<timestamp>/`.
- **R6** planner context budget (needs E) and **R7** adaptive arbitration (needs E + G).
- After the Wave-5 fix pass: ask the user before the first live click and before the first POC game (Training Camp, then a friendly battle).

### Not now (explicitly deferred by the user)
- Fine-tuning d1 (data is ready; wait for the go-ahead).
- Bulk re-download / re-running perception for dense act/hold negatives (approved for later). A single video for the C parity test only with the user's OK.

## 6. First actions for you
1. Wave 0 steps 1–4 (setup, background downloads, one batch of user questions, all Wave-1 specs).
2. Dispatch all twelve Wave-1 packages in one message.
3. Integrate, write `docs/mac/FIX_LIST.md`, run the parallel fix pass, push, and add a STATUS entry with the result tables.
4. Then Wave 4 with the user at the Mac.
