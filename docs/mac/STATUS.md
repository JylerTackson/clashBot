# Mac session status

One dated section per milestone (brief §4). Decisions: `docs/decisions/DECISIONS.md`. Specs: `docs/plans/MAC-*.md`.

## 2026-11-02 — Wave 0: setup, user questions, Wave-1 specs

**Machine.** Apple M5, **32 GB** (not 16 GB, D-025), macOS 26.5 (25F71), Python 3.12.8 venv at `../.venv`, Homebrew + Xcode toolchain.
**Installed.** torch 2.15 (MPS on), torchvision 0.29, numpy 2.5, OpenCV 5.0, ultralytics 8.1.24, hnswlib 0.8 (needs Apple clang; Anaconda's clang fails), tokenizers 0.23, mss 10.2, pyobjc Quartz, onnxruntime, rapidocr, mlx 0.32, mlx-lm 0.32, mlx-vlm 0.7.11, transformers 5.22, sentence-transformers 6.2, PyKEEN 1.11, LightGBM 4.7 (+ `brew install libomp`).
**Downloaded (outside git).** KataCR repo + `katacr_d1.pt`/`katacr_d2.pt` in `~/cr_scratch`; d1 tokenizer in `db/hf/`; HF cache: d1-omni-600M-MLX-fp16, d1-omni-600M, Qwen3-4B-Instruct-2507-4bit, bge-small-en-v1.5.

**Rebuild on the Mac** (all builders per `docs/PHASE3_OVERVIEW.md`): identical to the cloud — 10,923 states / 6,658 train-eligible / 197 games; KG 590 nodes, 12,030 edges; 3,654 documents; ranker joint 3,764 / 771 / 780; d1 joint/factorized/act exported.

| check | result |
|---|---|
| full test suite (incl. OpenCV perception tests, first run ever) | **112 passed**, 43 s |
| fast loop replay `LGb0mz8Sb8w-m0.0 --limit 15` | 29.8 ms p50 / 33.2 ms p95 per decision (cloud: 41 / 60) |
| candidate recall@10 exact key, val/test | kNN 0.366 / 0.317, popularity 0.503 / 0.408 |
| card-level recall@10, val/test | kNN 0.722 / 0.651, popularity 0.677 / 0.613 |
| 4-way baselines, val/test | placement prior 0.375 / 0.363, popularity 0.309 / 0.254, random 0.25 |

**User decisions (one batch).** d1 backend: both, MLX fp16 default (D-026). Planner: Qwen3-4B-Instruct-2507 4-bit (D-027). Installs: all four groups approved (D-028). One video for the parity test: yes (D-029).
**Defaults taken (Opus).** D-030 KataCR env paths · D-031 double live guard · D-032 eval outputs · D-033 bge-small embedder · D-034 synthetic edit · D-035 executor geometry · D-036 package ownership.

**Finding (D-037, proposed).** The joint 4-way options give the answer away: a rule that only looks at the option keys scores **0.85** on train/val/test, because the WP4 negatives N2/N3 are built around the positive. Fine-tuning d1 on `d1/joint` as is would learn the shortcut. Every 4-way eval now reports this `option_set_shortcut` row; the fix goes to the user.

**Wave-1 specs.** `docs/plans/MAC-00_common.md` (shared rules) + MAC-A config paths, B mirroring capture, C live state, D d1 zero-shot, E Qwen planner, F executor, G live loop, R1 retrieval ablation, R2 KG embeddings, R3 feature ranker, R4 next card, R5 memory + editing.

**Next.** Dispatch all 12 packages (Sonnet, isolated worktrees), integrate one by one, `docs/mac/FIX_LIST.md`, parallel fix pass, heavy runs (serialised), push.

## 2026-11-03 — Waves 1–3: 15 packages built, integrated, reviewed and fixed

**Done.** Wave 1 dispatched all 12 brief packages at once (Sonnet, isolated worktrees): A config paths, B iPhone Mirroring capture, C live-state builder, D d1 zero-shot ranker, E Qwen planner, F Mac executor, G live loop, R1–R5. Three more came out of the review: **H** 2×2 grid negatives (D-037), **P** perception rate gating, **Q** live perception latency (D-038). Everything is merged on `claude/clash-royale-kb-phase-1-4rtfm6`. `docs/mac/FIX_LIST.md` has 16 findings: 15 fixed through two parallel fix passes and 1 documented (won't fix).
**Tests.** The full suite on the merged code before the last four fixes gave **299 passed, 2 skipped** (the 2 opt-in real-model tests). Each later fix branch was green on its own (P 302, H2 309, G2 and Q green). The final combined run is pending: the memory guard stopped it (see "Stopped" below).

### Key findings (review)
- **D-037: the 4-way options leaked the answer.** With the WP4 negatives, a rule that reads only the option keys scored **0.85**. Rebuilt as a 2×2 card × placement grid, it is now 0.25 by construction. A second, smaller leak from my first grid spec (a fixed lane order made "right" a tell; lane-only rule 0.31) was fixed by randomising the second placement (lane-only rule 0.27).
- **Perception rate bug (pre-existing, #12).** OCR, deploy labels and detection were gated on the source's absolute frame index. With a strided video source, clock OCR and labels never ran. Fixed: gates count processed frames.
- **Live loop.** The trigger re-fired on consecutive frames (692 decisions in 1,499 frames). That is fixed with a 1 s minimum gap and handling each opponent play once. The decision latency target was missed (below), so the hand-read cache and background OCR were added.
- **Data hygiene.** Three old tests rebuilt `dataset/v1` in place, and parallel agents raced on those files (JSONDecodeError). All test builds now use temp dirs, and the dataset was rebuilt serially afterwards.

### Results (val / test; tables in `docs/data/*_v1.*`)
**4-way task, randomised grid** (n 761 / 766; train 3,641; 3% of examples skipped where no grid exists)

| reference | val | test |
|---|---|---|
| random | 0.250 | 0.250 |
| option-set shortcut (leak check) | 0.250 | 0.250 |
| lane-only rule (leak check) | 0.266 | 0.272 |
| (lane, zone)-only rule (mostly the real placement prior) | 0.353 | 0.345 |
| popularity | 0.468 | 0.325 |
| **placement prior** (bar to beat) | **0.493** | **0.384** |

**d1-omni-600M zero-shot.** Measured on the deterministic grid, before the lane fix; the re-run on the final grid is pending. Joint **0.273 / 0.238** (chance level). Factorized: card 0.134 / 0.162, lane 0.369 / 0.355, zone 0.354 / 0.332. d1 has no usable Clash Royale knowledge out of the box, which is what fine-tuning is for (D-022, your go-ahead). Latency per joint request: MLX fp16 47 / 44 ms p50 (p95 139 / 102), transformers MPS 61 / 63 ms. The two backends agree on 99.7% of top answers (max Δp 0.013).
**R3 feature rankers.** Also measured on the deterministic grid, so inflated by the lane tell; the re-run is pending. logreg 0.629 / 0.588, LightGBM 0.617 / 0.560, DCN-V2 0.593 / 0.470. Top LightGBM features: key popularity and p(zone | card).
**R1 retrieval** (n 916 / 902, train pool, bge-small text channel)

| list | hit@10 exact key | card hit@10 | ms p50 |
|---|---|---|---|
| BM25 | 0.142 / 0.159 | 0.396 / 0.483 | 24 |
| structured | 0.166 / 0.136 | 0.434 / 0.412 | 1.3 |
| text (bge-small) | 0.132 / 0.143 | 0.397 / 0.436 | 21 |
| struct + BM25 (runtime) | 0.138 / 0.160 | 0.434 / 0.487 | 26 |
| all three | 0.156 / 0.170 | 0.461 / 0.514 | 47 |

The channels are within noise of each other. The text channel adds about 20 ms per query for no clear gain, so it is not integrated (D-033). HNSW at ef 40 reaches recall@10 0.978 in 0.31 ms against 1.5 ms for exact search; exact search is affordable at 10.9 K vectors. Deck pre-filter: for 89% of val/test queries the pool is empty (their decks never occur in train). Post-filtering at K = 10 returns nothing for 50% (deck) and 87% (phase) of queries.
**R2 KG embeddings, held-out COUNTERS** (filtered, realistic; equal budget; 3 seeds): RotatE **MRR 0.159, H@10 0.326**; ComplEx 0.116; DistMult 0.091; TransE 0.082; popularity 0.110 / 0.239; random 0.022 / 0.113. Dropping the near-inverse ANSWERED_BY edges changes nothing (RotatE 0.160). DistMult is symmetric by construction (asymmetry score 0).
**R4 opponent next card** (acc@1): **copy-frequency 0.207 / 0.243**; SASRec 0.129 / 0.222 (± 0.01); the cycle rule never helps (best w = 0, because perception misses plays); popularity ≈ 0 (opponents vary across games). A tuned simple baseline wins.
**R5 knowledge editing.** Synthetic +1-elixir edits on 5 cards (D-034), pooled. Non-parametric route vs no edit: reliability 1.00 / 0.00, generalization 1.00 / 0.00 (n 465), locality 1.00 / 1.00, portability 1.00 / 0.07; re-weighting cuts the share of hits on edited cards from 0.124 to 0.002; versioning and rollback work.
**Planner (Qwen3-4B 4-bit, MLX).** Pregame 2.6–3.6 s, update 2.4–3.0 s, plans are plain text of 5 lines. Since the format fix it makes **0 tool calls**: it answers from the primed notes. Revisit in R6/R7.
**Live state.** Round trip on 400 dataset records: own/clock/phase fields identical, median feature cosine 1.0. End-to-end parity on the re-downloaded game (Mac perception vs the cloud-built dataset, D-029), after fix #12: phase 0.62, towers 0.77, elixir ±0.5 0.55, hand slots 0.57, units Jaccard 0.34, median feature cosine 0.67. The remaining gap mixes perception reproducibility (different label/detect cadences from the cloud run) with builder differences.
**Live dry run on recorded footage** (d1 ranker, KataCR on MPS, before fixes #14/#15): 171 ms p50 / **471 ms p95** per decision (target < 300). Perception was 87 / 386 ms: the hand reader took 77 ms every frame, clock OCR ≈ 290 ms and label OCR ≈ 250 ms on their frames. Candidates 37 ms, d1 rank 44 / 76 ms. Calibration from 6 towers: residuals 0.04–0.11 tiles.

### Decisions this milestone
D-037 decided by you (2×2 grid) plus an implementation note (randomised placement). D-038 default (Opus): cached hand reads, background OCR in live mode, 1 s minimum decision gap.

### Stopped (system memory guard, 2026-11-03 ~01:00, while the session was idle)
Claude Code stopped two queued background chains when the Mac ran critically low on memory. Not restarted, per the guard's rule. Not yet run:
1. d1 zero-shot (MLX + transformers) and R3 on the final randomised grid.
2. Full test suite on the final merged code.
3. Two live dry runs on the recorded game with fixes #14/#15 (d1 and score rankers).
The rebuild of `dataset/v1/ranker`, `act` and `d1` with the randomised grid did finish.

### Git note
`git pull --rebase` before the first push flattened my local `--no-ff` merge commits into a linear history. The tree was identical to the tested one, and the remote only fast-forwarded (no force push). Later merges keep `--no-ff`; pushes now use `git pull --rebase=merges`. Agent worktrees start from the last pushed commit, so I push before dispatching.

### Next
Re-run the stopped jobs (your OK); then Wave 4 with you at the Mac: iPhone Mirroring capture benchmark, calibration of the mirrored window, live dry run on the mirrored screen. Before any live click: your go-ahead, `--live` + `CLASHBOT_ALLOW_LIVE=1`, Training Camp first. Until d1 is fine-tuned, the POC should use `--ranker score` (kNN + KG order), because zero-shot d1 is at chance.

## 2026-11-03 — Wave 4 begins: POC on the mirrored iPhone (Training Camp)

**Done.** Capture of the iPhone Mirroring window works (Screen Recording already granted): 30 fps sustained. Switched to Quartz window-ID capture (#17) for clean 2× frames (632×1392, 22–38 ms/grab) instead of 1× region grabs with other windows bleeding through. Calibration from all 6 towers (residuals ≤ 0.18 tiles); HUD ROIs line up; executor overlay shows tap points on the right card slots and tiles (`runs/poc/mirror_overlay.png`, local). User deck: knight, fireball, zap, ram-rider, musketeer (hero), electro-dragon (evo), princess (evo), tesla.
**Dry run on the mirrored screen** (2 min, ranker = score, no input): 436 frames, 59 decisions (min gap working), 58 planned two-tap plays; hand / clock / phase / opponent deck / units read correctly. Latency 136 ms p50 / 672 ms p95 — the detector ran on **CPU** (592 ms per detect) because `--device` was not given; background OCR and the hand cache worked (OCR 0.2 ms on the main thread, 62% cache hits). 4 of 59 picks were fallback candidates (e.g. fireball on our own back) → fixed: hold instead (#18), mps default (#19).
**First live run** (launched by the user, Training Camp, 60 s, before #18/#19): 26 two-tap plays sent. For the first ~30 s macOS blocked the clicks until the user allowed them (the bot repeated the same play because nothing changed); from t ≈ 32 s **the taps work**: every verifiable play registered (knight elixir 10 → 7, musketeer 7 → 4, fireball 4 → 2, princess 3 → 2, each card cycling out of the hand). iPhone Mirroring accepts the executor's Quartz clicks as is (#20: not a bug). Observed: the executor's click sequence takes ~390 ms (two clicks with 120 ms gaps), and the hand reads all four slots as unknown on some low-elixir frames (greyed cards).
**Tests.** Full suite on the merged code: 330 passed, 2 skipped (before the last two small fixes; their branches were green).
**Next.** A full Training Camp match live with the merged fixes (mps detector, hold on fallback) to measure decision latency on MPS and plays per minute; then decide on shortening the click gaps and handling greyed hand cards.
