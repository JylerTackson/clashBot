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

## 2026-11-03 (evening) — live match 2, reactive play timing, fine-tuning started

**Live match 2** (user-launched, Training Camp, MPS detector, before the reactive policy): decision latency **138 ms p50 / 283 ms p95 — target met** (detector 102 ms on MPS vs 592 on CPU); 13 plays sent, 9 confirmed in game (elixir/hand change), 6 fallback plays held (#18 working), 6 skipped because all four hand slots read unknown (greyed cards → #21 fixed: keep the last confident hand ≤ 2 s); click sequence 383 ms (#22 fixed: 50 ms gaps). The bot lost quickly: it played on every trigger, as the user noted ("it doesn't always have to place a card … save elixir and react to the opponent").
**Decisions (user).** D-039 fine-tune d1 now (joint, factorized, act); D-040 dense re-perception of the 33 pro videos for real wait/act data; D-041 reactive play timing until the learned act head is good.
**Built (Sonnet, reviewed, merged).**
- MAC-W `clashbot/policy.py`: hold by default; **defend** (enemy unit ≤ 4 s from a tower or on our half, in its lane), **leak** (≥ 9.5 elixir, away from the enemy), **punish** (opponent estimate ≤ 3 → other lane, bridge/enemy side); no spell without an enemy unit in its lane and zone; live default (`--policy reactive`). Offline at near-random moments (periodic samples, pro play rate 23–24 %): act rate **22–34 %** vs 80–92 % for the old trigger.
- MAC-T `tools/train/train_d1.py`: differentiable copy of d1's text forward (exact parity with the model's own `_forward`), token-budget batching, AdamW + warmup/cosine, best-epoch restore, temperature re-fit, save + reload check. Smoke: 1.9 rows/s, ~19 GiB MPS at the default budget (overnight runs use half the token budget × 2 accumulation).
- MAC-V `tools/data/build_dense_act.py`: one sample per second from re-run perception, built with the live state builder (train/serve parity), label = the pro deploys within 1 s; one perception match group per game (#24).
**Zero-shot d1 on the final 4-way grid** (the "before" number): **0.258 / 0.221** (chance); placement prior 0.493 / 0.384.
**Running (serialised, overnight).** Joint fine-tune (~1.5–2 h) → eval → dense perception over the 33 videos (downloading one at a time after YouTube rate-limited the parallel batch; subtitles skipped). Then dense act data → act fine-tune → factorized fine-tune.
**Next.** Fine-tuned joint numbers; a live Training Camp match with the reactive policy + fine-tuned ranker (`--ranker d1` with the new checkpoint).

## 2026-11-04 — d1-omni-600M fine-tuned on the pro decisions (joint)

**Joint fine-tune** (D-039): 3,641 train decisions, 3 epochs, all 16 trunk layers + head (313.6 M trainable), fp32 on MPS, token budget 3,072 × 8 accumulation; 3 h 41 min (first epoch 6,577 s, last 2,481 s), peak 17.3 GiB. Val accuracy by epoch 0.460 → 0.514 → **0.526** (best = last; val loss rose in epoch 3, 1.09 → 1.20, so more epochs would overfit). Temperature re-fit: `choice:3-5` = 2.0. Checkpoint `db/models/d1-ft-joint` (gitignored, reload verified).

| 4-way accuracy (2×2 grid, D-037) | val | test |
|---|---|---|
| random | 0.250 | 0.250 |
| d1 zero-shot | 0.258 | 0.221 |
| popularity | 0.468 | 0.325 |
| placement prior | 0.493 | 0.384 |
| **d1 fine-tuned (joint)** | **0.524** | **0.483** |

Latency 67–70 ms p50 / 95 ms p95 per decision (transformers fp16, MPS). Reports in `docs/data/d1_finetune_joint_{train,eval}_v1.json`. (R3 feature rankers still have to be re-run on the final grid for a like-for-like comparison.)
**Dense perception** (D-040): all 33 videos downloaded (4.2 GB, video only after YouTube rate-limited subtitles). The first overnight pass picked up only 1 video because a stale download manifest listed only it; moved aside, the full pass over all videos is running (resumable), followed by the dense act dataset build.
**Next.** Act head on the dense data; factorized ablation (≈ 3× the rows of joint — likely ~10 h, schedule for another night); R3 re-run; a live Training Camp match with `--ranker d1 --d1-backend transformers --d1-model db/models/d1-ft-joint` and the reactive policy.
**Overnight chain (detached, user request).** `runs/overnight/chain.sh` (local, gitignored) runs dense perception over all 33 videos and then builds `dataset/v1/act_dense`, in its own process session under `caffeinate -i`, so it continues without the Claude session. Progress: `tail -f runs/overnight/chain.log`; done when it ends with `DONE`; the morning live-test command is in `runs/overnight/README.txt`. The interrupted single-video pass was stopped first and its half-written `states.jsonl` removed (append-mode recorder would otherwise duplicate rows).

## 2026-11-04 (day) — like-for-like ranker comparison on the final 4-way grid

| model (4-way, 2×2 grid) | val | test |
|---|---|---|
| placement prior | 0.493 | 0.384 |
| R3 logistic regression | 0.546 ± 0.003 | 0.476 ± 0.004 |
| **R3 LightGBM LambdaRank** | **0.548 ± 0.014** | **0.491 ± 0.013** |
| R3 DCN-V2-style MLP | 0.526 ± 0.013 | 0.412 ± 0.013 |
| **d1-omni-600M fine-tuned (joint)** | 0.524 | **0.483** |

The fine-tuned d1 is on par with a tuned GBDT feature ranker (within its seed spread on test; slightly behind on val) — the RecSys lesson "tuned simple baselines are competitive" measured on our data. LightGBM's top features are train popularity of the key and p(zone | card), which d1 never sees explicitly; giving d1 those priors in the option text (or an ensemble) is the obvious next experiment. Tables in `docs/data/feature_ranker_v1.{json,md}` (supersede the deterministic-grid numbers).
**Perception (overnight chain, detached):** 17 / 33 videos (6.6 / 12.0 match-hours) at 09:49; ≈ 0.7 match-hours per hour → done ≈ 17:30–18:00. The user chose to wait for it before the next live test.
**In progress:** MAC-X (act-head sampler, learned play/wait policy, offline eval with a reaction-to-opponent slice).

## 2026-11-05 — dense wait/act dataset built (D-040)

The detached overnight chain finished on 2026-11-04 at 17:47 (perception 00:04 → 17:46, 33 / 33 videos; dataset build 43 s). `dataset/v1/act_dense` (gitignored): one sample per second of every pro game, label = the pro deploys a card within the next 1.0 s, state rendered by the live `LiveStateBuilder`; state sidecars aligned with the act files.

| split | games | lines | act rate | act rate, threat on board | act rate, ≤ 3 s after an opponent play |
|---|---|---|---|---|---|
| train | 142 | 29,036 | 0.196 | 0.230 | 0.215 |
| val | 27 | 5,560 | 0.233 | 0.256 | 0.259 |
| test | 28 | 5,279 | 0.217 | 0.249 | 0.234 |
| total | 197 | 39,875 | 0.204 | 0.236 | 0.223 |

All 197 games of `games.jsonl` are covered (no game skipped for missing time or perception); 3,540 sample times (8 %) had no perception row within 0.6 s and were dropped (test 17 %, from gaps in a few videos). 176 game windows also contained rows of a neighbouring match; only the dominant match was used (FIX_LIST #24). Per game: 56–361 samples, pro act rate 0.04–0.45, no game without plays. Pros wait about 80 % of the seconds, and only slightly more often play right after an opponent play or when a unit threatens a tower: the act head has to learn *when*, not just *whether*, to react.
**Next.** Live Training Camp test with the fine-tuned d1 + reactive policy (command in `runs/overnight/README.txt`); then `runs/overnight/NEXT.sh` (sample → act-head fine-tune → offline eval → `--policy learned` live command).

## 2026-11-05 (evening) — live match 2, fixes, tactical engine (D-042/43/44)

**Live match 2** (`runs/poc/20261105-172506`, fine-tuned d1 + reactive rules, lost in about 1 min of play): 15 of 27 decisions were skipped because the hand read `?` (greyed cards are not recognized); spells landed on the pro median tile, not on units; 10 elixir went in the first 15 s. Only 3 of 22 moves came from kNN: 105 / 197 pro games share none of the user's cards. Fixed in MAC-Y (#26–#29): positional hand memory with deck-cycle inference, no taps from an unread hand, damage spells aimed at the best worthwhile cluster (with lead) and never on a lone tank.
**User decisions.**
- D-042: a tactical engine, with d1 reasoning over lookup plus tactics.
- D-043: perception may be fixed from specs.
- D-044: switch to the KB's best-covered deck, Goblin Barrel bait with Magic Archer (13 exact pro games, 43 sharing ≥ 5 cards).
**Tactical engine (MAC-Z1–Z3).**
- `clashbot/card_roles.py` + `resources/card_roles.json`: 122 cards; 90 stats-derived from RoyaleAPI cr-api-data, 12 deliberate manual overrides (chain/pierce/multi-unit cards).
- `clashbot/tactics.py`:
  - groups enemy units into pushes per lane, with needs (air / tank / swarm / building-targeter);
  - scores hand counters by role, KG and elixir trade;
  - places them relative to the threat (building pull tile, melee in front, ranged behind the tower, kamikaze ahead, spells on the cluster);
  - proposes counter-push / punish / leak moves.
- The live loop gives d1 the `tactics:` lines in its context and per-option notes (tactics reason vs "pros played this in similar states" vs "known counter of …"). Tactical moves keep their own tile. `--ranker tactics` is a baseline, and `--no-tactics` turns the engine off.

**Offline check vs the pro plays** (`docs/data/tactics_eval_v1.md`):
- The engine is only marginally better than chance at naming the pro's card when a push is on (top-1 19 % vs 17 % for a random affordable card). It is at chance on the 4-way grid (27.5 % / 25.5 % val / test vs 25 % random, 38–49 % placement prior, 48–52 % fine-tuned d1).
- These numbers are capped by perception: in the pro data the hand is fully read in only 34–37 % of moments, and the pro's card is in the visible affordable hand only 47 % of the time.
- Agreement with pros is therefore not the engine's yardstick. Two things are: whether it plays sensibly live, and whether d1 trained with its notes beats d1 without them.
**Next.**
- `dataset/v1/d1/joint_tac`, then fine-tune `d1-ft-joint-tac` from `d1-ft-joint` (after the act head; GPU serialised).
- Live Training Camp matches with the new deck: lookup-only (`--no-tactics`) vs `--ranker tactics`.
- Merge MAC-PF (perception fixes).

## 2026-11-05 19:22 — learned play/wait head (MAC-X heavy run)
`d1-ft-act`: initialised from `d1-ft-joint`, top 8 layers, 2 epochs on 8,405 balanced dense lines (1 h 45 min). Train loss barely fell (0.696 → 0.676, chance = 0.693). Val threshold 0.50.

| test (n = 3,221, pro act rate 0.206) | balanced acc | act rate | act recall | wait recall |
|---|---|---|---|---|
| learned head | **0.558** | 0.429 | 0.522 | 0.595 |
| reactive rules | 0.535 | 0.450 | 0.507 | 0.564 |
| combined (rules keep defend / leak) | 0.545 | 0.540 | 0.611 | 0.478 |

Only marginally better than the rules. Both act about twice as often as the pros. The 1 s label ("the pro deploys within the next second") is very noisy for text-only states, so this head is not worth its 70 ms in the live loop yet. Candidate next step: a structured-feature model (GBDT on elixir, opponent elixir, threats/ETAs, time since the opponent's last play, hand costs) with 1 / 2 / 3 s horizons. Tables in `docs/data/act_head_v1.md`.

## 2026-11-05 (night) — live matches A/B with the new deck, and the fixes they led to
| | match A (`20261105-193555`): LLM + lookup, `--no-tactics` | match B (`20261105-194101`): LLM + lookup + tactics context |
|---|---|---|
| length | ~1 min 30 s | ~2 min 10 s |
| damage dealt | one enemy tower −830 | both enemy towers −830 |
| LLM picks by source | kNN / KG, 11 fallback holds | 55 / 80 tactical, 18 kNN, 7 KG |
| taps / elixir they cost / ≈ available | 42 / 62 / 38 | 80 / 112 / 55 |
| ice-spirit taps | 26 | 57 |

The LLM uses the tactics context even untrained on it. Both losses had the same causes:
- permanent defend mode (any unit on our half);
- spending every elixir on the cheapest card;
- an own-elixir reading that lags our taps (most taps failed);
- the engine itself recommending ice spirit vs a balloon (trade without stopping power).

**Fixes merged:**
- MAC-AC C1 (#32/#33):
  - real threats only (no retreating / far / unknown units);
  - the engine judges whether an adequate answer is affordable, else the bot saves for it;
  - the LLM picks only among adequate cards.
  - Offline act rate 0.42 → 0.20, periodic accuracy 0.65 → 0.70.
- MAC-AC C2 (#34): local own-elixir accounting after taps, failed-tap detection, a 2 s cooldown per failed card.
- MAC-AB A2 (engine):
  - bait / pressure / support / cycle intents;
  - stopping power and already-answered pushes;
  - weights tuned on pro data with design guards (val objective 0.513 → 0.532, test 0.461 → 0.476);
  - no-push coverage 11–16 % → 36–42 %.
- MAC-AB A3 (#31): phantom opponent plays 9 → 2 on the 90 s clip.
- MAC-AB A1: a fast play/wait model (LightGBM, 29 features):
  - test balanced accuracy 0.607 vs rules 0.533 at h = 2 s, and 0.598 vs the d1 head's 0.558 at h = 1 s;
  - 0.05 ms per decision;
  - `--policy fast`.
- MAC-AB A4: the R7 arbiter. d1 always decides; the arbiter only narrows its options when it is unsure (D-045). There is no gain on the current d1 probabilities (test 48.3 % either way), so it is off by default.

**D-045:** the LLM makes every play decision; the engine, lookup, timing model and arbiter are its tools.
**Next:**
- Z3.10, card-level tactics notes. The pro card is among the engine's cards on 23 % of lines, but exact-key notes reach only 5 %.
- Rebuild `joint_tac`, then the overnight fine-tune `d1-ft-joint-tac` (`runs/overnight/NEXT2.sh`).
- A live match with all fixes.

## 2026-11-05 22:28 — d1 fine-tuned with the tactics context (`d1-ft-joint-tac`)
Fine-tuned from `d1-ft-joint` for 2 epochs on `dataset/v1/d1/joint_tac`: state plus `tactics:` lines, and per-option notes (tactics reason / card-level tactics note / pro lookup / counter). Engine as of MAC-AB A2 + Z3.10.

| 4-way accuracy | val | test |
|---|---|---|
| placement prior | 0.493 | 0.384 |
| R3 LightGBM (best feature ranker) | 0.548 | 0.491 |
| `d1-ft-joint` on `joint` (no tactics) | 0.524 | 0.483 |
| `d1-ft-joint` on `joint_tac` (notes never seen in training) | 0.480 | 0.470 |
| **`d1-ft-joint-tac` on `joint_tac`** | **0.540** | **0.497** |

Training the LLM on the tactics context gains +1.6 / +1.4 points over the no-tactics model, and puts it slightly above the best feature ranker on test. Without training, the notes hurt (−4.4 / −1.3), as expected from the train/serve mismatch. Latency is 62 ms p50 / 93 ms p95 per decision.

**Caveat:** MAC-AD (merged after this dataset build) changes some engine lines: pushes are listed from 10 s, and "answered by" now counts our buildings and recent plays. Rebuild `joint_tac` before the next fine-tune (d1-2B).
