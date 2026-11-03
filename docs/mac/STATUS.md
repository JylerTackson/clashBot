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
