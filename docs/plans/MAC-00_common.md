# MAC-00 — rules shared by every Mac Wave-1 package (read this first)

Author: Opus (Mac session) · 2026-11-02 · Applies to MAC-A … MAC-G and MAC-R1 … MAC-R5.
Decisions referenced: `docs/decisions/DECISIONS.md` (D-001 … D-037).

**You are implementing, not designing.** Your package spec names the exact files, signatures, algorithms and tests. If something is unclear or contradicts the code, do the smallest reasonable thing, write it under "deviations" in your report, and keep going. Do not redesign.

## 1. Environment
- Machine: Apple M5, 32 GB, macOS 26.5. Python **3.12.8** venv: `PY=/Users/tylerjackson/Projects/clashbot/.venv/bin/python`. Always run Python as `$PY` (never the system/anaconda `python3`). Do not `pip install` anything; if you need a package that is missing, report it.
- Installed: torch (MPS), torchvision, numpy, scipy, opencv-python-headless, pillow, onnxruntime, rapidocr-onnxruntime, mss, pyobjc Quartz, ultralytics 8.1.24, hnswlib, tokenizers, pytest, mlx, mlx-lm, mlx-vlm, transformers, sentence-transformers, pykeen, lightgbm, soundfile.
- You work in **your own git worktree** (the current directory when you start). The main checkout is `/Users/tylerjackson/Projects/clashbot/clashBot` — treat it as **read-only**.
- Built data lives only in the main checkout (gitignored). To run real-data tests or smoke runs in your worktree, create two symlinks at your worktree root (they are git-excluded, never add them):
  ```bash
  ln -s /Users/tylerjackson/Projects/clashbot/clashBot/dataset dataset
  ln -s /Users/tylerjackson/Projects/clashbot/clashBot/db db
  ```
  Then `clashbot.config.DATASET_DIR` / `DB_PATH` resolve to the real data. **Never** run `tools/data/*`, `tools/db/seed.py` or anything that writes into `dataset/` or `db/` through these links. Write smoke outputs under `runs/smoke/<package>/` in your worktree (`runs/` is gitignored).
- Model weights already in the Hugging Face cache (`~/.cache/huggingface`): `LiquidAI/d1-omni-600M`, `LiquidAI/d1-omni-600M-MLX-fp16`, `mlx-community/Qwen3-4B-Instruct-2507-4bit`, `BAAI/bge-small-en-v1.5`. KataCR: repo at `~/cr_scratch/katacr`, weights `~/cr_scratch/katacr_d1.pt`, `~/cr_scratch/katacr_d2.pt`. Do not download other models.

## 2. Hard rules
1. **Own only your files.** Create/edit only the files your spec lists under "Owns". Shared files (`requirements*.txt`, `README.md`, `.gitignore`, `docs/**`, `clashbot/config.py`, any file another package owns) are edited by Opus at integration — list what you need in your report.
2. **Read-only perception, no input.** Nothing you write may send mouse/keyboard input, except `clashbot/executor.py` (package F) behind the double live guard (D-031). Never run anything with `--live`, never set `CLASHBOT_ALLOW_LIVE`.
3. **No secrets.** Never print, log or commit cookies, tokens or keys.
4. **Heavy jobs are serialised.** Unit tests must run with small fakes in seconds and must not load real models. A real-model or real-data smoke run is allowed only where your spec says so, once, with the `--limit` it gives. Full evaluations are run by Opus at integration.
5. **Tests**: put tests in the file your spec names; use `pytest`; real-data tests are skipped when the data is missing (`pytest.mark.skipif(not path.exists(), ...)`), real-model tests are marked `@pytest.mark.slow` **and** skipped unless env `CLASHBOT_SLOW_TESTS=1`. Follow the import pattern of the existing tests (`ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))`).
6. **Style**: match the surrounding code — `from __future__ import annotations`, type hints, short module docstring, no new framework/abstraction beyond the spec, comment density like `clashbot/*.py`.
7. **Time box.** If a test still fails after three fix attempts on the same failure, stop fixing, commit what you have, and report the failing output. Broken work is expected; a fix pass follows.

## 3. Run before you commit
```bash
$PY -m pytest -q <your test file(s)>
$PY -m pytest -q tests -x -q -p no:cacheprovider 2>&1 | tail -5     # whole suite must not get worse
git status --short                                                  # only your owned files
```

## 4. Commit (in your worktree, on its branch)
Stage only your owned files (`git add <paths>`; never `git add -A`). One commit, message:
```
MAC-<pkg>: <one-line summary>

<2–5 lines: what was built, test status>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VaQ71W4LvG8WdXhES2BAQy
```
Do not push. Do not merge. Do not touch other branches.

## 5. Report back (≤ 40 lines, this order)
1. Branch name and commit hash (`git rev-parse --abbrev-ref HEAD; git rev-parse HEAD`).
2. Files created/edited.
3. Test results: the last lines of each pytest run (pass/fail/skip counts); paste failing assertions verbatim.
4. Smoke run output (if your spec has one), trimmed.
5. Dependencies or shared-file edits Opus must make.
6. Deviations from the spec and why.
7. Open issues / anything you could not finish.
