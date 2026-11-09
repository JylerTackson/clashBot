# MAC-AV — a Docker build the grader runs with one command, showing the agent end to end (spec for one Sonnet implementer)

Author: Opus · 2026-11-09 · Due: code submission to Canvas in about 23 h. Read `docs/plans/MAC-00_common.md` first. Code freeze D-051 holds. Allowed: portability guards that leave behaviour on the Mac unchanged, and new packaging files. Not allowed: decision-logic changes.

User:
- *"The grader will get it because I will upload a zip file to canvas."*
- *"we have to assume it will not be an apple silicone."*
- *"We need to just show it running end to end with the build; we don't need to include all the docs, benchmarks, etc. in the image, just need to show the end 2 end build with the image so the grader can see it working."*

## Goal
The grader unzips the submission and runs one command: `docker compose up --build`.
- The image builds natively on their machine: linux/amd64, arm64, Windows or an Intel Mac.
- It then runs the **live stack in dry-run on a bundled video clip**: perception (KataCR YOLOv8 on CPU, OCR, tracker) → live state → policy (rules + LightGBM) → tactics engine + KG → candidates (kNN/BM25/RRF retrieval, KG counters, engine moves) → d1 ranker → executor (dry run, taps logged).
- It also starts the decision dashboard (`clashbot.viz`) following that run at `http://localhost:8765`.
- The grader opens the page and watches decisions appear, with screenshots, candidates and probabilities, retrieved pro states, the KG view and the reasons.
- When the clip ends, the container prints the run summary and the dashboard keeps serving the finished run until Ctrl-C.
- No live taps; macOS / iPhone Mirroring is out of scope for Docker.

## Owns
```
docker/Dockerfile              docker/requirements-docker.txt     docker/entrypoint.sh
docker-compose.yml (repo root) scripts/make_docker_submission.sh  docs/DOCKER.md
clashbot/ and cr_perception/ (ONLY: import guards so the video path runs on Linux without Quartz / mss / mlx / pyobjc;
                               behaviour on the Mac must not change)
tests/test_docker_packaging.py (new)
```

## Contents of the submission zip (`dist/clashbot_docker_submission.zip`)
Built by `scripts/make_docker_submission.sh [--with-d1]` from `git archive HEAD` plus the artefacts below. No docs/report, no benchmark outputs, no `runs/` except the clip and its config.
- **Code:** `clashbot/`, `cr_perception/`, `tools/` (only what the live path and viz import), `knowledge_base/` or resources the runtime reads, `configs/`.
- **Docker files:** `docker/`, `docker-compose.yml`.
- **README:** a short `README.md` for the zip with the one command, what the grader will see, the hardware note (≥ 6 GB RAM for Docker; about 5–10 min first build) and the live-play note.
- **Runtime artefacts** (copied under `artifacts/` in the zip; measure each and list the sizes):
  - `db/clashbot.sqlite`. If it is large, a slimmed copy holding only what `StateIndex` and `KnowledgeGraph` read: states metadata + FTS for BM25 + `kg_edges`. Drop the `documents` table if unused on this path.
  - `db/state_features.npy` and `db/state_features.meta.json`.
  - `db/kg2.sqlite`.
  - `db/models/act-gbdt/`.
  - The calibration (`configs/mirror_calib.example.json`, or the per-video calib that matches the clip).
  - KataCR weights `katacr_d1.pt` and `katacr_d2.pt` (MIT, 87.8 MB each).
  - The one KataCR source file the checkpoints need to unpickle (`katacr/yolov8/custom_model.py` and its package `__init__` files, from github.com/wty-yy/katacr at commit `36ceb9f`), plus KataCR's LICENSE.
- **The clip:** 90 s cut from `data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4`, starting at 60 s, the segment every dry run has used. Re-encode to 720p H.264, about 10–20 MB, with cv2 (no ffmpeg dependency at runtime).
- **d1 (optional):**
  - `--with-d1` adds `db/models/d1-ft-joint-tac` converted to **fp16 safetensors** (about 1.2 GB) with its code files, tokenizer and the LFM Open License file. Convert into a temp dir and never modify `db/models`.
  - Without the flag the zip has no d1. In that case the entrypoint uses `D1_URL` (env or `.env`) to download and unpack the model at container start if it is set. Otherwise it falls back to `--ranker tactics` (the engine picks moves) and prints a clear banner: "d1 weights not present — the tactics engine is choosing moves; see README to add d1".

## Image
- `python:3.12-slim` (multi-arch).
- apt: `build-essential` for hnswlib, `libgl1`, `libglib2.0-0` for OpenCV.
- **Requirements:** `docker/requirements-docker.txt` = `requirements-mac.txt` minus the macOS-only packages (mlx, mlx-lm, mlx-vlm, pyobjc, mss), with `opencv-python-headless` instead of `opencv-python`. Install torch and torchvision **CPU** wheels from `https://download.pytorch.org/whl/cpu`, pinned to the versions in `requirements-mac.txt` (or the closest CPU wheel available for both amd64 and arm64; say which).
- Keep the layer order cache-friendly: deps first, then artefacts, then code.
- `ENV` sets `HF_HUB_OFFLINE=1`, `CLASHBOT_KATACR_ROOT`, `CLASHBOT_KATACR_WEIGHTS` and `CLASHBOT_TORCH_DEVICE=cpu`.
- **Entrypoint** (`docker/entrypoint.sh`):
  1. Start `python -m clashbot.viz --follow /data/runs --port 8765` (bind 0.0.0.0 inside the container) in the background.
  2. Run `python -m clashbot.live --source video --video /app/artifacts/clip.mp4 --calib … --deck <the clip's pro deck> --ranker d1 --d1-backend transformers --d1-model /app/artifacts/d1-ft-joint-tac --policy fast --act-model /app/artifacts/act-gbdt --record-shots --out /data/runs/docker_demo_<ts>`. Use the tactics fallback above when d1 is absent. Use the deck the clip's pro plays (take it from `dataset/v1/games.jsonl` for the game behind the clip) so hand reading and candidates make sense.
  3. Print the summary and keep the viz server in the foreground.
- **docker-compose.yml:** one service `clashbot`, `build: {context: ., dockerfile: docker/Dockerfile}`, port `8765:8765`, a named volume `clashbot-runs:/data/runs` (outputs persist), `mem_limit` unset. Also a `command` override documented for `docker compose run --rm clashbot tests`, which runs a small CPU test subset.
- **The viz server** must bind to 0.0.0.0 when `--host 0.0.0.0` is passed. Add that flag if missing; the default stays 127.0.0.1 on the Mac.

## Portability (Linux)
Make `python -m clashbot.live --source video` and `python -m clashbot.viz` import and run without Quartz, mss, pyobjc or mlx. Guard those imports (lazy, or try/except inside the mirroring / executor / MLX code paths). The video path must not import them. `--live` and `--source mirroring` on Linux exit with a clear message. The full suite stays green on the Mac.

## Verify
1. On this Mac (arm64), build and run with `docker compose up --build`. Fetch `http://localhost:8765/api/runs` and `/api/run/<id>/decisions?since=0` while the run progresses. Report the build time, image size, run wall time for the 90 s clip, decisions and the latency table.
2. **amd64**, since the grader is likely not on Apple Silicon:
   - `docker buildx build --platform linux/amd64 -f docker/Dockerfile -t clashbot:amd64 --load .` (QEMU emulation, slow);
   - then run `docker run --platform linux/amd64 … tests` and a 20 s smoke run.
   - Report whether it works and how long it took.
3. Test the zip from a clean temp dir: unzip it, then `docker compose up --build` with no access to the repo. Cover both the `--with-d1` zip and the no-d1 zip, the latter with the fallback banner.

Rules:
- Docker builds and runs are heavy. Wrap each in `/usr/bin/lockf /private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/heavy.lock <cmd>`, because another benchmark measures latency on this Mac.
- Clean up containers, images and build cache you no longer need. Keep the final images.
- Never run `--live`.

## Report
- Branch, commit and files.
- Zip sizes with and without d1, plus the artefact list with sizes.
- Image size, build times (arm64 / amd64), run time and decisions.
- What the grader sees.
- Deviations.

## Amendment 1 (user, 2026-11-09 ~01:30): d1 from Hugging Face at build time; a native live path for a Mac grader
**User:**
- *"we will host it on my hugging face account … just make it part of the build to use the weights so when he builds the images he also downloads the weights."*
- *"In the case that they do have a mac and the game downloaded ready to test we also need to make sure we are giving them the option to play the game live on the agent."*

**d1 weights.** Public repo `Jyler/clashbot-d1-ft-joint-tac`, pinned revision `5b984f0cce7c6db6e8a768cc0df2d6836022a3e2`, fp16 safetensors, 1.17 GB. It holds the code files, tokenizer, LFM license and model card. It picks the same option as the fp32 original on 100 of 100 check lines.
- In the Dockerfile, download it at **build time** into `/app/artifacts/d1-ft-joint-tac`:
  ```
  RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Jyler/clashbot-d1-ft-joint-tac', revision='5b984f0cce7c6db6e8a768cc0df2d6836022a3e2', local_dir='/app/artifacts/d1-ft-joint-tac')"
  ```
  Do it in its own layer, after the deps and before the code, so code edits don't re-download it. No token is needed. Keep `HF_HUB_OFFLINE=1` for runtime only, set after this layer.
- Drop the `--with-d1` zip variant: the zip never contains d1.
- Keep the tactics fallback only for a failed download, with the banner.

**Native live path (macOS + iPhone Mirroring + Clash Royale).** Docker cannot reach the macOS window server, so add `scripts/setup_mac.sh` to the zip. It must be idempotent and must not touch any existing `.venv` unless asked. It:
1. Checks for macOS on Apple Silicon or Intel, and Python 3.12.
2. Creates `.venv` in the unzipped folder and installs `requirements-mac.txt`.
3. Downloads d1 from the same HF revision into `db/models/d1-ft-joint-tac`.
4. Copies the zip's artefacts (db subset, kg2, act-gbdt, KataCR weights and source) into the paths the code expects, or sets the env vars: `CR_SCRATCH`, `CLASHBOT_KATACR_ROOT`, `CLASHBOT_KATACR_WEIGHTS`.
5. Copies `configs/mirror_calib.example.json` to `runs/poc/mirror_calib.json`, or uses FIX_LIST #64's fallback.
6. Runs `scripts/check_env.py`.
7. Prints the live command: `CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live --dashboard`, plus the two macOS permissions it needs (Screen Recording, Accessibility).

The README's "Live play on a Mac (optional)" section explains:
- what is needed: iPhone Mirroring with an iPhone that has Clash Royale;
- Training Camp only (D-050), and the double live guard (D-031);
- recalibrating with `tools/calibrate.py` if the mirrored window differs.

Test `setup_mac.sh` on this Mac from a clean unzip into a temp dir, with a fresh venv. Do NOT run `--live`. Run `check_env` and a 20 s video dry run from that venv instead.
