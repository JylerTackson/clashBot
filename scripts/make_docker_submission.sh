#!/bin/sh
# Build the Docker submission zip (MAC-AV): the code from `git archive HEAD` plus the runtime artefacts, so a grader can unzip it and
# run `docker compose up --build` on any machine (amd64 or arm64), or set up the native Mac path with scripts/setup_mac.sh.
# No docs/report, no benchmarks and no runs/ except the clip and its config and, under prior_runs/, a small copy of every prior game
# (scripts/pack_prior_runs.py, MAC-AW) so the dashboard can browse them.
#   scripts/make_docker_submission.sh [OUT]
#     OUT defaults to dist/clashbot_docker_submission.zip (dist/ is gitignored)
# The zip never contains the d1 weights: the Dockerfile downloads them from Hugging Face (Jyler/clashbot-d1-ft-joint-tac, pinned
# revision) while the image is built, and scripts/setup_mac.sh downloads the same revision for the native Mac path.
# Environment (all optional): PY (python with cv2), BUILD_DIR (staging, default dist/docker_build),
#   VIDEO (default data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4), CR_SCRATCH (default ~/cr_scratch: katacr/ clone and katacr_d{1,2}.pt),
#   PRIOR_RUNS (default 1; 0 leaves the prior games out), PRIOR_DIR (where the pack is built, default $BUILD_DIR/prior_runs).
# Only committed files are archived: commit first. Built data (db/, dataset/) and the video are read from this checkout.
set -eu
cd "$(dirname "$0")/.."

OUT=""
for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    -*) echo "error: unknown option $arg" >&2; exit 2 ;;
    *) OUT="$arg" ;;
  esac
done
OUT="${OUT:-dist/clashbot_docker_submission.zip}"

PY="${PY:-$( [ -x ../.venv/bin/python ] && echo ../.venv/bin/python || echo python3 )}"
BUILD_DIR="${BUILD_DIR:-dist/docker_build}"
VIDEO="${VIDEO:-data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4}"
CR_SCRATCH="${CR_SCRATCH:-$HOME/cr_scratch}"
TOP="clashbot_docker"
PRIOR_RUNS="${PRIOR_RUNS:-1}"
PRIOR_DIR="${PRIOR_DIR:-$BUILD_DIR/prior_runs}"
# the pro deck of the clip is fixed in docker/entrypoint.sh (game Z8-4VhLjrGU-m0.0); the clip is 60 s .. 150 s of that video

# what the zip carries from git: the code, the reference data the runtime reads (card tables and art), the packaging files, the
# test subset that `docker compose run --rm clashbot tests` runs (keep in step with docker/entrypoint.sh) and what the native Mac
# path needs (scripts/setup_mac.sh, the launcher, the environment check, requirements-mac.txt, the calibration tool)
TEST_FILES="tests/__init__.py tests/conftest.py tests/test_docker_packaging.py tests/test_live_loop.py tests/test_viz.py
tests/test_mirroring_source.py tests/test_executor_mac.py tests/test_policy.py tests/test_tactics.py
tests/test_d1_format.py tests/test_geometry.py tests/test_clock.py tests/test_tracking.py"
GIT_PATHS="clashbot cr_perception configs knowledge_base tools
docker docker-compose.yml .dockerignore docs/DOCKER.md requirements-mac.txt scripts/setup_mac.sh scripts/demo.sh scripts/check_env.py
$TEST_FILES"

git rev-parse --git-dir >/dev/null 2>&1 || { echo "error: not inside a git checkout (the code is taken with git archive)" >&2; exit 2; }
for f in docker/Dockerfile docker/entrypoint.sh docker/requirements-docker.txt docker-compose.yml docs/DOCKER.md scripts/setup_mac.sh; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "error: $f is not committed" >&2; exit 2; }
done
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "warning: uncommitted changes to tracked files are NOT in the zip (git archive HEAD):" >&2
  git status --short --untracked-files=no >&2
fi
need() { [ -e "$1" ] || { echo "error: missing $1 ($2)" >&2; exit 2; }; }
need "$VIDEO" "the source video of the clip; set VIDEO"
need db/clashbot.sqlite "built data; see README.md 'Rebuild the data'"
need db/kg2.sqlite "knowledge graph v2 (prebuilt)"
need db/state_features.npy "feature matrix"
need db/state_features.meta.json "feature metadata"
need db/models/act-gbdt/meta.json "LightGBM act model"
need dataset/v1/placement_prior.json "placement prior"
need dataset/v1/kg2/deck_cooc.json "opponent-deck co-occurrence"
need "$CR_SCRATCH/katacr_d1.pt" "KataCR weights; set CR_SCRATCH"
need "$CR_SCRATCH/katacr_d2.pt" "KataCR weights; set CR_SCRATCH"
need "$CR_SCRATCH/katacr/katacr/yolov8/custom_model.py" "KataCR source clone at commit 36ceb9f"
need "$CR_SCRATCH/katacr/LICENSE" "KataCR license"
if [ "$PRIOR_RUNS" != "0" ]; then
  need runs/poc "the prior runs (set PRIOR_RUNS=0 to build the zip without them)"
  need runs/eval/vs_pro "the bot-vs-pro runs (set PRIOR_RUNS=0 to build the zip without them)"
fi

STAGE="$BUILD_DIR/stage"
ROOT_DIR="$STAGE/$TOP"
ART="$ROOT_DIR/artifacts"
rm -rf "$STAGE"
mkdir -p "$ART/db" "$ART/dataset/v1/kg2" "$ART/act-gbdt" "$ART/katacr/katacr/yolov8"

echo "== code (git archive HEAD, $(git rev-parse --short HEAD))"
# shellcheck disable=SC2086
git archive --format=tar HEAD -- $GIT_PATHS | tar -x -C "$ROOT_DIR"
# the runtime reads the card pages and art, the decks, the hero / card index; the pro-state and match markdown only feed the database build
rm -rf "$ROOT_DIR/knowledge_base/states" "$ROOT_DIR/knowledge_base/matches"
cp "$ROOT_DIR/docs/DOCKER.md" "$ROOT_DIR/README.md"
rm -rf "$ROOT_DIR/docs"

echo "== artefacts"
"$PY" scripts/docker_artifacts.py clip "$VIDEO" "$ART/clip.mp4" --start 60 --seconds 90
cp configs/mirror_calib.example.json "$ART/calib.json"
# the retrieval pool without any state of the clip's source video (the demo must not retrieve the pro's own play from the same match)
"$PY" scripts/docker_artifacts.py slim-pool db "$ART/db" --exclude-video Z8-4VhLjrGU
cp db/kg2.sqlite "$ART/db/"
cp dataset/v1/placement_prior.json "$ART/dataset/v1/"
cp dataset/v1/kg2/deck_cooc.json "$ART/dataset/v1/kg2/"
cp db/models/act-gbdt/meta.json db/models/act-gbdt/model_h*.json "$ART/act-gbdt/"
cp "$CR_SCRATCH/katacr_d1.pt" "$CR_SCRATCH/katacr_d2.pt" "$ART/"
cp "$CR_SCRATCH/katacr/katacr/yolov8/custom_model.py" "$ART/katacr/katacr/yolov8/"
cp "$CR_SCRATCH/katacr/LICENSE" "$ART/katacr/LICENSE"
cat > "$ART/katacr/NOTICE.txt" <<'EOF'
KataCR (https://github.com/wty-yy/katacr), MIT license (LICENSE in this folder), commit 36ceb9f.
katacr/yolov8/custom_model.py is the one source file the checkpoints katacr_d1.pt and katacr_d2.pt (next folder up, MIT, from the
same project) need in order to unpickle. katacr/ and katacr/yolov8/ are namespace packages (no __init__.py in the original).
EOF

if [ "$PRIOR_RUNS" != "0" ]; then
  # every live game, the 24 bot-vs-pro runs and two smoke dry runs, at prior_runs/<group>/<run>/ with screenshots re-encoded smaller
  # (the Dockerfile copies it to /app/prior_runs and the dashboard lists it; scripts/setup_mac.sh copies it into runs/)
  echo "== prior runs (scripts/pack_prior_runs.py)"
  "$PY" scripts/pack_prior_runs.py --runs runs --out "$PRIOR_DIR"
  cp -Rp "$PRIOR_DIR" "$ROOT_DIR/prior_runs"
else
  mkdir -p "$ROOT_DIR/prior_runs"       # the Dockerfile copies this folder: an empty one keeps the build working (the picker lists nothing from it)
  echo "no prior runs in this build (PRIOR_RUNS=0)" > "$ROOT_DIR/prior_runs/NOTE.txt"
fi

echo "== zip"
OUT_ABS="$(cd "$(dirname "$OUT")" 2>/dev/null && pwd || true)"
[ -n "$OUT_ABS" ] || { mkdir -p "$(dirname "$OUT")"; OUT_ABS="$(cd "$(dirname "$OUT")" && pwd)"; }
OUT_ABS="$OUT_ABS/$(basename "$OUT")"
rm -f "$OUT_ABS"
(cd "$STAGE" && zip -r -X -q -1 "$OUT_ABS" "$TOP")
unzip -tq "$OUT_ABS" >/dev/null || { echo "error: the zip fails its integrity test" >&2; exit 1; }

echo "commit:  $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "zip:     $OUT_ABS"
echo "size:    $(wc -c < "$OUT_ABS" | tr -d ' ') bytes ($(du -h "$OUT_ABS" | cut -f1 | tr -d ' '))"
echo "files:   $(unzip -Z1 "$OUT_ABS" | grep -vc '/$' || true)"
echo "artefacts (unzipped sizes):"
(cd "$ART" && du -sh clip.mp4 calib.json db/* dataset act-gbdt katacr_d1.pt katacr_d2.pt katacr | sed 's/^/  /')
if [ -d "$ROOT_DIR/prior_runs" ]; then
  echo "prior runs (unzipped): $(du -sh "$ROOT_DIR/prior_runs" | cut -f1 | tr -d ' ') in $(ls -d "$ROOT_DIR"/prior_runs/*/* | wc -l | tr -d ' ') run folders (prior_runs/<group>/<run>)"
fi
echo "code (unzipped): $(du -sh "$ROOT_DIR" | cut -f1 | tr -d ' ') in total, $(du -sh "$ROOT_DIR/knowledge_base" | cut -f1 | tr -d ' ') of it knowledge_base/"
