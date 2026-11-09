#!/bin/sh
# Native Mac setup for live play (MAC-AV, amendment 1): Docker cannot reach the macOS window server, so playing through iPhone
# Mirroring runs natively. Run it once from the unzipped folder (or the repository root); it can be re-run safely.
#   scripts/setup_mac.sh [--python PATH] [--venv DIR] [--reinstall] [--no-d1]
#     --python PATH  the Python 3.12 to build the venv with (default: python3.12, else python3 when that is 3.12)
#     --venv DIR     where the virtual environment goes (default .venv in this folder)
#     --reinstall    install requirements-mac.txt again into an existing venv (an existing venv is otherwise left alone)
#     --no-d1        skip the 1.17 GB d1 download (the live command then needs it: db/models/d1-ft-joint-tac)
# It: (1) checks macOS and Python 3.12, (2) creates the venv and installs requirements-mac.txt, (3) downloads d1 from Hugging Face
# (public repo Jyler/clashbot-d1-ft-joint-tac, pinned revision), (4) copies the zip's artefacts (databases, act model, KataCR
# weights and source) to the paths the code expects, (5) copies the calibration, (6) runs scripts/check_env.py, (7) prints the
# live command and the two macOS permissions it needs. It never sends input and never starts a battle: live play is a separate
# command that needs --live AND CLASHBOT_ALLOW_LIVE=1 (D-031). No account, token or secret is read or written.
set -eu
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

D1_REPO="Jyler/clashbot-d1-ft-joint-tac"
D1_REV="5b984f0cce7c6db6e8a768cc0df2d6836022a3e2"
D1_DIR="db/models/d1-ft-joint-tac"
VENV=".venv"
PYTHON=""
REINSTALL=0
WITH_D1=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --python) PYTHON="${2:?--python needs a path}"; shift ;;
    --venv) VENV="${2:?--venv needs a directory}"; shift ;;
    --reinstall) REINSTALL=1 ;;
    --no-d1) WITH_D1=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "error: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n== %s\n' "$*"; }
fail() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. macOS and Python 3.12
say "1. system"
[ "$(uname -s)" = "Darwin" ] || fail "this script is for macOS (iPhone Mirroring does not exist elsewhere); on Linux or Windows use: docker compose up --build"
ARCH="$(uname -m)"
MACOS="$(sw_vers -productVersion 2>/dev/null || echo 0)"
echo "macOS $MACOS on $ARCH"
case "$ARCH" in
  arm64) ;;
  x86_64) echo "note: Intel Mac. The MLX packages (Apple silicon only) are skipped; the live command uses the PyTorch backend and works, more slowly." ;;
  *) fail "unsupported CPU architecture $ARCH" ;;
esac
[ "${MACOS%%.*}" -ge 15 ] 2>/dev/null || echo "warning: iPhone Mirroring needs macOS 15 or newer (this is $MACOS); the video dry run still works"
xcode-select -p >/dev/null 2>&1 || fail "Xcode command line tools are missing (hnswlib builds from source): run xcode-select --install, then this script again"

py_is_312() { [ -n "$1" ] && "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; }
if [ -z "$PYTHON" ]; then
  for cand in python3.12 python3; do
    p="$(command -v "$cand" 2>/dev/null || true)"
    if py_is_312 "$p"; then PYTHON="$p"; break; fi
  done
fi
py_is_312 "$PYTHON" || fail "Python 3.12 not found (the pins were verified on it). Install it (brew install python@3.12) and re-run, or pass --python /path/to/python3.12"
echo "python: $PYTHON ($("$PYTHON" -c 'import platform; print(platform.python_version())'))"
if ! command -v brew >/dev/null 2>&1 || ! brew list libomp >/dev/null 2>&1; then
  echo "note: libomp is not installed (brew install libomp); only LightGBM needs it, and the live path does not import LightGBM"
fi

# ---------------------------------------------------------------- 2. virtual environment
say "2. virtual environment ($VENV)"
PYV="$VENV/bin/python"
REQ="requirements-mac.txt"
if [ "$ARCH" != "arm64" ]; then
  REQ="$(mktemp -t clashbot_requirements)"
  grep -v '^mlx' requirements-mac.txt > "$REQ"
fi
NEED_INSTALL=0
if [ ! -e "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
  touch "$VENV/.clashbot_setup_incomplete"          # removed when the install below finishes: a re-run continues it
  NEED_INSTALL=1
elif [ -e "$VENV/.clashbot_setup_incomplete" ]; then
  echo "continuing an interrupted install"
  NEED_INSTALL=1
elif [ "$REINSTALL" = 1 ]; then
  NEED_INSTALL=1
else
  echo "$VENV already exists: left as it is (use --reinstall to install requirements-mac.txt into it again)"
fi
if [ "$NEED_INSTALL" = 1 ]; then
  echo "installing requirements-mac.txt (torch, MLX, transformers, ...: several GB, a few minutes)"
  "$PYV" -m pip install --quiet --upgrade pip
  CC=/usr/bin/clang CXX=/usr/bin/clang++ "$PYV" -m pip install -r "$REQ"
  rm -f "$VENV/.clashbot_setup_incomplete"
fi
[ -x "$PYV" ] || fail "$PYV not found"

# ---------------------------------------------------------------- 3. d1 from Hugging Face
say "3. d1 ranker ($D1_DIR)"
if [ -f "$D1_DIR/model.safetensors" ]; then
  echo "present: $D1_DIR"
elif [ "$WITH_D1" = 0 ]; then
  echo "skipped (--no-d1): the live command needs $D1_DIR; re-run without --no-d1"
else
  echo "downloading $D1_REPO @ $(printf %.8s "$D1_REV") (1.17 GB, public repo, no token)"
  mkdir -p "$D1_DIR"
  env -u HF_HUB_OFFLINE "$PYV" -c "from huggingface_hub import snapshot_download; snapshot_download('$D1_REPO', revision='$D1_REV', local_dir='$D1_DIR')" \
    || fail "the d1 download failed (network?). Re-run this script to retry; nothing else is affected"
fi

# ---------------------------------------------------------------- 4. artefacts of the zip
say "4. artefacts"
copy_missing() {            # copy_missing SRC DST: copy a file or folder unless DST exists (never overwrites)
  if [ -e "$1" ] && [ ! -e "$2" ]; then
    mkdir -p "$(dirname "$2")"
    cp -R "$1" "$2"
    echo "copied $2"
  elif [ -e "$2" ]; then
    echo "present: $2"
  else
    echo "note: $1 is not in this folder (a full checkout has $2 already, or build it: README.md 'Rebuild the data')"
  fi
}
copy_missing artifacts/db/clashbot.sqlite db/clashbot.sqlite
copy_missing artifacts/db/kg2.sqlite db/kg2.sqlite
copy_missing artifacts/db/state_features.npy db/state_features.npy
copy_missing artifacts/db/state_features.meta.json db/state_features.meta.json
copy_missing artifacts/act-gbdt db/models/act-gbdt
copy_missing artifacts/dataset/v1/placement_prior.json dataset/v1/placement_prior.json
copy_missing artifacts/dataset/v1/kg2/deck_cooc.json dataset/v1/kg2/deck_cooc.json
copy_missing artifacts/katacr_d1.pt cr_scratch/katacr_d1.pt
copy_missing artifacts/katacr_d2.pt cr_scratch/katacr_d2.pt
copy_missing artifacts/katacr cr_scratch/katacr
CRS="$ROOT/cr_scratch"
[ -d "$CRS" ] || CRS="${CR_SCRATCH:-$HOME/cr_scratch}"      # a full checkout keeps KataCR in ~/cr_scratch

# ---------------------------------------------------------------- 5. calibration
say "5. calibration"
mkdir -p runs/poc
copy_missing configs/mirror_calib.example.json runs/poc/mirror_calib.json

# ---------------------------------------------------------------- 6. environment check
say "6. environment check (scripts/check_env.py)"
CR_SCRATCH="$CRS" HF_HUB_OFFLINE=1 "$PYV" scripts/check_env.py || echo "check_env reported missing required packages (see above)"
echo "(check_env looks for the clip at data/videos/Z8-4VhLjrGU/..., so its 'video dry run' lines say no here and 'dataset/v1' is only"
echo " needed for the evaluations; this folder's clip is artifacts/clip.mp4 and the dry run command at the end uses it)"

# ---------------------------------------------------------------- 7. what to run
say "7. done. Live play"
cat <<EOF
Open iPhone Mirroring with your iPhone showing Clash Royale (Training Camp), then in a terminal:

  cd "$ROOT"
  source $VENV/bin/activate
  export CR_SCRATCH="$CRS"
  CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live --dashboard

Leave out CLASHBOT_ALLOW_LIVE=1 and --live to only watch (the bot decides and logs, nothing is tapped). It taps only in a battle whose
opponent banner reads "Trainer ..." (Training Camp). Ctrl-C stops it.

macOS permissions for the terminal app you run it from (System Settings > Privacy & Security):
  - Screen Recording   (to read the mirrored window)
  - Accessibility      (to post the two taps per card)
Restart the terminal after granting them. If the mirrored window is not the size the calibration was made for, recalibrate:
  python tools/calibrate.py --help

A dry run of the same stack on a recorded video (no phone):
  python -m clashbot.live --dry-run --source video --video artifacts/clip.mp4 --calib runs/poc/mirror_calib.json \\
    --deck berserker,dart-goblin,electro-spirit,elite-barbarians,fireball,goblin-gang,suspicious-bush,the-log \\
    --ranker d1 --d1-backend transformers --d1-model $D1_DIR --policy fast --act-model db/models/act-gbdt \\
    --record-shots --seconds 20 --out runs/smoke/native_check
EOF
