#!/usr/bin/env bash
# One-shot environment setup for a perception worker session (Linux container or macOS).
# Usage: bash tools/worker_setup.sh   (idempotent; ~5-10 min on first run)
#   PY=/path/to/venv/bin/python bash tools/worker_setup.sh     # pick the interpreter (default python3)
#   CR_SCRATCH=...  CLASHBOT_KATACR_ROOT=...                   # override locations (defaults below)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
export CR_SCRATCH="${CR_SCRATCH:-$HOME/cr_scratch}"
KATACR_ROOT="${CLASHBOT_KATACR_ROOT:-$CR_SCRATCH/katacr}"
export KATACR_ROOT
KATACR_REPO="${KATACR_REPO:-https://github.com/wty-yy/katacr}"
OS="$(uname)"
mkdir -p "$CR_SCRATCH" "$ROOT/data" "$ROOT/runs"
echo "== python deps ($PY, $OS)"
PIP_PKGS=(ultralytics==8.1.24 numpy opencv-python-headless pillow scipy onnxruntime rapidocr-onnxruntime pytest yt-dlp yt-dlp-ejs requests hnswlib tokenizers)
if [ "$OS" = "Darwin" ]; then
  # hnswlib has no wheels: build it with Apple clang. The default torch wheel already includes MPS.
  export CC=/usr/bin/clang CXX=/usr/bin/clang++
  PIP_PKGS+=(mss pyobjc-framework-Quartz)
  "$PY" -m pip install -q torch torchvision 2>&1 | tail -1 || true
else
  "$PY" -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision 2>&1 | tail -1 || true
fi
"$PY" -m pip install -q "${PIP_PKGS[@]}" 2>&1 | tail -1
echo "== KataCR source (needed to unpickle the checkpoints)"
if [ ! -d "$KATACR_ROOT/katacr" ]; then
  mkdir -p "$(dirname "$KATACR_ROOT")"
  git clone -q --depth 1 "$KATACR_REPO" "$KATACR_ROOT"
fi
echo "== KataCR weights"
cd "$CR_SCRATCH"
[ -s katacr_d1.pt ] || curl -sS -L -o katacr_d1.pt "https://drive.google.com/uc?export=download&id=1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_"
[ -s katacr_d2.pt ] || curl -sS -L -o katacr_d2.pt "https://drive.google.com/uc?export=download&id=1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD"
ls -la katacr_d1.pt katacr_d2.pt
"$PY" - <<'PY'
import os, sys, torch
sys.path.insert(0, os.environ["KATACR_ROOT"])
for f in ("katacr_d1.pt", "katacr_d2.pt"):
    p = os.path.join(os.environ["CR_SCRATCH"], f)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = ck.get("model") or ck.get("ema")
    print(f, "ok, nc =", getattr(m, "nc", None) or len(getattr(m, "names", {})))
PY
cd "$ROOT" && "$PY" -m pytest -q tests 2>&1 | tail -1
echo "== setup complete; export these before running the tools:"
echo "   export CR_SCRATCH=$CR_SCRATCH"
echo "   export CLASHBOT_KATACR_ROOT=$KATACR_ROOT"
echo "   export CLASHBOT_KATACR_WEIGHTS=$CR_SCRATCH/katacr_d1.pt:$CR_SCRATCH/katacr_d2.pt"
