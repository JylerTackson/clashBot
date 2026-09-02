#!/usr/bin/env bash
# One-shot environment setup for a perception worker session (fresh container).
# Usage: bash tools/worker_setup.sh   (idempotent; ~5-10 min on first run)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CR_SCRATCH="${CR_SCRATCH:-$HOME/cr_scratch}"
mkdir -p "$CR_SCRATCH" "$ROOT/data" "$ROOT/runs"
echo "== python deps"
python3 -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision 2>&1 | tail -1 || true
python3 -m pip install -q "ultralytics==8.1.24" numpy opencv-python-headless pillow scipy onnxruntime rapidocr-onnxruntime pytest yt-dlp yt-dlp-ejs requests 2>&1 | tail -1
echo "== KataCR source (needed to unpickle the checkpoints)"
if [ ! -d /home/user/wty-yy/katacr ]; then
  mkdir -p /home/user/wty-yy && git clone -q --depth 1 https://github.com/wty-yy/katacr /home/user/wty-yy/katacr
fi
echo "== KataCR weights"
cd "$CR_SCRATCH"
[ -s katacr_d1.pt ] || curl -sS -L -o katacr_d1.pt "https://drive.google.com/uc?export=download&id=1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_"
[ -s katacr_d2.pt ] || curl -sS -L -o katacr_d2.pt "https://drive.google.com/uc?export=download&id=1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD"
ls -la katacr_d1.pt katacr_d2.pt
python3 - <<'PY'
import os, sys, torch
sys.path.insert(0, "/home/user/wty-yy/katacr")
for f in ("katacr_d1.pt", "katacr_d2.pt"):
    p = os.path.join(os.environ["CR_SCRATCH"], f)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = ck.get("model") or ck.get("ema")
    print(f, "ok, nc =", getattr(m, "nc", None) or len(getattr(m, "names", {})))
PY
cd "$ROOT" && python3 -m pytest -q tests 2>&1 | tail -1
echo "== setup complete; export CR_SCRATCH=$CR_SCRATCH before running tools/batch_process.py"
