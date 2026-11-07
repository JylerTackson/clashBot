#!/bin/sh
# One-command demo (MAC-AN, D-050): asks the deck, loads the models, says READY, then watches iPhone Mirroring and plays
# Training Camp battles until Ctrl-C.
#   scripts/demo.sh                              watch only (the bot decides and logs; nothing is tapped)
#   CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live taps ON (both D-031 switches), Training Camp battles only
# Other flags: --deck --evo-cards --hero-cards --yes --dashboard --out   (see: scripts/demo.sh --help)
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
PY="$(pwd)/../.venv/bin/python"
[ -x "$PY" ] || PY=python3
exec "$PY" -m clashbot.demo "$@"
