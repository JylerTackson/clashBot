#!/bin/sh
# clashBot demo entrypoint (MAC-AV).
#   demo   (default)  start the dashboard, run the live stack in dry run on the bundled clip, keep the dashboard up
#   tests             a small CPU test subset
#   <anything else>   run that command (e.g. `bash`)
# Environment knobs (all optional): RANKER (d1|tactics), DEMO_SECONDS (video seconds to process, default 60 of the
# 90 s clip), VIZ_PORT (default 8765), KEEP_SERVING (default 1; 0 = exit when the run ends).
# Nothing here sends input anywhere: the executor is the dry-run one and `--live` is never passed.
set -eu

cd /app
RUNS=/data/runs
ART=/app/artifacts
VIZ_PORT="${VIZ_PORT:-8765}"
D1_NAME=d1-ft-joint-tac
# the pro deck of the clip (Z8-4VhLjrGU, game 0, from dataset/v1/games.jsonl): bait, no hero card
DECK="berserker,dart-goblin,electro-spirit,elite-barbarians,fireball,goblin-gang,suspicious-bush,the-log"

mode="${1:-demo}"
case "$mode" in
  demo) if [ "$#" -gt 0 ]; then shift; fi ;;
  tests)
    if [ "$#" -gt 0 ]; then shift; fi
    # (the one deselected test borrows fixtures from test modules that are not in the image)
    exec python -m pytest -q -p no:cacheprovider --deselect tests/test_live_loop.py::test_a_closed_tap_gate_also_holds_back_the_hero_ability_tap \
      "$@" tests/test_docker_packaging.py tests/test_live_loop.py tests/test_viz.py \
      tests/test_mirroring_source.py tests/test_executor_mac.py tests/test_policy.py tests/test_tactics.py \
      tests/test_d1_format.py tests/test_geometry.py tests/test_clock.py tests/test_tracking.py ;;
  *) exec "$@" ;;
esac

mkdir -p "$RUNS"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RUNS/docker_demo_$STAMP"

banner() { echo; echo "================================================================================"; for l in "$@"; do echo "  $l"; done; echo "================================================================================"; echo; }

# ---- d1: downloaded from Hugging Face while the image was built; if that download failed the tactics engine picks the moves ----
D1_DIR=""
if [ -f "$ART/$D1_NAME/model.safetensors" ]; then
  D1_DIR="$ART/$D1_NAME"
fi

RANKER="${RANKER:-d1}"
if [ "$RANKER" = "d1" ] && [ -z "$D1_DIR" ]; then
  RANKER=tactics
  banner "d1 weights not present - the tactics engine is choosing moves; see README to add d1" \
         "(the download during 'docker compose build' failed or was blocked; retry it with: D1_RETRY=1 docker compose up --build)" \
         "(the rest of the stack is unchanged: perception, retrieval, knowledge graph, policy, executor in dry run)"
fi
if [ "$RANKER" = "d1" ]; then
  RANKER_ARGS="--ranker d1 --d1-backend transformers --d1-model $D1_DIR"
else
  RANKER_ARGS="--ranker tactics"
fi

# ---- the dashboard, following the newest run under /data/runs (the run folder exists now, so the page leaves an old run at once) ----
mkdir -p "$OUT"
LIVE_PID=""
python -m clashbot.viz --follow "$RUNS" --host 0.0.0.0 --port "$VIZ_PORT" &
VIZ_PID=$!
# Ctrl-C / docker stop: stop the run and the dashboard (children of a non-interactive shell ignore SIGINT, so use SIGTERM)
trap 'kill $LIVE_PID $VIZ_PID 2>/dev/null || true; exit 0' INT TERM
banner "Dashboard:  http://localhost:$VIZ_PORT   (open it now; decisions appear as the run progresses)" \
       "Run:        $OUT" \
       "Ranker:     $RANKER" \
       "Dry run:    no taps are ever sent"

# ---- the live stack in dry run on the clip. --stride 3 turns the 30 fps video into 10 fps, the rate of the live capture and of the
# vs-pro runs; --seconds is video time (the clip is 90 s long, the default 60 s keeps the run short) ----
status=0
# shellcheck disable=SC2086
python -m clashbot.live --dry-run --source video --video "$ART/clip.mp4" --calib "$ART/calib.json" \
    --deck "$DECK" $RANKER_ARGS --policy fast --act-model "$ART/act-gbdt" --device cpu \
    --record-shots --stride 3 --seconds "${DEMO_SECONDS:-60}" --out "$OUT" &
LIVE_PID=$!
wait "$LIVE_PID" || status=$?

if [ "$status" -ne 0 ]; then
  banner "The run failed (exit code $status); see the messages above."
  kill "$VIZ_PID" 2>/dev/null || true
  exit "$status"
fi

banner "The demo run has finished. Run summary: $OUT/summary.json" \
       "The dashboard keeps serving the finished run at http://localhost:$VIZ_PORT  (Ctrl-C to stop)"
if [ "${KEEP_SERVING:-1}" = "0" ]; then
  kill "$VIZ_PID" 2>/dev/null || true
  exit 0
fi
wait "$VIZ_PID"
