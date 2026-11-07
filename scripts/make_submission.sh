#!/bin/sh
# Build the submission zip from git (MAC-AS): `git archive HEAD`, never the working tree (the working tree holds .claude/
# worktrees, data/, db/, runs/ and the model weights: tens of GB; none of it is tracked). Weights and built data stay on
# the author's Mac (D-052); the README says how to get / rebuild them.
#   scripts/make_submission.sh [OUT]          OUT defaults to dist/clashbot_submission.zip (dist/ is gitignored)
# Only committed files are archived: commit first (a warning is printed for uncommitted changes to tracked files).
# The zip has one top-level folder, clashbot/. Check the result with: scripts/check_submission.sh OUT
set -eu
cd "$(dirname "$0")/.."
OUT="${1:-dist/clashbot_submission.zip}"
CALIB="configs/mirror_calib.example.json"

git rev-parse --git-dir >/dev/null 2>&1 || { echo "error: not inside a git checkout (the zip is built with git archive)" >&2; exit 2; }
git ls-files --error-unmatch "$CALIB" >/dev/null 2>&1 || { echo "error: $CALIB is not committed; the zip would not run the demo or the video dry run" >&2; exit 2; }
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "warning: uncommitted changes to tracked files are NOT in the zip (git archive HEAD):" >&2
  git status --short --untracked-files=no >&2
fi

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
git archive --format=zip --prefix=clashbot/ -o "$OUT" HEAD
unzip -tq "$OUT" >/dev/null || { echo "error: the zip fails its integrity test" >&2; exit 1; }

FILES="$(unzip -Z1 "$OUT" | grep -vc '/$' || true)"
BYTES="$(wc -c < "$OUT" | tr -d ' ')"
echo "commit:  $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "zip:     $OUT"
echo "size:    $BYTES bytes ($(du -h "$OUT" | cut -f1 | tr -d " "))"
echo "files:   $FILES"
echo "next:    scripts/check_submission.sh $OUT"
