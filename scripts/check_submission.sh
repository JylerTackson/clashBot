#!/bin/sh
# Check a submission zip (MAC-AS): unzip into a temp dir, then
#   1. the zip passes its integrity test and has the required files (README, docs/QUICKSTART.md, requirements-mac.txt,
#      the calibration example, scripts/demo.sh);
#   2. nothing that must not ship is inside: .git, .claude, runs/, data/, db/, dataset/, weights (*.pt, *.safetensors,
#      *.onnx), databases / arrays (*.sqlite, *.npy, *.hnsw), videos, .env / cookie / key files, .DS_Store, __pycache__;
#   3. no file is larger than 50 MB;
#   4. `python -m compileall -q` passes on the code;
#   5. no secret patterns (API keys, tokens, private keys, credential assignments) in any text file.
# Prints every finding; exit 1 on any (0 = clean). Machine-specific user paths are counted and shown as a note only.
#   scripts/check_submission.sh [ZIP]          ZIP defaults to dist/clashbot_submission.zip
#   PYTHON=/path/to/python scripts/check_submission.sh ZIP      (default: python3)
set -u
ZIP="${1:-dist/clashbot_submission.zip}"
PYTHON="${PYTHON:-python3}"
FINDINGS=0
bad() { echo "FINDING: $*"; FINDINGS=$((FINDINGS + 1)); }

[ -f "$ZIP" ] || { echo "error: $ZIP not found (build it with scripts/make_submission.sh)" >&2; exit 2; }
TMP="$(mktemp -d "${TMPDIR:-/tmp}/clashbot_check.XXXXXX")" || exit 2
trap 'rm -rf "$TMP"' EXIT INT TERM

echo "== integrity"
unzip -tq "$ZIP" >/dev/null 2>&1 || bad "the zip fails 'unzip -t'"
unzip -q "$ZIP" -d "$TMP" || { echo "error: cannot unzip $ZIP" >&2; exit 2; }
# one top-level folder (git archive --prefix) or the files at the top
set -- "$TMP"/*
if [ $# -eq 1 ] && [ -d "$1" ]; then ROOT="$1"; else ROOT="$TMP"; fi
echo "unzipped to a temp dir: $(find "$ROOT" -type f | wc -l | tr -d ' ') files, $(du -sh "$ROOT" | cut -f1 | tr -d " ")"

echo "== required files"
for f in README.md docs/QUICKSTART.md requirements-mac.txt configs/mirror_calib.example.json scripts/demo.sh scripts/check_env.py; do
  [ -s "$ROOT/$f" ] || bad "missing or empty: $f"
done

echo "== files that must not ship"
NOSHIP="$(cd "$ROOT" && find . \( -name .git -o -name .claude -o -name .DS_Store -o -name __pycache__ -o -name .pytest_cache \
  -o -name .env -o -name '.env.*' -o -name 'cookies*.txt' -o -name '*.pem' -o -name '*.p12' -o -name id_rsa -o -name id_ed25519 \
  -o -name '*.pt' -o -name '*.safetensors' -o -name '*.onnx' -o -name '*.sqlite' -o -name '*.sqlite-wal' -o -name '*.npy' \
  -o -name '*.hnsw' -o -name '*.mp4' -o -name '*.mkv' -o -name '*.webm' \) -print)"
[ -z "$NOSHIP" ] || { echo "$NOSHIP" | sed 's/^/  /'; bad "$(echo "$NOSHIP" | wc -l | tr -d ' ') path(s) that must not be in the zip (listed above)"; }
TOPDIRS="$(cd "$ROOT" && ls -d runs data db dataset dist 2>/dev/null || true)"
[ -z "$TOPDIRS" ] || bad "top-level build/data folder(s) in the zip: $(echo $TOPDIRS)"

echo "== files over 50 MB"
BIG="$(cd "$ROOT" && find . -type f -size +50M -exec ls -l {} \; 2>/dev/null || true)"
[ -z "$BIG" ] || { echo "$BIG" | sed 's/^/  /'; bad "file(s) larger than 50 MB"; }
echo "largest: $(cd "$ROOT" && find . -type f -exec ls -l {} \; | sort -k5 -n -r | head -1 | awk '{printf "%s bytes %s", $5, $9}')"

echo "== compileall ($PYTHON)"
if "$PYTHON" -c 'import sys; sys.exit(0)' 2>/dev/null; then
  OUTC="$(cd "$ROOT" && "$PYTHON" -m compileall -q . 2>&1)"
  [ $? -eq 0 ] || { echo "$OUTC" | head -20 | sed 's/^/  /'; bad "python -m compileall reported errors"; }
else
  bad "$PYTHON not found; set PYTHON=/path/to/python"
fi

echo "== secret patterns"
# high-signal token formats, private keys, cookie files, and credential-looking assignments of a long literal
PATTERNS='(^|[^A-Za-z0-9])(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})|-----BEGIN [A-Z ]*PRIVATE KEY-----|# Netscape HTTP Cookie File|(api[_-]?key|secret|passwd|password|auth[_-]?token|access[_-]?token|bearer)[A-Za-z_]*["'"'"']? *[:=] *["'"'"'][A-Za-z0-9/+_.=-]{16,}["'"'"']'
HITS="$(cd "$ROOT" && grep -rIEn --exclude=check_submission.sh -e "$PATTERNS" . 2>/dev/null | cut -c1-200 || true)"
if [ -n "$HITS" ]; then
  echo "$HITS" | head -20 | sed 's/^/  /'
  bad "$(echo "$HITS" | wc -l | tr -d ' ') line(s) match a secret pattern (first 20 listed above)"
else
  echo "none"
fi

echo "== note: machine-specific paths (not a finding)"
USERPATHS="$(cd "$ROOT" && grep -rIl -e '/Users/[a-z]' -e '/home/user' . 2>/dev/null | wc -l | tr -d ' ')"
echo "$USERPATHS file(s) mention /Users/<name> or /home/user (documentation and old report JSON; no code path depends on them)"

echo "== result"
if [ "$FINDINGS" -eq 0 ]; then echo "OK: $ZIP is clean"; exit 0; fi
echo "$FINDINGS finding(s) in $ZIP"
exit 1
