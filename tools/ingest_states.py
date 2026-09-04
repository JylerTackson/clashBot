"""Pull enriched state files from a phase5 worker branch into this checkout.

Only knowledge_base/states/<key>.jsonl files that differ from the current
branch are restored (never whole directories), then the validator runs and
a summary is printed. Nothing is committed here; the caller commits.

  python3 tools/ingest_states.py origin/phase5/sworker-1 [--commit]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "origin/claude/clash-royale-kb-phase-1-4rtfm6"


def sh(*cmd, check=True):
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=True, text=True)


def enriched_count(path: Path) -> tuple[int, int]:
    tot = enr = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        tot += 1
        s = json.loads(line)
        if s.get("enrichment"):
            enr += 1
    return tot, enr


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    commit = "--commit" in sys.argv
    if not args:
        print(__doc__)
        return 2
    ref = args[0] if args[0].startswith("origin/") else f"origin/{args[0]}"
    if sh("git", "fetch", "-q", "origin", ref.split("/", 1)[1], check=False).returncode != 0:
        print(f"{ref}: branch not on origin yet")
        return 0
    # only files the worker itself changed since it forked (a stale fork still
    # carries un-enriched copies of games other workers have since finished)
    base = sh("git", "merge-base", "HEAD", ref).stdout.strip()
    changed = [p for p in sh("git", "diff", "--name-only", base, ref, "--",
                             "knowledge_base/states").stdout.split()
               if p.startswith("knowledge_base/states/") and p.endswith(".jsonl")]
    # only files that exist on the worker branch (never delete)
    on_ref = set(sh("git", "ls-tree", "-r", "--name-only", ref, "knowledge_base/states").stdout.split())
    changed = [p for p in changed if p in on_ref]
    if not changed:
        print(f"{ref}: no state files differ from HEAD")
        return 0
    sh("git", "restore", "--source", ref, "--worktree", "--", *changed)
    rows = []
    for p in changed:
        tot, enr = enriched_count(ROOT / p)
        rows.append((Path(p).stem, tot, enr))
    val = sh("python3", "tools/validate_states.py", "--summary", check=False)
    summary = json.loads(val.stdout.strip().splitlines()[-1]) if val.stdout.strip() else {}
    errors = summary.get("errors", "?")
    if errors:
        print(val.stdout[-3000:])
        print(f"VALIDATION FAILED ({errors} errors); restoring HEAD copies")
        sh("git", "restore", "--source", "HEAD", "--worktree", "--", *changed)
        return 1
    for key, tot, enr in rows:
        print(f"  {key}: {enr}/{tot} enriched")
    print(f"{ref}: {len(changed)} file(s) ingested; validator: {summary}")
    if commit:
        sh("git", "add", "--", *changed)
        msg = f"phase 2 data: ingest enriched states from {ref.split('/', 1)[1]} ({len(changed)} games)"
        sh("git", "commit", "-q", "-m", msg + "\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01YJapic6HNtu1AzCZskGtf3")
        print("committed:", sh("git", "log", "-1", "--format=%h %s").stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
