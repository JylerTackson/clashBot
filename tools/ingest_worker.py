"""Pull one video's outputs from a worker branch into this checkout (worktree
only, nothing staged from runs/), merge its insights into the knowledge base,
mark the games done. Idempotent.

  python3 tools/ingest_worker.py <branch> <video_id> [<video_id> ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(*cmd, check=True):
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=True, text=True)


def main() -> int:
    branch, vids = sys.argv[1], sys.argv[2:]
    sh("git", "fetch", "-q", "origin", branch.split("/", 1)[1] if branch.startswith("origin/") else branch)
    ref = branch if branch.startswith("origin/") else f"origin/{branch}"
    for vid in vids:
        paths = [f"runs/videos/{vid}"] + [p for p in sh("git", "ls-tree", "-r", "--name-only", ref, "knowledge_base/matches").stdout.split()
                                           if p.startswith(f"knowledge_base/matches/{vid}-m")]
        sh("git", "restore", "--source", ref, "--worktree", "--", *paths)
        # imported rather than shelled out: video ids can start with '-'
        sys.path.insert(0, str(ROOT / "tools"))
        import dispatch_matches as dm
        import merge_insights as mi
        keys = [r["key"] for r in dm.ready() if r["key"].startswith(vid + "-m")]
        if not keys:
            print(f"{vid}: no game contexts found on {ref}")
            continue
        missing = [k for k in keys if not (ROOT / "knowledge_base" / "matches" / f"{k}.md").exists()]
        idx = json.loads((ROOT / "knowledge_base" / "meta" / "card_index.json").read_text())
        names = {c["slug"]: c["name"] for c in idx["cards"]}
        cards_idx = {c["slug"]: c for c in idx["cards"]}
        n = 0
        for p in sorted((ROOT / "runs" / "videos" / vid).glob("match_*/**/insights.json")):
            ins = json.loads(p.read_text())
            if ins["key"] in keys:
                mi.apply(ins, names, cards_idx, False)
                n += 1
        out = f"{n} insight file(s) merged"
        m = dm.load()
        for k in keys:
            m["matches"][k] = {**m["matches"].get(k, {}), "status": "done"}
        dm.save(m)
        title = json.loads((ROOT / "runs" / "videos" / vid / "video_deck.json").read_text()) if (ROOT / "runs" / "videos" / vid / "video_deck.json").exists() else {}
        print(f"{vid}: {len(keys)} games merged; missing match files: {missing or 'none'}; deck: {title.get('deck_key') or ('mixed' if title.get('mixed') else '?')}")
        print("   " + out.strip().splitlines()[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
