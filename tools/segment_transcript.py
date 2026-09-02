"""Keep only the transcript cues spoken while a match was readable.

  python3 tools/segment_transcript.py --states runs/<name>/states.jsonl --vtt data/videos/<id>/<id>.en.vtt --out runs/<name>/transcript_segments.json

Match periods come from the perception states (readiness == "match",
merged with a small gap tolerance). Cues outside those periods are DISCARDED
(the commentator is in the menu / talking, not giving match context). Each
kept cue carries the video time, the match clock read at that time and the
nearest state summary, so a downstream agent can compare speech to visuals.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr_perception.recorder import read_jsonl  # noqa: E402

TS = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d\d\d)")


def parse_vtt(path: Path) -> list[dict]:
    cues, cur, text = [], None, []
    seen = set()
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if "-->" in line:
            if cur and text:
                cues.append({**cur, "text": " ".join(text)})
            a, b = line.split("-->")[:2]
            ma, mb = TS.search(a), TS.search(b)
            if not (ma and mb):
                cur = None
                continue
            f = lambda m: int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
            cur, text = {"start": f(ma), "end": f(mb)}, []
        elif line and cur is not None and not line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            clean = re.sub(r"<[^>]+>", "", line).strip()
            # auto-subs repeat lines as they scroll; keep each sentence once
            if clean and clean not in seen:
                seen.add(clean)
                text.append(clean)
    if cur and text:
        cues.append({**cur, "text": " ".join(text)})
    return cues


def match_periods(states_path: Path, gap: float = 3.0, min_len: float = 10.0) -> tuple[list[tuple[float, float]], list[dict]]:
    periods, cur = [], None
    states = []
    for rec in read_jsonl(states_path):
        if rec.get("type") != "state":
            continue
        states.append(rec)
        t = rec["t"]
        if rec["readiness"] == "match":
            if cur is None:
                cur = [t, t]
            elif t - cur[1] <= gap:
                cur[1] = t
            else:
                periods.append(tuple(cur))
                cur = [t, t]
    if cur:
        periods.append(tuple(cur))
    return [p for p in periods if p[1] - p[0] >= min_len], states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--vtt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    periods, states = match_periods(Path(a.states))
    cues = parse_vtt(Path(a.vtt))
    kept, dropped = [], 0
    import bisect
    ts = [s["t"] for s in states]
    for c in cues:
        mid = (c["start"] + c["end"]) / 2
        if not any(p0 <= mid <= p1 for p0, p1 in periods):
            dropped += 1
            continue
        i = min(len(states) - 1, bisect.bisect_left(ts, mid))
        s = states[i]
        kept.append({**c, "match_clock": s.get("match_clock"), "phase": s.get("phase"),
                     "own_elixir": s.get("own", {}).get("elixir"), "own_hand": s.get("own", {}).get("hand"),
                     "units": [(u["class"], u["side"], u["tile"]) for u in s.get("units", [])][:12]})
    out = {"match_periods": periods, "cues_total": len(cues), "cues_kept": len(kept), "cues_dropped": dropped, "cues": kept}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"match periods: {[(round(a0), round(b0)) for a0, b0 in periods]}; cues kept {len(kept)}/{len(cues)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
