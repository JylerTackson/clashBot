"""Match context packs: what the agents read.

From a run's states.jsonl (+ events) and the video's subtitle file, build
  context.json  - machine-readable: header, own/opponent decks, events,
                  1 Hz timeline (units with tiles, elixir, towers, hand),
                  transcript cues aligned to match time, quality metrics
  context.md    - the same as a readable document with transcript lines
                  interleaved into the timeline, for an LLM agent
"""
from __future__ import annotations

import bisect
import json
import re
from collections import Counter
from pathlib import Path

from .recorder import read_jsonl

TS = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d\d\d)")


def parse_vtt(path: Path) -> list[dict]:
    cues, cur, text, seen = [], None, [], set()
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
            if clean and clean not in seen:
                seen.add(clean)
                text.append(clean)
    if cur and text:
        cues.append({**cur, "text": " ".join(text)})
    return cues


def build_context(states_path: Path, vtt_path: Path | None, header: dict, card_names: dict[str, str]) -> dict:
    states, events = [], []
    for rec in read_jsonl(states_path):
        if rec.get("type") == "state":
            states.append(rec)
        elif rec.get("type") == "play_event":
            events.append(rec)
    match_states = [s for s in states if s["readiness"] == "match"]
    if not match_states:
        return {**header, "empty": True, "reason": "no readable match frames"}
    t0, t1 = match_states[0]["t"], match_states[-1]["t"]

    # 1 Hz timeline
    timeline, last_sec = [], None
    for s in match_states:
        sec = int(s["t"])
        if sec == last_sec:
            continue
        last_sec = sec
        timeline.append({
            "t": round(s["t"], 1), "clock": s.get("match_clock"), "phase": s.get("phase"),
            "own_elixir": s["own"].get("elixir"), "opp_elixir_est": s["opponent"].get("elixir_est"),
            "hand": s["own"].get("hand"), "next": s["own"].get("next_card"),
            "towers_own": s["own"].get("towers"), "towers_enemy": s["opponent"].get("towers"),
            "units": [{"class": u["class"], "side": u["side"], "tile": u["tile"]} for u in s.get("units", [])
                      if u.get("category") not in ("tower", "ui") and u.get("tile")],
        })

    # own deck: cards seen in hand/next + own plays
    hand_cards = Counter()
    for s in match_states:
        for c in (s["own"].get("hand") or []) + [s["own"].get("next_card")]:
            if c:
                hand_cards[c] += 1
    own_plays = [e for e in events if e["player"] == "own" and e.get("card")]
    for e in own_plays:
        hand_cards[e["card"]] += 30
    # keep cards seen consistently (>= 2 s of frames or played)
    own_deck = [c for c, n in hand_cards.most_common() if n >= 20][:8]
    last = match_states[-1]
    opp = last["opponent"]
    quality = {
        "readable_seconds": round(t1 - t0, 1), "match_frames": len(match_states),
        "own_elixir_drift": last.get("field_confidence", {}).get("own_elixir_drift"),
        "events_total": len(events), "events_unidentified": sum(1 for e in events if not e.get("card")),
        "events_by_source": dict(Counter(e["detect_source"] for e in events)),
        "hand_conf_mean": round(sum(sum(s["own"].get("hand_conf", [0])) / 4 for s in match_states) / len(match_states), 3),
    }
    cues = []
    if vtt_path and vtt_path.exists():
        for c in parse_vtt(vtt_path):
            mid = (c["start"] + c["end"]) / 2
            if t0 - 2 <= mid <= t1 + 2:
                cues.append({"t": round(c["start"], 1), "end": round(c["end"], 1), "text": c["text"]})
    return {**header, "start_t": round(t0, 1), "end_t": round(t1, 1),
            "own_deck_observed": own_deck, "own_deck_key": "-".join(sorted(own_deck)) if len(own_deck) == 8 else None,
            "opponent": {"deck_known": opp.get("deck_known"), "deck_complete": opp.get("deck_complete"),
                         "deck_predictions": opp.get("deck_predictions"), "kb_matches": opp.get("kb_matches"),
                         "cycle_confirmed": opp.get("cycle_confirmed"), "cycle_violations": opp.get("cycle_violations")},
            "events": events, "timeline": timeline, "transcript": cues, "quality": quality}


def render_context_md(ctx: dict, card_names: dict[str, str]) -> str:
    n = lambda slug: card_names.get(slug, slug) if slug else "?"
    if ctx.get("empty"):
        return f"# {ctx.get('video_id')} match {ctx.get('match_index')}: no readable match\n"
    L = [f"# Match context: {ctx.get('title', '')} (video {ctx.get('video_id')}, match {ctx.get('match_index')})", "",
         f"- Video time {ctx['start_t']}s to {ctx['end_t']}s ({ctx['quality']['readable_seconds']}s readable). "
         f"Calibration: {ctx.get('calibration_method')}.",
         f"- Own deck observed (Ryley): {', '.join(n(c) for c in ctx['own_deck_observed']) or 'unknown'}"
         + (f" (deck_key `{ctx['own_deck_key']}`)" if ctx.get('own_deck_key') else " (incomplete)"),
         f"- Opponent cards seen: {', '.join(n(c) for c in (ctx['opponent']['deck_known'] or []))}"
         + (" (complete)" if ctx['opponent'].get('deck_complete') else ""),
         f"- Quality: events {ctx['quality']['events_total']} ({ctx['quality']['events_unidentified']} unidentified), "
         f"sources {ctx['quality']['events_by_source']}, hand confidence {ctx['quality']['hand_conf_mean']}, "
         f"own-elixir drift {ctx['quality']['own_elixir_drift']}", "",
         "Tile coordinates: (col 0-17 left to right, row 0-31 bottom to top); Ryley's half is rows 0-14, river 15-16, "
         "opponent half 17-31. Unit positions come from a detector frozen in 2024 (newer cards may be missing or "
         "'unknown_unit'); deployments from the in-game deploy label are reliable for any card.", "",
         "## Plays", ""]
    for e in ctx["events"]:
        L.append(f"- t={e['timestamp']:.1f} clock {e.get('match_clock')} **{e['player']}** {n(e.get('card')) if e.get('card') else 'UNIDENTIFIED'}"
                 f" at tile {e.get('tile')} (elixir {e.get('elixir_before')}->{e.get('elixir_after')}) [{e['detect_source']}/{e['confidence']}] {e.get('detail', '')}")
    L += ["", "## Timeline with commentary", "",
          "Each second: clock, Ryley's elixir / opponent estimate, hand, units on the field (class, side, tile). "
          "Commentary lines (from the auto-transcript) are placed at the time they were spoken.", ""]
    cue_ts = [c["t"] for c in ctx["transcript"]]
    ci = 0
    for row in ctx["timeline"]:
        while ci < len(ctx["transcript"]) and ctx["transcript"][ci]["t"] <= row["t"]:
            L.append(f"> [{ctx['transcript'][ci]['t']:.0f}s] {ctx['transcript'][ci]['text']}")
            ci += 1
        units = ", ".join(f"{u['class']}({u['side'][0]})@{tuple(u['tile'])}" for u in row["units"][:10])
        L.append(f"- {row['t']:.0f}s clock {row['clock']} {row['phase'] or ''} | elixir {row['own_elixir']} / opp~{row['opp_elixir_est']} | "
                 f"hand {[n(c) if c else '-' for c in (row['hand'] or [])]} next {n(row['next'])} | towers own {row['towers_own']} enemy {row['towers_enemy']} | {units}")
    while ci < len(ctx["transcript"]):
        L.append(f"> [{ctx['transcript'][ci]['t']:.0f}s] {ctx['transcript'][ci]['text']}")
        ci += 1
    return "\n".join(L) + "\n"
