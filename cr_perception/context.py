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


def split_matches(states_path: Path, min_seconds: float = 60.0) -> list[tuple[int, float, float]]:
    """(game_index, t0, t1) for every game in a run, derived from the clock
    series with confirmation rules (misreads and overtime do not split)."""
    from .clock import segment_by_clock
    from .hud import parse_clock
    samples = []
    for rec in read_jsonl(states_path):
        if rec.get("type") == "state" and rec["readiness"] == "match":
            c = rec.get("match_clock")
            samples.append((rec["t"], parse_clock(c) if c else None))
    return segment_by_clock(samples, min_seconds)


def build_context(states_path: Path, vtt_path: Path | None, header: dict, card_names: dict[str, str],
                  window: tuple[float, float] | None = None, kb_decks: list | None = None) -> dict:
    states, events = [], []
    for rec in read_jsonl(states_path):
        if rec.get("type") == "state":
            if window is None or window[0] <= rec["t"] <= window[1]:
                states.append(rec)
        elif rec.get("type") == "play_event":
            events.append(rec)
    match_states = [s for s in states if s["readiness"] == "match"]
    if not match_states:
        return {**header, "empty": True, "reason": "no readable match frames"}
    t0, t1 = match_states[0]["t"], match_states[-1]["t"]
    events = [e for e in events if t0 - 1.0 <= e["timestamp"] <= t1 + 1.0]


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
            "units": [{"class": u["class"], "side": u["side"], "tile": u["tile"],
                       **({"heading": u["motion"]["heading"], "speed": u["motion"]["speed"], "pred_2s": u["motion"]["pred_2s"],
                           "pos_std": u["motion"]["pos_std"], "eta_tower": u["motion"].get("eta_tower")} if u.get("motion") else {})}
                      for u in s.get("units", []) if u.get("category") not in ("tower", "ui") and u.get("tile")],
        })
        row = timeline[-1]
        row["threats"] = [f"{u['class']}({u['side'][0]}) {u['heading']}, tower in {u['eta_tower']['s']}s"
                          for u in row["units"] if u.get("eta_tower") and u["eta_tower"]["s"] <= 8 and u["side"] == "enemy"]

    # implausible opponent events: detector-side troops starting deep in Ryley's
    # half are his own units (side channel wrong), not opponent plays
    from .detect import SPELL_CLASSES
    exempt = SPELL_CLASSES | {"miner", "goblin-drill", "graveyard", "goblin-barrel"}
    events = [e for e in events if not (e["player"] == "opponent" and e["detect_source"] == "arena" and e.get("card")
                                        and e.get("tile") and e["tile"][1] < 13 and e["card"] not in exempt)]
    # a deploy label for a card Ryley has HUD-confirmed twice in this game is
    # his play the hand reader missed, not the opponent's (mirror matches are
    # rare; the level line, when both differ, already separates them)
    hud_cards = Counter(e["card"] for e in events if e["player"] == "own" and e.get("card") and e["detect_source"] == "hud")
    for e in events:
        if e["player"] == "opponent" and e["detect_source"] == "deploy_label" and hud_cards.get(e.get("card"), 0) >= 2:
            e["player"] = "own"
            e["confidence"] = "medium"
            e["elixir_before"] = e["elixir_after"] = None
            e["detail"] += " [reattributed to Ryley: card is HUD-confirmed in his deck this game]"
    # own deck: confirmed plays first (HUD / deploy label), then confident hand reads
    played_hud = Counter(e["card"] for e in events if e["player"] == "own" and e.get("card") and e["detect_source"] == "hud")
    played_label = Counter(e["card"] for e in events if e["player"] == "own" and e.get("card") and e["detect_source"] == "deploy_label")
    opp_label = Counter(e["card"] for e in events if e["player"] == "opponent" and e.get("card") and e["detect_source"] == "deploy_label")
    # a label-only own card that the opponent also demonstrably deploys is an
    # attribution error (bridge label offset), not one of Ryley's cards
    played = Counter(played_hud)
    for c, n in played_label.items():
        if c not in played_hud and opp_label.get(c, 0) >= n:
            continue
        played[c] += n
    hand_cards = Counter()
    for s in match_states:
        for c, cf in zip(s["own"].get("hand") or [], s["own"].get("hand_conf") or []):
            if c and cf >= 0.6:
                hand_cards[c] += 1
    # deploy labels are far more reliable than hand reads: label-confirmed
    # cards rank first, then HUD plays by count
    own_deck = sorted(played, key=lambda c: (-(played_label.get(c, 0) > 0), -played[c]))
    for c, n in hand_cards.most_common():
        if c not in own_deck and n >= 30 and len(own_deck) < 8:
            own_deck.append(c)
    own_deck = own_deck[:8]
    own_deck_sources = {c: ("played" if c in played else "hand") for c in own_deck}
    from .decktracker import OpponentDeckTracker
    dt = OpponentDeckTracker(kb_decks or [])
    # opponent cards that are Ryley's confirmed cards are misattributions
    events = [e for e in events if not (e["player"] == "opponent" and e.get("card") in played and e["detect_source"] == "arena")]
    for e in events:   # opponent deck for THIS game, replayed from its (filtered) events
        if e["player"] == "opponent":
            if e.get("card"):
                dt.check_cycle(e["card"])
            dt.observe_play(e.get("card"))
    replayed = dt.summary()
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
    text = " ".join(c["text"] for c in cues).lower()
    mentioned_all = count_card_mentions(text, card_names)
    mentioned = dict(list(mentioned_all.items())[:15])
    # per-game sanity: a HUD-read deck card he never names, when a card he
    # names repeatedly has some visual evidence but no slot, is a hand misread
    deck_notes = []
    visual = set(played_label) | set(hand_cards) | {e["card"] for e in events if e.get("card") and e["player"] == "own"}
    for cand, m in mentioned_all.items():
        if m < 8 or cand in own_deck or cand not in visual:
            continue
        if len(own_deck) < 8:
            own_deck.append(cand)
            own_deck_sources[cand] = "transcript+visual"
            deck_notes.append(f"added {cand} ({m} mentions, visual evidence but no confident slot read)")
            continue
        # only weakly supported cards (<= 2 HUD plays, or hand-read only) can be displaced
        silent = [c for c in own_deck if mentioned_all.get(c, 0) == 0 and played_hud.get(c, 0) <= 2]
        if not silent:
            continue
        drop = min(silent, key=lambda c: played.get(c, 0))
        deck_notes.append(f"swapped {drop} (never named in this game's commentary, weak evidence) for {cand} ({m} mentions, visual evidence)")
        own_deck[own_deck.index(drop)] = cand
        own_deck_sources[cand] = "transcript+visual"
        own_deck_sources.pop(drop, None)
    return {**header, "start_t": round(t0, 1), "end_t": round(t1, 1),
            "cards_mentioned": mentioned,
            "own_deck_observed": own_deck, "own_deck_sources": own_deck_sources, "own_deck_notes": deck_notes,
            "own_deck_counts": {"hud": dict(played_hud), "label": dict(played_label), "hand": {c: k for c, k in hand_cards.items() if k >= 30}},
            "own_deck_key": "-".join(sorted(own_deck)) if len(own_deck) == 8 else None,
            "opponent": {"deck_known": replayed["deck_known"], "deck_complete": replayed["deck_complete"],
                         "deck_predictions": replayed["deck_predictions"], "kb_matches": replayed["kb_matches"],
                         "cycle_confirmed": replayed["cycle_confirmed"], "cycle_violations": replayed["cycle_violations"],
                         "live_tracker_deck_known": opp.get("deck_known")},
            "events": events, "timeline": timeline, "transcript": cues, "quality": quality}


def render_context_md(ctx: dict, card_names: dict[str, str]) -> str:
    n = lambda slug: card_names.get(slug, slug) if slug else "?"
    def src_tag(c: str) -> str:
        src = ctx.get("own_deck_sources", {}).get(c)
        return {"played": "", "hand": " (hand read only)", None: ""}.get(src, f" ({src})")
    if ctx.get("empty"):
        return f"# {ctx.get('video_id')} match {ctx.get('match_index')}: no readable match\n"
    L = [f"# Match context: {ctx.get('title', '')} (video {ctx.get('video_id')}, match {ctx.get('match_index')})", "",
         f"- Video time {ctx['start_t']}s to {ctx['end_t']}s ({ctx['quality']['readable_seconds']}s readable). "
         f"Calibration: {ctx.get('calibration_method')}.",
         f"- Own deck observed (Ryley): {', '.join(n(c) + src_tag(c) for c in ctx['own_deck_observed']) or 'unknown'}"
         + (f" (deck_key `{ctx['own_deck_key']}`)" if ctx.get('own_deck_key') else " (incomplete)")
         + (("; " + "; ".join(ctx["own_deck_notes"])) if ctx.get("own_deck_notes") else ""),
         *([f"- Own deck, video level: this video's games use DIFFERENT decks (deck-showcase format); rely on the per-game read above and the commentary"]
           if ctx.get("own_deck_video", {}).get("mixed") else []),
         *([f"- Own deck, video-level consensus across {ctx['own_deck_video']['games']} game(s) of this video: "
            f"{', '.join(n(c) for c in ctx['own_deck_video']['deck'])}"
            + (f" (deck_key `{ctx['own_deck_video']['deck_key']}`)" if ctx['own_deck_video'].get('deck_key') else " (incomplete)")
            + ("; use this when the per-game read above is incomplete unless the commentary says he switched decks" )]
           if ctx.get("own_deck_video") and not ctx["own_deck_video"].get("mixed") else []),
         *([f"- Hero variants possible: {ctx['hero_note']}"] if ctx.get("hero_note") else []),
         *([f"- Cards named in this game's commentary (mentions): " + ", ".join(f"{n(c)} {k}" for c, k in ctx["cards_mentioned"].items())]
           if ctx.get("cards_mentioned") else []),
         f"- Opponent cards seen: {', '.join(n(c) for c in (ctx['opponent']['deck_known'] or []))}"
         + (" (complete)" if ctx['opponent'].get('deck_complete') else ""),
         f"- Quality: events {ctx['quality']['events_total']} ({ctx['quality']['events_unidentified']} unidentified), "
         f"sources {ctx['quality']['events_by_source']}, hand confidence {ctx['quality']['hand_conf_mean']}, "
         f"own-elixir drift {ctx['quality']['own_elixir_drift']}", "",
         "Tile coordinates: (col 0-17 left to right, row 0-31 bottom to top); Ryley's half is rows 0-14, river 15-16, "
         "opponent half 17-31. Unit positions are Kalman-tracked estimates (heading and 2 s prediction from a "
         "constant-velocity model seeded with the card's speed class); 'THREATS' lists enemy units within ~8 s of a tower. "
         "Detections come from a detector frozen in 2024 (newer cards may be missing or "
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
        units = ", ".join(f"{u['class']}({u['side'][0]})@{tuple(u['tile'])}" + (f" {u['heading']}" if u.get("heading") and u["heading"] != "stationary" else "")
                          for u in row["units"][:10])
        if row.get("threats"):
            units += " | THREATS: " + "; ".join(row["threats"])
        L.append(f"- {row['t']:.0f}s clock {row['clock']} {row['phase'] or ''} | elixir {row['own_elixir']} / opp~{row['opp_elixir_est']} | "
                 f"hand {[n(c) if c else '-' for c in (row['hand'] or [])]} next {n(row['next'])} | towers own {row['towers_own']} enemy {row['towers_enemy']} | {units}")
    while ci < len(ctx["transcript"]):
        L.append(f"> [{ctx['transcript'][ci]['t']:.0f}s] {ctx['transcript'][ci]['text']}")
        ci += 1
    return "\n".join(L) + "\n"


def video_deck_consensus(ctx_paths: list[Path], card_names: dict[str, str]) -> dict:
    """Games in one video are almost always played with the same deck. Pool
    the per-game evidence (HUD-confirmed plays weigh 3, label-only 2, hand
    reads 1 per game), keep the top 8, stamp `own_deck_video` into every
    game's context.json and re-render its context.md. Returns the consensus."""
    ctxs = []
    for p in ctx_paths:
        try:
            c = json.loads(p.read_text())
        except Exception:
            continue
        if c.get("empty") or "own_deck_counts" not in c:
            continue
        ctxs.append((p, c))
    score: Counter = Counter()
    for _, c in ctxs:
        cnt = c["own_deck_counts"]
        for card in cnt.get("hud", {}):
            score[card] += 3
        for card in cnt.get("label", {}):
            score[card] += 2 if card in c.get("own_deck_observed", []) else 0
        for card in cnt.get("hand", {}):
            score[card] += 1
    # transcript evidence: how often Ryley names each candidate card across
    # the video. A hand-read card he never mentions is suspect when a card he
    # keeps naming has some visual evidence but was scored below the cut.
    text = " ".join(cue.get("text", "") for _, c in ctxs for cue in c.get("transcript", [])).lower()
    all_m = count_card_mentions(text, card_names)
    mentions = {c: all_m.get(c, 0) for c in score}
    ranked = [c for c, _ in score.most_common()]
    deck, notes = ranked[:8], []
    # deck-showcase videos switch decks between games: if two games with solid
    # HUD evidence barely overlap, there is no single deck to pool
    hud_sets = [set(c["own_deck_counts"].get("hud", {})) for _, c in ctxs]
    hud_sets = [h for h in hud_sets if len(h) >= 5]
    # hand-read noise costs 2-3 cards per game, so one deck shows a median
    # pairwise Jaccard of ~0.6-0.9 between games; deck-showcase videos sit at
    # 0.2-0.4 (measured on the first five videos). Titles like "TOP 5 decks"
    # are a second signal.
    import re as _re
    pj = sorted(len(a & b) / len(a | b) for i, a in enumerate(hud_sets) for b in hud_sets[i + 1:])
    median_j = pj[len(pj) // 2] if pj else 1.0
    title = next((c.get("title", "") for _, c in ctxs), "")
    title_hint = bool(_re.search(r"\btop\s*\d+\b|\bdecks\b", title, _re.I))
    mixed = median_j < 0.5 or title_hint
    if mixed:
        notes.append(f"games in this video use different decks (median HUD overlap {median_j:.2f}"
                     + (", title suggests a deck showcase" if title_hint else "") + "); no video-level deck")
        deck = []
    for cand in ranked[8:]:
        if mentions.get(cand, 0) < 20:
            continue
        silent = [c for c in deck if mentions.get(c, 0) == 0]
        if not silent:
            break
        drop = min(silent, key=lambda c: score[c])
        notes.append(f"swapped {drop} (score {score[drop]}, never mentioned) for {cand} (score {score[cand]}, {mentions[cand]} mentions)")
        deck[deck.index(drop)] = cand
    consensus = {"games": len(ctxs), "deck": deck, "deck_key": "-".join(sorted(deck)) if len(deck) == 8 else None,
                 "mixed": mixed, "scores": dict(score.most_common()), "transcript_mentions": {c: m for c, m in mentions.items() if m},
                 "notes": notes}
    heroes = _hero_abilities()
    hero_note = "; ".join(f"{card_names.get(c, c)} as a Hero has the ability '{heroes[c]['name']}' costing {heroes[c]['cost']} elixir "
                          f"(an own elixir drop of {heroes[c]['cost']} with no hand change is usually this ability, not a play)"
                          for c in deck if c in heroes)
    for p, c in ctxs:
        c["own_deck_video"] = consensus
        if hero_note:
            c["hero_note"] = hero_note
        p.write_text(json.dumps(c, indent=1))
        (p.parent / "context.md").write_text(render_context_md(c, card_names))
    return consensus


_ALIASES = {
    "elite-barbarians": ["e-barbs", "e barbs", "ebarbs", "elite barbs"], "valkyrie": ["valk"], "battle-ram": ["ram"],
    "hog-rider": ["hog"], "mega-knight": ["mk", "mega night"], "the-log": ["log"], "electro-wizard": ["e-wiz", "ewiz"],
    "electro-giant": ["e-giant", "egiant"], "electro-dragon": ["e-drag", "edrag"], "electro-spirit": ["e-spirit"],
    "p-e-k-k-a": ["pekka", "pecka"], "mini-p-e-k-k-a": ["mini pekka"], "goblin-barrel": ["barrel"], "royal-giant": ["rg"],
    "lava-hound": ["lava", "hound"], "x-bow": ["xbow", "x bow", "crossbow"], "inferno-tower": ["inferno"],
    "inferno-dragon": ["inferno drag"], "baby-dragon": ["baby drag"], "skeleton-army": ["skarmy", "skeleton army"],
    "mega-minion": ["mega minion"], "giant-skeleton": ["giant skelly"], "royal-hogs": ["hogs"], "goblin-giant": ["gob giant"],
    "fire-spirit": ["fire spirit"], "ice-spirit": ["ice spirit"], "dart-goblin": ["dart gob"], "spear-goblins": ["spear gobs"],
    "wall-breakers": ["wall breakers", "wb"], "heal-spirit": ["heal spirit"], "mother-witch": ["mother witch", "mw"],
    "goblin-drill": ["drill"], "little-prince": ["prince"], "archer-queen": ["aq", "queen"], "skeleton-king": ["sk"],
    "golden-knight": ["gk"], "monk": ["monk"], "mighty-miner": ["mighty miner"], "phoenix": ["phoenix"],
    "goblin-machine": ["gob machine"], "royal-recruits": ["recruits"], "cannon-cart": ["cart"], "bomb-tower": ["bomb tower"],
    "night-witch": ["night witch"], "magic-archer": ["magic archer", "marcher"], "tornado": ["nado"], "earthquake": ["eq"],
}


def transcript_mentions(text: str, slug: str, name: str) -> int:
    """Count whole-word mentions of a card (its name plus common abbreviations)."""
    import re as _re
    terms = {name.lower(), name.lower().replace("-", " "), slug.replace("-", " ")}
    terms.update(_ALIASES.get(slug, []))
    n = 0
    for t in terms:
        if len(t) < 2:
            continue
        n += len(_re.findall(rf"(?<![a-z]){_re.escape(t)}s?(?![a-z])", text))
    return n


def _hero_abilities() -> dict[str, dict]:
    """base card slug -> {name, cost} from the Phase 1 hero index."""
    p = Path(__file__).resolve().parents[1] / "knowledge_base" / "meta" / "hero_index.json"
    try:
        h = json.loads(p.read_text())
    except Exception:
        return {}
    out = {}
    for e in h.get("heroes", []):
        try:
            out[e["base_slug"]] = {"name": e.get("ability_name"), "cost": int(e.get("ability_cost"))}
        except (TypeError, ValueError, KeyError):
            continue
    return out


def count_card_mentions(text: str, card_names: dict[str, str]) -> dict[str, int]:
    """Mentions per card; longer names are matched first and blanked so
    'ice wizard' does not also count as 'wizard'."""
    import re as _re
    work = text.lower()
    out: dict[str, int] = {}
    order = sorted(card_names.items(), key=lambda kv: -len(kv[1]))
    for slug, name in order:
        terms = {name.lower(), name.lower().replace("-", " "), slug.replace("-", " ")} | set(_ALIASES.get(slug, []))
        n = 0
        for t in sorted(terms, key=len, reverse=True):
            if len(t) < 2:
                continue
            pat = _re.compile(rf"(?<![a-z]){_re.escape(t)}s?(?![a-z])")
            n += len(pat.findall(work))
            work = pat.sub(" ", work)
        if n:
            out[slug] = n
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
