"""Build the clean training dataset (WP1) from knowledge_base/states + matches.

Reads every `knowledge_base/states/<game_id>.jsonl`, the matching
`knowledge_base/matches/<game_id>.md` frontmatter, the card table and the unit
map, and writes into the output directory: states.jsonl (clean samples),
games.jsonl, provenance.jsonl, splits.json, CLEANING_REPORT.md, MANIFEST.json.

  python3 tools/data/build_clean_dataset.py [--out DIR] [--games ID1,ID2,...]

--out defaults to clashbot.config.DATASET_DIR; --games restricts the build to
those game ids (splits are still computed from all state files).

Deterministic: same inputs -> byte-identical outputs except MANIFEST.json's built_at.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import (  # noqa: E402
    DATASET_DIR, DATASET_VERSION, DECKS_DIR, MATCHES_DIR, SPLIT_FRACTIONS, STATES_DIR,
)
from clashbot.game_rules import action_key, is_playable, lane_of, load_cards, zone_of  # noqa: E402
from clashbot.state_text import render_context, render_state_text, render_threats  # noqa: E402
from clashbot.units import load_unit_map, resolve  # noqa: E402

DROPPED_KEYS = ("schema_version", "source", "commentary", "context_refs")
BLOCKING = frozenset({"deck_invalid", "action_unknown", "action_card_unknown",
                      "placement_unknown", "action_card_off_deck"})
ALL_FLAGS = ("deck_invalid", "deck_incomplete", "action_unknown", "action_card_unknown",
             "placement_unknown", "action_card_off_deck", "phase_unknown", "hand_unknown",
             "units_side_repaired", "units_unverified", "opp_deck_capped")
KINDS = ("key", "play", "periodic")
SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- frontmatter

def parse_frontmatter(path: Path) -> dict:
    """Tiny `key: value` parser for the lines between the first two `---` lines."""
    wanted = ("own_deck_key", "own_archetype", "opponent_archetype_guess", "result",
              "video_time", "match_index", "video_id", "opponent_deck_seen")
    out: dict = {k: None for k in wanted}
    if not path.exists():
        return out
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        return out
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not line or line[0] in " \t" or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key not in wanted:
            continue
        if value.startswith("[") and value.endswith("]"):
            out[key] = [x.strip() for x in value[1:-1].split(",") if x.strip()]
        else:
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                value = value[1:-1]
            out[key] = value
    return out


# --------------------------------------------------------------------------- scrub

def scrub(text: str) -> str:
    text = re.sub(r"\(\s*t\s*[=≈~][^)]*\)", "", text)
    text = re.sub(r"\bt\s*[=≈~]\s*[~≈]?\s*\d+(?:\.\d+)?(?:\s*[-–]\s*[~≈]?\d+(?:\.\d+)?)?\s*(?:—|–|:)?\s*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\((?:\.\./)*[^)]*?\.md\)", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def scrub_enrichment(enr):
    if not isinstance(enr, dict):
        return enr
    out = {}
    for k, v in enr.items():
        if isinstance(v, str):
            out[k] = scrub(v)
        elif isinstance(v, list):
            out[k] = [scrub(x) if isinstance(x, str) else x for x in v]
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- splits

def read_first_video_id(path: Path) -> str:
    with path.open(encoding="utf-8") as fh:
        return json.loads(fh.readline())["source"]["video_id"]


def compute_splits(files: list[Path]) -> dict[str, list[str]]:
    videos = sorted({read_first_video_id(p) for p in files})
    videos.sort(key=lambda v: hashlib.sha1(v.encode()).hexdigest())
    n = len(videos)
    n_train = round(SPLIT_FRACTIONS[0] * n)
    n_val = round(SPLIT_FRACTIONS[1] * n)
    return {"train": videos[:n_train], "val": videos[n_train:n_train + n_val],
            "test": videos[n_train + n_val:]}


# --------------------------------------------------------------------------- cleaning helpers

def clean_plays(plays, cards: dict) -> list[dict]:
    out = []
    for p in plays or []:
        q = dict(p)
        card = q.get("card")
        if not (card and is_playable(card, cards)):
            q["card"] = None
        out.append(q)
    return out


def own_deck_of(samples: list[dict], cards: dict) -> tuple[list[str], str]:
    raw = Counter(tuple(s["state"]["own"]["deck"] or ()) for s in samples).most_common(1)[0][0]
    deck = sorted({c for c in raw if is_playable(c, cards)})
    status = "valid" if len(deck) == 8 else "incomplete" if len(deck) == 7 else "invalid"
    return deck, status


def opp_play_counts(samples: list[dict], cards: dict) -> tuple[Counter, dict[str, float]]:
    """(play count per card, earliest raw play time per card)."""
    seen: set = set()
    first: dict[str, float] = {}
    for s in samples:
        plays = list(s["state"]["opponent"].get("recent_plays") or []) + \
            list((s.get("outcome") or {}).get("opponent_plays") or [])
        for p in plays:
            card = p.get("card")
            if card and is_playable(card, cards):
                seen.add((round(p["t"], 1), card))
                first[card] = min(first.get(card, p["t"]), p["t"])
    return Counter(card for _, card in seen), first


def game_opp_deck(samples: list[dict], play_count: Counter, fm_seen, cards: dict
                  ) -> tuple[list[str], list[str], str]:
    """(opp_deck, dropped, opponent_deck_source) - Amendment A1."""
    fm: list[str] = []
    for c in fm_seen or []:
        if is_playable(c, cards) and c not in fm:
            fm.append(c)
    if 1 <= len(fm) <= 8:
        return sorted(fm), [], "match_file"
    if len(fm) > 8:
        top = sorted(fm, key=lambda c: (-play_count.get(c, 0), fm.index(c)))[:8]
        return sorted(top), sorted(set(fm) - set(top)), "match_file_capped"
    union = {c for s in samples for c in (s["state"]["opponent"].get("deck_known") or [])
             if is_playable(c, cards)} | set(play_count)
    if len(union) <= 8:
        return sorted(union), [], "perception"
    top = sorted(sorted(union, key=lambda c: (-play_count.get(c, 0), c))[:8])
    return top, sorted(union - set(top)), "perception_capped"


def first_seen_times(samples: list[dict], opp_deck: list[str], play_first: dict[str, float]
                     ) -> dict[str, float]:
    """Amendment A2: earliest play or deck_known observation per opp_deck card."""
    first: dict[str, float] = {c: play_first[c] for c in opp_deck if c in play_first}
    for s in samples:
        for c in s["state"]["opponent"].get("deck_known") or []:
            if c in opp_deck:
                first[c] = min(first.get(c, s["t"]), s["t"])
    return first


def clean_units(raw_units: list[dict], deck_status: str, own_deck: list[str], opp_deck: list[str],
                cards: dict, unit_map: dict) -> tuple[list[dict], int, bool]:
    """Returns (units, number flipped, any unit with no candidates)."""
    units, flipped, unverified = [], 0, False
    own_set, opp_set = set(own_deck), set(opp_deck)
    for u in raw_units or []:
        r = resolve(u["unit"], cards, unit_map)
        side, repaired, eta = u["side"], False, u.get("eta_tower")
        if not r["candidates"]:
            unverified = True
        if deck_status != "invalid":
            cand = set(r["candidates"])
            own_ok, opp_ok = bool(cand & own_set), bool(cand & opp_set)
            if side == "ally" and not own_ok and opp_ok:
                side, repaired = "enemy", True
            elif side == "enemy" and not opp_ok and own_ok:
                side, repaired = "ally", True
            if repaired:
                eta = None
                flipped += 1
        units.append({"unit_class": u["unit"], "card": r["card"], "is_evo": r["is_evo"],
                      "side": side, "side_repaired": repaired, "tile": u.get("tile"),
                      "heading": u.get("heading"), "speed_tiles_s": u.get("speed_tiles_s"),
                      "predicted_tile_2s": u.get("predicted_tile_2s"), "eta_tower": eta})
    return units, flipped, unverified


def clean_sample(s: dict, game_id: str, split: str, own_deck: list[str], deck_status: str,
                 opp_deck: list[str], first_seen: dict[str, float], capped: bool,
                 cards: dict, unit_map: dict) -> tuple[dict, int]:
    st, act = s["state"], s["action"]
    own, opp = st["own"], st["opponent"]
    padded = (list(own.get("hand") or [])[:4] + [None] * 4)[:4]
    hand = [h if (h and is_playable(h, cards)) else None for h in padded]
    nxt = own.get("next_card")
    nxt = nxt if (nxt and is_playable(nxt, cards)) else None

    seen_dk = sorted((c for c in opp_deck if c in first_seen and first_seen[c] <= s["t"]),
                     key=lambda c: (first_seen[c], c))

    units, flipped, unverified = clean_units(st.get("units"), deck_status, own_deck, opp_deck,
                                             cards, unit_map)
    state = {
        "clock": st.get("clock"), "phase": st.get("phase"), "match_seconds": st.get("match_seconds"),
        "own": {
            "elixir": own.get("elixir"), "hand_confidence": own.get("hand_confidence"),
            "hero_cards": own.get("hero_cards"), "towers": own.get("towers"),
            "hand": hand, "next_card": nxt, "deck": own_deck, "deck_status": deck_status,
            "recent_plays": clean_plays(own.get("recent_plays"), cards),
        },
        "opponent": {
            "elixir_estimate": opp.get("elixir_estimate"), "towers": opp.get("towers"),
            "deck_known": seen_dk, "deck_complete": len(seen_dk) == len(opp_deck) == 8,
            "recent_plays": clean_plays(opp.get("recent_plays"), cards),
        },
        "units": units,
        "threats": render_threats(units),
    }

    a_type = act.get("type")
    a_card = act.get("card")
    a_card = a_card if (a_card and is_playable(a_card, cards)) else None
    tile = act.get("tile")
    lane, zone = lane_of(tile), zone_of(tile)
    key = action_key(a_card, lane, zone) if a_type == "play" else ("hold" if a_type == "hold" else None)
    action = {
        "type": a_type, "card": a_card, "tile": tile, "lane": lane, "zone": zone, "key": key,
        "delay_s": act.get("delay_s"), "elixir_before": act.get("elixir_before"),
        "elixir_after": act.get("elixir_after"), "detect_source": act.get("detect_source"),
        "confidence": act.get("confidence"), "window_s": act.get("window_s"),
        "following_plays": clean_plays(act.get("following_plays"), cards),
    }

    outcome = copy.deepcopy(s.get("outcome"))
    if isinstance(outcome, dict):
        outcome["opponent_plays"] = clean_plays(outcome.get("opponent_plays"), cards)

    km = copy.deepcopy(s.get("key_moment"))
    if isinstance(km, dict) and isinstance(km.get("text"), str):
        km["text"] = scrub(km["text"])

    quality = {k: v for k, v in (s.get("quality") or {}).items() if k != "state_source"}

    flags = set()
    if deck_status == "invalid":
        flags.add("deck_invalid")
    if deck_status == "incomplete":
        flags.add("deck_incomplete")
    if a_type == "unknown":
        flags.add("action_unknown")
    if a_type == "play" and a_card is None:
        flags.add("action_card_unknown")
    if a_type == "play" and tile is None:
        flags.add("placement_unknown")
    if a_type == "play" and a_card is not None and deck_status == "valid" and a_card not in own_deck:
        flags.add("action_card_off_deck")
    if state["phase"] is None:
        flags.add("phase_unknown")
    if all(h is None for h in hand):
        flags.add("hand_unknown")
    if any(u["side_repaired"] for u in units):
        flags.add("units_side_repaired")
    if unverified:
        flags.add("units_unverified")
    if capped:
        flags.add("opp_deck_capped")
    blocking = sorted(flags & BLOCKING)

    rec = {
        "id": s["id"], "game_id": game_id, "split": split, "kind": s.get("kind"), "t": s["t"],
        "state": state, "action": action, "outcome": outcome, "key_moment": km,
        "enrichment": scrub_enrichment(copy.deepcopy(s.get("enrichment"))),
        "quality": quality, "flags": sorted(flags), "train_eligible": not blocking,
        "exclusion_reasons": blocking,
        "state_text": render_state_text(state, action, s["t"]),
        "context_text": render_context(state, s["t"]),
    }
    return rec, flipped


# --------------------------------------------------------------------------- build

def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def read_samples(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["t"])


def build_game(path: Path, split_of_video: dict[str, str], cards: dict, unit_map: dict) -> dict:
    game_id = path.stem
    samples = read_samples(path)
    fm = parse_frontmatter(MATCHES_DIR / f"{game_id}.md")
    src = samples[0]["source"]
    split = split_of_video[src["video_id"]]
    own_deck, deck_status = own_deck_of(samples, cards)
    play_count, play_first = opp_play_counts(samples, cards)
    opp_deck, dropped, opp_source = game_opp_deck(samples, play_count, fm.get("opponent_deck_seen"), cards)
    capped = opp_source.endswith("_capped")
    first_seen = first_seen_times(samples, opp_deck, play_first)

    records, repairs = [], 0
    for s in samples:
        rec, flipped = clean_sample(s, game_id, split, own_deck, deck_status, opp_deck, first_seen,
                                    capped, cards, unit_map)
        records.append(rec)
        repairs += flipped

    key = fm.get("own_deck_key")
    if not (deck_status == "valid" and key and (DECKS_DIR / f"{key}.md").exists()):
        key = None
    result = fm.get("result")
    game_flags = sorted(({"opp_deck_capped"} if capped else set())
                        | ({"deck_invalid"} if deck_status == "invalid" else set())
                        | ({"deck_incomplete"} if deck_status == "incomplete" else set()))
    kinds = Counter(r["kind"] for r in records)
    game = {
        "game_id": game_id, "split": split, "own_deck": own_deck, "deck_status": deck_status,
        "own_deck_key": key, "own_archetype": fm.get("own_archetype"),
        "opponent_deck": opp_deck, "opponent_dropped": dropped,
        "opponent_deck_source": opp_source,
        "opponent_archetype_guess": fm.get("opponent_archetype_guess"),
        "result": result if isinstance(result, str) else None,
        "n_samples": {k: kinds.get(k, 0) for k in KINDS},
        "n_train_eligible": sum(1 for r in records if r["train_eligible"]),
        "side_repairs": repairs, "flags": game_flags,
    }
    prov = {"game_id": game_id, "video_id": src.get("video_id"), "match_index": src.get("match_index"),
            "creator": src.get("creator"), "video_title": src.get("video_title"),
            "video_url": src.get("video_url"), "video_time": fm.get("video_time")}
    return {"records": records, "game": game, "prov": prov, "dropped": dropped}


def write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def line_count(path: Path) -> int:
    return path.read_bytes().count(b"\n")


def md_table(headers: list[str], rows: list[list]) -> str:
    def esc(x) -> str:
        return str(x).replace("|", "\\|")

    out = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_report(records: list[dict], games: list[dict], unit_counts: Counter,
                 flipped_classes: Counter, unresolved: Counter) -> str:
    out = ["# Cleaning report", ""]
    out += ["## Totals", "", f"- games: {len(games)}", f"- samples: {len(records)}"]
    kc = Counter(r["kind"] for r in records)
    out += ["- samples by kind: " + ", ".join(f"{k}={kc.get(k, 0)}" for k in KINDS)]
    src = Counter(g["opponent_deck_source"] for g in games)
    out += ["- games by opponent_deck_source: " + ", ".join(f"{k}={v}" for k, v in sorted(src.items())), "",
            "Train-eligible by split x kind:", ""]
    rows = []
    for sp in SPLIT_NAMES:
        counts = [sum(1 for r in records if r["split"] == sp and r["kind"] == k and r["train_eligible"])
                  for k in KINDS]
        rows.append([sp, *counts, sum(counts)])
    out += [md_table(["split", *KINDS, "total"], rows), ""]

    out += ["## Flag counts", ""]
    fc = Counter(f for r in records for f in r["flags"])
    out += [md_table(["flag", "samples", "blocking"],
                     [[f, fc.get(f, 0), "yes" if f in BLOCKING else "no"] for f in ALL_FLAGS]), ""]

    out += ["## Deck status per game (non-valid)", ""]
    bad = [g for g in games if g["deck_status"] != "valid"]
    out += [f"- {g['game_id']}: {g['deck_status']} (own_deck length {len(g['own_deck'])})" for g in bad] \
        or ["- none"]
    out += [""]

    out += ["## Opponent decks capped", ""]
    capped = [g for g in games if "opp_deck_capped" in g["flags"]]
    out += [f"- {g['game_id']}: dropped {', '.join(g['opponent_dropped'])}" for g in capped] or ["- none"]
    out += [""]

    out += ["## Side repairs", "", f"- total flipped units: {sum(g['side_repairs'] for g in games)}", "",
            "Top 20 unit classes flipped:", ""]
    top = sorted(flipped_classes.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    out += [md_table(["unit_class", "count"], [[k, v] for k, v in top]) if top else "- none", ""]

    out += ["## Unresolved unit classes", ""]
    out += [md_table(["unit_class", "count"],
                     [[k, v] for k, v in sorted(unresolved.items(), key=lambda kv: (-kv[1], kv[0]))])
            if unresolved else "- none", ""]

    out += ["## Action keys", "", "Top 30 keys among train-eligible play samples:", ""]
    keys = Counter(r["action"]["key"] for r in records
                   if r["train_eligible"] and r["action"]["type"] == "play" and r["action"]["key"])
    topk = sorted(keys.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    out += [md_table(["key", "count"], [[k, v] for k, v in topk]) if topk else "- none", ""]
    holds = sum(1 for r in records if r["train_eligible"] and r["action"]["key"] == "hold")
    out += [f"- hold (train-eligible): {holds}", ""]
    return "\n".join(out)


def source_commit() -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True)
        return res.stdout.strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    ap.add_argument("--games", default=None, help="comma separated game ids")
    args = ap.parse_args(argv)

    all_files = sorted(STATES_DIR.glob("*.jsonl"), key=lambda p: p.name)
    splits = compute_splits(all_files)
    split_of_video = {v: sp for sp in SPLIT_NAMES for v in splits[sp]}
    files = all_files
    if args.games:
        wanted = [g.strip() for g in args.games.split(",") if g.strip()]
        by_stem = {p.stem: p for p in all_files}
        missing = [g for g in wanted if g not in by_stem]
        if missing:
            raise SystemExit(f"unknown game ids: {', '.join(missing)}")
        files = [p for p in all_files if p.stem in set(wanted)]

    cards, unit_map = load_cards(), load_unit_map()
    records: list[dict] = []
    games: list[dict] = []
    provs: list[dict] = []
    flipped_classes: Counter = Counter()
    unresolved: Counter = Counter()
    unit_counts: Counter = Counter()
    for p in files:
        g = build_game(p, split_of_video, cards, unit_map)
        records += g["records"]
        games.append(g["game"])
        provs.append(g["prov"])
        for r in g["records"]:
            for u in r["state"]["units"]:
                unit_counts[u["unit_class"]] += 1
                if u["side_repaired"]:
                    flipped_classes[u["unit_class"]] += 1
                if resolve(u["unit_class"], cards, unit_map)["kind"] == "unknown":
                    unresolved[u["unit_class"]] += 1

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "states.jsonl", "".join(dumps(r) + "\n" for r in records))
    write_text(out / "games.jsonl", "".join(dumps(g) + "\n" for g in games))
    write_text(out / "provenance.jsonl", "".join(dumps(x) + "\n" for x in provs))
    splits_doc = {"method": "sha1(video_id) order, 70/15/15 by video", **splits}
    write_text(out / "splits.json", json.dumps(splits_doc, ensure_ascii=False, indent=2) + "\n")
    write_text(out / "CLEANING_REPORT.md", build_report(records, games, unit_counts, flipped_classes, unresolved))
    manifest = {
        "dataset_version": DATASET_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": source_commit(),
        "builder": "tools/data/build_clean_dataset.py",
        "files": {name: {"sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
                         "lines": line_count(out / name)}
                  for name in ("states.jsonl", "games.jsonl", "provenance.jsonl", "splits.json")},
    }
    write_text(out / "MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"games: {len(games)}")
    print(f"samples: {len(records)}")
    print(f"train-eligible: {sum(1 for r in records if r['train_eligible'])}")
    print(f"side repairs: {sum(g['side_repairs'] for g in games)}")
    print(f"capped games: {sum(1 for g in games if 'opp_deck_capped' in g['flags'])}")
    print(f"output dir: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
