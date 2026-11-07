"""Build dataset/v1/kg2/deck_cooc.json (MAC-AK K3): the known decks and their card co-occurrence counts.

  python tools/data/build_deck_cooc.py [--dataset DIR] [--kb DIR] [--out PATH] [--temperature T] [--count-power P]

Deck sources (identical decks are merged and counted):
  kb   - knowledge_base/decks/*.md card lists (the 68 pro decks);
  own  - games.jsonl `own_deck` where deck_status == "valid" (exactly 8 cards);
  opp  - games.jsonl `opponent_deck` when 7 or 8 cards are known.
  live - with `--live-runs DIR` (e.g. runs/poc): every run directory under DIR adds the opponent cards it had seen at its
         last decision (the `opp deck:` line of the last decisions.jsonl context, else a `deck_known` in the frames'
         perception) as one deck of weight 3 tagged `training_camp` (FIX #42 d). Runs that saw fewer than
         `--live-min-cards` (4) cards are dropped: a deck that small would match any single card too well.
Archetype of a merged deck: the KB deck file, else the game's own_archetype, else the opponent_archetype_guess.
The JSON is `OppDeckModel.to_dict()`: cards, pair_counts, card_counts, decks, deck_counts, deck_archetypes,
deck_weights, deck_tags, params, meta (with the `training_camp` share of the weight).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import CARD_INDEX, DATASET_DIR, KB  # noqa: E402
from clashbot.opp_model import (DECK_SIZE, DEFAULT_COUNT_POWER, DEFAULT_TEMPERATURE, NO_ARCHETYPE,  # noqa: E402
                                TRAINING_CAMP, OppDeckModel)
from tools.data.kb_markdown import _CARD_LINK, bullets, parse_frontmatter, section  # noqa: E402

SOURCE_RANK = {"kb": 0, "own": 1, "opp": 2, "live": 3}        # archetype label priority
MIN_OPP_CARDS = 7
LIVE_WEIGHT = 3.0               # prior weight of a live Training Camp deck (a pro / game deck weighs 1)
LIVE_MIN_CARDS = 4              # a live run that saw fewer opponent cards than this adds no deck
_OPP_DECK_LINE = re.compile(r"opp deck:\s*([^\n]*)")


def parse_deck_file(path: Path) -> dict | None:
    """{deck_key, cards, archetype} of one knowledge_base/decks/*.md file (cards from the `## Cards` links)."""
    fm, body = parse_frontmatter(Path(path).read_text(encoding="utf-8"))
    cards = []
    for b in bullets(section(body, "Cards") or ""):
        m = _CARD_LINK.search(b)
        if m and m.group(1) not in cards:
            cards.append(m.group(1))
    if not cards:
        return None
    return {"deck_key": fm.get("deck_key") or Path(path).stem, "cards": cards, "archetype": fm.get("archetype_primary")}


def load_kb_decks(kb_dir: Path = KB) -> list[dict]:
    out = []
    for p in sorted(Path(kb_dir, "decks").glob("*.md")):
        d = parse_deck_file(p)
        if d:
            out.append(d)
    return out


def known_card_slugs(path: Path = CARD_INDEX) -> set[str] | None:
    try:
        return {c["slug"] for c in json.loads(Path(path).read_text(encoding="utf-8"))["cards"]}
    except (OSError, KeyError, ValueError):
        return None


def _label(value) -> str | None:
    return None if value in NO_ARCHETYPE else value


def deck_from_context(text: str | None) -> list[str]:
    """The opponent cards of an `opp deck: a, b, c` line of a decision context ([] for `unknown` or no line)."""
    m = _OPP_DECK_LINE.search(text or "")
    if not m:
        return []
    return [c.strip() for c in m.group(1).split(",") if c.strip() and c.strip() not in ("unknown", "?")]


def _deck_known_in(obj) -> list[str]:
    """A `deck_known` list anywhere inside a nested frame record (the perception state), [] when absent."""
    if isinstance(obj, dict):
        got = obj.get("deck_known")
        if isinstance(got, list) and got:
            return [c for c in got if isinstance(c, str)]
        for v in obj.values():
            r = _deck_known_in(v)
            if r:
                return r
    return []


def live_run_deck(run_dir: Path) -> list[str]:
    """The opponent cards a live run had seen by its end: the `opp deck:` line of the last decision that has one, else
    the last frame record that carries a `deck_known`. [] when the run has neither."""
    run_dir = Path(run_dir)
    dec = run_dir / "decisions.jsonl"
    if dec.is_file():
        for line in reversed(dec.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                cards = deck_from_context(json.loads(line).get("context_text"))
            except ValueError:
                continue
            if cards:
                return cards
    frames = run_dir / "frames.jsonl"
    if frames.is_file():
        last: list[str] = []
        with frames.open(encoding="utf-8") as fh:
            for line in fh:
                if "deck_known" not in line:
                    continue
                try:
                    cards = _deck_known_in(json.loads(line))
                except ValueError:
                    continue
                last = cards or last
        return last
    return []


def is_video_run(run_dir: Path) -> bool:
    """A run that replayed a recorded video (config `source` "video" or a `video` path): its opponent is a human, not a
    Training Camp bot. A run without a readable config.json counts as live."""
    try:
        cfg = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(cfg, dict) and (cfg.get("source") == "video" or bool(cfg.get("video")))


def collect_live_entries(runs_dir: Path, card_slugs: set[str] | None = None, min_cards: int = LIVE_MIN_CARDS,
                         weight: float = LIVE_WEIGHT, stats: Counter | None = None) -> list[dict]:
    """One `live` entry per run directory under `runs_dir` that saw at least `min_cards` known opponent cards, weight
    `weight`, tagged `training_camp`. Runs that replayed a video are skipped. `stats` counts live_runs, live_video,
    live_no_deck, live_too_small, live_unknown_card."""
    stats = stats if stats is not None else Counter()
    entries = []
    for run in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        if not ((run / "decisions.jsonl").is_file() or (run / "frames.jsonl").is_file()):
            continue
        stats["live_runs"] += 1
        if is_video_run(run):
            stats["live_video"] += 1
            continue
        cards = live_run_deck(run)
        if not cards:
            stats["live_no_deck"] += 1
            continue
        if card_slugs is not None:
            known = [c for c in cards if c in card_slugs]
            stats["live_unknown_card"] += len(cards) - len(known)
            cards = known
        uniq = tuple(sorted(set(cards)))
        if len(uniq) < min_cards or len(uniq) > DECK_SIZE:
            stats["live_too_small"] += 1
            continue
        entries.append({"cards": uniq, "source": "live", "game_id": run.name, "archetype": None,
                        "weight": float(weight), "tags": (TRAINING_CAMP,)})
    return entries


def collect_entries(dataset_dir: Path = DATASET_DIR, kb_dir: Path = KB, card_slugs: set[str] | None = None,
                    stats: Counter | None = None, live_runs: Path | None = None,
                    live_min_cards: int = LIVE_MIN_CARDS, live_weight: float = LIVE_WEIGHT) -> list[dict]:
    """One entry per deck occurrence: {cards: sorted tuple, source: kb|own|opp|live, game_id, archetype}; live entries
    (from `live_runs`, a directory of run directories) also carry `weight` and `tags`.

    `stats` (optional) counts what was dropped: own_not_valid, opp_too_small, unknown_card, and the live_* counters."""
    stats = stats if stats is not None else Counter()
    entries: list[dict] = []

    def add(cards, source, game_id, archetype):
        uniq = tuple(sorted(set(cards)))
        if card_slugs is not None and any(c not in card_slugs for c in uniq):
            stats["unknown_card"] += 1
            return
        entries.append({"cards": uniq, "source": source, "game_id": game_id, "archetype": _label(archetype)})

    for d in load_kb_decks(kb_dir):
        add(d["cards"], "kb", d["deck_key"], d["archetype"])
    with open(Path(dataset_dir) / "games.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            own = set(g.get("own_deck") or [])
            if g.get("deck_status") == "valid" and len(own) == DECK_SIZE:
                add(own, "own", g["game_id"], g.get("own_archetype"))
            else:
                stats["own_not_valid"] += 1
            opp = set(g.get("opponent_deck") or [])
            if MIN_OPP_CARDS <= len(opp) <= DECK_SIZE:
                add(opp, "opp", g["game_id"], g.get("opponent_archetype_guess"))
            else:
                stats["opp_too_small"] += 1
    if live_runs is not None:
        entries += collect_live_entries(Path(live_runs), card_slugs, live_min_cards, live_weight, stats)
    return entries


def aggregate_full(entries: list[dict]) -> tuple[list[tuple[str, ...]], list[int], list[str | None], list[float],
                                                  list[tuple[str, ...]]]:
    """Unique decks (sorted), how many times each occurs, the best archetype label of each, its prior weight (the largest
    `weight` of its entries, default 1) and its tags (the union of its entries')."""
    count: Counter = Counter()
    label: dict[tuple, tuple[int, str]] = {}
    weight: dict[tuple, float] = {}
    tags: dict[tuple, set] = {}
    for e in entries:
        count[e["cards"]] += 1
        weight[e["cards"]] = max(weight.get(e["cards"], 1.0), float(e.get("weight", 1.0)))
        tags.setdefault(e["cards"], set()).update(e.get("tags") or ())
        a = e["archetype"]
        if a is not None:
            rank = SOURCE_RANK[e["source"]]
            if e["cards"] not in label or rank < label[e["cards"]][0]:
                label[e["cards"]] = (rank, a)
    decks = sorted(count)
    return (decks, [count[d] for d in decks], [label[d][1] if d in label else None for d in decks],
            [weight[d] for d in decks], [tuple(sorted(tags[d])) for d in decks])


def aggregate(entries: list[dict]) -> tuple[list[tuple[str, ...]], list[int], list[str | None]]:
    """Unique decks (sorted), how many times each occurs, and the best archetype label of each."""
    return aggregate_full(entries)[:3]


def build_model(entries: list[dict], temperature: float = DEFAULT_TEMPERATURE,
                count_power: float = DEFAULT_COUNT_POWER) -> OppDeckModel:
    decks, counts, archetypes, weights, tags = aggregate_full(entries)
    return OppDeckModel(decks, counts, archetypes, temperature=temperature, count_power=count_power,
                        deck_weights=weights, deck_tags=tags)


def summary(entries: list[dict], model: OppDeckModel, stats: Counter) -> dict:
    by_source = Counter(e["source"] for e in entries)
    unique_by_source = {s: len({e["cards"] for e in entries if e["source"] == s}) for s in SOURCE_RANK}
    sizes = Counter(len(d) for d in model.decks)
    return {"total_decks": len(entries), "unique_decks": len(model.decks), "cards": len(model.cards),
            "total_by_source": dict(by_source), "unique_by_source": unique_by_source,
            "unique_deck_sizes": dict(sorted(sizes.items())), "dropped": dict(stats),
            "decks_with_archetype": sum(a is not None for a in model.archetypes),
            "live_decks": unique_by_source.get("live", 0), "training_camp_decks": model.training_camp_decks(),
            "total_weight": round(model.total_weight(), 3),
            "training_camp_share": round(model.training_camp_share(), 4)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--kb", type=Path, default=KB)
    ap.add_argument("--out", type=Path, default=None, help="default <dataset>/kg2/deck_cooc.json")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--count-power", type=float, default=DEFAULT_COUNT_POWER)
    ap.add_argument("--live-runs", type=Path, default=None,
                    help="directory of live run directories (e.g. runs/poc): adds each run's final opponent deck")
    ap.add_argument("--live-weight", type=float, default=LIVE_WEIGHT, help="prior weight of a live deck")
    ap.add_argument("--live-min-cards", type=int, default=LIVE_MIN_CARDS,
                    help="a live run that saw fewer opponent cards adds no deck")
    args = ap.parse_args(argv)
    out = args.out or args.dataset / "kg2" / "deck_cooc.json"

    stats: Counter = Counter()
    entries = collect_entries(args.dataset, args.kb, known_card_slugs(), stats, args.live_runs,
                              args.live_min_cards, args.live_weight)
    model = build_model(entries, args.temperature, args.count_power)
    info = summary(entries, model, stats)
    model.save(out, meta=info)
    print(json.dumps(info, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return info


if __name__ == "__main__":
    main()
