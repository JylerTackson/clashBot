"""Leave-one-game-out evaluation of the opponent-deck model (MAC-AK K3): CPU only, no models.

  python tools/eval/eval_opp_model.py [--dataset DIR] [--out PATH] [--temperature T] [--count-power P] [--sweep]

For every game whose opponent deck has 7-8 known cards, the first k = 1..5 cards of that deck are revealed in the order
they first appear in the opponent's `recent_plays` (`states.jsonl`, by play time; cards outside the cleaned deck are
dropped), and the model, trained on every deck except that game's own opponent deck, predicts the hidden cards:

  recall@(8-k)   share of the hidden cards among the top (8 - k) predictions;
  precision@3    share of the top 3 predictions that are hidden cards.

Baselines: global card popularity (the same weighted counts, seen cards excluded) and random (the exact expectation over
a uniform pick from the unseen vocabulary). Two protocols: `logo` removes only this game's occurrence of the deck
(leave-one-game-out); `lodo` removes every identical deck from the training set (strict, leave-one-deck-out).
Sections: `all` = every 7-8 card opponent deck, `full8` = the 8-card ones only; each also per split.
--sweep prints a (temperature, count-power) grid on the train split only and writes nothing.
Writes JSON to --out (default <dataset>/eval/opp_model.json) and a Markdown copy next to it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR, KB  # noqa: E402
from clashbot.opp_model import DECK_SIZE, DEFAULT_COUNT_POWER, DEFAULT_TEMPERATURE, OppDeckModel  # noqa: E402
from tools.data.build_deck_cooc import aggregate, build_model, collect_entries, known_card_slugs  # noqa: E402

KS = (1, 2, 3, 4, 5)
SYSTEMS = ("model", "popularity", "random")
PROTOCOLS = ("logo", "lodo")
EXAMPLES = (["giant", "balloon"], ["bats", "night-witch"])
SWEEP_TEMPERATURES = (0.02, 0.03, 0.05, 0.08, 0.12, 0.25)
SWEEP_POWERS = (0.0, 0.5, 1.0)


def reveal_orders(states_path: Path, decks: dict[str, set[str]]) -> dict[str, list[str]]:
    """game_id -> deck cards in order of first appearance in the opponent `recent_plays` (play time `t`)."""
    first: dict[str, dict[str, float]] = defaultdict(dict)
    with open(states_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            gid = rec["game_id"]
            deck = decks.get(gid)
            if deck is None:
                continue
            for p in (rec.get("state") or {}).get("opponent", {}).get("recent_plays") or []:
                card, t = p.get("card"), p.get("t")
                if card in deck and t is not None and t < first[gid].get(card, float("inf")):
                    first[gid][card] = t
    return {gid: [c for c, _ in sorted(times.items(), key=lambda kv: (kv[1], kv[0]))] for gid, times in first.items()}


def score(preds: list[str], hidden: set[str], n: int) -> tuple[float, float]:
    """(recall@n, precision@3) of a ranked card list against the hidden cards."""
    recall = len(set(preds[:n]) & hidden) / len(hidden)
    precision = len(set(preds[:3]) & hidden) / 3
    return recall, precision


def random_score(candidates: int, hidden_in_cand: int, hidden: int, n: int) -> tuple[float, float]:
    """Exact expected (recall@n, precision@3) of picking uniformly from `candidates` unseen cards."""
    if candidates <= 0:
        return 0.0, 0.0
    return min(n, candidates) / candidates * hidden_in_cand / hidden, hidden_in_cand / candidates


def evaluate_game(model: OppDeckModel, deck: set[str], order: list[str], ks=KS) -> dict[int, dict[str, tuple[float, float]]]:
    """{k: {system: (recall@(8-k), precision@3)}} for the ks the game has enough revealed cards for."""
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for k in ks:
        if len(order) < k:
            break
        seen = order[:k]
        hidden = deck - set(seen)
        if not hidden:
            continue
        n = DECK_SIZE - k
        cand = [c for c in model.cards if c not in seen]
        out[k] = {
            "model": score([c for c, _ in model.predict(seen, top=n)], hidden, n),
            "popularity": score([c for c, _ in model.popularity(seen, top=n)], hidden, n),
            "random": random_score(len(cand), len(hidden & set(cand)), len(hidden), n),
        }
    return out


def run_protocol(entries: list[dict], games: dict[str, dict], orders: dict[str, list[str]], protocol: str,
                 temperature: float, count_power: float, only_split: str | None = None) -> list[dict]:
    """One result row per evaluated game: {game_id, split, size, results}."""
    rows = []
    cache: dict[tuple, OppDeckModel] = {}
    for gid, g in games.items():
        if only_split and g["split"] != only_split:
            continue
        deck = set(g["opponent_deck"])
        if not (7 <= len(deck) <= DECK_SIZE) or not orders.get(gid):
            continue
        key = tuple(sorted(deck))
        if protocol == "logo":
            train = [e for e in entries if not (e["source"] == "opp" and e["game_id"] == gid)]
            model = build_model(train, temperature, count_power)
        else:                                                  # lodo: identical decks share one model
            if key not in cache:
                cache[key] = build_model([e for e in entries if e["cards"] != key], temperature, count_power)
            model = cache[key]
        rows.append({"game_id": gid, "split": g["split"], "size": len(deck),
                     "results": evaluate_game(model, deck, orders[gid])})
    return rows


def aggregate_rows(rows: list[dict]) -> dict:
    """{k: {n, <system>: {recall, precision3}}} averaged over the rows that reach k."""
    out = {}
    for k in KS:
        sel = [r["results"][k] for r in rows if k in r["results"]]
        cell: dict = {"n": len(sel)}
        for s in SYSTEMS:
            cell[s] = ({"recall": round(sum(x[s][0] for x in sel) / len(sel), 4),
                        "precision3": round(sum(x[s][1] for x in sel) / len(sel), 4)} if sel else None)
        out[str(k)] = cell
    return out


def sections(rows: list[dict]) -> dict:
    groups = {"all": rows, "full8": [r for r in rows if r["size"] == DECK_SIZE]}
    for split in ("train", "val", "test"):
        groups[f"split_{split}"] = [r for r in rows if r["split"] == split]
    return {name: {"games": len(rs), "by_k": aggregate_rows(rs)} for name, rs in groups.items()}


def examples(model: OppDeckModel) -> list[dict]:
    out = []
    for seen in EXAMPLES:
        arch = model.likely_archetype(seen)
        out.append({"seen": seen, "predict": [[c, round(p, 3)] for c, p in model.predict(seen, top=8)],
                    "archetype": list(arch) if arch else None,
                    "sum_all": round(sum(p for _, p in model.predict(seen, top=len(model.cards))), 4)})
    return out


def sweep(entries, games, orders, split="train") -> None:
    print(f"sweep on split={split}: mean recall@(8-k) for k=1..5 and their average (logo)")
    for power in SWEEP_POWERS:
        for temp in SWEEP_TEMPERATURES:
            by_k = aggregate_rows(run_protocol(entries, games, orders, "logo", temp, power, split))
            recalls = [by_k[str(k)]["model"]["recall"] for k in KS]
            print(f"  power={power:<4} T={temp:<5} " + " ".join(f"{r:.3f}" for r in recalls) + f"  avg={sum(recalls) / len(recalls):.3f}")


def markdown(report: dict) -> str:
    p = report["params"]
    lines = ["# Opponent-deck model: leave-one-game-out evaluation (MAC-AK K3)", "",
             f"Model: temperature {p['temperature']}, count power {p['count_power']}, mix 0.6 nearest decks + 0.4 co-occurrence. "
             f"Training pool: {report['deck_pool']['total_decks']} deck occurrences, {report['deck_pool']['unique_decks']} unique decks, "
             f"{report['deck_pool']['cards']} cards. Evaluated games: {report['sections_logo']['all']['games']} "
             f"({report['sections_logo']['full8']['games']} with a full 8-card deck).", "",
             "Cards are revealed in order of first opponent play; recall is over the hidden cards at top (8-k), precision over the top 3. "
             "`random` is the exact expectation. `logo` drops only this game's deck from training; `lodo` drops every identical deck.", ""]
    for protocol in PROTOCOLS:
        for name, title in (("all", "all 7-8 card opponent decks"), ("full8", "8-card opponent decks only"),
                            ("split_test", "test split only")):
            sec = report[f"sections_{protocol}"][name]
            lines += [f"## {protocol}: {title} ({sec['games']} games)", "",
                      "| k | n | recall@(8-k) model | popularity | random | precision@3 model | popularity | random |",
                      "|---|---|---|---|---|---|---|---|"]
            for k in KS:
                c = sec["by_k"][str(k)]
                if not c["n"]:
                    continue
                lines.append(f"| {k} | {c['n']} | " + " | ".join(f"{c[s]['recall']:.3f}" for s in SYSTEMS) + " | "
                             + " | ".join(f"{c[s]['precision3']:.3f}" for s in SYSTEMS) + " |")
            lines.append("")
    lines += ["## Example predictions (model trained on all decks)", ""]
    for ex in report["examples"]:
        arch = f"; likely archetype {ex['archetype'][0]} ({ex['archetype'][1]:.2f})" if ex["archetype"] else ""
        lines.append(f"- seen {ex['seen']}{arch}: " + ", ".join(f"{c} {p:.2f}" for c, p in ex["predict"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--kb", type=Path, default=KB)
    ap.add_argument("--out", type=Path, default=None, help="default <dataset>/eval/opp_model.json")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--count-power", type=float, default=DEFAULT_COUNT_POWER)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args(argv)

    entries = collect_entries(args.dataset, args.kb, known_card_slugs())
    games = {}
    with open(args.dataset / "games.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                g = json.loads(line)
                games[g["game_id"]] = g
    orders = reveal_orders(args.dataset / "states.jsonl", {gid: set(g["opponent_deck"]) for gid, g in games.items()})
    if args.sweep:
        sweep(entries, games, orders)
        return {}

    report: dict = {"params": {"temperature": args.temperature, "count_power": args.count_power}}
    full = build_model(entries, args.temperature, args.count_power)
    decks, _, _ = aggregate(entries)
    report["deck_pool"] = {"total_decks": len(entries), "unique_decks": len(decks), "cards": len(full.cards)}
    for protocol in PROTOCOLS:
        rows = run_protocol(entries, games, orders, protocol, args.temperature, args.count_power)
        report[f"sections_{protocol}"] = sections(rows)
    report["examples"] = examples(full)
    out = args.out or args.dataset / "eval" / "opp_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f"wrote {out} and {out.with_suffix('.md')}")
    return report


if __name__ == "__main__":
    main()
