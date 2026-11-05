"""Offline evaluation of the tactics engine (MAC-Z3): no models, CPU only.

  python3 tools/eval/eval_tactics.py [--dataset DIR] [--splits val,test] [--out PATH]

1. Play records: for every `states.jsonl` record of the splits whose action is a known-card play, run the engine on the
   state and report coverage (a best move exists), top-1 / top-3 card agreement with the pro play and, where the card
   matches, lane / zone agreement and the mean tile distance - overall and split by push / no push.
2. 4-way grid: for every `<dataset>/ranker/<split>.jsonl` line the tactics ranker picks the option with the highest
   `engine.score_key` (None = -inf, ties -> first); its accuracy is printed next to the reference rows.
Writes JSON to --out (default <dataset>/eval/tactics.json) and a Markdown copy next to it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.game_rules import lane_of, load_cards, zone_of  # noqa: E402
from clashbot.kg import KnowledgeGraph  # noqa: E402
from clashbot.tactics import TacticalEngine  # noqa: E402

DEFAULT_SPLITS = ("val", "test")
SITUATIONS = ("all", "push", "no_push")
# val / test accuracy of the other rankers on the 4-way files (docs/data/d1_finetune_joint_eval_v1.json)
REFERENCE = {"random": {"val": 0.25, "test": 0.25},
             "placement_prior": {"val": 0.493, "test": 0.384},
             "d1_finetuned": {"val": 0.524, "test": 0.483}}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_kg():
    """The knowledge graph if the db loads, else None (the engine then skips its KG bonus)."""
    try:
        return KnowledgeGraph()
    except Exception as exc:                      # noqa: BLE001 - any load failure just means "no KG"
        print(f"note: no knowledge graph ({type(exc).__name__}: {exc}); engine runs without KG scores")
        return None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _placement(move, action: dict) -> tuple[bool, bool, float | None]:
    """(lane agrees, zone agrees, tile distance or None) of an engine move against the pro action."""
    lane_ok = lane_of(move.tile) == action.get("lane")
    zone_ok = zone_of(move.tile) == action.get("zone")
    tile = action.get("tile")
    dist = math.dist(move.tile, tile[:2]) if isinstance(tile, (list, tuple)) and len(tile) >= 2 else None
    return lane_ok, zone_ok, dist


class _Row:
    """Counters of one (split, situation) cell of the play-record table."""

    def __init__(self) -> None:
        self.n = self.covered = self.top1 = self.top3 = 0
        self.place = {"top1": [], "top3": []}          # (lane_ok, zone_ok, dist) of the matched move

    def add(self, report, action: dict) -> None:
        self.n += 1
        self.covered += report.best is not None
        card = action["card"]
        if report.best is not None and report.best.card == card:
            self.top1 += 1
            self.place["top1"].append(_placement(report.best, action))
        top3 = [m for m in report.moves[:3] if m.card == card]
        if top3:
            self.top3 += 1
            self.place["top3"].append(_placement(top3[0], action))

    def summary(self) -> dict:
        out = {"n": self.n, "coverage": _rate(self.covered, self.n), "top1": _rate(self.top1, self.n),
               "top3": _rate(self.top3, self.n)}
        for name, items in self.place.items():
            dists = [d for _, _, d in items if d is not None]
            out[f"{name}_matched"] = {
                "n": len(items), "lane": _rate(sum(1 for lane, _, _ in items if lane), len(items)),
                "zone": _rate(sum(1 for _, zone, _ in items if zone), len(items)),
                "tile_dist": round(sum(dists) / len(dists), 2) if dists else None}
        return out


def play_table(records: list[dict], reports: dict[str, object], splits) -> dict:
    rows = {s: {sit: _Row() for sit in SITUATIONS} for s in splits}
    for rec in records:
        action = rec.get("action") or {}
        if action.get("type") != "play" or not action.get("card"):
            continue
        report = reports[rec["id"]]
        for sit in ("all", "push" if report.pushes else "no_push"):
            rows[rec["split"]][sit].add(report, action)
    return {s: {sit: row.summary() for sit, row in rows[s].items()} for s in splits}


def grid_table(engine, lines_by_split: dict[str, list[dict]], states: dict[str, dict],
               reports: dict[str, object]) -> dict:
    out = {}
    for split, lines in lines_by_split.items():
        n = hit = scored = scored_hit = missing = 0
        by_sit = {"push": [0, 0], "no_push": [0, 0]}               # [lines, hits]
        for line in lines:
            rec = states.get(line["id"])
            gold = next((i for i, o in enumerate(line["options"]) if o.get("source") == "pos"), None)
            if rec is None or gold is None:
                missing += 1
                continue
            report = reports[line["id"]]
            scores = [engine.score_key(o["key"], rec["state"], report) for o in line["options"]]
            vals = [-math.inf if s is None else s for s in scores]
            pick = 0
            for i, v in enumerate(vals):
                if v > vals[pick]:
                    pick = i                                       # ties -> the first option
            ok = pick == gold
            n += 1
            hit += ok
            if any(s is not None for s in scores):
                scored += 1
                scored_hit += ok
            sit = by_sit["push" if report.pushes else "no_push"]
            sit[0] += 1
            sit[1] += ok
        out[split] = {"n": n, "acc": _rate(hit, n), "scored_share": _rate(scored, n),
                      "acc_when_scored": _rate(scored_hit, scored), "skipped_no_state": missing,
                      "by_situation": {k: {"n": v[0], "acc": _rate(v[1], v[0])} for k, v in by_sit.items()}}
    return out


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def render_markdown(result: dict) -> str:
    splits = result["splits"]
    lines = ["## Play records: engine vs the pro play", "",
             "| split | situation | n | coverage | top-1 card | top-3 card | lane (top-1) | zone (top-1) "
             "| tile dist (top-1) | n top-1 | lane (top-3) | zone (top-3) | tile dist (top-3) | n top-3 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in splits:
        for sit in SITUATIONS:
            r = result["play"][s][sit]
            m1, m3 = r["top1_matched"], r["top3_matched"]
            lines.append(f"| {s} | {sit.replace('_', ' ')} | {r['n']} | {_pct(r['coverage'])} | {_pct(r['top1'])} "
                         f"| {_pct(r['top3'])} | {_pct(m1['lane'])} | {_pct(m1['zone'])} | {m1['tile_dist']} "
                         f"| {m1['n']} | {_pct(m3['lane'])} | {_pct(m3['zone'])} | {m3['tile_dist']} | {m3['n']} |")
    lines += ["", "## 4-way grid: tactics ranker vs the other rankers", "",
              "| split | n | tactics ranker | random | placement prior | d1 fine-tuned | scored share "
              "| acc when scored | acc push | acc no push |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in splits:
        g = result["grid"][s]
        ref = result["reference"]
        lines.append(f"| {s} | {g['n']} | {_pct(g['acc'])} | {_pct(ref['random'].get(s))} "
                     f"| {_pct(ref['placement_prior'].get(s))} | {_pct(ref['d1_finetuned'].get(s))} "
                     f"| {_pct(g['scored_share'])} | {_pct(g['acc_when_scored'])} "
                     f"| {_pct(g['by_situation']['push']['acc'])} | {_pct(g['by_situation']['no_push']['acc'])} |")
    lines += ["", "lane / zone / tile dist (top-1) are measured on the best move when its card is the pro card; (top-3) on "
              "the first of the top-3 moves with the pro card. Tile distance is in tiles. Scored share = grid lines "
              "where at least one option got an engine score (else the ranker picks the first option).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="comma-separated splits (default val,test)")
    ap.add_argument("--out", type=Path, default=None, help="JSON path (default <dataset>/eval/tactics.json)")
    args = ap.parse_args(argv)
    dataset = Path(args.dataset)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    out = Path(args.out) if args.out else dataset / "eval" / "tactics.json"

    cards = load_cards()
    kg = load_kg()
    engine = TacticalEngine(cards, kg=kg)
    states = {r["id"]: r for r in read_jsonl(dataset / "states.jsonl") if r.get("split") in splits}
    lines_by_split = {s: read_jsonl(dataset / "ranker" / f"{s}.jsonl") for s in splits}

    reports: dict[str, object] = {}
    for rid, rec in states.items():
        reports[rid] = engine.analyse(rec["state"], rec["t"])
    result = {"dataset": str(dataset), "splits": list(splits), "kg": kg is not None, "reference": REFERENCE,
              "play": play_table(list(states.values()), reports, splits),
              "grid": grid_table(engine, lines_by_split, states, reports)}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    md = render_markdown(result)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
