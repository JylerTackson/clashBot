"""Knowledge-editing evaluation (D-034): SYNTHETIC balance changes applied through the non-parametric memory route.

  python3 tools/eval/edit_eval.py [--dataset DIR] [--db PATH] [--cards auto|slug,slug,...] [--n-probe 300] [--out PATH]
  python3 tools/eval/edit_eval.py --card hog-rider [--attribute elixir_cost] [--value 5] [--patch P1] ...

Default (`--cards auto`): one +1-elixir edit for each of the 5 most-played cards (train-split, train-eligible plays),
every edit in its own fresh MemoryStore, criteria reported per edit and pooled (sum of passes / sum of n).
`--card` keeps the single-edit mode (any editable attribute).

The edit goes into a MemoryStore (versioned external fact store, SERAC/GRACE family) and is read back through
`edited_cards`; "no edit" is the unedited card table. Criteria follow the editing literature [46]: reliability,
generalization, locality (other cards / other states), portability, plus conflict handling, versioning and rollback.
Generalization and portability probe the val + test records where the card is in hand or possible (D-019) and the
elixir level makes legality flip. Probes are read from <dataset>/states.jsonl. Writes <out> (JSON) and <out>.md and
prints the table.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR, DB_PATH  # noqa: E402
from clashbot.d1_format import card_description  # noqa: E402
from clashbot.game_rules import legal_action_keys, legal_cards, load_cards, possible_cards  # noqa: E402
from clashbot.memory import (BASE_PATCH, EDITABLE_ATTRIBUTES, MemoryStore, reweight_hits,  # noqa: E402
                             stale_cards)

PROBE_SPLITS = ("val", "test")
SYNTHETIC_SOURCE = "synthetic_balance_change"
SYNTHETIC_NOTE = "SYNTHETIC balance change (D-034); no real patch is involved"
CONFLICT_POOL = 30      # retrieval depth; the share is measured on the top TOP_K before / after re-weighting
TOP_K = 10
AUTO_K = 5              # `--cards auto`: number of most-played cards to edit
NA_ATTRIBUTE = "n/a for this attribute"


def _iter_records(dataset: Path):
    with open(Path(dataset) / "states.jsonl") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_probe_records(dataset: Path) -> list[dict]:
    return [rec for rec in _iter_records(dataset)
            if rec.get("split") in PROBE_SPLITS and isinstance(rec.get("state"), dict)]


def train_play_counts(dataset: Path, base: dict) -> Counter:
    """action.card counts over train-split, train-eligible play records whose card has a numeric elixir_cost."""
    counts: Counter = Counter()
    for rec in _iter_records(dataset):
        action = rec.get("action") or {}
        card = action.get("card")
        if (rec.get("split") == "train" and rec.get("train_eligible") and action.get("type") == "play"
                and card in base and isinstance(base[card].get("elixir_cost"), (int, float))):
            counts[card] += 1
    return counts


def auto_cards(counts: Counter, k: int = AUTO_K) -> list[str]:
    """The k most frequent cards, ties broken by slug."""
    return [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def _hand(rec: dict) -> list[str]:
    return [h for h in rec["state"]["own"]["hand"] if h]


def _probe_state(state: dict, card: str) -> dict:
    """The state with `card` in hand: a card that is only *possible* (unread slot, D-019) is put into the first empty
    slot, so the legality check asks "would it be playable?" instead of passing trivially in both arms."""
    hand = list(state["own"]["hand"])
    if card in hand:
        return state
    slot = next((i for i, h in enumerate(hand) if not h), None)
    if slot is None:
        hand.append(card)
    else:
        hand[slot] = card
    return {**state, "own": {**state["own"], "hand": hand}}


def _rate(passes: list[bool]) -> float | None:
    return round(sum(passes) / len(passes), 4) if passes else None


def _crit(edited: list[bool], no_edit: list[bool] | float | None = None, **extra) -> dict:
    """One criterion: n, rates, and the raw pass counts (so edits can be pooled exactly)."""
    n = len(edited)
    if isinstance(no_edit, list):
        no_edit_pass, no_edit = sum(no_edit), _rate(no_edit)
    else:
        no_edit_pass = None if no_edit is None else round(no_edit * n)
    return {"n": n, "edited": _rate(edited), "no_edit": no_edit, **extra,
            "edited_pass": sum(edited), "no_edit_pass": no_edit_pass}


def _skipped(reason: str) -> dict:
    return {"n": 0, "edited": None, "no_edit": None, "skipped": reason}


def _parse_value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def _expected_description(attribute: str, value) -> str | None:
    """Substring `d1_format.card_description` must contain once the edit is visible (None: not rendered there)."""
    if attribute == "elixir_cost":
        fmt = str(int(value)) if float(value).is_integer() else str(value)
        return f"{fmt} elixir"
    if attribute == "card_type":
        return f"elixir {str(value).lower()}"
    return None


def _reliability(store: MemoryStore, base: dict, edited: dict, card: str, attribute: str, value) -> dict:
    needle = _expected_description(attribute, value)
    edited_pass = [store.card_fact(card, attribute, base_cards=base) == value, edited[card][attribute] == value]
    no_edit_pass = [store.card_fact(card, attribute, patch=BASE_PATCH, base_cards=base) == value,
                    base[card][attribute] == value]
    if needle is not None:
        edited_pass.append(needle in card_description(card, edited))
        no_edit_pass.append(needle in card_description(card, base))
    return _crit(edited_pass, no_edit_pass)


def _candidates_use(gen, rec: dict, state: dict, card: str) -> bool:
    cands = gen.generate(state, rec["t"], rec["context_text"], n=4, splits=["train"], exclude_game=rec["game_id"])
    return any(c.key.split("|")[0] == card for c in cands)


def _open_db(db: Path):
    """(StateIndex, KnowledgeGraph, note): index/kg are None when the DB is missing or unusable, note says why."""
    db = Path(db)
    if not db.exists():
        return None, None, "no db"
    try:
        from clashbot.kg import KnowledgeGraph
        from clashbot.retrieval import StateIndex
        return StateIndex(db), KnowledgeGraph(db), ""
    except Exception as exc:                      # missing sidecar files, version mismatch, bad schema
        return None, None, f"db unusable: {type(exc).__name__}: {exc}"


def evaluate_edit(records: list[dict], index, kg, db_note: str, base: dict, card: str, attribute: str, value,
                  patch: str, n_probe: int) -> dict:
    """Apply one synthetic edit in a fresh MemoryStore and score every criterion."""
    if card not in base:
        raise ValueError(f"unknown card {card!r}")
    if attribute not in EDITABLE_ATTRIBUTES:
        raise ValueError(f"attribute {attribute!r} is not editable; expected one of {EDITABLE_ATTRIBUTES}")
    old = base[card][attribute]
    if old == value:
        raise ValueError(f"{card}.{attribute} is already {value!r}; the edit would be a no-op")
    if attribute == "elixir_cost":
        float(old), float(value)        # numeric on both sides, else the legality window below is undefined

    store = MemoryStore(":memory:")
    fid = store.apply_edit(card, attribute, value, patch, SYNTHETIC_SOURCE, "edit_eval / D-034")
    edited = store.edited_cards(base)

    criteria: dict[str, dict] = {}
    criteria["reliability"] = _reliability(store, base, edited, card, attribute, value)

    # sets of probe records (first n_probe of each, in file order)
    in_hand = [r for r in records if card in _hand(r)]
    other_states = [r for r in records
                    if card not in _hand(r) and card not in (r["state"]["own"].get("deck") or [])][:n_probe]
    gen_set: list[dict] = []
    increase = False
    if attribute == "elixir_cost":
        # card in hand or possible (D-019), at an elixir where legality (cost <= e + 0.5) differs old vs new cost
        lo, hi = sorted((float(old), float(value)))
        increase = float(value) > float(old)
        gen_set = [r for r in records if r["state"]["own"].get("elixir") is not None
                   and lo - 0.5 <= r["state"]["own"]["elixir"] < hi - 0.5
                   and card in possible_cards(r["state"], base)][:n_probe]
    gen_states = [_probe_state(r["state"], card) for r in gen_set]
    counts = {"n_in_hand": sum(card in _hand(r) for r in gen_set), "n_possible": len(gen_set)}

    # generalization: the card's legality follows the new cost
    if attribute == "elixir_cost":
        criteria["generalization"] = _crit(
            [(card in legal_cards(s, edited)) == (not increase) for s in gen_states],
            [(card in legal_cards(s, base)) == (not increase) for s in gen_states], **counts)
    else:
        criteria["generalization"] = _skipped(NA_ATTRIBUTE)

    # locality: nothing else changes (the no-edit side is the unedited table, so trivially 1.0)
    others = [(c, a) for c in base if c != card for a in EDITABLE_ATTRIBUTES]
    criteria["locality_cards"] = _crit([edited[c][a] == base[c][a] for c, a in others], 1.0)
    criteria["locality_states"] = _crit(
        [legal_action_keys(r["state"], edited) == legal_action_keys(r["state"], base) for r in other_states], 1.0)

    # portability: downstream candidate generation stops proposing a card that became unaffordable
    if attribute != "elixir_cost":
        criteria["portability"] = _skipped(NA_ATTRIBUTE)
    elif not increase:
        criteria["portability"] = _skipped("n/a for a cost decrease")
    elif index is None:
        criteria["portability"] = _skipped(db_note)
    else:
        from clashbot.candidates import CandidateGenerator
        gen_edited, gen_base = CandidateGenerator(index, kg, edited), CandidateGenerator(index, kg, base)
        criteria["portability"] = _crit(
            [not _candidates_use(gen_edited, r, s, card) for r, s in zip(gen_set, gen_states)],
            [not _candidates_use(gen_base, r, s, card) for r, s in zip(gen_set, gen_states)], **counts)

    # conflict handling: pre-patch neighbours about the edited card are down-weighted (ClashEval-style)
    if index is None:
        conflict: dict = {"n": 0, "skipped": db_note}
    else:
        stale = stale_cards(store)
        before, after = [], []
        for r in in_hand[:n_probe]:
            pool = index.search(r["state"], r["t"], r["context_text"], k=CONFLICT_POOL, splits=["train"],
                                exclude_game=r["game_id"])
            top_before = pool[:TOP_K]
            top_after = reweight_hits(pool, stale)[:TOP_K]
            if top_before:
                before.append(sum(h.action_key is not None and h.action_key.split("|")[0] == card
                                  for h in top_before) / len(top_before))
                after.append(sum(h.action_key is not None and h.action_key.split("|")[0] == card
                                 for h in top_after) / len(top_after))
        conflict = {"n": len(before), "before": round(sum(before) / len(before), 4) if before else None,
                    "after": round(sum(after) / len(after), 4) if after else None,
                    "sum_before": sum(before), "sum_after": sum(after),
                    "pool": CONFLICT_POOL, "top_k": TOP_K, "stale": sorted(stale)}

    versioning = store.card_fact(card, attribute, patch=BASE_PATCH, base_cards=base) == old
    store.retract(fid)                                     # rollback runs last
    rollback = store.edited_cards(base) == base and store.card_fact(card, attribute, base_cards=base) == old
    store.close()

    return {"edit": {"card": card, "attribute": attribute, "old": old, "new": value, "patch": patch,
                     "synthetic": True, "note": SYNTHETIC_NOTE},
            "criteria": criteria, "conflict": conflict, "versioning": versioning, "rollback": rollback}


def run_eval(dataset: Path, db: Path, card: str, attribute: str, value, patch: str, n_probe: int) -> dict:
    """Single-edit mode (`--card`)."""
    index, kg, db_note = _open_db(db)
    result = evaluate_edit(read_probe_records(dataset), index, kg, db_note, load_cards(), card, attribute, value,
                           patch, n_probe)
    result["config"] = {"dataset": str(dataset), "db": str(db), "n_probe": n_probe,
                        "probe_splits": list(PROBE_SPLITS)}
    return result


def pool_results(edits: list[dict]) -> tuple[dict, dict]:
    """Pool per-edit criteria (sum of passes / sum of n) and the conflict share (n-weighted mean)."""
    names = list(dict.fromkeys(name for e in edits for name in e["criteria"]))
    pooled: dict[str, dict] = {}
    for name in names:
        parts = [e["criteria"][name] for e in edits if name in e["criteria"]]
        live = [p for p in parts if "skipped" not in p]
        if not live:
            pooled[name] = _skipped(parts[0]["skipped"])
            continue
        n = sum(p["n"] for p in live)
        edited_pass = sum(p["edited_pass"] for p in live)
        no_edit_pass = None if any(p["no_edit_pass"] is None for p in live) else sum(p["no_edit_pass"] for p in live)
        entry = {"n": n, "edited": round(edited_pass / n, 4) if n else None,
                 "no_edit": round(no_edit_pass / n, 4) if n and no_edit_pass is not None else None}
        for key in ("n_in_hand", "n_possible"):
            if any(key in p for p in live):
                entry[key] = sum(p.get(key, 0) for p in live)
        entry["edited_pass"], entry["no_edit_pass"] = edited_pass, no_edit_pass
        pooled[name] = entry

    live_c = [e["conflict"] for e in edits if "skipped" not in e["conflict"]]
    if not live_c:
        conflict = {"n": 0, "skipped": edits[0]["conflict"].get("skipped", "no conflict probes")}
    else:
        n = sum(c["n"] for c in live_c)
        conflict = {"n": n,
                    "before": round(sum(c["sum_before"] for c in live_c) / n, 4) if n else None,
                    "after": round(sum(c["sum_after"] for c in live_c) / n, 4) if n else None,
                    "pool": CONFLICT_POOL, "top_k": TOP_K,
                    "stale": sorted({s for c in live_c for s in c["stale"]})}
    return pooled, conflict


def run_multi(dataset: Path, db: Path, cards_arg: str, patch: str, n_probe: int) -> dict:
    """Several +1-elixir edits (`--cards auto|slug,slug`), one fresh MemoryStore each, plus the pooled criteria."""
    base = load_cards()
    counts = train_play_counts(dataset, base)
    if cards_arg == "auto":
        cards = auto_cards(counts)
        if not cards:
            raise ValueError("--cards auto: no train-eligible play records with a numeric elixir_cost in the dataset")
    else:
        cards = list(dict.fromkeys(c.strip() for c in cards_arg.split(",") if c.strip()))
        for c in cards:
            if c not in base:
                raise ValueError(f"unknown card {c!r}")
            if not isinstance(base[c].get("elixir_cost"), (int, float)):
                raise ValueError(f"{c} has no numeric elixir_cost")
    records = read_probe_records(dataset)
    index, kg, db_note = _open_db(db)
    edits = [evaluate_edit(records, index, kg, db_note, base, c, "elixir_cost", base[c]["elixir_cost"] + 1, patch,
                           n_probe) for c in cards]
    pooled, conflict_pooled = pool_results(edits)
    return {"synthetic": True, "note": SYNTHETIC_NOTE,
            "selection": {"mode": "auto" if cards_arg == "auto" else "explicit", "cards": cards,
                          "train_play_counts": {c: counts.get(c, 0) for c in cards}},
            "edits": edits, "pooled": pooled, "conflict_pooled": conflict_pooled,
            "versioning": all(e["versioning"] for e in edits), "rollback": all(e["rollback"] for e in edits),
            "config": {"dataset": str(dataset), "db": str(db), "n_probe": n_probe, "patch": patch,
                       "probe_splits": list(PROBE_SPLITS)}}


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _num(x) -> str:
    return f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def _label(name: str, c: dict) -> str:
    if "n_possible" in c:
        return f"{name} (in hand {c['n_in_hand']}, hand-or-possible {c['n_possible']})"
    return name


def _criteria_rows(criteria: dict, conflict: dict, card_label: str) -> list[str]:
    lines = []
    for name, c in criteria.items():
        if "skipped" in c:
            lines.append(f"| {name} | 0 | skipped: {c['skipped']} | skipped |")
        else:
            lines.append(f"| {_label(name, c)} | {c['n']} | {_fmt(c['edited'])} | {_fmt(c['no_edit'])} |")
    label = f"conflict: share of top-{TOP_K} hits using {card_label} (no edit = before, edited = after re-weighting)"
    if "skipped" in conflict:
        lines.append(f"| {label} | 0 | skipped: {conflict['skipped']} | skipped |")
    else:
        lines.append(f"| {label} | {conflict['n']} | {_fmt(conflict['after'])} | {_fmt(conflict['before'])} |")
    return lines


def render_table(result: dict) -> str:
    """Single-edit table."""
    e = result["edit"]
    lines = [f"{SYNTHETIC_NOTE}: {e['card']} {e['attribute']} {e['old']} -> {e['new']} in patch {e['patch']} "
             "(non-parametric route: versioned external fact store).", "",
             "| criterion | n | edited | no edit |", "|---|---|---|---|"]
    lines += _criteria_rows(result["criteria"], result["conflict"], e["card"])
    lines.append(f"| versioning (value as of {BASE_PATCH} is the old one) | 1 | {result['versioning']} | - |")
    lines.append(f"| rollback (retract restores the base table) | 1 | {result['rollback']} | - |")
    return "\n".join(lines) + "\n"


def render_pooled(result: dict) -> str:
    """Pooled table plus one row per edited card (n / edited score per criterion)."""
    edits = result["edits"]
    desc = ", ".join(f"{e['edit']['card']} {_num(e['edit']['old'])} -> {_num(e['edit']['new'])}" for e in edits)
    lines = [f"{SYNTHETIC_NOTE}: +1 elixir on each of {len(edits)} cards ({desc}) in patch "
             f"{edits[0]['edit']['patch']}; one fresh MemoryStore per edit; non-parametric route (versioned external "
             "fact store). Pooled = sum of passes / sum of n.", "",
             "| criterion | n | edited | no edit |", "|---|---|---|---|"]
    lines += _criteria_rows(result["pooled"], result["conflict_pooled"], "the edited card")
    lines.append(f"| versioning (value as of {BASE_PATCH} is the old one; all edits) | {len(edits)} | "
                 f"{result['versioning']} | - |")
    lines.append(f"| rollback (retract restores the base table; all edits) | {len(edits)} | {result['rollback']} | - |")

    names = list(result["pooled"])
    lines += ["", "Per card (n / edited score):", "",
              "| card | edit | " + " | ".join(names + ["conflict", "versioning", "rollback"]) + " |",
              "|---|---|" + "---|" * (len(names) + 3)]
    for e in edits:
        cells = []
        for name in names:
            c = e["criteria"][name]
            if "skipped" in c:
                cells.append("skipped")
            elif name in ("generalization", "portability"):
                cells.append(f"{c['n']} ({c['n_in_hand']} in hand) / {_fmt(c['edited'])}")
            else:
                cells.append(f"{c['n']} / {_fmt(c['edited'])}")
        c = e["conflict"]
        cells.append("skipped" if "skipped" in c else f"{c['n']} / {_fmt(c['after'])}")
        cells += [str(e["versioning"]), str(e["rollback"])]
        ed = e["edit"]
        lines.append(f"| {ed['card']} | {_num(ed['old'])} -> {_num(ed['new'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--cards", default=None,
                    help="auto (default without --card): +1 elixir on the 5 most-played cards; or slug,slug,...")
    ap.add_argument("--card", default=None, help="single-edit mode: the card to edit (e.g. hog-rider)")
    ap.add_argument("--attribute", default=None, choices=EDITABLE_ATTRIBUTES,
                    help="single-edit mode only (default elixir_cost)")
    ap.add_argument("--value", default=None,
                    help="single-edit mode only: new value (JSON, e.g. 5 or 4.5, else a plain string; default 5)")
    ap.add_argument("--patch", default="P1")
    ap.add_argument("--n-probe", type=int, default=300)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.card is not None and args.cards is not None:
        ap.error("--card (single edit) and --cards (several +1-elixir edits) are mutually exclusive")
    if args.card is None and (args.value is not None or args.attribute not in (None, "elixir_cost")):
        ap.error("--cards edits elixir_cost by +1; use --card for another attribute or value")

    out = args.out if args.out is not None else args.dataset / "eval" / "edit_eval.json"
    if args.card is not None:
        value = _parse_value(args.value) if args.value is not None else 5
        result = run_eval(args.dataset, args.db, args.card, args.attribute or "elixir_cost", value, args.patch,
                          args.n_probe)
        table = render_table(result)
    else:
        result = run_multi(args.dataset, args.db, args.cards or "auto", args.patch, args.n_probe)
        table = render_pooled(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    out.with_suffix(".md").write_text(table)
    print(table)
    return result


if __name__ == "__main__":
    main()
