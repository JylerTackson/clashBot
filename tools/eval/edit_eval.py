"""Knowledge-editing evaluation (D-034): one SYNTHETIC balance change applied through the non-parametric memory route.

  python3 tools/eval/edit_eval.py [--dataset DIR] [--db PATH] [--card hog-rider] [--attribute elixir_cost]
          [--value 5] [--patch P1] [--n-probe 300] [--out PATH]

The edit goes into a MemoryStore (versioned external fact store, SERAC/GRACE family) and is read back through
`edited_cards`; "no edit" is the unedited card table. Criteria follow the editing literature [46]: reliability,
generalization, locality (other cards / other states), portability, plus conflict handling, versioning and rollback.
Probes are the val + test records of <dataset>/states.jsonl. Writes <out> (JSON) and <out>.md and prints the table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR, DB_PATH  # noqa: E402
from clashbot.d1_format import card_description  # noqa: E402
from clashbot.game_rules import legal_action_keys, legal_cards, load_cards  # noqa: E402
from clashbot.memory import (BASE_PATCH, EDITABLE_ATTRIBUTES, MemoryStore, reweight_hits,  # noqa: E402
                             stale_cards)

PROBE_SPLITS = ("val", "test")
SYNTHETIC_SOURCE = "synthetic_balance_change"
SYNTHETIC_NOTE = "SYNTHETIC balance change (D-034); no real patch is involved"
CONFLICT_POOL = 30      # retrieval depth; the share is measured on the top TOP_K before / after re-weighting
TOP_K = 10
NA_ATTRIBUTE = "n/a for this attribute"


def read_probe_records(dataset: Path) -> list[dict]:
    records = []
    with open(Path(dataset) / "states.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") in PROBE_SPLITS and isinstance(rec.get("state"), dict):
                records.append(rec)
    return records


def _hand(rec: dict) -> list[str]:
    return [h for h in rec["state"]["own"]["hand"] if h]


def _rate(passes: list[bool]) -> float | None:
    return round(sum(passes) / len(passes), 4) if passes else None


def _crit(edited: list[bool], no_edit: list[bool] | float | None = None, **extra) -> dict:
    if isinstance(no_edit, list):
        no_edit = _rate(no_edit)
    return {"n": len(edited), "edited": _rate(edited), "no_edit": no_edit, **extra}


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


def _candidates_use(gen, rec: dict, card: str) -> bool:
    cands = gen.generate(rec["state"], rec["t"], rec["context_text"], n=4, splits=["train"],
                         exclude_game=rec["game_id"])
    return any(c.key.split("|")[0] == card for c in cands)


def run_eval(dataset: Path, db: Path, card: str, attribute: str, value, patch: str, n_probe: int) -> dict:
    base = load_cards()
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
    records = read_probe_records(dataset)

    criteria: dict[str, dict] = {}
    criteria["reliability"] = _reliability(store, base, edited, card, attribute, value)

    # sets of probe records (first n_probe of each, in file order)
    in_hand = [r for r in records if card in _hand(r)]
    other_states = [r for r in records
                    if card not in _hand(r) and card not in (r["state"]["own"].get("deck") or [])][:n_probe]
    gen_set: list[dict] = []
    increase = False
    if attribute == "elixir_cost":
        # elixir values where legality (cost <= e + 0.5) differs between the old and the new cost
        lo, hi = sorted((float(old), float(value)))
        increase = float(value) > float(old)
        gen_set = [r for r in in_hand if r["state"]["own"].get("elixir") is not None
                   and lo - 0.5 <= r["state"]["own"]["elixir"] < hi - 0.5][:n_probe]

    # generalization: the card's legality follows the new cost
    if attribute == "elixir_cost":
        criteria["generalization"] = _crit(
            [(card in legal_cards(r["state"], edited)) == (not increase) for r in gen_set],
            [(card in legal_cards(r["state"], base)) == (not increase) for r in gen_set])
    else:
        criteria["generalization"] = _skipped(NA_ATTRIBUTE)

    # locality: nothing else changes (the no-edit side is the unedited table, so trivially 1.0)
    others = [(c, a) for c in base if c != card for a in EDITABLE_ATTRIBUTES]
    criteria["locality_cards"] = _crit([edited[c][a] == base[c][a] for c, a in others], 1.0)
    criteria["locality_states"] = _crit(
        [legal_action_keys(r["state"], edited) == legal_action_keys(r["state"], base) for r in other_states], 1.0)

    # portability: downstream candidate generation stops proposing a card that became unaffordable
    db = Path(db)
    index = kg = None
    db_note = "no db"
    if db.exists():
        try:
            from clashbot.kg import KnowledgeGraph
            from clashbot.retrieval import StateIndex
            index, kg = StateIndex(db), KnowledgeGraph(db)
        except Exception as exc:                  # missing sidecar files, version mismatch, bad schema
            index = kg = None
            db_note = f"db unusable: {type(exc).__name__}: {exc}"
    if attribute != "elixir_cost":
        criteria["portability"] = _skipped(NA_ATTRIBUTE)
    elif not increase:
        criteria["portability"] = _skipped("n/a for a cost decrease")
    elif index is None:
        criteria["portability"] = _skipped(db_note)
    else:
        from clashbot.candidates import CandidateGenerator
        gen_edited, gen_base = CandidateGenerator(index, kg, edited), CandidateGenerator(index, kg, base)
        criteria["portability"] = _crit([not _candidates_use(gen_edited, r, card) for r in gen_set],
                                        [not _candidates_use(gen_base, r, card) for r in gen_set])

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
                    "pool": CONFLICT_POOL, "top_k": TOP_K, "stale": sorted(stale)}

    versioning = store.card_fact(card, attribute, patch=BASE_PATCH, base_cards=base) == old
    store.retract(fid)                                     # rollback runs last
    rollback = store.edited_cards(base) == base and store.card_fact(card, attribute, base_cards=base) == old
    store.close()

    return {"edit": {"card": card, "attribute": attribute, "old": old, "new": value, "patch": patch,
                     "synthetic": True, "note": SYNTHETIC_NOTE},
            "criteria": criteria, "conflict": conflict, "versioning": versioning, "rollback": rollback,
            "config": {"dataset": str(dataset), "db": str(db), "n_probe": n_probe,
                       "probe_splits": list(PROBE_SPLITS)}}


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render_table(result: dict) -> str:
    e = result["edit"]
    lines = [f"{SYNTHETIC_NOTE}: {e['card']} {e['attribute']} {e['old']} -> {e['new']} in patch {e['patch']} "
             "(non-parametric route: versioned external fact store).", "",
             "| criterion | n | edited | no edit |", "|---|---|---|---|"]
    for name, c in result["criteria"].items():
        if "skipped" in c:
            lines.append(f"| {name} | 0 | skipped: {c['skipped']} | skipped |")
        else:
            lines.append(f"| {name} | {c['n']} | {_fmt(c['edited'])} | {_fmt(c['no_edit'])} |")
    c = result["conflict"]
    label = f"conflict: share of top-{TOP_K} hits using {e['card']} (no edit = before, edited = after re-weighting)"
    if "skipped" in c:
        lines.append(f"| {label} | 0 | skipped: {c['skipped']} | skipped |")
    else:
        lines.append(f"| {label} | {c['n']} | {_fmt(c['after'])} | {_fmt(c['before'])} |")
    lines.append(f"| versioning (value as of {BASE_PATCH} is the old one) | 1 | {result['versioning']} | - |")
    lines.append(f"| rollback (retract restores the base table) | 1 | {result['rollback']} | - |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--card", default="hog-rider")
    ap.add_argument("--attribute", default="elixir_cost", choices=EDITABLE_ATTRIBUTES)
    ap.add_argument("--value", default="5", help="new value (JSON, e.g. 5 or 4.5, else a plain string)")
    ap.add_argument("--patch", default="P1")
    ap.add_argument("--n-probe", type=int, default=300)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out = args.out if args.out is not None else args.dataset / "eval" / "edit_eval.json"
    result = run_eval(args.dataset, args.db, args.card, args.attribute, _parse_value(args.value), args.patch,
                      args.n_probe)
    table = render_table(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    out.with_suffix(".md").write_text(table)
    print(table)
    return result


if __name__ == "__main__":
    main()
