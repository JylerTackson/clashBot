"""Offline evaluation of the lookup-vs-tactics arbiter (MAC-AB A4, D-045): no model, CPU only.

  python3 tools/eval/eval_arbiter.py [--dataset DIR] [--d1-eval PATH] [--lines DIR] [--out PATH]

The LLM (d1) makes every play decision; the arbiter only narrows its options. For every val / test line of
`<lines>` (default `<dataset>/d1/joint_tac`):
  - d1's probabilities come from the `per_example` of --d1-eval, matched on (split, id); they are in the line's criteria
    order (checked: the eval's `pred` / `gold` must equal the line's option at the argmax / the line's answer);
  - the candidate sources come from the criteria notes (`tactics:` -> tac 1.0; `pros played this in similar states
    (support X)` -> knn X; `known counter` -> kg 1; `; pros played this too` -> knn 0.5 unless a value is given);
  - `urgent` comes from the `tactics: push ... tower in Xs` lines of the state (X <= the ETA threshold).
Strategies, all with d1 picking (argmax of its probabilities restricted to a subset):
  d1 alone; d1 restricted to tac + the best lookup candidate when unsure; d1 restricted to kNN-supported + tac when
  unsure; the full RuleArbiter (grid over conf, knn_min, urgent ETA, chosen on val and applied to test). Two non-LLM
  baselines (tactics-first, knn-first) are printed as diagnostics only. Writes JSON to --out (default
  <dataset>/eval/arbiter.json) and a Markdown copy next to it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.arbiter import (ArbiterConfig, RuleArbiter, argmax, best_lookup, knn_indices,  # noqa: E402
                              pick_within, source_value, tac_indices)
from clashbot.config import DATASET_DIR  # noqa: E402

SPLITS = ("val", "test")
GRID_CONF = [round(0.30 + 0.05 * i, 2) for i in range(11)]          # 0.30 ... 0.80
GRID_KNN = [0.2, 0.4, 0.6, 0.8, 1.0]
GRID_ETA = [2.0, 4.0, 6.0]
DEFAULTS = {"conf": ArbiterConfig.conf, "knn_min": ArbiterConfig.knn_min, "urgent_eta_s": ArbiterConfig.urgent_eta_s}
SCALE = {"conf": 0.5, "knn_min": 0.8, "urgent_eta_s": 4.0}           # tie-break distance to the defaults

TAC_MARK = " — tactics: "
KNN_RE = re.compile(r"pros played this in similar states \(support ([0-9]*\.?[0-9]+)\)")
PROS_TOO = "; pros played this too"
KG_MARK = "known counter"
PUSH_RE = re.compile(r"^tactics: push \S+: .*tower in ([0-9]+(?:\.[0-9]+)?)s")


class Cand(NamedTuple):
    key: str
    sources: dict


@dataclass
class Example:
    id: str
    cands: list[Cand]
    gold: int
    probs: list[float]
    min_eta: float | None          # smallest `tower in Xs` of the state's tactics push lines (None = no push)

    @property
    def d1(self) -> int:
        return argmax(self.probs)

    def urgent(self, eta: float) -> bool:
        return self.min_eta is not None and self.min_eta <= eta


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# -- parsing ------------------------------------------------------------------------------------------------------------
def parse_sources(criterion: str) -> dict:
    """The candidate sources a criterion text (option description + " — note") stands for."""
    sources: dict = {}
    if TAC_MARK in criterion:
        sources["tac"] = 1.0
    m = KNN_RE.search(criterion)
    if m:
        sources["knn"] = float(m.group(1))
    elif PROS_TOO in criterion:
        sources["knn"] = 0.5
    if KG_MARK in criterion:
        sources["kg"] = 1
    return sources


def push_etas(state_text: str) -> list[float]:
    """Seconds until each enemy push reaches an own tower, from the state's `tactics: push ... tower in Xs` lines."""
    out = []
    for line in state_text.splitlines():
        m = PUSH_RE.match(line)
        if m:
            out.append(float(m.group(1)))
    return out


def load_examples(lines: list[dict], d1_entries: dict[tuple[str, str], list[dict]], split: str
                  ) -> tuple[list[Example], dict]:
    """Join the lines with d1's per-example probabilities; also returns the order / coverage checks."""
    examples: list[Example] = []
    check = {"lines": len(lines), "joined": 0, "missing_d1": 0, "size_mismatch": 0, "pred_mismatch": 0,
             "gold_mismatch": 0}
    for line in lines:
        q = (line.get("questions") or {}).get("play") or {}
        criteria = q.get("criteria") or {}
        keys = list(criteria)
        entries = d1_entries.get((split, line["id"]))
        if not entries:
            check["missing_d1"] += 1
            continue
        entry = next((e for e in entries if e.get("variant") == line.get("variant")), entries[0])
        probs = [float(p) for p in entry["probs"]]
        if len(probs) != len(keys):
            check["size_mismatch"] += 1
            continue
        gold_key = (line.get("answers") or {}).get("play")
        if gold_key not in keys:
            check["gold_mismatch"] += 1
            continue
        if entry.get("pred") is not None and entry["pred"] != keys[argmax(probs)]:
            check["pred_mismatch"] += 1               # the probs are not in this line's criteria order
        if entry.get("gold") is not None and entry["gold"] != gold_key:
            check["gold_mismatch"] += 1
        etas = push_etas(line.get("state", ""))
        examples.append(Example(line["id"], [Cand(k, parse_sources(criteria[k])) for k in keys], keys.index(gold_key),
                                probs, min(etas) if etas else None))
        check["joined"] += 1
    return examples, check


def load_d1_entries(path: Path) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for e in json.loads(Path(path).read_text())["per_example"]:
        out.setdefault((e["split"], e["id"]), []).append(e)
    return out


# -- strategies: (example, **params) -> (index, rule, subset) -----------------------------------------------------------
def s_d1(ex: Example) -> tuple[int, str, list[int]]:
    return ex.d1, "d1", list(range(len(ex.cands)))


def _best_by(ex: Example, name: str) -> int | None:
    best, best_v = None, 0.0
    for i, c in enumerate(ex.cands):
        v = source_value(c, name)
        if v is not None and v > best_v:
            best, best_v = i, v
    return best


def s_tactics_first(ex: Example) -> tuple[int, str, list[int]]:
    """Non-LLM baseline: the highest tac candidate, else d1."""
    i = _best_by(ex, "tac")
    return (ex.d1, "d1", list(range(len(ex.cands)))) if i is None else (i, "tactics", [i])


def s_knn_first(ex: Example) -> tuple[int, str, list[int]]:
    """Non-LLM baseline: the candidate with the highest kNN support, else d1."""
    i = _best_by(ex, "knn")
    return (ex.d1, "d1", list(range(len(ex.cands)))) if i is None else (i, "knn", [i])


def _restricted(ex: Example, conf: float, subset: list[int], name: str) -> tuple[int, str, list[int]]:
    everyone = list(range(len(ex.cands)))
    if max(ex.probs) >= conf:
        return ex.d1, "d1 confident", everyone
    if not subset:
        return ex.d1, "d1", everyone
    return pick_within(ex.probs, subset)[0], name, subset


def s_restrict_tac_knn(ex: Example, conf: float) -> tuple[int, str, list[int]]:
    """When unsure, d1 chooses among the tac candidates and the best lookup candidate (only if there is a tac one)."""
    tac = tac_indices(ex.cands)
    lookup = best_lookup(ex.cands)
    subset = sorted(set(tac) | ({lookup} if lookup is not None else set())) if tac else []
    return _restricted(ex, conf, subset, "d1 among tactics+lookup")


def s_restrict_knn_tac(ex: Example, conf: float, knn_min: float) -> tuple[int, str, list[int]]:
    """When unsure, d1 chooses among the kNN-supported (>= knn_min) and the tac candidates."""
    pros = knn_indices(ex.cands, knn_min)
    subset = sorted(set(pros) | set(tac_indices(ex.cands))) if pros else []
    return _restricted(ex, conf, subset, "d1 among pro-supported + tactics")


def s_arbiter(ex: Example, conf: float, knn_min: float, urgent_eta_s: float) -> tuple[int, str, list[int]]:
    d = RuleArbiter(ArbiterConfig(conf, urgent_eta_s, knn_min)).decide(ex.cands, ex.probs, ex.urgent(urgent_eta_s))
    return d.index, d.reason, d.subset


# -- scoring ------------------------------------------------------------------------------------------------------------
def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score(examples: list[Example], fn: Callable[[Example], tuple[int, str, list[int]]]) -> dict:
    """Accuracy of a strategy, how often it changed d1's pick (fixed / broken) and a breakdown by rule."""
    n = correct = changed = fixed = broken = 0
    rules: dict[str, dict] = {}
    for ex in examples:
        i, rule, subset = fn(ex)
        ok = i == ex.gold
        d1_ok = ex.d1 == ex.gold
        n += 1
        correct += ok
        if i != ex.d1:
            changed += 1
            fixed += ok and not d1_ok
            broken += d1_ok and not ok
        r = rules.setdefault(rule, {"fired": 0, "correct": 0, "d1_correct": 0, "gold_in_subset": 0,
                                    "d1_in_subset": 0, "subset_size": 0, "changed": 0})
        r["fired"] += 1
        r["correct"] += ok
        r["d1_correct"] += d1_ok
        r["gold_in_subset"] += ex.gold in subset
        r["d1_in_subset"] += ex.d1 in subset
        r["subset_size"] += len(subset)
        r["changed"] += i != ex.d1
    out_rules = {}
    for name, r in rules.items():
        f = r["fired"]
        out_rules[name] = {"fired": f, "share": _rate(f, n), "acc": _rate(r["correct"], f),
                           "d1_acc": _rate(r["d1_correct"], f), "gold_in_subset": _rate(r["gold_in_subset"], f),
                           "d1_in_subset": _rate(r["d1_in_subset"], f), "mean_subset": round(r["subset_size"] / f, 2),
                           "changed": r["changed"]}
    return {"n": n, "correct": correct, "acc": _rate(correct, n), "changed": changed, "fixed": fixed,
            "broken": broken, "rules": out_rules}


def _distance(params: dict) -> float:
    return sum(abs(v - DEFAULTS[k]) / SCALE[k] for k, v in params.items())


def run_grid(val: list[Example], test: list[Example], fn: Callable, grid: list[dict]) -> dict:
    """Score `fn(ex, **params)` for every grid point on val and test; the chosen point is the best on val alone
    (ties -> closest to the defaults, then the smallest parameters)."""
    rows = []
    for params in grid:
        bound = lambda ex, p=params: fn(ex, **p)          # noqa: E731
        rows.append({"params": params, "val": score(val, bound), "test": score(test, bound)})
    best = min(rows, key=lambda r: (-r["val"]["correct"], _distance(r["params"]), tuple(r["params"].values())))
    return {"chosen": best, "grid": [{"params": r["params"], "val_acc": r["val"]["acc"], "test_acc": r["test"]["acc"]}
                                     for r in rows]}


def coverage(examples: list[Example]) -> dict:
    n = len(examples)

    def share(pred) -> float | None:
        return _rate(sum(1 for ex in examples if pred(ex)), n)

    def has(ex, name, minimum=0.0):
        return any((source_value(c, name) or 0.0) > 0 and (source_value(c, name) or 0.0) >= minimum for c in ex.cands)

    def gold_has(ex, name, minimum=0.0):
        v = source_value(ex.cands[ex.gold], name) or 0.0
        return v > 0 and v >= minimum

    out = {"n": n, "mean_options": round(sum(len(ex.cands) for ex in examples) / n, 2) if n else None,
           "with_tac": share(lambda ex: has(ex, "tac")), "with_knn": share(lambda ex: has(ex, "knn")),
           "with_knn_ge_0.5": share(lambda ex: has(ex, "knn", 0.5)), "with_kg": share(lambda ex: has(ex, "kg")),
           "gold_has_tac": share(lambda ex: gold_has(ex, "tac")),
           "gold_has_knn": share(lambda ex: gold_has(ex, "knn")),
           "gold_has_knn_ge_0.5": share(lambda ex: gold_has(ex, "knn", 0.5)),
           "gold_has_kg": share(lambda ex: gold_has(ex, "kg")),
           "with_push": share(lambda ex: ex.min_eta is not None)}
    for eta in GRID_ETA:
        out[f"urgent_{eta:g}s"] = share(lambda ex, e=eta: ex.urgent(e))
        out[f"urgent_{eta:g}s_with_tac"] = share(lambda ex, e=eta: ex.urgent(e) and has(ex, "tac"))
    return out


def calibration(examples: list[Example]) -> list[dict]:
    """d1's accuracy when its top probability is >= conf (confident) vs below (unsure), for each conf of the grid."""
    rows = []
    for conf in GRID_CONF:
        sure = [ex for ex in examples if max(ex.probs) >= conf]
        unsure = [ex for ex in examples if max(ex.probs) < conf]
        rows.append({"conf": conf, "confident": len(sure),
                     "acc_confident": _rate(sum(ex.d1 == ex.gold for ex in sure), len(sure)),
                     "unsure": len(unsure), "acc_unsure": _rate(sum(ex.d1 == ex.gold for ex in unsure), len(unsure))})
    return rows


def evaluate(val: list[Example], test: list[Example]) -> dict:
    """All strategies on val and test; every parameter is chosen on val only."""
    def fixed(fn, **kw):
        return {"val": score(val, lambda ex: fn(ex, **kw)), "test": score(test, lambda ex: fn(ex, **kw))}

    arb_grid = [{"conf": c, "knn_min": k, "urgent_eta_s": e} for c in GRID_CONF for k in GRID_KNN for e in GRID_ETA]
    arbiter = run_grid(val, test, s_arbiter, arb_grid)
    tac_knn = run_grid(val, test, s_restrict_tac_knn, [{"conf": c} for c in GRID_CONF])
    knn_tac = run_grid(val, test, s_restrict_knn_tac, [{"conf": c, "knn_min": k} for c in GRID_CONF for k in GRID_KNN])

    def entry(label, row=None, llm=True, **vt):
        base = {"label": label, "llm_decides": llm}
        if row is not None:
            base.update(params=row["params"], val=row["val"], test=row["test"])
        else:
            base.update(params=None, **vt)
        return base

    strategies = {
        "d1": entry("d1 alone", **fixed(s_d1)),
        "restrict_tac_knn": entry("d1 restricted to tac + best lookup when unsure", tac_knn["chosen"]),
        "restrict_knn_tac": entry("d1 restricted to kNN + tac when unsure", knn_tac["chosen"]),
        "arbiter_default": entry("arbiter, default config", None, **fixed(s_arbiter, **DEFAULTS)),
        "arbiter": entry("arbiter, grid chosen on val", arbiter["chosen"]),
        "tactics_first": entry("tactics-first (non-LLM baseline)", llm=False, **fixed(s_tactics_first)),
        "knn_first": entry("knn-first (non-LLM baseline)", llm=False, **fixed(s_knn_first)),
    }
    return {"strategies": strategies, "arbiter_grid": arbiter["grid"], "restrict_tac_knn_grid": tac_knn["grid"],
            "restrict_knn_tac_grid": knn_tac["grid"]}


# -- output -------------------------------------------------------------------------------------------------------------
def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def _params(p: dict | None) -> str:
    if not p:
        return "-"
    return ", ".join(f"{k}={v:g}" for k, v in p.items())


def render_markdown(result: dict) -> str:
    st = result["strategies"]
    out = ["## Accuracy of the strategies (d1 decides; parameters chosen on val, then applied to test)", "",
           "| strategy | parameters | val acc | test acc | test changed d1 | test fixed / broken |",
           "|---|---|---|---|---|---|"]
    for name, s in st.items():
        v, t = s["val"], s["test"]
        out.append(f"| {s['label']} | {_params(s['params'])} | {_pct(v['acc'])} ({v['correct']}/{v['n']}) "
                   f"| {_pct(t['acc'])} ({t['correct']}/{t['n']}) | {t['changed']} | {t['fixed']} / {t['broken']} |")
    arb = st["arbiter"]
    out += ["", f"Chosen arbiter grid point (best val accuracy): {_params(arb['params'])} - val "
                f"{_pct(arb['val']['acc'])}, test {_pct(arb['test']['acc'])}; d1 alone: val {_pct(st['d1']['val']['acc'])}, "
                f"test {_pct(st['d1']['test']['acc'])}.", ""]
    for name in ("arbiter", "arbiter_default"):
        s = st[name]
        out += [f"## Rule firing: {s['label']} ({_params(s['params'] or DEFAULTS)})", "",
                "| split | rule | fired | share | acc when fired | d1 acc on the same | gold in subset | d1 pick in subset "
                "| mean subset size | pick != d1 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for split in SPLITS:
            for rule, r in s[split]["rules"].items():
                out.append(f"| {split} | {rule} | {r['fired']} | {_pct(r['share'])} | {_pct(r['acc'])} "
                           f"| {_pct(r['d1_acc'])} | {_pct(r['gold_in_subset'])} | {_pct(r['d1_in_subset'])} "
                           f"| {r['mean_subset']} | {r['changed']} |")
        out.append("")
    top = sorted(result["arbiter_grid"], key=lambda r: (-r["val_acc"], _distance(r["params"])))[:8]
    out += ["## Arbiter grid: top 8 points on val (with their test accuracy)", "", "| parameters | val acc | test acc |",
            "|---|---|---|"]
    out += [f"| {_params(r['params'])} | {_pct(r['val_acc'])} | {_pct(r['test_acc'])} |" for r in top]
    accs = [r["val_acc"] for r in result["arbiter_grid"]]
    tests = [r["test_acc"] for r in result["arbiter_grid"]]
    out += ["", f"All {len(accs)} grid points: val {_pct(min(accs))} .. {_pct(max(accs))}, test {_pct(min(tests))} .. "
                f"{_pct(max(tests))}.", "", "## d1 calibration: accuracy when its top probability is >= conf (confident) vs below "
                "(unsure)", "", "| conf | val confident | val acc | val unsure | val acc | test confident | test acc "
                "| test unsure | test acc |", "|---|---|---|---|---|---|---|---|---|"]
    for v, t in zip(result["calibration"]["val"], result["calibration"]["test"]):
        out.append(f"| {v['conf']:g} | {v['confident']} | {_pct(v['acc_confident'])} | {v['unsure']} | "
                   f"{_pct(v['acc_unsure'])} | {t['confident']} | {_pct(t['acc_confident'])} | {t['unsure']} | "
                   f"{_pct(t['acc_unsure'])} |")
    cov = result["coverage"]
    out += ["", "## Source coverage (share of lines)", "", "| measure | val | test |", "|---|---|---|"]
    for key in cov["val"]:
        if key != "n":
            fmt = (lambda x: x) if key == "mean_options" else _pct
            out.append(f"| {key} | {fmt(cov['val'][key])} | {fmt(cov['test'][key])} |")
    chk = result["checks"]
    out += ["", "## Input checks", "", "| split | lines | joined | missing d1 | size mismatch | pred mismatch | gold mismatch |",
            "|---|---|---|---|---|---|---|"]
    for split in SPLITS:
        c = chk[split]
        out.append(f"| {split} | {c['lines']} | {c['joined']} | {c['missing_d1']} | {c['size_mismatch']} "
                   f"| {c['pred_mismatch']} | {c['gold_mismatch']} |")
    out += ["", "pred / gold mismatch: the eval's `pred` is not the line's option at the argmax of the probs / its `gold` "
            "is not the line's answer (0 = the probs follow the criteria order). Rules not listed never fired. "
            "`gold in subset` is the ceiling of the rule; `fixed / broken` count lines where the pick differs from "
            "d1's own argmax and is right / wrong where d1 was wrong / right.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--d1-eval", type=Path, default=None,
                    help="d1 eval JSON with per_example probs (default <dataset>/eval/d1_eval_transformers_d1-ft-joint.json)")
    ap.add_argument("--lines", type=Path, default=None, help="dir with val / test jsonl (default <dataset>/d1/joint_tac)")
    ap.add_argument("--out", type=Path, default=None, help="JSON path (default <dataset>/eval/arbiter.json)")
    args = ap.parse_args(argv)
    dataset = Path(args.dataset)
    d1_eval = Path(args.d1_eval) if args.d1_eval else dataset / "eval" / "d1_eval_transformers_d1-ft-joint.json"
    lines_dir = Path(args.lines) if args.lines else dataset / "d1" / "joint_tac"
    out = Path(args.out) if args.out else dataset / "eval" / "arbiter.json"

    d1_entries = load_d1_entries(d1_eval)
    examples: dict[str, list[Example]] = {}
    checks: dict[str, dict] = {}
    for split in SPLITS:
        examples[split], checks[split] = load_examples(read_jsonl(lines_dir / f"{split}.jsonl"), d1_entries, split)
        if not examples[split]:
            print(f"error: no {split} line could be joined with {d1_eval}", file=sys.stderr)
            return 1
        c = checks[split]
        if c["missing_d1"] or c["size_mismatch"] or c["pred_mismatch"] or c["gold_mismatch"]:
            print(f"warning: {split} input checks {c}")

    result = {"d1_eval": str(d1_eval), "lines": str(lines_dir),
              "grid": {"conf": GRID_CONF, "knn_min": GRID_KNN, "urgent_eta_s": GRID_ETA}, "defaults": DEFAULTS,
              "checks": checks, "coverage": {s: coverage(examples[s]) for s in SPLITS},
              "calibration": {s: calibration(examples[s]) for s in SPLITS}}
    result.update(evaluate(examples["val"], examples["test"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    md = render_markdown(result)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
