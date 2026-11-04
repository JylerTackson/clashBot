"""Offline evaluation of the d1 act-head (MAC-X, D-041): learned vs rules vs combined play/wait on dense data.

  python3 tools/eval/eval_act_head.py --model PATH [--data <dataset>/act_dense_bal] [--backend transformers]
         [--splits val,test] [--limit N] [--out PATH]

Every line of the splits (from `sample_act_train.py` / `build_dense_act.py`, with their `states_<split>.jsonl`
sidecars) is scored once: p = P(act) of the model, label = the pro deployed a card within the horizon. The threshold is
the one with the best balanced accuracy on val (0.05..0.95, step 0.01; ties go to the one closest to 0.5) and is
written to <model>/act_threshold.json, where `clashbot.live --policy learned` reads it. Policies per split: `learned`
(p >= threshold), `rules` (ReactivePolicy.decide(...).act), `combined` (LearnedActPolicy logic: the rules' defend /
leak, else the head), `always`, `never`. The reaction slice repeats it on lines with `meta.since_opp_play <= 3.0`.
Writes <out> (JSON, default <data>/../eval/act_head.json) and <out>.md and prints the tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.policy import LearnedActPolicy, ReactivePolicy  # noqa: E402
from clashbot.ranker import D1Ranker  # noqa: E402

POLICIES = ("learned", "rules", "combined", "always", "never")
DEFAULT_SPLITS = ("val", "test")
REACTION_S = 3.0                      # "shortly after an opponent play": meta.since_opp_play <= this
THRESHOLDS = [round(0.05 + 0.01 * k, 2) for k in range(91)]       # 0.05 .. 0.95


# --------------------------------------------------------------------------- data

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_split(data: Path, split: str, limit: int | None) -> tuple[list[dict], dict[str, dict]]:
    """(lines, {id: state dict}) of one split; `limit` keeps that many evenly spaced lines."""
    lines = read_jsonl(data / f"{split}.jsonl")
    if limit is not None and 0 < limit < len(lines):
        lines = [lines[int(i * len(lines) / limit)] for i in range(limit)]
    sidecar = data / f"states_{split}.jsonl"
    if not sidecar.exists():
        raise SystemExit(f"missing {sidecar}: build the data with build_dense_act.py (state sidecar) / sample_act_train.py")
    states = {r["id"]: r for r in read_jsonl(sidecar)}
    missing = [l["id"] for l in lines if l["id"] not in states]
    if missing:
        raise SystemExit(f"{sidecar} has no state for {len(missing)} lines of {split}, e.g. {missing[0]}")
    return lines, states


class Precomputed:
    """Scorer stand-in for LearnedActPolicy: the probability already computed for each state string."""

    def __init__(self, p_by_state: dict[str, float]):
        self.p_by_state = p_by_state

    def probabilities(self, state: str, questions: dict) -> dict:
        p = self.p_by_state[state]
        return {"act": [1.0 - p, p]}


# --------------------------------------------------------------------------- metrics

def _div(a: int, b: int) -> float | None:
    return a / b if b else None


def metrics(labels: list[bool], preds: list[bool]) -> dict:
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and not p)
    fp = sum(1 for y, p in zip(labels, preds) if not y and p)
    tn = sum(1 for y, p in zip(labels, preds) if not y and not p)
    n = len(labels)
    act_recall, wait_recall = _div(tp, tp + fn), _div(tn, tn + fp)
    rates = [r for r in (act_recall, wait_recall) if r is not None]
    return {"n": n, "accuracy": _div(tp + tn, n),
            "balanced_accuracy": sum(rates) / len(rates) if rates else None,
            "act_rate": _div(tp + fp, n), "pro_act_rate": _div(tp + fn, n),
            "act_precision": _div(tp, tp + fp), "act_recall": act_recall,
            "wait_precision": _div(tn, tn + fn), "wait_recall": wait_recall,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def pick_threshold(labels: list[bool], probs: list[float]) -> tuple[float, float | None]:
    """(threshold, balanced accuracy) maximising balanced accuracy over THRESHOLDS; ties -> closest to 0.5."""
    best: tuple[float, float, float] | None = None          # (-bacc, |thr - 0.5|, thr)
    for thr in THRESHOLDS:
        bacc = metrics(labels, [p >= thr for p in probs])["balanced_accuracy"]
        if bacc is None:
            continue
        key = (-round(bacc, 12), abs(thr - 0.5), thr)
        if best is None or key < best:
            best = key
    if best is None:
        return 0.5, None
    return best[2], -best[0]


def policy_predictions(lines: list[dict], states: dict[str, dict], probs: list[float], threshold: float,
                       rules: ReactivePolicy) -> dict[str, list[bool]]:
    combined = LearnedActPolicy(Precomputed({l["state"]: p for l, p in zip(lines, probs)}), rules=rules,
                                threshold=threshold, min_check_gap_s=0.0)
    out: dict[str, list[bool]] = {name: [] for name in POLICIES}
    for line, p in zip(lines, probs):
        s = states[line["id"]]
        out["learned"].append(p >= threshold)
        out["rules"].append(bool(rules.decide(s["state"], s["t"]).act))
        out["combined"].append(bool(combined.decide(s["state"], s["t"], context_text=line["state"]).act))
        out["always"].append(True)
        out["never"].append(False)
    return out


def evaluate_split(lines: list[dict], states: dict[str, dict], probs: list[float], threshold: float,
                   rules: ReactivePolicy) -> dict:
    labels = [l["answers"]["act"] == "true" for l in lines]
    preds = policy_predictions(lines, states, probs, threshold, rules)
    block = {"n": len(lines), "pro_act_rate": _div(sum(labels), len(labels)),
             "policies": {name: metrics(labels, preds[name]) for name in POLICIES}}
    idx = [i for i, l in enumerate(lines)
           if (l.get("meta") or {}).get("since_opp_play") is not None and l["meta"]["since_opp_play"] <= REACTION_S]
    r_labels = [labels[i] for i in idx]
    block["reaction"] = {
        "n": len(idx), "pro_act_rate": _div(sum(r_labels), len(idx)),
        "policies": {name: {k: v for k, v in metrics(r_labels, [preds[name][i] for i in idx]).items()
                            if k in ("n", "act_rate", "pro_act_rate", "act_recall", "act_precision")}
                     for name in POLICIES}}
    return block


# --------------------------------------------------------------------------- report

def _f(x, nd=3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def format_markdown(result: dict) -> str:
    out = ["# Act-head evaluation (offline, D-041)", "",
           f"model `{result['model']}`, threshold {_f(result['threshold'], 2)} "
           f"(val balanced accuracy {_f(result['val_balanced_acc'])})", ""]
    for split, block in result["splits"].items():
        out += [f"## {split}  (n={block['n']}, pro act rate {_f(block['pro_act_rate'])})", "",
                "| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |",
                "|---|---|---|---|---|---|---|---|"]
        for name in POLICIES:
            m = block["policies"][name]
            out.append(f"| {name} | {_f(m['accuracy'])} | {_f(m['balanced_accuracy'])} | {_f(m['act_rate'])} | "
                       f"{_f(m['act_precision'])} | {_f(m['act_recall'])} | {_f(m['wait_precision'])} | "
                       f"{_f(m['wait_recall'])} |")
        r = block["reaction"]
        out += ["", f"reaction slice (since_opp_play <= {REACTION_S:g} s): n={r['n']}, "
                    f"pro act rate {_f(r['pro_act_rate'])}", "",
                "| policy | act rate | act recall | act prec |", "|---|---|---|---|"]
        for name in POLICIES:
            m = r["policies"][name]
            out.append(f"| {name} | {_f(m['act_rate'])} | {_f(m['act_recall'])} | {_f(m['act_precision'])} |")
        out.append("")
    return "\n".join(out)


def run(model: str, data: Path, backend: str, splits: tuple[str, ...], limit: int | None, out: Path) -> dict:
    ranker = D1Ranker(model_path=model, backend=backend)
    rules = ReactivePolicy(load_cards())
    loaded: dict[str, tuple[list[dict], dict, list[float]]] = {}
    for split in splits:
        lines, states = load_split(data, split, limit)
        probs = []
        for k, line in enumerate(lines, 1):
            probs.append(float(ranker.probabilities(line["state"], line["questions"])["act"][1]))
            if k % 500 == 0:
                print(f"{split}: scored {k}/{len(lines)}", file=sys.stderr)
        loaded[split] = (lines, states, probs)

    val_lines, _, val_probs = loaded["val"]
    threshold, val_bacc = pick_threshold([l["answers"]["act"] == "true" for l in val_lines], val_probs)
    thr_json = {"threshold": threshold, "val_balanced_acc": val_bacc}
    model_dir = Path(model)
    thr_path = (model_dir if model_dir.is_dir() else out.parent) / "act_threshold.json"
    thr_path.parent.mkdir(parents=True, exist_ok=True)
    thr_path.write_text(json.dumps(thr_json, indent=2) + "\n")

    result = {"model": str(model), "backend": backend, "data": str(data), "limit": limit,
              "threshold": threshold, "val_balanced_acc": val_bacc, "threshold_file": str(thr_path),
              "splits": {s: evaluate_split(*loaded[s][:2], loaded[s][2], threshold, rules) for s in splits}}
    md = format_markdown(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    out.with_suffix(".md").write_text(md + "\n")
    print(md)
    return result


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="act-head checkpoint (d1 fine-tuned with --variant act)")
    ap.add_argument("--data", type=Path, default=DATASET_DIR / "act_dense_bal")
    ap.add_argument("--backend", choices=("mlx", "transformers"), default="transformers")
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="comma-separated: val,test (val is required)")
    ap.add_argument("--limit", type=int, default=None, help="score only N evenly spaced lines per split")
    ap.add_argument("--out", type=Path, default=None, help="JSON path (default <data>/../eval/act_head.json)")
    args = ap.parse_args(argv)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    bad = [s for s in splits if s not in ("train", "val", "test")]
    if bad:
        ap.error(f"unknown splits {bad}; choose from train, val, test")
    if "val" not in splits:
        ap.error("--splits must include val (the threshold is chosen on it)")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be > 0")
    out = args.out or (args.data.parent / "eval" / "act_head.json")
    return run(args.model, args.data, args.backend, splits, args.limit, out)


if __name__ == "__main__":
    main()
