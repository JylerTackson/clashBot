"""Act/hold policy comparison (MAC-W, D-041): the reactive policy vs the old fixed-interval trigger, offline.

  python3 tools/eval/eval_act_policy.py [--dataset DIR] [--splits val,test] [--out PATH]

Every record of the split (all kinds) from <dataset>/states.jsonl is labelled 1 if the pro played a card there
(`action.type == "play"`) and 0 on a hold; `unknown` actions are skipped. Predictions: `reactive` =
ReactivePolicy.decide(...).act, `trigger` = should_decide(state, t, None, cards), `always` = 1. Per split and per
policy: accuracy, balanced accuracy, act rate vs the pro play rate, confusion matrix; the reactive policy also
counts its modes. The same is repeated on `kind == "periodic"` records only (closest to random moments).
The labels are biased toward plays (D-020), so this is a sanity check; the dense data of MAC-V is the real test.
Writes <out> (JSON, default <dataset>/eval/act_policy.json) and <out>.md (same name, .md) and prints the table.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.loop import should_decide  # noqa: E402
from clashbot.policy import ReactivePolicy  # noqa: E402

POLICIES = ("reactive", "trigger", "always")
SUBSETS = ("all", "periodic")
DEFAULT_SPLITS = ("val", "test")
MODES = ("defend", "leak", "punish", "save", "hold", "none")


def load_records(dataset: Path, splits: set[str]) -> list[dict]:
    """states.jsonl reduced to the fields the policies read (the file is large)."""
    out = []
    with open(Path(dataset) / "states.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") not in splits:
                continue
            action = r.get("action") or {}
            out.append({"split": r["split"], "kind": r.get("kind"), "t": r["t"], "state": r["state"],
                        "type": action.get("type")})
    return out


def score(pairs: list[tuple[int, int]]) -> dict:
    """pairs of (label, prediction) -> accuracy / balanced accuracy / act rate / confusion matrix."""
    tp = sum(1 for y, p in pairs if y and p)
    fn = sum(1 for y, p in pairs if y and not p)
    fp = sum(1 for y, p in pairs if not y and p)
    tn = sum(1 for y, p in pairs if not y and not p)
    n = len(pairs)
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    rates = [r for r in (tpr, tnr) if r is not None]
    return {"n": n,
            "accuracy": (tp + tn) / n if n else None,
            "balanced_accuracy": sum(rates) / len(rates) if rates else None,
            "act_rate": (tp + fp) / n if n else None,
            "pro_play_rate": (tp + fn) / n if n else None,
            "tpr": tpr, "tnr": tnr,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def evaluate(records: list[dict], cards: dict, policy: ReactivePolicy | None = None) -> dict:
    """{split: {subset: {"n", "pro_play_rate", "policies": {name: score}, "reactive_modes": {mode: n}}}}."""
    policy = policy or ReactivePolicy(cards)
    result: dict = {}
    for split in sorted({r["split"] for r in records}):
        rows = [r for r in records if r["split"] == split and r["type"] in ("play", "hold")]
        result[split] = {}
        for subset in SUBSETS:
            sub = [r for r in rows if subset == "all" or r["kind"] == "periodic"]
            pairs: dict[str, list[tuple[int, int]]] = {p: [] for p in POLICIES}
            modes: Counter = Counter()
            for r in sub:
                y = 1 if r["type"] == "play" else 0
                decision = policy.decide(r["state"], r["t"])
                modes[decision.mode] += 1
                pairs["reactive"].append((y, int(decision.act)))
                pairs["trigger"].append((y, int(should_decide(r["state"], r["t"], None, cards))))
                pairs["always"].append((y, 1))
            result[split][subset] = {
                "n": len(sub),
                "pro_play_rate": sum(y for y, _ in pairs["always"]) / len(sub) if sub else None,
                "policies": {p: score(pairs[p]) for p in POLICIES},
                "reactive_modes": {m: modes[m] for m in MODES if modes[m]},
            }
    return result


def _f(x, nd=3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def format_markdown(result: dict) -> str:
    lines = ["# Act/hold policy: reactive vs trigger (offline, D-041)", "",
             "Labels: 1 = the pro played a card, 0 = hold (unknown skipped). The labels are biased toward plays (D-020).", ""]
    for split, subsets in result.items():
        for subset, block in subsets.items():
            lines += [f"## {split} / {subset}  (n={block['n']}, pro play rate {_f(block['pro_play_rate'])})", "",
                      "| policy | accuracy | balanced acc | act rate | tpr | tnr | tp | fp | fn | tn |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for name in POLICIES:
                s = block["policies"][name]
                c = s["confusion"]
                lines.append(f"| {name} | {_f(s['accuracy'])} | {_f(s['balanced_accuracy'])} | {_f(s['act_rate'])} | "
                             f"{_f(s['tpr'])} | {_f(s['tnr'])} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} |")
            modes = ", ".join(f"{m} {n}" for m, n in block["reactive_modes"].items()) or "none"
            lines += ["", f"reactive modes: {modes}", ""]
    return "\n".join(lines)


def run(dataset: Path, splits: tuple[str, ...], out: Path) -> dict:
    cards = load_cards()
    records = load_records(dataset, set(splits))
    result = {"dataset": str(dataset), "splits": evaluate(records, cards)}
    md = format_markdown(result["splits"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    out.with_suffix(".md").write_text(md + "\n")
    print(md)
    return result


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="comma-separated: train,val,test")
    ap.add_argument("--out", type=Path, default=None, help="JSON path (default <dataset>/eval/act_policy.json)")
    args = ap.parse_args(argv)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    bad = [s for s in splits if s not in ("train", "val", "test")]
    if bad:
        ap.error(f"unknown splits {bad}; choose from train, val, test")
    return run(args.dataset, splits, args.out or (args.dataset / "eval" / "act_policy.json"))


if __name__ == "__main__":
    main()
