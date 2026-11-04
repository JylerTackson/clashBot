"""Zero-shot evaluation of d1-omni-600M on the exported d1 dataset (D-022, D-023, D-026; no fine-tuning).

  python3 tools/eval/eval_d1_zero_shot.py [--dataset DIR] [--backend mlx|transformers]
          [--variant joint|factorized|both] [--splits val,test] [--limit N] [--out PATH] [--model PATH]
  python3 tools/eval/eval_d1_zero_shot.py --compare OUT_A OUT_B     # no model: agreement of two result files

Reads <dataset>/d1/<variant>/<split>.jsonl and sends each line's `state` / `questions` as exported.
Writes <dataset>/eval/d1_zero_shot_<backend>.json (or --out) and prints markdown tables.
--model PATH evaluates a fine-tuned checkpoint (tools/train/train_d1.py) instead; the default --out is then
<dataset>/eval/d1_eval_<backend>_<basename(PATH)>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.ranker import D1Ranker, argmax_first  # noqa: E402

VARIANTS = ("joint", "factorized")
FACTORIZED_QUESTIONS = ("card", "lane", "zone")
BASELINE_COLUMNS = ("placement_prior", "popularity", "random_expected")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def shortcut_argmax(keys: list[str]) -> list[int]:
    """Option-set shortcut: score(k) = #options with k's card + #options with k's (lane, zone), k included.
    Returns the indices of the maximum scores (WP4 negatives N2/N3 make the positive stand out)."""
    parts = [k.split("|") for k in keys]
    scores = [sum(p[0] == q[0] for q in parts) + sum(p[1:] == q[1:] for q in parts) for p in parts]
    top = max(scores)
    return [i for i, s in enumerate(scores) if s == top]


def shortcut_accuracy(keys: list[str], gold: int) -> float:
    """Expected accuracy of the shortcut with a random tie-break."""
    top = shortcut_argmax(keys)
    return 1.0 / len(top) if gold in top else 0.0


def joint_metrics(rows: list[dict], times: list[float]) -> dict:
    n = len(rows)
    width = max([4] + [len(r["probs"]) for r in rows])
    pred_pos, gold_pos = [0] * width, [0] * width
    by_kind: dict[str, list[float]] = {}
    hits, gold_probs, shortcut, agree = [], [], [], []
    for r in rows:
        hit = float(r["pred_index"] == r["gold_index"])
        hits.append(hit)
        by_kind.setdefault(r["kind"], []).append(hit)
        gold_probs.append(r["probs"][r["gold_index"]])
        pred_pos[r["pred_index"]] += 1
        gold_pos[r["gold_index"]] += 1
        shortcut.append(shortcut_accuracy(r["keys"], r["gold_index"]))
        agree.append(float(r["pred_index"] in shortcut_argmax(r["keys"])))
    return {
        "n": n, "acc": mean(hits),
        "acc_by_kind": {k: {"acc": mean(v), "n": len(v)} for k, v in sorted(by_kind.items())},
        "mean_gold_prob": mean(gold_probs),
        "pred_position_counts": pred_pos, "gold_position_counts": gold_pos,
        "option_set_shortcut": mean(shortcut), "d1_agrees_with_shortcut": mean(agree),
        "ms_p50": pct(times, 50), "ms_p95": pct(times, 95),
    }


def factorized_metrics(rows: list[dict], times: list[float]) -> dict:
    out: dict = {"n": len(rows)}
    for name in FACTORIZED_QUESTIONS:
        out[name] = mean([float(r["pred"][name] == r["gold"][name]) for r in rows])
    all_three = [float(all(r["pred"][q] == r["gold"][q] for q in FACTORIZED_QUESTIONS)) for r in rows]
    key_hits = [float("|".join(r["pred"][q] for q in FACTORIZED_QUESTIONS)
                       == "|".join(r["gold"][q] for q in FACTORIZED_QUESTIONS)) for r in rows]
    out.update({"all_three": mean(all_three), "key_acc": mean(key_hits),
                "ms_p50": pct(times, 50), "ms_p95": pct(times, 95)})
    return out


def evaluate_split(ranker, lines: list[dict], variant: str) -> tuple[dict, list[dict]]:
    """Run d1 on every line; returns (metrics, per_example records)."""
    rows, times, per_example = [], [], []
    for line in lines:
        probs = ranker.probabilities(line["state"], line["questions"])
        times.append(ranker.last_ms)
        if variant == "joint":
            keys = list(line["questions"]["play"]["criteria"])
            p = probs["play"]
            pred_i, gold = argmax_first(p), line["answers"]["play"]
            rows.append({"kind": line.get("kind", "?"), "probs": p, "keys": keys,
                         "pred_index": pred_i, "gold_index": keys.index(gold)})
            pred = keys[pred_i]
        else:
            pred = {}
            for name in FACTORIZED_QUESTIONS:
                labels = list(line["questions"][name]["criteria"])
                pred[name] = labels[argmax_first(probs[name])]
            gold = {name: line["answers"][name] for name in FACTORIZED_QUESTIONS}
            rows.append({"pred": pred, "gold": gold})
            p = probs
        per_example.append({"id": line["id"], "variant": variant, "split": line["split"],
                            "pred": pred, "gold": gold, "probs": p})
    metrics = joint_metrics(rows, times) if variant == "joint" else factorized_metrics(rows, times)
    return metrics, per_example


def load_baselines(dataset: Path, splits: list[str]) -> dict:
    path = dataset / "ranker" / "baselines.json"
    if not path.exists():
        return {}
    ranker = json.loads(path.read_text()).get("ranker", {})
    return {s: {name: m.get("acc") for name, m in ranker.get(s, {}).items()} for s in splits}


def fmt(x) -> str:
    return "-" if x is None else f"{x:.4f}" if isinstance(x, float) and abs(x) <= 1 else f"{x}"


def print_tables(result: dict) -> None:
    variants, baselines = result["variants"], result["baselines"]
    if "joint" in variants:
        print("\n### joint (multiple choice over 4 plays)\n")
        print("| split | n | d1 acc | option_set_shortcut | placement_prior | popularity | random | ms p50 | ms p95 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for split, m in variants["joint"].items():
            b = baselines.get(split, {})
            print(f"| {split} | {m['n']} | {fmt(m['acc'])} | {fmt(m['option_set_shortcut'])} | "
                  f"{fmt(b.get('placement_prior'))} | {fmt(b.get('popularity'))} | {fmt(b.get('random_expected'))} | "
                  f"{fmt(m['ms_p50'])} | {fmt(m['ms_p95'])} |")
        for split, m in variants["joint"].items():
            print(f"\n{split}: d1_agrees_with_shortcut={fmt(m['d1_agrees_with_shortcut'])} "
                  f"mean_gold_prob={fmt(m['mean_gold_prob'])} pred_positions={m['pred_position_counts']} "
                  f"gold_positions={m['gold_position_counts']}")
    if "factorized" in variants:
        print("\n### factorized (card, lane, zone)\n")
        print("| split | n | card | lane | zone | all three | key | ms p50 |")
        print("|---|---|---|---|---|---|---|---|")
        for split, m in variants["factorized"].items():
            print(f"| {split} | {m['n']} | {fmt(m['card'])} | {fmt(m['lane'])} | {fmt(m['zone'])} | "
                  f"{fmt(m['all_three'])} | {fmt(m['key_acc'])} | {fmt(m['ms_p50'])} |")


def flatten(probs) -> list[float]:
    if isinstance(probs, dict):
        return [float(x) for k in sorted(probs) for x in probs[k]]
    return [float(x) for x in probs]


def compare(path_a: Path, path_b: Path) -> dict:
    """Agreement of two result files (e.g. mlx vs transformers), matched by (id, variant)."""
    def index(path: Path) -> dict:
        return {(e["id"], e["variant"]): e for e in json.loads(Path(path).read_text())["per_example"]}
    a, b = index(path_a), index(path_b)
    common = sorted(set(a) & set(b))
    agree, diffs = [], []
    for key in common:
        agree.append(float(a[key]["pred"] == b[key]["pred"]))
        diffs.extend(abs(x - y) for x, y in zip(flatten(a[key]["probs"]), flatten(b[key]["probs"])))
    result = {"n_common": len(common),
              "top_agreement": round(sum(agree) / len(agree), 4) if agree else None,
              "max_abs_prob_diff": max(diffs) if diffs else None,
              "mean_abs_prob_diff": sum(diffs) / len(diffs) if diffs else None}
    print(json.dumps(result, indent=1))
    return result


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--backend", choices=("mlx", "transformers"), default="mlx")
    ap.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--limit", type=int, default=None, help="first N lines per split and variant")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default=None, help="checkpoint dir (fine-tuned d1) instead of the zero-shot weights")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)

    splits = [s for s in args.splits.split(",") if s]
    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    data = {(v, s): read_jsonl(args.dataset / "d1" / v / f"{s}.jsonl")[:args.limit]
            for v in variants for s in splits}

    ranker = D1Ranker(model_path=args.model, backend=args.backend)
    first = next((lines[0] for lines in data.values() if lines), None)
    if first is not None:                                  # warm-up: model load + first call, not timed
        ranker.probabilities(first["state"], first["questions"])

    result: dict = {"backend": args.backend, "model_id": ranker.model_id, "variants": {},
                    "baselines": load_baselines(args.dataset, splits), "limit": args.limit, "per_example": []}
    if args.model:
        result["model_path"] = args.model
    for v in variants:
        result["variants"][v] = {}
        for s in splits:
            metrics, per_example = evaluate_split(ranker, data[(v, s)], v)
            result["variants"][v][s] = metrics
            result["per_example"].extend(per_example)

    name = f"d1_eval_{args.backend}_{Path(args.model).name}" if args.model else f"d1_zero_shot_{args.backend}"
    out = args.out or args.dataset / "eval" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print_tables(result)
    print(f"\nwrote {out}")
    return result


if __name__ == "__main__":
    main()
