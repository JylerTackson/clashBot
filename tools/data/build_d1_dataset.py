"""Build the d1-omni-600M training export (WP6): joint, factorized and act variants in d1's Decision Index format.

  python3 tools/data/build_d1_dataset.py [--dataset DIR] [--tokenizer PATH]

Writes <dataset>/d1/{joint,factorized,act}/{train,val,test}.jsonl, <dataset>/d1/stats.json and <dataset>/d1/README.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.d1_format import act_request, factorized_request, joint_request, validate  # noqa: E402
from clashbot.game_rules import load_cards, possible_cards  # noqa: E402

SPLITS = ("train", "val", "test")
VARIANTS = ("joint", "factorized", "act")
MAX_LEN = 16384


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def line_record(example: dict, split: str, variant: str, state: str, questions: dict, answers: dict) -> dict:
    return {"id": example["id"], "game_id": example["game_id"], "split": split, "kind": example["kind"],
            "variant": variant, "state": state, "questions": questions, "answers": answers}


def build_joint(examples: list[dict], split: str) -> tuple[list[dict], Counter]:
    out = []
    for ex in examples:
        keys = [o["key"] for o in ex["options"]]
        state, questions = joint_request(ex["context"], keys)
        out.append(line_record(ex, split, "joint", state, questions, {"play": ex["positive_key"]}))
    return out, Counter()


def build_factorized(examples: list[dict], split: str, states: dict, cards: dict) -> tuple[list[dict], Counter]:
    out, skips = [], Counter()
    for ex in examples:
        rec = states[ex["id"]]
        possible = possible_cards(rec["state"], cards)
        if len(possible) < 2:
            skips["skipped_few_cards"] += 1
            continue
        action = rec["action"]
        if action["card"] not in possible:
            skips["skipped_card_not_possible"] += 1
            continue
        state, questions = factorized_request(ex["context"], possible, cards)
        answers = {"card": action["card"], "lane": action["lane"], "zone": action["zone"]}
        out.append(line_record(ex, split, "factorized", state, questions, answers))
    return out, skips


def build_act(examples: list[dict], split: str) -> tuple[list[dict], Counter]:
    out = []
    for ex in examples:
        state, questions = act_request(ex["context"])
        line = line_record(ex, split, "act", state, questions, {"act": "true" if ex["label"] == 1 else "false"})
        line["provisional"] = True
        out.append(line)
    return out, Counter()


def minmedmax(values: list[int]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def answer_distribution(variant: str, lines: list[dict]) -> dict:
    if variant == "joint":
        c = Counter(list(l["questions"]["play"]["criteria"]).index(l["answers"]["play"]) for l in lines)
        return {"play": {str(k): c[k] for k in sorted(c)}}
    if variant == "factorized":
        return {"card": dict(Counter(l["answers"]["card"] for l in lines).most_common(10)),
                "lane": dict(Counter(l["answers"]["lane"] for l in lines)),
                "zone": dict(Counter(l["answers"]["zone"] for l in lines))}
    c = Counter(l["answers"]["act"] for l in lines)
    return {"act": {"true": c["true"], "false": c["false"]}}


def token_stats(lines: list[dict], tok) -> dict:
    state_tokens, option_max, over = [], 0, 0
    for l in lines:
        state_tokens.append(len(tok.encode(l["state"], add_special_tokens=False).ids))
        for q in l["questions"].values():
            criteria = q.get("criteria") or {}
            k = len(criteria)
            if not k:
                continue
            budget = (max(96, min(24 * k + 32, MAX_LEN // 2)) - 3 * k) // k
            for name, desc in criteria.items():
                rendered = f"{name}: {desc}" if desc else name
                n = len(tok.encode(" " + rendered, add_special_tokens=False).ids)
                option_max = max(option_max, n)
                if n > budget:
                    over += 1
    return {"state_tokens": minmedmax(state_tokens), "option_tokens": {"max": option_max},
            "options_over_budget": over}


def write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")


def readme(stats: dict) -> str:
    rows = []
    for v in VARIANTS:
        counts = ", ".join(f"{s}={stats[v][s]['examples']}" for s in SPLITS)
        rows.append(f"  {v}: {counts}")
    return f"""d1-omni-600M training data (WP6; D-021, D-022, D-023)

Layout: d1/{{joint,factorized,act}}/{{train,val,test}}.jsonl, d1/stats.json.

Every line is one JSON object:
  {{"id", "game_id", "split", "kind", "variant", "state", "questions", "answers"}}
  state      string shown to the model (our context text, never containing "<|")
  questions  {{name: {{"type", "instructions", "criteria"}}}} in d1's Decision Index format
  answers    {{name: gold answer}}: the gold option is line["answers"][name]

Variants and example counts:
  joint       one choice question "play" over the 4 shuffled plays of a ranker example (D-018: no hold option)
  factorized  three choice questions: card (cards the player could hold), lane, zone
  act         one noul question "act" (deploy now vs wait); answers {{"act": "true"|"false"}}
{chr(10).join(rows)}

Mapping to the model: for one line, call
  model.system_one(line["state"], line["questions"])
and compare the result for each question name with line["answers"][name]
(choice: the chosen criteria key; noul: P(yes) against "true"/"false").

The act variant is provisional (D-020): every act line carries "provisional": true.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    dataset = Path(args.dataset)
    out_dir = dataset / "d1"
    cards = load_cards()
    states = {r["id"]: r for r in read_jsonl(dataset / "states.jsonl")}

    tok = None
    if args.tokenizer:
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(args.tokenizer)
        except ImportError:
            print("tokenizers is not installed; skipping the token check")

    stats: dict = {v: {} for v in VARIANTS}
    for split in SPLITS:
        ranker = read_jsonl(dataset / "ranker" / f"{split}.jsonl")
        act = read_jsonl(dataset / "act" / f"{split}.jsonl")
        built = {"joint": build_joint(ranker, split),
                 "factorized": build_factorized(ranker, split, states, cards),
                 "act": build_act(act, split)}
        for variant, (lines, skips) in built.items():
            for l in lines:
                try:
                    validate(l["state"], l["questions"], l["answers"])
                except ValueError as e:
                    raise SystemExit(f"validation failed for {variant}/{split} id={l['id']}: {e}")
            write_jsonl(out_dir / variant / f"{split}.jsonl", lines)
            entry = {"examples": len(lines), **{k: skips[k] for k in sorted(skips)},
                     "answers": answer_distribution(variant, lines),
                     "state_chars": minmedmax([len(l["state"]) for l in lines])}
            if variant == "factorized":
                entry.setdefault("skipped_few_cards", 0)
                entry.setdefault("skipped_card_not_possible", 0)
            if tok is not None:
                entry.update(token_stats(lines, tok))
            stats[variant][split] = entry
            print(f"{variant}/{split}: {len(lines)} lines" + (f", skips {dict(skips)}" if skips else ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "README.md").write_text(readme(stats))


if __name__ == "__main__":
    main()
