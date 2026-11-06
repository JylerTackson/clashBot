"""Build the d1 joint training export with tactics (MAC-Z3): the joint questions of WP6 plus the engine's lines and notes.

  python3 tools/data/build_d1_tac.py [--dataset DIR] [--out DIR] [--plans FILE]

Reads <dataset>/d1/joint/{train,val,test}.jsonl and <dataset>/states.jsonl. For every line the tactics engine analyses
the state; the new line's state is the context followed by the engine's `tactics: ...` lines, and every option carries
the note the live loop would attach (`CandidateGenerator.option_notes`; kNN over the train split without the line's
own game); an option that plays the card of an engine move at another placement carries the card-level tactics note
(stats: `answer_has_tactics_card_note` counts exact or card-level). Answers and metadata are unchanged. Writes <out>/{train,val,test}.jsonl (default <dataset>/d1/joint_tac)
and <out>/stats.json. CPU only; no model is loaded.

`--plans FILE` (coach_plans.jsonl of build_coach_plans.py, D-046) adds to each line's state the newest coach plan of its
game made at or before the line's time, as `[plan] ...` after the tactics lines (`d1_format.state_string`), the way
the live loop shows it. Lines with no such plan stay as they are. Without `--plans` the output is unchanged.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.candidates import TAC_CARD_NOTE, CandidateGenerator  # noqa: E402
from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.d1_format import JOINT_NAME, joint_request, validate  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.kg import KnowledgeGraph  # noqa: E402
from clashbot.retrieval import StateIndex  # noqa: E402
from clashbot.state_text import render_context  # noqa: E402
from clashbot.tactics import TacticalEngine  # noqa: E402

SPLITS = ("train", "val", "test")
TAC_PREFIX = "tactics:"
LOOKUP_PREFIXES = ("pros played this in similar states", "known counter of")    # lookup notes with real support
PROGRESS_EVERY = 500


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _share(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def load_plans(path: Path) -> dict[str, tuple[list[float], list[str]]]:
    """game_id -> (sorted plan times, plans) from a coach_plans.jsonl; a plan that cannot go into a state is dropped."""
    rows: dict[str, list[tuple[float, str]]] = {}
    for row in read_jsonl(path):
        if row.get("plan") and "<|" not in row["plan"]:
            rows.setdefault(row["game_id"], []).append((row["t"], row["plan"]))
    return {gid: ([t for t, _ in sorted(r)], [p for _, p in sorted(r)]) for gid, r in rows.items()}


def latest_plan(plans: dict | None, game_id: str, t: float) -> str | None:
    """The plan of `game_id` with the largest t_plan <= t, else None."""
    times, texts = (plans or {}).get(game_id, ([], []))
    i = bisect.bisect_right(times, t)
    return texts[i - 1] if i else None


def build_line(line: dict, rec: dict, engine: TacticalEngine, generator: CandidateGenerator,
               plans: dict | None = None) -> tuple[dict, dict]:
    """(new line, facts about it) for one joint line and its state record; `plans` as `load_plans`, or None."""
    report = engine.analyse(rec["state"], rec["t"])
    context = render_context(rec["state"], rec["t"])
    mismatch = context != line["state"]
    if mismatch:
        context = line["state"]                    # the line's own text is what the model was trained on
    keys = list(line["questions"][JOINT_NAME]["criteria"])
    notes = generator.option_notes(rec["state"], rec["t"], context, keys, report, splits=("train",),
                                   exclude_game=rec["game_id"])
    plan = latest_plan(plans, rec["game_id"], rec["t"])
    state, questions = joint_request(context + "\n" + "\n".join(report.lines), keys, plan=plan, notes=notes)
    new = {**line, "state": state, "questions": questions}
    validate(new["state"], new["questions"], new["answers"])
    answer = line["answers"][JOINT_NAME]
    answer_note = notes.get(answer) or ""
    facts = {"mismatch": mismatch, "has_push": bool(report.pushes),
             "answer_tactics_note": answer_note.startswith(TAC_PREFIX),
             "answer_tactics_card_note": answer_note.startswith(TAC_PREFIX) or TAC_CARD_NOTE in answer_note,
             "top_is_answer": report.best is not None and report.best.key == answer,
             "any_lookup_note": any(n.startswith(LOOKUP_PREFIXES) for n in notes.values()),
             "has_plan": plan is not None, "chars": len(state)}
    return new, facts


def split_stats(facts: list[dict], missing: int, plans: bool = False) -> dict:
    n = len(facts)
    return {"lines": n,
            **({"lines_with_plan": _share(sum(f["has_plan"] for f in facts), n)} if plans else {}),
            "answer_has_tactics_note": _share(sum(f["answer_tactics_note"] for f in facts), n),
            "answer_has_tactics_card_note": _share(sum(f["answer_tactics_card_note"] for f in facts), n),
            "tactics_top_is_answer": _share(sum(f["top_is_answer"] for f in facts), n),
            "any_lookup_note": _share(sum(f["any_lookup_note"] for f in facts), n),
            "lines_with_push": _share(sum(f["has_push"] for f in facts), n),
            "mean_state_chars": round(sum(f["chars"] for f in facts) / n, 1) if n else None,
            "context_mismatches": sum(f["mismatch"] for f in facts),
            "skipped_no_state": missing}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--out", type=Path, default=None, help="output dir (default <dataset>/d1/joint_tac)")
    ap.add_argument("--plans", type=Path, default=None,
                    help="coach_plans.jsonl: add the newest coach plan at or before each line as '[plan] ...'")
    args = ap.parse_args(argv)
    dataset = Path(args.dataset)
    out_dir = Path(args.out) if args.out else dataset / "d1" / "joint_tac"

    # built exactly as clashbot/live.py::main builds them
    cards = load_cards()
    kg = KnowledgeGraph()
    generator = CandidateGenerator(StateIndex(), kg, cards)
    engine = TacticalEngine(cards, kg=kg)
    states = {r["id"]: r for r in read_jsonl(dataset / "states.jsonl")}
    plans = load_plans(Path(args.plans)) if args.plans else None

    stats: dict = {}
    for split in SPLITS:
        src = dataset / "d1" / "joint" / f"{split}.jsonl"
        if not src.exists():
            continue
        built, facts, missing = [], [], 0
        lines = read_jsonl(src)
        for i, line in enumerate(lines, 1):
            rec = states.get(line["id"])
            if rec is None:
                missing += 1
                continue
            new, fact = build_line(line, rec, engine, generator, plans)
            built.append(new)
            facts.append(fact)
            if i % PROGRESS_EVERY == 0:
                print(f"{split}: {i}/{len(lines)}", flush=True)
        write_jsonl(out_dir / f"{split}.jsonl", built)
        stats[split] = split_stats(facts, missing, plans is not None)
        print(f"{split}: {len(built)} lines -> {out_dir / (split + '.jsonl')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
