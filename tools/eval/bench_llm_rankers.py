"""Qwen decision models on the MAC-AU lines (Part A): zero-shot 4B / 14B and retrieval-augmented few-shot 4B.

  python tools/eval/bench_llm_rankers.py --render-prompts                 # write runs/bench/au/prompts.jsonl (no model)
  lockf <heavy.lock> python tools/eval/bench_llm_rankers.py --variant qwen3-4b [--limit N] [--resume]

One variant per invocation (and so per heavy-lock acquisition): accuracy and latency come from the same pass. Latency per
line is wall time from the chat template through tokenisation, the forward pass and parsing (and, for the few-shot
variant, the retrieval query); one untimed warm-up generation runs first. Predictions go to
`runs/bench/au/preds/<variant>.jsonl` (the MAC-AU contract); with `--limit` they go to `runs/bench/au/smoke/` instead.

Variants (every line, chat template, greedy, 8 new tokens): qwen3-4b; qwen3-14b (thinking off); qwen3-4b-rag3 (the 4B
plus the 3 most similar train-split pro states, from `clashbot.retrieval.StateIndex.search`, as examples).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.eval.decision_prompt import (LETTERS, build_messages, load_lines, parse_letter, read_jsonl,  # noqa: E402
                                        render_prompt_rows)

BENCH_DIR = ROOT / "runs" / "bench" / "au"
QWEN_4B = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
QWEN_14B = "mlx-community/Qwen3-14B-4bit"
RAG_K = 3
VARIANTS: dict[str, dict] = {
    "qwen3-4b": {"repo": QWEN_4B, "max_tokens": 8, "k": 0},
    "qwen3-14b": {"repo": QWEN_14B, "max_tokens": 8, "k": 0},
    "qwen3-4b-rag3": {"repo": QWEN_4B, "max_tokens": 8, "k": RAG_K},
}


def letter_probs(logprobs, token_ids: list[int]) -> list[float] | None:
    """Softmax over the A-D token logits of the first generated token (`logprobs`: the vocabulary-sized row)."""
    try:
        vals = [float(logprobs[i]) for i in token_ids]
    except Exception:
        return None
    top = max(vals)
    exps = [math.exp(v - top) for v in vals]
    total = sum(exps)
    return [e / total for e in exps]


# -- retrieval for the few-shot variant ---------------------------------------------------------------------------------
class Retriever:
    """k most similar train-split pro states for a line: `StateIndex.search` over the state dict and `context_text` of
    the line's own state (from dataset/v1/states.jsonl by state_id), `splits=["train"]`, its own game excluded."""

    def __init__(self, k: int, states_path: Path | None = None, db_path: Path | None = None):
        from clashbot.config import DATASET_DIR, DB_PATH
        from clashbot.retrieval import StateIndex
        self.k = k
        self.index = StateIndex(Path(db_path or DB_PATH))
        self.states: dict[str, dict] = {}
        for rec in read_jsonl(Path(states_path or DATASET_DIR / "states.jsonl")):
            self.states[rec["id"]] = {"state": rec["state"], "t": rec["t"], "context_text": rec["context_text"],
                                      "game_id": rec["game_id"], "split": rec["split"]}

    def examples(self, line: dict) -> tuple[list[tuple[str, str]], list[str]]:
        """([(context_text, action_key)], [hit ids]) for the line; every hit is a train-split state of another game."""
        rec = self.states[line["state_id"]]
        hits = self.index.search(rec["state"], rec["t"], rec["context_text"], k=self.k, splits=["train"],
                                 exclude_game=line["game_id"])
        out = []
        for h in hits:
            src = self.states[h.id]
            if src["split"] != "train" or h.game_id == line["game_id"]:
                raise RuntimeError(f"retrieval leakage: {h.id} ({src['split']}) for line {line['line_id']}")
            out.append((src["context_text"], h.action_key))
        return out, [h.id for h in hits]


# -- the runner ---------------------------------------------------------------------------------------------------------
class QwenRunner:
    """Loads one MLX model and answers lines; `decide` is the timed unit."""

    def __init__(self, variant: str, retriever: Retriever | None = None):
        from mlx_lm import load
        self.name = variant
        self.cfg = VARIANTS[variant]
        self.model, self.tok = load(self.cfg["repo"])
        self.retriever = retriever
        self.letter_ids = [self.tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]

    def generate(self, messages: list[dict], max_tokens: int) -> dict:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                              enable_thinking=False)
        text, first, last = "", None, None
        for last in stream_generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens,
                                    sampler=make_sampler(temp=0.0)):
            if first is None:
                first = last.logprobs
            text += last.text
        return {"text": text, "first_logprobs": first, "in_tokens": getattr(last, "prompt_tokens", None),
                "out_tokens": getattr(last, "generation_tokens", None)}

    def decide(self, line: dict) -> dict:
        t0 = time.perf_counter()
        examples, hit_ids, retrieval_ms = None, [], 0.0
        if self.retriever is not None:
            examples, hit_ids = self.retriever.examples(line)
            retrieval_ms = (time.perf_counter() - t0) * 1000.0
        out = self.generate(build_messages(line, examples), self.cfg["max_tokens"])
        pred = parse_letter(out["text"])
        probs = None if out["first_logprobs"] is None else letter_probs(out["first_logprobs"], self.letter_ids)
        ms = (time.perf_counter() - t0) * 1000.0
        row = {"pred": pred, "probs": probs, "latency_ms": round(ms, 1), "in_tokens": out["in_tokens"],
               "out_tokens": out["out_tokens"], "cost_usd": 0.0, "raw": out["text"][:200],
               "error": None if pred is not None else "unparseable"}
        if self.retriever is not None:
            row.update({"k": self.retriever.k, "prompt_tokens": out["in_tokens"], "retrieved": hit_ids,
                        "retrieval_ms": round(retrieval_ms, 1)})
        return row


def run_variant(runner, lines: list[dict], out_path: Path, resume: bool = False, warmup: bool = True) -> dict:
    """Answer every line (one warm-up generation first, untimed and unrecorded); rows go to `<out>.partial` as they
    finish and the file is renamed on completion."""
    partial = out_path.with_name(out_path.name + ".partial")
    done = {r["line_id"]: r for r in read_jsonl(partial)} if resume and partial.exists() else {}
    if warmup and lines:
        runner.generate(build_messages(lines[0]), 2)
    rows = []
    t0 = time.perf_counter()
    with open(partial, "a" if resume else "w") as f:
        for i, ln in enumerate(lines):
            if ln["line_id"] in done:
                rows.append(done[ln["line_id"]])
                continue
            try:
                res = runner.decide(ln)
            except Exception as exc:                      # one bad line must not lose the run
                res = {"pred": None, "probs": None, "latency_ms": None, "in_tokens": None, "out_tokens": None,
                       "cost_usd": 0.0, "raw": "", "error": f"{type(exc).__name__}: {exc}"[:200]}
            row = {"line_id": ln["line_id"], "game_id": ln["game_id"], "gold": ln["gold"], **res}
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"[{runner.name}] {i + 1}/{len(lines)} {time.perf_counter() - t0:.0f}s", flush=True)
    partial.replace(out_path)
    ok = sum(r["pred"] == r["gold"] for r in rows)
    return {"variant": runner.name, "n": len(rows), "correct": ok, "accuracy": round(ok / max(len(rows), 1), 4),
            "parse_failures": sum(r["pred"] is None for r in rows), "out": str(out_path),
            "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--render-prompts", action="store_true", help="write runs/bench/au/prompts.jsonl (for the Claude runner) and exit")
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--lines", type=Path, default=BENCH_DIR / "lines.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None, help="default runs/bench/au/preds (runs/bench/au/smoke with --limit)")
    ap.add_argument("--limit", type=int, default=None, help="first N lines only (smoke test)")
    ap.add_argument("--resume", action="store_true", help="continue from <variant>.jsonl.partial")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    lines = load_lines(args.lines)
    if args.render_prompts:
        rows = render_prompt_rows(lines)
        path = BENCH_DIR / "prompts.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"wrote {len(rows)} prompts to {path}")
        return 0
    if not args.variant:
        ap.error("--variant or --render-prompts is required")
    cfg = VARIANTS[args.variant]
    if args.limit:
        lines = lines[:args.limit]
    out_dir = args.out_dir or (BENCH_DIR / "smoke" if args.limit else BENCH_DIR / "preds")
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever = Retriever(cfg["k"]) if cfg["k"] else None
    runner = QwenRunner(args.variant, retriever)
    summary = run_variant(runner, lines, out_dir / f"{args.variant}.jsonl", resume=args.resume,
                          warmup=not args.no_warmup)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
