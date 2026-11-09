"""Claude Haiku 5.5 / Sonnet 5.5 as decision models on the MAC-AU lines (Part A).

  ANTHROPIC_API_KEY="$(cat ~/.clashbot/anthropic_api_key)" <claudeenv python> tools/eval/bench_claude_rankers.py \
      --variant claude-haiku-5-5 [--limit N] [--resume] [--budget-usd 15]

Run it with the isolated venv that has the `anthropic` SDK (not the project venv). It imports no torch, mlx or clashbot
module: it reads `runs/bench/au/lines.jsonl` and the prompts `bench_llm_rankers.py --render-prompts` wrote to
`runs/bench/au/prompts.jsonl`, calls the API one line at a time (so latency is clean), and writes
`runs/bench/au/preds/<variant>.jsonl` (the MAC-AU predictions contract). The key is read from the environment by the SDK
and is never logged or written.

Variants: Haiku 5.5 (thinking disabled) and Sonnet 5.5 (thinking "between_tools", since it rejects "disabled" with a
400), both on every line with max_tokens 16. No temperature / top_p / top_k, no prefill, no `fallbacks`. The extra
`claude-sonnet-5-5-long` (max_tokens 2000, answer read from the end of the reply) is a supplement to the spec.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.eval.decision_prompt import (cost_usd, load_lines, parse_final_letter, parse_letter,  # noqa: E402
                                        read_jsonl)

BENCH_DIR = ROOT / "runs" / "bench" / "au"
MAX_RETRIES = 4                 # the SDK retries 429 / 5xx / connection errors itself
TIMEOUT_S = 180.0
STOP_AFTER_CLIENT_ERRORS = 3    # consecutive non-retryable 4xx replies: the request itself is wrong, stop spending

VARIANTS: dict[str, dict] = {
    "claude-haiku-5-5": {"model": "claude-haiku-5-5", "max_tokens": 16, "thinking": {"type": "disabled"}},
    "claude-sonnet-5-5": {"model": "claude-sonnet-5-5", "max_tokens": 16,
                          "thinking": {"type": "between_tools"}},        # Sonnet 5.5 rejects "disabled" with a 400
    # Supplement, not in the spec: Sonnet 5.5 mostly reasons in prose before it answers, so max_tokens 16 cuts most replies
    # off (a parse failure). Same request with room to finish; the answer is read from the end of the reply.
    "claude-sonnet-5-5-long": {"model": "claude-sonnet-5-5", "max_tokens": 2000, "thinking": {"type": "between_tools"},
                               "parse": "final"},
}

_KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def scrub(text: str) -> str:
    """Drop anything that looks like an API key from an error string before it is stored."""
    return _KEY_RE.sub("[redacted]", text)


def build_request(variant: str, system: str, user: str) -> dict:
    """`messages.create` kwargs for one line. Deliberately no temperature / top_p / top_k, prefill or fallbacks."""
    v = VARIANTS[variant]
    return {"model": v["model"], "max_tokens": v["max_tokens"], "system": system,
            "messages": [{"role": "user", "content": user}], "thinking": dict(v["thinking"])}


def reply_text(response) -> str:
    """The text blocks of a response, joined (thinking blocks are skipped)."""
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")


def outcome(variant: str, response) -> dict:
    """pred / raw / error / tokens / cost of one API response."""
    text = reply_text(response)
    usage = response.usage
    in_tok, out_tok = usage.input_tokens, usage.output_tokens
    stop = getattr(response, "stop_reason", None)
    pred = (parse_final_letter if VARIANTS[variant].get("parse") == "final" else parse_letter)(text)
    error = None
    if stop == "refusal":
        pred, error = None, "refusal"
    elif pred is None:
        error = "max_tokens" if stop == "max_tokens" else "unparseable"
    return {"pred": pred, "raw": text[:200], "raw_tail": text[-100:] if len(text) > 200 else "", "error": error,
            "in_tokens": in_tok, "out_tokens": out_tok,
            "cost_usd": cost_usd(VARIANTS[variant]["model"], in_tok, out_tok), "stop_reason": stop}


def client_error(exc: Exception) -> tuple[str, bool]:
    """(error text, non-retryable 4xx?) for an SDK exception: typed errors, most specific first. Anything that is not an
    SDK error is raised again (a bug in this script must not turn into a quiet null prediction)."""
    try:
        import anthropic
    except ImportError:
        raise exc
    if isinstance(exc, anthropic.RateLimitError):
        return "RateLimitError: 429 after retries", False
    if isinstance(exc, anthropic.APIStatusError):
        client = 400 <= exc.status_code < 500 and exc.status_code not in (408, 409)
        return scrub(f"APIStatusError {exc.status_code}: {exc.message}")[:200], client
    if isinstance(exc, anthropic.APIConnectionError):
        return scrub(f"{type(exc).__name__}: {exc}")[:200], False
    raise exc


def call_line(client, variant: str, row: dict) -> tuple[dict, bool]:
    """One timed API call -> (the predictions-file fields, whether it was a non-retryable 4xx)."""
    req = build_request(variant, row["system"], row["user"])
    t0 = time.perf_counter()
    try:
        response = client.messages.create(**req)
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000.0
        text, client_side = client_error(exc)
        return {"pred": None, "raw": "", "error": text, "in_tokens": None, "out_tokens": None, "cost_usd": 0.0,
                "latency_ms": round(ms, 1)}, client_side
    ms = (time.perf_counter() - t0) * 1000.0
    return {**outcome(variant, response), "latency_ms": round(ms, 1)}, False


def spent_elsewhere(preds_dir: Path, skip: str) -> float:
    """Dollars already spent by the other Claude variants' predictions files (for the shared budget)."""
    total = 0.0
    for path in preds_dir.glob("claude-*.jsonl*"):
        if path.name.split(".jsonl")[0] != skip:
            total += sum(r.get("cost_usd") or 0.0 for r in read_jsonl(path))
    return total


def run_variant(client, variant: str, lines: list[dict], prompts: dict[str, dict], out_path: Path,
                budget_usd: float = 15.0, resume: bool = False, others_usd: float = 0.0) -> dict:
    """Run every line sequentially, appending to `<out>.partial` and renaming it at the end. Stops early (and says so)
    when the projected total spend would pass the budget or the requests keep failing with a 4xx."""
    partial = out_path.with_name(out_path.name + ".partial")
    done: dict[str, dict] = {}
    if resume and partial.exists():
        done = {r["line_id"]: r for r in read_jsonl(partial)}
    spent = sum(r.get("cost_usd") or 0.0 for r in done.values())
    status, bad_streak, rows = "ok", 0, []
    with open(partial, "a" if resume else "w") as f:
        for ln in lines:
            if ln["line_id"] in done:
                rows.append(done[ln["line_id"]])
                continue
            fields, client_side = call_line(client, variant, prompts[ln["line_id"]])
            row = {"line_id": ln["line_id"], "game_id": ln["game_id"], "gold": ln["gold"], "pred": fields["pred"],
                   "probs": None, "latency_ms": fields["latency_ms"], "in_tokens": fields["in_tokens"],
                   "out_tokens": fields["out_tokens"], "cost_usd": fields["cost_usd"], "raw": fields["raw"],
                   "error": fields["error"], "stop_reason": fields.get("stop_reason"),
                   "raw_tail": fields.get("raw_tail", "")}
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            spent += row["cost_usd"] or 0.0
            bad_streak = bad_streak + 1 if client_side else 0
            if bad_streak >= STOP_AFTER_CLIENT_ERRORS:
                status = f"stopped: {bad_streak} consecutive 4xx errors ({row['error']})"
                break
            projected = others_usd + spent / len(rows) * len(lines)
            if projected > budget_usd:
                status = f"stopped: projected spend ${projected:.2f} exceeds the ${budget_usd:.2f} budget"
                break
    if status == "ok":
        partial.replace(out_path)
    return {"variant": variant, "n": len(rows), "of": len(lines), "status": status, "spent_usd": round(spent, 4),
            "out": str(out_path if status == "ok" else partial)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--lines", type=Path, default=BENCH_DIR / "lines.jsonl")
    ap.add_argument("--prompts", type=Path, default=BENCH_DIR / "prompts.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None, help="default runs/bench/au/preds (runs/bench/au/smoke with --limit)")
    ap.add_argument("--limit", type=int, default=None, help="first N lines only (smoke test)")
    ap.add_argument("--resume", action="store_true", help="continue from <variant>.jsonl.partial")
    ap.add_argument("--budget-usd", type=float, default=15.0, help="stop when the projected total Claude spend passes it")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2
    import anthropic
    lines = load_lines(args.lines)
    prompts = {r["line_id"]: r for r in read_jsonl(args.prompts)}
    if args.limit:
        lines = lines[:args.limit]
    out_dir = args.out_dir or (BENCH_DIR / "smoke" if args.limit else BENCH_DIR / "preds")
    out_dir.mkdir(parents=True, exist_ok=True)
    others = spent_elsewhere(out_dir, args.variant)
    client = anthropic.Anthropic(max_retries=MAX_RETRIES, timeout=TIMEOUT_S)
    t0 = time.perf_counter()
    summary = run_variant(client, args.variant, lines, prompts, out_dir / f"{args.variant}.jsonl",
                          budget_usd=args.budget_usd, resume=args.resume, others_usd=others)
    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(summary))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
