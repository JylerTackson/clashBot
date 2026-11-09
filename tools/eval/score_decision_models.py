"""MAC-AU scorer: every model's predictions on the benchmark lines -> accuracy, card accuracy, latency, cost, tests.

  python3 tools/eval/score_decision_models.py [--bench-dir runs/bench/au] [--out docs/data/decision_models_v1]
          [--boot 10000] [--seed 0] [--reference d1-ft-joint-tac]

Reads <bench-dir>/lines.jsonl and every <bench-dir>/preds/*.jsonl (one file per model, the MAC-AU predictions contract:
{line_id, game_id, gold, pred, probs, latency_ms, in_tokens, out_tokens, cost_usd, raw, error}) and writes
<out>.json and <out>.md. Re-runnable on partial inputs: a model is scored on the lines its file covers (the table shows
n), models whose file is missing are listed as pending, and the paired tests use the lines both models cover.

Accuracy = the picked option is the pro's play (the gold option); `pred` null (unparseable / error) counts as wrong.
Card accuracy = the picked option has the pro's card. 95% CIs are cluster bootstraps over games (10,000 resamples,
seed 0, `eval_stats.make_ctx` / `ratio_boot`, as in docs/data/stats_audit_v1; Amendment 1 uses all 766 test lines, 27
games). McNemar is the exact two-sided test on the discordant lines against the reference model; it treats lines as
independent, so the cluster-bootstrap p of the paired difference is shown next to it. Numpy only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tools.eval.eval_stats import ci_of, make_ctx, paired_diff, ratio_boot  # noqa: E402

BENCH_DIR = ROOT / "runs" / "bench" / "au"
OUT_STEM = ROOT / "docs" / "data" / "decision_models_v1"
REFERENCE = "d1-ft-joint-tac"
N_BOOT = 10_000
SEED = 0

# model id -> (label, size / type, group). Group order is the table order; unknown ids sort last.
MODELS: dict[str, tuple[str, str, str]] = {
    "d1-ft-joint-tac": ("d1-ft-joint-tac (live)", "600M, fine-tuned", "d1"),
    "d1-zero-shot": ("d1-omni-600M zero-shot", "600M, zero-shot", "d1"),
    "d1-2b-ft-joint-tac": ("d1-2B fine-tuned", "1.9B, fine-tuned", "d1"),
    "qwen3-4b": ("Qwen3-4B-Instruct-2507 zero-shot", "4B LLM, 4-bit", "qwen"),
    "qwen3-4b-rag3": ("Qwen3-4B + 3 retrieved pro states", "4B LLM, 4-bit, RAG few-shot", "qwen"),
    "qwen3-14b": ("Qwen3-14B zero-shot", "14B LLM, 4-bit", "qwen"),
    "claude-haiku-5-5": ("Claude Haiku 5.5", "API", "claude"),
    "claude-sonnet-5-5": ("Claude Sonnet 5.5", "API", "claude"),
    "random": ("random (one seeded draw)", "baseline", "baseline"),
    "popularity": ("popularity", "baseline", "baseline"),
    "placement-prior": ("placement prior", "baseline", "baseline"),
    "lightgbm": ("LightGBM LambdaRank (seed 0)", "baseline, feature ranker", "baseline"),
}
GROUPS = ("d1", "qwen", "claude", "baseline", "other")
RANDOM_EXPECTED = "random-expected"


# ---------------------------------------------------------------------------------------------------------- loading
def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_preds(preds_dir: Path, lines: list[dict]) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """{model_id: {line_id: row}} for every preds/*.jsonl, plus warnings. Rows for unknown lines are dropped, a repeated
    line_id keeps the last row, and a row whose gold disagrees with lines.jsonl is a hard error (wrong line file)."""
    gold = {ln["line_id"]: ln["gold"] for ln in lines}
    out: dict[str, dict[str, dict]] = {}
    warnings: list[str] = []
    for path in sorted(Path(preds_dir).glob("*.jsonl")):
        rows: dict[str, dict] = {}
        dropped = dup = 0
        for r in read_jsonl(path):
            lid = r.get("line_id")
            if lid not in gold:
                dropped += 1
                continue
            if r.get("gold") is not None and r["gold"] != gold[lid]:
                raise ValueError(f"{path.name}: gold of {lid} is {r['gold']}, lines.jsonl says {gold[lid]}")
            dup += lid in rows
            rows[lid] = r
        if dropped:
            warnings.append(f"{path.stem}: {dropped} rows are not benchmark lines and were ignored")
        if dup:
            warnings.append(f"{path.stem}: {dup} repeated line_id rows (the last one is used)")
        if rows:
            out[path.stem] = rows
    return out, warnings


META_KEYS = ("backend", "dtype", "mps", "torch", "transformers", "load_and_warmup_s", "loader_note", "note", "finished")


def load_meta(preds_dir: Path) -> dict[str, dict]:
    """{model_id: how the run was made} from the optional <model_id>.meta.json sidecars of the local runs."""
    out = {}
    for path in sorted(Path(preds_dir).glob("*.meta.json")):
        meta = json.loads(path.read_text())
        out[path.name[: -len(".meta.json")]] = {k: meta[k] for k in META_KEYS if meta.get(k) is not None}
    return out


# --------------------------------------------------------------------------------------------------------- metrics
def card_of(key: str) -> str:
    return key.split("|")[0]


def pick_hits(rows: dict[str, dict], lines: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """(covered lines, exact-play hits, card hits) for `rows` over `lines` (a null pred is a miss)."""
    covered, hit, card_hit = [], [], []
    for ln in lines:
        r = rows.get(ln["line_id"])
        if r is None:
            continue
        pred = r.get("pred")
        ok = isinstance(pred, int) and not isinstance(pred, bool) and 0 <= pred < len(ln["keys"])
        covered.append(ln)
        hit.append(float(ok and pred == ln["gold"]))
        card_hit.append(float(ok and card_of(ln["keys"][pred]) == card_of(ln["keys"][ln["gold"]])))
    return covered, np.array(hit), np.array(card_hit)


class Boot:
    """Game-cluster bootstrap contexts, cached by the covered line set (equal sets share their resamples)."""

    def __init__(self, n_boot: int = N_BOOT, seed: int = SEED):
        self.n_boot, self.seed, self._cache = n_boot, seed, {}

    def ctx(self, covered: list[dict]) -> dict:
        key = tuple(ln["line_id"] for ln in covered)
        if key not in self._cache:
            self._cache[key] = make_ctx([ln["game_id"] for ln in covered], self.n_boot, self.seed)
        return self._cache[key]

    def ci(self, covered: list[dict], values: np.ndarray) -> dict:
        est, boot = ratio_boot(self.ctx(covered), values)
        return {k: round(float(v), 4) for k, v in ci_of(est, boot).items()}

    def diff(self, covered: list[dict], a: np.ndarray, b: np.ndarray) -> dict:
        d = paired_diff(self.ctx(covered), a, b)
        return {k: (round(float(v), 6) if isinstance(v, float) else v) for k, v in d.items()}


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    a = np.asarray(values, dtype=float)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "mean": round(float(a.mean()), 3), "n": int(len(a))}


def mean_or_none(values: list[float], nd: int = 6) -> float | None:
    return round(float(np.mean(values)), nd) if values else None


def score_model(rows: dict[str, dict], lines: list[dict], boot: Boot, ref_rows: dict[str, dict] | None = None) -> dict:
    """All metrics of one model over the `lines` its `rows` cover."""
    covered, hit, card_hit = pick_hits(rows, lines)
    used = [rows[ln["line_id"]] for ln in covered]
    out: dict = {"n": len(covered)}
    if not covered:
        return out
    out["n_games"] = len({ln["game_id"] for ln in covered})
    out["acc"] = boot.ci(covered, hit)
    out["card_acc"] = boot.ci(covered, card_hit)
    out["latency_ms"] = percentiles([r["latency_ms"] for r in used if r.get("latency_ms") is not None])
    out["cost_usd_per_decision"] = mean_or_none([r["cost_usd"] for r in used if r.get("cost_usd") is not None])
    out["in_tokens_mean"] = mean_or_none([r["in_tokens"] for r in used if r.get("in_tokens") is not None], 1)
    out["out_tokens_mean"] = mean_or_none([r["out_tokens"] for r in used if r.get("out_tokens") is not None], 1)
    fails = sum(r.get("pred") is None for r in used)
    out["parse_failures"] = {"n": int(fails), "rate": round(fails / len(used), 4)}
    out["errors"] = int(sum(bool(r.get("error")) for r in used))
    by_game: dict[str, list[float]] = {}
    for ln, h in zip(covered, hit):
        by_game.setdefault(ln["game_id"], []).append(h)
    out["per_game"] = {g: {"n": len(v), "acc": round(float(np.mean(v)), 4)} for g, v in sorted(by_game.items())}
    if ref_rows is not None and ref_rows is not rows:
        both = [ln for ln in lines if ln["line_id"] in rows and ln["line_id"] in ref_rows]
        if both:
            _, a, _ = pick_hits(rows, both)
            _, b, _ = pick_hits(ref_rows, both)
            d = boot.diff(both, a, b)
            out["vs_reference"] = {
                "n_common": len(both), "model_only": d["mcnemar_a_only"], "reference_only": d["mcnemar_b_only"],
                "mcnemar_p": round(d["mcnemar_p"], 6),
                "diff": {"est": round(d["est"], 4), "lo": round(d["lo"], 4), "hi": round(d["hi"], 4),
                         "p_boot": round(d["p_boot"], 4)}}
    return out


def random_expected(lines: list[dict]) -> dict:
    """Analytic row for uniform random picks: accuracy 0.25, card accuracy = mean share of options with the gold card."""
    card = [sum(card_of(k) == card_of(ln["keys"][ln["gold"]]) for k in ln["keys"]) / len(ln["keys"]) for ln in lines]
    acc = [1 / len(ln["keys"]) for ln in lines]
    return {"n": len(lines), "n_games": len({ln["game_id"] for ln in lines}),
            "acc": {"est": round(float(np.mean(acc)), 4), "lo": None, "hi": None},
            "card_acc": {"est": round(float(np.mean(card)), 4), "lo": None, "hi": None},
            "latency_ms": percentiles([]), "cost_usd_per_decision": 0.0, "parse_failures": {"n": 0, "rate": 0.0},
            "errors": 0, "analytic": True}


# ---------------------------------------------------------------------------------------------------------- report
def order_key(model_id: str) -> tuple[int, int]:
    info = MODELS.get(model_id)
    group = info[2] if info else "other"
    return GROUPS.index(group), list(MODELS).index(model_id) if model_id in MODELS else 999


def describe(model_id: str) -> tuple[str, str]:
    info = MODELS.get(model_id)
    return (info[0], info[1]) if info else (model_id, "?")


def build_report(lines: list[dict], preds: dict[str, dict[str, dict]], reference: str = REFERENCE,
                 n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    boot = Boot(n_boot, seed)
    ref_rows = preds.get(reference)
    models = {}
    for mid in sorted(preds, key=order_key):
        label, kind = describe(mid)
        models[mid] = {"label": label, "type": kind, **score_model(preds[mid], lines, boot, ref_rows)}
    models[RANDOM_EXPECTED] = {"label": "random (expected)", "type": "baseline, analytic", **random_expected(lines)}
    games = sorted({ln["game_id"] for ln in lines})
    return {
        "spec": "MAC-AU Part B (Amendment 1: all test lines)",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lines": {"n": len(lines), "n_games": len(games),
                  "by_game": {g: sum(ln["game_id"] == g for ln in lines) for g in games}},
        "reference": reference if ref_rows is not None else None,
        "bootstrap": {"n_boot": n_boot, "seed": seed, "unit": "game (cluster)"},
        "models": models,
        "pending": [m for m in MODELS if m not in preds],
    }


def ci_text(c: dict | None, nd: int = 3) -> str:
    if not c or c.get("est") is None:
        return "-"
    if c.get("lo") is None:
        return f"{c['est']:.{nd}f}"
    return f"{c['est']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


def _ms(x: float) -> str:
    return f"{x:.3f}" if x < 0.1 else f"{x:.1f}" if x < 10 else f"{x:.0f}" if x < 10_000 else f"{x / 1000:.1f}k"


def lat_text(m: dict) -> str:
    lat = m.get("latency_ms") or {}
    if lat.get("p50") is None:
        return "-"
    return f"{_ms(lat['p50'])} / {_ms(lat['p95'])}"


def cost_text(m: dict) -> str:
    c = m.get("cost_usd_per_decision")
    if c is None:
        return "-"
    return "0 (local)" if c == 0 else f"${c:.5f}"


def p_text(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def render_table(models: dict, reference: str | None, n_lines: int) -> list[str]:
    head = ("| model | size / type | n | accuracy [95% CI] | card accuracy [95% CI] | diff vs d1-ft-joint-tac [95% CI] "
            "| McNemar p | latency p50 / p95 (ms) | cost / decision | parse failures |")
    out = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for mid, m in models.items():
        if "acc" not in m:
            continue
        pf = m["parse_failures"]
        v = m.get("vs_reference")
        if mid == reference:
            diff_txt, p_txt = "(reference)", "(reference)"
        elif v:
            d = v["diff"]
            diff_txt = f"{d['est']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
            p_txt = p_text(v["mcnemar_p"])
        else:
            diff_txt, p_txt = "-", "-"
        n_txt = str(m["n"]) if m["n"] == n_lines else f"{m['n']} (partial)"
        analytic = m.get("analytic", False)
        cost_txt = "-" if analytic else cost_text(m)
        fail_txt = "-" if analytic else f"{pf['n']}/{m['n']} ({100 * pf['rate']:.1f}%)"
        out.append(f"| {m['label']} | {m['type']} | {n_txt} | {ci_text(m['acc'])} | {ci_text(m['card_acc'])} | "
                   f"{diff_txt} | {p_txt} | {lat_text(m)} | {cost_txt} | {fail_txt} |")
    return out


def render_markdown(rep: dict) -> str:
    info = rep["lines"]
    boot = rep["bootstrap"]
    out = ["# Decision models vs the pro: accuracy and latency on the held-out test games (MAC-AU)", "",
           f"Generated {rep['generated']} by `tools/eval/score_decision_models.py` from `runs/bench/au/preds/*.jsonl`; "
           "re-run it when predictions change.", "",
           f"**Lines:** all {info['n']} 4-way lines of `dataset/v1/d1/joint_tac_v2/test.jsonl`, {info['n_games']} games "
           "(Amendment 1; the first plan was the 5 largest games).", "",
           "**Accuracy** = the picked option is the pro's play. **Card accuracy** = the picked option has the pro's card. "
           "A null prediction (unparseable reply or error) counts as wrong. 95% CIs are cluster bootstraps over the "
           f"{info['n_games']} games ({boot['n_boot']:,} resamples, seed {boot['seed']}, as in `stats_audit_v1`). "
           "`diff` is the paired accuracy difference to d1-ft-joint-tac (same lines, same game resamples); McNemar is the "
           "exact two-sided test on the lines where the two disagree (lines treated as independent). Latency is wall "
           "time per decision (local models: validate, tokenise, forward, parse, measured under the heavy lock after a "
           "warm-up; API models: includes the network round trip).", "",
           "## Accuracy and latency", ""]
    out += render_table(rep["models"], rep["reference"], info["n"])
    if rep["pending"]:
        out += ["", "## Pending", "", "No predictions file yet for: " + ", ".join(f"`{m}`" for m in rep["pending"]) + "."]
    if rep.get("runs"):
        out += ["", "## How the local runs were made", ""]
        for mid, meta in rep["runs"].items():
            out.append(f"- `{mid}`: " + "; ".join(f"{k} {v}" for k, v in meta.items()))
    if rep.get("warnings"):
        out += ["", "## Warnings", ""] + [f"- {w}" for w in rep["warnings"]]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    ap.add_argument("--out", type=Path, default=OUT_STEM, help="output stem: writes <out>.json and <out>.md")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--reference", default=REFERENCE)
    args = ap.parse_args(argv)
    lines = read_jsonl(args.bench_dir / "lines.jsonl")
    preds, warnings = load_preds(args.bench_dir / "preds", lines)
    rep = build_report(lines, preds, args.reference, args.boot, args.seed)
    rep["warnings"] = warnings
    rep["runs"] = load_meta(args.bench_dir / "preds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    args.out.with_suffix(".md").write_text(render_markdown(rep))
    print(render_markdown(rep))
    print(f"wrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.md')}")
    return rep


if __name__ == "__main__":
    main()
