"""Latency table for the report (MAC-AP): per-stage p50 / p95 across runs, plus isolated micro-benchmarks.

  python3 tools/eval/latency_table.py [--h6 DIR ...] [--poc-root runs/poc] [--poc-glob '2026110*']
          [--out docs/data/latency_v2.json] [--skip-micro] [--micro katacr,d1,act,knn,candidates]

1. whole-stack dry runs (the three H6 repetitions, `<dir>/summary.json`): per stage, mean +- sd across runs of the p50
   and of the p95 of `decision_ms`;
2. every live or dry run under <poc-root>/<glob> that has a `summary.json`: one row each;
3. micro-benchmarks (200 calls after warm-up each; heavy: loads KataCR, d1, db/): KataCR detect on MPS vs CPU (one real
   frame), d1-ft-joint-tac 4-option ranking (transformers, MPS), act-gbdt `p_act`, exact kNN vs HNSW (ef 40) on the state
   index, `CandidateGenerator.generate` on 50 real states (lookup channels only, no tactics report).
Writes <out> and <out>.md. Read-only on dataset/ and db/.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

STAGES = ("capture", "perception", "live_state", "trigger", "candidates", "rank", "execute", "total")
N_CALLS = 200
STAT_KEYS = frozenset({"n", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"})
WARMUP = 5
MICRO_NAMES = ("katacr", "d1", "act", "knn", "candidates")
VIDEO = "data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4"
CALIB = "runs/poc/mirror_calib.json"
D1_MODEL = "db/models/d1-ft-joint-tac"
ACT_MODEL = "db/models/act-gbdt"
D1_LINES = "dataset/v1/d1/joint_tac_v2/val.jsonl"
STATES = "dataset/v1/states.jsonl"


# ----------------------------------------------------------------------------------------------------------------
# aggregation of summary.json files (pure)
# ----------------------------------------------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def stage_stats(summary: dict, key: str = "decision_ms") -> dict[str, dict[str, float]]:
    """{stage: {"p50", "p95"}} of one summary.json, for the stages it recorded."""
    block = summary.get(key) or {}
    return {s: {"p50": float(block[s]["p50"]), "p95": float(block[s]["p95"])} for s in STAGES
            if isinstance(block.get(s), dict) and block[s].get("p50") is not None and block[s].get("p95") is not None}


def mean_sd(values: list[float]) -> dict:
    """Mean and sample sd (n - 1) of per-run values; sd is None for a single run."""
    return {"n": len(values), "mean": float(statistics.fmean(values)),
            "sd": float(statistics.stdev(values)) if len(values) > 1 else None, "values": [float(v) for v in values]}


def aggregate_runs(summaries: list[dict], key: str = "decision_ms") -> dict[str, dict[str, dict]]:
    """{stage: {"p50": mean_sd, "p95": mean_sd}} across the runs that recorded the stage."""
    per_run = [stage_stats(s, key) for s in summaries]
    out: dict = {}
    for stage in STAGES:
        rows = [r[stage] for r in per_run if stage in r]
        if rows:
            out[stage] = {q: mean_sd([r[q] for r in rows]) for q in ("p50", "p95")}
    return out


def run_row(run_dir: Path) -> dict | None:
    """One row for a run dir with a summary.json: mode, decisions, per-stage p50 / p95 of `decision_ms`."""
    sp = Path(run_dir) / "summary.json"
    if not sp.exists():
        return None
    summary = load_json(sp)
    cfg = load_json(Path(run_dir) / "config.json") if (Path(run_dir) / "config.json").exists() else {}
    return {"run": Path(run_dir).name, "mode": "live" if cfg.get("live") else "dry",
            "source": cfg.get("source"), "ranker": cfg.get("ranker"), "policy": cfg.get("policy"),
            "seconds": cfg.get("seconds"), "decisions": summary.get("decisions"), "frames": summary.get("frames"),
            "target_met": summary.get("target_met"), "stages": stage_stats(summary)}


def live_rows(poc_root: Path, pattern: str = "2026110*") -> list[dict]:
    rows = [run_row(d) for d in sorted(Path(poc_root).glob(pattern)) if d.is_dir()]
    return [r for r in rows if r is not None]


def percentile_stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=float)
    return {"n": int(a.size), "mean_ms": round(float(a.mean()), 3), "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3), "min_ms": round(float(a.min()), 3),
            "max_ms": round(float(a.max()), 3)}


def time_calls(fn, n: int = N_CALLS, warmup: int = WARMUP, args_for=None) -> list[float]:
    """Wall time in ms of `fn(args_for(i))` (or `fn()`), `n` calls after `warmup` untimed ones."""
    call = (lambda i: fn()) if args_for is None else (lambda i: fn(args_for(i)))
    for i in range(warmup):
        call(-1 - i)
    out = []
    for i in range(n):
        t0 = time.perf_counter()
        call(i)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


# ----------------------------------------------------------------------------------------------------------------
# micro-benchmarks (heavy; imported lazily)
# ----------------------------------------------------------------------------------------------------------------

def _read_states(n: int, split: str = "val", need_action: bool = False) -> list[dict]:
    """`n` evenly spaced records of `split` from states.jsonl (state, t, context_text, game_id)."""
    pool = []
    with open(ROOT / STATES) as f:
        for line in f:
            if '"split": "%s"' % split in line or '"split":"%s"' % split in line:
                r = json.loads(line)
                if r.get("split") == split and r.get("context_text") and (not need_action or (r.get("action") or {}).get("key")):
                    pool.append(r)
    idx = np.linspace(0, len(pool) - 1, n).astype(int)
    return [pool[i] for i in idx]


def micro_katacr() -> dict:
    """KataCR YOLOv8 detect on one real frame (arena crop, imgsz 640 as in the live runs): MPS vs CPU."""
    import torch
    from cr_perception.config import Calibration
    from cr_perception.detect import default_katacr_detector
    from cr_perception.sources import VideoFrameSource
    calib = Calibration.load(ROOT / CALIB)
    src = VideoFrameSource(ROOT / VIDEO, start=60.0)
    frame = src.frame_at(90.0)
    src.close()
    fh, fw = frame.shape[:2]
    cx, cy, cw, ch = calib.content_rect(fw, fh)
    content = frame[cy:cy + ch, cx:cx + cw]
    crop = calib.arena_crop(cw, ch)
    out: dict = {"frame": f"{VIDEO} t=90 s, content {cw}x{ch}, arena crop {crop[2]}x{crop[3]}, imgsz 640"}
    for dev in ("mps", "cpu"):
        if dev == "mps" and not torch.backends.mps.is_available():
            out[dev] = {"error": "mps unavailable"}
            continue
        det = default_katacr_detector(imgsz=640, device=dev)
        res = percentile_stats(time_calls(lambda: det.detect(content, crop)))
        res["detections"] = len(det.detect(content, crop))
        out[dev] = res
        del det
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()
    return out


def micro_d1() -> dict:
    """d1-ft-joint-tac scoring one 4-option line (transformers backend, MPS, fp16): the lines of joint_tac_v2/val."""
    from clashbot.ranker import D1Ranker
    lines = []
    with open(ROOT / D1_LINES) as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            lines.append(json.loads(line))
    ranker = D1Ranker(model_path=str(ROOT / D1_MODEL), backend="transformers")
    def call(i):
        ln = lines[i % len(lines)]
        ranker.probabilities(ln["state"], ln["questions"])
    t0 = time.perf_counter()
    ranker.probabilities(lines[0]["state"], lines[0]["questions"])            # load + first call
    cold_ms = (time.perf_counter() - t0) * 1000.0
    res = percentile_stats(time_calls(call, args_for=lambda i: max(i, 0)))
    res["cold_first_call_ms"] = round(cold_ms, 1)
    res["note"] = "each call = ranker.probabilities on one joint_tac_v2 line (4 options); 50 distinct lines cycled"
    return res


def micro_act() -> dict:
    """act-gbdt: features + tree evaluation of one state (`ActGBDT.p_act`), plus the feature step alone."""
    from clashbot.act_model import ActGBDT, act_features
    from clashbot.game_rules import load_cards
    model = ActGBDT.load(ROOT / ACT_MODEL)
    cards = load_cards()
    recs = _read_states(200)
    full = percentile_stats(time_calls(lambda r: model.p_act(r["state"], r["t"]), args_for=lambda i: recs[i % 200]))
    feats = percentile_stats(time_calls(lambda r: act_features(r["state"], r["t"], cards), args_for=lambda i: recs[i % 200]))
    return {"p_act": full, "features_only": feats, "horizon_s": getattr(model, "horizon", None)}


def micro_knn() -> dict:
    """Exact vs HNSW (ef 40) nearest-neighbour query on the 10.9 k state vectors: the production `StateIndex.knn` path
    and the raw search (matmul + top-10 vs hnswlib.knn_query k = 10)."""
    import hnswlib
    from clashbot.retrieval import StateIndex
    exact = StateIndex()
    ann = StateIndex(use_hnsw=True)
    ann._hnsw.set_ef(40)
    rng = np.random.default_rng(0)
    rows = rng.integers(0, exact.n, N_CALLS)
    vecs = np.asarray(exact.M[rows], dtype=np.float32)
    M = np.asarray(exact.M)
    ids = np.arange(exact.n)
    raw = hnswlib.Index(space="ip", dim=int(M.shape[1]))
    raw.load_index(str(ROOT / "db" / "state_features.hnsw"), max_elements=exact.n)
    raw.set_ef(40)

    def raw_exact(v):
        s = np.asarray(M @ v, dtype=np.float64)
        return np.lexsort((ids, -s))[:10]

    def pick(i):
        return vecs[i % len(vecs)]
    res = {"n_vectors": int(exact.n), "dim": int(M.shape[1]), "ef": 40,
           "StateIndex.knn exact (k=10)": percentile_stats(time_calls(lambda v: exact.knn(v, 10), args_for=pick)),
           "StateIndex.knn hnsw (k=10)": percentile_stats(time_calls(lambda v: ann.knn(v, 10), args_for=pick)),
           "raw exact (matmul + top-10)": percentile_stats(time_calls(raw_exact, args_for=pick)),
           "raw hnswlib knn_query (k=10)": percentile_stats(time_calls(lambda v: raw.knn_query(v.reshape(1, -1), k=10), args_for=pick))}
    hits = [len(set(int(x) for x in raw.knn_query(v.reshape(1, -1), k=10)[0][0]) & set(int(x) for x in raw_exact(v))) / 10 for v in vecs]
    res["hnsw_recall_at_10_vs_exact"] = round(float(np.mean(hits)), 4)
    return res


def micro_candidates() -> dict:
    """`CandidateGenerator.generate` (kNN + BM25 + KG lookup, no tactics report) on 50 real val states, 4 passes."""
    from clashbot.candidates import CandidateGenerator
    from clashbot.game_rules import load_cards
    from clashbot.kg import KnowledgeGraph
    from clashbot.retrieval import StateIndex
    cards = load_cards()
    gen = CandidateGenerator(StateIndex(), KnowledgeGraph(), cards)
    recs = _read_states(50)
    ms = time_calls(lambda r: gen.generate(r["state"], r["t"], r["context_text"]), args_for=lambda i: recs[i % 50])
    res = percentile_stats(ms)
    res["note"] = "50 distinct val states cycled 4x; no tactics report, no exclude_game filter (as in the live stack)"
    return res


MICRO_FUNCS = {"katacr": micro_katacr, "d1": micro_d1, "act": micro_act, "knn": micro_knn, "candidates": micro_candidates}


def run_micro(names: list[str]) -> dict:
    out = {}
    for name in names:
        t0 = time.time()
        try:
            out[name] = MICRO_FUNCS[name]()
        except Exception as exc:                                  # noqa: BLE001 - report, keep going
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
        out[name]["bench_wall_s"] = round(time.time() - t0, 1)
        print(f"[micro] {name}: {time.time() - t0:.0f}s", file=sys.stderr)
        gc.collect()
    return out


# ----------------------------------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------------------------------

def _pm(c: dict) -> str:
    return f"{c['mean']:.1f}" if c["sd"] is None else f"{c['mean']:.1f} ± {c['sd']:.1f}"


def _cell(stats: dict | None, stage: str) -> str:
    s = (stats or {}).get(stage)
    return "-" if s is None else f"{s['p50']:.0f} / {s['p95']:.0f}"


def _micro_rows(name: str, m: dict) -> list[tuple[str, dict]]:
    if "error" in m:
        return [(name, m)]
    if "p50_ms" in m:
        return [(name, m)]
    return [(f"{name} / {k}", v) for k, v in m.items() if isinstance(v, dict) and ("p50_ms" in v or "error" in v)]


def render_markdown(res: dict) -> str:
    L = ["# Latency (MAC-AP)", "",
         f"git HEAD `{res['meta'].get('git_head')}` at run time; decision code frozen at `66917e7` (D-051). Times in ms. "
         "`decision_ms` stages are per decision, from each run's `summary.json`; the live runs include the 183 ms two-click `execute` stage "
         "(dry runs do not click, execute = 0).", ""]
    h6 = res.get("h6")
    if h6:
        L += [f"## Whole stack, {len(h6['runs'])} repeated dry runs (video `Z8-4VhLjrGU` 60-180 s, d1-ft-joint-tac + fast policy, MPS)", "",
              "Mean ± sd across runs (n-1) of each run's p50 and p95.", "",
              "| stage | p50 mean ± sd | p95 mean ± sd | p50 per run | p95 per run |", "|---|---|---|---|---|"]
        for stage in STAGES:
            c = h6["stages"].get(stage)
            if c:
                L.append(f"| {stage} | {_pm(c['p50'])} | {_pm(c['p95'])} | "
                         f"{', '.join(f'{v:.1f}' for v in c['p50']['values'])} | {', '.join(f'{v:.1f}' for v in c['p95']['values'])} |")
        L += ["", f"Runs: {', '.join(h6['runs'])}; decisions per run: {h6['decisions']}; target p95 < 300 ms met in {h6['target_met']} of {len(h6['runs'])} runs.", ""]
    rows = res.get("runs") or []
    if rows:
        L += ["## Every recorded run under `runs/poc` (p50 / p95 per stage)", "",
              "| run | mode | decisions | " + " | ".join(STAGES) + " |", "|---|---|---|" + "---|" * len(STAGES)]
        for r in rows:
            L.append(f"| {r['run']} | {r['mode']} | {r['decisions']} | " + " | ".join(_cell(r["stages"], s) for s in STAGES) + " |")
        live = [r for r in rows if r["mode"] == "live" and r["stages"]]
        if live:
            L.append("| median of live runs | live | - | " + " | ".join(
                (lambda vs: "-" if not vs else f"{statistics.median(v[0] for v in vs):.0f} / {statistics.median(v[1] for v in vs):.0f}")(
                    [(r["stages"][s]["p50"], r["stages"][s]["p95"]) for r in live if s in r["stages"]]) for s in STAGES) + " |")
        L.append("")
    micro = res.get("micro")
    if micro:
        L += ["## Isolated micro-benchmarks (200 calls each after warm-up)", "",
              "| benchmark | n | mean | p50 | p95 | min | max |", "|---|---|---|---|---|---|---|"]
        for name, m in micro.items():
            for label, v in _micro_rows(name, m):
                if "error" in v:
                    L.append(f"| {label} | ERROR: {v['error']} | | | | | |")
                else:
                    L.append(f"| {label} | {v['n']} | {v['mean_ms']:.2f} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} | {v['min_ms']:.2f} | {v['max_ms']:.2f} |")
        L.append("")
        for name, m in micro.items():
            extras = {k: v for k, v in m.items() if not isinstance(v, dict) and k not in STAT_KEYS and k != "bench_wall_s"}
            if "frame" in m:
                extras["frame"] = m["frame"]
            if extras:
                L.append(f"- {name}: " + "; ".join(f"{k} = {v}" for k, v in extras.items()))
        L.append("")
    return "\n".join(L)


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_result(h6_dirs: list[Path], poc_root: Path, pattern: str, micro: dict | None) -> dict:
    res: dict = {"meta": {"git_head": git_head(), "n_calls": N_CALLS, "warmup": WARMUP}}
    summaries = [load_json(Path(d) / "summary.json") for d in h6_dirs if (Path(d) / "summary.json").exists()]
    if summaries:
        res["h6"] = {"runs": [Path(d).name for d in h6_dirs if (Path(d) / "summary.json").exists()],
                     "stages": aggregate_runs(summaries), "frame_stages": aggregate_runs(summaries, "frame_ms"),
                     "decisions": [s.get("decisions") for s in summaries],
                     "target_met": sum(1 for s in summaries if s.get("target_met"))}
    res["runs"] = live_rows(poc_root, pattern)
    if micro is not None:
        res["micro"] = micro
    return res


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h6", type=Path, nargs="*", default=[ROOT / "runs" / "rerun" / f"lat_{i}" for i in (1, 2, 3)])
    ap.add_argument("--poc-root", type=Path, default=ROOT / "runs" / "poc")
    ap.add_argument("--poc-glob", default="2026110*")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "data" / "latency_v2.json")
    ap.add_argument("--skip-micro", action="store_true", help="aggregate the summaries only (no model is loaded)")
    ap.add_argument("--micro", default=",".join(MICRO_NAMES), help="comma-separated subset of " + ",".join(MICRO_NAMES))
    ap.add_argument("--keep-micro", type=Path, default=None,
                    help="JSON of a previous micro run to embed instead of running the micro-benchmarks")
    args = ap.parse_args(argv)
    micro = None
    if args.keep_micro:
        micro = load_json(args.keep_micro)
    elif not args.skip_micro:
        names = [n for n in args.micro.split(",") if n]
        bad = [n for n in names if n not in MICRO_FUNCS]
        if bad:
            ap.error(f"unknown micro-benchmarks {bad}")
        micro = run_micro(names)
    res = build_result(args.h6, args.poc_root, args.poc_glob, micro)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    md = render_markdown(res)
    args.out.with_suffix(".md").write_text(md)
    print(md)
    return res


if __name__ == "__main__":
    main()
