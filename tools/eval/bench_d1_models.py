"""MAC-AU Part B: the benchmark line file, the d1 family and the baselines (accuracy and latency vs the pro).

  python3 tools/eval/bench_d1_models.py lines                         # runs/bench/au/lines.jsonl (all 766 test lines)
  python3 tools/eval/bench_d1_models.py d1 --model-id d1-ft-joint-tac --model db/models/d1-ft-joint-tac
  python3 tools/eval/bench_d1_models.py d1 --model-id d1-zero-shot                          # LiquidAI/d1-omni-600M
  python3 tools/eval/bench_d1_models.py d1 --model-id d1-2b-ft-joint-tac --model db/models/d1-2b-ft-joint-tac
  python3 tools/eval/bench_d1_models.py baselines                     # random / popularity / placement prior / LightGBM

Lines = every row of dataset/v1/d1/joint_tac_v2/test.jsonl (MAC-AU Amendment 1: 766 lines, 27 games; the first
plan was the 5 largest games, `build_lines(games=...)` still filters). Predictions follow the
MAC-AU contract: runs/bench/au/preds/<model_id>.jsonl, one row per line
{line_id, game_id, gold, pred, probs, latency_ms, in_tokens, out_tokens, cost_usd, raw, error}; a sidecar
<model_id>.meta.json records how the run was made. `d1` runs are timed per decision (validate + tokenise + forward +
parse) after an untimed warm-up and must hold the global heavy lock (the caller wraps the command in `lockf`).

d1-2B loader fix (`make_loadable`): the fine-tuned d1-2B checkpoint's config.json names `configuration_lfm3_vl.Lfm3VlConfig`
as its AutoConfig, a module that looks `lfm3` up in transformers' CONFIG_MAPPING before anything has registered it
(transformers 5.22 ships no lfm3), so loading fails with KeyError 'lfm3'. The original LiquidAI/d1-2B snapshot names
`native_lfm3_vl.Lfm3VlConfig`, which imports `native_lfm3` first and registers the vendored Lfm3Config. The checkpoint
dir is never changed: it is mirrored (weights symlinked, small files copied) into runs/bench/au/d1-2b-loadable/ with
that one auto_map entry restored.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.config import DATASET_DIR  # noqa: E402

BENCH_DIR = ROOT / "runs" / "bench" / "au"
LINES_PATH = BENCH_DIR / "lines.jsonl"
PREDS_DIR = BENCH_DIR / "preds"
LOADABLE_2B = BENCH_DIR / "d1-2b-loadable"
SOURCE = "d1/joint_tac_v2/test.jsonl"
FEATURE_PREDS = ROOT / "docs" / "data" / "feature_ranker_preds_v2.json"
PRED_FIELDS = ("line_id", "game_id", "gold", "pred", "probs", "latency_ms", "in_tokens", "out_tokens", "cost_usd",
               "raw", "error")
BAD_AUTO_CONFIG = "configuration_lfm3_vl."          # a checkpoint whose AutoConfig skips the lfm3 registration
GOOD_AUTO_CONFIG = "native_lfm3_vl.Lfm3VlConfig"
WARMUP = 3


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------- the line file
def build_lines(dataset: Path = DATASET_DIR, games: tuple[str, ...] | None = None, source: str = SOURCE) -> list[dict]:
    """The benchmark lines, in source-file order: every 4-way line of `source` (Amendment 1: all 766 test lines, 27
    games), or only those of `games`. `line_id` is the d1 line id, which is also the id of the dataset state behind it
    (`state_id`, a row of states.jsonl); `keys` / `notes` are the 4 options in the order d1 sees them."""
    state_ids = {r["id"] for r in read_jsonl(dataset / "states.jsonl") if games is None or r["game_id"] in games}
    out = []
    for r in read_jsonl(dataset / source):
        if games is not None and r["game_id"] not in games:
            continue
        crit = r["questions"]["play"]["criteria"]
        keys = list(crit)
        if len(keys) != 4:
            continue
        if r["id"] not in state_ids:
            raise ValueError(f"line {r['id']} has no dataset state")
        out.append({"line_id": r["id"], "game_id": r["game_id"], "state_id": r["id"], "kind": r["kind"],
                    "gold": keys.index(r["answers"]["play"]), "keys": keys, "notes": [crit[k] for k in keys],
                    "state": r["state"]})
    return out


def write_lines(path: Path = LINES_PATH, **kw) -> list[dict]:
    lines = build_lines(**kw)
    write_jsonl(path, lines)
    return lines


def load_lines(path: Path = LINES_PATH) -> list[dict]:
    return read_jsonl(path)


def load_source_rows(line_ids: list[str], dataset: Path = DATASET_DIR, source: str = SOURCE) -> list[dict]:
    """The d1 source rows (state + questions as exported) of `line_ids`, in that order."""
    by_id = {r["id"]: r for r in read_jsonl(dataset / source)}
    return [by_id[i] for i in line_ids]


# ------------------------------------------------------------------------------------------------ prediction rows
def pred_row(line: dict, pred: int | None, probs: list[float] | None, latency_ms: float | None, *,
             in_tokens: int | None = None, out_tokens: int | None = None, cost_usd: float | None = 0.0,
             raw: str | None = None, error: str | None = None) -> dict:
    """One row of the predictions contract (floats rounded to keep the files small and diff-able)."""
    return {"line_id": line["line_id"], "game_id": line["game_id"], "gold": line["gold"], "pred": pred,
            "probs": None if probs is None else [round(float(p), 6) for p in probs],
            "latency_ms": None if latency_ms is None else round(float(latency_ms), 4),
            "in_tokens": in_tokens, "out_tokens": out_tokens, "cost_usd": cost_usd,
            "raw": None if raw is None else str(raw)[:200], "error": error}


def write_preds(model_id: str, rows: list[dict], meta: dict | None = None, preds_dir: Path = PREDS_DIR) -> Path:
    path = preds_dir / f"{model_id}.jsonl"
    write_jsonl(path, rows)
    if meta is not None:
        (preds_dir / f"{model_id}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return path


def softmax(scores) -> list[float]:
    x = np.asarray(scores, dtype=float)
    e = np.exp(x - x.max())
    return list(e / e.sum())


def argmax_first(scores) -> int:
    """Index of the first maximum (ties go to the earliest option, as in eval_stats / eval_d1_zero_shot)."""
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


# ------------------------------------------------------------------------------------------- the d1-2B loader fix
def needs_loader_fix(config: dict | None) -> bool:
    """True when config.json's AutoConfig is the vendored `configuration_lfm3_vl` module (see the module docstring)."""
    auto = ((config or {}).get("auto_map") or {}).get("AutoConfig", "")
    return str(auto).startswith(BAD_AUTO_CONFIG)


def make_loadable(src: Path, dst: Path = LOADABLE_2B) -> Path:
    """Mirror checkpoint dir `src` into `dst` with the AutoConfig entry restored to `native_lfm3_vl.Lfm3VlConfig`.
    Weights are symlinked, every other file copied; `src` is only read. Safe to call again."""
    src, dst = Path(src).resolve(), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_dir() or f.name == "config.json":
            continue
        target = dst / f.name
        if target.is_symlink() or target.exists():
            target.unlink()
        if f.suffix == ".safetensors":
            target.symlink_to(f)
        else:
            shutil.copyfile(f, target)
    config = json.loads((src / "config.json").read_text())
    config.setdefault("auto_map", {})["AutoConfig"] = GOOD_AUTO_CONFIG
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    return dst


def resolve_model_path(model: str | None, mirror: Path = LOADABLE_2B) -> tuple[str | None, str | None]:
    """(path to load, note). A local checkpoint whose config needs the d1-2B loader fix is mirrored first."""
    if model and (Path(model) / "config.json").exists():
        config = json.loads((Path(model) / "config.json").read_text())
        if needs_loader_fix(config):
            fixed = make_loadable(Path(model), mirror)
            shown = fixed.relative_to(ROOT) if ROOT in fixed.parents else fixed
            return str(fixed), f"loader fix: AutoConfig -> {GOOD_AUTO_CONFIG} in a mirror at {shown}"
    return model, None


# ------------------------------------------------------------------------------------------------------- d1 runs
def run_d1(model_id: str, model: str | None, *, backend: str = "transformers", dtype: str = "auto",
           limit: int | None = None, warmup: int = WARMUP, lines: list[dict] | None = None,
           ranker=None, preds_dir: Path = PREDS_DIR, dataset: Path = DATASET_DIR) -> list[dict]:
    """Score every line with D1Ranker and time each decision. `ranker` replaces the real one (tests)."""
    lines = (lines if lines is not None else load_lines())[:limit]
    src = {r["id"]: r for r in load_source_rows([ln["line_id"] for ln in lines], dataset)}
    meta: dict = {"model_id": model_id, "model": model, "backend": backend, "dtype": dtype, "n": len(lines),
                  "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "warmup": warmup,
                  "python": platform.python_version()}
    if ranker is None:
        from clashbot.ranker import D1Ranker
        load_path, note = resolve_model_path(model)
        meta["loader_note"] = note
        meta["loaded_from"] = load_path
        ranker = D1Ranker(model_path=load_path, backend=backend, dtype=dtype)
        import torch
        import transformers
        meta.update({"torch": torch.__version__, "transformers": transformers.__version__,
                     "mps": bool(torch.backends.mps.is_available())})
    t0 = time.perf_counter()
    for ln in lines[:warmup]:                               # model load + warm-up calls, not timed
        ranker.probabilities(src[ln["line_id"]]["state"], src[ln["line_id"]]["questions"])
    meta["load_and_warmup_s"] = round(time.perf_counter() - t0, 2)
    rows = []
    for ln in lines:
        s = src[ln["line_id"]]
        t = time.perf_counter()
        try:
            probs = ranker.probabilities(s["state"], s["questions"])["play"]
            pred = argmax_first(probs)
            err = None
        except Exception as exc:                            # a failed decision counts as wrong
            probs, pred, err = None, None, f"{type(exc).__name__}: {exc}"[:200]
        ms = (time.perf_counter() - t) * 1000.0
        rows.append(pred_row(ln, pred, probs, ms, cost_usd=0.0, error=err,
                             raw=None if pred is None else ln["keys"][pred]))
    meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_preds(model_id, rows, meta, preds_dir)
    return rows


# --------------------------------------------------------------------------------------------------------- baselines
def placement_score(key: str, zone_freq: dict) -> float:
    """Share of the key's card placements that went to the key's zone (eval_stats.baseline_scores)."""
    card, _, zone = key.split("|")
    zf = zone_freq.get(card, {})
    total = sum(zf.values())
    return zf.get(zone, 0) / total if total else 0.0


def baseline_rows(lines: list[dict], pop: dict, zone_freq: dict, seed: int = 0) -> dict[str, list[dict]]:
    """Rows for random (one seeded draw per line; expected accuracy 0.25), popularity (train play counts of the key) and
    the placement prior. The latency of each row is the cost of that line's scoring call."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[dict]] = {"random": [], "popularity": [], "placement-prior": []}
    for ln in lines:
        keys = ln["keys"]
        t = time.perf_counter()
        draw = int(rng.integers(len(keys)))
        ms = (time.perf_counter() - t) * 1000.0
        out["random"].append(pred_row(ln, draw, [1 / len(keys)] * len(keys), ms, raw=keys[draw]))
        for name, fn in (("popularity", lambda k: float(pop.get(k, 0))),
                         ("placement-prior", lambda k: placement_score(k, zone_freq))):
            t = time.perf_counter()
            scores = [fn(k) for k in keys]
            pred = argmax_first(scores)
            ms = (time.perf_counter() - t) * 1000.0
            total = sum(scores)
            probs = [s / total for s in scores] if total > 0 else [1 / len(keys)] * len(keys)
            out[name].append(pred_row(ln, pred, probs, ms, raw=keys[pred]))
    return out


def lightgbm_rows(lines: list[dict], dump: dict, seed: int = 0, latency_ms: dict[str, float] | None = None) -> list[dict]:
    """LightGBM LambdaRank (seed `seed`) rows from feature_ranker_preds_v2.json, filtered to `lines`. The dump holds
    scores only: `latency_ms` ({line_id: ms}, from `time_lightgbm`) is the measured cost of the real scoring call; without
    it the rows' latency is null (not measured)."""
    by_id = {e["id"]: i for i, e in enumerate(dump["lines"]["test"])}
    scores = dump["models"]["lgbm_lambdarank"][str(seed)]["test"]
    rows = []
    for ln in lines:
        i = by_id[ln["line_id"]]
        e = dump["lines"]["test"][i]
        if e["keys"] != ln["keys"] or e["label"] != ln["gold"]:
            raise ValueError(f"LightGBM dump disagrees with lines.jsonl on {ln['line_id']}")
        s = scores[i]
        pred = argmax_first(s)
        ms = None if latency_ms is None else latency_ms.get(ln["line_id"])
        rows.append(pred_row(ln, pred, softmax(s), ms, raw=ln["keys"][pred]))
    return rows


def time_lightgbm(lines: list[dict], dataset: Path = DATASET_DIR, seed: int = 0) -> tuple[dict[str, float], dict[str, int]]:
    """Retrain the chosen LightGBM LambdaRank config (docs/data/feature_ranker_v1.json) with `seed`, exactly as
    feature_ranker.fit_lgbm does, then time the real scoring call per line: `build_features` of the line's state and 4
    options + `predict`. Returns ({line_id: ms}, {line_id: picked option}); the picks check the retrain against the dump."""
    import lightgbm as lgb
    from tools.eval import feature_ranker as fr
    best = json.loads((ROOT / "docs" / "data" / "feature_ranker_v1.json").read_text())["models"]["lgbm_lambdarank"]["best"]
    cfg = best["config"]
    records = fr.read_jsonl(dataset / "states.jsonl")
    by_id = {r["id"]: r for r in records}
    stats, kg = fr.train_stats(records), fr.load_kg()
    X, y, _ = fr.build_split(fr.read_jsonl(dataset / "ranker" / "train.jsonl"), by_id, stats, kg)
    n, k, f = X.shape
    labels = np.zeros((n, k), dtype=np.int32)
    labels[np.arange(n), y] = 1
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
                           num_leaves=cfg["num_leaves"], min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1)
    model.fit(X.reshape(n * k, f), labels.reshape(-1), group=[k] * n)
    test = {e["id"]: e for e in fr.read_jsonl(dataset / "ranker" / "test.jsonl")}
    warm = test[lines[0]["line_id"]]
    model.predict(fr.build_features(warm, by_id[warm["id"]], stats, kg))        # warm-up: card table, vocab, trees
    times: dict[str, float] = {}
    picks: dict[str, int] = {}
    for ln in lines:
        ex = test[ln["line_id"]]
        t = time.perf_counter()
        scores = model.predict(fr.build_features(ex, by_id[ex["id"]], stats, kg))
        pick = argmax_first(list(scores))
        times[ln["line_id"]] = (time.perf_counter() - t) * 1000.0
        picks[ln["line_id"]] = pick
    return times, picks


def run_baselines(lines: list[dict] | None = None, dataset: Path = DATASET_DIR, preds_dir: Path = PREDS_DIR,
                  measure_lightgbm: bool = True) -> dict:
    from tools.eval.eval_stats import train_key_counts
    lines = lines if lines is not None else load_lines()
    pop = dict(train_key_counts(dataset))
    zone_freq = json.loads((dataset / "placement_prior.json").read_text())["zone_freq"]
    rows = baseline_rows(lines, pop, zone_freq)
    notes = {"random": "one seeded draw (numpy default_rng(0)); expected accuracy 0.25",
             "popularity": "argmax of the key's train-split play count, first maximum on ties",
             "placement-prior": "argmax of the share of the card's placements in the key's zone, first maximum on ties"}
    if FEATURE_PREDS.exists():
        times, picks = time_lightgbm(lines, dataset) if measure_lightgbm else (None, None)
        rows["lightgbm"] = lightgbm_rows(lines, json.loads(FEATURE_PREDS.read_text()), latency_ms=times)
        notes["lightgbm"] = "LightGBM LambdaRank seed 0, predictions from docs/data/feature_ranker_preds_v2.json"
        if times is None:
            notes["lightgbm"] += "; latency not measured (stored scores)"
        else:
            agree = sum(picks[r["line_id"]] == r["pred"] for r in rows["lightgbm"])
            notes["lightgbm"] += ("; latency = build_features + predict per line of a model retrained with the same config "
                                  f"and seed (its picks match the dump on {agree}/{len(rows['lightgbm'])} lines)")
    for name, r in rows.items():
        write_preds(name, r, {"model_id": name, "n": len(r), "note": notes[name]}, preds_dir)
    return rows


# ----------------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lines")
    bl = sub.add_parser("baselines")
    bl.add_argument("--stored-lightgbm", action="store_true", help="do not retrain LightGBM to time its scoring call")
    d1 = sub.add_parser("d1")
    d1.add_argument("--model-id", required=True, help="preds file stem, e.g. d1-ft-joint-tac")
    d1.add_argument("--model", default=None, help="checkpoint dir; omitted = zero-shot LiquidAI/d1-omni-600M")
    d1.add_argument("--backend", choices=("mlx", "transformers"), default="transformers")
    d1.add_argument("--dtype", default="auto")
    d1.add_argument("--limit", type=int, default=None)
    d1.add_argument("--warmup", type=int, default=WARMUP)
    args = ap.parse_args(argv)
    if args.cmd == "lines":
        lines = write_lines()
        print(f"wrote {LINES_PATH} ({len(lines)} lines)")
    elif args.cmd == "baselines":
        for name, rows in run_baselines(measure_lightgbm=not args.stored_lightgbm).items():
            acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
            print(f"{name:16s} n={len(rows)} acc={acc:.3f}")
    else:
        rows = run_d1(args.model_id, args.model, backend=args.backend, dtype=args.dtype, limit=args.limit,
                      warmup=args.warmup)
        ms = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
        acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
        fails = sum(r["pred"] is None for r in rows)
        p = lambda q: ms[min(len(ms) - 1, math.floor(q * len(ms)))] if ms else float("nan")  # noqa: E731
        print(f"{args.model_id}: n={len(rows)} acc={acc:.4f} p50={p(0.5):.1f}ms p95={p(0.95):.1f}ms failures={fails}")


if __name__ == "__main__":
    main()
