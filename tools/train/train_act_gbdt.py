"""Train the fast play/wait model: LightGBM on structured features, one binary model per horizon (MAC-AB A1).

  python3 tools/train/train_act_gbdt.py [--data DIR] [--out DIR] [--horizons 1,2,3] [--seed 0]
         [--bal DIR] [--d1-json PATH] [--report PATH]
  python3 tools/train/train_act_gbdt.py --export-json DIR        # only (re)write DIR/model_h<h>.json from the .txt files

Lines come from the FULL dense data (`<data>/{train,val,test}.jsonl` joined by id with the state sidecars
`states_<split>.jsonl`; build_dense_act.py), features from `clashbot.act_model.act_features`. Label for horizon h:
the pro deploys a card within h seconds (`meta.next_deploy_in <= h`); h = 1 is the line's own `answers.act` (it
differs from the rounded meta value on a handful of lines, counted below) and the other horizons are nested around it.
Per horizon one LightGBM model (is_unbalance, early stopping on val logloss) and the threshold with the best balanced
accuracy on val (0.05..0.95, step 0.01, ties -> closest to 0.5). Reports, per split: AUC, balanced accuracy, act rate,
precision / recall of act and wait for the model, the rules (`ReactivePolicy.decide(...).act`) and `fast` (rules'
none / defend / leak + the model, the `FastActPolicy` logic), plus the reaction slice (since_opp_play <= 3 s). At h = 1
it repeats the metrics on the ids of `<bal>/{val,test}.jsonl` next to the d1-head numbers in `--d1-json`. Writes
<out>/model_h<h>.txt, <out>/model_h<h>.json (the same trees as `Booster.dump_model()` JSON: the live loop reads these
with a numpy tree walker, never lightgbm, because torch + lightgbm in one process segfault) and <out>/meta.json
(features, thresholds, metrics, default_horizon 2) and <report> (JSON, default <data>/../eval/act_gbdt.json) + .md;
prints the tables, the top-15 gain importance and the single-row predict latency. `--export-json DIR` writes the
.json files of existing .txt models (also what `ActGBDT.load` runs in a subprocess when a .json is missing).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from clashbot.act_model import (DEFAULT_HORIZON, FEATURES, JSON_FILE, META_FILE, MODEL_FILE,  # noqa: E402
                                ActGBDT, act_features)
from clashbot.config import DATASET_DIR, DB_DIR  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.policy import ReactivePolicy  # noqa: E402
from tools.eval.eval_act_head import REACTION_S, metrics, pick_threshold  # noqa: E402

SPLITS = ("train", "val", "test")
DENSE_HORIZON = 1.0           # horizon of the dense labels (`answers.act`)
POLICIES = ("gbdt", "rules", "fast", "always", "never")
MAX_ROUNDS = 2000
EARLY_STOP = 50
TOP_N = 15
LATENCY_ROWS = 500
MAX_META_MISMATCH = 0.002     # fraction of lines where answers.act differs from the rounded meta.next_deploy_in <= 1
PARAMS = {"objective": "binary", "metric": "binary_logloss", "is_unbalance": True, "num_leaves": 31,
          "learning_rate": 0.05, "min_data_in_leaf": 50, "n_jobs": 1,        # libomp next to torch: one thread
          "deterministic": True, "force_row_wise": True, "verbose": -1}


# --------------------------------------------------------------------------- data

def _rel(path: Path) -> str:
    """`path` relative to the repo root when it lies under it (reports stay free of machine paths)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def meta_label(line: dict, h: float) -> bool:
    """The pro deploys within h seconds, from `meta.next_deploy_in` alone."""
    nd = (line.get("meta") or {}).get("next_deploy_in")
    return nd is not None and nd <= h


def act_label(line: dict, h: float) -> bool:
    """Label of `line` for horizon `h`. h = 1 is `answers.act` itself; longer horizons also accept it (so
    act_1 -> act_2 -> act_3 hold even where the rounded meta value disagrees), shorter ones also require it."""
    act1 = line["answers"]["act"] == "true"
    if h == DENSE_HORIZON:
        return act1
    return (meta_label(line, h) or act1) if h > DENSE_HORIZON else (meta_label(line, h) and act1)


def load_split(data: Path, split: str, cards: dict) -> dict:
    """Lines, sidecar states (same order, joined by id) and the feature matrix of one split."""
    lines = read_jsonl(data / f"{split}.jsonl")
    side = {r["id"]: r for r in read_jsonl(data / f"states_{split}.jsonl")}
    missing = [l["id"] for l in lines if l["id"] not in side]
    if missing:
        raise SystemExit(f"{data / f'states_{split}.jsonl'} has no state for {len(missing)} lines, e.g. {missing[0]}")
    states = [side[l["id"]] for l in lines]
    X = np.array([act_features(s["state"], s["t"], cards) for s in states], dtype=np.float64).reshape(-1, len(FEATURES))
    return {"lines": lines, "states": states, "X": X, "ids": [l["id"] for l in lines]}


def label_check(lines: list[dict]) -> dict:
    """How often `answers.act` and the rounded `meta.next_deploy_in <= 1` disagree (rounding at the 1.0 edge)."""
    bad = sum(1 for l in lines if meta_label(l, DENSE_HORIZON) != (l["answers"]["act"] == "true"))
    return {"n": len(lines), "meta_mismatch": bad}


# --------------------------------------------------------------------------- metrics

def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels.astype(bool)].sum() - pos * (pos + 1) / 2) / (pos * neg))


def rule_decisions(states: list[dict], rules: ReactivePolicy) -> list:
    return [rules.decide(s["state"], s["t"]) for s in states]


def fast_predictions(decisions: list, probs: np.ndarray, threshold: float) -> list[bool]:
    """`FastActPolicy.decide(...).act` from precomputed rule decisions and probabilities."""
    out: list[bool] = []
    for r, p in zip(decisions, probs):
        if r.mode == "none":
            out.append(False)
        elif r.act and r.mode in ("defend", "counter", "leak"):
            out.append(True)
        else:
            out.append(bool(p >= threshold))
    return out


def predictions(decisions: list, probs: np.ndarray, threshold: float) -> dict[str, list[bool]]:
    n = len(probs)
    return {"gbdt": [bool(p >= threshold) for p in probs], "rules": [bool(r.act) for r in decisions],
            "fast": fast_predictions(decisions, probs, threshold), "always": [True] * n, "never": [False] * n}


def split_block(lines: list[dict], labels: np.ndarray, probs: np.ndarray, decisions: list, threshold: float) -> dict:
    """AUC, the policy table and the reaction slice of one split (or of a subset of it)."""
    y = [bool(v) for v in labels]
    preds = predictions(decisions, probs, threshold)
    idx = [i for i, l in enumerate(lines)
           if (l.get("meta") or {}).get("since_opp_play") is not None and l["meta"]["since_opp_play"] <= REACTION_S]
    r_y = [y[i] for i in idx]
    keep = ("n", "act_rate", "pro_act_rate", "act_recall", "act_precision")
    return {"n": len(y), "pro_act_rate": float(np.mean(y)) if y else None,
            "auc": auc(labels.astype(int), probs),
            "policies": {name: metrics(y, preds[name]) for name in POLICIES},
            "reaction": {"n": len(idx), "pro_act_rate": float(np.mean(r_y)) if r_y else None,
                         "policies": {name: {k: v for k, v in metrics(r_y, [preds[name][i] for i in idx]).items()
                                             if k in keep} for name in POLICIES}}}


def subset(lines: list, labels: np.ndarray, probs: np.ndarray, decisions: list, idx: list[int]):
    return ([lines[i] for i in idx], labels[idx], probs[idx], [decisions[i] for i in idx])


# --------------------------------------------------------------------------- training

def fit(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, seed: int,
        max_rounds: int = MAX_ROUNDS, early_stop: int = EARLY_STOP):
    import lightgbm as lgb
    params = {**PARAMS, "seed": seed}
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=list(FEATURES), free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(early_stop, verbose=False)])


def predict(booster, X: np.ndarray) -> np.ndarray:
    return np.asarray(booster.predict(X, num_iteration=booster.best_iteration or None, num_threads=1))


def export_json(txt: Path) -> Path:
    """Write the trees of the LightGBM text model `txt` (model_h<h>.txt) as `Booster.dump_model()` JSON next to it
    (model_h<h>.json; atomic, so a running live loop never reads half a file). Returns the .json path."""
    import lightgbm as lgb
    dump = lgb.Booster(model_file=str(txt)).dump_model()
    out = txt.with_suffix(".json")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(dump, separators=(",", ":")), encoding="utf-8")
    tmp.replace(out)
    return out


def export_dir(directory: Path) -> list[Path]:
    """`export_json` for every model_h<h>.txt of `directory`."""
    txts = sorted(directory.glob("model_h*.txt"))
    if not txts:
        raise SystemExit(f"no model_h*.txt in {directory}")
    return [export_json(t) for t in txts]


def importance(booster, top: int = TOP_N) -> list[dict]:
    gain = booster.feature_importance(importance_type="gain")
    total = float(gain.sum()) or 1.0
    order = np.argsort(-gain, kind="stable")[:top]
    return [{"feature": FEATURES[i], "gain": float(gain[i]), "share": float(gain[i] / total)} for i in order]


def latency_ms(out: Path, states: list[dict], cards: dict) -> dict:
    """Single-row timings in ms of the saved default-horizon model: features, tree-walk predict, `p_act` end to end."""
    model = ActGBDT.load(out)
    rows = states[:LATENCY_ROWS]
    if not rows:
        return {}
    feats = [act_features(s["state"], s["t"], cards) for s in rows]
    for s, x in zip(rows[:20], feats[:20]):                               # warm up
        model.p_act(s["state"], s["t"])
        model.trees.predict_row(x)

    def timed(fn) -> dict:
        ts = []
        for i in range(len(rows)):
            a = time.perf_counter()
            fn(i)
            ts.append((time.perf_counter() - a) * 1000.0)
        arr = np.array(ts)
        return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
                "max": float(arr.max())}

    return {"horizon": model.horizon, "rows": len(rows),
            "features": timed(lambda i: act_features(rows[i]["state"], rows[i]["t"], cards)),
            "predict": timed(lambda i: model.trees.predict_row(feats[i])),
            "p_act": timed(lambda i: model.p_act(rows[i]["state"], rows[i]["t"]))}


# --------------------------------------------------------------------------- report

def _f(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _policy_rows(block: dict, names=POLICIES) -> list[str]:
    out = ["| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |",
           "|---|---|---|---|---|---|---|---|"]
    for name in names:
        m = block["policies"][name]
        out.append(f"| {name} | {_f(m['accuracy'])} | {_f(m['balanced_accuracy'])} | {_f(m['act_rate'])} | "
                   f"{_f(m['act_precision'])} | {_f(m['act_recall'])} | {_f(m['wait_precision'])} | "
                   f"{_f(m['wait_recall'])} |")
    return out


def _reaction_rows(block: dict, names=POLICIES) -> list[str]:
    r = block["reaction"]
    out = [f"reaction slice (since_opp_play <= {REACTION_S:g} s): n={r['n']}, pro act rate {_f(r['pro_act_rate'])}", "",
           "| policy | act rate | act recall | act prec |", "|---|---|---|---|"]
    for name in names:
        m = r["policies"][name]
        out.append(f"| {name} | {_f(m['act_rate'])} | {_f(m['act_recall'])} | {_f(m['act_precision'])} |")
    return out


def format_markdown(res: dict) -> str:
    out = ["# Fast act model (LightGBM on structured features, MAC-AB A1)", "",
           f"data `{res['data']}`, features {len(res['features'])}, seed {res['seed']}, "
           f"default horizon {res['default_horizon']:g} s; lines " +
           ", ".join(f"{s} {n}" for s, n in res["lines"].items()) +
           "; `answers.act` vs rounded `meta.next_deploy_in <= 1` disagree on " +
           ", ".join(f"{s} {c['meta_mismatch']}" for s, c in res["label_check"].items()) + " lines", ""]
    out += ["## Overview", "", "| horizon | threshold | rounds | val pro act | val AUC | val bacc | test pro act | test AUC | "
            "test bacc | test rules bacc | test fast bacc |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, hres in res["horizons"].items():
        v, t = hres["splits"]["val"], hres["splits"]["test"]
        out.append(f"| {key} s | {_f(hres['threshold'], 2)} | {hres['best_iteration']} | {_f(v['pro_act_rate'])} | "
                   f"{_f(v['auc'])} | {_f(v['policies']['gbdt']['balanced_accuracy'])} | {_f(t['pro_act_rate'])} | "
                   f"{_f(t['auc'])} | {_f(t['policies']['gbdt']['balanced_accuracy'])} | "
                   f"{_f(t['policies']['rules']['balanced_accuracy'])} | "
                   f"{_f(t['policies']['fast']['balanced_accuracy'])} |")
    out.append("")
    for key, hres in res["horizons"].items():
        out += [f"## Horizon {key} s  (threshold {_f(hres['threshold'], 2)}, best round {hres['best_iteration']})", ""]
        for split in ("val", "test"):
            b = hres["splits"][split]
            out += [f"### {split}  (n={b['n']}, pro act rate {_f(b['pro_act_rate'])}, AUC {_f(b['auc'])})", ""]
            out += _policy_rows(b) + [""] + _reaction_rows(b) + [""]
        out += [f"### Feature importance (gain, top {len(hres['importance'])})", "",
                "| feature | gain | share |", "|---|---|---|"]
        out += [f"| {r['feature']} | {r['gain']:.1f} | {r['share']:.3f} |" for r in hres["importance"]] + [""]
    if res.get("bal"):
        out += _bal_markdown(res["bal"])
    lat = res.get("latency_ms")
    if lat:
        out += [f"## Predict latency (horizon {lat['horizon']:g} s, {lat['rows']} val rows, single row, 1 thread)", "",
                "| step | mean ms | p50 ms | p95 ms | max ms |", "|---|---|---|---|---|"]
        for step in ("features", "predict", "p_act"):
            m = lat[step]
            out.append(f"| {step} | {m['mean']:.3f} | {m['p50']:.3f} | {m['p95']:.3f} | {m['max']:.3f} |")
        out += ["", f"p_act mean < 1 ms: {lat['p_act']['mean'] < 1.0}", ""]
    return "\n".join(out)


def _bal_markdown(bal: dict) -> list[str]:
    out = ["## Apples to apples with the d1 head (horizon 1 s, ids of `act_dense_bal`)", ""]
    if bal.get("d1_json"):
        out += [f"d1 numbers from `{bal['d1_json']}` (threshold {bal['d1_threshold']}); `gbdt` is the h=1 model at the "
                f"threshold chosen on the full val ({_f(bal['threshold_full'], 2)}), `gbdt@bal` at the threshold chosen "
                f"on the balanced val ({_f(bal['threshold_bal'], 2)}), `fast` = rules' none / defend / leak + gbdt.", ""]
    for split, block in bal["splits"].items():
        out += [f"### {split}  (n={block['n']}, pro act rate {_f(block['pro_act_rate'])}, gbdt AUC {_f(block['auc'])})",
                "", "| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |",
                "|---|---|---|---|---|---|---|---|"]
        for name, m in block["rows"].items():
            out.append(f"| {name} | {_f(m['accuracy'])} | {_f(m['balanced_accuracy'])} | {_f(m['act_rate'])} | "
                       f"{_f(m['act_precision'])} | {_f(m['act_recall'])} | {_f(m['wait_precision'])} | "
                       f"{_f(m['wait_recall'])} |")
        r = block["reaction"]
        out += ["", f"reaction slice (since_opp_play <= {REACTION_S:g} s): n={r['n']}", "",
                "| policy | act rate | act recall | act prec |", "|---|---|---|---|"]
        for name, m in r["rows"].items():
            out.append(f"| {name} | {_f(m['act_rate'])} | {_f(m['act_recall'])} | {_f(m['act_precision'])} |")
        out.append("")
    return out


# --------------------------------------------------------------------------- bal comparison

def bal_comparison(bal_dir: Path, data: dict, labels: dict, probs: dict, decisions: dict, thr_full: float,
                   d1_json: Path | None) -> dict | None:
    """h = 1 metrics on the ids of <bal>/{val,test}.jsonl, next to the d1-head rows of `d1_json`."""
    if not all((bal_dir / f"{s}.jsonl").exists() for s in ("val", "test")):
        return None
    d1 = json.loads(d1_json.read_text(encoding="utf-8")) if d1_json and d1_json.exists() else None
    idx: dict[str, list[int]] = {}
    for split in ("val", "test"):
        wanted = {l["id"] for l in read_jsonl(bal_dir / f"{split}.jsonl")}
        pos = {i: k for k, i in enumerate(data[split]["ids"])}
        idx[split] = sorted(pos[i] for i in wanted if i in pos)
        if len(idx[split]) != len(wanted):
            print(f"warning: {len(wanted) - len(idx[split])} ids of {bal_dir.name}/{split} not in the dense data",
                  file=sys.stderr)
    sub = {s: subset(data[s]["lines"], labels[s], probs[s], decisions[s], idx[s]) for s in idx}
    thr_bal, _ = pick_threshold([bool(v) for v in sub["val"][1]], list(sub["val"][2]))
    out: dict = {"data": _rel(bal_dir), "threshold_full": thr_full, "threshold_bal": thr_bal,
                 "d1_json": _rel(d1_json) if d1 else None, "d1_threshold": d1["threshold"] if d1 else None,
                 "splits": {}}
    keep = ("n", "act_rate", "pro_act_rate", "act_recall", "act_precision")
    for split, (lines, y, p, dec) in sub.items():
        b_full = split_block(lines, y, p, dec, thr_full)
        b_bal = split_block(lines, y, p, dec, thr_bal)
        rows: dict[str, dict] = {}
        reaction: dict[str, dict] = {}
        if d1:
            for name in ("learned", "rules", "combined"):
                rows[f"d1 {name}"] = d1["splits"][split]["policies"][name]
                reaction[f"d1 {name}"] = {k: v for k, v in d1["splits"][split]["reaction"]["policies"][name].items()
                                          if k in keep}
        for name, label in (("rules", "rules (recomputed)"), ("gbdt", "gbdt"), ("fast", "fast (rules + gbdt)")):
            rows[label] = b_full["policies"][name]
            reaction[label] = b_full["reaction"]["policies"][name]
        rows["gbdt@bal"] = b_bal["policies"]["gbdt"]
        reaction["gbdt@bal"] = b_bal["reaction"]["policies"]["gbdt"]
        for name in ("always", "never"):
            rows[name] = b_full["policies"][name]
            reaction[name] = b_full["reaction"]["policies"][name]
        out["splits"][split] = {"n": b_full["n"], "pro_act_rate": b_full["pro_act_rate"], "auc": b_full["auc"],
                                "rows": rows, "reaction": {**b_full["reaction"], "rows": reaction}}
    return out


# --------------------------------------------------------------------------- main

def run(data_dir: Path, out: Path, horizons: list[float], seed: int, bal_dir: Path | None, d1_json: Path | None,
        report: Path, max_rounds: int = MAX_ROUNDS, early_stop: int = EARLY_STOP, quiet: bool = False) -> dict:
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, file=sys.stderr))
    cards = load_cards()
    rules = ReactivePolicy(cards)
    t0 = time.perf_counter()
    data = {s: load_split(data_dir, s, cards) for s in SPLITS}
    log(f"loaded {', '.join(f'{s} {len(data[s]['lines'])}' for s in SPLITS)} lines, {len(FEATURES)} features "
        f"({time.perf_counter() - t0:.1f} s)")
    checks = {s: label_check(data[s]["lines"]) for s in SPLITS}
    total = sum(c["n"] for c in checks.values())
    bad = sum(c["meta_mismatch"] for c in checks.values())
    if total and bad / total > MAX_META_MISMATCH:
        raise SystemExit(f"answers.act disagrees with meta.next_deploy_in <= 1 on {bad}/{total} lines: "
                         f"the label semantics changed (build_dense_act.py)")
    decisions = {s: rule_decisions(data[s]["states"], rules) for s in ("val", "test")}

    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"data": _rel(data_dir), "seed": seed, "params": {**PARAMS, "max_rounds": max_rounds,
                 "early_stopping_rounds": early_stop}, "features": list(FEATURES), "default_horizon": DEFAULT_HORIZON,
                 "lines": {s: len(data[s]["lines"]) for s in SPLITS}, "label_check": checks, "horizons": {},
                 "bal": None}
    labels_by_h: dict[float, dict[str, np.ndarray]] = {}
    probs_by_h: dict[float, dict[str, np.ndarray]] = {}
    thr_by_h: dict[float, float] = {}
    prev: dict[str, np.ndarray] | None = None
    for h in sorted(horizons):
        labels = {s: np.array([act_label(l, h) for l in data[s]["lines"]], dtype=np.int32) for s in SPLITS}
        if prev is not None:
            assert all((prev[s] <= labels[s]).all() for s in SPLITS), "horizons must be nested"
        prev = labels
        if h == DENSE_HORIZON:
            assert all(((labels[s] == 1) == np.array([l["answers"]["act"] == "true" for l in data[s]["lines"]])).all()
                       for s in SPLITS)
        t1 = time.perf_counter()
        booster = fit(data["train"]["X"], labels["train"], data["val"]["X"], labels["val"], seed, max_rounds,
                      early_stop)
        probs = {s: predict(booster, data[s]["X"]) for s in ("val", "test")}
        thr, val_bacc = pick_threshold([bool(v) for v in labels["val"]], list(probs["val"]))
        booster.save_model(str(out / MODEL_FILE.format(h=h)))
        export_json(out / MODEL_FILE.format(h=h))
        key = f"{h:g}"
        log(f"h={key}: {booster.best_iteration} rounds, threshold {thr:.2f} (val bacc {val_bacc:.3f}), "
            f"{time.perf_counter() - t1:.1f} s")
        res["horizons"][key] = {
            "horizon": h, "threshold": thr, "val_balanced_acc": val_bacc, "best_iteration": int(booster.best_iteration),
            "pro_act_rate": {s: float(labels[s].mean()) for s in SPLITS},
            "splits": {s: split_block(data[s]["lines"], labels[s], probs[s], decisions[s], thr)
                       for s in ("val", "test")},
            "importance": importance(booster)}
        labels_by_h[h], probs_by_h[h], thr_by_h[h] = labels, probs, thr

    if bal_dir is not None and DENSE_HORIZON in thr_by_h:
        res["bal"] = bal_comparison(bal_dir, data, labels_by_h[DENSE_HORIZON], probs_by_h[DENSE_HORIZON], decisions,
                                    thr_by_h[DENSE_HORIZON], d1_json)
    default = DEFAULT_HORIZON if DEFAULT_HORIZON in thr_by_h else sorted(thr_by_h)[0]
    meta = {"features": list(FEATURES), "horizons": sorted(horizons), "default_horizon": default,
            "thresholds": {f"{h:g}": thr_by_h[h] for h in sorted(horizons)}, "seed": seed,
            "params": res["params"], "data": _rel(data_dir),
            "model_files": {f"{h:g}": MODEL_FILE.format(h=h) for h in sorted(horizons)},
            "json_files": {f"{h:g}": JSON_FILE.format(h=h) for h in sorted(horizons)},
            "metrics": {key: {"threshold": v["threshold"], "best_iteration": v["best_iteration"],
                              "pro_act_rate": v["pro_act_rate"],
                              **{s: {"auc": v["splits"][s]["auc"], **{k: v["splits"][s]["policies"]["gbdt"][k]
                                     for k in ("balanced_accuracy", "act_rate", "act_precision", "act_recall",
                                               "wait_precision", "wait_recall")}} for s in ("val", "test")}}
                        for key, v in res["horizons"].items()}}
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    res["latency_ms"] = latency_ms(out, data["val"]["states"], cards)

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    md = format_markdown(res)
    report.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    if not quiet:
        print(md)
    return res


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATASET_DIR / "act_dense", help="full dense data (with state sidecars)")
    ap.add_argument("--out", type=Path, default=DB_DIR / "models" / "act-gbdt")
    ap.add_argument("--horizons", default="1,2,3", help="comma-separated seconds; 1 is the dense label itself")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bal", type=Path, default=None, help="balanced subsets for the d1 comparison "
                                                           "(default <data>_bal)")
    ap.add_argument("--d1-json", type=Path, default=ROOT / "docs" / "data" / "act_head_v1.json")
    ap.add_argument("--report", type=Path, default=None, help="JSON path (default <data>/../eval/act_gbdt.json)")
    ap.add_argument("--export-json", type=Path, default=None, metavar="DIR",
                    help="only write model_h<h>.json next to the model_h<h>.txt files of DIR, then exit")
    a = ap.parse_args(argv)
    if a.export_json is not None:
        for path in export_dir(a.export_json):
            print(f"wrote {path}", file=sys.stderr)
        return {}
    try:
        horizons = sorted({float(x) for x in a.horizons.split(",") if x.strip()})
    except ValueError:
        ap.error(f"bad --horizons {a.horizons!r}")
    if not horizons or horizons[0] <= 0:
        ap.error("--horizons needs positive seconds")
    bal = a.bal if a.bal is not None else a.data.parent / f"{a.data.name}_bal"
    report = a.report or (a.data.parent / "eval" / "act_gbdt.json")
    return run(a.data, a.out, horizons, a.seed, bal if bal.exists() else None, a.d1_json, report)


if __name__ == "__main__":
    main()
