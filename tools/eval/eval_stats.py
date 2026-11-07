"""E1 statistics audit of the 4-way multiple-choice table (MAC-AP, docs/research/course_gap_analysis.md section 7.3).

  python3 tools/eval/eval_stats.py [--dataset DIR] [--docs-data DIR] [--out docs/data/stats_audit_v1.json]
          [--boot 10000] [--seed N]

Reads the saved per-line outputs, never a model: the d1 eval JSONs (`per_example`, 4 probabilities per line), the
feature rankers' per-line scores (`feature_ranker.py --dump-preds`), and recomputes the three baselines (random,
popularity, placement prior) from `dataset/v1`. Everything is scored on `ranker/{val,test}.jsonl` (761 / 766 lines,
option order identical to the d1 `joint` and `joint_tac` files). Thresholds and the ensemble weight are chosen on val only.

Reports (val and test): accuracy with a 95% cluster-bootstrap CI over `game_id`; paired differences (CI, exact McNemar,
cluster-bootstrap p); calibration of d1-tac (ECE 15 bins, Brier, reliability); risk-coverage and the +1/0/-1 utility;
accuracy by popularity tercile of the gold key; the d1-tac + LightGBM ensemble. Writes <out>, <out>.md and
<out stem>_risk_coverage.csv. numpy only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.config import DATASET_DIR  # noqa: E402

SPLITS = ("val", "test")
N_BOOT = 10_000
SEED = 20261107
ECE_BINS = 15
WEIGHT_GRID = [round(float(w), 2) for w in np.linspace(0.0, 1.0, 21)]   # weight of d1 in the ensemble
RANKER_MODELS = (("logreg", "logreg"), ("lgbm", "lgbm_lambdarank"), ("dcn_v2", "dcn_v2"))   # (audit name, dump name)

# (audit name, label, file in docs/data, extra row?)
D1_SOURCES = (
    ("d1_zero_shot", "d1 zero-shot", "d1_zero_shot_v1.json", False),
    ("d1_joint", "d1-ft-joint (plain lines)", "d1_finetune_joint_eval_v1.json", False),
    ("d1_tac", "d1-ft-joint-tac (HEAD lines)", "d1_finetune_joint_tac_eval_v2.json", False),
    ("d1_kg", "d1-ft-joint-kg (HEAD lines)", "d1_joint_kg_on_tac_v2.json", False),
    ("d1_joint_on_tac", "d1-ft-joint on HEAD tactics lines", "d1_joint_on_tac_v2.json", True),
    ("d1_tac_v1", "d1-ft-joint-tac (as-trained lines)", "d1_finetune_joint_tac_eval_v1.json", True),
)
D1_FALLBACK = {"d1_kg": "d1_finetune_joint_kg_eval_v1.json"}
# (a, b, why); delta = acc(a) - acc(b). The first four are the spec's; the rest are context.
PAIRS = (
    ("d1_tac", "lgbm", "spec"), ("d1_tac", "d1_joint", "spec"), ("d1_tac", "placement_prior", "spec"),
    ("d1_kg", "d1_tac", "spec"),
    ("d1_tac", "d1_joint_on_tac", "extra"), ("d1_tac", "d1_tac_v1", "extra"), ("d1_tac", "logreg", "extra"),
    ("d1_tac", "dcn_v2", "extra"), ("d1_tac", "popularity", "extra"), ("d1_joint", "lgbm", "extra"),
    ("d1_tac", "lgbm_s1", "robustness"), ("d1_tac", "lgbm_s2", "robustness"),
)
MAIN_ORDER = ("random", "popularity", "placement_prior", "logreg", "lgbm", "dcn_v2", "d1_zero_shot", "d1_joint",
              "d1_tac", "d1_kg")


# ----------------------------------------------------------------------------------------------------------------
# statistics (pure functions)
# ----------------------------------------------------------------------------------------------------------------

def argmax_hits(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1.0 where the first maximal option is the gold option."""
    return (np.argmax(scores, axis=1) == labels).astype(float)


def make_ctx(game_ids, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Cluster-bootstrap context: one (n_boot, n_games) matrix of resampled game indices, shared by every statistic of a
    split so that differences and ratios are resampled on the same games."""
    _, inv = np.unique(np.asarray(game_ids), return_inverse=True)
    g = int(inv.max()) + 1
    idx = np.random.default_rng(seed).integers(0, g, size=(n_boot, g))
    return {"inv": inv, "G": g, "n_g": np.bincount(inv, minlength=g).astype(float), "idx": idx}


def ratio_boot(ctx: dict, num, den=None) -> tuple[float, np.ndarray]:
    """(estimate, bootstrap replicates) of sum(num) / sum(den) with whole games resampled; den defaults to the line
    count. Replicates whose resample holds no denominator mass are dropped."""
    num_g = np.bincount(ctx["inv"], weights=np.asarray(num, float), minlength=ctx["G"])
    den_g = ctx["n_g"] if den is None else np.bincount(ctx["inv"], weights=np.asarray(den, float), minlength=ctx["G"])
    total = den_g.sum()
    est = float(num_g.sum() / total) if total > 0 else float("nan")
    nb, db = num_g[ctx["idx"]].sum(axis=1), den_g[ctx["idx"]].sum(axis=1)
    ok = db > 0
    return est, nb[ok] / db[ok]


def ci_of(est: float, boot: np.ndarray, alpha: float = 0.05) -> dict:
    if len(boot) == 0:
        return {"est": est, "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"est": est, "lo": float(lo), "hi": float(hi)}


def cluster_bootstrap_ci(values, game_ids, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> dict:
    """Mean of per-line `values` with a percentile CI from resampling whole `game_ids`."""
    est, boot = ratio_boot(make_ctx(game_ids, n_boot, seed), values)
    return ci_of(est, boot, alpha)


def mcnemar_exact(hit_a, hit_b) -> dict:
    """Exact two-sided McNemar test on paired 0/1 outcomes: under H0 the discordant pairs split Binomial(n, 1/2)."""
    a, b = np.asarray(hit_a) > 0.5, np.asarray(hit_b) > 0.5
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1))
    return {"a_only": a_only, "b_only": b_only, "p": float(min(1.0, 2 * tail / 2 ** n))}


def paired_diff(ctx: dict, hit_a, hit_b) -> dict:
    """acc(a) - acc(b) with a cluster-bootstrap CI and p, and the exact McNemar test (which treats lines as independent)."""
    est, boot = ratio_boot(ctx, np.asarray(hit_a, float) - np.asarray(hit_b, float))
    out = ci_of(est, boot)
    p_boot = 1.0 if len(boot) == 0 else min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
    out.update({"p_boot": float(max(p_boot, 2.0 / (len(boot) + 1))), **{f"mcnemar_{k}": v for k, v in mcnemar_exact(hit_a, hit_b).items()}})
    return out


def calibration_bins(conf: np.ndarray, bins: int = ECE_BINS) -> np.ndarray:
    return np.minimum((np.asarray(conf) * bins).astype(int), bins - 1)


def expected_calibration_error(conf, hit, bins: int = ECE_BINS) -> tuple[float, list[dict]]:
    """ECE over `bins` equal-width confidence bins on [0, 1]: sum_k n_k/N |acc_k - conf_k|. Returns (ece, reliability rows)."""
    conf, hit = np.asarray(conf, float), np.asarray(hit, float)
    b, rows, ece = calibration_bins(conf, bins), [], 0.0
    for k in range(bins):
        m = b == k
        n = int(m.sum())
        row = {"bin": k, "lo": k / bins, "hi": (k + 1) / bins, "n": n}
        if n:
            row.update({"conf": float(conf[m].mean()), "acc": float(hit[m].mean())})
            row["gap"] = row["acc"] - row["conf"]
            ece += n / len(conf) * abs(row["gap"])
        rows.append(row)
    return float(ece), rows


def ece_ci(ctx: dict, conf, hit, bins: int = ECE_BINS) -> dict:
    conf, hit = np.asarray(conf, float), np.asarray(hit, float)
    b = calibration_bins(conf, bins)
    n_gb, c_gb, h_gb = (np.zeros((ctx["G"], bins)) for _ in range(3))
    np.add.at(n_gb, (ctx["inv"], b), 1.0)
    np.add.at(c_gb, (ctx["inv"], b), conf)
    np.add.at(h_gb, (ctx["inv"], b), hit)
    idx = ctx["idx"]
    boot = np.abs(h_gb[idx].sum(axis=1) - c_gb[idx].sum(axis=1)).sum(axis=1) / n_gb[idx].sum(axis=(1, 2))
    return ci_of(float(np.abs(h_gb.sum(0) - c_gb.sum(0)).sum() / len(conf)), boot)


def brier_per_line(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Multi-class Brier score per line: sum over options of (p - onehot)^2, in [0, 2]."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return ((probs - onehot) ** 2).sum(axis=1)


def risk_coverage(conf, hit) -> dict:
    """Answer a line when conf >= threshold. One point per distinct confidence (ties answered together): coverage,
    accuracy on the answered lines, risk = 1 - accuracy, utility = (right - wrong) / N with abstention = 0 (CRAG).
    `aurc` is the mean risk over the prefixes of the confidence ranking."""
    conf, hit = np.asarray(conf, float), np.asarray(hit, float)
    order = np.argsort(-conf, kind="stable")
    c, h = conf[order], hit[order]
    k = np.arange(1, len(c) + 1)
    cum = np.cumsum(h)
    last = np.r_[c[1:] != c[:-1], True]
    n_ans, correct = k[last], cum[last]
    return {"threshold": c[last], "n_answered": n_ans, "coverage": n_ans / len(c), "accuracy": correct / n_ans,
            "risk": 1 - correct / n_ans, "utility": (2 * correct - n_ans) / len(c),
            "aurc": float(np.mean(1 - cum / k))}


def choose_threshold(conf, hit) -> float:
    """The confidence threshold with the best +1/0/-1 utility on these lines (val). Ties go to the lowest threshold (most
    coverage); +inf (abstain always) only when every threshold has a negative utility."""
    rc = risk_coverage(conf, hit)
    best = float(rc["utility"].max())
    if best < 0:
        return float("inf")
    return float(rc["threshold"][np.flatnonzero(rc["utility"] >= best - 1e-12)[-1]])


def zscore_rows(x) -> np.ndarray:
    """z-score each line across its options; a line whose options are all equal becomes all zeros."""
    x = np.asarray(x, float)
    sd = x.std(axis=1, keepdims=True)
    return np.where(sd > 1e-12, (x - x.mean(axis=1, keepdims=True)) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def pick_ensemble_weight(z_a, z_b, labels, grid=WEIGHT_GRID) -> tuple[float, list[float]]:
    """Weight w of member a in w*z_a + (1-w)*z_b with the best accuracy on the given (val) lines; ties go to the weight
    closest to an even blend. Sees only the lines it is given."""
    accs = [float(np.mean(np.argmax(w * z_a + (1 - w) * z_b, axis=1) == labels)) for w in grid]
    best = max(accs)
    tied = [w for w, a in zip(grid, accs) if a >= best - 1e-12]
    return float(min(tied, key=lambda w: (abs(w - 0.5), w))), accs


def tercile_edges(counts) -> tuple[float, float]:
    lo, hi = np.quantile(np.asarray(counts, float), [1 / 3, 2 / 3], method="lower")
    return float(lo), float(hi)


def tercile_of(counts, edges: tuple[float, float]) -> np.ndarray:
    c = np.asarray(counts, float)
    return np.where(c <= edges[0], 0, np.where(c <= edges[1], 1, 2))


# ----------------------------------------------------------------------------------------------------------------
# the audit
# ----------------------------------------------------------------------------------------------------------------

def _log_probs(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-12, 1.0))


def audit(lines: dict, models: dict, gold_counts: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """`lines[split]` = {"game_ids", "labels"}; `models[name]` = {"label", "kind" ("prob" | "score"), "extra",
    "scores": {split: (n, 4)}}; `gold_counts[split]` = train count of each line's gold key. Models named d1_tac / lgbm /
    d1_joint / d1_kg / placement_prior feed the pairs, calibration, selective and ensemble blocks when present."""
    ctx = {sp: make_ctx(lines[sp]["game_ids"], n_boot, seed + i) for i, sp in enumerate(SPLITS)}
    labels = {sp: np.asarray(lines[sp]["labels"]) for sp in SPLITS}
    hits = {m: {sp: argmax_hits(v["scores"][sp], labels[sp]) for sp in SPLITS} for m, v in models.items()}
    out: dict = {"n_boot": n_boot, "seed": seed,
                 "n_lines": {sp: len(labels[sp]) for sp in SPLITS},
                 "n_games": {sp: ctx[sp]["G"] for sp in SPLITS}}

    out["accuracy"] = {sp: {} for sp in SPLITS}
    for sp in SPLITS:
        for m, v in models.items():
            est, boot = ratio_boot(ctx[sp], hits[m][sp])
            out["accuracy"][sp][m] = {**ci_of(est, boot), "label": v["label"], "extra": bool(v.get("extra"))}

    out["paired"] = {sp: [] for sp in SPLITS}
    for a, b, why in PAIRS:
        if a in models and b in models:
            for sp in SPLITS:
                out["paired"][sp].append({"a": a, "b": b, "why": why, **paired_diff(ctx[sp], hits[a][sp], hits[b][sp])})

    if "d1_tac" in models:
        out["calibration"] = _calibration(ctx, models, labels, hits)
        out["selective"] = _selective(ctx, models["d1_tac"]["scores"], labels, hits["d1_tac"])
    out["popularity"] = _popularity(ctx, models, gold_counts, hits)
    if "d1_tac" in models and "lgbm" in models:
        out["ensemble"] = _ensemble(ctx, models, labels, hits)
    return out


def _calibration(ctx, models, labels, hits) -> dict:
    out: dict = {"bins": ECE_BINS, "model": "d1_tac", "splits": {}, "others": {}}
    for m in [m for m, v in models.items() if v["kind"] == "prob"]:
        for sp in SPLITS:
            p = np.asarray(models[m]["scores"][sp], float)
            conf, hit = p.max(axis=1), hits[m][sp]
            ece, table = expected_calibration_error(conf, hit)
            est, boot = ratio_boot(ctx[sp], brier_per_line(p, labels[sp]))
            row = {"ece": ece, "ece_ci": ece_ci(ctx[sp], conf, hit), "brier": ci_of(est, boot),
                   "mean_conf": float(conf.mean()), "acc": float(hit.mean()), "reliability": table}
            if m == "d1_tac":
                out["splits"][sp] = row
            else:
                out["others"].setdefault(m, {})[sp] = {k: v for k, v in row.items() if k != "reliability"}
    return out


def _selective(ctx, scores, labels, hit) -> dict:
    conf = {sp: np.asarray(scores[sp], float).max(axis=1) for sp in SPLITS}
    tau = choose_threshold(conf["val"], hit["val"])
    out: dict = {"threshold": tau, "rule": "answer when max probability >= threshold; threshold = best +1/0/-1 utility on val",
                 "curves": {sp: risk_coverage(conf[sp], hit[sp]) for sp in SPLITS}, "at_threshold": {}}
    for sp in SPLITS:
        ans = (conf[sp] >= tau).astype(float)
        sign = 2 * hit[sp] - 1
        u_est, u_boot = ratio_boot(ctx[sp], ans * sign)
        c_est, c_boot = ratio_boot(ctx[sp], ans)
        a_est, a_boot = ratio_boot(ctx[sp], ans * hit[sp], ans)
        all_est, all_boot = ratio_boot(ctx[sp], sign)
        out["at_threshold"][sp] = {"utility": ci_of(u_est, u_boot), "coverage": ci_of(c_est, c_boot),
                                   "accuracy_answered": ci_of(a_est, a_boot), "answer_all_utility": ci_of(all_est, all_boot),
                                   "aurc": out["curves"][sp]["aurc"]}
    return out


def _popularity(ctx, models, gold_counts, hits) -> dict:
    pooled = np.concatenate([np.asarray(gold_counts[sp], float) for sp in SPLITS])
    edges = tercile_edges(pooled)
    out: dict = {"edges": list(edges), "tercile_rule": f"low: count <= {edges[0]:g}; mid: <= {edges[1]:g}; high: above",
                 "n": {}, "unseen_share": {}, "accuracy": {}}
    terc = {sp: tercile_of(gold_counts[sp], edges) for sp in SPLITS}
    for sp in SPLITS:
        out["n"][sp] = [int(np.sum(terc[sp] == t)) for t in range(3)]
        out["unseen_share"][sp] = float(np.mean(np.asarray(gold_counts[sp]) == 0))
        out["accuracy"][sp] = {}
        for m in models:
            if m.startswith("lgbm_s"):
                continue
            row = []
            for t in range(3):
                mask = (terc[sp] == t).astype(float)
                est, boot = ratio_boot(ctx[sp], mask * hits[m][sp], mask)
                row.append(ci_of(est, boot))
            out["accuracy"][sp][m] = row
    return out


def _ensemble(ctx, models, labels, hits) -> dict:
    """z-scored (per line) log-probabilities of d1-tac and scores of LightGBM; the d1 weight is picked on val."""
    z_d1 = {sp: zscore_rows(_log_probs(np.asarray(models["d1_tac"]["scores"][sp], float))) for sp in SPLITS}
    out: dict = {"grid": WEIGHT_GRID, "rows": []}
    for name in ["lgbm"] + sorted(m for m in models if m.startswith("lgbm_s")):
        z_gb = {sp: zscore_rows(models[name]["scores"][sp]) for sp in SPLITS}
        w, val_accs = pick_ensemble_weight(z_d1["val"], z_gb["val"], labels["val"])
        row: dict = {"lightgbm": name, "weight_d1": w, "val_acc_by_weight": val_accs}
        for sp in SPLITS:
            hit_e = argmax_hits(w * z_d1[sp] + (1 - w) * z_gb[sp], labels[sp])
            est, boot = ratio_boot(ctx[sp], hit_e)
            row[sp] = {"acc": ci_of(est, boot),
                       "vs_d1_tac": paired_diff(ctx[sp], hit_e, hits["d1_tac"][sp]),
                       "vs_lightgbm": paired_diff(ctx[sp], hit_e, hits[name][sp])}
        out["rows"].append(row)
    out["primary"] = out["rows"][0]
    return out


# ----------------------------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_lines(dataset: Path) -> dict:
    """ranker/{val,test}.jsonl: ids, game ids, option keys and the gold option index of every 4-way line."""
    out = {}
    for sp in SPLITS:
        ex = read_jsonl(Path(dataset) / "ranker" / f"{sp}.jsonl")
        out[sp] = {"ids": [e["id"] for e in ex], "game_ids": np.array([e["game_id"] for e in ex]),
                   "labels": np.array([e["label"] for e in ex], dtype=int),
                   "keys": [[o["key"] for o in e["options"]] for e in ex], "kinds": [e["kind"] for e in ex]}
    return out


def train_key_counts(dataset: Path) -> Counter:
    """Train-split play counts per action key (train_eligible records of states.jsonl), as in eval_ranker_baselines."""
    pop: Counter = Counter()
    for r in read_jsonl(Path(dataset) / "states.jsonl"):
        if r.get("split") == "train" and r.get("train_eligible", True) and (r.get("action") or {}).get("key"):
            pop[r["action"]["key"]] += 1
    return pop


def baseline_scores(lines: dict, pop: Counter, zone_freq: dict, seed: int = SEED) -> dict:
    """Per-line option scores of the three baselines. `random` is one seeded random pick per line (expected accuracy 0.25)."""
    def placement(key: str) -> float:
        card, _, zone = key.split("|")
        zf = zone_freq.get(card, {})
        tot = sum(zf.values())
        return zf.get(zone, 0) / tot if tot else 0.0

    rng = np.random.default_rng(seed + 99)
    out: dict = {"random": {}, "popularity": {}, "placement_prior": {}}
    for sp in SPLITS:
        keys = lines[sp]["keys"]
        out["random"][sp] = rng.random((len(keys), 4))
        out["popularity"][sp] = np.array([[float(pop[k]) for k in ks] for ks in keys])
        out["placement_prior"][sp] = np.array([[placement(k) for k in ks] for ks in keys])
    return out


def load_d1_probs(path: Path, lines: dict, variant: str = "joint") -> tuple[dict, dict]:
    """Probabilities of a d1 eval JSON aligned to `lines` (by id). Checks that the saved pred / gold keys match the
    line's options. Returns ({split: (n, 4)}, {split: recorded accuracy})."""
    d = json.loads(Path(path).read_text())
    rec = {(r["split"], r["id"]): r for r in d["per_example"] if r["variant"] == variant}
    probs, recorded = {}, {}
    for sp in SPLITS:
        rows = []
        for i, id_ in enumerate(lines[sp]["ids"]):
            r = rec.get((sp, id_))
            if r is None:
                raise ValueError(f"{path}: no {variant} record for {sp} line {id_}")
            keys = lines[sp]["keys"][i]
            if len(r["probs"]) != len(keys) or keys[int(np.argmax(r["probs"]))] != r["pred"] or keys[lines[sp]["labels"][i]] != r["gold"]:
                raise ValueError(f"{path}: {sp} line {id_} disagrees with ranker/{sp}.jsonl (options / pred / gold)")
            rows.append(r["probs"])
        probs[sp] = np.array(rows, dtype=float)
        recorded[sp] = (d.get("variants", {}).get(variant, {}).get(sp) or {}).get("acc")
    return probs, recorded


def load_ranker_preds(path: Path, lines: dict) -> dict:
    """{dump name: {seed: {split: (n, 4)}}} from `feature_ranker.py --dump-preds`, checked against `lines`."""
    d = json.loads(Path(path).read_text())
    for sp in SPLITS:
        got = d["lines"][sp]
        if [g["id"] for g in got] != lines[sp]["ids"] or [g["label"] for g in got] != list(lines[sp]["labels"]):
            raise ValueError(f"{path}: {sp} lines differ from ranker/{sp}.jsonl")
    return {name: {int(s): {sp: np.array(v[sp], dtype=float) for sp in SPLITS} for s, v in runs.items()}
            for name, runs in d["models"].items()}


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_models(dataset: Path, docs_data: Path, lines: dict) -> tuple[dict, dict, list[str]]:
    """Every model's per-line scores plus the reproduction checks. Missing optional inputs are listed in `notes`."""
    notes: list[str] = []
    checks: dict = {"d1_recorded_vs_recomputed": {}, "baselines_vs_ranker_baselines": {}, "rankers_vs_v1": {}}
    models: dict = {}
    pop = train_key_counts(dataset)
    prior = json.loads((Path(dataset) / "placement_prior.json").read_text())
    base = baseline_scores(lines, pop, prior["zone_freq"])
    labels_of = {"random": "random (one seeded pick per line)", "popularity": "popularity (train key counts)",
                 "placement_prior": "placement prior"}
    for name in ("random", "popularity", "placement_prior"):
        models[name] = {"label": labels_of[name], "kind": "score", "extra": False, "scores": base[name]}
    bl = Path(dataset) / "ranker" / "baselines.json"
    if bl.exists():
        ref = json.loads(bl.read_text())["ranker"]
        for name in ("popularity", "placement_prior"):
            checks["baselines_vs_ranker_baselines"][name] = {
                sp: {"recomputed": float(argmax_hits(base[name][sp], lines[sp]["labels"]).mean()),
                     "recorded": ref[sp][name]["acc"]} for sp in SPLITS}

    fr_preds = Path(docs_data) / "feature_ranker_preds_v2.json"
    if fr_preds.exists():
        preds = load_ranker_preds(fr_preds, lines)
        v1 = Path(docs_data) / "feature_ranker_v1.json"
        v1_models = json.loads(v1.read_text())["models"] if v1.exists() else {}
        for name, dump in RANKER_MODELS:
            if dump not in preds:
                notes.append(f"ranker {dump} missing from {fr_preds.name}")
                continue
            seeds = sorted(preds[dump])
            models[name] = {"label": {"logreg": "logreg", "lgbm": "LightGBM LambdaRank", "dcn_v2": "DCN-V2"}[name] + " (seed 0)",
                            "kind": "score", "extra": False, "scores": preds[dump][seeds[0]]}
            if name == "lgbm":
                for s in seeds[1:]:
                    models[f"lgbm_s{s}"] = {"label": f"LightGBM (seed {s})", "kind": "score", "extra": True,
                                            "scores": preds[dump][s]}
            best = (v1_models.get(dump) or {}).get("best") or {}
            if best.get("seeds"):
                rows = []
                for j, s in enumerate(best["seeds"]):
                    if s in preds[dump]:
                        rows.append({"seed": s, **{sp: {"recomputed": float(argmax_hits(preds[dump][s][sp], lines[sp]["labels"]).mean()),
                                                        "v1": best[f"{sp}_acc"][j]} for sp in SPLITS}})
                checks["rankers_vs_v1"][name] = rows
    else:
        notes.append(f"{fr_preds} missing: no logreg / LightGBM / DCN-V2 rows (run feature_ranker.py --dump-preds)")

    for name, label, fname, extra in D1_SOURCES:
        path = Path(docs_data) / fname
        if not path.exists() and name in D1_FALLBACK:
            path = Path(docs_data) / D1_FALLBACK[name]
            notes.append(f"{name}: {fname} missing, using {path.name}")
        if not path.exists():
            notes.append(f"{name}: {path.name} missing, row skipped")
            continue
        probs, recorded = load_d1_probs(path, lines)
        models[name] = {"label": label, "kind": "prob", "extra": extra, "scores": probs, "source": path.name}
        checks["d1_recorded_vs_recomputed"][name] = {
            sp: {"recomputed": float(argmax_hits(probs[sp], lines[sp]["labels"]).mean()), "recorded": recorded[sp]} for sp in SPLITS}
    gold = {sp: np.array([pop[ks[g]] for ks, g in zip(lines[sp]["keys"], lines[sp]["labels"])], dtype=float) for sp in SPLITS}
    return models, {"checks": checks, "gold_counts": gold, "notes": notes}, notes


def reproduction_verdict(checks: dict) -> dict:
    """Largest absolute gap, in accuracy points, between the recomputed accuracies and the recorded ones."""
    gaps = {"rankers": 0.0, "baselines": 0.0, "d1": 0.0}
    for rows in checks.get("rankers_vs_v1", {}).values():
        for r in rows:
            for sp in SPLITS:
                gaps["rankers"] = max(gaps["rankers"], abs(r[sp]["recomputed"] - r[sp]["v1"]) * 100)
    for grp, key in (("baselines", "baselines_vs_ranker_baselines"), ("d1", "d1_recorded_vs_recomputed")):
        for per in checks.get(key, {}).values():
            for sp in SPLITS:
                if per[sp]["recorded"] is not None:
                    gaps[grp] = max(gaps[grp], abs(per[sp]["recomputed"] - per[sp]["recorded"]) * 100)
    return {"max_gap_points": {k: round(v, 3) for k, v in gaps.items()},
            "rankers_within_0.5pt": gaps["rankers"] <= 0.5 if checks.get("rankers_vs_v1") else None}


# ----------------------------------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------------------------------

def _round(x, nd: int = 7):
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return _round(x.tolist(), nd)
    if isinstance(x, (np.floating, float)):
        return None if (isinstance(x, float) and math.isnan(x)) else (float("inf") if math.isinf(x) else round(float(x), nd))
    if isinstance(x, np.integer):
        return int(x)
    return x


def _acc(c: dict) -> str:
    return f"{c['est']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def _pts(c: dict) -> str:
    return f"{100 * c['est']:+.1f} [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]"


def _p(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def render_markdown(res: dict) -> str:
    n, g = res["n_lines"], res["n_games"]
    L = [f"# Statistics audit of the 4-way table (E1, MAC-AP)", "",
         f"- git HEAD `{res['meta'].get('git_head')}` at run time; frozen decision code `66917e7` (D-051). Lines: val {n['val']} "
         f"({g['val']} games), test {n['test']} ({g['test']} games).",
         f"- CIs: percentile, {res['n_boot']:,} cluster-bootstrap resamples of whole `game_id`s (seed {res['seed']}); "
         "the same resampled games are used for every statistic of a split, so differences are paired.",
         "- McNemar p is exact and treats lines as independent (lines of one game are correlated, so it is the optimistic one); "
         "`p boot` is the two-sided cluster-bootstrap p of the difference.",
         "- Thresholds and the ensemble weight are chosen on val only. LightGBM / DCN-V2 / logreg configs were selected on val "
         "accuracy (feature_ranker.py), so the val column is optimistic for them; test is the clean column.", ""]
    for note in res["meta"].get("notes", []):
        L.append(f"- NOTE: {note}")
    rep = res["meta"].get("reproduction", {})
    if rep:
        mg = rep["max_gap_points"]
        L += ["", f"Reproduction: recomputed vs recorded accuracies, largest gap in points: rankers vs `feature_ranker_v1.json` "
              f"{mg['rankers']} (within 0.5 pt: {rep['rankers_within_0.5pt']}), baselines vs `ranker/baselines.json` {mg['baselines']}, "
              f"d1 vs the eval JSON `acc` {mg['d1']}."]
    L += ["", "## 1. Accuracy with 95% CI", "", "| model | val | test |", "|---|---|---|"]
    order = [m for m in MAIN_ORDER if m in res["accuracy"]["test"]]
    extras = [m for m in res["accuracy"]["test"] if m not in order and not m.startswith("lgbm_s")]
    for m in order + extras:
        a = res["accuracy"]
        tag = " (extra)" if a["test"][m]["extra"] else ""
        L.append(f"| {a['test'][m]['label']}{tag} | {_acc(a['val'][m])} | {_acc(a['test'][m])} |")
    L += ["", "## 2. Paired differences (accuracy points, a minus b)", "",
          "| a vs b | split | delta [95% CI] | a only right | b only right | McNemar p | p boot |", "|---|---|---|---|---|---|---|"]
    for sp in ("test", "val"):
        for r in res["paired"][sp]:
            L.append(f"| {r['a']} vs {r['b']} ({r['why']}) | {sp} | {_pts(r)} | {r['mcnemar_a_only']} | {r['mcnemar_b_only']} | "
                     f"{_p(r['mcnemar_p'])} | {_p(r['p_boot'])} |")
    if "calibration" in res:
        cal = res["calibration"]
        L += ["", "## 3. Calibration of d1-tac", "", "| split | ECE (15 bins) [CI] | Brier [CI] | mean confidence | accuracy |", "|---|---|---|---|---|"]
        for sp in SPLITS:
            c = cal["splits"][sp]
            L.append(f"| {sp} | {_acc(c['ece_ci'])} | {_acc(c['brier'])} | {c['mean_conf']:.3f} | {c['acc']:.3f} |")
        L += ["", "The ECE interval sits above the point estimate: ECE is biased upward on resamples (every resample re-bins noisy "
              "bins), so read the point estimate with the width of the interval, not the interval's centre."]
        for sp in SPLITS:
            L += ["", f"Reliability, {sp} (empty bins omitted)", "", "| confidence bin | n | mean conf | accuracy | gap |", "|---|---|---|---|---|"]
            for r in cal["splits"][sp]["reliability"]:
                if r["n"]:
                    L.append(f"| {r['lo']:.2f}-{r['hi']:.2f} | {r['n']} | {r['conf']:.3f} | {r['acc']:.3f} | {r['gap']:+.3f} |")
        L += ["", "Other d1 models (ECE / Brier, test): " + "; ".join(
            f"{m} {v['test']['ece']:.3f} / {v['test']['brier']['est']:.3f}" for m, v in cal["others"].items())]
    if "selective" in res:
        s = res["selective"]
        L += ["", "## 4. Selective prediction (hold when unsure), d1-tac", "",
              f"Rule: {s['rule']}. Val-chosen threshold **{s['threshold']:.4f}**. Utility = (right - wrong) / N, abstain = 0.", "",
              "| split | coverage | accuracy answered | utility at threshold | utility answering everything | AURC (risk) |", "|---|---|---|---|---|---|"]
        for sp in SPLITS:
            a = s["at_threshold"][sp]
            L.append(f"| {sp} | {_acc(a['coverage'])} | {_acc(a['accuracy_answered'])} | {_acc(a['utility'])} | "
                     f"{_acc(a['answer_all_utility'])} | {a['aurc']:.3f} |")
        L += ["", "Risk-coverage curve: `stats_audit_v1_risk_coverage.csv`."]
    pop = res["popularity"]
    L += ["", "## 5. Accuracy by popularity tercile of the gold key (train counts)", "",
          f"Terciles pooled over val + test lines; {pop['tercile_rule']}. Share of gold keys never seen in train: "
          f"val {pop['unseen_share']['val']:.2f}, test {pop['unseen_share']['test']:.2f}.", "",
          "| model | split | low | mid | high |", "|---|---|---|---|---|"]
    for sp in ("test", "val"):
        L.append(f"| n lines | {sp} | {pop['n'][sp][0]} | {pop['n'][sp][1]} | {pop['n'][sp][2]} |")
        for m in order + [e for e in extras if e in pop["accuracy"][sp]]:
            row = pop["accuracy"][sp][m]
            L.append(f"| {res['accuracy'][sp][m]['label']} | {sp} | " + " | ".join(_acc(c) for c in row) + " |")
    if "ensemble" in res:
        e = res["ensemble"]
        L += ["", "## 6. Ensemble d1-tac + LightGBM", "",
              "Score = w * z(log p of d1-tac) + (1 - w) * z(LightGBM score), each z-scored across the 4 options of a line; "
              f"w (the d1 weight) picked on val from {{0, 0.05, ..., 1}}, ties to the most even blend.", "",
              "| LightGBM seed | w (d1) | split | ensemble acc [CI] | vs d1-tac (pts) | p McNemar | vs LightGBM (pts) | p McNemar |", "|---|---|---|---|---|---|---|---|"]
        for r in e["rows"]:
            for sp in ("test", "val"):
                d = r[sp]
                L.append(f"| {r['lightgbm']} | {r['weight_d1']:.2f} | {sp} | {_acc(d['acc'])} | {_pts(d['vs_d1_tac'])} | "
                         f"{_p(d['vs_d1_tac']['mcnemar_p'])} | {_pts(d['vs_lightgbm'])} | {_p(d['vs_lightgbm']['mcnemar_p'])} |")
    L.append("")
    return "\n".join(L)


def write_curve_csv(path: Path, res_raw: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "threshold", "n_answered", "coverage", "accuracy", "risk", "utility"])
        for sp in SPLITS:
            c = res_raw["selective"]["curves"][sp]
            for i in range(len(c["threshold"])):
                w.writerow(["d1_tac", sp, f"{c['threshold'][i]:.6f}", int(c["n_answered"][i]), f"{c['coverage'][i]:.5f}",
                            f"{c['accuracy'][i]:.5f}", f"{c['risk'][i]:.5f}", f"{c['utility'][i]:.5f}"])


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--docs-data", type=Path, default=ROOT / "docs" / "data")
    ap.add_argument("--out", type=Path, default=None, help="default <docs-data>/stats_audit_v1.json")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    out = args.out if args.out is not None else Path(args.docs_data) / "stats_audit_v1.json"

    lines = load_lines(args.dataset)
    models, extra, notes = build_models(args.dataset, args.docs_data, lines)
    res = audit(lines, models, extra["gold_counts"], args.boot, args.seed)
    res["meta"] = {"git_head": git_head(), "notes": notes, "checks": extra["checks"],
                   "reproduction": reproduction_verdict(extra["checks"]),
                   "sources": {m: v.get("source") for m, v in models.items() if v.get("source")}}
    md = render_markdown(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    if "selective" in res:
        write_curve_csv(out.with_name(out.stem + "_risk_coverage.csv"), res)
        res["selective"].pop("curves")                # the CSV holds them
    out.write_text(json.dumps(_round(res), indent=1) + "\n")
    out.with_suffix(".md").write_text(md)
    print(md)
    return res


if __name__ == "__main__":
    main()
