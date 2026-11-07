"""Feature-interaction ranker baselines on the 4-way multiple-choice task (MAC-R3).

  python3 tools/eval/feature_ranker.py [--dataset DIR] [--models logreg,lgbm_lambdarank,dcn_v2]
          [--seeds 0,1,2] [--limit N] [--quick] [--out PATH]

Each option is scored from the state and that option alone (pointwise features, no look at the other
options, no legality, no `source`/`positive_key`/`pos_in_legal`, no kNN vote). Popularity and zone statistics
come from the train split and are leave-one-game-out. Three scorers are trained with a listwise (or lambdarank)
objective: a linear model, LightGBM LambdaRank and a DCN-V2-style cross network + MLP. Writes <out> (JSON) and
<out>.md (table); the reference rows include the option-set shortcut, which is NOT a valid model.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as nnf  # noqa: E402
from torch import nn  # noqa: E402

from clashbot.config import DATASET_DIR, DB_PATH, LANES, ZONES  # noqa: E402
from clashbot.features import feature_names, featurize, vocab  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.kg import KnowledgeGraph  # noqa: E402

SPLITS = ("train", "val", "test")
MODEL_NAMES = ("logreg", "lgbm_lambdarank", "dcn_v2")
REFERENCE_NAMES = ("random_expected", "placement_prior", "popularity")
N_OPTIONS = 4
CARD_TYPES = ("Troop", "Spell", "Building", "other")
OPT_SCALARS = ("cost", "in_hand", "elixir_after", "p_zone_given_card", "log_pop_key", "log_pop_card",
               "log_pop_key_phase", "kg_counter")
THREAT_RECENT_S = 5.0     # same rule as CandidateGenerator._threats
THREAT_ETA_S = 8.0
CLIP = 10.0               # standardised features are clipped to +-CLIP (rare one-hots would explode the cross layers)
QUICK_EPOCHS = 5


# ----------------------------------------------------------------------------------------------------------------
# train statistics (leave-one-game-out)
# ----------------------------------------------------------------------------------------------------------------

class _Counts:
    __slots__ = ("zone", "key", "card", "key_phase")

    def __init__(self) -> None:
        self.zone: Counter = Counter()        # (card, zone)
        self.key: Counter = Counter()         # card|lane|zone
        self.card: Counter = Counter()
        self.key_phase: Counter = Counter()   # (phase, key)

    def add(self, key: str, phase) -> None:
        card, _lane, zone = key.split("|")
        self.zone[(card, zone)] += 1
        self.key[key] += 1
        self.card[card] += 1
        self.key_phase[(phase, key)] += 1


class Stats:
    """Train play counts, kept per game so that one game can be left out when a feature is read."""

    def __init__(self) -> None:
        self.total = _Counts()
        self.per_game: dict[str, _Counts] = defaultdict(_Counts)

    def add(self, game_id: str, key: str, phase) -> None:
        self.total.add(key, phase)
        self.per_game[game_id].add(key, phase)

    def _count(self, table: str, item, game: str | None) -> int:
        n = getattr(self.total, table)[item]
        if game is not None and game in self.per_game:
            n -= getattr(self.per_game[game], table)[item]
        return n

    def zone_count(self, card: str, zone: str, game: str | None = None) -> int:
        return self._count("zone", (card, zone), game)

    def key_count(self, key: str, game: str | None = None) -> int:
        return self._count("key", key, game)

    def card_count(self, card: str, game: str | None = None) -> int:
        return self._count("card", card, game)

    def key_phase_count(self, phase, key: str, game: str | None = None) -> int:
        return self._count("key_phase", (phase, key), game)

    def p_zone_given_card(self, card: str, zone: str, game: str | None = None) -> float:
        """P(zone | card) from train zone counts, add-one smoothing over the 4 zones."""
        tot = sum(self.zone_count(card, z, game) for z in ZONES)
        return (self.zone_count(card, zone, game) + 1.0) / (tot + float(len(ZONES)))


def train_stats(records: list[dict], exclude_game: str | None = None) -> Stats:
    """Counts over train-split played actions of states.jsonl records, skipping `exclude_game` entirely."""
    stats = Stats()
    for r in records:
        if r.get("split") != "train" or not r.get("train_eligible", True):
            continue
        if exclude_game is not None and r.get("game_id") == exclude_game:
            continue
        a = r.get("action") or {}
        key = a.get("key")
        if a.get("type") != "play" or not key or len(key.split("|")) != 3:
            continue
        stats.add(r["game_id"], key, (r.get("state") or {}).get("phase"))
    return stats


# ----------------------------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------------------------

def option_feature_names() -> list[str]:
    names = [f"opt:card:{s}" for s in vocab()]
    names += [f"opt:lane:{x}" for x in LANES]
    names += [f"opt:zone:{x}" for x in ZONES]
    names += ["opt:cost"]
    names += [f"opt:type:{x}" for x in CARD_TYPES]
    names += [f"opt:{x}" for x in OPT_SCALARS[1:]]
    return names


def all_feature_names() -> list[str]:
    return [f"state:{n}" for n in feature_names()] + option_feature_names()


def n_features() -> int:
    return len(all_feature_names())


def column_groups() -> dict[str, list[int]]:
    """Column indices used by the ablations."""
    names = all_feature_names()
    pick = lambda pred: [i for i, n in enumerate(names) if pred(n)]  # noqa: E731
    pop = {"opt:log_pop_key", "opt:log_pop_card", "opt:log_pop_key_phase"}
    return {"popularity": pick(lambda n: n in pop),
            "state_block": pick(lambda n: n.startswith("state:")),
            "kg_counter": pick(lambda n: n == "opt:kg_counter"),
            "train_stats": pick(lambda n: n in pop or n == "opt:p_zone_given_card")}


def threat_cards(state: dict, t: float) -> list[str]:
    """Cards currently threatening us; copies the rule of CandidateGenerator._threats."""
    out: list[str] = []
    for p in (state.get("opponent") or {}).get("recent_plays") or []:
        if p.get("card") and p.get("t") is not None and 0 <= t - p["t"] <= THREAT_RECENT_S:
            out.append(p["card"])
    for u in state.get("units") or []:
        eta = u.get("eta_tower")
        if (u.get("side") == "enemy" and u.get("card") and isinstance(eta, dict)
                and str(eta.get("tower", "")).startswith("own")
                and eta.get("s") is not None and eta["s"] <= THREAT_ETA_S):
            out.append(u["card"])
    return out


def build_features(example: dict, record: dict, stats: Stats, kg: KnowledgeGraph | None) -> np.ndarray:
    """Feature matrix (n_options, F) of one example. Reads only record['state'], record['t'], the game id and
    each option's key; each row depends on the state and that option alone (pointwise)."""
    cards = load_cards()
    v = vocab()
    cidx = {s: i for i, s in enumerate(v)}
    V = len(v)
    state, t = record["state"], record["t"]
    game = record.get("game_id", example.get("game_id"))
    state_vec = featurize(state, t)
    S = len(state_vec)

    own = state.get("own") or {}
    hand = {h for h in own.get("hand") or [] if h}
    elixir = own.get("elixir") or 0.0
    phase = state.get("phase")
    kg_scores = [kg.counter_scores(th) for th in threat_cards(state, t)] if kg is not None else []

    o_lane, o_zone, o_cost = V, V + len(LANES), V + len(LANES) + len(ZONES)
    o_type = o_cost + 1
    o_scalar = o_type + len(CARD_TYPES)
    opt_dim = o_scalar + len(OPT_SCALARS) - 1
    rows = np.zeros((len(example["options"]), S + opt_dim), dtype=np.float32)
    for i, opt in enumerate(example["options"]):
        key = opt["key"]
        parts = key.split("|")
        card, lane, zone = parts if len(parts) == 3 else (None, None, None)
        row = rows[i]
        row[:S] = state_vec
        o = S
        if card in cidx:
            row[o + cidx[card]] = 1.0
        if lane in LANES:
            row[o + o_lane + LANES.index(lane)] = 1.0
        if zone in ZONES:
            row[o + o_zone + ZONES.index(zone)] = 1.0
        info = cards.get(card) if card else None
        cost = float((info or {}).get("elixir_cost") or 0.0)
        ctype = (info or {}).get("card_type")
        row[o + o_cost] = cost / 10.0
        row[o + o_type + (CARD_TYPES.index(ctype) if ctype in CARD_TYPES[:3] else 3)] = 1.0
        p_zone = stats.p_zone_given_card(card, zone, game) if card and zone else 1.0 / len(ZONES)
        row[o + o_scalar:o + opt_dim] = [
            float(card in hand), (elixir - cost) / 10.0, p_zone,
            np.log1p(stats.key_count(key, game)), np.log1p(stats.card_count(card, game)) if card else 0.0,
            np.log1p(stats.key_phase_count(phase, key, game)),
            max((s.get(card, 0.0) for s in kg_scores), default=0.0) if card else 0.0]
    return rows


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_kg() -> KnowledgeGraph | None:
    """The knowledge graph for kg_counter, or None (feature = 0.0) if db/ is missing or unreadable."""
    path = Path(DB_PATH)
    if not path.exists():
        return None
    try:
        return KnowledgeGraph(path)
    except Exception as exc:  # sqlite3.Error / OSError / missing table
        print(f"warning: knowledge graph unavailable ({exc}); kg_counter = 0.0", file=sys.stderr)
        return None


def build_split(examples: list[dict], by_id: dict, stats: Stats, kg) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.stack([build_features(ex, by_id[ex["id"]], stats, kg) for ex in examples]).astype(np.float32)
    y = np.array([ex["label"] for ex in examples], dtype=np.int64)
    return X, y, [ex["kind"] for ex in examples]


def option_set_scores(example: dict) -> list[float]:
    """score(k) = #options with k's card + #options with k's (lane, zone). Uses the other options: a leak, not a model."""
    parsed = [o["key"].split("|") for o in example["options"]]
    cards = Counter(p[0] for p in parsed)
    lz = Counter((p[1], p[2]) for p in parsed)
    return [float(cards[p[0]] + lz[(p[1], p[2])]) for p in parsed]


def option_set_accuracy(example: dict) -> float:
    """Expected accuracy of the arg-max of option_set_scores with a uniformly random tie-break."""
    s = option_set_scores(example)
    top = max(s)
    ties = [i for i, x in enumerate(s) if x == top]
    return 1.0 / len(ties) if example["label"] in ties else 0.0


def by_kind(values: np.ndarray, kinds: list[str]) -> dict[str, float]:
    return {k: float(np.mean([v for v, kk in zip(values, kinds) if kk == k])) for k in sorted(set(kinds))}


# ----------------------------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------------------------

def listwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy over the options of each example. scores (B, K), labels (B,)."""
    return nnf.cross_entropy(scores, labels)


def accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(scores, axis=1) == y))


class LinearScorer(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.lin = nn.Linear(f, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, K, F) -> (B, K)
        return self.lin(x).squeeze(-1)


class DCNv2Scorer(nn.Module):
    """x_{l+1} = x0 * (W_l x_l + b_l) + x_l (full-rank cross layers) in parallel with a ReLU MLP."""

    def __init__(self, f: int, n_cross: int = 3, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        self.cross = nn.ModuleList(nn.Linear(f, f) for _ in range(n_cross))
        layers: list[nn.Module] = []
        d = f
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.deep = nn.Sequential(*layers)
        self.out = nn.Linear(f + d, 1)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for lin in self.cross:
            x = x0 * lin(x) + x
        return self.out(torch.cat([x, self.deep(x0)], dim=-1)).squeeze(-1)


def standardise(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((X - mean) / std, -CLIP, CLIP).astype(np.float32)


def fit_standardiser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    return mean, np.where(std < 1e-6, 1.0, std).astype(np.float32)


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def fit_torch(kind: str, cfg: dict, d: dict, seed: int) -> dict:
    """Train the linear or DCN-V2 scorer; DCN-V2 early-stops (best epoch restored) on val accuracy."""
    torch.manual_seed(seed)
    Xtr, Xva, Xte = d["Xtr"], d["Xva"], d["Xte"]
    if kind == "dcn_v2":
        mean, std = fit_standardiser(Xtr)
        Xtr, Xva, Xte = (standardise(x, mean, std) for x in (Xtr, Xva, Xte))
    f = Xtr.shape[-1]
    model: nn.Module = LinearScorer(f) if kind == "logreg" else DCNv2Scorer(f)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(d["ytr"])
    n = len(yt)
    best_acc, best_state, best_epoch, bad = -1.0, None, -1, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            listwise_loss(model(xt[idx]), yt[idx]).backward()
            opt.step()
        if kind != "dcn_v2":
            continue
        acc = accuracy(predict(model, Xva), d["yva"])
        if acc > best_acc:
            best_acc, best_state, best_epoch, bad = acc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_scores": predict(model, Xva), "test_scores": predict(model, Xte), "best_epoch": best_epoch}


def fit_lgbm(cfg: dict, d: dict, seed: int, cols: np.ndarray | None = None) -> dict:
    import lightgbm as lgb
    sel = (lambda X: X) if cols is None else (lambda X: X[..., cols])
    Xtr = sel(d["Xtr"])
    n, k, f = Xtr.shape
    labels = np.zeros((n, k), dtype=np.int32)
    labels[np.arange(n), d["ytr"]] = 1
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
                           num_leaves=cfg["num_leaves"], min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, verbose=-1,
                           n_jobs=1)    # torch + lightgbm each ship a libomp on macOS; multi-threaded LightGBM segfaults after torch is imported
    model.fit(Xtr.reshape(n * k, f), labels.reshape(-1), group=[k] * n)

    def score(X: np.ndarray) -> np.ndarray:
        X = sel(X)
        return model.predict(X.reshape(-1, X.shape[-1])).reshape(len(X), k)

    return {"val_scores": score(d["Xva"]), "test_scores": score(d["Xte"]),
            "gain": model.booster_.feature_importance(importance_type="gain")}


def model_grid(name: str, quick: bool) -> list[dict]:
    if name == "logreg":
        grid = [{"lr": lr, "wd": 1e-4, "epochs": 40, "batch": 128} for lr in (1e-2, 1e-3)]
    elif name == "dcn_v2":
        grid = [{"lr": lr, "wd": 1e-5, "epochs": 40, "patience": 5, "batch": 128} for lr in (1e-3, 3e-4)]
    elif name == "lgbm_lambdarank":
        grid = [{"n_estimators": n, "learning_rate": 0.05, "num_leaves": nl} for n in (200, 500) for nl in (15, 31)]
    else:
        raise ValueError(f"unknown model {name!r}")
    if quick:
        grid = grid[:1]
        if "epochs" in grid[0]:
            grid[0]["epochs"] = QUICK_EPOCHS
    return grid


def fit_model(name: str, cfg: dict, d: dict, seed: int) -> dict:
    return fit_lgbm(cfg, d, seed) if name == "lgbm_lambdarank" else fit_torch(name, cfg, d, seed)


def summarise(name: str, grid: list[dict], best: int, runs: dict[int, dict], d: dict) -> dict:
    """Mean/std over seeds of val and test accuracy (+ test accuracy by kind) for the chosen config."""
    seeds = sorted(runs)
    val = [accuracy(runs[s]["val_scores"], d["yva"]) for s in seeds]
    test = [accuracy(runs[s]["test_scores"], d["yte"]) for s in seeds]
    per_kind: dict[str, list[float]] = defaultdict(list)
    for s in seeds:
        hit = (np.argmax(runs[s]["test_scores"], axis=1) == d["yte"]).astype(float)
        for kd, a in by_kind(hit, d["kte"]).items():
            per_kind[kd].append(a)
    counts = Counter(d["kte"])
    return {"grid": grid,
            "best": {"config": grid[best]["config"], "seeds": seeds, "val_acc": val, "test_acc": test},
            "val_acc_mean": float(np.mean(val)), "val_acc_std": float(np.std(val)),
            "test_acc_mean": float(np.mean(test)), "test_acc_std": float(np.std(test)),
            "test_by_kind": {kd: {"acc_mean": float(np.mean(v)), "acc_std": float(np.std(v)), "n": counts[kd]}
                             for kd, v in sorted(per_kind.items())}}


def run_model(name: str, d: dict, seeds: list[int], quick: bool, collect: dict | None = None) -> tuple[dict, dict]:
    """Choose the config on val accuracy (seed 0), then train the chosen config on every seed. `collect`, when given,
    receives {name: {seed: {"val": scores, "test": scores}}} for `--dump-preds`."""
    t0 = time.time()
    cache: dict[tuple[int, int], dict] = {}
    grid = []
    for ci, cfg in enumerate(model_grid(name, quick)):
        cache[(ci, 0)] = fit_model(name, cfg, d, 0)
        entry = {"config": cfg, "val_acc": accuracy(cache[(ci, 0)]["val_scores"], d["yva"])}
        if "best_epoch" in cache[(ci, 0)]:
            entry["best_epoch"] = cache[(ci, 0)]["best_epoch"]
        grid.append(entry)
    best = max(range(len(grid)), key=lambda i: (grid[i]["val_acc"], -i))
    runs = {}
    for s in seeds:
        if (best, s) not in cache:
            cache[(best, s)] = fit_model(name, grid[best]["config"], d, s)
        runs[s] = cache[(best, s)]
    print(f"[{name}] grid val acc {[round(g['val_acc'], 4) for g in grid]} -> config {best} "
          f"({time.time() - t0:.0f}s)", file=sys.stderr)
    if collect is not None:
        collect[name] = {s: {"val": runs[s]["val_scores"], "test": runs[s]["test_scores"]} for s in seeds}
    return summarise(name, grid, best, runs, d), cache[(best, 0)]


def run_ablation(cfg: dict, d: dict, seed: int = 0) -> dict:
    """LightGBM with feature groups removed (chosen config, one seed)."""
    n_total = d["Xtr"].shape[-1]
    out = {}
    for label, drop in [("full", [])] + sorted(column_groups().items()):
        keep = np.setdiff1d(np.arange(n_total), np.array(drop, dtype=int))
        r = fit_lgbm(cfg, d, seed, cols=keep)
        out[label if label == "full" else f"no_{label}"] = {
            "n_features": int(len(keep)), "val_acc": accuracy(r["val_scores"], d["yva"]),
            "test_acc": accuracy(r["test_scores"], d["yte"])}
    return out


def top_importance(gain: np.ndarray, k: int = 20) -> list[dict]:
    names = all_feature_names()
    total = float(gain.sum()) or 1.0
    return [{"feature": names[i], "gain": float(gain[i]), "gain_frac": float(gain[i]) / total}
            for i in np.argsort(-gain, kind="stable")[:k]]


# ----------------------------------------------------------------------------------------------------------------
# reference rows and report
# ----------------------------------------------------------------------------------------------------------------

def build_reference(dataset: Path, ex_val: list[dict], ex_test: list[dict]) -> dict:
    ref: dict = {}
    path = Path(dataset) / "ranker" / "baselines.json"
    if path.exists():
        ranker = json.loads(path.read_text())["ranker"]
        for name in REFERENCE_NAMES:
            ref[name] = {"val_acc": ranker["val"][name]["acc"], "test_acc": ranker["test"][name]["acc"],
                         "test_by_kind": {k: v["acc"] for k, v in ranker["test"][name]["by_kind"].items()},
                         "source": "copied from ranker/baselines.json (full val/test files)"}
    shortcut = {}
    for split, examples in (("val", ex_val), ("test", ex_test)):
        accs = np.array([option_set_accuracy(ex) for ex in examples])
        shortcut[f"{split}_acc"] = float(accs.mean()) if len(accs) else 0.0
        if split == "test":
            shortcut["test_by_kind"] = by_kind(accs, [ex["kind"] for ex in examples])
    shortcut["note"] = "not a valid model: scores an option by how many other options share its card / (lane, zone); shows the option-set leak"
    ref["option_set_shortcut"] = shortcut
    return ref


def _pm(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def render_markdown(res: dict) -> str:
    lines = [f"Feature-interaction rankers on the 4-way task (n val {res['n']['val']}, n test {res['n']['test']}, "
             f"{res['features']} features; mean ± std over seeds)", "",
             "| model | val acc | test acc |", "|---|---|---|"]
    ref = res["reference"]
    for name in REFERENCE_NAMES:
        if name in ref:
            lines.append(f"| {name} (reference) | {ref[name]['val_acc']:.4f} | {ref[name]['test_acc']:.4f} |")
    sc = ref["option_set_shortcut"]
    lines.append(f"| option_set_shortcut (NOT A VALID MODEL: shows the option-set leak) | "
                 f"{sc['val_acc']:.4f} | {sc['test_acc']:.4f} |")
    for name, m in res["models"].items():
        lines.append(f"| {name} | {_pm(m['val_acc_mean'], m['val_acc_std'])} | {_pm(m['test_acc_mean'], m['test_acc_std'])} |")
    kinds = sorted({k for m in res["models"].values() for k in m["test_by_kind"]})
    if kinds and res["models"]:
        lines += ["", "Test accuracy by kind", "", "| model | " + " | ".join(kinds) + " |", "|---" * (len(kinds) + 1) + "|"]
        for name, m in res["models"].items():
            lines.append(f"| {name} | " + " | ".join(
                _pm(m["test_by_kind"][k]["acc_mean"], m["test_by_kind"][k]["acc_std"]) + f" (n={m['test_by_kind'][k]['n']})"
                if k in m["test_by_kind"] else "-" for k in kinds) + " |")
    if res.get("ablation"):
        lines += ["", "LightGBM ablation (chosen config, seed 0)", "", "| variant | features | val acc | test acc |", "|---|---|---|---|"]
        for name, a in res["ablation"].items():
            lines.append(f"| {name} | {a['n_features']} | {a['val_acc']:.4f} | {a['test_acc']:.4f} |")
    if res.get("importance"):
        lines += ["", "LightGBM top-20 features by gain (chosen config, seed 0)", "", "| feature | gain share |", "|---|---|"]
        lines += [f"| {r['feature']} | {r['gain_frac']:.4f} |" for r in res["importance"]]
    if not res["kg_counter_available"]:
        lines += ["", "kg_counter: knowledge graph unavailable (db/ missing or unreadable), feature = 0.0 for every option."]
    lines += ["", "Config selection and DCN-V2 early stopping use val accuracy, so the val column is optimistic; test is untouched."]
    return "\n".join(lines) + "\n"


def dump_preds(path: Path, examples: dict[str, list[dict]], collected: dict, args_seeds: list[int]) -> None:
    """Per-line option scores of every model and seed (val and test), for tools/eval/eval_stats.py. Options are in the
    order of ranker/<split>.jsonl; `label` is the gold option index."""
    out: dict = {"seeds": args_seeds, "note": "raw option scores (logreg / LambdaRank / DCN-V2 logits); argmax = the pick",
                 "lines": {sp: [{"id": ex["id"], "game_id": ex["game_id"], "kind": ex["kind"], "label": ex["label"],
                                 "keys": [o["key"] for o in ex["options"]]} for ex in examples[sp]]
                           for sp in ("val", "test")},
                 "models": {name: {str(s): {sp: np.round(v[sp], 6).tolist() for sp in ("val", "test")}
                                   for s, v in runs.items()} for name, runs in collected.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--models", default=",".join(MODEL_NAMES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--limit", type=int, default=None, help="first N examples per split")
    ap.add_argument("--quick", action="store_true", help="one config per model, one seed, 5 epochs")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump-preds", type=Path, default=None,
                    help="also write the per-line option scores of every model and seed (val, test) to this JSON")
    args = ap.parse_args(argv)
    dataset = Path(args.dataset)
    out = args.out if args.out is not None else dataset / "eval" / "feature_ranker.json"
    models = [m for m in args.models.split(",") if m]
    bad = [m for m in models if m not in MODEL_NAMES]
    if bad:
        ap.error(f"unknown models {bad}; choose from {list(MODEL_NAMES)}")
    seeds = [int(s) for s in args.seeds.split(",") if s] or [0]
    if args.quick:
        seeds = seeds[:1]

    t0 = time.time()
    records = read_jsonl(dataset / "states.jsonl")
    by_id = {r["id"]: r for r in records}
    stats = train_stats(records)
    kg = load_kg()
    examples = {sp: read_jsonl(dataset / "ranker" / f"{sp}.jsonl")[:args.limit] for sp in SPLITS}
    data = {}
    for sp, key in zip(SPLITS, ("tr", "va", "te")):
        data[f"X{key}"], data[f"y{key}"], data[f"k{key}"] = build_split(examples[sp], by_id, stats, kg)
    print(f"features built: {data['Xtr'].shape} train, {data['Xva'].shape} val, {data['Xte'].shape} test "
          f"({time.time() - t0:.0f}s); kg_counter {'on' if kg is not None else 'OFF (db missing)'}", file=sys.stderr)

    res: dict = {"n": {sp: len(examples[sp]) for sp in SPLITS}, "features": int(data["Xtr"].shape[-1]),
                 "models": {}, "ablation": {}, "reference": build_reference(dataset, examples["val"], examples["test"]),
                 "importance": [], "kg_counter_available": kg is not None}
    collected: dict = {}
    for name in models:
        res["models"][name], first = run_model(name, data, seeds, args.quick, collected if args.dump_preds else None)
        if name == "lgbm_lambdarank":
            res["importance"] = top_importance(first["gain"])
            res["ablation"] = run_ablation(res["models"][name]["best"]["config"], data)

    if args.dump_preds:
        dump_preds(Path(args.dump_preds), examples, collected, seeds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    md = render_markdown(res)
    out.with_suffix(".md").write_text(md)
    print(md)
    return res


if __name__ == "__main__":
    main()
