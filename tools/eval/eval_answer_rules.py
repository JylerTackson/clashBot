"""E6 (MAC-AR R1): are the mined "opponent played X -> pro answered Y" rules predictive on held-out games?

  python3 tools/eval/eval_answer_rules.py [--dataset DIR] [--kg2 PATH] [--out docs/data/answer_rules_v1.json]
          [--windows 2,4,8] [--supports 1,3,5,10] [--boot 10000] [--seed 0] [--rerun-dir DIR] [--force-remine]
          [--no-sweep]

Rules come from the TRAIN events of `<dataset>/kg2/answer_events.jsonl` only (the shipped `db/kg2.sqlite` ANSWERED_BY
edges, i.e. answers.jsonl, support >= 3, window 8 s). Every val / test event (one opponent play and the first
defensive pro play after it) is turned into a ranking task: rank the pro's own deck (games.jsonl) for the threat and
see where the card the pro really played lands. Methods: the full `KG2.answers_for`, each evidence source alone and
without it (pro, role transfer, wiki, simulation), wiki COUNTERS only (`KnowledgeGraph`), train popularity of the answer
card, random in the deck. Ties are broken in expectation (a no-evidence card shares its rank with all the other
no-evidence cards), so a method with no evidence scores exactly like random. Sweeps: support threshold n >= 1/3/5/10 and
answer window 2/4/8 s (the events are re-mined with the window; rules from train only; written under runs/rerun/, never
over dataset/ or db/). CIs: bootstrap over games (clusters), paired for differences. Writes <out> (JSON) and <out>.md.
Read-only use of clashbot/kg2.py, clashbot/kg.py and tools/data/mine_pro_answers.py (D-051 code freeze).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.config import DATASET_DIR, DB_PATH, KG2_PATH  # noqa: E402
from clashbot.kg2 import KG2, SOURCE_WEIGHTS, Answer  # noqa: E402

EVAL_SPLITS = ("val", "test")
SOURCES = tuple(SOURCE_WEIGHTS)                    # pro, role_transfer, wiki, sim
DEFAULT_SUPPORT = 3                                # mine_pro_answers.MIN_SUPPORT (the shipped rules)
DEFAULT_WINDOW = 8.0                               # mine_pro_answers.WINDOW_S
SUPPORTS = (1, 3, 5, 10)
WINDOWS = (2.0, 4.0, 8.0)
BOOT = 10_000
KS = (1, 3)
SCORE_BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01))
SCORE_DECIMALS = 9                                 # scores equal to this many decimals count as tied

METHOD_LABELS = {
    "full": "full KG2.answers_for",
    **{f"only:{s}": f"{s} only" for s in SOURCES},
    **{f"without:{s}": f"all but {s}" for s in SOURCES},
    "wiki_counters": "wiki COUNTERS only (KnowledgeGraph)",
    "counter_scores_v1": "KnowledgeGraph.counter_scores (COUNTERS + v1 train-mined ANSWERED_BY)",
    "popularity": "train popularity of the answer card",
    "random": "random in deck (expected)",
}
METHODS = tuple(METHOD_LABELS)


# --------------------------------------------------------------------------- io
def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def git_head() -> dict:
    """HEAD hash and whether the working tree has uncommitted changes (D-051: benchmarks record the hash they ran at)."""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"head": head, "dirty": dirty}
    except Exception:                              # pragma: no cover - not a git checkout
        return {"head": None, "dirty": None}


# --------------------------------------------------------------------------- rules and scoring
def answers_to_answered(answers: list[dict]) -> dict[str, dict[str, dict]]:
    """answers.jsonl rows -> KG2._answered layout: threat -> answer -> aggregate props (as KG2._load builds it)."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for a in answers:
        props = dict(a)
        props.setdefault("support", 0)
        out[a["threat"]][a["answer"]] = props
    return out


def set_rules(kg: KG2, answers: list[dict]) -> None:
    """Swap the pro-mined ANSWERED_BY relation of an already loaded KG2 for `answers` (in memory only). The other
    relations (wiki COUNTERS, SIMILAR_ROLE, STOPS) do not depend on answer events and stay."""
    kg._answered = answers_to_answered(answers)


def combine_sources(answers: list[Answer], weights: dict[str, float]) -> dict[str, float]:
    """KG2.answers_for's rule restricted to `weights`' sources: the weighted mean of the per-source scores,
    renormalised over the sources that have any evidence among the deck cards. {card: score}; empty = no evidence."""
    active = [s for s in weights if weights[s] > 0 and any(a.sources.get(s, 0.0) > 0 for a in answers)]
    if not active:
        return {}
    norm = sum(weights[s] for s in active)
    return {a.card: sum(weights[s] * a.sources.get(s, 0.0) for s in active) / norm for a in answers}


def source_scores(kg: KG2, threat: str, deck: list[str], answers: list[Answer] | None = None
                  ) -> dict[str, dict[str, float]]:
    """{method: {card: score}} for full, each source alone and each leave-one-source-out, from ONE answers_for call."""
    if answers is None:
        answers = kg.answers_for(threat, deck, k=max(len(deck), 1))
    base = dict(kg.source_weights)
    out = {"full": combine_sources(answers, {s: base[s] for s in SOURCES})}
    for s in SOURCES:
        out[f"only:{s}"] = combine_sources(answers, {s: base[s]})
        out[f"without:{s}"] = combine_sources(answers, {t: base[t] for t in SOURCES if t != s})
    return out


def score_gap(kg: KG2, answers: list[Answer]) -> float:
    """Largest |recomputed full score - Answer.score| (the parts are rounded to 4 decimals): the ablation wiring check."""
    mine = combine_sources(answers, {s: kg.source_weights[s] for s in SOURCES})
    return max((abs(mine.get(a.card, 0.0) - a.score) for a in answers), default=0.0)


def counters_only(wiki_kg, threat: str) -> dict[str, float]:
    """Wiki COUNTERS edges only: how many say card Z counters the threat (KnowledgeGraph.counter_scores minus its v1
    ANSWERED_BY term)."""
    score: dict[str, float] = defaultdict(float)
    for src, _props in wiki_kg.inc.get(("card:" + threat, "COUNTERS"), []):
        score[src.removeprefix("card:")] += 1.0
    return dict(score)


def popularity_counts(events: list[dict]) -> Counter:
    """How often each card was the pro's answer in the TRAIN events."""
    return Counter(e["answer"] for e in events if e["split"] == "train")


# --------------------------------------------------------------------------- ranking metrics
def tie_metrics(scores: dict[str, float], deck: list[str], answer: str) -> dict:
    """Rank `deck` by `scores` (a missing card scores 0) and locate `answer`. Cards tied with the answer share their
    positions: hit@k and the reciprocal rank are the expectations over a random order of the tie group, so a method
    without evidence is exactly random. covered = some deck card has a positive score; answer_scored = the answer has."""
    deck = list(dict.fromkeys(deck))
    if answer not in deck:
        raise ValueError(f"answer {answer!r} is not in the deck")
    sc = {c: round(float(scores.get(c, 0.0)), SCORE_DECIMALS) for c in deck}
    mine = sc[answer]
    greater = sum(1 for c in deck if sc[c] > mine)
    tied = sum(1 for c in deck if sc[c] == mine)
    out = {f"hit{k}": min(1.0, max(0.0, (k - greater) / tied)) for k in KS}
    out["rr"] = float(np.mean([1.0 / (greater + j) for j in range(1, tied + 1)]))
    out["covered"] = float(any(v > 0 for v in sc.values()))
    out["answer_scored"] = float(mine > 0)
    out["rank_best"] = greater + 1
    out["score"] = mine
    return out


class GameBootstrap:
    """Cluster bootstrap over games: resample whole games, recompute a ratio of sums. One resample matrix per event
    set, so every method and every difference on the same events is paired."""

    def __init__(self, groups: list[str], boot: int = BOOT, seed: int = 0):
        uniq, self.inv = np.unique(np.asarray(groups), return_inverse=True)
        self.n_games = len(uniq)
        self.idx = np.random.default_rng(seed).integers(0, max(self.n_games, 1), size=(boot, max(self.n_games, 1)))

    def ci(self, values, mask=None) -> dict:
        """Mean over the events (those with mask true) and its 95% percentile CI; NaNs when there are none."""
        v = np.asarray(values, float)
        m = np.ones(len(v)) if mask is None else np.asarray(mask, float)
        if len(v) == 0 or m.sum() == 0:
            return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
        num_g = np.bincount(self.inv, weights=v * m, minlength=self.n_games)
        den_g = np.bincount(self.inv, weights=m, minlength=self.n_games)
        num, den = num_g[self.idx].sum(1), den_g[self.idx].sum(1)
        est = num[den > 0] / den[den > 0]
        lo, hi = np.percentile(est, [2.5, 97.5])
        return {"mean": float((v * m).sum() / m.sum()), "lo": float(lo), "hi": float(hi), "n": int(m.sum())}

    def diff(self, a, b) -> dict:
        """Paired mean difference a - b with CI and a two-sided bootstrap p (share of resamples on either side of 0)."""
        d = np.asarray(a, float) - np.asarray(b, float)
        out = self.ci(d)
        if out["n"]:
            num = np.bincount(self.inv, weights=d, minlength=self.n_games)[self.idx].sum(1)
            den = np.bincount(self.inv, minlength=self.n_games).astype(float)[self.idx].sum(1)
            est = num / den
            out["p"] = float(min(1.0, 2.0 * min((est <= 0).mean(), (est >= 0).mean())))
        return out


# --------------------------------------------------------------------------- evaluation of one rule set
def eval_rows(kg: KG2, events: list[dict], decks: dict[str, list[str]], pop: Counter, wiki_kg=None) -> dict:
    """Score every event with every method. Returns {"rows": [event dict + per-method metrics], "skipped": counts}.
    Events whose answer is not in the game's recorded deck cannot be ranked and are counted, not scored."""
    rows, skipped = [], Counter()
    worst = 0.0
    for e in events:
        deck = list(dict.fromkeys(decks.get(e["game_id"], [])))
        if e["answer"] not in deck:
            skipped["answer_not_in_deck"] += 1
            continue
        answers = kg.answers_for(e["threat"], deck, k=max(len(deck), 1))
        methods = source_scores(kg, e["threat"], deck, answers)
        worst = max(worst, score_gap(kg, answers))
        if wiki_kg is not None:
            methods["wiki_counters"] = counters_only(wiki_kg, e["threat"])
            methods["counter_scores_v1"] = wiki_kg.counter_scores(e["threat"])
        methods["popularity"] = {c: float(pop.get(c, 0)) for c in deck}
        methods["random"] = {}
        rows.append({"game_id": e["game_id"], "split": e["split"], "threat": e["threat"], "answer": e["answer"],
                     "threat_source": e.get("threat_source"), "held": e.get("held"), "deck_size": len(deck),
                     "pro_support_max": max((a.support for a in answers if a.sources.get("pro", 0.0) > 0), default=0),
                     "m": {name: tie_metrics(sc, deck, e["answer"]) for name, sc in methods.items()}})
    return {"rows": rows, "skipped": dict(skipped), "max_score_gap": worst}


def column(rows: list[dict], method: str, key: str) -> np.ndarray:
    return np.array([r["m"][method][key] for r in rows], float)


def summarise(rows: list[dict], boot: GameBootstrap, methods) -> dict:
    out = {}
    for name in methods:
        if not rows or name not in rows[0]["m"]:
            continue
        cov = column(rows, name, "covered")
        out[name] = {
            "hit@1": boot.ci(column(rows, name, "hit1")), "hit@3": boot.ci(column(rows, name, "hit3")),
            "mrr": boot.ci(column(rows, name, "rr")), "coverage": boot.ci(cov),
            "answer_scored": boot.ci(column(rows, name, "answer_scored")),
            "hit@1_covered": boot.ci(column(rows, name, "hit1"), mask=cov),
            "mrr_covered": boot.ci(column(rows, name, "rr"), mask=cov)}
    return out


def paired(rows: list[dict], boot: GameBootstrap, pairs) -> dict:
    out = {}
    for a, b in pairs:
        if not rows or a not in rows[0]["m"] or b not in rows[0]["m"]:
            continue
        out[f"{a} - {b}"] = {"hit@1": boot.diff(column(rows, a, "hit1"), column(rows, b, "hit1")),
                             "hit@3": boot.diff(column(rows, a, "hit3"), column(rows, b, "hit3")),
                             "mrr": boot.diff(column(rows, a, "rr"), column(rows, b, "rr"))}
    return out


STRATA = (("no pro rule for this threat in the deck", lambda r: r["pro_support_max"] == 0),
          ("best pro rule support 3-4", lambda r: 0 < r["pro_support_max"] < 5),
          ("best pro rule support 5-9", lambda r: 5 <= r["pro_support_max"] < 10),
          ("best pro rule support 10+", lambda r: r["pro_support_max"] >= 10))
STRATA_METHODS = ("full", "only:pro", "only:role_transfer", "only:wiki", "only:sim", "popularity", "random")


def by_pro_support(rows: list[dict], seed: int, boot_n: int) -> dict:
    """Accuracy by how much pro evidence the deck has for the threat: the largest support of any pro rule (threat ->
    deck card) in the event's deck. Tests 'pro evidence dominates where support is 5 or more' and 'role transfer / sim /
    wiki cover the rest'. The bootstrap resamples the games of each stratum."""
    out = {}
    for name, pred in STRATA:
        sub = [r for r in rows if pred(r)]
        if not sub:
            continue
        boot = GameBootstrap([r["game_id"] for r in sub], boot_n, seed)
        out[name] = {"events": len(sub), "games": boot.n_games, "methods": summarise(sub, boot, STRATA_METHODS)}
    return out


def held_by_rank(rows: list[dict], boot: GameBootstrap, method: str = "full") -> dict:
    """Share of events where the answer held, by where `method` ranked the pro's answer (best position in its tie
    group; 'none' = the answer had no evidence). Events with an unknown held flag are left out of each share."""
    buckets = {"rank 1": lambda r: r["answer_scored"] and r["rank_best"] == 1,
               "rank 2": lambda r: r["answer_scored"] and r["rank_best"] == 2,
               "rank 3": lambda r: r["answer_scored"] and r["rank_best"] == 3,
               "rank 4+": lambda r: r["answer_scored"] and r["rank_best"] >= 4,
               "no evidence": lambda r: not r["answer_scored"]}
    known = np.array([r["held"] is not None for r in rows], float)
    held = np.array([1.0 if r["held"] else 0.0 for r in rows])
    out = {}
    for name, test in buckets.items():
        sel = np.array([float(test(r["m"][method])) for r in rows])
        out[name] = {"events": int(sel.sum()), "held": boot.ci(held, mask=sel * known)}
    return out


def score_calibration(rows: list[dict], boot: GameBootstrap, method: str = "full") -> dict:
    """Held share by the score the method gave the pro's answer (0 = no evidence), plus the AUROC of that score for
    held vs not held (events with a known flag; ties count half)."""
    known = np.array([r["held"] is not None for r in rows], float)
    held = np.array([1.0 if r["held"] else 0.0 for r in rows])
    score = column(rows, method, "score")
    out = {"bins": {}}
    sel0 = (score <= 0).astype(float)
    out["bins"]["0 (none)"] = {"events": int(sel0.sum()), "held": boot.ci(held, mask=sel0 * known)}
    for lo, hi in SCORE_BINS:
        sel = ((score > lo) & (score <= hi)).astype(float)
        out["bins"][f"({lo:g}, {min(hi, 1.0):g}]"] = {"events": int(sel.sum()), "held": boot.ci(held, mask=sel * known)}
    pos, neg = score[(known > 0) & (held > 0)], score[(known > 0) & (held == 0)]
    if len(pos) and len(neg):
        gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        out["auroc_held"] = float(gt / (len(pos) * len(neg)))
    else:
        out["auroc_held"] = None
    out["n_pos"], out["n_neg"] = int(len(pos)), int(len(neg))
    return out


# --------------------------------------------------------------------------- mining (rules from train only)
def mining_module():
    from tools.data import mine_pro_answers as M
    return M


def aggregate_rules(events: list[dict], min_support: int) -> list[dict]:
    """answers.jsonl rows for `events` (the miner's own aggregate; it keeps train events only)."""
    return mining_module().aggregate_answers(events, min_support)


def leak_check(events: list[dict], shipped: list[dict], min_support: int = DEFAULT_SUPPORT) -> dict:
    """No val / test event may build a rule: the aggregate over all events equals the one over train events alone, and
    equals the shipped answers.jsonl; every rule's support is bounded by its train events."""
    train = [e for e in events if e["split"] == "train"]
    all_rules = aggregate_rules(events, min_support)
    train_rules = aggregate_rules(train, min_support)
    n_pair = Counter((e["threat"], e["answer"]) for e in train)
    return {"rules_all_events_equal_train_only": all_rules == train_rules,
            "rules_equal_shipped_answers_jsonl": all_rules == shipped,
            "n_rules": len(all_rules), "n_shipped_rules": len(shipped),
            "support_within_train_events": all(r["support"] <= n_pair[(r["threat"], r["answer"])] for r in all_rules),
            "events": dict(Counter(e["split"] for e in events))}


def remine(window_s: float, dataset: Path, out_dir: Path, force: bool = False) -> list[dict]:
    """Re-mine every answer event with the answer window `window_s` into `out_dir` (a temp dir under runs/, never
    dataset/ or db/); cached there for reruns. Returns the events."""
    out_dir = Path(out_dir).resolve()
    for guard in (Path(DATASET_DIR).resolve(), Path(DB_PATH).resolve().parent):
        if guard == out_dir or guard in out_dir.parents:
            raise ValueError(f"refusing to write re-mined rules into {out_dir}")
    path = out_dir / "answer_events.jsonl"
    if path.exists() and not force:
        return read_jsonl(path)
    M = mining_module()
    old = M.WINDOW_S
    M.WINDOW_S = float(window_s)
    try:
        events, _pairs, stats = M.mine_dataset(dataset)
    finally:
        M.WINDOW_S = old
    events.sort(key=lambda e: (e["split"], e["game_id"], e["threat_t"], e["threat"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    M.write_jsonl(path, events)
    M.write_jsonl(out_dir / "answers.jsonl", M.aggregate_answers(events))
    (out_dir / "mine_stats.json").write_text(json.dumps({"window_s": window_s, "stats": dict(stats)}, indent=1))
    return read_jsonl(path)


# --------------------------------------------------------------------------- the experiment
def decks_of(games: list[dict]) -> dict[str, list[str]]:
    return {g["game_id"]: list(g.get("own_deck") or []) for g in games}


def split_rows(rows: list[dict], split: str) -> list[dict]:
    return [r for r in rows if r["split"] == split]


def headline(kg: KG2, events: list[dict], decks: dict, wiki_kg, boot_n: int, seed: int) -> dict:
    pop = popularity_counts(events)
    ev = [e for e in events if e["split"] in EVAL_SPLITS]
    res = eval_rows(kg, ev, decks, pop, wiki_kg)
    out = {"skipped": res["skipped"], "max_score_gap_vs_answers_for": res["max_score_gap"], "splits": {}}
    pairs = [("full", "popularity"), ("full", "random"), ("full", "wiki_counters"), ("only:pro", "popularity"),
             ("popularity", "random"), ("full", "only:pro"), ("without:pro", "random")]
    pooled_rows = res["rows"]
    for split in (*EVAL_SPLITS, "val+test"):
        rows = pooled_rows if split == "val+test" else split_rows(pooled_rows, split)
        if not rows:
            continue
        boot = GameBootstrap([r["game_id"] for r in rows], boot_n, seed)
        by_source = {}
        for src in sorted({r["threat_source"] for r in rows if r["threat_source"]}):
            sub = [r for r in rows if r["threat_source"] == src]
            b = GameBootstrap([r["game_id"] for r in sub], boot_n, seed)
            by_source[src] = summarise(sub, b, ("full", "popularity", "random"))
        out["splits"][split] = {
            "events": len(rows), "games": boot.n_games, "mean_deck_size": float(np.mean([r["deck_size"] for r in rows])),
            "methods": summarise(rows, boot, METHODS), "paired": paired(rows, boot, pairs),
            "held_by_rank": held_by_rank(rows, boot), "score_calibration": score_calibration(rows, boot),
            "by_threat_source": by_source, "by_pro_support": by_pro_support(rows, seed, boot_n)}
    out["_rows"] = pooled_rows
    return out


def sweep(kg: KG2, canonical: list[dict], decks: dict, windows, supports, dataset: Path, rerun_dir: Path,
          force: bool, boot_n: int, seed: int) -> dict:
    """Support x window grid for the full score and the pro source alone. For each window the events are re-mined;
    rules = train events of that window with support >= n. 'fixed' evaluates on the canonical (8 s) val / test
    events; 'matched' on the val / test events re-mined with the same window as the rules."""
    saved = kg._answered
    cells, remined = [], {}
    try:
        for w in windows:
            events_w = remine(w, dataset, Path(rerun_dir) / f"ar_w{int(w)}", force)
            remined[w] = events_w
            pop = popularity_counts(events_w)
            for n in supports:
                rules = aggregate_rules(events_w, n)
                set_rules(kg, rules)
                cell = {"window_s": w, "min_support": n, "n_rules": len(rules),
                        "train_events": sum(1 for e in events_w if e["split"] == "train")}
                for mode, evs in (("fixed", canonical), ("matched", events_w)):
                    ev = [e for e in evs if e["split"] in EVAL_SPLITS]
                    res = eval_rows(kg, ev, decks, pop)
                    cell[mode] = {}
                    for split in EVAL_SPLITS:
                        rows = split_rows(res["rows"], split)
                        boot = GameBootstrap([r["game_id"] for r in rows], boot_n, seed)
                        cell[mode][split] = {"events": len(rows),
                                             **summarise(rows, boot, ("full", "only:pro", "popularity", "random"))}
                cells.append(cell)
    finally:
        kg._answered = saved
    return {"cells": cells, "events_per_window": {str(w): dict(Counter(e["split"] for e in evs))
                                                  for w, evs in remined.items()}}


def best_on_val(cells: list[dict], mode: str = "fixed", method: str = "full") -> dict | None:
    """The (window, support) cell with the best val MRR, and its test numbers (selection on val only)."""
    scored = [c for c in cells if c[mode].get("val", {}).get(method)]
    if not scored:
        return None
    best = max(scored, key=lambda c: c[mode]["val"][method]["mrr"]["mean"])
    return {"window_s": best["window_s"], "min_support": best["min_support"], "mode": mode, "method": method,
            "val_mrr": best[mode]["val"][method]["mrr"]["mean"], "test": best[mode]["test"][method]}


def run(args) -> dict:
    t0 = time.time()
    ds = Path(args.dataset)
    games = read_jsonl(ds / "games.jsonl")
    decks = decks_of(games)
    events = read_jsonl(ds / "kg2" / "answer_events.jsonl")
    shipped = read_jsonl(ds / "kg2" / "answers.jsonl")
    kg = KG2(args.kg2)
    try:
        from clashbot.kg import KnowledgeGraph
        wiki_kg = KnowledgeGraph(DB_PATH)
    except Exception as exc:                       # pragma: no cover - db/clashbot.sqlite missing
        print(f"warning: wiki baselines skipped ({exc})")
        wiki_kg = None
    out = {"version": "answer_rules_v1", "experiment": "E6 held-out evaluation of the mined answer rules",
           **{"git": git_head()}, "dataset": str(ds), "kg2": str(args.kg2), "bootstrap": args.boot, "seed": args.seed,
           "source_weights": dict(kg.source_weights),
           "leak_check": leak_check(events, shipped),
           "ties": "expected value over a random order of tied cards (no evidence = random)",
           "methods": METHOD_LABELS}
    h = headline(kg, events, decks, wiki_kg, args.boot, args.seed)
    rows = h.pop("_rows")
    out["headline"] = h
    out["held_out_games"] = {s: len({r["game_id"] for r in rows if r["split"] == s}) for s in EVAL_SPLITS}
    if not args.no_sweep:
        windows = [float(w) for w in args.windows.split(",") if w.strip()]
        supports = [int(n) for n in args.supports.split(",") if n.strip()]
        out["sweep"] = sweep(kg, events, decks, windows, supports, ds, Path(args.rerun_dir), args.force_remine,
                             args.boot, args.seed)
        out["sweep"]["best_on_val"] = {m: best_on_val(out["sweep"]["cells"], "fixed", m) for m in ("full", "only:pro")}
        w8 = remine(DEFAULT_WINDOW, ds, Path(args.rerun_dir) / f"ar_w{int(DEFAULT_WINDOW)}")
        out["sweep"]["window8_remine_equals_shipped_events"] = w8 == events
    out["seconds"] = round(time.time() - t0, 1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float))
    md = render_markdown(out)
    out_path.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out_path} and {out_path.with_suffix('.md')}")
    return out


# --------------------------------------------------------------------------- markdown
def _ci(d: dict, digits: int = 3) -> str:
    if not d or d.get("n", 1) == 0 or d["mean"] != d["mean"]:
        return "-"
    return f"{d['mean']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def _pct(d: dict) -> str:
    if not d or d["mean"] != d["mean"]:
        return "-"
    return f"{100 * d['mean']:.0f}% [{100 * d['lo']:.0f}, {100 * d['hi']:.0f}]"


def _dci(d: dict) -> str:
    if not d or d["mean"] != d["mean"]:
        return "-"
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] p={d.get('p', float('nan')):.3f}"


def render_markdown(out: dict) -> str:
    L: list[str] = ["# E6 - held-out evaluation of the mined answer rules (MAC-AR R1)", ""]
    g = out["git"]
    L.append(f"Run at HEAD `{(g.get('head') or '?')[:10]}`{' (+ uncommitted changes)' if g.get('dirty') else ''}; bootstrap "
             f"{out['bootstrap']} resamples over games (seed {out['seed']}); source weights {out['source_weights']}.")
    lc = out["leak_check"]
    L += ["", "## Setup and leakage check", "",
          "Rules are the train-split ANSWERED_BY edges (support >= 3, answer window 8 s). Each val / test event is one "
          "opponent play and the first defensive pro play after it; the task is to rank the pro's own deck (7-8 cards) "
          "for that threat. Ties are scored in expectation, so a method with no evidence equals random.", "",
          f"- events per split: {lc['events']}; held-out games: {out['held_out_games']}",
          f"- rules over all events equal rules over train events only: {lc['rules_all_events_equal_train_only']}; "
          f"equal the shipped answers.jsonl ({lc['n_rules']} vs {lc['n_shipped_rules']} rules): "
          f"{lc['rules_equal_shipped_answers_jsonl']}; every rule's support is within its train events: "
          f"{lc['support_within_train_events']}",
          f"- events left out because the pro's answer is not in the recorded deck: {out['headline']['skipped']}"]
    for split, s in out["headline"]["splits"].items():
        L += ["", f"## Headline - {split} ({s['events']} events, {s['games']} games, mean deck {s['mean_deck_size']:.2f})",
              "", "| method | hit@1 | hit@3 | MRR | coverage | hit@1 if covered |", "|---|---|---|---|---|---|"]
        for name, m in s["methods"].items():
            cov = ("-", "-") if name == "random" else (_pct(m["coverage"]), _ci(m["hit@1_covered"]))
            L.append(f"| {METHOD_LABELS[name]} | {_ci(m['hit@1'])} | {_ci(m['hit@3'])} | {_ci(m['mrr'])} | "
                     f"{cov[0]} | {cov[1]} |")
        L += ["", f"Paired differences ({split}; CI over games, p = bootstrap two-sided):", "",
              "| a - b | hit@1 | hit@3 | MRR |", "|---|---|---|---|"]
        for name, p in s["paired"].items():
            L.append(f"| {name} | {_dci(p['hit@1'])} | {_dci(p['hit@3'])} | {_dci(p['mrr'])} |")
        L += ["", f"Held by the rank the full score gave the pro's answer ({split}):", "",
              "| rank of the answer | events | share held |", "|---|---|---|"]
        for name, b in s["held_by_rank"].items():
            L.append(f"| {name} | {b['events']} | {_pct(b['held'])} |")
        cal = s["score_calibration"]
        L += ["", f"Held by the full score of the pro's answer ({split}); AUROC of the score for held vs tower-hit = "
              f"{'-' if cal['auroc_held'] is None else format(cal['auroc_held'], '.3f')} "
              f"({cal['n_pos']} held, {cal['n_neg']} not):", "",
              "| score | events | share held |", "|---|---|---|"]
        for name, b in cal["bins"].items():
            L.append(f"| {name} | {b['events']} | {_pct(b['held'])} |")
        L += ["", f"By the strongest pro rule available in the deck ({split}; hit@1 / MRR):", "",
              "| stratum | events | " + " | ".join(METHOD_LABELS[m] for m in STRATA_METHODS) + " |",
              "|---|---|" + "---|" * len(STRATA_METHODS)]
        for name, st in s["by_pro_support"].items():
            cells = " | ".join(f"{st['methods'][m]['hit@1']['mean']:.3f} / {st['methods'][m]['mrr']['mean']:.3f}"
                               for m in STRATA_METHODS)
            L.append(f"| {name} | {st['events']} | {cells} |")
        if len(s["by_threat_source"]) > 1:
            L += ["", f"By how the threat was found ({split}; play = the opponent was seen playing the card, unit = a unit "
                  "crossed with no play seen):", "", "| threat source | method | hit@1 | MRR | coverage |", "|---|---|---|---|---|"]
            for src, ms in s["by_threat_source"].items():
                for name in ("full", "popularity", "random"):
                    m = ms[name]
                    L.append(f"| {src} | {METHOD_LABELS[name]} | {_ci(m['hit@1'])} | {_ci(m['mrr'])} | {_pct(m['coverage'])} |")
    if "sweep" in out:
        sw = out["sweep"]
        L += ["", "## Sweep - support threshold n x answer window (rules from train events of that window)", "",
              "'fixed' evaluates on the canonical 8 s val / test events, so the rows are comparable; 'matched' re-mines the "
              "val / test events with the same window as the rules. Cells: MRR [95% CI] / hit@1.", ""]
        for mode in ("fixed", "matched"):
            for method, label in (("full", "full score"), ("only:pro", "pro source only")):
                L += [f"### {mode}, {label}", "",
                      "| window | n >= | rules | val events | val MRR | val hit@1 | test events | test MRR | test hit@1 |",
                      "|---|---|---|---|---|---|---|---|---|"]
                for c in sw["cells"]:
                    v, t = c[mode]["val"], c[mode]["test"]
                    L.append(f"| {c['window_s']:g} s | {c['min_support']} | {c['n_rules']} | {v['events']} | "
                             f"{_ci(v[method]['mrr'])} | {v[method]['hit@1']['mean']:.3f} | {t['events']} | "
                             f"{_ci(t[method]['mrr'])} | {t[method]['hit@1']['mean']:.3f} |")
                L.append("")
        b = sw["best_on_val"]["full"]
        if b:
            L.append(f"Best on val (full score, fixed events): window {b['window_s']:g} s, n >= {b['min_support']} "
                     f"(val MRR {b['val_mrr']:.3f}); its test MRR {_ci(b['test']['mrr'])}, hit@1 {_ci(b['test']['hit@1'])}.")
        L.append(f"Events per window (all splits): {sw['events_per_window']}; the re-mined 8 s events equal the shipped "
                 f"answer_events.jsonl: {out['sweep'].get('window8_remine_equals_shipped_events')}.")
    L += ["", "## Notes", "",
          "- Coverage = some deck card has evidence under the method; 'hit@1 if covered' is the accuracy on those events. "
          "Ties are expectations, so the uncovered events count as random.",
          "- Standard vs PCA-style confidence (AMIE) is not reported: a threat's candidate answers share one denominator, so "
          "the ranking inside a threat is the same under both, and answer_events.jsonl holds answered threats only (the "
          "unanswered ones are counted in the miner's stats, not stored).",
          "- 'KnowledgeGraph.counter_scores' adds the v1 ANSWERED_BY edges (mined from train states, window 4 s) to the wiki "
          "COUNTERS; the 'wiki COUNTERS only' row drops them.",
          "- The 'pro only' score is the support-shrunk held rate of the mined edge; the other three sources never see an "
          "answer event."]
    L.append(f"\nRuntime {out['seconds']} s.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--kg2", type=Path, default=KG2_PATH)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "data" / "answer_rules_v1.json")
    ap.add_argument("--windows", default=",".join(str(int(w)) for w in WINDOWS))
    ap.add_argument("--supports", default=",".join(str(n) for n in SUPPORTS))
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rerun-dir", type=Path, default=ROOT / "runs" / "rerun")
    ap.add_argument("--force-remine", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
