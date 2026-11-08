"""E2 (MAC-AR R2): does d1-ft-joint-tac follow the notes and tactics / KG lines even when they are wrong?

  python3 tools/eval/context_conflict.py run    [--variant joint_tac] [--model db/models/d1-ft-joint-tac]
          [--splits val,test] [--conditions C0,C1,C2,C3,C4] [--limit N] [--cache-dir DIR] [--force]
  python3 tools/eval/context_conflict.py report [--variant joint_tac] [--cache-dir DIR] [--out docs/data/context_conflict_v1.json]

`run` sends the corrupted prompts of one or more (split, condition) pairs through `D1Ranker(backend="transformers")` and
caches the per-line probabilities under runs/rerun/ar_cc/<variant>/ (heavy: one d1 call per line, ~65 ms on MPS; the
caller wraps each invocation in the shared lock). `report` needs no model: it rebuilds every perturbation (they are
deterministic given --seed), joins them with the cached probabilities and writes <out> (JSON) and <out>.md.

Conditions (the line's option order, the gold key and its position are never touched; only text changes):
  C0 as exported (tactics: / KG: lines in the state, a note after every option)
  C1 notes and tactics / KG lines removed (the `joint` format the model was not trained on)
  C2 notes shuffled across the 4 options of the line
  C3 poisoned: the gold option's note swapped with a distractor's (when that makes the gold note worse), and the
     `tactics: best <card>` line renamed to the distractor's card
  C4 transplant: the tactics / KG block replaced by the block of a random line of another game
  C5 (edit line, IKE style) and the d1-2B variants are not run here (optional in the spec).
Contrastive decoding (CAD, AdaCAD) is computed from the cached C-condition and C1 probabilities, no extra d1 calls.
Read-only use of clashbot/ranker.py and clashbot/d1_format.py (D-051 code freeze).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.d1_format import JOINT_NAME, NOTE_SEP  # noqa: E402
from tools.eval.eval_answer_rules import GameBootstrap  # noqa: E402

CONDITIONS = ("C0", "C1", "C2", "C3", "C4")
CONDITION_LABELS = {"C0": "as exported", "C1": "notes and tactics / KG lines removed", "C2": "notes shuffled",
                    "C3": "poisoned (gold note swapped, wrong tactics card)", "C4": "tactics / KG block transplanted"}
CORRUPTED = ("C2", "C3", "C4")
SPLITS = ("val", "test")
BLOCK_PREFIXES = ("tactics:", "KG:")
FILLER_PREFIX = "filler:"
BEST_RE = re.compile(r"^(tactics: best )(\S+)( @.*)$")
CAD_ALPHAS = (0.0, 0.25, 0.5, 1.0)
CAD_CONTEXTS = ("C0", "C2", "C3", "C4")
ECE_BINS = 15
EPS = 1e-9
DEFAULT_MODEL = "db/models/d1-ft-joint-tac"
BOOT = 10_000


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def git_head() -> dict:
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"head": head, "dirty": dirty}
    except Exception:                              # pragma: no cover - not a git checkout
        return {"head": None, "dirty": None}


# --------------------------------------------------------------------------- parsing a line
def split_option(text: str) -> tuple[str, str | None]:
    """An option's criterion text -> (description, note or None); the note follows the first NOTE_SEP."""
    desc, sep, note = text.partition(NOTE_SEP)
    return desc, (note if sep else None)


def join_option(desc: str, note: str | None) -> str:
    return desc if note is None else f"{desc}{NOTE_SEP}{note}"


def informative(note: str | None) -> bool:
    """A note that says something about the option: anything but none / the `filler: ...` note."""
    return bool(note) and not note.startswith(FILLER_PREFIX)


def split_state(state: str) -> tuple[list[str], list[str]]:
    """(body lines, block lines): the block is every `tactics:` / `KG:` line, the body is the rest in order."""
    lines = state.split("\n")
    return ([l for l in lines if not l.startswith(BLOCK_PREFIXES)], [l for l in lines if l.startswith(BLOCK_PREFIXES)])


def join_state(body: list[str], block: list[str]) -> str:
    return "\n".join(body + block)


def best_card(block: list[str]) -> str | None:
    """The card named by the engine's `tactics: best <card> @...` line, if there is one."""
    for l in block:
        m = BEST_RE.match(l)
        if m:
            return m.group(2)
    return None


def option_card(key: str) -> str:
    return key.split("|")[0]


def parse_line(line: dict) -> dict:
    """The pieces every perturbation works on: option keys (d1 order), descriptions, notes, gold index, state parts."""
    crit = line["questions"][JOINT_NAME]["criteria"]
    keys = list(crit)
    descs, notes = zip(*(split_option(crit[k]) for k in keys))
    body, block = split_state(line["state"])
    return {"keys": keys, "descs": list(descs), "notes": list(notes), "gold": keys.index(line["answers"][JOINT_NAME]),
            "body": body, "block": block}


def note_status(p: dict) -> str:
    """Is the gold option's note right? right = gold carries an informative note; wrong = gold carries filler while a
    distractor carries an informative one (the notes point at the wrong option); none = every note is filler."""
    if informative(p["notes"][p["gold"]]):
        return "right"
    return "wrong" if any(informative(n) for i, n in enumerate(p["notes"]) if i != p["gold"]) else "none"


def tactics_status(p: dict) -> str:
    """Does the state's `tactics: best` line name the gold option's card?"""
    card = best_card(p["block"])
    if card is None:
        return "no best line"
    return "names the gold card" if card == option_card(p["keys"][p["gold"]]) else "names another card"


# --------------------------------------------------------------------------- perturbations
def rng_for(seed: int, cond: str, line_id: str) -> random.Random:
    return random.Random(f"{seed}|{cond}|{line_id}")


def fav_of_notes(notes: list[str | None]) -> list[int]:
    return [i for i, n in enumerate(notes) if informative(n)]


def perturb(line: dict, cond: str, seed: int = 0, pool: list[tuple[str, list[str]]] | None = None) -> dict:
    """The condition's version of one line: {"state", "criteria" (key -> text, same key order), "fav" (option indices the
    corrupted context points at), "changed" (does the text differ from C0), "info"}. `pool` = [(game_id, block lines)]
    of the lines a C4 transplant may draw from. Deterministic in (seed, cond, line id); the answers are not touched."""
    p = parse_line(line)
    keys, descs, notes, gold = p["keys"], p["descs"], list(p["notes"]), p["gold"]
    body, block = p["body"], list(p["block"])
    rng = rng_for(seed, cond, line["id"])
    fav: list[int] = []
    info: dict = {}
    if cond == "C0":
        fav = fav_of_notes(notes)
    elif cond == "C1":
        notes, block = [None] * len(keys), []
    elif cond == "C2":
        order = list(range(len(keys)))
        if len(set(notes)) > 1:                       # a shuffle of identical notes changes nothing: draw until it does
            for _ in range(50):
                rng.shuffle(order)
                if [notes[i] for i in order] != notes:
                    break
        notes = [notes[i] for i in order]
        fav = fav_of_notes(notes)
        info["order"] = order
    elif cond == "C3":
        d = poison_target(p, rng)
        info["distractor"] = d
        if d is not None:
            if informative(notes[gold]) and not informative(notes[d]):
                notes[gold], notes[d] = notes[d], notes[gold]        # the useful note moves to the wrong option
                info["swapped"] = True
            wrong_card = option_card(keys[d])
            block = [BEST_RE.sub(lambda m: f"{m.group(1)}{wrong_card}{m.group(3)}", l) if BEST_RE.match(l) else l
                     for l in block]
            fav = [d]
    elif cond == "C4":
        donors = [b for g, b in (pool or []) if g != line["game_id"] and b]
        if donors:
            block = list(rng.choice(donors))
        card = best_card(block)
        fav = [i for i, k in enumerate(keys) if card is not None and option_card(k) == card]
        info["card"] = card
    else:
        raise ValueError(f"unknown condition {cond!r}")
    criteria = {k: join_option(d, n) for k, d, n in zip(keys, descs, notes)}
    state = join_state(body, block)
    changed = state != line["state"] or criteria != line["questions"][JOINT_NAME]["criteria"]
    assert list(criteria) == keys and p["gold"] == keys.index(line["answers"][JOINT_NAME])    # gold untouched
    return {"state": state, "criteria": criteria, "fav": fav, "changed": changed, "info": info}


def poison_target(p: dict, rng: random.Random) -> int | None:
    """The distractor C3 points the context at: a card other than the gold card if there is one; among those, the one
    whose note is worse than the gold's when the gold's note is informative, else one that already carries an informative
    note (the notes then already point at it). None when the line has only gold-card distractors and no note to swap."""
    gold, keys, notes = p["gold"], p["keys"], p["notes"]
    others = [i for i in range(len(keys)) if i != gold]
    pool = [i for i in others if option_card(keys[i]) != option_card(keys[gold])] or others
    if informative(notes[gold]):
        better = [i for i in pool if not informative(notes[i])]
        pool = better or pool
    else:
        better = [i for i in pool if informative(notes[i])]
        pool = better or pool
    return rng.choice(pool) if pool else None


def make_pool(lines: list[dict]) -> list[tuple[str, list[str]]]:
    return [(l["game_id"], split_state(l["state"])[1]) for l in lines]


# --------------------------------------------------------------------------- running d1
def questions_of(line: dict, version: dict) -> dict:
    q = json.loads(json.dumps(line["questions"]))
    q[JOINT_NAME]["criteria"] = version["criteria"]
    return q


def run_condition(ranker, lines: list[dict], cond: str, seed: int = 0, pool=None, progress: int = 0) -> list[dict]:
    """d1's probabilities for every line under `cond`: [{id, game_id, probs, ms, fav, changed}]."""
    rows = []
    t0 = time.time()
    for i, line in enumerate(lines):
        v = perturb(line, cond, seed, pool)
        probs = ranker.probabilities(v["state"], questions_of(line, v))[JOINT_NAME]
        rows.append({"id": line["id"], "game_id": line["game_id"], "probs": [float(x) for x in probs],
                     "ms": round(float(getattr(ranker, "last_ms", 0.0)), 2), "fav": v["fav"], "changed": v["changed"]})
        if progress and (i + 1) % progress == 0:
            print(f"  {cond} {i + 1}/{len(lines)}  {time.time() - t0:.0f}s", flush=True)
    return rows


def cache_path(cache_dir: Path, variant: str, split: str, cond: str) -> Path:
    return Path(cache_dir) / variant / f"{split}_{cond}.json"


def load_lines(dataset: Path, variant: str, split: str, limit: int | None = None) -> list[dict]:
    lines = read_jsonl(Path(dataset) / "d1" / variant / f"{split}.jsonl")
    return lines[:limit] if limit else lines


def cmd_run(args) -> None:
    from clashbot.ranker import D1Ranker
    ranker = D1Ranker(model_path=args.model, backend="transformers", dtype=args.dtype)
    for split in args.splits.split(","):
        lines = load_lines(args.dataset, args.variant, split.strip(), args.limit)
        pool = make_pool(lines)
        for cond in args.conditions.split(","):
            cond = cond.strip()
            path = cache_path(args.cache_dir, args.variant, split.strip(), cond)
            if path.exists() and not args.force:
                print(f"{path} exists; skipped")
                continue
            t0 = time.time()
            rows = run_condition(ranker, lines, cond, args.seed, pool, progress=250)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"model": args.model, "variant": args.variant, "split": split.strip(),
                                       "condition": cond, "seed": args.seed, "limit": args.limit, "dtype": args.dtype,
                                       "git": git_head(), "seconds": round(time.time() - t0, 1), "rows": rows}))
            tmp.replace(path)
            acc = np.mean([float(int(np.argmax(r["probs"])) == parse_line(l)["gold"]) for r, l in zip(rows, lines)])
            print(f"{split} {cond}: {len(rows)} lines, acc {acc:.4f}, {time.time() - t0:.0f}s -> {path}", flush=True)


# --------------------------------------------------------------------------- statistics
def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired 0/1 correctness vectors: only discordant pairs count, p = 2 P(X <= min)
    under X ~ Binomial(n01 + n10, 1/2)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())     # a right b wrong, a wrong b right
    n = n10 + n01
    if n == 0:
        return {"a_only": n10, "b_only": n01, "p": 1.0}
    k = min(n10, n01)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n10, "b_only": n01, "p": p}


def ece(probs: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error of the top-1 confidence, equal-width bins."""
    conf = probs.max(1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def jsd_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits (0..1) per row of two probability matrices."""
    p, q = np.clip(p, EPS, 1.0), np.clip(q, EPS, 1.0)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log2(a) - np.log2(b))).sum(1)         # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cad_scores(p_ctx: np.ndarray, p_free: np.ndarray, alpha) -> np.ndarray:
    """Contrastive decoding scores (1 + a) log p(with context) - a log p(without), a a scalar or one value per row."""
    a = np.asarray(alpha, float).reshape(-1, 1) if np.ndim(alpha) else float(alpha)
    return (1.0 + a) * np.log(np.clip(p_ctx, EPS, 1.0)) - a * np.log(np.clip(p_free, EPS, 1.0))


def adacad_alpha(p_ctx: np.ndarray, p_free: np.ndarray) -> np.ndarray:
    """AdaCAD: the weight is the Jensen-Shannon divergence (bits) of the two distributions, one value per line."""
    return jsd_bits(p_ctx, p_free)


def follow_stats(pick_c0: np.ndarray, pick_x: np.ndarray, fav: list[list[int]], changed: list[bool],
                 fav_c0: list[list[int]]) -> dict:
    """Follow-rate of a corrupted condition. Active lines = the corruption changed the text, points at some options, and
    those options differ from the ones the C0 notes point at; among them the lines whose C0 pick is not already a favoured
    option. 'moved' = the condition's pick is a favoured option. Also the raw share of picks on a favoured option."""
    act = [i for i in range(len(pick_x)) if changed[i] and fav[i] and set(fav[i]) != set(fav_c0[i])]
    movable = [i for i in act if int(pick_c0[i]) not in fav[i]]
    moved = [i for i in movable if int(pick_x[i]) in fav[i]]
    on_fav = [i for i in act if int(pick_x[i]) in fav[i]]
    return {"active_lines": len(act), "movable_lines": len(movable), "moved": len(moved),
            "follow_rate": (len(moved) / len(movable)) if movable else None,
            "pick_on_favoured": (len(on_fav) / len(act)) if act else None,
            "c0_pick_on_favoured": (sum(int(pick_c0[i]) in fav[i] for i in act) / len(act)) if act else None,
            "indices": movable, "moved_indices": moved}


# --------------------------------------------------------------------------- report
class Cell:
    """The cached probabilities of one (split, condition), aligned with the split's lines."""

    def __init__(self, lines: list[dict], rows: list[dict]):
        self.probs = np.array([r["probs"] for r in rows], float)
        self.pick = self.probs.argmax(1)
        self.gold = np.array([parse_line(l)["gold"] for l in lines])
        self.correct = (self.pick == self.gold).astype(float)
        self.gold_prob = self.probs[np.arange(len(lines)), self.gold]
        self.fav = [r["fav"] for r in rows]
        self.changed = [r["changed"] for r in rows]
        self.ms = [r["ms"] for r in rows]


def load_cells(cache_dir: Path, variant: str, split: str, lines: list[dict], conds) -> dict[str, Cell]:
    out = {}
    for cond in conds:
        path = cache_path(cache_dir, variant, split, cond)
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["rows"]
        if [r["id"] for r in rows] != [l["id"] for l in lines[:len(rows)]]:
            raise ValueError(f"{path}: line ids do not match {split}.jsonl")
        out[cond] = Cell(lines[:len(rows)], rows)
    return out


def condition_table(cells: dict[str, Cell], boot: GameBootstrap) -> dict:
    out = {}
    base = cells.get("C0")
    for cond, c in cells.items():
        row = {"n": len(c.correct), "acc": boot.ci(c.correct), "mean_gold_prob": float(c.gold_prob.mean()),
               "ece": ece(c.probs, c.correct), "mean_confidence": float(c.probs.max(1).mean()),
               "changed_lines": int(sum(c.changed))}
        if base is not None and cond != "C0":
            row["vs_c0"] = {**boot.diff(c.correct, base.correct), **mcnemar_exact(c.correct > 0, base.correct > 0),
                            "pick_changed": float((c.pick != base.pick).mean())}
        out[cond] = row
    return out


def follow_table(cells: dict[str, Cell], boot: GameBootstrap) -> dict:
    out = {}
    base = cells.get("C0")
    if base is None:
        return out
    for cond in CORRUPTED:
        c = cells.get(cond)
        if c is None:
            continue
        f = follow_stats(base.pick, c.pick, c.fav, c.changed, base.fav)
        idx = f.pop("indices")
        moved = set(f.pop("moved_indices"))
        mask = np.zeros(len(c.pick))
        mask[idx] = 1.0
        hit = np.array([1.0 if i in moved else 0.0 for i in range(len(c.pick))])
        f["follow_rate_ci"] = boot.ci(hit, mask=mask)
        out[cond] = f
    return out


def strata_table(cells: dict[str, Cell], lines: list[dict], boot: GameBootstrap) -> dict:
    """Accuracy per condition within strata defined on the C0 context: is the gold option's note right / wrong / absent
    and does the tactics `best` line name the gold card. Lines are the same under every condition (a paired view)."""
    parsed = [parse_line(l) for l in lines[:len(next(iter(cells.values())).correct)]]
    out = {}
    for name, fn in (("note", note_status), ("tactics_best", tactics_status)):
        labels = [fn(p) for p in parsed]
        out[name] = {}
        for lab in sorted(set(labels)):
            mask = np.array([1.0 if x == lab else 0.0 for x in labels])
            out[name][lab] = {"n": int(mask.sum()), **{cond: boot.ci(c.correct, mask=mask) for cond, c in cells.items()}}
    return out


def cad_rows(ctx: Cell, free: Cell) -> dict[str, np.ndarray]:
    """Correctness vectors of contrastive decoding of `ctx` against the context-free probabilities: one per alpha of the
    grid and AdaCAD (alpha = JSD of the two distributions per line)."""
    rows = {f"a={a:g}": (cad_scores(ctx.probs, free.probs, a).argmax(1) == ctx.gold).astype(float) for a in CAD_ALPHAS}
    ada = adacad_alpha(ctx.probs, free.probs)
    rows["adacad"] = (cad_scores(ctx.probs, free.probs, ada).argmax(1) == ctx.gold).astype(float)
    return rows


def cad_report(cells_by: dict, boots: dict) -> dict:
    """CAD / AdaCAD for each context condition against C1: accuracy with CIs on val and test, the paired delta against
    the plain context (a = 0) and against C0, and the alpha of the grid that is best on val (ties: the smaller)."""
    out: dict = {}
    grid = [f"a={a:g}" for a in CAD_ALPHAS]
    for ctx in CAD_CONTEXTS:
        if not all(ctx in cells_by.get(s, {}) and "C1" in cells_by[s] and "C0" in cells_by[s] for s in SPLITS):
            continue
        rows = {s: cad_rows(cells_by[s][ctx], cells_by[s]["C1"]) for s in SPLITS}
        tuned = max(grid, key=lambda k: (rows["val"][k].mean(), -grid.index(k)))
        out[ctx] = {"tuned_on_val": tuned,
                    "mean_adacad_alpha": {s: float(adacad_alpha(cells_by[s][ctx].probs, cells_by[s]["C1"].probs).mean())
                                          for s in SPLITS}}
        for s in SPLITS:
            c0 = cells_by[s]["C0"].correct
            out[ctx][s] = {k: {"acc": boots[s].ci(v), "vs_plain": boots[s].diff(v, rows[s]["a=0"]),
                               "vs_c0": boots[s].diff(v, c0)} for k, v in rows[s].items()}
    return out


def build_report(cache_dir: Path, variant: str, dataset: Path, boot_n: int, seed: int,
                 stored_eval: Path | None = None) -> dict:
    out: dict = {"version": "context_conflict_v1", "experiment": "E2 context-conflict test of the d1 ranker",
                 "variant": variant, "git": git_head(), "bootstrap": boot_n, "seed": seed, "conditions": CONDITION_LABELS}
    cells_by, lines_by, boots = {}, {}, {}
    meta = {}
    for split in SPLITS:
        lines = load_lines(dataset, variant, split)
        cells = load_cells(cache_dir, variant, split, lines, CONDITIONS)
        if not cells:
            continue
        n = len(next(iter(cells.values())).correct)
        lines_by[split], cells_by[split] = lines[:n], cells
        boots[split] = GameBootstrap([l["game_id"] for l in lines[:n]], boot_n, seed)
        meta[split] = {"lines": n, "games": boots[split].n_games, "conditions_run": sorted(cells)}
    out["runs"] = meta
    per_split = {}
    for split, cells in cells_by.items():
        lines = lines_by[split]
        entry = {"conditions": condition_table(cells, boots[split]),
                 "follow": follow_table(cells, boots[split])}
        if "C0" in cells:
            entry["strata"] = strata_table(cells, lines, boots[split])
            p0 = [parse_line(l) for l in lines]
            entry["c0_context"] = {"note": _counts([note_status(p) for p in p0]),
                                   "tactics_best": _counts([tactics_status(p) for p in p0]),
                                   "poisonable_c3": sum(1 for c in cells["C3"].changed if c) if "C3" in cells else None}
        per_split[split] = entry
    out["splits"] = per_split
    if "C0" in cells_by.get("val", {}) and "C0" in cells_by.get("test", {}):
        pooled_lines = lines_by["val"] + lines_by["test"]
        pooled = {c: _concat(cells_by["val"][c], cells_by["test"][c]) for c in CONDITIONS
                  if c in cells_by["val"] and c in cells_by["test"]}
        pb = GameBootstrap([l["game_id"] for l in pooled_lines], boot_n, seed)
        out["splits"]["val+test"] = {"conditions": condition_table(pooled, pb), "follow": follow_table(pooled, pb),
                                     "strata": strata_table(pooled, pooled_lines, pb)}
    cad = cad_report(cells_by, boots)
    if cad:
        out["cad"] = cad
    if stored_eval is not None and Path(stored_eval).exists() and "C0" in cells_by.get("test", {}):
        out["c0_vs_stored_eval"] = compare_stored(Path(stored_eval), cells_by, lines_by)
    return out


def _counts(labels: list[str]) -> dict:
    return {k: labels.count(k) for k in sorted(set(labels))}


def _concat(a: Cell, b: Cell) -> Cell:
    c = Cell.__new__(Cell)
    c.probs = np.concatenate([a.probs, b.probs])
    c.pick, c.gold = np.concatenate([a.pick, b.pick]), np.concatenate([a.gold, b.gold])
    c.correct = np.concatenate([a.correct, b.correct])
    c.gold_prob = np.concatenate([a.gold_prob, b.gold_prob])
    c.fav, c.changed, c.ms = a.fav + b.fav, a.changed + b.changed, a.ms + b.ms
    return c


def compare_stored(path: Path, cells_by: dict, lines_by: dict) -> dict:
    """C0 against the stored d1_eval file of the same model (same ids): pick agreement and the accuracy it reports."""
    stored = {e["id"]: e for e in json.loads(path.read_text()).get("per_example", [])}
    out = {"file": str(path)}
    for split, cells in cells_by.items():
        lines = lines_by[split]
        pairs = [(i, stored[l["id"]]) for i, l in enumerate(lines) if l["id"] in stored]
        keys = [list(l["questions"][JOINT_NAME]["criteria"]) for l in lines]
        agree = [keys[i][int(cells["C0"].pick[i])] == e["pred"] for i, e in pairs]
        out[split] = {"lines_matched": len(pairs), "pick_agreement": float(np.mean(agree)) if agree else None,
                      "stored_acc": float(np.mean([e["pred"] == e["gold"] for _, e in pairs])) if pairs else None,
                      "rerun_acc": float(cells["C0"].correct.mean())}
    return out


# --------------------------------------------------------------------------- markdown
def _ci(d: dict, digits: int = 3) -> str:
    if not d or d.get("mean") != d.get("mean"):
        return "-"
    return f"{d['mean']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def _dci(d: dict) -> str:
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def render_markdown(rep: dict) -> str:
    g = rep["git"]
    L = ["# E2 - context-conflict test of d1-ft-joint-tac (MAC-AR R2)", "",
         f"Variant `{rep['variant']}`; run at HEAD `{(g.get('head') or '?')[:10]}`{' (+ uncommitted changes)' if g.get('dirty') else ''}; "
         f"{rep['bootstrap']} bootstrap resamples over games (seed {rep['seed']}); lines {rep['runs']}.", "",
         "Conditions: " + "; ".join(f"{k} = {v}" for k, v in rep["conditions"].items()) + ".", ""]
    for split, s in rep["splits"].items():
        L += [f"## Accuracy by condition - {split}", "",
              "| condition | n | accuracy [95% CI] | vs C0 | McNemar p (right only here / right only in C0) | picks changed "
              "| mean P(gold) | ECE | mean confidence |",
              "|---|---|---|---|---|---|---|---|---|"]
        for cond, r in s["conditions"].items():
            v = r.get("vs_c0")
            L.append(f"| {cond} | {r['n']} | {_ci(r['acc'])} | {_dci(v) if v else '-'} | "
                     f"{('%.4f (%d/%d)' % (v['p'], v['a_only'], v['b_only'])) if v else '-'} | "
                     f"{('%.0f%%' % (100 * v['pick_changed'])) if v else '-'} | {r['mean_gold_prob']:.3f} | {r['ece']:.3f} | "
                     f"{r['mean_confidence']:.3f} |")
        if s.get("follow"):
            L += ["", f"Follow-rate - {split}: among lines where the corruption points at options other than the C0 notes "
                  "and the C0 pick is not already one of them, the share whose pick moves to a favoured option.", "",
                  "| condition | active lines | movable | moved | follow-rate [95% CI] | pick on favoured | C0 pick on favoured |",
                  "|---|---|---|---|---|---|---|"]
            for cond, f in s["follow"].items():
                pf = "-" if f["pick_on_favoured"] is None else f"{f['pick_on_favoured']:.3f}"
                p0 = "-" if f["c0_pick_on_favoured"] is None else f"{f['c0_pick_on_favoured']:.3f}"
                L.append(f"| {cond} | {f['active_lines']} | {f['movable_lines']} | {f['moved']} | {_ci(f['follow_rate_ci'])} | "
                         f"{pf} | {p0} |")
        for name, title in (("note", "gold option's note (C0): right = informative, wrong = filler while a distractor is "
                                     "informative, none = all filler"),
                            ("tactics_best", "state's `tactics: best` card vs the gold card (C0)")):
            st = (s.get("strata") or {}).get(name)
            if not st:
                continue
            conds = [c for c in CONDITIONS if c in s["conditions"]]
            L += ["", f"Accuracy by {title} - {split}:", "", "| stratum | n | " + " | ".join(conds) + " |",
                  "|---|---|" + "---|" * len(conds)]
            for lab, row in st.items():
                L.append(f"| {lab} | {row['n']} | " + " | ".join(_ci(row[c]) for c in conds) + " |")
        if s.get("c0_context"):
            L += ["", f"C0 context - {split}: note status {s['c0_context']['note']}; tactics best line "
                  f"{s['c0_context']['tactics_best']}; lines C3 changed {s['c0_context']['poisonable_c3']}."]
        L.append("")
    if "cad" in rep:
        L += ["## Contrastive decoding (CAD) and AdaCAD against C1", "",
              "score = (1 + a) log p(context) - a log p(no context); a from the grid, tuned on val; AdaCAD a = Jensen-Shannon "
              "divergence (bits) of the two distributions. Delta columns are paired bootstrap differences (CI over games).", ""]
        for ctx, c in rep["cad"].items():
            ada = c["mean_adacad_alpha"]
            L += [f"### context {ctx} (a tuned on val: {c['tuned_on_val']}; mean AdaCAD a {ada['val']:.3f} val / "
                  f"{ada['test']:.3f} test)", "",
                  "| decoding | val acc | test acc | test vs plain | test vs C0 |", "|---|---|---|---|---|"]
            for k in c["val"]:
                L.append(f"| {k} | {_ci(c['val'][k]['acc'])} | {_ci(c['test'][k]['acc'])} | {_dci(c['test'][k]['vs_plain'])} | "
                         f"{_dci(c['test'][k]['vs_c0'])} |")
            L.append("")
    if "c0_vs_stored_eval" in rep:
        L += ["## Reproduction check", "", f"C0 against the stored eval file: {json.dumps(rep['c0_vs_stored_eval'])}"]
    L += ["", "## How to read the tables", "",
          "- A model that follows its context loses accuracy in C3 and C4 and gains it in the 'right' note stratum only while "
          "the notes are present (C0), so a gap between strata that is also there in C1 (no notes, no tactics / KG lines) comes "
          "from the lines, not from reading the notes.",
          "- Follow-rate is compared with the C0 column ('C0 pick on favoured'): the same share of picks lands on the favoured "
          "options when the context is not corrupted, so only the difference between the two is movement caused by the context.",
          "- C3 is applied where it can bite: the gold note moves only when it is informative and a distractor's is not; lines "
          "with nothing to corrupt are left as exported (`changed lines` above).",
          "- CAD uses the C1 probabilities as the context-free distribution; no extra d1 call."]
    return "\n".join(L) + "\n"


def cmd_report(args) -> dict:
    rep = build_report(args.cache_dir, args.variant, args.dataset, args.boot, args.seed,
                       args.dataset / "eval" / "d1_eval_transformers_d1-ft-joint-tac_on_kg.json")
    md = render_markdown(rep)
    if args.replicate:                                  # the same run on another export of the lines (e.g. joint_tac_v2)
        rep["replicate"] = build_report(args.cache_dir, args.replicate, args.dataset, args.boot, args.seed)
        sub = render_markdown(rep["replicate"])
        md += "\n" + "\n".join("#" + l if l.startswith("#") else l for l in sub.split("\n"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float))
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out} and {out.with_suffix('.md')}")
    return rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--dataset", type=Path, default=DATASET_DIR)
        sp.add_argument("--variant", default="joint_tac")
        sp.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "rerun" / "ar_cc")
        sp.add_argument("--seed", type=int, default=0)
        if name == "run":
            sp.add_argument("--model", default=str(ROOT / DEFAULT_MODEL))
            sp.add_argument("--dtype", default="auto")
            sp.add_argument("--splits", default=",".join(SPLITS))
            sp.add_argument("--conditions", default=",".join(CONDITIONS))
            sp.add_argument("--limit", type=int, default=None)
            sp.add_argument("--force", action="store_true")
        else:
            sp.add_argument("--boot", type=int, default=BOOT)
            sp.add_argument("--replicate", default=None, help="a second variant (cached runs) to report after the first")
            sp.add_argument("--out", type=Path, default=ROOT / "docs" / "data" / "context_conflict_v1.json")
    args = ap.parse_args(argv)
    cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    main()
