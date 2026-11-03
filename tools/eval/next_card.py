"""Opponent next-card prediction: popularity / copy / cycle baselines vs a SASRec-style transformer (MAC-R4).

  python3 tools/eval/next_card.py [--dataset DIR] [--methods popularity,copy_frequency,cycle,sasrec,sasrec+cycle]
          [--seeds 0,1,2] [--epochs 60] [--quick] [--out PATH]

One opponent play sequence per game (from states.jsonl); every position i >= 1 is a target: predict card_i from
plays[:i] as a ranked list of all vocab cards. Writes <out> (JSON) and <out>.md and prints the markdown table.
Ideas: CyGNet copy mechanism (repeat what was seen), the Clash Royale cycle rule (a card played in the last 4
plays cannot be in hand), SASRec (causal self-attention over item ids, tied output embedding).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.features import vocab as default_vocab  # noqa: E402
from clashbot.game_rules import is_playable, load_cards  # noqa: E402

SPLITS = ("train", "val", "test")
EVAL_SPLITS = ("val", "test")
METHODS = ("popularity", "copy_frequency", "cycle", "sasrec", "sasrec+cycle")
CYCLE_WS = (0, 1, 2, 3, 4)
KS = (1, 3, 5)
EARLY_DISTINCT = 8        # bucket: fewer than 8 distinct opponent cards seen so far -> "early"
COLLAPSE_S = 1.0          # same card re-read within this many seconds = perception double-read
MAX_LEN = 50
IGNORE = -100


# --------------------------------------------------------------------------- sequences

def opponent_sequences(records: list[dict], cards: dict) -> dict[str, dict]:
    """game_id -> {"split", "plays": [(t, card), ...]} from state.opponent.recent_plays and outcome.opponent_plays."""
    raw: dict[str, dict] = {}
    for r in records:
        g = raw.setdefault(r["game_id"], {"split": r.get("split"), "plays": {}})
        entries = list(((r.get("state") or {}).get("opponent") or {}).get("recent_plays") or [])
        out = r.get("outcome")
        if isinstance(out, dict):
            entries += list(out.get("opponent_plays") or [])
        for p in entries:
            card, t = p.get("card"), p.get("t")
            if t is None or not isinstance(card, str) or not is_playable(card, cards):
                continue
            key = (round(float(t), 1), card)
            if key not in g["plays"] or t < g["plays"][key]:
                g["plays"][key] = float(t)
    result: dict[str, dict] = {}
    for gid, g in raw.items():
        kept: list[tuple[float, str]] = []
        last: dict[str, float] = {}
        for (_, card), t in sorted(((k, t) for k, t in g["plays"].items()), key=lambda kv: (kv[1], kv[0][1])):
            if card in last and t - last[card] < COLLAPSE_S:
                continue
            last[card] = t
            kept.append((t, card))
        result[gid] = {"split": g["split"], "plays": kept}
    return result


def load_play_records(dataset: Path) -> list[dict]:
    """states.jsonl reduced to the fields opponent_sequences reads (the file is large)."""
    out = []
    with open(Path(dataset) / "states.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            o = r.get("outcome")
            out.append({"game_id": r["game_id"], "split": r.get("split"),
                        "state": {"opponent": {"recent_plays": ((r.get("state") or {}).get("opponent") or {})
                                               .get("recent_plays")}},
                        "outcome": {"opponent_plays": o.get("opponent_plays")} if isinstance(o, dict) else None})
    return out


@dataclass(frozen=True)
class Target:
    game_id: str
    split: str
    i: int
    history: tuple[str, ...]       # plays[:i] (cards)
    truth: str

    @property
    def bucket(self) -> str:
        return "early" if len(set(self.history)) < EARLY_DISTINCT else "late"


def make_targets(seqs: dict[str, dict]) -> list[Target]:
    """Every game and position i >= 1: predict card_i from plays[:i]."""
    out = []
    for gid in sorted(seqs):
        cards = [c for _, c in seqs[gid]["plays"]]
        for i in range(1, len(cards)):
            out.append(Target(gid, seqs[gid]["split"], i, tuple(cards[:i]), cards[i]))
    return out


# --------------------------------------------------------------------------- baselines

def popularity_order(train_card_lists: list[list[str]], vocab: list[str]) -> list[str]:
    """All vocab cards by train frequency desc, ties by slug."""
    cnt: Counter = Counter()
    for cs in train_card_lists:
        cnt.update(cs)
    return sorted(vocab, key=lambda c: (-cnt[c], c))


def copy_frequency(history, pop_order: list[str]) -> list[str]:
    """Seen cards by count desc, then most recent last play first; then unseen cards by popularity."""
    cnt: Counter = Counter(history)
    last = {c: i for i, c in enumerate(history)}
    seen = sorted(cnt, key=lambda c: (-cnt[c], -last[c]))
    return seen + [c for c in pop_order if c not in cnt]


def cycle(history, pop_order: list[str], w: int) -> list[str]:
    """copy_frequency with the cards among the last w plays moved to the end of the seen block."""
    base = copy_frequency(history, pop_order)
    n_seen = len(set(history))
    seen, rest = base[:n_seen], base[n_seen:]
    recent = set(history[-w:]) if w > 0 else set()
    return [c for c in seen if c not in recent] + [c for c in seen if c in recent] + rest


def ranked_from_scores(scores: np.ndarray, vocab: list[str]) -> list[str]:
    """Descending score; ties by vocab order (stable)."""
    return [vocab[j] for j in np.argsort(-scores, kind="stable")]


# --------------------------------------------------------------------------- metrics

def rank_of(ranked: list[str], truth: str) -> int:
    return ranked.index(truth) + 1


def rank_metrics(ranks: list[int]) -> dict:
    n = len(ranks)
    if n == 0:
        return {**{f"acc@{k}": 0.0 for k in KS}, "MRR": 0.0, "n": 0}
    a = np.asarray(ranks, dtype=np.float64)
    return {**{f"acc@{k}": float((a <= k).mean()) for k in KS}, "MRR": float((1.0 / a).mean()), "n": n}


def summarize(targets: list[Target], ranks: list[int]) -> dict:
    """Metrics over all targets plus the early / late buckets."""
    out = rank_metrics(ranks)
    for b in ("early", "late"):
        out[b] = rank_metrics([r for t, r in zip(targets, ranks) if t.bucket == b])
    return out


def aggregate_seeds(per_seed: list[dict]) -> dict:
    """Mean over seeds of every numeric metric (recursively); `std` (population) alongside at each level."""
    def agg(dicts: list[dict]) -> dict:
        out: dict = {}
        std: dict = {}
        for k in dicts[0]:
            if isinstance(dicts[0][k], dict):
                out[k] = agg([d[k] for d in dicts])
            elif k == "n":
                out[k] = dicts[0][k]
            else:
                vals = np.array([d[k] for d in dicts], dtype=np.float64)
                out[k], std[k] = float(vals.mean()), float(vals.std())
        out["std"] = std
        return out
    return agg(per_seed)


# --------------------------------------------------------------------------- SASRec

class _Block(nn.Module):
    """Pre-norm transformer block: x + attn(LN(x)); x + ff(LN(x))."""

    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.ln2(x)))


class SASRec(nn.Module):
    """Causal transformer over card ids (pad 0, cards 1..V); logits tied to the item embedding over the V cards."""

    def __init__(self, n_items: int, d: int = 64, n_blocks: int = 2, n_heads: int = 2, d_ff: int = 128,
                 dropout: float = 0.2, max_len: int = MAX_LEN) -> None:
        super().__init__()
        self.n_items, self.max_len = n_items, max_len
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(_Block(d, n_heads, d_ff, dropout) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.normal_(self.pos_emb.weight, std=0.1)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, T] ids -> [B, T, d]; position j sees only tokens <= j."""
        T = ids.shape[1]
        if T > self.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.max_len}")
        x = self.drop(self.item_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device)))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, T] ids -> [B, T, V] logits over the V cards (card id j+1 is column j)."""
        return self.encode(ids) @ self.item_emb.weight[1:].T


def _pad(rows: list[list[int]], fill: int = 0) -> torch.Tensor:
    width = max(len(r) for r in rows)
    return torch.tensor([r + [fill] * (width - len(r)) for r in rows], dtype=torch.long)


def training_windows(seq: list[int], max_len: int = MAX_LEN) -> list[list[int]]:
    """Windows of at most max_len + 1 ids (inputs = w[:-1], targets = w[1:]); long games get overlapping windows."""
    n = len(seq)
    if n < 2:
        return []
    if n <= max_len + 1:
        return [seq]
    starts = sorted(set(list(range(0, n - max_len - 1, max(1, max_len // 2))) + [n - max_len - 1]))
    return [seq[s:s + max_len + 1] for s in starts]


@torch.no_grad()
def sasrec_scores(model: SASRec, contexts: list[list[int]], batch_size: int = 512) -> np.ndarray:
    """Logits over the V cards for the next play after each context (last max_len ids kept) -> [n, V]."""
    model.eval()
    out = np.zeros((len(contexts), model.n_items), dtype=np.float32)
    for s in range(0, len(contexts), batch_size):
        chunk = [c[-model.max_len:] for c in contexts[s:s + batch_size]]
        ids = _pad(chunk)
        h = model.encode(ids)[torch.arange(len(chunk)), torch.tensor([len(c) - 1 for c in chunk])]
        out[s:s + batch_size] = (h @ model.item_emb.weight[1:].T).numpy()
    return out


def _acc1(model: SASRec, ctx: list[tuple[list[int], int]]) -> float:
    if not ctx:
        return 0.0
    scores = sasrec_scores(model, [c for c, _ in ctx])
    return float((scores.argmax(1) + 1 == np.array([t for _, t in ctx])).mean())


def train_sasrec(train_seqs: list[list[int]], val_ctx: list[tuple[list[int], int]], n_items: int, *,
                 d: int = 64, n_blocks: int = 2, n_heads: int = 2, d_ff: int = 128, dropout: float = 0.2,
                 max_len: int = MAX_LEN, lr: float = 1e-3, batch_size: int = 32, epochs: int = 60,
                 patience: int = 6, seed: int = 0) -> tuple[SASRec, dict]:
    """Next-card cross-entropy at every non-pad position; early stop on val acc@1 (best weights restored).

    val_ctx = [(context ids, true id), ...]. Without val targets all epochs run and the last weights are kept."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SASRec(n_items, d, n_blocks, n_heads, d_ff, dropout, max_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    windows = [w for s in train_seqs for w in training_windows(s, max_len)]
    best_acc, best_state, best_epoch, bad = -1.0, None, 0, 0
    ran = 0
    for epoch in range(1, epochs + 1):
        model.train()
        ran = epoch
        order = rng.permutation(len(windows))
        for s in range(0, len(order), batch_size):
            batch = [windows[j] for j in order[s:s + batch_size]]
            x = _pad([w[:-1] for w in batch])
            y = _pad([[t - 1 for t in w[1:]] for w in batch], fill=IGNORE)
            loss = nn.functional.cross_entropy(model(x).reshape(-1, n_items), y.reshape(-1), ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if not val_ctx:
            continue
        acc = _acc1(model, val_ctx)
        if acc > best_acc:
            best_acc, best_epoch, bad = acc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs_run": ran, "best_epoch": best_epoch if val_ctx else ran,
                   "val_acc@1": round(best_acc, 4) if val_ctx else None}


# --------------------------------------------------------------------------- evaluation driver

def _by_split(targets: list[Target]) -> dict[str, list[Target]]:
    return {s: [t for t in targets if t.split == s] for s in SPLITS}


def _ctx_ids(history, card_id: dict[str, int]) -> list[int]:
    return [card_id[c] for c in history]


def run(dataset: Path, methods: list[str], seeds: list[int], epochs: int, out: Path,
        vocab: list[str] | None = None, cards: dict | None = None, verbose: bool = True) -> dict:
    t0 = time.time()
    vocab = list(vocab) if vocab is not None else default_vocab()
    cards = cards if cards is not None else load_cards()
    card_id = {c: j + 1 for j, c in enumerate(vocab)}
    seqs = opponent_sequences(load_play_records(Path(dataset)), cards)
    seqs = {g: s for g, s in seqs.items() if s["split"] in SPLITS}
    targets = make_targets(seqs)
    by = _by_split(targets)
    train_lists = [[c for _, c in s["plays"]] for s in seqs.values() if s["split"] == "train"]
    pop = popularity_order(train_lists, vocab)

    result: dict = {
        "n_games": {s: sum(1 for g in seqs.values() if g["split"] == s and len(g["plays"]) >= 2) for s in SPLITS},
        "n_targets": {s: len(by[s]) for s in SPLITS},
        "oracle_seen": {s: round(float(np.mean([t.truth in t.history for t in by[s]])), 4) if by[s] else 0.0
                        for s in SPLITS},
    }

    def eval_lists(fn) -> dict:
        out_m = {}
        for s in EVAL_SPLITS:
            out_m[s] = summarize(by[s], [rank_of(fn(t), t.truth) for t in by[s]])
        return out_m

    # cycle@w sweep on val (cheap, also needed to pick w* for sasrec+cycle)
    cycle_by_w = {w: eval_lists(lambda t, w=w: cycle(t.history, pop, w)) for w in CYCLE_WS}
    val_acc = {w: cycle_by_w[w]["val"]["acc@1"] for w in CYCLE_WS}
    chosen_w = max(CYCLE_WS, key=lambda w: (val_acc[w], -w))
    result["cycle_w_val"] = {str(w): round(val_acc[w], 4) for w in CYCLE_WS}
    result["chosen_w"] = chosen_w
    result["cycle_by_w"] = {str(w): cycle_by_w[w] for w in CYCLE_WS}

    res: dict = {}
    if "popularity" in methods:
        res["popularity"] = eval_lists(lambda t: pop)
    if "copy_frequency" in methods:
        res["copy_frequency"] = eval_lists(lambda t: copy_frequency(t.history, pop))
    if "cycle" in methods:
        res["cycle"] = {**cycle_by_w[chosen_w], "w": chosen_w}

    if "sasrec" in methods or "sasrec+cycle" in methods:
        train_seqs = [_ctx_ids([c for _, c in s["plays"]], card_id) for s in seqs.values() if s["split"] == "train"]
        val_ctx = [(_ctx_ids(t.history[-MAX_LEN:], card_id), card_id[t.truth]) for t in by["val"]]
        per_seed: dict[str, list[dict]] = {"sasrec": [], "sasrec+cycle": []}
        train_info = {}
        for seed in seeds:
            model, info = train_sasrec(train_seqs, val_ctx, len(vocab), epochs=epochs, seed=seed)
            train_info[str(seed)] = info
            if verbose:
                print(f"sasrec seed {seed}: {info} ({time.time() - t0:.0f}s)", flush=True)
            scores = {s: sasrec_scores(model, [_ctx_ids(t.history[-MAX_LEN:], card_id) for t in by[s]])
                      for s in EVAL_SPLITS}
            plain: dict = {}
            masked: dict = {}
            for s in EVAL_SPLITS:
                ranks_p, ranks_m = [], []
                for t, row in zip(by[s], scores[s]):
                    ranks_p.append(rank_of(ranked_from_scores(row, vocab), t.truth))
                    row2 = row.copy()
                    for c in set(t.history[-chosen_w:] if chosen_w > 0 else ()):
                        row2[card_id[c] - 1] = -np.inf
                    ranks_m.append(rank_of(ranked_from_scores(row2, vocab), t.truth))
                plain[s], masked[s] = summarize(by[s], ranks_p), summarize(by[s], ranks_m)
            per_seed["sasrec"].append(plain)
            per_seed["sasrec+cycle"].append(masked)
        for m in ("sasrec", "sasrec+cycle"):
            if m in methods:
                res[m] = aggregate_seeds(per_seed[m])
                res[m]["seeds"] = list(seeds)
        result["sasrec_train"] = train_info
    result["methods"] = {m: res[m] for m in methods if m in res}
    result["config"] = {"vocab": len(vocab), "seeds": list(seeds), "epochs": epochs, "max_len": MAX_LEN,
                        "elapsed_s": round(time.time() - t0, 1)}

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    md = to_markdown(result)
    out.with_name(out.name + ".md").write_text(md)
    if verbose:
        print(md)
    return result


def _cell(m: dict, key: str) -> str:
    v = m[key]
    sd = m.get("std", {}).get(key)
    return f"{v:.3f} ± {sd:.3f}" if sd else f"{v:.3f}"


def to_markdown(result: dict) -> str:
    lines = [
        f"n_games {result['n_games']}  n_targets {result['n_targets']}  oracle_seen {result['oracle_seen']}",
        f"cycle@w val acc@1 {result['cycle_w_val']}  chosen w = {result['chosen_w']}",
        "",
        "| method | val acc@1 | val acc@3 | val MRR | test acc@1 | test acc@3 | test MRR |",
        "|---|---|---|---|---|---|---|"]
    for name, m in result["methods"].items():
        label = f"{name} (w={m['w']})" if name == "cycle" else (
            f"{name} (w={result['chosen_w']})" if name == "sasrec+cycle" else name)
        cells = [_cell(m[s], k) for s in EVAL_SPLITS for k in ("acc@1", "acc@3", "MRR")]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "| method | val early acc@1 | val late acc@1 | test early acc@1 | test late acc@1 |",
              "|---|---|---|---|---|"]
    for name, m in result["methods"].items():
        cells = [_cell(m[s][b], "acc@1") + f" (n={m[s][b]['n']})" for s in EVAL_SPLITS for b in ("early", "late")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--quick", action="store_true", help="3 epochs, seed 0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        ap.error(f"unknown methods {bad}; choose from {list(METHODS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    epochs = args.epochs
    if args.quick:
        epochs, seeds = 3, [0]
    out = args.out or (args.dataset / "eval" / "next_card.json")
    return run(args.dataset, methods, seeds, epochs, out)


if __name__ == "__main__":
    main()
