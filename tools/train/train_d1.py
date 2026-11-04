"""Fine-tune LiquidAI/d1-omni-600M on the pro decisions (D-039, D-022, D-026).

  python tools/train/train_d1.py --variant joint|factorized|act [--data DIR] [--out DIR] [--init ID_OR_PATH]
          [--epochs 3] [--lr 2e-5] [--weight-decay 0.01] [--warmup 0.05] [--max-tokens 6144] [--accum 4]
          [--train-layers 16] [--balance] [--limit N] [--val-limit N] [--device mps] [--seed 0]

Trains the text path (`model.encoder` top layers + `model.head`; fp32) with a differentiable copy of the
model's own text forward pass, a cross-entropy over each question's options, token-budget batches and
AdamW + warmup/cosine. After every epoch it scores the val file; the best epoch is restored, the per-key
temperatures are re-fitted on val, and the checkpoint is saved (and reloaded as a check) in --out.
Data: <dataset>/d1/<variant>/{train,val}.jsonl. Checkpoints live under db/ (gitignored, D-013).
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from clashbot.config import DATASET_DIR, DB_DIR  # noqa: E402

VARIANTS = ("joint", "factorized", "act")
PRIMARY = {"joint": "play", "factorized": "all_three", "act": "balanced_acc"}   # selects the best epoch
INIT_ID = "LiquidAI/d1-omni-600M"
COPY_FILES = ("modeling_d1.py", "prompt.py", "encoder.py", "vision.py", "audio.py", "tokenizer.json",
              "tokenizer_config.json", "LICENSE")
TEMPERATURE_GRID = np.arange(0.5, 3.0001, 0.05)
PAD_LOGIT = -1e4                                  # what the head writes at unused option slots


class Row(NamedTuple):
    """One question over one state: token ids, option-marker positions and the gold option."""
    ids: list[int]
    markers: list[int]
    qtype: int
    n_options: int
    target: int
    line_id: str
    name: str
    tkey: str = ""                                # mod.temperature_key(q), for the temperature re-fit


# ---------------------------------------------------------------------------------------------- data
def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    lines = []
    with open(path) as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
                if limit is not None and len(lines) >= limit:
                    break
    return lines


def build_rows(lines: list[dict], model, mod) -> list[Row]:
    """Every question of every line becomes one Row (a noul's target is in the model's [false, true] order)."""
    rows = []
    for line in lines:
        for name, raw in line["questions"].items():
            q = mod.as_question(raw)
            ids, markers = mod.encode(model.tokenizer, line["state"], q, model.config.max_length, None, False)
            answer = line["answers"][name]
            if q.type == "choice":
                target = list(q.criteria).index(answer)
            elif q.type == "noul":
                target = 1 if str(answer).lower() == "true" else 0
            else:
                raise ValueError(f"question {name!r}: type {q.type!r} is not trainable here")
            rows.append(Row(ids, markers, mod.QTYPES[q.type], q.options, target, line["id"], name,
                            mod.temperature_key(q)))
    return rows


def make_batches(rows: list[Row], max_tokens: int, rng: random.Random | None = None,
                 chunk: int = 64) -> list[list[int]]:
    """Row indices -> batches. Shuffle (when rng is given), split into chunks of 64, sort each chunk by length
    (longest first), then cut greedily so that len(batch) * longest <= max_tokens (an over-long row is alone)."""
    order = list(range(len(rows)))
    if rng is not None:
        rng.shuffle(order)
    batches = []
    for c in range(0, len(order), chunk):
        part = sorted(order[c:c + chunk], key=lambda i: -len(rows[i].ids))
        i = 0
        while i < len(part):
            longest, j = len(rows[part[i]].ids), i + 1
            while j < len(part) and (j - i + 1) * longest <= max_tokens:
                j += 1
            batches.append(part[i:j])
            i = j
    return batches


def class_weights(rows: list[Row]) -> dict[int, float]:
    """Inverse class frequency per target (normalised to mean 1 over classes); for the imbalanced act set."""
    counts: dict[int, int] = defaultdict(int)
    for r in rows:
        counts[r.target] += 1
    return {t: len(rows) / (len(counts) * c) for t, c in counts.items()}


# ---------------------------------------------------------------------------------------------- model
def set_trainable(model, train_layers: int) -> list[torch.nn.Parameter]:
    """Freeze everything, then enable the head and the top `train_layers` trunk layers (all 16 also the final
    norm). Embeddings, vision and audio stay frozen."""
    for p in model.parameters():
        p.requires_grad = False
    mods = [model.head]
    layers = list(model.encoder.layers)
    if train_layers > 0:
        mods.extend(layers[-train_layers:])
    if train_layers >= len(layers):
        mods.append(model.encoder.embedding_norm)
    for m in mods:
        for p in m.parameters():
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def option_logits(model, rows: list[Row]) -> list[torch.Tensor]:
    """The text branch of D1OmniModel._forward, with gradients: one (n_options,) logit vector per row."""
    device = model.encoder.embed_tokens.weight.device
    h = torch.nn.utils.rnn.pad_sequence(
        [model.encoder.embed_tokens(torch.tensor(r.ids, device=device)) for r in rows], batch_first=True)
    lengths = torch.tensor([len(r.ids) for r in rows], device=device)
    pad = torch.arange(h.shape[1], device=device)[None] < lengths[:, None]
    h = model.encoder(h, pad, torch.zeros(len(rows), dtype=torch.long, device=device))
    k = max(len(r.markers) for r in rows)
    marker_pos = torch.zeros(len(rows), k, dtype=torch.long, device=device)
    marker_mask = torch.zeros(len(rows), k, dtype=torch.bool, device=device)
    for i, r in enumerate(rows):
        marker_pos[i, :len(r.markers)] = torch.tensor(r.markers, device=device)
        marker_mask[i, :len(r.markers)] = True
    qtype = torch.tensor([r.qtype for r in rows], device=device)
    logits = model.head(h, pad, marker_pos, marker_mask, qtype)
    return [z[:r.n_options] for r, z in zip(rows, logits)]


def batch_loss(zs: list[torch.Tensor], rows: list[Row], weights: dict[int, float] | None = None) -> torch.Tensor:
    """Mean cross-entropy over the rows of a batch (weighted mean, sum(w * loss) / sum(w), with class weights)."""
    k = max(z.shape[0] for z in zs)
    logits = torch.stack([F.pad(z, (0, k - z.shape[0]), value=PAD_LOGIT) for z in zs])
    target = torch.tensor([r.target for r in rows], device=logits.device)
    per_row = F.cross_entropy(logits, target, reduction="none")
    if weights is None:
        return per_row.mean()
    w = torch.tensor([weights[r.target] for r in rows], device=logits.device, dtype=per_row.dtype)
    return (per_row * w).sum() / w.sum()


def lr_factor(step: int, total: int, warmup: float) -> float:
    """Linear warmup over warmup * total steps, then cosine decay to 0."""
    warm = max(1, round(warmup * total))
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


# ---------------------------------------------------------------------------------------------- evaluation
def softmax_nll(z: np.ndarray, target: int) -> float:
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()) - z[target])


@torch.no_grad()
def evaluate(model, rows: list[Row], max_tokens: int) -> tuple[float, list[np.ndarray]]:
    """Val loss (mean cross-entropy) and the raw (T=1) option logits of every row, in row order."""
    model.eval()
    logits: list[np.ndarray | None] = [None] * len(rows)
    for idx in make_batches(rows, max_tokens):
        for i, z in zip(idx, option_logits(model, [rows[i] for i in idx])):
            logits[i] = z.detach().float().cpu().numpy()
    loss = float(np.mean([softmax_nll(z, r.target) for z, r in zip(logits, rows)])) if rows else float("nan")
    return loss, logits


def compute_metrics(rows: list[Row], logits: list[np.ndarray], variant: str) -> dict:
    """Per-question accuracy; factorized adds all_three; act adds acc, balanced_acc and precision/recall of
    `false` (= wait)."""
    pred = [int(np.argmax(z)) for z in logits]
    hit = [float(p == r.target) for p, r in zip(pred, rows)]
    by_name: dict[str, list[float]] = defaultdict(list)
    by_line: dict[str, list[float]] = defaultdict(list)
    for r, h in zip(rows, hit):
        by_name[r.name].append(h)
        by_line[r.line_id].append(h)
    metrics: dict = {name: float(np.mean(v)) for name, v in sorted(by_name.items())}
    if variant == "factorized":
        metrics["all_three"] = float(np.mean([float(all(v)) for v in by_line.values()]))
    if variant == "act":
        gold = np.array([r.target for r in rows])
        guess = np.array(pred)
        recalls = [float((guess[gold == c] == c).mean()) for c in (0, 1) if (gold == c).any()]
        tp = float(((guess == 0) & (gold == 0)).sum())
        metrics.update({"acc": float(np.mean(hit)), "balanced_acc": float(np.mean(recalls)),
                        "precision_wait": tp / max(1, int((guess == 0).sum())),
                        "recall_wait": tp / max(1, int((gold == 0).sum()))})
    metrics["n_rows"], metrics["n_lines"] = len(rows), len(by_line)
    return metrics


def fit_temperatures(rows: list[Row], logits: list[np.ndarray]) -> dict[str, float]:
    """Per temperature key: the T in 0.5..3.0 (step 0.05) that minimises the mean NLL of softmax(z / T)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.tkey].append(i)
    fitted = {}
    for key, idx in groups.items():
        k = max(len(logits[i]) for i in idx)
        z = np.full((len(idx), k), -1e9)
        for j, i in enumerate(idx):
            z[j, :len(logits[i])] = logits[i]
        target = np.array([rows[i].target for i in idx])
        nll = []
        for t in TEMPERATURE_GRID:
            s = z / t
            m = s.max(axis=1, keepdims=True)
            lse = (m + np.log(np.exp(s - m).sum(axis=1, keepdims=True)))[:, 0]
            nll.append(float((lse - s[np.arange(len(idx)), target]).mean()))
        fitted[key] = round(float(TEMPERATURE_GRID[int(np.argmin(nll))]), 4)
    return fitted


# ---------------------------------------------------------------------------------------------- training
def sync(device: str) -> None:
    if str(device).startswith("mps"):
        torch.mps.synchronize()


def mps_bytes(device: str) -> int:
    """Memory the MPS driver holds right now (0 off mps)."""
    if not str(device).startswith("mps"):
        return 0
    fn = getattr(torch.mps, "driver_allocated_memory", None)
    return int(fn()) if fn else 0


def train(model, train_rows: list[Row], val_rows: list[Row], *, variant: str, epochs: int = 3, lr: float = 2e-5,
          weight_decay: float = 0.01, warmup: float = 0.05, max_tokens: int = 6144, accum: int = 4,
          balance: bool = False, seed: int = 0, device: str = "cpu", log=print) -> dict:
    """AdamW + warmup/cosine over the trainable parameters; scores val after every epoch, restores the best
    epoch (by PRIMARY[variant]) and returns {history, best_epoch, best_logits, peak_mps_bytes, ...}."""
    params = [p for p in model.parameters() if p.requires_grad]
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    plans = [make_batches(train_rows, max_tokens, rng) for _ in range(epochs)]
    total = sum(math.ceil(len(b) / accum) for b in plans)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warmup))
    weights = class_weights(train_rows) if balance else None
    primary = PRIMARY[variant]

    history, best, best_epoch, best_state, best_logits = [], -math.inf, 0, None, None
    peak, step, seen, window = 0, 0, 0, []
    batch_log: list[tuple[int, int, float]] = []              # (rows, tokens, seconds) per batch
    t_start = time.time()
    for epoch, batches in enumerate(plans):
        model.train()
        t_epoch, loss_sum, n_rows, epoch_rows_time = time.time(), 0.0, 0, 0.0
        for g in range(0, len(batches), accum):
            group = batches[g:g + accum]
            for idx in group:
                t0 = time.time()
                rows = [train_rows[i] for i in idx]
                loss = batch_loss(option_logits(model, rows), rows, weights)
                (loss / len(group)).backward()
                value = float(loss.detach())
                sync(device)
                dt = time.time() - t0
                loss_sum += value * len(rows)
                n_rows += len(rows)
                seen += len(rows)
                window.append(value)
                batch_log.append((len(rows), len(rows) * max(len(r.ids) for r in rows), dt))
                epoch_rows_time += dt
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            peak = max(peak, mps_bytes(device))
            if step == 1 or step % 50 == 0 or step == total:
                el = time.time() - t_start
                log(f"step {step}/{total}  loss {np.mean(window):.4f}  {seen / el:.1f} rows/s  "
                    f"elapsed {el:.0f}s  eta {el / step * (total - step):.0f}s  lr {sched.get_last_lr()[0]:.2e}")
                window = []
        train_seconds = time.time() - t_epoch

        val_loss, logits = evaluate(model, val_rows, max_tokens)
        if str(device).startswith("mps"):
            peak = max(peak, mps_bytes(device))
            torch.mps.empty_cache()
        metrics = compute_metrics(val_rows, logits, variant)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / max(1, n_rows), "val_loss": val_loss,
                        "val_metrics": metrics, "train_rows_per_s": n_rows / max(1e-9, epoch_rows_time),
                        "seconds": time.time() - t_epoch, "train_seconds": train_seconds})
        log(f"epoch {epoch + 1}/{epochs}  train_loss {history[-1]['train_loss']:.4f}  val_loss {val_loss:.4f}  "
            f"{primary} {metrics[primary]:.4f}  ({history[-1]['seconds']:.0f}s)")
        if best_state is None or metrics[primary] > best:
            best, best_logits = metrics[primary], logits
            best_state = {n: p.detach().cpu().clone() for n, p in named.items()}
            best_epoch = epoch + 1
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(best_state[n].to(p.device))
    warm = batch_log[1:] or batch_log                          # the first batch pays the kernel warm-up
    return {"history": history, "best_epoch": best_epoch, "best_logits": best_logits,
            "peak_mps_bytes": peak, "seconds": time.time() - t_start, "steps": step,
            "warm_rows_per_s": sum(b[0] for b in warm) / max(1e-9, sum(b[2] for b in warm)),
            "warm_tokens_per_s": sum(b[1] for b in warm) / max(1e-9, sum(b[2] for b in warm))}


# ---------------------------------------------------------------------------------------------- save / reload
def source_dir(init: str) -> Path:
    """The local directory of --init: itself, or the cached snapshot of a hub id."""
    if Path(init).is_dir():
        return Path(init)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(init, local_files_only=True))


def save_checkpoint(model, out: Path, src: Path) -> None:
    """save_pretrained + the remote code and tokenizer files, so `out` loads on its own."""
    from transformers import PretrainedConfig
    out.mkdir(parents=True, exist_ok=True)
    # d1's config keeps `max_length` (a context length); transformers' save check mistakes it for a generation
    # parameter and refuses to write the config
    with mock.patch.object(PretrainedConfig, "_get_generation_parameters", lambda self: {}):
        model.save_pretrained(out)
    for name in COPY_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    saved = json.loads((out / "config.json").read_text())
    if "auto_map" not in saved and (src / "config.json").exists():     # AutoModel needs it to find modeling_d1.py
        saved["auto_map"] = json.loads((src / "config.json").read_text()).get("auto_map", {})
        (out / "config.json").write_text(json.dumps(saved, indent=2))


def reload_check(out: Path, line: dict, expected: list[list[float]] | None = None) -> dict:
    """Reload `out` on the cpu and run system_one on a val line: every distribution must sum to 1 +- 1e-3.
    `expected` (the in-memory probabilities, [true, false] for a noul) gives the max abs difference."""
    from transformers import AutoModel
    reloaded = AutoModel.from_pretrained(out, trust_remote_code=True, dtype=torch.float32).eval()
    questions = line["questions"]
    answers = reloaded.system_one(line["state"], questions)["answers"]
    probs = reloaded.probabilities(line["state"], list(questions.values()))
    sums_ok = all(abs(sum(p) - 1.0) <= 1e-3 for p in probs)
    diff = None
    if expected is not None:
        diff = max(abs(a - b) for pe, pr in zip(expected, probs) for a, b in zip(pe, pr))
    del reloaded
    gc.collect()
    return {"reload_ok": bool(sums_ok and set(answers) == set(questions)), "reload_max_abs_diff": diff}


def expected_probabilities(model, rows: list[Row], temperatures: dict[str, float]) -> list[list[float]]:
    """In-memory probabilities of `rows`, in the order the reloaded model returns them (noul: [true, false])."""
    with torch.no_grad():
        model.eval()
        out = []
        for r, z in zip(rows, option_logits(model, rows)):
            p = (z.float() / temperatures.get(r.tkey, 1.0)).softmax(-1).tolist()
            out.append(p[::-1] if r.qtype == 2 else p)
        return out


# ---------------------------------------------------------------------------------------------- main
def fmt_table(history: list[dict], best_epoch: int) -> str:
    keys = [k for k in history[0]["val_metrics"] if k not in ("n_rows", "n_lines")]
    lines = ["| epoch | train_loss | val_loss | " + " | ".join(keys) + " | seconds |",
             "|---|---|---|" + "---|" * len(keys) + "---|"]
    for h in history:
        cells = " | ".join(f"{h['val_metrics'][k]:.4f}" for k in keys)
        mark = " *" if h["epoch"] == best_epoch else ""
        lines.append(f"| {h['epoch']}{mark} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | {cells} | "
                     f"{h['seconds']:.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--data", type=Path, default=None, help="default <dataset>/d1/<variant>")
    ap.add_argument("--out", type=Path, default=None, help="default db/models/d1-ft-<variant>")
    ap.add_argument("--init", default=INIT_ID, help="hub id (cached) or checkpoint dir")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--train-layers", type=int, default=16)
    ap.add_argument("--balance", action=argparse.BooleanOptionalAction, default=None,
                    help="inverse-class-frequency loss weights (default: on for act)")
    ap.add_argument("--limit", type=int, default=None, help="first N train lines")
    ap.add_argument("--val-limit", type=int, default=None, help="first N val lines")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    data = args.data or DATASET_DIR / "d1" / args.variant
    out = args.out or DB_DIR / "models" / f"d1-ft-{args.variant}"
    balance = args.variant == "act" if args.balance is None else args.balance
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t_start = time.time()

    from transformers import AutoModel
    src = source_dir(args.init)
    model = AutoModel.from_pretrained(src, trust_remote_code=True, dtype=torch.float32)   # fp32: bf16 flips answers
    mod = importlib.import_module(type(model).__module__)
    model.encoder.to(args.device)                              # vision / audio are unused and stay on the cpu
    model.head.to(args.device)
    trainable = sum(p.numel() for p in set_trainable(model, args.train_layers))
    print(f"trainable parameters: {trainable:,}  (train layers {args.train_layers}, device {args.device})")

    train_lines = read_jsonl(data / "train.jsonl", args.limit)
    val_lines = read_jsonl(data / "val.jsonl", args.val_limit)
    train_rows, val_rows = build_rows(train_lines, model, mod), build_rows(val_lines, model, mod)
    print(f"{args.variant}: {len(train_lines)} train lines / {len(train_rows)} rows, "
          f"{len(val_lines)} val lines / {len(val_rows)} rows, balance {balance}")

    res = train(model, train_rows, val_rows, variant=args.variant, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, warmup=args.warmup, max_tokens=args.max_tokens, accum=args.accum,
                balance=balance, seed=args.seed, device=args.device)
    print(f"best epoch {res['best_epoch']} ({PRIMARY[args.variant]} "
          f"{res['history'][res['best_epoch'] - 1]['val_metrics'][PRIMARY[args.variant]]:.4f})")

    temperatures = fit_temperatures(val_rows, res["best_logits"])    # best_logits = the restored weights on val
    model.config.temperatures.update(temperatures)
    print("temperatures:", temperatures)

    save_checkpoint(model, out, src)
    first = val_lines[0]
    first_rows = [r for r in val_rows if r.line_id == first["id"]]
    check = reload_check(out, first, expected_probabilities(model, first_rows, model.config.temperatures))
    print(f"saved {out}  reload_ok {check['reload_ok']}  max|diff| vs in-memory {check['reload_max_abs_diff']}")

    peak = res["peak_mps_bytes"]
    report = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {"balance": balance},
              "rows": {"train": len(train_rows), "val": len(val_rows),
                       "train_lines": len(train_lines), "val_lines": len(val_lines)},
              "trainable_params": trainable, "primary_metric": PRIMARY[args.variant],
              "epochs": res["history"], "best_epoch": res["best_epoch"], "temperatures": temperatures,
              "reload_ok": check["reload_ok"], "reload_max_abs_diff": check["reload_max_abs_diff"],
              "peak_mps_bytes": peak or None, "optimizer_steps": res["steps"],
              "warm_train_rows_per_s": res["warm_rows_per_s"], "warm_train_tokens_per_s": res["warm_tokens_per_s"],
              "loop_seconds": res["seconds"], "total_seconds": time.time() - t_start}
    (out / "train_report.json").write_text(json.dumps(report, indent=1))
    print("\n" + fmt_table(res["history"], res["best_epoch"]))
    print(f"\nwarm train throughput {res['warm_rows_per_s']:.1f} rows/s ({res['warm_tokens_per_s']:.0f} tokens/s)  "
          f"peak mps {peak / 2**30:.2f} GiB  total {report['total_seconds']:.0f}s\nwrote {out / 'train_report.json'}")
    return report


if __name__ == "__main__":
    main()
