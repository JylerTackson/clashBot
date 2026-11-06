"""Fine-tune LiquidAI d1 on the pro decisions: d1-omni-600M (D-039, D-022, D-026) or d1-2B (D-046, MAC-AE E2).

  python tools/train/train_d1.py --variant joint|factorized|act [--data DIR] [--out DIR] [--init ID_OR_PATH]
          [--epochs 3] [--lr 2e-5] [--weight-decay 0.01] [--warmup 0.05] [--max-tokens N] [--accum 4]
          [--train-layers N] [--balance] [--limit N] [--val-limit N] [--device mps] [--seed 0]
          [--fp32] [--grad-ckpt] [--max-gb 26] [--estimate-only]          (the last four: d1-2B)

Trains the text path with a differentiable copy of the model's own text forward pass, a cross-entropy over each
question's options, token-budget batches and AdamW + warmup/cosine. After every epoch it scores the val file;
the best epoch is restored, the per-key temperatures are re-fitted on val, and the checkpoint is saved (and
reloaded as a check) in --out. Data: <dataset>/d1/<variant>/{train,val}.jsonl. Checkpoints live under db/
(gitignored, D-013).

Two architectures, detected from the config of --init (`detect_architecture`):
  d1_omni  d1-omni-600M: `model.encoder` top layers + `model.head`, fp32 (the original path, MAC-T §0).
  lfm3     d1-2B (`Lfm3Adapter`): the LFM3-VL language model's top --train-layers of 24 + the final norm + the
           option-token rows of the tied LM head; frozen weights bf16, trainable fp32 (all fp32 with --fp32).
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
ARCHS = ("d1_omni", "lfm3")
DEFAULT_MAX_TOKENS = {"d1_omni": 6144, "lfm3": 4096}
COPY_SUFFIXES = (".py", ".json", ".jinja")        # d1-2B: every remote-code, config, template and tokenizer file
KEEP_SAVED = ("config.json", "generation_config.json", "model.safetensors.index.json")   # written by save_pretrained
GB = 2 ** 30


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
    extra: object = None                          # d1-2B: Lfm3Extra (the attention plan and the readout table)


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


def adapter_of(model):
    """The `Lfm3Adapter` attached to a d1-2B model (`model.d1_adapter`), else None (d1-omni-600M)."""
    return getattr(model, "d1_adapter", None)


def build_rows(lines: list[dict], model, mod) -> list[Row]:
    """Every question of every line becomes one Row (a noul's target is in the model's [false, true] order)."""
    if adapter_of(model) is not None:
        return adapter_of(model).build_rows(lines)
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
    norm). Embeddings, vision and audio stay frozen. (d1-2B: `Lfm3Adapter.set_trainable`.)"""
    if adapter_of(model) is not None:
        return adapter_of(model).set_trainable(train_layers)
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
    if adapter_of(model) is not None:
        return adapter_of(model).option_logits(rows)
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


# ---------------------------------------------------------------------------------------------- d1-2B (lfm3)
def detect_architecture(config) -> str:
    """"lfm3" (d1-2B: an LFM3-VL language model read through its LM head) or "d1_omni" (d1-omni-600M: encoder +
    scorer head), from a checkpoint's config.json (a dict or a config object)."""
    cfg = config if isinstance(config, dict) else config.to_dict()
    text = cfg.get("text_config") or {}
    text = text if isinstance(text, dict) else text.to_dict()
    kinds = [str(cfg.get("model_type", "")), str(text.get("model_type", ""))]
    names = " ".join(str(a) for a in cfg.get("architectures") or [])
    if any(k.startswith("lfm3") for k in kinds) or "Lfm3" in names:
        return "lfm3"
    if kinds[0] == "d1_omni" or "D1Omni" in names:
        return "d1_omni"
    raise ValueError(f"unknown d1 architecture (model_type {kinds[0]!r}, text {kinds[1]!r}, {names!r}); "
                     f"expected one of {ARCHS}")


def read_config(src: Path) -> dict:
    return json.loads((Path(src) / "config.json").read_text())


class Lfm3Extra(NamedTuple):
    """What a d1-2B row needs beyond its ids: the attention plan and the readout table of its question."""
    state_len: int | None                         # tokens that read each other both ways (None: causal)
    question: list[int] | None                    # each token's question under `set` attention (-1: state)
    tokens: tuple[int, ...]                       # the vocabulary ids the options read
    forms: np.ndarray                             # (n_options, width): positions in `tokens`, best form per option


class Lfm3Codec:
    """The remote code of a d1 LFM3 checkpoint as training needs it: `prompt.py`, `runner.py`, `attention.py` and
    `calibration.py` of the checkpoint's own package, so a training row is exactly the prompt `system_one` serves
    (`SlotReader._sequences`, a lone question) and its readout table is the one `system_one` reads."""

    def __init__(self, model, src: Path):
        from transformers import AutoTokenizer
        pkg = type(model).__module__.rpartition(".")[0]
        self.prompt = importlib.import_module(f"{pkg}.prompt")
        self.runner = importlib.import_module(f"{pkg}.runner")
        self.attn = importlib.import_module(f"{pkg}.attention")
        self.calib = importlib.import_module(f"{pkg}.calibration")
        self.tokenizer = AutoTokenizer.from_pretrained(str(src))
        self.pad_id = self.tokenizer.pad_token_id or 0
        bos = getattr(self.tokenizer, "bos_token", None)
        reader = self.runner.SlotReader()                 # SystemOne.__init__ without its model (and its fusing)
        reader.tokenizer, reader.bos = self.tokenizer, bos if isinstance(bos, str) else ""
        reader.lead = self.prompt.default_lead(model.config.model_type, self.tokenizer)
        reader.state_style, reader.system = self.prompt.DEFAULT_STATE_STYLE, self.prompt.DEFAULT_SYSTEM
        reader.option_style, reader.calibration = "desc", None
        reader._serve(model.config)                       # d1_codes / d1_attention / d1_set_size of the checkpoint
        self.reader = reader
        self.attention = reader.attention

    def question(self, raw: dict):
        self.prompt.check_question(raw, self.tokenizer)
        return self.prompt.as_question(raw)

    def encode(self, state: str, q) -> tuple[list[int], int | None, list[int] | None, int]:
        """(ids, state length, each token's question, answer slot) of one question alone after its state."""
        (_, _, ids, state_len, question, ends, _), = self.reader._sequences([(state, [q], [])], set_size=1)
        return ids, state_len, question, ends[0][0]

    def readout(self, q) -> tuple[tuple[int, ...], np.ndarray]:
        tokens, forms, _ = self.prompt.readout_table(self.tokenizer, q)
        return tuple(tokens), np.asarray(forms)

    def bucket(self, q) -> str:
        return self.calib.question_bucket(q)

    def cardinality(self, q) -> int:
        return self.prompt.cardinality(q)

    def option_tokens(self) -> list[int]:
        """Every vocabulary id an option can read: A..Z and ' A'..' Z' (up to 26 options), yes/no forms."""
        p = self.prompt
        both = (p.Choice("x", {c: None for c in p.LETTERS}), p.Noul("x"))
        return sorted({t for q in both for t in p.readout_table(self.tokenizer, q)[0]})

    def context(self, model, state_len, question):
        return self.attn.state_attention(model, state_len, question)

    def backbone(self, model):
        return self.attn.text_backbone(model)


class OptionRows(torch.nn.Module):
    """Trainable fp32 copies of the tied embedding rows of the option tokens. The LM head is tied to the input
    embeddings, so a row is read twice: as the option's readout vector and as the prompt's embedding of its code
    letter. Both reads use the copy here (the prompt's as a constant: the gradient reaches the rows through the
    readout only), and `fold` writes it back into the tied matrix before saving, so the trained function is the
    served one (a full 128k x 2048 matrix would cost 4 GiB of weights, gradients and Adam)."""

    def __init__(self, weight: torch.Tensor, token_ids: list[int]):
        super().__init__()
        ids = torch.tensor(sorted(set(token_ids)), dtype=torch.long)
        index = torch.full((weight.shape[0],), -1, dtype=torch.long)
        index[ids] = torch.arange(len(ids))
        self.register_buffer("ids", ids)
        self.register_buffer("index", index)
        self.rows = torch.nn.Parameter(weight.detach()[ids.to(weight.device)].float().clone())

    def lookup(self, tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The readout rows of `tokens` (n, D): the trainable copy where there is one, else the frozen matrix."""
        pos = self.index[tokens]
        return torch.where((pos >= 0)[:, None], self.rows[pos.clamp(min=0)], weight[tokens].float())

    def embed(self, input_ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The prompt's embeddings with the copies' current values, as constants: the rows learn through the
        readout only, so no gradient flows from the trained layers down through the frozen ones to the embeddings."""
        pos = self.index[input_ids]
        return torch.where((pos >= 0)[..., None], self.rows.detach()[pos.clamp(min=0)].to(out.dtype), out)

    @torch.no_grad()
    def fold(self, weight: torch.Tensor) -> None:
        """Write the rows into the tied matrix (rounded to its dtype); the copies take the rounded values, so the
        in-memory model computes what the saved one will."""
        ids = self.ids.to(weight.device)
        weight[ids] = self.rows.detach().to(weight.device, weight.dtype)
        self.rows.copy_(weight[ids].to(self.rows.device, self.rows.dtype))


class Lfm3Adapter:
    """The d1-2B text path, trained (equivalent of MAC-T §0 for the 600M model). d1-2B is LFM3-VL (24 layers: 13 gated
    short-conv, 5 sliding-window attention, 6 global attention; 1.92B parameters) behind `D1Model.system_one`. There
    is no separate scorer head: the answer is read off the language model's own LM head at one slot.

    1. ids / slot: `SlotReader._sequences([(state, [q], [])], set_size=1)` of the checkpoint's runner: ids =
       `<bos><|im_start|>user\\n{state}\\n\\n\\nQUESTION:\\n{instructions}\\n\\nOptions:\\n{code} {description}...\\n\\n
       Reply with the option code only.<|im_end|>\\n<|im_start|>assistant\\n<think></think>`; the answer slot is the
       last token. Option codes are A, B, C ... (one token each, also read with a leading space). A noul asks for
       yes or no. Every question is its own sequence: a lone question under `set` attention reads exactly as under
       `prompt`, the training-time case; serve a factorized checkpoint with `set_size=1`.
    2. attention: `attention.state_attention(model, state_len, question)` swaps the full-attention layers' mask for
       `Sees` (config `d1_attention`, "set": the whole lone prompt reads itself both ways); the conv and sliding
       layers stay causal. Right padding, attention_mask 1 on real tokens.
    3. hidden state: `text_backbone(model)(input_ids, attention_mask, use_cache=False).last_hidden_state`, after the
       language model's final norm (`embedding_norm`), taken at the slot.
    4. option logits: `hidden @ lm_head.weight[tokens].T` (the LM head is tied to `embed_tokens`), each option the
       max over its forms (code, ' code'; yes/Yes/YES, no/No/NO) -- `prompt.readout_table`. The vector is in
       criteria order; a noul is [false (no), true (yes)] here (the model itself orders [yes, no]).
    5. probabilities: softmax(z / T), T = `calibration.json` `temperatures[bucket]`, else its `default`; the buckets
       are `calibration.question_bucket`: noul, choice:small (<=3 options), choice:mid (<=8), choice:large (<=26).
       `calibration.json` also holds the typed answers' `tables`, which this trainer leaves alone.

    Training: frozen weights stay bf16, the trainable ones are fp32 (all fp32 with `fp32`), cast at the boundary
    of the first trainable layer by a forward pre-hook (so the trained layers run in fp32 end to end); trainable =
    the top N of the 24 layers + the final norm + `OptionRows` (the head). Gradient checkpointing is optional."""
    arch = "lfm3"

    def __init__(self, model, codec, *, fp32: bool = False, device: str = "cpu", head_rows: bool = True,
                 calibration: dict | None = None):
        self.model, self.codec = model, codec
        self.lm = model.model.language_model
        self.fp32, self.device, self.head_rows = fp32, device, head_rows
        self.calibration = calibration if calibration is not None else {"default": 1.0, "temperatures": {}}
        self.calibration.setdefault("temperatures", {})
        self.option_rows: OptionRows | None = None
        self.train_layers = 0
        self._hook = None

    # ------------------------------------------------------------------------------------------ rows
    def build_rows(self, lines: list[dict]) -> list[Row]:
        """Every question of every line becomes one Row (a noul's target and options are [false, true])."""
        c, rows = self.codec, []
        for line in lines:
            for name, raw in line["questions"].items():
                q = c.question(raw)
                ids, state_len, question, slot = c.encode(line["state"], q)
                tokens, forms = c.readout(q)
                answer = line["answers"][name]
                if q.type == "choice":
                    target, n, qtype = list(q.criteria).index(answer), c.cardinality(q), 0
                elif q.type == "noul":
                    target, n, qtype, forms = (1 if str(answer).lower() == "true" else 0), 2, 2, forms[::-1]
                else:
                    raise ValueError(f"question {name!r}: type {q.type!r} is not trainable here")
                rows.append(Row(ids, [slot], qtype, n, target, line["id"], name, c.bucket(q),
                                Lfm3Extra(state_len, question, tokens, np.ascontiguousarray(forms))))
        return rows

    # ------------------------------------------------------------------------------------------ freezing
    def set_trainable(self, train_layers: int) -> list[torch.nn.Parameter]:
        """Freeze everything, then enable the top `train_layers` of the language model's layers (counted from the
        top), its final norm and the option rows; cast those to fp32 (frozen weights keep their dtype)."""
        layers = list(self.lm.layers)
        k = max(0, min(train_layers, len(layers)))
        self.train_layers = k
        for p in self.model.parameters():
            p.requires_grad = False
        mods = [self.lm.embedding_norm] + (layers[len(layers) - k:] if k else [])
        for m in mods:
            if not self.fp32:
                m.float()
            for p in m.parameters():
                p.requires_grad = True
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if k and not self.fp32:       # frozen layers hand over bf16: the trained ones compute in fp32
            self._hook = layers[len(layers) - k].register_forward_pre_hook(
                lambda module, args: (args[0].float(), *args[1:]))
        if self.head_rows and self.option_rows is None:
            weight = self.lm.embed_tokens.weight
            self.option_rows = OptionRows(weight, self.codec.option_tokens()).to(weight.device)
            embed, plain = self.lm.embed_tokens, self.lm.embed_tokens.forward
            embed.forward = lambda input_ids: self.option_rows.embed(input_ids, plain(input_ids))
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.extra_params().values())

    def extra_params(self) -> dict[str, torch.nn.Parameter]:
        """Trainable parameters that are not registered in the model (they are folded in at save)."""
        return {} if self.option_rows is None else {"d1_option_rows": self.option_rows.rows}

    def to_device(self) -> None:
        """The language model and the LM head to the training device; vision and the projector stay on the cpu."""
        self.lm.to(self.device)
        self.model.lm_head.to(self.device)
        if self.option_rows is not None:
            self.option_rows.to(self.device)

    def enable_grad_ckpt(self) -> bool:
        """Non-reentrant checkpointing of the decoder layers (reentrant would drop the grads of the first trained
        layer, whose input does not require grad); False when the model does not support it. transformers hooks the
        input embeddings to require grad when it enables checkpointing, which would pull the backward pass (and its
        recompute) through the frozen layers: that hook is removed."""
        if not getattr(self.model, "supports_gradient_checkpointing", False):
            return False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(self.model, "disable_input_require_grads"):
            self.model.disable_input_require_grads()
        return True

    # ------------------------------------------------------------------------------------------ forward
    def readout_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        weight = self.model.lm_head.weight
        return weight[tokens].float() if self.option_rows is None else self.option_rows.lookup(tokens, weight)

    def option_logits(self, rows: list[Row]) -> list[torch.Tensor]:
        """The text path of `system_one`, with gradients: one (n_options,) logit vector per row (no temperature)."""
        dev, width = self.device, max(len(r.ids) for r in rows)
        lengths = torch.tensor([len(r.ids) for r in rows], device=dev)
        ids = torch.tensor([r.ids + [self.codec.pad_id] * (width - len(r.ids)) for r in rows], device=dev)
        mask = (torch.arange(width, device=dev)[None] < lengths[:, None]).long()
        state = question = None
        if rows[0].extra.state_len is not None:
            state = torch.tensor([r.extra.state_len for r in rows])
            if rows[0].extra.question is not None:
                question = torch.tensor([r.extra.question + [-1] * (width - len(r.extra.question)) for r in rows])
        with self.codec.context(self.model, state, question):
            out = self.codec.backbone(self.model)(input_ids=ids, attention_mask=mask, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        hidden = out[0] if hidden is None else hidden
        slot = torch.tensor([r.markers[0] for r in rows], device=hidden.device)
        slots = hidden[torch.arange(len(rows), device=hidden.device), slot].float()
        zs = []
        for h, r in zip(slots, rows):
            w = self.readout_rows(torch.tensor(r.extra.tokens, device=h.device))
            z = w @ h
            zs.append(z[torch.as_tensor(r.extra.forms, device=h.device)].max(-1).values)
        return zs

    # ------------------------------------------------------------------------------------------ memory
    def memory_estimate(self, train_layers: int, max_tokens: int, grad_ckpt: bool = False) -> dict:
        """A rough byte budget (GiB) of training with the top `train_layers` layers: weights (frozen bf16, trained
        fp32), gradients, Adam, the CPU copy of the best epoch, activations of the trained layers (30 hidden-widths
        per token and layer in fp32; with checkpointing one layer's plus the layer inputs) and a fixed 2 GiB. Measured
        on the M5 (top 8 layers, 4096 tokens): 15.1 GiB peak with checkpointing, 22.1 without."""
        layers = list(self.lm.layers)
        k = max(0, min(train_layers, len(layers)))
        count = lambda mods: sum(p.numel() for m in mods for p in m.parameters())  # noqa: E731
        top = count(layers[len(layers) - k:]) if k else 0
        norm = count([self.lm.embedding_norm])
        hidden = int(self.model.config.text_config.hidden_size)
        rows = len(self.codec.option_tokens()) * hidden if self.head_rows else 0
        trained = top + norm
        frozen = count([self.model]) - trained
        wide = 4 if self.fp32 else 2
        units = (k + 30) if grad_ckpt else 30 * k                  # hidden-widths kept per token (fp32)
        activations = max_tokens * hidden * (4 * units + wide * 30)  # + one frozen layer's working set
        parts = {"weights_frozen": frozen * wide, "weights_trained": (trained + rows) * 4,
                 "grads": (trained + rows) * 4, "adam": (trained + rows) * 8,
                 "best_state_copy": (trained + rows) * 4, "activations": activations, "overhead": 2 * GB}
        parts = {name: value / GB for name, value in parts.items()}
        parts["total"] = sum(parts.values())
        parts["trainable_params"] = trained + rows
        return parts

    # ------------------------------------------------------------------------------------------ temperatures
    def temperature(self, tkey: str) -> float:
        return float(self.calibration["temperatures"].get(tkey, self.calibration.get("default", 1.0)))

    def set_temperatures(self, temperatures: dict[str, float]) -> None:
        self.calibration["temperatures"].update(temperatures)

    def expected_probabilities(self, rows: list[Row]) -> list[list[float]]:
        """In-memory probabilities of `rows` as `system_one` returns them: criteria order, a noul [false, true]."""
        with torch.no_grad():
            self.model.eval()
            return [(z.float() / self.temperature(r.tkey)).softmax(-1).tolist()
                    for r, z in zip(rows, self.option_logits(rows))]

    # ------------------------------------------------------------------------------------------ save / reload
    def save(self, out: Path, src: Path) -> None:
        """Fold the option rows into the tied matrix, store every weight in the serving dtype (bf16, or fp32 with
        --fp32), `save_pretrained`, then copy the remote code, tokenizer and templates of `src` and write the
        re-fitted `calibration.json`."""
        from transformers import PretrainedConfig
        if self.option_rows is not None:
            self.option_rows.fold(self.lm.embed_tokens.weight)
        storage = torch.float32 if self.fp32 else torch.bfloat16
        for p in self.model.parameters():
            if p.is_floating_point() and p.dtype != storage:
                p.data = p.data.to(storage)
        if self._hook is not None:                 # everything is storage-dtype now: no fp32 boundary any more
            self._hook.remove()
            self._hook = None
        out.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(PretrainedConfig, "_get_generation_parameters", lambda self: {}):
            self.model.save_pretrained(out)
        copy_remote_files(Path(src), out)
        (out / "calibration.json").write_text(json.dumps(self.calibration, indent=1))

    def reload_check(self, out: Path, line: dict, rows: list[Row], release: bool = False) -> dict:
        """Reload `out` on the cpu (in the storage dtype) and run system_one on a val line: every distribution sums to
        1 +- 1e-3; the max abs difference to the in-memory probabilities is reported (`rows` are the line's rows).
        `release`: drop the in-memory weights (to the meta device) before the reload, which needs the same memory."""
        from transformers import AutoModel
        want = self.expected_probabilities(rows)   # first: loading `out` registers its own copy of the "d1" attention
        if release:
            self.model.to("meta")
            gc.collect()
        reloaded = AutoModel.from_pretrained(out, trust_remote_code=True, dtype=torch.float32 if self.fp32
                                             else torch.bfloat16).eval()       # on the cpu (it replaces our attention)
        answers = reloaded.system_one(line["state"], line["questions"])["answers"]
        got = []
        for r in rows:
            a = answers[r.name]
            if r.qtype == 2:
                got.append([1.0 - float(a["noul"]), float(a["noul"])])
            else:
                got.append([float(x) for x in a["probabilities"].values()])
        diff = max(abs(a - b) for pe, pr in zip(want, got) for a, b in zip(pe, pr))
        sums_ok = all(abs(sum(p) - 1.0) <= 1e-3 for p in got)
        del reloaded
        gc.collect()
        return {"reload_ok": bool(sums_ok and set(answers) == set(line["questions"])), "reload_max_abs_diff": diff}


def copy_remote_files(src: Path, out: Path) -> list[str]:
    """Copy every `*.py`, `*.json`, `*.jinja`, tokenizer file and the LICENSE of `src` (a snapshot or checkpoint
    dir) into `out`, except what `save_pretrained` wrote there; keys of the source config that the saved one lacks
    (`auto_map`, the `d1_*` serving fields) are added to `out/config.json`. Returns the copied names."""
    copied = []
    for f in sorted(Path(src).iterdir()):
        if not f.is_file() or f.name in KEEP_SAVED or f.name == "train_report.json":
            continue
        if f.suffix in COPY_SUFFIXES or f.name.startswith("tokenizer") or f.name == "LICENSE":
            shutil.copy2(f, out / f.name)
            copied.append(f.name)
    saved = json.loads((out / "config.json").read_text())
    for key, value in read_config(src).items():
        saved.setdefault(key, value)
    (out / "config.json").write_text(json.dumps(saved, indent=2))
    return copied


def check_memory(estimate: dict, max_gb: float) -> None:
    """Abort (SystemExit with the reason) when the estimated total exceeds `max_gb`."""
    if estimate["total"] > max_gb:
        raise SystemExit(
            f"memory estimate {estimate['total']:.1f} GiB exceeds --max-gb {max_gb:g}: trained weights/grads/Adam "
            f"{estimate['weights_trained'] + estimate['grads'] + estimate['adam']:.1f} GiB, activations "
            f"{estimate['activations']:.1f} GiB. Lower --train-layers, --max-tokens or add --grad-ckpt, or raise "
            f"--max-gb.")


def fmt_estimate(estimate: dict) -> str:
    keys = ("weights_frozen", "weights_trained", "grads", "adam", "best_state_copy", "activations", "overhead")
    return (" + ".join(f"{k.replace('_', ' ')} {estimate[k]:.1f}" for k in keys)
            + f" = {estimate['total']:.1f} GiB ({estimate['trainable_params'] / 1e6:.0f}M trainable)")


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
    extra = adapter_of(model).extra_params() if adapter_of(model) is not None else {}   # d1-2B: the option rows
    named = {n: p for n, p in model.named_parameters() if p.requires_grad} | extra
    params = list(named.values())
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
    if adapter_of(model) is not None:
        return adapter_of(model).save(out, src)
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


def load_lfm3(src: Path, args) -> tuple:
    """d1-2B: the model (bf16, or fp32 with --fp32) and its `Lfm3Adapter`, attached as `model.d1_adapter`.
    With --estimate-only the model is built on the meta device: parameter counts, no weights read."""
    from transformers import AutoConfig, AutoModel
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    if args.estimate_only:
        with torch.device("meta"):
            model = AutoModel.from_config(AutoConfig.from_pretrained(src, trust_remote_code=True),
                                          trust_remote_code=True, dtype=dtype)
    else:
        model = AutoModel.from_pretrained(src, trust_remote_code=True, dtype=dtype)
    calibration = Path(src) / "calibration.json"
    adapter = Lfm3Adapter(model, Lfm3Codec(model, Path(src)), fp32=args.fp32, device=args.device,
                          head_rows=args.head_rows,
                          calibration=json.loads(calibration.read_text()) if calibration.exists() else None)
    model.d1_adapter = adapter
    return model, adapter


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--data", type=Path, default=None, help="default <dataset>/d1/<variant>")
    ap.add_argument("--out", type=Path, default=None, help="default db/models/d1-ft-<variant>")
    ap.add_argument("--init", default=INIT_ID, help="hub id (cached) or checkpoint dir; LiquidAI/d1-2B for d1-2B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=None, help="tokens per batch (default 6144; d1-2B 4096)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--train-layers", type=int, default=None,
                    help="top N layers (default: all 16 of the 600M; 8 of the 24 of d1-2B)")
    ap.add_argument("--balance", action=argparse.BooleanOptionalAction, default=None,
                    help="inverse-class-frequency loss weights (default: on for act)")
    ap.add_argument("--limit", type=int, default=None, help="first N train lines")
    ap.add_argument("--val-limit", type=int, default=None, help="first N val lines")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="d1-2B: all weights fp32 (default: frozen bf16, trained fp32)")
    ap.add_argument("--grad-ckpt", action="store_true", help="d1-2B: gradient checkpointing of the decoder layers")
    ap.add_argument("--max-gb", type=float, default=26.0, help="d1-2B: abort when the memory estimate is above")
    ap.add_argument("--head-rows", action=argparse.BooleanOptionalAction, default=True,
                    help="d1-2B: train the option-token rows of the tied LM head (the head), default on")
    ap.add_argument("--estimate-only", action="store_true",
                    help="d1-2B: print the memory estimates (from the config, no weights) and exit")
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
    arch = detect_architecture(read_config(src))
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS[arch]
    if args.train_layers is None:
        args.train_layers = 16 if arch == "d1_omni" else 8
    adapter, estimate, mod = None, None, None
    if arch == "lfm3":
        model, adapter = load_lfm3(src, args)
        if args.estimate_only:
            table = {n: adapter.memory_estimate(n, max_tokens, args.grad_ckpt) for n in sorted({args.train_layers, 8, 12, 24})}
            for n, est in table.items():
                print(f"train layers {n:>2}: {fmt_estimate(est)}")
            return {"estimates": table}
        estimate = adapter.memory_estimate(args.train_layers, max_tokens, args.grad_ckpt)
        print(f"memory estimate (train layers {args.train_layers}, max tokens {max_tokens}, "
              f"grad ckpt {args.grad_ckpt}): {fmt_estimate(estimate)}   limit {args.max_gb:g} GiB")
        check_memory(estimate, args.max_gb)
        trainable = sum(p.numel() for p in set_trainable(model, args.train_layers))
        adapter.to_device()
        if args.grad_ckpt and not adapter.enable_grad_ckpt():
            print("warning: this model does not support gradient checkpointing; running without")
    else:
        model = AutoModel.from_pretrained(src, trust_remote_code=True, dtype=torch.float32)   # fp32: bf16 flips answers
        mod = importlib.import_module(type(model).__module__)
        model.encoder.to(args.device)                          # vision / audio are unused and stay on the cpu
        model.head.to(args.device)
        trainable = sum(p.numel() for p in set_trainable(model, args.train_layers))
    print(f"trainable parameters: {trainable:,}  (train layers {args.train_layers}, device {args.device}, "
          f"arch {arch})")

    train_lines = read_jsonl(data / "train.jsonl", args.limit)
    val_lines = read_jsonl(data / "val.jsonl", args.val_limit)
    train_rows, val_rows = build_rows(train_lines, model, mod), build_rows(val_lines, model, mod)
    print(f"{args.variant}: {len(train_lines)} train lines / {len(train_rows)} rows, "
          f"{len(val_lines)} val lines / {len(val_rows)} rows, balance {balance}")

    res = train(model, train_rows, val_rows, variant=args.variant, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, warmup=args.warmup, max_tokens=max_tokens, accum=args.accum,
                balance=balance, seed=args.seed, device=args.device)
    print(f"best epoch {res['best_epoch']} ({PRIMARY[args.variant]} "
          f"{res['history'][res['best_epoch'] - 1]['val_metrics'][PRIMARY[args.variant]]:.4f})")

    temperatures = fit_temperatures(val_rows, res["best_logits"])    # best_logits = the restored weights on val
    if adapter is not None:
        adapter.set_temperatures(temperatures)                       # -> calibration.json[temperatures]
    else:
        model.config.temperatures.update(temperatures)
    print("temperatures:", temperatures)

    save_checkpoint(model, out, src)
    first = val_lines[0]
    first_rows = [r for r in val_rows if r.line_id == first["id"]]
    if adapter is not None:
        check = adapter.reload_check(out, first, first_rows, release=True)
    else:
        check = reload_check(out, first, expected_probabilities(model, first_rows, model.config.temperatures))
    print(f"saved {out}  reload_ok {check['reload_ok']}  max|diff| vs in-memory {check['reload_max_abs_diff']}")

    peak = res["peak_mps_bytes"]
    report = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {"balance": balance},
              "arch": arch, "max_tokens": max_tokens, "memory_estimate": estimate,
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
