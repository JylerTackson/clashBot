"""MAC-AE E2 tests: d1-2B (LFM3) support -- the trainer's `Lfm3Adapter` on a fake LFM3-like model set and on a tiny
random-weight model built from the cached remote code, D1Ranker's dtype handling, and `eval_d1_zero_shot --dtype`.
No real weights are loaded except in the slow test (CLASHBOT_SLOW_TESTS=1)."""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot import ranker as ranker_mod  # noqa: E402
from clashbot.ranker import D1Ranker  # noqa: E402
from tools.eval import eval_d1_zero_shot as ev  # noqa: E402
from tools.train import train_d1 as T  # noqa: E402

DATASET = ROOT / "dataset" / "v1"
SNAPSHOTS = sorted((Path.home() / ".cache/huggingface/hub/models--LiquidAI--d1-2B/snapshots").glob("*"))
SNAPSHOTS = [s for s in SNAPSHOTS if (s / "prompt.py").exists() and (s / "modeling_d1_lfm3_vl.py").exists()]
needs_code = pytest.mark.skipif(not SNAPSHOTS, reason="d1-2B remote code not in the Hugging Face cache")


# --------------------------------------------------------------------------- a fake LFM3-like model set
class Block(nn.Module):
    """One layer plus a mean-pooled context, so the answer slot can see the state."""

    def __init__(self, d: int):
        super().__init__()
        self.a, self.b = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, h, pad):
        w = pad[..., None].to(h.dtype)
        ctx = (self.b(h) * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)
        return h + torch.tanh(self.a(h) + ctx)


class FakeLM(nn.Module):
    """`model.model.language_model`: embed_tokens, layers, embedding_norm (the names the adapter reads)."""

    def __init__(self, vocab: int, d: int, n: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.embedding_norm = nn.LayerNorm(d)

    def forward(self, input_ids, attention_mask, use_cache=False):
        h, pad = self.embed_tokens(input_ids), attention_mask.bool()
        for layer in self.layers:
            h = layer(h, pad)
        return types.SimpleNamespace(last_hidden_state=self.embedding_norm(h))


class FakeVL(nn.Module):
    def __init__(self, vocab: int, d: int, n: int):
        super().__init__()
        self.language_model = FakeLM(vocab, d, n)

    def forward(self, input_ids, attention_mask, use_cache=False):
        return self.language_model(input_ids, attention_mask, use_cache)


class FakeLfm3(nn.Module):
    supports_gradient_checkpointing = False

    def __init__(self, vocab: int = 64, d: int = 8, n: int = 4, dtype=torch.float32, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = FakeVL(vocab, d, n)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight          # tied, as in d1-2B
        self.config = types.SimpleNamespace(model_type="lfm3_vl", text_config=types.SimpleNamespace(hidden_size=d))
        self.to(dtype)

    def save_pretrained(self, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "config.json").write_text(json.dumps({"model_type": "lfm3_vl", "dtype": "bfloat16"}))
        torch.save(self.state_dict(), Path(out) / "model.pt")


@dataclass
class FakeQuestion:
    type: str
    criteria: dict | None
    n: int


class FakeCodec:
    """Same interface as `Lfm3Codec`: one token per state character, option j read at token 41 + j (a noul: yes 56,
    no 57), the answer slot is the last token."""
    attention = "set"
    pad_id = 0

    def question(self, raw):
        crit = raw.get("criteria")
        return FakeQuestion(raw["type"], crit, 2 if raw["type"] == "noul" else len(crit))

    def encode(self, state, q):
        ids = [1 + ord(c) % 40 for c in state] + [40, 40]
        return ids, len(state), [-1] * len(state) + [1] * 2, len(ids) - 1

    def readout(self, q):
        if q.type == "noul":
            return (56, 57), np.array([[0], [1]])                                   # [yes, no], as the model orders
        return tuple(41 + j for j in range(q.n)), np.arange(q.n)[:, None]

    def bucket(self, q):
        return f"{q.type}:{q.n}"

    def cardinality(self, q):
        return q.n

    def option_tokens(self):
        return list(range(41, 58))

    def context(self, model, state_len, question):
        return contextlib.nullcontext()

    def backbone(self, model):
        return model.model


def choice_q(n: int = 3) -> dict:
    return {"type": "choice", "instructions": "?", "criteria": {f"o{j}": f"option {j}" for j in range(n)}}


NOUL = {"type": "noul", "instructions": "act?", "criteria": {"true": "deploy", "false": "wait"}}


def line(i: int, state: str, questions: dict, answers: dict) -> dict:
    return {"id": f"l{i}", "game_id": "g", "split": "train", "state": state, "questions": questions, "answers": answers}


def toy_lines(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = "abc"[int(rng.integers(3))]
        out.append(line(i, c + "xy", {"play": choice_q(3)}, {"play": f"o{'abc'.index(c)}"}))
    return out


def fake_adapter(dtype=torch.float32, fp32: bool = True, **kw) -> tuple[FakeLfm3, T.Lfm3Adapter]:
    model = FakeLfm3(dtype=dtype)
    adapter = T.Lfm3Adapter(model, FakeCodec(), fp32=fp32, device="cpu",
                            calibration={"default": 1.5, "temperatures": {}, "tables": {"box": [[0.0, 0.0]]}}, **kw)
    model.d1_adapter = adapter
    return model, adapter


# --------------------------------------------------------------------------- architecture detection
def test_detect_architecture():
    assert T.detect_architecture({"model_type": "lfm3_vl", "text_config": {"model_type": "lfm3"}}) == "lfm3"
    assert T.detect_architecture({"model_type": "x", "text_config": {"model_type": "lfm3"}}) == "lfm3"
    assert T.detect_architecture({"model_type": "d1_omni", "architectures": ["D1OmniModel"]}) == "d1_omni"
    assert T.detect_architecture({"architectures": ["D1OmniModel"]}) == "d1_omni"
    assert T.detect_architecture(types.SimpleNamespace(to_dict=lambda: {"model_type": "lfm3_vl"})) == "lfm3"
    with pytest.raises(ValueError, match="unknown d1 architecture"):
        T.detect_architecture({"model_type": "llama"})


@needs_code
def test_detect_architecture_real_configs():
    assert T.detect_architecture(T.read_config(SNAPSHOTS[-1])) == "lfm3"
    omni = sorted((Path.home() / ".cache/huggingface/hub/models--LiquidAI--d1-omni-600M/snapshots").glob("*"))
    if omni:
        assert T.detect_architecture(T.read_config(omni[-1])) == "d1_omni"


# --------------------------------------------------------------------------- rows
def test_build_rows_lfm3_targets_and_orientation():
    model, adapter = fake_adapter()
    lines = [line(0, "abc", {"play": choice_q(4)}, {"play": "o2"}),
             line(1, "abc", {"act": NOUL}, {"act": "true"}),
             line(2, "abc", {"act": NOUL}, {"act": "false"}),
             line(3, "abc", {"card": choice_q(5), "lane": choice_q(3)}, {"card": "o4", "lane": "o0"})]
    rows = T.build_rows(lines, model, None)
    assert [r.target for r in rows] == [2, 1, 0, 4, 0]
    assert [r.n_options for r in rows] == [4, 2, 2, 5, 3] and [r.qtype for r in rows[:3]] == [0, 2, 2]
    assert [r.tkey for r in rows] == ["choice:4", "noul:2", "noul:2", "choice:5", "choice:3"]
    assert [r.name for r in rows[3:]] == ["card", "lane"] and all(r.line_id == "l3" for r in rows[3:])
    assert rows[0].markers == [len(rows[0].ids) - 1] and rows[0].extra.state_len == 3
    # a noul's options are [false (no token 57), true (yes token 56)]: forms are the model's [yes, no] reversed
    assert rows[1].extra.tokens == (56, 57) and rows[1].extra.forms.tolist() == [[1], [0]]
    assert rows[0].extra.forms.tolist() == [[0], [1], [2], [3]]


# --------------------------------------------------------------------------- freezing, logits, gradients
def test_option_logits_match_manual_readout_and_carry_gradients():
    model, adapter = fake_adapter()
    rows = T.build_rows(toy_lines(3) + [line(9, "abcd", {"act": NOUL}, {"act": "false"})], model, None)
    T.set_trainable(model, 2)
    with torch.no_grad():
        zs = T.option_logits(model, rows)
    for r, z in zip(rows, zs):                                    # one row at a time: padding must not matter
        ids = torch.tensor([r.ids])
        h = model.model(ids, torch.ones_like(ids)).last_hidden_state[0, r.markers[0]]
        w = model.lm_head.weight[list(r.extra.tokens)]
        want = (w @ h)[torch.as_tensor(r.extra.forms)].max(-1).values
        assert torch.allclose(z, want, atol=1e-5) and z.shape == (r.n_options,)
    model.train()
    seen = []
    handle = model.model.language_model.layers[2].register_forward_pre_hook(
        lambda module, args: seen.append(args[0].requires_grad))
    T.batch_loss(T.option_logits(model, rows), rows).backward()
    handle.remove()
    assert seen == [False]            # nothing flows back into the frozen layers (not even via the option rows)
    layers = model.model.language_model.layers
    assert layers[3].a.weight.grad is not None and layers[2].a.weight.grad is not None
    assert layers[1].a.weight.grad is None and layers[0].a.weight.grad is None
    assert model.model.language_model.embedding_norm.weight.grad is not None
    assert model.model.language_model.embed_tokens.weight.grad is None            # the matrix itself stays frozen
    assert adapter.option_rows.rows.grad is not None and float(adapter.option_rows.rows.grad.abs().sum()) > 0


def test_freezing_top_layers_keeps_the_bottom_bit_identical_in_bf16():
    model, adapter = fake_adapter(dtype=torch.bfloat16, fp32=False)
    rows = T.build_rows(toy_lines(30), model, None)
    params = T.set_trainable(model, 2)
    assert adapter._hook is not None and len(params) == sum(1 for p in model.parameters() if p.requires_grad) + 1
    layers = model.model.language_model.layers
    assert layers[3].a.weight.dtype == layers[2].a.weight.dtype == torch.float32      # trained: fp32
    assert layers[0].a.weight.dtype == layers[1].a.weight.dtype == torch.bfloat16     # frozen: bf16
    assert model.model.language_model.embedding_norm.weight.dtype == torch.float32
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rows_before = adapter.option_rows.rows.detach().clone()
    T.train(model, rows, rows[:10], variant="joint", epochs=2, lr=3e-2, max_tokens=120, accum=2, log=lambda *_: None)
    after = dict(model.named_parameters())
    for name in before:
        same = torch.equal(before[name], after[name].detach())
        if name.startswith(("model.language_model.layers.0.", "model.language_model.layers.1.",
                            "model.language_model.embed_tokens")):
            assert same, f"{name} must stay frozen and bit-identical"
    assert not torch.equal(before["model.language_model.layers.3.a.weight"], after["model.language_model.layers.3.a.weight"].detach())
    assert not torch.equal(before["model.language_model.embedding_norm.weight"], after["model.language_model.embedding_norm.weight"].detach())
    assert not torch.equal(rows_before, adapter.option_rows.rows.detach())            # the head moved
    # N counts from the top and clamps; 0 leaves only the head (final norm + option rows)
    T.set_trainable(model, 99)
    assert all(p.requires_grad for p in layers.parameters()) and not model.model.language_model.embed_tokens.weight.requires_grad
    T.set_trainable(model, 0)
    assert not any(p.requires_grad for p in layers.parameters()) and adapter._hook is None
    assert model.model.language_model.embedding_norm.weight.requires_grad


def test_training_learns_toy_task_with_the_adapter():
    model, adapter = fake_adapter()
    T.set_trainable(model, 4)
    rows = T.build_rows(toy_lines(60), model, None)
    val = T.build_rows(toy_lines(30, seed=5), model, None)
    res = T.train(model, rows, val, variant="joint", epochs=40, lr=3e-2, max_tokens=120, accum=1, log=lambda *_: None)
    assert res["history"][-1]["train_loss"] < 0.5 * res["history"][0]["train_loss"]
    _, logits = T.evaluate(model, rows, 120)
    assert T.compute_metrics(rows, logits, "joint")["play"] >= 0.95
    _, again = T.evaluate(model, val, 120)                                              # best epoch restored
    assert all(np.allclose(a, b, atol=1e-5) for a, b in zip(again, res["best_logits"]))


def test_no_head_rows_reads_the_frozen_matrix():
    model, adapter = fake_adapter(head_rows=False)
    rows = T.build_rows(toy_lines(2), model, None)
    params = T.set_trainable(model, 1)
    assert adapter.option_rows is None and adapter.extra_params() == {}
    assert len(params) == sum(1 for p in model.parameters() if p.requires_grad)
    assert len(T.option_logits(model, rows)) == 2


# --------------------------------------------------------------------------- temperatures
def test_temperatures_default_and_refit_buckets():
    model, adapter = fake_adapter()
    rows = T.build_rows([line(0, "abc", {"play": choice_q(3)}, {"play": "o0"})], model, None)
    assert adapter.temperature("choice:3") == 1.5                                       # calibration default
    with torch.no_grad():
        z = T.option_logits(model, rows)[0]
    assert adapter.expected_probabilities(rows)[0] == pytest.approx((z / 1.5).softmax(-1).tolist(), abs=1e-6)
    adapter.set_temperatures({"choice:3": 0.8})
    assert adapter.expected_probabilities(rows)[0] == pytest.approx((z / 0.8).softmax(-1).tolist(), abs=1e-6)
    assert adapter.calibration["default"] == 1.5 and "tables" in adapter.calibration


# --------------------------------------------------------------------------- save: the remote-code files
def make_source(tmp_path: Path) -> Path:
    src = tmp_path / "snapshot"
    src.mkdir()
    for name in ("modeling_d1_lfm3_vl.py", "prompt.py", "runner.py", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "processor_config.json", "LICENSE", "README.md",
                 "model.safetensors.index.json", "train_report.json"):
        (src / name).write_text(f"x:{name}")
    (src / "config.json").write_text(json.dumps({
        "model_type": "lfm3_vl", "auto_map": {"AutoModel": "modeling_d1_lfm3_vl.D1Model"}, "d1_codes": 2,
        "d1_attention": "set", "d1_set_size": 16, "dtype": "bfloat16"}))
    (src / "calibration.json").write_text(json.dumps({"default": 1.3, "temperatures": {}, "tables": {"box": [[0, 0]]}}))
    return src


def test_save_folds_rows_copies_remote_files_and_writes_calibration(tmp_path):
    src, out = make_source(tmp_path), tmp_path / "ckpt"
    model, adapter = fake_adapter(dtype=torch.bfloat16, fp32=False)
    T.set_trainable(model, 1)
    with torch.no_grad():
        adapter.option_rows.rows.add_(0.5)
    adapter.set_temperatures({"choice:3": 1.9})
    T.save_checkpoint(model, out, src)
    names = {p.name for p in out.iterdir()}
    assert {"modeling_d1_lfm3_vl.py", "prompt.py", "runner.py", "tokenizer.json", "tokenizer_config.json",
            "chat_template.jinja", "processor_config.json", "LICENSE", "calibration.json", "config.json",
            "model.pt"} <= names
    assert not names & {"README.md", "model.safetensors.index.json", "train_report.json"}
    assert (out / "prompt.py").read_text() == "x:prompt.py"
    cfg = json.loads((out / "config.json").read_text())                    # saved config + the source's missing keys
    assert cfg["auto_map"]["AutoModel"].endswith("D1Model") and cfg["d1_attention"] == "set" and cfg["d1_codes"] == 2
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["temperatures"] == {"choice:3": 1.9} and cal["default"] == 1.5 and cal["tables"] == {"box": [[0.0, 0.0]]}
    state = torch.load(out / "model.pt")
    emb = state["model.language_model.embed_tokens.weight"]
    ids = adapter.option_rows.ids
    assert emb.dtype == torch.bfloat16 and state["model.language_model.layers.3.a.weight"].dtype == torch.bfloat16
    assert torch.equal(emb[ids], adapter.option_rows.rows.detach().to(torch.bfloat16))   # the rows were folded in
    assert adapter._hook is None                                            # the model is storage-dtype now: usable
    assert len(adapter.expected_probabilities(T.build_rows(toy_lines(1), model, None))[0]) == 3


def test_copy_remote_files_keeps_what_save_pretrained_wrote(tmp_path):
    src, out = make_source(tmp_path), tmp_path / "o"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"model_type": "lfm3_vl", "d1_codes": 7}))
    (out / "generation_config.json").write_text("{}")
    copied = T.copy_remote_files(src, out)
    assert "config.json" not in copied and "generation_config.json" not in copied
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["d1_codes"] == 7 and cfg["d1_set_size"] == 16              # the saved value wins, missing keys are added


# --------------------------------------------------------------------------- the memory estimate
def test_memory_estimate_and_abort():
    model, adapter = fake_adapter(dtype=torch.bfloat16, fp32=False)
    e0, e2, e4 = (adapter.memory_estimate(n, 4096) for n in (0, 2, 4))
    assert e0["total"] < e2["total"] < e4["total"]
    assert e4["trainable_params"] > e2["trainable_params"] > e0["trainable_params"]
    assert e2["adam"] == pytest.approx(2 * e2["grads"]) and e2["grads"] == pytest.approx(e2["weights_trained"])
    assert adapter.memory_estimate(4, 4096, grad_ckpt=True)["activations"] < e4["activations"]
    assert adapter.memory_estimate(4, 8192)["activations"] > e4["activations"]
    T.check_memory(e2, max_gb=e2["total"] + 1)
    with pytest.raises(SystemExit, match="exceeds --max-gb"):
        T.check_memory(e4, max_gb=e4["total"] - 0.5)
    assert "GiB" in T.fmt_estimate(e2)


def test_main_aborts_before_training_when_over_budget(tmp_path, monkeypatch):
    src = make_source(tmp_path)
    model, adapter = fake_adapter(dtype=torch.bfloat16, fp32=False)
    monkeypatch.setattr(T, "source_dir", lambda init: src)
    monkeypatch.setattr(T, "load_lfm3", lambda path, args: (model, adapter))
    args = ["--variant", "joint", "--init", "LiquidAI/d1-2B", "--out", str(tmp_path / "o"), "--device", "cpu"]
    with pytest.raises(SystemExit, match="--max-gb"):
        T.main(args + ["--max-gb", "0.001", "--data", str(tmp_path)])
    assert not (tmp_path / "o").exists()                                   # nothing was trained or written
    est = T.main(args + ["--estimate-only", "--train-layers", "2"])["estimates"]
    assert set(est) == {2, 8, 12, 24}


# --------------------------------------------------------------------------- the real remote code, tiny random weights
def tiny_2b():
    """A tiny random-init D1Model (4 layers) built from the cached d1-2B remote code, and that snapshot."""
    snap = SNAPSHOTS[-1]
    pkg = types.ModuleType("d1_2b_pkg")
    pkg.__path__ = [str(snap)]
    sys.modules["d1_2b_pkg"] = pkg
    modeling = importlib.import_module("d1_2b_pkg.modeling_d1_lfm3_vl")
    native = importlib.import_module("d1_2b_pkg.native_lfm3_vl")
    attn = importlib.import_module("d1_2b_pkg.attention")
    attn.AttentionInterface.register("d1", attn.d1_attention)   # a reloaded checkpoint's copy replaces this one
    cfg = json.loads((snap / "config.json").read_text())
    text = dict(cfg["text_config"], hidden_size=64, intermediate_size=192, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, sliding_window=32,
                layer_types=["conv", "full_attention", "conv", "sliding_attention"])
    vision = dict(cfg["vision_config"], hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1, layer_types=["conv"])
    rest = {k: v for k, v in cfg.items() if k not in ("text_config", "vision_config", "auto_map", "architectures")}
    config = native.Lfm3VlConfig(text_config=text, vision_config=vision, **{**rest, "projector_hidden_size": 64})
    torch.manual_seed(0)
    model = modeling.D1Model._from_config(config, attn_implementation="d1", dtype=torch.float32).eval()
    return model, modeling, snap


TINY_LINES = [
    line(0, "[single_elixir|2:48] elixir 10. hand: knight, ?\nfield: none\ntactics: best knight @[5,2.5]",
         {"play": {"type": "choice", "instructions": "Which of these plays is best right now?",
                   "criteria": {f"knight|left|{z}": f"knight left lane, {z}" for z in ("back", "mid", "bridge", "enemy_half")}}},
         {"play": "knight|left|mid"}),
    line(1, "x " * 220 + "elixir 4", {"act": {"type": "noul", "instructions": "Should we deploy a card right now?",
                                              "criteria": {"true": "deploy a card now", "false": "wait"}}}, {"act": "true"}),
    line(2, "elixir 9", {"act": {"type": "noul", "instructions": "Should we deploy a card right now?"}}, {"act": "false"}),
]


def tiny_adapter(fp32: bool = True):
    model, modeling, snap = tiny_2b()
    runner = importlib.import_module("d1_2b_pkg.runner")
    import copy
    reference = runner.SystemOne(model=copy.deepcopy(model), tokenizer=T.Lfm3Codec(model, snap).tokenizer, calibration=None)
    codec = T.Lfm3Codec(model, snap)
    adapter = T.Lfm3Adapter(model, codec, fp32=fp32, device="cpu", calibration=json.loads((snap / "calibration.json").read_text()))
    model.d1_adapter = adapter
    return model, adapter, reference, snap


@needs_code
def test_adapter_option_logits_equal_system_one():
    """The trainer's forward is `system_one`'s: same ids, attention plan, slot, readout rows (T = 1 here)."""
    model, adapter, reference, _ = tiny_adapter()
    rows = T.build_rows(TINY_LINES, model, None)
    assert [r.tkey for r in rows] == ["choice:mid", "noul", "noul"]
    with torch.no_grad():
        zs = T.option_logits(model, rows)                                   # one padded batch of three lengths
    for ln, r, z in zip(TINY_LINES, rows, zs):
        want = reference.probabilities(ln["state"], [ln["questions"][r.name]])[0]
        got = z.softmax(-1).tolist()
        got = got[::-1] if r.qtype == 2 else got                            # system_one orders a noul [yes, no]
        assert got == pytest.approx(want, abs=1e-5)


@needs_code
def test_adapter_real_model_freezing_gradients_and_checkpointing():
    model, adapter, _, _ = tiny_adapter(fp32=True)
    rows = T.build_rows(TINY_LINES, model, None)
    params = T.set_trainable(model, 2)
    assert adapter.option_rows.rows.shape[1] == 64 and any(p is adapter.option_rows.rows for p in params)
    lm = model.model.language_model
    grads = {}
    for ckpt in (False, True):
        if ckpt:
            assert adapter.enable_grad_ckpt()
        model.train()
        for p in params:
            p.grad = None
        seen = []
        handle = lm.layers[2].register_forward_pre_hook(lambda module, args: seen.append(args[0].requires_grad))
        loss = T.batch_loss(T.option_logits(model, rows), rows)
        loss.backward()
        handle.remove()
        assert seen[0] is False       # the first trained layer's input carries no graph: frozen layers keep none
        grads[ckpt] = (float(loss.detach()), lm.layers[3].feed_forward.w1.weight.grad.clone(), adapter.option_rows.rows.grad.clone())
        assert lm.layers[1].feed_forward.w1.weight.grad is None and lm.layers[0].conv.in_proj.weight.grad is None
        assert lm.layers[2].conv.in_proj.weight.grad is not None and lm.embedding_norm.weight.grad is not None
        assert lm.embed_tokens.weight.grad is None
    assert grads[False][0] == pytest.approx(grads[True][0], abs=1e-6)
    assert torch.allclose(grads[False][1], grads[True][1], atol=1e-6) and torch.allclose(grads[False][2], grads[True][2], atol=1e-6)


@needs_code
def test_adapter_real_model_save_and_reload(tmp_path):
    model, adapter, _, snap = tiny_adapter(fp32=True)
    rows = T.build_rows(TINY_LINES, model, None)
    T.set_trainable(model, 1)
    with torch.no_grad():
        adapter.option_rows.rows.add_(0.05 * torch.randn_like(adapter.option_rows.rows))
        model.model.language_model.layers[3].feed_forward.w1.weight.mul_(1.1)
    adapter.set_temperatures({"choice:mid": 1.7, "noul": 0.9})
    out = tmp_path / "ckpt"
    T.save_checkpoint(model, out, snap)
    for name in ("model.safetensors", "config.json", "modeling_d1_lfm3_vl.py", "lfm3.py", "runner.py", "prompt.py",
                 "attention.py", "calibration.py", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "calibration.json"):
        assert (out / name).exists(), name
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["auto_map"]["AutoModel"] == "modeling_d1_lfm3_vl.D1Model" and cfg["d1_attention"] == "set"
    assert json.loads((out / "calibration.json").read_text())["temperatures"] == {"choice:mid": 1.7, "noul": 0.9}
    first = [r for r in rows if r.line_id == "l0"]
    check = adapter.reload_check(out, TINY_LINES[0], first)
    assert check["reload_ok"] and check["reload_max_abs_diff"] < 1e-4          # trained rows + layer + temperature


# --------------------------------------------------------------------------- D1Ranker: d1-2B answers, dtype
@needs_code
def test_ranker_parses_d1_2b_answers():
    """Answers as d1-2B's own `api.answer` builds them: a choice's probability dict, a noul's P(yes)."""
    importlib.import_module("d1_2b_pkg") if "d1_2b_pkg" in sys.modules else tiny_2b()
    api = importlib.import_module("d1_2b_pkg.api")
    prompt = importlib.import_module("d1_2b_pkg.prompt")
    keys = ["knight|left|mid", "giant|right|back", "archers|middle|mid", "knight|right|mid"]
    questions = {"play": {"type": "choice", "instructions": "?", "criteria": {k: k for k in keys}},
                 "act": {"type": "noul", "instructions": "go?", "criteria": {"true": "yes", "false": "no"}}}
    probs = {"play": [0.1, 0.2, 0.6, 0.1], "act": [0.8, 0.2]}                  # a noul's probs are [P(yes), P(no)]

    def fake(state, qs):
        return {"answers": {n: api.answer(prompt.as_question(q), probs[n]) for n, q in qs.items()},
                "usage": {"input_tokens": 1, "output_tokens": 0}}

    out = D1Ranker(predict_fn=fake, backend="transformers").probabilities("s", questions)
    assert out["play"] == pytest.approx([0.1, 0.2, 0.6, 0.1]) and out["act"] == pytest.approx([0.2, 0.8])


def test_ranker_dtype_validation_and_resolution(monkeypatch):
    with pytest.raises(ValueError, match="unknown d1 dtype"):
        D1Ranker(dtype="int8")
    assert D1Ranker().dtype == "auto" and D1Ranker(dtype="bf16").dtype == "bf16"
    lfm3 = {"model_type": "lfm3_vl", "text_config": {"model_type": "lfm3"}, "architectures": ["D1Model"]}
    omni = {"model_type": "d1_omni", "architectures": ["D1OmniModel"]}
    assert ranker_mod.is_lfm3_config(lfm3) and not ranker_mod.is_lfm3_config(omni) and not ranker_mod.is_lfm3_config(None)
    assert ranker_mod.is_lfm3_config({"architectures": ["Lfm3VlForConditionalGeneration"]})
    r = ranker_mod.resolve_dtype
    assert r("auto", True, "mps") == torch.bfloat16 and r("auto", False, "mps") == torch.float16
    assert r("auto", True, "cpu") == torch.float32 and r("auto", False, "cpu") == torch.float32
    assert r("bf16", False, "mps") == torch.bfloat16 and r("fp16", True, "mps") == torch.float16
    assert r("fp32", True, "mps") == torch.float32


def test_ranker_load_passes_the_resolved_dtype(tmp_path, monkeypatch):
    import transformers
    seen = []

    class FakeLoaded:
        def to(self, dev):
            seen.append(("to", dev))
            return self

        def eval(self):
            return self

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        staticmethod(lambda mid, **kw: seen.append((mid, kw["dtype"], kw["trust_remote_code"])) or FakeLoaded()))
    two_b, omni = tmp_path / "d1-2b-ft", tmp_path / "d1-omni-ft"
    two_b.mkdir()
    omni.mkdir()
    (two_b / "config.json").write_text(json.dumps({"model_type": "lfm3_vl", "text_config": {"model_type": "lfm3"}}))
    (omni / "config.json").write_text(json.dumps({"model_type": "d1_omni"}))
    for path, dtype, device, want in ((two_b, "auto", "mps", torch.bfloat16), (two_b, "fp16", "mps", torch.float16),
                                      (omni, "auto", "mps", torch.float16), (two_b, "auto", "cpu", torch.float32),
                                      (omni, "bf16", "mps", torch.bfloat16)):
        seen.clear()
        D1Ranker(model_path=str(path), backend="transformers", device=device, dtype=dtype)._load()
        assert seen[0] == (str(path), want, True), (path.name, dtype, device)


def test_eval_dtype_pass_through(tmp_path, monkeypatch):
    ds = tmp_path / "ds"
    play = {"type": "choice", "instructions": "?", "criteria": {f"k{i}|left|mid": f"k{i}" for i in range(4)}}
    (ds / "d1" / "joint").mkdir(parents=True)
    (ds / "d1" / "joint" / "val.jsonl").write_text(json.dumps(
        {"id": "g#1", "game_id": "g", "split": "val", "kind": "key", "variant": "joint", "state": "s",
         "questions": {"play": play}, "answers": {"play": "k3|left|mid"}}) + "\n")
    seen = []

    def factory(model_path=None, backend="mlx", **kw):
        seen.append((model_path, backend, kw.get("dtype")))
        return D1Ranker(model_path=model_path, backend=backend, predict_fn=lambda s, q: {"answers": {
            "play": {"type": "choice", "probabilities": dict(zip(q["play"]["criteria"], [0.1, 0.1, 0.1, 0.7]))}}})

    monkeypatch.setattr(ev, "D1Ranker", factory)
    base = ["--dataset", str(ds), "--variant", "joint", "--splits", "val", "--backend", "transformers"]
    res = ev.main(base + ["--model", "LiquidAI/d1-2B", "--dtype", "bf16"])
    assert seen[-1] == ("LiquidAI/d1-2B", "transformers", "bf16") and res["dtype"] == "bf16"
    assert (ds / "eval" / "d1_eval_transformers_d1-2B.json").exists()
    ev.main(base)
    assert seen[-1] == (None, "transformers", "auto")
    with pytest.raises(SystemExit):
        ev.main(base + ["--dtype", "fp8"])


# --------------------------------------------------------------------------- real model (slow)
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CLASHBOT_SLOW_TESTS") != "1", reason="set CLASHBOT_SLOW_TESTS=1")
@pytest.mark.skipif(not (DATASET / "d1" / "joint" / "val.jsonl").exists() or not SNAPSHOTS
                    or not (SNAPSHOTS[-1] / "model.safetensors").exists(),
                    reason="dataset/v1/d1 or the d1-2B weights are missing")
def test_real_d1_2b_one_epoch_cpu(tmp_path):
    """The real d1-2B on the cpu (fp32; bf16 is far too slow there): 4 joint lines, 1 epoch, the top layer only."""
    data = tmp_path / "joint"
    data.mkdir()
    for split, n in (("train", 4), ("val", 2)):
        lines = (DATASET / "d1" / "joint" / f"{split}.jsonl").read_text().splitlines()[:n]
        (data / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    out = tmp_path / "ckpt"
    report = T.main(["--variant", "joint", "--data", str(data), "--out", str(out), "--init", "LiquidAI/d1-2B",
                     "--epochs", "1", "--train-layers", "1", "--device", "cpu", "--max-tokens", "2048", "--accum", "2",
                     "--fp32", "--grad-ckpt"])
    assert report["arch"] == "lfm3" and report["reload_ok"] and (out / "model.safetensors").exists()
    assert (out / "modeling_d1_lfm3_vl.py").exists() and (out / "calibration.json").exists()
    assert json.loads((out / "train_report.json").read_text())["best_epoch"] == 1
    assert "choice:mid" in report["temperatures"]
