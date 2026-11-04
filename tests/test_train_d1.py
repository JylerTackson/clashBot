"""MAC-T tests: d1 fine-tuning (tools/train/train_d1.py) on a tiny fake model, and eval_d1_zero_shot --model."""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.ranker import D1Ranker  # noqa: E402
from tools.eval import eval_d1_zero_shot as ev  # noqa: E402
from tools.train import train_d1 as T  # noqa: E402

DATASET = ROOT / "dataset" / "v1"
SNAPSHOTS = sorted((Path.home() / ".cache/huggingface/hub/models--LiquidAI--d1-omni-600M/snapshots").glob("*"))


# --------------------------------------------------------------------------- fakes with d1's interface
class Block(nn.Module):
    """One position-wise layer plus a mean-pooled context, so a marker can see the state."""

    def __init__(self, d: int):
        super().__init__()
        self.a, self.b = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, h, pad):
        w = pad[..., None].to(h.dtype)
        ctx = (self.b(h) * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)
        return h + torch.tanh(self.a(h) + ctx)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(50, 8)
        self.layers = nn.ModuleList([Block(8), Block(8)])
        self.embedding_norm = nn.LayerNorm(8)

    def forward(self, h, pad, offsets):
        for layer in self.layers:
            h = layer(h, pad)
        return self.embedding_norm(h)


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(8, 1)

    def forward(self, h, pad, marker_pos, marker_mask, qtype):
        g = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
        return self.out(g).squeeze(-1).masked_fill(~marker_mask, -1e4)


@dataclass
class FakeConfig:
    max_length: int = 64
    temperatures: dict | None = None

    def __post_init__(self):
        self.temperatures = {}


class FakeModel(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder, self.head = FakeEncoder(), FakeHead()
        self.config = FakeConfig()
        self.tokenizer = None


@dataclass
class FakeQuestion:
    type: str
    criteria: dict | None
    options: int


class FakeMod:
    QTYPES = {"choice": 0, "score": 1, "noul": 2}

    @staticmethod
    def as_question(d: dict) -> FakeQuestion:
        crit = d.get("criteria")
        return FakeQuestion(d["type"], crit, 2 if d["type"] == "noul" else len(crit))

    @staticmethod
    def encode(tok, state, q, max_len, noul_default=None, audio=False):
        ids, markers = [ord(c) % 40 for c in state], []        # one token per character
        for j in range(q.options):
            markers.append(len(ids))
            ids += [40 + j, 7]                                 # the option's marker, then one text token
        ids.append(49)
        return ids[:max_len], markers

    @staticmethod
    def temperature_key(q) -> str:
        return f"{q.type}:{q.options}"


def choice_q(n: int = 3) -> dict:
    return {"type": "choice", "instructions": "?", "criteria": {f"o{j}": f"option {j}" for j in range(n)}}


def line(i: int, state: str, questions: dict, answers: dict) -> dict:
    return {"id": f"l{i}", "game_id": "g", "split": "train", "state": state, "questions": questions,
            "answers": answers}


def toy_lines(n: int, seed: int = 0) -> list[dict]:
    """The state's first character ('a', 'b' or 'c') decides the answer."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        c = rng.choice("abc")
        out.append(line(i, c + "xy", {"play": choice_q(3)}, {"play": f"o{'abc'.index(c)}"}))
    return out


def row(n_tokens: int, target: int = 0, name: str = "play", n_options: int = 3, line_id: str = "l") -> T.Row:
    return T.Row(list(range(n_tokens)), [0], 0, n_options, target, line_id, name, f"choice:{n_options}")


# --------------------------------------------------------------------------- 1. build_rows
def test_build_rows_targets_and_counts():
    noul = {"type": "noul", "instructions": "act?", "criteria": {"true": "deploy", "false": "wait"}}
    lines = [
        line(0, "abc", {"play": choice_q(4)}, {"play": "o2"}),
        line(1, "abc", {"act": noul}, {"act": "true"}),
        line(2, "abc", {"act": noul}, {"act": "false"}),
        line(3, "abc", {"card": choice_q(5), "lane": choice_q(3), "zone": choice_q(4)},
             {"card": "o4", "lane": "o0", "zone": "o3"}),
    ]
    rows = T.build_rows(lines, FakeModel(), FakeMod)
    assert [r.target for r in rows] == [2, 1, 0, 4, 0, 3]
    assert [r.name for r in rows[3:]] == ["card", "lane", "zone"] and len(rows) == 1 + 1 + 1 + 3   # 3 rows for the factorized line
    assert all(r.line_id == "l3" for r in rows[3:])
    assert [r.n_options for r in rows] == [4, 2, 2, 5, 3, 4]
    assert [r.qtype for r in rows[:2]] == [0, 2]
    assert rows[1].tkey == "noul:2" and rows[0].markers == [3, 5, 7, 9]
    assert rows[0].ids == FakeMod.encode(None, "abc", FakeMod.as_question(choice_q(4)), 64)[0]


def test_read_jsonl_limit(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text("\n".join(json.dumps({"i": i}) for i in range(5)) + "\n\n")
    assert [d["i"] for d in T.read_jsonl(path, 3)] == [0, 1, 2]
    assert len(T.read_jsonl(path)) == 5


# --------------------------------------------------------------------------- 2. make_batches
def test_make_batches_cover_and_budget():
    rng = random.Random(1)
    rows = [row(rng.randint(5, 60)) for _ in range(300)] + [row(500)]      # one row over the budget on its own
    batches = T.make_batches(rows, 200, random.Random(7))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(rows)))                           # every row exactly once
    for b in batches:
        longest = max(len(rows[i].ids) for i in b)
        assert len(b) * longest <= 200 or len(b) == 1
    assert [len(rows[i].ids) for b in batches for i in b if len(rows[i].ids) > 200] == [500]
    assert any(len(b) > 1 for b in batches)
    assert T.make_batches(rows, 200, random.Random(7)) == batches           # deterministic for a fixed rng
    assert T.make_batches(rows, 200, random.Random(8)) != batches
    plain = T.make_batches(rows, 200)                                       # no rng: no shuffling, same cover
    assert sorted(i for b in plain for i in b) == list(range(len(rows)))
    assert plain == T.make_batches(rows, 200)


# --------------------------------------------------------------------------- 3. training on a toy set
def test_training_learns_toy_task():
    model = FakeModel()
    T.set_trainable(model, 2)
    rows = T.build_rows(toy_lines(60), model, FakeMod)
    val = T.build_rows(toy_lines(30, seed=5), model, FakeMod)
    res = T.train(model, rows, val, variant="joint", epochs=40, lr=3e-2, warmup=0.05, max_tokens=120, accum=1,
                  log=lambda *_: None)
    assert res["steps"] <= 200
    hist = res["history"]
    assert hist[-1]["train_loss"] < 0.5 * hist[0]["train_loss"]
    _, logits = T.evaluate(model, rows, 120)
    assert T.compute_metrics(rows, logits, "joint")["play"] >= 0.95
    assert res["best_epoch"] >= 1 and len(res["best_logits"]) == len(val)
    # the restored weights reproduce the best epoch's val logits
    _, again = T.evaluate(model, val, 120)
    assert all(np.allclose(a, b, atol=1e-5) for a, b in zip(again, res["best_logits"]))


def test_best_epoch_is_restored():
    model = FakeModel()
    T.set_trainable(model, 2)
    rows = T.build_rows(toy_lines(30), model, FakeMod)
    val = T.build_rows(toy_lines(20, seed=3), model, FakeMod)
    res = T.train(model, rows, val, variant="joint", epochs=4, lr=3e-2, max_tokens=120, accum=1, log=lambda *_: None)
    best = res["history"][res["best_epoch"] - 1]["val_metrics"]["play"]
    assert best == max(h["val_metrics"]["play"] for h in res["history"])
    _, logits = T.evaluate(model, val, 120)
    assert T.compute_metrics(val, logits, "joint")["play"] == best


# --------------------------------------------------------------------------- 4. freezing
def test_freezing_train_layers():
    model = FakeModel()
    params = T.set_trainable(model, 1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rows = T.build_rows(toy_lines(30), model, FakeMod)
    T.train(model, rows, rows[:10], variant="joint", epochs=2, lr=3e-2, max_tokens=120, accum=2, log=lambda *_: None)
    after = dict(model.named_parameters())
    for name in before:
        same = torch.equal(before[name], after[name].detach())
        if name.startswith(("encoder.layers.0.", "encoder.embed_tokens", "encoder.embedding_norm")):
            assert same, f"{name} must stay frozen and bit-identical"
    assert not torch.equal(before["head.out.weight"], after["head.out.weight"].detach())
    assert not torch.equal(before["encoder.layers.1.a.weight"], after["encoder.layers.1.a.weight"].detach())
    assert len(params) == len([p for p in model.parameters() if p.requires_grad])
    assert not any(p.requires_grad for p in model.encoder.layers[0].parameters())
    # all layers: the final norm trains too, the embeddings never do
    T.set_trainable(model, 2)
    assert model.encoder.embedding_norm.weight.requires_grad and not model.encoder.embed_tokens.weight.requires_grad


# --------------------------------------------------------------------------- 5. temperature fit
def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    n = 20000
    z = rng.normal(0, 2.0, (n, 4))
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    target = (p.cumsum(1) > rng.random((n, 1))).argmax(1)               # labels drawn from the calibrated model
    rows = [row(5, int(t), n_options=4) for t in target]
    assert T.fit_temperatures(rows, list(z))["choice:4"] == pytest.approx(1.0, abs=0.1)
    assert T.fit_temperatures(rows, list(z * 3))["choice:4"] == pytest.approx(3.0, abs=0.1)
    # two keys are fitted separately; the grid is 0.5 .. 3.0
    mixed = rows[:1000] + [row(5, int(t), n_options=4)._replace(tkey="noul:2") for t in target[1000:2000]]
    out = T.fit_temperatures(mixed, list(z[:1000]) + list(z[1000:2000] * 2))
    assert set(out) == {"choice:4", "noul:2"} and out["noul:2"] == pytest.approx(2.0, abs=0.25)
    assert 0.5 <= min(out.values()) and max(out.values()) <= 3.0


# --------------------------------------------------------------------------- 6. balanced weights, loss, metrics
def test_class_weights_imbalanced_act():
    rows = [row(5, 1, "act", 2) for _ in range(90)] + [row(5, 0, "act", 2) for _ in range(10)]
    w = T.class_weights(rows)
    assert w[0] / w[1] == pytest.approx(9.0)
    # weighted mean: the minority class counts as much as the majority in total
    zs = [torch.tensor([0.0, 0.0]) for _ in rows]
    assert float(T.batch_loss(zs, rows, w)) == pytest.approx(float(np.log(2)), abs=1e-6)
    skew = [torch.tensor([-4.0, 4.0]) for _ in rows]                     # always predicts true
    unweighted, weighted = float(T.batch_loss(skew, rows)), float(T.batch_loss(skew, rows, w))
    assert weighted > 4 * unweighted


def test_batch_loss_pads_options():
    zs = [torch.tensor([2.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 3.0])]
    rows = [row(5, 0, n_options=4), row(5, 1, n_options=2)]
    expect = (float(torch.nn.functional.cross_entropy(zs[0][None], torch.tensor([0])))
              + float(torch.nn.functional.cross_entropy(zs[1][None], torch.tensor([1])))) / 2
    assert float(T.batch_loss(zs, rows)) == pytest.approx(expect, abs=1e-6)


def test_metrics_factorized_and_act():
    names = ("card", "lane", "zone")
    rows = [row(5, t, n, 3, f"l{i}") for i, tt in enumerate([(0, 1, 2), (1, 1, 1)]) for n, t in zip(names, tt)]
    best = lambda t: np.eye(3)[t] * 5.0                                  # noqa: E731
    logits = [best(0), best(1), best(2), best(1), best(0), best(1)]     # second line: lane wrong
    m = T.compute_metrics(rows, logits, "factorized")
    assert m["card"] == 1.0 and m["lane"] == 0.5 and m["zone"] == 1.0 and m["all_three"] == 0.5
    act = [row(5, t, "act", 2, f"a{i}") for i, t in enumerate([0, 0, 1, 1, 1, 1])]
    pred = [0, 1, 1, 1, 1, 0]
    m = T.compute_metrics(act, [np.eye(2)[p] for p in pred], "act")
    assert m["acc"] == pytest.approx(4 / 6) and m["balanced_acc"] == pytest.approx((0.5 + 0.75) / 2)
    assert m["precision_wait"] == pytest.approx(0.5) and m["recall_wait"] == pytest.approx(0.5)


def test_lr_factor_warmup_then_cosine():
    total = 100
    f = [T.lr_factor(s, total, 0.05) for s in range(total)]
    assert f[0] < f[4] <= 1.0 and f[4] == pytest.approx(1.0)             # linear warmup over 5 steps
    assert all(a >= b for a, b in zip(f[5:], f[6:])) and f[-1] < 0.01


# --------------------------------------------------------------------------- 7. eval_d1_zero_shot --model
def fake_predict(state, questions):
    answers = {}
    for name, q in questions.items():
        keys = list(q["criteria"])
        probs = [0.1] * (len(keys) - 1) + [0.7]
        answers[name] = {"type": "choice", "value": keys[-1], "probabilities": dict(zip(keys, probs))}
    return {"answers": answers}


def test_eval_model_option(tmp_path, monkeypatch):
    ds = tmp_path / "ds"
    play = {"type": "choice", "instructions": "?", "criteria": {f"k{i}|left|mid": f"k{i}" for i in range(4)}}
    (ds / "d1" / "joint").mkdir(parents=True)
    (ds / "d1" / "joint" / "val.jsonl").write_text(json.dumps(
        {"id": "g#1", "game_id": "g", "split": "val", "kind": "key", "variant": "joint", "state": "s",
         "questions": {"play": play}, "answers": {"play": "k3|left|mid"}}) + "\n")
    seen = []

    def factory(model_path=None, backend="mlx", **kw):
        seen.append((model_path, backend))
        return D1Ranker(model_path=model_path, backend=backend, predict_fn=fake_predict)

    monkeypatch.setattr(ev, "D1Ranker", factory)
    base = ["--dataset", str(ds), "--variant", "joint", "--splits", "val", "--backend", "transformers"]
    res = ev.main(base + ["--model", "db/models/d1-ft-joint/"])
    out = ds / "eval" / "d1_eval_transformers_d1-ft-joint.json"
    assert out.exists() and seen[-1] == ("db/models/d1-ft-joint/", "transformers")
    saved = json.loads(out.read_text())
    assert saved["model_path"] == "db/models/d1-ft-joint/" == res["model_path"]
    assert saved["model_id"] == "db/models/d1-ft-joint/" and saved["variants"]["joint"]["val"]["acc"] == 1.0
    explicit = tmp_path / "o.json"
    ev.main(base + ["--model", "x", "--out", str(explicit)])
    assert json.loads(explicit.read_text())["model_path"] == "x" and seen[-1][0] == "x"
    plain = ev.main(base)                                                # without --model nothing changes
    assert seen[-1][0] is None and "model_path" not in plain
    assert (ds / "eval" / "d1_zero_shot_transformers.json").exists()


# --------------------------------------------------------------------------- the real code, random tiny weights
def tiny_d1():
    """A tiny random-init D1OmniModel built from the cached remote code (no pretrained weights)."""
    import importlib
    import types
    snap = SNAPSHOTS[-1]
    pkg = types.ModuleType("d1_remote_pkg")
    pkg.__path__ = [str(snap)]
    sys.modules["d1_remote_pkg"] = pkg
    modeling = importlib.import_module("d1_remote_pkg.modeling_d1")
    cfg = json.loads((snap / "config.json").read_text())
    text = dict(cfg["text_config"], hidden_size=64, intermediate_size=192, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, layer_types=["conv", "full_attention", "conv", "full_attention"])
    vision = dict(cfg["vision_config"], hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2)
    audio = dict(cfg["audio_config"], n_layers=1, d_model=32, n_heads=2, subsampling_conv_channels=16, residual_width=32)
    torch.manual_seed(0)
    model = modeling.D1OmniModel(modeling.D1OmniConfig(
        text_config=text, vision_config=vision, audio_config=audio, projector_hidden_size=64)).eval()
    from transformers import AutoTokenizer
    model.__dict__["tokenizer"] = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=True)
    return model, modeling, snap


NOUL = {"type": "noul", "instructions": "Should we act?", "criteria": {"true": "deploy", "false": "wait"}}
REAL_LINES = [line(0, "[single_elixir|2:48] elixir 10. hand: knight, ?", {"play": choice_q(4), "act": NOUL},
                   {"play": "o1", "act": "true"}),
              line(1, "x" * 400, {"card": choice_q(7), "lane": choice_q(3)}, {"card": "o6", "lane": "o2"})]


@pytest.mark.skipif(not SNAPSHOTS, reason="d1-omni-600M remote code not in the Hugging Face cache")
def test_option_logits_match_model_forward():
    """option_logits == D1OmniModel._forward (text branch), and the gradients reach only the trainable part."""
    model, modeling, _ = tiny_d1()
    rows = T.build_rows(REAL_LINES, model, modeling)
    questions = [modeling.as_question(q) for ln in REAL_LINES for q in ln["questions"].values()]
    assert [r.tkey for r in rows] == [modeling.temperature_key(q) for q in questions]
    with torch.no_grad():
        mine = T.option_logits(model, rows)
        ref = model._forward([(None, r.ids, r.markers, q, False) for r, q in zip(rows, questions)])
    for z, p, r in zip(mine, ref, rows):
        got = z.softmax(-1).tolist()
        assert (got[::-1] if r.qtype == 2 else got) == pytest.approx(p, abs=1e-6)
    T.set_trainable(model, 1)
    model.train()
    T.batch_loss(T.option_logits(model, rows), rows).backward()
    assert model.head.scorer[3].weight.grad is not None and model.encoder.layers[0].feed_forward.w1.weight.grad is None
    assert model.encoder.layers[3].feed_forward.w1.weight.grad is not None


@pytest.mark.skipif(not SNAPSHOTS, reason="d1-omni-600M remote code not in the Hugging Face cache")
def test_save_checkpoint_and_reload(tmp_path):
    model, modeling, snap = tiny_d1()
    rows = T.build_rows(REAL_LINES[:1], model, modeling)
    temps = {"choice:3-5": 1.7, "noul:2": 0.9}
    model.config.temperatures.update(temps)
    out = tmp_path / "ckpt"
    T.save_checkpoint(model, out, snap)
    for name in ("model.safetensors", "config.json", "modeling_d1.py", "prompt.py", "encoder.py", "tokenizer.json"):
        assert (out / name).exists(), name
    assert json.loads((out / "config.json").read_text())["temperatures"]["choice:3-5"] == 1.7
    check = T.reload_check(out, REAL_LINES[0], T.expected_probabilities(model, rows, model.config.temperatures))
    assert check["reload_ok"] and check["reload_max_abs_diff"] < 1e-4


# --------------------------------------------------------------------------- 8. real model (slow)
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CLASHBOT_SLOW_TESTS") != "1", reason="set CLASHBOT_SLOW_TESTS=1")
@pytest.mark.skipif(not (DATASET / "d1" / "joint" / "val.jsonl").exists() or not SNAPSHOTS,
                    reason="dataset/v1/d1 or the d1-omni-600M weights are missing")
def test_real_model_one_epoch_cpu(tmp_path):
    data = tmp_path / "joint"
    data.mkdir()
    for split, n in (("train", 8), ("val", 4)):
        lines = (DATASET / "d1" / "joint" / f"{split}.jsonl").read_text().splitlines()[:n]
        (data / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    out = tmp_path / "ckpt"
    report = T.main(["--variant", "joint", "--data", str(data), "--out", str(out), "--epochs", "1",
                     "--train-layers", "2", "--device", "cpu", "--max-tokens", "2048", "--accum", "2"])
    assert report["reload_ok"] and (out / "model.safetensors").exists() and (out / "modeling_d1.py").exists()
    assert json.loads((out / "train_report.json").read_text())["best_epoch"] == 1
    assert "choice:3-5" in report["temperatures"]
