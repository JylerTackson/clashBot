"""MAC-D tests: D1Ranker output normalisation, choose/choose_keys, and the zero-shot evaluation script (fakes only)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot import d1_format  # noqa: E402
from clashbot.ranker import D1Ranker, RankResult, argmax_first  # noqa: E402
from tools.eval import eval_d1_zero_shot as ev  # noqa: E402

KEYS = ["knight|left|bridge", "giant|right|back", "archers|middle|mid", "knight|right|mid"]
STATE = "[single_elixir|1:00] elixir 5. hand: knight, giant, archers, ?."
DATASET = ROOT / "dataset" / "v1"


def boosted(keys: list[str], hot: int | None = None) -> list[float]:
    """The hot option (default: the last) gets 0.7, the others share 0.3."""
    hot = len(keys) - 1 if hot is None else hot
    return [0.7 if i == hot else 0.3 / (len(keys) - 1) for i in range(len(keys))]


def make_fake(style: str = "mlx", hot_by_question: dict | None = None):
    """predict_fn returning backend-shaped answers built from the questions; counts its calls."""
    calls: list[tuple] = []

    def fake(state, questions):
        calls.append((state, questions))
        answers = {}
        for name, q in questions.items():
            keys = list(q["criteria"])
            hot = (hot_by_question or {}).get(name)
            if q["type"] == "noul":
                answers[name] = {"type": "bool", "value": True, "probability": 0.8}
                continue
            probs = boosted(keys, hot)
            if style == "mlx":
                answers[name] = {"type": "choice", "value": keys[probs.index(max(probs))],
                                 "probabilities": dict(zip(keys, probs))}
            elif style == "transformers":
                answers[name] = {"type": "choice", "choice": keys[probs.index(max(probs))],
                                 "confidence": max(probs), "probabilities": dict(zip(keys, probs))}
            else:   # list-style probabilities
                answers[name] = {"type": "choice", "value": keys[probs.index(max(probs))], "probabilities": probs}
        return {"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}

    fake.calls = calls
    return fake


def test_choose_keys_returns_boosted_index():
    ranker = D1Ranker(predict_fn=make_fake())
    res = ranker.choose_keys(STATE, KEYS)
    assert isinstance(res, RankResult)
    assert res.index == 3 and res.model == "d1-mlx"
    assert len(res.probs) == 4 and abs(sum(res.probs) - 1.0) < 1e-9
    assert res.probs == pytest.approx(boosted(KEYS))
    assert ranker.last_ms >= 0.0
    res2 = D1Ranker(predict_fn=make_fake(hot_by_question={"play": 1})).choose_keys(STATE, KEYS)
    assert res2.index == 1


def test_answer_styles_normalise_identically():
    outs = []
    for style in ("mlx", "transformers", "list"):
        ranker = D1Ranker(predict_fn=make_fake(style))
        state, q = ranker.build_request(STATE, KEYS)
        outs.append(ranker.probabilities(state, q)["play"])
    assert outs[0] == pytest.approx(outs[1])
    assert outs[0] == pytest.approx(outs[2])
    assert outs[0] == pytest.approx(boosted(KEYS))


def test_list_probabilities_wrong_length_raises():
    def bad(state, questions):
        return {"answers": {"play": {"probabilities": [0.5, 0.5]}}}
    ranker = D1Ranker(predict_fn=bad)
    with pytest.raises(ValueError):
        ranker.choose_keys(STATE, KEYS)


def test_noul_normalisation():
    questions = {"act": {"type": "noul", "instructions": "Deploy now?", "criteria": {"true": "yes", "false": "no"}}}
    cases = [({"probability": 0.8}, [0.2, 0.8]), ({"noul": 0.3}, [0.7, 0.3]),
             ({"probabilities": {"true": 0.9, "false": 0.1}}, [0.1, 0.9])]
    for ans, want in cases:
        ranker = D1Ranker(predict_fn=lambda s, q, ans=ans: {"answers": {"act": ans}})
        assert ranker.probabilities(STATE, questions)["act"] == pytest.approx(want)


def test_score_dict_sorted_by_level():
    questions = {"urg": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]}}
    ans = {"probabilities": {"2": 0.5, "0": 0.2, "1": 0.3}}
    ranker = D1Ranker(predict_fn=lambda s, q: {"answers": {"urg": ans}})
    assert ranker.probabilities(STATE, questions)["urg"] == pytest.approx([0.2, 0.3, 0.5])


def test_choose_needs_keys():
    ranker = D1Ranker(predict_fn=make_fake())
    with pytest.raises(ValueError, match="action keys"):
        ranker.choose(STATE, ["play knight in the left lane, at the bridge", "hold elixir and play nothing yet"])
    with pytest.raises(ValueError, match="action keys"):
        ranker.choose(STATE, ["knight|left|bridge", "hold"])
    a, b = ranker.choose(STATE, KEYS), ranker.choose_keys(STATE, KEYS)
    assert (a.index, a.probs, a.model) == (b.index, b.probs, b.model)


def test_backend_validation_and_lazy_imports():
    with pytest.raises(ValueError):
        D1Ranker(backend="onnx")
    saved = {name: sys.modules.pop(name) for name in list(sys.modules)
             if name.split(".")[0] in ("mlx_vlm", "transformers")}
    try:
        ranker = D1Ranker()
        assert ranker.model_id == "LiquidAI/d1-omni-600M-MLX-fp16"
        assert D1Ranker(backend="transformers").model_id == "LiquidAI/d1-omni-600M"
        assert D1Ranker(model_path="/some/where").model_id == "/some/where"
        assert not any(name.split(".")[0] in ("mlx_vlm", "transformers") for name in sys.modules)
    finally:
        sys.modules.update(saved)


def test_ties_go_to_first_option():
    assert argmax_first([0.25] * 4) == 0
    assert argmax_first([0.1, 0.4, 0.4, 0.1]) == 1
    fake = lambda s, q: {"answers": {"play": {"probabilities": [0.25] * 4}}}   # noqa: E731
    assert D1Ranker(predict_fn=fake).choose_keys(STATE, KEYS).index == 0


def test_request_is_valid_and_bad_context_raises_before_backend():
    fake = make_fake()
    ranker = D1Ranker(predict_fn=fake)
    state, questions = ranker.build_request(STATE, KEYS, plan="pressure the left lane")
    d1_format.validate(state, questions)
    assert state.endswith("[plan] pressure the left lane")
    assert list(questions["play"]["criteria"]) == KEYS
    with pytest.raises(ValueError):
        ranker.choose_keys("bad <|context|>", KEYS)
    assert fake.calls == []
    ranker.choose_keys(STATE, KEYS)
    assert len(fake.calls) == 1


# --- evaluation script --------------------------------------------------------------------------

def joint_line(i: int, gold: int, kind: str = "play", split: str = "val") -> dict:
    keys = [f"{c}|left|mid" for c in ("a", "b", "c", "d")]
    q = {"play": {"type": "choice", "instructions": d1_format.JOINT_INSTRUCTIONS,
                  "criteria": {k: f"{k} desc" for k in keys}}}
    return {"id": f"g#{i}", "game_id": "g", "split": split, "kind": kind, "variant": "joint", "state": f"state {i}",
            "questions": q, "answers": {"play": keys[gold]}}


def fact_line(i: int, card: str, lane: str, zone: str, split: str = "val") -> dict:
    q = {"card": {"type": "choice", "instructions": "card?", "criteria": {"a": "A", "b": "B", "c": "C"}},
         "lane": {"type": "choice", "instructions": "lane?", "criteria": dict(d1_format.LANE_CRITERIA)},
         "zone": {"type": "choice", "instructions": "zone?", "criteria": dict(d1_format.ZONE_CRITERIA)}}
    return {"id": f"g#{i}", "game_id": "g", "split": split, "kind": "play", "variant": "factorized",
            "state": f"state {i}", "questions": q, "answers": {"card": card, "lane": lane, "zone": zone}}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def tiny_dataset(tmp_path):
    ds = tmp_path / "ds"
    # fake predicts the last option for every question -> joint pred index 3; factorized pred (c, right, enemy_half)
    write_jsonl(ds / "d1" / "joint" / "val.jsonl",
                [joint_line(0, 3), joint_line(1, 0, kind="key"), joint_line(2, 3, kind="key")])
    write_jsonl(ds / "d1" / "factorized" / "val.jsonl",
                [fact_line(0, "c", "right", "enemy_half"), fact_line(1, "c", "left", "enemy_half")])
    (ds / "ranker").mkdir()
    (ds / "ranker" / "baselines.json").write_text(json.dumps(
        {"ranker": {"val": {"random_expected": {"acc": 0.25}, "popularity": {"acc": 0.31},
                            "placement_prior": {"acc": 0.37}}}}))
    return ds


def test_eval_script_on_tiny_dataset(tiny_dataset, tmp_path, monkeypatch, capsys):
    fake = make_fake("mlx")
    monkeypatch.setattr(ev, "D1Ranker", lambda backend="mlx", **kw: D1Ranker(backend=backend, predict_fn=fake))
    out = tmp_path / "out" / "r.json"
    res = ev.main(["--dataset", str(tiny_dataset), "--variant", "both", "--splits", "val", "--out", str(out)])
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["backend"] == "mlx" and saved["limit"] is None
    assert saved["model_id"] == "LiquidAI/d1-omni-600M-MLX-fp16"
    assert saved["baselines"]["val"]["placement_prior"] == 0.37
    joint = saved["variants"]["joint"]["val"]
    assert joint["n"] == 3
    assert joint["acc"] == pytest.approx(2 / 3, abs=1e-4)           # golds 3, 0, 3 vs pred 3
    assert joint["acc_by_kind"] == {"key": {"acc": 0.5, "n": 2}, "play": {"acc": 1.0, "n": 1}}
    assert joint["pred_position_counts"] == [0, 0, 0, 3] and joint["gold_position_counts"] == [1, 0, 0, 2]
    assert joint["mean_gold_prob"] == pytest.approx((0.7 + 0.1 + 0.7) / 3, abs=1e-4)
    assert joint["option_set_shortcut"] == pytest.approx(0.25)       # distinct cards, same lane|zone: all tie
    assert joint["d1_agrees_with_shortcut"] == 1.0
    assert joint["ms_p50"] is not None and joint["ms_p95"] is not None
    fz = saved["variants"]["factorized"]["val"]
    for k in ("card", "lane", "zone", "all_three", "key_acc"):
        assert k in fz
    assert (fz["card"], fz["lane"], fz["zone"]) == (1.0, 0.5, 1.0)
    assert fz["all_three"] == 0.5 and fz["key_acc"] == 0.5 and fz["n"] == 2
    assert len(saved["per_example"]) == 5
    ex = next(e for e in saved["per_example"] if e["variant"] == "joint" and e["id"] == "g#1")
    assert ex["pred"] == "d|left|mid" and ex["gold"] == "a|left|mid" and len(ex["probs"]) == 4
    assert res["variants"]["joint"]["val"]["acc"] == joint["acc"]
    printed = capsys.readouterr().out
    assert "| split | n | d1 acc | option_set_shortcut | placement_prior | popularity | random | ms p50 | ms p95 |" in printed
    assert "| split | n | card | lane | zone | all three | key | ms p50 |" in printed
    # warm-up call is not part of the evaluated lines: 1 warm-up + 3 joint + 2 factorized
    assert len(fake.calls) == 6


def test_eval_limit(tiny_dataset, tmp_path, monkeypatch):
    fake = make_fake("transformers")
    monkeypatch.setattr(ev, "D1Ranker", lambda backend="mlx", **kw: D1Ranker(backend=backend, predict_fn=fake))
    res = ev.main(["--dataset", str(tiny_dataset), "--variant", "joint", "--splits", "val", "--limit", "2",
                   "--out", str(tmp_path / "r.json")])
    assert res["variants"]["joint"]["val"]["n"] == 2 and res["limit"] == 2
    assert "factorized" not in res["variants"]


def test_compare(tmp_path):
    def entry(i, pred, probs, variant="joint"):
        return {"id": f"g#{i}", "variant": variant, "split": "val", "pred": pred, "gold": "x", "probs": probs}
    a = {"per_example": [entry(0, "k1", [0.7, 0.1, 0.1, 0.1]), entry(1, "k2", [0.2, 0.5, 0.2, 0.1]),
                         entry(2, "k3", [0.25] * 4), entry(3, "only_a", [0.25] * 4)]}
    b = {"per_example": [entry(0, "k1", [0.6, 0.2, 0.1, 0.1]), entry(1, "k9", [0.2, 0.3, 0.4, 0.1]),
                         entry(2, "k3", [0.25] * 4)]}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    res = ev.main(["--compare", str(pa), str(pb)])
    assert res["n_common"] == 3
    assert res["top_agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert res["max_abs_prob_diff"] == pytest.approx(0.2)
    assert res["mean_abs_prob_diff"] == pytest.approx((0.1 + 0.1 + 0.2 + 0.2) / 12)


def test_option_set_shortcut():
    keys = ["a|left|mid", "a|left|back", "b|left|mid", "c|right|bridge"]
    assert ev.shortcut_argmax(keys) == [0]
    assert ev.shortcut_accuracy(keys, 0) == 1.0
    assert ev.shortcut_accuracy(keys, 1) == 0.0
    distinct = ["a|left|mid", "b|right|back", "c|middle|bridge", "d|left|enemy_half"]
    for gold in range(4):
        assert ev.shortcut_accuracy(distinct, gold) == 0.25


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CLASHBOT_SLOW_TESTS") != "1", reason="set CLASHBOT_SLOW_TESTS=1 for real-model tests")
@pytest.mark.skipif(not (DATASET / "d1" / "joint" / "val.jsonl").exists(), reason="dataset/v1/d1/joint/val.jsonl missing")
def test_real_mlx_backend_one_line():
    with open(DATASET / "d1" / "joint" / "val.jsonl") as f:
        line = json.loads(f.readline())
    ranker = D1Ranker(backend="mlx")
    probs = ranker.probabilities(line["state"], line["questions"])["play"]
    assert len(probs) == 4 and abs(sum(probs) - 1.0) < 1e-3
