"""WP6 tests: d1 request format, validation and the d1 dataset export."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.d1_format import (ACT_CRITERIA, ACT_NAME, card_description, factorized_request,  # noqa: E402
                                joint_request, option_description, state_string, validate, act_request)
from clashbot.game_rules import load_cards  # noqa: E402

DATASET = ROOT / "dataset" / "v1"
needs_dataset = pytest.mark.skipif(not (DATASET / "ranker" / "val.jsonl").exists(),
                                   reason="dataset/v1/ranker/val.jsonl missing")
KEYS = ["knight|left|bridge", "giant|right|back", "archers|middle|mid", "knight|right|mid"]


def test_option_description():
    assert option_description("skeletons|right|mid") == "skeletons right lane, mid, on our side"
    with pytest.raises(ValueError):
        option_description("hold")


def test_joint_request_and_state_string():
    state, q = joint_request("ctx", KEYS)
    assert state == "ctx"
    assert list(q["play"]["criteria"]) == KEYS
    assert q["play"]["type"] == "choice"
    with pytest.raises(ValueError):
        joint_request("ctx", KEYS[:1])
    with pytest.raises(ValueError):
        joint_request("ctx", [KEYS[0], KEYS[0]])
    assert state_string("ctx", "push left") == "ctx\n[plan] push left"
    assert state_string("ctx", "") == "ctx"
    with pytest.raises(ValueError):
        state_string("bad <|pad|> text")
    with pytest.raises(ValueError):
        state_string("ctx", "<|x|>")


def test_card_description():
    assert card_description("skeletons", load_cards()) == "Skeletons, 1 elixir troop"


def _factorized():
    cards = load_cards()
    state, q = factorized_request("ctx", ["skeletons", "knight"], cards)
    return state, q, {"card": "knight", "lane": "left", "zone": "mid"}


def test_validate():
    state, q, ans = _factorized()
    assert list(q) == ["card", "lane", "zone"]
    validate(state, q, ans)
    validate(*act_request("ctx"), {ACT_NAME: "true"})

    bad = json.loads(json.dumps(q))
    bad["lane"]["type"] = "ranking"
    with pytest.raises(ValueError):
        validate(state, bad)

    bad = json.loads(json.dumps(q))
    bad["lane"]["criteria"] = {"left": "left lane"}
    with pytest.raises(ValueError):
        validate(state, bad)

    with pytest.raises(ValueError):
        validate(state, q, {**ans, "lane": "top"})

    s, aq = act_request("ctx")
    with pytest.raises(ValueError):
        validate(s, aq, {ACT_NAME: "yes"})

    bad = json.loads(json.dumps(q))
    bad["zone"]["criteria"]["back"] = "back <|pad|>"
    with pytest.raises(ValueError):
        validate(state, bad)


@needs_dataset
def test_builder_integration(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "tools" / "data" / "build_d1_dataset.py")], check=True, cwd=ROOT)
    d1 = DATASET / "d1"
    stats = json.loads((d1 / "stats.json").read_text())
    for variant in ("joint", "factorized", "act"):
        for split in ("train", "val", "test"):
            lines = [json.loads(l) for l in (d1 / variant / f"{split}.jsonl").read_text().splitlines() if l.strip()]
            assert stats[variant][split]["examples"] == len(lines)
            for line in lines:
                if variant == "joint":
                    crit = line["questions"]["play"]["criteria"]
                    assert len(crit) == 4
                    assert line["answers"]["play"] in crit
                elif variant == "factorized":
                    assert line["answers"]["card"] in line["questions"]["card"]["criteria"]
                else:
                    assert line["provisional"] is True
