"""WP1: shared rules package (clashbot/) and the clean dataset builder."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.config import ZONES  # noqa: E402
from clashbot.game_rules import (  # noqa: E402
    action_key, lane_of, legal_cards, legal_zones, load_cards, parse_action_key, zone_of,
)
from clashbot.state_text import render_context, render_state_text, unit_token  # noqa: E402
from clashbot.units import load_unit_map, resolve  # noqa: E402

BUILDER = ROOT / "tools" / "data" / "build_clean_dataset.py"
GAMES = "LGb0mz8Sb8w-m0.0,vp7jP_vjxcU-m0.1"

CARDS = {
    "knight": {"card_type": "Troop", "elixir_cost": 3.0},
    "fireball": {"card_type": "Spell", "elixir_cost": 4.0},
    "miner": {"card_type": "Troop", "elixir_cost": 3.0},
    "golem": {"card_type": "Troop", "elixir_cost": 8.0},
    "princess-tower": {"card_type": "Tower Troop", "elixir_cost": 0.0},
    "mystery": {"card_type": "Troop", "elixir_cost": None},
}


def _state(hand, deck, elixir, left=4000, right=4000):
    return {"own": {"hand": hand, "deck": deck, "elixir": elixir},
            "opponent": {"towers": {"king": 1, "left": left, "right": right}}}


def test_lane_zone_boundaries():
    assert [lane_of([c, 0]) for c in (5, 6, 11, 12)] == ["left", "middle", "middle", "right"]
    assert [zone_of([0, r]) for r in (4, 5, 11, 12, 16, 17)] == [
        "back", "mid", "mid", "bridge", "bridge", "enemy_half"]
    assert lane_of(None) is None and zone_of(None) is None


def test_action_key_roundtrip():
    key = action_key("knight", "left", "mid")
    assert key == "knight|left|mid"
    assert parse_action_key(key) == {"type": "play", "card": "knight", "lane": "left", "zone": "mid"}
    assert parse_action_key("hold") == {"type": "hold", "card": None, "lane": None, "zone": None}
    assert action_key("knight", None, "mid") is None
    with pytest.raises(ValueError):
        parse_action_key("x|y")


def test_legal_cards():
    st = _state(["knight", "golem", None, "princess-tower"], ["fireball"], 3.0)
    assert legal_cards(st, CARDS) == ["knight"]
    assert legal_cards(_state(["golem", None, None, None], [], 7.5), CARDS) == ["golem"]
    assert legal_cards(_state(["golem", None, None, None], [], 7.4), CARDS) == []
    st2 = _state([None] * 4, ["knight", "fireball", "knight", "golem", "mystery"], 4.0)
    assert legal_cards(st2, CARDS) == ["knight", "fireball"]


def test_legal_zones():
    st = _state([], [], 5)
    assert legal_zones("fireball", st, CARDS) == list(ZONES)
    assert legal_zones("knight", st, CARDS) == ["back", "mid", "bridge"]
    assert legal_zones("knight", _state([], [], 5, left=0), CARDS) == ["back", "mid", "bridge", "enemy_half"]
    assert legal_zones("knight", _state([], [], 5, left=None), CARDS) == ["back", "mid", "bridge"]
    assert len(legal_zones("miner", st, CARDS)) == 4


def test_resolve():
    r = resolve("skeleton-evolution")
    assert r["card"] == "skeletons" and r["is_evo"] is True
    r = resolve("knight")
    assert r["card"] == "knight" and r["is_evo"] is False
    assert resolve("nonsense")["kind"] == "unknown"
    cards = load_cards()
    for name, e in load_unit_map().items():
        for c in e["candidates"]:
            assert c in cards, (name, c)


def _full_state():
    units = [{"unit_class": "skeleton-evolution", "card": "skeletons", "is_evo": True, "side": "enemy",
              "tile": [8, 6], "heading": "advancing", "eta_tower": {"tower": "own_left", "s": 3.0}}]
    return {
        "clock": "2:00", "phase": "single_elixir", "match_seconds": 30,
        "own": {"elixir": 5.0, "hand": ["knight", None, None, None], "next_card": None,
                "deck": ["knight"], "deck_status": "invalid", "recent_plays": [],
                "towers": {"king": None, "left": 1, "right": 2}},
        "opponent": {"elixir_estimate": 3.5, "deck_known": [], "recent_plays": [],
                     "towers": {"king": None, "left": 1, "right": 2}},
        "units": units, "threats": ["skeletons*(e) advancing, tower in 3.0s"],
    }


def test_render_text():
    st = _full_state()
    action = {"type": "play", "card": "knight", "tile": [3, 9], "lane": "left", "zone": "mid", "delay_s": 1.5}
    ctx = render_context(st, 10.0)
    assert "action:" not in ctx
    assert "skeletons*(e)@[8,6]" in ctx and "[deck invalid]" in ctx
    full = render_state_text(st, action, 10.0)
    assert full.endswith("action: play knight at [3,9] (mid left) after 1.5s")
    assert full.startswith(ctx)
    assert unit_token({"unit_class": "x", "card": "skeletons", "is_evo": True}) == "skeletons*"
    assert unit_token({"unit_class": "x", "card": None}) == "x"


def _build(out: Path) -> None:
    subprocess.run([sys.executable, str(BUILDER), "--games", GAMES, "--out", str(out)],
                   check=True, capture_output=True, text=True)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_integration(tmp_path):
    _build(tmp_path)
    games = {g["game_id"]: g for g in _jsonl(tmp_path / "games.jsonl")}
    assert set(games) == set(GAMES.split(","))
    recs = _jsonl(tmp_path / "states.jsonl")
    bad = [r for r in recs if r["game_id"] == "vp7jP_vjxcU-m0.1"]
    assert games["vp7jP_vjxcU-m0.1"]["deck_status"] == "invalid"
    assert bad and all(r["train_eligible"] is False for r in bad)
    for r in recs:
        assert not ({"source", "commentary", "context_refs"} & set(r))
        assert len(r["state"]["opponent"]["deck_known"]) <= 8
        assert "action:" not in r["context_text"]
    prov = {p["game_id"]: p for p in _jsonl(tmp_path / "provenance.jsonl")}
    splits = json.loads((tmp_path / "splits.json").read_text())
    for gid, g in games.items():
        vid = prov[gid]["video_id"]
        holders = [name for name in ("train", "val", "test") if vid in splits[name]]
        assert holders == [g["split"]]
    lg = games["LGb0mz8Sb8w-m0.0"]
    assert lg["opponent_deck_source"] == "match_file"
    assert "giant" in lg["opponent_deck"] and "royal-hogs" not in lg["opponent_deck"]
    prev = 0
    for r in sorted((r for r in recs if r["game_id"] == "LGb0mz8Sb8w-m0.0"), key=lambda r: r["t"]):
        dk = r["state"]["opponent"]["deck_known"]
        assert set(dk) <= set(lg["opponent_deck"])
        assert len(dk) >= prev
        prev = len(dk)


def test_determinism(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _build(a)
    _build(b)
    assert (a / "states.jsonl").read_bytes() == (b / "states.jsonl").read_bytes()


def test_data_dir_not_ignored():
    res = subprocess.run(["git", "check-ignore", "tools/data/build_clean_dataset.py"], cwd=ROOT,
                         capture_output=True, text=True)
    assert res.returncode == 1
