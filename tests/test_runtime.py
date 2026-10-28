import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.candidates import CandidateGenerator  # noqa: E402
from clashbot.config import DB_PATH, HOLD_KEY  # noqa: E402
from clashbot.executor import Calibration, homography, tile_to_screen  # noqa: E402
from clashbot.game_rules import legal_action_keys, load_cards  # noqa: E402
from clashbot.kg import KnowledgeGraph  # noqa: E402
from clashbot.loop import FastLoop, should_decide  # noqa: E402
from clashbot.ranker import format_multiple_choice  # noqa: E402
from clashbot.retrieval import StateIndex  # noqa: E402
from clashbot.tools import TOOLS, ToolContext, call_tool  # noqa: E402

STATES = ROOT / "dataset" / "v1" / "states.jsonl"
needs_db = pytest.mark.skipif(not DB_PATH.exists(), reason="db/clashbot.sqlite missing")


def _state(elixir=5.0, units=None, opp_plays=None, hand=None):
    return {
        "phase": "double_elixir", "match_seconds": 150,
        "own": {"elixir": elixir, "hand": hand or ["zap", "archers", "knight", "giant"],
                "deck": ["zap", "archers", "knight", "giant"],
                "towers": {"king": None, "left": 4000, "right": 5000}, "recent_plays": []},
        "opponent": {"elixir_estimate": 4.0, "towers": {"king": None, "left": 3000, "right": 4858},
                     "deck_known": [], "recent_plays": opp_plays or []},
        "units": units or [], "threats": [],
    }


def test_format_multiple_choice():
    out = format_multiple_choice("CTX", ["a", "b", "c"])
    assert out == "CTX\nOptions:\nA) a\nB) b\nC) c\nAnswer:"
    out = format_multiple_choice("CTX", ["a", "b", "c"], plan="go left")
    assert out == "CTX\n[plan] go left\nOptions:\nA) a\nB) b\nC) c\nAnswer:"
    assert format_multiple_choice("CTX", ["a"], plan="") == "CTX\nOptions:\nA) a\nAnswer:"
    with pytest.raises(ValueError):
        format_multiple_choice("CTX", [])
    with pytest.raises(ValueError):
        format_multiple_choice("CTX", [str(i) for i in range(9)])


def test_homography_axis_aligned():
    src = [[0, 0], [10, 0], [10, 10], [0, 10]]
    dst = [[100 + 10 * x, 100 + 10 * y] for x, y in src]
    H = homography(Calibration(src, dst))
    x, y = tile_to_screen([1, 2], H)
    assert abs(x - 110) < 1e-6 and abs(y - 120) < 1e-6


def test_call_tool_validation():
    ctx = ToolContext()
    with pytest.raises(ValueError):
        call_tool("nope", {}, ctx)
    with pytest.raises(ValueError):
        call_tool("get_game_state", {"bogus": 1}, ctx)
    with pytest.raises(ValueError):
        call_tool("search_similar_states", {"k": 99}, ctx)
    with pytest.raises(ValueError):
        call_tool("kg_query", {}, ctx)
    with pytest.raises(ValueError):
        call_tool("kg_query", {"node": "card:zap", "direction": "sideways"}, ctx)
    assert call_tool("get_current_plan", {}, ToolContext(plan_text="x")) == {"plan": "x"}
    assert len(TOOLS) == 6
    for tool in TOOLS:
        for bad in ("write", "play", "tap", "set", "delete"):
            assert bad not in tool["name"]


def test_should_decide_cases():
    cards = load_cards()
    quiet = _state(elixir=5.0)
    assert not should_decide(quiet, 10.0, 9.0, cards)
    # (a) enemy unit about to hit an own tower
    near = _state(units=[{"side": "enemy", "eta_tower": {"tower": "own_left", "s": 3.0}}])
    assert should_decide(near, 10.0, 9.0, cards)
    far = _state(units=[{"side": "enemy", "eta_tower": {"tower": "own_left", "s": 6.0}}])
    assert not should_decide(far, 10.0, 9.0, cards)
    # (b) elixir cap
    assert should_decide(_state(elixir=9.5), 10.0, 9.0, cards)
    # (c) time since last decision
    assert should_decide(quiet, 10.0, None, cards)
    assert should_decide(quiet, 12.0, 9.0, cards)
    # (d) opponent just played
    assert should_decide(_state(opp_plays=[{"t": 9.0, "card": "giant"}]), 10.0, 9.5, cards)
    assert not should_decide(_state(opp_plays=[{"t": 5.0, "card": "giant"}]), 10.0, 9.5, cards)
    # no legal card -> False even when everything else fires
    broke = _state(elixir=0.0, units=near["units"])
    assert not should_decide(broke, 10.0, None, cards)


@needs_db
def test_kg_counter_scores():
    kg = KnowledgeGraph()
    assert kg.counter_scores("hog-rider")
    assert kg.neighbors("card:hog-rider", "COUNTERS", "in", limit=3)


@needs_db
def test_candidate_generator_first_val_record():
    rec = next(json.loads(l) for l in STATES.read_text().splitlines()
               if l.strip() and json.loads(l)["split"] == "val")
    cards = load_cards()
    gen = CandidateGenerator(StateIndex(), KnowledgeGraph(), cards)
    cands = gen.generate(rec["state"], rec["t"], rec["context_text"], exclude_game=rec["game_id"])
    legal = set(legal_action_keys(rec["state"], cards))
    assert len(cands) == 4
    assert all(c.key != HOLD_KEY for c in cands)
    assert all(c.key in legal for c in cands)


def test_decide_without_legal_cards_holds():
    class Boom:
        def generate(self, *a, **k):
            return []

        def choose(self, *a, **k):
            raise AssertionError("ranker must not be called")

        def execute(self, *a, **k):
            raise AssertionError("executor must not be called")

    boom = Boom()
    loop = FastLoop(boom, boom, boom, {})
    d = loop.decide(_state(elixir=0.0, hand=["giant", "giant", "giant", "giant"]), 10.0, "ctx")
    assert d.choice == HOLD_KEY and d.tile is None and d.candidates == []


@needs_db
def test_generator_empty_when_no_legal_cards():
    cards = load_cards()
    gen = CandidateGenerator(StateIndex(), KnowledgeGraph(), cards)
    st = _state(elixir=0.0, hand=["giant", "giant", "giant", "giant"])
    assert legal_action_keys(st, cards) == [HOLD_KEY]
    assert gen.generate(st, 10.0, "ctx") == []


@needs_db
def test_search_documents_tool():
    docs = call_tool("search_documents", {"query": "graveyard"}, ToolContext())
    assert len(docs) >= 1
    assert {"doc_id", "type", "title", "text"} <= set(docs[0])
    info = call_tool("card_info", {"slug": "zap"}, ToolContext())
    assert info.get("slug") == "zap"
    assert call_tool("card_info", {"slug": "no-such-card"}, ToolContext()) == {"error": "unknown card"}


@needs_db
def test_replay_cli():
    game_id = json.loads(STATES.open().readline())["game_id"]
    proc = subprocess.run([sys.executable, "-m", "clashbot.loop", f"--replay={game_id}", "--limit", "5"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "summary:" in proc.stdout


@needs_db
def test_replay_cli_dash_game_id():
    proc = subprocess.run([sys.executable, "-m", "clashbot.loop", "--replay", "-V4H_YeMGGk-m0.0", "--limit", "2"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "summary:" in proc.stdout
