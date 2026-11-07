"""MAC-AO: the candidate generator keeps what its last call looked at (retrieval hits, KG contributions) for the
decision dashboard, and nothing about the candidates changes."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.candidates import CandidateGenerator  # noqa: E402
from clashbot.config import HOLD_KEY  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402

CARDS = load_cards()


def _state():
    """Hand zap / archers / knight / giant at 5 elixir; an enemy knight 3 s from the left tower (a KG threat)."""
    return {
        "phase": "double_elixir", "match_seconds": 150,
        "own": {"elixir": 5.0, "hand": ["zap", "archers", "knight", "giant"], "deck": ["zap", "archers", "knight", "giant"],
                "towers": {"king": None, "left": 4000, "right": 5000}, "recent_plays": []},
        "opponent": {"elixir_estimate": 4.0, "towers": {"king": None, "left": 3000, "right": 4858},
                     "deck_known": [], "recent_plays": []},
        "units": [{"unit_class": "knight", "card": "knight", "side": "enemy", "tile": [4, 12],
                   "eta_tower": {"tower": "own_left", "s": 3.0}}],
        "threats": [],
    }


class Index:
    """StateIndex stand-in returning full Hit-like rows."""

    def __init__(self, hits):
        self.hits = hits

    def search(self, state, t, context_text, k=10, depth=50, **filters):
        return [types.SimpleNamespace(row=i, id=f"v#{i}.0", game_id=f"v-m0.{i}", kind="play", action_key=a, verdict="positive",
                                      elixir_trade=1.5, score=s, ranks={"knn": i + 1, "bm25": 3})
                for i, (a, s) in enumerate(self.hits)]


class KG:
    def __init__(self, table):
        self.table = table

    def counter_scores(self, threat):
        return dict(self.table.get(threat, {}))


HITS = [("knight|left|mid", 0.03), ("zap|left|mid", 0.02), (HOLD_KEY, 0.015), ("fireball|left|mid", 0.01)]


def _gen():
    return CandidateGenerator(Index(HITS), KG({"knight": {"archers": 1.0, "giant": 0.5, "fireball": 3.0}}), CARDS)


def test_last_retrieval_and_last_kg_are_filled_by_generate():
    gen = _gen()
    assert gen.last_retrieval == [] and gen.last_kg == []
    gen.generate(_state(), 10.0, "ctx", n=4)
    assert [h["action_key"] for h in gen.last_retrieval] == [a for a, _ in HITS]           # every hit, in order, hold included
    first = gen.last_retrieval[0]
    assert first == {"id": "v#0.0", "game_id": "v-m0.0", "kind": "play", "action_key": "knight|left|mid",
                     "verdict": "positive", "elixir_trade": 1.5, "score": 0.03, "ranks": {"knn": 1, "bm25": 3}}
    # only the contributions that reached a legal key: fireball is not in the hand
    assert [(k["threat"], k["card"], k["key"]) for k in gen.last_kg] == [
        ("knight", "archers", "archers|left|mid"), ("knight", "giant", "giant|left|mid")]
    assert gen.last_kg[0]["score"] == 1.0 and gen.last_kg[1]["score"] == 0.5


def test_each_call_resets_the_logs():
    gen = _gen()
    gen.generate(_state(), 10.0, "ctx", n=4)
    quiet = _state()
    quiet["units"] = []                                       # no threat this time
    gen.generate(quiet, 11.0, "ctx", n=4)
    assert gen.last_kg == [] and len(gen.last_retrieval) == len(HITS)
    gen.index = Index([])
    gen.generate(quiet, 12.0, "ctx", n=4)
    assert gen.last_retrieval == []


def test_hits_without_the_optional_fields_still_log():
    class Bare:
        def search(self, *a, **k):
            return [types.SimpleNamespace(action_key="zap|left|mid", score=0.02)]

    gen = CandidateGenerator(Bare(), KG({}), CARDS)
    gen.generate(_state(), 10.0, "ctx", n=4)
    assert gen.last_retrieval == [{"id": None, "game_id": None, "kind": None, "action_key": "zap|left|mid",
                                   "verdict": None, "elixir_trade": None, "score": 0.02, "ranks": {}}]


def test_candidates_do_not_depend_on_the_logging():
    a, b = _gen(), _gen()
    before = [(c.key, c.score, c.sources, c.tile, c.note) for c in a.generate(_state(), 10.0, "ctx", n=4)]
    b.last_retrieval, b.last_kg = ["junk"], ["junk"]          # stale values from an earlier call are not read
    after = [(c.key, c.score, c.sources, c.tile, c.note) for c in b.generate(_state(), 10.0, "ctx", n=4)]
    assert before == after
    assert [k for k, *_ in before] == ["knight|left|mid", "zap|left|mid", "archers|left|mid", "giant|left|mid"]
    assert before[2][1:3] == (0.5, {"knn": 0.0, "kg": 1.0})                                  # score = knn + 0.5 * kg
    notes = a.option_notes(_state(), 10.0, "ctx", ["zap|left|mid"])                          # the training export path
    assert notes == {"zap|left|mid": "pros played this in similar states (support 0.67)"}


@pytest.mark.parametrize("n", [1, 2, 4])
def test_generate_sizes_unchanged(n):
    assert len(_gen().generate(_state(), 10.0, "ctx", n=n)) == n
