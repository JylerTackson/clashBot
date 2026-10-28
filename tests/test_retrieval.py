import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.db import fts_query  # noqa: E402
from clashbot.features import block_slices, dim, feature_names, featurize  # noqa: E402
from clashbot.retrieval import StateIndex  # noqa: E402

STATES = ROOT / "dataset" / "v1" / "states.jsonl"


def _state(**over):
    s = {
        "phase": "double_elixir", "match_seconds": 150,
        "own": {"elixir": 5.0, "hand": ["zap", None, "archers", "knight"], "deck": ["zap", "archers", "knight"],
                "towers": {"king": None, "left": 4000, "right": 5000}, "recent_plays": [
                    {"t": 98.0, "card": "zap"}]},
        "opponent": {"elixir_estimate": 4.0, "towers": {"king": None, "left": 3000, "right": 4858},
                     "deck_known": ["giant"], "recent_plays": [{"t": 50.0, "card": "giant"}]},
        "units": [{"side": "enemy", "tile": [3, 8], "eta_tower": {"tower": "own_left", "s": 2.0}}],
        "threats": [],
    }
    s.update(over)
    return s


def test_featurize_shape_norm_and_slices():
    v = featurize(_state(), 100.0)
    assert v.shape == (dim(),) and v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    sl = block_slices()
    pos = 0
    for start, stop in sl.values():
        assert start == pos
        pos = stop
    assert pos == dim() == len(feature_names())
    empty = {"phase": None, "match_seconds": None,
             "own": {"elixir": None, "hand": [None] * 4, "deck": [], "towers": {}, "recent_plays": []},
             "opponent": {"elixir_estimate": None, "towers": {}, "deck_known": [], "recent_plays": []},
             "units": [], "threats": []}
    v2 = featurize(empty, 0.0)
    assert v2.shape == (dim(),) and np.isfinite(v2).all()


def test_field_block_unit_position():
    st = _state(units=[{"side": "ally", "tile": [3, 20]}])
    v = featurize(st, 100.0)
    a, b = block_slices()["field"]
    idx = feature_names().index("field:ally:left:enemy_half")
    assert a <= idx < b
    assert v[idx] > 0


def test_fts_query():
    q = fts_query("hand: skeletons, the-log (next ?)")
    assert '"skeletons"' in q and '"log"' in q
    assert '"hand"' not in q and '"next"' not in q


def test_rrf():
    out = StateIndex.rrf({"knn": [(5, 0.9), (2, 0.8), (7, 0.1)], "bm25": [(2, 3.0), (9, 1.0)]})
    assert [r for r, _, _ in out] == [2, 5, 9, 7]
    assert out[0][2] == {"knn": 2, "bm25": 1}
    assert out[1][2] == {"knn": 1}
    assert out[2][2] == {"bm25": 2}
    assert out[0][1] == pytest.approx(1 / 62 + 1 / 61)


@pytest.mark.skipif(not STATES.exists(), reason="dataset/v1/states.jsonl missing")
def test_integration(tmp_path):
    db = tmp_path / "t.sqlite"
    subprocess.run([sys.executable, str(ROOT / "tools" / "db" / "seed.py"), "--db", str(db), "--hnsw"],
                   check=True, cwd=ROOT)
    with open(STATES) as f:
        lines = [l for l in f if l.strip()]
    rec = next(json.loads(l) for l in lines if json.loads(l)["split"] == "train")
    idx = StateIndex(db)
    hits = idx.search(rec["state"], rec["t"], rec["context_text"], k=5, exclude_game=rec["game_id"])
    assert len(hits) == 5
    assert all(h.game_id != rec["game_id"] and h.action_key is not None for h in hits)
    hidx = StateIndex(db, use_hnsw=True)
    assert len(hidx.knn(featurize(rec["state"], rec["t"]), k=5)) == 5
    import sqlite3
    n = sqlite3.connect(db).execute("SELECT count(*) FROM states").fetchone()[0]
    assert n == len(lines)
