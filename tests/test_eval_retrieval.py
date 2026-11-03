import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.db import connect, create_schema  # noqa: E402
from clashbot.text_embed import (HashEmbedder, TextEmbedder, build_state_text_matrix,  # noqa: E402
                                 text_knn)
from tools.eval.eval_retrieval import (LIST_NAMES, hit_at_k, main, ndcg_at_k,  # noqa: E402
                                       recall_overlap, vote_recall_at_k)

DB = ROOT / "db" / "clashbot.sqlite"
STATES = ROOT / "dataset" / "v1" / "states.jsonl"


def test_hash_embedder():
    emb = HashEmbedder()
    texts = ["elixir 5 hand: zap, knight", "elixir 5 hand: zap, knight", "tower in 2s", ""]
    E = emb.encode(texts)
    assert E.shape == (4, 384) and E.dtype == np.float32
    assert np.array_equal(E, HashEmbedder().encode(texts))
    assert np.array_equal(E[0], E[1])
    assert abs(float(np.linalg.norm(E[0])) - 1.0) < 1e-5
    assert abs(float(np.linalg.norm(E[2])) - 1.0) < 1e-5
    assert not E[3].any()
    assert emb.model_id == "hash-384"


def test_text_embedder_encode_fn_normalises_without_sentence_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)    # any import would raise ImportError
    emb = TextEmbedder(encode_fn=lambda texts: np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 2.0]])[:len(texts)])
    out = emb.encode(["a", "b", "c"])
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0], [0.0, 1.0]])
    assert emb.model_id == "BAAI/bge-small-en-v1.5"


def test_text_knn_mask_ties_and_k():
    E = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0.6, 0.8]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    allowed = np.array([True, False, True, True, True])
    out = text_knn(E, q, allowed, 10)
    assert [r for r, _ in out] == [0, 3, 4, 2]                  # row 1 masked; tie 0 vs 3 -> lower row first
    assert out[0][1] == pytest.approx(1.0) and out[2][1] == pytest.approx(0.6)
    assert [r for r, _ in text_knn(E, q, allowed, 2)] == [0, 3]
    assert text_knn(E, q, np.zeros(5, dtype=bool), 3) == []


def test_hit_at_k():
    keys = ["a|l|x", "hold", "b|m|y", "a|m|x"]
    assert hit_at_k(keys, "a|l|x", 1) == 1.0
    assert hit_at_k(keys, "b|m|y", 2) == 0.0
    assert hit_at_k(keys, "b|m|y", 3) == 1.0
    assert hit_at_k(keys, "hold", 4) == 0.0
    assert hit_at_k([], "a|l|x", 5) == 0.0


def test_vote_recall_at_k():
    # weights: a|l|x = 1/61 + 1/63, b|m|y = 1/62 + 1/65 + 1/66; "hold" (rank 4) is ignored
    keys = ["a|l|x", "b|m|y", "a|l|x", "hold", "b|m|y", "b|m|y"]
    wa, wb = 1 / 61 + 1 / 63, 1 / 62 + 1 / 66 + 1 / 65
    assert wb > wa                                              # b wins on weight despite a's rank-1 hit
    assert vote_recall_at_k(keys, "b|m|y", 1) == 1.0
    assert vote_recall_at_k(keys, "a|l|x", 1) == 0.0
    assert vote_recall_at_k(keys, "a|l|x", 3) == 1.0
    assert vote_recall_at_k(keys, "hold", 10) == 0.0
    assert vote_recall_at_k(["hold", "hold"], "a|l|x", 10) == 0.0
    # equal weights -> alphabetical key order
    assert vote_recall_at_k(["z|l|x", "a|l|x"], "a|l|x", 1) == 0.0
    assert vote_recall_at_k(["a|l|x", "a|l|x"], "a|l|x", 1) == 1.0


def test_ndcg_at_k():
    target = "a|l|x"
    assert ndcg_at_k([target, "a|m|x"], target, n_exact=1, n_card=1) == pytest.approx(1.0)
    assert ndcg_at_k(["b|l|x", "c|l|x"], target, n_exact=1, n_card=1) == 0.0
    assert ndcg_at_k([], target, n_exact=1, n_card=0) == 0.0
    assert ndcg_at_k([target], target, n_exact=0, n_card=0) == 0.0          # empty ideal list
    # mixed: rels [2, 0, 1]; ideal with 2 exact + 1 same-card = [2, 2, 1]
    dcg = 2 / math.log2(2) + 0 + 1 / math.log2(4)
    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    got = ndcg_at_k([target, "b|l|x", "a|r|y"], target, n_exact=2, n_card=1)
    assert got == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["hold"], "hold", n_exact=1, n_card=0) == 0.0


def test_recall_overlap():
    assert recall_overlap([1, 2, 3], [3, 2, 1]) == 1.0
    assert recall_overlap([1, 2, 9, 8], [1, 2, 3, 4]) == 0.5
    assert recall_overlap([], [1, 2], 10) == 0.0
    assert recall_overlap([5], [], 10) == 0.0
    assert recall_overlap([1, 2, 3, 4], [1, 2, 3, 4], 2) == 1.0


def test_build_state_text_matrix_caches(tmp_path):
    db = tmp_path / "t.sqlite"
    conn = connect(db)
    create_schema(conn)
    for i, text in enumerate(["elixir 5 zap", "elixir 9 giant", "tower in 2s knight"]):
        conn.execute("INSERT INTO states(vec_row, id, game_id, split, context_text) VALUES (?,?,?,?,?)",
                     (i, f"s{i}", "g", "train", text))
    conn.commit()
    conn.close()

    cache = tmp_path / "cache"
    emb = HashEmbedder()
    E = build_state_text_matrix(db, emb, cache)
    assert E.shape == (3, 384)
    assert (cache / "state_text_emb.npy").exists()
    meta = json.loads((cache / "state_text_emb.meta.json").read_text())
    assert meta["model"] == "hash-384" and meta["n"] == 3 and meta["dim"] == 384 and len(meta["texts_sha1"]) == 40
    assert np.array_equal(E, np.load(cache / "state_text_emb.npy"))

    def boom(texts, batch_size=64):
        raise AssertionError("cache not reused")

    monkey = HashEmbedder()
    monkey.encode = boom
    assert np.array_equal(build_state_text_matrix(db, monkey, cache), E)
    with pytest.raises(AssertionError):                           # force bypasses the cache
        build_state_text_matrix(db, monkey, cache, force=True)

    # changed text invalidates the cache
    conn = connect(db)
    conn.execute("UPDATE states SET context_text = 'changed' WHERE vec_row = 1")
    conn.commit()
    conn.close()
    with pytest.raises(AssertionError):
        build_state_text_matrix(db, monkey, cache)


@pytest.mark.skipif(not DB.exists() or not STATES.exists(), reason="db/clashbot.sqlite or dataset missing")
def test_main_real_data_hash(tmp_path):
    out = tmp_path / "r.json"
    main(["--embedder", "hash", "--limit", "20", "--skip-hnsw", "--skip-filter",
          "--cache-dir", str(tmp_path), "--out", str(out)])
    res = json.loads(out.read_text())
    assert out.with_suffix(".md").exists()
    assert res["embedder"] == "hash-384" and res["depth"] == 50
    for split in ("val", "test"):
        assert res["n_queries"][split] == 20
        assert set(res["channels"][split]) == set(LIST_NAMES) and len(LIST_NAMES) == 7
        for name, metrics in res["channels"][split].items():
            for metric, v in metrics.items():
                if metric.startswith("ms_"):
                    assert v is not None and v >= 0
                else:
                    assert 0.0 <= v <= 1.0, (split, name, metric, v)
    assert res["hnsw"] == [] and res["filtering"] == {}
