"""Seed the SQLite database, state feature matrix and optional HNSW index from dataset/v1 (WP3)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from clashbot.config import DATASET_DIR, DATASET_VERSION, DB_PATH  # noqa: E402
from clashbot.db import connect, create_schema  # noqa: E402
from clashbot.features import FEATURE_VERSION, block_slices, dim, featurize  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def js(value) -> str:
    return json.dumps(value)


def js_if_needed(value):
    """JSON-encode lists/dicts, pass scalars and None through."""
    if value is None or isinstance(value, (str, int, float)):
        return value
    return json.dumps(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--hnsw", action="store_true")
    args = ap.parse_args()
    dataset: Path = args.dataset
    db_path: Path = args.db
    folder = db_path.parent
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 1. fresh db
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    create_schema(conn)

    # 2. games, provenance
    games = read_jsonl(dataset / "games.jsonl")
    conn.executemany(
        "INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(g["game_id"], g.get("split"), js(g.get("own_deck")), g.get("deck_status"), g.get("own_deck_key"),
          g.get("own_archetype"), js(g.get("opponent_deck")), g.get("opponent_archetype_guess"),
          g.get("result"), js(g.get("flags"))) for g in games])
    prov = read_jsonl(dataset / "provenance.jsonl")
    conn.executemany(
        "INSERT INTO provenance VALUES (?,?,?,?,?,?,?)",
        [(p["game_id"], p.get("video_id"), p.get("match_index"), p.get("creator"), p.get("video_title"),
          p.get("video_url"), js_if_needed(p.get("video_time"))) for p in prov])
    deck_key_of = {g["game_id"]: g.get("own_deck_key") for g in games}

    # 3. states
    records = read_jsonl(dataset / "states.jsonl")
    rows = []
    for i, r in enumerate(records):
        st = r["state"]
        act = r.get("action") or {}
        out = r.get("outcome") or {}
        rows.append((
            i, r["id"], r["game_id"], r.get("split"), r.get("kind"), r.get("t"), st.get("phase"),
            st.get("match_seconds"), (st.get("own") or {}).get("elixir"), (st.get("own") or {}).get("deck_status"),
            deck_key_of.get(r["game_id"]),
            act.get("type"), act.get("card"), act.get("lane"), act.get("zone"), act.get("key"),
            out.get("verdict"), out.get("elixir_trade"), 1 if r.get("train_eligible") else 0,
            js(r.get("flags")), r.get("state_text"), r.get("context_text"), js(r)))
    conn.executemany("INSERT INTO states VALUES (" + ",".join("?" * 23) + ")", rows)

    # 4. FTS over context text
    conn.executemany(
        "INSERT INTO states_fts(rowid, id, context_text) VALUES (?,?,?)",
        [(i + 1, r["id"], r.get("context_text")) for i, r in enumerate(records)])

    # 5. features
    M = np.stack([featurize(r["state"], r["t"]) for r in records]).astype(np.float32)
    np.save(folder / "state_features.npy", M)
    (folder / "state_features.meta.json").write_text(json.dumps({
        "feature_version": FEATURE_VERSION, "dim": dim(), "n": int(M.shape[0]),
        "block_slices": block_slices(), "built_at": built_at}, indent=2))

    # 6. HNSW
    if args.hnsw:
        import hnswlib
        index = hnswlib.Index(space="ip", dim=M.shape[1])
        index.init_index(max_elements=M.shape[0], ef_construction=200, M=16)
        index.add_items(M, np.arange(M.shape[0]))
        index.save_index(str(folder / "state_features.hnsw"))

    # 7. WP2 outputs, if built
    stats_path = dataset / "card_stats.jsonl"
    if stats_path.exists():
        conn.executemany(
            "INSERT INTO cards VALUES (?,?,?,?,?,?,?,?)",
            [(c["slug"], c.get("name"), c.get("card_type"), c.get("rarity"), c.get("elixir_cost"),
              js_if_needed(c.get("targets")), js(c.get("attributes")), js(c.get("stats_by_level")))
             for c in read_jsonl(stats_path)])
    else:
        print("skip card_stats.jsonl: not built")
    nodes_path, edges_path = dataset / "kg" / "nodes.jsonl", dataset / "kg" / "edges.jsonl"
    if nodes_path.exists() and edges_path.exists():
        conn.executemany(
            "INSERT INTO kg_nodes VALUES (?,?,?,?)",
            [(n["id"], n.get("type"), n.get("name"), js(n.get("props"))) for n in read_jsonl(nodes_path)])
        conn.executemany(
            "INSERT INTO kg_edges VALUES (?,?,?,?,?)",
            [(i, e["src"], e["rel"], e["dst"], js(e.get("props"))) for i, e in enumerate(read_jsonl(edges_path))])
    else:
        print("skip kg: not built")
    docs_path = dataset / "documents.jsonl"
    if docs_path.exists():
        docs = read_jsonl(docs_path)
        conn.executemany(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
            [(d["doc_id"], d.get("type"), d.get("title"), d.get("text"), js(d.get("cards")), d.get("deck_key"),
              d.get("archetype"), d.get("game_id"), d.get("split")) for d in docs])
        conn.executemany(
            "INSERT INTO documents_fts(rowid, doc_id, title, text) VALUES (?,?,?,?)",
            [(i + 1, d["doc_id"], d.get("title"), d.get("text")) for i, d in enumerate(docs)])
    else:
        print("skip documents.jsonl: not built")

    # 8. meta
    manifest = dataset / "MANIFEST.json"
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else "missing"
    conn.executemany("INSERT INTO meta VALUES (?,?)", [
        ("dataset_version", DATASET_VERSION), ("feature_version", str(FEATURE_VERSION)),
        ("built_at", built_at), ("n_states", str(len(records))), ("feature_dim", str(dim())),
        ("source_manifest_sha256", sha)])

    # 9. commit
    conn.commit()
    for table in ("meta", "games", "provenance", "states", "states_fts", "cards", "kg_nodes", "kg_edges",
                  "documents", "documents_fts"):
        n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table:14s} {n}")
    conn.close()


if __name__ == "__main__":
    main()
