"""Hybrid state retrieval: structured-vector kNN + BM25 over context text, fused with RRF (D-007)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clashbot.config import DB_PATH
from clashbot.db import connect, fts_query
from clashbot.features import FEATURE_VERSION, dim, featurize


@dataclass
class Hit:
    row: int
    id: str
    game_id: str
    kind: str
    action_key: str | None
    verdict: str | None
    elixir_trade: float | None
    score: float
    ranks: dict          # e.g. {"knn": 1, "bm25": 4}; a missing channel is absent


class StateIndex:
    def __init__(self, db_path: Path = DB_PATH, use_hnsw: bool = False):
        db_path = Path(db_path)
        folder = db_path.parent
        conn = connect(db_path, readonly=True)
        try:
            rows = conn.execute(
                "SELECT vec_row, id, game_id, split, kind, own_deck_key, phase, action_key, verdict, "
                "elixir_trade, train_eligible FROM states ORDER BY vec_row").fetchall()
        finally:
            conn.close()
        self.db_path = db_path
        self.n = len(rows)
        self.id = np.array([r["id"] for r in rows], dtype=object)
        self.game_id = np.array([r["game_id"] for r in rows], dtype=object)
        self.split = np.array([r["split"] for r in rows], dtype=object)
        self.kind = np.array([r["kind"] for r in rows], dtype=object)
        self.own_deck_key = np.array([r["own_deck_key"] for r in rows], dtype=object)
        self.phase = np.array([r["phase"] for r in rows], dtype=object)
        self.action_key = np.array([r["action_key"] for r in rows], dtype=object)
        self.verdict = np.array([r["verdict"] for r in rows], dtype=object)
        self.elixir_trade = np.array([r["elixir_trade"] for r in rows], dtype=object)
        self.train_eligible = np.array([r["train_eligible"] for r in rows], dtype=object)
        self._has_action = np.array([a is not None for a in self.action_key], dtype=bool)

        meta = json.loads((folder / "state_features.meta.json").read_text())
        if meta.get("feature_version") != FEATURE_VERSION:
            raise RuntimeError(
                f"feature_version mismatch: db {meta.get('feature_version')} vs code {FEATURE_VERSION}")
        if meta.get("dim") != dim():
            raise RuntimeError(f"feature dim mismatch: db {meta.get('dim')} vs code {dim()}")
        self.M = np.load(folder / "state_features.npy", mmap_mode="r")
        if self.M.shape != (self.n, dim()):
            raise RuntimeError(f"state_features.npy shape {self.M.shape} != ({self.n}, {dim()})")

        self.use_hnsw = use_hnsw
        self._hnsw = None
        if use_hnsw:
            import hnswlib
            path = folder / "state_features.hnsw"
            if not path.exists():
                raise FileNotFoundError(str(path))
            self._hnsw = hnswlib.Index(space="ip", dim=dim())
            self._hnsw.load_index(str(path), max_elements=self.n)
            self._hnsw.set_ef(64)

    def mask(self, *, splits=None, exclude_game=None, deck_key=None, phase=None,
             require_action=True) -> np.ndarray:
        m = np.ones(self.n, dtype=bool)
        if splits is not None:
            m &= np.isin(self.split, list(splits))
        if exclude_game is not None:
            m &= self.game_id != exclude_game
        if deck_key is not None:
            m &= self.own_deck_key == deck_key
        if phase is not None:
            m &= self.phase == phase
        if require_action:
            m &= self._has_action
        return m

    def knn(self, vec: np.ndarray, k: int = 20, **filters) -> list[tuple[int, float]]:
        allowed = self.mask(**filters)
        vec = np.asarray(vec, dtype=np.float32)
        if self.use_hnsw:
            kk = min(k * 10, self.n)
            if kk <= 0:
                return []
            labels, dists = self._hnsw.knn_query(vec.reshape(1, -1), k=kk)
            out: list[tuple[int, float]] = []
            for lab, d in zip(labels[0], dists[0]):
                if allowed[int(lab)]:
                    out.append((int(lab), float(1.0 - d)))
                    if len(out) >= k:
                        break
            return out
        scores = np.asarray(self.M @ vec, dtype=np.float64)
        scores[~allowed] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores))
        order = np.lexsort((valid, -scores[valid]))[:k]
        return [(int(valid[i]), float(scores[valid[i]])) for i in order]

    def bm25(self, text: str, k: int = 20, **filters) -> list[tuple[int, float]]:
        q = fts_query(text)
        if q is None:
            return []
        allowed = self.mask(**filters)
        conn = connect(self.db_path, readonly=True)
        try:
            rows = conn.execute(
                "SELECT rowid, bm25(states_fts) FROM states_fts WHERE states_fts MATCH ? "
                "ORDER BY bm25(states_fts) LIMIT ?", (q, k * 10)).fetchall()
        finally:
            conn.close()
        out: list[tuple[int, float]] = []
        for rowid, score in rows:
            row = rowid - 1
            if allowed[row]:
                out.append((row, -float(score)))
                if len(out) >= k:
                    break
        return out

    @staticmethod
    def rrf(lists: dict[str, list[tuple[int, float]]], k_rrf: int = 60) -> list[tuple[int, float, dict]]:
        scores: dict[int, float] = {}
        ranks: dict[int, dict] = {}
        for name, items in lists.items():
            for r, (row, _score) in enumerate(items, start=1):
                scores[row] = scores.get(row, 0.0) + 1.0 / (k_rrf + r)
                ranks.setdefault(row, {})[name] = r
        ordered = sorted(scores, key=lambda row: (-scores[row], row))
        return [(row, scores[row], ranks[row]) for row in ordered]

    def search(self, state: dict, t: float, context_text: str, k: int = 10, depth: int = 50,
               **filters) -> list[Hit]:
        fused = self.rrf({
            "knn": self.knn(featurize(state, t), depth, **filters),
            "bm25": self.bm25(context_text, depth, **filters),
        })
        hits = []
        for row, score, ranks in fused[:k]:
            et = self.elixir_trade[row]
            hits.append(Hit(row=row, id=self.id[row], game_id=self.game_id[row], kind=self.kind[row],
                            action_key=self.action_key[row], verdict=self.verdict[row],
                            elixir_trade=None if et is None else float(et), score=score, ranks=ranks))
        return hits
