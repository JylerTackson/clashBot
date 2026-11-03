"""Dense text channel for state retrieval: embed context_text and cosine-kNN over it (D-033)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from clashbot.config import DB_PATH
from clashbot.db import connect

DEFAULT_TEXT_MODEL = "BAAI/bge-small-en-v1.5"      # 384-d (D-033)
CACHE_NAME = "state_text_emb"                       # <name>.npy + <name>.meta.json


def _normalise_rows(arr: np.ndarray) -> np.ndarray:
    """L2-normalise rows to float32; all-zero rows stay zero."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


class TextEmbedder:
    """Sentence-transformers wrapper; the model is loaded lazily on the first encode()."""

    def __init__(self, model_id: str = DEFAULT_TEXT_MODEL, device: str | None = None, encode_fn=None):
        self.model_id = model_id
        self.device = device
        self.encode_fn = encode_fn
        self._model = None
        self._dim = 0

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = self.device
            if device is None:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_id, device=device)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        if self.encode_fn is not None:
            out = _normalise_rows(self.encode_fn(list(texts)))
        else:
            out = self._load().encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        self._dim = out.shape[1]
        return out


class HashEmbedder:
    """Deterministic, model-free bag-of-hashed-tokens embedding (tests, --embedder hash)."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.model_id = f"hash-{dim}"

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z][a-z0-9\-]*", (text or "").lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _normalise_rows(out)


def build_state_text_matrix(db_path: Path = DB_PATH, embedder=None, cache_dir: Path | None = None,
                            force: bool = False) -> np.ndarray:
    """(n_states, d) embedding of states.context_text in vec_row order, cached next to the db."""
    db_path = Path(db_path)
    embedder = embedder if embedder is not None else TextEmbedder()
    cache_dir = Path(cache_dir) if cache_dir is not None else db_path.parent
    conn = connect(db_path, readonly=True)
    try:
        texts = [r[0] or "" for r in conn.execute("SELECT context_text FROM states ORDER BY vec_row")]
    finally:
        conn.close()
    model = getattr(embedder, "model_id", type(embedder).__name__)
    sha = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()
    npy_path, meta_path = cache_dir / f"{CACHE_NAME}.npy", cache_dir / f"{CACHE_NAME}.meta.json"

    if not force and npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("model") == model and meta.get("n") == len(texts) and meta.get("texts_sha1") == sha:
            E = np.load(npy_path)
            if E.shape == (meta["n"], meta.get("dim")):
                return E

    E = embedder.encode(texts, batch_size=64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, E)
    meta_path.write_text(json.dumps({"model": model, "n": len(texts), "dim": int(E.shape[1]),
                                     "texts_sha1": sha}, indent=2))
    return E


def text_knn(E: np.ndarray, q: np.ndarray, allowed: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Cosine kNN over the allowed rows, ordered exactly like StateIndex.knn (score desc, row asc)."""
    scores = np.asarray(E @ np.asarray(q, dtype=np.float32), dtype=np.float64)
    scores[~allowed] = -np.inf
    valid = np.flatnonzero(np.isfinite(scores))
    order = np.lexsort((valid, -scores[valid]))[:k]
    return [(int(valid[i]), float(scores[valid[i]])) for i in order]
