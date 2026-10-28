"""SQLite access helpers (D-009)."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from clashbot.config import DB_PATH, RESOURCES

SCHEMA_PATH = RESOURCES / "schema.sql"

_STOP = frozenset({"own", "opp", "elixir", "hand", "next", "deck", "field", "threats", "recent",
                   "towers", "enemy", "none", "ago", "eta", "unknown"})


def connect(path: Path = DB_PATH, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())


def fts_query(text: str, max_terms: int = 64) -> str | None:
    tokens: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"[a-z][a-z0-9]*", text.lower()):
        if len(tok) < 3 or tok in _STOP or tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
    tokens = tokens[:max_terms]
    if not tokens:
        return None
    return " OR ".join(f'"{tok}"' for tok in tokens)
