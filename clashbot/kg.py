"""In-memory read-only view of the knowledge graph (kg_edges) for the runtime (D-003, D-004)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from clashbot.config import DB_PATH
from clashbot.db import connect


class KnowledgeGraph:
    def __init__(self, db_path: Path = DB_PATH):
        self.out: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
        self.inc: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
        conn = connect(Path(db_path), readonly=True)
        try:
            rows = conn.execute("SELECT src, rel, dst, props FROM kg_edges ORDER BY edge_id").fetchall()
        finally:
            conn.close()
        for r in rows:
            props = json.loads(r["props"]) if r["props"] else {}
            self.out[(r["src"], r["rel"])].append((r["dst"], props))
            self.inc[(r["dst"], r["rel"])].append((r["src"], props))

    def counter_scores(self, threat_card: str) -> dict[str, float]:
        score: dict[str, float] = defaultdict(float)
        for src, _props in self.inc.get(("card:" + threat_card, "COUNTERS"), []):
            score[src.removeprefix("card:")] += 1.0
        for dst, props in self.out.get(("card:" + threat_card, "ANSWERED_BY"), []):
            score[dst.removeprefix("card:")] += props["confidence"] * min(1.0, props["support"] / 10)
        return dict(score)

    def neighbors(self, node_id: str, rel: str | None = None, direction: str = "out",
                  limit: int = 50) -> list[dict]:
        table = self.out if direction == "out" else self.inc
        if rel is None:
            keys = sorted(k for k in table if k[0] == node_id)
        else:
            keys = [(node_id, rel)]
        items: list[dict] = []
        for key in keys:
            for other, props in table.get(key, []):
                items.append({"node": other, "rel": key[1], "props": props})
                if len(items) >= limit:
                    return items
        return items
