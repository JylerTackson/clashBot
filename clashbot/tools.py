"""Read-only tools for the slow-loop LLM (D-003: the agent reads, code executes)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clashbot.config import DB_PATH
from clashbot.db import connect, fts_query
from clashbot.kg import KnowledgeGraph
from clashbot.retrieval import StateIndex

RELATIONS = ["COUNTERS", "ANSWERED_BY", "SYNERGY", "SUBSTITUTE", "IN_DECK", "SPAWNED_BY",
             "EVOLUTION_OF", "HERO_OF"]


@dataclass
class ToolContext:
    state: dict | None = None
    t: float | None = None
    context_text: str | None = None
    plan_text: str | None = None
    index: StateIndex | None = None
    kg: KnowledgeGraph | None = None
    db_path: Path = DB_PATH


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    out = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        out["required"] = required
    return out


TOOLS: list[dict] = [
    {"name": "get_game_state",
     "description": "Current game time, the text context and the structured state of the live game.",
     "parameters": _schema({})},
    {"name": "search_similar_states",
     "description": "Find similar situations from professional games; returns what the pro did and how it went.",
     "parameters": _schema({"k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}})},
    {"name": "kg_query",
     "description": "Neighbours of a knowledge-graph node (e.g. card:hog-rider) along one relation or all.",
     "parameters": _schema({
         "node": {"type": "string"},
         "relation": {"type": "string", "enum": RELATIONS},
         "direction": {"type": "string", "enum": ["out", "in"], "default": "out"}}, ["node"])},
    {"name": "card_info",
     "description": "Stats and attributes of a card by slug.",
     "parameters": _schema({"slug": {"type": "string"}}, ["slug"])},
    {"name": "search_documents",
     "description": "Full-text search over professional notes, deck explanations and archetype guides.",
     "parameters": _schema({
         "query": {"type": "string"},
         "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
         "type": {"type": "string"}}, ["query"])},
    {"name": "get_current_plan",
     "description": "The current game plan text.",
     "parameters": _schema({})},
]

_SPECS = {t["name"]: t["parameters"] for t in TOOLS}


def _validate(name: str, args: dict) -> dict:
    if name not in _SPECS:
        raise ValueError(f"unknown tool: {name!r}")
    if not isinstance(args, dict):
        raise ValueError("args must be a dict")
    spec = _SPECS[name]
    props = spec["properties"]
    unknown = set(args) - set(props)
    if unknown:
        raise ValueError(f"{name}: unknown argument(s) {sorted(unknown)}")
    for req in spec.get("required", []):
        if req not in args:
            raise ValueError(f"{name}: missing required argument {req!r}")
    out: dict = {}
    for key, p in props.items():
        if key not in args:
            if "default" in p:
                out[key] = p["default"]
            continue
        val = args[key]
        if p["type"] == "integer":
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{name}: {key} must be an integer")
            if val < p["minimum"] or val > p["maximum"]:
                raise ValueError(f"{name}: {key} must be in {p['minimum']}..{p['maximum']}")
        elif p["type"] == "string":
            if not isinstance(val, str):
                raise ValueError(f"{name}: {key} must be a string")
            if "enum" in p and val not in p["enum"]:
                raise ValueError(f"{name}: {key} must be one of {p['enum']}")
        out[key] = val
    return out


def _need(ctx_value, what: str):
    if ctx_value is None:
        raise ValueError(f"{what} not available in tool context")
    return ctx_value


def _get_game_state(a: dict, ctx: ToolContext) -> dict:
    return {"t": ctx.t, "context_text": ctx.context_text, "state": ctx.state}


def _search_similar_states(a: dict, ctx: ToolContext) -> list[dict]:
    index = _need(ctx.index, "index")
    hits = index.search(_need(ctx.state, "state"), _need(ctx.t, "t"), ctx.context_text or "", k=a["k"])
    return [{"id": h.id, "action_key": h.action_key, "verdict": h.verdict, "score": h.score} for h in hits]


def _kg_query(a: dict, ctx: ToolContext) -> list[dict]:
    kg = _need(ctx.kg, "kg")
    return kg.neighbors(a["node"], a.get("relation"), a["direction"], limit=25)


def _card_info(a: dict, ctx: ToolContext) -> dict:
    conn = connect(Path(ctx.db_path), readonly=True)
    try:
        row = conn.execute("SELECT * FROM cards WHERE slug = ?", (a["slug"],)).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"error": "unknown card"}
    d = dict(row)
    for col in ("attributes", "stats_by_level"):
        if d.get(col):
            try:
                d[col] = json.loads(d[col])
            except (TypeError, ValueError):
                pass
    return d


def _search_documents(a: dict, ctx: ToolContext) -> list[dict]:
    q = fts_query(a["query"])
    if q is None:
        return []
    sql = ("SELECT d.doc_id, d.type, d.title, d.text FROM documents_fts "
           "JOIN documents d ON d.doc_id = documents_fts.doc_id WHERE documents_fts MATCH ?")
    params: list = [q]
    if a.get("type"):
        sql += " AND d.type = ?"
        params.append(a["type"])
    sql += " ORDER BY bm25(documents_fts) LIMIT ?"
    params.append(a["k"])
    conn = connect(Path(ctx.db_path), readonly=True)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [{"doc_id": r["doc_id"], "type": r["type"], "title": r["title"], "text": (r["text"] or "")[:800]}
            for r in rows]


def _get_current_plan(a: dict, ctx: ToolContext) -> dict:
    return {"plan": ctx.plan_text}


_IMPL = {"get_game_state": _get_game_state, "search_similar_states": _search_similar_states,
         "kg_query": _kg_query, "card_info": _card_info, "search_documents": _search_documents,
         "get_current_plan": _get_current_plan}


def call_tool(name: str, args: dict, ctx: ToolContext) -> Any:
    validated = _validate(name, args)      # raises ValueError for unknown tool / bad arguments
    return _IMPL[name](validated, ctx)
