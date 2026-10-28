"""Slow-loop planner: rule-based pre-game plan from the documents table; LLM planner is a stub (D-004)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from clashbot.config import DB_PATH
from clashbot.db import connect

PLAN_MAX_CHARS = 600

PREGAME_TEMPLATE = (
    "You are the strategist for a Clash Royale agent. You cannot act; you can only read via tools: {tools}.\n"
    "Our deck: {deck}\n"
    "Professional notes:\n"
    "{docs}\n"
    "Write a game plan in at most 5 short lines: win condition, what to save each spell for, preferred lane, "
    "how to defend the opponent's likely win condition, and when to stop spending."
)


@dataclass
class Plan:
    text: str
    source: str           # "rules" | "llm"


class RuleBasedPlanner:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)

    def _doc_text(self, conn, doc_id: str) -> str | None:
        row = conn.execute("SELECT text FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return row["text"] if row else None

    def pregame(self, own_deck: list[str], deck_key: str | None) -> Plan:
        text: str | None = None
        conn = connect(self.db_path, readonly=True)
        try:
            if deck_key:
                text = self._doc_text(conn, f"deck_why:{deck_key}")
                if text is None:
                    node = conn.execute("SELECT props FROM kg_nodes WHERE id = ?",
                                        (f"deck:{deck_key}",)).fetchone()
                    if node:
                        archetype = (json.loads(node["props"]) if node["props"] else {}).get("archetype_primary")
                        if archetype:
                            text = self._doc_text(conn, f"archetype_guide:{archetype}")
        finally:
            conn.close()
        if text is None:
            text = "Deck: " + ", ".join(own_deck)
        return Plan(text=text[:PLAN_MAX_CHARS], source="rules")

    def update(self, plan: Plan, state: dict, t: float) -> Plan:
        return plan


def build_pregame_prompt(own_deck: list[str], deck_docs: list[str], tool_names: list[str]) -> str:
    return PREGAME_TEMPLATE.format(deck=", ".join(own_deck), docs="\n".join(deck_docs),
                                   tools=", ".join(tool_names))


class LLMPlanner:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def pregame(self, own_deck: list[str], deck_key: str | None = None) -> Plan:
        raise NotImplementedError("Qwen planner runs on the Mac via MLX or llama.cpp; "
                                  "see docs/design/pipeline_review.md §4")
