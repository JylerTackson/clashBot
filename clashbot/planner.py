"""Slow-loop planner: rule-based pre-game plan from the documents table, and a Qwen ReAct planner over the six
read-only tools (D-003, D-007, D-027). The planner reads and writes text only; it never acts on the game."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from clashbot.config import DB_PATH
from clashbot.db import connect
from clashbot.state_text import render_context
from clashbot.tools import TOOLS, ToolContext, call_tool

PLAN_MAX_CHARS = 600
DEFAULT_QWEN_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_STEPS = 6                # ReAct steps for the pre-game plan
UPDATE_STEPS = 2             # ReAct steps for an in-game revision
UPDATE_PERIOD_S = 10.0       # minimum game seconds between plan revisions
MAX_CALLS_PER_STEP = 3
OBS_MAX_CHARS = 2000

PREGAME_TEMPLATE = (
    "You are the strategist for a Clash Royale agent. You cannot act; you can only read via tools: {tools}.\n"
    "Our deck: {deck}\n"
    "Professional notes:\n"
    "{docs}\n"
    "Write a game plan in at most 5 short lines: win condition, what to save each spell for, preferred lane, "
    "how to defend the opponent's likely win condition, and when to stop spending."
)

SYSTEM_PROMPT = ("You are the strategist for a Clash Royale agent. You cannot act or play cards: you can only read "
                 "through the tools you are given. Call a tool when you need a fact (card stats, counters, professional "
                 "notes, similar situations). When you know enough, reply with the plan and no tool call.")
FINAL_NUDGE = "Write the final plan now, in at most 5 short lines, without calling any tool."
UPDATE_TEMPLATE = ("Current plan:\n{plan}\n\nThe game now (t={t}s):\n{context}\n\nEnemy cards not seen when the plan "
                   "was made: {new_cards}.\nKeep or revise the plan in at most 5 short lines.")

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Plan:
    text: str
    source: str           # "rules" | "llm" | "rules_fallback"
    trace: list = field(default_factory=list)   # tool calls made while planning


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


class MLXChat:
    """Chat wrapper over mlx-lm; the model loads on the first `chat` call."""

    def __init__(self, model_id: str = DEFAULT_QWEN_ID, max_tokens: int = 384, temperature: float = 0.0):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.last_stats: dict = {}
        self._model = None
        self._tok = None

    def _load(self) -> None:
        from mlx_lm import load
        self._model, self._tok = load(self.model_id)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        if self._model is None:
            self._load()
        kwargs = {"tools": tools} if tools is not None else {}
        prompt = self._tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
        t0 = time.perf_counter()
        text = generate(self._model, self._tok, prompt=prompt, max_tokens=self.max_tokens,
                        sampler=make_sampler(temp=self.temperature), verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        self.last_stats = {"prompt_tokens": len(self._tok.encode(prompt)),
                           "gen_tokens": len(self._tok.encode(text)), "gen_ms": round(ms, 1)}
        return text


def openai_tools(tools: list[dict] = TOOLS) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["parameters"]}} for t in tools]


def parse_tool_calls(text: str) -> list[dict]:
    """Every <tool_call>{json}</tool_call> block (the Qwen3 format) as {"name", "arguments"[, "error"]}."""
    calls: list[dict] = []
    for block in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(block.strip())
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            calls.append({"name": obj.get("name"), "arguments": obj.get("arguments") or {}})
        else:
            calls.append({"name": None, "arguments": {}, "error": "bad tool_call json"})
    return calls


def strip_tool_calls(text: str) -> str:
    return _THINK_RE.sub("", _TOOL_CALL_RE.sub("", text)).strip()


def should_update(last_update_t: float | None, t: float, opp_deck_known: list[str], seen: set[str],
                  period_s: float = UPDATE_PERIOD_S) -> bool:
    """Revise the plan on a timer or as soon as a new enemy card shows up. The caller owns `seen`."""
    if last_update_t is None or t - last_update_t >= period_s:
        return True
    return any(c not in seen for c in opp_deck_known)


class LLMPlanner:
    def __init__(self, chat=None, model_path: str | None = None, max_steps: int = MAX_STEPS,
                 db_path: Path = DB_PATH):
        self.model_path = model_path
        self.chat = chat if chat is not None else MLXChat(model_path or DEFAULT_QWEN_ID)   # loads lazily
        self.max_steps = max_steps
        self.db_path = Path(db_path)
        self._seen: set[str] = set()      # opponent cards already shown to the model

    def _notes(self, own_deck: list[str], deck_key: str | None) -> str:
        """Rule-based pro context for the prompt (D-007 priming); falls back to the bare deck without a DB."""
        try:
            return RuleBasedPlanner(self.db_path).pregame(own_deck, deck_key).text
        except (sqlite3.Error, FileNotFoundError):
            return "Deck: " + ", ".join(own_deck)

    def run(self, system: str, user: str, ctx: ToolContext, max_steps: int) -> tuple[str, list[dict]]:
        """ReAct loop over the read-only tools; returns (final text, tool-call trace)."""
        messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        trace: list[dict] = []
        for step in range(max_steps):
            text = self.chat.chat(messages, openai_tools())
            calls = parse_tool_calls(text)
            if not calls:
                return strip_tool_calls(text), trace
            messages.append({"role": "assistant", "content": text})
            for c in calls[:MAX_CALLS_PER_STEP]:
                try:
                    result = call_tool(c["name"], c["arguments"], ctx) if c["name"] else {"error": c.get("error")}
                except (ValueError, KeyError, TypeError, sqlite3.Error) as e:
                    result = {"error": str(e)}
                obs = json.dumps(result, default=str, ensure_ascii=False)[:OBS_MAX_CHARS]
                messages.append({"role": "tool", "name": c["name"] or "invalid", "content": obs})
                trace.append({"step": step, "tool": c["name"], "arguments": c["arguments"],
                              "ok": not (isinstance(result, dict) and "error" in result)})
        messages.append({"role": "user", "content": FINAL_NUDGE})
        return strip_tool_calls(self.chat.chat(messages, None)), trace

    def pregame(self, own_deck: list[str], deck_key: str | None = None, ctx: ToolContext | None = None) -> Plan:
        ctx = ctx or ToolContext(db_path=self.db_path)
        self._seen = set()                # new game: forget the enemy cards of the previous one
        notes = self._notes(own_deck, deck_key)
        user = build_pregame_prompt(own_deck, [notes], [t["name"] for t in TOOLS])
        text, trace = self.run(SYSTEM_PROMPT, user, ctx, self.max_steps)
        text = text[:PLAN_MAX_CHARS].strip()
        plan = Plan(text, "llm", trace) if text else Plan(notes[:PLAN_MAX_CHARS], "rules_fallback", trace)
        ctx.plan_text = plan.text
        return plan

    def update(self, plan: Plan, state: dict, t: float, ctx: ToolContext | None = None) -> Plan:
        context = ctx.context_text if ctx and ctx.context_text else render_context(state, t)
        known = state["opponent"].get("deck_known") or []
        new_cards = [c for c in known if c not in self._seen]
        self._seen.update(known)
        user = UPDATE_TEMPLATE.format(plan=plan.text, t=round(t, 1), context=context,
                                      new_cards=", ".join(new_cards) if new_cards else "none")
        if ctx is None:
            ctx = ToolContext(state=state, t=t, context_text=context, plan_text=plan.text, db_path=self.db_path)
        elif ctx.plan_text is None:
            ctx.plan_text = plan.text
        text, trace = self.run(SYSTEM_PROMPT, user, ctx, UPDATE_STEPS)
        text = text[:PLAN_MAX_CHARS].strip()
        if not text:
            return plan
        ctx.plan_text = text
        return Plan(text, "llm", trace)
