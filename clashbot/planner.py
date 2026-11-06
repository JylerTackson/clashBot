"""Slow-loop planner: rule-based pre-game plan from the documents table, a Qwen ReAct planner over the six read-only
tools (D-003, D-007, D-027) and the Qwen3-14B coach that rewrites a short plan every few game seconds (D-046). The
planner reads and writes text only; it never acts on the game."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from clashbot.config import DB_PATH, HOLD_KEY
from clashbot.db import connect
from clashbot.state_text import render_context
from clashbot.tools import TOOLS, ToolContext, call_tool

PLAN_MAX_CHARS = 600
DEFAULT_QWEN_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
COACH_MODEL_ID = "mlx-community/Qwen3-14B-4bit"
MAX_STEPS = 6                # ReAct steps for the pre-game plan
UPDATE_STEPS = 2             # ReAct steps for an in-game revision
UPDATE_PERIOD_S = 10.0       # minimum game seconds between plan revisions
MAX_CALLS_PER_STEP = 3
OBS_MAX_CHARS = 2000
COACH_PERIOD_S = 10.0        # game seconds between coach plans (CoachWorker; also a card-seen trigger)
COACH_MAX_LINES = 3
COACH_LINE_CHARS = 100
COACH_MAX_CHARS = COACH_MAX_LINES * COACH_LINE_CHARS + COACH_MAX_LINES - 1
LOOKUP_N = 3                 # pro moves quoted to the coach
NO_THINK = "/no_think"       # Qwen3 soft switch, for a tokenizer whose template has no `enable_thinking`

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
                 "notes, similar situations). When you know enough, reply with the plan and no tool call. "
                 "Write plain text without markdown: at most 5 lines, each under 100 characters. Mention only cards "
                 "from our deck or opponent cards seen so far.")
FINAL_NUDGE = "Write the final plan now: plain text, at most 5 lines, each under 100 characters, no tool calls."
UPDATE_TEMPLATE = ("Current plan:\n{plan}\n\nThe game now (t={t}s):\n{context}\n\nEnemy cards not seen when the plan "
                   "was made: {new_cards}.\nKeep or revise the plan in at most 5 short lines.")

COACH_SYSTEM_PROMPT = ("You are the coach of a Clash Royale agent. You cannot act or play cards: another model picks "
                       "every move and reads your plan before it does. Write plain text without markdown. Mention "
                       "only cards from our deck or opponent cards seen so far.")
COACH_UPDATE_TEMPLATE = (
    "Our deck: {deck}\n"
    "Opponent cards seen: {opp}\n\n"
    "The game now (t={t}s):\n{context}\n\n"
    "Tactics engine:\n{tactics}\n\n"
    "Similar situations:\n{lookup}\n\n"
    "Current plan:\n{plan}\n\n"
    "Write the next plan in at most 3 plain lines, one point per line, each at most 100 characters: what to save "
    "for which threat, where to attack next, and when to spend. Mention only cards from our deck or the opponent "
    "cards seen. No markdown, no tool calls.")

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL)      # a think block cut off by max_tokens
_HEADER_RE = re.compile(r"^#+\s*")
_BULLET_RE = re.compile(r"^[-*]\s+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


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


def with_no_think(messages: list[dict]) -> list[dict]:
    """Copy of `messages` whose last user turn ends with the Qwen3 `/no_think` switch (the input is not changed)."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            if NO_THINK not in m["content"]:
                m["content"] = m["content"].rstrip() + "\n" + NO_THINK
            break
    return out


def has_thinking_switch(tokenizer) -> bool:
    """True when the tokenizer's chat template reads `enable_thinking` (Qwen3 hybrid models)."""
    template = getattr(tokenizer, "chat_template", None)
    if template is not None and not isinstance(template, str):
        template = json.dumps(template, default=str)
    return isinstance(template, str) and "enable_thinking" in template


class MLXChat:
    """Chat wrapper over mlx-lm; the model loads on the first `chat` call. `enable_thinking=False` turns Qwen3's
    thinking off: the template switch when the tokenizer has one, else `/no_think` on the last user turn."""

    def __init__(self, model_id: str = DEFAULT_QWEN_ID, max_tokens: int = 200, temperature: float = 0.0,
                 enable_thinking: bool | None = None):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.last_stats: dict = {}
        self._model = None
        self._tok = None

    def _load(self) -> None:
        from mlx_lm import load
        self._model, self._tok = load(self.model_id)

    def load(self) -> None:
        """Load the model now (idempotent); `chat` does it on the first call otherwise."""
        if self._model is None:
            self._load()

    def render_prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        kwargs = {"tools": tools} if tools is not None else {}
        if self.enable_thinking is False:
            if has_thinking_switch(self._tok):
                kwargs["enable_thinking"] = False
            else:
                messages = with_no_think(messages)
        return self._tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.load()
        prompt = self.render_prompt(messages, tools)
        t0 = time.perf_counter()
        text, last = "", None
        for last in stream_generate(self._model, self._tok, prompt=prompt, max_tokens=self.max_tokens,
                                    sampler=make_sampler(temp=self.temperature)):
            text += last.text
        ms = (time.perf_counter() - t0) * 1000
        self.last_stats = {"prompt_tokens": getattr(last, "prompt_tokens", 0),
                           "gen_tokens": getattr(last, "generation_tokens", 0), "gen_ms": round(ms, 1),
                           "prompt_tps": round(getattr(last, "prompt_tps", 0.0), 1),
                           "gen_tps": round(getattr(last, "generation_tps", 0.0), 1),
                           "peak_gb": round(getattr(last, "peak_memory", 0.0), 2)}
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


def strip_think(text: str) -> str:
    """Drop `<think>...</think>` blocks, and a trailing think block that never closed."""
    return _THINK_OPEN_RE.sub("", _THINK_RE.sub("", text)).strip()


def strip_tool_calls(text: str) -> str:
    return strip_think(_TOOL_CALL_RE.sub("", text))


def clip_plan(text: str, max_chars: int = PLAN_MAX_CHARS, max_lines: int | None = None) -> str:
    """Plain-text plan of whole lines within `max_chars` (and `max_lines` when given): markdown emphasis, headers and
    bullets removed, blank lines dropped; an over-long first line is cut at a space, never mid-word."""
    lines = []
    for raw in text.replace("**", "").replace("__", "").splitlines():
        line = _BULLET_RE.sub("", _HEADER_RE.sub("", raw.strip())).strip()
        if line:
            lines.append(line)
    if max_lines is not None:
        lines = lines[:max_lines]
    kept: list[str] = []
    for line in lines:
        if len("\n".join(kept + [line])) > max_chars:
            break
        kept.append(line)
    if kept:
        return "\n".join(kept)
    if not lines:
        return ""
    head = lines[0][:max_chars + 1]
    cut = head.rfind(" ")
    return (head[:cut] if cut > 0 else lines[0][:max_chars]).rstrip()


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
        text = clip_plan(text)
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
        text = clip_plan(text)
        if not text:
            return plan
        ctx.plan_text = text
        return Plan(text, "llm", trace)


def coach_lines(text: str) -> str:
    """A coach reply as at most COACH_MAX_LINES plain lines within COACH_MAX_CHARS. The 14B tends to pack all its
    sentences into one line, so a line longer than COACH_LINE_CHARS is split at its sentence ends first."""
    lines: list[str] = []
    for line in clip_plan(text, 10 * COACH_MAX_CHARS).splitlines():
        lines += _SENTENCE_END_RE.split(line) if len(line) > COACH_LINE_CHARS else [line]
    return clip_plan("\n".join(lines), COACH_MAX_CHARS, COACH_MAX_LINES)


def lookup_summary(hits, n: int = LOOKUP_N) -> list[str]:
    """`"{card} {lane} {zone}"` of the first `n` distinct play keys among `StateIndex.search` hits (what the pros did)."""
    out: list[str] = []
    for hit in hits:
        key = getattr(hit, "action_key", None)
        parts = key.split("|") if key and key != HOLD_KEY else []
        if len(parts) != 3:
            continue
        line = " ".join(parts)
        if line not in out:
            out.append(line)
            if len(out) >= n:
                break
    return out


def coach_update_prompt(plan_text: str, state: dict, t: float, context: str, tactics_lines: list[str] | None = None,
                        lookup: list[str] | None = None) -> str:
    """The user turn of a coach update (`COACH_UPDATE_TEMPLATE`): deck, enemy cards seen, the state, the tactics
    engine's lines, up to three pro moves from similar states and the current plan."""
    deck = state["own"].get("deck") or []
    seen = state["opponent"].get("deck_known") or []
    return COACH_UPDATE_TEMPLATE.format(
        deck=", ".join(deck) if deck else "unknown", opp=", ".join(seen) if seen else "none yet",
        t=round(t, 1), context=context,
        tactics="\n".join(tactics_lines) if tactics_lines else "none",
        lookup="\n".join(f"pros in similar states played {x}" for x in (lookup or [])[:LOOKUP_N]) or "none",
        plan=plan_text or "none yet")


class CoachPlanner(LLMPlanner):
    """The Qwen3-14B coach (D-046). `pregame` is the ReAct pre-game plan of `LLMPlanner` (tools allowed) on the coach
    model; `update` is one chat turn with no tools, so it is fast: at most three plain lines about what to save, where
    to attack and when to spend, written into the decision model's context. Thinking is off. The coach never acts."""

    def __init__(self, chat=None, model_path: str | None = None, max_steps: int = MAX_STEPS,
                 db_path: Path = DB_PATH):
        model_path = model_path or COACH_MODEL_ID
        if chat is None:
            chat = MLXChat(model_path, enable_thinking=False)       # loads lazily, on the thread that first chats
        super().__init__(chat=chat, model_path=model_path, max_steps=max_steps, db_path=db_path)

    def update(self, plan: Plan, state: dict, t: float, tactics_lines: list[str] | None = None,
               lookup: list[str] | None = None, context_text: str | None = None) -> Plan:
        """The next plan, or `plan` itself when the model wrote nothing usable. `lookup` is `lookup_summary(hits)`."""
        context = context_text or render_context(state, t)
        user = coach_update_prompt(plan.text, state, t, context, tactics_lines, lookup)
        text = self.chat.chat([{"role": "system", "content": COACH_SYSTEM_PROMPT},
                               {"role": "user", "content": user}], None)
        text = coach_lines(strip_tool_calls(text))
        return Plan(text, "coach") if text else plan
