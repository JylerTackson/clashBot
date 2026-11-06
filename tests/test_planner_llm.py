"""MAC-E: Qwen ReAct planner — tool-call parsing, the ReAct loop, update cadence and isolation (fake chat, no model)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.db import connect, create_schema  # noqa: E402
from clashbot.planner import (FINAL_NUDGE, PLAN_MAX_CHARS, SYSTEM_PROMPT, LLMPlanner, MLXChat, Plan,  # noqa: E402
                              clip_plan, openai_tools, parse_tool_calls, should_update, strip_tool_calls)
from clashbot.state_text import render_context  # noqa: E402
from clashbot.tools import TOOLS, ToolContext  # noqa: E402


class FakeChat:
    """Replays `replies` in order; with repeat=True the last reply is returned forever."""

    def __init__(self, replies, repeat: bool = False):
        self.replies = list(replies)
        self.repeat = repeat
        self.calls: list[tuple[list[dict], list | None]] = []
        self.kwargs: list[dict] = []         # extra per-call options (the coach passes max_tokens)
        self.last_stats: dict = {}

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append(([dict(m) for m in messages], tools))
        self.kwargs.append(kwargs)
        if len(self.calls) <= len(self.replies):
            return self.replies[len(self.calls) - 1]
        if self.repeat:
            return self.replies[-1]
        raise AssertionError("FakeChat ran out of replies")


def tc(name, arguments=None) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments or {}}) + "\n</tool_call>"


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = connect(path)
    create_schema(conn)
    conn.execute("INSERT INTO cards VALUES (?,?,?,?,?,?,?,?)",
                 ("hog-rider", "Hog Rider", "troop", "rare", 4, "buildings",
                  json.dumps({"speed": "very fast"}), json.dumps({"1": {"hitpoints": 1000}})))
    conn.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                 ("note:1", "pro_note", "Hog cycle", "Hog Rider cycle decks punish an empty elixir bar.",
                  "[]", None, None, None, None))
    conn.execute("INSERT INTO documents_fts VALUES (?,?,?)",
                 ("note:1", "Hog cycle", "Hog Rider cycle decks punish an empty elixir bar."))
    conn.commit()
    conn.close()
    return path


def make_state(deck_known: list[str]) -> dict:
    return {
        "clock": "2:10", "phase": "double_elixir", "match_seconds": 50,
        "own": {"elixir": 6.0, "hand": ["hog-rider", "zap", None, "knight"], "next_card": "archers",
                "deck": ["hog-rider", "zap", "knight", "archers"], "towers": {"king": None, "left": 4000, "right": 5000},
                "recent_plays": []},
        "opponent": {"elixir_estimate": 5.0, "deck_known": deck_known, "towers": {"king": None, "left": 3000, "right": 4858},
                     "recent_plays": []},
        "units": [], "threats": [],
    }


# --- helpers -------------------------------------------------------------------------------------------------------

def test_parse_and_strip_tool_calls():
    reply = ("<think>hmm</think>Let me look.\n" + tc("card_info", {"slug": "hog-rider"}) + "\n" +
             tc("search_documents", {"query": "hog cycle", "k": 2}))
    calls = parse_tool_calls(reply)
    assert calls == [{"name": "card_info", "arguments": {"slug": "hog-rider"}},
                     {"name": "search_documents", "arguments": {"query": "hog cycle", "k": 2}}]
    assert strip_tool_calls(reply) == "Let me look."

    bad = parse_tool_calls("<tool_call>{not json}</tool_call>" + tc("get_current_plan") + "<tool_call>[1]</tool_call>")
    assert bad[0] == {"name": None, "arguments": {}, "error": "bad tool_call json"}
    assert bad[1] == {"name": "get_current_plan", "arguments": {}}
    assert bad[2]["name"] is None and "error" in bad[2]
    assert parse_tool_calls("Plan: push right.") == []
    assert strip_tool_calls("<think>a\nb</think>\n  Plan: push right.  ") == "Plan: push right."


def test_openai_tools():
    tools = openai_tools()
    assert len(tools) == 6
    assert all(t["type"] == "function" for t in tools)
    assert [t["function"]["name"] for t in tools] == [t["name"] for t in TOOLS]
    assert all(set(t["function"]) == {"name", "description", "parameters"} for t in tools)


def test_clip_plan_strips_markdown_and_keeps_whole_lines():
    md = ("## Game plan\n\n"
          "- **Win condition:** Hog Rider cycle through the right lane.\n"
          "* __Spells:__ save Zap for the Inferno Dragon.\n"
          "\n\n"
          "**Defend:** Knight on the Giant, then counter-push.\n")
    out = clip_plan(md)
    assert "**" not in out and "__" not in out and "#" not in out
    assert out.splitlines() == ["Game plan",
                                "Win condition: Hog Rider cycle through the right lane.",
                                "Spells: save Zap for the Inferno Dragon.",
                                "Defend: Knight on the Giant, then counter-push."]
    assert not out.startswith(("-", "*")) and "\n\n" not in out

    long_md = "\n".join(f"- **Line {i}:** " + "word " * 12 for i in range(30))     # ~75 chars a line
    out = clip_plan(long_md)
    assert len(out) <= PLAN_MAX_CHARS and "**" not in out
    kept = out.splitlines()
    assert 0 < len(kept) < 30                                       # ends at a line boundary, whole lines only
    assert all(line.startswith("Line ") and line.endswith("word") for line in kept)
    assert kept == [f"Line {i}: " + "word " * 11 + "word" for i in range(len(kept))]
    assert len("\n".join(kept + [f"Line {len(kept)}: " + "word " * 11 + "word"])) > PLAN_MAX_CHARS
    assert clip_plan(long_md, max_chars=140).count("\n") == 1       # custom limit: two whole lines


def test_clip_plan_cuts_a_single_long_line_at_a_space():
    line = "push right with hog and zap " * 25                      # 700 chars
    line = line[:700]
    out = clip_plan(line)
    assert 0 < len(out) <= PLAN_MAX_CHARS and "\n" not in out
    assert line[len(out)] == " " and not out.endswith(" ")          # cut sits on a word boundary
    assert line.startswith(out)
    exact = ("ab " * 400)[:601]                                       # a space exactly at max_chars
    assert clip_plan(exact) == exact[:600].rstrip()
    assert clip_plan("x" * 2000) == "x" * 600                       # no space anywhere: hard cut
    assert clip_plan("  \n \n") == "" and clip_plan("") == ""
    assert clip_plan("short plan") == "short plan"


def test_prompts_ask_for_short_plain_text():
    assert SYSTEM_PROMPT.endswith("Write plain text without markdown: at most 5 lines, each under 100 characters. "
                                  "Mention only cards from our deck or opponent cards seen so far.")
    assert FINAL_NUDGE == ("Write the final plan now: plain text, at most 5 lines, each under 100 characters, "
                           "no tool calls.")
    assert MLXChat().max_tokens == 200


def test_should_update():
    seen = {"giant"}
    assert should_update(None, 0.0, [], set()) is True
    assert should_update(100.0, 109.9, ["giant"], seen) is False
    assert should_update(100.0, 110.0, ["giant"], seen) is True
    assert should_update(100.0, 101.0, ["giant", "zap"], seen) is True      # new enemy card
    assert should_update(100.0, 101.0, [], seen) is False
    assert should_update(100.0, 104.0, [], seen, period_s=3.0) is True


# --- ReAct loop ----------------------------------------------------------------------------------------------------

def test_react_tool_then_plan(db):
    chat = FakeChat([tc("card_info", {"slug": "hog-rider"}), "Plan: push right."])
    plan = LLMPlanner(chat, db_path=db).pregame(["hog-rider", "zap"], "k")
    assert isinstance(plan, Plan)
    assert plan.source == "llm" and plan.text == "Plan: push right."
    assert len(plan.trace) == 1 and plan.trace[0]["tool"] == "card_info" and plan.trace[0]["ok"] is True
    assert len(chat.calls) == 2
    first_msgs, first_tools = chat.calls[0]
    assert first_msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "hog-rider, zap" in first_msgs[1]["content"] and "Deck: hog-rider, zap" in first_msgs[1]["content"]
    assert len(first_tools) == 6
    second_msgs, _ = chat.calls[1]
    tool_msgs = [m for m in second_msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and "hog-rider" in tool_msgs[0]["content"] and tool_msgs[0]["name"] == "card_info"
    assert second_msgs[-2]["role"] == "assistant"


def test_pregame_sets_ctx_plan_text_and_uses_documents(db):
    conn = connect(db)
    conn.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                 ("deck_why:k", "deck_why", "why", "Cycle fast and punish the double-elixir lull.", "[]", "k",
                  None, None, None))
    conn.commit()
    conn.close()
    chat = FakeChat(["Plan: cycle."])
    ctx = ToolContext(db_path=db)
    plan = LLMPlanner(chat, db_path=db).pregame(["hog-rider"], "k", ctx)
    assert ctx.plan_text == "Plan: cycle." == plan.text
    assert "Cycle fast and punish" in chat.calls[0][0][1]["content"]       # rule-based notes prime the prompt


def test_missing_db_uses_bare_deck_notes(tmp_path):
    chat = FakeChat(["Plan: x."])
    plan = LLMPlanner(chat, db_path=tmp_path / "nope.sqlite").pregame(["zap", "knight"], "k")
    assert plan.source == "llm"
    assert "Deck: zap, knight" in chat.calls[0][0][1]["content"]


def test_bad_tools_and_arguments_become_error_observations(db):
    reply = "\n".join([tc("tap", {"x": 1}),                                   # unknown tool
                       tc("card_info", {"slug": 5}),                          # bad argument type
                       "<tool_call>{broken</tool_call>"])                     # invalid JSON
    chat = FakeChat([reply, tc("kg_query", {"node": "card:hog-rider"}), "Plan: ok."])   # kg missing from ctx
    plan = LLMPlanner(chat, db_path=db).pregame(["hog-rider"], None)
    assert plan.source == "llm" and plan.text == "Plan: ok."
    assert [e["ok"] for e in plan.trace] == [False, False, False, False]
    assert [e["tool"] for e in plan.trace] == ["tap", "card_info", None, "kg_query"]
    msgs = chat.calls[1][0]
    obs = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
    assert len(obs) == 3 and all("error" in o for o in obs)
    assert msgs[-1]["name"] == "invalid"


def test_at_most_three_calls_per_step(db):
    chat = FakeChat([tc("get_current_plan") * 4, "Plan: ok."])
    plan = LLMPlanner(chat, db_path=db).pregame(["zap"])
    assert len(plan.trace) == 3


def test_step_cap_forces_final_answer(db):
    chat = FakeChat([tc("get_current_plan")], repeat=True)
    planner = LLMPlanner(chat, max_steps=3, db_path=db)
    text, trace = planner.run(SYSTEM_PROMPT, "go", ToolContext(db_path=db), 3)
    assert len(chat.calls) == 4 and len(trace) == 3          # max_steps + 1 chats
    assert all(tools is not None for _, tools in chat.calls[:3])
    last_msgs, last_tools = chat.calls[-1]
    assert last_tools is None
    assert last_msgs[-1] == {"role": "user", "content": FINAL_NUDGE}
    assert text == ""                                          # the final reply was still a tool call

    chat2 = FakeChat([tc("get_current_plan")], repeat=True)
    plan = LLMPlanner(chat2, max_steps=2, db_path=db).pregame(["zap"])
    assert len(chat2.calls) == 3 and plan.source == "rules_fallback" and len(plan.trace) == 2


def test_empty_final_text_falls_back_to_rules(db):
    plan = LLMPlanner(FakeChat(["  <think>nothing</think> "]), db_path=db).pregame(["hog-rider", "zap"], "k")
    assert plan.source == "rules_fallback"
    assert plan.text == "Deck: hog-rider, zap"


def test_plan_text_is_capped(db):
    plan = LLMPlanner(FakeChat(["x" * 2000]), db_path=db).pregame(["zap"])
    assert plan.source == "llm" and len(plan.text) == 600


def test_pregame_and_update_clip_markdown_plans(db):
    md = "**Plan**\n\n- **Win:** hog cycle right.\n- **Spells:** zap for the dragon.\n" + "- filler " * 120
    planner = LLMPlanner(FakeChat([md, md]), db_path=db)
    plan = planner.pregame(["hog-rider", "zap"])
    assert plan.source == "llm" and "**" not in plan.text and len(plan.text) <= PLAN_MAX_CHARS
    assert plan.text.splitlines()[:3] == ["Plan", "Win: hog cycle right.", "Spells: zap for the dragon."]
    revised = planner.update(plan, make_state([]), 30.0)
    assert revised.text == plan.text and "-" not in revised.text.splitlines()[1][:1]


# --- update --------------------------------------------------------------------------------------------------------

def test_update_passes_context_and_new_enemy_cards(db):
    chat = FakeChat(["Plan: defend giant.", "Plan: still defend."])
    planner = LLMPlanner(chat, db_path=db)
    old = Plan("Plan: push right.", "llm")
    state = make_state(["giant", "witch"])
    p1 = planner.update(old, state, 61.25)
    user1 = chat.calls[0][0][1]["content"]
    assert "Plan: push right." in user1 and render_context(state, 61.25) in user1
    assert "t=61.2s" in user1 or "t=61.3s" in user1
    assert "not seen when the plan was made: giant, witch." in user1
    assert p1.source == "llm" and p1.text == "Plan: defend giant."

    p2 = planner.update(p1, state, 72.0)
    user2 = chat.calls[1][0][1]["content"]
    assert "not seen when the plan was made: none." in user2
    assert p2.text == "Plan: still defend."


def test_update_uses_given_ctx_and_keeps_plan_on_empty_reply(db):
    chat = FakeChat(["", ])
    planner = LLMPlanner(chat, db_path=db)
    old = Plan("Plan: push right.", "llm")
    ctx = ToolContext(state=make_state([]), t=5.0, context_text="CTX-MARKER", db_path=db)
    assert planner.update(old, make_state([]), 5.0, ctx) is old
    assert "CTX-MARKER" in chat.calls[0][0][1]["content"]
    assert ctx.plan_text == "Plan: push right."


def test_update_step_budget(db):
    chat = FakeChat([tc("get_current_plan")], repeat=True)
    LLMPlanner(chat, db_path=db).update(Plan("p", "llm"), make_state([]), 20.0)
    assert len(chat.calls) == 3                                  # UPDATE_STEPS (2) + the forced final answer


def test_pregame_resets_seen_cards(db):
    chat = FakeChat(["a", "b", "c", "d"])
    planner = LLMPlanner(chat, db_path=db)
    plan = planner.pregame(["zap"])
    planner.update(plan, make_state(["giant"]), 30.0)
    planner.pregame(["zap"])
    planner.update(plan, make_state(["giant"]), 30.0)
    assert "made: giant." in chat.calls[3][0][1]["content"]


# --- isolation ----------------------------------------------------------------------------------------------------

def test_planner_source_has_no_input_paths():
    src = (ROOT / "clashbot" / "planner.py").read_text()
    for banned in ("executor", "CGEvent", "subprocess", "clashbot.loop", "clashbot.live"):
        assert banned not in src, banned


def test_import_is_isolated_and_mlx_is_lazy(db):
    code = (
        "import sys\n"
        "import clashbot.planner as P\n"
        "from clashbot.planner import LLMPlanner, MLXChat\n"
        "class F:\n"
        "    def chat(self, messages, tools=None):\n"
        "        return 'Plan: ok.'\n"
        "planner = LLMPlanner(chat=F(), db_path=sys.argv[1])\n"
        "assert planner.pregame(['zap']).text == 'Plan: ok.'\n"
        "MLXChat()\n"
        "LLMPlanner()\n"
        "bad = [m for m in ('clashbot.executor', 'clashbot.loop', 'clashbot.live', 'mlx', 'mlx_lm') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    proc = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r)\n%s" % (str(ROOT), code),
                           str(db)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_bench_deck_and_state_selection(tmp_path):
    from tools.eval.bench_planner import first_phase_states, pick_decks
    games = [{"game_id": "g1", "split": "train", "own_deck_key": "a"},
             {"game_id": "g2", "split": "val", "own_deck_key": "b"},
             {"game_id": "g3", "split": "test", "own_deck_key": "b"},
             {"game_id": "g4", "split": "val", "own_deck_key": None},
             {"game_id": "g5", "split": "test", "own_deck_key": "c"}]
    states = [{"game_id": "g2", "t": 1.0, "state": {"phase": None}},
              {"game_id": "g3", "t": 2.0, "state": {"phase": None}},
              {"game_id": "g3", "t": 9.0, "state": {"phase": "single_elixir"}},
              {"game_id": "g5", "t": 3.0, "state": {"phase": None}}]
    gp, sp = tmp_path / "games.jsonl", tmp_path / "states.jsonl"
    gp.write_text("\n".join(json.dumps(g) for g in games))
    sp.write_text("\n".join(json.dumps(s) for s in states))
    first = first_phase_states(sp)
    assert first["g3"]["t"] == 9.0 and first["g2"]["t"] == 1.0 and first["g5"]["t"] == 3.0
    has_phase = {g for g, r in first.items() if r["state"]["phase"] is not None}
    picked = pick_decks(gp, 5, has_phase)
    assert [g["game_id"] for g in picked] == ["g3", "g5"]               # keys b, c; train/None-key games excluded
    assert [g["game_id"] for g in pick_decks(gp, 1, has_phase)] == ["g3"]


# --- real model (slow) ---------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CLASHBOT_SLOW_TESTS") != "1", reason="set CLASHBOT_SLOW_TESTS=1 for real-model tests")
def test_real_qwen_replies():
    chat = MLXChat()
    out = chat.chat([{"role": "user", "content": "Reply with the word ok."}])
    assert isinstance(out, str) and out.strip()
    assert chat.last_stats["gen_tokens"] > 0
