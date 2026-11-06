"""MAC-AE E1: the Qwen3-14B coach (D-046) - prompt, thinking switch, CoachWorker, LiveLoop wiring, build_coach_plans and
build_d1_tac --plans. Fakes only: no model, no screen, no GPU."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot import d1_format  # noqa: E402
from clashbot import live as live_mod  # noqa: E402
from clashbot.live import CoachPlan, CoachWorker, build_parser, coach_quiet  # noqa: E402
from clashbot.planner import (COACH_MAX_CHARS, COACH_MODEL_ID, COACH_PERIOD_S, COACH_SYSTEM_PROMPT,  # noqa: E402
                              COACH_UPDATE_TOKENS, SYSTEM_PROMPT, CoachPlanner, MLXChat, Plan, PrefixCache,
                              clip_plan, coach_chat, coach_lines, coach_update_prompt, has_thinking_switch,
                              lookup_summary, strip_think, strip_tool_calls, with_no_think)
from clashbot.policy import ActDecision  # noqa: E402
from tests.test_live_loop import (FakeKeyRanker, RangeSource, TimedBuilder, TimedPerception,  # noqa: E402
                                  _argv, _loop)
from tests.test_planner_llm import FakeChat, db, make_state, tc  # noqa: E402, F401  (db is a fixture)
from tests.test_tactics_integration import FakeIndex, FakeKG, synthetic  # noqa: E402, F401  (synthetic is a fixture)
from tools.data import build_coach_plans, build_d1_tac  # noqa: E402

TACTICS = ["tactics: push right in 4 s, answer with knight", "tactics: save zap for the swarm"]
LOOKUP = ["hog-rider left back", "zap middle mid"]


def wait_until(pred, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


# --- the planner ---------------------------------------------------------------------------------------------------

def test_coach_defaults_to_the_14b_model_with_thinking_off_a_prefix_cache_and_loads_lazily():
    assert COACH_MODEL_ID == "mlx-community/Qwen3-14B-4bit"
    coach = CoachPlanner()
    assert isinstance(coach.chat, MLXChat) and coach.chat.model_id == COACH_MODEL_ID
    assert coach.chat.enable_thinking is False and coach.chat._model is None
    assert coach.chat.prefix_cache is True
    tuned = CoachPlanner(prefill_step=64, yield_s=0.01, token_yield_s=0.02).chat
    assert tuned.prefill_step == 64 and tuned.yield_s == 0.01 and tuned.token_yield_s == 0.02
    assert coach_chat("x", prefix_cache=False).prefix_cache is False
    assert CoachPlanner(model_path="/m/other").chat.model_id == "/m/other"


def test_update_prompt_holds_deck_enemy_cards_context_tactics_lookup_and_plan():
    state = make_state(["miner", "bats"])
    chat = FakeChat(["Save zap for bats.\nPush right next."])
    out = CoachPlanner(chat).update(Plan("old plan line", "llm"), state, 70.0, TACTICS, LOOKUP)
    [(messages, tools)] = chat.calls
    assert tools is None                                  # no tool calls in an update
    assert messages[0] == {"role": "system", "content": COACH_SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert "Our deck: hog-rider, zap, knight, archers" in user
    assert "Opponent cards seen: miner, bats" in user
    assert "[double_elixir|2:10]" in user and "elixir 6" in user          # render_context(state, t)
    assert "t=70.0s" in user
    assert all(line in user for line in TACTICS)
    assert "pros in similar states played hog-rider left back" in user
    assert "pros in similar states played zap middle mid" in user
    assert "Current plan:\nold plan line" in user
    assert user.startswith("Our deck: hog-rider, zap, knight, archers\n")      # static part first: prefix-cacheable
    assert user.endswith("Current plan:\nold plan line\n\nWrite the next plan.")   # dynamic part last
    system = messages[0]["content"]
    assert "at most 3 plain lines" in system and "100" in system and "deck" in system
    assert chat.kwargs == [{"max_tokens": COACH_UPDATE_TOKENS}] and COACH_UPDATE_TOKENS == 48
    assert out.text == "Save zap for bats.\nPush right next." and out.source == "coach"


def test_update_prompt_without_lookup_tactics_or_plan_and_with_given_context():
    state = make_state([])
    prompt = coach_update_prompt("", state, 5.0, "CTX TEXT")
    assert "CTX TEXT" in prompt and "Opponent cards seen: none yet" in prompt
    assert "Tactics engine:\nnone" in prompt and "Similar situations:\nnone" in prompt and "Current plan:\nnone yet" in prompt
    chat = FakeChat(["go"])
    CoachPlanner(chat).update(Plan("p", "llm"), state, 5.0, context_text="GIVEN CONTEXT")
    assert "GIVEN CONTEXT" in chat.calls[0][0][1]["content"]
    assert len(prompt.split("pros in similar states")) == 1                # no lookup lines


def test_update_quotes_at_most_three_lookup_lines():
    prompt = coach_update_prompt("p", make_state([]), 1.0, "C", None, ["a b c", "d e f", "g h i", "j k l"])
    assert prompt.count("pros in similar states played") == 3 and "j k l" not in prompt


def test_update_strips_think_and_markdown_and_keeps_three_lines():
    reply = ("<think>the enemy has bats</think>\n**Save zap for bats**\n- push right with hog-rider\n"
             "# spend at 8 elixir\nfourth line\nfifth line")
    out = CoachPlanner(FakeChat([reply])).update(Plan("p", "llm"), make_state([]), 1.0)
    assert out.text.splitlines() == ["Save zap for bats", "push right with hog-rider", "spend at 8 elixir"]
    assert "<think>" not in out.text and "**" not in out.text and len(out.text) <= COACH_MAX_CHARS


def test_update_drops_an_unclosed_think_block_and_tool_calls():
    out = CoachPlanner(FakeChat(["Hold elixir.\n<think>wait, let me reconsider the lanes and"])).update(
        Plan("p", "llm"), make_state([]), 1.0)
    assert out.text == "Hold elixir."
    out = CoachPlanner(FakeChat([tc("card_info", {"slug": "zap"}) + "\nPush left."])).update(
        Plan("p", "llm"), make_state([]), 1.0)
    assert out.text == "Push left."
    assert strip_think("a<think>b</think>c<think>d") == "ac"
    assert strip_tool_calls("<think>x</think> ok") == "ok"


def test_empty_reply_keeps_the_same_plan_object():
    plan = Plan("keep me", "llm")
    assert CoachPlanner(FakeChat(["<think>only thoughts</think>"])).update(plan, make_state([]), 1.0) is plan
    assert CoachPlanner(FakeChat(["   "])).update(plan, make_state([]), 1.0) is plan


def test_pregame_is_the_react_plan_with_tools(db):  # noqa: F811
    chat = FakeChat([tc("card_info", {"slug": "hog-rider"}), "Plan: push right with hog."])
    plan = CoachPlanner(chat, db_path=db).pregame(["hog-rider", "zap"], None)
    assert plan.source == "llm" and plan.text == "Plan: push right with hog."
    assert chat.calls[0][1] is not None and chat.calls[0][0][0]["content"] == SYSTEM_PROMPT
    assert any("hog-rider" in m["content"] for m in chat.calls[1][0] if m["role"] == "tool")


def test_a_long_one_line_reply_is_split_at_sentence_ends_into_at_most_three_lines():
    reply = ("Save Fireball for Baby Dragon. Attack left with Goblin Gang and Dart Goblin. Spend when the Baby Dragon "
             "reveals. And a fourth sentence that must not survive.")
    out = CoachPlanner(FakeChat([reply])).update(Plan("p", "llm"), make_state([]), 1.0)
    assert out.text.splitlines() == ["Save Fireball for Baby Dragon.", "Attack left with Goblin Gang and Dart Goblin.",
                                     "Spend when the Baby Dragon reveals."]
    assert coach_lines("Short line one.\nShort line two. Still short.") == "Short line one.\nShort line two. Still short."
    assert coach_lines("") == "" and coach_lines("   \n ") == ""
    assert all(len(x) <= 100 for x in out.text.splitlines())


def test_prime_prefills_the_static_prefix_with_one_token_and_is_a_no_op_without_a_cache():
    chat = FakeChat(["x"])
    chat.prefix_cache = True
    deck = ["hog-rider", "zap", "knight", "archers"]
    CoachPlanner(chat).prime(deck)
    [(messages, tools)] = chat.calls
    assert tools is None and chat.kwargs == [{"max_tokens": 1}]
    assert messages[0]["content"] == COACH_SYSTEM_PROMPT and messages[1]["content"].startswith(
        "Our deck: hog-rider, zap, knight, archers\nOpponent cards seen: none yet\n")
    # the real update's prompt shares that prefix
    real = coach_update_prompt("p", make_state([]), 70.0, "CTX")
    assert real.startswith("Our deck: hog-rider, zap, knight, archers\nOpponent cards seen: none yet\n")
    quiet = FakeChat([])
    CoachPlanner(quiet).prime(deck)                      # a chat without a prefix cache: nothing to prime
    assert quiet.calls == []


def test_clip_plan_max_lines():
    text = "one\ntwo\nthree\nfour"
    assert clip_plan(text, 600, 3) == "one\ntwo\nthree" and clip_plan(text, 600) == text
    assert clip_plan(text, 600, 1) == "one"


@pytest.mark.parametrize("keys,expected", [
    (["hog-rider|left|back", "hog-rider|left|back", "zap|middle|mid", "hold", None, "knight|right|mid",
      "archers|left|mid"], ["hog-rider left back", "zap middle mid", "knight right mid"]),
    ([None, "hold"], []),
    (["bad-key", "a|b|c"], ["a b c"]),
])
def test_lookup_summary_is_the_top_distinct_play_keys(keys, expected):
    hits = [types.SimpleNamespace(action_key=k) for k in keys]
    assert lookup_summary(hits) == expected
    assert lookup_summary(hits, n=1) == expected[:1]


# --- thinking switch -----------------------------------------------------------------------------------------------

class FakeTok:
    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        self.calls.append((messages, add_generation_prompt, tokenize, kwargs))
        return "PROMPT"


QWEN3_TEMPLATE = "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>{% endif %}"
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "plan the game"}]


def _chat_with(tok, enable_thinking):
    chat = MLXChat("x", enable_thinking=enable_thinking)
    chat._tok = tok
    return chat


def test_thinking_off_passes_enable_thinking_when_the_template_has_the_switch():
    tok = FakeTok(QWEN3_TEMPLATE)
    assert _chat_with(tok, False).render_prompt(MESSAGES, None) == "PROMPT"
    [(messages, gen, tokenize, kwargs)] = tok.calls
    assert kwargs == {"enable_thinking": False} and gen is True and tokenize is False
    assert messages == MESSAGES                                           # unchanged: the switch is in the template
    assert has_thinking_switch(tok) and has_thinking_switch(FakeTok([{"name": "d", "template": QWEN3_TEMPLATE}]))


def test_thinking_off_appends_no_think_when_the_template_lacks_the_switch():
    tok = FakeTok("{{ messages }}")
    _chat_with(tok, False).render_prompt(MESSAGES, [{"type": "function"}])
    [(messages, _gen, _tok, kwargs)] = tok.calls
    assert kwargs == {"tools": [{"type": "function"}]}
    assert messages[-1]["content"] == "plan the game\n/no_think" and messages[0]["content"] == "sys"
    assert MESSAGES[-1]["content"] == "plan the game"                     # the caller's list is not mutated
    assert not has_thinking_switch(tok) and not has_thinking_switch(object())


def test_thinking_default_leaves_the_prompt_alone():
    tok = FakeTok(QWEN3_TEMPLATE)
    _chat_with(tok, None).render_prompt(MESSAGES, None)
    assert tok.calls[0][3] == {} and tok.calls[0][0] == MESSAGES


def test_with_no_think_marks_only_the_last_user_turn_once():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"},
            {"role": "tool", "name": "t", "content": "d"}]
    out = with_no_think(msgs)
    assert [m["content"] for m in out] == ["a", "b", "c\n/no_think", "d"]
    assert with_no_think(out)[2]["content"] == "c\n/no_think"


# --- CoachWorker ---------------------------------------------------------------------------------------------------

class FakeCoachPlanner:
    def __init__(self, gate: threading.Event | None = None, fail_at=(), empty_at=()):
        self.gate, self.fail_at, self.empty_at = gate, set(fail_at), set(empty_at)
        self.calls: list[dict] = []
        self.pregame_threads: list[str] = []

    def pregame(self, own_deck, deck_key=None, ctx=None):
        self.pregame_threads.append(threading.current_thread().name)
        if deck_key == "boom":
            raise RuntimeError("no model")
        return Plan("pregame plan", "llm")

    def update(self, plan, state, t, tactics_lines=None, lookup=None, context_text=None):
        self.calls.append({"plan": plan.text, "state": state, "t": t, "tactics": tactics_lines, "lookup": lookup,
                           "context": context_text, "thread": threading.current_thread().name})
        if self.gate is not None:
            assert self.gate.wait(5), "gate never opened"
        if t in self.fail_at:
            raise RuntimeError("boom")
        if t in self.empty_at:
            return plan
        return Plan(f"plan@{t}", "coach")


def opp(*cards) -> dict:
    return {"opponent": {"deck_known": list(cards)}}


def make_worker(planner=None, **kw) -> CoachWorker:
    kw.setdefault("period_s", 10.0)            # the live default is 20 s; these tests are written around 10
    return CoachWorker(planner or FakeCoachPlanner(), **kw)


def test_pregame_primes_the_planner_on_the_worker_thread_and_a_failing_prime_is_only_logged():
    class Primed(FakeCoachPlanner):
        def __init__(self, fail=False):
            super().__init__()
            self.primed: list[tuple] = []
            self.fail = fail

        def prime(self, own_deck):
            self.primed.append((threading.current_thread().name, list(own_deck)))
            if self.fail:
                raise RuntimeError("no cache")

    planner = Primed()
    w = make_worker(planner)
    assert w.pregame(["zap", "knight"], "k").text == "pregame plan"
    assert planner.primed == [("coach", ["zap", "knight"])]
    w.close()
    bad = Primed(fail=True)
    w = make_worker(bad)
    assert w.pregame(["zap"], "k").text == "pregame plan" and w.errors == 0       # the pregame plan survives
    w.close()


def test_pregame_runs_on_the_worker_thread_and_a_failure_keeps_the_bare_deck():
    planner = FakeCoachPlanner()
    w = make_worker(planner)
    plan = w.pregame(["zap", "knight"], "k")
    assert plan.text == "pregame plan" and w.pregame_plan is plan and w.pregame_s is not None
    assert planner.pregame_threads == ["coach"]
    bad = make_worker(planner)
    plan = bad.pregame(["zap", "knight"], "boom")
    assert plan.text == "Deck: zap, knight" and plan.source == "rules_fallback" and bad.errors == 1
    w.close()
    bad.close()


def test_submit_is_non_blocking_and_one_job_runs_at_a_time():
    gate = threading.Event()
    planner = FakeCoachPlanner(gate)
    w = make_worker(planner)
    w.pregame(["zap"], "k")
    t0 = time.perf_counter()
    assert w.submit(opp(), "CTX", TACTICS, LOOKUP, 0.0) is True
    assert time.perf_counter() - t0 < 0.5                                  # the model is blocked on the gate
    wait_until(lambda: planner.calls)
    assert w.busy and w.latest_plan is None
    assert w.submit(opp("miner"), "CTX", TACTICS, LOOKUP, 50.0) is False   # busy, even though it is long overdue
    assert w.is_due(opp(), 50.0) is False
    gate.set()
    wait_until(lambda: not w.busy)
    assert len(planner.calls) == 1 and w.jobs == 1
    w.close()


def test_job_receives_the_state_copy_context_tactics_lookup_and_current_plan_on_the_worker_thread():
    planner = FakeCoachPlanner()
    w = make_worker(planner)
    w.pregame(["zap"], "k")
    state = {"opponent": {"deck_known": ["miner"]}, "units": [1, 2]}
    assert w.submit(state, "CTX", TACTICS, LOOKUP, 3.0)
    state["units"].append(3)                                               # the loop keeps going; the job has a copy
    wait_until(lambda: not w.busy)
    [call] = planner.calls
    assert call["thread"] == "coach" and call["t"] == 3.0 and call["context"] == "CTX"
    assert call["tactics"] == TACTICS and call["lookup"] == LOOKUP
    assert call["plan"] == "pregame plan" and call["state"]["units"] == [1, 2]
    assert w.latest_plan.text == "plan@3.0" and w.latest_plan.t == 3.0 and w.latest_plan.latency_s >= 0.0
    assert w.submit(opp("miner"), "CTX", [], LOOKUP, 14.0)                 # the next job revises the coach's own plan
    wait_until(lambda: not w.busy)
    assert planner.calls[1]["plan"] == "plan@3.0"
    w.close()


def test_lookup_fn_runs_on_the_worker_thread_only_when_no_lookup_is_given():
    seen = []

    def lookup_fn(state, t, ctx):
        seen.append((threading.current_thread().name, t, ctx))
        return ["from lookup_fn"]

    planner = FakeCoachPlanner()
    w = make_worker(planner, lookup_fn=lookup_fn)
    assert w.submit(opp(), "CTX", [], None, 0.0)
    wait_until(lambda: not w.busy)
    assert seen == [("coach", 0.0, "CTX")] and planner.calls[0]["lookup"] == ["from lookup_fn"]
    assert w.submit(opp(), "CTX", [], ["given"], 10.0)
    wait_until(lambda: not w.busy)
    assert len(seen) == 1 and planner.calls[1]["lookup"] == ["given"]
    w.close()


def test_a_failing_lookup_fn_gives_an_empty_lookup_not_a_failed_job():
    def lookup_fn(state, t, ctx):
        raise OSError("db gone")

    planner = FakeCoachPlanner()
    w = make_worker(planner, lookup_fn=lookup_fn)
    w.submit(opp(), "CTX", [], None, 0.0)
    wait_until(lambda: not w.busy)
    assert planner.calls[0]["lookup"] == [] and w.jobs == 1 and w.errors == 0
    w.close()


def test_due_first_time_after_the_period_and_on_a_new_opponent_card():
    planner = FakeCoachPlanner()
    w = make_worker(planner, period_s=10.0)

    def go(state, t) -> bool:
        ok = w.submit(state, "C", [], [], t)
        wait_until(lambda: not w.busy)
        return ok

    assert go(opp(), 5.0) is True                       # the first time
    assert go(opp(), 5.0 + 9.9) is False                # inside the period
    assert go(opp(), 5.0 + 10.0) is True                # period elapsed
    assert go(opp("miner"), 15.0 + 1.0) is True         # a new opponent card, early
    assert go(opp("miner"), 16.0 + 5.0) is False        # no longer new
    assert go(opp("miner", "bats"), 16.0 + 6.0) is True
    assert [c["t"] for c in planner.calls] == [5.0, 15.0, 16.0, 22.0]
    w.close()


def test_an_exception_keeps_the_old_plan_and_the_worker_goes_on():
    planner = FakeCoachPlanner(fail_at={10.0})
    w = make_worker(planner)
    w.pregame(["zap"], "k")
    w.submit(opp(), "C", [], [], 0.0)
    wait_until(lambda: not w.busy)
    assert w.latest_plan.text == "plan@0.0"
    assert w.submit(opp(), "C", [], [], 10.0)
    wait_until(lambda: not w.busy)
    assert w.errors == 1 and w.latest_plan.text == "plan@0.0" and w.jobs == 1       # old plan kept
    assert w.submit(opp(), "C", [], [], 20.0)                                     # not stuck busy, retried by the clock
    wait_until(lambda: not w.busy)
    assert w.latest_plan.text == "plan@20.0" and w.jobs == 2
    w.close()


def test_an_empty_plan_is_not_published_and_the_next_job_is_not_blocked():
    planner = FakeCoachPlanner(empty_at={0.0})
    w = make_worker(planner)
    w.pregame(["zap"], "k")
    w.submit(opp(), "C", [], [], 0.0)
    wait_until(lambda: not w.busy)
    assert w.latest_plan is None and w.empty == 1 and w.jobs == 0
    w.submit(opp(), "C", [], [], 10.0)
    wait_until(lambda: not w.busy)
    assert w.latest_plan.text == "plan@10.0"
    w.close()


def test_plans_jsonl_gets_t_plan_and_latency(tmp_path):
    w = make_worker()
    w.attach(tmp_path)
    assert (tmp_path / "plans.jsonl").read_text() == ""
    for t in (0.0, 10.0):
        w.submit(opp(), "C", [], [], t)
        wait_until(lambda: not w.busy)
    rows = [json.loads(x) for x in (tmp_path / "plans.jsonl").read_text().splitlines()]
    assert [(r["t"], r["plan"]) for r in rows] == [(0.0, "plan@0.0"), (10.0, "plan@10.0")]
    assert all(r["latency_s"] >= 0 for r in rows)
    assert w.summary()["plans"] == 2 and w.summary()["latency_s"]["p50"] is not None
    w.close()


def test_close_stops_the_thread_and_later_submits_do_nothing():
    planner = FakeCoachPlanner()
    w = make_worker(planner)
    w.submit(opp(), "C", [], [], 0.0)
    wait_until(lambda: not w.busy)
    w.close()
    assert not w._thread.is_alive()
    assert w.submit(opp("miner"), "C", [], [], 99.0) is False and len(planner.calls) == 1
    make_worker().close()                                                  # closing a worker that never started


# --- LiveLoop -----------------------------------------------------------------------------------------------------

class FakeCoach:
    """CoachWorker stand-in for the loop: records offers, serves a settable latest_plan."""

    def __init__(self, latest_plan=None, due=True):
        self.latest_plan = latest_plan
        self.due = due
        self.offers: list[tuple] = []
        self.modes: list = []
        self.closed = 0
        self.attached = None

    def attach(self, out_dir):
        self.attached = out_dir

    def is_due(self, state, t):
        return self.due

    def submit(self, state, ctx, tactics_lines, lookup, t, mode=None):
        self.offers.append((t, ctx, list(tactics_lines), lookup))
        self.modes.append(mode)
        return True

    def close(self):
        self.closed += 1

    def summary(self):
        return {"plans": len(self.offers)}


class FakeTactics:
    def __init__(self):
        self.calls = 0

    def analyse(self, state, t):
        self.calls += 1
        return types.SimpleNamespace(lines=list(TACTICS), pushes=[], moves=[], best=None, hold=None)


class ScriptedPolicy:
    def __init__(self, acts):
        self.acts = acts

    def decide(self, state, t, context_text=None):
        return ActDecision(self.acts(t), "learned", None, "scripted")

    def filter_candidates(self, cands, verdict, state):
        return cands


def test_rank_keys_gets_the_latest_coach_plan_else_the_pregame_plan(tmp_path):
    ranker = FakeKeyRanker()
    coach = FakeCoach()
    lp = _loop(tmp_path, ranker=ranker, coach=coach, plan_text="PREGAME", builder=TimedBuilder(elixir=9.5),
               perception=TimedPerception())
    d0 = lp.step("f", 0.0, 0, 2.0)
    assert ranker.calls[-1][2] == "PREGAME" and d0.plan_t is None and d0.plan_age_s is None
    coach.latest_plan = CoachPlan("SAVE ZAP FOR BATS", 0.15, 1.4)
    d1 = lp.step("f", 4.2, 1, 2.0)
    assert ranker.calls[-1][2] == "SAVE ZAP FOR BATS"
    assert d1.plan_t == 0.15 and d1.plan_age_s == pytest.approx(4.05)
    row = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[-1])
    assert row["plan_t"] == 0.15 and row["plan_age_s"] == pytest.approx(4.05)
    assert coach.attached == tmp_path


def test_without_a_coach_the_plan_is_the_pregame_plan_and_the_log_has_no_age():
    ranker = FakeKeyRanker()
    lp = _loop(ranker=ranker, builder=TimedBuilder(elixir=9.5), perception=TimedPerception(), plan_text="PRE")
    d = lp.step("f", 0.0, 0, 2.0)
    assert ranker.calls[0][2] == "PRE" and d.plan_t is None and d.plan_age_s is None
    assert "coach" not in lp.summary()


def test_the_coach_is_offered_a_job_on_decision_and_hold_steps_at_most_once_a_second():
    coach, tactics = FakeCoach(), FakeTactics()
    acts = lambda t: 1.0 <= t < 1.5                                      # noqa: E731
    lp = _loop(source=RangeSource(0.0, 3.0), builder=TimedBuilder(elixir=5.0), perception=TimedPerception(),
               policy=ScriptedPolicy(acts), tactics=tactics, coach=coach, min_gap_s=1.0)
    summary = lp.run()
    assert [o[0] for o in coach.offers] == [0.0, 1.0, 2.0]               # hold, decision (t=1.0), hold
    assert all(o[1] == "CTX" and o[2] == TACTICS and o[3] is None for o in coach.offers)
    assert tactics.calls == 3                                            # the decision's report is reused, not redone
    assert [d.t for d in lp.decisions] == [1.0] and coach.closed == 1
    assert summary["coach"] == {"plans": 3}


def test_a_coach_that_is_not_due_costs_no_tactics_analysis():
    coach, tactics = FakeCoach(due=False), FakeTactics()
    lp = _loop(source=RangeSource(0.0, 3.0), builder=TimedBuilder(elixir=5.0), perception=TimedPerception(),
               policy=ScriptedPolicy(lambda t: False), tactics=tactics, coach=coach)
    lp.run()
    assert coach.offers == [] and tactics.calls == 0


def test_step_does_not_wait_for_the_coach(tmp_path):
    gate = threading.Event()
    planner = FakeCoachPlanner(gate)
    worker = CoachWorker(planner)
    worker.pregame(["zap"], "k")
    lp = _loop(tmp_path, builder=TimedBuilder(elixir=5.0), perception=TimedPerception(),
               policy=ScriptedPolicy(lambda t: False), coach=worker, plan_text="PRE")
    t0 = time.perf_counter()
    for i in range(30):
        lp.step("f", i * 0.1, i, 1.0)
    assert time.perf_counter() - t0 < 2.0                                # 30 steps while the model is blocked
    wait_until(lambda: planner.calls)
    assert worker.latest_plan is None and len(planner.calls) == 1        # one job at a time, nothing queued behind it
    gate.set()
    worker.close()
    assert worker.latest_plan.text == "plan@0.0"
    rows = [json.loads(x) for x in (tmp_path / "plans.jsonl").read_text().splitlines()]
    assert rows[0]["t"] == 0.0 and rows[0]["plan"] == "plan@0.0"


# --- quiet-time gating --------------------------------------------------------------------------------------------

PUSH_NEAR = "tactics: push left: knight, giant (8 elixir), tower in 6.2s; needs tank"
PUSH_FAR = "tactics: push right: hog-rider (4 elixir), tower in 12.5s; needs nothing special"


@pytest.mark.parametrize("mode,lines,quiet", [
    (None, [], True), ("hold", [], True), ("trigger", [], True), ("leak", [], True), ("punish", [], True),
    ("defend", [], False), ("save", [], False), ("save", [PUSH_FAR], False),
    ("hold", [PUSH_NEAR], False), ("hold", [PUSH_FAR], True), ("hold", [PUSH_FAR, PUSH_NEAR], False),
    ("hold", ["tactics: push left: knight (3 elixir), tower in 10.0s; needs nothing"], False),     # X <= 10
    ("hold", ["tactics: push left: knight (3 elixir), tower in 10.1s; needs nothing"], True),
    ("hold", ["tactics: best knight @[4,12] defend: meets the push, trade +2", "tactics: hold: saving elixir"], True),
])
def test_coach_quiet(mode, lines, quiet):
    assert coach_quiet(mode, lines) is quiet


def test_submit_waits_for_quiet_time_and_keeps_the_job_due():
    planner = FakeCoachPlanner()
    w = make_worker(planner)
    assert w.submit(opp(), "C", [], [], 0.0, mode="defend") is False            # defending
    assert w.submit(opp(), "C", [PUSH_NEAR], [], 1.0, mode="hold") is False     # a push 6 s from a tower
    assert w.submit(opp(), "C", [PUSH_NEAR], [], 2.0, mode="save") is False
    assert planner.calls == [] and not w.busy and w.is_due(opp(), 2.0)          # still due: nothing was consumed
    assert w.submit(opp(), "C", [PUSH_FAR], [], 3.0, mode="hold") is True       # quiet at last
    wait_until(lambda: not w.busy)
    assert [c["t"] for c in planner.calls] == [3.0]
    w.close()


def test_a_new_opponent_card_triggers_a_plan_only_once_things_are_quiet():
    planner = FakeCoachPlanner()
    w = make_worker(planner, period_s=20.0)
    assert w.submit(opp(), "C", [], [], 0.0, mode="hold")
    wait_until(lambda: not w.busy)
    assert w.submit(opp("miner"), "C", [], [], 3.0, mode="defend") is False      # the card is new, but we are defending
    assert w.submit(opp("miner"), "C", [PUSH_NEAR], [], 4.0, mode="hold") is False
    assert w.submit(opp("miner"), "C", [], [], 5.0, mode="hold") is True
    wait_until(lambda: not w.busy)
    assert [c["t"] for c in planner.calls] == [0.0, 5.0]
    assert w.submit(opp("miner"), "C", [], [], 6.0, mode="hold") is False        # not new any more, period not over
    w.close()


def test_submit_without_a_mode_is_gated_by_the_tactics_lines_only():
    w = make_worker()
    assert w.submit(opp(), "C", [PUSH_NEAR], [], 0.0) is False
    assert w.submit(opp(), "C", [], [], 0.0) is True
    w.close()


def test_the_loop_passes_the_policy_mode_and_the_tactics_lines_to_the_coach():
    coach, tactics = FakeCoach(), FakeTactics()
    lp = _loop(source=RangeSource(0.0, 3.0), builder=TimedBuilder(elixir=5.0), perception=TimedPerception(),
               policy=ScriptedPolicy(lambda t: 1.0 <= t < 1.5), tactics=tactics, coach=coach, min_gap_s=1.0)
    lp.run()
    assert coach.modes == ["learned", "learned", "learned"]              # ScriptedPolicy's mode; one per offer
    lp = _loop(source=RangeSource(0.0, 1.5), builder=TimedBuilder(elixir=9.5), perception=TimedPerception(),
               coach=FakeCoach(), min_gap_s=1.0)                         # the trigger policy
    lp.run()
    assert lp.coach.modes and set(lp.coach.modes) == {"trigger"}


class ModePolicy(ScriptedPolicy):
    def __init__(self, mode):
        super().__init__(lambda t: False)
        self.mode = mode

    def decide(self, state, t, context_text=None):
        return ActDecision(False, self.mode, None, "scripted")


def test_a_defending_policy_keeps_the_real_worker_idle_and_a_push_line_does_too():
    planner = FakeCoachPlanner()
    worker = CoachWorker(planner)
    lp = _loop(source=RangeSource(0.0, 3.0), builder=TimedBuilder(elixir=5.0), perception=TimedPerception(),
               policy=ModePolicy("save"), tactics=FakeTactics(), coach=worker)
    lp.run()
    assert planner.calls == [] and worker.jobs == 0
    push = types.SimpleNamespace(analyse=lambda state, t: types.SimpleNamespace(lines=[PUSH_NEAR]))
    worker = CoachWorker(planner)
    lp = _loop(source=RangeSource(0.0, 3.0), builder=TimedBuilder(elixir=5.0), perception=TimedPerception(),
               policy=ModePolicy("hold"), tactics=push, coach=worker)
    lp.run()
    assert planner.calls == []
    worker = CoachWorker(planner)
    lp = _loop(builder=TimedBuilder(elixir=5.0), perception=TimedPerception(),
               policy=ModePolicy("hold"), tactics=FakeTactics(), coach=worker)
    for i in range(30):
        lp.step("f", i * 0.1, i, 1.0)
    wait_until(lambda: planner.calls and not worker.busy)
    assert [c["t"] for c in planner.calls] == [0.0]                      # quiet: the first job, once (period 20 s)
    assert planner.calls[0]["tactics"] == TACTICS
    worker.close()


# --- prompt cache, chunked prefill, token budget ------------------------------------------------------------------

class FakeKV:
    def __init__(self):
        self.offset = 0


class FakeCacheOps:
    """make / trim / can_trim stand-ins for mlx_lm.models.cache: a 'cache' is a list of FakeKV layers."""

    def __init__(self, trimmable=True):
        self.trimmable = trimmable
        self.made = 0
        self.trims: list[int] = []

    def make(self):
        self.made += 1
        return [FakeKV(), FakeKV()]

    def trim(self, cache, n):
        self.trims.append(n)
        for c in cache:
            c.offset -= n

    def can_trim(self, cache):
        return self.trimmable


def feed(cache, n):
    for c in cache:
        c.offset += n


def test_prefix_cache_reuses_the_common_prefix_and_always_prefills_one_token():
    ops = FakeCacheOps()
    pc = PrefixCache(ops.make, ops.trim, ops.can_trim)
    ids1 = [1, 2, 3, 4, 5, 6]
    cache, n = pc.prepare(ids1)
    assert n == 0 and ops.made == 1
    feed(cache, len(ids1) + 3)                         # the prompt, then 3 generated tokens
    pc.commit(ids1)
    assert cache[0].offset == 6 and pc.ids == ids1 and ops.trims == [3]
    ids2 = [1, 2, 3, 4, 9, 9, 9]                       # shares 4 tokens
    cache2, n = pc.prepare(ids2)
    assert cache2 is cache and n == 4 and cache[0].offset == 4 and pc.ids == [1, 2, 3, 4]
    feed(cache, 3 + 2)
    pc.commit(ids2)
    assert cache[0].offset == 7 and pc.ids == ids2
    _, n = pc.prepare(ids2)                            # the identical prompt: still one token to prefill
    assert n == 6 and cache[0].offset == 6
    _, n = pc.prepare([1, 2, 3, 4, 9, 9, 9, 8, 8])     # a longer prompt that extends the previous one
    assert n == 6 + 0 and pc.ids == ids2[:6]


def test_prefix_cache_without_a_trimmable_cache_starts_over_and_reset_clears():
    ops = FakeCacheOps(trimmable=False)
    pc = PrefixCache(ops.make, ops.trim, ops.can_trim)
    cache, _ = pc.prepare([1, 2, 3])
    feed(cache, 3)
    pc.commit([1, 2, 3])                               # nothing to trim: the cache holds exactly the prompt
    cache, n = pc.prepare([1, 2, 3, 4])
    assert n == 3 and ops.made == 1                    # a pure extension needs no trim
    feed(cache, 1)
    pc.commit([1, 2, 3, 4])
    _, n = pc.prepare([1, 2, 9, 9])                    # a divergence cannot be trimmed: a fresh cache
    assert n == 0 and ops.made == 2 and ops.trims == []
    pc.reset()
    assert pc.cache is None and pc.ids == []


def test_prefix_cache_commit_resets_when_the_cache_is_shorter_than_the_prompt():
    ops = FakeCacheOps()
    pc = PrefixCache(ops.make, ops.trim, ops.can_trim)
    cache, _ = pc.prepare([1, 2, 3])
    feed(cache, 2)                                     # inconsistent: fewer tokens than the prompt
    pc.commit([1, 2, 3])
    assert pc.cache is None


class FakeBatchTok:
    chat_template = QWEN3_TEMPLATE
    bos_token = None

    def __init__(self):
        self.encodes: list[tuple] = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def encode(self, text, add_special_tokens=True):
        self.encodes.append((text, add_special_tokens))
        return [ord(c) for c in text]


@pytest.fixture
def fake_mlx(monkeypatch):
    """Fake mlx_lm modules: stream_generate records its kwargs and grows the cache like the real one does."""
    ops = FakeCacheOps()
    calls: list[dict] = []

    def stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
        cache = kwargs.get("prompt_cache")
        cb = kwargs.get("prompt_progress_callback")
        calls.append({"prompt": prompt, "max_tokens": max_tokens, **kwargs})
        if cb:
            cb(0, len(prompt))
            cb(len(prompt), len(prompt))
        n_gen = 3
        if cache is not None:
            feed(cache, len(prompt) + n_gen - 1)
        for i in range(n_gen):
            yield types.SimpleNamespace(text="ab"[i % 2], prompt_tokens=len(prompt), prompt_tps=100.0,
                                        generation_tokens=i + 1, generation_tps=10.0, peak_memory=1.5)

    mods = {
        "mlx_lm": types.SimpleNamespace(stream_generate=stream_generate, load=lambda mid: ("MODEL", FakeBatchTok())),
        "mlx_lm.sample_utils": types.SimpleNamespace(make_sampler=lambda temp=0.0: "SAMPLER"),
        "mlx_lm.models": types.ModuleType("mlx_lm.models"),
        "mlx_lm.models.cache": types.SimpleNamespace(make_prompt_cache=lambda model: ops.make(),
                                                     trim_prompt_cache=ops.trim,
                                                     can_trim_prompt_cache=ops.can_trim),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return ops, calls


MSGS_A = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "deck: zap\ndynamic one"}]
MSGS_B = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "deck: zap\ndynamic two"}]


def test_chat_with_a_prefix_cache_prefills_only_the_changed_tail(fake_mlx):
    ops, calls = fake_mlx
    chat = MLXChat("x", enable_thinking=False, prefix_cache=True)
    assert chat.chat(MSGS_A, None, max_tokens=48) == "aba"
    first = calls[0]
    assert first["max_tokens"] == 48 and len(first["prompt"]) > 40 and first["prompt_cache"] is chat._prefix.cache
    assert chat.last_stats["cached_tokens"] == 0
    assert chat.last_stats["prompt_tokens"] == len(first["prompt"])
    assert chat.last_stats["prefill_ms"] == pytest.approx(len(first["prompt"]) / 100.0 * 1000, abs=0.1)
    assert chat.chat(MSGS_B).startswith("ab")
    second = calls[1]
    assert second["max_tokens"] == 200                                      # the default when no per-call value is given
    assert chat.last_stats["cached_tokens"] > 20 and len(second["prompt"]) < len(first["prompt"]) / 2
    assert chat.last_stats["prompt_tokens"] == chat.last_stats["cached_tokens"] + len(second["prompt"])
    assert chat._prefix.cache[0].offset == len(chat._prefix.ids)            # trimmed back to the prompt: no generated tokens
    assert ops.made == 1                                                    # one cache for the whole conversation


def test_chat_without_a_prefix_cache_passes_the_whole_prompt_and_no_cache(fake_mlx):
    _, calls = fake_mlx
    chat = MLXChat("x", enable_thinking=False)
    chat.chat(MSGS_A)
    assert isinstance(calls[0]["prompt"], str) and "prompt_cache" not in calls[0]
    assert "prefill_step_size" not in calls[0] and "prompt_progress_callback" not in calls[0]
    assert chat.last_stats["cached_tokens"] == 0


def test_chat_passes_the_prefill_chunk_and_yields_between_chunks_and_tokens(fake_mlx, monkeypatch):
    _, calls = fake_mlx
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    chat = MLXChat("x", prefix_cache=True, prefill_step=64, yield_s=0.01, token_yield_s=0.02)
    chat.chat(MSGS_A)
    assert calls[0]["prefill_step_size"] == 64
    assert sleeps == [0.01, 0.01, 0.02, 0.02, 0.02]               # 2 prefill progress callbacks, then 3 generated tokens
    sleeps.clear()
    MLXChat("x", yield_s=0.01).chat(MSGS_A)
    assert sleeps == [0.01, 0.01]                                  # no token yield unless asked for
    sleeps.clear()
    MLXChat("x", token_yield_s=0.02).chat(MSGS_A)
    assert sleeps == [0.02] * 3 and "prompt_progress_callback" not in calls[-1]


def test_chat_drops_the_prefix_cache_when_generation_fails(monkeypatch, fake_mlx):
    ops, _ = fake_mlx

    def boom(*a, **k):
        raise RuntimeError("metal error")
        yield

    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", boom)
    chat = MLXChat("x", prefix_cache=True)
    with pytest.raises(RuntimeError):
        chat.chat(MSGS_A)
    assert chat._prefix.cache is None


def test_a_reply_cut_at_the_token_budget_loses_its_unfinished_last_line():
    chat = FakeChat(["Save zap for bats.\nAttack left with the hog.\nSpend when the bar is fu"])
    chat.last_stats = {"gen_tokens": COACH_UPDATE_TOKENS}
    out = CoachPlanner(chat).update(Plan("p", "llm"), make_state([]), 1.0)
    assert out.text == "Save zap for bats.\nAttack left with the hog."
    chat = FakeChat(["Save zap for bats.\nAttack left with the hog.\nSpend when the bar is fu"])
    chat.last_stats = {"gen_tokens": 30}                                   # not cut: kept as written
    assert CoachPlanner(chat).update(Plan("p", "llm"), make_state([]), 1.0).text.endswith("is fu")
    assert coach_lines("Only one unfinished line", truncated=True) == "Only one unfinished line"
    assert coach_lines("A done line.\nA cut line.", truncated=True) == "A done line.\nA cut line."


# --- CLI wiring ----------------------------------------------------------------------------------------------------

def test_cli_has_the_coach_flags():
    args = build_parser().parse_args(_argv())
    assert args.planner == "rules" and args.coach_model == COACH_MODEL_ID and args.coach_period == 20.0
    assert COACH_PERIOD_S == 20.0 and args.coach_prefill_step == build_parser().get_default("coach_prefill_step")
    assert args.coach_yield_ms == build_parser().get_default("coach_yield_ms")
    args = build_parser().parse_args(_argv("--planner", "coach", "--coach-model", "/m/q", "--coach-period", "6"))
    assert args.planner == "coach" and args.coach_model == "/m/q" and args.coach_period == 6.0
    for choice in ("rules", "llm", "none"):
        assert build_parser().parse_args(_argv("--planner", choice)).planner == choice


def test_build_coach_makes_the_pregame_plan_with_tools_context_and_a_lookup_fn(monkeypatch):
    made = {}

    class StubPlanner(FakeCoachPlanner):
        db_path = Path("x.sqlite")

        def __init__(self, model_path=None, **kw):
            super().__init__()
            made["model"] = model_path
            made["kw"] = kw

        def pregame(self, own_deck, deck_key=None, ctx=None):
            made["ctx"] = ctx
            return super().pregame(own_deck, deck_key, ctx)

    import clashbot.planner as planner_mod
    monkeypatch.setattr(planner_mod, "CoachPlanner", StubPlanner)
    index = FakeIndex([("hog-rider|left|back", 0.1), ("zap|middle|mid", 0.05)])
    args = build_parser().parse_args(_argv("--planner", "coach", "--coach-model", "/m/q", "--coach-period", "7",
                                           "--coach-prefill-step", "64", "--coach-yield-ms", "5",
                                           "--coach-token-yield-ms", "2"))
    worker = live_mod._build_coach(args, ["zap"] * 8, index, "KG")
    assert made["model"] == "/m/q" and worker.period_s == 7.0 and worker.pregame_plan.text == "pregame plan"
    assert made["kw"] == {"prefill_step": 64, "yield_s": 0.005, "token_yield_s": 0.002}
    assert made["ctx"].index is index and made["ctx"].kg == "KG"
    assert worker.lookup_fn({"s": 1}, 3.0, "CTX") == ["hog-rider left back", "zap middle mid"]
    assert index.calls == [{}]                                           # live looks at every split
    worker.close()


def test_main_runs_the_loop_with_the_coach_and_its_pregame_plan(monkeypatch, tmp_path):
    seen = {}

    class StubLoop:
        def __init__(self, *a, **k):
            seen.update(k)
            self.summary = _loop().summary()

        def run(self, **k):
            return self.summary

    worker = types.SimpleNamespace(pregame_plan=Plan("COACH PREGAME", "llm"))
    monkeypatch.setattr(live_mod, "_build_source", lambda args: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(live_mod, "_build_perception", lambda args, source: None)
    monkeypatch.setattr(live_mod, "_build_coach", lambda args, deck, index=None, kg=None: worker)
    monkeypatch.setattr(live_mod, "_build_plan_text", lambda args, deck: pytest.fail("coach has its own pregame"))
    monkeypatch.setattr(live_mod, "load_cards", lambda: {})
    monkeypatch.setattr(live_mod, "load_placement_prior", lambda: {})
    monkeypatch.setattr(live_mod, "StateIndex", lambda: None)
    monkeypatch.setattr(live_mod, "KnowledgeGraph", lambda: None)
    monkeypatch.setattr(live_mod, "CandidateGenerator", lambda *a: None)
    monkeypatch.setattr(live_mod, "LiveLoop", StubLoop)
    out = tmp_path / "run"
    rc = live_mod.main(_argv("--source", "video", "--video", "x.mp4", "--device", "cpu", "--planner", "coach",
                             "--coach-period", "6", "--out", str(out)))
    assert rc == 0 and seen["coach"] is worker and seen["plan_text"] == "COACH PREGAME"
    config = json.loads((out / "config.json").read_text())
    assert config["planner"] == "coach" and config["coach_period"] == 6.0 and config["coach_model"] == COACH_MODEL_ID


def test_build_plan_text_one_shot_coach(monkeypatch):
    import clashbot.planner as planner_mod

    class OneShot:
        def __init__(self, model_path=None, **kw):
            self.model_path = model_path

        def pregame(self, deck, deck_key=None):
            return Plan(f"one-shot {self.model_path} {deck_key}", "llm")

    monkeypatch.setattr(planner_mod, "CoachPlanner", OneShot)
    args = build_parser().parse_args(_argv("--planner", "coach", "--deck-key", "kk"))
    assert live_mod._build_plan_text(args, ["zap"] * 8) == f"one-shot {COACH_MODEL_ID} kk"
    assert live_mod._build_plan_text(build_parser().parse_args(_argv("--planner", "none")), ["zap"] * 8) is None


# --- build_coach_plans ---------------------------------------------------------------------------------------------

class RecordingIndex:
    def __init__(self):
        self.calls: list[tuple] = []

    def search(self, state, t, context_text, k=10, depth=50, **filters):
        self.calls.append((t, context_text, filters))
        return [types.SimpleNamespace(action_key=f"card{int(t)}|left|mid"), types.SimpleNamespace(action_key="hold")]


class FakeEngine:
    def __init__(self, *a, **k):
        pass

    def analyse(self, state, t=None):
        return types.SimpleNamespace(lines=[f"tactics: at {t}"])


class ChainPlanner:
    """Records every update; the plan text carries the time so the chain is visible."""

    def __init__(self, *a, empty_at=(), **k):
        self.calls: list[dict] = []
        self.empty_at = set(empty_at)

    def update(self, plan, state, t, tactics_lines=None, lookup=None, context_text=None):
        self.calls.append({"plan": plan.text, "t": t, "tactics": tactics_lines, "lookup": lookup,
                           "context": context_text})
        if t in self.empty_at:
            return plan
        return Plan(f"coach plan {t}", "coach")


def rec(game_id, t, known=()):
    return {"id": f"{game_id}#{t}", "game_id": game_id, "split": "train", "t": t, "context_text": f"CTX {t}",
            "state": {"opponent": {"deck_known": list(known)}}}


RECORDS = [rec("g1", 0.0), rec("g1", 3.0), rec("g1", 9.0), rec("g1", 10.0), rec("g1", 12.0, ["miner"]),
           rec("g1", 15.0, ["miner"]), rec("g1", 25.0, ["miner"])]


def run_plan_game(done=None, planner=None, records=RECORDS, period=10.0):
    planner = planner or ChainPlanner()
    index = RecordingIndex()
    rows = list(build_coach_plans.plan_game("g1", list(reversed(records)), planner, FakeEngine(), index,
                                            Plan("RULES PLAN", "rules"), period, done,
                                            clock=iter(range(0, 1000, 3)).__next__))
    return rows, planner, index


def test_plan_cadence_period_and_new_opponent_card_in_time_order():
    rows, planner, index = run_plan_game()
    assert [r["t"] for r in rows] == [0.0, 10.0, 12.0, 25.0]            # 3 and 9 are inside the period; 15 is not new
    assert {k for r in rows for k in r} == {"game_id", "t", "plan", "latency_s"}
    assert rows[0]["game_id"] == "g1" and rows[0]["plan"] == "coach plan 0.0" and rows[0]["latency_s"] == 3
    assert [c["plan"] for c in planner.calls] == ["RULES PLAN", "coach plan 0.0", "coach plan 10.0", "coach plan 12.0"]
    first = planner.calls[0]
    assert first["tactics"] == ["tactics: at 0.0"] and first["lookup"] == ["card0 left mid"]
    assert first["context"] == "CTX 0.0"
    assert all(call[2] == {"splits": ("train",), "exclude_game": "g1"} for call in index.calls)


def test_resume_skips_written_pairs_but_keeps_the_cadence_and_the_plan_chain():
    done = {0.0: "written plan 0", 10.0: "written plan 10"}
    rows, planner, _ = run_plan_game(done)
    assert [r["t"] for r in rows] == [12.0, 25.0]
    assert [c["plan"] for c in planner.calls] == ["written plan 10", "coach plan 12.0"]
    rows, planner, _ = run_plan_game({t: f"p{t}" for t in (0.0, 10.0, 12.0, 25.0)})
    assert rows == [] and planner.calls == []


def test_an_empty_model_reply_writes_no_row_and_keeps_the_previous_plan():
    rows, planner, _ = run_plan_game(planner=ChainPlanner(empty_at={10.0}))
    assert [r["t"] for r in rows] == [0.0, 12.0, 25.0]
    assert planner.calls[2]["plan"] == "coach plan 0.0"                  # t=12 revises the plan from t=0


def test_the_first_record_always_gets_a_plan_even_late():
    rows, _, _ = run_plan_game(records=[rec("g1", 44.0), rec("g1", 47.0)])
    assert [r["t"] for r in rows] == [44.0]


@pytest.fixture
def plan_dataset(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    games = [{"game_id": g, "own_deck": ["zap"] * 8, "own_deck_key": f"deck-{g}"} for g in ("g1", "g2", "g3")]
    (ds / "games.jsonl").write_text("\n".join(json.dumps(g) for g in games))
    states = [rec("g1", t) for t in (0.0, 5.0, 12.0)] + [rec("g2", t) for t in (1.0, 14.0)] + [rec("g0", 0.0)]
    (ds / "states.jsonl").write_text("\n".join(json.dumps(r) for r in states))
    return ds


def _patch_build(monkeypatch, planner):
    monkeypatch.setattr(build_coach_plans, "load_cards", lambda: {})
    monkeypatch.setattr(build_coach_plans, "KnowledgeGraph", lambda path: None)
    monkeypatch.setattr(build_coach_plans, "StateIndex", lambda path: RecordingIndex())
    monkeypatch.setattr(build_coach_plans, "TacticalEngine", FakeEngine)
    monkeypatch.setattr(build_coach_plans, "CoachPlanner", lambda **k: planner)
    monkeypatch.setattr(build_coach_plans, "RuleBasedPlanner", lambda path: types.SimpleNamespace(
        pregame=lambda deck, key: Plan(f"rules {key}", "rules")))


def rows_of(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_build_coach_plans_main_writes_resumes_and_limits(plan_dataset, monkeypatch, capsys):
    planner = ChainPlanner()
    _patch_build(monkeypatch, planner)
    out = plan_dataset / "coach_plans.jsonl"
    assert build_coach_plans.main(["--dataset", str(plan_dataset)]) == 0
    rows = rows_of(out)
    assert [(r["game_id"], r["t"]) for r in rows] == [("g1", 0.0), ("g1", 12.0), ("g2", 1.0), ("g2", 14.0)]
    assert planner.calls[0]["plan"] == "rules deck-g1"                    # the chain starts at the rules pre-game plan
    assert planner.calls[2]["plan"] == "rules deck-g2"                    # and restarts for the next game
    # a rerun adds nothing; dropping the last line replans exactly that pair
    planner.calls.clear()
    assert build_coach_plans.main(["--dataset", str(plan_dataset)]) == 0
    assert len(rows_of(out)) == 4 and planner.calls == []
    out.write_text("\n".join(json.dumps(r) for r in rows[:3]) + "\n")
    assert build_coach_plans.main(["--dataset", str(plan_dataset)]) == 0
    assert [(r["game_id"], r["t"]) for r in rows_of(out)] == [("g1", 0.0), ("g1", 12.0), ("g2", 1.0), ("g2", 14.0)]
    assert [(c["t"], c["plan"]) for c in planner.calls] == [(14.0, "coach plan 1.0")]
    # --limit-games and --out
    other = plan_dataset / "limited.jsonl"
    assert build_coach_plans.main(["--dataset", str(plan_dataset), "--limit-games", "1", "--out", str(other)]) == 0
    assert {r["game_id"] for r in rows_of(other)} == {"g1"}
    assert "plans ->" in capsys.readouterr().out


def test_seed_plan_falls_back_to_the_bare_deck_without_a_db(tmp_path):
    from clashbot.planner import RuleBasedPlanner
    game = {"own_deck": ["zap", "knight"], "own_deck_key": "k"}
    plan = build_coach_plans.seed_plan(game, RuleBasedPlanner(tmp_path / "missing.sqlite"))
    assert plan.text == "Deck: zap, knight"


# --- build_d1_tac --plans ------------------------------------------------------------------------------------------

def _tac_run(ds, monkeypatch, best, extra=()):
    index = FakeIndex([(best.key, 0.02)])
    monkeypatch.setattr(build_d1_tac, "StateIndex", lambda: index)
    monkeypatch.setattr(build_d1_tac, "KnowledgeGraph", lambda: FakeKG())
    out = ds / "d1" / ("joint_tac_plans" if extra else "joint_tac")
    assert build_d1_tac.main(["--dataset", str(ds), "--out", str(out), *extra]) == 0
    lines = {s: [json.loads(x) for x in (out / f"{s}.jsonl").read_text().splitlines()] for s in ("train", "val", "test")}
    return lines, json.loads((out / "stats.json").read_text())


def test_build_d1_tac_plans_adds_the_latest_plan_at_or_before_each_line(synthetic, monkeypatch):  # noqa: F811
    ds, best, joint = synthetic                                         # one record per split at t=10: g1 train, g2 val, g3 test
    plans = ds / "coach_plans.jsonl"
    rows = [{"game_id": "g1", "t": 2.0, "plan": "old plan", "latency_s": 1.0},
            {"game_id": "g1", "t": 10.0, "plan": "Save zap for bats.\nPush right.", "latency_s": 1.0},
            {"game_id": "g1", "t": 11.0, "plan": "future plan", "latency_s": 1.0},
            {"game_id": "g2", "t": 10.5, "plan": "not yet made", "latency_s": 1.0},
            {"game_id": "g3", "t": 1.0, "plan": "bad <|plan|>", "latency_s": 1.0}]
    plans.write_text("\n".join(json.dumps(r) for r in rows))
    plain, plain_stats = _tac_run(ds, monkeypatch, best)
    withp, stats = _tac_run(ds, monkeypatch, best, ["--plans", str(plans)])
    [tr], [va], [te] = withp["train"], withp["val"], withp["test"]
    expected = d1_format.state_string(plain["train"][0]["state"], "Save zap for bats.\nPush right.")
    assert tr["state"] == expected and tr["state"].endswith("\n[plan] Save zap for bats.\nPush right.")
    assert "\ntactics: " in tr["state"] and tr["state"].index("tactics:") < tr["state"].index("[plan]")
    d1_format.validate(tr["state"], tr["questions"], tr["answers"])
    assert va == plain["val"][0] and te == plain["test"][0]             # no plan at or before t / an unusable plan
    assert "[plan]" not in va["state"] and "[plan]" not in te["state"]
    assert tr["questions"] == plain["train"][0]["questions"] and tr["answers"] == plain["train"][0]["answers"]
    assert stats["train"]["lines_with_plan"] == 1.0 and stats["val"]["lines_with_plan"] == 0.0
    assert "lines_with_plan" not in plain_stats["train"]               # default output is unchanged
    assert all("[plan]" not in x[0]["state"] for x in plain.values())


def test_latest_plan_picks_the_largest_time_not_after_t():
    plans = {"g": ([1.0, 5.0, 9.0], ["a", "b", "c"])}
    assert build_d1_tac.latest_plan(plans, "g", 0.5) is None
    assert build_d1_tac.latest_plan(plans, "g", 1.0) == "a"
    assert build_d1_tac.latest_plan(plans, "g", 8.9) == "b" and build_d1_tac.latest_plan(plans, "g", 99) == "c"
    assert build_d1_tac.latest_plan(plans, "other", 5.0) is None and build_d1_tac.latest_plan(None, "g", 5.0) is None


def test_load_plans_sorts_by_time_and_drops_unusable_plans(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"game_id": "g", "t": 9.0, "plan": "late"}, {"game_id": "g", "t": 1.0, "plan": "early"},
        {"game_id": "g", "t": 5.0, "plan": ""}, {"game_id": "g", "t": 6.0, "plan": "x <|y"}]))
    assert build_d1_tac.load_plans(path) == {"g": ([1.0, 9.0], ["early", "late"])}


# --- real model (slow) ---------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CLASHBOT_SLOW_TESTS") != "1", reason="set CLASHBOT_SLOW_TESTS=1 for real-model tests")
def test_real_coach_update_has_no_think_block_and_at_most_three_lines():
    out = CoachPlanner().update(Plan("Push right with the hog.", "llm"), make_state(["miner", "bats"]), 70.0,
                                TACTICS, LOOKUP)
    assert out.text and "<think>" not in out.text and len(out.text.splitlines()) <= 3
