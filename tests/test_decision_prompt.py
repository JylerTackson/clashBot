"""Tests for tools/eval/decision_prompt.py and the two LLM runners (MAC-AU Part A): prompt rendering, letter parsing, the
few-shot block, the leakage filter, cost, and the Claude request / outcome handling with fake clients. No model, no network."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval import bench_claude_rankers as bc  # noqa: E402
from tools.eval import bench_llm_rankers as bl  # noqa: E402
from tools.eval import decision_prompt as dp  # noqa: E402


def make_line(i: int = 0, game: str = "g1", gold: int = 1, notes: bool = True) -> dict:
    keys = ["skeletons|left|bridge", "electro-spirit|left|mid", "skeletons|left|mid", "electro-spirit|left|bridge"]
    return {"line_id": f"{game}#{i}.0", "game_id": game, "state_id": f"{game}#{i}.0", "gold": gold, "keys": keys,
            "notes": [f"{k.split('|')[0]} option {n} — filler: no pro or counter support" for n, k in enumerate(keys)]
            if notes else [], "state": "[x|1:00] elixir 9\nfield: none\ntactics: hold: tower handles it\nKG: fire-spirit L15"}


# -- parsing -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reply,expected", [
    ("A", 0), ("B", 1), ("C.", 2), ("D)", 3), ("  b", None), ("Answer: C", 2), ("**D**", 3), ("(B)", 1),
    ("The answer is B.", 1), ("I pick option C because...", 2), ("", None), (None, None), ("no letter here", None),
    ("ABC", None), ("Bomb", None), ("E", None), ("a", None), ("B\nC", 1), ("\n\nC", 2),
])
def test_parse_letter_takes_the_first_standalone_letter(reply, expected):
    assert dp.parse_letter(reply) == expected


def test_parse_letter_skips_letters_inside_words():
    assert dp.parse_letter("Bridge defence, then D") == 3
    assert dp.parse_letter("CB1 A2 then C") == 2


@pytest.mark.parametrize("reply,expected", [
    ("A fire spirit is coming. Placing at the bridge (A) is aggressive.\n\nI'll go with C.\n\n**C**", 2),
    ("reasoning with (B) and A\n\nAnswer: D", 3),
    ("reasoning...\n**Answer: B**", 1),
    ("A", 0), ("**D**", 3), ("C.", 2),
    ("The best is A) skeletons but I would pick C) goblinstein, final answer is C, since it is cheaper.", 2),
    ("A fire spirit is heading left and the pro likely held", None),      # prose with no answer: not the article "A"
    ("", None), (None, None),
])
def test_parse_final_letter_reads_the_end_of_a_reasoning_reply(reply, expected):
    assert dp.parse_final_letter(reply) == expected


# -- rendering ---------------------------------------------------------------------------------------------------------
def test_user_prompt_has_state_four_lettered_options_and_the_fixed_question():
    line = make_line()
    text = dp.build_user_prompt(line)
    assert text.startswith("Game state:\n[x|1:00] elixir 9")
    assert "tactics: hold" in text and "KG: fire-spirit" in text
    for i, letter in enumerate("ABCD"):
        assert f"\n{letter}) {line['notes'][i]}\n" in text + "\n"
    assert text.endswith("Which option did the professional player choose? Answer with one letter: A, B, C or D.")
    assert text.index("A) ") < text.index("B) ") < text.index("C) ") < text.index("D) ") < text.index("Which option")


def test_options_fall_back_to_the_keys_without_notes():
    assert "A) skeletons|left|bridge" in dp.render_options(make_line(notes=False))


def test_render_options_rejects_a_line_that_is_not_four_way():
    bad = make_line()
    bad["keys"] = bad["keys"][:3]
    with pytest.raises(ValueError):
        dp.render_options(bad)


def test_messages_have_a_system_prompt_that_explains_the_options():
    msgs = dp.build_messages(make_line())
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "lettered A to D" in msgs[0]["content"] and "professional" in msgs[0]["content"]
    assert len(msgs[0]["content"]) < 1200            # a short system prompt


# -- few-shot ----------------------------------------------------------------------------------------------------------
def test_examples_are_truncated_and_end_with_the_pro_play():
    ctx = "x" * 900
    block = dp.format_examples([(ctx, "skeletons|left|mid"), ("short state", "bomb-tower|right|back")])
    assert block.startswith("Here are 2 similar situations")
    assert "Example 1:\n" + "x" * 600 + "\nthe pro played skeletons|left|mid" in block
    assert "x" * 601 not in block
    assert "Example 2:\nshort state\nthe pro played bomb-tower|right|back" in block


def test_few_shot_prompt_puts_examples_before_the_lines_own_prompt():
    line = make_line()
    plain = dp.build_user_prompt(line)
    shot = dp.build_user_prompt(line, [("a state", "skeletons|left|mid")])
    assert shot.endswith(plain)
    assert shot.index("Example 1:") < shot.index("Game state:")
    assert dp.build_user_prompt(line, []) == plain


def test_prompt_rows_cover_every_line_once():
    lines = [make_line(i, "g1") for i in range(3)] + [make_line(i, "g2") for i in range(2)]
    rows = dp.render_prompt_rows(lines)
    assert [r["line_id"] for r in rows] == [ln["line_id"] for ln in lines]
    assert all(r["system"] == dp.SYSTEM_PROMPT and r["user"].startswith("Game state:") for r in rows)
    assert [r["gold"] for r in rows] == [ln["gold"] for ln in lines]


# -- cost --------------------------------------------------------------------------------------------------------------
def test_cost_uses_the_list_prices():
    assert dp.cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert dp.cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert dp.cost_usd("claude-sonnet-5-5", 700, 3) == pytest.approx(0.00143)
    assert dp.cost_usd("qwen3-4b", 700, 3) == 0.0
    assert dp.cost_usd("claude-haiku-5-5", None, None) == 0.0


# -- Qwen runner pieces ------------------------------------------------------------------------------------------------
def test_letter_probs_is_a_softmax_over_the_four_letter_tokens():
    logprobs = {10: -0.1, 11: -2.0, 12: -3.0, 13: -9.0, 99: 0.0}
    p = bl.letter_probs(logprobs, [10, 11, 12, 13])
    assert sum(p) == pytest.approx(1.0) and p[0] > p[1] > p[2] > p[3]
    assert bl.letter_probs({}, [1, 2, 3, 4]) is None


def test_variants_match_the_spec():
    assert set(bl.VARIANTS) == {"qwen3-4b", "qwen3-14b", "qwen3-4b-rag3"}
    assert bl.VARIANTS["qwen3-4b"]["repo"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert bl.VARIANTS["qwen3-14b"]["repo"] == "mlx-community/Qwen3-14B-4bit"
    assert all(v["max_tokens"] == 8 for v in bl.VARIANTS.values())
    assert bl.VARIANTS["qwen3-4b-rag3"]["k"] == 3 and bl.VARIANTS["qwen3-4b"]["k"] == 0


class FakeHit:
    def __init__(self, hid, game, key):
        self.id, self.game_id, self.action_key = hid, game, key


class FakeIndex:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, state, t, context_text, k=10, depth=50, **filters):
        self.calls.append({"state": state, "t": t, "context_text": context_text, "k": k, **filters})
        return self.hits[:k]


def make_retriever(hits, states) -> bl.Retriever:
    r = bl.Retriever.__new__(bl.Retriever)             # skip the real index and states.jsonl
    r.k, r.index, r.states = 3, FakeIndex(hits), states
    return r


def test_retriever_searches_train_only_with_the_lines_own_context_and_excludes_its_game():
    line = make_line(0, "gT")
    states = {line["state_id"]: {"state": {"s": 1}, "t": 12.5, "context_text": "own ctx", "game_id": "gT", "split": "test"},
              "a#1": {"state": {}, "t": 0, "context_text": "ctx a", "game_id": "ga", "split": "train"},
              "b#1": {"state": {}, "t": 0, "context_text": "ctx b", "game_id": "gb", "split": "train"}}
    r = make_retriever([FakeHit("a#1", "ga", "skeletons|left|mid"), FakeHit("b#1", "gb", "x|right|back")], states)
    examples, ids = r.examples(line)
    assert examples == [("ctx a", "skeletons|left|mid"), ("ctx b", "x|right|back")] and ids == ["a#1", "b#1"]
    call = r.index.calls[0]
    assert call["splits"] == ["train"] and call["exclude_game"] == "gT" and call["k"] == 3
    assert call["context_text"] == "own ctx" and call["state"] == {"s": 1} and call["t"] == 12.5


def test_retriever_refuses_a_hit_from_outside_train_or_from_the_lines_game():
    line = make_line(0, "gT")
    base = {line["state_id"]: {"state": {}, "t": 0, "context_text": "c", "game_id": "gT", "split": "test"}}
    leak_split = {**base, "v#1": {"state": {}, "t": 0, "context_text": "c", "game_id": "gv", "split": "val"}}
    with pytest.raises(RuntimeError, match="leakage"):
        make_retriever([FakeHit("v#1", "gv", "a|left|mid")], leak_split).examples(line)
    leak_game = {**base, "t#2": {"state": {}, "t": 0, "context_text": "c", "game_id": "gT", "split": "train"}}
    with pytest.raises(RuntimeError, match="leakage"):
        make_retriever([FakeHit("t#2", "gT", "a|left|mid")], leak_game).examples(line)


class FakeRunner:
    name = "fake"

    def __init__(self, fail_on=None):
        self.fail_on, self.generated = fail_on, 0

    def generate(self, messages, max_tokens):
        self.generated += 1

    def decide(self, line):
        if line["line_id"] == self.fail_on:
            raise RuntimeError("boom")
        pred = line["gold"]
        return {"pred": pred, "probs": None, "latency_ms": 5.0, "in_tokens": 10, "out_tokens": 1, "cost_usd": 0.0,
                "raw": "B", "error": None}


def test_run_variant_writes_the_contract_rows_and_survives_a_failing_line(tmp_path):
    lines = [make_line(i, "g1", gold=i % 4) for i in range(4)]
    out = tmp_path / "fake.jsonl"
    summary = bl.run_variant(FakeRunner(fail_on=lines[2]["line_id"]), lines, out)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["line_id"] for r in rows] == [ln["line_id"] for ln in lines]
    assert not out.with_name("fake.jsonl.partial").exists()
    for key in ("line_id", "game_id", "gold", "pred", "probs", "latency_ms", "in_tokens", "out_tokens", "cost_usd", "raw", "error"):
        assert key in rows[0]
    assert rows[2]["pred"] is None and "RuntimeError" in rows[2]["error"]
    assert summary["n"] == 4 and summary["correct"] == 3 and summary["parse_failures"] == 1


def test_run_variant_warms_up_once_and_resumes_from_the_partial_file(tmp_path):
    lines = [make_line(i, "g1", gold=0) for i in range(3)]
    out = tmp_path / "fake.jsonl"
    done = {"line_id": lines[0]["line_id"], "game_id": "g1", "gold": 0, "pred": 0, "probs": None, "latency_ms": 1.0,
            "in_tokens": 1, "out_tokens": 1, "cost_usd": 0.0, "raw": "A", "error": None}
    out.with_name("fake.jsonl.partial").write_text(json.dumps(done) + "\n")
    runner = FakeRunner()
    bl.run_variant(runner, lines, out, resume=True)
    assert runner.generated == 1
    assert [json.loads(x)["line_id"] for x in out.read_text().splitlines()] == [ln["line_id"] for ln in lines]


# -- Claude runner -----------------------------------------------------------------------------------------------------
def fake_response(text="B", stop="end_turn", in_tok=700, out_tok=3, extra_blocks=()):
    blocks = list(extra_blocks) + ([SimpleNamespace(type="text", text=text)] if text is not None else [])
    return SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
                           stop_reason=stop)


def test_variants_and_requests_follow_the_spec():
    assert set(bc.VARIANTS) == {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-sonnet-5-5-long"}
    haiku = bc.build_request("claude-haiku-5-5", "sys", "usr")
    sonnet = bc.build_request("claude-sonnet-5-5", "sys", "usr")
    assert haiku["thinking"] == {"type": "disabled"}
    assert sonnet["thinking"] == {"type": "between_tools"}            # Sonnet 5.5 400s on "disabled"
    for req in (haiku, sonnet):
        assert req["max_tokens"] == 16 and req["system"] == "sys"
        assert req["messages"] == [{"role": "user", "content": "usr"}]      # no prefill
        assert not {"temperature", "top_p", "top_k", "fallbacks", "betas"} & set(req)


def test_outcome_parses_costs_and_flags_the_failure_modes():
    ok = bc.outcome("claude-haiku-5-5", fake_response("C", in_tok=1000, out_tok=2))
    assert ok["pred"] == 2 and ok["error"] is None and ok["in_tokens"] == 1000 and ok["out_tokens"] == 2
    assert ok["cost_usd"] == pytest.approx(1000 * 0.10e-6 + 2 * 0.50e-6)
    refusal = bc.outcome("claude-sonnet-5-5", fake_response("A", stop="refusal"))
    assert refusal["pred"] is None and refusal["error"] == "refusal"
    cut = bc.outcome("claude-sonnet-5-5", fake_response("The fire spirit is coming down the left lane, and", stop="max_tokens", out_tok=16))
    assert cut["pred"] is None and cut["error"] == "max_tokens"
    odd = bc.outcome("claude-haiku-5-5", fake_response("no idea"))
    assert odd["pred"] is None and odd["error"] == "unparseable"
    assert bc.outcome("claude-haiku-5-5", fake_response("x" * 500, stop="end_turn"))["raw"] == "x" * 200


def test_the_long_sonnet_supplement_reads_the_answer_from_the_end_of_the_reply():
    long_req = bc.build_request("claude-sonnet-5-5-long", "sys", "usr")
    assert long_req["max_tokens"] == 2000 and long_req["thinking"] == {"type": "between_tools"}
    assert long_req["model"] == "claude-sonnet-5-5"
    prose = "A fire spirit is coming left (B) is too passive, so I will go with D.\n\n" + "pad " * 60 + "\n\n**D**"
    out = bc.outcome("claude-sonnet-5-5-long", fake_response(prose, out_tok=300))
    assert out["pred"] == 3 and out["raw_tail"].endswith("**D**") and len(out["raw"]) == 200
    assert bc.outcome("claude-sonnet-5-5", fake_response(prose))["pred"] == 0      # the spec's first-letter rule: the article A


def test_reply_text_skips_thinking_blocks():
    resp = fake_response("D", extra_blocks=[SimpleNamespace(type="thinking", thinking="A is good")])
    assert bc.reply_text(resp) == "D"
    assert bc.reply_text(fake_response(None)) == ""


def test_scrub_removes_anything_shaped_like_an_api_key():
    assert "sk-ant" not in bc.scrub("invalid x-api-key sk-ant-api03-AbC_123-xyz rejected")


class FakeMessages:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def prompt_rows(lines):
    return {ln["line_id"]: {"system": "sys", "user": "usr"} for ln in lines}


def test_run_variant_is_sequential_and_writes_contract_rows(tmp_path):
    lines = [make_line(i, "g1", gold=i % 4) for i in range(3)]
    client = SimpleNamespace(messages=FakeMessages([fake_response("A"), fake_response("B"), fake_response("C")]))
    out = tmp_path / "claude-haiku-5-5.jsonl"
    summary = bc.run_variant(client, "claude-haiku-5-5", lines, prompt_rows(lines), out)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert summary["status"] == "ok" and summary["n"] == 3 and not out.with_name(out.name + ".partial").exists()
    assert [r["pred"] for r in rows] == [0, 1, 2] and all(r["latency_ms"] >= 0 for r in rows)
    assert [r["gold"] for r in rows] == [0, 1, 2]
    assert all(r["probs"] is None and r["cost_usd"] > 0 for r in rows)
    assert len(client.messages.requests) == 3


def test_run_variant_stops_when_the_projected_spend_passes_the_budget(tmp_path):
    lines = [make_line(i, "g1") for i in range(10)]
    client = SimpleNamespace(messages=FakeMessages([fake_response("A", in_tok=1_000_000)] * 10))
    summary = bc.run_variant(client, "claude-sonnet-5-5", lines, prompt_rows(lines), tmp_path / "s.jsonl", budget_usd=15.0)
    assert summary["status"].startswith("stopped: projected spend") and summary["n"] == 1
    assert (tmp_path / "s.jsonl.partial").exists() and not (tmp_path / "s.jsonl").exists()


def test_run_variant_counts_the_other_variants_spend_against_the_budget(tmp_path):
    lines = [make_line(i, "g1") for i in range(4)]
    client = SimpleNamespace(messages=FakeMessages([fake_response("A")] * 4))
    summary = bc.run_variant(client, "claude-haiku-5-5", lines, prompt_rows(lines), tmp_path / "h.jsonl",
                             budget_usd=15.0, others_usd=14.9999)
    assert summary["status"].startswith("stopped: projected spend")


def test_typed_api_errors_become_null_predictions_without_stopping_the_run(tmp_path):
    anthropic = pytest.importorskip("anthropic")
    import httpx2 as httpx
    req = httpx.Request("POST", "https://api.example.invalid/v1/messages")

    def err(cls, status):
        return cls("boom sk-ant-secret-1234", response=httpx.Response(status, request=req), body=None)

    lines = [make_line(i, "g1", gold=0) for i in range(3)]
    client = SimpleNamespace(messages=FakeMessages([err(anthropic.RateLimitError, 429), err(anthropic.InternalServerError, 500),
                                                   fake_response("A")]))
    out = tmp_path / "claude-haiku-5-5.jsonl"
    summary = bc.run_variant(client, "claude-haiku-5-5", lines, prompt_rows(lines), out)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert summary["status"] == "ok"
    assert rows[0]["pred"] is None and rows[0]["error"].startswith("RateLimitError")
    assert rows[1]["pred"] is None and rows[1]["error"].startswith("APIStatusError 500")
    assert rows[2]["pred"] == 0
    assert "sk-ant" not in out.read_text()


def test_repeated_client_errors_stop_the_run_early(tmp_path):
    anthropic = pytest.importorskip("anthropic")
    import httpx2 as httpx
    req = httpx.Request("POST", "https://api.example.invalid/v1/messages")
    bad = [anthropic.BadRequestError("bad thinking param", response=httpx.Response(400, request=req), body=None)
           for _ in range(10)]
    lines = [make_line(i, "g1") for i in range(10)]
    client = SimpleNamespace(messages=FakeMessages(bad))
    summary = bc.run_variant(client, "claude-sonnet-5-5", lines, prompt_rows(lines), tmp_path / "s.jsonl")
    assert summary["status"].startswith("stopped: 3 consecutive 4xx") and summary["n"] == 3
