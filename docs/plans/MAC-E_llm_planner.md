# MAC-E — Qwen slow-loop planner with mlx-lm (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Implements D-003, D-007 (slow loop), D-027. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
clashbot/planner.py             (add the LLM planner; keep RuleBasedPlanner / build_pregame_prompt behaviour)
tools/eval/bench_planner.py     (new)
tests/test_planner_llm.py       (new)
```

## Contract (R6/R7 and package G code against this)
```python
# clashbot/planner.py
DEFAULT_QWEN_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_STEPS = 6
UPDATE_STEPS = 2
UPDATE_PERIOD_S = 10.0

@dataclass
class Plan:
    text: str
    source: str                          # "rules" | "llm" | "rules_fallback"
    trace: list = field(default_factory=list)   # tool calls made while planning

class MLXChat:
    def __init__(self, model_id: str = DEFAULT_QWEN_ID, max_tokens: int = 384, temperature: float = 0.0): ...
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str
    last_stats: dict                     # {"prompt_tokens", "gen_tokens", "gen_ms"}

def openai_tools(tools: list[dict] = TOOLS) -> list[dict]
def parse_tool_calls(text: str) -> list[dict]
def strip_tool_calls(text: str) -> str
def should_update(last_update_t: float | None, t: float, opp_deck_known: list[str], seen: set[str],
                  period_s: float = UPDATE_PERIOD_S) -> bool

class LLMPlanner:
    def __init__(self, chat=None, model_path: str | None = None, max_steps: int = MAX_STEPS,
                 db_path: Path = DB_PATH): ...
    def pregame(self, own_deck: list[str], deck_key: str | None = None, ctx: ToolContext | None = None) -> Plan
    def update(self, plan: Plan, state: dict, t: float, ctx: ToolContext | None = None) -> Plan
    def run(self, system: str, user: str, ctx: ToolContext, max_steps: int) -> tuple[str, list[dict]]
```
`chat` is any object with `.chat(messages, tools) -> str` (tests pass a fake). `LLMPlanner()` must not import `mlx`/`mlx_lm` until the first `chat` call.

## 1. `MLXChat` (mlx-lm 0.32 installed)
Lazy load on first `chat`:
```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
self._model, self._tok = load(self.model_id)
```
`chat`: `prompt = self._tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)` (pass `tools` only when not `None`); `t0 = perf_counter()`; `text = generate(self._model, self._tok, prompt=prompt, max_tokens=self.max_tokens, sampler=make_sampler(temp=self.temperature), verbose=False)`; `last_stats = {"prompt_tokens": len(self._tok.encode(prompt)), "gen_tokens": len(self._tok.encode(text)), "gen_ms": round(ms, 1)}`; return `text`.

## 2. Helpers
- `openai_tools`: `[{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools]` (TOOLS from `clashbot.tools`).
- `parse_tool_calls(text)`: every `<tool_call>…</tool_call>` block (regex, DOTALL); inner text stripped and `json.loads`; valid → `{"name": obj.get("name"), "arguments": obj.get("arguments") or {}}`; invalid JSON → `{"name": None, "arguments": {}, "error": "bad tool_call json"}`. Qwen3 writes exactly this format.
- `strip_tool_calls(text)`: remove those blocks and any `<think>…</think>` block, then `.strip()`.
- `should_update`: `True` if `last_update_t is None`, or `t - last_update_t >= period_s`, or any card in `opp_deck_known` is not in `seen`; else `False`. (Caller owns `seen`.)

## 3. Prompts (exact strings, module constants)
```python
SYSTEM_PROMPT = ("You are the strategist for a Clash Royale agent. You cannot act or play cards: you can only read "
                 "through the tools you are given. Call a tool when you need a fact (card stats, counters, professional "
                 "notes, similar situations). When you know enough, reply with the plan and no tool call.")
FINAL_NUDGE = "Write the final plan now, in at most 5 short lines, without calling any tool."
UPDATE_TEMPLATE = ("Current plan:\n{plan}\n\nThe game now (t={t}s):\n{context}\n\nEnemy cards not seen when the plan "
                   "was made: {new_cards}.\nKeep or revise the plan in at most 5 short lines.")
```
Pregame user message: `build_pregame_prompt(own_deck, [notes], [t["name"] for t in TOOLS])` (existing function) where `notes = RuleBasedPlanner(self.db_path).pregame(own_deck, deck_key).text` (≤ 600 chars of pro context, D-007 priming). If the DB is missing (`sqlite3.Error`/`FileNotFoundError`), `notes = "Deck: " + ", ".join(own_deck)`.

## 4. ReAct loop `run(system, user, ctx, max_steps)`
```
messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]; trace = []
for step in range(max_steps):
    text = self.chat.chat(messages, openai_tools())
    calls = parse_tool_calls(text)
    if not calls: return strip_tool_calls(text), trace
    messages.append({"role": "assistant", "content": text})
    for c in calls[:3]:                                     # at most 3 tool calls per step
        try:    result = call_tool(c["name"], c["arguments"], ctx) if c["name"] else {"error": c.get("error")}
        except (ValueError, KeyError, TypeError, sqlite3.Error) as e: result = {"error": str(e)}
        obs = json.dumps(result, default=str, ensure_ascii=False)[:2000]
        messages.append({"role": "tool", "name": c["name"] or "invalid", "content": obs})
        trace.append({"step": step, "tool": c["name"], "arguments": c["arguments"],
                      "ok": not (isinstance(result, dict) and "error" in result)})
messages.append({"role": "user", "content": FINAL_NUDGE})
return strip_tool_calls(self.chat.chat(messages, None)), trace
```
Only `clashbot.tools.call_tool` is reachable: the planner never imports `clashbot.executor`, `clashbot.loop` or `clashbot.live` (D-003).

## 5. `pregame` / `update`
- `pregame`: `ctx = ctx or ToolContext(db_path=self.db_path)`; `text, trace = self.run(SYSTEM_PROMPT, user, ctx, self.max_steps)`; `text = text[:PLAN_MAX_CHARS].strip()`; empty → `Plan(RuleBasedPlanner(...).pregame(...).text, "rules_fallback", trace)`; else `Plan(text, "llm", trace)`. Also set `ctx.plan_text = text`.
- `update(plan, state, t, ctx)`: `context = (ctx.context_text if ctx and ctx.context_text else render_context(state, t))`; `new_cards` = `state["opponent"]["deck_known"]` minus cards mentioned in `plan.text` — keep simple: `[c for c in deck_known if c not in self._seen]` where `self._seen` is the set of opponent cards seen at the previous pregame/update (starts empty), then add them to `_seen`; text `"none"` when empty. `run(SYSTEM_PROMPT, UPDATE_TEMPLATE.format(...), ctx or ToolContext(state=state, t=t, context_text=context, plan_text=plan.text, db_path=self.db_path), UPDATE_STEPS)`; empty → return `plan` unchanged; else `Plan(text[:PLAN_MAX_CHARS], "llm", trace)`.
- Keep `RuleBasedPlanner.update` as is.

## 6. `tools/eval/bench_planner.py`
```
$PY tools/eval/bench_planner.py [--dataset DIR] [--db PATH] [--decks 5] [--model ID] [--out PATH]
```
1. Decks: games in `<dataset>/games.jsonl` with split in (val, test) and non-null `own_deck_key`, distinct by key, sorted by key, first `--decks`.
2. One `MLXChat(--model)`; measure `load_s` with a warm-up `chat([{"role":"user","content":"Say ok."}])` (not counted).
3. Per deck: `ctx = ToolContext(index=StateIndex(db), kg=KnowledgeGraph(db), db_path=db)` (build both once, reuse); `t0`; `plan = LLMPlanner(chat).pregame(deck, key, ctx)`; record `pregame_s`, `steps` (max step in trace + 1, or 0), `tool_calls`, `tools_used` (Counter), `tool_errors`, `plan_chars`, `source`, `plan` text. Then one `update`: first record of that game with `phase` not None from `states.jsonl`: `ctx` with `state`, `t`, `context_text`; record `update_s`.
4. Output JSON (`--out`, default `<dataset>/eval/planner_bench.json`): `{"model", "load_s", "decks": [...], "summary": {"pregame_s_p50", "pregame_s_max", "update_s_p50", "mean_tool_calls", "tool_error_rate", "fallbacks"}}`; print a markdown table `| deck | pregame s | steps | tool calls | errors | update s | plan chars |`.

## 7. Tests — `tests/test_planner_llm.py` (fake chat; no model)
`class FakeChat: def __init__(self, replies): ...; def chat(self, messages, tools): self.calls.append((messages copy, tools)); return next reply`.
Tiny DB fixture: `clashbot.db.connect(tmp/"t.sqlite")`, `create_schema`, insert one `cards` row (`hog-rider`, …) and one `documents` row; pass `db_path`.
1. `parse_tool_calls` on a Qwen-style reply with two calls → two dicts; bad JSON → `error`; `strip_tool_calls` removes tool_call and think blocks.
2. ReAct: replies `[<tool_call card_info hog-rider>, "Plan: push right."]` → plan `source == "llm"`, text `"Plan: push right."`, trace has one ok call, and the second chat call's messages contain a `tool` message whose content includes `"hog-rider"`.
3. Unknown tool `"tap"` and bad arguments → error observations (`ok: False`), no exception, loop continues.
4. Step cap: chat always returns a tool call → exactly `max_steps + 1` chat calls, the last with `tools is None` and the last user message `== FINAL_NUDGE`.
5. Empty final text → `source == "rules_fallback"`.
6. `openai_tools()` has 6 entries with `type == "function"` and names equal to `[t["name"] for t in TOOLS]`.
7. `should_update` cases: None → True; 9.9 s → False; 10 s → True; new enemy card → True.
8. `update` passes the current context and the new enemy cards in the user message; second update with no new cards says `none`.
9. Isolation: `import clashbot.planner` does not put `clashbot.executor` into `sys.modules` (pop it first); `planner.py` source contains no `executor`, `CGEvent` or `subprocess`.
10. `LLMPlanner(chat=FakeChat([...]))` never imports `mlx_lm` (check `sys.modules`).
11. `@pytest.mark.slow` + skip unless `CLASHBOT_SLOW_TESTS=1`: real `MLXChat().chat([{"role":"user","content":"Reply with the word ok."}])` returns a non-empty string.

## 8. Smoke (allowed once, after the symlinks; the Qwen model is cached)
`$PY tools/eval/bench_planner.py --decks 1 --out runs/smoke/E/bench.json` — paste the table and the plan text. If the real model's tool-call format differs from §2, adapt `parse_tool_calls` and report one raw reply.

## 9. Heavy run (Opus, at integration)
`bench_planner.py --decks 5` → `docs/data/planner_bench_v1.json`, table in STATUS.md.

## 10. Report
Per MAC-00 §5.
