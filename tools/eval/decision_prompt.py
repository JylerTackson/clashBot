"""The one decision prompt every LLM in the MAC-AU benchmark shares (Qwen via MLX, Claude via the API), plus the reply parser.

A line (a row of `runs/bench/au/lines.jsonl`) is rendered the way d1 sees it: the state text, with its `tactics:` and
`KG:` lines, then the four options lettered A-D with their notes, then one fixed question. `parse_letter` takes the first
standalone A-D letter of a reply. Standard library only, so the Claude runner can import it from a venv without torch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LETTERS = "ABCD"
QUESTION = "Which option did the professional player choose? Answer with one letter: A, B, C or D."
EXAMPLE_MAX_CHARS = 600           # retrieved few-shot states are cut to this many characters

SYSTEM_PROMPT = (
    "You are an expert Clash Royale player. You are shown one moment of a professional player's match as a text "
    "summary: elixir, the cards in hand and the decks, the units on the field, the threats heading for towers, recent "
    "plays and tower hit points. It may also hold `tactics:` lines (what a rule engine suggests) and `KG:` lines "
    "(card facts and which cards answer which). Four candidate plays follow, lettered A to D. Each is 'card, lane, "
    "zone', sometimes followed by ' — ' and a note (pros played it in similar states, the tactics engine suggests "
    "it, or filler with no support). Exactly one of them is the play the professional made. Reply with that letter "
    "only.")

# USD per million tokens (input, output), Anthropic list prices for the two models benchmarked.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_lines(path: Path) -> list[dict]:
    """The benchmark lines (`runs/bench/au/lines.jsonl`), in file order."""
    return read_jsonl(Path(path))


def render_options(line: dict) -> str:
    """'A) <option text>' per option; the text is the d1 criterion (description plus note), the key if no note."""
    keys, notes = line["keys"], line.get("notes") or []
    if len(keys) != len(LETTERS):
        raise ValueError(f"line {line.get('line_id')}: expected {len(LETTERS)} options, got {len(keys)}")
    texts = [(notes[i] if i < len(notes) and notes[i] else keys[i]) for i in range(len(keys))]
    return "\n".join(f"{LETTERS[i]}) {t}" for i, t in enumerate(texts))


def format_examples(examples: list[tuple[str, str]], max_chars: int = EXAMPLE_MAX_CHARS) -> str:
    """Few-shot block from (context_text, action_key) pairs: each state cut to `max_chars`, then 'the pro played <key>'."""
    if not examples:
        return ""
    parts = [f"Here are {len(examples)} similar situations from other professional games and what the pro played."]
    for i, (ctx, key) in enumerate(examples, start=1):
        parts.append(f"Example {i}:\n{ctx[:max_chars].rstrip()}\nthe pro played {key}")
    return "\n\n".join(parts)


def build_user_prompt(line: dict, examples: list[tuple[str, str]] | None = None) -> str:
    """The user turn for one line; `examples` (retrieved train states) go first when given."""
    body = f"Game state:\n{line['state']}\n\nOptions:\n{render_options(line)}\n\n{QUESTION}"
    if not examples:
        return body
    return f"{format_examples(examples)}\n\nNow the situation to decide.\n\n{body}"


def build_messages(line: dict, examples: list[tuple[str, str]] | None = None) -> list[dict]:
    """Chat messages (system + user) for an OpenAI/Qwen-style chat template."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user_prompt(line, examples)}]


_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")


def parse_letter(reply: str | None) -> int | None:
    """Index (0-3) of the first standalone A-D letter in the reply, else None (an unparseable reply counts as wrong)."""
    if not reply:
        return None
    m = _LETTER_RE.search(reply)
    return LETTERS.index(m.group(1)) if m else None


_FINAL_LINE_RE = re.compile(
    r"^(?:(?i:(?:final\s+)?answer|option|choice|i['\u2019]ll\s+(?:go\s+with|choose|pick)|(?:the\s+)?pro\s+(?:played|chose))"
    r"\s*(?:is\s*)?:?\s*)?([A-D])$")
_STATED_RE = re.compile(r"(?i:answer|choose|chose|pick|go\s+with|select|final)\W+(?:(?:is|was|would\s+be)\W+)?"
                        r"(?i:option\W+)?([A-D])(?![A-Za-z0-9])")


def parse_final_letter(reply: str | None) -> int | None:
    """For a model that reasons in prose before it answers: the letter its last line is ('**C**', 'Answer: C'), else the
    last 'answer / I'll go with <letter>' statement, else None. The first-letter rule would read the article 'A' or an
    option the reasoning merely mentions."""
    text = (reply or "").strip()
    if not text:
        return None
    tail = re.sub(r"[*_`()\[\].:!\s]+", " ", text.splitlines()[-1]).strip()
    m = _FINAL_LINE_RE.match(tail)
    if m:
        return LETTERS.index(m.group(1))
    found = _STATED_RE.findall(text)
    return LETTERS.index(found[-1]) if found else None


def cost_usd(model: str, in_tokens: int | None, out_tokens: int | None) -> float:
    """Dollar cost of one call from its token counts and `PRICES`; 0.0 for a model without a price (local runs)."""
    if model not in PRICES:
        return 0.0
    p_in, p_out = PRICES[model]
    return ((in_tokens or 0) * p_in + (out_tokens or 0) * p_out) / 1e6


def render_prompt_rows(lines: list[dict]) -> list[dict]:
    """The rows of `runs/bench/au/prompts.jsonl` (the Claude runner reads them): system + user text per line."""
    return [{"line_id": ln["line_id"], "game_id": ln["game_id"], "gold": ln["gold"], "system": SYSTEM_PROMPT,
             "user": build_user_prompt(ln)} for ln in lines]
