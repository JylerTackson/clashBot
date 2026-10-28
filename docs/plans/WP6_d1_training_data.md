# WP6 — d1-omni-600M training data (spec for a Sonnet implementer)

Author: Opus · 2026-10-28 · Implements D-021, D-022, D-023. Builds on WP1–WP5 (approved).
**Do not fine-tune, download model weights, or install torch/transformers.** This WP only prepares data and validates it.

**You are implementing, not designing.** If anything is unclear, stop and report.

## 0. What d1 expects (from the model card and its `prompt.py`, verified by Opus)
- A request is `(state, questions)`: `state` is a string (or JSON); `questions` is `{name: {"type", "instructions", "criteria"}}`.
- `type: "choice"` → `criteria` is `{option_name: description}` with ≥ 2 options; the model renders each option as `"<name>: <description>"` (or just `<name>` when the description is empty) and gives each option text a budget of about `(max(96, min(24k+32, max_len/2)) - 3k) // k` tokens (k options; ≈29 tokens for k=4). Longer option texts are cut.
- `type: "noul"` → optional `criteria` `{"true": "...", "false": "..."}`; the answer is P(yes).
- Text containing `<|name|>` is escaped by the model; our text must never contain `<|`.
- Context limit 16,384 tokens (text only).

## 1. Files to create
```
clashbot/d1_format.py
tools/data/build_d1_dataset.py
tests/test_d1_format.py
```
Also edit **only** `clashbot/ranker.py`: in `D1Ranker`, add a method `build_request(self, context: str, candidate_keys: list[str], plan: str | None = None) -> tuple[str, dict]` that returns `d1_format.joint_request(context, candidate_keys, plan)`; `choose` stays a stub (NotImplementedError).

## 2. `clashbot/d1_format.py`
Module docstring: "Requests for LiquidAI d1-omni-600M in its Decision Index format (D-023). Shared by the training export and the runtime ranker."

Constants (exact strings):
```python
JOINT_NAME = "play"
JOINT_INSTRUCTIONS = ("You are playing Clash Royale with the deck shown in the state. "
                      "Which of these plays is best right now?")
CARD_NAME, LANE_NAME, ZONE_NAME = "card", "lane", "zone"
CARD_INSTRUCTIONS = "You are playing Clash Royale with the deck shown in the state. Which card should we play now?"
LANE_INSTRUCTIONS = "Which lane should that card go in?"
ZONE_INSTRUCTIONS = "How far forward should it be placed?"
LANE_CRITERIA = {"left": "left lane", "middle": "middle, between the lanes", "right": "right lane"}
ZONE_CRITERIA = {"back": "back, behind the king tower", "mid": "mid, on our side",
                 "bridge": "at the bridge", "enemy_half": "on the enemy side"}
ACT_NAME = "act"
ACT_INSTRUCTIONS = "Should we deploy a card right now rather than wait?"
ACT_CRITERIA = {"true": "deploy a card now", "false": "wait and keep elixir"}
```

```python
def state_string(context_text: str, plan: str | None = None) -> str
```
`context_text` unchanged; if `plan` is a non-empty string, append `"\n[plan] " + plan`. Raise `ValueError` if the result contains `"<|"`.

```python
def option_description(key: str) -> str
```
Play key `card|lane|zone` → `f"{card} {lane} lane, {ZONE_CRITERIA[zone]}"` (example: `"skeletons right lane, mid, on our side"`). `"hold"` → `ValueError` (the joint question never contains hold, D-018).

```python
def joint_request(context_text: str, candidate_keys: list[str], plan: str | None = None) -> tuple[str, dict]
```
Returns `(state_string(context_text, plan), {JOINT_NAME: {"type": "choice", "instructions": JOINT_INSTRUCTIONS, "criteria": {key: option_description(key) for key in candidate_keys}}})`. Keys keep the given order. Raise `ValueError` if fewer than 2 keys or duplicate keys.

```python
def card_description(slug: str, cards: dict) -> str
```
`f"{cards[slug]['name']}, {fmt} elixir {cards[slug]['card_type'].lower()}"` where `fmt` is the cost as an int string when integral, else as given (e.g. `"Skeletons, 1 elixir troop"`).

```python
def factorized_request(context_text: str, possible: list[str], cards: dict, plan: str | None = None) -> tuple[str, dict]
```
Questions, in this order: `card` = choice over `possible` (`{slug: card_description(slug, cards)}`, given order), `lane` = choice over `LANE_CRITERIA`, `zone` = choice over `ZONE_CRITERIA`. Raise `ValueError` if `len(possible) < 2`.

```python
def act_request(context_text: str, plan: str | None = None) -> tuple[str, dict]
```
`{ACT_NAME: {"type": "noul", "instructions": ACT_INSTRUCTIONS, "criteria": ACT_CRITERIA}}`.

```python
def validate(state: str, questions: dict, answers: dict | None = None) -> None
```
Mirrors d1's `Question` checks; raises `ValueError` with a clear message on: state not a str or containing `"<|"`; a question without `type`/`instructions`; type not in `{"choice","score","noul"}`; choice criteria not a dict or < 2 options; any criteria key or description containing `"<|"`. If `answers` is given: same key set as `questions`; a choice answer must be one of its criteria keys; a noul answer must be `"true"` or `"false"`.

## 3. `tools/data/build_d1_dataset.py`
CLI: `python3 tools/data/build_d1_dataset.py [--dataset DIR] [--tokenizer PATH]`. Writes under `<dataset>/d1/`.

Inputs: `<dataset>/ranker/{train,val,test}.jsonl`, `<dataset>/act/{train,val,test}.jsonl`, `<dataset>/states.jsonl` (to get each ranker example's `state` by id), `load_cards()`.

Output line format for every file (one JSON object per line):
```
{"id", "game_id", "split", "kind", "variant", "state", "questions", "answers"}
```

1. **joint** → `d1/joint/<split>.jsonl`: for each ranker example (file order): `keys = [o["key"] for o in example["options"]]` (already shuffled); `state, questions = joint_request(example["context"], keys)`; `answers = {"play": example["positive_key"]}`; `variant = "joint"`.
2. **factorized** → `d1/factorized/<split>.jsonl`: for each ranker example: `rec` = its states.jsonl record; `possible = possible_cards(rec["state"], cards)`; if `len(possible) < 2` → skip and count `skipped_few_cards`; `state, questions = factorized_request(example["context"], possible, cards)`; `answers = {"card": action.card, "lane": action.lane, "zone": action.zone}` from `rec`; skip and count `skipped_card_not_possible` if `action.card not in possible`; `variant = "factorized"`.
3. **act** → `d1/act/<split>.jsonl`: for each act example: `state, questions = act_request(example["context"])`; `answers = {"act": "true" if example["label"] == 1 else "false"}`; `variant = "act"`. Add `"provisional": true` to every act line (D-020).
4. Call `validate(state, questions, answers)` on every line before writing; any failure aborts with the id.
5. `d1/stats.json`: per variant and split: `examples`, skips, answer distribution (`play`: index position of the gold key among criteria keys → counts; `card`: top 10 cards; `lane`/`zone`: counts; `act`: true/false counts), `state_chars` (min/median/max).
6. **Optional token check** — only when `--tokenizer PATH` is given: `from tokenizers import Tokenizer`; `tok = Tokenizer.from_file(PATH)`; for each line compute `len(tok.encode(state, add_special_tokens=False).ids)` and, for every option, `len(tok.encode(" " + rendered_option, add_special_tokens=False).ids)` where `rendered_option` = `f"{name}: {description}"`. Add to stats per variant/split: `state_tokens` (min/median/max) and `option_tokens` (max), and `options_over_budget` = count of options whose token length exceeds `(max(96, min(24*k + 32, 16384 // 2)) - 3*k) // k` for that question's k (lane/zone/act included). If the `tokenizers` import fails, print a message and skip the check (do not fail).
7. `d1/README.md` (generated, plain text): the format above, the three variants with counts, how one line maps to `model.system_one(line["state"], line["questions"])`, that the gold option is `line["answers"]`, and that `act` is provisional (D-020).

## 4. Tests — `tests/test_d1_format.py`
1. `option_description("skeletons|right|mid") == "skeletons right lane, mid, on our side"`; `"hold"` raises.
2. `joint_request` keeps key order; < 2 or duplicate keys raise; `state_string` appends the plan line; `"<|"` raises.
3. `card_description("skeletons", load_cards())` == `"Skeletons, 1 elixir troop"`.
4. `validate` accepts a correct factorized request + answers; rejects: unknown type, 1-option choice, answer not in criteria, noul answer `"yes"`, `"<|"` in a description.
5. Integration (skip if `dataset/v1/ranker/val.jsonl` missing): run the builder; for every variant and split, line counts in stats equal file line counts; every joint line has exactly 4 criteria; every joint answer is in its criteria; every factorized `card` answer is among its card criteria; act lines all have `provisional: true`.

## 5. Acceptance (run, paste outputs)
```
pip install tokenizers
mkdir -p db/hf && curl -sSL -o db/hf/d1_tokenizer.json https://huggingface.co/LiquidAI/d1-omni-600M/resolve/main/tokenizer.json
python3 -m pytest -q tests/test_d1_format.py tests/test_runtime.py
python3 tools/data/build_d1_dataset.py --tokenizer db/hf/d1_tokenizer.json
cat dataset/v1/d1/stats.json
head -c 1500 dataset/v1/d1/joint/val.jsonl; echo
head -c 1500 dataset/v1/d1/factorized/val.jsonl; echo
git status --short
```
(`db/` is gitignored, so the tokenizer file stays local.)

## 6. Report back (≤ 40 lines)
Files; tests; counts per variant/split and skips; token stats incl. `options_over_budget`; deviations and why.
