"""Requests for LiquidAI d1-omni-600M in its Decision Index format (D-023). Shared by the training export and the runtime ranker."""
from __future__ import annotations

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

_QUESTION_TYPES = {"choice", "score", "noul"}


def state_string(context_text: str, plan: str | None = None) -> str:
    text = context_text
    if isinstance(plan, str) and plan:
        text = text + "\n[plan] " + plan
    if "<|" in text:
        raise ValueError("state text must not contain '<|'")
    return text


def option_description(key: str) -> str:
    if key == "hold":
        raise ValueError("the joint question never contains hold (D-018)")
    parts = key.split("|")
    if len(parts) != 3:
        raise ValueError(f"bad play key: {key!r}")
    card, lane, zone = parts
    return f"{card} {lane} lane, {ZONE_CRITERIA[zone]}"


def joint_request(context_text: str, candidate_keys: list[str], plan: str | None = None) -> tuple[str, dict]:
    keys = list(candidate_keys)
    if len(keys) < 2:
        raise ValueError(f"need at least 2 candidate keys, got {len(keys)}")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate candidate keys")
    state = state_string(context_text, plan)
    questions = {JOINT_NAME: {"type": "choice", "instructions": JOINT_INSTRUCTIONS,
                              "criteria": {key: option_description(key) for key in keys}}}
    return state, questions


def card_description(slug: str, cards: dict) -> str:
    card = cards[slug]
    cost = card["elixir_cost"]
    fmt = str(int(cost)) if isinstance(cost, (int, float)) and float(cost).is_integer() else str(cost)
    return f"{card['name']}, {fmt} elixir {card['card_type'].lower()}"


def factorized_request(context_text: str, possible: list[str], cards: dict,
                       plan: str | None = None) -> tuple[str, dict]:
    if len(possible) < 2:
        raise ValueError(f"need at least 2 possible cards, got {len(possible)}")
    state = state_string(context_text, plan)
    questions = {
        CARD_NAME: {"type": "choice", "instructions": CARD_INSTRUCTIONS,
                    "criteria": {slug: card_description(slug, cards) for slug in possible}},
        LANE_NAME: {"type": "choice", "instructions": LANE_INSTRUCTIONS, "criteria": dict(LANE_CRITERIA)},
        ZONE_NAME: {"type": "choice", "instructions": ZONE_INSTRUCTIONS, "criteria": dict(ZONE_CRITERIA)},
    }
    return state, questions


def act_request(context_text: str, plan: str | None = None) -> tuple[str, dict]:
    state = state_string(context_text, plan)
    questions = {ACT_NAME: {"type": "noul", "instructions": ACT_INSTRUCTIONS, "criteria": dict(ACT_CRITERIA)}}
    return state, questions


def validate(state: str, questions: dict, answers: dict | None = None) -> None:
    """Mirror of d1's Question checks; raises ValueError with a clear message."""
    if not isinstance(state, str):
        raise ValueError("state must be a str")
    if "<|" in state:
        raise ValueError("state contains '<|'")
    if not isinstance(questions, dict):
        raise ValueError("questions must be a dict")
    for name, q in questions.items():
        if not isinstance(q, dict) or "type" not in q or "instructions" not in q:
            raise ValueError(f"question {name!r}: needs 'type' and 'instructions'")
        qtype = q["type"]
        if qtype not in _QUESTION_TYPES:
            raise ValueError(f"question {name!r}: unknown type {qtype!r}")
        criteria = q.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"question {name!r}: choice criteria must be a dict with at least 2 options")
        if isinstance(criteria, dict):
            for key, desc in criteria.items():
                if "<|" in str(key) or "<|" in str(desc):
                    raise ValueError(f"question {name!r}: criteria contain '<|'")
    if answers is None:
        return
    if set(answers) != set(questions):
        raise ValueError(f"answer keys {sorted(answers)} != question keys {sorted(questions)}")
    for name, ans in answers.items():
        q = questions[name]
        if q["type"] == "choice":
            if ans not in q["criteria"]:
                raise ValueError(f"question {name!r}: answer {ans!r} is not one of its criteria keys")
        elif q["type"] == "noul":
            if ans not in ("true", "false"):
                raise ValueError(f"question {name!r}: noul answer must be 'true' or 'false', got {ans!r}")
