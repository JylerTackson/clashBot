"""Lane/zone geometry, card table and action keys (D-010). Single source of truth for builders and the live loop."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from clashbot.config import ANYWHERE_CARDS, CARD_INDEX, HOLD_KEY, LANES, ZONES


def lane_of(tile) -> str | None:
    if not tile:
        return None
    col = tile[0]
    if col <= 5:
        return "left"
    if col >= 12:
        return "right"
    return "middle"


def zone_of(tile) -> str | None:
    if not tile:
        return None
    row = tile[1]
    if row <= 4:
        return "back"
    if row <= 11:
        return "mid"
    if row <= 16:
        return "bridge"
    return "enemy_half"


def _cost(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _read_cards(path: Path) -> dict[str, dict]:
    raw = json.loads(Path(path).read_text())["cards"]
    out: dict[str, dict] = {}
    for c in raw:
        out[c["slug"]] = {
            "slug": c["slug"], "name": c.get("name"), "card_type": c.get("card_type"),
            "rarity": c.get("rarity"), "targets": c.get("targets"),
            "elixir_cost": _cost(c.get("elixir_cost")),
            "has_evolution": c.get("has_evolution"), "has_hero_variant": c.get("has_hero_variant"),
        }
    return out


@lru_cache(maxsize=1)
def _cards_cached() -> dict[str, dict]:
    return _read_cards(CARD_INDEX)


def load_cards(path: Path = CARD_INDEX) -> dict[str, dict]:
    if Path(path) == CARD_INDEX:
        return _cards_cached()
    return _read_cards(path)


def is_playable(slug: str, cards: dict) -> bool:
    return slug in cards and cards[slug]["card_type"] != "Tower Troop" and cards[slug]["elixir_cost"] is not None


def action_key(card: str | None, lane: str | None, zone: str | None) -> str | None:
    if all(isinstance(x, str) and x for x in (card, lane, zone)):
        return f"{card}|{lane}|{zone}"
    return None


def parse_action_key(key: str) -> dict:
    if key == HOLD_KEY:
        return {"type": "hold", "card": None, "lane": None, "zone": None}
    parts = key.split("|")
    if len(parts) != 3:
        raise ValueError(f"bad action key: {key!r}")
    return {"type": "play", "card": parts[0], "lane": parts[1], "zone": parts[2]}


def legal_cards(state: dict, cards: dict, elixir_tolerance: float = 0.5) -> list[str]:
    own = state["own"]
    hand = [h for h in own["hand"] if h]
    pool = hand if hand else list(own["deck"])
    elixir = own["elixir"] or 0
    out: list[str] = []
    for c in pool:
        if c in out:
            continue
        if is_playable(c, cards) and cards[c]["elixir_cost"] <= elixir + elixir_tolerance:
            out.append(c)
    return out


def legal_zones(card: str, state: dict, cards: dict) -> list[str]:
    if cards[card]["card_type"] == "Spell" or card in ANYWHERE_CARDS:
        return list(ZONES)
    zones = ["back", "mid", "bridge"]
    towers = state["opponent"]["towers"]
    if towers["left"] == 0 or towers["right"] == 0:
        zones.append("enemy_half")
    return zones


def legal_action_keys(state: dict, cards: dict, elixir_tolerance: float = 0.5) -> list[str]:
    keys: list[str] = []
    for card in legal_cards(state, cards, elixir_tolerance):
        for lane in LANES:
            for zone in legal_zones(card, state, cards):
                keys.append(action_key(card, lane, zone))
    keys.append(HOLD_KEY)
    return keys


def possible_cards(state: dict, cards: dict) -> list[str]:
    """Cards the player could plausibly hold: perceived hand plus, for unread slots, the rest of the deck. No elixir filter."""
    own = state["own"]
    known = [h for h in own["hand"] if h]
    unknown_slots = 4 - len(known)
    out: list[str] = []
    for c in known:
        if is_playable(c, cards) and c not in out:
            out.append(c)
    if unknown_slots > 0:
        for c in own["deck"]:
            if is_playable(c, cards) and c not in known and c not in out:
                out.append(c)
    return out


def possible_action_keys(state: dict, cards: dict) -> list[str]:
    keys: list[str] = []
    for card in possible_cards(state, cards):
        for lane in LANES:
            for zone in legal_zones(card, state, cards):
                keys.append(action_key(card, lane, zone))
    return keys
