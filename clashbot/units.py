"""Resolve YOLO unit classes to cards (D-014)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from clashbot.config import UNIT_MAP_PATH
from clashbot.game_rules import load_cards


def _read_unit_map(path: Path) -> dict[str, dict]:
    raw = json.loads(Path(path).read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def _unit_map_cached() -> dict[str, dict]:
    return _read_unit_map(UNIT_MAP_PATH)


def load_unit_map(path: Path = UNIT_MAP_PATH) -> dict[str, dict]:
    if Path(path) == UNIT_MAP_PATH:
        return _unit_map_cached()
    return _read_unit_map(path)


def resolve(unit_class: str, cards: dict | None = None, unit_map: dict | None = None) -> dict:
    cards = load_cards() if cards is None else cards
    unit_map = load_unit_map() if unit_map is None else unit_map
    if unit_class in unit_map:
        e = unit_map[unit_class]
        return {"card": e["primary"], "is_evo": e["is_evo"], "kind": e["kind"],
                "candidates": [c for c in e["candidates"] if c in cards]}
    if unit_class in cards:
        return {"card": unit_class, "is_evo": False, "kind": "card", "candidates": [unit_class]}
    return {"card": None, "is_evo": False, "kind": "unknown", "candidates": []}
