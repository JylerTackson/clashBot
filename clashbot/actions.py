"""Option text and placement-prior lookup for ranker options (D-010)."""
from __future__ import annotations

import json
from pathlib import Path

from clashbot.config import DATASET_DIR, HOLD_KEY

ZONE_TEXT = {"back": "back, behind the king tower", "mid": "mid, on our side",
             "bridge": "at the bridge", "enemy_half": "on the enemy side"}


def option_text(key: str) -> str:
    if key == HOLD_KEY:
        return "hold elixir and play nothing yet"
    card, lane, zone = key.split("|")
    return f"play {card} in the {lane} lane, {ZONE_TEXT[zone]}"


def load_placement_prior(path: Path = DATASET_DIR / "placement_prior.json") -> dict:
    return json.loads(Path(path).read_text())


def placement_tile(key: str, prior: dict) -> list[float] | None:
    if key == HOLD_KEY:
        return None
    entry = prior.get("by_key", {}).get(key)
    if entry and entry.get("median_tile") is not None:
        return entry["median_tile"]
    parts = key.split("|")
    if len(parts) != 3:
        return None
    entry = prior.get("by_lane_zone", {}).get(f"{parts[1]}|{parts[2]}")
    if entry and entry.get("median_tile") is not None:
        return entry["median_tile"]
    return None
