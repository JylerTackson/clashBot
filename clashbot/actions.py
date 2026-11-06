"""Option text and placement-prior lookup for ranker options (D-010)."""
from __future__ import annotations

import json
from pathlib import Path

from clashbot.config import DATASET_DIR, HOLD_KEY

ZONE_TEXT = {"back": "back, behind the king tower", "mid": "mid, on our side",
             "bridge": "at the bridge", "enemy_half": "on the enemy side"}

# Hero ability (MAC-AI): a key "ability:musketeer|<lane>|<zone>" is not a card play. Lane and zone are where the turret
# lands (3 tiles in front of the ally Musketeer); the executor taps the on-screen ability button instead of a card.
ABILITY_PREFIX = "ability:"
ABILITY_KEY = "ability:musketeer"              # the card slot of the ability in own plays / elixir accounting
ABILITY_COST = 3                               # elixir
ABILITY_TURRET_TILES = 3                       # the turret spawns this many tiles ahead of the Musketeer (row + 3)
ABILITY_TEXT = ("activate Hero Musketeer's Trusty Turret (3 elixir): a turret 3 tiles in front of her, range 4, "
                "pulls win conditions, 10 s")


def is_ability_key(key: str | None) -> bool:
    return isinstance(key, str) and key.startswith(ABILITY_PREFIX)


def ability_key(lane: str, zone: str) -> str:
    return f"{ABILITY_KEY}|{lane}|{zone}"


def option_text(key: str) -> str:
    if key == HOLD_KEY:
        return "hold elixir and play nothing yet"
    if is_ability_key(key):
        return ABILITY_TEXT
    card, lane, zone = key.split("|")
    return f"play {card} in the {lane} lane, {ZONE_TEXT[zone]}"


def load_placement_prior(path: Path = DATASET_DIR / "placement_prior.json") -> dict:
    return json.loads(Path(path).read_text())


def placement_tile(key: str, prior: dict) -> list[float] | None:
    if key == HOLD_KEY or is_ability_key(key):      # an ability key has no pro-median tile: the executor taps a button
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
