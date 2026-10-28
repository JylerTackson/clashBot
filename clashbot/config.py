"""Paths and constants shared by the dataset builders and the live loop."""
from __future__ import annotations
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
KB = ROOT / "knowledge_base"
CARD_INDEX = KB / "meta" / "card_index.json"
STATES_DIR = KB / "states"
MATCHES_DIR = KB / "matches"
DECKS_DIR = KB / "decks"
RESOURCES = PKG / "resources"
UNIT_MAP_PATH = RESOURCES / "unit_map.json"
DATASET_VERSION = "v1"
DATASET_DIR = ROOT / "dataset" / DATASET_VERSION
DB_DIR = ROOT / "db"
DB_PATH = DB_DIR / "clashbot.sqlite"

LANES = ("left", "middle", "right")
ZONES = ("back", "mid", "bridge", "enemy_half")
PHASES = ("single_elixir", "double_elixir", "triple_elixir", "overtime")
HOLD_KEY = "hold"
ANYWHERE_CARDS = frozenset({"miner", "goblin-drill"})
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train, val, test — by video (D-006)
