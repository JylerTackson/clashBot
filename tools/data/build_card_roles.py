"""Build clashbot/resources/card_roles.json (MAC-Z1): per-card roles for the tactical engine.

  python tools/data/build_card_roles.py [--stats-dir data/external/cr-api-data] [--out clashbot/resources/card_roles.json] [--fetch]

Stats-derived fields (flying, targets, splash, range, speed) come from RoyaleAPI cr-api-data (`cards_stats.json`: the
`troop` / `building` / `spell` sections map a card `key` to its `summon_character`, whose per-unit stats are in
`characters`; buildings carry their stats in the section entry itself). Everything else is the MANUAL table below.
Where stats and MANUAL disagree on a stats-derived field, the stats win and the disagreement is printed, except for the
(card, field) pairs in MANUAL_WINS, where the manual value is kept (multi-unit, chain and pierce cards; printed as "kept
manual"). Count, hp_class and spell radius are manual-only but are sanity-checked against the stats (printed as notes).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.card_roles import (  # noqa: E402
    DPS_CLASSES, GROUND_ONLY_SPELLS, HP_CLASSES, RANGES, RESOURCE, ROLES, SPEEDS, SPELL_CLASSES, TARGETS, TYPES,
    validate_roles,
)
from clashbot.game_rules import is_playable, load_cards  # noqa: E402

BASE_URL = "https://raw.githubusercontent.com/RoyaleAPI/cr-api-data/master/docs/json/"
STATS_FILES = ("cards_stats.json", "cards.json")
DEFAULT_STATS_DIR = ROOT / "data" / "external" / "cr-api-data"

# our slug -> cr-api `key` where they differ
KEY_ALIASES = {"p-e-k-k-a": "pekka", "mini-p-e-k-k-a": "mini-pekka"}

STATS_FIELDS = ("flying", "targets", "splash", "range", "speed")
MELEE_MAX_RANGE = 1600                # game units (1,000 per tile); the spec said 1,500, amended so 1.6-tile melee units are melee
SPLASH_PROJECTILE_MIN = 500           # game units: a projectile radius of at least half a tile counts as splash
TOURNAMENT_LEVEL = 11
RARITY_START_LEVEL = {"Common": 1, "Rare": 3, "Epic": 6, "Legendary": 9, "Champion": 11}   # first entry of *_per_level
HP_BAND_SWARM, HP_BAND_LIGHT, HP_BAND_TANK = 250, 690, 3000      # level-11 hitpoints: swarm <= 250 < light <= 690 < medium < 3000 <= tank

G, AG, B = "ground", "air_ground", "buildings"


def troop(targets, hp, dps, rng, speed, count, roles, flying=False, splash=False) -> dict:
    return {"type": "troop", "targets": targets, "flying": flying, "hp_class": hp, "dps_class": dps, "splash": splash,
            "range": rng, "speed": speed, "count": count, "roles": roles}


def building(targets, hp, dps, rng, roles, splash=False) -> dict:
    return {"type": "building", "targets": targets, "flying": False, "hp_class": hp, "dps_class": dps, "splash": splash,
            "range": rng, "speed": None, "count": 1, "roles": roles}


def spell(spell_class, radius, kills, roles) -> dict:
    return {"type": "spell", "spell_class": spell_class, "radius": radius, "kills": kills, "roles": roles}


# Hand-written for every playable card (plus mirror and spirit-empress, which the card index has without an elixir cost).
# hp_class is the level-11 hitpoints of one unit: swarm dies to zap/log, light to fireball, tank >= ~3000.
# dps_class is the card's total single-target dps at level 11 (low < 130 <= medium < 300 <= high); ramping inferno
# cards are high, suicide cards (spirits, wall breakers) low. Values marked "guess" are for cards cr-api-data lacks.
MANUAL: dict[str, dict] = {
    "archer-queen": troop(AG, "medium", "medium", "long", "medium", 1, ["support", "air_defense"]),
    "archers": troop(AG, "light", "medium", "long", "medium", 2, ["support", "air_defense"]),
    "arrows": spell("small", 4.0, ["swarm"], ["spell_small"]),
    "baby-dragon": troop(AG, "medium", "low", "short", "fast", 1, ["splash_defense", "air_defense"], flying=True, splash=True),
    "balloon": troop(B, "medium", "high", "melee", "medium", 1, ["win_condition"], flying=True),
    "bandit": troop(G, "medium", "medium", "melee", "fast", 1, ["support"]),
    "barbarian-barrel": spell("small", 2.0, ["swarm"], ["spell_small"]),
    "barbarian-hut": building(G, "medium", "low", "melee", ["building"]),
    "barbarians": troop(G, "light", "high", "melee", "medium", 5, ["swarm", "tank_killer"]),
    "bats": troop(AG, "swarm", "high", "melee", "very_fast", 5, ["swarm", "cycle"], flying=True),
    "battle-healer": troop(G, "medium", "low", "melee", "medium", 1, ["support"]),
    "battle-ram": troop(B, "medium", "high", "melee", "medium", 1, ["win_condition", "kamikaze"]),
    "berserker": troop(G, "light", "medium", "melee", "medium", 1, ["cycle"]),                       # guess
    "bomb-tower": building(G, "medium", "medium", "long", ["building", "splash_defense"], splash=True),
    "bomber": troop(G, "light", "low", "short", "medium", 1, ["splash_defense"], splash=True),
    "boss-bandit": troop(G, "medium", "high", "melee", "fast", 1, ["mini_tank"]),                    # guess
    "bowler": troop(G, "medium", "low", "short", "slow", 1, ["splash_defense", "support"], splash=True),
    "cannon": building(G, "medium", "medium", "long", ["building"]),
    "cannon-cart": troop(G, "medium", "medium", "long", "medium", 1, ["support"]),
    "clone": spell("utility", 3.0, [], []),
    "dark-prince": troop(G, "medium", "medium", "melee", "medium", 1, ["mini_tank", "splash_defense"], splash=True),
    "dart-goblin": troop(AG, "light", "medium", "long", "very_fast", 1, ["support", "air_defense", "bait"]),
    "earthquake": spell("small", 3.5, [], ["spell_small"]),
    "electro-dragon": troop(AG, "medium", "low", "short", "medium", 1, ["air_defense", "splash_defense"], flying=True, splash=True),
    "electro-giant": troop(B, "tank", "low", "melee", "slow", 1, ["tank", "win_condition"]),
    "electro-spirit": troop(AG, "swarm", "low", "short", "very_fast", 1, ["cycle", "kamikaze"]),
    "electro-wizard": troop(AG, "medium", "low", "long", "fast", 1, ["support", "air_defense"]),
    "elite-barbarians": troop(G, "medium", "high", "melee", "fast", 2, ["tank_killer", "mini_tank"]),
    "elixir-collector": building(G, "medium", "low", "melee", ["building"]),
    "elixir-golem": troop(B, "medium", "medium", "melee", "slow", 1, ["mini_tank", "win_condition"]),
    "executioner": troop(AG, "medium", "medium", "short", "medium", 1, ["splash_defense", "air_defense"], splash=True),
    "fire-spirit": troop(AG, "swarm", "low", "short", "very_fast", 1, ["cycle", "kamikaze"], splash=True),
    "fireball": spell("medium", 2.5, ["swarm", "light"], ["spell_medium"]),
    "firecracker": troop(AG, "light", "low", "long", "fast", 1, ["support", "air_defense"], splash=True),
    "fisherman": troop(G, "medium", "medium", "melee", "medium", 1, ["support"]),
    "flying-machine": troop(AG, "light", "medium", "long", "fast", 1, ["air_defense", "support"], flying=True),
    "freeze": spell("utility", 3.0, [], []),
    "furnace": building(AG, "medium", "low", "short", ["building"], splash=True),
    "giant": troop(B, "tank", "medium", "melee", "slow", 1, ["tank", "win_condition"]),
    "giant-skeleton": troop(G, "tank", "medium", "melee", "medium", 1, ["tank"]),
    "giant-snowball": spell("small", 2.5, ["swarm"], ["spell_small"]),
    "goblin-barrel": spell("utility", 1.5, [], ["win_condition"]),
    "goblin-cage": building(G, "medium", "low", "long", ["building"]),
    "goblin-curse": spell("utility", 3.0, [], []),                                                   # guess
    "goblin-demolisher": troop(G, "medium", "medium", "short", "medium", 1, ["splash_defense", "support"], splash=True),   # guess
    "goblin-drill": building(G, "medium", "low", "melee", ["building", "win_condition"]),
    "goblin-gang": troop(AG, "swarm", "high", "short", "very_fast", 6, ["swarm", "bait"]),
    "goblin-giant": troop(B, "tank", "medium", "melee", "medium", 1, ["tank", "win_condition"]),
    "goblin-hut": building(AG, "medium", "low", "long", ["building"]),
    "goblin-machine": troop(AG, "medium", "medium", "melee", "medium", 1, ["mini_tank"]),            # guess
    "goblins": troop(G, "swarm", "high", "melee", "very_fast", 4, ["swarm", "cycle"]),
    "goblinstein": troop(AG, "tank", "medium", "short", "slow", 1, ["tank"]),                        # guess
    "golden-knight": troop(G, "medium", "medium", "melee", "medium", 1, ["mini_tank"]),
    "golem": troop(B, "tank", "low", "melee", "slow", 1, ["tank", "win_condition"]),
    "graveyard": spell("utility", 4.0, [], ["win_condition"]),
    "guards": troop(G, "light", "high", "melee", "fast", 3, ["swarm"]),
    "heal-spirit": troop(AG, "swarm", "low", "short", "very_fast", 1, ["kamikaze", "support"], splash=True),
    "hog-rider": troop(B, "medium", "medium", "melee", "very_fast", 1, ["win_condition"]),
    "hunter": troop(AG, "medium", "high", "short", "medium", 1, ["tank_killer", "air_defense"], splash=True),
    "ice-golem": troop(B, "medium", "low", "melee", "slow", 1, ["mini_tank"]),
    "ice-spirit": troop(AG, "swarm", "low", "short", "very_fast", 1, ["cycle", "kamikaze"], splash=True),
    "ice-wizard": troop(AG, "light", "low", "long", "medium", 1, ["support", "splash_defense"], splash=True),
    "inferno-dragon": troop(AG, "medium", "high", "short", "medium", 1, ["tank_killer", "air_defense"], flying=True),
    "inferno-tower": building(AG, "medium", "high", "long", ["building", "tank_killer", "air_defense"]),
    "knight": troop(G, "medium", "medium", "melee", "medium", 1, ["mini_tank"]),
    "lava-hound": troop(B, "tank", "low", "short", "slow", 1, ["tank", "win_condition"], flying=True),
    "lightning": spell("large", 3.5, ["swarm", "light", "medium"], ["spell_large"]),
    "little-prince": troop(AG, "medium", "medium", "long", "medium", 1, ["support", "air_defense"]),  # guess
    "lumberjack": troop(G, "medium", "high", "melee", "very_fast", 1, ["tank_killer"]),
    "magic-archer": troop(AG, "light", "low", "long", "medium", 1, ["support", "air_defense"], splash=True),
    "mega-knight": troop(G, "tank", "medium", "melee", "medium", 1, ["tank", "splash_defense"], splash=True),
    "mega-minion": troop(AG, "medium", "medium", "melee", "medium", 1, ["air_defense", "tank_killer"], flying=True),
    "mighty-miner": troop(G, "medium", "medium", "melee", "medium", 1, ["mini_tank", "tank_killer"]),
    "miner": troop(G, "medium", "medium", "melee", "fast", 1, ["win_condition"]),
    "mini-p-e-k-k-a": troop(G, "medium", "high", "melee", "fast", 1, ["tank_killer"]),
    "minion-horde": troop(AG, "swarm", "high", "melee", "fast", 6, ["swarm", "air_defense"], flying=True),
    "minions": troop(AG, "swarm", "high", "melee", "fast", 3, ["swarm", "air_defense"], flying=True),
    "mirror": spell("utility", 0.0, [], []),
    "monk": troop(G, "medium", "medium", "melee", "medium", 1, ["mini_tank"]),
    "mortar": building(G, "medium", "low", "long", ["building"], splash=True),
    "mother-witch": troop(AG, "light", "medium", "long", "medium", 1, ["support", "air_defense"]),
    "musketeer": troop(AG, "medium", "medium", "long", "medium", 1, ["support", "air_defense"]),
    "night-witch": troop(G, "medium", "medium", "melee", "medium", 1, ["support"]),
    "p-e-k-k-a": troop(G, "tank", "high", "melee", "slow", 1, ["tank", "tank_killer"]),
    "phoenix": troop(AG, "medium", "medium", "melee", "medium", 1, ["air_defense", "support"], flying=True),
    "poison": spell("medium", 3.5, ["swarm", "light"], ["spell_medium"]),
    "prince": troop(G, "medium", "medium", "melee", "medium", 1, ["mini_tank"]),
    "princess": troop(AG, "light", "low", "long", "medium", 1, ["support", "bait"], splash=True),
    "rage": spell("utility", 3.0, [], []),
    "ram-rider": troop(B, "medium", "medium", "melee", "medium", 1, ["win_condition"]),
    "rascals": troop(AG, "medium", "medium", "short", "medium", 3, ["swarm", "support"]),
    "rocket": spell("large", 2.0, ["swarm", "light", "medium"], ["spell_large"]),
    "ronin": troop(G, "medium", "high", "melee", "medium", 1, ["mini_tank"]),                        # guess
    "royal-delivery": spell("small", 3.0, ["swarm"], ["spell_small"]),
    "royal-ghost": troop(G, "medium", "medium", "melee", "fast", 1, ["support", "splash_defense"], splash=True),
    "royal-giant": troop(B, "tank", "medium", "long", "slow", 1, ["tank", "win_condition"]),
    "royal-hogs": troop(B, "medium", "medium", "melee", "very_fast", 4, ["win_condition"]),
    "royal-recruits": troop(G, "light", "high", "melee", "medium", 6, ["swarm"]),
    "rune-giant": troop(B, "tank", "low", "melee", "slow", 1, ["tank", "win_condition"]),            # guess
    "skeleton-army": troop(G, "swarm", "high", "melee", "fast", 15, ["swarm", "tank_killer"]),
    "skeleton-barrel": troop(B, "light", "low", "melee", "fast", 1, ["win_condition", "bait"], flying=True),
    "skeleton-dragons": troop(AG, "light", "medium", "short", "fast", 2, ["air_defense", "splash_defense"], flying=True, splash=True),
    "skeleton-king": troop(G, "medium", "medium", "melee", "medium", 1, ["mini_tank", "splash_defense"], splash=True),
    "skeletons": troop(G, "swarm", "medium", "melee", "fast", 3, ["swarm", "cycle"]),
    "sparky": troop(G, "medium", "high", "long", "slow", 1, ["tank_killer", "splash_defense"], splash=True),
    "spear-goblins": troop(AG, "swarm", "medium", "long", "very_fast", 3, ["swarm", "cycle", "air_defense"]),
    "spirit-empress": troop(AG, "medium", "medium", "long", "medium", 1, ["support"]),               # guess
    "suspicious-bush": troop(G, "light", "low", "melee", "medium", 1, ["cycle", "bait"]),            # guess
    "tesla": building(AG, "medium", "medium", "long", ["building", "air_defense"]),
    "the-log": spell("small", 2.0, ["swarm"], ["spell_small"]),
    "three-musketeers": troop(AG, "medium", "high", "long", "medium", 3, ["support", "air_defense"]),
    "tombstone": building(G, "light", "low", "melee", ["building"]),
    "tornado": spell("utility", 5.5, [], []),
    "valkyrie": troop(G, "medium", "medium", "melee", "medium", 1, ["mini_tank", "splash_defense"], splash=True),
    "vines": spell("utility", 2.5, [], []),                                                          # guess
    "void": spell("medium", 2.5, ["swarm", "light"], ["spell_medium"]),                              # guess
    "wall-breakers": troop(B, "light", "low", "melee", "very_fast", 2, ["win_condition", "kamikaze"], splash=True),
    "witch": troop(AG, "medium", "medium", "long", "medium", 1, ["support", "splash_defense", "air_defense"], splash=True),
    "wizard": troop(AG, "medium", "medium", "long", "medium", 1, ["splash_defense", "air_defense"], splash=True),
    "x-bow": building(G, "medium", "medium", "long", ["building"]),
    "zap": spell("small", 2.5, ["swarm"], ["spell_small"]),
    "zappies": troop(AG, "light", "medium", "short", "medium", 3, ["support", "air_defense"]),
}

# (card, field) pairs where the manual value beats the stats: the stats describe only the card's first unit or a
# single hit, and the game behaviour differs. Printed as "kept manual", not as disagreements.
MANUAL_WINS: set[tuple[str, str]] = {
    ("electro-dragon", "splash"),      # chain lightning hits up to three targets
    ("firecracker", "splash"),         # the shot bursts into a small area
    ("magic-archer", "splash"),        # the arrow pierces in a line, which is what matters against swarms
    ("goblin-gang", "targets"),        # half the gang are spear goblins, which hit air
    ("goblin-gang", "range"),          # mixed melee goblins and ranged spear goblins
    ("rascals", "targets"),            # the two girls are ranged and hit air
    ("rascals", "range"),
    ("furnace", "targets"),            # the furnace itself does not attack; the fire spirits it spawns hit air and splash
    ("furnace", "range"),
    ("furnace", "splash"),
    ("goblin-hut", "targets"),         # likewise: the spear goblins it spawns hit air from range
    ("goblin-hut", "range"),
}


# --------------------------------------------------------------------------------------------- stats loading

def fetch_stats(stats_dir: Path) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    for name in STATS_FILES:
        with urllib.request.urlopen(BASE_URL + name, timeout=60) as resp:   # public data, no credentials
            (stats_dir / name).write_bytes(resp.read())
        print(f"fetched {name} -> {stats_dir / name}")


def load_stats(stats_dir: Path) -> dict | None:
    """The cr-api-data tables indexed for lookup, or None when the files are missing."""
    stats_path, cards_path = stats_dir / "cards_stats.json", stats_dir / "cards.json"
    if not stats_path.exists():
        return None
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    cards = json.loads(cards_path.read_text(encoding="utf-8")) if cards_path.exists() else []
    sections: dict[str, list[dict]] = {k: raw.get(k) or [] for k in ("troop", "building", "spell")}
    return {
        "cards": {c["key"]: c for c in cards if c.get("key")},
        "sections": sections,
        "characters": {c["name"]: c for c in raw.get("characters") or [] if c.get("name")},
        "projectiles": {p["name"]: p for p in raw.get("projectile") or [] if p.get("name")},
    }


def stats_entry(slug: str, stats: dict) -> tuple[dict | None, dict | None]:
    """(section entry, unit stats) for our slug: slug -> cr-api key (cards.json) -> troop/building/spell entry ->
    `characters` row named by its summon_character. A building's stats sit in its own entry when no such row exists."""
    key = KEY_ALIASES.get(slug, slug)
    if stats["cards"] and key not in stats["cards"]:
        return None, None
    sc_key = (stats["cards"].get(key) or {}).get("sc_key")
    entry = None
    for section in ("troop", "building", "spell"):
        for e in stats["sections"][section]:
            if e.get("key") == key:
                entry = e
                break
        if entry:
            break
    if entry is None and sc_key:                      # no section entry carries the key: match by sc_key instead
        entry = next((e for sec in stats["sections"].values() for e in sec if e.get("summon_character") == sc_key), None)
    if entry is None:
        return None, None
    unit = stats["characters"].get(entry.get("summon_character"))
    if unit is None and ("range" in entry or "hitpoints" in entry):
        unit = entry
    return entry, unit


def derive_stats(unit: dict, projectiles: dict, is_building: bool) -> dict:
    """The stats-derived fields for one unit, by the spec's mapping rules (splash also counts a projectile radius)."""
    out: dict = {}
    out["flying"] = (unit.get("flying_height") or 0) > 0
    if unit.get("target_only_buildings"):
        out["targets"] = "buildings"
    elif unit.get("attacks_air"):
        out["targets"] = "air_ground"
    else:
        out["targets"] = "ground"
    proj = projectiles.get(unit.get("projectile")) or {}
    out["splash"] = (unit.get("area_damage_radius") or 0) > 0 or (proj.get("radius") or 0) >= SPLASH_PROJECTILE_MIN
    rng = unit.get("range")
    if rng is not None:
        out["range"] = "melee" if rng <= MELEE_MAX_RANGE else "short" if rng < 5000 else "long"
    speed = unit.get("speed")
    if is_building:
        out["speed"] = None
    elif speed is not None:
        out["speed"] = "slow" if speed <= 45 else "medium" if speed <= 60 else "fast" if speed <= 90 else "very_fast"
    return out


def level_value(per_level: list | None, rarity: str | None, level: int = TOURNAMENT_LEVEL):
    if not per_level:
        return None
    i = level - RARITY_START_LEVEL.get(rarity or "", 1)
    return per_level[max(0, min(i, len(per_level) - 1))]


def hp_band(hp: float) -> str:
    return "swarm" if hp <= HP_BAND_SWARM else "light" if hp <= HP_BAND_LIGHT else "medium" if hp < HP_BAND_TANK else "tank"


def notes_for(slug: str, manual: dict, entry: dict, unit: dict | None, rarity: str | None) -> list[str]:
    """Non-overriding sanity checks of the manual-only fields (count, hp_class, spell radius) against the stats."""
    notes: list[str] = []
    if manual["type"] == "spell":
        radius = entry.get("radius")
        if radius and abs(radius / 1000 - manual["radius"]) > 0.01:
            notes.append(f"{slug}.radius: manual={manual['radius']} stats={radius / 1000}")
        return notes
    n = max(1, entry.get("summon_number") or 0) + (entry.get("summon_character_second_count") or 0)
    if manual["type"] == "troop" and manual["count"] != n:
        notes.append(f"{slug}.count: manual={manual['count']} stats={n}")
    if unit:
        hp = level_value(unit.get("hitpoints_per_level"), rarity) or unit.get("hitpoints")
        if hp and hp_band(hp) != manual["hp_class"]:
            notes.append(f"{slug}.hp_class: manual={manual['hp_class']} stats={hp_band(hp)} (level-{TOURNAMENT_LEVEL} hp {hp})")
    return notes


# ---------------------------------------------------------------------------------------------- the build

def build_roles(stats: dict | None, manual: dict[str, dict], cards: dict[str, dict],
                wins: set[tuple[str, str]] | None = None) -> tuple[dict, dict]:
    """({slug: role} sorted by slug, report). Stats-derived fields overwrite the manual ones (report lists them), except
    the (card, field) pairs in `wins` (default MANUAL_WINS), which keep the manual value (report "kept_manual")."""
    wins = MANUAL_WINS if wins is None else wins
    roles: dict[str, dict] = {}
    report = {"disagreements": [], "kept_manual": [], "notes": [], "stats": 0, "manual": 0, "no_card": []}
    for slug in sorted(manual):
        m = dict(manual[slug])
        entry, unit = stats_entry(slug, stats) if stats else (None, None)
        if m["type"] == "spell":
            unit = None                      # spells have no stats-derived fields; the entry only feeds the radius note
        derived = derive_stats(unit, stats["projectiles"], m["type"] == "building") if unit else {}
        for field, value in derived.items():
            if (slug, field) in wins:
                if m.get(field) != value:
                    report["kept_manual"].append(f"{slug}.{field}: manual={m.get(field)!r} stats={value!r}")
                continue
            if m.get(field) != value:
                report["disagreements"].append(f"{slug}.{field}: manual={m.get(field)!r} stats={value!r}")
            m[field] = value
        if entry:
            report["notes"].extend(notes_for(slug, m, entry, unit, (cards.get(slug) or {}).get("rarity")))
        if derived:
            report["stats"] += 1
        else:
            report["manual"] += 1
        if m["type"] == "spell":
            role = {"type": "spell", "spell_class": m["spell_class"], "radius": m["radius"], "kills": m["kills"],
                    "roles": m["roles"], "source": "manual"}
            if slug in GROUND_ONLY_SPELLS:
                role["hits_air"] = False
        else:
            role = {k: m[k] for k in ("type", "targets", "flying", "hp_class", "dps_class", "splash", "range", "speed", "count", "roles")}
            role["source"] = "stats+manual" if derived else "manual"
        roles[slug] = role
    report["no_card"] = sorted(s for s in manual if s not in cards)
    return roles, report


def check_manual(manual: dict[str, dict], wins: set[tuple[str, str]] | None = None) -> list[str]:
    """Vocabulary problems in the MANUAL table itself, and MANUAL_WINS entries that name no stats-derived manual field."""
    problems = [f"MANUAL_WINS {slug}.{field}: not a stats-derived field of a manual card"
                for slug, field in sorted(MANUAL_WINS if wins is None else wins)
                if field not in STATS_FIELDS or field not in manual.get(slug, {})]
    for slug, m in manual.items():
        for field, vocab in (("type", TYPES), ("targets", TARGETS), ("hp_class", HP_CLASSES), ("dps_class", DPS_CLASSES),
                             ("range", RANGES), ("speed", SPEEDS), ("spell_class", SPELL_CLASSES)):
            if m.get(field) is not None and m[field] not in vocab:
                problems.append(f"{slug}.{field}: {m[field]!r}")
        problems.extend(f"{slug}.roles: {r!r}" for r in m["roles"] if r not in ROLES)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    ap.add_argument("--out", type=Path, default=RESOURCE)
    ap.add_argument("--fetch", action="store_true", help="download cards_stats.json and cards.json into --stats-dir first")
    args = ap.parse_args(argv)

    if args.fetch:
        fetch_stats(args.stats_dir)
    stats = load_stats(args.stats_dir)
    if stats is None:
        print(f"warning: no stats in {args.stats_dir} (use --fetch); building from the manual table only", file=sys.stderr)
    problems = check_manual(MANUAL)
    cards = load_cards()
    roles, report = build_roles(stats, MANUAL, cards)
    problems += validate_roles(roles, cards)
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(f" {json.dumps(slug)}: {json.dumps(role, ensure_ascii=False)}" for slug, role in roles.items())
    args.out.write_text('{"version": 1, "cards": {\n' + body + "\n}}\n", encoding="utf-8")        # one line per card

    playable = sorted(s for s in cards if is_playable(s, cards))
    print(f"wrote {args.out}")
    print("coverage")
    print(f"  playable cards in the index : {len(playable)}")
    print(f"  cards in the role table     : {len(roles)}   (not playable in the index: {sorted(set(roles) - set(playable))})")
    print(f"  stats-derived               : {report['stats']}")
    print(f"  manual-only                 : {report['manual']}")
    print(f"  disagreements (stats win)   : {len(report['disagreements'])}")
    for line in report["disagreements"]:
        print("    " + line)
    print(f"  kept manual (MANUAL_WINS)   : {len(report['kept_manual'])}")
    for line in report["kept_manual"]:
        print("    " + line)
    print(f"  notes on manual-only fields : {len(report['notes'])}")
    for line in report["notes"]:
        print("    " + line)
    if report["no_card"]:
        print(f"  manual entries not in the card index: {report['no_card']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
