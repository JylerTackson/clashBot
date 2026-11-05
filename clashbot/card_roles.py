"""Card role table (MAC-Z1): what each playable card is for, as plain vocabularies the tactical engine can reason over.

The table is built by `tools/data/build_card_roles.py` (RoyaleAPI cr-api-data stats + a hand-written MANUAL table) and
committed as `resources/card_roles.json`. `load_roles()` returns it as {slug: role}; do not mutate the result (cached).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from clashbot.game_rules import is_playable

RESOURCE = Path(__file__).resolve().parent / "resources" / "card_roles.json"

TYPES = frozenset({"troop", "building", "spell"})
TARGETS = frozenset({"ground", "air_ground", "buildings"})
HP_CLASSES = frozenset({"swarm", "light", "medium", "tank"})
DPS_CLASSES = frozenset({"low", "medium", "high"})
RANGES = frozenset({"melee", "short", "long"})
SPEEDS = frozenset({"slow", "medium", "fast", "very_fast"})          # null for buildings
SPELL_CLASSES = frozenset({"small", "medium", "large", "utility"})
ROLES = frozenset({
    "win_condition", "tank", "mini_tank", "tank_killer", "splash_defense", "air_defense", "swarm", "building",
    "kamikaze", "support", "cycle", "spell_small", "spell_medium", "spell_large", "bait",
})

# spells that cannot hit air (the role dicts of these four also carry "hits_air": false)
GROUND_ONLY_SPELLS = frozenset({"the-log", "barbarian-barrel", "earthquake", "giant-snowball"})


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cards"]


def load_roles(path: Path | str = RESOURCE) -> dict[str, dict]:
    """{slug: role dict} from the committed JSON (cached per path)."""
    return _load(str(path))


def _check_enum(problems: list[str], slug: str, role: dict, field: str, vocab, required: bool, nullable: bool = False) -> None:
    value = role.get(field)
    if value is None:
        if required and not (nullable and field in role):
            problems.append(f"{slug}: missing {field}")
        return
    if value not in vocab:
        problems.append(f"{slug}: unknown {field} {value!r}")


def validate_roles(roles: dict[str, dict], cards: dict[str, dict]) -> list[str]:
    """Problems in `roles`: a missing playable card, an unknown vocabulary value, a troop/building without
    targets/hp_class, a spell without spell_class/radius. Empty list = valid."""
    problems: list[str] = []
    for slug in sorted(cards):
        if is_playable(slug, cards) and slug not in roles:
            problems.append(f"{slug}: missing from roles")
    for slug in sorted(roles):
        role = roles[slug]
        kind = role.get("type")
        if kind not in TYPES:
            problems.append(f"{slug}: unknown type {kind!r}")
            continue
        if kind == "spell":
            _check_enum(problems, slug, role, "spell_class", SPELL_CLASSES, required=True)
            radius = role.get("radius")
            if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius < 0:
                problems.append(f"{slug}: bad radius {radius!r}")
            for hp in role.get("kills") or []:
                if hp not in HP_CLASSES:
                    problems.append(f"{slug}: unknown kills value {hp!r}")
        else:
            _check_enum(problems, slug, role, "targets", TARGETS, required=True)
            _check_enum(problems, slug, role, "hp_class", HP_CLASSES, required=True)
            _check_enum(problems, slug, role, "dps_class", DPS_CLASSES, required=False)
            _check_enum(problems, slug, role, "range", RANGES, required=False)
            _check_enum(problems, slug, role, "speed", SPEEDS, required=False, nullable=True)
            if kind == "building" and role.get("speed") is not None:
                problems.append(f"{slug}: a building has no speed")
            count = role.get("count")
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
                problems.append(f"{slug}: bad count {count!r}")
        for r in role.get("roles") or []:
            if r not in ROLES:
                problems.append(f"{slug}: unknown role {r!r}")
    return problems


def hits_air(role: dict) -> bool:
    """Spells hit air except the-log / barbarian-barrel / earthquake / giant-snowball (their role carries
    "hits_air": false); everything else hits air iff `targets == "air_ground"`."""
    if role.get("type") == "spell":
        return role.get("hits_air", True) is not False
    return role.get("targets") == "air_ground"


def is_tank(role: dict) -> bool:
    return role.get("hp_class") == "tank" or "tank" in (role.get("roles") or [])


def is_win_condition(role: dict) -> bool:
    return "win_condition" in (role.get("roles") or [])
