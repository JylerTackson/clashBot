"""Build clashbot/resources/unit_stats.json (MAC-AH): exact per-level unit stats for the threat model.

  python tools/data/build_unit_stats.py [--kb-dir knowledge_base] [--stats-dir data/external/cr-api-data]
                                        [--out clashbot/resources/unit_stats.json] [--fetch]

Sources, in priority order (the output records which one filled each group in `source`):
  1. the knowledge base (`knowledge_base/cards|heroes|evolutions/*.md`): the attribute tables (hit speed, speed, range,
     target, count, transport, lifetime, radius ...) and the `## Stats by level` tables, which hold the exact per-level
     numbers (hitpoints, damage, death damage, crown tower damage ...). Column names vary per card ("Night Witch
     Hitpoints", "Area Damage", "Damage per second"); `column_key` maps them to (unit prefix, field).
  2. RoyalAPI cr-api-data (`cards_stats.json`, `cards.json`; `--fetch` downloads them), found through the same slug ->
     character mapping as `build_card_roles.py`: fills the fields a KB card lacks (per-level series, splash radius,
     spawn counts, death spawns) and the numbers of cards the KB parse could not read.
  3. `MANUAL_STATS` below, last.
The princess tower comes from `knowledge_base/cards/tower-princess.md` (level 15: 4,424 HP); the king tower is derived
(see `build_towers`). Units: 1,000 game units per tile; speed 60 = 1 tile per second (slow 45, medium 60, fast 90, very
fast 120); hit speeds in seconds. Series are stored as {"l0": first level, "v": [values]}; `clashbot.unit_stats` reads
them and extrapolates beyond the table with x1.1 per level.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_card_roles as bcr  # noqa: E402  (slug -> cr-api mapping: load_stats, stats_entry)
from clashbot.game_rules import load_cards  # noqa: E402

KB_DIR = ROOT / "knowledge_base"
DEFAULT_OUT = ROOT / "clashbot" / "resources" / "unit_stats.json"
UNITS_PER_TILE = 1000.0
SPEED_PER_TILE_S = 60.0                 # a speed of 60 game units is one tile per second
LEVEL_GROWTH = 1.1                      # per level, used only beyond a table
RARITY_START_LEVEL = {"Common": 1, "Rare": 3, "Epic": 6, "Legendary": 9, "Champion": 11}
TOWER_LEVEL = 15
PRINCESS_L15_HP = 4424                  # observed in our screenshots; the KB tower table must reproduce it
KING_ANCHORS = {11: 4824, 16: 7728}     # king tower HP seen in the recorded pro matches (tournament standard, level 16)

# unit tables the KB lists under "<name> Attributes" that are abilities / effects, not units
NOT_UNITS = {"charge", "dash", "dashing dash", "cloaking cape", "kamikaze goblin demolisher", "rocket launcher",
             "broken cannon cart", "lightning link", "jump", "zap", "zap pack", "enchant", "hook", "death nova",
             "spawn damage", "healing", "active healing", "spawn healing", "getaway grenade", "royal rescue",
             "soul summoning", "explosive escape", "pensive protection", "cooking", "ranged attack", "melee attack",
             "evolution", "dagger", "sound effects", "generic", "unique", "current", "removed", "unused", "rage"}
DEATH_TABLES = {"bomb", "skeleton container"}      # death bomb: splash radius + fuse time of the main unit

# Fallback numbers for cards neither the KB nor cr-api describe well (best game knowledge, `source: manual`). Empty
# when both sources cover every card; add {slug: {"units": [{...unit fields...}]}} entries here if the coverage report
# lists a card.
MANUAL_STATS: dict[str, dict] = {}

# the variants of our deck (heroes / evolutions), by (slug, kind)
VARIANTS = (("musketeer", "hero"), ("tesla", "evo"), ("electro-dragon", "evo"))


# ------------------------------------------------------------------------------------------------- markdown parsing
def frontmatter(text: str) -> dict:
    out: dict = {}
    if not text.startswith("---"):
        return out
    for line in text.split("---", 2)[1].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def md_tables(lines: list[str]) -> list[tuple[str, list[str], list[list[str]]]]:
    """[(bold title above the table, header cells, rows)] of the markdown tables in `lines`."""
    res = []
    title = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("**") and line.endswith("**") and not line.startswith("|"):
            title = line.strip("*").strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            hdr = [c.strip() for c in line.strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            res.append((title, hdr, rows))
            i = j
            continue
        i += 1
    return res


def norm(text: str) -> str:
    """Name key: lowercase letters and digits only, a trailing plural s dropped ('Fire Spirits' = 'fire spirit')."""
    s = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    return s[:-1] if s.endswith("s") and len(s) > 3 else s


def num(cell) -> float | None:
    """The first number in a cell ('1,400' -> 1400, '119 x3' -> 119, '2 sec' -> 2); None for 'N/A' and text."""
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(cell or ""))
    if not m:
        return None
    return float(m.group(0).replace(",", ""))


def speed_tiles(cell) -> float | None:
    m = re.search(r"\((\d+)\)", str(cell or ""))
    v = float(m.group(1)) if m else num(cell)
    return None if v is None else round(v / SPEED_PER_TILE_S, 3)


def range_tiles(cell) -> float | None:
    m = re.search(r"\((\d+(?:\.\d+)?)\)", str(cell or ""))
    return float(m.group(1)) if m else num(cell)


def targets_of(cell) -> str | None:
    t = str(cell or "").lower()
    if not t or t.startswith("n/a"):
        return None
    if "building" in t:
        return "buildings"
    if "air" in t:
        return "air_ground" if "ground" in t else "air"
    if "ground" in t:
        return "ground"
    return None


def damage_num(cell) -> float | None:
    """A damage cell: '119 x3' -> 119 (per target), '25 x5 (125)' -> 125 (the total one target takes), '1,007' -> 1007."""
    text = str(cell or "")
    m = re.search(r"x\s*\d+\s*\((\d[\d,]*(?:\.\d+)?)\)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    return num(text)


def count_of(cell) -> int | None:
    v = num(cell)
    return None if v is None else int(v)


# the stats-table columns: (regex over the lowercase column name without parentheses, field). The first group is the
# unit prefix ("night witch hitpoints" -> prefix "night witch").
COLUMN_PATTERNS = (
    (r"^(.*?)\s*hitpoints lost per second$", "decay"),
    (r"^(.*?)\s*shield hitpoints$", "shield"),
    (r"^(.*?)\s*hitpoints$", "hp"),
    (r"^(.*?)\s*death damage$", "death_damage"),
    (r"^(.*?)\s*crown tower damage$", "ct_damage"),
    (r"^(.*?)\s*damage per second.*$", "dps"),
    (r"^(.*?)\s*damage$", "damage"),
)


def column_key(col: str) -> tuple[str, str]:
    """(unit prefix, field) of a stats column; the field is 'other' (prefix = the whole name) when unknown."""
    c = re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", col).strip().lower())
    c = re.sub(r"\barea\b\s*", "", c).strip()                 # "Area Damage" is a damage column of the unit / spell
    for pat, field in COLUMN_PATTERNS:
        m = re.match(pat, c)
        if m:
            return m.group(1).strip(), field
    return c, "other"


def make_series(levels: list[int], values: list[float | None]) -> dict | None:
    """{"l0": first level, "v": [..]} from (level, value) pairs sorted by level; gaps are filled geometrically, a
    non-increasing step in the middle of a rising series (a typo in the source) is repaired. None when no value."""
    pairs = sorted((lv, v) for lv, v in zip(levels, values) if v is not None)
    if not pairs:
        return None
    l0, lmax = pairs[0][0], pairs[-1][0]
    by_level = dict(pairs)
    out: list[float] = []
    for lv in range(l0, lmax + 1):
        if lv in by_level:
            out.append(by_level[lv])
        else:                                                         # a missing row: x1.1 from the previous one
            out.append(out[-1] * LEVEL_GROWTH)
    return {"l0": l0, "v": [int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 2) for v in out]}


def repair_series(s: dict | None, notes: list[str], what: str) -> dict | None:
    """Fix a value that falls out of a rising series (source typos such as a 721 between 1396 and 1536, or a 1,906
    between 824 and 997): replaced by the geometric mean of its neighbours. Notes are appended to `notes`."""
    if not s or len(s["v"]) < 3:
        return s
    v = s["v"]
    for i in range(1, len(v) - 1):
        if v[i - 1] < v[i + 1] and (v[i] < v[i - 1] or v[i] > v[i + 1]) and v[i] > 0:     # a dip or a spike in a rising series
            fixed = int(round((v[i - 1] * v[i + 1]) ** 0.5))
            notes.append(f"{what}: level {s['l0'] + i} value {v[i]} -> {fixed}")
            v[i] = fixed
    return s


# ------------------------------------------------------------------------------------------------- one KB card file
def table_row(header: list[str], row: list[str]) -> dict[str, str]:
    return {h.strip().lower(): (row[i] if i < len(row) else "") for i, h in enumerate(header)}


def title_name(title: str) -> str:
    t = re.sub(r"\s*attributes$", "", title.strip(), flags=re.I).strip().lower()
    return t


def unit_from_row(name: str, d: dict[str, str]) -> dict:
    return {
        "name": name,
        "count": count_of(d.get("count")),
        "hit_speed_s": num(d.get("hit speed") or d.get("hit speed (stage 1)") or d.get("attack period")),
        "first_hit_s": num(d.get("first hit speed") or d.get("first attack period")),
        "speed_tiles_s": speed_tiles(d.get("speed")),
        "range_tiles": range_tiles(d.get("range")),
        "targets": targets_of(d.get("target")),
        "flying": (d.get("transport") or "").strip().lower() == "air" if "transport" in d else None,
        "splash_radius_tiles": num(d.get("splash radius")),
        "death_radius_tiles": num(d.get("death damage splash radius")),
        "deploy_s": num(d.get("deploy time")),
        "lifetime_s": num(d.get("lifetime")),
        "spawn_every_s": num(d.get("spawn speed")),
    }


def parse_card_text(text: str, slug: str, kind: str = "card") -> dict:
    """The raw KB parse of one card / hero / evolution file: {"meta", "units", "series", "extras", "spell", "notes"}.
    `units` are dicts from the attribute tables (series not attached yet); `series` is {unit key: {field: series}}."""
    fm = frontmatter(text)
    secs = sections(text)
    name = fm.get("name") or slug
    card_type = fm.get("card_type") or ""
    if kind == "evo":
        attr_lines = stat_lines = secs.get("Evolution-specific stats/behavior", [])
    else:
        attr_lines, stat_lines = secs.get("Attributes", []), secs.get("Stats by level", [])
    attr_tables = [t for t in md_tables(attr_lines) if "level" not in [h.lower() for h in t[1]]]
    stat_tables = [t for t in md_tables(stat_lines) if "level" in [h.lower() for h in t[1]]]

    meta: dict = {"name": name, "slug": slug, "rarity": fm.get("rarity"), "card_type": card_type,
                  "elixir": num(fm.get("elixir_cost")), "kb_targets": fm.get("targets")}
    units: list[dict] = []
    card_attr: dict[str, str] = {}
    death_attr: dict[str, str] = {}
    extras_attr: dict[str, dict] = {}
    for title, header, rows in attr_tables:
        if not rows:
            continue
        d = table_row(header, rows[0])
        tname = title_name(title)
        hset = set(d)
        if tname in DEATH_TABLES:
            death_attr = d
            continue
        if tname in NOT_UNITS:
            extras_attr[tname] = d
            continue
        is_unit = bool(hset & {"hit speed", "hit speed (stage 1)", "attack period", "speed"}) or (
            "range" in hset and "cost" not in hset)
        if "cost" in hset and not card_attr:
            card_attr = d
        if is_unit and not (kind == "card" and card_type == "Spell" and "radius" in hset and "speed" not in hset):
            unit_name = name if tname in ("", "attributes", "card") else re.sub(r"\s*attributes$", "", title.strip(), flags=re.I)
            units.append(unit_from_row(unit_name, d))
            if "cost" in hset:
                card_attr = d
        elif "cost" in hset or not card_attr:
            card_attr = d if "cost" in hset else card_attr
    meta["card_attr"] = card_attr
    meta["death_attr"] = death_attr
    meta["extras_attr"] = extras_attr

    # the main unit: the one named like the card, else the first; a building / hut without a unit table gets one
    main_norm = norm(name)
    main_idx = next((i for i, u in enumerate(units) if norm(u["name"]) == main_norm), 0 if units else None)
    if main_idx is not None and main_idx != 0:
        units.insert(0, units.pop(main_idx))
    if card_type in ("Building",) and not units:
        u = unit_from_row(name, card_attr)
        units.append(u)
    # the card's own attribute row also carries lifetime / deploy / spawn speed (a hut or a card table with units)
    if units and card_attr:
        for k, src in (("deploy_s", "deploy time"), ("lifetime_s", "lifetime"), ("spawn_every_s", "spawn speed")):
            if units[0].get(k) is None and num(card_attr.get(src)) is not None:
                units[0][k] = num(card_attr.get(src))
    if units and death_attr:
        if units[0].get("death_radius_tiles") is None:
            units[0]["death_radius_tiles"] = num(death_attr.get("death damage splash radius"))
        units[0]["death_delay_s"] = num(death_attr.get("deploy time"))

    # the stats tables: (unit key -> field -> series), the spell's own damage, other columns kept as extras
    keys = {norm(u["name"]): i for i, u in enumerate(units)}
    series: dict[int, dict[str, dict]] = {}
    spell: dict[str, dict] = {}
    extras: dict[str, dict] = {}
    notes: list[str] = []
    first_unit_damage_to_spell = card_type == "Spell"
    for _, header, rows in stat_tables:
        lv_col = next(i for i, h in enumerate(header) if h.strip().lower() == "level")
        levels, ok_rows = [], []
        for r in rows:
            lv = num(r[lv_col]) if lv_col < len(r) else None
            if lv is not None and float(lv).is_integer():
                levels.append(int(lv))
                ok_rows.append(r)
        if not levels:
            continue
        for ci, h in enumerate(header):
            if ci == lv_col:
                continue
            prefix, field = column_key(h)
            if field == "decay":
                continue
            to_val = damage_num if field in ("damage", "death_damage", "ct_damage") else num
            values = [to_val(r[ci]) if ci < len(r) else None for r in ok_rows]
            s = repair_series(make_series(levels, values), notes, f"{slug} {h}")
            if s is None:
                continue
            pn = norm(prefix)
            ui = keys.get(pn) if pn else None
            if ui is None and pn:                          # 'Bush Hitpoints' of the Suspicious Bush: a part of the name
                ui = next((i for k, i in keys.items() if k.endswith(pn) and len(pn) >= 4), None)
            if ui is None and (pn == main_norm or not pn):
                ui = 0 if units else None
            if card_type == "Spell":
                area = "area" in h.lower()
                if field in ("damage", "ct_damage") and (area or pn == main_norm or not units
                                                         or (not pn and field == "ct_damage")):
                    spell["damage" if field == "damage" else "crown_tower_damage"] = s
                    continue
                if field == "ct_damage" and ui is None:
                    spell["crown_tower_damage"] = s
                    continue
            if ui is not None and field in ("hp", "damage", "death_damage", "shield", "ct_damage", "dps"):
                series.setdefault(ui, {}).setdefault(field, s)
            else:
                extras.setdefault(re.sub(r"\s+", " ", h.strip().lower()), s)
    return {"meta": meta, "units": units, "series": series, "spell": spell, "extras": extras, "notes": notes}


# ------------------------------------------------------------------------------------------------- cr-api fallback
def crapi_units(slug: str, stats: dict | None) -> list[dict]:
    """The cr-api characters of a card as fallback unit dicts [{"name", "count", "char", "entry"}], main first."""
    if not stats:
        return []
    entry, unit = bcr.stats_entry(slug, stats)
    if entry is None:
        return []
    chars = stats["characters"]
    out = []
    if unit is not None:
        out.append({"name": entry.get("summon_character") or entry.get("name"), "char": unit, "entry": entry,
                    "count": max(1, entry.get("summon_number") or 0)})
    second = chars.get(entry.get("summon_character_second"))
    if second is not None:
        out.append({"name": entry.get("summon_character_second"), "char": second, "entry": entry,
                    "count": entry.get("summon_character_second_count") or 1})
    for key in ("spawn_character", "death_spawn_character"):
        for host in list(out):
            ch = chars.get(host["char"].get(key))
            if ch is not None and all(o["name"] != ch["name"] for o in out):
                out.append({"name": ch["name"], "char": ch, "entry": entry, "count": 0})
    return out


def cr_series(char: dict, field: str, rarity: str | None, projectiles: dict) -> dict | None:
    arr = char.get(field + "_per_level")
    if not arr and field == "damage":
        proj = projectiles.get(char.get("projectile")) or {}
        arr = proj.get("damage_per_level")
    if not arr:
        return None
    l0 = RARITY_START_LEVEL.get(char.get("rarity") or rarity or "", 1)
    return {"l0": l0, "v": list(arr)}


def fill_from_crapi(unit: dict, card_slug: str, cr: list[dict], rarity: str | None, projectiles: dict,
                    source: dict[str, str], unit_index: int) -> None:
    """Fill the None fields of `unit` from the cr-api character that matches it (main unit: the first character)."""
    match = None
    for c in cr:
        # same name, or a part of the character's ('Rider' is the cr-api 'RamRider')
        if norm(c["name"]) == norm(unit["name"]) or (unit_index > 0 and norm(c["name"]).endswith(norm(unit["name"]))):
            match = c
            break
    if match is None and unit_index == 0 and cr:
        match = cr[0]
    if match is None:
        return
    ch = match["char"]
    proj = projectiles.get(ch.get("projectile")) or {}

    def put(key, value, group):
        if unit.get(key) is None and value not in (None, 0):
            unit[key] = value
            source.setdefault(group, "cr-api")

    put("hit_speed_s", (ch.get("hit_speed") or 0) / 1000.0, "attrs")
    put("first_hit_s", (ch.get("load_time") or 0) / 1000.0, "attrs")
    put("speed_tiles_s", round((ch.get("speed") or 0) / SPEED_PER_TILE_S, 3), "attrs")
    put("range_tiles", (ch.get("range") or 0) / UNITS_PER_TILE, "attrs")
    if unit.get("targets") is None:
        unit["targets"] = ("buildings" if ch.get("target_only_buildings") else
                           "air_ground" if ch.get("attacks_air") else "ground")
        source.setdefault("attrs", "cr-api")
    if unit.get("flying") is None:
        unit["flying"] = (ch.get("flying_height") or 0) > 0
    splash = (ch.get("area_damage_radius") or 0) / UNITS_PER_TILE or (proj.get("radius") or 0) / UNITS_PER_TILE
    put("splash_radius_tiles", splash, "attrs")
    put("death_radius_tiles", (ch.get("death_damage_radius") or 0) / UNITS_PER_TILE, "attrs")
    put("lifetime_s", (ch.get("life_time") or 0) / 1000.0, "attrs")
    if unit.get("count") is None:
        unit["count"] = match["count"]
    for fld in ("hp", "damage", "death_damage"):
        if unit.get("series_" + fld) is None:
            key = {"hp": "hitpoints", "damage": "damage", "death_damage": "death_damage"}[fld]
            s = cr_series(ch, key, rarity, projectiles) if fld != "death_damage" else None
            if s is not None:
                unit["series_" + fld] = s
                source.setdefault("series", "cr-api")
    # spawns: the character spawns / death-spawns another unit
    if ch.get("spawn_character") and not unit.get("spawn"):
        unit["_cr_spawn"] = {"unit": ch["spawn_character"], "count": max(1, ch.get("spawn_number") or 1),
                             "start_s": (ch.get("spawn_start_time") or 0) / 1000.0,
                             "every_s": (ch.get("spawn_interval") or 0) / 1000.0}
    if ch.get("death_spawn_character"):
        unit["_cr_death_spawn"] = {"unit": ch["death_spawn_character"], "count": max(1, ch.get("death_spawn_count") or 1)}
    if ch.get("death_damage") and unit.get("death_damage_base") is None:
        unit["death_damage_base"] = ch["death_damage"]
    if ch.get("kamikaze"):                         # a spirit: it explodes on its first hit and is gone
        unit["kamikaze"] = True


# ------------------------------------------------------------------------------------------------- the card record
def check_hit_speed(u: dict, notes: list[str], slug: str) -> None:
    """The KB's "Hit Speed" is sometimes only the swing (an executioner's axe takes 1.5 s more to come back): when the
    table's damage and damage-per-second imply another period (> 25% off, or no hit speed at all), the implied one
    is used."""
    dmg, dps = u.get("series_damage"), u.get("series_dps")
    if not dmg or not dps:
        return
    level = min(dmg["l0"] + len(dmg["v"]), dps["l0"] + len(dps["v"])) - 1
    d, ps = series_value(dmg, level), series_value(dps, level)
    if not d or not ps:
        return
    implied = round(d / ps * 20) / 20
    hs = u.get("hit_speed_s")
    if 0.2 <= implied <= 6.0 and (hs is None or abs(implied / hs - 1.0) > 0.25):
        notes.append(f"{slug} {u['name']}: hit speed {hs} -> {implied} (damage {d} / dps {ps})")
        u["hit_speed_s"] = implied


def build_card(slug: str, kb: dict, cr: list[dict], stats: dict | None, card: dict) -> tuple[dict, dict]:
    """(record, report) of one card from its raw KB parse `kb`, the cr-api fallback characters `cr` and the card index."""
    meta = kb["meta"]
    rarity = meta.get("rarity") or card.get("rarity")
    ctype = {"Building": "building", "Spell": "spell"}.get(meta.get("card_type") or card.get("card_type") or "", "troop")
    projectiles = stats["projectiles"] if stats else {}
    source: dict[str, str] = {}
    units = [dict(u) for u in kb["units"]]
    if not units and cr and ctype != "spell":
        units = [{"name": cr[0]["name"]}]                    # the KB parse found no unit table: the cr-api character
        source["units"] = "cr-api"
    for i, u in enumerate(units):
        for fld, s in (kb["series"].get(i) or {}).items():
            u["series_" + fld] = s
        if any(u.get("series_" + f) for f in ("hp", "damage")):
            source.setdefault("series", "kb")
        fill_from_crapi(u, slug, cr, rarity, projectiles, source, i)
        check_hit_speed(u, kb["notes"], slug)
    source.setdefault("attrs", "kb")
    # spawn links: a unit with a Spawn Speed spawns the first unit that has no initial count
    spawned = [u for u in units[1:] if u.get("count") in (None, 0)]
    out_units = []
    for i, u in enumerate(units):
        spawn = None
        if u.get("spawn_every_s") and (spawned or u.get("_cr_spawn")):
            target = spawned[0]["name"] if spawned else u["_cr_spawn"]["unit"]
            cnt = (u.get("_cr_spawn") or {}).get("count") or 1
            if ctype == "building":
                cnt = 1
            spawn = {"unit": target, "every_s": u["spawn_every_s"], "count": cnt,
                     "start_s": (u.get("_cr_spawn") or {}).get("start_s") or 1.0}
        death_spawn = None
        ds = u.get("_cr_death_spawn")
        if ds and any(norm(o["name"]) == norm(ds["unit"]) for o in units):
            death_spawn = {"unit": next(o["name"] for o in units if norm(o["name"]) == norm(ds["unit"])),
                           "count": ds["count"]}
        initial = u.get("count")
        if initial is None:
            initial = 0 if (i > 0 and any(u["name"] == s["name"] for s in spawned)) else 1
        hs = u.get("hit_speed_s")
        out_units.append({
            "name": u["name"], "count": int(initial),
            "hp": u.get("series_hp"), "damage": u.get("series_damage"),
            "hit_speed_s": hs, "first_hit_s": u.get("first_hit_s"), "range_tiles": u.get("range_tiles"),
            "speed_tiles_s": u.get("speed_tiles_s"), "targets": u.get("targets") or "ground",
            "flying": bool(u.get("flying")), "kamikaze": bool(u.get("kamikaze")),
            "splash_radius_tiles": u.get("splash_radius_tiles") or 0.0,
            "death_damage": u.get("series_death_damage"), "death_radius_tiles": u.get("death_radius_tiles") or 0.0,
            "death_delay_s": u.get("death_delay_s"),
            "death_spawn": death_spawn, "spawn": spawn,
            "deploy_s": u.get("deploy_s"), "lifetime_s": u.get("lifetime_s"),
            "shield": u.get("series_shield"),
        })
    rec: dict = {"type": ctype, "rarity": rarity, "elixir": meta.get("elixir") if meta.get("elixir") is not None
                 else card.get("elixir_cost"), "start_level": RARITY_START_LEVEL.get(rarity or "", 1),
                 "units": out_units, "source": source}
    if ctype == "building" and out_units:
        rec["lifetime_s"] = out_units[0].get("lifetime_s")
    if ctype == "spell" or kb["spell"]:
        d = meta["card_attr"]
        rec["spell"] = {"damage": kb["spell"].get("damage"), "crown_tower_damage": kb["spell"].get("crown_tower_damage"),
                        "radius_tiles": num(d.get("radius")), "duration_s": num(d.get("duration")),
                        "stun_s": num(d.get("stun duration"))}
        if rec["spell"]["damage"] is not None:
            source["spell"] = "kb"
    if kb["extras"]:
        rec["extras"] = kb["extras"]
    return rec, {"notes": kb["notes"]}


def apply_manual(cards: dict[str, dict]) -> None:
    for slug, m in MANUAL_STATS.items():
        rec = cards.setdefault(slug, {"type": "troop", "units": [], "source": {}})
        rec.update({k: v for k, v in m.items() if k != "units"})
        if m.get("units"):
            rec["units"] = m["units"]
        rec.setdefault("source", {})["manual"] = "manual"


# ------------------------------------------------------------------------------------------------- variants, towers
def parse_variant(kb_dir: Path, slug: str, kind: str) -> dict | None:
    """The hero / evolution record of a card: the series that differ from the base card, and the ability numbers."""
    folder, suffix = ("heroes", "hero") if kind == "hero" else ("evolutions", "evolution")
    path = kb_dir / folder / f"{slug}-{suffix}.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    raw = parse_card_text(text, slug, "evo" if kind == "evo" else "hero")
    out: dict = {"kind": kind, "source": "kb"}
    series: dict = {}
    if raw["series"].get(0):
        series = {k: v for k, v in raw["series"][0].items()}
    out["series"] = series
    abilities: dict = {}
    out["_notes"] = raw["notes"]
    if kind == "hero":
        # "Trusty Turret": cost, turret range / lifetime and per-level turret numbers
        sec = sections(text)
        for title, header, rows in md_tables(sec.get("Attributes", [])):
            if "turret range" in [h.lower() for h in header] and rows:
                d = table_row(header, rows[0])
                turret = {"cost": num(d.get("cost")), "range_tiles": num(d.get("turret range")),
                          "targets": targets_of(d.get("turret target")), "lifetime_s": num(d.get("turret lifetime")),
                          "deploy_s": num(d.get("deploy time"))}
                for k, s in raw["extras"].items():
                    if k.startswith("turret "):
                        turret[k[len("turret "):].replace(" ", "_")] = s
                abilities["trusty-turret"] = {"name": "Trusty Turret", "cost": turret["cost"], "turret": turret}
    if kind == "evo":
        evo_attr = next((d for t, d in raw["meta"]["extras_attr"].items() if t == "evolution"), {})
        abilities["evolution"] = {"cycles": num(evo_attr.get("cycles")), "pulse_radius_tiles": num(evo_attr.get("pulse radius")),
                                  "pulse_stun_s": num(evo_attr.get("pulse stun duration"))}
        for k, s in raw["extras"].items():
            abilities.setdefault("series", {})[k.replace(" ", "_")] = s
    out["abilities"] = abilities
    return out


def build_towers(kb_dir: Path, stats: dict | None) -> tuple[dict, list[str]]:
    """The tower entries. Princess: the exact per-level table of `cards/tower-princess.md` (hit speed, range from its
    attribute row). King: the same damage curve; hitpoints geometric between the two levels seen in the pro matches
    (11: 4,824, 16: 7,728), hit speed 1.0 s, range 7.0 (cr-api), because the KB has no king-tower table."""
    msgs: list[str] = []
    path = kb_dir / "cards" / "tower-princess.md"
    raw = parse_card_text(path.read_text(encoding="utf-8"), "tower-princess") if path.exists() else None
    s0 = (raw["series"].get(0) if raw else None) or {}
    hp, dmg = s0.get("hp"), s0.get("damage")
    d = (raw["meta"]["card_attr"] if raw else {}) or {}
    hit = num(d.get("attack period")) or 0.8
    rng = num(d.get("range")) or 7.5
    if hp is None:                                                # no KB file: cr-api tower numbers scaled to the anchor
        msgs.append("tower-princess.md missing: using the observed level-15 anchor only")
        l0 = 1
        base = [PRINCESS_L15_HP / (LEVEL_GROWTH ** (TOWER_LEVEL - 1 - i)) for i in range(16)]
        hp = {"l0": l0, "v": [int(round(v)) for v in base]}
        dmg = {"l0": l0, "v": [int(round(v / 28.0)) for v in hp["v"]]}
    princess = {"hp": hp, "damage": dmg, "hit_speed_s": hit, "first_hit_s": hit, "range_tiles": rng,
                "source": "kb:tower-princess" if raw else "anchor"}
    # king: geometric growth between the anchors, extended at the same rate
    (la, va), (lb, vb) = sorted(KING_ANCHORS.items())
    g = (vb / va) ** (1.0 / (lb - la))
    l_first, l_last = hp["l0"], hp["l0"] + len(hp["v"]) - 1
    king_hp = {"l0": l_first, "v": [int(round(va * g ** (lv - la))) for lv in range(l_first, l_last + 1)]}
    king = {"hp": king_hp, "damage": dmg, "hit_speed_s": 1.0, "first_hit_s": 0.5, "range_tiles": 7.0,
            "source": "derived: pro-match anchors 11:4824, 16:7728; damage curve of the princess tower"}
    got = princess["hp"]["v"][TOWER_LEVEL - princess["hp"]["l0"]]
    msgs.append(f"princess tower level {TOWER_LEVEL}: {got} HP (observed {PRINCESS_L15_HP})"
                + ("" if got == PRINCESS_L15_HP else "  <-- MISMATCH: the tower table no longer reproduces the screenshots"))
    return {"princess": princess, "king": king}, msgs


# ------------------------------------------------------------------------------------------------- the build
def build(kb_dir: Path, stats: dict | None, cards: dict[str, dict]) -> tuple[dict, dict]:
    records: dict[str, dict] = {}
    report: dict = {"no_kb": [], "no_hp": [], "no_damage": [], "notes": [], "cr_only": [], "counts": {}}
    for path in sorted((kb_dir / "cards").glob("*.md")):
        slug = path.stem
        if slug not in cards:
            continue
        kb = parse_card_text(path.read_text(encoding="utf-8"), slug)
        cr = crapi_units(slug, stats)
        rec, rep = build_card(slug, kb, cr, stats, cards[slug])
        records[slug] = rec
        report["notes"].extend(rep["notes"])
        if rec["type"] != "spell":
            if not rec["units"]:
                report["no_kb"].append(slug)
            elif not any(u.get("hp") for u in rec["units"]):
                report["no_hp"].append(slug)
            if rec["type"] == "troop" and not any(u.get("damage") for u in rec["units"]):
                report["no_damage"].append(slug)
        elif not (rec.get("spell") or {}).get("damage") and not any(u.get("damage") for u in rec["units"]):
            report["no_damage"].append(slug + " (spell)")
        if rec["source"].get("series") == "cr-api" or rec["source"].get("units") == "cr-api":
            report["cr_only"].append(slug)
    for slug, kind in VARIANTS:
        v = parse_variant(kb_dir, slug, kind)
        if v and slug in records:
            report["notes"].extend(f"{slug} {kind}: {n}" for n in v.pop("_notes", []))
            records[slug].setdefault("variants", {})[kind] = v
    apply_manual(records)
    towers, tmsgs = build_towers(kb_dir, stats)
    report["notes"].extend(tmsgs)
    return {"version": 1, "units_per_tile": UNITS_PER_TILE, "speed_per_tile_s": SPEED_PER_TILE_S,
            "level_growth": LEVEL_GROWTH, "tower": towers, "cards": records}, report


def dump(data: dict) -> str:
    """One line per card, so a change shows as a small diff."""
    head = {k: v for k, v in data.items() if k not in ("cards", "tower")}
    lines = ["{" + ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in head.items()) + ",",
             ' "tower": ' + json.dumps(data["tower"], ensure_ascii=False) + ",", ' "cards": {']
    items = [f"  {json.dumps(slug)}: {json.dumps(rec, ensure_ascii=False)}" for slug, rec in sorted(data["cards"].items())]
    lines.append(",\n".join(items))
    lines.append(" }}")
    return "\n".join(lines) + "\n"


def series_value(s: dict | None, level: int) -> int | None:
    if not s:
        return None
    v = s["v"]
    i = level - s["l0"]
    if 0 <= i < len(v):
        return int(round(v[i]))
    if i >= len(v):
        return int(round(v[-1] * LEVEL_GROWTH ** (i - len(v) + 1)))
    return int(round(v[0] / LEVEL_GROWTH ** (-i)))


CALIBRATION_CARDS = (("knight", 15), ("knight", 16), ("giant", 15), ("balloon", 15), ("balloon", 16), ("bats", 15),
                     ("musketeer", 15), ("tesla", 15), ("fireball", 15), ("zap", 15), ("night-witch", 15))


def calibration_table(data: dict) -> list[str]:
    """KB value at the level vs the x1.1 rule from the first table row, and vs cr-api when given (hitpoints / damage)."""
    rows = ["card level  field   kb_table  x1.1_rule  diff%"]
    for slug, level in CALIBRATION_CARDS:
        rec = data["cards"].get(slug)
        if not rec:
            continue
        for fld in ("hp", "damage"):
            u = rec["units"][0] if rec["type"] != "spell" else None
            if u is None and fld == "hp":
                continue
            s = u.get(fld) if u else (rec.get("spell") or {}).get("damage")
            if not s:
                continue
            kbv = series_value(s, level)
            rule = int(round(s["v"][0] * LEVEL_GROWTH ** (level - s["l0"])))
            rows.append(f"{slug:<12} {level:>3}  {fld:<7} {kbv:>8}  {rule:>9}  {100.0 * (rule - kbv) / kbv:+6.2f}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kb-dir", type=Path, default=KB_DIR)
    ap.add_argument("--stats-dir", type=Path, default=bcr.DEFAULT_STATS_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fetch", action="store_true", help="download cards_stats.json and cards.json into --stats-dir first")
    args = ap.parse_args(argv)
    if args.fetch:
        bcr.fetch_stats(args.stats_dir)
    stats = bcr.load_stats(args.stats_dir)
    if stats is None:
        print(f"warning: no cr-api data in {args.stats_dir} (use --fetch); building from the knowledge base only",
              file=sys.stderr)
    cards = load_cards()
    data, report = build(args.kb_dir, stats, cards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump(data), encoding="utf-8")
    json.loads(args.out.read_text(encoding="utf-8"))               # the file must parse
    print(f"wrote {args.out}: {len(data['cards'])} cards")
    src: dict[str, int] = {}
    for rec in data["cards"].values():
        for grp, s in rec["source"].items():
            src[f"{grp}:{s}"] = src.get(f"{grp}:{s}", 0) + 1
    print("sources     :", dict(sorted(src.items())))
    print("no units    :", report["no_kb"])
    print("no hitpoints:", report["no_hp"])
    print("no damage   :", report["no_damage"])
    print("cr-api-only series:", report["cr_only"])
    print("notes:")
    for n in report["notes"]:
        print("  " + n)
    print("calibration (KB table vs the x1.1 rule):")
    for line in calibration_table(data):
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
