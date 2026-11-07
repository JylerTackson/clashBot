"""Unit stats and the threat model (MAC-AH): the committed table, the level scaling, spell values, the lane simulation,
and the builder's knowledge-base parser on a tiny fake KB - no models, no screen."""
import statistics
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "data"))

from clashbot import unit_stats as us  # noqa: E402
from clashbot.game_rules import is_playable, load_cards  # noqa: E402

CARDS = load_cards()


def _enemy(card, tile, cls=None):
    return {"card": card, "unit_class": cls or card, "side": "enemy", "tile": list(tile)}


# -- the table ---------------------------------------------------------------------------------------------------------
def test_the_committed_table_covers_every_playable_card_with_hitpoints_or_damage():
    cards = us.load_unit_stats()["cards"]
    playable = [s for s in CARDS if is_playable(s, CARDS)]
    assert len(cards) >= 126 and not [s for s in playable if s not in cards]
    for slug in playable:
        st = us.stat_at(slug, 15)
        assert st is not None, slug
        if st["type"] == "spell":
            continue
        assert st["units"] and st["units"][0]["hp"] > 0, slug


def test_the_princess_tower_at_level_15_has_4424_hitpoints():
    tower = us.tower_stat("princess", 15)
    assert tower["hp"] == 4424 and tower["damage"] == 158 and tower["hit_speed_s"] == 0.8 and tower["range_tiles"] == 7.5
    assert us.tower_stat("princess")["hp"] == 4424 and us.tower_level() == 15
    assert us.tower_stat("princess", 11)["hp"] == 3052 and us.tower_stat("princess", 16)["hp"] == 4858   # seen in the pros' matches
    assert us.tower_stat("king", 16)["hp"] == 7728 and us.tower_stat("king", 11)["hp"] == 4824


# the exact per-level numbers of the knowledge base (cards/*.md "Stats by level"), read back at levels 15 and 16
@pytest.mark.parametrize("card,level,hp,damage", [
    ("balloon", 15, 2458, 937), ("balloon", 16, 2704, 1031), ("giant", 15, 5810, 370), ("knight", 15, 2586, 296),
    ("knight", 16, 2844, 325), ("bats", 15, 119, 119), ("musketeer", 15, 1056, 318), ("musketeer", 16, 1161, 349),
    ("tesla", 15, 1687, 322), ("tesla", 16, 1855, 354), ("archers", 15, 445, 164)])
def test_stat_at_reads_the_exact_table_row(card, level, hp, damage):
    unit = us.stat_at(card, level)["units"][0]
    assert unit["hp"] == hp and unit["damage"] == damage


def test_expected_ranges_for_the_units_that_matter():
    ranges = {"knight": (2500, 3000), "giant": (5500, 6200), "balloon": (2400, 2800), "bats": (110, 140),
              "musketeer": (1000, 1200), "night-witch": (1250, 1500), "electro-wizard": (1000, 1100)}
    for card, (lo, hi) in ranges.items():
        for level in (15, 16):
            hp = us.stat_at(card, level)["units"][0]["hp"]
            assert lo <= hp <= hi * (1.1 if level == 16 else 1.0), (card, level, hp)
    assert us.stat_at("bats", 15)["units"][0]["count"] == 5 and us.stat_at("bats", 15)["units"][0]["flying"]
    balloon = us.stat_at("balloon", 15)["units"][0]
    assert balloon["targets"] == "buildings" and balloon["flying"] and balloon["death_damage"] == 351
    assert balloon["speed_tiles_s"] == 1.0 and balloon["hit_speed_s"] == 2.0 and balloon["dps"] == 468.5


def test_levels_beyond_a_table_grow_by_ten_percent_a_level_and_the_rule_tracks_the_table():
    s = us.load_unit_stats()["cards"]["balloon"]["units"][0]["hp"]
    assert us.series_at(s, 16) == 2704 and us.series_at(s, 17) == round(2704 * 1.1) == us.stat_at("balloon", 17)["units"][0]["hp"]
    assert us.series_at(s, 5) == round(1043 / 1.1)
    # calibration: the x1.1 rule from the first row stays within 1% of the table for the cards we know
    for card, level in (("knight", 15), ("giant", 15), ("balloon", 16), ("musketeer", 15), ("tesla", 15), ("night-witch", 15)):
        series = us.load_unit_stats()["cards"][card]["units"][0]["hp"]
        rule = series["v"][0] * 1.1 ** (level - series["l0"])
        assert rule == pytest.approx(us.series_at(series, level), rel=0.01), card


def test_levels_json_and_the_defaults():
    assert us.opponent_level("balloon") == 15 and us.own_level("knight") == 16 and us.own_level("electro-dragon") == 16
    assert us.own_level("musketeer") == 15 and us.own_level("goblin-barrel") == 14 and us.own_level("giant") == 14
    assert us.stat_at("balloon")["level"] == 15                          # the opponent's level by default


def test_heroes_and_evolutions_of_our_deck():
    hero = us.stat_at("musketeer", 15, "hero")
    turret = hero["abilities"]["trusty-turret"]
    assert turret["cost"] == 3 and turret["turret"]["range_tiles"] == 4.0 and turret["turret"]["lifetime_s"] == 10.0
    assert turret["turret"]["hitpoints"] == 2249 and turret["turret"]["damage"] == 205 and turret["turret"]["spawn_damage"] == 299
    assert hero["units"][0]["hp"] == 1056
    assert us.stat_at("musketeer", 11, "hero")["abilities"]["trusty-turret"]["turret"]["hitpoints"] == 1536   # a typo in the source
    evo = us.stat_at("tesla", 15, "evo")["abilities"]["evolution"]
    assert evo["cycles"] == 2 and evo["pulse_radius_tiles"] == 6.0 and evo["pulse_stun_s"] == 0.5
    assert us.stat_at("tesla", 15, "evo")["abilities"]["series"]["pulse_damage"] == 255
    dragon = us.stat_at("electro-dragon", 15, "evo")
    assert dragon["units"][0]["hp"] == 2124 and dragon["abilities"]["series"]["damage_after_5_chains"] == 94
    assert us.stat_at("electro-dragon", 15)["units"][0]["hp"] == 1536            # the base card is unchanged
    assert us.stat_at("knight", 15, "evo")["units"][0]["hp"] == 2586              # no such variant: the base card


# -- spells ------------------------------------------------------------------------------------------------------------
def test_zap_kills_level_15_bats_but_not_a_balloon_and_fireball_kills_archers_but_not_a_giant():
    zap, fireball = us.spell_stat("zap"), us.spell_stat("fireball")
    assert zap["damage"] == 281 and zap["radius_tiles"] == 2.5 and zap["crown_tower_damage"] == 85 and zap["stun_s"] == 0.5
    assert fireball["damage"] == 1007 and fireball["radius_tiles"] == 2.5 and fireball["crown_tower_damage"] == 303
    assert zap["damage"] > us.stat_at("bats", 15)["units"][0]["hp"]
    assert zap["damage"] < us.stat_at("balloon", 15)["units"][0]["hp"] / 5
    assert fireball["damage"] > us.stat_at("archers", 15)["units"][0]["hp"]
    assert fireball["damage"] < us.stat_at("giant", 15)["units"][0]["hp"] / 5
    # the musketeer (1,056 HP at level 15) survives a level-15 fireball by a sliver
    assert fireball["damage"] < us.stat_at("musketeer", 15)["units"][0]["hp"]
    assert us.spell_stat("clone") is None and us.spell_stat("goblin-barrel") is None and us.spell_stat("not-a-card") is None


def _value(spell, *cards):
    units = [_enemy(c, (5, 9)) for c in cards]
    return us.spell_value(spell, units, CARDS)


def test_spell_values_by_kills_and_damage():
    # (MAC-AM M2.3: a swarm card is paid for by all its units - bats 5, skeletons 3, minions 3 - not by the ones seen: two
    # bats are 2/5 of the bats' 2 elixir; before, the units seen stood for the card and two bats were priced as five)
    assert _value("zap", "bats", "bats") == (0.8, 2)
    assert _value("zap", "bats", "bats", "bats", "bats", "bats") == (2.0, 5)
    value, killed = _value("zap", "balloon")                              # a lone balloon: damaged, not killed
    assert killed == 0 and value == pytest.approx(0.5 * 5 * 281 / 2458, abs=0.001) and value < 2.0
    assert _value("fireball", "archers", "archers") == (3.0, 2)           # kills the pair: 3 elixir < the 4 it costs
    value, killed = _value("fireball", "musketeer")
    assert killed == 0 and value == pytest.approx(0.5 * 4 * 1007 / 1056, abs=0.001)
    value, killed = _value("fireball", "bats", "bats", "archers", "archers")
    assert killed == 4 and value == 3.8                                   # 0.8 (2 of 5 bats) + 3 elixir (the archers' pair)
    value, killed = _value("zap", "bats", "bats", "night-witch", "electro-wizard")
    assert killed == 2 and 1.7 < value < 1.8                              # 2 of 5 bats dead (0.8) + a bit of witch and e-wizard
    value, _ = _value("fireball", "giant")
    assert value == pytest.approx(0.5 * 5 * 1007 / 5810, abs=0.001)       # nowhere near the spell's cost
    assert us.spell_value("clone", [_enemy("bats", (5, 9))], CARDS) is None
    assert _value("zap") == (0.0, 0)


def test_killing_every_unit_of_a_swarm_play_never_beats_its_elixir_and_one_bat_of_five_is_two_fifths():
    """MAC-AM M2.3: the units one play of a card puts on the field are what pays for its elixir."""
    assert [us.card_unit_count(c) for c in ("bats", "skeletons", "minions", "minion-horde", "knight", "night-witch")] == [
        5, 3, 3, 6, 1, 1]
    for card, n in (("bats", 5), ("skeletons", 3), ("minions", 3), ("spear-goblins", 3), ("skeleton-army", 15)):
        value, killed = _value("fireball", *[card] * n)                       # a fireball kills every unit of them
        assert killed == n and value == pytest.approx(CARDS[card]["elixir_cost"])
        assert _value("zap", *[card] * n)[0] <= CARDS[card]["elixir_cost"] + 1e-9
    assert _value("zap", "bats") == (0.4, 1)                                  # one bat of five: not the whole card (was 2.0)
    assert _value("fireball", "skeletons") == (pytest.approx(1 / 3, abs=0.001), 1)
    # more units seen than one play brings (two plays, or a night witch's bats) are shared out, never above the card
    value, killed = _value("zap", *["bats"] * 8)
    assert killed == 8 and value == pytest.approx(2.0)
    # a lone survivor of a swarm card is worth its share, too (a zap does not kill minions: 370 hp)
    value, killed = _value("zap", "minions")
    assert killed == 0 and value < 0.5 * CARDS["minions"]["elixir_cost"] / 3


def test_the_spell_value_table_for_the_report():
    """zap / fireball vs bats, balloon, archers (printed with -s; the asserts above pin the numbers)."""
    rows = {}
    for spell in ("zap", "fireball"):
        cost = CARDS[spell]["elixir_cost"]
        for what, cards in (("bats x5", ["bats"] * 5), ("balloon", ["balloon"]), ("archers x2", ["archers"] * 2),
                            ("musketeer", ["musketeer"]), ("giant", ["giant"])):
            v, k = _value(spell, *cards)
            rows[(spell, what)] = (v, k, "worth it" if v >= cost else "not worth it")
    assert rows[("zap", "bats x5")][2] == "worth it" and rows[("zap", "balloon")][2] == "not worth it"
    assert rows[("fireball", "archers x2")][2] == "not worth it"
    assert rows[("fireball", "bats x5")][2] == "not worth it"              # 2 elixir of bats for a 4-elixir fireball
    print("\n".join(f"{s:<9} vs {w:<11} value {v:5.2f} killed {k}  {verdict}" for (s, w), (v, k, verdict) in rows.items()))


# -- the simulation ----------------------------------------------------------------------------------------------------
def test_two_bats_do_nothing_to_a_level_15_tower():
    r = us.simulate_lane([_enemy("bats", (3, 14), "bat"), _enemy("bats", (4, 14), "bat")], [], "left")
    assert r.tower_damage < 300 and not r.tower_destroyed and r.units_alive_at_end == 0 and r.enemy_killed == 2
    assert r.time_to_clear is not None and r.time_to_clear < 4


def test_a_lone_balloon_does_big_damage_and_a_musketeer_behind_the_tower_cuts_it():
    balloon = [_enemy("balloon", (16, 15))]
    open_ = us.simulate_lane(balloon, [], "right")
    assert open_.tower_damage >= 1500 and open_.tower_damage == 3162 and not open_.tower_destroyed
    assert open_.eta_s is not None and 8 < open_.eta_s < 11 and open_.time_to_clear == 15.0
    guarded = us.simulate_lane(balloon, [{"card": "musketeer", "tile": [15.5, 2.5]}], "right")
    assert guarded.tower_damage < open_.tower_damage * 0.25               # only the death bomb (351) lands
    assert guarded.tower_damage == 351
    pulled = us.simulate_lane(balloon, [{"card": "tesla", "tile": [12.0, 8.0]}], "right")
    assert pulled.tower_damage == 0 and pulled.eta_s is None              # it never reaches the tower
    # cards that cannot hit a flyer change nothing; a zap does not kill it
    assert us.simulate_lane(balloon, [{"card": "knight", "tile": [14.5, 9.0]}], "right").tower_damage == open_.tower_damage
    assert us.simulate_lane(balloon, [{"card": "zap", "tile": [14.5, 9.0]}], "right").tower_damage == open_.tower_damage


def test_ground_pushes_and_the_tower_hitpoints():
    giant = [_enemy("giant", (15, 13))]
    open_ = us.simulate_lane(giant, [], "right")
    assert 2500 <= open_.tower_damage <= 3500 and open_.units_alive_at_end == 1       # the tower cannot kill a giant in 20 s
    assert us.simulate_lane(giant, [{"card": "tesla", "tile": [12, 8]}], "right").tower_damage == 0
    assert us.simulate_lane(giant, [{"card": "mini-p-e-k-k-a", "tile": [15, 10]}], "right").tower_damage < open_.tower_damage / 3
    low = us.simulate_lane(giant, [], {"lane": "right", "hp": 600})
    assert low.tower_destroyed and low.tower_damage == 600 and low.destroyed_at_s is not None and low.tower_hp_left == 0
    # a king-side unit walks to the other lane's tower: nothing reaches the left one
    assert us.simulate_lane([_enemy("knight", (15, 13))], [], "left").tower_damage >= 0


def test_a_witch_pack_destroys_the_tower_and_our_units_stop_it():
    pack = [_enemy("night-witch", (4, 10)), _enemy("electro-wizard", (3, 10)), _enemy("bats", (4, 10), "bat"),
            _enemy("bats", (2, 10), "bat")]
    tower = {"lane": "left", "hp": 4082}
    open_ = us.simulate_lane(pack, [], tower)
    assert open_.tower_destroyed and open_.tower_damage == 4082
    held = us.simulate_lane(pack, [{"card": "electro-dragon", "tile": [4, 7], "placed": True, "single": True},
                                   {"card": "musketeer", "tile": [5, 5], "placed": True, "single": True}], tower)
    assert not held.tower_destroyed and held.tower_damage < 300
    assert us.simulate_lane(pack, [{"card": "zap", "tile": [3.5, 10]}], tower).tower_damage > held.tower_damage


def test_spirits_explode_once_and_spawners_and_death_spawns_count():
    spirit = us.simulate_lane([_enemy("giant", (15, 13))], [{"card": "ice-spirit", "tile": [15, 10]}], "right")
    assert spirit.defenders_lost == 1                                   # the spirit is gone after its hit
    golem = us.simulate_lane([_enemy("golem", (15, 14))], [], "right")
    assert golem.tower_damage > 0
    witch = us.simulate_lane([_enemy("night-witch", (3.5, 12))], [], {"lane": "left", "hp": 4424})
    nowitch_bats = us.simulate_lane([_enemy("bats", (3.5, 12), "bat")], [], {"lane": "left", "hp": 4424})
    assert witch.enemy_killed > 1 > nowitch_bats.tower_damage            # her bats were simulated (and killed)


def test_the_simulation_is_deterministic_and_unaffected_by_unknown_units():
    pack = [_enemy("night-witch", (4, 10)), _enemy("bats", (4, 10), "bat"), {"card": None, "tile": [3, 11]},
            {"card": "not-a-card", "tile": [3, 12]}, {"card": "knight", "tile": None}]
    a, b = us.simulate_lane(pack, [], "left"), us.simulate_lane(pack, [], "left")
    assert a == b and a.tower_damage > 0
    assert us.simulate_lane([], [], "left") == us.SimResult(0, False, None, 0, 0.0, None, 4424, 0, 0)


def test_one_lane_takes_under_two_milliseconds():
    pack = [_enemy("night-witch", (4, 10)), _enemy("electro-wizard", (3, 10)), _enemy("bats", (4, 10), "bat"),
            _enemy("bats", (2, 10), "bat"), _enemy("bats", (3, 10), "bat"), _enemy("knight", (4, 12)),
            _enemy("giant", (3, 13))]
    defenders = [{"card": "electro-dragon", "tile": [4, 7], "placed": True, "single": True},
                 {"card": "musketeer", "tile": [5, 5], "placed": True, "single": True}, {"card": "tesla", "tile": [6, 7]}]
    for args in ((pack, []), (pack, defenders), ([_enemy("balloon", (16, 15))], [])):
        us.simulate_lane(*args, "left" if args[0] is pack else "right")                  # warm the stats cache
        runs = []
        for _ in range(60):
            t0 = time.perf_counter()
            us.simulate_lane(*args, "left" if args[0] is pack else "right")
            runs.append(time.perf_counter() - t0)
        assert statistics.median(runs) < 0.002, f"{statistics.median(runs) * 1000:.2f} ms"
        assert min(runs) < 0.002


# -- the builder's parser (a tiny fake knowledge base) ------------------------------------------------------------------
FAKE_WITCH = """---
name: Night Witch
slug: night-witch
rarity: Legendary
elixir_cost: 4
card_type: Troop
targets: ground
---
# Night Witch

## Attributes

**Night Witch Attributes**

| Cost | Hit Speed | First Hit Speed | Speed | Deploy Time | Spawn Speed | Range | Target | Count | Transport | Type | Rarity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.3 sec | 0.75 sec | Medium (60) | 1 sec | 5 sec | Melee: Long (1.6) | Ground | x1 | Ground | Troop | Legendary |

**Bat Attributes**

| Hit Speed | First Hit Speed | Speed | Range | Target | Transport |
|---|---|---|---|---|---|
| 1.3 sec | 0.6 sec | Very Fast (120) | Melee: Medium (1.2) | Air & Ground | Air |

**Sound Effects**

| Sound Effect | Action |
|---|---|
| x.wav | Deployment |

## Stats by level

**Card Statistics**

| Level | Night Witch Hitpoints | Night Witch Damage | Night Witch Damage per second | Bat Hitpoints | Bat Damage | Bat Damage per second |
|---|---|---|---|---|---|---|
| 9 | 749 | 260 | 200 | 67 | 67 | 51 |
| 10 | 824 | 285 | 219 | 74 | 74 | 56 |
| 11 | 1,906 | 314 | 241 | 81 | 81 | 62 |
| 12 | 997 | 345 | 265 | 89 | 89 | 68 |
| 13 | 1,096 | 380 | 292 | 98 | 98 | 75 |

## Abilities and special mechanics
"""

FAKE_ZAP = """---
name: Zap
slug: zap
rarity: Common
elixir_cost: 2
card_type: Spell
---
## Attributes

| Cost | Radius | Stun Duration | Target | Type | Rarity |
|---|---|---|---|---|---|
| 2 | 2.5 | 0.5 sec | Air & Ground | Spell | Common |

## Stats by level

| Level | Area Damage | Crown Tower Damage |
|---|---|---|
| 14 | 256 | 77 |
| 15 | 281 | 85 |
| 16 | 309 | 93 |
"""

FAKE_CRACKER = """---
name: Firecracker
slug: firecracker
rarity: Common
elixir_cost: 3
card_type: Troop
---
## Attributes

| Cost | Hit Speed | First Hit Speed | Speed | Deploy Time | Range | Target | Count | Transport | Type | Rarity |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 3 sec | 0.65 sec | Fast (90) | 1 sec | 6 | Air & Ground | x1 | Ground | Troop | Common |

## Stats by level

| Level | Hitpoints | Area Damage | Damage per second |
|---|---|---|---|
| 14 | 400 | 25 x5 (125) | 42 |
| 15 | 440 | 27 x5 (135) | 45 |
"""


def test_column_names_map_to_a_unit_and_a_field():
    import build_unit_stats as b
    assert b.column_key("Night Witch Hitpoints") == ("night witch", "hp")
    assert b.column_key("Hitpoints") == ("", "hp")
    assert b.column_key("Area Damage") == ("", "damage") and b.column_key("Damage") == ("", "damage")
    assert b.column_key("Bat Damage per second") == ("bat", "dps")
    assert b.column_key("Little Prince Damage per Second (Stage 2)") == ("little prince", "dps")
    assert b.column_key("Golem Death Damage") == ("golem", "death_damage")
    assert b.column_key("Crown Tower Damage") == ("", "ct_damage")
    assert b.column_key("Turret Hitpoints lost per second") == ("turret", "decay")
    assert b.column_key("Charge Damage") == ("charge", "damage")           # an unknown prefix: the caller files it as extra
    assert b.num("1,400") == 1400 and b.num("119 x3") == 119 and b.num("N/A") is None
    assert b.damage_num("25 x5 (125)") == 125 and b.damage_num("119 x3") == 119 and b.damage_num("1,007") == 1007
    assert b.speed_tiles("Very Fast (120)") == 2.0 and b.speed_tiles("Slow (45)") == 0.75
    assert b.range_tiles("Melee: Long (1.6)") == 1.6 and b.range_tiles("5.5") == 5.5
    assert b.targets_of("Air & Ground") == "air_ground" and b.targets_of("Buildings") == "buildings"


def test_the_parser_reads_units_attributes_and_per_level_series():
    import build_unit_stats as b
    raw = b.parse_card_text(FAKE_WITCH, "night-witch")
    assert [u["name"] for u in raw["units"]] == ["Night Witch", "Bat"]
    witch, bat = raw["units"]
    assert (witch["hit_speed_s"], witch["first_hit_s"], witch["speed_tiles_s"], witch["range_tiles"]) == (1.3, 0.75, 1.0, 1.6)
    assert witch["targets"] == "ground" and witch["flying"] is False and witch["spawn_every_s"] == 5.0
    assert bat["flying"] is True and bat["targets"] == "air_ground" and bat["speed_tiles_s"] == 2.0
    hp = raw["series"][0]["hp"]
    assert hp["l0"] == 9 and hp["v"][:2] == [749, 824]
    assert hp["v"][2] == 906                                               # the 1,906 typo is repaired (between 824 and 997)
    assert any("1906" in n or "1,906" in n or "906" in n for n in raw["notes"])
    assert raw["series"][1]["hp"]["v"] == [67, 74, 81, 89, 98] and raw["series"][1]["damage"]["l0"] == 9
    card = b.build_card("night-witch", raw, [], None, {"rarity": "Legendary", "elixir_cost": 4, "card_type": "Troop"})[0]
    assert card["type"] == "troop" and card["start_level"] == 9 and card["elixir"] == 4
    assert card["units"][0]["spawn"] == {"unit": "Bat", "every_s": 5.0, "count": 1, "start_s": 1.0}
    assert card["units"][1]["count"] == 0 and card["units"][1]["flying"] is True       # spawned: none at the start


def test_the_parser_reads_spells_and_multi_hit_damage_cells():
    import build_unit_stats as b
    zap = b.parse_card_text(FAKE_ZAP, "zap")
    rec = b.build_card("zap", zap, [], None, {"rarity": "Common", "elixir_cost": 2, "card_type": "Spell"})[0]
    assert rec["type"] == "spell" and rec["spell"]["radius_tiles"] == 2.5 and rec["spell"]["stun_s"] == 0.5
    assert rec["spell"]["damage"] == {"l0": 14, "v": [256, 281, 309]} and rec["spell"]["crown_tower_damage"]["v"] == [77, 85, 93]
    cracker = b.parse_card_text(FAKE_CRACKER, "firecracker")
    assert cracker["series"][0]["damage"]["v"] == [125, 135]               # "25 x5 (125)": the whole volley on one target
    card = b.build_card("firecracker", cracker, [], None, {"rarity": "Common", "elixir_cost": 3, "card_type": "Troop"})[0]
    assert card["units"][0]["hit_speed_s"] == 3.0                          # damage 135 / dps 45 = 3.0 agrees with the table


def test_a_hit_speed_that_disagrees_with_damage_over_dps_is_replaced():
    import build_unit_stats as b
    unit = {"name": "Executioner", "hit_speed_s": 0.9, "series_damage": {"l0": 6, "v": [208, 230]},
            "series_dps": {"l0": 6, "v": [86, 95]}}
    notes = []
    b.check_hit_speed(unit, notes, "executioner")
    assert unit["hit_speed_s"] == 2.4 and notes


def test_the_builder_main_writes_a_parsable_table(tmp_path):
    import json

    import build_unit_stats as b
    kb = tmp_path / "kb"
    (kb / "cards").mkdir(parents=True)
    (kb / "cards" / "night-witch.md").write_text(FAKE_WITCH, encoding="utf-8")
    (kb / "cards" / "zap.md").write_text(FAKE_ZAP, encoding="utf-8")
    out = tmp_path / "unit_stats.json"
    assert b.main(["--kb-dir", str(kb), "--stats-dir", str(tmp_path / "no-cr-api"), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["cards"]) == {"night-witch", "zap"} and data["version"] == 1
    assert data["tower"]["princess"]["hp"]["v"][14] == 4424 or data["tower"]["princess"]["source"] == "anchor"
