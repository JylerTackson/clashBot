"""Card role table (MAC-Z1): committed JSON validates against the card index, spot checks, and the builder's stats mapping."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.card_roles import (  # noqa: E402
    HP_CLASSES, RESOURCE, ROLES, hits_air, is_tank, is_win_condition, load_roles, validate_roles,
)
from clashbot.game_rules import is_playable, load_cards  # noqa: E402
from tools.data import build_card_roles as brc  # noqa: E402


def test_roles_validate_against_card_index():
    cards = load_cards()
    assert validate_roles(load_roles(), cards) == []
    assert RESOURCE.exists()


def test_every_playable_card_has_a_role_and_manual_entry():
    cards = load_cards()
    playable = {s for s in cards if is_playable(s, cards)}
    assert playable <= set(load_roles())
    assert playable <= set(brc.MANUAL)
    assert len(playable) >= 100


def test_json_is_sorted_and_versioned():
    raw = json.loads(RESOURCE.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert list(raw["cards"]) == sorted(raw["cards"])
    assert all(r["source"] in ("stats+manual", "manual") for r in raw["cards"].values())


def test_spot_checks():
    r = load_roles()
    knight = r["knight"]
    assert (knight["type"], knight["range"], knight["targets"], knight["hp_class"]) == ("troop", "melee", "ground", "medium")
    assert (r["tesla"]["type"], r["tesla"]["targets"]) == ("building", "air_ground")
    assert r["tesla"]["speed"] is None and "air_defense" in r["tesla"]["roles"]
    giant = r["giant"]
    assert giant["hp_class"] == "tank" and giant["targets"] == "buildings" and is_win_condition(giant) and is_tank(giant)
    bats = r["bats"]
    assert bats["flying"] and bats["hp_class"] == "swarm" and bats["targets"] == "air_ground"
    assert (r["zap"]["spell_class"], r["zap"]["kills"]) == ("small", ["swarm"])
    assert (r["fireball"]["spell_class"], sorted(r["fireball"]["kills"])) == ("medium", ["light", "swarm"])
    assert not hits_air(r["the-log"])
    assert r["magic-archer"]["range"] == "long"
    assert is_win_condition(r["goblin-barrel"])
    skeletons = r["skeletons"]
    assert skeletons["hp_class"] == "swarm" and skeletons["count"] == 3


def test_hits_air_helper():
    r = load_roles()
    for slug in ("the-log", "barbarian-barrel", "earthquake", "giant-snowball"):
        assert not hits_air(r[slug]), slug
    for slug in ("zap", "fireball", "arrows", "lightning", "rocket", "poison", "freeze"):
        assert hits_air(r[slug]), slug
    assert hits_air(r["musketeer"]) and hits_air(r["tesla"])
    assert not hits_air(r["knight"]) and not hits_air(r["cannon"]) and not hits_air(r["giant"])


def test_multi_unit_chain_and_pierce_cards_keep_their_manual_values():
    r = load_roles()
    assert r["magic-archer"]["splash"] and r["electro-dragon"]["splash"] and r["firecracker"]["splash"]
    for slug in ("goblin-gang", "rascals"):
        assert r[slug]["targets"] == "air_ground" and r[slug]["range"] == "short", slug
        assert hits_air(r[slug]), slug
    assert hits_air(r["goblin-hut"]) and hits_air(r["furnace"])
    assert r["goblin-giant"]["targets"] == "buildings"                         # the giant drives targeting
    for slug in ("prince", "night-witch", "guards", "royal-recruits", "phoenix", "mighty-miner", "battle-healer"):
        assert r[slug]["range"] == "melee", slug                              # 1.6 tiles counts as melee


def test_vocabulary_of_the_committed_table():
    for slug, role in load_roles().items():
        assert set(role["roles"]) <= ROLES, slug
        if role["type"] == "spell":
            assert set(role["kills"]) <= HP_CLASSES, slug
        else:
            assert (role["speed"] is None) == (role["type"] == "building"), slug
            assert role["count"] >= 1, slug


def test_validate_roles_reports_problems():
    cards = {"a": {"card_type": "Troop", "elixir_cost": 3.0}, "b": {"card_type": "Spell", "elixir_cost": 2.0},
             "c": {"card_type": "Tower Troop", "elixir_cost": None}}
    ok = {"a": {"type": "troop", "targets": "ground", "hp_class": "medium", "roles": []},
          "b": {"type": "spell", "spell_class": "small", "radius": 2.0, "kills": ["swarm"], "roles": []}}
    assert validate_roles(ok, cards) == []
    assert any("b: missing" in p for p in validate_roles({"a": ok["a"]}, cards))
    assert any("unknown targets" in p for p in validate_roles({**ok, "a": {**ok["a"], "targets": "sky"}}, cards))
    assert any("missing hp_class" in p for p in validate_roles({**ok, "a": {"type": "troop", "targets": "ground"}}, cards))
    assert any("missing targets" in p for p in validate_roles({**ok, "a": {"type": "building", "hp_class": "light"}}, cards))
    assert any("spell_class" in p for p in validate_roles({**ok, "b": {"type": "spell", "radius": 2.0}}, cards))
    assert any("radius" in p for p in validate_roles({**ok, "b": {"type": "spell", "spell_class": "small"}}, cards))
    assert any("unknown role" in p for p in validate_roles({**ok, "a": {**ok["a"], "roles": ["boss"]}}, cards))
    assert any("unknown type" in p for p in validate_roles({**ok, "a": {"type": "wall"}}, cards))


def test_manual_table_uses_the_vocabularies():
    assert brc.check_manual(brc.MANUAL) == []


# ---------------------------------------------------------------------------------------- builder on fake stats

def _fake_stats(tmp_path):
    chars = [
        {"name": "Knight", "range": 1200, "speed": 60, "flying_height": 0, "attacks_air": False, "attacks_ground": True,
         "area_damage_radius": 0, "target_only_buildings": False, "hitpoints": 690,
         "hitpoints_per_level": [690, 759, 834, 917, 1007, 1104, 1214, 1331, 1462, 1607, 1766]},
        {"name": "Bat", "range": 5500, "speed": 120, "flying_height": 2000, "attacks_air": True, "attacks_ground": True,
         "area_damage_radius": 1000, "target_only_buildings": False, "hitpoints": 81},
    ]
    sections = {"troop": [{"key": "knight", "summon_character": "Knight", "summon_number": 0},
                          {"key": "bats", "summon_character": "Bat", "summon_number": 5}],
                "building": [], "spell": [], "projectile": [], "characters": chars}
    (tmp_path / "cards_stats.json").write_text(json.dumps(sections))
    (tmp_path / "cards.json").write_text(json.dumps([{"key": "knight", "name": "Knight", "sc_key": "Knight"},
                                                     {"key": "bats", "name": "Bats", "sc_key": "Bats"}]))
    return brc.load_stats(tmp_path)


def test_builder_derives_stats_fields_from_a_fake_stats_dir(tmp_path):
    stats = _fake_stats(tmp_path)
    manual = {   # every stats-derived field deliberately wrong, so each one has to be overridden
        "knight": brc.troop("air_ground", "medium", "medium", "long", "very_fast", 1, ["mini_tank"], flying=True, splash=True),
        "bats": brc.troop("ground", "swarm", "low", "melee", "slow", 5, ["swarm"], flying=False, splash=False),
        "zap": brc.spell("small", 2.5, ["swarm"], ["spell_small"]),
    }
    cards = {"knight": {"rarity": "Common"}, "bats": {"rarity": "Common"}, "zap": {"rarity": "Common"}}
    roles, report = brc.build_roles(stats, manual, cards)
    knight, bats = roles["knight"], roles["bats"]
    assert (knight["flying"], knight["targets"], knight["splash"], knight["range"], knight["speed"]) == (False, "ground", False, "melee", "medium")
    assert (bats["flying"], bats["targets"], bats["splash"], bats["range"], bats["speed"]) == (True, "air_ground", True, "long", "very_fast")
    assert knight["source"] == bats["source"] == "stats+manual" and roles["zap"]["source"] == "manual"
    assert knight["hp_class"] == "medium" and knight["count"] == 1 and bats["count"] == 5      # manual fields untouched
    assert "knight.range: manual='long' stats='melee'" in report["disagreements"]
    assert "bats.speed: manual='slow' stats='very_fast'" in report["disagreements"]
    assert len(report["disagreements"]) == 10 and report["stats"] == 2 and report["manual"] == 1
    assert list(roles) == sorted(roles)


def test_builder_agreeing_manual_reports_no_disagreement(tmp_path):
    stats = _fake_stats(tmp_path)
    manual = {"knight": brc.troop("ground", "medium", "medium", "melee", "medium", 1, ["mini_tank"])}
    roles, report = brc.build_roles(stats, manual, {"knight": {"rarity": "Common"}})
    assert report["disagreements"] == [] and report["notes"] == []
    assert roles["knight"]["source"] == "stats+manual"


def test_builder_notes_do_not_override_manual_fields(tmp_path):
    stats = _fake_stats(tmp_path)
    manual = {"knight": brc.troop("ground", "tank", "medium", "melee", "medium", 3, ["mini_tank"])}
    roles, report = brc.build_roles(stats, manual, {"knight": {"rarity": "Common"}})
    assert roles["knight"]["hp_class"] == "tank" and roles["knight"]["count"] == 3
    assert any(n.startswith("knight.count: manual=3 stats=1") for n in report["notes"])
    assert any(n.startswith("knight.hp_class: manual=tank stats=medium") for n in report["notes"])


def test_derive_stats_mapping_boundaries():
    def d(**kw):
        return brc.derive_stats(kw, {}, is_building=False)
    assert d(range=1600)["range"] == "melee" and d(range=1601)["range"] == "short"        # 1.6-tile units (prince) are melee
    assert d(range=4999)["range"] == "short" and d(range=5000)["range"] == "long"
    assert [d(speed=s)["speed"] for s in (45, 46, 60, 61, 90, 91)] == ["slow", "medium", "medium", "fast", "fast", "very_fast"]
    assert d(flying_height=1)["flying"] and not d(flying_height=0)["flying"]
    assert d(target_only_buildings=True, attacks_air=True)["targets"] == "buildings"       # buildings-only wins
    assert d(attacks_air=True)["targets"] == "air_ground" and d()["targets"] == "ground"
    assert d(area_damage_radius=1)["splash"] and not d(area_damage_radius=0)["splash"]
    wiz = brc.derive_stats({"projectile": "P"}, {"P": {"radius": 1500}}, is_building=False)
    assert wiz["splash"]                                                                   # splash carried by the projectile
    assert not brc.derive_stats({"projectile": "P"}, {"P": {"radius": 250}}, is_building=False)["splash"]
    assert brc.derive_stats({"speed": 0, "range": 5500}, {}, is_building=True)["speed"] is None


def test_stats_entry_maps_slugs_through_cards_json(tmp_path):
    stats = _fake_stats(tmp_path)
    entry, unit = brc.stats_entry("bats", stats)
    assert entry["summon_character"] == "Bat" and unit["name"] == "Bat"
    assert brc.stats_entry("not-a-card", stats) == (None, None)
    assert brc.KEY_ALIASES["p-e-k-k-a"] == "pekka"


def test_missing_stats_dir_falls_back_to_manual(tmp_path):
    assert brc.load_stats(tmp_path) is None
    roles, report = brc.build_roles(None, {"zap": brc.spell("small", 2.5, ["swarm"], ["spell_small"])}, {"zap": {}})
    assert roles["zap"]["source"] == "manual" and report["disagreements"] == []


def test_manual_wins_override_stats_and_are_reported_as_kept(tmp_path):
    stats = _fake_stats(tmp_path)
    manual = {"knight": brc.troop("air_ground", "medium", "medium", "long", "very_fast", 1, ["mini_tank"], flying=True)}
    cards = {"knight": {"rarity": "Common"}}
    wins = {("knight", "range"), ("knight", "targets")}
    roles, report = brc.build_roles(stats, manual, cards, wins=wins)
    k = roles["knight"]
    assert (k["range"], k["targets"]) == ("long", "air_ground")               # kept manual
    assert (k["flying"], k["speed"]) == (False, "medium")                     # not in wins: stats still win
    assert sorted(report["kept_manual"]) == ["knight.range: manual='long' stats='melee'", "knight.targets: manual='air_ground' stats='ground'"]
    assert sorted(d.split(":")[0] for d in report["disagreements"]) == ["knight.flying", "knight.speed"]
    roles2, report2 = brc.build_roles(stats, manual, cards, wins=set())       # without wins the stats take over
    assert (roles2["knight"]["range"], roles2["knight"]["targets"]) == ("melee", "ground") and report2["kept_manual"] == []


def test_manual_wins_entries_are_valid_and_survive_in_the_committed_json():
    assert brc.MANUAL_WINS and brc.check_manual(brc.MANUAL) == []
    roles = load_roles()
    for slug, field in brc.MANUAL_WINS:
        assert roles[slug][field] == brc.MANUAL[slug][field], (slug, field)
    bad = brc.check_manual(brc.MANUAL, wins={("knight", "range"), ("zap", "range"), ("knight", "hp_class")})
    assert sorted(bad) == ["MANUAL_WINS knight.hp_class: not a stats-derived field of a manual card",
                           "MANUAL_WINS zap.range: not a stats-derived field of a manual card"]


@pytest.mark.skipif(not (brc.DEFAULT_STATS_DIR / "cards_stats.json").exists(), reason="cr-api-data not fetched")
def test_every_manual_wins_entry_still_differs_from_the_stats():
    _, report = brc.build_roles(brc.load_stats(brc.DEFAULT_STATS_DIR), brc.MANUAL, load_cards())
    kept = {line.split(":")[0] for line in report["kept_manual"]}
    assert kept == {f"{slug}.{field}" for slug, field in brc.MANUAL_WINS}      # none is stale (equal to the stats)


@pytest.mark.skipif(not (brc.DEFAULT_STATS_DIR / "cards_stats.json").exists(), reason="cr-api-data not fetched")
def test_committed_json_matches_a_rebuild_from_the_stats():
    roles, _ = brc.build_roles(brc.load_stats(brc.DEFAULT_STATS_DIR), brc.MANUAL, load_cards())
    assert roles == load_roles()
