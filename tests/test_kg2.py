"""MAC-AK K1: knowledge graph v2 store, builder and query API (fixture KB + fixture K2 / K3 files in tmp)."""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.card_roles import load_roles  # noqa: E402
from clashbot.kg2 import KG2, SOURCE_WEIGHTS, WIKI_WEIGHT_WITH_SIM, elixir_band, eta_band, situation_key  # noqa: E402
from tools.data import build_kg_v2 as B  # noqa: E402

REAL_DB = ROOT / "db" / "kg2.sqlite"
REAL_KB = ROOT / "knowledge_base"
REAL_UNIT_STATS = ROOT / "clashbot" / "resources" / "unit_stats.json"
HAVE_UNIT_STATS = REAL_UNIT_STATS.exists() and (REAL_KB / "cards").exists()
CARDS = {"balloon": 5, "musketeer": 4, "cannon": 3, "tesla": 4, "inferno-tower": 5, "knight": 3, "valkyrie": 4,
         "zap": 2, "fireball": 4, "golem": 8, "bats": 2, "arrows": 3, "night-witch": 4, "lightning": 6, "giant": 5}
OUR = ["musketeer", "tesla", "zap", "knight"]

CARD_MD = """---
name: {name}
slug: {slug}
elixir_cost: {cost}
---
# {name}

## Stats by level

**Card Statistics**

{table}

## Strong against
"""
TABLES = {
    "balloon": """| Level | Hitpoints | Damage | Damage per second | Death Damage |
|---|---|---|---|---|
| 14 | 2235 | 852 | 426 | 319 |
| 15 | 2,458 | 937 | 468 | 351 |
| 16 | 2704 | 1031 | 515 | 387 |""",
    "bats": """| Level | Hit Points | Damage | Damage Per Second |
|---|---|---|---|
| 15 | 119 | 119 | 91 |""",
    "arrows": """| Level | Area Damage | Crown Tower Damage |
|---|---|---|
| 15 | 83 x3 (249) | 21 x3 (63) |""",
    "golem": """| Level | Golem Hitpoints | Golem Damage | Golem Damage per second | Golem Death Damage | Golemite Hitpoints | Golemite Damage |
|---|---|---|---|---|---|---|
| 15 | 7496 | 457 | 182 | 329 | 1521 | 123 |""",
    "giant": """| Level | Hitpoints | Damage | Damage per second |
|---|---|---|---|
| 15 | 5810 | 370 | 246 |""",
}
HERO_MD = """---
name: Heroic Musketeer
slug: musketeer-hero
base_card_slug: musketeer
elixir_cost: 4
ability_name: Trusty Turret
ability_cost: 3
total_elixir_cost: 7
---
# Heroic Musketeer

## Ability: Trusty Turret

*Spawns a short-range, rapid-firing auto-turret that deals landing damage in front of the Musketeer.* (costs 3 Elixir; 4 + 3 = 7 total).

### Ability: Trusty Turret

> "Spawns a short-range auto-turret in front of the Musketeer."
"""


def _jl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _counter(src, dst, count=1, source="wiki_weak_against"):
    return {"src": f"card:{src}", "rel": "COUNTERS", "dst": f"card:{dst}", "props": {"source": source, "count": count}}


def make_kb(tmp: Path) -> Path:
    kb = tmp / "kb"
    (kb / "cards").mkdir(parents=True, exist_ok=True)
    (kb / "heroes").mkdir(exist_ok=True)
    for slug, table in TABLES.items():
        (kb / "cards" / f"{slug}.md").write_text(
            CARD_MD.format(name=slug.title(), slug=slug, cost=CARDS[slug], table=table), encoding="utf-8")
    (kb / "heroes" / "musketeer-hero.md").write_text(HERO_MD, encoding="utf-8")
    return kb


def make_dataset(tmp: Path, with_k2: bool = True, with_k3: bool = True) -> Path:
    ds = tmp / "ds"
    nodes = [{"id": f"card:{s}", "type": "card", "name": s, "props": {"elixir_cost": float(c)}}
             for s, c in CARDS.items()]
    nodes += [
        {"id": "evo:tesla-evolution", "type": "evolution", "name": "Evolved Tesla", "props": {"base_card_slug": "tesla"}},
        {"id": "hero:musketeer-hero", "type": "hero", "name": "Heroic Musketeer", "props": {}},
        {"id": "deck:d1", "type": "deck", "name": "d1", "props": {}},
        {"id": "archetype:cycle", "type": "archetype", "name": "Cycle", "props": {}},
        {"id": "game:g1", "type": "game", "name": "g1", "props": {}},
    ]
    edges = [
        _counter("musketeer", "balloon", 2), _counter("tesla", "balloon", 1), _counter("zap", "bats", 1),
        _counter("knight", "giant", 1),
        {"src": "card:tesla", "rel": "SYNERGY", "dst": "card:knight", "props": {"source": "wiki_notes", "count": 1}},
        {"src": "card:tesla", "rel": "IN_DECK", "dst": "deck:d1", "props": {}},
        {"src": "deck:d1", "rel": "HAS_ARCHETYPE", "dst": "archetype:cycle", "props": {"role": "primary"}},
        {"src": "evo:tesla-evolution", "rel": "EVOLUTION_OF", "dst": "card:tesla", "props": {}},
        {"src": "hero:musketeer-hero", "rel": "HERO_OF", "dst": "card:musketeer", "props": {}},
        {"src": "game:g1", "rel": "GAME_OPP_CARD", "dst": "card:balloon", "props": {}},
        {"src": "card:balloon", "rel": "ANSWERED_BY", "dst": "card:zap", "props": {"support": 9, "confidence": 0.1}},
        {"src": "card:balloon", "rel": "COUNTERS", "dst": "card:ghost-node", "props": {"count": 1}},     # dangling
    ]
    _jl(ds / "kg" / "nodes.jsonl", nodes)
    _jl(ds / "kg" / "edges.jsonl", edges)
    kg2 = ds / "kg2"
    if with_k2:
        _jl(kg2 / "answers.jsonl", [
            {"threat": "balloon", "answer": "cannon", "support": 22, "held": 11, "held_rate": 0.5, "mean_trade": 0.5,
             "mean_dt_s": 2.0, "offset": {"dx_med": 1.0, "dy_med": -1.0, "dx_iqr": 1.0, "dy_iqr": 1.0},
             "eta_at_answer_med": 6.5, "dist_at_answer_med": 9.0},
            {"threat": "balloon", "answer": "musketeer", "support": 48, "held": 34, "held_rate": 0.7,
             "mean_trade": 1.0, "mean_dt_s": 3.0, "offset": {"dx_med": 0.0, "dy_med": -2.0, "dx_iqr": 1.0, "dy_iqr": 2.0},
             "eta_at_answer_med": 5.0, "dist_at_answer_med": 8.0},
            {"threat": "balloon", "answer": "fireball", "support": 4, "held": None, "held_rate": None},
            {"threat": "giant", "answer": "inferno-tower", "support": 10, "held_rate": 1.0},
            {"threat": "balloon", "answer": "no-such-card", "support": 3, "held_rate": 1.0},               # dropped
        ])
        _jl(kg2 / "situations.jsonl", [
            {"key": "balloon+giant|same|3-6|4-6", "threat_cards": ["balloon", "giant"], "lane_rel": "same",
             "eta_band": "3-6", "elixir_band": "4-6", "n": 12, "answers": {"musketeer": {"n": 8, "held_rate": 0.75}}},
            {"key": "balloon|split|6-10|7-10", "threat_cards": ["balloon"], "lane_rel": "split",
             "eta_band": "6-10", "elixir_band": "7-10", "n": 30, "answers": {"tesla": {"n": 9, "held_rate": 0.6}}},
        ])
        _jl(kg2 / "sequences.jsonl", [
            {"who": "own", "first": "giant", "then": "musketeer", "support": 7, "lag_med_s": 2.5}])
    if with_k3:
        (kg2).mkdir(parents=True, exist_ok=True)
        (kg2 / "deck_cooc.json").write_text(json.dumps({
            "cards": ["night-witch", "balloon", "lightning", "zap"],
            "pair_counts": {"night-witch": {"balloon": 4, "lightning": 2}, "balloon": {"night-witch": 4, "zap": 1},
                            "lightning": {"night-witch": 2}},
            "card_counts": {"night-witch": 5, "balloon": 5, "lightning": 2, "zap": 1},
            "decks": [["night-witch", "balloon"]] * 5}), encoding="utf-8")
    return ds


def make_roles() -> dict:
    real = load_roles()
    return {s: real[s] for s in CARDS if s in real}


def build(tmp: Path, **kw) -> Path:
    ds = kw.pop("ds", None) or make_dataset(tmp)
    out = tmp / "kg2.sqlite"
    B.build(ds, out, kb=make_kb(tmp), roles=make_roles(), unit_stats_json=tmp / "no_unit_stats.json", **kw)
    return out


@pytest.fixture()
def db(tmp_path):
    return build(tmp_path)


@pytest.fixture()
def kg(db, tmp_path):
    k = KG2(db, deck_cooc_path=tmp_path / "missing.json")        # no K3 model file: CO_OCCURS fallback
    yield k
    k.close()


def rows(path: Path, sql: str, *args):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


# ------------------------------------------------------------------------------------------ builder
def test_tables_indices_and_v1_copy(db):
    names = {r[0] for r in rows(db, "SELECT name FROM sqlite_master")}
    assert {"nodes", "edges", "stats", "meta", "idx_edges_src", "idx_edges_dst", "idx_stats_card"} <= names
    rel = dict(rows(db, "SELECT rel, COUNT(*) FROM edges GROUP BY rel"))
    for r in ("COUNTERS", "SYNERGY", "IN_DECK", "HAS_ARCHETYPE", "EVOLUTION_OF", "HERO_OF", "GAME_OPP_CARD"):
        assert rel.get(r), r
    # v1 ANSWERED_BY (card:balloon -> card:zap, support 9) is superseded by K2's; the dangling COUNTERS is dropped
    answered = rows(db, "SELECT dst, source FROM edges WHERE rel = 'ANSWERED_BY'")
    assert ("card:zap", "pro_mined") not in answered and all(s == "pro_mined" for _, s in answered)
    assert not rows(db, "SELECT 1 FROM edges WHERE dst = 'card:ghost-node'")
    # `evo:` is renamed to the contract's `evolution:` in nodes and edges
    assert rows(db, "SELECT type FROM nodes WHERE id = 'evolution:tesla-evolution'") == [("evolution",)]
    assert rows(db, "SELECT dst FROM edges WHERE src = 'evolution:tesla-evolution'") == [("card:tesla",)]
    # wiki bullets carry their count as weight and support
    assert rows(db, "SELECT weight, support, source FROM edges WHERE src = 'card:musketeer' AND rel = 'COUNTERS'") \
        == [(2.0, 2, "wiki")]
    assert rows(db, "SELECT source FROM edges WHERE rel = 'IN_DECK'") == [("kb_table",)]


def test_meta_counts_and_stats_path(db):
    meta = dict(rows(db, "SELECT key, value FROM meta"))
    assert meta["stats_path"] == "kb_tables"
    assert int(meta["count.rel.COUNTERS"]) == rows(db, "SELECT COUNT(*) FROM edges WHERE rel = 'COUNTERS'")[0][0]
    assert int(meta["count.source.pro_mined"]) == rows(db, "SELECT COUNT(*) FROM edges WHERE source = 'pro_mined'")[0][0]
    assert json.loads(meta["edges_by_rel"])["ANSWERED_BY"] == 4          # the unknown-card row is dropped
    assert meta["built_at"] and meta["answers_file"] == "True" and meta["deck_cooc_file"] == "True"
    assert meta["k2_rows_skipped_unknown_card"] == "1"


def test_missing_k2_and_k3_files_are_fine(tmp_path):
    ds = make_dataset(tmp_path, with_k2=False, with_k3=False)
    out = build(tmp_path, ds=ds)
    meta = dict(rows(out, "SELECT key, value FROM meta"))
    assert meta["answers_file"] == "False" and meta["deck_cooc_file"] == "False"
    assert not rows(out, "SELECT 1 FROM edges WHERE rel IN ('ANSWERED_BY', 'PLAYED_AFTER', 'CO_OCCURS')")
    k = KG2(out, deck_cooc_path=tmp_path / "missing.json")
    assert k.answers_for("balloon", OUR)[0].card == "musketeer"      # wiki only
    assert k.predict_opp_deck(["balloon"]) == []
    assert k.situation(["balloon"], "same", 4.0, 5.0) is None
    k.close()


def test_k2_k3_edges_and_situation_nodes(db):
    pe = rows(db, "SELECT weight, support, source, props FROM edges WHERE src = 'card:balloon' AND dst = 'card:musketeer' "
                  "AND rel = 'ANSWERED_BY'")
    assert len(pe) == 1
    w, support, source, props = pe[0]
    assert source == "pro_mined" and support == 48
    assert w == pytest.approx(0.7 * math.log1p(48))
    assert json.loads(props)["offset"]["dy_med"] == -2.0 and json.loads(props)["held"] == 34      # whole aggregate
    # an unknown held rate counts as 0.5 in the edge weight
    assert rows(db, "SELECT weight FROM edges WHERE dst = 'card:fireball' AND rel = 'ANSWERED_BY'")[0][0] \
        == pytest.approx(0.5 * math.log1p(4))
    assert rows(db, "SELECT COUNT(*) FROM nodes WHERE type = 'situation'") == [(2,)]
    assert rows(db, "SELECT name FROM nodes WHERE id = 'situation:balloon|split|6-10|7-10'")
    assert rows(db, "SELECT src, dst, weight FROM edges WHERE rel = 'PLAYED_AFTER'") == [
        ("card:musketeer", "card:giant", 7.0)]
    co = rows(db, "SELECT weight, support, source, props FROM edges WHERE src = 'card:night-witch' AND dst = 'card:balloon' "
                  "AND rel = 'CO_OCCURS'")[0]
    assert co[0] == pytest.approx(4 / 5) and co[1] == 4 and co[2] == "deck_cooc"
    assert json.loads(co[3])["count"] == 4 and json.loads(co[3])["lift"] == pytest.approx(0.8 / (5 / 5))
    assert rows(db, "SELECT 1 FROM edges WHERE src = 'card:zap' AND dst = 'card:balloon' AND rel = 'CO_OCCURS'")  # reverse of pair


# ------------------------------------------------------------------------------------------ stats
def test_parse_cell_and_headers():
    assert B.parse_cell("2,458") == (2458, None)
    assert B.parse_cell("83 x3 (249)") == (83, 249)
    assert B.parse_cell("12.5") == (12.5, None)
    assert B.parse_cell("Not specified") == (None, None)
    for h, want in {"Hitpoints": ("", "hp"), "Hit Points": ("", "hp"), "HP": ("", "hp"), "Damage": ("", "damage"),
                    "Damage Per Second": ("", "dps"), "Damage per second": ("", "dps"), "DPS": ("", "dps"),
                    "Death Damage": ("", "death_damage"), "Area Damage": ("", "area_damage"),
                    "Crown Tower Damage": ("", "crown_tower_damage"), "Golemite Hitpoints": ("golemite", "hp"),
                    "Ram Charge Damage": ("ram charge", "damage")}.items():
        assert B.split_header(h) == want, h
    assert B.split_header("Healing Per Pulse") == ("healing per pulse", None)


def test_stats_from_kb_tables(db, kg):
    s = kg.stats("balloon", 15)
    assert (s["hp"], s["damage"], s["dps"], s["death_damage"]) == (2458, 937, 468, 351)
    assert s["unit"] == "balloon"
    assert kg.stats("balloon", 16)["hp"] == 2704
    assert kg.stats("balloon", 9) is None and kg.stats("no-such-card", 15) is None
    assert kg.stats_near("balloon", 20)["hp"] == 2704                  # closest level in the table
    assert kg.stats("bats", 15)["dps"] == 91                           # "Hit Points" / "Damage Per Second" variants
    z = kg.stats("arrows", 15)
    assert z["area_damage"] == 83 and z["area_damage_total"] == 249 and z["crown_tower_damage"] == 21
    g = kg.stats("golem", 15)
    assert g["hp"] == 7496 and g["death_damage"] == 329
    assert g["units"]["golemite"]["hp"] == 1521 and g["units"]["golemite"]["damage"] == 123
    assert rows(db, "SELECT unit FROM stats WHERE card = 'golem' ORDER BY unit") == [("golem",), ("golemite",)]


def test_stats_from_unit_stats_when_importable(tmp_path):
    cards = {"balloon": {"start_level": 6, "units": []}, "fireball": {"start_level": 1}}
    js = tmp_path / "unit_stats.json"
    js.write_text(json.dumps({"cards": cards}), encoding="utf-8")
    calls = []

    def fake_stat_at(card, level):
        calls.append((card, level))
        if card == "balloon":
            return {"units": [{"name": "Balloon", "count": 1, "hp": 2458, "damage": 937, "dps": 468.0, "flying": True,
                               "death_spawn": None, "spawn": {"x": 1}}]}
        return {"units": [], "spell": {"damage": 1007, "crown_tower_damage": 303, "radius_tiles": 2.5}}

    rows_, path = B.load_stats(make_kb(tmp_path), js, fake_stat_at)
    assert path == "unit_stats"
    by_key = {(c, lv, u): p for c, lv, u, p in rows_}
    assert ("balloon", 5, "balloon") not in by_key and ("balloon", 6, "balloon") in by_key   # below start_level
    assert ("balloon", 16, "balloon") in by_key and ("fireball", 1, "fireball") in by_key
    b = by_key[("balloon", 15, "balloon")]
    assert b["hp"] == 2458 and b["flying"] is True and "death_spawn" not in b and "spawn" not in b
    assert by_key[("fireball", 15, "fireball")] == {"damage": 1007, "area_damage": 1007, "crown_tower_damage": 303,
                                                    "radius_tiles": 2.5}
    # the builder records the path it used
    out = tmp_path / "kg2.sqlite"
    B.build(make_dataset(tmp_path), out, kb=make_kb(tmp_path), roles=make_roles(), unit_stats_json=js,
            stat_at=fake_stat_at)
    assert dict(rows(out, "SELECT key, value FROM meta"))["stats_path"] == "unit_stats"
    k = KG2(out)
    assert k.stats("balloon", 15)["damage"] == 937
    k.close()


def test_unit_stats_json_without_module_falls_back_to_kb(tmp_path, monkeypatch):
    js = tmp_path / "unit_stats.json"
    js.write_text(json.dumps({"cards": {"balloon": {}}}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "clashbot.unit_stats", None)        # import raises ImportError
    rows_, path = B.load_stats(make_kb(tmp_path), js)
    assert path == "kb_tables" and any(r[0] == "balloon" for r in rows_)


# ------------------------------------------------------------------------------------------ abilities
def test_ability_node_and_edge(db):
    node = rows(db, "SELECT type, name, props FROM nodes WHERE id = 'ability:musketeer-hero'")[0]
    props = json.loads(node[2])
    assert node[:2] == ("ability", "Trusty Turret") and props["cost"] == 3 and props["base_card_slug"] == "musketeer"
    assert props["description"].startswith("Spawns a short-range, rapid-firing auto-turret")
    assert rows(db, "SELECT dst, source FROM edges WHERE src = 'hero:musketeer-hero' AND rel = 'HAS_ABILITY'") == [
        ("ability:musketeer-hero", "kb_table")]


# ------------------------------------------------------------------------------------------ SIMILAR_ROLE
def _sim(db, a, b):
    r = rows(db, "SELECT weight FROM edges WHERE src = ? AND dst = ? AND rel = 'SIMILAR_ROLE'", f"card:{a}", f"card:{b}")
    return r[0][0] if r else None


def test_similar_role_is_sensible_and_symmetric_ish(db):
    # tesla <-> cannon / inferno-tower high; knight <-> valkyrie medium (below the tesla pair)
    assert _sim(db, "tesla", "inferno-tower") > 0.85
    assert _sim(db, "tesla", "cannon") > 0.7
    kv = _sim(db, "knight", "valkyrie")
    assert 0.6 <= kv < _sim(db, "tesla", "inferno-tower")
    assert _sim(db, "knight", "tesla") is None                       # a ground melee troop is not a tesla
    assert _sim(db, "tesla", "inferno-tower") == _sim(db, "inferno-tower", "tesla")
    # every edge >= 0.6, at most 3 per card, weight = similarity, source role_transfer
    per = {}
    for src, w, source in rows(db, "SELECT src, weight, source FROM edges WHERE rel = 'SIMILAR_ROLE'"):
        per[src] = per.get(src, 0) + 1
        assert w >= 0.6 and source == "role_transfer"
    assert max(per.values()) <= 3
    # weights agree wherever both directions exist
    for a, b, w in rows(db, "SELECT src, dst, weight FROM edges WHERE rel = 'SIMILAR_ROLE'"):
        back = _sim(db, b[5:], a[5:])
        assert back is None or back == w


def test_role_similarity_on_the_real_roles():
    roles = load_roles()
    v = {s: B.role_vector(roles[s]) for s in ("tesla", "cannon", "inferno-tower", "knight", "valkyrie", "zap", "arrows")}
    assert B.cosine(v["tesla"], v["inferno-tower"]) > B.cosine(v["tesla"], v["cannon"]) > 0.65
    assert 0.6 < B.cosine(v["knight"], v["valkyrie"]) < B.cosine(v["tesla"], v["inferno-tower"])
    assert B.cosine(v["zap"], v["arrows"]) > 0.9 and B.cosine(v["tesla"], v["zap"]) < 0.3


# ------------------------------------------------------------------------------------------ answers_for
def _pro(support, held):
    return (0.5 if held is None else held) * support / (support + 5)


def test_answers_for_scores_pro_role_transfer_and_wiki(kg, db):
    ans = {a.card: a for a in kg.answers_for("balloon", OUR)}
    # pro: musketeer (support 48, held 0.7) shrunk by 48 / 53
    m = ans["musketeer"]
    assert m.sources["pro"] == pytest.approx(_pro(48, 0.7), abs=1e-3)
    assert m.sources["wiki"] == pytest.approx(1 - 0.5 ** 2, abs=1e-3)         # two wiki bullets
    assert m.support == 48 and m.held_rate == 0.7 and m.via is None
    assert m.offset == {"dx_med": 0.0, "dy_med": -2.0, "dx_iqr": 1.0, "dy_iqr": 2.0} and m.eta_at_answer == 5.0
    assert m.share == pytest.approx(48 / (48 + 22 + 4), abs=1e-3)
    # role transfer: cannon (not ours, pro score) -> tesla via SIMILAR_ROLE
    t = ans["tesla"]
    sim_ct = max(_sim(db, "cannon", "tesla") or 0, _sim(db, "tesla", "cannon") or 0)
    assert sim_ct > 0.7
    assert t.sources["role_transfer"] == pytest.approx(sim_ct * _pro(22, 0.5) * 0.7, abs=1e-3)
    assert t.via == "cannon" and t.support == 22 and t.held_rate == 0.5 and t.offset["dy_med"] == -1.0
    assert t.sources["pro"] == 0.0 and t.sources["wiki"] == pytest.approx(0.5)
    # all four sources exist for this threat except sim (0 until K4): renormalised over pro / role_transfer / wiki
    w = SOURCE_WEIGHTS
    norm = w["pro"] + w["role_transfer"] + w["wiki"]
    assert m.score == pytest.approx((w["pro"] * m.sources["pro"] + w["role_transfer"] * m.sources["role_transfer"]
                                     + w["wiki"] * m.sources["wiki"]) / norm, abs=1e-3)
    assert [a.card for a in kg.answers_for("balloon", OUR)][0] == "musketeer"
    # no evidence at all for a card -> not returned; k limits
    assert "knight" not in ans and "zap" not in ans
    assert len(kg.answers_for("balloon", OUR, k=1)) == 1


def test_answers_for_renormalises_over_missing_sources(kg):
    # giant: wiki says knight counters it, pro says inferno-tower (not ours) -> role transfer needs a similar card
    giant = {a.card: a for a in kg.answers_for("giant", ["knight", "tesla", "inferno-tower"])}
    assert giant["inferno-tower"].sources["pro"] == pytest.approx(_pro(10, 1.0), abs=1e-3)
    assert giant["knight"].sources["wiki"] == pytest.approx(0.5)
    # bats: wiki only -> score is the wiki value itself (the other weights drop out)
    bats = kg.answers_for("bats", ["zap", "tesla"])
    assert [(a.card, a.score) for a in bats] == [("zap", 0.5)] and bats[0].support == 0 and bats[0].via is None
    # nothing known -> empty
    assert kg.answers_for("lightning", OUR) == []
    # a pro-only threat (no wiki bullets): the score is the weighted mean of pro and role_transfer only
    pro_only = kg.answers_for("giant", ["inferno-tower"])
    assert pro_only[0].score == pytest.approx(_pro(10, 1.0), abs=1e-3)


def test_answers_for_sim_source_uses_stops_edges(tmp_path):
    out = build(tmp_path)
    con = sqlite3.connect(out)                                       # a K4 STOPS edge: our knight stops their giant
    con.execute("INSERT INTO edges VALUES ('card:knight', 'STOPS', 'card:giant', 0.8, 0, 'sim', '{}')")
    con.commit()
    con.close()
    k = KG2(out, deck_cooc_path=tmp_path / "missing.json")
    ans = {a.card: a for a in k.answers_for("giant", ["knight", "tesla"])}
    kn = ans["knight"]
    assert kn.sources["sim"] == pytest.approx(0.8) and kn.sources["wiki"] == pytest.approx(0.5)
    # pro is absent (inferno-tower, the pro answer, is not ours) so its weight drops out; role transfer reaches tesla
    assert ans["tesla"].sources["role_transfer"] > 0 and ans["tesla"].via == "inferno-tower"
    w = dict(SOURCE_WEIGHTS, wiki=WIKI_WEIGHT_WITH_SIM)                # a STOPS edge exists: the wiki weight halves
    assert k.source_weights["wiki"] == 0.05
    assert kn.score == pytest.approx((w["wiki"] * 0.5 + w["sim"] * 0.8) / (w["role_transfer"] + w["wiki"] + w["sim"]),
                                     abs=1e-3)
    assert kn.prevented is None and kn.adequate is None                # '{}' props: nothing to quote
    k.close()


# ------------------------------------------------------------------------------------------ opp deck, situation
def test_predict_opp_deck_cooccurrence_fallback(kg):
    p = kg.predict_opp_deck(["night-witch"], top=5)
    cards = [c for c, _ in p]
    assert "night-witch" not in cards and cards[0] == "balloon"
    assert dict(p)["balloon"] == pytest.approx(0.8) and dict(p)["lightning"] == pytest.approx(0.4)   # P(c | seen)
    assert kg.predict_opp_deck(["night-witch", "lightning"])[0][0] == "balloon"
    assert kg.predict_opp_deck([]) == [] and kg.predict_opp_deck(["unknown-card"]) == []


def test_predict_opp_deck_delegates_to_k3_model(db, tmp_path, monkeypatch):
    cooc = tmp_path / "deck_cooc.json"
    cooc.write_text("{}", encoding="utf-8")

    class FakeModel:
        @classmethod
        def load(cls, path):
            assert Path(path) == cooc
            return cls()

        def predict(self, seen, top=8):
            return [("lightning", 0.9), ("zap", 0.4)][:top]

    fake = type(sys)("clashbot.opp_model")
    fake.OppDeckModel = FakeModel
    monkeypatch.setitem(sys.modules, "clashbot.opp_model", fake)
    k = KG2(db, deck_cooc_path=cooc)
    assert k.predict_opp_deck(["balloon"], top=2) == [("lightning", 0.9), ("zap", 0.4)]
    k.close()


def test_situation_nearest(kg):
    assert eta_band(2.0) == "0-3" and eta_band(4.0) == "3-6" and eta_band(8) == "6-10" and eta_band(None) == ">10"
    assert eta_band(10) == "6-10" and eta_band(10.5) == ">10"                  # same bands as K2's situation_key
    assert elixir_band(2) == "0-3" and elixir_band(5.5) == "4-6" and elixir_band(9) == "7-10"
    assert elixir_band(None) == "?" and situation_key(["balloon"], "same", None, None) == "balloon|same|>10|?"
    assert situation_key(["giant", "balloon", "bats"], "same", 4.0, 5.0, {"giant": 5, "balloon": 5, "bats": 2}) \
        == "balloon+giant|same|3-6|4-6"
    s = kg.situation(["giant", "balloon"], "same", 4.0, 5.0)
    assert s["key"] == "balloon+giant|same|3-6|4-6" and s["exact"] and s["n"] == 12
    assert s["answers"]["musketeer"]["held_rate"] == 0.75
    near = kg.situation(["balloon"], "split", 7.0, 9.0)
    assert near["key"] == "balloon|split|6-10|7-10" and near["exact"]
    far = kg.situation(["balloon"], "centre", 1.0, 2.0)                     # nothing exact: closest by threat overlap
    assert far["key"] == "balloon|split|6-10|7-10" and not far["exact"]
    assert kg.situation(["night-witch"], "same", 4.0, 5.0) is None


# ------------------------------------------------------------------------------------------ facts_lines
def _state(units, known=()):
    return {"units": units, "opponent": {"deck_known": list(known)}, "own": {"elixir": 6.0}}


def _unit(card, tile, eta=None, side="enemy"):
    return {"unit_class": card, "card": card, "side": side, "tile": tile,
            "eta_tower": {"tower": "own_left", "s": eta} if eta is not None else None}


def test_facts_lines_content_and_limits(kg):
    state = _state([_unit("bats", [5, 12], 3.0), _unit("balloon", [12, 20], 7.0), _unit("knight", [3, 4], side="ally"),
                    _unit("giant", [9, 30])], known=["night-witch"])
    lines = kg.facts_lines(state, OUR)
    assert 1 <= len(lines) <= 3 and all(len(x) <= 160 and x.startswith("KG: ") for x in lines)
    first = lines[0]
    assert first.startswith("KG: balloon L15 2458hp 937/drop; answers musketeer (pros 65% n=48 held 70%)")
    assert "tesla (role of cannon, n=22)" in first
    assert lines[1].startswith("KG: bats L15 x5 119hp 119/hit; answers zap (wiki)")
    assert lines[2] == "KG: opp likely also has balloon 0.80, lightning 0.40"
    # giant is far on the enemy half (row 30, no ETA): not a threat; our knight is not either
    assert not any("giant" in x or "knight" in x.split(";")[0] for x in lines)
    assert kg.facts_lines(state, OUR, max_lines=1) == lines[:1]
    assert kg.facts_lines({"units": [], "opponent": {"deck_known": []}}, OUR) == []
    assert kg.facts_lines({"state": state}, OUR) == lines                       # a whole dataset record also works


def test_facts_lines_shrink_to_fit(kg, monkeypatch):
    import clashbot.kg2 as mod
    state = _state([_unit("balloon", [12, 20], 7.0)])
    full = kg.facts_lines(state, OUR)[0]
    monkeypatch.setattr(mod, "LINE_MAX", 100)
    short = kg.facts_lines(state, OUR)[0]
    assert len(short) <= 100 < len(full) and short.startswith("KG: balloon L15 2458hp 937/drop; answers")
    monkeypatch.setattr(mod, "LINE_MAX", 40)
    assert len(kg.facts_lines(state, OUR)[0]) <= 40
    # a card without stats and without any answer gives no line
    assert kg.facts_lines(_state([_unit("no-such-card", [9, 10], 4.0)]), OUR) == []


# ------------------------------------------------------------------------------------------ K4a: numeric role vectors
def test_soft_bins_share_between_neighbours_and_sum_to_one():
    mid = B.soft_bin("hp", 1000.0)                                   # between the 900 and 1500 centres, nearer 900
    assert set(mid) == {"hp~900", "hp~1500"} and sum(mid.values()) == pytest.approx(1.0)
    assert mid["hp~900"] > mid["hp~1500"]
    assert B.soft_bin("hp", 10.0) == {"hp~150": 1.0} and B.soft_bin("hp", 1e6) == {"hp~6500": 1.0}
    assert B.soft_bin("range", 1.5)["range~1.5"] == 1.0                    # exactly on a centre
    assert B.soft_bin("speed", 0.0) == {}


def test_numeric_stats_totals_and_levels():
    rows_ = [("swarm", 14, "swarm", {"hp": 100, "dps": 10, "count": 4}),
             ("swarm", 15, "swarm", {"hp": 110, "dps": 11, "count": 4, "range_tiles": 1.2, "damage": 20,
                                     "speed_tiles_s": 2.0}),
             ("swarm", 15, "spawned", {"hp": 50, "dps": 5, "count": 0}),
             ("fireball", 15, "fireball", {"damage": 1000, "area_damage": 1000, "radius_tiles": 2.5})]
    num = B.numeric_stats(rows_)
    assert num["swarm"] == {"hp": 440.0, "dps": 44.0, "range": 1.2, "speed": 2.0}      # level 15, units x count, count 0 skipped
    assert num["fireball"] == {"spell_dmg": 1000.0, "spell_radius": 2.5}
    assert B.numeric_stats(rows_, level=16)["swarm"]["hp"] == 440.0                    # nearest level


def test_numeric_bins_tell_apart_cards_that_share_every_coarse_class():
    roles = {"a": {"type": "troop", "targets": "ground", "hp_class": "medium", "dps_class": "medium", "range": "melee",
                   "roles": ["mini_tank"]}}
    base = B.role_vector(roles["a"])
    big = B.role_vector(roles["a"], {"hp": 2200.0, "dps": 200.0, "range": 1.0, "speed": 1.5})
    small = B.role_vector(roles["a"], {"hp": 1500.0, "dps": 140.0, "range": 1.0, "speed": 1.5})
    assert B.cosine(base, base) == pytest.approx(1.0)
    assert B.cosine(big, small) < 1.0 and B.cosine(big, big) == pytest.approx(1.0)
    assert B.cosine(big, base) > 0.6                                                    # still the same class


@pytest.mark.skipif(not HAVE_UNIT_STATS, reason="unit_stats.json / knowledge_base missing")
def test_knight_is_no_longer_identical_to_prince_and_monk():
    roles = load_roles()
    stat_rows, path = B.load_stats(REAL_KB, REAL_UNIT_STATS)
    assert path == "unit_stats"
    num = B.numeric_stats(stat_rows)
    v = {s: B.role_vector(roles[s], num.get(s)) for s in ("knight", "prince", "monk", "tesla", "inferno-tower", "zap")}
    coarse = {s: B.role_vector(roles[s]) for s in ("knight", "prince", "monk")}
    assert B.cosine(coarse["knight"], coarse["prince"]) == pytest.approx(1.0)           # (the old vectors: identical)
    assert B.cosine(v["knight"], v["prince"]) < 0.97 and B.cosine(v["knight"], v["monk"]) < 0.99
    assert B.cosine(v["tesla"], v["inferno-tower"]) > 0.7 and B.cosine(v["tesla"], v["zap"]) < 0.3


# ------------------------------------------------------------------------------------------ K4a: mechanics edges
UNIT_NODES = [("unit:bat", "bat", "bats"), ("unit:balloon", "balloon", "balloon"), ("unit:giant", "giant", "giant"),
              ("unit:knight", "knight", "knight")]


def build_sim(tmp: Path, **kw) -> Path:
    """The fixture graph built with the real unit_stats.json: mechanics edges (STOPS / KILLS / PULLS) are on."""
    ds = make_dataset(tmp)
    with (ds / "kg" / "nodes.jsonl").open("a", encoding="utf-8") as fh:
        for nid, name, primary in UNIT_NODES:
            fh.write(json.dumps({"id": nid, "type": "unit", "name": name,
                                 "props": {"kind": "card", "primary": primary, "is_evo": False}}) + "\n")
    out = tmp / "kg2_sim.sqlite"
    B.build(ds, out, kb=make_kb(tmp), roles=make_roles(), unit_stats_json=REAL_UNIT_STATS, **kw)
    return out


def _stops(path: Path, src: str, dst: str):
    r = rows(path, "SELECT weight, support, source, props FROM edges WHERE src = ? AND dst = ? AND rel = 'STOPS'",
             f"card:{src}", f"card:{dst}")
    if not r:
        return None
    w, support, source, props = r[0]
    return {"weight": w, "support": support, "source": source, **json.loads(props)}


@pytest.fixture(scope="module")
def sim_db(tmp_path_factory):
    if not HAVE_UNIT_STATS:
        pytest.skip("unit_stats.json / knowledge_base missing")
    return build_sim(tmp_path_factory.mktemp("sim"))


def test_without_unit_stats_there_are_no_mechanics_edges(db):
    assert not rows(db, "SELECT 1 FROM edges WHERE rel IN ('STOPS', 'KILLS', 'PULLS') OR source = 'sim'")
    meta = dict(rows(db, "SELECT key, value FROM meta"))
    assert meta["mechanics"] == "False"


def test_stops_edges_follow_the_lane_simulation(sim_db):
    meta = dict(rows(sim_db, "SELECT key, value FROM meta"))
    assert meta["stats_path"] == "unit_stats" and meta["mechanics"] == "True"
    assert int(meta["count.rel.STOPS"]) > 50
    assert int(meta["count.source.sim"]) == sum(int(meta.get(f"count.rel.{r}", 0)) for r in ("STOPS", "KILLS", "PULLS"))
    musk = _stops(sim_db, "musketeer", "balloon")
    tesla = _stops(sim_db, "tesla", "balloon")
    assert musk["source"] == "sim" and musk["support"] == 0
    assert 0.0 < musk["weight"] <= 1.0 and musk["prevented"] >= 1500 and musk["adequate"] is True
    assert musk["damage_without"] > 2000 and musk["damage_with"] < 600                    # a lone L15 balloon vs the tower
    assert tesla["adequate"] is True and tesla["prevented"] >= 1500 and tesla["level"] == 15
    # a knight cannot touch a flyer; a zap only delays the balloon's first hit (not adequate)
    assert _stops(sim_db, "knight", "balloon") is None
    zap = _stops(sim_db, "zap", "balloon")
    assert zap is None or zap["adequate"] is False
    # weight = prevented per elixir / the reference, so the cheaper equal answer weighs more
    assert musk["weight"] == pytest.approx(min(1.0, musk["prevented"] / musk["elixir"] / B.STOPS_EFFICIENCY_REF), abs=1e-3)
    # a building stops a giant; the knight alone does not stop it in time
    assert _stops(sim_db, "tesla", "giant")["adequate"] is True
    kn = _stops(sim_db, "knight", "giant")
    assert kn["prevented"] > 1000 and kn["adequate"] is False


def test_kills_edges_compare_spell_damage_with_unit_hitpoints(sim_db):
    kills = {(s, d) for s, d in rows(sim_db, "SELECT src, dst FROM edges WHERE rel = 'KILLS'")}
    assert ("card:zap", "unit:bat") in kills and ("card:fireball", "unit:bat") in kills       # L15 bats have 119 hp
    assert ("card:zap", "unit:balloon") not in kills and ("card:fireball", "unit:giant") not in kills
    assert not any(s not in ("card:zap", "card:fireball", "card:arrows", "card:lightning") for s, _ in kills)
    props = json.loads(rows(sim_db, "SELECT props FROM edges WHERE src = 'card:zap' AND dst = 'unit:bat'")[0][0])
    assert props["damage"] >= props["hp"] and props["card"] == "bats" and props["opp_level"] == 15


def test_pulls_edges_building_vs_building_targeter(sim_db):
    pulls = {(s, d) for s, d in rows(sim_db, "SELECT src, dst FROM edges WHERE rel = 'PULLS'")}
    assert ("card:tesla", "card:balloon") in pulls and ("card:cannon", "card:giant") in pulls
    assert not any(d in ("card:knight", "card:bats", "card:musketeer") for _, d in pulls)
    assert not any(s in ("card:knight", "card:balloon") for s, _ in pulls)                # only buildings pull
    q = "SELECT props FROM edges WHERE rel = 'PULLS' AND src = 'card:tesla' AND dst = ?"
    p = json.loads(rows(sim_db, q, "card:giant")[0][0])
    assert p["pull_tile"] == list(B.SIM_PULL_TILE) and p["window_s"] > 2 and p["survives_window"] is True
    assert json.loads(rows(sim_db, q, "card:balloon")[0][0])["flying"] is True


def test_answers_for_uses_stops_and_the_lower_wiki_weight(sim_db, tmp_path):
    k = KG2(sim_db, deck_cooc_path=tmp_path / "missing.json")
    assert k.source_weights["wiki"] == WIKI_WEIGHT_WITH_SIM
    ans = {a.card: a for a in k.answers_for("night-witch", OUR)}                        # no pro, no wiki: the sim carries it
    assert ans and all(a.support == 0 and a.sources["sim"] > 0 for a in ans.values())
    best = next(iter(ans.values()))
    assert best.prevented and best.prevented >= 1000 and best.adequate is True
    assert best.score == pytest.approx(best.sources["sim"])                             # renormalised over the one source
    bal = {a.card: a for a in k.answers_for("balloon", OUR)}
    assert bal["musketeer"].sources["sim"] > 0 and bal["musketeer"].sources["pro"] > 0
    assert bal["musketeer"].prevented >= 1500
    state = _state([_unit("night-witch", [5, 12], 3.0)])
    line = k.facts_lines(state, OUR)[0]
    assert len(line) <= 160 and "sim saves" in line
    # threats are ranked by the simulated tower damage: 5 bats (~120) rank below a night witch (~2,200)
    both = k.facts_lines(_state([_unit("bats", [5, 12], 3.0), _unit("night-witch", [5, 13], 3.0)]), OUR)
    assert both[0].startswith("KG: night-witch") and both[1].startswith("KG: bats")
    assert k._magnitude("night-witch") > k._magnitude("bats") and k._magnitude("balloon") > 2000
    k.close()


# ------------------------------------------------------------------------------------------ real build
@pytest.mark.skipif(not REAL_DB.exists(), reason="db/kg2.sqlite not built")
def test_real_db_balloon_l15():
    k = KG2(REAL_DB)
    s = k.stats("balloon", 15)
    assert s["hp"] == 2458 and s["damage"] == 937
    assert k.answers_for("balloon", ["tesla", "musketeer", "knight", "zap", "fireball"])
    meta = k.meta()
    assert int(meta["count.rel.SIMILAR_ROLE"]) > 100 and int(meta["count.rel.COUNTERS"]) > 1000
    assert meta["stats_path"] in ("kb_tables", "unit_stats")
    if meta.get("mechanics") == "True":                                   # built after K4a: STOPS / KILLS / PULLS are in
        assert int(meta["count.rel.STOPS"]) > 1000 and int(meta["count.rel.KILLS"]) > 50
        assert int(meta["count.rel.PULLS"]) > 50 and int(meta["count.source.sim"]) > 1000
        top = k.answers_for("giant", ["tesla", "mini-p-e-k-k-a", "zap", "musketeer"])[0]
        assert top.card in ("tesla", "mini-p-e-k-k-a") and top.prevented > 2000 and top.adequate is True
        assert k.source_weights["wiki"] == 0.05
    k.close()
