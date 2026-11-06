"""Build knowledge graph v2 (D-048, MAC-AK K1): db/kg2.sqlite from the v1 graph, the KB and the K2 / K3 mined files.

  python3 tools/data/build_kg_v2.py [--dataset DIR] [--out db/kg2.sqlite] [--kb DIR]

Steps: (1) v1 nodes / edges (ANSWERED_BY dropped; K2 supersedes it); (2) per-card / unit stats at every level, from
clashbot/resources/unit_stats.json through `clashbot.unit_stats.stat_at` when MAC-AH's module is there, else parsed
from the KB "Stats by level" tables; (3) hero ability nodes + HAS_ABILITY; (4) SIMILAR_ROLE from card_roles;
(5) K2 (answers / situations / sequences) and K3 (deck_cooc) files from <dataset>/kg2/ when present; (6) meta counts.
Missing K2 / K3 files are logged and skipped. Only the output database is written.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR, KB, KG2_PATH, RESOURCES  # noqa: E402
from clashbot.kg2 import SCHEMA  # noqa: E402
from tools.data.kb_markdown import first_table, parse_frontmatter, section  # noqa: E402

log = logging.getLogger("build_kg_v2")

UNIT_STATS_JSON = RESOURCES / "unit_stats.json"
V1_KEEP = frozenset({"COUNTERS", "SYNERGY", "SUBSTITUTE", "IN_DECK", "HAS_ARCHETYPE", "SPAWNED_BY", "EVOLUTION_OF",
                     "HERO_OF", "GAME_OWN_CARD", "GAME_OPP_CARD", "GAME_DECK"})
V1_WIKI = frozenset({"COUNTERS", "SYNERGY", "SUBSTITUTE"})       # wiki bullets; the rest are KB / record structure
NODE_PREFIX_RENAME = {"evo:": "evolution:"}                      # v1 used `evo:`, the v2 contract says `evolution:`
STAT_LEVELS = range(1, 17)
SIM_TOP = 3
SIM_MIN = 0.6


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL file ([] when missing); unparsable lines are skipped and counted in the log."""
    if not path.exists():
        return []
    rows, bad = [], 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    if bad:
        log.warning("%s: %d unparsable lines skipped", path, bad)
    return rows


def _rename(node_id: str) -> str:
    for old, new in NODE_PREFIX_RENAME.items():
        if node_id.startswith(old):
            return new + node_id[len(old):]
    return node_id


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def edge_row(src: str, rel: str, dst: str, weight: float, support: int, source: str, props: dict | None = None) -> tuple:
    return (src, rel, dst, float(weight), int(support), source, json.dumps(props or {}, ensure_ascii=False))


# ------------------------------------------------------------------------------------------ 1. v1 graph
def load_v1(dataset: Path) -> tuple[list[dict], list[tuple]]:
    """v1 nodes (as dicts) and the kept v1 edges (as rows), `evo:` renamed to `evolution:`, dangling edges dropped."""
    raw_nodes = read_jsonl(dataset / "kg" / "nodes.jsonl")
    if not raw_nodes:
        log.warning("no v1 nodes at %s", dataset / "kg" / "nodes.jsonl")
    nodes = [{"id": _rename(n["id"]), "type": n["type"], "name": n.get("name"), "props": n.get("props") or {}}
             for n in raw_nodes]
    ids = {n["id"] for n in nodes}
    edges, dropped, dangling = [], Counter(), 0
    for e in read_jsonl(dataset / "kg" / "edges.jsonl"):
        rel = e["rel"]
        if rel not in V1_KEEP:
            dropped[rel] += 1
            continue
        src, dst = _rename(e["src"]), _rename(e["dst"])
        if src not in ids or dst not in ids:
            dangling += 1
            continue
        props = e.get("props") or {}
        count = props.get("count")
        edges.append(edge_row(src, rel, dst, count if count is not None else 1.0, count or 0,
                              "wiki" if rel in V1_WIKI else "kb_table", props))
    log.info("v1: %d nodes, %d edges kept; dropped by relation %s, dangling %d", len(nodes), len(edges),
             dict(dropped), dangling)
    return nodes, edges


# ------------------------------------------------------------------------------------------ 2. stats
_SUFFIXES = sorted([
    ("crown tower damage per second", "crown_tower_dps"),
    ("hitpoints lost per second", "hp_lost_per_s"),
    ("shield hitpoints", "shield_hp"),
    ("damage per second", "dps"),
    ("crown tower damage", "crown_tower_damage"),
    ("area damage", "area_damage"),
    ("death damage", "death_damage"),
    ("hitpoints", "hp"),
    ("damage", "damage"),
], key=lambda p: -len(p[0]))
_NUM = re.compile(r"^(-?\d[\d,]*(?:\.\d+)?)\s*(?:x\s*\d+)?\s*(?:\((-?\d[\d,]*(?:\.\d+)?)\))?$")


def _num(text: str):
    v = float(text.replace(",", ""))
    return int(v) if v == int(v) else v


def parse_cell(cell: str) -> tuple[float | int | None, float | int | None]:
    """(value, total) of a table cell: `2,458` -> (2458, None); `47 x3 (141)` -> (47, 141); text -> (None, None)."""
    m = _NUM.match((cell or "").strip())
    if not m:
        return None, None
    return _num(m.group(1)), (_num(m.group(2)) if m.group(2) else None)


def norm_header(header: str) -> str:
    """Lower-cased header with the case / plural / spelling variants folded (Hit Points, HP, Damage Per Sec, DPS ...)."""
    h = re.sub(r"\s+", " ", header.strip().lower())
    h = re.sub(r"\bhit ?points?\b|\bhp\b", "hitpoints", h)
    h = re.sub(r"\bdamages\b|\bdmg\b", "damage", h)
    h = re.sub(r"\bper sec\b", "per second", h)
    h = re.sub(r"\bdps\b", "damage per second", h)
    return h


def split_header(header: str) -> tuple[str, str | None]:
    """(unit prefix, canonical stat) of a header: `Golemite Death Damage` -> ('golemite', 'death_damage');
    `Hitpoints` -> ('', 'hp'); a header that is not one of the known stats -> (normalised header, None)."""
    h = norm_header(header)
    for suffix, canon in _SUFFIXES:
        if h == suffix:
            return "", canon
        if h.endswith(" " + suffix):
            return h[:-len(suffix) - 1].strip(), canon
    return h, None


def parse_stats_table(card: str, rows: list[dict]) -> dict[tuple[int, str], dict]:
    """{(level, unit): props} from a KB "Stats by level" table (rows as `first_table` returns them).

    Columns Hitpoints / Damage / Damage per second / Death Damage / Area Damage / Crown Tower Damage (and shield hp,
    hp lost per second) go to hp / damage / dps / death_damage / area_damage / crown_tower_damage / ... . A prefix that
    has its own Hitpoints column is another unit (`Golemite Hitpoints` -> unit golemite; the card's own unit is named
    after the card); any other prefix is a qualifier kept on the main unit (`Ram Charge Damage` -> ram_charge_damage).
    `47 x3 (141)` cells give the per-hit value and a `<stat>_total`."""
    if not rows:
        return {}
    level_key = next((h for h in rows[0] if h.strip().lower() == "level"), None)
    if level_key is None:
        return {}
    headers = [h for h in rows[0] if h != level_key]
    parsed = {h: split_header(h) for h in headers}
    unit_prefixes = [p for h in headers for p, c in [parsed[h]] if c == "hp" and p]
    plain = any(not p and c in ("hp", "damage", "dps") for p, c in parsed.values())
    main_prefix = next((p for p in unit_prefixes if slugify(p) == card), None)
    if main_prefix is None and not plain and unit_prefixes:
        main_prefix = unit_prefixes[0]                     # no plain columns: the first listed unit is the card's
    out: dict[tuple[int, str], dict] = {}
    for row in rows:
        try:
            level = int(float(row[level_key]))
        except (TypeError, ValueError):
            continue
        for h in headers:
            prefix, canon = parsed[h]
            value, total = parse_cell(row.get(h, ""))
            if value is None:
                continue
            if canon is None:
                unit, key = card, _snake(prefix)           # unknown column: kept as is on the main unit
            elif not prefix or prefix == main_prefix:
                unit, key = card, canon
            elif prefix in unit_prefixes:
                unit, key = slugify(prefix), canon
            else:
                unit, key = card, _snake(prefix) + "_" + canon
            props = out.setdefault((level, unit), {})
            props[key] = value
            if total is not None:
                props[key + "_total"] = total
    return out


def kb_stats_rows(kb: Path) -> list[tuple[str, int, str, dict]]:
    """(card, level, unit, props) for every card file with a "Stats by level" table."""
    rows = []
    for p in sorted((kb / "cards").glob("*.md")):
        fm, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        slug = fm.get("slug") or p.stem
        table = first_table(section(body, "Stats by level") or "")
        for (level, unit), props in sorted(parse_stats_table(slug, table).items()):
            rows.append((slug, level, unit, props))
    return rows


def unit_stats_rows(json_path: Path, stat_at, levels=STAT_LEVELS) -> list[tuple[str, int, str, dict]]:
    """(card, level, unit, props) from MAC-AH's `stat_at(card, level)` for every card in unit_stats.json.

    `stat_at` returns {"units": [{name, count, hp, damage, dps, ...}], "spell": {damage, crown_tower_damage,
    radius_tiles}}; the first unit is stored under the card slug, others under their slugified name. Levels below the
    card's `start_level` are skipped (stat_at extrapolates them)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cards = data.get("cards", data)
    rows = []
    for card in sorted(cards):
        rec = cards[card]
        start = rec.get("start_level", 1) if isinstance(rec, dict) else 1
        for level in levels:
            if level < start:
                continue
            st = stat_at(card, level)
            if not st:
                continue
            used: set[str] = set()
            main: dict | None = None
            for i, u in enumerate(st.get("units") or []):
                name = card if i == 0 else slugify(u.get("name") or f"unit{i}")
                if name in used:
                    name = f"{name}-{i}"
                used.add(name)
                props = {k: v for k, v in u.items() if k != "name" and v is not None and not isinstance(v, (dict, list))}
                rows.append((card, level, name, props))
                main = main if main is not None else props
            sp = st.get("spell")
            if sp:
                props = {"damage": sp.get("damage"), "area_damage": sp.get("damage"),
                         "crown_tower_damage": sp.get("crown_tower_damage"), "radius_tiles": sp.get("radius_tiles")}
                props = {k: v for k, v in props.items() if v is not None}
                if main is not None:
                    main.update({"spell_" + k: v for k, v in props.items()})
                else:
                    rows.append((card, level, card, props))
    return rows


def load_stats(kb: Path, unit_stats_json: Path = UNIT_STATS_JSON, stat_at=None) -> tuple[list[tuple], str]:
    """(rows, path name): `unit_stats` when the JSON exists and `clashbot.unit_stats.stat_at` imports, else `kb_tables`."""
    if unit_stats_json.exists():
        if stat_at is None:
            try:
                from clashbot.unit_stats import stat_at as stat_at_fn
                stat_at = stat_at_fn
            except Exception as exc:
                log.warning("unit_stats.json exists but clashbot.unit_stats is not importable (%s); using KB tables", exc)
        if stat_at is not None:
            rows = unit_stats_rows(unit_stats_json, stat_at)
            if rows:
                return rows, "unit_stats"
            log.warning("unit_stats gave no rows; using KB tables")
    return kb_stats_rows(kb), "kb_tables"


# ------------------------------------------------------------------------------------------ 3. abilities
def ability_nodes(kb: Path, known_nodes: set[str]) -> tuple[list[dict], list[tuple]]:
    """`ability:<hero-slug>` nodes (name, cost, description) and `hero:<slug>` HAS_ABILITY edges from heroes/*-hero.md.
    A hero node missing from the v1 graph is added. musketeer-hero -> Trusty Turret, cost 3."""
    nodes, edges = [], []
    for p in sorted((kb / "heroes").glob("*-hero.md")):
        fm, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        name = fm.get("ability_name")
        if not name:
            continue
        slug = fm.get("slug") or p.stem
        sec = section(body, f"Ability: {name}") or ""
        m = re.search(r"^\*(.+?)\*", sec, re.M)
        cost = fm.get("ability_cost")
        cost = _num(cost) if cost and re.fullmatch(r"\d+(\.\d+)?", cost) else cost
        hero_id, ab_id = f"hero:{slug}", f"ability:{slug}"
        if hero_id not in known_nodes:
            nodes.append({"id": hero_id, "type": "hero", "name": fm.get("name"),
                          "props": {"base_card_slug": fm.get("base_card_slug")}})
        nodes.append({"id": ab_id, "type": "ability", "name": name, "props": {
            "cost": cost, "hero": slug, "base_card_slug": fm.get("base_card_slug"),
            "description": m.group(1).strip() if m else None,
            "total_elixir_cost": fm.get("total_elixir_cost")}})
        edges.append(edge_row(hero_id, "HAS_ABILITY", ab_id, 1.0, 0, "kb_table"))
    return nodes, edges


# ------------------------------------------------------------------------------------------ 4. SIMILAR_ROLE
_HP = ("swarm", "light", "medium", "tank")
_DPS = ("low", "medium", "high")
_RANGE = ("melee", "short", "long")
ROLE_W = {"kind": 1.0, "targets": 1.0, "flying": 1.0, "splash": 1.0, "hp": 0.5, "dps": 0.5, "range": 0.7,
          "role": 1.0, "spell": 1.0}


def role_vector(role: dict) -> dict[str, float]:
    """Sparse feature vector of a card_roles entry: kind (troop / building / spell), targets, flying, splash, the
    ordinal hp_class / dps_class (thermometer coded, so medium is nearer tank than swarm), range, every role in the
    roles set, and for spells the spell class, whether they hit air and which hp classes they kill."""
    w = ROLE_W
    f: dict[str, float] = {"kind:" + str(role.get("type")): w["kind"]}
    if role.get("targets"):
        f["targets:" + role["targets"]] = w["targets"]
    if role.get("flying"):
        f["flying"] = w["flying"]
    if role.get("splash"):
        f["splash"] = w["splash"]
    for field, order, key in (("hp_class", _HP, "hp"), ("dps_class", _DPS, "dps")):
        if role.get(field) in order:
            for i in range(order.index(role[field]) + 1):
                f[f"{key}>={order[i]}"] = w[key]
    if role.get("range") in _RANGE:
        f["range:" + role["range"]] = w["range"]
    for r in role.get("roles") or []:
        f["role:" + r] = w["role"]
    if role.get("type") == "spell":
        f["spell_class:" + str(role.get("spell_class"))] = w["spell"]
        if role.get("hits_air", True) is not False:
            f["hits_air"] = w["spell"]
        for hp in role.get("kills") or []:
            f["kills:" + hp] = w["spell"]
    return f


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    num = sum(v * b[k] for k, v in a.items() if k in b)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def similar_role_edges(roles: dict[str, dict], card_nodes: dict[str, dict], top: int = SIM_TOP,
                       min_sim: float = SIM_MIN) -> list[tuple]:
    """SIMILAR_ROLE edges: for every card with a node, its `top` most similar cards (cosine >= `min_sim`), weight =
    similarity; ties go to the nearer elixir cost, then the slug. Two cards that pick each other share a weight."""
    slugs = sorted(s for s in roles if f"card:{s}" in card_nodes)
    vecs = {s: role_vector(roles[s]) for s in slugs}

    def cost(s: str):
        return (card_nodes[f"card:{s}"].get("props") or {}).get("elixir_cost")

    out = []
    for a in slugs:
        cands = []
        for b in slugs:
            if b == a:
                continue
            sim = cosine(vecs[a], vecs[b])
            if sim >= min_sim:
                ca, cb = cost(a), cost(b)
                gap = abs(float(ca) - float(cb)) if ca is not None and cb is not None else 99.0
                cands.append((-round(sim, 6), gap, b, sim))
        cands.sort()
        for rank, (_, _, b, sim) in enumerate(cands[:top], 1):
            out.append(edge_row(f"card:{a}", "SIMILAR_ROLE", f"card:{b}", round(sim, 4), 0, "role_transfer",
                                {"sim": round(sim, 4), "rank": rank}))
    return out


# ------------------------------------------------------------------------------------------ 5. K2 / K3 files
def k2_graph(kg2_dir: Path, card_nodes: dict[str, dict]) -> tuple[list[dict], list[tuple], dict]:
    """K2 files -> ANSWERED_BY (weight held_rate x log1p(support), props = the aggregate), PLAYED_AFTER (then -> first)
    and `situation:<key>` nodes. Rows naming a card that has no node are dropped and counted."""
    nodes, edges, info = [], [], {}
    answers = read_jsonl(kg2_dir / "answers.jsonl")
    sits = read_jsonl(kg2_dir / "situations.jsonl")
    seqs = read_jsonl(kg2_dir / "sequences.jsonl")
    info.update(answers_file=bool(answers), situations_file=bool(sits), sequences_file=bool(seqs))
    skipped = 0
    for a in answers:
        src, dst = f"card:{a.get('threat')}", f"card:{a.get('answer')}"
        if src not in card_nodes or dst not in card_nodes:
            skipped += 1
            continue
        support = int(a.get("support") or 0)
        held = a.get("held_rate")
        weight = (0.5 if held is None else float(held)) * math.log1p(support)
        edges.append(edge_row(src, "ANSWERED_BY", dst, weight, support, "pro_mined", a))
    for s in sits:
        key = s.get("key")
        if key:
            nodes.append({"id": f"situation:{key}", "type": "situation", "name": key, "props": s})
    for q in seqs:
        first, then = f"card:{q.get('first')}", f"card:{q.get('then')}"
        if first not in card_nodes or then not in card_nodes:
            skipped += 1
            continue
        support = int(q.get("support") or 0)
        edges.append(edge_row(then, "PLAYED_AFTER", first, support, support, "pro_mined", q))
    info["k2_rows_skipped_unknown_card"] = skipped
    if not (answers or sits or seqs):
        log.info("no K2 files in %s (answers / situations / sequences); skipped", kg2_dir)
    return nodes, edges, info


def k3_edges(path: Path, card_nodes: dict[str, dict]) -> tuple[list[tuple], dict]:
    """deck_cooc.json -> CO_OCCURS a -> b (weight P(b | a), props count and lift)."""
    if not path.exists():
        log.info("no K3 file %s; skipped", path)
        return [], {"deck_cooc_file": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs, counts = data.get("pair_counts") or {}, data.get("card_counts") or {}
    # card_counts count each deck in `decks` once (K3, count_power 0); without the list, estimate from 8 cards a deck
    n_decks = float(len(data.get("decks") or [])) or (sum(counts.values()) / 8.0 if counts else 0.0)
    names = sorted(set(pairs) | {b for d in pairs.values() for b in d})
    edges = []
    for a in names:
        if f"card:{a}" not in card_nodes:
            continue
        for b in names:
            n = max((pairs.get(a) or {}).get(b, 0), (pairs.get(b) or {}).get(a, 0))
            if a == b or n <= 0 or f"card:{b}" not in card_nodes:
                continue
            denom = counts.get(a) or max((pairs.get(a) or {}).values(), default=0) or 1
            p = n / denom
            base = (counts.get(b) or 0) / n_decks if n_decks else 0.0
            edges.append(edge_row(f"card:{a}", "CO_OCCURS", f"card:{b}", round(p, 4), n, "deck_cooc",
                                  {"count": n, "lift": round(p / base, 3) if base else None}))
    return edges, {"deck_cooc_file": True}


# ------------------------------------------------------------------------------------------ build
def build(dataset: Path, out: Path, kb: Path = KB, roles: dict | None = None,
          unit_stats_json: Path = UNIT_STATS_JSON, stat_at=None, kg2_dir: Path | None = None) -> dict:
    """Write `out` (atomically) and return the summary that is also stored in `meta`."""
    t0 = time.time()
    kg2_dir = kg2_dir or dataset / "kg2"
    if roles is None:
        from clashbot.card_roles import load_roles
        roles = load_roles()

    nodes, edges = load_v1(dataset)
    node_by_id = {n["id"]: n for n in nodes}

    ab_nodes, ab_edges = ability_nodes(kb, set(node_by_id))
    for n in ab_nodes:
        node_by_id[n["id"]] = n
    edges += ab_edges

    card_nodes = {i: n for i, n in node_by_id.items() if n["type"] == "card"}
    edges += similar_role_edges(roles, card_nodes)

    k2_nodes, k2_edges, k2_info = k2_graph(kg2_dir, card_nodes)
    for n in k2_nodes:
        node_by_id[n["id"]] = n
    edges += k2_edges
    k3, k3_info = k3_edges(kg2_dir / "deck_cooc.json", card_nodes)
    edges += k3

    stat_rows, stats_path = load_stats(kb, unit_stats_json, stat_at)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?)", [
            (n["id"], n["type"], n.get("name"), json.dumps(n.get("props") or {}, ensure_ascii=False))
            for n in node_by_id.values()])
        conn.executemany("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?)", edges)
        conn.executemany("INSERT OR REPLACE INTO stats VALUES (?, ?, ?, ?)", [
            (c, lv, u, json.dumps(p, ensure_ascii=False)) for c, lv, u, p in stat_rows])
        by_rel = Counter(e[1] for e in edges)
        by_source = Counter(e[5] for e in edges)
        by_type = Counter(n["type"] for n in node_by_id.values())
        n_stat_cards = len({r[0] for r in stat_rows})
        summary = {
            "version": "kg2-1", "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_seconds": round(time.time() - t0, 2), "stats_path": stats_path,
            "nodes": dict(sorted(by_type.items())), "edges_by_rel": dict(sorted(by_rel.items())),
            "edges_by_source": dict(sorted(by_source.items())), "stats_rows": len(stat_rows),
            "stats_cards": n_stat_cards, **k2_info, **k3_info,
        }
        meta = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in summary.items()}
        for rel, n in by_rel.items():
            meta[f"count.rel.{rel}"] = str(n)
        for src, n in by_source.items():
            meta[f"count.source.{src}"] = str(n)
        conn.executemany("INSERT INTO meta VALUES (?, ?)", sorted(meta.items()))
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, out)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--out", type=Path, default=KG2_PATH)
    ap.add_argument("--kb", type=Path, default=KB)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = build(args.dataset, args.out, kb=args.kb)
    print("nodes:", s["nodes"])
    print("edges by relation:", s["edges_by_rel"])
    print("edges by source:", s["edges_by_source"])
    print(f"stats: {s['stats_rows']} rows, {s['stats_cards']} cards, path={s['stats_path']}")
    print("K2 files:", {k: s[k] for k in ("answers_file", "situations_file", "sequences_file")},
          "K3 file:", s["deck_cooc_file"])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
