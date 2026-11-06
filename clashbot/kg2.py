"""Knowledge graph v2 (D-048, MAC-AK K1): the SQLite store `db/kg2.sqlite` and its read-only query API.

The store is built by `tools/data/build_kg_v2.py` (v1 nodes / edges + per-level stats + hero abilities + SIMILAR_ROLE +
K2 pro-answer and K3 deck co-occurrence files). `KG2` loads the small relations into memory and answers:

  stats(card, level)                        -> numbers for one card at one level
  answers_for(threat, our_cards)            -> which of our cards answer a threat, with pro evidence
  predict_opp_deck(seen)                    -> likely hidden opponent cards (K3 model, else CO_OCCURS edges)
  situation(threat_cards, lane_rel, eta, elixir) -> nearest mined situation (K2)
  facts_lines(state, our_deck)              -> <= 3 short lines for the LLM context

Pure sqlite3 / json: no model, no network, no screen.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from clashbot.config import DATASET_DIR, KG2_PATH

SCHEMA = """
CREATE TABLE nodes(id TEXT PRIMARY KEY, type TEXT, name TEXT, props TEXT);
CREATE TABLE edges(src TEXT, rel TEXT, dst TEXT, weight REAL, support INTEGER, source TEXT, props TEXT);
CREATE TABLE stats(card TEXT, level INTEGER, unit TEXT, props TEXT, PRIMARY KEY(card, level, unit));
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX idx_edges_src ON edges(src, rel);
CREATE INDEX idx_edges_dst ON edges(dst, rel);
CREATE INDEX idx_stats_card ON stats(card);
"""

SOURCES = ("wiki", "pro_mined", "deck_cooc", "role_transfer", "sim", "kb_table")
SOURCE_WEIGHTS = {"pro": 0.45, "role_transfer": 0.30, "wiki": 0.10, "sim": 0.15}
ROLE_TRANSFER_DISCOUNT = 0.7
SUPPORT_SHRINK = 5.0                 # pro score is shrunk by support / (support + SUPPORT_SHRINK)
NEUTRAL_HELD_RATE = 0.5              # a pro answer whose held rate is unknown (null tower HP)
OPP_LEVEL = 15                       # opponent level shown in facts_lines (MAC-AH levels.json opponent_default)
LINE_MAX = 160                       # facts_lines: characters per line
MAX_THREAT_ROW = 20                  # an enemy unit up to this tile row (just across the river) counts as a threat
MAX_THREAT_ETA_S = 12.0
DECK_SIZE = 8
_EDGE_RELS = ("ANSWERED_BY", "COUNTERS", "SIMILAR_ROLE", "STOPS", "CO_OCCURS")


@dataclass
class Answer:
    card: str
    score: float
    sources: dict                     # {"pro": x, "role_transfer": y, "wiki": z, "sim": w}, each 0..1 before weighting
    support: int = 0                  # events behind the pro (or via) evidence; 0 = wiki / sim only
    held_rate: float | None = None
    offset: dict | None = None        # {dx_med, dy_med, dx_iqr, dy_iqr}, canonical lane (centre-ward dx > 0)
    eta_at_answer: float | None = None
    via: str | None = None            # the pro card whose role this card shares (role transfer)
    share: float | None = None        # support / all pro support against this threat


# ---------------------------------------------------------------------------------- situation keys
# Local copy of K2's `tools.data.mine_pro_answers.situation_key` (the module wins when present and compatible).

def eta_band(eta: float | None) -> str:
    if eta is None or eta > 10:
        return ">10"
    return "0-3" if eta < 3 else "3-6" if eta < 6 else "6-10"


def elixir_band(elixir: float | None) -> str:
    if elixir is None:
        return "?"
    return "0-3" if elixir < 4 else "4-6" if elixir < 7 else "7-10"


def situation_key(threat_cards: list[str], lane_rel: str, eta: float | None, elixir: float | None,
                  elixir_of: dict | None = None) -> str:
    """`a+b|lane_rel|eta_band|elixir_band`: the (up to) 2 costliest threat cards, sorted."""
    top = _top2(threat_cards, elixir_of or {})
    return "|".join(["+".join(top), lane_rel, eta_band(eta), elixir_band(elixir)])


def _top2(cards: list[str], elixir_of: dict) -> list[str]:
    uniq = list(dict.fromkeys(cards))
    uniq.sort(key=lambda c: (-(elixir_of.get(c) or 0.0), c))
    return sorted(uniq[:2])


def _pro_score(props: dict) -> float:
    support = float(props.get("support") or 0)
    held = props.get("held_rate")
    held = NEUTRAL_HELD_RATE if held is None else float(held)
    return held * support / (support + SUPPORT_SHRINK)


def _card_id(slug: str) -> str:
    return slug if slug.startswith("card:") else "card:" + slug


def _slug(node_id: str) -> str:
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


class KG2:
    def __init__(self, path: Path | str = KG2_PATH, deck_cooc_path: Path | str | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"knowledge graph v2 not built: {self.path} (run tools/data/build_kg_v2.py)")
        self.deck_cooc_path = Path(deck_cooc_path) if deck_cooc_path else DATASET_DIR / "kg2" / "deck_cooc.json"
        self._conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._answered: dict[str, dict[str, dict]] = defaultdict(dict)    # threat -> answer -> aggregate props
        self._counters: dict[str, dict[str, float]] = defaultdict(dict)   # threat -> counter -> wiki bullet count
        self._sim: dict[str, dict[str, float]] = defaultdict(dict)        # SIMILAR_ROLE, both directions
        self._stops: dict[str, dict[str, float]] = defaultdict(dict)      # threat -> our card -> weight (K4)
        self._cooc: dict[str, dict[str, float]] = defaultdict(dict)       # a -> b -> P(b | a)
        self._elixir: dict[str, float] = {}
        self._situations: list[dict] = []
        self._stat_cache: dict[tuple[str, int], dict | None] = {}
        self._opp_model = False                                           # False = not tried yet, None = unavailable
        self._roles: dict[str, dict] | None = None
        self._load()

    # ------------------------------------------------------------------------------------ loading
    def _load(self) -> None:
        marks = ",".join("?" * len(_EDGE_RELS))
        rows = self._conn.execute(
            f"SELECT src, rel, dst, weight, support, props FROM edges WHERE rel IN ({marks})", _EDGE_RELS)
        for r in rows:
            if not (r["src"].startswith("card:") and r["dst"].startswith("card:")):
                continue
            src, dst, rel = _slug(r["src"]), _slug(r["dst"]), r["rel"]
            props = json.loads(r["props"]) if r["props"] else {}
            weight = float(r["weight"] if r["weight"] is not None else 1.0)
            if rel == "ANSWERED_BY":
                props.setdefault("support", r["support"] or 0)
                self._answered[src][dst] = props
            elif rel == "COUNTERS":                      # src counters dst (wiki: "src strong against dst")
                self._counters[dst][src] = self._counters[dst].get(src, 0.0) + float(props.get("count") or 1)
            elif rel == "SIMILAR_ROLE":
                for a, b in ((src, dst), (dst, src)):
                    self._sim[a][b] = max(self._sim[a].get(b, 0.0), weight)
            elif rel == "STOPS":                         # our card src stops opponent card dst
                self._stops[dst][src] = max(self._stops[dst].get(src, 0.0), weight)
            elif rel == "CO_OCCURS":
                self._cooc[src][dst] = weight
        for r in self._conn.execute("SELECT id, type, props FROM nodes WHERE type IN ('card', 'situation')"):
            props = json.loads(r["props"]) if r["props"] else {}
            if r["type"] == "card" and props.get("elixir_cost") is not None:
                self._elixir[_slug(r["id"])] = float(props["elixir_cost"])
            elif r["type"] == "situation":
                self._situations.append(props)

    def close(self) -> None:
        self._conn.close()

    def meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self._conn.execute("SELECT key, value FROM meta")}

    def roles(self) -> dict[str, dict]:
        if self._roles is None:
            try:
                from clashbot.card_roles import load_roles
                self._roles = load_roles()
            except Exception:                            # pragma: no cover - resource missing
                self._roles = {}
        return self._roles

    # -------------------------------------------------------------------------------------- stats
    def stats(self, card: str, level: int) -> dict | None:
        """Main-unit numbers of `card` at `level` (hp, damage, dps, death_damage, ... as the store holds them) plus
        `unit` (the unit's name) and, when the card has several units (golem / golemite), `units` = {name: props} of
        the others. None when the card or level is not in the table. A unit class (e.g. `bat`) resolves to its card."""
        key = (card, int(level))
        if key in self._stat_cache:
            return self._stat_cache[key]
        out = self._stats_uncached(card, int(level))
        self._stat_cache[key] = out
        return out

    def _stat_rows(self, card: str, level: int) -> dict[str, dict]:
        rows = self._conn.execute("SELECT unit, props FROM stats WHERE card = ? AND level = ?", (card, level))
        return {r["unit"]: json.loads(r["props"]) for r in rows}

    def _stats_uncached(self, card: str, level: int) -> dict | None:
        units = self._stat_rows(card, level)
        if not units:
            try:
                from clashbot.units import resolve
                alt = resolve(card).get("card")
            except Exception:
                alt = None
            if alt and alt != card:
                card = alt
                units = self._stat_rows(card, level)
        if not units:
            return None
        main = card if card in units else sorted(units)[0]
        out = dict(units[main])
        out["unit"] = main
        others = {u: p for u, p in units.items() if u != main}
        if others:
            out["units"] = others
        return out

    def stats_near(self, card: str, level: int) -> dict | None:
        """`stats`, falling back to the closest level in the table (cards whose table stops below `level`)."""
        got = self.stats(card, level)
        if got is not None:
            return got
        rows = self._conn.execute("SELECT DISTINCT level FROM stats WHERE card = ?", (card,)).fetchall()
        if not rows:
            return None
        best = min((r["level"] for r in rows), key=lambda lv: (abs(lv - level), lv))
        return self.stats(card, best)

    # ----------------------------------------------------------------------------------- answers
    def answers_for(self, threat: str, our_cards: list[str], k: int = 5) -> list[Answer]:
        """The best of `our_cards` against `threat`, strongest first, at most `k`, each with its evidence.

        Four sources, each 0..1 per card: pro = held rate x support / (support + 5) from the mined ANSWERED_BY edge;
        role_transfer = SIMILAR_ROLE(Y, Z) x pro(Y) x 0.7 for a pro answer Y that is not in our cards (best Y, kept in
        `via`); wiki = 1 - 0.5 ** (wiki bullets saying Z beats the threat); sim = K4 STOPS weight (0 until K4).
        score = 0.45 pro + 0.30 role_transfer + 0.10 wiki + 0.15 sim, renormalised over the sources that have any
        evidence for this threat among our cards. Cards with no evidence at all are not returned."""
        threat = _slug(threat)
        ours = list(dict.fromkeys(_slug(c) for c in our_cards))
        ours_set = set(ours)
        pro_edges = self._answered.get(threat, {})
        total_support = sum(float(p.get("support") or 0) for p in pro_edges.values())

        pro = {z: _pro_score(pro_edges[z]) for z in ours if z in pro_edges}
        transfer: dict[str, tuple[float, str]] = {}
        for y in sorted(pro_edges):
            if y in ours_set:
                continue
            ps = _pro_score(pro_edges[y])
            if ps <= 0:
                continue
            for z in ours:
                v = self._sim.get(y, {}).get(z, 0.0) * ps * ROLE_TRANSFER_DISCOUNT
                if v > transfer.get(z, (0.0, ""))[0]:
                    transfer[z] = (v, y)
        wiki = {}
        for z in ours:
            n = self._counters.get(threat, {}).get(z, 0.0)
            if n > 0:
                wiki[z] = 1.0 - 0.5 ** n
        sim = {z: min(1.0, self._stops[threat][z]) for z in ours if self._stops.get(threat, {}).get(z, 0.0) > 0}

        per_source = {"pro": pro, "role_transfer": {z: t[0] for z, t in transfer.items()}, "wiki": wiki, "sim": sim}
        active = [s for s, d in per_source.items() if any(v > 0 for v in d.values())]
        if not active:
            return []
        norm = sum(SOURCE_WEIGHTS[s] for s in active)
        out: list[Answer] = []
        for z in ours:
            parts = {s: float(per_source[s].get(z, 0.0)) for s in SOURCE_WEIGHTS}
            score = sum(SOURCE_WEIGHTS[s] * parts[s] for s in active) / norm
            if score <= 0:
                continue
            via = None
            props = None
            if parts["pro"] > 0:
                props = pro_edges[z]
            elif parts["role_transfer"] > 0:
                via = transfer[z][1]
                props = pro_edges[via]
            support = int(props.get("support") or 0) if props else 0
            out.append(Answer(
                card=z, score=round(score, 4), sources={s: round(v, 4) for s, v in parts.items()},
                support=support, held_rate=props.get("held_rate") if props else None,
                offset=props.get("offset") if props else None,
                eta_at_answer=props.get("eta_at_answer_med") if props else None, via=via,
                share=round(support / total_support, 4) if props and total_support else None))
        out.sort(key=lambda a: (-a.score, a.card))
        return out[:k]

    # ---------------------------------------------------------------------------- opponent deck
    def _opp_deck_model(self):
        if self._opp_model is False:
            self._opp_model = None
            try:                                         # K3: clashbot/opp_model.py (optional)
                from clashbot.opp_model import OppDeckModel
                if self.deck_cooc_path.exists():
                    self._opp_model = OppDeckModel.load(self.deck_cooc_path)
            except Exception:
                self._opp_model = None
        return self._opp_model

    def predict_opp_deck(self, seen: list[str], top: int = 8) -> list[tuple[str, float]]:
        """Likely hidden opponent cards [(card, prob)], seen cards excluded. K3's `OppDeckModel` when importable and
        its file exists, else the mean of P(c | s) over the seen cards s, read from the CO_OCCURS edges (cards no seen
        card ever co-occurred with are left out)."""
        seen = list(dict.fromkeys(_slug(c) for c in seen))
        model = self._opp_deck_model()
        if model is not None:
            try:
                return [(c, float(p)) for c, p in model.predict(seen, top=top)]
            except Exception:
                pass
        seen_set = set(seen)
        score: dict[str, float] = defaultdict(float)
        for s in seen:
            for c, p in self._cooc.get(s, {}).items():
                if c not in seen_set:
                    score[c] += p / len(seen)
        ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        return [(c, round(min(1.0, v), 4)) for c, v in ranked]

    # ---------------------------------------------------------------------------------- situation
    def situation(self, threat_cards: list[str], lane_rel: str, eta: float | None,
                  elixir: float | None) -> dict | None:
        """The mined situation (K2) nearest to the given one: the same threat pair, lane relation, ETA band and elixir
        band first, else the closest by threat overlap then the other fields. {key, threat_cards, lane_rel, eta_band,
        elixir_band, n, answers, exact}; None when no mined situation shares a threat card."""
        if not self._situations:
            return None
        top = _top2([_slug(c) for c in threat_cards], self._elixir)
        want = set(top)
        eb, xb = eta_band(eta), elixir_band(elixir)
        key = None
        try:                                             # K2's own key function when present (format may differ)
            from tools.data.mine_pro_answers import situation_key as k2_key
            got = k2_key(top, lane_rel, eta, elixir)
            key = got if isinstance(got, str) else None
        except Exception:
            key = None
        key = key or situation_key(top, lane_rel, eta, elixir, self._elixir)
        best: tuple[float, dict] | None = None
        for s in self._situations:
            have = s.get("threat_cards") or []
            if isinstance(have, str):
                have = have.split("+")
            have = set(have)
            if not (have & want):
                continue
            sc = 4.0 * len(have & want) / len(have | want)
            sc += 1.0 if s.get("lane_rel") == lane_rel else 0.0
            sc += _band_score(s.get("eta_band"), eb, ("0-3", "3-6", "6-10", ">10"))
            sc += 0.5 * (_band_score(s.get("elixir_band"), xb, ("0-3", "4-6", "7-10")) if xb != "?" else 0.5)
            sc += min(float(s.get("n") or 0), 1000.0) / 1e6       # tie-break: more evidence
            if s.get("key") == key:
                sc += 10.0
            if best is None or sc > best[0]:
                best = (sc, s)
        if best is None:
            return None
        out = dict(best[1])
        out["exact"] = best[1].get("key") == key or (
            set(best[1].get("threat_cards") or []) == want and best[1].get("lane_rel") == lane_rel
            and best[1].get("eta_band") == eb and (xb == "?" or best[1].get("elixir_band") == xb))
        return out

    # --------------------------------------------------------------------------------- LLM lines
    def facts_lines(self, state: dict, our_deck: list[str], max_lines: int = 3) -> list[str]:
        """At most `max_lines` (<= 3) lines of <= 160 characters for the LLM, e.g.
        `KG: balloon L15 2458hp 937/drop; answers musketeer (pros 31% n=48 held 70%), tesla (role of cannon, n=22)`:
        the top two enemy threats on the board (stats at level 15 + the best answers in our deck), then the opponent
        cards predicted but not yet seen. A part with no data is left out."""
        if "units" not in state and isinstance(state.get("state"), dict):
            state = state["state"]
        lines: list[str] = []
        for card, _unit in self._rank_threats(state)[:2]:
            line = self._threat_line(card, our_deck)
            if line:
                lines.append(line)
        line = self._opp_line(state)
        if line:
            lines.append(line)
        return lines[:max(0, min(max_lines, 3))]

    def _rank_threats(self, state: dict) -> list[tuple[str, dict]]:
        best: dict[str, tuple[tuple, dict]] = {}
        for u in state.get("units") or []:
            card = u.get("card")
            if u.get("side") != "enemy" or not card:
                continue
            eta = (u.get("eta_tower") or {}).get("s")
            tile = u.get("tile") or [0, 0]
            if not ((eta is not None and eta <= MAX_THREAT_ETA_S) or tile[1] <= MAX_THREAT_ROW):
                continue
            rank = (-self._magnitude(card), eta if eta is not None else 99.0, card)
            if card not in best or rank < best[card][0]:
                best[card] = (rank, u)
        return [(c, v[1]) for c, v in sorted(best.items(), key=lambda kv: kv[1][0])]

    def _magnitude(self, card: str) -> float:
        """Rough damage a unit does to a tower: hit damage x count, tripled for building-targeters; elixir x 50 when
        the card has no stats."""
        st = self.stats_near(card, OPP_LEVEL)
        role = self.roles().get(card) or {}
        dmg = _hit_damage(st) if st else 0
        if not dmg:
            return (self._elixir.get(card) or 3.0) * 50.0
        return float(dmg) * (role.get("count") or 1) * (3.0 if role.get("targets") == "buildings" else 1.0)

    def _threat_line(self, card: str, our_deck: list[str]) -> str | None:
        st = self.stats_near(card, OPP_LEVEL)
        role = self.roles().get(card) or {}
        head = f"KG: {card}"
        if st:
            bits = [f"L{OPP_LEVEL}"]
            if (role.get("count") or 1) > 1:
                bits.append(f"x{role['count']}")
            if st.get("hp"):
                bits.append(f"{_n(st['hp'])}hp")
            dmg = _hit_damage(st)
            if dmg:
                per = "drop" if role.get("flying") and role.get("targets") == "buildings" else "hit"
                bits.append(f"{_n(dmg)}/{per}")
            head += " " + " ".join(bits)
        answers = self.answers_for(card, our_deck, k=2)
        if not st and not answers:
            return None
        variants = []
        for n in (2, 1):
            sel = answers[:n]
            if sel:
                for detail in (True, False):
                    variants.append(head + "; answers " + ", ".join(_answer_text(a, detail) for a in sel))
        variants.append(head + ("" if answers else "; no known answer in deck"))
        for v in variants:
            if len(v) <= LINE_MAX:
                return v
        return variants[-1][:LINE_MAX - 3] + "..."

    def _opp_line(self, state: dict) -> str | None:
        seen = list((state.get("opponent") or {}).get("deck_known") or [])
        if not seen:
            return None
        picks = [(c, p) for c, p in self.predict_opp_deck(seen, top=6) if p >= 0.2][:3]
        if not picks:
            return None
        line = "KG: opp likely also has " + ", ".join(f"{c} {p:.2f}" for c, p in picks)
        return line if len(line) <= LINE_MAX else line[:LINE_MAX - 3] + "..."


def _band_score(have, want, order: tuple[str, ...]) -> float:
    if have == want:
        return 1.0
    if have in order and want in order and abs(order.index(have) - order.index(want)) == 1:
        return 0.5
    return 0.0


def _hit_damage(st: dict) -> float:
    """Damage of one hit: `damage`, else `area_damage`, else the first qualified `*_damage` (ram-rider's ram_damage)."""
    if st.get("damage") or st.get("area_damage"):
        return st.get("damage") or st.get("area_damage")
    for k, v in st.items():
        if k.endswith("_damage") and not k.startswith(("death_", "crown_tower_")) and isinstance(v, (int, float)) and v:
            return v
    return 0


def _n(x) -> str:
    return str(int(round(x))) if float(x) == int(x) or abs(x) >= 10 else f"{x:.1f}"


def _answer_text(a: Answer, detail: bool) -> str:
    if a.via:
        return f"{a.card} (role of {a.via}, n={a.support})" if detail else f"{a.card} (via {a.via})"
    if a.support:
        if not detail:
            return f"{a.card} (n={a.support})"
        bits = []
        if a.share is not None:
            bits.append(f"pros {a.share * 100:.0f}%")
        else:
            bits.append("pros")
        bits.append(f"n={a.support}")
        if a.held_rate is not None:
            bits.append(f"held {a.held_rate * 100:.0f}%")
        return f"{a.card} ({' '.join(bits)})"
    return f"{a.card} (wiki)" if a.sources.get("wiki") else f"{a.card} (sim)"
