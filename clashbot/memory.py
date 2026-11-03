"""Agent memory stores (D-034): episodic, semantic and procedural, in one SQLite file (course AGT §4.1).

Semantic memory holds patch-versioned card facts with provenance: an external fact store consulted at run time
(non-parametric editing, SERAC/GRACE family), so a balance change is an `apply_edit`, never a weight update, and it
can be listed, read as of an older patch, and retracted. Procedural memory holds playbook lessons written with
Mem0-style ADD / UPDATE / DELETE / NOOP operations. Episodic memory holds observed or played situations.
`conflict_weight` / `reweight_hits` down-weight pre-patch retrieval evidence about edited cards (ClashEval-style).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from clashbot.config import DB_DIR
from clashbot.game_rules import load_cards

MEMORY_DB_PATH = DB_DIR / "memory.sqlite"
BASE_PATCH = "base"
EDITABLE_ATTRIBUTES = ("elixir_cost", "card_type", "rarity", "targets")   # keys of load_cards() entries
LESSON_OPS = ("ADD", "UPDATE", "DELETE", "NOOP")                         # Mem0-style write operations

SCHEMA = """
CREATE TABLE IF NOT EXISTS patches(
    seq INTEGER PRIMARY KEY, patch TEXT UNIQUE NOT NULL, note TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS card_facts(
    fact_id INTEGER PRIMARY KEY, card TEXT NOT NULL, attribute TEXT NOT NULL, value TEXT NOT NULL,
    patch TEXT NOT NULL, valid_from INTEGER NOT NULL, valid_to INTEGER, source TEXT, provenance TEXT,
    created_at TEXT);
CREATE TABLE IF NOT EXISTS episodes(
    episode_id INTEGER PRIMARY KEY, game_id TEXT, t REAL, source TEXT, patch TEXT, state TEXT, action_key TEXT,
    outcome TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS lessons(
    lesson_id INTEGER PRIMARY KEY, scope TEXT NOT NULL, text TEXT NOT NULL, source TEXT, support INTEGER DEFAULT 1,
    created_at TEXT, updated_at TEXT, deleted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS memory_log(
    log_id INTEGER PRIMARY KEY, op TEXT, store TEXT, ref INTEGER, detail TEXT, created_at TEXT);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fact_dict(row: sqlite3.Row) -> dict:
    out = dict(row)
    out["value"] = json.loads(out["value"])
    return out


class MemoryStore:
    def __init__(self, path: Path | str = MEMORY_DB_PATH, now=None):
        self._now = now if now is not None else _utc_now
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.executescript(SCHEMA)
            self.conn.execute("INSERT OR IGNORE INTO patches(seq, patch, note, created_at) VALUES (0, ?, ?, ?)",
                              (BASE_PATCH, "unedited card index", self._now()))

    def close(self) -> None:
        self.conn.close()

    def _log(self, op: str, store: str, ref: int | None, detail: dict) -> None:
        self.conn.execute("INSERT INTO memory_log(op, store, ref, detail, created_at) VALUES (?,?,?,?,?)",
                          (op, store, ref, json.dumps(detail, default=str), self._now()))

    # ---- patches -------------------------------------------------------------------------------------------

    def _add_patch(self, patch: str, note: str = "") -> int:
        row = self.conn.execute("SELECT seq FROM patches WHERE patch = ?", (patch,)).fetchone()
        if row is not None:
            return row["seq"]
        seq = self.conn.execute("SELECT max(seq) FROM patches").fetchone()[0] + 1
        self.conn.execute("INSERT INTO patches(seq, patch, note, created_at) VALUES (?,?,?,?)",
                          (seq, patch, note, self._now()))
        return seq

    def add_patch(self, patch: str, note: str = "") -> int:
        with self.conn:
            return self._add_patch(patch, note)

    def patch_seq(self, patch: str) -> int:
        row = self.conn.execute("SELECT seq FROM patches WHERE patch = ?", (patch,)).fetchone()
        if row is None:
            raise KeyError(patch)
        return row["seq"]

    def patches(self) -> list[str]:
        return [r["patch"] for r in self.conn.execute("SELECT patch FROM patches ORDER BY seq")]

    def _latest_seq(self) -> int:
        return self.conn.execute("SELECT max(seq) FROM patches").fetchone()[0]

    # ---- semantic memory: patch-versioned card facts -------------------------------------------------------

    def apply_edit(self, card: str, attribute: str, value, patch: str, source: str, provenance: str) -> int:
        if attribute not in EDITABLE_ATTRIBUTES:
            raise ValueError(f"attribute {attribute!r} is not editable; expected one of {EDITABLE_ATTRIBUTES}")
        with self.conn:
            s = self._add_patch(patch)
            prev = self.conn.execute(
                "SELECT fact_id, valid_from FROM card_facts WHERE card = ? AND attribute = ? AND valid_to IS NULL "
                "ORDER BY fact_id DESC LIMIT 1", (card, attribute)).fetchone()
            if prev is not None and s < prev["valid_from"]:
                raise ValueError(f"patch {patch!r} precedes the open fact for {card}.{attribute}")
            if prev is not None:
                # a re-edit inside the same patch makes the earlier fact never valid (valid_to == valid_from)
                self.conn.execute("UPDATE card_facts SET valid_to = ? WHERE fact_id = ?",
                                  (s if prev["valid_from"] < s else prev["valid_from"], prev["fact_id"]))
            cur = self.conn.execute(
                "INSERT INTO card_facts(card, attribute, value, patch, valid_from, valid_to, source, provenance, "
                "created_at) VALUES (?,?,?,?,?,NULL,?,?,?)",
                (card, attribute, json.dumps(value), patch, s, source, provenance, self._now()))
            fact_id = cur.lastrowid
            self._log("ADD" if prev is None else "UPDATE", "semantic", fact_id,
                      {"card": card, "attribute": attribute, "value": value, "patch": patch, "source": source,
                       "provenance": provenance, "replaces": None if prev is None else prev["fact_id"]})
        return fact_id

    def retract(self, fact_id: int) -> None:
        with self.conn:
            fact = self.conn.execute("SELECT * FROM card_facts WHERE fact_id = ?", (fact_id,)).fetchone()
            if fact is None:
                raise KeyError(fact_id)
            if fact["valid_to"] is not None and fact["valid_to"] == fact["valid_from"]:
                return    # already retracted (or never valid)
            self.conn.execute("UPDATE card_facts SET valid_to = valid_from WHERE fact_id = ?", (fact_id,))
            # the fact this one superseded becomes current again (never-valid facts are not candidates)
            prev = self.conn.execute(
                "SELECT fact_id FROM card_facts WHERE card = ? AND attribute = ? AND fact_id < ? AND valid_to = ? "
                "AND valid_from < valid_to ORDER BY fact_id DESC LIMIT 1",
                (fact["card"], fact["attribute"], fact_id, fact["valid_from"])).fetchone()
            if prev is not None:
                self.conn.execute("UPDATE card_facts SET valid_to = NULL WHERE fact_id = ?", (prev["fact_id"],))
            self._log("DELETE", "semantic", fact_id,
                      {"card": fact["card"], "attribute": fact["attribute"], "patch": fact["patch"],
                       "reopened": None if prev is None else prev["fact_id"]})

    def _facts_as_of(self, seq: int) -> list[sqlite3.Row]:
        # valid_from <= seq < valid_to; a never-valid fact has valid_to == valid_from and so never matches
        return self.conn.execute(
            "SELECT * FROM card_facts WHERE valid_from <= ? AND (valid_to IS NULL OR ? < valid_to) "
            "ORDER BY fact_id", (seq, seq)).fetchall()

    def card_fact(self, card: str, attribute: str, patch: str | None = None, base_cards: dict | None = None):
        s = self._latest_seq() if patch is None else self.patch_seq(patch)
        row = self.conn.execute(
            "SELECT value FROM card_facts WHERE card = ? AND attribute = ? AND valid_from <= ? "
            "AND (valid_to IS NULL OR ? < valid_to) ORDER BY fact_id DESC LIMIT 1", (card, attribute, s, s)).fetchone()
        if row is not None:
            return json.loads(row["value"])
        cards = load_cards() if base_cards is None else base_cards
        return cards[card][attribute]

    def edited_cards(self, base_cards: dict, patch: str | None = None) -> dict:
        s = self._latest_seq() if patch is None else self.patch_seq(patch)
        out = copy.deepcopy(base_cards)
        for row in self._facts_as_of(s):
            if row["card"] in out:
                out[row["card"]][row["attribute"]] = json.loads(row["value"])
        return out

    def edits_since(self, patch: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM card_facts WHERE valid_from > ? AND (valid_to IS NULL OR valid_to > valid_from) "
            "ORDER BY fact_id", (self.patch_seq(patch),)).fetchall()
        return [_fact_dict(r) for r in rows]

    # ---- episodic memory -----------------------------------------------------------------------------------

    def add_episode(self, game_id: str, t: float, state: dict, action_key: str | None, outcome: dict | None,
                    source: str = "own", patch: str | None = None) -> int:
        with self.conn:
            if patch is None:
                patch = self.conn.execute("SELECT patch FROM patches ORDER BY seq DESC LIMIT 1").fetchone()["patch"]
            cur = self.conn.execute(
                "INSERT INTO episodes(game_id, t, source, patch, state, action_key, outcome, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (game_id, t, source, patch, json.dumps(state), action_key,
                 None if outcome is None else json.dumps(outcome), self._now()))
            episode_id = cur.lastrowid
            self._log("ADD", "episodic", episode_id,
                      {"game_id": game_id, "t": t, "source": source, "patch": patch, "action_key": action_key})
        return episode_id

    def episodes(self, game_id: str | None = None, source: str | None = None) -> list[dict]:
        where, args = [], []
        if game_id is not None:
            where.append("game_id = ?")
            args.append(game_id)
        if source is not None:
            where.append("source = ?")
            args.append(source)
        sql = "SELECT * FROM episodes" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY episode_id"
        out = []
        for row in self.conn.execute(sql, args):
            d = dict(row)
            d["state"] = json.loads(d["state"]) if d["state"] is not None else None
            d["outcome"] = json.loads(d["outcome"]) if d["outcome"] is not None else None
            out.append(d)
        return out

    # ---- procedural memory: playbook lessons ---------------------------------------------------------------

    def _live_lesson(self, lesson_id: int | None) -> sqlite3.Row:
        row = None
        if lesson_id is not None:
            row = self.conn.execute("SELECT * FROM lessons WHERE lesson_id = ? AND deleted = 0",
                                    (lesson_id,)).fetchone()
        if row is None:
            raise KeyError(lesson_id)
        return row

    def write_lesson(self, op: str, scope: str, text: str = "", lesson_id: int | None = None,
                     source: str = "reflection") -> int | None:
        if op not in LESSON_OPS:
            raise ValueError(f"lesson op {op!r} must be one of {LESSON_OPS}")
        text = text.strip()
        now = self._now()
        with self.conn:
            if op == "ADD":
                if not text:
                    raise ValueError("ADD needs lesson text")
                dup = self.conn.execute("SELECT lesson_id FROM lessons WHERE scope = ? AND text = ? AND deleted = 0 "
                                        "ORDER BY lesson_id LIMIT 1", (scope, text)).fetchone()
                if dup is not None:
                    self.conn.execute("UPDATE lessons SET support = support + 1, updated_at = ? WHERE lesson_id = ?",
                                      (now, dup["lesson_id"]))
                    self._log("NOOP", "procedural", dup["lesson_id"],
                              {"scope": scope, "text": text, "reason": "duplicate", "source": source})
                    return dup["lesson_id"]
                cur = self.conn.execute(
                    "INSERT INTO lessons(scope, text, source, support, created_at, updated_at, deleted) "
                    "VALUES (?,?,?,1,?,?,0)", (scope, text, source, now, now))
                self._log("ADD", "procedural", cur.lastrowid, {"scope": scope, "text": text, "source": source})
                return cur.lastrowid
            if op == "UPDATE":
                if not text:
                    raise ValueError("UPDATE needs lesson text")
                old = self._live_lesson(lesson_id)
                self.conn.execute("UPDATE lessons SET text = ?, updated_at = ?, support = support + 1 "
                                  "WHERE lesson_id = ?", (text, now, lesson_id))
                self._log("UPDATE", "procedural", lesson_id,
                          {"scope": old["scope"], "old_text": old["text"], "text": text, "source": source})
                return lesson_id
            if op == "DELETE":
                old = self._live_lesson(lesson_id)
                self.conn.execute("UPDATE lessons SET deleted = 1, updated_at = ? WHERE lesson_id = ?",
                                  (now, lesson_id))
                self._log("DELETE", "procedural", lesson_id,
                          {"scope": old["scope"], "text": old["text"], "source": source})
                return lesson_id
            self._log("NOOP", "procedural", lesson_id, {"scope": scope, "text": text, "source": source})
            return lesson_id

    def lessons(self, scope: str | None = None) -> list[dict]:
        sql = "SELECT * FROM lessons WHERE deleted = 0"
        args: list = []
        if scope is not None:
            sql += " AND scope = ?"
            args.append(scope)
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY support DESC, lesson_id", args)]

    def log(self) -> list[dict]:
        out = []
        for row in self.conn.execute("SELECT * FROM memory_log ORDER BY log_id"):
            d = dict(row)
            d["detail"] = json.loads(d["detail"]) if d["detail"] else None
            out.append(d)
        return out


# ---- conflict handling (pre-patch evidence about edited cards) ---------------------------------------------

def stale_cards(store: MemoryStore, sample_patch: str = BASE_PATCH, current_patch: str | None = None) -> set[str]:
    """Cards edited after `sample_patch` up to `current_patch` (default: latest) whose edit has not been retracted."""
    hi = store.patch_seq(current_patch) if current_patch is not None else store.patch_seq(store.patches()[-1])
    return {f["card"] for f in store.edits_since(sample_patch) if f["valid_from"] <= hi}


def conflict_weight(action_key: str | None, stale: set[str], weight: float = 0.5) -> float:
    """`weight` for a play key whose card is stale (retrieved from before its edit), else 1.0."""
    if not action_key:
        return 1.0
    parts = action_key.split("|")
    return weight if len(parts) == 3 and parts[0] in stale else 1.0


def reweight_hits(hits: list, stale: set[str], weight: float = 0.5) -> list:
    """New `clashbot.retrieval.Hit`s with stale-card scores down-weighted, re-sorted by (-score, row)."""
    out = [dataclasses.replace(h, score=h.score * conflict_weight(h.action_key, stale, weight)) for h in hits]
    return sorted(out, key=lambda h: (-h.score, h.row))
