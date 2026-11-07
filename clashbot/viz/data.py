"""Data layer of the decision dashboard (MAC-AO): run directories, incremental JSON-lines readers, the reference data.

A run dir (runs/poc/<ts>, runs/demo/<ts>, runs/smoke/<name>) is read the way the live loop writes it: `decisions.jsonl`,
`holds.jsonl` and `battles.jsonl` are tailed by byte offset (a partial last line waits for its newline), `shots/` is
listed, `config.json` / `summary.json` are small. The reference data (the pro states behind a retrieval hit, the
knowledge graph) is opened lazily, once, read-only. Python standard library only; nothing here writes to a run or to db/.

Run ids are `<parent folder>:<run folder>` (e.g. `poc:20261107-111554`) so they fit one URL segment.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

from clashbot.config import DB_PATH, KG2_PATH, ROOT

RUNS_DIR = ROOT / "runs"
RUN_PARENTS = ("demo", "poc", "smoke", "eval/vs_pro")   # listed by /api/runs, in this order (eval/vs_pro: MAC-AQ, the bot on pro videos)
PRO_FILE = "pro_plays.jsonl"                    # MAC-AQ: the pro's plays of a run made on a pro video (t in run time)
PRO_WINDOW_S = 3.0                              # a decision shows the nearest pro play within this many seconds
LIVE_WINDOW_S = 15.0                            # a run without summary.json is live while a file changed this recently
MAX_OPEN_RUNS = 8                               # runs whose rows are kept in memory
SHOT_NAME_RE = re.compile(r"^d(\d+)_t([0-9.]+)_([abc])\.jpg$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*:[A-Za-z0-9][A-Za-z0-9_.\-]*$")
TIMESTAMP_RE = re.compile(r"^(\d{8})-(\d{6})$")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
UNIT_RE = re.compile(r"([a-z0-9?][a-z0-9?\-]*)\((a|e)\)@\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")
THREAT_RE = re.compile(r"([a-z0-9][a-z0-9\-]*)\(e\)")
KG_RELS = ("COUNTERS", "ANSWERED_BY", "SYNERGY")       # relations drawn in the knowledge-graph panel
MAX_SYNERGY_EDGES = 6
TOP_COUNTERS = 3


def bucket_reason(reason: str | None) -> str:
    """A hold reason with its numbers replaced by '#' ("fast wait p=0.32 h=2s" -> "fast wait p=# h=#s")."""
    return NUMBER_RE.sub("#", reason or "")


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100), the numpy default; None for no values."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def parse_context(text: str | None) -> dict:
    """The cards on the field and in the threat list of a d1 `context_text`, and the hand it names.

    `field: knight(a)@[3,15] advancing ...; bats(e)@[11,16] ...` -> units (card, side a|e, tile); `threats: bats(e)
    advancing right lane, tower in 3.8s; ...` -> threat cards; `hand: a, b, ?, d (next x)` -> hand."""
    out = {"units": [], "enemy": [], "ally": [], "threats": [], "hand": []}
    for line in (text or "").split("\n"):
        if line.startswith("field:"):
            for m in UNIT_RE.finditer(line):
                card, side = m.group(1), m.group(2)
                if card == "?":
                    continue
                out["units"].append({"card": card, "side": "ally" if side == "a" else "enemy",
                                     "tile": [float(m.group(3)), float(m.group(4))]})
        elif line.startswith("threats:"):
            out["threats"] = list(dict.fromkeys(THREAT_RE.findall(line)))
        elif "hand:" in line:                                    # the header line: "... hand: a, b, ?, d (next x). deck: ..."
            head = line.split("hand:", 1)[1].split("(next", 1)[0].split(". deck", 1)[0]
            out["hand"] = [c.strip() for c in head.split(",") if c.strip() and c.strip() != "?"]
    out["enemy"] = list(dict.fromkeys(u["card"] for u in out["units"] if u["side"] == "enemy"))
    out["ally"] = list(dict.fromkeys(u["card"] for u in out["units"] if u["side"] == "ally"))
    for c in out["threats"]:
        if c not in out["enemy"]:
            out["enemy"].append(c)
    return out


class Tail:
    """Incremental reader of a JSON-lines file: `poll()` appends the complete new lines to `rows`.

    A last line without its newline is left alone until it is complete (or until it already parses as a whole JSON
    object, for a file that was closed without a final newline). A line that is not a JSON object is counted in
    `bad` and skipped. A file that shrank (rewritten) is read again from the start."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: list[dict] = []
        self.offset = 0
        self.bad = 0
        self._lock = threading.Lock()

    def _add(self, raw: bytes) -> bool:
        raw = raw.strip()
        if not raw:
            return True
        try:
            row = json.loads(raw)
        except ValueError:
            self.bad += 1
            return False
        if not isinstance(row, dict):
            self.bad += 1
            return False
        self.rows.append(row)
        return True

    def poll(self) -> list[dict]:
        with self._lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                return self.rows
            if size < self.offset:
                self.rows, self.offset, self.bad = [], 0, 0
            if size > self.offset:
                with open(self.path, "rb") as fh:
                    fh.seek(self.offset)
                    chunk = fh.read(size - self.offset)
                end = chunk.rfind(b"\n") + 1
                for line in chunk[:end].splitlines():
                    self._add(line)
                rest = chunk[end:]
                if rest.strip():
                    mark = self.bad
                    if self._add(rest):
                        end = len(chunk)
                    else:
                        self.bad = mark                     # a partial line is not a bad one yet
                self.offset += end
            return self.rows

    def since(self, n: int) -> tuple[list[dict], int]:
        rows = self.poll()
        n = max(int(n), 0)
        return rows[n:], len(rows)


def nearest_pro(row: dict, plays: list[dict], window: float = PRO_WINDOW_S) -> dict | None:
    """The pro play nearest in time to a decision row (within `window` s) with a match badge: `same card` (the bot's card
    is the pro's), else `same lane`, else `different`; `bot held` when the decision is a hold, `pro card unknown` when the
    dataset could not name the pro's card (and the lane is not shared). `dt` = bot time minus pro
    time (negative = the bot was earlier). None when no pro play is that close."""
    best = None
    for p in plays:
        d = abs(float(p["t"]) - float(row.get("t", 0.0)))
        if d <= window and (best is None or d < best[0]):
            best = (d, p)
    if best is None:
        return None
    p = best[1]
    key = row.get("key")
    status = str((row.get("executor") or {}).get("status", "dry_run"))
    parts = key.split("|") if isinstance(key, str) else []
    held = key in (None, "hold") or status.startswith("hold")
    card = parts[0] if len(parts) == 3 else None
    lane = parts[1] if len(parts) == 3 else None
    same_card = bool(card and card == p.get("card"))
    same_lane = bool(lane and lane == p.get("lane"))
    tile, ptile = row.get("tile"), p.get("tile")
    dist = None
    if isinstance(tile, (list, tuple)) and isinstance(ptile, (list, tuple)) and len(tile) >= 2 and len(ptile) >= 2:
        dist = round(((float(tile[0]) - float(ptile[0])) ** 2 + (float(tile[1]) - float(ptile[1])) ** 2) ** 0.5, 2)
    badge = ("bot held" if held else "pro card unknown" if not p.get("card") and not same_lane else "same card" if same_card
             else "same lane" if same_lane else "different")
    return {**{k: p.get(k) for k in ("t", "card", "tile", "lane", "zone", "key", "confidence")},
            "dt": round(float(row.get("t", 0.0)) - float(p["t"]), 2), "badge": badge, "same_card": same_card,
            "same_lane": same_lane, "tile_dist": dist}


class Run:
    """One run directory, read incrementally."""

    def __init__(self, run_id: str, path: Path):
        self.id = run_id
        self.path = Path(path)
        self.decisions = Tail(self.path / "decisions.jsonl")
        self.holds = Tail(self.path / "holds.jsonl")
        self.battles = Tail(self.path / "battles.jsonl")
        self._shots: dict[str, dict[str, str]] = {}
        self._shots_mark: tuple | None = None
        self._lat: list[float] = []
        self._lat_n = 0
        self._taps = 0
        self._buckets: dict[str, int] = {}
        self._buckets_n = 0
        self._pro: list[dict] | None = None                     # pro_plays.jsonl (MAC-AQ), re-read when the file changed
        self._pro_mark: tuple | None = None
        self._lock = threading.RLock()                          # one request at a time per run (the caches above)

    # ---------------------------------------------------------------------------------- small files
    def _json(self, name: str) -> dict | None:
        try:
            data = json.loads((self.path / name).read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def config(self) -> dict:
        return self._json("config.json") or {}

    def summary(self) -> dict | None:
        return self._json("summary.json")

    def is_live(self) -> bool:
        return run_is_live(self.path)

    # ---------------------------------------------------------------------------------- shots
    def _shot_index(self) -> dict[str, dict[str, str]]:
        """shots/ listed as {"d0217_t22.5": {"a": file, "b": file, "c": file}}; rescanned when the folder changed."""
        folder = self.path / "shots"
        try:
            st = folder.stat()
        except OSError:
            self._shots, self._shots_mark = {}, None
            return self._shots
        mark = (st.st_mtime_ns, st.st_size)
        if mark != self._shots_mark:
            index: dict[str, dict[str, str]] = {}
            with os.scandir(folder) as it:
                for e in it:
                    m = SHOT_NAME_RE.match(e.name)
                    if m:
                        index.setdefault(e.name[: e.name.rindex("_")], {})[m.group(3)] = e.name
            self._shots, self._shots_mark = index, mark
        return self._shots

    def shot_files(self, row: dict) -> dict[str, str | None]:
        """The shot file names (a, b, c) of a decision row; None where there is none."""
        files = self._shot_index().get(f"d{int(row.get('idx', -1)):04d}_t{float(row.get('t', 0.0)):.1f}", {})
        return {tag: files.get(tag) for tag in "abc"}

    def shot_path(self, name: str) -> Path | None:
        """The jpg `name` in this run's shots/, or None when the name is not a plain existing jpg there."""
        if not SAFE_NAME_RE.match(name or "") or not name.lower().endswith(".jpg") or ".." in name:
            return None
        folder = (self.path / "shots").resolve()
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file():
            return None
        return path

    # ---------------------------------------------------------------------------------- the pro (MAC-AQ)
    def has_pro(self) -> bool:
        return (self.path / PRO_FILE).is_file()

    def pro_plays(self) -> list[dict]:
        """The plays of pro_plays.jsonl sorted by time ([] for a run without the file); re-read when the file changed."""
        path = self.path / PRO_FILE
        try:
            st = path.stat()
        except OSError:
            self._pro, self._pro_mark = None, None
            return []
        mark = (st.st_mtime_ns, st.st_size)
        if self._pro is None or mark != self._pro_mark:
            rows = Tail(path).poll()
            self._pro = sorted((r for r in rows if isinstance(r.get("t"), (int, float))), key=lambda r: r["t"])
            self._pro_mark = mark
        return self._pro

    # ---------------------------------------------------------------------------------- rows
    def _enrich(self, i: int, row: dict) -> dict:
        out = dict(row)
        out["i"] = i
        out["shots"] = self.shot_files(row)
        out["parsed"] = parse_context(row.get("context_text"))
        if self.has_pro():                                      # only runs made on a pro video carry the comparison
            out["pro"] = nearest_pro(row, self.pro_plays())
        return out

    def _stats(self) -> dict:
        rows = self.decisions.rows
        if self._lat_n > len(rows):
            self._lat, self._lat_n, self._taps = [], 0, 0
        for row in rows[self._lat_n:]:
            total = (row.get("timings_ms") or {}).get("total")
            if isinstance(total, (int, float)):
                self._lat.append(float(total))
            if (row.get("executor") or {}).get("sent"):
                self._taps += 1
        self._lat_n = len(rows)
        p50, p95 = percentile(self._lat, 50), percentile(self._lat, 95)
        return {"decisions": len(rows), "taps": self._taps,
                "p50_ms": None if p50 is None else round(p50, 1), "p95_ms": None if p95 is None else round(p95, 1)}

    def decisions_since(self, since: int = 0, limit: int | None = None) -> dict:
        with self._lock:
            new, total = self.decisions.since(since)
            start = max(int(since), 0)
            if limit is not None:
                new = new[: max(int(limit), 0)]
            return {"rows": [self._enrich(start + k, r) for k, r in enumerate(new)], "next": start + len(new),
                    "total": total, "stats": self._stats(), "live": self.is_live()}

    def holds_since(self, since: int = 0) -> dict:
        with self._lock:
            new, total = self.holds.since(since)
            start = max(int(since), 0)
            rows = [{**r, "i": start + k, "bucket": bucket_reason(r.get("reason"))} for k, r in enumerate(new)]
            if self._buckets_n > len(self.holds.rows):
                self._buckets, self._buckets_n = {}, 0
            for r in self.holds.rows[self._buckets_n:]:         # cumulative counts, kept as rows arrive
                b = bucket_reason(r.get("reason"))
                self._buckets[b] = self._buckets.get(b, 0) + 1
            self._buckets_n = len(self.holds.rows)
            return {"rows": rows, "next": start + len(new), "total": total,
                    "buckets": dict(sorted(self._buckets.items(), key=lambda kv: -kv[1])), "live": self.is_live()}

    def meta(self, cards: dict | None = None) -> dict:
        with self._lock:
            return self._meta(cards)

    def _meta(self, cards: dict | None) -> dict:
        cfg = self.config()
        battles = self.battles.poll()
        self.decisions.poll()
        deck = [c.strip() for c in str(cfg.get("deck") or "").split(",") if c.strip()]
        heroes = _slugs(cfg.get("hero_cards"))
        evos = _slugs(cfg.get("evo_cards"))
        cards = cards or {}
        return {
            "id": self.id, "name": self.path.name, "path": str(self.path), "live": self.is_live(),
            "config": cfg, "summary": self.summary(), "battles": battles,
            "deck": [{"slug": s, "name": (cards.get(s) or {}).get("name") or s, "hero": s in heroes, "evo": s in evos}
                     for s in deck],
            "has_shots": (self.path / "shots").is_dir(), "stats": self._stats(),
            "has_pro": self.has_pro(),
        }


def run_is_live(path: Path) -> bool:
    """A run without summary.json whose files changed in the last LIVE_WINDOW_S seconds (frames.jsonl moves every frame)."""
    if (path / "summary.json").exists():
        return False
    newest = 0.0
    for name in ("frames.jsonl", "decisions.jsonl", "holds.jsonl", "battles.jsonl"):
        try:
            newest = max(newest, (path / name).stat().st_mtime)
        except OSError:
            pass
    return newest > 0 and time.time() - newest < LIVE_WINDOW_S


def _slugs(value) -> list[str]:
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, (list, tuple)):
        return [str(s).strip() for s in value if str(s).strip()]
    return []


def run_start(path: Path) -> str:
    """ISO start time of a run: from a timestamp folder name, else the config / folder modification time."""
    m = TIMESTAMP_RE.match(path.name)
    if m:
        d, t = m.groups()
        return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}"
    try:
        mtime = (path / "config.json").stat().st_mtime if (path / "config.json").exists() else path.stat().st_mtime
    except OSError:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime))


def _first_line(path: Path, limit: int = 1 << 20) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.readline(limit)
    except OSError:
        return b""


class Store:
    """The runs the dashboard can show: `follow` (always the newest run dir under a parent) or a fixed `run`."""

    def __init__(self, run: Path | None = None, follow: Path | None = None, runs_dir: Path = RUNS_DIR,
                 cards: dict | None = None):
        if (run is None) == (follow is None):
            raise ValueError("give exactly one of run / follow")
        self.fixed = Path(run).resolve() if run is not None else None
        self.follow = Path(follow).resolve() if follow is not None else None
        self.runs_dir = Path(runs_dir)
        self.cards = cards or {}
        self._runs: OrderedDict[str, Run] = OrderedDict()
        self._known: dict[str, Path] = {}
        self._list_cache: dict[str, tuple] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------------------- ids
    @staticmethod
    def run_id(path: Path) -> str:
        return f"{path.parent.name}:{path.name}"

    def _roots(self) -> list[Path]:
        roots: list[Path] = []
        for r in [self.runs_dir / p for p in RUN_PARENTS] + [self.follow, self.fixed.parent if self.fixed else None]:
            if r is not None and r.resolve() not in [x.resolve() for x in roots]:
                roots.append(r)
        return roots

    def resolve(self, run_id: str) -> Path | None:
        """The folder of a run id: one we listed, the fixed run, or `<root>/<name>` under a known root."""
        if not RUN_ID_RE.match(run_id or "") or ".." in run_id:
            return None
        with self._lock:
            known = self._known.get(run_id)
        if known is not None and known.is_dir():
            return known
        if self.fixed is not None and run_id == self.run_id(self.fixed) and self.fixed.is_dir():
            return self.fixed
        parent, name = run_id.split(":", 1)
        for root in self._roots():
            if root.name == parent and (root / name).is_dir():
                return root / name
        return None

    def run(self, run_id: str) -> Run | None:
        path = self.resolve(run_id)
        if path is None:
            return None
        with self._lock:
            r = self._runs.get(run_id)
            if r is None:
                r = self._runs[run_id] = Run(run_id, path)
                while len(self._runs) > MAX_OPEN_RUNS:
                    self._runs.popitem(last=False)
            self._runs.move_to_end(run_id)
            return r

    # ---------------------------------------------------------------------------------- current / list
    def newest(self) -> Path | None:
        """The newest run folder under the followed parent (by folder time, then name)."""
        if self.follow is None or not self.follow.is_dir():
            return None
        best, best_key = None, None
        for p in self.follow.iterdir():
            if not p.is_dir() or not SAFE_NAME_RE.match(p.name):
                continue
            try:
                key = (p.stat().st_mtime_ns, p.name)
            except OSError:
                continue
            if best_key is None or key > best_key:
                best, best_key = p, key
        return best

    def current(self) -> dict:
        path = self.newest() if self.follow is not None else self.fixed
        run_id = None
        live = False
        if path is not None:
            run_id = self.run_id(path)
            with self._lock:
                self._known[run_id] = path
            live = run_is_live(path)
        return {"follow": self.follow is not None, "parent": str(self.follow) if self.follow else None,
                "id": run_id, "live": live}

    def list_runs(self) -> list[dict]:
        out = []
        for root in self._roots():
            if not root.is_dir():
                continue
            for p in sorted(root.iterdir()):
                if not p.is_dir() or not SAFE_NAME_RE.match(p.name):
                    continue
                if not ((p / "decisions.jsonl").exists() or (p / "config.json").exists()):
                    continue
                run_id = self.run_id(p)
                with self._lock:
                    self._known[run_id] = p
                out.append(self._summary_row(run_id, p))
        out.sort(key=lambda r: (r["start"], r["id"]), reverse=True)
        return out

    def _summary_row(self, run_id: str, p: Path) -> dict:
        dec = p / "decisions.jsonl"
        try:
            st = dec.stat()
            mark = (st.st_size, st.st_mtime_ns, (p / "summary.json").exists())
        except OSError:
            mark = (0, 0, (p / "summary.json").exists())
        cached = self._list_cache.get(run_id)
        if cached is None or cached[0] != mark:
            count, has_retrieval = 0, False
            summary = None
            try:
                summary = json.loads((p / "summary.json").read_text())
            except (OSError, ValueError):
                pass
            if isinstance(summary, dict) and isinstance(summary.get("decisions"), int):
                count = summary["decisions"]
            elif mark[0]:
                with open(dec, "rb") as fh:
                    count = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))
            has_retrieval = b'"retrieval"' in _first_line(dec)
            cached = self._list_cache[run_id] = (mark, count, has_retrieval)
        _, count, has_retrieval = cached
        return {"id": run_id, "name": p.name, "parent": p.parent.name, "start": run_start(p), "decisions": count,
                "has_retrieval": has_retrieval, "has_shots": (p / "shots").is_dir(),
                "has_summary": (p / "summary.json").exists(), "live": run_is_live(p), "has_pro": (p / PRO_FILE).is_file()}


# -------------------------------------------------------------------------------------- reference data
class Reference:
    """The pro states and the knowledge graphs, opened lazily and read-only (db/clashbot.sqlite, db/kg2.sqlite)."""

    def __init__(self, db_path: Path = DB_PATH, kg2_path: Path = KG2_PATH, cards: dict | None = None):
        self.db_path = Path(db_path)
        self.kg2_path = Path(kg2_path)
        self.cards = cards or {}
        self._conn: sqlite3.Connection | None = None
        self._kg = None
        self._kg2 = None
        self._kg2_tried = False
        self._lock = threading.RLock()

    def available(self) -> bool:
        return self.db_path.exists()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(f"file:{self.db_path.resolve()}?mode=ro", uri=True, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _graph(self):
        if self._kg is None:
            from clashbot.kg import KnowledgeGraph
            self._kg = KnowledgeGraph(self.db_path)
        return self._kg

    def _graph2(self):
        if not self._kg2_tried:
            self._kg2_tried = True
            try:
                from clashbot.kg2 import KG2
                self._kg2 = KG2(self.kg2_path)
            except Exception:                                    # not built / unreadable: the page just has no v2 answers
                self._kg2 = None
        return self._kg2

    def name(self, slug: str) -> str:
        return (self.cards.get(slug) or {}).get("name") or slug.replace("-", " ").title()

    # ---------------------------------------------------------------------------------- a pro state
    def state(self, state_id: str) -> dict | None:
        """One retrieved pro state: game / video, time in the video, context text, the pro's action, verdict, trade."""
        if not self.available() or not state_id:
            return None
        with self._lock:
            row = self._db().execute(
                "SELECT s.id, s.game_id, s.kind, s.t, s.phase, s.match_seconds, s.own_elixir, s.action_type, "
                "s.action_card, s.action_lane, s.action_zone, s.action_key, s.verdict, s.elixir_trade, s.flags, "
                "s.context_text, g.own_deck, g.opponent_deck, g.result, p.video_id, p.video_title, p.video_url, "
                "p.creator, p.video_time FROM states s LEFT JOIN games g ON g.game_id = s.game_id "
                "LEFT JOIN provenance p ON p.game_id = s.game_id WHERE s.id = ?", (state_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("own_deck", "opponent_deck", "flags", "video_time"):
            try:
                d[k] = json.loads(d[k]) if d.get(k) else None
            except ValueError:
                pass
        t = d.get("t")
        d["video_time_s"] = t                                    # states.t is the second in the source video
        url = d.get("video_url")
        d["video_link"] = f"{url}&t={int(t)}s" if url and isinstance(t, (int, float)) else url
        d["clock"] = _clock(d.get("context_text"))
        return d

    # ---------------------------------------------------------------------------------- the knowledge graph
    def kg(self, cards: list[str], own: list[str] | None = None, enemy: list[str] | None = None,
           focus: str | None = None, deck: list[str] | None = None) -> dict:
        """The subgraph among `cards` (KG v1: COUNTERS, ANSWERED_BY, a few SYNERGY) plus the top-3 counters of each
        enemy card (of every card when `enemy` is not given), and the KG v2 answers of our `deck` / `own` cards to each
        enemy card. Nodes: {id, label, side: own|enemy|other}; edges: {src, dst, rel, weight, counter, threat, ...}.

        Directions are the graph's own: a COUNTERS edge is stored src = the counter, dst = the card it counters;
        an ANSWERED_BY edge is stored src = the threat, dst = the card that answers it (mined from pro games).
        `counter` / `threat` name the two roles of either relation so a client can draw both the same way."""
        cards = list(dict.fromkeys(c for c in cards if c))
        own_set = set(own or [])
        enemy_list = list(dict.fromkeys(enemy if enemy is not None else cards))
        enemy_set = set(enemy or [])
        with self._lock:
            kg = self._graph()
            nodes: dict[str, dict] = {}

            def add(slug: str, side: str | None = None) -> None:
                if slug not in nodes:
                    nodes[slug] = {"id": slug, "label": self.name(slug),
                                   "side": side or ("enemy" if slug in enemy_set and slug not in own_set
                                                    else "own" if slug in own_set else "other")}

            for c in cards:
                add(c)
            edges: dict[tuple, dict] = {}

            def add_edge(src: str, dst: str, rel: str, props: dict, kind: str) -> None:
                key = (src, rel, dst)
                if key in edges:
                    return
                if rel == "ANSWERED_BY":
                    weight = float(props.get("confidence", 0.0)) * min(1.0, float(props.get("support", 0)) / 10.0)
                    counter, threat = dst, src
                elif rel == "COUNTERS":
                    weight = float(props.get("count", 1.0))
                    counter, threat = src, dst
                else:
                    weight = float(props.get("count", 1.0))
                    counter = threat = None
                ev = (props.get("evidence") or [None])[0]
                edges[key] = {"src": src, "dst": dst, "rel": rel, "weight": round(weight, 4), "counter": counter,
                              "threat": threat, "kind": kind, "source": props.get("source"),
                              "evidence": (ev[:220] if isinstance(ev, str) else None)}

            card_set = set(cards)
            synergy = 0
            for a in cards:
                for rel in KG_RELS:
                    for n in kg.neighbors(f"card:{a}", rel, "out", limit=500):
                        b = n["node"].removeprefix("card:")
                        if b not in card_set or b == a:
                            continue
                        if rel == "SYNERGY":
                            if not (a in own_set and b in own_set) or synergy >= MAX_SYNERGY_EDGES:
                                continue
                            synergy += 1
                        add_edge(a, b, rel, n["props"], "among")
            tops: dict[str, list[dict]] = {}
            for threat in enemy_list:
                scores = kg.counter_scores(threat)
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_COUNTERS]
                tops[threat] = [{"card": c, "score": round(s, 4)} for c, s in ranked]
                add(threat)
                for c, _s in ranked:
                    add(c)
                    for n in kg.neighbors(f"card:{c}", "COUNTERS", "out", limit=500):
                        if n["node"] == f"card:{threat}":
                            add_edge(c, threat, "COUNTERS", n["props"], "top_counter")
                    for n in kg.neighbors(f"card:{threat}", "ANSWERED_BY", "out", limit=500):
                        if n["node"] == f"card:{c}":
                            add_edge(threat, c, "ANSWERED_BY", n["props"], "top_counter")
            kg2 = self._kg2_answers(enemy_list, deck or own or [])
        return {"nodes": list(nodes.values()), "edges": list(edges.values()), "focus": focus, "top_counters": tops,
                "kg2": kg2,
                "convention": {"COUNTERS": "stored src = the counter, dst = the card it counters (src counters dst)",
                               "ANSWERED_BY": "stored src = the threat, dst = the card that answers it",
                               "counter/threat": "the answering card and the card being answered, for either relation"}}

    def _kg2_answers(self, threats: list[str], ours: list[str]) -> dict | None:
        kg2 = self._graph2()
        if kg2 is None or not ours:
            return None
        out: dict[str, list[dict]] = {}
        for threat in threats:
            try:
                answers = kg2.answers_for(threat, list(ours), k=3)
            except Exception:
                continue
            out[threat] = [{"card": a.card, "score": a.score, "sources": a.sources, "support": a.support,
                            "held_rate": a.held_rate, "via": a.via, "prevented": a.prevented, "adequate": a.adequate}
                           for a in answers]
        return out


def _clock(text: str | None) -> str | None:
    m = re.match(r"\[[a-z_?]+\|([0-9?]+:[0-9?]+)\]", text or "")
    return m.group(1) if m else None
