"""HTTP server of the decision dashboard (MAC-AO): a JSON API over a run directory plus the static page.

Standard library only (`http.server.ThreadingHTTPServer`), bound to 127.0.0.1 (`--host` changes it), read-only: it never writes to a run, to
db/ or to the screen, and sends no input. Routes (all GET):

  /                                  the page (static/index.html, app.js, style.css)
  /api/current                       which run the page shows: {follow, parent, id, live}
  /api/runs                          the run folders under runs/demo, runs/poc, runs/smoke
  /api/run/<id>/meta                 config, summary, battles, deck marks, counts
  /api/run/<id>/decisions?since=N    decision rows from index N, each with its shot files and parsed context
  /api/run/<id>/holds?since=N        hold rows from index N (+ reason buckets, digits replaced by "#")
  /api/run/<id>/pro                  the pro's plays of a run made on a pro video (pro_plays.jsonl, MAC-AQ)
  /api/run/<id>/livemeta             the Recording panel's data: live_meta.json (or derived from config.json), has_video, live,
                                     finished, the server's clock and the agent's progress / lag (MAC-AW)
  /api/run/<id>/video                the run's source video (the .mp4 named in its config.json), HTTP Range supported (MAC-AW)
  /api/run/<id>/player               the frames of the screenshot player, ordered by time (MAC-AW)
  /api/shot/<id>/<file>              a jpg from that run's shots/
  /api/state/<state id>              one retrieved pro state
  /api/kg?cards=a,b&own=..&enemy=..&focus=x&deck=..   the knowledge-graph subgraph
"""
from __future__ import annotations

import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from clashbot.game_rules import load_cards
from clashbot.viz.data import Reference, Store

STATIC = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css",
                "/recording.js": "recording.js"}
MAX_STATE_ID = 200
CHUNK = 1 << 16                                               # bytes of a video copied to the socket at a time
RANGE_RE = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", re.I)


class FileRange:
    """A byte range of a file to send as the body (the video endpoint): `start`..`end` inclusive of `size` bytes in all."""

    def __init__(self, path: Path, start: int, end: int, size: int, partial: bool):
        self.path, self.start, self.end, self.size, self.partial = path, start, end, size, partial
        self.length = end - start + 1
        self.headers = {"Accept-Ranges": "bytes"}
        if partial:
            self.headers["Content-Range"] = f"bytes {start}-{end}/{size}"

    def chunks(self):
        left = self.length
        with open(self.path, "rb") as fh:
            fh.seek(self.start)
            while left > 0:
                data = fh.read(min(CHUNK, left))
                if not data:
                    return                                     # the file shrank under us
                left -= len(data)
                yield data

    def read(self) -> bytes:
        return b"".join(self.chunks())


def parse_range(header: str | None, size: int) -> tuple[int, int] | str | None:
    """A Range header against a file of `size` bytes: (start, end) inclusive for a satisfiable single byte range, "unsatisfiable"
    for a range outside the file (answer 416), None when there is no usable Range header (answer 200 with the whole file: a
    header that is not `bytes=a-b`, a malformed one, or several ranges is ignored, as the RFC allows)."""
    m = RANGE_RE.match(header or "")
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return None
    first, last = m.group(1), m.group(2)
    if first == "":                                            # bytes=-N: the last N bytes
        n = int(last)
        return "unsatisfiable" if n == 0 or size == 0 else (max(size - n, 0), size - 1)
    start = int(first)
    end = int(last) if last else size - 1
    if last and end < start:
        return None                                            # backwards range: invalid, ignored
    if start >= size:
        return "unsatisfiable"
    return start, min(end, size - 1)


class Api:
    """The routes as plain functions of (path segments, query) -> (status, content type, bytes); no socket in here."""

    def __init__(self, store: Store, reference: Reference | None = None, cards: dict | None = None):
        self.store = store
        self.cards = cards if cards is not None else _cards()
        self.reference = reference or Reference(cards=self.cards)

    @staticmethod
    def _json(obj, status: int = 200) -> tuple[int, str, bytes]:
        return status, "application/json", json.dumps(obj, separators=(",", ":")).encode()

    def _err(self, status: int, msg: str) -> tuple[int, str, bytes]:
        return self._json({"error": msg}, status)

    @staticmethod
    def _int(q: dict, name: str, default: int = 0) -> int:
        try:
            return max(int(q.get(name, [default])[0]), 0)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _list(q: dict, name: str) -> list[str]:
        return [c.strip() for c in (q.get(name, [""])[0]).split(",") if c.strip()]

    def handle(self, path: str, headers=None) -> tuple[int, str, bytes | FileRange]:
        """`headers` (the request's, case-insensitive for a real message, a plain dict in tests) only matters for the video's Range."""
        parts = urlsplit(path)
        if parts.path in STATIC_FILES:
            return self.static(STATIC_FILES[parts.path])
        # split before unquoting so an encoded slash stays inside its segment (state ids contain '#')
        segs = [unquote(s) for s in parts.path.split("/")[1:]]
        if len(segs) < 2 or segs[0] != "api":
            return self._err(404, "not found")
        q = parse_qs(parts.query, keep_blank_values=True)
        route = segs[1]
        try:
            if route == "current" and len(segs) == 2:
                return self._json(self.store.current())
            if route == "runs" and len(segs) == 2:
                return self._json({"runs": self.store.list_runs()})
            if route == "run" and len(segs) == 4:
                return self.run_route(segs[2], segs[3], q, headers)
            if route == "shot" and len(segs) == 4:
                return self.shot(segs[2], segs[3])
            if route == "state" and len(segs) >= 3:
                state_id = "/".join(segs[2:])
                if len(state_id) > MAX_STATE_ID:
                    return self._err(400, "state id too long")
                state = self.reference.state(state_id)
                if state is None:
                    return self._err(404, "no such state (or db/ missing)")
                return self._json(state)
            if route == "kg" and len(segs) == 2:
                return self.kg(q)
        except FileNotFoundError as exc:
            return self._err(503, f"reference data missing: {exc}")
        return self._err(404, "not found")

    def run_route(self, run_id: str, what: str, q: dict, headers=None) -> tuple[int, str, bytes | FileRange]:
        run = self.store.run(run_id)
        if run is None:
            return self._err(404, "no such run")
        if what == "video":
            return self.video(run, headers)
        if what == "livemeta":
            return self._json(run.recording())
        if what == "player":
            return self._json(run.player_frames())
        if what == "meta":
            return self._json(run.meta(self.cards))
        if what == "decisions":
            limit = self._int(q, "limit") if "limit" in q else None
            return self._json(run.decisions_since(self._int(q, "since"), limit))
        if what == "holds":
            return self._json(run.holds_since(self._int(q, "since")))
        if what == "pro":
            return self._json({"has_pro": run.has_pro(), "plays": run.pro_plays()})
        return self._err(404, "not found")

    def video(self, run, headers=None) -> tuple[int, str, bytes | FileRange]:
        """The run's source video with Range support (206 + Content-Range for a satisfiable range, 416 outside the file, 200 for no
        Range header). Only the `.mp4` that run's config.json names is ever served: the URL carries no file name."""
        path = run.video_file()
        if path is None:
            return self._err(404, "this run has no playable video")
        try:
            size = path.stat().st_size
        except OSError:
            return self._err(404, "this run has no playable video")
        wanted = parse_range(headers.get("Range") if headers is not None else None, size)
        if wanted == "unsatisfiable":
            return 416, "application/json", json.dumps({"error": "range not satisfiable", "size": size}).encode()
        if wanted is None:
            return 200, "video/mp4", FileRange(path, 0, size - 1, size, partial=False)
        return 206, "video/mp4", FileRange(path, wanted[0], wanted[1], size, partial=True)

    def shot(self, run_id: str, name: str) -> tuple[int, str, bytes]:
        run = self.store.run(run_id)
        if run is None:
            return self._err(404, "no such run")
        if "/" in name or "\\" in name or ".." in name or name.startswith(".") or not name.lower().endswith(".jpg"):
            return self._err(400, "bad shot name")
        path = run.shot_path(name)
        if path is None:
            return self._err(404, "no such shot")
        return 200, "image/jpeg", path.read_bytes()

    def kg(self, q: dict) -> tuple[int, str, bytes]:
        cards = self._list(q, "cards")
        if not cards:
            return self._err(400, "cards= is required (comma-separated card slugs)")
        if len(cards) > 40:
            return self._err(400, "too many cards")
        if not self.reference.available():
            return self._err(503, "db/clashbot.sqlite is missing")
        focus = (q.get("focus", [""])[0] or None)
        enemy = self._list(q, "enemy") if "enemy" in q else None          # absent: every card counts as a threat
        return self._json(self.reference.kg(cards, own=self._list(q, "own"), enemy=enemy, focus=focus,
                                            deck=self._list(q, "deck")))

    @staticmethod
    def static(name: str) -> tuple[int, str, bytes]:
        path = STATIC / name
        if not path.is_file():
            return 404, "text/plain", b"missing static file"
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        return 200, ctype, path.read_bytes()


def _cards() -> dict:
    try:
        return load_cards()
    except Exception:
        return {}


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        server_version = "clashbot-viz"

        def _respond(self, send_body: bool) -> None:
            try:
                status, ctype, body = api.handle(self.path, self.headers)
            except Exception as exc:                              # a bad row or file must not kill the page
                status, ctype = 500, "application/json"
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
            streamed = isinstance(body, FileRange)
            self.send_response(status, HTTPStatus(status).phrase)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(body.length if streamed else len(body)))
            if streamed:
                for k, v in body.headers.items():
                    self.send_header(k, v)
                self.send_header("Cache-Control", "no-cache")
            else:
                if status == 416 and ctype == "application/json":
                    self.send_header("Content-Range", f"bytes */{json.loads(body).get('size', 0)}")
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not send_body:
                return
            try:
                if streamed:
                    for chunk in body.chunks():
                        self.wfile.write(chunk)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):       # the player seeks: it drops a request it no longer needs
                self.close_connection = True

        def do_GET(self) -> None:                                 # noqa: N802 (http.server API)
            self._respond(True)

        def do_HEAD(self) -> None:                                # noqa: N802
            self._respond(False)

        def log_message(self, format: str, *args) -> None:        # noqa: A002 (the page polls every second: stay quiet)
            return

    return Handler


def make_server(store: Store, port: int = 8765, host: str = "127.0.0.1", reference: Reference | None = None,
                cards: dict | None = None) -> ThreadingHTTPServer:
    """A ready (bound, not yet serving) server; `port=0` picks a free port (`server.server_address[1]`)."""
    api = Api(store, reference, cards)
    server = ThreadingHTTPServer((host, port), make_handler(api))
    server.daemon_threads = True
    return server
