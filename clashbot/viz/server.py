"""HTTP server of the decision dashboard (MAC-AO): a JSON API over a run directory plus the static page.

Standard library only (`http.server.ThreadingHTTPServer`), bound to 127.0.0.1, read-only: it never writes to a run, to
db/ or to the screen, and sends no input. Routes (all GET):

  /                                  the page (static/index.html, app.js, style.css)
  /api/current                       which run the page shows: {follow, parent, id, live}
  /api/runs                          the run folders under runs/demo, runs/poc, runs/smoke
  /api/run/<id>/meta                 config, summary, battles, deck marks, counts
  /api/run/<id>/decisions?since=N    decision rows from index N, each with its shot files and parsed context
  /api/run/<id>/holds?since=N        hold rows from index N (+ reason buckets, digits replaced by "#")
  /api/shot/<id>/<file>              a jpg from that run's shots/
  /api/state/<state id>              one retrieved pro state
  /api/kg?cards=a,b&own=..&enemy=..&focus=x&deck=..   the knowledge-graph subgraph
"""
from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from clashbot.game_rules import load_cards
from clashbot.viz.data import Reference, Store

STATIC = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
MAX_STATE_ID = 200


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

    def handle(self, path: str) -> tuple[int, str, bytes]:
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
                return self.run_route(segs[2], segs[3], q)
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

    def run_route(self, run_id: str, what: str, q: dict) -> tuple[int, str, bytes]:
        run = self.store.run(run_id)
        if run is None:
            return self._err(404, "no such run")
        if what == "meta":
            return self._json(run.meta(self.cards))
        if what == "decisions":
            limit = self._int(q, "limit") if "limit" in q else None
            return self._json(run.decisions_since(self._int(q, "since"), limit))
        if what == "holds":
            return self._json(run.holds_since(self._int(q, "since")))
        return self._err(404, "not found")

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

        def do_GET(self) -> None:                                 # noqa: N802 (http.server API)
            try:
                status, ctype, body = api.handle(self.path)
            except Exception as exc:                              # a bad row or file must not kill the page
                status, ctype = 500, "application/json"
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
            self.send_response(status, HTTPStatus(status).phrase)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

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
