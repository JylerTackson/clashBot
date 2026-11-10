"""`python -m clashbot.viz [--run DIR | --follow PARENT] [--parents A,B] [--port 8765] [--open]` - serve the decision dashboard.

`--follow PARENT` (default runs/demo) shows the newest run folder under PARENT and switches when a newer one appears;
`--run DIR` shows one fixed run (replay). `--parents A,B,...` (or env CLASHBOT_VIZ_PARENTS, comma-separated) adds more run
parents to the run picker, e.g. the prior games packed into the Docker image (MAC-AW); they are listed, never followed.
Binds 127.0.0.1 unless `--host` says otherwise (the Docker image passes `--host 0.0.0.0` so the published port reaches
it). Ctrl-C (or SIGTERM from the launcher) stops it."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import webbrowser
from pathlib import Path

from clashbot.config import ROOT
from clashbot.viz.data import Store
from clashbot.viz.server import make_server


def _path(arg: str) -> Path:
    p = Path(arg).expanduser()
    if not p.is_absolute() and not p.exists():
        p = ROOT / p                                           # runs/demo from any working directory (it may not exist yet)
    return p


def parent_paths(arg_values: list[str] | None, env: str | None = None) -> list[Path]:
    """The extra run parents of `--parents` (repeatable, each a comma-separated list) followed by those of the environment
    variable CLASHBOT_VIZ_PARENTS; empty items dropped, duplicates removed, order kept."""
    items: list[str] = []
    for value in list(arg_values or []) + [os.environ.get("CLASHBOT_VIZ_PARENTS", "") if env is None else env]:
        items += [x.strip() for x in str(value).split(",") if x.strip()]
    out: list[Path] = []
    for item in items:
        p = _path(item)
        if p not in out:
            out.append(p)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m clashbot.viz", description=__doc__.splitlines()[0])
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--run", metavar="DIR", help="show this run folder (replay)")
    which.add_argument("--follow", metavar="PARENT", help="show the newest run folder under PARENT (default runs/demo)")
    ap.add_argument("--parents", metavar="A,B", action="append",
                    help="more run parents for the run picker (comma-separated; env CLASHBOT_VIZ_PARENTS): their run folders are listed")
    ap.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1; 0.0.0.0 inside a container)")
    ap.add_argument("--port", type=int, default=8765, help="port to listen on (0 = any free port)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parents = parent_paths(args.parents)
    if args.run:
        run = _path(args.run)
        if not run.is_dir():
            print(f"viz: run folder not found: {run}", file=sys.stderr)
            return 2
        store = Store(run=run, parents=parents)
    else:
        store = Store(follow=_path(args.follow or "runs/demo"), parents=parents)
    try:
        server = make_server(store, args.port, host=args.host)
    except OSError as exc:
        print(f"viz: cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 2
    shown = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown}:{server.server_address[1]}/"
    what = f"run {store.fixed}" if store.fixed else f"newest run under {store.follow}"
    print(f"decision dashboard: {url}  ({what}; read-only; Ctrl-C to stop)", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    if args.open:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
